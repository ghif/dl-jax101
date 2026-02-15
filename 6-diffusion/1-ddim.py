import jax
import jax.numpy as jnp
from flax import nnx
import matplotlib.pyplot as plt
import sys, os
import numpy as np
import time as timer
from tqdm import tqdm
import grain.python as grain
import optax
import urllib.request
import tarfile
import pickle

# Add parent directory to path to import utils
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
import viz_utils as vu
import model_utils as mu

# Define constants
DATA_DIR = os.path.join(parent_dir, "data")
MODEL_DIR = os.path.join(parent_dir, "models") 

BATCH_SIZE = 64
NUM_EPOCH = 50
IMAGE_SIZE = 32 # CIFAR-10 default size
NC = 3 
LR = 2e-4
TIMESTEPS = 1.0 # Continuous time [0, 1]
SAMPLE_STEPS = 20 # DDIM sampling steps
NVIZ = 64
DTYPE = jnp.bfloat16 # A100 optimized
EMA_DECAY = 0.999
MIN_SIGNAL_RATE = 0.02
MAX_SIGNAL_RATE = 0.95

DATASET = 'cifar10'
checkpoint_dir = os.path.join(MODEL_DIR, f"ddim_{DATASET}")
sample_dir = os.path.join(checkpoint_dir, "samples")

for d in [checkpoint_dir, sample_dir, DATA_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# 1. Dataset Loading (CIFAR-10)
def download_and_extract_cifar10(dest_dir):
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = os.path.join(dest_dir, "cifar-10-python.tar.gz")
    extract_path = os.path.join(dest_dir, "cifar-10-batches-py")
    if not os.path.exists(extract_path):
        if not os.path.exists(filename):
            print(f"Downloading {url}...")
            urllib.request.urlretrieve(url, filename)
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(path=dest_dir)
    return extract_path

def load_cifar10_local(data_dir):
    def unpickle(file):
        with open(file, 'rb') as fo:
            d = pickle.load(fo, encoding='bytes')
        return d
    images = []
    for i in range(1, 6):
        batch = unpickle(os.path.join(data_dir, f"data_batch_{i}"))
        images.append(batch[b'data'])
    X_train = np.vstack(images)
    return X_train

print(f"Loading {DATASET}...")
cifar_path = download_and_extract_cifar10(DATA_DIR)
X_train_all = load_cifar10_local(cifar_path)

class CIFARSource(grain.RandomAccessDataSource):
    def __init__(self, images):
        self._images = images
    def __len__(self): return len(self._images)
    def __getitem__(self, index):
        # CIFAR-10 raw (3, 32, 32) -> (32, 32, 3)
        img = self._images[index].reshape(3, 32, 32).transpose(1, 2, 0).astype(np.uint8)
        img = np.array(img).astype(np.float32)
        # Normalize to [-1, 1]
        image = (img / 127.5) - 1.0
        return {'image': image}

def create_loader(data_source, batch_size, shuffle=True, seed=0):
    sampler = grain.IndexSampler(num_records=len(data_source), shard_options=grain.NoSharding(), shuffle=shuffle, num_epochs=1, seed=seed)
    dataloader = grain.DataLoader(data_source=data_source, sampler=sampler, worker_count=0)
    
    class BatchIterator:
        def __init__(self, loader, batch_size, num_records):
            self.loader, self.batch_size, self.num_records = loader, batch_size, num_records
        def __len__(self): return (self.num_records + self.batch_size - 1) // self.batch_size
        def __iter__(self):
            batch_images = []
            for record in self.loader:
                batch_images.append(record['image'])
                if len(batch_images) == self.batch_size:
                    yield np.stack(batch_images)
                    batch_images = []
            if batch_images: yield np.stack(batch_images)
    return BatchIterator(dataloader, batch_size, len(data_source))

train_loader = create_loader(CIFARSource(X_train_all), BATCH_SIZE)

# 2. DDIM Scheduler Logic (Cosine-based)
class DDIMScheduler:
    def __init__(self, min_signal_rate=0.02, max_signal_rate=0.95):
        self.min_signal_rate = min_signal_rate
        self.max_signal_rate = max_signal_rate
        
    def get_schedule(self, diffusion_times):
        # diffusion_times: (N, 1, 1, 1) or scalar
        start_angle = jnp.arccos(self.max_signal_rate)
        end_angle = jnp.arccos(self.min_signal_rate)
        
        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
        
        signal_rates = jnp.cos(diffusion_angles)
        noise_rates = jnp.sin(diffusion_angles)
        return noise_rates, signal_rates

    def add_noise(self, x_start, noise, noise_rates, signal_rates):
        # x_start: (N, H, W, C), noise: (N, H, W, C)
        return signal_rates * x_start + noise_rates * noise

scheduler = DDIMScheduler(MIN_SIGNAL_RATE, MAX_SIGNAL_RATE)

# 3. Model Architecture (Simplified U-Net)
class TimeEmbedding(nnx.Module):
    def __init__(self, dim, rngs: nnx.Rngs, dtype=jnp.float32):
        self.dim = dim
        self.dtype = dtype
        self.mlp = nnx.Sequential(
            nnx.Linear(dim, dim * 4, param_dtype=jnp.float32, dtype=dtype, rngs=rngs),
            nnx.swish,
            nnx.Linear(dim * 4, dim, param_dtype=jnp.float32, dtype=dtype, rngs=rngs)
        )

    def __call__(self, noise_rates):
        # noise_rates: (N, 1, 1, 1)
        # Sinusoidal embedding matching the reference logic
        half_dim = self.dim // 2
        frequencies = jnp.exp(
            jnp.linspace(
                jnp.log(1.0),
                jnp.log(1000.0),
                half_dim,
            )
        )
        angular_speeds = 2.0 * jnp.pi * frequencies
        # Reshape noise_rates to (N, 1) for embedding then back to (N, 1, 1, dim)
        x = noise_rates.reshape(-1, 1)
        embeddings = jnp.concatenate(
            [jnp.sin(angular_speeds * x), jnp.cos(angular_speeds * x)], axis=-1
        )
        # Resulting embeddings: (N, dim)
        return self.mlp(embeddings.astype(self.dtype))

class ResBlock(nnx.Module):
    def __init__(self, in_ch, out_ch, time_dim, rngs: nnx.Rngs, dtype=jnp.float32):
        self.dtype = dtype
        self.conv1 = nnx.Conv(in_ch, out_ch, (3, 3), padding='SAME', param_dtype=jnp.float32, dtype=dtype, rngs=rngs)
        self.bn1 = nnx.BatchNorm(out_ch, param_dtype=jnp.float32, dtype=dtype, rngs=rngs)
        self.time_proj = nnx.Linear(time_dim, out_ch, param_dtype=jnp.float32, dtype=dtype, rngs=rngs)
        self.conv2 = nnx.Conv(out_ch, out_ch, (3, 3), padding='SAME', param_dtype=jnp.float32, dtype=dtype, rngs=rngs)
        self.bn2 = nnx.BatchNorm(out_ch, param_dtype=jnp.float32, dtype=dtype, rngs=rngs)
        self.shortcut = nnx.Conv(in_ch, out_ch, (1, 1), param_dtype=jnp.float32, dtype=dtype, rngs=rngs) if in_ch != out_ch else (lambda x: x)

    def __call__(self, x, t_emb, train=True):
        h = nnx.swish(self.bn1(self.conv1(x), use_running_average=not train))
        h = h + self.time_proj(nnx.swish(t_emb))[:, None, None, :]
        h = nnx.swish(self.bn2(self.conv2(h), use_running_average=not train))
        return h + self.shortcut(x)

class UNet(nnx.Module):
    def __init__(self, in_ch, widths, rngs: nnx.Rngs, dtype=jnp.float32):
        time_dim = widths[0] * 4
        self.dtype = dtype
        self.time_mlp = TimeEmbedding(time_dim, rngs, dtype=dtype)
        
        # Encoder
        self.inc = nnx.Conv(in_ch, widths[0], (3, 3), padding='SAME', param_dtype=jnp.float32, dtype=dtype, rngs=rngs)
        self.down1 = ResBlock(widths[0], widths[1], time_dim, rngs, dtype=dtype)
        self.down2 = ResBlock(widths[1], widths[2], time_dim, rngs, dtype=dtype)
        
        # Bottleneck
        self.mid1 = ResBlock(widths[2], widths[3], time_dim, rngs, dtype=dtype)
        self.mid2 = ResBlock(widths[3], widths[3], time_dim, rngs, dtype=dtype)
        
        # Decoder
        self.up1 = ResBlock(widths[3] + widths[2], widths[1], time_dim, rngs, dtype=dtype) # + skip
        self.up2 = ResBlock(widths[1] + widths[1], widths[0], time_dim, rngs, dtype=dtype) # + skip
        self.outc = nnx.Conv(widths[0], in_ch, (1, 1), param_dtype=jnp.float32, dtype=dtype, rngs=rngs, kernel_init=nnx.initializers.zeros)

    def __call__(self, x, noise_rates, train=True):
        t_emb = self.time_mlp(noise_rates)
        
        # Cast input to dtype
        x = x.astype(self.dtype)
        
        x1 = self.inc(x) 
        x2 = self.down1(x1, t_emb, train)
        x2_pool = nnx.avg_pool(x2, (2, 2), strides=(2, 2)) 
        
        x3 = self.down2(x2_pool, t_emb, train)
        x3_pool = nnx.avg_pool(x3, (2, 2), strides=(2, 2)) 
        
        h = self.mid1(x3_pool, t_emb, train)
        h = self.mid2(h, t_emb, train)
        
        # Upsample 1
        h = jax.image.resize(h, (h.shape[0], IMAGE_SIZE//2, IMAGE_SIZE//2, h.shape[-1]), method='bilinear')
        h = jnp.concatenate([h, x3], axis=-1)
        h = self.up1(h, t_emb, train)
        
        # Upsample 2
        h = jax.image.resize(h, (h.shape[0], IMAGE_SIZE, IMAGE_SIZE, h.shape[-1]), method='bilinear')
        h = jnp.concatenate([h, x2], axis=-1)
        h = self.up2(h, t_emb, train)
        
        return self.outc(h).astype(jnp.float32)

# 4. Training Logic
rngs = nnx.Rngs(0)
WIDTHS = [32, 64, 96, 128]
model = UNet(in_ch=NC, widths=WIDTHS, rngs=rngs, dtype=DTYPE)
# EMA Model for inference
ema_model = UNet(in_ch=NC, widths=WIDTHS, rngs=rngs, dtype=DTYPE)
# Initialize EMA weights with model weights
nnx.update(ema_model, nnx.state(model, nnx.Param))
optimizer = nnx.Optimizer(model, optax.adamw(LR, weight_decay=1e-4), wrt=nnx.Param)

@nnx.jit
def train_step(model, ema_model, optimizer, x_start, noise, diffusion_times):
    # Prepare rates
    noise_rates, signal_rates = scheduler.get_schedule(diffusion_times)
    
    # Mix images with noise (forward diffusion)
    x_noisy = scheduler.add_noise(x_start, noise, noise_rates, signal_rates)
    
    def loss_fn(model):
        pred_noise = model(x_noisy, noise_rates, train=True)
        # Loss calculation in float32: MAE as per reference
        return jnp.mean(jnp.abs(noise.astype(jnp.float32) - pred_noise.astype(jnp.float32)))
    
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    
    # Update EMA model weights
    # Get states for Parameters only
    model_state = nnx.state(model, nnx.Param)
    ema_state = nnx.state(ema_model, nnx.Param)
    
    new_ema_state = jax.tree.map(
        lambda p, e: EMA_DECAY * e + (1 - EMA_DECAY) * p,
        model_state, ema_state
    )
    nnx.update(ema_model, new_ema_state)
    
    return loss

# 5. DDIM Sampling Logic
@nnx.jit
def sample_ddim(model, x_t, noise_rates, signal_rates, next_noise_rates, next_signal_rates):
    # Deterministic sampling (eta=0) as per reference remix logic
    pred_noise = model(x_t, noise_rates, train=False)
    
    # 1. pred_image (x0 prediction)
    pred_image = (x_t - noise_rates * pred_noise) / signal_rates
    
    # 2. next_noisy_image (x_prev)
    x_prev = next_signal_rates * pred_image + next_noise_rates * pred_noise
    
    return x_prev, pred_image

def generate_samples(model, num_samples=16):
    x = jax.random.normal(jax.random.PRNGKey(42), (num_samples, IMAGE_SIZE, IMAGE_SIZE, NC))
    
    step_size = 1.0 / SAMPLE_STEPS
    
    for i in tqdm(range(SAMPLE_STEPS), desc="Sampling", leave=False):
        diffusion_times = jnp.full((num_samples, 1, 1, 1), 1.0 - i * step_size)
        next_diffusion_times = diffusion_times - step_size
        
        noise_rates, signal_rates = scheduler.get_schedule(diffusion_times)
        next_noise_rates, next_signal_rates = scheduler.get_schedule(next_diffusion_times)
        
        x, pred_image = sample_ddim(model, x, noise_rates, signal_rates, next_noise_rates, next_signal_rates)
    
    return pred_image # Reference returns the final predicted images

# 6. Main Training Loop
print("Starting DDIM Training...")
step_rng = jax.random.PRNGKey(0)

for epoch in range(NUM_EPOCH):
    total_loss = 0
    num_batches = 0
    with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}") as tepoch:
        for x_start in tepoch:
            step_rng, rng_t, rng_n = jax.random.split(step_rng, 3)
            
            # Sample continuous random timesteps [0, 1]
            diffusion_times = jax.random.uniform(rng_t, (x_start.shape[0], 1, 1, 1))
            # Sample noise
            noise = jax.random.normal(rng_n, x_start.shape)
            
            loss = train_step(model, ema_model, optimizer, x_start, noise, diffusion_times)
            total_loss += loss
            num_batches += 1
            tepoch.set_postfix(loss=f"{loss:.4f}")
            
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch+1} complete. Average Loss: {avg_loss:.4f}. Generating samples...")
    samples = generate_samples(ema_model, num_samples=NVIZ)
    grid = vu.set_grid(samples, num_cells=NVIZ)
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(np.array(vu.normalize(grid, 0, 1)), (1, 2, 0)))
    plt.axis('off')
    plt.title(f"DDIM Samples - Epoch {epoch+1}")
    plt.savefig(os.path.join(sample_dir, f"epoch_{epoch+1}.png"))
    plt.close()
    
    # Checkpointing
    mu.save_checkpoint(model, epoch + 1, filedir=checkpoint_dir)
