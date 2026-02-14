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
TIMESTEPS = 1000
SAMPLE_STEPS = 50 # DDIM sampling steps
NVIZ = 64
DTYPE = jnp.bfloat16 # A100 optimized

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

# 2. DDIM Scheduler Logic
class DDIMScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = jnp.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
        
    def add_noise(self, x_start, noise, t):
        # x_start: (N, H, W, C), noise: (N, H, W, C), t: (N,)
        sqrt_alphas_cumprod_t = jnp.sqrt(self.alphas_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = jnp.sqrt(1 - self.alphas_cumprod[t])[:, None, None, None]
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

scheduler = DDIMScheduler(TIMESTEPS)

# 3. Model Architecture (Simplified U-Net)
class TimeEmbedding(nnx.Module):
    def __init__(self, dim, rngs: nnx.Rngs, dtype=jnp.float32):
        self.dim = dim
        self.dtype = dtype
        self.mlp = nnx.Sequential(
            nnx.Linear(dim, dim * 4, param_dtype=jnp.float32, dtype=dtype, rngs=rngs),
            nnx.relu,
            nnx.Linear(dim * 4, dim, param_dtype=jnp.float32, dtype=dtype, rngs=rngs)
        )

    def __call__(self, t):
        # Sinusoidal embedding
        half_dim = self.dim // 2
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return self.mlp(emb.astype(self.dtype))

class ResBlock(nnx.Module):
    def __init__(self, in_ch, out_ch, time_dim, rngs: nnx.Rngs, dtype=jnp.float32):
        self.dtype = dtype
        self.conv1 = nnx.Conv(in_ch, out_ch, (3, 3), padding='SAME', param_dtype=jnp.float32, dtype=dtype, rngs=rngs)
        self.bn1 = nnx.BatchNorm(out_ch, param_dtype=jnp.float32, dtype=dtype, rngs=rngs)
        self.time_proj = nnx.Linear(time_dim, out_ch, param_dtype=jnp.float32, dtype=dtype, rngs=rngs)
        self.conv2 = nnx.Conv(out_ch, out_ch, (3, 3), padding='SAME', param_dtype=jnp.float32, dtype=dtype, rngs=rngs)
        self.bn2 = nnx.BatchNorm(out_ch, param_dtype=jnp.float32, dtype=dtype, rngs=rngs)
        self.shortcut = nnx.Linear(in_ch, out_ch, param_dtype=jnp.float32, dtype=dtype, rngs=rngs) if in_ch != out_ch else (lambda x: x)

    def __call__(self, x, t_emb, train=True):
        h = nnx.relu(self.bn1(self.conv1(x), use_running_average=not train))
        h = h + self.time_proj(nnx.relu(t_emb))[:, None, None, :]
        h = nnx.relu(self.bn2(self.conv2(h), use_running_average=not train))
        return h + self.shortcut(x)

class UNet(nnx.Module):
    def __init__(self, in_ch, base_ch, rngs: nnx.Rngs, dtype=jnp.float32):
        time_dim = base_ch * 4
        self.dtype = dtype
        self.time_mlp = TimeEmbedding(time_dim, rngs, dtype=dtype)
        
        # Encoder
        self.inc = nnx.Conv(in_ch, base_ch, (3, 3), padding='SAME', param_dtype=jnp.float32, dtype=dtype, rngs=rngs)
        self.down1 = ResBlock(base_ch, base_ch * 2, time_dim, rngs, dtype=dtype)
        self.down2 = ResBlock(base_ch * 2, base_ch * 4, time_dim, rngs, dtype=dtype)
        
        # Bottleneck
        self.mid1 = ResBlock(base_ch * 4, base_ch * 4, time_dim, rngs, dtype=dtype)
        self.mid2 = ResBlock(base_ch * 4, base_ch * 4, time_dim, rngs, dtype=dtype)
        
        # Decoder
        self.up1 = ResBlock(base_ch * 8, base_ch * 2, time_dim, rngs, dtype=dtype) # + skip
        self.up2 = ResBlock(base_ch * 4, base_ch, time_dim, rngs, dtype=dtype) # + skip
        self.outc = nnx.Conv(base_ch, in_ch, (1, 1), param_dtype=jnp.float32, dtype=dtype, rngs=rngs)

    def __call__(self, x, t, train=True):
        t_emb = self.time_mlp(t)
        
        # Cast input to dtype
        x = x.astype(self.dtype)
        
        x1 = self.inc(x) # (32, 32, 64)
        x2 = self.down1(x1, t_emb, train)
        x2_pool = nnx.avg_pool(x2, (2, 2), strides=(2, 2)) # (16, 16, 128)
        
        x3 = self.down2(x2_pool, t_emb, train)
        x3_pool = nnx.avg_pool(x3, (2, 2), strides=(2, 2)) # (8, 8, 256)
        
        h = self.mid1(x3_pool, t_emb, train)
        h = self.mid2(h, t_emb, train)
        
        # Upsample 1
        h = jax.image.resize(h, (h.shape[0], 16, 16, h.shape[-1]), method='bilinear')
        h = jnp.concatenate([h, x3], axis=-1)
        h = self.up1(h, t_emb, train)
        
        # Upsample 2
        h = jax.image.resize(h, (h.shape[0], 32, 32, h.shape[-1]), method='bilinear')
        h = jnp.concatenate([h, x2], axis=-1)
        h = self.up2(h, t_emb, train)
        
        return self.outc(h).astype(jnp.float32)

# 4. Training Logic
rngs = nnx.Rngs(0)
model = UNet(in_ch=NC, base_ch=64, rngs=rngs, dtype=DTYPE)
optimizer = nnx.Optimizer(model, optax.adam(LR), wrt=nnx.Param)

@nnx.jit
def train_step(model, optimizer, x_start, t, noise):
    # Prepare x_noisy in float32 for scheduler stability, then cast during model call
    x_noisy = scheduler.add_noise(x_start, noise, t)
    
    def loss_fn(model):
        pred_noise = model(x_noisy, t, train=True)
        # Loss calculation in float32
        return jnp.mean((noise.astype(jnp.float32) - pred_noise.astype(jnp.float32))**2)
    
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

# 5. DDIM Sampling Logic
@nnx.jit
def sample_ddim(model, x_t, t, t_prev, eta=0.0):
    # Deterministic sampling (eta=0)
    pred_noise = model(x_t, t, train=False)
    
    alpha_t = scheduler.alphas_cumprod[t][:, None, None, None]
    alpha_prev = scheduler.alphas_cumprod[t_prev][:, None, None, None]
    
    # 1. x0 prediction
    pred_x0 = (x_t - jnp.sqrt(1 - alpha_t) * pred_noise) / jnp.sqrt(alpha_t)
    
    # 2. direction pointing to xt
    dir_xt = jnp.sqrt(1 - alpha_prev) * pred_noise
    
    # 3. next step
    x_prev = jnp.sqrt(alpha_prev) * pred_x0 + dir_xt
    return x_prev

def generate_samples(model, num_samples=16):
    x = jax.random.normal(jax.random.PRNGKey(42), (num_samples, IMAGE_SIZE, IMAGE_SIZE, NC))
    
    # Subsample timesteps for DDIM
    indices = jnp.linspace(TIMESTEPS - 1, 0, SAMPLE_STEPS).astype(jnp.int32)
    
    for i in tqdm(range(len(indices) - 1), desc="Sampling", leave=False):
        t = jnp.full((num_samples,), indices[i])
        t_prev = jnp.full((num_samples,), indices[i+1])
        x = sample_ddim(model, x, t, t_prev)
    
    return x

# 6. Main Training Loop
print("Starting DDIM Training...")
step_rng = jax.random.PRNGKey(0)

for epoch in range(NUM_EPOCH):
    total_loss = 0
    num_batches = 0
    with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}") as tepoch:
        for x_start in tepoch:
            step_rng, rng_t, rng_n = jax.random.split(step_rng, 3)
            
            # Sample random timesteps
            t = jax.random.randint(rng_t, (x_start.shape[0],), 0, TIMESTEPS)
            # Sample noise
            noise = jax.random.normal(rng_n, x_start.shape)
            
            loss = train_step(model, optimizer, x_start, t, noise)
            total_loss += loss
            num_batches += 1
            tepoch.set_postfix(loss=f"{loss:.4f}")
            
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch+1} complete. Average Loss: {avg_loss:.4f}. Generating samples...")
    samples = generate_samples(model, num_samples=NVIZ)
    grid = vu.set_grid(samples, num_cells=NVIZ)
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(np.array(vu.normalize(grid, 0, 1)), (1, 2, 0)))
    plt.axis('off')
    plt.title(f"DDIM Samples - Epoch {epoch+1}")
    plt.savefig(os.path.join(sample_dir, f"epoch_{epoch+1}.png"))
    plt.close()
    
    # Checkpointing
    mu.save_checkpoint(model, epoch + 1, filedir=checkpoint_dir)
