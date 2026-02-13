import jax
import jax.numpy as jnp
from flax import nnx
import matplotlib.pyplot as plt
import sys, os
import numpy as np
import time as timer
from tqdm import tqdm
import grain.python as grain
from sklearn.datasets import fetch_openml
import optax

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
import viz_utils as vu
import model_utils as mu

# Define constants
DATA_DIR = "../data"
# Ensure we use the jax/models directory
MODEL_DIR = "../models" 

BATCH_SIZE = 128
NUM_EPOCH = 50
NC = 1 # num channels
NZ = 100 # num latent variables
LR = 2e-4 # learning rate
BETA1 = 0.5 # beta1 for Adam optimizer
NVIZ = 64

DATASET = 'mnist'

mname = "gan"
checkpoint_dir = os.path.join(MODEL_DIR, f"{mname}_{DATASET}_z{NZ}")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    print(f'The new directory {checkpoint_dir} has been created')

sample_dir = os.path.join(checkpoint_dir, f"samples")
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
    print(f'The new directory {sample_dir} has been created')

# Load MNIST using scikit-learn
print("Loading MNIST via OpenML (may take a minute)...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
X_all, y_all = mnist.data, mnist.target.astype(np.int32)

# Split into train and test (60k / 10k)
X_train_all, X_test_all = X_all[:60000], X_all[60000:]
y_train_all, y_test_all = y_all[:60000], y_all[60000:]

class MNISTSource(grain.RandomAccessDataSource):
    def __init__(self, images, labels):
        self._images = images
        self._labels = labels
        
    def __len__(self):
        return len(self._images)
        
    def __getitem__(self, index):
        # MNIST in OpenML is flattened (784,)
        # Reshape to (1, 28, 28) and normalize to [-1, 1]
        image = (self._images[index].reshape(1, 28, 28).astype(np.float32) / 255.0) * 2.0 - 1.0
        label = self._labels[index]
        return {'image': image, 'label': label}

train_source = MNISTSource(X_train_all, y_train_all)
test_source = MNISTSource(X_test_all, y_test_all)

def create_loader(data_source, batch_size, shuffle=False, seed=0):
    sampler = grain.IndexSampler(
        num_records=len(data_source),
        shard_options=grain.NoSharding(),
        shuffle=shuffle,
        num_epochs=1,
        seed=seed,
    )
    dataloader = grain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        worker_count=0,
    )
    
    class BatchIterator:
        def __init__(self, loader, batch_size, num_records):
            self.loader = loader
            self.batch_size = batch_size
            self.num_records = num_records
        
        def __len__(self):
            return (self.num_records + self.batch_size - 1) // self.batch_size

        def __iter__(self):
            batch_images = []
            batch_labels = []
            for record in self.loader:
                batch_images.append(record['image'])
                batch_labels.append(record['label'])
                if len(batch_images) == self.batch_size:
                    yield np.stack(batch_images), np.array(batch_labels)
                    batch_images = []
                    batch_labels = []
            if batch_images:
                 yield np.stack(batch_images), np.array(batch_labels)
    
    return BatchIterator(dataloader, batch_size, len(data_source))

train_loader = create_loader(train_source, BATCH_SIZE, shuffle=True, seed=42)
test_loader = create_loader(test_source, BATCH_SIZE, shuffle=False, seed=42)


# Define Generator
class Generator(nnx.Module):
    def __init__(self, input_size=100, output_size=784, rngs: nnx.Rngs = None):
        self.layer = nnx.Sequential(
            nnx.Linear(input_size, 128, rngs=rngs),
            nnx.leaky_relu, # Default negative_slope=0.01, PyTorch uses 0.2
            # nnx.leaky_relu(negative_slope=0.2) is not directly callable in Sequential if using func, need lambda
            # But let's stick to simple structure or define custom layer
            nnx.Linear(128, 256, rngs=rngs),
            nnx.BatchNorm(256, rngs=rngs),
            nnx.leaky_relu,
            nnx.Linear(256, 512, rngs=rngs),
            nnx.BatchNorm(512, rngs=rngs),
            nnx.leaky_relu,
            nnx.Linear(512, 1024, rngs=rngs),
            nnx.BatchNorm(1024, rngs=rngs),
            nnx.leaky_relu,
            nnx.Linear(1024, output_size, rngs=rngs),
            nnx.tanh
        )
        # Note: In PyTorch code, leaky_relu slope is 0.2. 
        # Flax nnx.leaky_relu defaults to 0.01. 
        # For strict parity, we should use a lambda or custom module, but 0.01 is often fine.
        # Let's use 0.2 to match PyTorch exactly.
    
    def __call__(self, z):
        # We need to manually handle LeakyReLU with slope 0.2 if we want exact match
        # But nnx.Sequential with nnx.leaky_relu uses defaults.
        # Let's reconstruct to be explicit about slope.
        # Actually, let's redefine __init__ to separate layers to apply activation manually
        pass 

# Redefining Generator for explicit control
class Generator(nnx.Module):
    def __init__(self, input_size=100, output_size=784, rngs: nnx.Rngs = None):
        self.fc1 = nnx.Linear(input_size, 128, rngs=rngs)
        self.fc2 = nnx.Linear(128, 256, rngs=rngs)
        self.bn2 = nnx.BatchNorm(256, rngs=rngs)
        self.fc3 = nnx.Linear(256, 512, rngs=rngs)
        self.bn3 = nnx.BatchNorm(512, rngs=rngs)
        self.fc4 = nnx.Linear(512, 1024, rngs=rngs)
        self.bn4 = nnx.BatchNorm(1024, rngs=rngs)
        self.fc5 = nnx.Linear(1024, output_size, rngs=rngs)
    
    def __call__(self, z):
        h = self.fc1(z)
        h = nnx.leaky_relu(h, negative_slope=0.2)
        
        h = self.fc2(h)
        h = self.bn2(h)
        h = nnx.leaky_relu(h, negative_slope=0.2)
        
        h = self.fc3(h)
        h = self.bn3(h)
        h = nnx.leaky_relu(h, negative_slope=0.2)
        
        h = self.fc4(h)
        h = self.bn4(h)
        h = nnx.leaky_relu(h, negative_slope=0.2)
        
        h = self.fc5(h)
        h = nnx.tanh(h)
        
        # Reshape to image (N, C, H, W) is done outside or here. 
        # PyTorch code reshapes at return.
        h = h.reshape(h.shape[0], NC, 28, 28)
        return h

# Define Discriminator
class Discriminator(nnx.Module):
    def __init__(self, input_size=784, num_classes=1, rngs: nnx.Rngs = None):
        self.fc1 = nnx.Linear(input_size, 512, rngs=rngs)
        self.fc2 = nnx.Linear(512, 256, rngs=rngs)
        self.fc3 = nnx.Linear(256, num_classes, rngs=rngs)
        
    def __call__(self, x):
        # x input is (N, C, H, W) or (N, 784)
        x = x.reshape(x.shape[0], -1) 
        
        h = self.fc1(x)
        h = nnx.leaky_relu(h, negative_slope=0.2)
        
        h = self.fc2(h)
        h = nnx.leaky_relu(h, negative_slope=0.2)
        
        h = self.fc3(h)
        h = nnx.sigmoid(h)
        return h.flatten()

# Initialize models
input_size = 28 * 28
rngs = nnx.Rngs(0)

netG = Generator(input_size=NZ, output_size=input_size, rngs=rngs)
netD = Discriminator(input_size=input_size, num_classes=1, rngs=rngs)

# Optimizers
optimizerG = nnx.Optimizer(netG, optax.adam(LR, b1=BETA1, b2=0.999), wrt=nnx.Param)
optimizerD = nnx.Optimizer(netD, optax.adam(LR, b1=BETA1, b2=0.999), wrt=nnx.Param)

# Metrics
metrics_history = {
    'd_loss': [],
    'g_loss': [],
    'dx': [],
    'dgz1': [],
    'dgz2': []
}

# Binary Cross Entropy Loss
def binary_cross_entropy(logits, labels):
    # Since D returns sigmoid probabilities, we can use simple log loss
    # loss = - (y * log(p) + (1-y) * log(1-p))
    # However, for numerical stability, usually better to use logits with stable sigmoid cross entropy
    # But here Discriminator already has Sigmoid.
    # Let's use explicit BCE
    epsilon = 1e-12
    logits = jnp.clip(logits, epsilon, 1.0 - epsilon)
    loss = -(labels * jnp.log(logits) + (1 - labels) * jnp.log(1 - logits))
    return jnp.mean(loss)

@nnx.jit
def train_step_D(netD, netG, optimizerD, real_x, noise, rng):
    # Train D: maximize log(D(x)) + log(1 - D(G(z)))
    
    # Generate fake images
    fake_x = netG(noise)
    
    def loss_fn(netD):
        # Real
        real_pred = netD(real_x)
        real_labels = jnp.ones_like(real_pred)
        errD_real = binary_cross_entropy(real_pred, real_labels)
        
        # Fake
        fake_pred = netD(fake_x)
        fake_labels = jnp.zeros_like(fake_pred)
        errD_fake = binary_cross_entropy(fake_pred, fake_labels)
        
        errD = errD_real + errD_fake
        return errD, (real_pred, fake_pred)
    
    (loss, (real_pred, fake_pred)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(netD)
    optimizerD.update(netD, grads)
    
    return loss, jnp.mean(real_pred), jnp.mean(fake_pred)

@nnx.jit
def train_step_G(netD, netG, optimizerG, noise, rng):
    # Train G: maximize log(D(G(z))) -> minimize log(1 - D(G(z))) or maximize log(D(G(z)))
    # Standard GAN generator loss: minimize log(1 - D(G(z))) is saturating.
    # Non-saturating: maximize log(D(G(z))) <=> minimize -log(D(G(z))) <=> binary_cross_entropy(D(G(z)), 1)
    
    def loss_fn(netG):
        fake_x = netG(noise)
        outputD = netD(fake_x)
        
        # Genuine labels (1) for fake images to fool D
        labels = jnp.ones_like(outputD)
        errG = binary_cross_entropy(outputD, labels)
        
        return errG, outputD

    (loss, outputD), grads = nnx.value_and_grad(loss_fn, has_aux=True)(netG)
    optimizerG.update(netG, grads)
    
    return loss, jnp.mean(outputD)


# Fixed noise for visualization
fixed_latent = jax.random.normal(jax.random.PRNGKey(42), (64, NZ))

# Training Loop
step_rng = jax.random.PRNGKey(0)

print(f"Starting Training Loop...")
for epoch in range(NUM_EPOCH):
    start_t = timer.time()
    
    d_losses = []
    g_losses = []
    
    with tqdm(train_loader, unit="batch") as tepoch:
        for batch_idx, (real_x, _) in enumerate(tepoch):
            batch_size = real_x.shape[0]
            
            # Update step RNG
            step_rng, rng_d, rng_g = jax.random.split(step_rng, 3)
            
            # Train Discriminator
            noise_d = jax.random.normal(rng_d, (batch_size, NZ))
            errD, D_x, D_G_z1 = train_step_D(netD, netG, optimizerD, real_x, noise_d, rng_d)
            
            # Train Generator
            # Generate new noise for G update
            noise_g = jax.random.normal(rng_g, (batch_size, NZ))
            errG, D_G_z2 = train_step_G(netD, netG, optimizerG, noise_g, rng_g)
            
            d_losses.append(errD)
            g_losses.append(errG)
            
            if batch_idx % 100 == 0:
                tepoch.set_postfix(
                    Loss_D=f"{errD:.4f}",
                    Loss_G=f"{errG:.4f}",
                    D_x=f"{D_x:.4f}",
                    D_G_z1=f"{D_G_z1:.4f}",
                    D_G_z2=f"{D_G_z2:.4f}"
                )

    elapsed_t = timer.time() - start_t
    avg_d_loss = np.mean(d_losses)
    avg_g_loss = np.mean(g_losses)
    
    print(f'[{epoch+1}/{NUM_EPOCH}] Loss_D: {avg_d_loss:.4f} Loss_G: {avg_g_loss:.4f} Elapsed: {elapsed_t:.2f} s')
    
    # Save Real Samples (Only once)
    if epoch == 0:
        vutils_real = real_x[:NVIZ] if real_x.shape[0] >= NVIZ else real_x
        # real_x is (B, 1, 28, 28)
        grid_real = vu.set_grid(vutils_real, num_cells=64) # NVIZ used in original was 512, let's use 64
        plt.figure(figsize=(10, 10))
        plt.imshow(np.transpose(np.array(vu.normalize(grid_real, 0, 1)), (1, 2, 0)), cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(sample_dir, 'real_samples.jpg'), bbox_inches='tight')
        plt.close()

    if epoch % 10 != 0:
        continue

    # Save Fake Samples
    fake_samples = netG(fixed_latent)
    # fake_samples is (B, 1, 28, 28)
    grid_fake = vu.set_grid(fake_samples, num_cells=64)
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(np.array(vu.normalize(grid_fake, 0, 1)), (1, 2, 0)), cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(sample_dir, f'fake_samples_epoch-{epoch+1}.jpg'), bbox_inches='tight')
    plt.close()
    
    # Checkpointing (Save mostly locally or periodically)
    mu.save_checkpoint(netD, epoch + 1, filedir=checkpoint_dir) # Might need to differentiate dicts if saving both in one dir
    # wait, model_utils save_checkpoint uses 'epoch_{epoch}.safetensors'. 
    # If we run this for both, they overwrite.
    # We should probably modify save_checkpoint to accept prefix or handle separate folders.
    # For now, let's save G and D in separate subfolders or just rename manually/hackily?
    # Or just use the model_utils standard and assume single model.
    # Let's save them as:
    
    # Saving Generator
    path_g = os.path.join(checkpoint_dir, "generator")
    if not os.path.exists(path_g): os.makedirs(path_g)
    mu.save_checkpoint(netG, epoch + 1, filedir=path_g)
    
    # Saving Discriminator
    path_d = os.path.join(checkpoint_dir, "discriminator")
    if not os.path.exists(path_d): os.makedirs(path_d)
    mu.save_checkpoint(netD, epoch + 1, filedir=path_d)
    
    print(f" --- Models stored ---")

