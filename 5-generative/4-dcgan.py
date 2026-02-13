import jax
# jax.config.update('jax_platform_name', 'cpu')
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
IMAGE_SIZE = 64
NC = 3 # RGB for CIFAR-10
NZ = 100 
NGF = 64 
NDF = 64 
LR = 1e-4 
BETA1 = 0.5 
NVIZ = 64

DATASET = 'cifar10'
checkpoint_dir = os.path.join(MODEL_DIR, f"dcgan_{DATASET}_z{NZ}_v2")
sample_dir = os.path.join(checkpoint_dir, "samples")

for d in [checkpoint_dir, sample_dir, DATA_DIR, 
          os.path.join(checkpoint_dir, "generator"), 
          os.path.join(checkpoint_dir, "discriminator")]:
    if not os.path.exists(d):
        os.makedirs(d)

def download_and_extract_cifar10(dest_dir):
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = os.path.join(dest_dir, "cifar-10-python.tar.gz")
    extract_path = os.path.join(dest_dir, "cifar-10-batches-py")
    
    if not os.path.exists(extract_path):
        if not os.path.exists(filename):
            print(f"Downloading {url}...")
            urllib.request.urlretrieve(url, filename)
            print("Download complete.")
        
        print(f"Extracting {filename}...")
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(path=dest_dir)
        print("Extraction complete.")
    return extract_path

def load_cifar10_local(data_dir):
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    images = []
    labels = []
    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f"data_batch_{i}")
        batch = unpickle(batch_file)
        images.append(batch[b'data'])
        labels.append(batch[b'labels'])
    
    X_train = np.vstack(images)
    y_train = np.hstack(labels).astype(np.int32)
    return X_train, y_train

# Load Data
print(f"Loading {DATASET} locally...")
cifar_path = download_and_extract_cifar10(DATA_DIR)
X_train_all, y_train_all = load_cifar10_local(cifar_path)

print(f"Dataset loaded. X_train_all shape: {X_train_all.shape}")

class CIFARSource(grain.RandomAccessDataSource):
    def __init__(self, images, labels):
        self._images = images
        self._labels = labels
        
    def __len__(self):
        return len(self._images)
        
    def __getitem__(self, index):
        from PIL import Image
        # CIFAR-10 in OpenML is flattened (3072,) -> (3, 32, 32) -> transpose to (32, 32, 3)
        img = self._images[index].reshape(3, 32, 32).transpose(1, 2, 0).astype(np.uint8)
        img = Image.fromarray(img)
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        img = np.array(img).astype(np.float32)
        # Normalize to [-1, 1]
        image = (img / 255.0) * 2.0 - 1.0
        label = self._labels[index]
        return {'image': image, 'label': label}

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
            batch_images, batch_labels = [], []
            for record in self.loader:
                batch_images.append(record['image'])
                batch_labels.append(record['label'])
                if len(batch_images) == self.batch_size:
                    yield np.stack(batch_images), np.array(batch_labels)
                    batch_images, batch_labels = [], []
            if batch_images:
                 yield np.stack(batch_images), np.array(batch_labels)
    return BatchIterator(dataloader, batch_size, len(data_source))

train_loader = create_loader(CIFARSource(X_train_all, y_train_all), BATCH_SIZE, shuffle=True, seed=42)

# Models
class Generator(nnx.Module):
    def __init__(self, nz, ngf, nc, rngs: nnx.Rngs):
        normal_init = nnx.initializers.normal(0.02)
        
        # Output: (N, 4, 4, ngf * 8)
        self.convt1 = nnx.ConvTranspose(nz, ngf * 8, kernel_size=(4, 4), strides=(1, 1), padding='VALID', 
                                        use_bias=False, rngs=rngs, kernel_init=normal_init)
        self.bn1 = nnx.BatchNorm(ngf * 8, rngs=rngs)
        
        # Output: (N, 8, 8, ngf * 4)
        self.convt2 = nnx.ConvTranspose(ngf * 8, ngf * 4, kernel_size=(4, 4), strides=(2, 2), padding='SAME', 
                                        use_bias=False, rngs=rngs, kernel_init=normal_init)
        self.bn2 = nnx.BatchNorm(ngf * 4, rngs=rngs)
        
        # Output: (N, 16, 16, ngf * 2)
        self.convt3 = nnx.ConvTranspose(ngf * 4, ngf * 2, kernel_size=(4, 4), strides=(2, 2), padding='SAME', 
                                        use_bias=False, rngs=rngs, kernel_init=normal_init)
        self.bn3 = nnx.BatchNorm(ngf * 2, rngs=rngs)

        # Output: (N, 32, 32, ngf)
        self.convt4 = nnx.ConvTranspose(ngf * 2, ngf, kernel_size=(4, 4), strides=(2, 2), padding='SAME', 
                                        use_bias=False, rngs=rngs, kernel_init=normal_init)
        self.bn4 = nnx.BatchNorm(ngf, rngs=rngs)
        
        # Output: (N, 64, 64, nc)
        self.convt5 = nnx.ConvTranspose(ngf, nc, kernel_size=(4, 4), strides=(2, 2), padding='SAME', 
                                        use_bias=False, rngs=rngs, kernel_init=normal_init)

    def __call__(self, z, train: bool = True, use_running_average: bool = None):
        if use_running_average is None:
            use_running_average = not train
        # z is (N, NZ). Reshape to (N, 1, 1, NZ) for NHWC ConvTranspose
        h = z.reshape(z.shape[0], 1, 1, -1)
        h = nnx.relu(self.bn1(self.convt1(h), use_running_average=use_running_average))
        h = nnx.relu(self.bn2(self.convt2(h), use_running_average=use_running_average))
        h = nnx.relu(self.bn3(self.convt3(h), use_running_average=use_running_average))
        h = nnx.relu(self.bn4(self.convt4(h), use_running_average=use_running_average))
        return nnx.tanh(self.convt5(h))

class Discriminator(nnx.Module):
    def __init__(self, nc, ndf, rngs: nnx.Rngs):
        normal_init = nnx.initializers.normal(0.02)
        
        self.conv1 = nnx.Conv(nc, ndf, kernel_size=(4, 4), strides=(2, 2), padding='SAME', 
                              use_bias=False, rngs=rngs, kernel_init=normal_init)
        self.conv2 = nnx.Conv(ndf, ndf * 2, kernel_size=(4, 4), strides=(2, 2), padding='SAME', 
                              use_bias=False, rngs=rngs, kernel_init=normal_init)
        self.bn2 = nnx.BatchNorm(ndf * 2, rngs=rngs)
        self.conv3 = nnx.Conv(ndf * 2, ndf * 4, kernel_size=(4, 4), strides=(2, 2), padding='SAME', 
                              use_bias=False, rngs=rngs, kernel_init=normal_init)
        self.bn3 = nnx.BatchNorm(ndf * 4, rngs=rngs)
        self.conv4 = nnx.Conv(ndf * 4, ndf * 8, kernel_size=(4, 4), strides=(2, 2), padding='SAME', 
                              use_bias=False, rngs=rngs, kernel_init=normal_init)
        self.bn4 = nnx.BatchNorm(ndf * 8, rngs=rngs)
        self.conv5 = nnx.Conv(ndf * 8, 1, kernel_size=(4, 4), strides=(1, 1), padding='VALID', 
                              use_bias=False, rngs=rngs, kernel_init=normal_init)

    def __call__(self, x, train: bool = True, use_running_average: bool = None):
        if use_running_average is None:
            use_running_average = not train
        h = nnx.leaky_relu(self.conv1(x), negative_slope=0.2)
        h = nnx.leaky_relu(self.bn2(self.conv2(h), use_running_average=use_running_average), negative_slope=0.2)
        h = nnx.leaky_relu(self.bn3(self.conv3(h), use_running_average=use_running_average), negative_slope=0.2)
        h = nnx.leaky_relu(self.bn4(self.conv4(h), use_running_average=use_running_average), negative_slope=0.2)
        return self.conv5(h).flatten()

class DCGAN(nnx.Module):
    def __init__(self, nz, ngf, nc, ndf, rngs: nnx.Rngs):
        self.netG = Generator(nz, ngf, nc, rngs)
        self.netD = Discriminator(nc, ndf, rngs)

# Init Models
rngs = nnx.Rngs(0)
model = DCGAN(NZ, NGF, NC, NDF, rngs=rngs)
optimizerG = nnx.Optimizer(model.netG, optax.adam(LR, b1=BETA1), wrt=nnx.Param)
optimizerD = nnx.Optimizer(model.netD, optax.adam(LR, b1=BETA1), wrt=nnx.Param)

def loss_bce(logits, labels):
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))

# Flax version check for API compatibility
def update_optimizer(optimizer, module, grads):
    try:
        optimizer.update(module, grads)
    except TypeError:
        optimizer.update(grads)

@nnx.jit
def train_step_D(model, optimizerD, real_x, noise):
    # G should be in train mode to update its own stats (though we discard gradients here)
    fake_x = model.netG(noise, train=True) 
    
    # Detach fake_x to treat it as constant for D
    fake_x = jax.lax.stop_gradient(fake_x)

    def loss_fn(model):
        # Separate passes for correct Batch Norm statistics (Real vs Fake distributions)
        real_logits = model.netD(real_x, train=True)
        fake_logits = model.netD(fake_x, train=True)
        
        errD_real = loss_bce(real_logits, jnp.ones_like(real_logits))
        errD_fake = loss_bce(fake_logits, jnp.zeros_like(fake_logits))
        errD = errD_real + errD_fake
        
        real_p = nnx.sigmoid(real_logits)
        fake_p = nnx.sigmoid(fake_logits)
        return errD, (real_p, fake_p)
        
    (loss, (real_p, fake_p)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    update_optimizer(optimizerD, model.netD, grads.netD)
    return loss, jnp.mean(real_p), jnp.mean(fake_p)

@nnx.jit
def train_step_G(model, optimizerG, noise):
    def loss_fn(model):
        fake_x = model.netG(noise, train=True)
        # Use train=True for D during G-update to ensure non-vanishing gradients.
        # D evaluates fakes using their own batch statistics, preventing mode mismatch.
        fake_logits = model.netD(fake_x, train=True)
        errG = loss_bce(fake_logits, jnp.ones_like(fake_logits))
        return errG, nnx.sigmoid(fake_logits)
    (loss, outD), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    update_optimizer(optimizerG, model.netG, grads.netG)
    return loss, jnp.mean(outD)

print("Starting Training Loop...")
fixed_latent = jax.random.normal(jax.random.PRNGKey(42), (NVIZ, NZ))
step_rng = jax.random.PRNGKey(0)

for epoch in range(NUM_EPOCH):
    start_t = timer.time()
    with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}") as tepoch:
        for batch_idx, (real_x, _) in enumerate(tepoch):
            step_rng, rng_d, rng_g = jax.random.split(step_rng, 3)
            noise_d = jax.random.normal(rng_d, (real_x.shape[0], NZ))
            errD, D_x, D_G_z1 = train_step_D(model, optimizerD, real_x, noise_d)
            noise_g = jax.random.normal(rng_g, (real_x.shape[0], NZ))
            errG, D_G_z2 = train_step_G(model, optimizerG, noise_g)
            
            if jnp.isnan(errD) or jnp.isnan(errG):
                print(f"\nNaN detected at batch {batch_idx}! LossD: {errD}, LossG: {errG}")
                sys.exit(1)

            if batch_idx % 10 == 0:
                tepoch.set_postfix(Loss_D=f"{errD:.4f}", Loss_G=f"{errG:.4f}", Dx=f"{D_x:.4f}", Dgz=f"{D_G_z2:.4f}")

    # if epoch % 10 == 0 or epoch == NUM_EPOCH - 1:
    fake_samples = model.netG(fixed_latent, train=False)
    grid = vu.set_grid(fake_samples, num_cells=NVIZ)
    plt.figure(figsize=(8, 8))
    # NHWC to HW(C) for grid? set_grid usually returns (C, H, W)
    plt.imshow(np.transpose(np.array(vu.normalize(grid, 0, 1)), (1, 2, 0)))
    plt.axis('off')
    plt.savefig(os.path.join(sample_dir, f'samples_epoch_{epoch+1}.png'))
    plt.close()
    
    mu.save_checkpoint(model.netG, epoch + 1, filedir=os.path.join(checkpoint_dir, "generator"))
    mu.save_checkpoint(model.netD, epoch + 1, filedir=os.path.join(checkpoint_dir, "discriminator"))
