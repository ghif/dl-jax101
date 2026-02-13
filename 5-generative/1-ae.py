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
from skimage.util import random_noise
import optax

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
import viz_utils as vu
import model_utils as mu

# Define constants
MODEL_DIR = "../models"

BATCH_SIZE = 128
NUM_EPOCH = 1
IS_DENOISING = True # True: Denoising Autoencoder, False: Standard Autoencoder
NOISE_TYPE = "gaussian" # {"gaussian", "salt"}
NVIZ = 64
nrow = np.floor(np.sqrt(NVIZ)).astype(int)

DATASET = "mnist"
mname = "ae" if not IS_DENOISING else f"dae_{NOISE_TYPE}"

checkpoint_dir = os.path.join(MODEL_DIR, f"{mname}_{DATASET}")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    print(f'The new directory {checkpoint_dir} has been created')

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
    
    # Convert record dictionaries to tuples for multi-assignment
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

# Noise generator using skimage
def add_noise(x, noise_type="gaussian", seed=None):
    # x is in range [-1, 1], convert to [0, 1] for skimage
    x_01 = (x + 1.0) / 2.0
    
    # Map noise types
    mode = 'gaussian' if noise_type == 'gaussian' else 's&p' if noise_type == 'salt' else noise_type
    
    # Apply noise via skimage (operates on numpy arrays)
    x_noisy_01 = random_noise(np.array(x_01), mode=mode, rng=seed)
    
    # Convert back to [-1, 1]
    return jnp.array(x_noisy_01 * 2.0 - 1.0, dtype=jnp.float32)

# Define model architecture (NNX)
class Autoencoder(nnx.Module):
    def __init__(self, input_size, hidden_size, rngs: nnx.Rngs):
        self.encoder = nnx.Sequential(
            nnx.Linear(input_size, hidden_size, rngs=rngs),
            nnx.tanh,
        )
        self.decoder = nnx.Sequential(
            nnx.Linear(hidden_size, input_size, rngs=rngs),
            nnx.tanh,
        )

    def __call__(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y

# Set training configuration
input_size = 28 * 28
hidden_size = 128
rngs = nnx.Rngs(0)
model = Autoencoder(input_size, hidden_size, rngs=rngs)

optimizer = nnx.Optimizer(model, optax.adam(3e-4), wrt=nnx.Param)
metrics = nnx.MultiMetric(
    loss=nnx.metrics.Average('loss'),
)

@nnx.jit
def train_step(model, optimizer, metrics, x, xn):
    def loss_fn(model):
        xr = model(xn)
        loss = jnp.mean((xr - x)**2) # MSE Loss
        return loss
    
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    metrics.update(loss=loss)
    return loss

def get_fixed_samples(loader, n_viz):
    imgs, labels = [], []
    count = 0
    for batch_X, batch_y in loader:
        x, y = batch_X, batch_y
        imgs.append(x)
        labels.append(y)
        count += x.shape[0]
        if count >= n_viz: break
    return jnp.concatenate(imgs, axis=0)[:n_viz], jnp.concatenate(labels, axis=0)[:n_viz]

fixed_in_imgs, fixed_labels = get_fixed_samples(test_loader, NVIZ)

# Save real samples
grid_real = vu.set_grid(fixed_in_imgs, num_cells=NVIZ)
plt.figure(figsize=(10, 10))
plt.imshow(np.transpose(np.array(vu.normalize(grid_real, 0, 1)), (1, 2, 0)), cmap='gray')
plt.axis('off')
plt.savefig(os.path.join(checkpoint_dir, 'real_samples.jpg'), bbox_inches='tight')
plt.close()

# Train model
for epoch in range(NUM_EPOCH):
    start_t = timer.time()
    metrics.reset()
    
    with tqdm(train_loader, unit="batch") as tepoch:
        for batch_idx, (X, y) in enumerate(tepoch):
            # X is (B, 1, 28, 28)
            X_flat = X.reshape(X.shape[0], -1)
            
            if IS_DENOISING:
                # Use batch_idx as seed for noise reproducibility if needed
                noise_seed = epoch * 10000 + batch_idx
                Xn = add_noise(X, noise_type=NOISE_TYPE, seed=noise_seed)
                Xn_flat = Xn.reshape(Xn.shape[0], -1)
            else:
                Xn_flat = X_flat
            
            train_step(model, optimizer, metrics, jnp.array(X_flat), jnp.array(Xn_flat))
            
            if batch_idx % 100 == 0:
                tepoch.set_postfix(loss=metrics.compute()['loss'])

    elapsed_t = timer.time() - start_t
    epoch_loss = metrics.compute()['loss']
    print(f'Epoch [{epoch+1}/{NUM_EPOCH}], loss: {epoch_loss:.4f}, elapsed_t: {elapsed_t: 0.2f} secs')

    # Save model
    mu.save_checkpoint(model, epoch + 1, filedir=checkpoint_dir)
    
    # Visualization
    fixed_in_flatten = fixed_in_imgs.reshape(fixed_in_imgs.shape[0], -1)
    rec_imgs_flat = model(fixed_in_flatten)
    rec_imgs = rec_imgs_flat.reshape(-1, 1, 28, 28)
    
    grid_rec = vu.set_grid(rec_imgs, num_cells=NVIZ)
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(np.array(vu.normalize(grid_rec, 0, 1)), (1, 2, 0)), cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(checkpoint_dir, f'reconstructed_samples_{epoch}.jpg'), bbox_inches='tight')
    plt.close()

    if IS_DENOISING:
        # Static noise for visualization
        fixed_in_n = add_noise(fixed_in_imgs, noise_type=NOISE_TYPE, seed=42)
        
        grid_noisy = vu.set_grid(fixed_in_n, num_cells=NVIZ)
        plt.figure(figsize=(10, 10))
        plt.imshow(np.transpose(np.array(vu.normalize(grid_noisy, 0, 1)), (1, 2, 0)), cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(checkpoint_dir, 'noisy_samples.jpg'), bbox_inches='tight')
        plt.close()

    # t-SNE
    Z = model.encoder(fixed_in_flatten)
    tsne_path = os.path.join(checkpoint_dir, f"tsne_{epoch}.jpg")
    vu.plot_features_tsne(np.array(Z), np.array(fixed_labels), tsne_path)

print("Training finished!")
