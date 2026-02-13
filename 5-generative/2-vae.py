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
MODEL_DIR = "../models"
BATCH_SIZE = 128
NUM_EPOCH = 50
HIDDEN_SIZE = 40  # As in reference 2-vae.py
BETA = 1
NVIZ = 64
nrow = np.floor(np.sqrt(NVIZ)).astype(int)

DATASET = "mnist"
MNAME = "vae"
DAY = "12feb" # Current date for folder naming consistency with reference

checkpoint_dir = os.path.join(MODEL_DIR, f"{MNAME}_{DATASET}_z{HIDDEN_SIZE}")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    print(f'The new directory {checkpoint_dir} has been created')

sample_dir = os.path.join(checkpoint_dir, f"samples_{DAY}")
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
        # Reshape to (1, 28, 28) and normalize to [-1, 1] as in 2-vae.py
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

# Define Variational Autoencoder architecture
class VariationalAutoencoder(nnx.Module):
    def __init__(self, input_size, hidden_size, rngs: nnx.Rngs):
        self.hidden_size = hidden_size
        
        # Encoder: Linear -> Tanh -> Linear (outputs mu and logvar)
        self.encoder = nnx.Sequential(
            nnx.Linear(input_size, hidden_size ** 2, rngs=rngs),
            nnx.tanh,
            nnx.Linear(hidden_size ** 2, hidden_size * 2, rngs=rngs),
        )

        # Decoder: Linear -> Tanh -> Linear -> Tanh
        self.decoder = nnx.Sequential(
            nnx.Linear(hidden_size, hidden_size ** 2, rngs=rngs),
            nnx.tanh,
            nnx.Linear(hidden_size ** 2, input_size, rngs=rngs),
            nnx.tanh,
        )

    def reparameterise(self, mu, logvar, rng):
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(rng, mu.shape)
        return mu + eps * std

    def encode(self, x, rng=None, train=True):
        mu_logvar = self.encoder(x)
        # Reshape to (batch, 2, hidden_size)
        mu_logvar = mu_logvar.reshape(-1, 2, self.hidden_size)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        
        if train and rng is not None:
            z = self.reparameterise(mu, logvar, rng)
        else:
            z = mu
        return z, mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def __call__(self, x, rng=None, train=True):
        z, mu, logvar = self.encode(x, rng, train)
        xr = self.decode(z)
        return xr, mu, logvar

# Training configuration
input_size = 28 * 28
rngs = nnx.Rngs(0)
model = VariationalAutoencoder(input_size, HIDDEN_SIZE, rngs=rngs)

optimizer = nnx.Optimizer(model, optax.adam(3e-4), wrt=nnx.Param)
metrics = nnx.MultiMetric(
    total_loss=nnx.metrics.Average('total_loss'),
    rec_loss=nnx.metrics.Average('rec_loss'),
    kl_loss=nnx.metrics.Average('kl_loss'),
)

@nnx.jit
def train_step(model, optimizer, metrics, x, rng):
    def loss_fn(model):
        xr, mu, logvar = model(x, rng=rng, train=True)
        
        # Reconstruction loss (MSE sum over pixels, mean over batch)
        # Note: PyTorch reference uses reduction="sum". 
        # Here we follow that by summing over dx then taking average over batch.
        rec_loss = jnp.sum((xr - x)**2, axis=1) 
        
        # KL Divergence
        # 0.5 * sum(mu^2 + exp(logvar) - logvar - 1)
        kl_loss = 0.5 * jnp.sum(jnp.square(mu) + jnp.exp(logvar) - logvar - 1, axis=1)
        
        total_loss = jnp.mean(rec_loss + BETA * kl_loss)
        return total_loss, (jnp.mean(rec_loss), jnp.mean(kl_loss))
    
    (loss, (rec_l, kl_l)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    metrics.update(total_loss=loss, rec_loss=rec_l, kl_loss=kl_l)
    return loss

def get_fixed_samples(loader, n_viz):
    imgs, labels = [], []
    count = 0
    for batch_X, batch_y in loader:
        imgs.append(batch_X)
        labels.append(batch_y)
        count += batch_X.shape[0]
        if count >= n_viz: break
    return jnp.concatenate(imgs, axis=0)[:n_viz], jnp.concatenate(labels, axis=0)[:n_viz]

fixed_in_imgs, fixed_labels = get_fixed_samples(test_loader, NVIZ)

# Save real samples
grid_real = vu.set_grid(fixed_in_imgs, num_cells=NVIZ)
plt.figure(figsize=(10, 10))
plt.imshow(np.transpose(np.array(vu.normalize(grid_real, 0, 1)), (1, 2, 0)), cmap='gray')
plt.axis('off')
plt.savefig(os.path.join(sample_dir, 'real_samples.jpg'), bbox_inches='tight')
plt.close()

fixed_latent = jax.random.normal(jax.random.PRNGKey(42), (64, HIDDEN_SIZE))

# Train model
step_rng = jax.random.PRNGKey(0)

for epoch in range(NUM_EPOCH):
    start_t = timer.time()
    metrics.reset()
    
    with tqdm(train_loader, unit="batch") as tepoch:
        for batch_idx, (X, y) in enumerate(tepoch):
            X_flat = X.reshape(X.shape[0], -1)
            
            # Update RNG for each step
            step_rng, sub_rng = jax.random.split(step_rng)
            
            train_step(model, optimizer, metrics, jnp.array(X_flat), sub_rng)
            
            if batch_idx % 100 == 0:
                m_results = metrics.compute()
                tepoch.set_postfix(
                    total_loss=m_results['total_loss'], 
                    rec_loss=m_results['rec_loss'], 
                    kl_loss=m_results['kl_loss']
                )

    elapsed_t = timer.time() - start_t
    m_results = metrics.compute()
    print(f'Epoch [{epoch+1}/{NUM_EPOCH}], '
          f'total_loss: {m_results["total_loss"]:.4f}, '
          f'rec_loss: {m_results["rec_loss"]:.4f}, '
          f'kl_loss: {m_results["kl_loss"]:.4f}, '
          f'elapsed_t: {elapsed_t: 0.2f} secs')

    if epoch % 10 != 0:
        continue

    # Save checkpoint
    mu.save_checkpoint(model, epoch + 1, filedir=checkpoint_dir)
    
    # Visualization: Reconstruction
    fixed_in_flatten = fixed_in_imgs.reshape(fixed_in_imgs.shape[0], -1)
    # Use deterministic encoding for reconstruction visualization
    rec_imgs_flat, _, _ = model(fixed_in_flatten, train=False)
    rec_imgs = rec_imgs_flat.reshape(-1, 1, 28, 28)
    
    grid_rec = vu.set_grid(rec_imgs, num_cells=NVIZ)
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(np.array(vu.normalize(grid_rec, 0, 1)), (1, 2, 0)), cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(sample_dir, f'reconstructed_samples_{epoch}.jpg'), bbox_inches='tight')
    plt.close()

    # Visualization: Random Generation
    gen_imgs_flat = model.decode(fixed_latent)
    gen_imgs = gen_imgs_flat.reshape(-1, 1, 28, 28)
    grid_gen = vu.set_grid(gen_imgs, num_cells=64)
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(np.array(vu.normalize(grid_gen, 0, 1)), (1, 2, 0)), cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(sample_dir, f'fixed_rec_{epoch}.jpg'), bbox_inches='tight')
    plt.close()

    # t-SNE
    _, mu_feat, _ = model.encode(fixed_in_flatten, train=False)
    tsne_path = os.path.join(sample_dir, f"tsne_{epoch}.jpg")
    vu.plot_features_tsne(np.array(mu_feat), np.array(fixed_labels), tsne_path)

print("Training finished!")
