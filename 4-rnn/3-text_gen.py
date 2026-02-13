import sys, os
script_dir = os.path.dirname(os.path.abspath(__file__))
jax_dir = os.path.dirname(script_dir)
sys.path.append(jax_dir)

from time import process_time
import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp
from flax import nnx
import optax

# Force CPU to avoid Metal issues on some Macs (can still be slow or unstable)
jax.config.update("jax_platform_name", "cpu")

import seq_processor as sp
import model_utils as mu

def loss_fn(model, xb, yb):
    logits = model(xb)
    # logits shape: (B, T, vocab_size)
    # yb shape: (B, T)
    B, T, C = logits.shape
    logits_flat = logits.reshape(B * T, C)
    targets_flat = yb.reshape(B * T)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits_flat, targets_flat).mean()
    return loss

@nnx.jit
def train_step(model, optimizer, xb, yb):
    loss, grads = nnx.value_and_grad(loss_fn)(model, xb, yb)
    optimizer.update(model, grads)
    return loss

@nnx.jit(static_argnums=(2, 3, 4))
def estimate_loss(model, data, eval_iters=10, batch_size=32, seq_len=64, key=None):
    model.eval()
    losses = []
    for k in range(eval_iters):
        # We need to be careful with key management if we want reproducibility
        # But for estimation, any key is fine
        curr_key = jax.random.fold_in(key, k) if key is not None else jax.random.PRNGKey(k)
        xb, yb = sp.get_batch(data, batch_size=batch_size, block_size=seq_len, key=curr_key)
        loss = loss_fn(model, xb, yb)
        losses.append(loss)
    
    avg_loss = jnp.mean(jnp.array(losses))
    model.train()
    return avg_loss

# Constants
model_dir = "../models"
os.makedirs(model_dir, exist_ok=True)

max_iters = 5000
data_dir = "../data"
data_path = os.path.join(data_dir, "chairilanwar.txt")
seq_len = 256
n_embed = 384
n_hidden = 512
batch_size = 64
eval_interval = 10 # Increased for speed

# Load sequence data
chproc = sp.CharProcessor(data_path)
data = jnp.array(chproc.encode(chproc.text), dtype=jnp.int32)

# Construct training data
train_data = data

# Initialize model
rngs = nnx.Rngs(1337)
model = mu.SimpleBigram(
    chproc.vocab_size,
    seq_len,
    n_embed,
    n_hidden,
    num_layers=1,
    rngs=rngs
)

optimizer = nnx.Optimizer(model, optax.adamw(3e-4), wrt=nnx.Param)

print("Starting training...")

key = jax.random.PRNGKey(0)

for step in range(max_iters):
    key, subkey = jax.random.split(key)
    xb, yb = sp.get_batch(train_data, 
        batch_size=batch_size, 
        block_size=seq_len,
        key=subkey
    )

    start_t = process_time()
    loss = train_step(model, optimizer, xb, yb)
    elapsed_t = process_time() - start_t

    if step % eval_interval == 0 or step == max_iters - 1:
        key, subkey = jax.random.split(key)
        train_loss = estimate_loss(model, train_data, eval_iters=10, batch_size=batch_size, seq_len=seq_len, key=subkey)

        print(f"[Step-{step+1}/{max_iters} - training time: {elapsed_t:.3f} secs]: train loss: {train_loss:.4f}")

        # Generate some text
        # Start with a zero token
        idx = jnp.zeros((1, 1), dtype=jnp.int32)
        # generate expects rngs
        pred_idx = model.generate(idx, 100, rngs=rngs)
        pred_str = chproc.decode(np.array(pred_idx[0]))
        print(f"Generated text:\n{pred_str}\n")

print("Training finished.")
