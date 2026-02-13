import jax
import jax.numpy as jnp
from flax import nnx
import optax
from time import process_time
from tqdm import tqdm

def train(model, dataloader, optimizer, metrics, rngs=None):
    """
    Refactored training loop for Flax NNX.
    """
    model.train()
    train_time = 0.0
    losses = 0.
    
    # In NNX, we define a train_step function
    @nnx.jit
    def train_step(model, optimizer, metrics, batch):
        inputs, targets = batch
        def loss_fn(model):
            logits = model(inputs)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
            return loss, logits

        (loss, logits), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        optimizer.update(model, grads) # optimizer is stateful in NNX
        metrics.update(values=loss, logits=logits, labels=targets)
        return loss

    with tqdm(dataloader, unit="batch") as tepoch:
        for (X, y) in tepoch:
            start_t = process_time()
            
            # Convert to jnp
            X_jax = jnp.asarray(X)
            y_jax = jnp.asarray(y)
            
            loss = train_step(model, optimizer, metrics, (X_jax, y_jax))
            
            # Since we use jnp.array(X.numpy()), it might be slow for large datasets.
            # In a real scenario, we'd use a JAX-friendly data loader.
            
            losses += loss.item()

            elapsed_time = process_time() - start_t
            train_time += elapsed_time

            tepoch.set_postfix({
                'loss': f"{loss.item():.4f}",
                'time(secs)': f"{train_time:.2f}"
            })

    losses /= len(dataloader)
    return losses, train_time

def evaluate(model, dataloader, metrics):
    """
    Refactored evaluation loop for Flax NNX.
    """
    model.eval()
    eval_time = 0.0
    losses = 0.0
    
    @nnx.jit
    def eval_step(model, metrics, batch):
        inputs, targets = batch
        logits = model(inputs)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
        metrics.update(values=loss, logits=logits, labels=targets)
        return loss

    with tqdm(dataloader, unit="batch") as tepoch:
        for (X, y) in tepoch:
            start_t = process_time()
            
            X_jax = jnp.asarray(X)
            y_jax = jnp.asarray(y)
            
            loss = eval_step(model, metrics, (X_jax, y_jax))
            
            losses += loss.item()

            elapsed_time = process_time() - start_t
            eval_time += elapsed_time

            tepoch.set_postfix({
                'loss': f"{loss.item():.4f}",
                'time(secs)': f"{eval_time:.2f}"
            })
            
    losses /= len(dataloader)
    model.train()

    return losses, eval_time
