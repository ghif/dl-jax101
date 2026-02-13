import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
# For make_grid equivalent, we might still use torchvision or implement a JAX version
# If we want to stay pure JAX, we can implement a simple grid function.

try:
    from sklearn.manifold import TSNE
    import seaborn as sns
    import pandas as pd
    _HAS_SKLEARN = True
except (ImportError, ValueError):
    print("Warning: sklearn/pandas not found or incompatible. t-SNE plotting will be disabled.")
    _HAS_SKLEARN = False
    TSNE = None
    sns = None
    pd = None

import os

plt.rcParams["savefig.bbox"] = 'tight'

def set_grid(D, num_cells=1):
    """
    Produce a grid of images.
    Args: 
        D (Array): (n, d1, d2), (n, c, d1, d2), or (n, d1, d2, c) collection of image arrays
    Return:
        grid (Array): The resulting grid in (c, H, W) format for consistency with show()
    """
    # 1. Ensure 4D (n, c, h, w)
    if len(D.shape) == 3:
        n, h, w = D.shape
        D = D[:, jnp.newaxis, :, :]
    
    # Detect channel ordering: (n, c, h, w) vs (n, h, w, c)
    if D.shape[1] in [1, 3] and D.shape[3] not in [1, 3]:
        # (n, c, h, w) - channel first
        n, c, d1, d2 = D.shape
    elif D.shape[3] in [1, 3]:
        # (n, h, w, c) - channel last -> convert to (n, c, h, w)
        D = jnp.transpose(D, (0, 3, 1, 2))
        n, c, d1, d2 = D.shape
    else:
        # Fallback/Unknown, assume (n, c, h, w)
        n, c, d1, d2 = D.shape
    
    grid_size = int(jnp.ceil(jnp.sqrt(num_cells)))
    grid = jnp.zeros((c, grid_size * d1, grid_size * d2))
    
    for i in range(num_cells):
        if i >= n: break
        r = i // grid_size
        col = i % grid_size
        grid = grid.at[:, r*d1:(r+1)*d1, col*d2:(col+1)*d2].set(D[i])
        
    return grid

def show(imgs, cmap=plt.cm.gray):
    """
    Args:
        imgs (Array): images in the form of jnp.array (C, H, W) or (H, W, C)
    """
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
    
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        # Ensure 3D (H, W, C)
        if len(img.shape) == 2:
            img = img[:, :, jnp.newaxis]
        
        # Determine channel ordering
        if img.shape[0] in [1, 3] and img.shape[2] not in [1, 3]:
            # CHW -> HWC
            img = jnp.transpose(img, (1, 2, 0))
        
        img_np = np.array(img)
        img_np = normalize(img_np, new_min=0, new_max=1)
        
        if img_np.shape[2] == 1:
            img_np = img_np.squeeze()
            
        axs[0, i].imshow(img_np,
            interpolation="nearest",
            cmap=cmap,
        )
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def plot_mnist(X, y, rows, cols, cmap=plt.cm.gray):
    """
    Plot a grid of MNIST-like images.
    Args:
        X (Array): batch of images (N, C, H, W) or (N, H, W, C)
        y (Array): labels (N,)
        rows (int): number of rows in grid
        cols (int): number of columns in grid
    """
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    for i, ax in enumerate(axes.flat):
        if i < len(X):
            img = X[i]
            # Ensure 3D (H, W, C)
            if len(img.shape) == 2:
                img = img[:, :, jnp.newaxis]
            
            # Determine channel ordering
            if img.shape[0] in [1, 3] and img.shape[2] not in [1, 3]:
                # CHW -> HWC
                img = jnp.transpose(img, (1, 2, 0))
            
            if img.shape[2] == 1:
                img = img.squeeze()
            
            ax.imshow(img, cmap=cmap)
            ax.set_title(f"Label: {y[i]}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()
        

def show_components(images, image_shape, n_row=2, n_col=3, cmap=plt.cm.gray):
    fig, axs = plt.subplots(
        nrows=n_row,
        ncols=n_col,
        figsize=(2.0 * n_col, 2.3 * n_row),
        facecolor="white",
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.02, hspace=0, wspace=0)
    fig.set_edgecolor("black")
    for ax, vec in zip(axs.flat, images):
        vmax = max(vec.max(), -vec.min())
        im = ax.imshow(
            np.array(vec).reshape(image_shape),
            cmap=cmap,
            interpolation="nearest",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.axis("off")

    fig.colorbar(im, ax=axs, orientation="horizontal", shrink=0.99, aspect=40, pad=0.01)
    plt.show()
    
    
def normalize(x, new_min=0, new_max=255):
    old_min = np.min(x)
    old_max = np.max(x)
    xn = (x - old_min) * ((new_max - new_min) / (old_max - old_min)) + new_min
    return xn


def plot_features_tsne(feat, labels, save_path, show=False):
    if not _HAS_SKLEARN:
        print("Skipping t-SNE plot due to missing/incompatible sklearn/pandas.")
        return

    print('generating t-SNE plot...')
    tsne = TSNE(random_state=0)
    tsne_output = tsne.fit_transform(np.array(feat))

    df = pd.DataFrame(tsne_output, columns=['x', 'y'])
    df['targets'] = np.array(labels)

    plt.rcParams['figure.figsize'] = 10, 10
    sns.scatterplot(
        x='x', y='y',
        hue='targets',
        palette=sns.color_palette("hls", 10),
        data=df,
        marker='o',
        legend="full",
        alpha=0.5
    )

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')

    plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.clf()
