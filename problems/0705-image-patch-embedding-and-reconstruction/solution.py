import numpy as np

def patch_embed_reconstruct(image: np.ndarray, patch_size: int):
    """
    Split image into patches, flatten to patch embedding matrix,
    then reconstruct the original image.

    Args:
        image: 2D numpy array of shape (H, W)
        patch_size: int, side length of each square patch

    Returns:
        Reconstructed image as a nested list of shape (H, W),
        or -1 if dimensions are invalid.
    """
    H, W = img.shape
    return img if H % patch_size ==0 and W % patch_size == 0 else -1