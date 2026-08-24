import numpy as np
from PIL import Image


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converts an image to uint8 if it is a float image.

    This is important for reducing the size of the image when sending it over the network.
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image

import jax
import jax.numpy as jnp

def resize_with_pad_jax(images, height, width, method="bilinear"):
    # images: [..., H, W, C]
    orig_shape = images.shape
    *batch_dims, H, W, C = orig_shape

    # Compute scale ratio (broadcast over batch dims)
    scale_h = height / H
    scale_w = width / W
    scale = jnp.minimum(scale_h, scale_w)

    new_h = jnp.floor(H * scale).astype(int)
    new_w = jnp.floor(W * scale).astype(int)

    # Resize (JAX handles batches)
    resized = jax.image.resize(
        images,
        (*batch_dims, new_h, new_w, C),
        method,
    )

    # Compute pad sizes
    pad_h = (height - new_h) // 2
    pad_w = (width - new_w) // 2

    pad_top = pad_h
    pad_bottom = height - new_h - pad_h
    pad_left = pad_w
    pad_right = width - new_w - pad_w

    padded = jnp.pad(resized,
                     (*[(0,0)]*len(batch_dims),
                      (pad_top, pad_bottom),
                      (pad_left, pad_right),
                      (0,0)),
                     mode='constant')

    return padded

