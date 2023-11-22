""" Tiling utilities for images. 

CHW format assumed for images.
Tiling done in HWC, on H and W.
The input can be cropped or padded to a multiple of tile_size.
"""
import numpy as np

# import decorator
from functools import wraps

def check_numpy_input(func):
    """ Decorator to check if the first input is a numpy array. """
    @wraps(func)
    def wrapper(x, *args, **kwargs):
        if not isinstance(x, np.ndarray):
            raise TypeError(f'Input must be a numpy array, got {type(x)}')
        return func(x, *args, **kwargs)
    return wrapper

@check_numpy_input
def tile_loop_hwc(x, tile_size):
    """ Tile an image (N*D, N*D, C) -> (N*N, D, D, C). """
    n_tiles = x.shape[0] // tile_size
    tiles = []
    for i in range(0, n_tiles):
        ilow = i * tile_size
        ihigh = (i + 1) * tile_size
        for j in range(n_tiles):
            jlow = j * tile_size
            jhigh = (j + 1) * tile_size
            tiles.append(x[ilow:ihigh, jlow:jhigh, :])
    return tiles

@check_numpy_input
def tile_hwc(x, tile_size):
    """ Tile an image (N*D, N*D, C) -> (N*N, D, D, C). """
    n_tiles = x.shape[0] // tile_size
    x = x.reshape(n_tiles, tile_size, n_tiles, tile_size, -1)  # -> (N, D, N, D, C)
    x = x.transpose(0, 2, 1, 3, 4)  # -> (N, N, D, D, C)
    x = x.reshape(n_tiles * n_tiles, tile_size, tile_size, -1)  # -> (N*N, D, D, C)
    return x

@check_numpy_input
def tile(x, tile_size):
    """ Tile an image (C, N*D, N*D) -> (N*N, C, D, D). """
    x = x.transpose(1, 2, 0)  # -> (N*D, N*D, C)
    x = tile_hwc(x, tile_size)  # alternatively, tile_loop(x)
    x = x.transpose(0, 3, 1, 2)  # -> (N*N, C, D, D)
    return x

@check_numpy_input
def untile_hwc(x, n_tiles, tile_size):
    """ Untile an image (N*N, D, D, C) -> (N*D, N*D, C). """
    x = x.reshape(n_tiles, n_tiles, tile_size, tile_size, -1)  # -> (N, N, D, D, C)
    x = x.transpose(0, 2, 1, 3, 4)  # -> (N, D, N, D, C)
    x = x.reshape(n_tiles * tile_size, n_tiles * tile_size, -1)  # -> (N*D, N*D, C)
    return x

@check_numpy_input
def untile(x, n_tiles, tile_size, hwc=False):
    """ Untile an image (N*N, C, D, D) -> (C, N*D, N*D). """
    x = x.transpose(0, 2, 3, 1)  # -> (N*N, D, D, C)
    x = untile_hwc(x, n_tiles, tile_size)
    if not hwc:
        x = x.transpose(2, 0, 1)  # -> (C, N*D, N*D)
    return x

@check_numpy_input
def crop(x, tile_size):
    """ Crop an image to a multiple of tile_size (CHW). """
    n_tiles = x.shape[1] // tile_size  # assume input larger than tile_size
    dim_crop = n_tiles * tile_size
    x = x[:, :dim_crop, :dim_crop]
    return x

@check_numpy_input
def pad(x, tile_size):
    """ Pad an image to a multiple of tile_size (CHW). """
    n_tiles = x.shape[1] // tile_size + (x.shape[1] % tile_size > 0)
    dim_pad = n_tiles * tile_size - x.shape[1]
    x = np.pad(x, ((0, 0), (0, dim_pad), (0, dim_pad)), mode='constant')
    return x

@check_numpy_input
def pad_with_mask(x, tile_size):
    """ Pad an image to a multiple of tile_size (CHW), and append a mask channel. """
    n_tiles = x.shape[1] // tile_size + (x.shape[1] % tile_size > 0)
    mask_valid = np.zeros_like(x[0:1])
    mask_valid[:n_tiles * tile_size, :n_tiles * tile_size] = 1
    x = np.concatenate([x, mask_valid], axis=0)
    dim_pad = n_tiles * tile_size - x.shape[1]
    x = np.pad(x, ((0, 0), (0, dim_pad), (0, dim_pad)), mode='constant')
    return x