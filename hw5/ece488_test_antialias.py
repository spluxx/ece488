"""ECE 488, Assignment: Antialiasing.

Test script for antialiasing.

Should produce a figure showing the original, decimated, and downsampled images.

On the Python path should be a file <NETID>_downsample
defining the method <NETID>_downsample(im, factor) where
isinstance(im, numpy.ndarray) and isinstance(factor, int) and factor >= 1

Patrick Wang, 2020-02-04
"""

import importlib
import numpy as np
import matplotlib.pyplot as plt
import cv2


def test_antialias(module):
    """Test antialias function."""
    method = getattr(module, module.__name__)

    im = cv2.imread('striped_shirt.jpg')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    factor = 3
    im_decimated = im[::factor, ::factor]
    im_downsampled = method(im, factor)
    assert im_downsampled.shape == im_decimated.shape

    fig, ax = plt.subplots(
        nrows=2, ncols=2, figsize=(10, 7),
    )

    ax[0, 0].imshow(
        im,
        cmap='gray',
        vmin=0, vmax=255
    )
    ax[0, 0].set_title('original image')

    ax[1, 0].remove()

    ax[0, 1].imshow(
        im_decimated,
        cmap='gray',
        vmin=0, vmax=255
    )
    ax[0, 1].set_title('decimated image')

    ax[1, 1].imshow(
        im_downsampled,
        cmap='gray',
        vmin=0, vmax=255
    )
    ax[1, 1].set_title('downsampled image')

    plt.show()


if __name__ == "__main__":
    module = importlib.import_module('ih33_downsample')
    test_antialias(module)
