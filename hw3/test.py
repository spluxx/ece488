"""ECE 488, Assignment 00.

Test script for histogram equalization.

Should produce a figure showing the original and contrast-adjusted images.

On the Python path should be a file <NETID>_histogram_equalize
defining the method <NETID>_histogram_equalize(im, ax) where
isinstance(im, numpy.ndarray) and isinstance(ax, matplotlib.axes.Axes)

Patrick Wang, 2019-01-15, 2020-01-16
"""

import importlib
import numpy as np
import matplotlib.pyplot as plt
import cv2


def test_histeq(module):
    """Test histeq function."""
    method = getattr(module, module.__name__)

    im = cv2.imread('poorContrast.jpg')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    im_equalized, func = method(im)

    fig, ax = plt.subplots(
        nrows=2, ncols=3, figsize=(10, 7),
    )

    ax[0, 0].imshow(
        im,
        cmap='gray',
        vmin=0, vmax=255
    )
    ax[0, 0].set_title('original image')

    bin_edges = np.linspace(-0.5, 255.5, 257)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    hist, _ = np.histogram(im, bin_edges)
    ax[1, 0].bar(bin_edges[:-1], hist, width=1.0)
    ax[1, 0].set_xlabel('intensity')
    ax[1, 0].set_ylabel('pixel count')
    ax[1, 0].set_title('original histogram')

    ax[0, 2].imshow(
        im_equalized,
        cmap='gray',
        vmin=0, vmax=255
    )
    ax[0, 2].set_title('histogram-equalized image')

    hist, _ = np.histogram(im_equalized, bin_edges)
    ax[1, 2].bar(bin_edges[:-1], hist, width=1.0)
    ax[1, 2].set_xlabel('intensity')
    ax[1, 2].set_ylabel('pixel count')
    ax[1, 2].set_title('equalized histogram')

    ax[0, 1].axis('off')

    ax[1, 1].plot(bin_centers, func(bin_centers))
    ax[1, 1].set_title('intensity transformation')
    ax[1, 1].set_xlabel('input intensity')
    ax[1, 1].set_ylabel('output intensity')

    plt.show()


if __name__ == "__main__":
    module = importlib.import_module('ih33_histogram_equalize')
    test_histeq(module)
