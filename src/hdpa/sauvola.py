"""Sauvola adaptive thresholding built on integral images.

I binarize scanned historical documents with Sauvola's local thresholding
method. Each pixel gets its own threshold derived from the mean and standard
deviation of the surrounding window, which copes far better with stained and
unevenly lit pages than a single global threshold.

Computing those window statistics directly would cost O(window^2) per pixel.
Instead I run them on OpenCV integral images: a summed-area table and its
squared counterpart let me read any window's sum (and sum of squares) from four
corner lookups, so each threshold is constant time regardless of window size.
That is what makes the method fast enough to be practical on full-page scans.
"""

from math import sqrt

import cv2
import numpy as np

# Sauvola's paper suggests a window around 15 px and k in [0.2, 0.5]. On the
# high-resolution document scans I worked with, a 35 px window with k = 0.1 gave
# the cleanest separation between ink and aged paper. R is the dynamic range of
# the standard deviation (128 for 8-bit images).
DEFAULT_WINDOW = 35
DEFAULT_K = 0.1
DEFAULT_R = 128


def sauvola_binarize(gray, window=DEFAULT_WINDOW, k=DEFAULT_K, r=DEFAULT_R):
    """Binarize a grayscale image with Sauvola's method.

    ``gray`` is a 2-D ``uint8`` array. Returns a ``uint8`` array of the same
    shape where 0 marks ink and 255 marks background.
    """
    if gray.ndim != 2:
        raise ValueError("sauvola_binarize expects a single-channel grayscale image")

    img_h, img_w = gray.shape
    if window > img_h or window > img_w:
        raise ValueError("window must not be larger than the image")

    # Summed-area tables. sum_img[y, x] holds the sum of every pixel above and to
    # the left of (x, y); sqsum_img holds the same for squared pixels. OpenCV
    # returns them padded to (H+1, W+1) so the corner lookups never fall off the
    # top or left edge. The sum table is integer, the squared table is float.
    sum_img, sqsum_img = cv2.integral2(gray)

    h, w = sqsum_img.shape  # padded integral dimensions, i.e. (H + 1, W + 1)
    half = window // 2
    area = window * window

    def box(table, y1, x1, y2, x2):
        """Sum of ``table`` over the rectangle (x1, y1)-(x2, y2) from its corners."""
        return table[y2, x2] + table[y1, x1] - table[y2, x1] - table[y1, x2]

    thresholds = np.empty((img_h, img_w), dtype=np.float64)

    # The window is anchored at its top-left corner (j - half, i - half); the
    # four cases clamp its bottom-right corner so it never runs past the right or
    # bottom edge. Mean and variance carry over along the single seam row/column
    # where the window sits exactly at the edge.
    mean_val = 0
    sq_val = 0.0
    for i in range(1, w):
        for j in range(1, h):
            if j > h - half and i > w - half:
                mean_val = box(sum_img, j - half, i - half, j, i) // area
                sq_val = box(sqsum_img, j - half, i - half, j, i) / area
            elif i > w - half and j < h - half:
                mean_val = box(sum_img, j - half, i - half, j + half, i) // area
                sq_val = box(sqsum_img, j - half, i - half, j + half, i) / area
            elif j > h - half and i < w - half:
                mean_val = box(sum_img, j - half, i - half, j, i + half) // area
                sq_val = box(sqsum_img, j - half, i - half, j, i + half) / area
            elif j < h - half and i < w - half:
                mean_val = box(sum_img, j - half, i - half, j + half, i + half) // area
                sq_val = box(sqsum_img, j - half, i - half, j + half, i + half) / area

            # Window mean is taken in integer arithmetic from the summed-area
            # table; the variance is the second moment minus the squared mean.
            variance = (sq_val - mean_val**2) / area
            std = sqrt(abs(variance))
            thresholds[j - 1, i - 1] = mean_val * (1 + k * ((std / r) - 1))

    # A pixel at or below its local Sauvola threshold is ink (0); anything
    # brighter is background (255).
    binary = np.where(gray <= thresholds, 0, 255).astype(np.uint8)

    # The window cannot be centred within half its width of the top/left edge, so
    # those bands have only partial statistics. I clear them to background rather
    # than trust a half-filled window.
    binary[:, :half] = 255
    binary[:half, :] = 255
    return binary
