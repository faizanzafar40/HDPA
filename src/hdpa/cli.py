"""Command-line entry point for binarizing a document image."""

import argparse
import os

import cv2

from .sauvola import DEFAULT_K, DEFAULT_R, DEFAULT_WINDOW, sauvola_binarize


def _default_output(input_path):
    """Place the result next to the input as ``<name>_binarized.png``."""
    stem, _ = os.path.splitext(input_path)
    return f"{stem}_binarized.png"


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="hdpa",
        description="Binarize a scanned document with Sauvola's adaptive thresholding.",
    )
    parser.add_argument("input", help="path to the input image")
    parser.add_argument(
        "-o",
        "--output",
        help="path for the binarized image (defaults to <input>_binarized.png)",
    )
    parser.add_argument(
        "-w",
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help="side length of the local window in pixels (default: %(default)s)",
    )
    parser.add_argument(
        "-k",
        type=float,
        default=DEFAULT_K,
        help="Sauvola k parameter (default: %(default)s)",
    )
    parser.add_argument(
        "-r",
        type=float,
        default=DEFAULT_R,
        help="dynamic range of the standard deviation (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    image = cv2.imread(args.input)
    if image is None:
        parser.error(f"could not read image: {args.input}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = sauvola_binarize(gray, window=args.window, k=args.k, r=args.r)

    output = args.output or _default_output(args.input)
    if not cv2.imwrite(output, binary):
        parser.error(f"could not write image: {output}")
    print(f"Wrote {output}")
    return 0
