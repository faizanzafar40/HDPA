# Historical Document Processing Application (HDPA)

Binarize scanned historical documents with Sauvola's adaptive thresholding,
implemented on integral images for speed.

I built HDPA to clean up scans of old, stained, and unevenly lit documents.
Instead of applying one global threshold to the whole page, it computes a
separate threshold for every pixel from the mean and standard deviation of the
window around it (Sauvola's method), which is what lets it keep faint ink while
dropping background noise. I run the window statistics on OpenCV integral images
(a summed-area table and its squared counterpart), so each threshold is a handful
of array lookups rather than a full window scan — that is what makes it fast
enough to use on full-page scans.

## Tech stack

- Python 3.9+
- [NumPy](https://numpy.org/) for the array math
- [OpenCV](https://opencv.org/) (`opencv-python`) for image I/O and integral images

## Features

- Sauvola local adaptive thresholding for document binarization
- Constant-time window statistics via integral images, independent of window size
- Tunable window size, `k`, and dynamic range `R`
- A small command-line tool plus an importable `sauvola_binarize` function

## Installation

```bash
git clone https://github.com/faizanzafar40/HDPA.git
cd HDPA
pip install .
```

For development (tests and linting):

```bash
pip install -e ".[dev]"
```

## Usage

Command line:

```bash
hdpa path/to/scan.jpg -o binarized.png
```

Equivalently, without installing the console script:

```bash
python -m hdpa path/to/scan.jpg -o binarized.png
```

Options:

| Flag | Meaning | Default |
|------|---------|---------|
| `input` | path to the input image | — |
| `-o`, `--output` | path for the binarized image | `<input>_binarized.png` |
| `-w`, `--window` | side length of the local window, in pixels | `35` |
| `-k` | Sauvola `k` parameter | `0.1` |
| `-r` | dynamic range of the standard deviation | `128` |

From Python:

```python
import cv2
from hdpa import sauvola_binarize

gray = cv2.imread("scan.jpg", cv2.IMREAD_GRAYSCALE)
binary = sauvola_binarize(gray, window=35, k=0.1, r=128)
cv2.imwrite("binarized.png", binary)
```

## Sample output

Run the tool on any document scan to see the effect; the output is a black-on-white
image where ink is `0` and background is `255`.

> _Drop a before/after image pair here (e.g. `docs/before.png` and `docs/after.png`)
> to showcase the result._

## Running the tests

```bash
pip install -e ".[dev]"
pytest
```

The tests build small synthetic pages and check that ink is separated from paper,
that the output is strictly binary, and that the edge bands and input validation
behave as expected.

## Project structure

```
HDPA/
├── src/hdpa/
│   ├── __init__.py      # package exports
│   ├── sauvola.py       # Sauvola thresholding on integral images
│   ├── cli.py           # argparse command-line interface
│   └── __main__.py      # `python -m hdpa` entry point
├── tests/
│   └── test_sauvola.py  # synthetic-image characterization tests
├── pyproject.toml       # packaging, dependencies, tooling config
├── LICENSE
└── README.md
```

## Context / what I learned

This started as a machine-learning course project on document image analysis. The
thing I took away from it was how much an integral image buys you: once the
summed-area tables are built, the cost of a window's mean and variance no longer
depends on the window size, so a per-pixel local threshold becomes practical on a
whole page. Working through Sauvola's formula also made the trade-off between `k`
and the window size concrete — both control how aggressively faint strokes are
kept versus how much background texture leaks through.

## License

Released under the [MIT License](LICENSE).
