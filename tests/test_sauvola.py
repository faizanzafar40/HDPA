import numpy as np
import pytest

from hdpa import sauvola_binarize


def _page_with_strokes(h=80, w=80):
    """A light 'paper' with a couple of thin dark 'ink' strokes."""
    img = np.full((h, w), 220, dtype=np.uint8)
    img[:, 40:42] = 30  # vertical stroke
    img[40:42, :] = 30  # horizontal stroke
    return img


def test_output_is_binary_with_same_shape():
    gray = _page_with_strokes()
    out = sauvola_binarize(gray, window=15)
    assert out.shape == gray.shape
    assert out.dtype == np.uint8
    assert set(np.unique(out)).issubset({0, 255})


def test_separates_ink_from_paper():
    gray = _page_with_strokes()
    out = sauvola_binarize(gray, window=15)
    assert out[60, 40] == 0  # on a stroke -> ink
    assert out[60, 60] == 255  # plain paper -> background
    assert 0 in out and 255 in out


def test_border_is_background():
    gray = _page_with_strokes()
    window = 15
    out = sauvola_binarize(gray, window=window)
    half = window // 2
    assert (out[:, :half] == 255).all()
    assert (out[:half, :] == 255).all()


def test_is_deterministic():
    gray = _page_with_strokes()
    assert np.array_equal(sauvola_binarize(gray, window=15), sauvola_binarize(gray, window=15))


def test_rejects_color_image():
    color = np.zeros((20, 20, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        sauvola_binarize(color)


def test_rejects_oversized_window():
    gray = _page_with_strokes(20, 20)
    with pytest.raises(ValueError):
        sauvola_binarize(gray, window=25)
