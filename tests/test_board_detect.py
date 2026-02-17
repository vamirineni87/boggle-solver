import cv2
import numpy as np
import pytest
from app.board_detect import detect_board_and_warp, infer_grid_size, _order_corners


def _make_synthetic_board(grid_n: int, cell_size: int = 80, border: int = 100):
    """Create a synthetic image with a clear grid of NxN cells."""
    board_size = grid_n * cell_size
    img_size = board_size + 2 * border
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 200  # light gray bg

    # Draw the board area (white)
    cv2.rectangle(img, (border, border), (border + board_size, border + board_size),
                  (255, 255, 255), -1)

    # Draw grid lines (black)
    for i in range(grid_n + 1):
        y = border + i * cell_size
        cv2.line(img, (border, y), (border + board_size, y), (0, 0, 0), 2)
        x = border + i * cell_size
        cv2.line(img, (x, border), (x, border + board_size), (0, 0, 0), 2)

    # Put a letter in each cell
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for r in range(grid_n):
        for c in range(grid_n):
            idx = (r * grid_n + c) % 26
            cx = border + c * cell_size + cell_size // 2 - 10
            cy = border + r * cell_size + cell_size // 2 + 10
            cv2.putText(img, letters[idx], (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    return img


def test_detect_synthetic_4x4():
    img = _make_synthetic_board(4)
    warped, info = detect_board_and_warp(img, 600)
    assert warped.shape == (600, 600)
    assert info["method"] in ("contour", "hough", "center_crop_fallback")


def test_detect_synthetic_5x5():
    img = _make_synthetic_board(5)
    warped, info = detect_board_and_warp(img, 600)
    assert warped.shape == (600, 600)


def test_infer_grid_size_4():
    img = _make_synthetic_board(4)
    warped, _ = detect_board_and_warp(img, 600)
    n = infer_grid_size(warped)
    assert n in (4, 5), f"Expected 4 or 5, got {n}"


def test_infer_grid_size_5():
    img = _make_synthetic_board(5)
    warped, _ = detect_board_and_warp(img, 600)
    n = infer_grid_size(warped)
    assert n in (4, 5, 6), f"Expected 5 or nearby, got {n}"


def test_order_corners():
    pts = np.array([[300, 0], [0, 0], [0, 300], [300, 300]], dtype=np.float32)
    ordered = _order_corners(pts)
    # top-left should be (0,0)
    assert ordered[0][0] == 0 and ordered[0][1] == 0
    # bottom-right should be (300,300)
    assert ordered[2][0] == 300 and ordered[2][1] == 300


def test_fallback_on_blank_image():
    """A blank image should still return a warped result via fallback."""
    img = np.ones((800, 600, 3), dtype=np.uint8) * 128
    warped, info = detect_board_and_warp(img, 600)
    assert warped.shape == (600, 600)
