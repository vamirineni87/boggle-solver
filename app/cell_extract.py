import cv2
import numpy as np


def split_cells(board_gray: np.ndarray, grid_size: int, inset: float = 0.15) -> list[np.ndarray]:
    """Split a warped square board image into individual cell images."""
    h, w = board_gray.shape
    cell_h = h / grid_size
    cell_w = w / grid_size
    inset_y = int(cell_h * inset)
    inset_x = int(cell_w * inset)

    cells = []
    for r in range(grid_size):
        for c in range(grid_size):
            y1 = int(r * cell_h) + inset_y
            y2 = int((r + 1) * cell_h) - inset_y
            x1 = int(c * cell_w) + inset_x
            x2 = int((c + 1) * cell_w) - inset_x
            cell = board_gray[y1:y2, x1:x2]
            cells.append(cell)

    return cells


def preprocess_cell(cell_gray: np.ndarray, target_size: int = 64) -> np.ndarray:
    """Preprocess a cell image for OCR: resize, CLAHE, binarize."""
    # Resize
    resized = cv2.resize(cell_gray, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(resized)

    # Otsu binarization
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure letter is dark on light background (most common)
    # If more than half the pixels are dark, invert
    if np.mean(binary) < 128:
        binary = cv2.bitwise_not(binary)

    return binary


def make_cell_montage(cells: list[np.ndarray], grid_size: int, cell_display_size: int = 80) -> np.ndarray:
    """Create a montage image of all cells for debugging."""
    rows = []
    for r in range(grid_size):
        row_cells = []
        for c in range(grid_size):
            idx = r * grid_size + c
            cell = cells[idx]
            resized = cv2.resize(cell, (cell_display_size, cell_display_size))
            # Add border
            bordered = cv2.copyMakeBorder(resized, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=128)
            row_cells.append(bordered)
        rows.append(np.hstack(row_cells))
    return np.vstack(rows)
