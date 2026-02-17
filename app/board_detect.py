import cv2
import numpy as np
import logging

logger = logging.getLogger("boggle")


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]   # top-left: smallest x+y
    rect[2] = pts[np.argmax(s)]   # bottom-right: largest x+y
    rect[1] = pts[np.argmin(d)]   # top-right: smallest x-y
    rect[3] = pts[np.argmax(d)]   # bottom-left: largest x-y
    return rect


def _find_board_contour(image_bgr: np.ndarray):
    """Find the largest roughly-square contour in the image."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Try adaptive threshold first
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Also try Canny
    edges = cv2.Canny(blurred, 50, 150)

    best_contour = None
    best_area = 0
    img_area = image_bgr.shape[0] * image_bgr.shape[1]

    for source in [thresh, edges]:
        # Dilate to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(source, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < img_area * 0.05:  # too small
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
                ordered = _order_corners(pts)
                w = np.linalg.norm(ordered[1] - ordered[0])
                h = np.linalg.norm(ordered[3] - ordered[0])
                if h == 0:
                    continue
                aspect = w / h
                if 0.6 < aspect < 1.6 and area > best_area:
                    best_area = area
                    best_contour = ordered

    return best_contour


def _find_board_hough(image_bgr: np.ndarray):
    """Fallback: use Hough lines to find the board rectangle."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    if lines is None:
        return None

    h_lines = []
    v_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1))
        if angle < 0.3:  # roughly horizontal
            h_lines.append((y1 + y2) / 2)
        elif angle > 1.27:  # roughly vertical
            v_lines.append((x1 + x2) / 2)

    if len(h_lines) < 2 or len(v_lines) < 2:
        return None

    h_lines.sort()
    v_lines.sort()

    top = h_lines[0]
    bottom = h_lines[-1]
    left = v_lines[0]
    right = v_lines[-1]

    corners = np.array([
        [left, top],
        [right, top],
        [right, bottom],
        [left, bottom],
    ], dtype=np.float32)

    return corners


def _find_board_from_cells(image_bgr: np.ndarray):
    """Fallback: find individual cell contours and compute their collective bounding box."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    img_h, img_w = gray.shape
    img_area = img_h * img_w

    # Find cell-like contours (each cell is ~1-4% of image area)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter for square-ish contours in the cell size range (1-5% of image)
    cell_rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if img_area * 0.008 < area < img_area * 0.06:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / h if h > 0 else 0
            if 0.6 < aspect < 1.6:
                cell_rects.append((x, y, x + w, y + h))

    # Need at least 16 cell-like contours to be a valid grid
    if len(cell_rects) < 12:
        return None

    # Compute collective bounding box with small padding
    min_x = min(r[0] for r in cell_rects)
    min_y = min(r[1] for r in cell_rects)
    max_x = max(r[2] for r in cell_rects)
    max_y = max(r[3] for r in cell_rects)

    # Small padding to include cell edges
    pad = int(min(max_x - min_x, max_y - min_y) * 0.02)
    min_x = max(0, min_x - pad)
    min_y = max(0, min_y - pad)
    max_x = min(img_w, max_x + pad)
    max_y = min(img_h, max_y + pad)

    corners = np.array([
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y],
    ], dtype=np.float32)

    logger.info("Cell grouping found %d cells, bbox=(%d,%d)-(%d,%d)",
                len(cell_rects), min_x, min_y, max_x, max_y)
    return corners


def detect_board_and_warp(image_bgr: np.ndarray, warp_size: int = 600):
    """Detect the Boggle board and return a perspective-warped square grayscale image."""
    debug_info = {}

    corners = _find_board_contour(image_bgr)
    debug_info["method"] = "contour"

    if corners is None:
        corners = _find_board_from_cells(image_bgr)
        debug_info["method"] = "cell_grouping"

    if corners is None:
        corners = _find_board_hough(image_bgr)
        debug_info["method"] = "hough"

    if corners is None:
        # Last resort: use center crop assuming board is roughly centered
        h, w = image_bgr.shape[:2]
        margin_x = int(w * 0.1)
        margin_y = int(h * 0.2)
        corners = np.array([
            [margin_x, margin_y],
            [w - margin_x, margin_y],
            [w - margin_x, h - margin_y],
            [margin_x, h - margin_y],
        ], dtype=np.float32)
        debug_info["method"] = "center_crop_fallback"
        logger.warning("Board detection fell back to center crop")

    debug_info["corners"] = corners.tolist()

    dst = np.array([
        [0, 0],
        [warp_size - 1, 0],
        [warp_size - 1, warp_size - 1],
        [0, warp_size - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(image_bgr, M, (warp_size, warp_size))
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    return warped_gray, debug_info


def infer_grid_size(board_gray: np.ndarray) -> int:
    """Infer grid size (4, 5, or 6) by counting cell-like contours in the warped board."""
    h, w = board_gray.shape

    # Primary: count cell contours
    # Cells are dark rounded rectangles on a lighter background
    _, thresh = cv2.threshold(board_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Invert so dark cells become white (foreground)
    cell_mask = cv2.bitwise_not(thresh)

    # Erode to separate cells that might be touching at rounded corners
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cell_mask = cv2.erode(cell_mask, kernel, iterations=2)

    contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter for cell-sized, roughly-square contours
    min_area = (w / 8) * (h / 8)
    max_area = (w / 3) * (h / 3)

    cell_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / ch if ch > 0 else 0
            if 0.5 < aspect < 2.0:
                cell_count += 1

    # Map to nearest grid size: 16=4x4, 25=5x5, 36=6x6
    candidates = {16: 4, 25: 5, 36: 6}
    best_n = min(candidates.keys(), key=lambda n: abs(n - cell_count))
    n = candidates[best_n]

    logger.info("Grid inference: %d cell contours -> %dx%d", cell_count, n, n)
    return n
