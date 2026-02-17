import logging
from pathlib import Path

import cv2
import numpy as np

from app.cell_extract import preprocess_cell

logger = logging.getLogger("boggle")

# Common OCR confusion mappings
CONFUSION_MAP = {
    "0": "O",
    "1": "I",
    "5": "S",
    "|": "I",
    "!": "I",
    "{": "C",
    "(": "C",
}


def load_templates(templates_dir: str) -> dict[str, np.ndarray] | None:
    """Load letter template images from a directory. Returns None if no templates found."""
    tpl_dir = Path(templates_dir)
    if not tpl_dir.exists():
        return None

    templates = {}
    for img_path in tpl_dir.glob("*.png"):
        letter = img_path.stem.upper()
        if letter.isalpha() or letter == "QU":
            tpl = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if tpl is not None:
                tpl = cv2.resize(tpl, (64, 64))
                _, tpl = cv2.threshold(tpl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if np.mean(tpl) < 128:
                    tpl = cv2.bitwise_not(tpl)
                templates[letter] = tpl

    return templates if templates else None


def _generate_synthetic_templates(size: int = 64) -> dict[str, np.ndarray]:
    """Generate synthetic letter templates using OpenCV's built-in font.

    These are a rough fallback for cells that EasyOCR's text detector misses
    (typically thin letters like I, V). Not accurate enough for primary recognition
    but useful for filling gaps.
    """
    templates = {}
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        img = np.ones((size, size), dtype=np.uint8) * 255
        font_scale = 1.5
        thickness = 2
        text_size = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        x = (size - text_size[0]) // 2
        y = (size + text_size[1]) // 2
        cv2.putText(img, letter, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 0, thickness)
        templates[letter] = img
    return templates


def _count_holes(binary_img: np.ndarray) -> int:
    """Count enclosed regions (holes) in a binarized letter image."""
    inv = cv2.bitwise_not(binary_img)
    contours, hierarchy = cv2.findContours(inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    return sum(1 for i in range(len(hierarchy[0])) if hierarchy[0][i][3] != -1)


# Expected hole counts per letter (from game font templates)
TEMPLATE_HOLES = {
    "A": 1, "B": 2, "C": 0, "D": 1, "E": 0, "F": 0, "G": 0, "H": 0,
    "I": 0, "J": 0, "K": 0, "L": 0, "M": 0, "N": 0, "O": 1, "P": 1,
    "QU": 1, "R": 1, "S": 0, "T": 0, "U": 0, "V": 0, "W": 0, "X": 0,
    "Y": 0, "Z": 0,
}


def _template_match(cell_processed: np.ndarray, templates: dict[str, np.ndarray]) -> tuple[str, float]:
    """Match a preprocessed cell against all templates using normalized cross-correlation."""
    best_letter = "?"
    best_score = -1.0

    for letter, tpl in templates.items():
        result = cv2.matchTemplate(cell_processed, tpl, cv2.TM_CCOEFF_NORMED)
        score = result.max()
        if score > best_score:
            best_score = score
            best_letter = letter

    return best_letter, float(best_score)


def _disambiguate_rp(cell_processed: np.ndarray) -> str:
    """Distinguish R from P using lower-right quadrant pixel density.

    R has a diagonal leg extending into the lower-right area that P lacks.
    Across 76 validated samples: R min=0.121, P max=0.056 — clean separation.
    """
    h, w = cell_processed.shape
    lower_right = cell_processed[h // 2:, w // 2:]
    total_dark = max(np.sum(cell_processed < 128), 1)
    lr_dark = np.sum(lower_right < 128)
    ratio = lr_dark / total_dark
    return "R" if ratio > 0.08 else "P"


def _template_match_verified(cell_processed: np.ndarray, templates: dict[str, np.ndarray]) -> tuple[str, float]:
    """Template match with structural verification.

    Applies hole-count verification and R/P structural disambiguation.
    """
    # Get all scores sorted
    scores = []
    for letter, tpl in templates.items():
        result = cv2.matchTemplate(cell_processed, tpl, cv2.TM_CCOEFF_NORMED)
        scores.append((letter, result.max()))
    scores.sort(key=lambda x: -x[1])

    if not scores:
        return "?", 0.0

    cell_holes = _count_holes(cell_processed)

    # Try each candidate, pick the first one with compatible hole count
    for letter, score in scores:
        expected_holes = TEMPLATE_HOLES.get(letter, 0)
        # Allow ±1 tolerance since hole detection isn't perfectly reliable
        if abs(cell_holes - expected_holes) <= 1:
            # R/P disambiguation: if match is R or P, verify structurally
            if letter in ("R", "P"):
                correct = _disambiguate_rp(cell_processed)
                if correct != letter:
                    # Find the score for the correct letter
                    correct_score = next((s for l, s in scores if l == correct), score)
                    return correct, float(correct_score)
            return letter, float(score)

    # If nothing matches with compatible holes, return top match anyway
    return scores[0][0], float(scores[0][1])


def init_easyocr():
    """Initialize EasyOCR reader for English."""
    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        logger.info("EasyOCR reader initialized (CPU)")
        return reader
    except Exception as e:
        logger.error("Failed to init EasyOCR: %s", e)
        return None


def _easyocr_full_board(
    warped_gray: np.ndarray,
    grid_size: int,
    reader,
) -> dict[tuple[int, int], tuple[str, float]]:
    """Run EasyOCR on the full warped board image.

    Much more accurate than per-cell EasyOCR because the text detector
    works better with spatial context from multiple characters.
    Returns {(row, col): (letter, confidence)}.
    """
    # Invert: game has white letters on dark cells -> dark letters on white bg
    inverted = cv2.bitwise_not(warped_gray)

    results = reader.readtext(
        inverted,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        detail=1,
        paragraph=False,
        min_size=5,
        text_threshold=0.3,
        low_text=0.2,
    )

    cell_size = warped_gray.shape[0] / grid_size
    detections: dict[tuple[int, int], tuple[str, float]] = {}

    for bbox, text, conf in results:
        cx = (bbox[0][0] + bbox[2][0]) / 2
        cy = (bbox[0][1] + bbox[2][1]) / 2
        r = min(int(cy / cell_size), grid_size - 1)
        c = min(int(cx / cell_size), grid_size - 1)

        text = _clean_ocr_text(text)
        if not text:
            continue

        # Keep highest-confidence detection per cell
        if (r, c) not in detections or conf > detections[(r, c)][1]:
            detections[(r, c)] = (text, conf)

    return detections


def _clean_ocr_text(text: str) -> str:
    """Clean and normalize OCR output to a single letter."""
    text = text.upper().strip()
    if not text:
        return ""

    # Apply confusion map
    result = []
    for ch in text:
        result.append(CONFUSION_MAP.get(ch, ch))
    text = "".join(result)

    # Keep only alpha characters
    alpha = "".join(c for c in text if c.isalpha())
    if not alpha:
        return ""

    # Return first letter (or "QU" if Q detected)
    if alpha[0] == "Q":
        return "QU"
    return alpha[0]


def recognize_cells(
    cells: list[np.ndarray],
    templates: dict[str, np.ndarray] | None,
    easyocr_reader,
    confidence_threshold: float = 0.75,
    warped_gray: np.ndarray | None = None,
    grid_size: int | None = None,
) -> tuple[list[list[str]], list[list[float]]]:
    """Recognize letters using hybrid approach:

    1. Primary: Full-board EasyOCR (high accuracy on most letters)
    2. Fallback: Template matching for cells EasyOCR missed (catches I, V, etc.)

    If templates/ has game-specific templates (from calibration), uses those.
    Otherwise falls back to synthetic OpenCV-font templates.
    """
    n = grid_size if grid_size else int(len(cells) ** 0.5)
    letters = ["?"] * len(cells)
    confidences = [0.0] * len(cells)

    # Step 1: Full-board EasyOCR (primary — gets ~80-90% of cells)
    easyocr_detections = {}
    if easyocr_reader is not None and warped_gray is not None:
        easyocr_detections = _easyocr_full_board(warped_gray, n, easyocr_reader)
        for (r, c), (letter, conf) in easyocr_detections.items():
            idx = r * n + c
            letters[idx] = letter
            confidences[idx] = conf
        detected = len(easyocr_detections)
        logger.info("Full-board EasyOCR: %d/%d cells detected", detected, n * n)

    # Step 2: Template matching for undetected cells
    tpl = templates if templates else _generate_synthetic_templates()
    undetected = [i for i in range(len(cells)) if letters[i] == "?"]
    if undetected:
        processed = {i: preprocess_cell(cells[i]) for i in undetected}

        for i in undetected:
            letter, conf = _template_match_verified(processed[i], tpl)
            letters[i] = letter
            confidences[i] = conf
            r, c = divmod(i, n)
            logger.info(
                "Template fallback (%d,%d): %s (conf=%.3f)",
                r, c, letter, conf,
            )

    # Step 3: Smart merge — cross-check EasyOCR with template matching
    if templates:
        for (r, c), (ocr_letter, ocr_conf) in easyocr_detections.items():
            idx = r * n + c
            cell_processed = preprocess_cell(cells[idx])
            tpl_letter, tpl_conf = _template_match_verified(cell_processed, tpl)
            if tpl_letter != ocr_letter and ocr_conf < 0.9:
                # Template disagrees and EasyOCR isn't highly confident — override when:
                # a) template is very confident (>0.85), or
                # b) template scores higher than EasyOCR and EasyOCR is uncertain (<0.7)
                if tpl_conf > 0.85 or (tpl_conf > ocr_conf and ocr_conf < 0.7):
                    letters[idx] = tpl_letter
                    confidences[idx] = tpl_conf
                    logger.info(
                        "Smart merge (%d,%d): EasyOCR=%s(%.3f) -> Template=%s(%.3f)",
                        r, c, ocr_letter, ocr_conf, tpl_letter, tpl_conf,
                    )
            else:
                # Same letter — boost confidence to the higher of the two
                confidences[idx] = max(confidences[idx], tpl_conf)

    # R/P structural disambiguation — apply to all cells detected as R or P
    for i in range(len(cells)):
        if letters[i] in ("R", "P"):
            proc = preprocess_cell(cells[i])
            correct = _disambiguate_rp(proc)
            if correct != letters[i]:
                r, c = divmod(i, n)
                logger.info(
                    "R/P disambig (%d,%d): %s -> %s",
                    r, c, letters[i], correct,
                )
                letters[i] = correct

    # Handle Q -> QU
    for i in range(len(letters)):
        if letters[i] == "Q":
            letters[i] = "QU"

    # Reshape to 2D
    board = [letters[r * n:(r + 1) * n] for r in range(n)]
    conf_2d = [confidences[r * n:(r + 1) * n] for r in range(n)]

    return board, conf_2d
