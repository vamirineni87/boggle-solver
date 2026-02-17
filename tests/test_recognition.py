import cv2
import numpy as np
import pytest
from app.recognition import _clean_ocr_text, _template_match, load_templates


def test_clean_ocr_basic():
    assert _clean_ocr_text("A") == "A"
    assert _clean_ocr_text("a") == "A"
    assert _clean_ocr_text("Q") == "QU"
    assert _clean_ocr_text("q") == "QU"


def test_clean_ocr_confusion():
    assert _clean_ocr_text("0") == "O"
    assert _clean_ocr_text("1") == "I"
    assert _clean_ocr_text("5") == "S"


def test_clean_ocr_empty():
    assert _clean_ocr_text("") == ""
    assert _clean_ocr_text("  ") == ""
    assert _clean_ocr_text("123") == "I"  # "1" maps to "I"


def test_clean_ocr_multi_char():
    # Only first alpha char returned
    assert _clean_ocr_text("AB") == "A"
    assert _clean_ocr_text("QU") == "QU"


def test_load_templates_nonexistent():
    result = load_templates("/nonexistent/path")
    assert result is None


def test_template_match_synthetic():
    """Create two synthetic templates and verify matching works."""
    # Template for "A" - distinctive pattern
    tpl_a = np.zeros((64, 64), dtype=np.uint8)
    tpl_a[10:54, 10:54] = 255
    cv2.putText(tpl_a, "A", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)

    # Template for "B"
    tpl_b = np.zeros((64, 64), dtype=np.uint8)
    tpl_b[10:54, 10:54] = 255
    cv2.putText(tpl_b, "B", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)

    templates = {"A": tpl_a, "B": tpl_b}

    # Create a test cell that looks like "A"
    test_cell = np.zeros((64, 64), dtype=np.uint8)
    test_cell[10:54, 10:54] = 255
    cv2.putText(test_cell, "A", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)

    letter, conf = _template_match(test_cell, templates)
    assert letter == "A"
    assert conf > 0.5
