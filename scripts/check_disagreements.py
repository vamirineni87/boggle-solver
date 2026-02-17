"""Find all cells where EasyOCR and template matching disagree."""
import cv2
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.settings import settings
from app.cell_extract import split_cells, preprocess_cell
from app.recognition import load_templates, _template_match_verified

templates = load_templates(str(settings.TEMPLATES_DIR))
debug_dir = PROJECT_ROOT / "debug"

disagreements = []
for fname in sorted(os.listdir(debug_dir)):
    if not fname.endswith("_result.json"):
        continue
    ts = fname.replace("_result.json", "")
    with open(debug_dir / fname) as f:
        result = json.load(f)
    board = result["board"]
    confs = result["confidences"]
    grid_size = result["grid_size"]
    warped = cv2.imread(str(debug_dir / f"{ts}_warp.png"), cv2.IMREAD_GRAYSCALE)
    if warped is None:
        continue
    cells = split_cells(warped, grid_size, settings.CELL_INSET)
    for r in range(grid_size):
        for c in range(grid_size):
            idx = r * grid_size + c
            proc = preprocess_cell(cells[idx])
            tpl_letter, tpl_conf = _template_match_verified(proc, templates)
            ocr_letter = board[r][c]
            ocr_conf = confs[r][c]
            if tpl_letter != ocr_letter and tpl_conf > 0.5:
                disagreements.append((ts, r, c, ocr_letter, ocr_conf, tpl_letter, tpl_conf))

print(f"Total disagreements: {len(disagreements)}")
print()
for ts, r, c, ol, oc, tl, tc in sorted(disagreements, key=lambda x: -x[6]):
    print(f"  {ts} ({r},{c}) OCR={ol}({oc:.3f}) TPL={tl}({tc:.3f})")
