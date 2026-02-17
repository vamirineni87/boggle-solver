"""
Calibration helper for Boggle Solver.

Usage:
    python -m scripts.calibration <screenshot_path> [--grid-size N]

Examples:
    python -m scripts.calibration screenshots/sample.png
    python -m scripts.calibration screenshots/5x5.png --grid-size 5
    python -m scripts.calibration screenshots/sample.png --save-templates

This will:
  1. Auto-detect the board region and perspective-warp it
  2. Infer or verify grid size (4, 5, or 6)
  3. Split into cells and display a montage for visual verification
  4. Run OCR on each cell and print the detected board
  5. Optionally save cell crops as letter templates (--save-templates)
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.settings import settings
from app.board_detect import detect_board_and_warp, infer_grid_size
from app.cell_extract import split_cells, preprocess_cell, make_cell_montage
from app.recognition import load_templates, _template_match, init_easyocr, _clean_ocr_text


def main():
    parser = argparse.ArgumentParser(description="Boggle Solver Calibration Tool")
    parser.add_argument("screenshot", help="Path to a screenshot image")
    parser.add_argument("--grid-size", type=int, choices=[4, 5, 6], default=None,
                        help="Override auto-detected grid size")
    parser.add_argument("--save-templates", action="store_true",
                        help="Save each cell crop as a letter template (prompts for labels)")
    parser.add_argument("--warp-size", type=int, default=settings.WARP_SIZE,
                        help=f"Warp target size in pixels (default: {settings.WARP_SIZE})")
    parser.add_argument("--output-dir", type=str, default="calibration_output",
                        help="Directory to save debug outputs (default: calibration_output)")
    args = parser.parse_args()

    img_path = Path(args.screenshot)
    if not img_path.exists():
        print(f"Error: {img_path} does not exist")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load image
    image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: Could not decode image at {img_path}")
        sys.exit(1)

    print(f"Image: {img_path} ({image.shape[1]}x{image.shape[0]})")

    # Step 1: Detect board and warp
    print("\n--- Board Detection ---")
    warped, debug_info = detect_board_and_warp(image, args.warp_size)
    print(f"Detection method: {debug_info['method']}")
    print(f"Corners: {debug_info['corners']}")

    warp_path = output_dir / "warped_board.png"
    cv2.imwrite(str(warp_path), warped)
    print(f"Warped board saved to: {warp_path}")

    # Step 2: Infer grid size
    if args.grid_size:
        grid_size = args.grid_size
        print(f"\nGrid size (override): {grid_size}x{grid_size}")
    else:
        grid_size = infer_grid_size(warped)
        print(f"\nAuto-detected grid size: {grid_size}x{grid_size}")

    # Step 3: Split cells
    print(f"\n--- Cell Extraction ({grid_size}x{grid_size} = {grid_size**2} cells) ---")
    cells = split_cells(warped, grid_size, settings.CELL_INSET)

    # Save montage
    montage = make_cell_montage(cells, grid_size)
    montage_path = output_dir / "cell_montage.png"
    cv2.imwrite(str(montage_path), montage)
    print(f"Cell montage saved to: {montage_path}")

    # Step 4: OCR each cell
    print(f"\n--- OCR Results ---")
    templates = load_templates(str(settings.TEMPLATES_DIR))

    # Try template matching first
    board = []
    processed_cells = [preprocess_cell(c) for c in cells]

    use_easyocr = False
    if templates:
        print(f"Using Tier A (template matching, {len(templates)} templates loaded)")
        for i, cell_proc in enumerate(processed_cells):
            letter, conf = _template_match(cell_proc, templates)
            if conf < settings.OCR_CONFIDENCE_THRESHOLD:
                use_easyocr = True
            board.append((letter, conf))
    else:
        print("No templates found — using EasyOCR directly")
        use_easyocr = True
        board = [("?", 0.0)] * len(cells)

    # EasyOCR fallback
    if use_easyocr:
        print("Initializing EasyOCR for uncertain cells...")
        reader = init_easyocr()
        if reader:
            for i, cell in enumerate(cells):
                if board[i][1] < settings.OCR_CONFIDENCE_THRESHOLD:
                    try:
                        results = reader.readtext(
                            cell,
                            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                            detail=1,
                            paragraph=False,
                        )
                        if results:
                            text = _clean_ocr_text(results[0][1])
                            conf = float(results[0][2])
                            if text and conf > board[i][1]:
                                board[i] = (text, conf)
                        else:
                            if board[i][0] == "?":
                                board[i] = ("?", 0.0)
                    except Exception as e:
                        print(f"  EasyOCR error on cell {i}: {e}")

    # Handle Q -> QU
    board = [(("QU" if l == "Q" else l), c) for l, c in board]

    # Print board
    print(f"\nDetected board ({grid_size}x{grid_size}):")
    print("-" * (grid_size * 6 + 1))
    for r in range(grid_size):
        row_letters = []
        row_confs = []
        for c in range(grid_size):
            idx = r * grid_size + c
            letter, conf = board[idx]
            row_letters.append(f" {letter:>2} ")
            row_confs.append(f"{conf:.2f}")
        print("|" + "|".join(row_letters) + "|")
    print("-" * (grid_size * 6 + 1))

    print("\nConfidences:")
    for r in range(grid_size):
        confs = []
        for c in range(grid_size):
            idx = r * grid_size + c
            _, conf = board[idx]
            marker = " " if conf >= settings.OCR_CONFIDENCE_THRESHOLD else "!"
            confs.append(f"{conf:.2f}{marker}")
        print(f"  Row {r}: {' '.join(confs)}")

    low_conf = sum(1 for _, c in board if c < settings.OCR_CONFIDENCE_THRESHOLD)
    if low_conf > 0:
        print(f"\nWarning: {low_conf} cell(s) below confidence threshold ({settings.OCR_CONFIDENCE_THRESHOLD})")

    # Step 5: Save templates if requested
    if args.save_templates:
        _save_templates_interactive(processed_cells, board, grid_size)

    print(f"\nAll outputs saved to: {output_dir}/")


def _save_templates_interactive(processed_cells, board, grid_size):
    """Interactively save cell crops as letter templates."""
    templates_dir = settings.TEMPLATES_DIR
    templates_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Template Saving ---")
    print(f"Templates directory: {templates_dir}")
    print("For each cell, enter the correct letter (or press Enter to use detected, 's' to skip):\n")

    saved = 0
    for i, cell_proc in enumerate(processed_cells):
        r, c = divmod(i, grid_size)
        detected = board[i][0]
        prompt = f"  Cell ({r},{c}) detected='{detected}': "

        user_input = input(prompt).strip().upper()
        if user_input == "S":
            continue
        elif user_input == "":
            letter = detected
        else:
            letter = user_input[0]

        if letter == "?" or not letter.isalpha():
            print(f"    Skipping invalid: '{letter}'")
            continue

        tpl_path = templates_dir / f"{letter}.png"
        cv2.imwrite(str(tpl_path), cell_proc)
        print(f"    Saved: {tpl_path}")
        saved += 1

    print(f"\n{saved} templates saved to {templates_dir}")


if __name__ == "__main__":
    main()
