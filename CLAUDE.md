# Boggle Solver — Project Guide

## What This Project Does
FastAPI server that receives iPhone screenshots of Boggle boards, auto-detects the grid (4x4/5x5/6x6), OCRs letters, solves via Trie+DFS, and pushes 4-5 letter word results via ntfy.sh.

## Architecture
```
POST /solve (multipart image)
  → decode (OpenCV)
  → detect_board_and_warp()    [board_detect.py]  — contour/Hough detection + perspective warp
  → infer_grid_size()          [board_detect.py]  — morphological line counting
  → split_cells()              [cell_extract.py]  — uniform grid on warped square + CLAHE/Otsu
  → recognize_cells()          [recognition.py]   — Tier A templates + Tier B EasyOCR fallback
  → solve()                    [solver.py]        — Trie + DFS with bitmask visited
  → send_notification()        [notifier.py]      — ntfy.sh (background, non-blocking)
  → return JSON response
```

## Project Layout
```
app/
  server.py          FastAPI app, /health, /solve, /debug/cells endpoints
  settings.py        Dataclass config with env overrides
  board_detect.py    Board localization + perspective warp + grid size inference
  cell_extract.py    Cell cropping, CLAHE preprocessing, montage debug view
  recognition.py     Two-tier OCR: template matching (fast) + EasyOCR (fallback)
  solver.py          TrieNode (__slots__), Trie, DFS with bitmask visited
  notifier.py        Async ntfy.sh POST (4-5 letter words only)
  metrics.py         StageTimer context manager for per-stage timing
scripts/
  calibration.py     CLI tool: board detection verify, OCR test, template capture
templates/letters/   Letter template PNGs for Tier A matching (populated via calibration)
tests/               pytest suite: test_solver, test_board_detect, test_recognition
dictionary.txt       TWL06 word list (~137k words)
```

## Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Run server (port 10001)
uvicorn app.server:app --host 0.0.0.0 --port 10001

# Run tests
python -m pytest tests/ -v

# Run calibration
python -m scripts.calibration <screenshot.png> [--grid-size 4] [--save-templates]

# Test solve endpoint
curl -F "file=@screenshot.png" http://localhost:10001/solve

# Debug cell extraction
curl -F "file=@screenshot.png" http://localhost:10001/debug/cells -o cells.png
```

## Key Design Decisions
- **Board detection**: Contour + perspective warp (not fixed pixel coords) — works across iPhone models
- **Grid size inference**: Morphological line detection on warped board (not template matching)
- **Two-tier OCR**: Fast template cross-correlation first; EasyOCR only for low-confidence cells
- **Solver**: Bitmask visited (int bitset) + TrieNode walking for O(1) prefix checks
- **EasyOCR threading**: Single-threaded batch, torch.set_num_threads(4) — no ThreadPoolExecutor
- **Notification**: Background task via FastAPI BackgroundTasks, filtered to 4-5 letter words
- **1 uvicorn worker**: Avoids duplicating PyTorch model + Trie in RAM

## Configuration
All settings in `app/settings.py` are overridable via environment variables:
- `NTFY_TOPIC` — ntfy.sh topic name (default: "boggle-solver")
- `MIN_WORD_LENGTH` — minimum word length to find (default: 3)
- `MAX_RESULTS` — max words returned in JSON (default: 50)
- `CELL_INSET` — fraction to crop from cell edges (default: 0.15)
- `OCR_CONFIDENCE_THRESHOLD` — Tier A confidence cutoff (default: 0.75)
- `DEBUG` — save debug artifacts per request (default: false)
- `TORCH_NUM_THREADS` — CPU threads for EasyOCR (default: 4)
- `PORT` — server port (default: 10001)

## Testing Notes
- Solver tests use hardcoded boards + mini dictionaries — no external deps
- Board detection tests use synthetic OpenCV-generated grid images
- Recognition tests verify confusion mapping (0→O, 1→I, 5→S) and template matching
- Performance test asserts solver completes 4x4 board in <500ms

## Common Issues
- **EasyOCR first run**: Downloads ~100MB model on first use. Subsequent runs use cache.
- **Windows firewall**: Must allow inbound TCP on port 10001 for iPhone access.
- **Empty templates/letters/**: Tier A is disabled until calibration populates templates.
- **"Q" handling**: OCR converts Q→QU at recognition layer; solver iterates all chars per cell.
