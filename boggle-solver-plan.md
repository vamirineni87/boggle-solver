# Boggle Solver System — Claude Code Plan

## Overview
Build a Python FastAPI server that receives full iPhone screenshots, crops a Boggle grid (4x4 up to 6x6) at known pixel coordinates, OCRs each cell, solves the board using trie+DFS, and sends results as a push notification via ntfy.sh.

## Architecture
```
iPhone Screenshot → iOS Shortcut (one tap, share sheet POST) → FastAPI server (local desktop) → Auto-detect grid size → OCR + Solve → ntfy.sh push notification → iPhone
```

## Project Structure
```
boggle-solver/
├── server.py              # FastAPI app with /solve endpoint
├── grid_detect.py         # Auto-detect grid size (4x4/5x5/6x6) from screenshot
├── ocr.py                 # Screenshot cropping + per-cell OCR (Pillow + Tesseract)
├── solver.py              # Trie construction + DFS board solver
├── notifier.py            # ntfy.sh push notification sender
├── config.py              # Grid pixel coordinates, ntfy topic, detection templates
├── templates/             # Reference crops saved during calibration (one per grid size)
├── dictionary.txt         # Word list (use TWL06 or SOWPODS scrabble dictionary)
├── requirements.txt       # fastapi, uvicorn, pillow, pytesseract, httpx
├── calibration.py         # Helper: calibrate pixel coords + save detection templates per grid size
└── README.md              # Setup instructions including iOS Shortcut steps
```

## Build Order

### Step 1: config.py
- Define dataclass/dict for grid configs by size (4x4, 5x5, 6x6)
- Each config holds: `grid_top_left_x`, `grid_top_left_y`, `grid_bottom_right_x`, `grid_bottom_right_y`, `cell_width`, `cell_height`, `grid_size` (rows/cols)
- Also store a `detection_signature` per grid size — e.g. total grid pixel dimensions, or a known unique region/crop hash — used by auto-detection
- Default ntfy topic name (user configurable)
- Placeholder pixel values (user calibrates later)

### Step 2: grid_detect.py
- `detect_grid_size(image_bytes: bytes) -> int`
- Strategy (pick the most reliable during calibration):
  - **Option A — Template matching**: During calibration, save a small reference crop from each grid size (e.g. a corner region or UI element that differs). At runtime, compare against all templates, best match wins.
  - **Option B — Grid line analysis**: Convert to grayscale, edge detect (Canny), look for horizontal/vertical lines. Count evenly-spaced divisions to determine 4/5/6.
  - **Option C — Cell count by contour**: Threshold + find contours in the grid region. Count the number of letter cells (16=4x4, 25=5x5, 36=6x6).
- Implement Option A as default (simplest and most reliable for a known app with fixed layouts). Fall back to Option C if needed.
- Calibration script saves the reference crops automatically.

### Step 3: solver.py
- Load dictionary.txt into a Trie (prefix tree)
- `solve(board: list[list[str]]) -> list[str]`
- DFS from every cell, track visited, prune on prefix misses
- Filter results: minimum 3-letter words, deduplicate, sort by length descending then alpha
- Handle "Qu" cells (Boggle treats Q as QU)
- Return top N results (configurable, default 50) to keep notification readable

### Step 4: ocr.py
- `process_screenshot(image_bytes: bytes, grid_size: int) -> list[list[str]]`
- grid_size is provided by grid_detect.py (auto-detected, not user-supplied)
- Open image with Pillow
- Crop to grid region using config pixel coords for the given grid_size
- Segment into individual cells (evenly divide grid region)
- Per cell: convert to grayscale, threshold (binarize), resize up to ~100x100 for OCR accuracy
- Run pytesseract on each cell with `--psm 10` (single character mode), whitelist A-Z
- Return 2D list of uppercase letters
- Include a confidence check — log warnings if Tesseract confidence is low on any cell

### Step 5: notifier.py
- `send_notification(words: list[str], grid_size: int, ntfy_topic: str)`
- Format message: grid size header + word count + top words grouped by length
- POST to `https://ntfy.sh/{topic}` with title and formatted body
- Use httpx async client

### Step 6: server.py
- FastAPI app
- `POST /solve` — accepts multipart file upload (the screenshot image)
  - No grid_size param needed — auto-detected from the image
  - Pipeline: grid_detect.detect_grid_size → ocr.process_screenshot → solver.solve → notifier.send_notification
  - Return JSON with detected grid size, extracted board, and found words (for debugging)
- `GET /health` — simple health check
- Add basic error handling and logging
- Run with uvicorn on port 8000

### Step 7: calibration.py
- CLI helper: `python calibration.py <screenshot_path> <grid_size>`
- Displays the image with matplotlib, overlays the current config grid lines for that size
- User visually confirms alignment or adjusts coords in config.py
- Prints what OCR reads from each cell so user can verify accuracy
- **Saves a detection reference crop** to `templates/` for that grid size (used by grid_detect.py at runtime)
- Run once per grid size (3 times total: 4x4, 5x5, 6x6)

### Step 8: requirements.txt
```
fastapi
uvicorn
pillow
pytesseract
httpx
matplotlib
```
Also note: user needs `tesseract-ocr` installed on system (`brew install tesseract` on macOS or equivalent)

### Step 9: README.md
Include:
- System dependencies (Python 3.10+, Tesseract)
- pip install instructions
- How to run calibration with a sample screenshot
- How to start the server
- iOS Shortcut setup instructions:
  1. Create shortcut named "Solve Boggle"
  2. Action: "Get Latest Screenshot" or receive from Share Sheet
  3. Action: "Get Contents of URL" — POST multipart to `http://<desktop-ip>:8000/solve` (no params needed, grid size auto-detected)
  4. Shortcut runs without leaving current app
- How to subscribe to ntfy.sh topic on iPhone (install ntfy app, subscribe to your topic)

## Key Design Decisions
- **Auto grid detection** — template matching against calibration reference crops means zero user interaction at runtime; single Shortcut handles all grid sizes
- **No dynamic grid detection** — fixed pixel coords per grid size since screenshot dimensions are consistent (grid detection just picks which config to use)
- **ntfy.sh over Pushover** — free, no account needed, open source
- **Tesseract PSM 10** — single character mode is ideal for isolated cell crops
- **Trie not hashset** — enables prefix pruning which is critical for 6x6 boards (36 cells = huge search space)
- **All processing server-side** — iPhone only takes screenshot and fires one HTTP request

## Stretch / Optional
- Add WebSocket endpoint for live result streaming
- Cache dictionary trie on startup (already planned, just noting)
- Support Boggle variants (e.g., WordHunt which allows diagonal-only paths)
- Add a simple web UI at GET / that shows last solved board visually
