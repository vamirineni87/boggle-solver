# Boggle Solver

A FastAPI server that solves Boggle boards from iPhone screenshots. Take a screenshot, fire an iOS Shortcut, and get the answers as a push notification — all in under 10 seconds.

## How It Works

```
iPhone Screenshot
  → iOS Shortcut (POST to server)
    → Auto-detect board region (contour detection + perspective warp)
      → Infer grid size (4x4 / 5x5 / 6x6)
        → OCR each cell (template matching + EasyOCR fallback)
          → Solve via Trie + DFS
            → Push notification (ntfy.sh) with 4-5 letter words
```

## Prerequisites

- **Python 3.10+**
- **pip** (included with Python)

No system-level OCR install needed — EasyOCR is pure Python.

## Installation

```bash
# Clone / download the project
cd boggle

# Install dependencies
pip install -r requirements.txt
```

On the first run, EasyOCR will download its English model (~100 MB). This is cached for future use.

## Quick Start

### 1. Start the server

```bash
uvicorn app.server:app --host 0.0.0.0 --port 10001
```

The server loads the dictionary (~137k words) and EasyOCR model at startup. This takes a few seconds.

### 2. Test with a screenshot

```bash
curl -F "file=@your_screenshot.png" http://localhost:10001/solve
```

Response:
```json
{
  "grid_size": 4,
  "board": [["C","A","T","S"], ["R","E","P","O"], ["B","O","N","E"], ["D","I","G","S"]],
  "words": ["BONES", "OPENS", "PONES", "..."],
  "word_count": 42,
  "processing_time": 2800.5,
  "stage_timings": {"decode": 15.2, "board_detect": 180.3, "...": "..."},
  "cell_confidences": [[0.95, 0.88, ...], ...]
}
```

### 3. Set up push notifications (ntfy.sh)

1. Install **ntfy** from the App Store (search "ntfy", by Philipp Heckel — free)
2. Open the app → tap **+** (Subscribe to topic)
3. Topic name: `boggle-solver`
4. Server: leave as `https://ntfy.sh` (default)
5. Tap **Subscribe**
6. **Verify it works** — run this from your PC:
   ```bash
   curl -d "test notification" https://ntfy.sh/boggle-solver
   ```
   You should see a notification on your phone within 1-2 seconds.

If no notification appears:
- Check iOS Settings → Notifications → ntfy → make sure Allow Notifications is ON
- Make sure Background App Refresh is enabled for ntfy
- Try opening the ntfy app and pulling to refresh

### 4. Find your PC's IP address

Run this on your PC:
```bash
ipconfig
```
Look for your Wi-Fi or Ethernet adapter's **IPv4 Address** (e.g., `192.168.1.100`). Your iPhone must be on the **same Wi-Fi network**.

## iOS Shortcut Setup

Pick one of these three options:

### Option A: Share Sheet Shortcut (recommended)

Take a screenshot, tap Share, tap "Solve Boggle".

1. Open the **Shortcuts** app → tap **+** to create new
2. Tap the name at top → rename to **Solve Boggle**
3. Add actions in order:

**Action 1 — Receive the screenshot**
- Tap **Add Action** → search **"Receive"**
- Select **"Receive ___ input from ___"**
- Configure: **Receive Images input from Share Sheet**

**Action 2 — Send to server**
- Tap **+** → search **"Get Contents of URL"**
- URL: `http://<your-pc-ip>:10001/solve`
- Tap **Show More**:
  - Method: **POST**
  - Request Body: **Form**
  - Tap **Add new field** → choose **File**
  - Key: `file`
  - Value: tap the field → select **Shortcut Input**

**Action 3 (optional) — Confirmation**
- Tap **+** → search **"Show Notification"**
- Text: `Boggle sent! Check ntfy for results.`

4. Tap the **settings icon** (top right ⓘ):
   - Enable **Show in Share Sheet**
   - Under Share Sheet Types → check **Images**
5. Tap **Done**

**Usage**: Screenshot → tap thumbnail → Share → **Solve Boggle**

### Option B: One-Tap Home Screen Button

Grabs the latest screenshot automatically — one tap to solve.

1. Create a new shortcut → name it **Solve Boggle**
2. Add actions:

**Action 1 — Get latest screenshot**
- **Add Action** → search **"Find Photos"**
- Tap **Add Filter** → set **Media Type** is **Screenshot**
- Tap **Sort By** → **Creation Date**, **Latest First**
- Toggle **Limit** ON → set to **1**

**Action 2 — Send to server**
- **Add Action** → search **"Get Contents of URL"**
- URL: `http://<your-pc-ip>:10001/solve`
- Tap **Show More**:
  - Method: **POST**
  - Request Body: **Form**
  - **Add new field** → **File** → Key: `file` → Value: select **Photos** (from step 1)

**Action 3 (optional)** — **Show Notification**: `Boggle solved!`

3. Tap **Done**
4. Add to Home Screen: settings icon (ⓘ) → **Add to Home Screen** → choose an icon

**Usage**: Screenshot the Boggle board → tap the Home Screen icon

### Option C: Fully Automatic (zero taps)

Runs automatically every time you take a screenshot. iOS 15.4+ required.

1. Open Shortcuts → **Automation** tab → tap **+**
2. Select **Create Personal Automation**
3. Scroll to **Screenshot** (under Events) → tap it → **Next**
4. Add the same two actions as Option B:
   - Find Photos (latest 1 screenshot)
   - Get Contents of URL (POST to your server)
5. Tap **Next**
6. Toggle **OFF** "Ask Before Running" → confirm
7. Tap **Done**

**Usage**: Just take a screenshot. The solve fires automatically in the background.

> **Tip**: This triggers on *every* screenshot. To make it Boggle-only, add a **"Choose from Menu"** action before the HTTP request with options "Solve" / "Cancel".

## Calibration (Optional but Recommended)

Calibration creates letter templates that dramatically speed up OCR (Tier A: ~20ms vs Tier B EasyOCR: ~1-3s).

```bash
# Auto-detect board and verify OCR accuracy
python -m scripts.calibration screenshot.png

# Override grid size if auto-detection is wrong
python -m scripts.calibration screenshot.png --grid-size 5

# Save cell crops as letter templates for fast Tier A matching
python -m scripts.calibration screenshot.png --save-templates
```

The calibration tool will:
- Display the detected + warped board region
- Show a cell montage for visual inspection
- Print OCR results per cell with confidence scores
- Optionally save templates to `templates/letters/`

Run `--save-templates` once with a clean screenshot where you know all the letters. This populates the Tier A template cache for fast recognition.

## API Reference

### `GET /health`
Health check. Returns `{"status": "ok", "trie_loaded": true}`.

### `POST /solve`
Main endpoint. Accepts a screenshot as multipart form upload.

**Request**: `multipart/form-data` with field `file` (image/png or image/jpeg)

**Response** (JSON):
| Field | Type | Description |
|-------|------|-------------|
| `grid_size` | int | Detected grid size (4, 5, or 6) |
| `board` | string[][] | 2D array of detected letters |
| `words` | string[] | Found words, sorted by length desc |
| `word_count` | int | Number of words found |
| `processing_time` | float | Total time in milliseconds |
| `stage_timings` | object | Per-stage breakdown (ms) |
| `cell_confidences` | float[][] | OCR confidence per cell (0-1) |

**Errors**:
- `400` — Not an image file or couldn't decode
- `413` — File too large (max 5 MB)
- `422` — Board detection failed
- `503` — Dictionary not loaded yet (server still starting)

### `POST /debug/cells`
Debug endpoint. Returns a PNG montage image of all extracted cells.

**Request**: Same as `/solve`
**Response**: `image/png`

## Configuration

All settings can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `NTFY_TOPIC` | `boggle-solver` | ntfy.sh topic for push notifications |
| `NTFY_URL` | `https://ntfy.sh` | ntfy.sh server URL |
| `MIN_WORD_LENGTH` | `3` | Minimum word length to find |
| `MAX_RESULTS` | `50` | Maximum words in JSON response |
| `CELL_INSET` | `0.15` | Fraction cropped from cell edges (avoids grid lines) |
| `OCR_CONFIDENCE_THRESHOLD` | `0.75` | Below this, cell falls back to EasyOCR |
| `MAX_UPLOAD_BYTES` | `5000000` | Maximum upload file size |
| `DEBUG` | `false` | Save debug artifacts per request |
| `TORCH_NUM_THREADS` | `4` | CPU threads for EasyOCR inference |
| `WARP_SIZE` | `600` | Board warp resolution in pixels |
| `PORT` | `10001` | Server port |

Example:
```bash
DEBUG=true NTFY_TOPIC=my-boggle uvicorn app.server:app --host 0.0.0.0 --port 10001
```

## Debug Mode

Set `DEBUG=true` to save per-request artifacts:

```
debug/
  req_<timestamp>_orig.jpg      Original screenshot
  req_<timestamp>_warp.png      Perspective-warped board
  req_<timestamp>_cells.png     Cell extraction montage
  req_<timestamp>_result.json   Board, words, and timing data
```

Use the `/debug/cells` endpoint for quick visual verification without enabling full debug mode.

## Windows Firewall

The server binds to `0.0.0.0:10001`. Windows Firewall blocks inbound connections by default. To allow your iPhone to reach the server:

1. Open **Windows Defender Firewall with Advanced Security**
2. Click **Inbound Rules** > **New Rule**
3. Rule Type: **Port**
4. Protocol: **TCP**, Specific port: **10001**
5. Action: **Allow the connection**
6. Profile: **Private** (your home network)
7. Name: `Boggle Solver`

Or via PowerShell (admin):
```powershell
New-NetFirewallRule -DisplayName "Boggle Solver" -Direction Inbound -Protocol TCP -LocalPort 10001 -Action Allow -Profile Private
```

## How the Solver Works

1. **Board Detection**: OpenCV finds the largest square-ish contour in the screenshot, applies perspective warp to get a clean square. Falls back to Hough line detection, then center crop.

2. **Grid Size Inference**: Morphological operations isolate horizontal and vertical grid lines in the warped image. Counting line clusters determines N (4, 5, or 6).

3. **Cell Extraction**: The warped square is divided into an NxN grid. Each cell is cropped with a 15% inset (to avoid grid line artifacts), resized to 64x64, contrast-enhanced (CLAHE), and binarized (Otsu).

4. **Two-Tier OCR**:
   - **Tier A** (fast): Normalized cross-correlation against saved letter templates (~20-150ms total). Used when templates exist.
   - **Tier B** (fallback): EasyOCR on cells where Tier A confidence is below 0.75 (~200-1200ms for uncertain cells only).
   - Common confusions are auto-corrected: 0→O, 1→I, 5→S.
   - Q is always treated as QU (standard Boggle rule).

5. **Trie + DFS Solver**: The dictionary is loaded into a prefix tree at startup. DFS explores all paths from every cell (8-directional adjacency), using bitmask tracking to prevent cell revisits. Branches are pruned when no dictionary word starts with the current prefix. Results are sorted longest-first, capped at 50.

6. **Push Notification**: Words of 4-5 letters are sent to ntfy.sh as a background task (doesn't block the response).

## Running Tests

```bash
python -m pytest tests/ -v
```

19 tests covering:
- Solver correctness, QU handling, no-revisit, sort order, performance
- Board detection with synthetic grid images
- OCR text cleaning, confusion mapping, template matching

## Project Structure

```
boggle/
  app/
    __init__.py
    server.py              FastAPI app with lifespan hooks
    settings.py            Dataclass config + env overrides
    board_detect.py        Contour/Hough detection + perspective warp
    cell_extract.py        Cell cropping + CLAHE/Otsu preprocessing
    recognition.py         Template matching + EasyOCR two-tier OCR
    solver.py              Trie + DFS solver with bitmask visited
    notifier.py            ntfy.sh async push notifications
    metrics.py             Per-stage timing utility
  scripts/
    calibration.py         CLI calibration + template capture tool
  templates/
    letters/               Letter template PNGs (populated by calibration)
  tests/
    test_solver.py         Solver unit tests
    test_board_detect.py   Board detection tests
    test_recognition.py    OCR/recognition tests
  dictionary.txt           TWL06 Scrabble word list (~137k words)
  requirements.txt         Python dependencies
  CLAUDE.md                Developer guide for Claude Code
  README.md                This file
```
