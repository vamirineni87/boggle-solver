import logging

from fastapi import FastAPI, File, Request, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse

from app.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("boggle")

# These will be populated at startup
_trie = None
_templates = None
_easyocr_reader = None


def create_app() -> FastAPI:
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        global _trie, _templates, _easyocr_reader

        import torch
        torch.set_num_threads(settings.TORCH_NUM_THREADS)

        from app.solver import load_trie
        dict_path = settings.DICTIONARY_COMMON_PATH if settings.COMMON_WORDS_ONLY else settings.DICTIONARY_PATH
        logger.info("Loading dictionary from %s (common_only=%s)", dict_path, settings.COMMON_WORDS_ONLY)
        _trie = load_trie(str(dict_path), settings.MIN_WORD_LENGTH)
        logger.info("Trie loaded")

        from app.recognition import load_templates, init_easyocr
        _templates = load_templates(str(settings.TEMPLATES_DIR))
        if _templates:
            logger.info("Loaded %d letter templates", len(_templates))
        else:
            logger.warning("No letter templates found — Tier A disabled")

        _easyocr_reader = init_easyocr()
        logger.info("EasyOCR reader initialized")

        yield

    application = FastAPI(title="Boggle Solver", lifespan=lifespan)

    @application.get("/health")
    async def health():
        return {"status": "ok", "trie_loaded": _trie is not None}

    @application.post("/solve")
    async def solve(
        request: Request,
        background_tasks: BackgroundTasks,
        file: UploadFile = File(None),
    ):
        import cv2
        import numpy as np
        from app.metrics import StageTimer
        from app.board_detect import detect_board_and_warp, infer_grid_size
        from app.cell_extract import split_cells
        from app.recognition import recognize_cells
        from app.solver import solve as solve_board
        from app.notifier import send_notification

        content_type = request.headers.get("content-type", "")
        logger.info("POST /solve content-type=%s", content_type)

        data = None
        if file is not None and file.filename:
            # Standard multipart form upload
            logger.info("Received file: name=%s, type=%s", file.filename, file.content_type)
            if file.content_type and not file.content_type.startswith("image/"):
                raise HTTPException(400, f"Only image files accepted, got: {file.content_type}")
            data = await file.read()
        else:
            # Fallback: read raw body (iOS Shortcut may send image directly)
            data = await request.body()
            logger.info("No 'file' field — reading raw body (%d bytes)", len(data))

        if not data:
            raise HTTPException(400, "Empty request body — no image data received")

        timer = StageTimer()

        with timer.stage("decode"):
            logger.info("Received %d bytes", len(data))
            if len(data) > settings.MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"File too large (max {settings.MAX_UPLOAD_BYTES} bytes)")
            arr = np.frombuffer(data, dtype=np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if image is None:
                raise HTTPException(400, "Could not decode image")

        with timer.stage("board_detect"):
            warped, debug_info = detect_board_and_warp(image, settings.WARP_SIZE)

        with timer.stage("grid_infer"):
            grid_size = infer_grid_size(warped)

        with timer.stage("cell_split"):
            cells = split_cells(warped, grid_size, settings.CELL_INSET)

        with timer.stage("recognize"):
            board, confidences = recognize_cells(
                cells, _templates, _easyocr_reader,
                settings.OCR_CONFIDENCE_THRESHOLD,
                warped_gray=warped,
                grid_size=grid_size,
            )

        # Log detected board for debugging
        board_str = " / ".join(" ".join(row) for row in board)
        logger.info("Board %dx%d: %s", grid_size, grid_size, board_str)

        with timer.stage("solve"):
            all_words, word_positions = solve_board(board, grid_size, _trie, 0)

        words = all_words[:settings.MAX_RESULTS]
        logger.info("Found %d words (returning top %d)", len(all_words), len(words))

        # Send notification in background (with ALL words so 4-5 letter filter works)
        background_tasks.add_task(
            send_notification, all_words, grid_size, board,
            timer.summary(), settings.NTFY_TOPIC, settings.NTFY_URL,
            settings.NOTIFY_WORDS_PER_GROUP, word_positions
        )

        _save_debug_artifacts(image, warped, cells, board, confidences, all_words, timer, grid_size, debug_info)

        return JSONResponse({
            "grid_size": grid_size,
            "board": board,
            "words": words,
            "word_count": len(words),
            "processing_time": timer.total_ms,
            "stage_timings": timer.summary(),
            "cell_confidences": confidences,
        })

    @application.post("/debug/cells")
    async def debug_cells(file: UploadFile = File(...)):
        import cv2
        import numpy as np
        from io import BytesIO
        from fastapi.responses import StreamingResponse
        from app.board_detect import detect_board_and_warp, infer_grid_size
        from app.cell_extract import split_cells, make_cell_montage

        data = await file.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(400, "Could not decode image")

        warped, _ = detect_board_and_warp(image, settings.WARP_SIZE)
        grid_size = infer_grid_size(warped)
        cells = split_cells(warped, grid_size, settings.CELL_INSET)
        montage = make_cell_montage(cells, grid_size)

        _, buf = cv2.imencode(".png", montage)
        return StreamingResponse(BytesIO(buf.tobytes()), media_type="image/png")

    @application.get("/api/settings")
    async def api_get_settings():
        from app.settings import get_editable_settings, EDITABLE_FIELDS
        values = get_editable_settings(settings)
        field_types = {k: v.__name__ for k, v in EDITABLE_FIELDS.items()}
        return JSONResponse({"settings": values, "field_types": field_types})

    @application.post("/api/settings")
    async def api_post_settings(request: Request):
        from app.settings import update_settings, get_editable_settings
        body = await request.json()
        errors = update_settings(settings, **body)
        if errors:
            return JSONResponse({"updated": get_editable_settings(settings), "errors": errors}, status_code=400)
        logger.info("Settings updated: %s", body)
        return JSONResponse({"updated": get_editable_settings(settings)})

    @application.get("/settings", response_class=HTMLResponse)
    async def settings_page():
        return SETTINGS_HTML

    return application


SETTINGS_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Boggle Settings</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #0f172a; color: #e2e8f0; padding: 16px; max-width: 480px; margin: 0 auto; }
  h1 { font-size: 1.4rem; margin-bottom: 16px; color: #38bdf8; }
  .field { background: #1e293b; border-radius: 10px; padding: 14px; margin-bottom: 10px;
           display: flex; justify-content: space-between; align-items: center; }
  .field-info { flex: 1; margin-right: 12px; }
  .field-name { font-size: 0.85rem; font-weight: 600; color: #94a3b8; }
  .field-desc { font-size: 0.75rem; color: #64748b; margin-top: 2px; }
  input[type="number"], input[type="text"] {
    width: 90px; padding: 8px; border-radius: 8px; border: 1px solid #334155;
    background: #0f172a; color: #e2e8f0; font-size: 1rem; text-align: right; }
  input[type="number"]:focus, input[type="text"]:focus { outline: none; border-color: #38bdf8; }
  .toggle { position: relative; width: 50px; height: 28px; flex-shrink: 0; }
  .toggle input { opacity: 0; width: 0; height: 0; }
  .toggle .slider { position: absolute; inset: 0; background: #334155; border-radius: 28px;
                    cursor: pointer; transition: background 0.2s; }
  .toggle .slider::before { content: ""; position: absolute; left: 3px; top: 3px;
    width: 22px; height: 22px; background: #e2e8f0; border-radius: 50%; transition: transform 0.2s; }
  .toggle input:checked + .slider { background: #38bdf8; }
  .toggle input:checked + .slider::before { transform: translateX(22px); }
  .btn { display: block; width: 100%; padding: 14px; margin-top: 16px; border: none;
         border-radius: 10px; background: #38bdf8; color: #0f172a; font-size: 1rem;
         font-weight: 700; cursor: pointer; transition: opacity 0.2s; }
  .btn:active { opacity: 0.7; }
  .toast { position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
           padding: 10px 20px; border-radius: 8px; font-size: 0.9rem; font-weight: 600;
           opacity: 0; transition: opacity 0.3s; pointer-events: none; z-index: 10; }
  .toast.ok { background: #22c55e; color: #fff; }
  .toast.err { background: #ef4444; color: #fff; }
  .toast.show { opacity: 1; }
</style>
</head>
<body>
<h1>Boggle Settings</h1>
<div id="fields"></div>
<button class="btn" onclick="save()">Save</button>
<div id="toast" class="toast"></div>
<script>
const DESC = {
  NOTIFY_WORDS_PER_GROUP: "Words per length group in notifications",
  MAX_RESULTS: "Max words in JSON response",
  MIN_WORD_LENGTH: "Minimum word length to solve",
  COMMON_WORDS_ONLY: "Use common dictionary subset",
  DEBUG: "Save debug artifacts per request",
  NTFY_TOPIC: "ntfy.sh topic name",
  OCR_CONFIDENCE_THRESHOLD: "OCR confidence cutoff (0-1)"
};
let fieldTypes = {};

function esc(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

async function load() {
  const r = await fetch("/api/settings");
  const d = await r.json();
  fieldTypes = d.field_types;
  const c = document.getElementById("fields");
  c.innerHTML = "";
  for (const [k, v] of Object.entries(d.settings)) {
    const ft = fieldTypes[k];
    const div = document.createElement("div");
    div.className = "field";
    const info = `<div class="field-info"><div class="field-name">${k}</div><div class="field-desc">${DESC[k] || ""}</div></div>`;
    let input;
    if (ft === "bool") {
      input = `<label class="toggle"><input type="checkbox" id="f_${k}" ${v ? "checked" : ""}><span class="slider"></span></label>`;
    } else if (ft === "int") {
      input = `<input type="number" id="f_${k}" value="${esc(v)}" step="1">`;
    } else if (ft === "float") {
      input = `<input type="number" id="f_${k}" value="${esc(v)}" step="0.05">`;
    } else {
      input = `<input type="text" id="f_${k}" value="${esc(v)}" style="width:140px">`;
    }
    div.innerHTML = info + input;
    c.appendChild(div);
  }
}

function gather() {
  const data = {};
  for (const [k, ft] of Object.entries(fieldTypes)) {
    const el = document.getElementById("f_" + k);
    if (!el) continue;
    if (ft === "bool") data[k] = el.checked;
    else if (ft === "int") data[k] = parseInt(el.value, 10);
    else if (ft === "float") data[k] = parseFloat(el.value);
    else data[k] = el.value;
  }
  return data;
}

function toast(msg, ok) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.className = "toast " + (ok ? "ok" : "err") + " show";
  setTimeout(() => t.className = "toast", 2000);
}

async function save() {
  try {
    const r = await fetch("/api/settings", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(gather())
    });
    const d = await r.json();
    if (r.ok) toast("Saved!", true);
    else toast("Error: " + JSON.stringify(d.errors), false);
  } catch (e) { toast("Network error", false); }
}

load();
</script>
</body>
</html>
"""


def _save_debug_artifacts(image, warped, cells, board, confidences, words, timer, grid_size, debug_info):
    import cv2
    import json
    from datetime import datetime
    from pathlib import Path
    from app.cell_extract import make_cell_montage

    from app.settings import settings as _s
    debug_dir = _s.BASE_DIR / "debug"
    debug_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    cv2.imwrite(str(debug_dir / f"{ts}_orig.png"), image)
    cv2.imwrite(str(debug_dir / f"{ts}_warp.png"), warped)

    montage = make_cell_montage(cells, grid_size)
    cv2.imwrite(str(debug_dir / f"{ts}_cells.png"), montage)

    result = {
        "timestamp": ts,
        "grid_size": grid_size,
        "board": board,
        "confidences": confidences,
        "detection_method": debug_info.get("method"),
        "word_count": len(words),
        "words": words,
        "timings": timer.summary(),
        "total_ms": timer.total_ms,
    }
    with open(debug_dir / f"{ts}_result.json", "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Saved debug artifacts to debug/%s_*", ts)


app = create_app()
