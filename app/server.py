import logging

from fastapi import FastAPI, File, Request, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

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
            all_words = solve_board(board, grid_size, _trie, 0)

        words = all_words[:settings.MAX_RESULTS]
        logger.info("Found %d words (returning top %d)", len(all_words), len(words))

        # Send notification in background (with ALL words so 4-5 letter filter works)
        background_tasks.add_task(
            send_notification, all_words, grid_size, board,
            timer.summary(), settings.NTFY_TOPIC, settings.NTFY_URL
        )

        if settings.DEBUG:
            _save_debug_artifacts(image, warped, cells, board, words, timer)

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

    return application


def _save_debug_artifacts(image, warped, cells, board, words, timer):
    import cv2
    import json
    import time
    from pathlib import Path
    from app.cell_extract import make_cell_montage

    from app.settings import settings as _s
    debug_dir = _s.BASE_DIR / "debug"
    debug_dir.mkdir(exist_ok=True)
    ts = int(time.time())

    cv2.imwrite(str(debug_dir / f"req_{ts}_orig.jpg"), image)
    cv2.imwrite(str(debug_dir / f"req_{ts}_warp.png"), warped)

    n = int(len(cells) ** 0.5)
    montage = make_cell_montage(cells, n)
    cv2.imwrite(str(debug_dir / f"req_{ts}_cells.png"), montage)

    with open(debug_dir / f"req_{ts}_result.json", "w") as f:
        json.dump({"board": board, "words": words, "timings": timer.summary()}, f, indent=2)


app = create_app()
