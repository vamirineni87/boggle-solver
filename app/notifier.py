import logging
from collections import defaultdict

import httpx

logger = logging.getLogger("boggle")


async def send_notification(
    words: list[str],
    grid_size: int,
    board: list[list[str]],
    timings: dict,
    topic: str,
    ntfy_url: str = "https://ntfy.sh",
):
    """Send solve results to ntfy.sh. Best-effort — failures are logged, not raised."""
    try:
        by_length: dict[int, list[str]] = defaultdict(list)
        for w in words:
            by_length[len(w)].append(w)

        title = f"Boggle {grid_size}x{grid_size} - {len(words)} words"

        min_len = 3 if grid_size <= 5 else 4

        selected = []
        for length in sorted(by_length.keys()):
            if length < min_len:
                continue
            selected.extend(by_length[length][:10])

        counts = " | ".join(f"{l}L:{len(g)}" for l, g in sorted(by_length.items()) if l >= min_len)
        body = ",".join(selected) + "\n\n" + counts

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{ntfy_url}/{topic}",
                content=body.encode("utf-8"),
                headers={
                    "Title": title,
                    "Priority": "high",
                    "Tags": "game_die",
                },
            )
            resp.raise_for_status()
            logger.info("Notification sent to %s/%s (status %d)", ntfy_url, topic, resp.status_code)

    except Exception as e:
        logger.error("Failed to send notification: %s", e)
