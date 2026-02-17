import logging
import time
from contextlib import contextmanager

logger = logging.getLogger("boggle")


class StageTimer:
    """Collects per-stage timing for a single request."""

    def __init__(self):
        self.timings: dict[str, float] = {}
        self._start = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        yield
        elapsed = time.perf_counter() - t0
        self.timings[name] = round(elapsed * 1000, 1)  # ms
        logger.info("stage=%s elapsed=%.1fms", name, self.timings[name])

    @property
    def total_ms(self) -> float:
        return round((time.perf_counter() - self._start) * 1000, 1)

    def summary(self) -> dict:
        return {**self.timings, "total": self.total_ms}
