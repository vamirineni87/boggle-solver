import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    DICTIONARY_PATH: Path = field(init=False)
    TEMPLATES_DIR: Path = field(init=False)

    NTFY_TOPIC: str = "boggle-solver"
    NTFY_URL: str = "https://ntfy.sh"

    MIN_WORD_LENGTH: int = 3
    MAX_RESULTS: int = 50

    CELL_INSET: float = 0.15
    OCR_CONFIDENCE_THRESHOLD: float = 0.75

    MAX_UPLOAD_BYTES: int = 5_000_000
    COMMON_WORDS_ONLY: bool = False
    DEBUG: bool = False

    TORCH_NUM_THREADS: int = 4
    WARP_SIZE: int = 400
    PORT: int = 10001

    DICTIONARY_COMMON_PATH: Path = field(init=False)

    def __post_init__(self):
        self.DICTIONARY_PATH = self.BASE_DIR / "dictionary.txt"
        self.DICTIONARY_COMMON_PATH = self.BASE_DIR / "dictionary_common.txt"
        self.TEMPLATES_DIR = self.BASE_DIR / "templates" / "letters"

        # Override from environment
        for fld in self.__dataclass_fields__:
            env_val = os.environ.get(fld)
            if env_val is not None:
                current = getattr(self, fld)
                if isinstance(current, bool):
                    setattr(self, fld, env_val.lower() in ("1", "true", "yes"))
                elif isinstance(current, int):
                    setattr(self, fld, int(env_val))
                elif isinstance(current, float):
                    setattr(self, fld, float(env_val))
                elif isinstance(current, Path):
                    setattr(self, fld, Path(env_val))
                else:
                    setattr(self, fld, env_val)


settings = Settings()
