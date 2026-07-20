from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    # Qdrant REST
    qdrant_url: str = _env("QDRANT_URL", "http://localhost:6334")
    qdrant_collection: str = _env("QDRANT_COLLECTION", "flickr8k_images")

    # Data
    images_dir: str = _env("IMAGES_DIR", "./data/flickr8k/images")

    # Model
    model_name: str = _env("MODEL_NAME", "ViT-B-32")
    model_pretrained: str = _env("MODEL_PRETRAINED", "openai")
    device: str = _env("DEVICE", "cpu")  # "cpu" for docker by default
    use_fp16: bool = _env_bool("USE_FP16", False)

    # Defaults
    default_top_k: int = int(_env("DEFAULT_TOP_K", "5"))


SETTINGS = Settings()