from __future__ import annotations

from typing import Callable

from inferential_data_generation.base import GenerationConfig

PREDICTOR_REGISTRY: dict[str, Callable[[GenerationConfig], object]] = {}


def register_predictor(name: str, builder: Callable[[GenerationConfig], object]) -> None:
    PREDICTOR_REGISTRY[name] = builder


def build_predictor(name: str, config: GenerationConfig):
    if name not in PREDICTOR_REGISTRY:
        available = ", ".join(sorted(PREDICTOR_REGISTRY))
        raise ValueError(f"Unknown predictor '{name}'. Available predictors: {available}")
    return PREDICTOR_REGISTRY[name](config)

