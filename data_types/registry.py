from __future__ import annotations

from typing import Callable

from inferential_data_generation.main.base import GenerationConfig

DATA_TYPE_REGISTRY: dict[str, Callable[[GenerationConfig], object]] = {}


def register_data_type(name: str, builder: Callable[[GenerationConfig], object]) -> None:
    DATA_TYPE_REGISTRY[name] = builder


def build_data_type(name: str, config: GenerationConfig):
    if name not in DATA_TYPE_REGISTRY:
        available = ", ".join(sorted(DATA_TYPE_REGISTRY))
        raise ValueError(f"Unknown data type '{name}'. Available data types: {available}")
    return DATA_TYPE_REGISTRY[name](config)
