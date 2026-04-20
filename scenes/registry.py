from __future__ import annotations

from typing import Callable

from inferential_data_generation.base import GenerationConfig

SCENE_REGISTRY: dict[str, Callable[[GenerationConfig], object]] = {}


def register_scene(name: str, builder: Callable[[GenerationConfig], object]) -> None:
    SCENE_REGISTRY[name] = builder


def build_scene(name: str, config: GenerationConfig):
    if name not in SCENE_REGISTRY:
        available = ", ".join(sorted(SCENE_REGISTRY))
        raise ValueError(f"Unknown scene '{name}'. Available scenes: {available}")
    return SCENE_REGISTRY[name](config)
