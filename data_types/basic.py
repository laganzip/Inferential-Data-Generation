from __future__ import annotations

from inferential_data_generation.main.base import GenerationConfig
from inferential_data_generation.data_types.registry import build_data_type


def build_data_type_assembler(config: GenerationConfig):
    return build_data_type(config.data_type_name, config)
