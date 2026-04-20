from __future__ import annotations

from typing import Any

from inferential_data_generation.main.llm_correction import LLMCorrectionGenerator
from inferential_data_generation.models.adapters.base import BaseCorrectionAdapter


class LLMCorrectionAdapter(BaseCorrectionAdapter):
    def __init__(self, generator: LLMCorrectionGenerator, model_name: str | None) -> None:
        self._generator = generator
        self._model_name = model_name

    @property
    def model_name(self) -> str | None:
        return self._model_name

    def generate(self, sample: dict[str, Any]) -> dict[str, Any]:
        return self._generator.generate(sample)


class NoopCorrectionAdapter(BaseCorrectionAdapter):
    @property
    def model_name(self) -> str | None:
        return None

    def generate(self, sample: dict[str, Any]) -> dict[str, Any]:
        del sample
        return {}
