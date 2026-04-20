from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseInitialPredictorAdapter(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @abstractmethod
    def predict(
        self,
        *,
        history_target_values: list[float],
        history_covariate_values: list[float],
        future_covariate_values: list[float],
        metadata: dict[str, Any],
    ) -> list[float]:
        ...


class BaseCorrectionAdapter(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str | None:
        ...

    @abstractmethod
    def generate(self, sample: dict[str, Any]) -> dict[str, Any]:
        ...
