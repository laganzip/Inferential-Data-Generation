from __future__ import annotations

from typing import Any

from inferential_data_generation.main.base import GenerationConfig
from inferential_data_generation.models.adapters.base import BaseInitialPredictorAdapter
from inferential_data_generation.predictors.registry import build_predictor


class RegistryPredictorAdapter(BaseInitialPredictorAdapter):
    def __init__(self, config: GenerationConfig) -> None:
        self._predictor = build_predictor(config.predictor_name, config)

    @property
    def model_name(self) -> str:
        return self._predictor.predictor_name

    def predict(
        self,
        *,
        history_target_values: list[float],
        history_covariate_values: list[float],
        future_covariate_values: list[float],
        metadata: dict[str, Any],
    ) -> list[float]:
        return self._predictor.predict(
            history_target_values=history_target_values,
            history_context_values=history_covariate_values,
            future_context_values=future_covariate_values,
            metadata=metadata,
        )
