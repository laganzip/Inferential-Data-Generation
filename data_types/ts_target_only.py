from __future__ import annotations

from typing import Any

from inferential_data_generation.base import BaseDataTypeAssembler, GenerationConfig, SceneSample


class TimeSeriesTargetOnlyAssembler(BaseDataTypeAssembler):
    data_type_name = "ts_target_only"

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config

    def build_public_sample(
        self,
        *,
        scene_sample: SceneSample,
        initial_prediction_values: list[float],
        predictor_name: str,
        config: GenerationConfig,
    ) -> dict[str, Any]:
        # Target-only data type: excludes covariate values and narrative text by design.
        return {
            "data_type": self.data_type_name,
            "target_name": scene_sample.target_name,
            "initial_prediction_values": [round(v, 1) for v in initial_prediction_values],
            "predictor_name": predictor_name,
            "sampling_minutes": config.sampling.sampling_minutes,
            "steps_per_day": (24 * 60) // config.sampling.sampling_minutes,
        }
