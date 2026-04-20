from __future__ import annotations

from typing import Any

from inferential_data_generation.main.base import BaseDataTypeAssembler, GenerationConfig, SceneSample


class TimeSeriesCovariateTextAssembler(BaseDataTypeAssembler):
    data_type_name = "ts_cov_text"

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
        record: dict[str, Any] = {
            "data_type": self.data_type_name,
            "target_name": scene_sample.target_name,
            "context_name": scene_sample.covariate_name,
            "initial_prediction_values": [round(v, 1) for v in initial_prediction_values],
            "predictor_name": predictor_name,
            "sampling_minutes": config.sampling.sampling_minutes,
            "steps_per_day": (24 * 60) // config.sampling.sampling_minutes,
            "forecast_days": config.sampling.forecast_len / ((24 * 60) // config.sampling.sampling_minutes),
            "forecast_horizon_text": (
                f"预测未来 {config.sampling.forecast_len} 个点，"
                f"每 {config.sampling.sampling_minutes} 分钟一个点。"
            ),
        }
        if config.output.include_covariate_values:
            record["history_context_values"] = [round(v, 1) for v in scene_sample.history_covariate_values]
            record["future_context_values"] = [round(v, 1) for v in scene_sample.future_covariate_values]

        if config.output.include_context_description:
            record["context_description"] = scene_sample.context_description

        if config.output.include_structured_event_context:
            record["structured_event_context"] = scene_sample.structured_event_context

        return record
