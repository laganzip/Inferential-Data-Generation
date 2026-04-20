from __future__ import annotations

from dataclasses import dataclass

from inferential_data_generation.main.base import GenerationConfig
from inferential_data_generation.data_types.basic import build_data_type_assembler
from inferential_data_generation.models.basic import build_initial_predictor_adapter
from inferential_data_generation.scenes.registry import build_scene


@dataclass(frozen=True)
class SampleBundle:
    public_sample: dict
    history_target: list[float]
    future_target: list[float]
    future_initial_prediction: list[float]
    future_corrected_prediction: list[float]


class ModularGenerator:
    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self.scene = build_scene(config.scene_name, config)
        self.data_type = build_data_type_assembler(config)
        self.predictor = build_initial_predictor_adapter(config)

    def generate_sample(self, sample_id: int) -> dict:
        return self.generate_sample_bundle(sample_id).public_sample

    def generate_sample_bundle(self, sample_id: int) -> SampleBundle:
        scene_sample = self.scene.generate_scene_sample(sample_id)
        initial_prediction = self.predictor.predict(
            history_target_values=scene_sample.history_target_values,
            history_covariate_values=scene_sample.history_covariate_values,
            future_covariate_values=scene_sample.future_covariate_values,
            metadata={
                **scene_sample.metadata,
                "target_name": scene_sample.target_name,
                "context_name": scene_sample.covariate_name,
            },
        )
        rounded_prediction = [round(v, 1) for v in initial_prediction]
        correction_delta = [
            round(truth - pred, 1)
            for pred, truth in zip(rounded_prediction, scene_sample.future_target_values)
        ]
        future_corrected_prediction = [
            round(pred + delta, 1) for pred, delta in zip(rounded_prediction, correction_delta)
        ]

        public_sample = self.data_type.build_public_sample(
            scene_sample=scene_sample,
            initial_prediction_values=rounded_prediction,
            predictor_name=self.predictor.model_name,
            config=self.config,
        )
        public_sample["correction_delta_values"] = correction_delta

        return SampleBundle(
            public_sample=public_sample,
            history_target=scene_sample.history_target_values,
            future_target=scene_sample.future_target_values,
            future_initial_prediction=rounded_prediction,
            future_corrected_prediction=future_corrected_prediction,
        )
