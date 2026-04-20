from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from typing import Any


@dataclass(frozen=True)
class SamplingConfig:
    seq_len: int = 384
    history_len: int = 192
    sampling_minutes: int = 15

    @property
    def forecast_len(self) -> int:
        return self.seq_len - self.history_len


@dataclass(frozen=True)
class OutputConfig:
    include_covariate_values: bool = True
    include_context_description: bool = True
    include_structured_event_context: bool = True


@dataclass(frozen=True)
class GenerationConfig:
    num_samples: int
    sampling: SamplingConfig
    seed: int | None = None
    predictor_name: str = "chronos2"
    chronos2_model_path: str = "/data/yichenglu/pre_train_model/Chronos2"
    predictor_device: str | None = None
    allow_predictor_fallback: bool = True
    context_generation_mode: str = "template"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_timeout_seconds: int = 60
    allow_context_fallback: bool = True
    # New modular switches
    scene_name: str = "temporal_physical_event"
    data_type_name: str = "ts_cov_text"
    output: OutputConfig = OutputConfig()

    def __post_init__(self) -> None:
        if self.llm_base_url is None:
            object.__setattr__(self, "llm_base_url", os.getenv("OPENAI_BASE_URL"))
        if self.llm_api_key is None:
            object.__setattr__(self, "llm_api_key", os.getenv("OPENAI_API_KEY"))
        if self.llm_model is None:
            object.__setattr__(self, "llm_model", os.getenv("OPENAI_MODEL"))


@dataclass(frozen=True)
class SceneSample:
    history_target_values: list[float]
    future_target_values: list[float]
    history_covariate_values: list[float]
    future_covariate_values: list[float]
    target_name: str
    covariate_name: str
    context_description: str
    structured_event_context: list[dict[str, Any]]
    metadata: dict[str, Any]


class BaseTaskGenerator(ABC):
    task_name: str

    @abstractmethod
    def generate_sample(self, sample_id: int) -> dict:
        """Generate one serializable sample."""


class BaseInitialPredictor(ABC):
    predictor_name: str

    @abstractmethod
    def predict(
        self,
        history_target_values: list[float],
        history_context_values: list[float],
        future_context_values: list[float],
        metadata: dict[str, Any],
    ) -> list[float]:
        """Generate an initial forecast for the hidden target series."""


class BaseSceneGenerator(ABC):
    scene_name: str

    @abstractmethod
    def generate_scene_sample(self, sample_id: int) -> SceneSample:
        """Generate one scene-level sample before data-type assembly."""


class BaseDataTypeAssembler(ABC):
    data_type_name: str

    @abstractmethod
    def build_public_sample(
        self,
        *,
        scene_sample: SceneSample,
        initial_prediction_values: list[float],
        predictor_name: str,
        config: GenerationConfig,
    ) -> dict[str, Any]:
        """Build final JSON sample fields for one data type."""
