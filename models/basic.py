from __future__ import annotations

from inferential_data_generation.base import GenerationConfig
from inferential_data_generation.llm_correction import LLMCorrectionConfig, LLMCorrectionGenerator
from inferential_data_generation.models.adapters.correction import LLMCorrectionAdapter, NoopCorrectionAdapter
from inferential_data_generation.models.adapters.predictor import RegistryPredictorAdapter


def build_initial_predictor_adapter(config: GenerationConfig) -> RegistryPredictorAdapter:
    return RegistryPredictorAdapter(config)


def build_correction_adapter(
    *,
    skip_llm_correction: bool,
    base_url: str | None,
    api_key: str | None,
    model: str | None,
    timeout_seconds: int,
    max_retries: int,
    allow_fallback: bool,
):
    if skip_llm_correction:
        return NoopCorrectionAdapter()

    generator = LLMCorrectionGenerator(
        LLMCorrectionConfig(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            allow_fallback=allow_fallback,
        )
    )
    return LLMCorrectionAdapter(generator=generator, model_name=model)
