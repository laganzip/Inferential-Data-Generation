from __future__ import annotations

import argparse
import json
from pathlib import Path

from inferential_data_generation.main.base import GenerationConfig, OutputConfig, SamplingConfig
from inferential_data_generation.pipelines.generate import ModularGenerator
import inferential_data_generation.data_types  # noqa: F401
import inferential_data_generation.predictors  # noqa: F401
import inferential_data_generation.scenes  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate inferential time-series datasets.")
    parser.add_argument("--task", default=None, help="Backward-compatible alias for --scene.")
    parser.add_argument("--scene", default="temporal_physical_event", help="Scene name.")
    parser.add_argument("--data-type", default="ts_cov_text", help="Data type name.")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to generate.")
    parser.add_argument(
        "--output",
        default="data/temporal_physical_event_samples.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--seq-len", type=int, default=384, help="Total sequence length.")
    parser.add_argument("--history-len", type=int, default=192, help="History window length.")
    parser.add_argument(
        "--sampling-minutes",
        type=int,
        default=15,
        help="Sampling frequency in minutes.",
    )
    parser.add_argument(
        "--predictor",
        default="chronos2",
        help="Initial predictor registered in the predictor registry.",
    )
    parser.add_argument(
        "--context-generation-mode",
        default="template",
        choices=["template", "llm"],
        help="How to generate context_description.",
    )
    parser.add_argument("--llm-base-url", default=None, help="OpenAI-compatible base URL.")
    parser.add_argument("--llm-api-key", default=None, help="API key for the external LLM.")
    parser.add_argument("--llm-model", default=None, help="Model name for the external LLM.")
    parser.add_argument(
        "--llm-timeout-seconds",
        type=int,
        default=60,
        help="Timeout for each external LLM request.",
    )
    parser.add_argument(
        "--disable-context-fallback",
        action="store_true",
        help="Fail immediately if LLM context generation is enabled but unavailable.",
    )
    parser.add_argument(
        "--chronos2-model-path",
        default="/data/yichenglu/pre_train_model/Chronos2",
        help="Local path used when predictor=chronos2.",
    )
    parser.add_argument(
        "--predictor-device",
        default=None,
        help="Optional device override such as cpu or cuda:0.",
    )
    parser.add_argument(
        "--disable-predictor-fallback",
        action="store_true",
        help="Fail immediately if Chronos2 runtime is unavailable.",
    )
    parser.add_argument("--seed", type=int, default=20260402, help="Random seed.")

    parser.add_argument(
        "--exclude-covariate-values",
        action="store_true",
        help="Do not include history/future covariate values in output.",
    )
    parser.add_argument(
        "--exclude-context-description",
        action="store_true",
        help="Do not include context_description in output.",
    )
    parser.add_argument(
        "--exclude-structured-event-context",
        action="store_true",
        help="Do not include structured_event_context in output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sampling = SamplingConfig(
        seq_len=args.seq_len,
        history_len=args.history_len,
        sampling_minutes=args.sampling_minutes,
    )
    scene_name = args.task or args.scene
    config = GenerationConfig(
        num_samples=args.num_samples,
        sampling=sampling,
        seed=args.seed,
        predictor_name=args.predictor,
        chronos2_model_path=args.chronos2_model_path,
        predictor_device=args.predictor_device,
        allow_predictor_fallback=not args.disable_predictor_fallback,
        context_generation_mode=args.context_generation_mode,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        llm_model=args.llm_model,
        llm_timeout_seconds=args.llm_timeout_seconds,
        allow_context_fallback=not args.disable_context_fallback,
        scene_name=scene_name,
        data_type_name=args.data_type,
        output=OutputConfig(
            include_covariate_values=not args.exclude_covariate_values,
            include_context_description=not args.exclude_context_description,
            include_structured_event_context=not args.exclude_structured_event_context,
        ),
    )
    generator = ModularGenerator(config)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for sample_id in range(args.num_samples):
            bundle = generator.generate_sample_bundle(sample_id=sample_id)
            sample = {
                **bundle.public_sample,
                "history_target_values": bundle.history_target,
                "future_target_values": bundle.future_target,
            }
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Generated {args.num_samples} samples to {output_path}")


if __name__ == "__main__":
    main()
