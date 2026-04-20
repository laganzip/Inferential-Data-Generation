from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random

from inferential_data_generation.main.base import GenerationConfig, OutputConfig, SamplingConfig
from inferential_data_generation.models.basic import build_correction_adapter
from inferential_data_generation.pipelines.generate import ModularGenerator
from inferential_data_generation.pipelines.llm_correction import apply_correction_to_records, load_jsonl
import inferential_data_generation.data_types  # noqa: F401
import inferential_data_generation.predictors  # noqa: F401
import inferential_data_generation.scenes  # noqa: F401

RANDOM_SCENE_POOL = [
    "temporal_physical_event",
    "data_center_cooling_event",
    "commercial_hvac_event",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate demo data and fill LLM correction results.")
    parser.add_argument("--task", default="temporal_physical_event", help="Backward-compatible alias for --scene.")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--data-type", default="ts_cov_text")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--output", default="data/temporal_physical_event_llm_demo.jsonl")
    parser.add_argument("--input-data", default=None, help="Existing JSONL file. If provided, skip generation.")
    parser.add_argument("--seq-len", type=int, default=384)
    parser.add_argument("--history-len", type=int, default=192)
    parser.add_argument("--sampling-minutes", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260402)
    parser.add_argument("--predictor", default="chronos2")
    parser.add_argument("--chronos2-model-path", default="/data/yichenglu/pre_train_model/Chronos2")
    parser.add_argument("--predictor-device", default=None)
    parser.add_argument("--disable-predictor-fallback", action="store_true")
    parser.add_argument("--context-generation-mode", default="template", choices=["template", "llm"])
    parser.add_argument("--context-llm-base-url", default=None)
    parser.add_argument("--context-llm-api-key", default=None)
    parser.add_argument("--context-llm-model", default=None)
    parser.add_argument("--context-llm-timeout-seconds", type=int, default=60)
    parser.add_argument("--disable-context-fallback", action="store_true")
    parser.add_argument("--correction-llm-base-url", default=None)
    parser.add_argument("--correction-llm-api-key", default=None)
    parser.add_argument("--correction-llm-model", default=None)
    parser.add_argument("--correction-llm-timeout-seconds", type=int, default=120)
    parser.add_argument("--correction-llm-max-retries", type=int, default=2)
    parser.add_argument("--disable-correction-fallback", action="store_true")
    parser.add_argument("--skip-llm-correction", action="store_true")
    parser.add_argument("--include-hidden-ground-truth", action="store_true")
    parser.add_argument("--exclude-covariate-values", action="store_true")
    parser.add_argument("--exclude-context-description", action="store_true")
    parser.add_argument("--exclude-structured-event-context", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.input_data:
        records = load_jsonl(Path(args.input_data))
    else:
        records = generate_records(args)

    correction_adapter = build_correction_adapter(
        skip_llm_correction=args.skip_llm_correction,
        base_url=args.correction_llm_base_url,
        api_key=args.correction_llm_api_key,
        model=args.correction_llm_model,
        timeout_seconds=args.correction_llm_timeout_seconds,
        max_retries=args.correction_llm_max_retries,
        allow_fallback=not args.disable_correction_fallback,
    )
    records = apply_correction_to_records(records, correction_adapter)

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} samples to {output_path}")


def generate_records(args: argparse.Namespace) -> list[dict]:
    sampling = SamplingConfig(
        seq_len=args.seq_len,
        history_len=args.history_len,
        sampling_minutes=args.sampling_minutes,
    )
    chosen_scene = args.scene or args.task
    base_config = GenerationConfig(
        num_samples=args.num_samples,
        sampling=sampling,
        seed=args.seed,
        predictor_name=args.predictor,
        chronos2_model_path=args.chronos2_model_path,
        predictor_device=args.predictor_device,
        allow_predictor_fallback=not args.disable_predictor_fallback,
        context_generation_mode=args.context_generation_mode,
        llm_base_url=args.context_llm_base_url,
        llm_api_key=args.context_llm_api_key,
        llm_model=args.context_llm_model,
        llm_timeout_seconds=args.context_llm_timeout_seconds,
        allow_context_fallback=not args.disable_context_fallback,
        scene_name="temporal_physical_event",
        data_type_name=args.data_type,
        output=OutputConfig(
            include_covariate_values=not args.exclude_covariate_values,
            include_context_description=not args.exclude_context_description,
            include_structured_event_context=not args.exclude_structured_event_context,
        ),
    )

    scene_generators: dict[str, ModularGenerator] = {}
    picker = random.Random(args.seed)

    if chosen_scene == "random":
        for idx, scene_name in enumerate(RANDOM_SCENE_POOL):
            scene_seed = None if args.seed is None else args.seed + (idx + 1) * 1000
            scene_generators[scene_name] = ModularGenerator(
                replace(base_config, seed=scene_seed, scene_name=scene_name)
            )
    else:
        scene_generators[chosen_scene] = ModularGenerator(replace(base_config, scene_name=chosen_scene))

    records: list[dict] = []
    for sample_id in range(args.num_samples):
        if chosen_scene == "random":
            selected_scene = picker.choice(RANDOM_SCENE_POOL)
        else:
            selected_scene = chosen_scene

        generator = scene_generators[selected_scene]
        bundle = generator.generate_sample_bundle(sample_id)
        record = build_demo_record(bundle, include_hidden_ground_truth=args.include_hidden_ground_truth)
        record["task_name"] = selected_scene
        records.append(record)

    return records


def build_demo_record(bundle, *, include_hidden_ground_truth: bool) -> dict:
    public = dict(bundle.public_sample)
    public.pop("correction_delta_values", None)
    public.pop("correction_reasoning_chain", None)
    record = {
        **public,
        "history_target_values": bundle.history_target,
        "future_target_values": bundle.future_target,
        "correction_llm_name": None,
    }
    if include_hidden_ground_truth:
        record["future_corrected_prediction_values"] = bundle.future_corrected_prediction
        record["ground_truth_correction_delta_values"] = [
            round(truth - pred, 1)
            for pred, truth in zip(bundle.future_initial_prediction, bundle.future_target)
        ]
    return record


if __name__ == "__main__":
    main()
