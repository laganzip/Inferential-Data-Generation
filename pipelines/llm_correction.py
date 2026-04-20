from __future__ import annotations

import json
from pathlib import Path

from inferential_data_generation.models.adapters.base import BaseCorrectionAdapter


def load_jsonl(path: Path) -> list[dict]:
    # Prefer strict JSONL first (one JSON object per line).
    with path.open("r", encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]
    try:
        return [json.loads(line) for line in lines]
    except json.JSONDecodeError:
        # Fallback: accept multi-line JSON object stream.
        text = "".join(lines)
        decoder = json.JSONDecoder()
        idx = 0
        size = len(text)
        records: list[dict] = []
        while idx < size:
            while idx < size and text[idx].isspace():
                idx += 1
            if idx >= size:
                break
            obj, end = decoder.raw_decode(text, idx)
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at position {idx}, got {type(obj).__name__}")
            records.append(obj)
            idx = end
        return records


def dump_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_correction_llm_input(record: dict) -> dict:
    forecast_length = len(record["initial_prediction_values"])
    sampling_minutes = int(record.get("sampling_minutes", 15))
    steps_per_day = int(record.get("steps_per_day", (24 * 60) // sampling_minutes))
    raw_events = record.get("structured_event_context", [])
    structured_event_context = []
    for event in raw_events:
        structured_event_context.append(
            {
                "event_type": event.get("event_type"),
                "start_text": event.get("start_text"),
                "duration_text": event.get("duration_text"),
                "target_direction": event.get("target_direction"),
                "impact_strength": event.get("impact_strength"),
                "narrative_hint": event.get("narrative_hint"),
                "future_start_index": event.get("future_start_index"),
                "future_end_index": event.get("future_end_index"),
                "future_duration_steps": event.get("future_duration_steps"),
                "expected_delta_sign": event.get("expected_delta_sign"),
            }
        )

    return {
        "target_name": record["target_name"],
        "context_name": record.get("context_name", "上下文变量"),
        "context_description": record.get("context_description", ""),
        "initial_prediction_values": record["initial_prediction_values"],
        "history_target_values": record["history_target_values"],
        "structured_event_context": structured_event_context,
        "sampling_minutes": sampling_minutes,
        "steps_per_day": steps_per_day,
        "forecast_days": forecast_length / steps_per_day,
        "forecast_horizon_text": (
            f"预测未来两天，共 {forecast_length} 个点，每 {sampling_minutes} 分钟一个点，每天 {steps_per_day} 个点。"
        ),
    }


def apply_correction_to_records(records: list[dict], correction_adapter: BaseCorrectionAdapter) -> list[dict]:
    if correction_adapter.model_name is None:
        return records

    output: list[dict] = []
    for record in records:
        updated = dict(record)
        llm_input = build_correction_llm_input(updated)
        llm_result = correction_adapter.generate(llm_input)
        updated["correction_llm_name"] = correction_adapter.model_name
        updated["llm_correction_delta_values"] = llm_result["correction_delta_values"]
        updated["llm_correction_reasoning_chain"] = llm_result["correction_reasoning_chain"]
        updated["llm_correction_generation_status"] = llm_result["correction_generation_status"]
        if "correction_error" in llm_result:
            updated["llm_correction_error"] = llm_result["correction_error"]
        updated["llm_corrected_prediction_values"] = [
            round(pred + delta, 1)
            for pred, delta in zip(
                updated["initial_prediction_values"],
                updated["llm_correction_delta_values"],
            )
        ]
        output.append(updated)
    return output
