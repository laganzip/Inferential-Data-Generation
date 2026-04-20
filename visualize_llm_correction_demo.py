from __future__ import annotations

import argparse
import json
from pathlib import Path

from matplotlib import pyplot as plt

from inferential_data_generation.base import SamplingConfig
from inferential_data_generation.visualize_demo import build_figure_from_series, resolve_font


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize JSONL demo data with LLM correction results.")
    parser.add_argument("--input", default="data/temporal_physical_event_llm_demo.jsonl")
    parser.add_argument("--output-dir", default="data/temporal_reasoning_llm_demo_plots")
    parser.add_argument("--output-format", default="png", choices=["png", "svg"])
    parser.add_argument("--font-path", default=None)
    parser.add_argument("--title-width", type=int, default=170)
    parser.add_argument("--seq-len", type=int, default=384)
    parser.add_argument("--history-len", type=int, default=192)
    parser.add_argument("--sampling-minutes", type=int, default=15)
    parser.add_argument("--corrected-key", default="llm_corrected_prediction_values")
    parser.add_argument("--reasoning-key", default="llm_correction_reasoning_chain")
    parser.add_argument("--num-plots", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sampling = SamplingConfig(
        seq_len=args.seq_len,
        history_len=args.history_len,
        sampling_minutes=args.sampling_minutes,
    )
    font_prop = resolve_font(args.font_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(Path(args.input))
    if args.num_plots is not None:
        records = records[: args.num_plots]

    for idx, record in enumerate(records):
        if args.corrected_key not in record:
            raise KeyError(
                f"Record {idx} does not contain '{args.corrected_key}'. "
                "Run the LLM correction generation step first."
            )
        public_sample = {
            "context_description": record["context_description"],
            "reference_reasoning_chain": record.get("correction_reasoning_chain", ""),
            "correction_reasoning_chain": record.get(args.reasoning_key, ""),
            "target_name": record["target_name"],
            "predictor_name": record["predictor_name"],
        }
        figure = build_figure_from_series(
            history_target=record["history_target_values"],
            future_target=record["future_target_values"],
            future_initial_prediction=record["initial_prediction_values"],
            future_corrected_prediction=record[args.corrected_key],
            public_sample=public_sample,
            idx=idx,
            sampling=sampling,
            font_prop=font_prop,
            title_width=args.title_width,
        )
        predictor_name = sanitize_filename_part(record.get("predictor_name", "unknown_predictor"))
        correction_llm_name = sanitize_filename_part(record.get("correction_llm_name") or "no_correction_llm")
        output_path = output_dir / (
            f"sample_{idx:02d}__{predictor_name}__{correction_llm_name}.{args.output_format}"
        )
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)

    print(f"Generated {len(records)} visualizations to {output_dir}")


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sanitize_filename_part(value: str) -> str:
    cleaned = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    text = "".join(cleaned).strip("._")
    return text or "unknown"


if __name__ == "__main__":
    main()
