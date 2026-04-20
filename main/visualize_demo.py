from __future__ import annotations

import argparse
from pathlib import Path
import textwrap
import unicodedata

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib import font_manager
import numpy as np

from types import SimpleNamespace

from inferential_data_generation.main.base import GenerationConfig, SamplingConfig
from inferential_data_generation.pipelines.generate import ModularGenerator
import inferential_data_generation.data_types  # noqa: F401
import inferential_data_generation.predictors  # noqa: F401
import inferential_data_generation.scenes  # noqa: F401


DEFAULT_FONT_CANDIDATES = [
    "/data/yichenglu/fonts/siyuanheiti/SourceHanSansCN-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate matplotlib demo plots for temporal reasoning samples.")
    parser.add_argument("--task", default=None, help="Backward-compatible alias for --scene.")
    parser.add_argument("--scene", default="temporal_physical_event", help="Scene name.")
    parser.add_argument("--data-type", default="ts_cov_text", help="Data type name.")
    parser.add_argument("--num-plots", type=int, default=4, help="Number of samples to visualize.")
    parser.add_argument("--output-dir", default="data/temporal_reasoning_demo_plots", help="Directory for demo plots.")
    parser.add_argument("--output-format", default="png", choices=["png", "svg"], help="Image format for saved plots.")
    parser.add_argument("--font-path", default=None, help="Optional font path for rendering Chinese text.")
    parser.add_argument("--title-width", type=int, default=65, help="Wrap width used for the long title text.")
    parser.add_argument("--seq-len", type=int, default=384)
    parser.add_argument("--history-len", type=int, default=192)
    parser.add_argument("--sampling-minutes", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260402)
    parser.add_argument("--predictor", default="chronos2")
    parser.add_argument("--context-generation-mode", default="template", choices=["template", "llm"])
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-timeout-seconds", type=int, default=60)
    parser.add_argument("--disable-context-fallback", action="store_true")
    parser.add_argument("--chronos2-model-path", default="/data/yichenglu/pre_train_model/Chronos2")
    parser.add_argument("--predictor-device", default=None)
    parser.add_argument("--disable-predictor-fallback", action="store_true")
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
        num_samples=args.num_plots,
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
    )
    generator = ModularGenerator(config)
    font_prop = resolve_font(args.font_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(args.num_plots):
        bundle = generator.generate_sample_bundle(idx)
        figure = build_figure(bundle, idx, sampling, font_prop=font_prop, title_width=args.title_width)
        output_path = output_dir / f"sample_{idx:02d}.{args.output_format}"
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)

    print(f"Generated {args.num_plots} {args.output_format.upper()} plots to {output_dir}")


def build_figure(
    bundle,
    idx: int,
    sampling: SamplingConfig,
    *,
    font_prop: font_manager.FontProperties | None,
    title_width: int,
) -> plt.Figure:
    return build_figure_from_series(
        history_target=bundle.history_target,
        future_target=bundle.future_target,
        future_initial_prediction=bundle.future_initial_prediction,
        future_corrected_prediction=bundle.future_corrected_prediction,
        public_sample=bundle.public_sample,
        idx=idx,
        sampling=sampling,
        font_prop=font_prop,
        title_width=title_width,
    )


def build_figure_from_series(
    *,
    history_target: list[float],
    future_target: list[float],
    future_initial_prediction: list[float],
    future_corrected_prediction: list[float],
    public_sample: dict,
    idx: int,
    sampling: SamplingConfig,
    font_prop: font_manager.FontProperties | None,
    title_width: int,
) -> plt.Figure:
    seq = np.asarray(history_target, dtype=float)
    true = np.asarray(future_target, dtype=float)
    predicts = np.asarray(future_initial_prediction, dtype=float)
    changed_pred = np.asarray(future_corrected_prediction, dtype=float)

    mse_org = float(np.mean((true - predicts) ** 2))
    mse_changed = float(np.mean((true - changed_pred) ** 2))

    mae_org = float(np.mean(np.abs(true - predicts)))
    mae_changed = float(np.mean(np.abs(true - changed_pred)))

    context = public_sample.get("context_description", "（当前 data_type 未提供上下文文本）")
    reference_reasoning = public_sample.get("reference_reasoning_chain", "")
    reasoning = public_sample.get("correction_reasoning_chain", "")
    raw_title = (
        f"样本 {idx}\n"
        f"| 上下文: {context}\n"
        f"| 参考推理链: {reference_reasoning}\n"
        f"| 修正模型推理链: {reasoning}"
    )

    def wrap_text_visual(text, width):
        lines, cur_line, cur_w = [], "", 0
        for char in text:
            if char == '\n':
                lines.append(cur_line)
                cur_line, cur_w = "", 0
                continue

            w = 2 if unicodedata.east_asian_width(char) in ('F', 'W', 'A') else 1

            if cur_w + w > width:
                lines.append(cur_line)
                cur_line, cur_w = char, w
            else:
                cur_line += char
                cur_w += w

        if cur_line: 
            lines.append(cur_line)
        return lines

    title = "\n".join(wrap_text_visual(raw_title, width=title_width))
    title_lines = max(title.count("\n") + 1, 1)

    figure_height = 6.0 + 0.36 * max(title_lines - 2, 0)
    fig, ax = plt.subplots(figsize=(13.5, figure_height))

    future_x = np.arange(len(seq), len(seq) + len(true))
    full_values = np.concatenate([seq, true, predicts, changed_pred])
    y_margin = max((float(full_values.max()) - float(full_values.min())) * 0.1, 8.0)

    ax.plot(np.arange(len(seq)), seq, label="seq", color="#375a7f", linewidth=2.0)
    ax.plot(future_x, true, label="true", color="#2e8b57", linewidth=2.0)
    predictor_label = f"{public_sample['predictor_name']}_pred"
    ax.plot(future_x, predicts, label=f"{predictor_label} (MSE: {mse_org:.2f}, MAE: {mae_org:.2f})", color="#222222", linewidth=2.0)
    ax.plot(
        future_x,
        changed_pred,
        label=f"changed_pred (MSE: {mse_changed:.2f}, MAE: {mae_changed:.2f})",
        color="#d04a36",
        linewidth=2.0,
    )

    ax.axvline(len(seq) - 0.5, color="#7b4b4b", linestyle="--", linewidth=1.5, alpha=0.85)
    ax.axvspan(len(seq) - 0.5, len(seq) + len(true) - 0.5, color="#fff3ed", alpha=0.45)
    ax.set_xlim(0, sampling.seq_len - 1)
    ax.set_ylim(float(full_values.min()) - y_margin, float(full_values.max()) + y_margin)
    ax.grid(True, color="#d9d9d9", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("time index", fontproperties=font_prop)
    ax.set_ylabel(public_sample["target_name"], fontproperties=font_prop)
    ax.legend(prop=font_prop, loc="upper left", frameon=True)

    title_kwargs = {"fontsize": 13, "loc": "left", "pad": 18}
    if font_prop is not None:
        title_kwargs["fontproperties"] = font_prop
    ax.set_title(title, **title_kwargs)

    info_text = (
        f"predictor={public_sample['predictor_name']} | "
        f"context_field=history_context_values,future_context_values | "
        "initial_prediction + correction_delta = true"
    )
    if font_prop is not None:
        ax.text(
            0.0,
            -0.18,
            info_text,
            transform=ax.transAxes,
            fontsize=10.5,
            color="#444444",
            fontproperties=font_prop,
        )
    else:
        ax.text(
            0.0,
            -0.18,
            info_text,
            transform=ax.transAxes,
            fontsize=10.5,
            color="#444444",
        )

    top_margin = max(0.72, 0.92 - 0.035 * min(title_lines, 8))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=top_margin)
    return fig


def make_bundle_like(record: dict, *, corrected_prediction_key: str, reasoning_key: str):
    public_sample = {
        "context_description": record["context_description"],
        "reference_reasoning_chain": record.get("correction_reasoning_chain", ""),
        "correction_reasoning_chain": record.get(reasoning_key, ""),
        "target_name": record["target_name"],
        "predictor_name": record["predictor_name"],
    }
    return SimpleNamespace(
        history_target=record["history_target_values"],
        future_target=record["future_target_values"],
        future_initial_prediction=record["initial_prediction_values"],
        future_corrected_prediction=record[corrected_prediction_key],
        public_sample=public_sample,
    )


def wrap_text(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)


def resolve_font(font_path: str | None) -> font_manager.FontProperties | None:
    candidates = [font_path] if font_path else []
    candidates.extend(DEFAULT_FONT_CANDIDATES)
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return font_manager.FontProperties(fname=path)
    return None


if __name__ == "__main__":
    main()
