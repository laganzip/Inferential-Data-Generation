from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal

import numpy as np


SeriesMode = Literal["raw", "stats"]


class TESMultimodalRegressionSFTDataset:
    """把 TES 多模态回归数据（npz + json）拼成标准 SFT messages。"""

    REQUIRED_NPZ_KEYS = {
        "X_ts",
        "y",
        "meta",
        "keep_ts_features",
        "target_columns",
    }

    def __init__(
        self,
        npz_path: str | Path,
        json_path: str | Path,
        system_prompt: str = "You are a regression assistant. Return numeric predictions only.",
        series_mode: SeriesMode = "raw",
        float_precision: int = 4,
        assistant_precision: int = 2,
        raw_series_precision: int = 1,
    ) -> None:
        self.npz_path = Path(npz_path)
        self.json_path = Path(json_path)
        self.system_prompt = system_prompt
        self.series_mode = series_mode
        self.float_precision = float_precision
        self.assistant_precision = assistant_precision
        self.raw_series_precision = raw_series_precision

        if self.series_mode not in ("raw", "stats"):
            raise ValueError(f"series_mode 只能是 'raw' 或 'stats'，当前: {self.series_mode}")

        self._npz = np.load(self.npz_path, allow_pickle=True)
        self._validate_npz_keys()

        self.X_ts = self._npz["X_ts"]
        self.y = self._npz["y"]
        self.meta = self._npz["meta"]
        self.keep_ts_features = [str(x) for x in self._npz["keep_ts_features"].tolist()]
        self.target_columns = [str(x) for x in self._npz["target_columns"].tolist()]

        with self.json_path.open("r", encoding="utf-8") as f:
            self.meta_json = json.load(f)

        self.records = self.meta_json.get("records")
        if not isinstance(self.records, list):
            raise ValueError("json 必须包含 records: list")

        self._id2record: Dict[int, Dict[str, Any]] = {}
        for record in self.records:
            if "sample_id" not in record:
                raise ValueError("records 中每条都必须有 sample_id")
            sid = int(record["sample_id"])
            if sid in self._id2record:
                raise ValueError(f"sample_id 重复: {sid}")
            self._id2record[sid] = record

        self._validate_alignment()

    def _validate_npz_keys(self) -> None:
        missing = self.REQUIRED_NPZ_KEYS - set(self._npz.files)
        if missing:
            raise ValueError(f"npz 缺少关键字段: {sorted(missing)}")

    def _validate_alignment(self) -> None:
        n = len(self.X_ts)
        if self.y.shape[0] != n or self.meta.shape[0] != n:
            raise ValueError(
                f"样本数不一致: len(X_ts)={n}, len(y)={self.y.shape[0]}, len(meta)={self.meta.shape[0]}"
            )

        if len(self.records) != n:
            raise ValueError(f"json records 数量与 npz 不一致: {len(self.records)} vs {n}")
        declared_num_samples = self.meta_json.get("num_samples")
        if declared_num_samples is None or int(declared_num_samples) != n:
            raise ValueError(
                f"json num_samples 与 npz 不一致: {declared_num_samples} vs {n}"
            )

        for i in range(n):
            if i not in self._id2record:
                raise ValueError(f"json 缺失 sample_id={i}")

            rec_meta = self._id2record[i].get("meta")
            if not isinstance(rec_meta, dict):
                raise ValueError(f"sample_id={i} 的 meta 不是 dict")

            required_meta_fields = ("start", "end", "target_index")
            if any(field not in rec_meta for field in required_meta_fields):
                raise ValueError(f"sample_id={i} 的 meta 缺少字段: {required_meta_fields}")

            npz_triplet = tuple(int(x) for x in self.meta[i].tolist())
            json_triplet = (
                int(rec_meta["start"]),
                int(rec_meta["end"]),
                int(rec_meta["target_index"]),
            )
            if npz_triplet != json_triplet:
                raise ValueError(
                    f"sample_id={i} 元数据不一致: npz={npz_triplet}, json={json_triplet}"
                )

        if self.y.ndim != 2:
            raise ValueError(f"y 必须是二维数组，当前 shape={self.y.shape}")
        if self.y.shape[1] != len(self.target_columns):
            raise ValueError(
                f"y 列数与 target_columns 不一致: {self.y.shape[1]} vs {len(self.target_columns)}"
            )

    def __len__(self) -> int:
        return self.X_ts.shape[0]

    def _format_float(self, x: float) -> str:
        return f"{x:.{self.float_precision}f}"

    def _series_to_text_stats(self, ts: np.ndarray) -> str:
        # ts: [T, F]
        lines: List[str] = []
        for feat_idx, feat_name in enumerate(self.keep_ts_features):
            col = ts[:, feat_idx]
            line = (
                f"- {feat_name}: "
                f"last={self._format_float(float(col[-1]))}, "
                f"mean={self._format_float(float(col.mean()))}, "
                f"std={self._format_float(float(col.std()))}, "
                f"min={self._format_float(float(col.min()))}, "
                f"max={self._format_float(float(col.max()))}"
            )
            lines.append(line)
        return "\n".join(lines)

    def _series_to_text_raw(self, ts: np.ndarray) -> str:
        payload = {
            feat_name: [round(float(x), self.raw_series_precision) for x in ts[:, feat_idx].tolist()]
            for feat_idx, feat_name in enumerate(self.keep_ts_features)
        }
        return json.dumps(payload, ensure_ascii=False)

    def _build_user_content(self, idx: int) -> str:
        rec = self._id2record[idx]
        ts = self.X_ts[idx]
        text_description = str(rec.get("text_description", "")).strip()

        series_text = (
            self._series_to_text_stats(ts)
            if self.series_mode == "stats"
            else self._series_to_text_raw(ts)
        )

        schema = self.meta_json.get("schema", {})
        scene = schema.get("background description", "")
        task_definition = schema.get("task_definition", "")
        variable_description = schema.get("variable_description", "")
        target_columns_schema = schema.get("target_columns", self.target_columns)
        if not isinstance(target_columns_schema, list):
            target_columns_schema = self.target_columns

        return (
            # "Task: Predict regression targets from numerical time-series inputs and text covariates.\n"
            f"Task: {task_definition}\n"
            f"Scene: {scene}\n"
            f"Target: {target_columns_schema}\n"
            f"Input Variable(X): {self.keep_ts_features}\n"
            f"Variable Description: {variable_description}\n"
            "\n"
            "Numerical Time-Series Inputs (X, raw):\n"
            f"{series_text}\n"
            "\n"
            "Text Covariates:\n"
            f"{text_description}\n"
            "\n"
            "Return a JSON object only. Keys must exactly match Target Columns."
        )

    def _build_assistant_content(self, idx: int) -> str:
        values = self.y[idx]
        payload = {
            col: round(float(values[j]), self.assistant_precision)
            for j, col in enumerate(self.target_columns)
        }
        return json.dumps(payload, ensure_ascii=False)

    def build_sft_item(self, idx: int) -> Dict[str, Any]:
        return {
            "sample_id": idx,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self._build_user_content(idx)},
                {"role": "assistant", "content": self._build_assistant_content(idx)},
            ],
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"样本索引越界: {idx}")
        return self.build_sft_item(idx)


def sft_collate_fn(batch: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    items = list(batch)
    return {
        "sample_ids": [x["sample_id"] for x in items],
        "messages": [x["messages"] for x in items],
    }


def build_sft_dataloader(
    npz_path: str | Path,
    json_path: str | Path,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    system_prompt: str = "You are a regression assistant. Return numeric predictions only.",
    series_mode: SeriesMode = "raw",
    assistant_precision: int = 2,
    raw_series_precision: int = 1,
):
    from torch.utils.data import DataLoader

    dataset = TESMultimodalRegressionSFTDataset(
        npz_path=npz_path,
        json_path=json_path,
        system_prompt=system_prompt,
        series_mode=series_mode,
        assistant_precision=assistant_precision,
        raw_series_precision=raw_series_precision,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=sft_collate_fn,
    )
    return dataset, dataloader


def export_sft_jsonl(
    dataset: TESMultimodalRegressionSFTDataset,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for i in range(len(dataset)):
            f.write(json.dumps(dataset[i], ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TES 多模态回归数据 -> SFT messages")
    parser.add_argument("--npz", required=True, help="npz 文件路径")
    parser.add_argument("--json", required=True, help="json 文件路径")
    parser.add_argument("--out", required=True, help="输出 jsonl 路径")
    parser.add_argument("--series-mode", choices=["stats", "raw"], default="raw")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--raw-series-precision", type=int, default=1)
    args = parser.parse_args()

    dataset, dataloader = build_sft_dataloader(
        npz_path=args.npz,
        json_path=args.json,
        batch_size=args.batch_size,
        shuffle=False,
        series_mode=args.series_mode,
        raw_series_precision=args.raw_series_precision,
    )
    export_sft_jsonl(dataset, args.out)

    first_batch = next(iter(dataloader))
    print(f"dataset size={len(dataset)}")
    print(f"first batch size={len(first_batch['sample_ids'])}")
    print(json.dumps(dataset[0], ensure_ascii=False)[:800])
