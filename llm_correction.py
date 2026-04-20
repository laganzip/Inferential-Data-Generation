from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time
from typing import Any

import requests


@dataclass(frozen=True)
class LLMCorrectionConfig:
    base_url: str | None
    api_key: str | None
    model: str | None
    timeout_seconds: int = 120
    temperature: float = 0.1
    allow_fallback: bool = True
    max_retries: int = 2
    retry_backoff_seconds: float = 3.0


class LLMCorrectionGenerator:
    def __init__(self, config: LLMCorrectionConfig) -> None:
        self.config = config

    def generate(self, sample: dict[str, Any]) -> dict[str, Any]:
        expected_len = len(sample["initial_prediction_values"])
        if not self.config.base_url or not self.config.api_key or not self.config.model:
            error = "LLM correction generation requires base_url, api_key, and model."
            if self.config.allow_fallback:
                return self._fallback_result(expected_len, error)
            raise ValueError(error)

        prompt = self._build_prompt(sample)
        last_error: Exception | None = None

        for attempt in range(1, self.config.max_retries + 2):
            try:
                response_text = self._request(prompt)
                parsed = self._parse_response(response_text)
                values = self._render_piecewise_series(parsed["segments"], expected_len)
                return {
                    "correction_delta_values": values,
                    "correction_reasoning_chain": parsed["correction_reasoning_chain"],
                    "correction_generation_status": "ok",
                }
            except Exception as exc:
                last_error = exc
                print(
                    f"[LLMCorrection] attempt {attempt} failed for model={self.config.model}: {exc}"
                )
                if attempt <= self.config.max_retries:
                    time.sleep(self.config.retry_backoff_seconds * attempt)

        if self.config.allow_fallback:
            return self._fallback_result(expected_len, str(last_error))
        raise RuntimeError(f"LLM correction generation failed: {last_error}") from last_error

    def _request(self, prompt: str) -> str:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        body = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是一个时序修正助手。你的任务不是直接输出每一个时刻的全部修正值，"
                        "而是先给出少量分段修正计划，再由程序展开成完整序列。"
                        "你必须返回 JSON 对象，包含 segments 和 correction_reasoning_chain 两个字段。"
                        "segments 是按时间顺序排列的区间列表。每个区间都要给出 start、end、start_delta、end_delta、reason。"
                        "修正偏差定义为 true - initial_prediction，也就是初始预测 + 修正偏差 = 真实值。正值表示需要上调预测，负值表示需要下调预测。"
                        "输入里会显式给出采样频率、每天点数和预测时长；你要按这个时间结构理解索引含义，例如 15 分钟一个点、每天 96 个点、预测未来两天时，索引变化对应真实的日内节律与跨天事件演化。"
                        "在给出修正前，你要联合历史目标序列摘要和初始预测摘要进行分析：先判断初始预测已经捕捉到了哪些常规节律、趋势和基线变化，再判断它遗漏了哪些由上下文描述和结构化事件信息揭示的事件性扰动、强度变化或恢复过程。"
                        "修正幅度不能只给方向性微调，而要尽量匹配这些遗漏影响的真实量级；如果上下文描述的是明显扰动、持续高位压力、短时强控制或恢复速度显著变化，delta 幅度也应相应明显，不要习惯性压缩到很小的数值。"
                        "区间数量尽量控制在 3 到 8 段，优先描述关键事件切入、持续和退出过程。"
                        "correction_reasoning_chain 需要简要说明你如何根据上下文描述、结构化事件信息、历史目标摘要和初始预测摘要，分析初始预测已捕捉到什么、未捕捉到什么，并据此形成这些分段修正。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            url,
            headers=headers,
            json=body,
            timeout=self.config.timeout_seconds,
            proxies={"http": None, "https": None},
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def _build_prompt(self, sample: dict[str, Any]) -> str:
        forecast_len = len(sample["initial_prediction_values"])
        init_summary = self._series_summary(sample["initial_prediction_values"], "initial_prediction")
        history_summary = self._series_summary(sample["history_target_values"], "history_target")
        example = {
            "segments": [
                {
                    "start": 0,
                    "end": 23,
                    "start_delta": 0.0,
                    "end_delta": 12.0,
                    "reason": "事件开始前后逐步抬升",
                },
                {
                    "start": 24,
                    "end": 71,
                    "start_delta": 12.0,
                    "end_delta": 34.0,
                    "reason": "偏热和效率走弱叠加，修正持续加深",
                },
                {
                    "start": 72,
                    "end": 95,
                    "start_delta": 34.0,
                    "end_delta": -8.0,
                    "reason": "控制动作切入，短时压低负荷",
                },
            ],
            "correction_reasoning_chain": "先根据上下文描述与结构化事件信息判断扰动发生时段，再结合历史负荷摘要与初始预测摘要判断初始预测已覆盖了常规日周期，但低估了事件造成的额外抬升和控制动作带来的短时压低，因此需要给出与事件强度相匹配的较明显分段修正，而不是只做小幅微调。",
        }
        payload = {
            "target_name": sample["target_name"],
            "context_name": sample["context_name"],
            "forecast_length": forecast_len,
            "sampling_minutes": sample.get("sampling_minutes"),
            "steps_per_day": sample.get("steps_per_day"),
            "forecast_days": sample.get("forecast_days"),
            "forecast_horizon_text": sample.get("forecast_horizon_text"),
            "structured_event_context": sample.get("structured_event_context", []),
            "history_target_summary": history_summary,
            "history_target_dense_anchors": self._series_dense_anchors(sample["history_target_values"]),
            "initial_prediction_summary": init_summary,
            "initial_prediction_dense_anchors": self._series_dense_anchors(
                sample["initial_prediction_values"]
            ),
            "context_description": sample["context_description"],
        }
        return (
            "请基于以下输入输出修正计划 JSON。\n"
            "要求：\n"
            "1. 只输出 JSON，不要额外解释。\n"
            "2. 用 segments 描述少量关键区间，不要输出完整长度的点值列表。\n"
            "3. start 和 end 用未来预测区间的 0-based 索引，必须覆盖主要变化阶段。\n"
            "4. start_delta 和 end_delta 是该区间两端的修正偏差值，允许正负，数值保留到 1 位小数即可。\n"
            "5. 注意输入中给出的采样频率、每天点数和预测时长，按真实时间去理解各索引区间对应的日内和跨天阶段。\n"
            "6. 协变量相关信息只能来自 context_description 和结构化事件信息，不要假设任何未提供的协变量点值。\n"
            "7. 必须同时利用结构化事件信息、中文上下文描述以及目标序列锚点，不要只依赖概括性摘要。\n"
            "8. 先分析历史目标摘要与初始预测摘要的一致部分，说明初始预测已经捕捉到哪些常规模式；再分析结合上下文描述、结构化事件信息和更密锚点后，初始预测还遗漏了哪些事件性影响；最后再输出对应的分段修正。\n"
            "9. 修正量级要与遗漏影响的强度匹配；如果事件描述体现的是明显升温、持续偏高、强控制或恢复变慢，不要只输出接近 0 的小幅 delta。\n"
            # "5. 相邻区间的修正变化应尽量平滑，不要设计无依据的剧烈跳变。\n"
            f"输入: {json.dumps(payload, ensure_ascii=False)}\n"
            f"示例输出格式: {json.dumps(example, ensure_ascii=False)}"
        )

    def _parse_response(self, response_text: str) -> dict[str, Any]:
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("LLM response does not contain a JSON object.")
        obj = json.loads(response_text[start : end + 1])
        segments = obj.get("segments")
        if not isinstance(segments, list) or not segments:
            raise ValueError("LLM response does not contain valid segments.")
        parsed_segments = []
        for item in segments:
            parsed_segments.append(
                {
                    "start": int(item["start"]),
                    "end": int(item["end"]),
                    "start_delta": float(item["start_delta"]),
                    "end_delta": float(item["end_delta"]),
                    "reason": str(item.get("reason", "")).strip(),
                }
            )
        return {
            "segments": parsed_segments,
            "correction_reasoning_chain": str(obj.get("correction_reasoning_chain", "")).strip(),
        }

    def _render_piecewise_series(self, segments: list[dict[str, Any]], expected_len: int) -> list[float]:
        values = [0.0] * expected_len
        normalized = self._normalize_segments(segments, expected_len)
        for segment in normalized:
            start = segment["start"]
            end = segment["end"]
            span = max(end - start, 1)
            for idx in range(start, end + 1):
                ratio = (idx - start) / span
                values[idx] = segment["start_delta"] + ratio * (
                    segment["end_delta"] - segment["start_delta"]
                )
        return [round(value, 1) for value in values]

    def _normalize_segments(self, segments: list[dict[str, Any]], expected_len: int) -> list[dict[str, Any]]:
        prepared = []
        for segment in segments:
            start = max(0, min(expected_len - 1, segment["start"]))
            end = max(start, min(expected_len - 1, segment["end"]))
            prepared.append({**segment, "start": start, "end": end})
        prepared.sort(key=lambda item: item["start"])

        if prepared[0]["start"] > 0:
            prepared.insert(
                0,
                {
                    "start": 0,
                    "end": prepared[0]["start"],
                    "start_delta": 0.0,
                    "end_delta": prepared[0]["start_delta"],
                    "reason": "自动补齐起始平滑段",
                },
            )
        if prepared[-1]["end"] < expected_len - 1:
            prepared.append(
                {
                    "start": prepared[-1]["end"],
                    "end": expected_len - 1,
                    "start_delta": prepared[-1]["end_delta"],
                    "end_delta": 0.0,
                    "reason": "自动补齐结束回落段",
                }
            )
        return prepared

    def _smooth(self, values: list[float]) -> list[float]:
        if len(values) < 3:
            return values
        result = values[:]
        for idx in range(1, len(values) - 1):
            result[idx] = (values[idx - 1] + 2 * values[idx] + values[idx + 1]) / 4.0
        return result

    def _series_summary(self, values: list[float], label: str) -> dict[str, Any]:
        anchors = []
        step = max(len(values) // 16, 1)
        for idx in range(0, len(values), step):
            anchors.append([idx, round(float(values[idx]), 1)])
        if anchors[-1][0] != len(values) - 1:
            anchors.append([len(values) - 1, round(float(values[-1]), 1)])
        mean_value = sum(values) / max(len(values), 1)
        min_value = min(values)
        max_value = max(values)
        slope = values[-1] - values[0]
        return {
            "label": label,
            "length": len(values),
            "mean": round(mean_value, 1),
            "min": round(min_value, 1),
            "max": round(max_value, 1),
            "end_minus_start": round(slope, 1),
            "anchors": anchors,
        }

    def _series_dense_anchors(self, values: list[float]) -> list[list[float]]:
        anchors = []
        step = 4
        for idx in range(0, len(values), step):
            anchors.append([idx, round(float(values[idx]), 1)])
        if anchors[-1][0] != len(values) - 1:
            anchors.append([len(values) - 1, round(float(values[-1]), 1)])
        return anchors

    def _fallback_result(self, expected_len: int, error_message: str) -> dict[str, Any]:
        return {
            "correction_delta_values": [0.0] * expected_len,
            "correction_reasoning_chain": "修正 LLM 调用失败，当前样本使用零修正占位。",
            "correction_generation_status": "fallback",
            "correction_error": error_message,
        }
