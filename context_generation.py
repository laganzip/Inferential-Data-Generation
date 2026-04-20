from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import requests

from inferential_data_generation.base import GenerationConfig


@dataclass(frozen=True)
class ContextGenerationPayload:
    target_name: str
    context_name: str
    history_summary: str
    future_context_summary: str
    continuity_instruction: str
    events: list[dict[str, Any]]


class ContextGenerator:
    def __init__(self, config: GenerationConfig) -> None:
        self.config = config

    def generate(self, payload: ContextGenerationPayload, fallback_text: str) -> str:
        if self.config.context_generation_mode != "llm":
            return fallback_text
        if not self.config.llm_base_url or not self.config.llm_api_key or not self.config.llm_model:
            if self.config.allow_context_fallback:
                return fallback_text
            raise ValueError(
                "LLM context generation requires llm_base_url, llm_api_key, and llm_model."
            )

        try:
            return self._generate_with_openai_compatible_api(payload)
        except Exception:
            print("LLM context generation failed. Falling back to default context text.")
            if self.config.allow_context_fallback:
                return fallback_text
            raise

    def _generate_with_openai_compatible_api(self, payload: ContextGenerationPayload) -> str:
        url = self.config.llm_base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.llm_api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.config.llm_model,
            "temperature": 0.8,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是一个工业时序数据生成助手。你的任务是把结构化事实写成自然、专业、可用于后续修正预测的中文上下文描述。"
                        "不要输出列表、JSON、标题、引号包裹的标签或额外解释，只输出一段连续中文。"
                        "你可以根据事件类型、方向、强度和持续时间灵活组织措辞，使未来影响可以表现为上升、下降、持续偏高、回落变慢或短时压低。"
                        # "文字应明确关键时段、影响方向、影响强弱，并暗示修正值应随事件演化平滑变化，避免无依据跳变。"
                        "风格要求稳定、克制、像运维分析说明，不要过度文学化，不要口语化。"
                    ),
                },
                {
                    "role": "user",
                    "content": self._build_user_prompt(payload),
                },
            ],
        }

        empty_proxies = {
            "http": None,
            "https": None,
        }

        response = requests.post(
            url,
            headers=headers,
            json=body,
            timeout=self.config.llm_timeout_seconds,
            proxies=empty_proxies,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def _build_user_prompt(self, payload: ContextGenerationPayload) -> str:
        event_lines = []
        for idx, event in enumerate(payload.events, start=1):
            event_lines.append(
                f"{idx}. 类型={event['event_type']}，开始={event['start_text']}，持续={event['duration_text']}，"
                f"对{payload.context_name}影响={event['context_effect']}，对{payload.target_name}影响方向={event['target_direction']}，"
                f"影响强度={event['impact_strength']}，补充说明={event['narrative_hint']}"
            )
        events_text = "\n".join(event_lines) if event_lines else "无显著事件。"

        facts = {
            "target_name": payload.target_name,
            "context_name": payload.context_name,
            "history_summary": payload.history_summary,
            "future_context_summary": payload.future_context_summary,
            "continuity_instruction": payload.continuity_instruction,
        }
        writing_template = (
            "请尽量按下面的段落骨架组织，但不要机械照抄：\n"
            "1. 先用 1 到 2 句说明预测目标、上下文信息，以及历史上的总体关系和近期节律。\n"
            "2. 再用 1 句概括未来上下文相对历史是偏强、偏弱还是接近常态。\n"
            "3. 然后按时间顺序写各个关键事件，说明开始时间、持续时间、影响方向、影响强弱，以及影响更像持续抬升、短时压低、恢复变慢还是逐步回落。\n"
            # "4. 最后用 1 句自然收束，暗示后续修正应顺着事件演化平滑展开，不要出现突兀跳变。\n"
        )
        style_example = (
            f"示例风格（只作为写法参考，不要复述内容）："
            f"{payload.target_name}的历史变化与{payload.context_name}整体保持同向联动，近期仍以日内起伏为主。"
            f"从未来两天的背景看，{payload.context_name}较历史常态略强，因此常规工况下{payload.target_name}的高位压力不会太早缓解。"
            f"若未来第1天上午开始出现一段中等强度的扰动，{payload.target_name}更可能在该时段逐步抬升，而不是单点跳高；"
            f"若后续再叠加一次短时控制动作，则对应时段的水平会在以上基础上再进行短时上升或下降。"
            # f"整体上，后续偏离更像沿事件过程缓慢展开和收敛。"
        )
        return (
            "请基于以下事实生成一段中文上下文描述。\n"
            "要求：自然叙事、避免模板腔、突出未来关键时段和影响方向，可以体现影响偏大/偏小、上升/下降、持续偏高/恢复变慢等差异。\n"
            "不要泄漏真实未来值，不要写成项目符号，不要输出额外解释，不要重复事件事实原文，不要出现“根据输入”“如下所示”“综上模板”等元话语。\n"
            f"{writing_template}\n"
            f"{style_example}\n"
            f"基础事实: {json.dumps(facts, ensure_ascii=False)}\n"
            f"事件事实:\n{events_text}"
        )
