from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any

from inferential_data_generation.base import BaseSceneGenerator, GenerationConfig, SceneSample
from inferential_data_generation.context_generation import ContextGenerationPayload, ContextGenerator


@dataclass(frozen=True)
class Event:
    name: str
    start: int
    end: int
    magnitude: float


class DataCenterCoolingEventScene(BaseSceneGenerator):
    scene_name = "data_center_cooling_event"
    target_name = "数据中心冷却功率"
    covariate_name = "机房进风温度"

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self.sampling = config.sampling
        self.rng = random.Random(config.seed)
        self.context_generator = ContextGenerator(config)

        if self.sampling.seq_len <= self.sampling.history_len:
            raise ValueError("seq_len must be larger than history_len")
        if self.sampling.seq_len != 384:
            raise ValueError("The current scene expects seq_len=384")

    def generate_scene_sample(self, sample_id: int) -> SceneSample:
        del sample_id

        day_steps = self._steps_per_day()
        it_base_kw = self.rng.uniform(540.0, 940.0)
        it_daily_ratio = self.rng.uniform(0.04, 0.09)
        inlet_base = self.rng.uniform(23.0, 25.2)
        ambient_offset = self.rng.uniform(26.0, 31.0)
        ambient_amp = self.rng.uniform(4.5, 7.5)
        ambient_trend = self.rng.uniform(-0.015, 0.03)
        chiller_cop_base = self.rng.uniform(4.8, 6.3)
        fan_power_base = self.rng.uniform(75.0, 145.0)
        pump_power_base = self.rng.uniform(52.0, 98.0)

        future_events = self._generate_future_events(day_steps)
        inlet_temperature: list[float] = []
        true_power: list[float] = []
        baseline_power: list[float] = []

        for t in range(self.sampling.seq_len):
            hour = (t % day_steps) * self.sampling.sampling_minutes / 60.0

            ambient_temp = (
                ambient_offset
                + ambient_amp * math.sin(2.0 * math.pi * (hour - 5.5) / 24.0)
                + 0.7 * math.sin(2.0 * math.pi * hour / 12.0 + 0.4)
                + ambient_trend * t
                + self.rng.uniform(-0.5, 0.5)
            )
            ambient_temp = round(ambient_temp, 1)

            it_load_ratio = 1.0 + it_daily_ratio * math.sin(2.0 * math.pi * (hour - 13.0) / 24.0)
            it_load_ratio += self.rng.uniform(-0.015, 0.015)
            it_load_ratio += self._it_load_event_effect_ratio(t, future_events)
            it_load_ratio = max(it_load_ratio, 0.75)
            it_load_kw = it_base_kw * it_load_ratio

            inlet_temp = inlet_base + 0.0085 * (it_load_kw - it_base_kw) + 0.085 * max(ambient_temp - 26.0, 0.0)
            inlet_temp += self._temperature_event_effect(t, future_events)
            inlet_temp += self.rng.uniform(-0.18, 0.18)
            inlet_temp = round(inlet_temp, 1)
            inlet_temperature.append(inlet_temp)

            cooling_stress = max(inlet_temp - 24.0, 0.0)
            cooling_reject_kw = 0.93 * it_load_kw + 22.0 * cooling_stress
            cop_penalty = self._cop_penalty_event_effect(t, future_events)
            chiller_cop = chiller_cop_base - 0.08 * max(ambient_temp - 29.0, 0.0) - cop_penalty
            chiller_cop = max(chiller_cop, 2.6)

            chiller_power = cooling_reject_kw / chiller_cop
            fan_power = fan_power_base * (1.0 + 0.025 * cooling_stress)
            pump_power = pump_power_base * (1.0 + 0.018 * max(ambient_temp - 27.0, 0.0))

            baseline_value = chiller_power + fan_power + pump_power + self.rng.uniform(-4.0, 4.0)
            baseline_value = round(max(baseline_value, 60.0), 1)
            baseline_power.append(baseline_value)

            true_value = baseline_value + self._load_event_effect(t, future_events, baseline_value)
            true_value = round(max(true_value, 60.0), 1)
            true_power.append(true_value)

        history_end = self.sampling.history_len
        history_target = self._round_series(true_power[:history_end])
        future_target = self._round_series(true_power[history_end:])
        history_covariate = self._round_series(inlet_temperature[:history_end])
        future_covariate = self._round_series(inlet_temperature[history_end:])

        context_text = self._build_context_text(
            history_target=history_target,
            history_covariate_values=history_covariate,
            future_covariate_values=future_covariate,
            events=future_events,
        )
        structured_event_context = self._build_structured_event_payloads(future_events)

        return SceneSample(
            history_target_values=history_target,
            future_target_values=future_target,
            history_covariate_values=history_covariate,
            future_covariate_values=future_covariate,
            target_name=self.target_name,
            covariate_name=self.covariate_name,
            context_description=context_text,
            structured_event_context=structured_event_context,
            metadata={
                "day_steps": day_steps,
                "future_baseline": self._round_series(baseline_power[history_end:]),
                "scene_name": self.scene_name,
            },
        )

    def _steps_per_day(self) -> int:
        minutes_per_day = 24 * 60
        if minutes_per_day % self.sampling.sampling_minutes != 0:
            raise ValueError("sampling_minutes must divide 1440 for this scene")
        return minutes_per_day // self.sampling.sampling_minutes

    def _generate_future_events(self, day_steps: int) -> list[Event]:
        history = self.sampling.history_len
        forecast_end = self.sampling.seq_len - 1
        events: list[Event] = []

        ai_start = self.rng.randint(history + 8, history + day_steps // 2)
        ai_end = min(ai_start + self.rng.randint(day_steps // 3, day_steps + 16), forecast_end)
        events.append(
            Event(
                name="ai_training_window",
                start=ai_start,
                end=ai_end,
                magnitude=self.rng.uniform(0.10, 0.24),
            )
        )

        if self.rng.random() < 0.72:
            foul_start = self.rng.randint(history + day_steps // 2, history + day_steps + 26)
            foul_end = min(foul_start + self.rng.randint(20, 78), forecast_end)
            events.append(
                Event(
                    name="chiller_fouling",
                    start=foul_start,
                    end=foul_end,
                    magnitude=self.rng.uniform(0.14, 0.32),
                )
            )

        if self.rng.random() < 0.62:
            eco_start = self.rng.randint(history + 24, forecast_end - 20)
            eco_end = min(eco_start + self.rng.randint(16, 60), forecast_end)
            events.append(
                Event(
                    name="waterside_economizer",
                    start=eco_start,
                    end=eco_end,
                    magnitude=self.rng.uniform(0.12, 0.30),
                )
            )

        events.sort(key=lambda item: item.start)
        return events

    def _event_profile(self, t: int, event: Event) -> float:
        if t < event.start or t > event.end:
            return 0.0
        progress = (t - event.start) / max(event.end - event.start, 1)
        return 0.35 + 0.65 * math.sin(math.pi * progress)

    def _it_load_event_effect_ratio(self, t: int, events: list[Event]) -> float:
        delta = 0.0
        for event in events:
            if event.name != "ai_training_window":
                continue
            delta += event.magnitude * self._event_profile(t, event)
        return delta

    def _temperature_event_effect(self, t: int, events: list[Event]) -> float:
        delta = 0.0
        for event in events:
            profile = self._event_profile(t, event)
            if profile <= 0:
                continue
            if event.name == "ai_training_window":
                delta += 1.8 * event.magnitude * profile
            elif event.name == "waterside_economizer":
                delta -= 3.6 * event.magnitude * profile
        return delta

    def _cop_penalty_event_effect(self, t: int, events: list[Event]) -> float:
        penalty = 0.0
        for event in events:
            if event.name != "chiller_fouling":
                continue
            penalty += 1.8 * event.magnitude * self._event_profile(t, event)
        return penalty

    def _load_event_effect(self, t: int, events: list[Event], baseline_value: float) -> float:
        delta = 0.0
        for event in events:
            profile = self._event_profile(t, event)
            if profile <= 0:
                continue
            if event.name == "chiller_fouling":
                delta += baseline_value * event.magnitude * (0.26 + 0.22 * profile)
            elif event.name == "waterside_economizer":
                delta -= baseline_value * event.magnitude * (0.42 + 0.15 * profile)
        return delta

    def _build_context_text(
        self,
        history_target: list[float],
        history_covariate_values: list[float],
        future_covariate_values: list[float],
        events: list[Event],
    ) -> str:
        history_target_mean = sum(history_target[-32:]) / min(len(history_target), 32)
        history_context_mean = sum(history_covariate_values) / len(history_covariate_values)
        future_context_mean = sum(future_covariate_values) / len(future_covariate_values)
        trend_text = self._describe_context_trend(history_context_mean, future_context_mean)
        load_shape_text = self._series_shape_text(history_target[-96:])

        fallback_segments = [
            f"当前预测目标是“{self.target_name}”，可用上下文是“{self.covariate_name}”。在常规工况下，进风温度抬升会推高冷却功率。",
            f"历史上负荷整体呈现“{load_shape_text}”节律，最近平均约为{history_target_mean:.1f}。未来两天进风温度相对历史整体{trend_text}。",
        ]
        for event in events:
            start_text = self._format_future_time(event.start)
            duration_text = self._format_duration(event.end - event.start + 1)
            if event.name == "ai_training_window":
                fallback_segments.append(
                    f"{start_text}起可能出现持续约{duration_text}的高密度训练窗口，会带来更高机柜散热压力。"
                )
            elif event.name == "chiller_fouling":
                fallback_segments.append(
                    f"{start_text}前后制冷机组可能有约{duration_text}的换热效率走弱，同等冷量下功率会偏高。"
                )
            elif event.name == "waterside_economizer":
                fallback_segments.append(
                    f"{start_text}附近若开启约{duration_text}的水侧自然冷却，冷却功率会阶段性回落。"
                )

        fallback_text = "".join(fallback_segments)
        payload = ContextGenerationPayload(
            target_name=self.target_name,
            context_name=self.covariate_name,
            history_summary=(
                f"历史上{self.target_name}呈现“{load_shape_text}”节律，最近平均约{history_target_mean:.1f}，"
                f"且{self.covariate_name}升高时通常带来{self.target_name}同步抬升。"
            ),
            future_context_summary=f"未来两天{self.covariate_name}相对历史整体{trend_text}。",
            continuity_instruction="",
            events=self._build_structured_event_payloads(events),
        )
        return self.context_generator.generate(payload, fallback_text)

    def _build_structured_event_payloads(self, events: list[Event]) -> list[dict[str, Any]]:
        event_payloads: list[dict[str, Any]] = []
        for event in events:
            start_text = self._format_future_time(event.start)
            duration_text = self._format_duration(event.end - event.start + 1)
            future_start_index = event.start - self.sampling.history_len
            future_end_index = event.end - self.sampling.history_len
            if event.name == "ai_training_window":
                event_payloads.append(
                    {
                        **self._build_event_payload(
                            event=event,
                            start_text=start_text,
                            duration_text=duration_text,
                            context_effect="进风温度偏高且热斑风险上升",
                            target_direction="上升",
                            impact_strength=self._describe_event_strength(event),
                            narrative_hint=f"{self.target_name}会随散热需求上行，并在高位段更容易放大。",
                        ),
                        "future_start_index": future_start_index,
                        "future_end_index": future_end_index,
                        "future_duration_steps": future_end_index - future_start_index + 1,
                        "expected_delta_sign": "positive",
                    }
                )
            elif event.name == "chiller_fouling":
                event_payloads.append(
                    {
                        **self._build_event_payload(
                            event=event,
                            start_text=start_text,
                            duration_text=duration_text,
                            context_effect="对进风温度影响有限，主要体现在效率下降",
                            target_direction="上升或持续偏高",
                            impact_strength=self._describe_event_strength(event),
                            narrative_hint=f"同等冷量下{self.target_name}会偏高且回落更慢。",
                        ),
                        "future_start_index": future_start_index,
                        "future_end_index": future_end_index,
                        "future_duration_steps": future_end_index - future_start_index + 1,
                        "expected_delta_sign": "positive",
                    }
                )
            elif event.name == "waterside_economizer":
                event_payloads.append(
                    {
                        **self._build_event_payload(
                            event=event,
                            start_text=start_text,
                            duration_text=duration_text,
                            context_effect="进风温度阶段性回落",
                            target_direction="下降",
                            impact_strength=self._describe_event_strength(event),
                            narrative_hint=f"{self.target_name}会被显著压低，退出后再回到常规水平。",
                        ),
                        "future_start_index": future_start_index,
                        "future_end_index": future_end_index,
                        "future_duration_steps": future_end_index - future_start_index + 1,
                        "expected_delta_sign": "negative",
                    }
                )
        return event_payloads

    def _build_event_payload(
        self,
        event: Event,
        start_text: str,
        duration_text: str,
        context_effect: str,
        target_direction: str,
        impact_strength: str,
        narrative_hint: str,
    ) -> dict[str, str]:
        return {
            "event_type": event.name,
            "start_text": start_text,
            "duration_text": duration_text,
            "context_effect": context_effect,
            "target_direction": target_direction,
            "impact_strength": impact_strength,
            "narrative_hint": narrative_hint,
        }

    def _describe_event_strength(self, event: Event) -> str:
        magnitude = abs(event.magnitude)
        if event.name == "ai_training_window":
            if magnitude < 0.14:
                return "较小"
            if magnitude < 0.2:
                return "中等"
            return "较强"
        if event.name == "chiller_fouling":
            if magnitude < 0.18:
                return "较小"
            if magnitude < 0.26:
                return "中等"
            return "较强"
        if magnitude < 0.18:
            return "较小"
        if magnitude < 0.25:
            return "中等"
        return "较强"

    def _describe_context_trend(self, history_mean: float, future_mean: float) -> str:
        delta = future_mean - history_mean
        if delta > 1.8:
            return "明显偏热"
        if delta > 0.6:
            return "略偏热"
        if delta < -1.8:
            return "明显回落"
        if delta < -0.6:
            return "略有回落"
        return "与近期相近"

    def _format_future_time(self, global_index: int) -> str:
        future_index = global_index - self.sampling.history_len
        minutes_from_start = future_index * self.sampling.sampling_minutes
        day = minutes_from_start // (24 * 60) + 1
        minute_of_day = minutes_from_start % (24 * 60)
        hour = minute_of_day // 60
        minute = minute_of_day % 60
        period = self._period_of_day(hour)
        return f"未来第{day}天{period}{hour:02d}:{minute:02d}"

    def _format_duration(self, steps: int) -> str:
        total_minutes = steps * self.sampling.sampling_minutes
        if total_minutes % 60 == 0:
            hours = total_minutes // 60
            if hours >= 24 and total_minutes % (24 * 60) == 0:
                days = total_minutes // (24 * 60)
                return f"{days}天"
            return f"{hours}小时"
        if total_minutes >= 60:
            hours = total_minutes // 60
            minutes = total_minutes % 60
            return f"{hours}小时{minutes}分钟"
        return f"{total_minutes}分钟"

    def _period_of_day(self, hour: int) -> str:
        if 0 <= hour < 6:
            return "凌晨"
        if 6 <= hour < 12:
            return "上午"
        if 12 <= hour < 18:
            return "下午"
        return "晚上"

    def _series_shape_text(self, values: list[float]) -> str:
        first_half = sum(values[: len(values) // 2]) / max(len(values) // 2, 1)
        second_half = sum(values[len(values) // 2 :]) / max(len(values) - len(values) // 2, 1)
        if second_half - first_half > 12:
            return "后半段更强"
        if first_half - second_half > 12:
            return "前高后低"
        return "围绕日周期上下波动"

    def _round_series(self, values: list[float]) -> list[float]:
        return [round(value, 1) for value in values]
