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


class TemporalPhysicalEventScene(BaseSceneGenerator):
    scene_name = "temporal_physical_event"
    target_name = "建筑冷站负荷"
    covariate_name = "室外气温"

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
        occupancy = self.rng.uniform(0.88, 1.15)
        daily_temp_amp = self.rng.uniform(5.0, 8.8)
        semi_daily_amp = self.rng.uniform(0.4, 1.3)
        temp_offset = self.rng.uniform(20.5, 24.5)
        temp_trend = self.rng.uniform(-0.02, 0.04)
        building_base_load = self.rng.uniform(155.0, 210.0)
        cooling_sensitivity = self.rng.uniform(9.0, 14.5)
        latent_gain = self.rng.uniform(10.0, 18.0)

        future_events = self._generate_future_events(day_steps)
        temperature: list[float] = []
        true_load: list[float] = []
        baseline_load: list[float] = []

        for t in range(self.sampling.seq_len):
            hour = (t % day_steps) * self.sampling.sampling_minutes / 60.0
            daily_cycle = daily_temp_amp * math.sin(2.0 * math.pi * (hour - 5.0) / 24.0)
            semi_daily_cycle = semi_daily_amp * math.sin(2.0 * math.pi * hour / 12.0 + 0.8)
            weather_noise = self.rng.uniform(-0.5, 0.5)

            temp_value = temp_offset + daily_cycle + semi_daily_cycle + temp_trend * t + weather_noise
            temp_value += self._temperature_event_effect(t, future_events)
            temp_value = round(temp_value, 1)
            temperature.append(temp_value)

            occupied = 1.0 if 7.0 <= hour <= 20.0 else 0.35
            occupancy_load = occupancy * occupied * latent_gain
            cooling_degree = max(temp_value - 23.0, 0.0)
            base_load = building_base_load + 6.0 * math.sin(2.0 * math.pi * hour / 24.0 - 0.6)

            baseline_value = base_load + occupancy_load + cooling_sensitivity * (cooling_degree ** 1.12)
            baseline_value += self.rng.uniform(-2.0, 2.0)
            baseline_value = round(max(baseline_value, 20.0), 1)
            baseline_load.append(baseline_value)

            true_value = baseline_value + self._load_event_effect(t, future_events, cooling_degree)
            true_value = round(max(true_value, 20.0), 1)
            true_load.append(true_value)

        history_end = self.sampling.history_len
        history_target = self._round_series(true_load[:history_end])
        future_target = self._round_series(true_load[history_end:])
        history_covariate = self._round_series(temperature[:history_end])
        future_covariate = self._round_series(temperature[history_end:])

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
                "future_baseline": self._round_series(baseline_load[history_end:]),
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
        late_future_start = history + day_steps // 2
        events: list[Event] = []

        heatwave_start = self.rng.randint(history + 18, history + 54)
        heatwave_duration = self.rng.randint(day_steps // 2, day_steps + 10)
        heatwave_end = min(heatwave_start + heatwave_duration, forecast_end)
        heatwave_mag = self.rng.uniform(2.8, 5.2)
        events.append(
            Event(
                name="heatwave",
                start=heatwave_start,
                end=heatwave_end,
                magnitude=heatwave_mag,
            )
        )

        if self.rng.random() < 0.7:
            eff_start = self.rng.randint(late_future_start, history + day_steps + 38)
            eff_end = min(eff_start + self.rng.randint(18, 60), forecast_end)
            eff_mag = self.rng.uniform(0.12, 0.23)
            events.append(
                Event(
                    name="efficiency_drop",
                    start=eff_start,
                    end=eff_end,
                    magnitude=eff_mag,
                )
            )

        if self.rng.random() < 0.55:
            dr_start = self.rng.randint(max(late_future_start, history + 72), forecast_end - 18)
            dr_end = min(dr_start + self.rng.randint(8, 24), forecast_end)
            dr_mag = self.rng.uniform(30.0, 55.0)
            events.append(
                Event(
                    name="demand_response",
                    start=dr_start,
                    end=dr_end,
                    magnitude=dr_mag,
                )
            )

        events.sort(key=lambda item: item.start)
        return events

    def _temperature_event_effect(self, t: int, events: list[Event]) -> float:
        delta = 0.0
        for event in events:
            if event.name != "heatwave":
                continue
            if t < event.start or t > event.end:
                continue
            progress = (t - event.start) / max(event.end - event.start, 1)
            ramp = math.sin(math.pi * progress)
            delta += event.magnitude * (0.45 + 0.55 * ramp)
        return delta

    def _load_event_effect(self, t: int, events: list[Event], cooling_degree: float) -> float:
        delta = 0.0
        for event in events:
            if t < event.start or t > event.end:
                continue
            if event.name == "efficiency_drop":
                delta += event.magnitude * (120.0 + 9.0 * cooling_degree)
            elif event.name == "demand_response":
                edge = min(t - event.start, event.end - t)
                smoothing = 0.75 + 0.25 * max(edge, 0) / max(event.end - event.start, 1)
                delta -= event.magnitude * smoothing
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
            f"当前预测目标是“{self.target_name}”，可用的上下文信息是“{self.covariate_name}”。从历史运行表现看，两者整体保持正相关。",
            f"近一段历史里，{self.target_name}整体呈现“{load_shape_text}”的节律，最近平均水平约为{history_target_mean:.1f}。未来两天的{self.covariate_name}相较历史整体{trend_text}。",
        ]

        event_payloads = self._build_structured_event_payloads(events)
        for event in events:
            start_text = self._format_future_time(event.start)
            duration_text = self._format_duration(event.end - event.start + 1)
            if event.name == "heatwave":
                fallback_segments.append(
                    f"{start_text}前后预计会有一段持续约{duration_text}的偏热过程，这会把{self.covariate_name}推到高于常规同类时段的水平。"
                )
            elif event.name == "efficiency_drop":
                fallback_segments.append(
                    f"运维侧还提示，{start_text}开始制冷侧可能经历约{duration_text}的效率走弱。"
                )
            elif event.name == "demand_response":
                fallback_segments.append(
                    f"控制侧消息显示，{start_text}附近可能落地一段约{duration_text}的短时压负荷策略。"
                )

        fallback_text = "".join(fallback_segments)
        payload = ContextGenerationPayload(
            target_name=self.target_name,
            context_name=self.covariate_name,
            history_summary=(
                f"历史上{self.target_name}呈现“{load_shape_text}”的节律，最近平均水平约为{history_target_mean:.1f}，"
                f"且{self.covariate_name}升高时通常带来{self.target_name}同步抬升。"
            ),
            future_context_summary=(
                f"未来两天的{self.covariate_name}相较历史整体{trend_text}，需要重点关注高位时段、回落节奏和异常偏离。"
            ),
            continuity_instruction="",
            events=event_payloads,
        )
        return self.context_generator.generate(payload, fallback_text)

    def _build_structured_event_payloads(self, events: list[Event]) -> list[dict[str, Any]]:
        event_payloads: list[dict[str, Any]] = []
        for event in events:
            start_text = self._format_future_time(event.start)
            duration_text = self._format_duration(event.end - event.start + 1)
            future_start_index = event.start - self.sampling.history_len
            future_end_index = event.end - self.sampling.history_len
            if event.name == "heatwave":
                event_payloads.append(
                    {
                        **self._build_event_payload(
                            event=event,
                            start_text=start_text,
                            duration_text=duration_text,
                            context_effect="明显抬升并阶段性偏热",
                            target_direction="上升",
                            impact_strength=self._describe_event_strength(event),
                            narrative_hint=f"{self.target_name}更容易连续上冲，高位时段压力增加。",
                        ),
                        "future_start_index": future_start_index,
                        "future_end_index": future_end_index,
                        "future_duration_steps": future_end_index - future_start_index + 1,
                        "expected_delta_sign": "positive",
                    }
                )
            elif event.name == "efficiency_drop":
                event_payloads.append(
                    {
                        **self._build_event_payload(
                            event=event,
                            start_text=start_text,
                            duration_text=duration_text,
                            context_effect="不一定继续变化，可能接近常规水平",
                            target_direction="上升或持续偏高",
                            impact_strength=self._describe_event_strength(event),
                            narrative_hint=f"{self.target_name}可能持续偏高，且恢复速度偏慢。",
                        ),
                        "future_start_index": future_start_index,
                        "future_end_index": future_end_index,
                        "future_duration_steps": future_end_index - future_start_index + 1,
                        "expected_delta_sign": "positive",
                    }
                )
            elif event.name == "demand_response":
                event_payloads.append(
                    {
                        **self._build_event_payload(
                            event=event,
                            start_text=start_text,
                            duration_text=duration_text,
                            context_effect="主要是控制动作，对上下文信息本身影响有限",
                            target_direction="下降",
                            impact_strength=self._describe_event_strength(event),
                            narrative_hint=f"{self.target_name}会被短时压低，随后逐步回到常规节律。",
                        ),
                        "future_start_index": future_start_index,
                        "future_end_index": future_end_index,
                        "future_duration_steps": future_end_index - future_start_index + 1,
                        "expected_delta_sign": "negative",
                    }
                )
        return event_payloads

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
        if event.name == "demand_response":
            if magnitude < 35:
                return "较小"
            if magnitude < 48:
                return "中等"
            return "较强"
        if event.name == "efficiency_drop":
            if magnitude < 0.15:
                return "较小"
            if magnitude < 0.2:
                return "中等"
            return "较强"
        if magnitude < 3.4:
            return "较小"
        if magnitude < 4.4:
            return "中等"
        return "较强"

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


