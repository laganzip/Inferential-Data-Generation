from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any

from inferential_data_generation.main.base import BaseSceneGenerator, GenerationConfig, SceneSample
from inferential_data_generation.main.context_generation import ContextGenerationPayload, ContextGenerator


@dataclass(frozen=True)
class Event:
    name: str
    start: int
    end: int
    magnitude: float


class CommercialHVACEventScene(BaseSceneGenerator):
    scene_name = "commercial_hvac_event"
    target_name = "商业综合体空调负荷"
    covariate_name = "体感温度指数"

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
        dry_temp_offset = self.rng.uniform(24.0, 29.0)
        dry_temp_amp = self.rng.uniform(5.0, 8.5)
        humidity_offset = self.rng.uniform(50.0, 66.0)
        humidity_amp = self.rng.uniform(6.0, 13.0)
        humidity_trend = self.rng.uniform(-0.03, 0.03)

        base_load = self.rng.uniform(160.0, 260.0)
        envelope_sensitivity = self.rng.uniform(8.5, 14.0)
        latent_gain = self.rng.uniform(0.9, 1.6)
        occupancy_gain = self.rng.uniform(165.0, 240.0)

        future_events = self._generate_future_events(day_steps)
        thermal_index_series: list[float] = []
        true_load: list[float] = []
        baseline_load: list[float] = []

        for t in range(self.sampling.seq_len):
            hour = (t % day_steps) * self.sampling.sampling_minutes / 60.0
            day_id = t // day_steps

            dry_temp = (
                dry_temp_offset
                + dry_temp_amp * math.sin(2.0 * math.pi * (hour - 4.2) / 24.0)
                + 0.8 * math.sin(2.0 * math.pi * hour / 12.0 + 0.6)
                + self.rng.uniform(-0.45, 0.45)
            )
            humidity = (
                humidity_offset
                + humidity_amp * math.sin(2.0 * math.pi * (hour - 7.5) / 24.0 + 0.7)
                + humidity_trend * t
                + self.rng.uniform(-1.2, 1.2)
            )

            thermal_index = dry_temp + 0.11 * (humidity - 50.0)
            thermal_index += self._temperature_event_effect(t, future_events)
            thermal_index = round(thermal_index, 1)
            thermal_index_series.append(thermal_index)

            occupancy_ratio = self._occupancy_ratio(day_id=day_id, hour=hour)
            occupancy_ratio += self._occupancy_event_effect(t, future_events)
            occupancy_ratio = min(max(occupancy_ratio, 0.12), 1.35)

            cooling_degree = max(thermal_index - 23.8, 0.0)
            latent_component = max(humidity - 55.0, 0.0) * latent_gain * occupancy_ratio
            baseline_value = (
                base_load
                + occupancy_gain * occupancy_ratio
                + envelope_sensitivity * (cooling_degree ** 1.08)
                + latent_component
                + self.rng.uniform(-3.5, 3.5)
            )
            baseline_value = round(max(baseline_value, 40.0), 1)
            baseline_load.append(baseline_value)

            true_value = baseline_value + self._load_event_effect(t, future_events, baseline_value)
            true_value = round(max(true_value, 40.0), 1)
            true_load.append(true_value)

        history_end = self.sampling.history_len
        history_target = self._round_series(true_load[:history_end])
        future_target = self._round_series(true_load[history_end:])
        history_covariate = self._round_series(thermal_index_series[:history_end])
        future_covariate = self._round_series(thermal_index_series[history_end:])

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
        events: list[Event] = []

        if self.rng.random() < 0.78:
            promo_start = self.rng.randint(history + 24, history + day_steps + 18)
            promo_end = min(promo_start + self.rng.randint(16, 52), forecast_end)
            events.append(
                Event(
                    name="mall_promotion",
                    start=promo_start,
                    end=promo_end,
                    magnitude=self.rng.uniform(0.15, 0.34),
                )
            )

        cold_start = self.rng.randint(history + 10, history + day_steps // 2 + 30)
        cold_end = min(cold_start + self.rng.randint(18, 64), forecast_end)
        events.append(
            Event(
                name="cold_front",
                start=cold_start,
                end=cold_end,
                magnitude=self.rng.uniform(1.2, 3.8),
            )
        )

        if self.rng.random() < 0.65:
            maint_start = self.rng.randint(history + day_steps // 2, forecast_end - 12)
            maint_end = min(maint_start + self.rng.randint(10, 44), forecast_end)
            events.append(
                Event(
                    name="chiller_maintenance",
                    start=maint_start,
                    end=maint_end,
                    magnitude=self.rng.uniform(22.0, 68.0),
                )
            )

        events.sort(key=lambda item: item.start)
        return events

    def _event_profile(self, t: int, event: Event) -> float:
        if t < event.start or t > event.end:
            return 0.0
        progress = (t - event.start) / max(event.end - event.start, 1)
        return 0.35 + 0.65 * math.sin(math.pi * progress)

    def _occupancy_ratio(self, day_id: int, hour: float) -> float:
        is_weekend_like = day_id % 7 in (5, 6)
        if is_weekend_like:
            if 10.0 <= hour <= 22.0:
                return 0.72 + 0.18 * math.sin(math.pi * (hour - 10.0) / 12.0)
            return 0.2
        if 8.0 <= hour <= 21.0:
            return 0.78 + 0.22 * math.sin(math.pi * (hour - 8.0) / 13.0)
        return 0.24

    def _occupancy_event_effect(self, t: int, events: list[Event]) -> float:
        delta = 0.0
        for event in events:
            if event.name != "mall_promotion":
                continue
            delta += event.magnitude * self._event_profile(t, event)
        return delta

    def _temperature_event_effect(self, t: int, events: list[Event]) -> float:
        delta = 0.0
        for event in events:
            if event.name != "cold_front":
                continue
            delta -= event.magnitude * self._event_profile(t, event)
        return delta

    def _load_event_effect(self, t: int, events: list[Event], baseline_value: float) -> float:
        delta = 0.0
        for event in events:
            profile = self._event_profile(t, event)
            if profile <= 0:
                continue
            if event.name == "chiller_maintenance":
                delta += event.magnitude * (0.7 + 0.3 * profile)
            elif event.name == "cold_front":
                delta -= baseline_value * 0.012 * event.magnitude * profile
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
            f"当前预测目标是“{self.target_name}”，可用上下文是“{self.covariate_name}”。体感温度指数越高，空调负荷通常越高。",
            f"历史上负荷主要呈现“{load_shape_text}”节律，最近平均约{history_target_mean:.1f}。未来两天体感温度指数相对历史整体{trend_text}。",
        ]
        for event in events:
            start_text = self._format_future_time(event.start)
            duration_text = self._format_duration(event.end - event.start + 1)
            if event.name == "mall_promotion":
                fallback_segments.append(
                    f"{start_text}前后若有约{duration_text}的促销活动，客流增大会抬升空调负荷。"
                )
            elif event.name == "cold_front":
                fallback_segments.append(
                    f"{start_text}起若有约{duration_text}的冷空气过程，体感温度指数将阶段性回落。"
                )
            elif event.name == "chiller_maintenance":
                fallback_segments.append(
                    f"{start_text}附近若进行约{duration_text}的制冷侧维护，系统运行效率会临时下降。"
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
            if event.name == "mall_promotion":
                event_payloads.append(
                    {
                        **self._build_event_payload(
                            event=event,
                            start_text=start_text,
                            duration_text=duration_text,
                            context_effect="体感温度指数不一定上行，但内部客流和散热会增加",
                            target_direction="上升",
                            impact_strength=self._describe_event_strength(event),
                            narrative_hint=f"{self.target_name}会在活动时段持续偏高。",
                        ),
                        "future_start_index": future_start_index,
                        "future_end_index": future_end_index,
                        "future_duration_steps": future_end_index - future_start_index + 1,
                        "expected_delta_sign": "positive",
                    }
                )
            elif event.name == "cold_front":
                event_payloads.append(
                    {
                        **self._build_event_payload(
                            event=event,
                            start_text=start_text,
                            duration_text=duration_text,
                            context_effect="体感温度指数明显回落",
                            target_direction="下降",
                            impact_strength=self._describe_event_strength(event),
                            narrative_hint=f"{self.target_name}会被气象侧拉低，随后逐步回归日常节律。",
                        ),
                        "future_start_index": future_start_index,
                        "future_end_index": future_end_index,
                        "future_duration_steps": future_end_index - future_start_index + 1,
                        "expected_delta_sign": "negative",
                    }
                )
            elif event.name == "chiller_maintenance":
                event_payloads.append(
                    {
                        **self._build_event_payload(
                            event=event,
                            start_text=start_text,
                            duration_text=duration_text,
                            context_effect="体感温度指数不一定变化，主要是设备效率暂时变差",
                            target_direction="上升或持续偏高",
                            impact_strength=self._describe_event_strength(event),
                            narrative_hint=f"同等需求下{self.target_name}会有额外抬升且恢复偏慢。",
                        ),
                        "future_start_index": future_start_index,
                        "future_end_index": future_end_index,
                        "future_duration_steps": future_end_index - future_start_index + 1,
                        "expected_delta_sign": "positive",
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
        if event.name == "mall_promotion":
            if magnitude < 0.2:
                return "较小"
            if magnitude < 0.28:
                return "中等"
            return "较强"
        if event.name == "cold_front":
            if magnitude < 2.0:
                return "较小"
            if magnitude < 3.0:
                return "中等"
            return "较强"
        if magnitude < 32:
            return "较小"
        if magnitude < 52:
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
