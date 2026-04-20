from __future__ import annotations

from typing import Any

from inferential_data_generation.main.base import BaseInitialPredictor, GenerationConfig


class HeuristicLoadPredictor(BaseInitialPredictor):
    predictor_name = "heuristic"

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config

    def predict(
        self,
        history_target_values: list[float],
        history_context_values: list[float],
        future_context_values: list[float],
        metadata: dict[str, Any],
    ) -> list[float]:
        del history_context_values

        day_steps = metadata["day_steps"]
        future_baseline = metadata["future_baseline"]
        last_day_history = history_target_values[-day_steps:]
        recent_mean = sum(history_target_values[-16:]) / 16.0
        prediction = []

        for idx, baseline in enumerate(future_baseline):
            seasonal_anchor = last_day_history[idx % day_steps]
            blended = 0.58 * seasonal_anchor + 0.42 * baseline
            temp_adjust = 1.8 * max(
                future_context_values[idx] - future_context_values[max(idx - 1, 0)],
                0.0,
            )
            inertia = 0.08 * (recent_mean - seasonal_anchor)
            predicted = blended + temp_adjust + inertia
            prediction.append(round(max(predicted, 20.0), 1))

        return prediction
