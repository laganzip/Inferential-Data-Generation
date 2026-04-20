from __future__ import annotations

from importlib import import_module
from typing import Any
import sys
import warnings

from inferential_data_generation.base import BaseInitialPredictor, GenerationConfig
from inferential_data_generation.predictors.heuristic import HeuristicLoadPredictor


class Chronos2InitialPredictor(BaseInitialPredictor):
    predictor_name = "chronos2"

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self._fallback_predictor = HeuristicLoadPredictor(config)
        self._pipeline = None
        self._torch = None
        self._np = None

        try:
            sys.path.insert(0, "/home/yichenglu/git-tpt/tpt-prophet")
            self._np = import_module("numpy")
            self._torch = import_module("torch")
            pipeline_module = import_module("src.models.chronos_2_forecast.core.pipeline")
            pipeline_cls = getattr(pipeline_module, "Chronos2Pipeline")
            device = self._resolve_device()
            self._pipeline = pipeline_cls.from_pretrained(
                self.config.chronos2_model_path,
                device_map=device,
            )
        except Exception as exc:  # pragma: no cover - depends on local Chronos2 runtime
            if not self.config.allow_predictor_fallback:
                raise RuntimeError(
                    "Unable to initialize Chronos2 predictor. "
                    "Check that the Chronos2 runtime module and local model path are available."
                ) from exc
            self.predictor_name = "chronos2_fallback"
            warnings.warn(
                "Chronos2 runtime is unavailable; falling back to heuristic predictor. "
                f"Reason: {exc}",
                RuntimeWarning,
            )

    def predict(
        self,
        history_target_values: list[float],
        history_context_values: list[float],
        future_context_values: list[float],
        metadata: dict[str, Any],
    ) -> list[float]:
        if self._pipeline is None or self._torch is None or self._np is None:
            return self._fallback_predictor.predict(
                history_target_values=history_target_values,
                history_context_values=history_context_values,
                future_context_values=future_context_values,
                metadata=metadata,
            )
        with self._torch.no_grad():  # pragma: no cover - depends on local Chronos2 runtime
            quantiles, predicts = self._pipeline.predict_quantiles(
                inputs=[self._np.array(history_target_values)],
                prediction_length=len(future_context_values),
                quantile_levels=[0.1, 0.5, 0.9],
            )

        del quantiles

        predicts = self._torch.cat(predicts, dim=0).squeeze().cpu().numpy()

        values = self._tensor_to_list(predicts)
        
        return values

    def _resolve_device(self):
        if self.config.predictor_device:
            return self._torch.device(self.config.predictor_device)
        if self._torch.cuda.is_available():
            return self._torch.device("cuda:0")
        return self._torch.device("cpu")

    def _tensor_to_list(self, values) -> list[float]:
        if hasattr(values, "detach"):
            values = values.detach()
        if hasattr(values, "cpu"):
            values = values.cpu()
        if hasattr(values, "numpy"):
            values = values.numpy()
        if hasattr(values, "tolist"):
            values = values.tolist()
        while isinstance(values, list) and len(values) == 1:
            values = values[0]
        return [round(float(item), 1) for item in values]
