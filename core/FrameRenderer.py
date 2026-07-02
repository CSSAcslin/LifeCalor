from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrameRenderParams:
    min_value: float | None = None
    max_value: float | None = None
    auto_range: bool = True
    use_colormap: bool = False
    colormap: str = "gray"


@dataclass(frozen=True)
class RenderedFrame:
    image: np.ndarray
    mode: str
    min_value: float = 0.0
    max_value: float = 0.0


class FrameRenderer:
    @staticmethod
    def render(frame: np.ndarray, params: FrameRenderParams | None = None) -> RenderedFrame:
        params = params or FrameRenderParams()
        normalized, min_value, max_value = FrameRenderer._normalize(frame, params)
        gray = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)

        if params.use_colormap:
            return RenderedFrame(
                image=FrameRenderer._apply_colormap(gray, params.colormap),
                mode="RGBA",
                min_value=min_value,
                max_value=max_value,
            )

        return RenderedFrame(image=gray, mode="L", min_value=min_value, max_value=max_value)

    @staticmethod
    def _normalize(frame: np.ndarray, params: FrameRenderParams):
        values = np.asarray(frame)
        if np.iscomplexobj(values):
            values = np.abs(values)
        values = values.astype(np.float32, copy=False)

        finite = values[np.isfinite(values)]
        if finite.size == 0:
            min_value = 0.0 if params.min_value is None else float(params.min_value)
            max_value = 1.0 if params.max_value is None else float(params.max_value)
            return np.zeros(values.shape, dtype=np.float32), min_value, max_value

        if params.auto_range:
            min_value = float(np.min(finite)) if params.min_value is None else float(params.min_value)
            max_value = float(np.max(finite)) if params.max_value is None else float(params.max_value)
        else:
            min_value = float(params.min_value if params.min_value is not None else np.min(finite))
            max_value = float(params.max_value if params.max_value is not None else np.max(finite))

        span = max_value - min_value
        if span <= 0:
            return np.zeros(values.shape, dtype=np.float32), min_value, max_value

        clean = np.nan_to_num(values, nan=min_value, posinf=max_value, neginf=min_value)
        return (clean - min_value) / span, min_value, max_value

    @staticmethod
    def _apply_colormap(gray: np.ndarray, colormap: str) -> np.ndarray:
        if colormap == "gray":
            alpha = np.full(gray.shape, 255, dtype=np.uint8)
            return np.stack((gray, gray, gray, alpha), axis=-1)

        try:
            import matplotlib.cm as cm

            cmap = cm.get_cmap(colormap)
            return (cmap(gray.astype(np.float32) / 255.0) * 255).astype(np.uint8)
        except Exception:
            alpha = np.full(gray.shape, 255, dtype=np.uint8)
            return np.stack((gray, gray, gray, alpha), axis=-1)
