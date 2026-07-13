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
        values = np.asarray(frame)
        if values.ndim == 3 and values.shape[-1] in (3, 4) and not params.use_colormap:
            color = FrameRenderer._color_uint8(values)
            if color.shape[-1] == 3:
                alpha = np.full(color.shape[:2] + (1,), 255, dtype=np.uint8)
                color = np.concatenate((color, alpha), axis=-1)
            return RenderedFrame(image=color, mode="RGBA", min_value=float(np.min(values)), max_value=float(np.max(values)))
        if values.ndim == 3 and values.shape[-1] in (3, 4):
            values = values[..., 0] * 0.2126 + values[..., 1] * 0.7152 + values[..., 2] * 0.0722
        normalized, min_value, max_value = FrameRenderer._normalize(values, params)
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
    def _color_uint8(values: np.ndarray) -> np.ndarray:
        if values.dtype == np.uint8:
            return values.copy()
        numeric = values.astype(np.float32)
        finite = numeric[np.isfinite(numeric)]
        if finite.size == 0:
            return np.zeros(values.shape, dtype=np.uint8)
        low, high = float(finite.min()), float(finite.max())
        if high <= low:
            return np.zeros(values.shape, dtype=np.uint8)
        return np.clip((numeric - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)

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

        if colormap == "Rainbow*":
            cmap = FrameRenderer._rainbow_colormap()
        else:
            cmap = FrameRenderer._matplotlib_colormap(colormap)
        return (cmap(gray.astype(np.float32) / 255.0) * 255).astype(np.uint8)

    @staticmethod
    def _matplotlib_colormap(colormap: str):
        try:
            from matplotlib import colormaps
            try:
                return colormaps.get_cmap(colormap)
            except ValueError:
                return colormaps.get_cmap("jet")
        except Exception:
            pass

        try:
            import matplotlib.cm as cm
            get_cmap = getattr(cm, "get_cmap", None)
            if callable(get_cmap):
                try:
                    return get_cmap(colormap)
                except Exception:
                    return get_cmap("jet")
            jet = getattr(cm, "jet", None)
            if jet is not None:
                return jet
        except Exception:
            pass

        return FrameRenderer._rainbow_colormap()

    @staticmethod
    def _rainbow_colormap():
        stops = np.array([0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 1.0], dtype=np.float32)
        red = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)
        green = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        blue = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        def cmap(values):
            values = np.asarray(values, dtype=np.float32)
            rgba = np.empty(values.shape + (4,), dtype=np.float32)
            rgba[..., 0] = np.interp(values, stops, red)
            rgba[..., 1] = np.interp(values, stops, green)
            rgba[..., 2] = np.interp(values, stops, blue)
            rgba[..., 3] = 1.0
            return rgba

        return cmap

