from .source import DisplaySource, DisplaySourceFactory
from .renderer import FrameRenderParams, FrameRenderer, RenderedFrame
from .cache import FrameCache, frame_cache_key
from .service import FrameRenderService
from .worker import FrameRenderWorker
from .status import render_status_update
from .canvas_signals import CanvasSignalBinder, disconnect_canvas_signal
from .render_controller import RenderRequestState, RenderController


def __getattr__(name):
    if name == "DisplayCanvasController":
        from .canvas_controller import DisplayCanvasController
        return DisplayCanvasController
    raise AttributeError(name)


__all__ = [
    "DisplaySource",
    "DisplaySourceFactory",
    "FrameRenderParams",
    "FrameRenderer",
    "RenderedFrame",
    "FrameCache",
    "frame_cache_key",
    "FrameRenderService",
    "FrameRenderWorker",
    "render_status_update",
    "CanvasSignalBinder",
    "disconnect_canvas_signal",
    "RenderRequestState",
    "RenderController",
    "DisplayCanvasController",
]
