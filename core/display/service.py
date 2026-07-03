from __future__ import annotations

from .cache import FrameCache, frame_cache_key
from .renderer import FrameRenderParams, FrameRenderer, RenderedFrame


class FrameRenderService:
    def __init__(self, cache_capacity: int = 12):
        self.cache = FrameCache(capacity=cache_capacity)

    @property
    def cache_size(self) -> int:
        return len(self.cache)

    def clear_cache(self) -> None:
        self.cache.clear()

    def render_source(self, source, frame_index: int = 0, params: FrameRenderParams | None = None) -> RenderedFrame:
        params = params or FrameRenderParams()
        key = frame_cache_key(source.source_id, frame_index, params)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        frame = source.get_frame(frame_index)
        rendered = FrameRenderer.render(frame, params)
        self.cache.put(key, rendered)
        return rendered
