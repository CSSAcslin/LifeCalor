from __future__ import annotations

from collections import OrderedDict
from typing import Hashable

from FrameRenderer import FrameRenderParams, RenderedFrame


def frame_cache_key(source_id: str, frame_index: int, params: FrameRenderParams) -> tuple[Hashable, ...]:
    return (
        source_id,
        int(frame_index),
        params.min_value,
        params.max_value,
        params.auto_range,
        params.use_colormap,
        params.colormap,
    )


class FrameCache:
    def __init__(self, max_items: int = 8, capacity: int | None = None):
        if capacity is not None:
            max_items = capacity
        if max_items < 1:
            raise ValueError("max_items must be >= 1")
        self.max_items = max_items
        self._items: OrderedDict[tuple[Hashable, ...], RenderedFrame] = OrderedDict()

    def __len__(self) -> int:
        return len(self._items)

    def get(self, key):
        item = self._items.get(key)
        if item is not None:
            self._items.move_to_end(key)
        return item

    def put(self, key, value: RenderedFrame) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)

    def clear(self) -> None:
        self._items.clear()
