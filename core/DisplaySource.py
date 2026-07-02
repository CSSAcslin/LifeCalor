from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class DisplaySource:
    source_id: str
    source_type: str
    source_name: str
    source_format: str
    array: np.ndarray
    time_point: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def shape(self):
        return self.array.shape

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def frame_count(self) -> int:
        if self.array.ndim <= 2:
            return 1
        return int(self.array.shape[0])

    @property
    def frame_shape(self):
        if self.array.ndim <= 2:
            return self.array.shape
        return self.array.shape[1:]

    def get_frame(self, index: int = 0) -> np.ndarray:
        if self.array.ndim <= 2:
            if index not in (0, -1):
                raise IndexError(f"frame index {index} out of range for 2D source")
            return self.array
        if index < 0:
            index += self.frame_count
        if index < 0 or index >= self.frame_count:
            raise IndexError(f"frame index {index} out of range for {self.frame_count} frames")
        return self.array[index]


class DisplaySourceFactory:
    @staticmethod
    def from_data(data_obj: Any, key: str | None = None) -> DisplaySource:
        class_name = data_obj.__class__.__name__
        source_name = getattr(data_obj, "data_name", "")
        source_format = getattr(data_obj, "data_format", "")
        timestamp = getattr(data_obj, "timestamp", "")
        metadata = {"timestamp": timestamp}

        if class_name == "ProcessedData":
            source_format = getattr(data_obj, "type_processed", source_format)
            if key is not None:
                if hasattr(data_obj, "out_processed_array"):
                    array = data_obj.out_processed_array(key)
                else:
                    array = data_obj.out_processed[key]
                metadata["out_processed_key"] = key
                source_type = "ProcessedData"
            else:
                array = data_obj.data_processed
                source_type = "ProcessedData"
        else:
            array = data_obj.data_origin
            source_type = class_name

        if not isinstance(array, np.ndarray):
            array = np.asarray(array)

        return DisplaySource(
            source_id=f"{class_name}:{source_name}:{timestamp}:{key or 'primary'}:{id(array)}",
            source_type=source_type,
            source_name=source_name,
            source_format=source_format,
            array=array,
            time_point=getattr(data_obj, "time_point", None),
            metadata=metadata,
        )
