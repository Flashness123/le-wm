from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.temporal_dataset import TemporalSample


class ModelAdapterInterface(ABC):
    """
    Converts a generic TemporalSample into whatever one concrete model needs.

    The generic data pipeline must not know about LeWM-specific concepts such
    as 224x224 images, ImageNet normalization, one-hot actions, or a 192-d
    hidden representation.
    """

    @abstractmethod
    def adapt(self, sample: "TemporalSample") -> dict[str, Any]:
        ...
