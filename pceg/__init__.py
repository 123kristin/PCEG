"""Public implementation of Prior-Calibrated Energy Gating (PCEG)."""

from .data import PCEGDataset, build_popularity_map
from .model import PCEG, PCEGConfig

__all__ = ["PCEG", "PCEGConfig", "PCEGDataset", "build_popularity_map"]

