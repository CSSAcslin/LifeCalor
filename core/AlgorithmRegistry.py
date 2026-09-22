"""Backward-compatible import path for the runtime callable registry."""

from compute.registry import AlgorithmRegistry, LegacyAlgorithmSpec as AlgorithmSpec

__all__ = ["AlgorithmRegistry", "AlgorithmSpec"]
