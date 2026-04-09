"""
ML Backends

Concrete implementations of :class:`~hpc_bottleneck_detector.ml.IMLBackend`.
"""

from .default_backend import DefaultBackend

__all__ = ["DefaultBackend"]
