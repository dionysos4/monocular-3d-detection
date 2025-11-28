"""
Monocular 3D Object Detection Package

A modular package for estimating 3D objects from a single calibrated camera
and a known transformation between camera and plane coordinate system.

Public API:
    MonoDetection: Main facade class for monocular 3D object detection
"""

from .core import MonoDetection

__all__ = ["MonoDetection"]
__version__ = "1.0.0"
