"""
Aerodynamics Engine for Advanced Fighter Jet Design SDK

This module provides comprehensive aerodynamic analysis capabilities including:
- Computational Fluid Dynamics (CFD) solver integration
- Stability and control analysis
- Stealth-aerodynamic optimization
- Multi-speed regime analysis (subsonic to hypersonic)
"""

from .engine import AerodynamicsEngine
from .cfd_solver import CFDSolver, MeshGenerator, ConvergenceMonitor
from .stability_analyzer import StabilityAnalyzer
from .stealth_shape_optimizer import StealthShapeOptimizer

__all__ = [
    'AerodynamicsEngine',
    'CFDSolver',
    'MeshGenerator', 
    'ConvergenceMonitor',
    'StabilityAnalyzer',
    'StealthShapeOptimizer'
]