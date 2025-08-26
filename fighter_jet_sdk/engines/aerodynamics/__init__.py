"""
Aerodynamics Engine for Advanced Fighter Jet Design SDK

This module provides comprehensive aerodynamic analysis capabilities including:
- Computational Fluid Dynamics (CFD) solver integration
- Stability and control analysis
- Stealth-aerodynamic optimization
- Multi-speed regime analysis (subsonic to hypersonic)
- Plasma flow analysis for extreme hypersonic conditions
- Non-equilibrium chemistry modeling for Mach 60+ flight
"""

from .engine import AerodynamicsEngine
from .cfd_solver import CFDSolver, MeshGenerator, ConvergenceMonitor
from .stability_analyzer import StabilityAnalyzer
from .stealth_shape_optimizer import StealthShapeOptimizer
from .plasma_flow_solver import PlasmaFlowSolver, PlasmaFlowConditions, PlasmaFlowResults
from .non_equilibrium_cfd import NonEquilibriumCFD, ChemicalKineticsCalculator, ChemicalSpecies, ChemicalReaction

__all__ = [
    'AerodynamicsEngine',
    'CFDSolver',
    'MeshGenerator', 
    'ConvergenceMonitor',
    'StabilityAnalyzer',
    'StealthShapeOptimizer',
    'PlasmaFlowSolver',
    'PlasmaFlowConditions',
    'PlasmaFlowResults',
    'NonEquilibriumCFD',
    'ChemicalKineticsCalculator',
    'ChemicalSpecies',
    'ChemicalReaction'
]