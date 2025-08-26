"""
Deployable Assets Engine

This module provides simulation capabilities for deployable assets including
small UAV swarms and decoy systems that can be deployed from the main aircraft.
"""

from .uav_swarm import UAVSwarmSimulator
from .decoy_system import DecoySystemSimulator
# from .engine import DeployableAssetsEngine  # Will be implemented after subtasks

__all__ = [
    'UAVSwarmSimulator',
    'DecoySystemSimulator', 
    # 'DeployableAssetsEngine'
]