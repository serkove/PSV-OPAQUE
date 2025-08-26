"""Engine modules for specialized aircraft design capabilities."""

from .design import DesignEngine
from .materials import MaterialsEngine
from .propulsion import PropulsionEngine
from .sensors import SensorsEngine
from .aerodynamics import AerodynamicsEngine
from .manufacturing import ManufacturingEngine
from .structural import StructuralEngine

__all__ = [
    'DesignEngine', 'MaterialsEngine', 'PropulsionEngine',
    'SensorsEngine', 'AerodynamicsEngine', 'ManufacturingEngine',
    'StructuralEngine'
]