"""Design Engine for modular aircraft management."""

from .engine import DesignEngine
from .module_library import ModuleLibrary, ModuleSearchCriteria
from .interface_validator import InterfaceValidator
from .configuration_optimizer import ConfigurationOptimizer

__all__ = [
    'DesignEngine',
    'ModuleLibrary', 
    'ModuleSearchCriteria',
    'InterfaceValidator',
    'ConfigurationOptimizer'
]