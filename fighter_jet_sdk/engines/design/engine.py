"""Main Design Engine for modular aircraft management."""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging

from ...common.interfaces import BaseEngine
from ...common.data_models import AircraftConfiguration, Module, BasePlatform
from ...common.enums import ModuleType
from ...core.errors import ValidationError, ConfigurationError

from .module_library import ModuleLibrary, ModuleSearchCriteria
from .interface_validator import InterfaceValidator, CompatibilityResult, InterfaceConflict


class DesignEngine(BaseEngine):
    """Main engine for modular aircraft design and management."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Design Engine.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        self.module_library: Optional[ModuleLibrary] = None
        self.interface_validator: Optional[InterfaceValidator] = None
        self.logger = logging.getLogger(__name__)
        
        # Configuration options
        self.library_path = self.config.get('library_path', 'data/module_library.json')
        self.auto_save = self.config.get('auto_save', True)
        self.validation_level = self.config.get('validation_level', 'strict')
    
    def initialize(self) -> bool:
        """Initialize the Design Engine.
        
        Returns:
            True if initialization successful
        """
        try:
            # Initialize module library
            self.module_library = ModuleLibrary(self.library_path)
            if not self.module_library.initialize():
                raise ConfigurationError("Failed to initialize module library")
            
            # Initialize interface validator
            self.interface_validator = InterfaceValidator()
            
            self.initialized = True
            self.logger.info("Design Engine initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Design Engine: {e}")
            return False
    
    def validate_input(self, data: Any) -> bool:
        """Validate input data.
        
        Args:
            data: Input data to validate
            
        Returns:
            True if data is valid
        """
        if not self.initialized:
            return False
        
        if isinstance(data, AircraftConfiguration):
            errors = data.validate_configuration()
            return len(errors) == 0
        elif isinstance(data, Module):
            errors = data.validate_module()
            return len(errors) == 0
        
        return True
    
    def process(self, data: Any) -> Any:
        """Process input data.
        
        Args:
            data: Data to process
            
        Returns:
            Processed results
        """
        if not self.initialized:
            raise ConfigurationError("Design Engine not initialized")
        
        # This is a placeholder - specific processing would be implemented
        # based on the type of operation requested
        return {"status": "processed", "data": data}
    
    # Module Library Operations
    
    def add_module_to_library(self, module: Module) -> bool:
        """Add a module to the library.
        
        Args:
            module: Module to add
            
        Returns:
            True if module was added successfully
        """
        if not self.module_library:
            raise ConfigurationError("Module library not initialized")
        
        try:
            result = self.module_library.add_module(module)
            
            if result and self.auto_save:
                self.save_module_library()
            
            return result
        except ValidationError as e:
            self.logger.error(f"Failed to add module: {e}")
            return False
    
    def remove_module_from_library(self, module_id: str) -> bool:
        """Remove a module from the library.
        
        Args:
            module_id: ID of module to remove
            
        Returns:
            True if module was removed
        """
        if not self.module_library:
            raise ConfigurationError("Module library not initialized")
        
        result = self.module_library.remove_module(module_id)
        
        if result and self.auto_save:
            self.save_module_library()
        
        return result
    
    def get_module_from_library(self, module_id: str) -> Optional[Module]:
        """Get a module from the library.
        
        Args:
            module_id: Module ID
            
        Returns:
            Module if found, None otherwise
        """
        if not self.module_library:
            return None
        
        return self.module_library.get_module(module_id)
    
    def search_modules(self, criteria: ModuleSearchCriteria) -> List[Module]:
        """Search for modules matching criteria.
        
        Args:
            criteria: Search criteria
            
        Returns:
            List of matching modules
        """
        if not self.module_library:
            return []
        
        return self.module_library.search_modules(criteria)
    
    def get_modules_by_type(self, module_type: ModuleType) -> List[Module]:
        """Get all modules of a specific type.
        
        Args:
            module_type: Type of modules to retrieve
            
        Returns:
            List of modules of the specified type
        """
        if not self.module_library:
            return []
        
        return self.module_library.get_modules_by_type(module_type)
    
    def get_compatible_modules(self, module_id: str) -> List[Module]:
        """Get modules compatible with the specified module.
        
        Args:
            module_id: ID of the reference module
            
        Returns:
            List of compatible modules
        """
        if not self.module_library:
            return []
        
        return self.module_library.get_compatible_modules(module_id)
    
    def get_library_statistics(self) -> Dict[str, Any]:
        """Get module library statistics.
        
        Returns:
            Dictionary with library statistics
        """
        if not self.module_library:
            return {}
        
        return self.module_library.get_module_statistics()
    
    # Interface Validation Operations
    
    def validate_module_compatibility(self, module1: Module, module2: Module) -> CompatibilityResult:
        """Validate compatibility between two modules.
        
        Args:
            module1: First module
            module2: Second module
            
        Returns:
            CompatibilityResult with detailed compatibility information
        """
        if not self.interface_validator:
            raise ConfigurationError("Interface validator not initialized")
        
        return self.interface_validator.validate_module_compatibility(module1, module2)
    
    def validate_configuration_interfaces(self, config: AircraftConfiguration) -> List[InterfaceConflict]:
        """Validate all interfaces in an aircraft configuration.
        
        Args:
            config: Aircraft configuration to validate
            
        Returns:
            List of interface conflicts found
        """
        if not self.interface_validator:
            raise ConfigurationError("Interface validator not initialized")
        
        return self.interface_validator.validate_configuration_interfaces(config)
    
    def get_compatibility_matrix(self, modules: List[Module]) -> Dict[str, Dict[str, CompatibilityResult]]:
        """Generate compatibility matrix for a list of modules.
        
        Args:
            modules: List of modules to analyze
            
        Returns:
            Matrix of compatibility results
        """
        if not self.interface_validator:
            raise ConfigurationError("Interface validator not initialized")
        
        return self.interface_validator.get_compatibility_matrix(modules)
    
    def suggest_interface_solutions(self, conflict: InterfaceConflict) -> List[Dict[str, Any]]:
        """Suggest solutions for an interface conflict.
        
        Args:
            conflict: Interface conflict to resolve
            
        Returns:
            List of suggested solutions
        """
        if not self.interface_validator:
            return []
        
        return self.interface_validator.suggest_interface_solutions(conflict)
    
    # Configuration Management Operations
    
    def create_base_configuration(self, platform: BasePlatform, name: str = "New Configuration") -> AircraftConfiguration:
        """Create a new base aircraft configuration.
        
        Args:
            platform: Base platform to use
            name: Name for the configuration
            
        Returns:
            New aircraft configuration
        """
        config = AircraftConfiguration(
            name=name,
            description=f"Aircraft configuration based on {platform.name}",
            base_platform=platform
        )
        
        return config
    
    def add_module_to_configuration(self, config: AircraftConfiguration, module: Module) -> bool:
        """Add a module to an aircraft configuration with validation.
        
        Args:
            config: Aircraft configuration
            module: Module to add
            
        Returns:
            True if module was added successfully
        """
        # Validate module compatibility with existing modules
        if self.validation_level == 'strict':
            for existing_module in config.modules:
                compat_result = self.validate_module_compatibility(existing_module, module)
                if compat_result.level.value == 'incompatible':
                    self.logger.warning(f"Module {module.name} incompatible with {existing_module.name}: {compat_result.issues}")
                    return False
        
        # Add module to configuration
        return config.add_module(module)
    
    def remove_module_from_configuration(self, config: AircraftConfiguration, module_id: str) -> bool:
        """Remove a module from an aircraft configuration.
        
        Args:
            config: Aircraft configuration
            module_id: ID of module to remove
            
        Returns:
            True if module was removed
        """
        return config.remove_module(module_id)
    
    def validate_complete_configuration(self, config: AircraftConfiguration) -> Dict[str, Any]:
        """Perform complete validation of an aircraft configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'interface_conflicts': [],
            'compatibility_issues': []
        }
        
        # Basic configuration validation
        config_errors = config.validate_configuration()
        results['errors'].extend(config_errors)
        
        # Interface validation
        if self.interface_validator:
            conflicts = self.validate_configuration_interfaces(config)
            results['interface_conflicts'] = conflicts
            
            # Add critical conflicts to errors
            critical_conflicts = [c for c in conflicts if c.severity == 'critical']
            for conflict in critical_conflicts:
                results['errors'].append(f"Critical interface conflict: {conflict.description}")
        
        # Module compatibility validation
        for i, module1 in enumerate(config.modules):
            for module2 in config.modules[i+1:]:
                compat_result = self.validate_module_compatibility(module1, module2)
                if compat_result.level.value == 'incompatible':
                    results['compatibility_issues'].append({
                        'module1': module1.name,
                        'module2': module2.name,
                        'issues': compat_result.issues
                    })
        
        # Determine overall validity
        results['valid'] = len(results['errors']) == 0 and len(results['compatibility_issues']) == 0
        
        return results
    
    def optimize_module_selection(self, config: AircraftConfiguration, 
                                criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest optimal module selections for a configuration.
        
        Args:
            config: Current configuration
            criteria: Optimization criteria (weight, cost, performance, etc.)
            
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        if not self.module_library:
            return suggestions
        
        # For each module type in the configuration, suggest alternatives
        for module in config.modules:
            # Find alternative modules of the same type
            alternatives = self.get_modules_by_type(module.module_type)
            
            for alt_module in alternatives:
                if alt_module.module_id != module.module_id:
                    # Check if alternative is better based on criteria
                    improvement = self._evaluate_module_improvement(module, alt_module, criteria)
                    if improvement['score'] > 0:
                        suggestions.append({
                            'current_module': module.name,
                            'suggested_module': alt_module.name,
                            'improvement': improvement,
                            'compatibility': self.validate_module_compatibility(module, alt_module)
                        })
        
        # Sort suggestions by improvement score
        suggestions.sort(key=lambda x: x['improvement']['score'], reverse=True)
        
        return suggestions
    
    # Utility Methods
    
    def save_module_library(self) -> bool:
        """Save the module library to file.
        
        Returns:
            True if save successful
        """
        if not self.module_library:
            return False
        
        try:
            self.module_library.save_to_file()
            return True
        except Exception as e:
            self.logger.error(f"Failed to save module library: {e}")
            return False
    
    def load_module_library(self, file_path: Optional[Union[str, Path]] = None) -> bool:
        """Load module library from file.
        
        Args:
            file_path: Path to load from (uses default if not provided)
            
        Returns:
            True if load successful
        """
        if not self.module_library:
            return False
        
        try:
            load_path = file_path or self.library_path
            self.module_library.load_from_file(load_path)
            return True
        except Exception as e:
            self.logger.error(f"Failed to load module library: {e}")
            return False
    
    def _evaluate_module_improvement(self, current: Module, alternative: Module, 
                                   criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate if an alternative module is an improvement.
        
        Args:
            current: Current module
            alternative: Alternative module
            criteria: Evaluation criteria
            
        Returns:
            Improvement evaluation
        """
        improvement = {'score': 0.0, 'details': {}}
        
        # Weight optimization
        if 'minimize_weight' in criteria and criteria['minimize_weight']:
            if (current.physical_properties and alternative.physical_properties and
                alternative.physical_properties.mass < current.physical_properties.mass):
                weight_saving = current.physical_properties.mass - alternative.physical_properties.mass
                improvement['score'] += weight_saving * criteria.get('weight_factor', 1.0)
                improvement['details']['weight_saving'] = weight_saving
        
        # Power optimization
        if 'minimize_power' in criteria and criteria['minimize_power']:
            current_power = current.calculate_total_power_consumption()
            alt_power = alternative.calculate_total_power_consumption()
            if alt_power < current_power:
                power_saving = current_power - alt_power
                improvement['score'] += power_saving * criteria.get('power_factor', 0.1)
                improvement['details']['power_saving'] = power_saving
        
        # Performance optimization
        if 'maximize_performance' in criteria and criteria['maximize_performance']:
            # This would need specific performance metrics defined
            # For now, use a simple heuristic
            current_perf = sum(current.performance_characteristics.values())
            alt_perf = sum(alternative.performance_characteristics.values())
            if alt_perf > current_perf:
                perf_gain = alt_perf - current_perf
                improvement['score'] += perf_gain * criteria.get('performance_factor', 0.01)
                improvement['details']['performance_gain'] = perf_gain
        
        return improvement