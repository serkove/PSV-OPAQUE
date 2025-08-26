"""Module library for component management."""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Any
from dataclasses import dataclass, field
from datetime import datetime

from ...common.data_models import Module, ModuleInterface, PhysicalProperties
from ...common.enums import ModuleType
from ...core.errors import ValidationError, ConfigurationError


@dataclass
class ModuleSearchCriteria:
    """Criteria for searching modules in the library."""
    module_type: Optional[ModuleType] = None
    max_mass: Optional[float] = None
    max_power: Optional[float] = None
    required_interfaces: List[str] = field(default_factory=list)
    compatibility_tags: List[str] = field(default_factory=list)


class ModuleLibrary:
    """Manages a library of aircraft modules and their compatibility."""
    
    def __init__(self, library_path: Optional[Union[str, Path]] = None):
        """Initialize the module library.
        
        Args:
            library_path: Path to load/save the module library
        """
        self.library_path = Path(library_path) if library_path else None
        self._modules: Dict[str, Module] = {}
        self._module_categories: Dict[ModuleType, Set[str]] = {}
        self._compatibility_matrix: Dict[str, Set[str]] = {}
        self._initialized = False
        
        # Initialize category tracking
        for module_type in ModuleType:
            self._module_categories[module_type] = set()
    
    def initialize(self) -> bool:
        """Initialize the module library."""
        try:
            if self.library_path and self.library_path.exists():
                self.load_from_file(self.library_path)
            else:
                self._create_default_modules()
            
            self._build_compatibility_matrix()
            self._initialized = True
            return True
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize module library: {e}")
    
    def add_module(self, module: Module) -> bool:
        """Add a module to the library.
        
        Args:
            module: Module to add
            
        Returns:
            True if module was added successfully
            
        Raises:
            ValidationError: If module validation fails
        """
        # Validate module
        validation_errors = module.validate_module()
        if validation_errors:
            raise ValidationError(f"Module validation failed: {validation_errors}")
        
        # Check for duplicate ID
        if module.module_id in self._modules:
            raise ValidationError(f"Module with ID {module.module_id} already exists")
        
        # Add module
        self._modules[module.module_id] = module
        self._module_categories[module.module_type].add(module.module_id)
        
        # Update compatibility matrix
        self._update_compatibility_for_module(module)
        
        return True
    
    def remove_module(self, module_id: str) -> bool:
        """Remove a module from the library.
        
        Args:
            module_id: ID of module to remove
            
        Returns:
            True if module was removed
        """
        if module_id not in self._modules:
            return False
        
        module = self._modules[module_id]
        
        # Remove from categories
        self._module_categories[module.module_type].discard(module_id)
        
        # Remove from compatibility matrix
        if module_id in self._compatibility_matrix:
            del self._compatibility_matrix[module_id]
        
        # Remove references from other modules' compatibility
        for compat_set in self._compatibility_matrix.values():
            compat_set.discard(module_id)
        
        # Remove module
        del self._modules[module_id]
        
        return True
    
    def get_module(self, module_id: str) -> Optional[Module]:
        """Get a module by ID.
        
        Args:
            module_id: Module ID
            
        Returns:
            Module if found, None otherwise
        """
        return self._modules.get(module_id)
    
    def get_modules_by_type(self, module_type: ModuleType) -> List[Module]:
        """Get all modules of a specific type.
        
        Args:
            module_type: Type of modules to retrieve
            
        Returns:
            List of modules of the specified type
        """
        module_ids = self._module_categories.get(module_type, set())
        return [self._modules[mid] for mid in module_ids if mid in self._modules]
    
    def search_modules(self, criteria: ModuleSearchCriteria) -> List[Module]:
        """Search for modules matching the given criteria.
        
        Args:
            criteria: Search criteria
            
        Returns:
            List of matching modules
        """
        results = []
        
        for module in self._modules.values():
            if self._matches_criteria(module, criteria):
                results.append(module)
        
        return results
    
    def get_compatible_modules(self, module_id: str) -> List[Module]:
        """Get all modules compatible with the specified module.
        
        Args:
            module_id: ID of the reference module
            
        Returns:
            List of compatible modules
        """
        if module_id not in self._compatibility_matrix:
            return []
        
        compatible_ids = self._compatibility_matrix[module_id]
        return [self._modules[mid] for mid in compatible_ids if mid in self._modules]
    
    def check_module_compatibility(self, module1_id: str, module2_id: str) -> bool:
        """Check if two modules are compatible.
        
        Args:
            module1_id: ID of first module
            module2_id: ID of second module
            
        Returns:
            True if modules are compatible
        """
        if module1_id not in self._compatibility_matrix:
            return False
        
        return module2_id in self._compatibility_matrix[module1_id]
    
    def get_module_statistics(self) -> Dict[str, Any]:
        """Get statistics about the module library.
        
        Returns:
            Dictionary with library statistics
        """
        stats = {
            'total_modules': len(self._modules),
            'modules_by_type': {},
            'average_mass': 0.0,
            'average_power': 0.0,
            'compatibility_coverage': 0.0
        }
        
        # Count modules by type
        for module_type, module_ids in self._module_categories.items():
            stats['modules_by_type'][module_type.name] = len(module_ids)
        
        # Calculate averages
        if self._modules:
            total_mass = 0.0
            total_power = 0.0
            mass_count = 0
            power_count = 0
            
            for module in self._modules.values():
                if module.physical_properties and module.physical_properties.mass > 0:
                    total_mass += module.physical_properties.mass
                    mass_count += 1
                
                power = module.calculate_total_power_consumption()
                if power > 0:
                    total_power += power
                    power_count += 1
            
            if mass_count > 0:
                stats['average_mass'] = total_mass / mass_count
            if power_count > 0:
                stats['average_power'] = total_power / power_count
        
        # Calculate compatibility coverage
        if self._modules:
            total_possible_pairs = len(self._modules) * (len(self._modules) - 1)
            if total_possible_pairs > 0:
                compatible_pairs = sum(len(compat_set) for compat_set in self._compatibility_matrix.values())
                stats['compatibility_coverage'] = compatible_pairs / total_possible_pairs
        
        return stats
    
    def save_to_file(self, file_path: Optional[Union[str, Path]] = None) -> None:
        """Save the module library to file.
        
        Args:
            file_path: Path to save to (uses library_path if not provided)
        """
        save_path = Path(file_path) if file_path else self.library_path
        if not save_path:
            raise ConfigurationError("No file path specified for saving")
        
        # Prepare data for serialization
        data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'version': '1.0',
                'total_modules': len(self._modules)
            },
            'modules': {},
            'compatibility_matrix': {}
        }
        
        # Serialize modules
        for module_id, module in self._modules.items():
            data['modules'][module_id] = module.to_dict()
        
        # Serialize compatibility matrix
        for module_id, compat_set in self._compatibility_matrix.items():
            data['compatibility_matrix'][module_id] = list(compat_set)
        
        # Save based on file extension
        if save_path.suffix.lower() in ['.yaml', '.yml']:
            with open(save_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
        else:
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
    
    def load_from_file(self, file_path: Union[str, Path]) -> None:
        """Load the module library from file.
        
        Args:
            file_path: Path to load from
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ConfigurationError(f"Library file not found: {file_path}")
        
        # Load data based on file extension
        with open(file_path, 'r') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        # Clear existing data
        self._modules.clear()
        self._module_categories.clear()
        self._compatibility_matrix.clear()
        
        # Initialize categories
        for module_type in ModuleType:
            self._module_categories[module_type] = set()
        
        # Load modules
        if 'modules' in data:
            for module_id, module_data in data['modules'].items():
                try:
                    module = Module.from_dict(module_data)
                    self._modules[module_id] = module
                    self._module_categories[module.module_type].add(module_id)
                except Exception as e:
                    print(f"Warning: Failed to load module {module_id}: {e}")
        
        # Load compatibility matrix
        if 'compatibility_matrix' in data:
            for module_id, compat_list in data['compatibility_matrix'].items():
                if module_id in self._modules:
                    self._compatibility_matrix[module_id] = set(compat_list)
    
    def _matches_criteria(self, module: Module, criteria: ModuleSearchCriteria) -> bool:
        """Check if a module matches the search criteria."""
        # Check module type
        if criteria.module_type and module.module_type != criteria.module_type:
            return False
        
        # Check mass constraint
        if criteria.max_mass is not None:
            if not module.physical_properties or module.physical_properties.mass > criteria.max_mass:
                return False
        
        # Check power constraint
        if criteria.max_power is not None:
            total_power = module.calculate_total_power_consumption()
            if total_power > criteria.max_power:
                return False
        
        # Check required interfaces
        if criteria.required_interfaces:
            module_interface_ids = {ei.interface_id for ei in module.electrical_interfaces}
            module_interface_ids.update({mi.interface_id for mi in module.mechanical_interfaces})
            
            for required_interface in criteria.required_interfaces:
                if required_interface not in module_interface_ids:
                    return False
        
        # Check compatibility tags
        if criteria.compatibility_tags:
            for tag in criteria.compatibility_tags:
                if tag not in module.compatibility_requirements:
                    return False
        
        return True
    
    def _update_compatibility_for_module(self, module: Module) -> None:
        """Update compatibility matrix for a new module."""
        module_id = module.module_id
        self._compatibility_matrix[module_id] = set()
        
        # Check compatibility with all existing modules
        for other_id, other_module in self._modules.items():
            if other_id != module_id:
                if module.is_compatible_with(other_module):
                    self._compatibility_matrix[module_id].add(other_id)
                    
                    # Update reverse compatibility
                    if other_id not in self._compatibility_matrix:
                        self._compatibility_matrix[other_id] = set()
                    self._compatibility_matrix[other_id].add(module_id)
    
    def _build_compatibility_matrix(self) -> None:
        """Build the complete compatibility matrix."""
        self._compatibility_matrix.clear()
        
        module_ids = list(self._modules.keys())
        
        for i, module1_id in enumerate(module_ids):
            self._compatibility_matrix[module1_id] = set()
            module1 = self._modules[module1_id]
            
            for j, module2_id in enumerate(module_ids):
                if i != j:
                    module2 = self._modules[module2_id]
                    if module1.is_compatible_with(module2):
                        self._compatibility_matrix[module1_id].add(module2_id)
    
    def _create_default_modules(self) -> None:
        """Create a set of default modules for testing and demonstration."""
        # Create default cockpit module
        cockpit = Module(
            name="Standard Cockpit",
            module_type=ModuleType.COCKPIT,
            description="Single-seat fighter cockpit with advanced avionics",
            physical_properties=PhysicalProperties(
                mass=500.0,
                center_of_gravity=(0.0, 0.0, 0.0),
                moments_of_inertia=(100.0, 150.0, 200.0),
                dimensions=(3.0, 1.5, 1.2)
            ),
            performance_characteristics={
                'visibility_angle': 360.0,
                'g_force_rating': 9.0,
                'ejection_altitude_max': 15000.0
            },
            compatibility_requirements=['requires_avionics', 'requires_life_support']
        )
        
        # Create default sensor module
        sensor = Module(
            name="AESA Radar System",
            module_type=ModuleType.SENSOR,
            description="Active Electronically Scanned Array radar",
            physical_properties=PhysicalProperties(
                mass=200.0,
                center_of_gravity=(0.0, 0.0, 0.5),
                moments_of_inertia=(50.0, 50.0, 25.0),
                dimensions=(1.0, 1.0, 0.3)
            ),
            performance_characteristics={
                'detection_range': 150.0,
                'tracking_targets': 20.0,
                'frequency_band': 10.0
            },
            compatibility_requirements=['requires_cooling', 'requires_high_power']
        )
        
        # Create default payload module
        payload = Module(
            name="Multi-Role Payload Bay",
            module_type=ModuleType.PAYLOAD,
            description="Configurable payload bay for various mission types",
            physical_properties=PhysicalProperties(
                mass=300.0,
                center_of_gravity=(0.0, 0.0, -0.5),
                moments_of_inertia=(75.0, 100.0, 125.0),
                dimensions=(4.0, 2.0, 1.0)
            ),
            performance_characteristics={
                'max_payload_mass': 2000.0,
                'volume_capacity': 8.0,
                'hardpoint_count': 6.0
            },
            compatibility_requirements=['structural_reinforcement']
        )
        
        # Add modules to library
        self._modules[cockpit.module_id] = cockpit
        self._modules[sensor.module_id] = sensor
        self._modules[payload.module_id] = payload
        
        # Update categories
        self._module_categories[ModuleType.COCKPIT].add(cockpit.module_id)
        self._module_categories[ModuleType.SENSOR].add(sensor.module_id)
        self._module_categories[ModuleType.PAYLOAD].add(payload.module_id)