"""Tests for the Design Engine module library and interface validator."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from fighter_jet_sdk.engines.design import (
    DesignEngine, ModuleLibrary, InterfaceValidator, ModuleSearchCriteria
)
from fighter_jet_sdk.engines.design.interface_validator import (
    CompatibilityLevel, CompatibilityResult, InterfaceConflict
)
from fighter_jet_sdk.common.data_models import (
    Module, AircraftConfiguration, BasePlatform, PhysicalProperties,
    ElectricalInterface, MechanicalInterface
)
from fighter_jet_sdk.common.enums import ModuleType
from fighter_jet_sdk.core.errors import ValidationError, ConfigurationError


class TestModuleLibrary:
    """Test cases for ModuleLibrary class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.library_path = Path(self.temp_dir) / "test_library.json"
        self.module_library = ModuleLibrary(self.library_path)
        
        # Create test modules
        self.test_cockpit = Module(
            name="Test Cockpit",
            module_type=ModuleType.COCKPIT,
            description="Test cockpit module",
            physical_properties=PhysicalProperties(
                mass=500.0,
                center_of_gravity=(0.0, 0.0, 0.0),
                moments_of_inertia=(100.0, 150.0, 200.0),
                dimensions=(3.0, 1.5, 1.2)
            ),
            electrical_interfaces=[
                ElectricalInterface(
                    interface_id="cockpit_power",
                    voltage=28.0,
                    current_capacity=50.0,
                    power_consumption=1000.0,
                    protocol="MIL-STD-1553"
                )
            ],
            performance_characteristics={'visibility_angle': 360.0}
        )
        
        self.test_sensor = Module(
            name="Test Sensor",
            module_type=ModuleType.SENSOR,
            description="Test sensor module",
            physical_properties=PhysicalProperties(
                mass=200.0,
                center_of_gravity=(0.0, 0.0, 0.5),
                moments_of_inertia=(50.0, 50.0, 25.0),
                dimensions=(1.0, 1.0, 0.3)
            ),
            electrical_interfaces=[
                ElectricalInterface(
                    interface_id="sensor_power",
                    voltage=28.0,
                    current_capacity=20.0,
                    power_consumption=500.0,
                    protocol="MIL-STD-1553"
                )
            ],
            performance_characteristics={'detection_range': 150.0}
        )
    
    def test_initialize_empty_library(self):
        """Test initializing an empty module library."""
        assert self.module_library.initialize()
        assert self.module_library._initialized
        assert len(self.module_library._modules) > 0  # Should have default modules
    
    def test_add_module_success(self):
        """Test successfully adding a module to the library."""
        self.module_library.initialize()
        
        result = self.module_library.add_module(self.test_cockpit)
        assert result is True
        assert self.test_cockpit.module_id in self.module_library._modules
        assert self.test_cockpit.module_id in self.module_library._module_categories[ModuleType.COCKPIT]
    
    def test_add_duplicate_module_fails(self):
        """Test that adding a duplicate module fails."""
        self.module_library.initialize()
        self.module_library.add_module(self.test_cockpit)
        
        with pytest.raises(ValidationError):
            self.module_library.add_module(self.test_cockpit)
    
    def test_add_invalid_module_fails(self):
        """Test that adding an invalid module fails."""
        self.module_library.initialize()
        
        invalid_module = Module(name="", module_type=ModuleType.COCKPIT)  # Empty name
        
        with pytest.raises(ValidationError):
            self.module_library.add_module(invalid_module)
    
    def test_remove_module_success(self):
        """Test successfully removing a module from the library."""
        self.module_library.initialize()
        self.module_library.add_module(self.test_cockpit)
        
        result = self.module_library.remove_module(self.test_cockpit.module_id)
        assert result is True
        assert self.test_cockpit.module_id not in self.module_library._modules
        assert self.test_cockpit.module_id not in self.module_library._module_categories[ModuleType.COCKPIT]
    
    def test_remove_nonexistent_module(self):
        """Test removing a non-existent module."""
        self.module_library.initialize()
        
        result = self.module_library.remove_module("nonexistent_id")
        assert result is False
    
    def test_get_module_success(self):
        """Test successfully retrieving a module."""
        self.module_library.initialize()
        self.module_library.add_module(self.test_cockpit)
        
        retrieved = self.module_library.get_module(self.test_cockpit.module_id)
        assert retrieved is not None
        assert retrieved.module_id == self.test_cockpit.module_id
        assert retrieved.name == self.test_cockpit.name
    
    def test_get_nonexistent_module(self):
        """Test retrieving a non-existent module."""
        self.module_library.initialize()
        
        retrieved = self.module_library.get_module("nonexistent_id")
        assert retrieved is None
    
    def test_get_modules_by_type(self):
        """Test retrieving modules by type."""
        self.module_library.initialize()
        self.module_library.add_module(self.test_cockpit)
        self.module_library.add_module(self.test_sensor)
        
        cockpit_modules = self.module_library.get_modules_by_type(ModuleType.COCKPIT)
        sensor_modules = self.module_library.get_modules_by_type(ModuleType.SENSOR)
        
        # Should include default modules plus our test modules
        assert len(cockpit_modules) >= 1
        assert len(sensor_modules) >= 1
        
        # Check that our test modules are included
        cockpit_ids = {m.module_id for m in cockpit_modules}
        sensor_ids = {m.module_id for m in sensor_modules}
        
        assert self.test_cockpit.module_id in cockpit_ids
        assert self.test_sensor.module_id in sensor_ids
    
    def test_search_modules_by_type(self):
        """Test searching modules by type."""
        self.module_library.initialize()
        self.module_library.add_module(self.test_cockpit)
        
        criteria = ModuleSearchCriteria(module_type=ModuleType.COCKPIT)
        results = self.module_library.search_modules(criteria)
        
        assert len(results) >= 1
        assert all(m.module_type == ModuleType.COCKPIT for m in results)
    
    def test_search_modules_by_mass(self):
        """Test searching modules by mass constraint."""
        self.module_library.initialize()
        self.module_library.add_module(self.test_cockpit)  # 500kg
        self.module_library.add_module(self.test_sensor)   # 200kg
        
        criteria = ModuleSearchCriteria(max_mass=300.0)
        results = self.module_library.search_modules(criteria)
        
        # Should include sensor but not cockpit
        result_ids = {m.module_id for m in results}
        assert self.test_sensor.module_id in result_ids
        assert self.test_cockpit.module_id not in result_ids
    
    def test_search_modules_by_power(self):
        """Test searching modules by power constraint."""
        self.module_library.initialize()
        self.module_library.add_module(self.test_cockpit)  # 1000W
        self.module_library.add_module(self.test_sensor)   # 500W
        
        criteria = ModuleSearchCriteria(max_power=750.0)
        results = self.module_library.search_modules(criteria)
        
        # Should include sensor but not cockpit
        result_ids = {m.module_id for m in results}
        assert self.test_sensor.module_id in result_ids
        assert self.test_cockpit.module_id not in result_ids
    
    def test_check_module_compatibility(self):
        """Test checking module compatibility."""
        self.module_library.initialize()
        self.module_library.add_module(self.test_cockpit)
        self.module_library.add_module(self.test_sensor)
        
        # These modules should be compatible (different types, compatible interfaces)
        compatible = self.module_library.check_module_compatibility(
            self.test_cockpit.module_id, self.test_sensor.module_id
        )
        assert compatible is True
    
    def test_get_compatible_modules(self):
        """Test getting compatible modules."""
        self.module_library.initialize()
        self.module_library.add_module(self.test_cockpit)
        self.module_library.add_module(self.test_sensor)
        
        compatible = self.module_library.get_compatible_modules(self.test_cockpit.module_id)
        compatible_ids = {m.module_id for m in compatible}
        
        # Sensor should be compatible with cockpit
        assert self.test_sensor.module_id in compatible_ids
    
    def test_get_module_statistics(self):
        """Test getting module library statistics."""
        self.module_library.initialize()
        self.module_library.add_module(self.test_cockpit)
        self.module_library.add_module(self.test_sensor)
        
        stats = self.module_library.get_module_statistics()
        
        assert 'total_modules' in stats
        assert 'modules_by_type' in stats
        assert 'average_mass' in stats
        assert 'average_power' in stats
        assert 'compatibility_coverage' in stats
        
        assert stats['total_modules'] >= 2
        assert stats['average_mass'] > 0
        assert stats['average_power'] > 0
    
    def test_save_and_load_library(self):
        """Test saving and loading the module library."""
        self.module_library.initialize()
        self.module_library.add_module(self.test_cockpit)
        
        # Save library
        self.module_library.save_to_file()
        assert self.library_path.exists()
        
        # Create new library and load
        new_library = ModuleLibrary(self.library_path)
        new_library.load_from_file(self.library_path)
        
        # Check that module was loaded
        loaded_module = new_library.get_module(self.test_cockpit.module_id)
        assert loaded_module is not None
        assert loaded_module.name == self.test_cockpit.name


class TestInterfaceValidator:
    """Test cases for InterfaceValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = InterfaceValidator()
        
        # Create test modules with different interface configurations
        self.compatible_module1 = Module(
            name="Compatible Module 1",
            module_type=ModuleType.SENSOR,
            electrical_interfaces=[
                ElectricalInterface(
                    interface_id="power_28v",
                    voltage=28.0,
                    current_capacity=10.0,
                    power_consumption=280.0,
                    protocol="MIL-STD-1553"
                )
            ],
            mechanical_interfaces=[
                MechanicalInterface(
                    interface_id="standard_mount",
                    attachment_type="NATO-Standard",
                    load_capacity=(1000.0, 1000.0, 2000.0),
                    moment_capacity=(500.0, 500.0, 1000.0),
                    position=(0.0, 0.0, 0.0)
                )
            ]
        )
        
        self.compatible_module2 = Module(
            name="Compatible Module 2",
            module_type=ModuleType.AVIONICS,
            electrical_interfaces=[
                ElectricalInterface(
                    interface_id="power_28v",
                    voltage=28.0,
                    current_capacity=5.0,
                    power_consumption=140.0,
                    protocol="MIL-STD-1553"
                )
            ],
            mechanical_interfaces=[
                MechanicalInterface(
                    interface_id="standard_mount",
                    attachment_type="NATO-Standard",
                    load_capacity=(500.0, 500.0, 1000.0),
                    moment_capacity=(250.0, 250.0, 500.0),
                    position=(1.0, 0.0, 0.0)
                )
            ]
        )
        
        self.incompatible_module = Module(
            name="Incompatible Module",
            module_type=ModuleType.PAYLOAD,
            electrical_interfaces=[
                ElectricalInterface(
                    interface_id="power_115v",
                    voltage=115.0,  # Different voltage
                    current_capacity=20.0,
                    power_consumption=2300.0,
                    protocol="ARINC-429"  # Different protocol
                )
            ],
            mechanical_interfaces=[
                MechanicalInterface(
                    interface_id="custom_mount",
                    attachment_type="Custom-Hardpoint",  # Different attachment type
                    load_capacity=(5000.0, 5000.0, 10000.0),
                    moment_capacity=(2500.0, 2500.0, 5000.0),
                    position=(2.0, 0.0, 0.0)
                )
            ],
            compatibility_requirements=["incompatible_with:SENSOR"]  # Explicit incompatibility
        )
    
    def test_validate_compatible_modules(self):
        """Test validating compatible modules."""
        result = self.validator.validate_module_compatibility(
            self.compatible_module1, self.compatible_module2
        )
        
        assert isinstance(result, CompatibilityResult)
        assert result.level in [CompatibilityLevel.FULLY_COMPATIBLE, CompatibilityLevel.COMPATIBLE_WITH_ADAPTER]
        assert result.confidence_score > 0.5
    
    def test_validate_incompatible_modules(self):
        """Test validating incompatible modules."""
        result = self.validator.validate_module_compatibility(
            self.compatible_module1, self.incompatible_module
        )
        
        assert isinstance(result, CompatibilityResult)
        assert result.level in [CompatibilityLevel.INCOMPATIBLE, CompatibilityLevel.REQUIRES_MODIFICATION]
        assert len(result.issues) > 0
    
    def test_validate_configuration_interfaces(self):
        """Test validating configuration interfaces."""
        # Create test configuration
        platform = BasePlatform(
            name="Test Platform",
            base_mass=5000.0,
            power_generation_capacity=10000.0,  # Sufficient power
            attachment_points=[
                MechanicalInterface(
                    interface_id="mount_1",
                    attachment_type="NATO-Standard",
                    load_capacity=(2000.0, 2000.0, 4000.0),
                    moment_capacity=(1000.0, 1000.0, 2000.0),
                    position=(0.0, 0.0, 0.0)
                ),
                MechanicalInterface(
                    interface_id="mount_2",
                    attachment_type="NATO-Standard",
                    load_capacity=(2000.0, 2000.0, 4000.0),
                    moment_capacity=(1000.0, 1000.0, 2000.0),
                    position=(1.0, 0.0, 0.0)
                )
            ]
        )
        
        config = AircraftConfiguration(
            name="Test Configuration",
            base_platform=platform,
            modules=[self.compatible_module1, self.compatible_module2]
        )
        
        conflicts = self.validator.validate_configuration_interfaces(config)
        
        # Should have no critical conflicts for compatible modules
        critical_conflicts = [c for c in conflicts if c.severity == 'critical']
        assert len(critical_conflicts) == 0
    
    def test_validate_power_overload(self):
        """Test detecting power overload conflicts."""
        # Create platform with insufficient power
        platform = BasePlatform(
            name="Low Power Platform",
            base_mass=5000.0,
            power_generation_capacity=100.0,  # Insufficient power
            attachment_points=[
                MechanicalInterface(
                    interface_id="mount_1",
                    attachment_type="NATO-Standard",
                    load_capacity=(2000.0, 2000.0, 4000.0),
                    moment_capacity=(1000.0, 1000.0, 2000.0),
                    position=(0.0, 0.0, 0.0)
                )
            ]
        )
        
        config = AircraftConfiguration(
            name="Overloaded Configuration",
            base_platform=platform,
            modules=[self.compatible_module1]  # Requires 280W, platform only has 100W
        )
        
        conflicts = self.validator.validate_configuration_interfaces(config)
        
        # Should detect power overload
        power_conflicts = [c for c in conflicts if c.conflict_type == 'power_overload']
        assert len(power_conflicts) > 0
    
    def test_suggest_interface_solutions(self):
        """Test suggesting solutions for interface conflicts."""
        conflict = InterfaceConflict(
            module1_id="module1",
            module2_id="module2",
            interface1_id="interface1",
            interface2_id="interface2",
            conflict_type="voltage_mismatch",
            description="Voltage mismatch: 28V vs 115V",
            severity="warning"
        )
        
        solutions = self.validator.suggest_interface_solutions(conflict)
        
        assert len(solutions) > 0
        assert any(sol['type'] == 'voltage_converter' for sol in solutions)
    
    def test_get_compatibility_matrix(self):
        """Test generating compatibility matrix."""
        modules = [self.compatible_module1, self.compatible_module2, self.incompatible_module]
        
        matrix = self.validator.get_compatibility_matrix(modules)
        
        assert len(matrix) == 3
        
        # Check that each module has entries for all other modules
        for module_id in matrix:
            assert len(matrix[module_id]) == 3  # Including self
        
        # Check self-compatibility
        for module in modules:
            self_compat = matrix[module.module_id][module.module_id]
            assert self_compat.level == CompatibilityLevel.FULLY_COMPATIBLE


class TestDesignEngine:
    """Test cases for DesignEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.library_path = Path(self.temp_dir) / "test_library.json"
        
        config = {
            'library_path': str(self.library_path),
            'auto_save': False,  # Disable auto-save for tests
            'validation_level': 'strict'
        }
        
        self.engine = DesignEngine(config)
        
        # Create test module
        self.test_module = Module(
            name="Test Engine Module",
            module_type=ModuleType.PROPULSION,
            description="Test propulsion module",
            physical_properties=PhysicalProperties(
                mass=1000.0,
                center_of_gravity=(0.0, 0.0, 0.0),
                moments_of_inertia=(500.0, 750.0, 1000.0),
                dimensions=(4.0, 2.0, 2.0)
            )
        )
    
    def test_initialize_engine(self):
        """Test initializing the design engine."""
        assert self.engine.initialize()
        assert self.engine.initialized
        assert self.engine.module_library is not None
        assert self.engine.interface_validator is not None
    
    def test_add_module_to_library(self):
        """Test adding a module to the library through the engine."""
        self.engine.initialize()
        
        result = self.engine.add_module_to_library(self.test_module)
        assert result is True
        
        # Verify module was added
        retrieved = self.engine.get_module_from_library(self.test_module.module_id)
        assert retrieved is not None
        assert retrieved.name == self.test_module.name
    
    def test_search_modules(self):
        """Test searching modules through the engine."""
        self.engine.initialize()
        self.engine.add_module_to_library(self.test_module)
        
        criteria = ModuleSearchCriteria(module_type=ModuleType.PROPULSION)
        results = self.engine.search_modules(criteria)
        
        # Should find our test module
        result_ids = {m.module_id for m in results}
        assert self.test_module.module_id in result_ids
    
    def test_validate_module_compatibility_through_engine(self):
        """Test module compatibility validation through the engine."""
        self.engine.initialize()
        
        # Create two compatible modules
        module1 = Module(name="Module 1", module_type=ModuleType.SENSOR)
        module2 = Module(name="Module 2", module_type=ModuleType.AVIONICS)
        
        result = self.engine.validate_module_compatibility(module1, module2)
        
        assert isinstance(result, CompatibilityResult)
        assert result.confidence_score >= 0.0
    
    def test_create_base_configuration(self):
        """Test creating a base aircraft configuration."""
        self.engine.initialize()
        
        platform = BasePlatform(
            name="Test Platform",
            base_mass=5000.0,
            power_generation_capacity=20000.0
        )
        
        config = self.engine.create_base_configuration(platform, "Test Aircraft")
        
        assert config.name == "Test Aircraft"
        assert config.base_platform == platform
        assert len(config.modules) == 0
    
    def test_add_module_to_configuration(self):
        """Test adding a module to a configuration through the engine."""
        self.engine.initialize()
        
        platform = BasePlatform(
            name="Test Platform",
            base_mass=5000.0,
            power_generation_capacity=20000.0
        )
        
        config = self.engine.create_base_configuration(platform)
        
        result = self.engine.add_module_to_configuration(config, self.test_module)
        assert result is True
        assert len(config.modules) == 1
        assert config.modules[0].module_id == self.test_module.module_id
    
    def test_validate_complete_configuration(self):
        """Test complete configuration validation."""
        self.engine.initialize()
        
        platform = BasePlatform(
            name="Test Platform",
            base_mass=5000.0,
            power_generation_capacity=20000.0,
            attachment_points=[
                MechanicalInterface(
                    interface_id="mount_1",
                    attachment_type="NATO-Standard",
                    load_capacity=(5000.0, 5000.0, 10000.0),
                    moment_capacity=(2500.0, 2500.0, 5000.0),
                    position=(0.0, 0.0, 0.0)
                )
            ]
        )
        
        config = AircraftConfiguration(
            name="Test Configuration",
            base_platform=platform,
            modules=[self.test_module]
        )
        
        results = self.engine.validate_complete_configuration(config)
        
        assert 'valid' in results
        assert 'errors' in results
        assert 'warnings' in results
        assert 'interface_conflicts' in results
        assert 'compatibility_issues' in results
    
    def test_get_library_statistics(self):
        """Test getting library statistics through the engine."""
        self.engine.initialize()
        self.engine.add_module_to_library(self.test_module)
        
        stats = self.engine.get_library_statistics()
        
        assert 'total_modules' in stats
        assert stats['total_modules'] > 0
    
    def test_engine_not_initialized_error(self):
        """Test that operations fail when engine is not initialized."""
        with pytest.raises(ConfigurationError):
            self.engine.add_module_to_library(self.test_module)
        
        with pytest.raises(ConfigurationError):
            self.engine.validate_module_compatibility(self.test_module, self.test_module)


if __name__ == "__main__":
    pytest.main([__file__])