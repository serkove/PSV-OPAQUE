"""Unit tests for core data models."""

import unittest
import tempfile
import json
import yaml
from pathlib import Path
from datetime import datetime

from fighter_jet_sdk.common.data_models import (
    AircraftConfiguration, Module, ModuleInterface, BasePlatform,
    PhysicalProperties, ElectricalInterface, MechanicalInterface,
    PerformanceEnvelope, MissionRequirements
)
from fighter_jet_sdk.common.enums import ModuleType, FlightRegime


class TestPhysicalProperties(unittest.TestCase):
    """Test PhysicalProperties class."""

    def test_creation(self):
        """Test basic creation of PhysicalProperties."""
        props = PhysicalProperties(
            mass=1000.0,
            center_of_gravity=(0.0, 0.0, 0.0),
            moments_of_inertia=(100.0, 200.0, 150.0),
            dimensions=(10.0, 5.0, 3.0)
        )
        
        self.assertEqual(props.mass, 1000.0)
        self.assertEqual(props.center_of_gravity, (0.0, 0.0, 0.0))
        self.assertEqual(props.moments_of_inertia, (100.0, 200.0, 150.0))
        self.assertEqual(props.dimensions, (10.0, 5.0, 3.0))


class TestElectricalInterface(unittest.TestCase):
    """Test ElectricalInterface class."""

    def test_creation(self):
        """Test basic creation of ElectricalInterface."""
        interface = ElectricalInterface(
            interface_id="test_electrical",
            voltage=28.0,
            current_capacity=50.0,
            power_consumption=1400.0,
            data_rate=1000.0,
            protocol="MIL-STD-1553"
        )
        
        self.assertEqual(interface.interface_id, "test_electrical")
        self.assertEqual(interface.voltage, 28.0)
        self.assertEqual(interface.current_capacity, 50.0)
        self.assertEqual(interface.power_consumption, 1400.0)
        self.assertEqual(interface.data_rate, 1000.0)
        self.assertEqual(interface.protocol, "MIL-STD-1553")


class TestMechanicalInterface(unittest.TestCase):
    """Test MechanicalInterface class."""

    def test_creation(self):
        """Test basic creation of MechanicalInterface."""
        interface = MechanicalInterface(
            interface_id="test_mechanical",
            attachment_type="bolt_pattern",
            load_capacity=(10000.0, 5000.0, 15000.0),
            moment_capacity=(1000.0, 2000.0, 1500.0),
            position=(1.0, 0.0, -0.5)
        )
        
        self.assertEqual(interface.interface_id, "test_mechanical")
        self.assertEqual(interface.attachment_type, "bolt_pattern")
        self.assertEqual(interface.load_capacity, (10000.0, 5000.0, 15000.0))
        self.assertEqual(interface.moment_capacity, (1000.0, 2000.0, 1500.0))
        self.assertEqual(interface.position, (1.0, 0.0, -0.5))


class TestModuleInterface(unittest.TestCase):
    """Test ModuleInterface class."""

    def setUp(self):
        """Set up test fixtures."""
        self.electrical = ElectricalInterface(
            interface_id="elec1",
            voltage=28.0,
            current_capacity=10.0,
            power_consumption=280.0
        )
        
        self.mechanical = MechanicalInterface(
            interface_id="mech1",
            attachment_type="standard",
            load_capacity=(1000.0, 1000.0, 1000.0),
            moment_capacity=(100.0, 100.0, 100.0),
            position=(0.0, 0.0, 0.0)
        )

    def test_creation(self):
        """Test basic creation of ModuleInterface."""
        interface = ModuleInterface(
            electrical=self.electrical,
            mechanical=self.mechanical
        )
        
        self.assertIsNotNone(interface.interface_id)
        self.assertEqual(interface.electrical, self.electrical)
        self.assertEqual(interface.mechanical, self.mechanical)

    def test_validation_empty_interface(self):
        """Test validation of empty interface."""
        interface = ModuleInterface()
        errors = interface.validate_interface()
        
        self.assertIn("Interface must have at least one connection type", errors)

    def test_validation_valid_interface(self):
        """Test validation of valid interface."""
        interface = ModuleInterface(electrical=self.electrical)
        errors = interface.validate_interface()
        
        self.assertEqual(len(errors), 0)

    def test_validation_invalid_electrical(self):
        """Test validation of invalid electrical interface."""
        bad_electrical = ElectricalInterface(
            interface_id="bad",
            voltage=-28.0,  # Invalid negative voltage
            current_capacity=0.0,  # Invalid zero capacity
            power_consumption=280.0
        )
        
        interface = ModuleInterface(electrical=bad_electrical)
        errors = interface.validate_interface()
        
        self.assertIn("Electrical interface voltage must be positive", errors)
        self.assertIn("Electrical interface current capacity must be positive", errors)

    def test_compatibility_check(self):
        """Test interface compatibility checking."""
        interface1 = ModuleInterface(electrical=self.electrical)
        
        compatible_electrical = ElectricalInterface(
            interface_id="elec2",
            voltage=28.0,  # Same voltage
            current_capacity=15.0,
            power_consumption=420.0,
            protocol=None  # Same protocol (None)
        )
        interface2 = ModuleInterface(electrical=compatible_electrical)
        
        self.assertTrue(interface1.is_compatible_with(interface2))

    def test_incompatibility_voltage(self):
        """Test interface incompatibility due to voltage difference."""
        interface1 = ModuleInterface(electrical=self.electrical)
        
        incompatible_electrical = ElectricalInterface(
            interface_id="elec3",
            voltage=115.0,  # Different voltage
            current_capacity=15.0,
            power_consumption=420.0
        )
        interface2 = ModuleInterface(electrical=incompatible_electrical)
        
        self.assertFalse(interface1.is_compatible_with(interface2))

    def test_serialization(self):
        """Test interface serialization and deserialization."""
        interface = ModuleInterface(
            electrical=self.electrical,
            mechanical=self.mechanical
        )
        
        # Test to_dict
        data = interface.to_dict()
        self.assertIsInstance(data, dict)
        self.assertIn('interface_id', data)
        self.assertIn('electrical', data)
        self.assertIn('mechanical', data)
        
        # Test from_dict
        restored_interface = ModuleInterface.from_dict(data)
        self.assertEqual(restored_interface.interface_id, interface.interface_id)
        self.assertEqual(restored_interface.electrical.voltage, interface.electrical.voltage)
        self.assertEqual(restored_interface.mechanical.attachment_type, interface.mechanical.attachment_type)


class TestModule(unittest.TestCase):
    """Test Module class."""

    def setUp(self):
        """Set up test fixtures."""
        self.physical_props = PhysicalProperties(
            mass=500.0,
            center_of_gravity=(0.0, 0.0, 0.0),
            moments_of_inertia=(50.0, 100.0, 75.0),
            dimensions=(2.0, 1.0, 0.5)
        )
        
        self.electrical_interface = ElectricalInterface(
            interface_id="sensor_power",
            voltage=28.0,
            current_capacity=20.0,
            power_consumption=560.0
        )

    def test_creation(self):
        """Test basic creation of Module."""
        module = Module(
            name="Test Sensor",
            module_type=ModuleType.SENSOR,
            description="Test sensor module",
            physical_properties=self.physical_props,
            electrical_interfaces=[self.electrical_interface]
        )
        
        self.assertEqual(module.name, "Test Sensor")
        self.assertEqual(module.module_type, ModuleType.SENSOR)
        self.assertEqual(module.description, "Test sensor module")
        self.assertEqual(module.physical_properties, self.physical_props)
        self.assertEqual(len(module.electrical_interfaces), 1)

    def test_validation_valid_module(self):
        """Test validation of valid module."""
        module = Module(
            name="Valid Module",
            physical_properties=self.physical_props,
            electrical_interfaces=[self.electrical_interface]
        )
        
        errors = module.validate_module()
        self.assertEqual(len(errors), 0)

    def test_validation_empty_name(self):
        """Test validation of module with empty name."""
        module = Module(name="")
        errors = module.validate_module()
        
        self.assertIn("Module must have a name", errors)

    def test_validation_invalid_physical_properties(self):
        """Test validation of invalid physical properties."""
        bad_props = PhysicalProperties(
            mass=-100.0,  # Invalid negative mass
            center_of_gravity=(0.0, 0.0, 0.0),
            moments_of_inertia=(-50.0, 100.0, 75.0),  # Invalid negative moment
            dimensions=(2.0, -1.0, 0.5)  # Invalid negative dimension
        )
        
        module = Module(
            name="Bad Module",
            physical_properties=bad_props
        )
        
        errors = module.validate_module()
        self.assertIn("Module mass must be positive", errors)
        self.assertIn("Moment of inertia 0 must be positive", errors)
        self.assertIn("Dimension 1 must be positive", errors)

    def test_power_consumption_calculation(self):
        """Test total power consumption calculation."""
        interface1 = ElectricalInterface(
            interface_id="power1",
            voltage=28.0,
            current_capacity=10.0,
            power_consumption=280.0
        )
        
        interface2 = ElectricalInterface(
            interface_id="power2",
            voltage=28.0,
            current_capacity=5.0,
            power_consumption=140.0
        )
        
        module = Module(
            name="Power Test",
            electrical_interfaces=[interface1, interface2]
        )
        
        total_power = module.calculate_total_power_consumption()
        self.assertEqual(total_power, 420.0)

    def test_compatibility_check(self):
        """Test module compatibility checking."""
        module1 = Module(
            name="Module 1",
            module_type=ModuleType.SENSOR
        )
        
        module2 = Module(
            name="Module 2",
            module_type=ModuleType.PAYLOAD
        )
        
        # Should be compatible by default
        self.assertTrue(module1.is_compatible_with(module2))

    def test_incompatibility_check(self):
        """Test module incompatibility checking."""
        module1 = Module(
            name="Module 1",
            module_type=ModuleType.SENSOR,
            compatibility_requirements=["incompatible_with:PAYLOAD"]
        )
        
        module2 = Module(
            name="Module 2",
            module_type=ModuleType.PAYLOAD
        )
        
        # Should be incompatible due to requirement
        self.assertFalse(module1.is_compatible_with(module2))

    def test_serialization(self):
        """Test module serialization and deserialization."""
        module = Module(
            name="Test Module",
            module_type=ModuleType.SENSOR,
            physical_properties=self.physical_props,
            electrical_interfaces=[self.electrical_interface]
        )
        
        # Test to_dict
        data = module.to_dict()
        self.assertIsInstance(data, dict)
        self.assertEqual(data['name'], "Test Module")
        self.assertEqual(data['module_type'], "SENSOR")
        
        # Test from_dict
        restored_module = Module.from_dict(data)
        self.assertEqual(restored_module.name, module.name)
        self.assertEqual(restored_module.module_type, module.module_type)
        self.assertEqual(restored_module.physical_properties.mass, module.physical_properties.mass)


class TestAircraftConfiguration(unittest.TestCase):
    """Test AircraftConfiguration class."""

    def setUp(self):
        """Set up test fixtures."""
        self.base_platform = BasePlatform(
            name="Test Platform",
            base_mass=5000.0,
            power_generation_capacity=10000.0,
            fuel_capacity=2000.0
        )
        
        # Add some attachment points
        for i in range(3):
            attachment_point = MechanicalInterface(
                interface_id=f"attach_{i}",
                attachment_type="standard",
                load_capacity=(5000.0, 5000.0, 10000.0),
                moment_capacity=(1000.0, 1000.0, 2000.0),
                position=(i * 2.0, 0.0, 0.0)
            )
            self.base_platform.attachment_points.append(attachment_point)
        
        self.module1 = Module(
            name="Sensor Module",
            module_type=ModuleType.SENSOR,
            electrical_interfaces=[
                ElectricalInterface(
                    interface_id="sensor_power",
                    voltage=28.0,
                    current_capacity=10.0,
                    power_consumption=280.0
                )
            ]
        )
        
        self.module2 = Module(
            name="Payload Module",
            module_type=ModuleType.PAYLOAD,
            electrical_interfaces=[
                ElectricalInterface(
                    interface_id="payload_power",
                    voltage=28.0,
                    current_capacity=20.0,
                    power_consumption=560.0
                )
            ]
        )

    def test_creation(self):
        """Test basic creation of AircraftConfiguration."""
        config = AircraftConfiguration(
            name="Test Aircraft",
            description="Test configuration",
            base_platform=self.base_platform,
            modules=[self.module1, self.module2]
        )
        
        self.assertEqual(config.name, "Test Aircraft")
        self.assertEqual(config.description, "Test configuration")
        self.assertEqual(config.base_platform, self.base_platform)
        self.assertEqual(len(config.modules), 2)

    def test_validation_valid_configuration(self):
        """Test validation of valid configuration."""
        config = AircraftConfiguration(
            name="Valid Aircraft",
            base_platform=self.base_platform,
            modules=[self.module1, self.module2]
        )
        
        errors = config.validate_configuration()
        self.assertEqual(len(errors), 0)

    def test_validation_empty_name(self):
        """Test validation of configuration with empty name."""
        config = AircraftConfiguration(
            name="",
            base_platform=self.base_platform
        )
        
        errors = config.validate_configuration()
        self.assertIn("Aircraft configuration must have a name", errors)

    def test_validation_no_platform(self):
        """Test validation of configuration without base platform."""
        config = AircraftConfiguration(
            name="No Platform Aircraft"
        )
        
        errors = config.validate_configuration()
        self.assertIn("Aircraft configuration must have a base platform", errors)

    def test_validation_no_modules(self):
        """Test validation of configuration without modules."""
        config = AircraftConfiguration(
            name="No Modules Aircraft",
            base_platform=self.base_platform
        )
        
        errors = config.validate_configuration()
        self.assertIn("Aircraft configuration must have at least one module", errors)

    def test_validation_power_exceeded(self):
        """Test validation when power requirements exceed capacity."""
        # Create high-power module
        high_power_module = Module(
            name="High Power Module",
            electrical_interfaces=[
                ElectricalInterface(
                    interface_id="high_power",
                    voltage=28.0,
                    current_capacity=500.0,
                    power_consumption=15000.0  # Exceeds platform capacity
                )
            ]
        )
        
        config = AircraftConfiguration(
            name="Over Power Aircraft",
            base_platform=self.base_platform,
            modules=[high_power_module]
        )
        
        errors = config.validate_configuration()
        power_error = next((e for e in errors if "Power requirements exceed" in e), None)
        self.assertIsNotNone(power_error)

    def test_add_module(self):
        """Test adding module to configuration."""
        config = AircraftConfiguration(
            name="Test Aircraft",
            base_platform=self.base_platform
        )
        
        success = config.add_module(self.module1)
        self.assertTrue(success)
        self.assertEqual(len(config.modules), 1)
        self.assertEqual(config.modules[0], self.module1)

    def test_add_duplicate_module(self):
        """Test adding duplicate module to configuration."""
        config = AircraftConfiguration(
            name="Test Aircraft",
            base_platform=self.base_platform,
            modules=[self.module1]
        )
        
        # Try to add same module again
        success = config.add_module(self.module1)
        self.assertFalse(success)
        self.assertEqual(len(config.modules), 1)

    def test_remove_module(self):
        """Test removing module from configuration."""
        config = AircraftConfiguration(
            name="Test Aircraft",
            base_platform=self.base_platform,
            modules=[self.module1, self.module2]
        )
        
        success = config.remove_module(self.module1.module_id)
        self.assertTrue(success)
        self.assertEqual(len(config.modules), 1)
        self.assertEqual(config.modules[0], self.module2)

    def test_remove_nonexistent_module(self):
        """Test removing nonexistent module from configuration."""
        config = AircraftConfiguration(
            name="Test Aircraft",
            base_platform=self.base_platform,
            modules=[self.module1]
        )
        
        success = config.remove_module("nonexistent_id")
        self.assertFalse(success)
        self.assertEqual(len(config.modules), 1)

    def test_serialization_json(self):
        """Test configuration serialization to JSON."""
        config = AircraftConfiguration(
            name="Test Aircraft",
            base_platform=self.base_platform,
            modules=[self.module1]
        )
        
        # Test to_dict
        data = config.to_dict()
        self.assertIsInstance(data, dict)
        self.assertEqual(data['name'], "Test Aircraft")
        
        # Test from_dict
        restored_config = AircraftConfiguration.from_dict(data)
        self.assertEqual(restored_config.name, config.name)
        self.assertEqual(len(restored_config.modules), len(config.modules))

    def test_file_operations_json(self):
        """Test saving and loading configuration to/from JSON file."""
        config = AircraftConfiguration(
            name="File Test Aircraft",
            base_platform=self.base_platform,
            modules=[self.module1]
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save to file
            config.save_to_file(temp_path)
            
            # Load from file
            loaded_config = AircraftConfiguration.load_from_file(temp_path)
            
            self.assertEqual(loaded_config.name, config.name)
            self.assertEqual(len(loaded_config.modules), len(config.modules))
            self.assertEqual(loaded_config.modules[0].name, config.modules[0].name)
            
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_file_operations_yaml(self):
        """Test saving and loading configuration to/from YAML file."""
        config = AircraftConfiguration(
            name="YAML Test Aircraft",
            base_platform=self.base_platform,
            modules=[self.module1]
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save to file
            config.save_to_file(temp_path)
            
            # Load from file
            loaded_config = AircraftConfiguration.load_from_file(temp_path)
            
            self.assertEqual(loaded_config.name, config.name)
            self.assertEqual(len(loaded_config.modules), len(config.modules))
            
        finally:
            Path(temp_path).unlink(missing_ok=True)


if __name__ == '__main__':
    unittest.main()