"""Unit tests for sensor system functionality."""

import unittest
from fighter_jet_sdk.common.data_models import (
    SensorSystem, DetectionCapabilities, PowerRequirements,
    AtmosphericConstraints, IntegrationRequirements
)
from fighter_jet_sdk.common.enums import SensorType


class TestSensorSystem(unittest.TestCase):
    """Test SensorSystem class."""

    def setUp(self):
        """Set up test fixtures."""
        self.detection_caps = DetectionCapabilities(
            detection_range={"fighter_aircraft": 150.0, "bomber": 200.0, "missile": 50.0},
            resolution={"range": 1.0, "azimuth": 0.1, "elevation": 0.1},
            accuracy={"range": 0.5, "azimuth": 0.05, "elevation": 0.05},
            update_rate=10.0,  # Hz
            field_of_view=(120.0, 60.0)  # azimuth, elevation degrees
        )
        
        self.power_reqs = PowerRequirements(
            peak_power=50000.0,  # 50 kW
            average_power=5000.0,  # 5 kW
            startup_power=75000.0,  # 75 kW
            voltage_requirement=28.0,  # V
            power_quality_requirements={"ripple": 0.01, "regulation": 0.02}
        )
        
        self.atmo_constraints = AtmosphericConstraints(
            altitude_range=(0.0, 20000.0),  # 0 to 20 km
            temperature_range=(223.0, 323.0),  # -50°C to 50°C
            humidity_limits=(0.0, 95.0),  # 0 to 95%
            pressure_range=(20000.0, 101325.0),  # 20 kPa to 1 atm
            weather_limitations=["heavy_rain", "snow", "fog"]
        )
        
        self.integration_reqs = IntegrationRequirements(
            cooling_requirements={"primary": 3000.0, "secondary": 1000.0},
            vibration_isolation=True,
            electromagnetic_shielding=True,
            data_interfaces=["MIL-STD-1553", "Ethernet"],
            physical_constraints={"weight": 500.0, "volume": 0.5}
        )

    def test_creation(self):
        """Test basic creation of SensorSystem."""
        sensor = SensorSystem(
            name="Test AESA Radar",
            sensor_type=SensorType.AESA_RADAR,
            detection_capabilities=self.detection_caps,
            power_requirements=self.power_reqs,
            atmospheric_limitations=self.atmo_constraints,
            integration_requirements=self.integration_reqs
        )
        
        self.assertEqual(sensor.name, "Test AESA Radar")
        self.assertEqual(sensor.sensor_type, SensorType.AESA_RADAR)
        self.assertEqual(sensor.detection_capabilities, self.detection_caps)
        self.assertEqual(sensor.power_requirements, self.power_reqs)

    def test_validation_valid_sensor(self):
        """Test validation of valid sensor system."""
        sensor = SensorSystem(
            name="Valid Sensor",
            sensor_type=SensorType.AESA_RADAR,
            detection_capabilities=self.detection_caps,
            power_requirements=self.power_reqs,
            atmospheric_limitations=self.atmo_constraints,
            integration_requirements=self.integration_reqs
        )
        
        errors = sensor.validate_sensor()
        self.assertEqual(len(errors), 0)

    def test_validation_empty_name(self):
        """Test validation of sensor with empty name."""
        sensor = SensorSystem(name="")
        errors = sensor.validate_sensor()
        
        self.assertIn("Sensor system must have a name", errors)

    def test_validation_invalid_detection_capabilities(self):
        """Test validation of invalid detection capabilities."""
        bad_detection_caps = DetectionCapabilities(
            detection_range={"fighter": -150.0},  # Invalid negative range
            resolution={"range": -1.0},  # Invalid negative resolution
            accuracy={"range": 0.0},  # Invalid zero accuracy
            update_rate=-10.0,  # Invalid negative update rate
            field_of_view=(400.0, 200.0)  # Invalid FOV (>360°, >180°)
        )
        
        sensor = SensorSystem(
            name="Bad Detection Sensor",
            detection_capabilities=bad_detection_caps
        )
        
        errors = sensor.validate_sensor()
        self.assertIn("Detection range for fighter must be positive", errors)
        self.assertIn("Resolution for range must be positive", errors)
        self.assertIn("Accuracy for range must be positive", errors)
        self.assertIn("Update rate must be positive", errors)
        self.assertIn("Azimuth field of view must be between 0 and 360 degrees", errors)
        self.assertIn("Elevation field of view must be between 0 and 180 degrees", errors)

    def test_validation_invalid_power_requirements(self):
        """Test validation of invalid power requirements."""
        bad_power_reqs = PowerRequirements(
            peak_power=-50000.0,  # Invalid negative
            average_power=60000.0,  # Greater than peak power
            startup_power=200000.0,  # Unreasonably high (>2x peak)
            voltage_requirement=-28.0,  # Invalid negative
            power_quality_requirements={}
        )
        
        sensor = SensorSystem(
            name="Bad Power Sensor",
            power_requirements=bad_power_reqs
        )
        
        errors = sensor.validate_sensor()
        self.assertIn("Peak power cannot be negative", errors)
        self.assertIn("Average power cannot exceed peak power", errors)
        self.assertIn("Startup power seems unreasonably high (>2x peak power)", errors)
        self.assertIn("Voltage requirement must be positive", errors)

    def test_validation_invalid_atmospheric_constraints(self):
        """Test validation of invalid atmospheric constraints."""
        bad_atmo_constraints = AtmosphericConstraints(
            altitude_range=(20000.0, 10000.0),  # Invalid range (min > max)
            temperature_range=(323.0, 223.0),  # Invalid range (min > max)
            humidity_limits=(50.0, 120.0),  # Invalid humidity (>100%)
            pressure_range=(101325.0, 20000.0),  # Invalid range (min > max)
            weather_limitations=[]
        )
        
        sensor = SensorSystem(
            name="Bad Atmospheric Sensor",
            atmospheric_limitations=bad_atmo_constraints
        )
        
        errors = sensor.validate_sensor()
        self.assertIn("Invalid altitude range: min must be less than max", errors)
        self.assertIn("Invalid temperature range: min must be less than max", errors)
        self.assertIn("Humidity limits must be between 0 and 100 percent", errors)
        self.assertIn("Invalid pressure range: min must be less than max", errors)

    def test_validation_aesa_radar_requirements(self):
        """Test validation of AESA radar specific requirements."""
        # AESA radar without power requirements
        sensor = SensorSystem(
            name="AESA Without Power",
            sensor_type=SensorType.AESA_RADAR
        )
        
        errors = sensor.validate_sensor()
        self.assertIn("AESA radar must have power requirements defined", errors)
        
        # AESA radar with insufficient power
        low_power_reqs = PowerRequirements(
            peak_power=500.0,  # Too low for AESA
            average_power=100.0,
            startup_power=750.0,
            voltage_requirement=28.0
        )
        
        sensor_low_power = SensorSystem(
            name="Low Power AESA",
            sensor_type=SensorType.AESA_RADAR,
            power_requirements=low_power_reqs
        )
        
        errors = sensor_low_power.validate_sensor()
        self.assertIn("AESA radar peak power should be at least 1kW", errors)

    def test_validation_laser_sensor_requirements(self):
        """Test validation of laser sensor specific requirements."""
        sensor = SensorSystem(
            name="Laser Without Requirements",
            sensor_type=SensorType.LASER_BASED
        )
        
        errors = sensor.validate_sensor()
        self.assertIn("Laser-based sensors must have power requirements defined", errors)
        self.assertIn("Laser-based sensors must have atmospheric limitations defined", errors)

    def test_validation_plasma_sensor_requirements(self):
        """Test validation of plasma sensor specific requirements."""
        # Plasma sensor with insufficient power
        low_power_reqs = PowerRequirements(
            peak_power=5000.0,  # Too low for plasma
            average_power=1000.0,
            startup_power=7500.0,
            voltage_requirement=28.0
        )
        
        sensor = SensorSystem(
            name="Low Power Plasma",
            sensor_type=SensorType.PLASMA_BASED,
            power_requirements=low_power_reqs
        )
        
        errors = sensor.validate_sensor()
        self.assertIn("Plasma-based sensors require at least 10kW peak power", errors)

    def test_power_consumption_calculation(self):
        """Test power consumption calculation."""
        sensor = SensorSystem(
            name="Power Test Sensor",
            power_requirements=self.power_reqs
        )
        
        # Test continuous operation (duty cycle = 1.0)
        continuous_power = sensor.calculate_power_consumption(1.0)
        self.assertEqual(continuous_power, self.power_reqs.average_power)
        
        # Test 50% duty cycle
        half_duty_power = sensor.calculate_power_consumption(0.5)
        expected = (self.power_reqs.average_power + 
                   (self.power_reqs.peak_power - self.power_reqs.average_power) * 0.5)
        self.assertEqual(half_duty_power, expected)
        
        # Test invalid duty cycle
        with self.assertRaises(ValueError):
            sensor.calculate_power_consumption(1.5)

    def test_detection_probability_calculation(self):
        """Test detection probability calculation."""
        sensor = SensorSystem(
            name="Detection Test Sensor",
            detection_capabilities=self.detection_caps
        )
        
        # Test detection at close range with good RCS
        prob_close = sensor.calculate_detection_probability(
            target_rcs=10.0,  # 10 m² RCS
            range_km=50.0,    # 50 km range
            atmospheric_attenuation=1.0
        )
        
        self.assertGreater(prob_close, 0.0)
        self.assertLessEqual(prob_close, 1.0)
        
        # Test detection at maximum range
        prob_max = sensor.calculate_detection_probability(
            target_rcs=1.0,   # 1 m² RCS
            range_km=150.0,   # At max range
            atmospheric_attenuation=1.0
        )
        
        self.assertGreaterEqual(prob_max, 0.0)
        self.assertLess(prob_max, prob_close)  # Should be lower than close range
        
        # Test beyond maximum range
        prob_beyond = sensor.calculate_detection_probability(
            target_rcs=1.0,
            range_km=300.0,   # Beyond max range
            atmospheric_attenuation=1.0
        )
        
        self.assertEqual(prob_beyond, 0.0)

    def test_atmospheric_compatibility_check(self):
        """Test atmospheric compatibility checking."""
        sensor = SensorSystem(
            name="Atmospheric Test Sensor",
            atmospheric_limitations=self.atmo_constraints
        )
        
        # Test compatible conditions
        compatible = sensor.check_atmospheric_compatibility(
            altitude=10000.0,    # Within range
            temperature=273.0,   # Within range
            humidity=50.0,       # Within range
            pressure=50000.0     # Within range
        )
        
        self.assertTrue(compatible)
        
        # Test incompatible altitude
        incompatible_alt = sensor.check_atmospheric_compatibility(
            altitude=25000.0,    # Above max altitude
            temperature=273.0,
            humidity=50.0,
            pressure=50000.0
        )
        
        self.assertFalse(incompatible_alt)
        
        # Test incompatible temperature
        incompatible_temp = sensor.check_atmospheric_compatibility(
            altitude=10000.0,
            temperature=350.0,   # Above max temperature
            humidity=50.0,
            pressure=50000.0
        )
        
        self.assertFalse(incompatible_temp)

    def test_cooling_requirements_calculation(self):
        """Test cooling requirements calculation."""
        sensor = SensorSystem(
            name="Cooling Test Sensor",
            sensor_type=SensorType.AESA_RADAR,
            power_requirements=self.power_reqs,
            integration_requirements=self.integration_reqs
        )
        
        cooling_reqs = sensor.calculate_cooling_requirements(
            ambient_temperature=298.0,  # 25°C
            duty_cycle=0.8
        )
        
        self.assertIsInstance(cooling_reqs, dict)
        self.assertIn("primary_cooling", cooling_reqs)
        self.assertIn("secondary_cooling", cooling_reqs)  # AESA should have secondary
        self.assertIn("electronics_cooling", cooling_reqs)
        
        # All cooling values should be positive
        for cooling_type, cooling_power in cooling_reqs.items():
            self.assertGreater(cooling_power, 0)

    def test_detection_range_estimation(self):
        """Test detection range estimation."""
        sensor = SensorSystem(
            name="Range Test Sensor",
            detection_capabilities=self.detection_caps
        )
        
        # Test range for large RCS target
        range_large = sensor.estimate_detection_range(
            target_rcs=100.0,  # Large RCS
            detection_probability=0.9
        )
        
        # Test range for small RCS target
        range_small = sensor.estimate_detection_range(
            target_rcs=0.1,    # Small RCS
            detection_probability=0.9
        )
        
        self.assertGreater(range_large, range_small)  # Larger RCS should have longer range
        self.assertGreater(range_large, 0)
        self.assertGreater(range_small, 0)
        
        # Test invalid inputs
        range_invalid = sensor.estimate_detection_range(
            target_rcs=-1.0,   # Invalid negative RCS
            detection_probability=0.9
        )
        
        self.assertEqual(range_invalid, 0.0)

    def test_serialization(self):
        """Test sensor system serialization and deserialization."""
        sensor = SensorSystem(
            name="Serialization Test",
            sensor_type=SensorType.AESA_RADAR,
            detection_capabilities=self.detection_caps,
            power_requirements=self.power_reqs
        )
        
        # Test to_dict
        data = sensor.to_dict()
        self.assertIsInstance(data, dict)
        self.assertEqual(data['name'], "Serialization Test")
        self.assertEqual(data['sensor_type'], "AESA_RADAR")
        
        # Test from_dict
        restored_sensor = SensorSystem.from_dict(data)
        self.assertEqual(restored_sensor.name, sensor.name)
        self.assertEqual(restored_sensor.sensor_type, sensor.sensor_type)
        self.assertEqual(restored_sensor.detection_capabilities.update_rate, 
                        sensor.detection_capabilities.update_rate)

    def test_sensor_without_capabilities(self):
        """Test sensor operations without detection capabilities."""
        sensor = SensorSystem(
            name="Minimal Sensor",
            sensor_type=SensorType.PASSIVE_RF
        )
        
        # Should return 0 for detection probability without capabilities
        prob = sensor.calculate_detection_probability(1.0, 100.0)
        self.assertEqual(prob, 0.0)
        
        # Should return 0 for range estimation without capabilities
        range_est = sensor.estimate_detection_range(1.0, 0.9)
        self.assertEqual(range_est, 0.0)
        
        # Should return True for atmospheric compatibility without constraints
        compatible = sensor.check_atmospheric_compatibility(10000.0, 273.0, 50.0, 50000.0)
        self.assertTrue(compatible)


if __name__ == '__main__':
    unittest.main()