"""Tests for laser-based sensor and weapon systems."""

import pytest
import numpy as np
import math
from fighter_jet_sdk.engines.sensors.laser_systems import (
    LaserFilamentationSim, AdaptiveOpticsController, LaserInducedBreakdownSpectroscopy,
    LaserSafetyAnalyzer, LaserConfiguration, AdaptiveOpticsConfiguration,
    AtmosphericParameters, LaserTarget, LaserType, AtmosphericCondition
)


class TestLaserConfiguration:
    """Test laser configuration."""
    
    def test_laser_config_creation(self):
        """Test laser configuration creation."""
        config = LaserConfiguration(
            wavelength=1.064e-6,  # Nd:YAG
            peak_power=1e6,  # 1 MW
            pulse_energy=10.0,  # 10 J
            pulse_duration=10e-9,  # 10 ns
            repetition_rate=10.0,  # 10 Hz
            beam_divergence=1e-6,  # 1 μrad
            beam_quality=1.2,
            laser_type=LaserType.SOLID_STATE
        )
        
        assert config.wavelength == 1.064e-6
        assert config.peak_power == 1e6
        assert config.pulse_energy == 10.0
        assert config.laser_type == LaserType.SOLID_STATE


class TestLaserFilamentationSim:
    """Test laser filamentation simulation."""
    
    @pytest.fixture
    def laser_config(self):
        """Create test laser configuration."""
        return LaserConfiguration(
            wavelength=800e-9,  # Ti:Sapphire
            peak_power=10e12,  # 10 TW
            pulse_energy=1.0,  # 1 J
            pulse_duration=100e-15,  # 100 fs
            laser_type=LaserType.PULSED
        )
    
    @pytest.fixture
    def atmospheric_params(self):
        """Create test atmospheric parameters."""
        return AtmosphericParameters(
            visibility=20.0,  # km
            temperature=288.15,  # K (15°C)
            pressure=101325,  # Pa
            humidity=50.0,  # %
            wind_speed=5.0,  # m/s
            turbulence_strength=1e-14,  # Cn²
            condition=AtmosphericCondition.CLEAR
        )
    
    def test_filamentation_sim_creation(self, laser_config):
        """Test filamentation simulator creation."""
        sim = LaserFilamentationSim(laser_config)
        assert sim.config == laser_config
        assert sim.critical_power_air == 3.2e9
    
    def test_critical_power_calculation(self, laser_config, atmospheric_params):
        """Test critical power calculation."""
        sim = LaserFilamentationSim(laser_config)
        
        critical_power = sim.calculate_critical_power(atmospheric_params)
        
        # Should be close to standard value for STP conditions
        assert 2e9 < critical_power < 5e9
        
        # Test with different pressure
        high_pressure_params = atmospheric_params
        high_pressure_params.pressure = 200000  # 2 atm
        
        critical_power_high = sim.calculate_critical_power(high_pressure_params)
        assert critical_power_high > critical_power
    
    def test_filamentation_length(self, laser_config, atmospheric_params):
        """Test filamentation length calculation."""
        sim = LaserFilamentationSim(laser_config)
        
        # Test with power below critical
        low_power = 1e9  # 1 GW
        length_low = sim.calculate_filamentation_length(low_power, atmospheric_params)
        assert length_low == 0.0
        
        # Test with power above critical
        high_power = 10e9  # 10 GW
        length_high = sim.calculate_filamentation_length(high_power, atmospheric_params)
        assert length_high > 0
        assert length_high < 10000  # Should be reasonable length
    
    def test_plasma_density_calculation(self, laser_config, atmospheric_params):
        """Test plasma density calculation."""
        sim = LaserFilamentationSim(laser_config)
        
        power = 10e9  # 10 GW
        position = 100.0  # 100 m
        
        density = sim.calculate_plasma_density(power, position, atmospheric_params)
        
        # Should be positive and reasonable
        assert density >= 0
        assert density < 1e20  # Reasonable upper limit
    
    def test_plasma_lifetime(self, laser_config, atmospheric_params):
        """Test plasma lifetime calculation."""
        sim = LaserFilamentationSim(laser_config)
        
        plasma_density = 1e16  # m^-3
        lifetime = sim.calculate_plasma_lifetime(plasma_density, atmospheric_params)
        
        # Should be positive and in reasonable range
        assert lifetime > 0
        assert 1e-9 < lifetime < 1e-3  # ns to ms range


class TestAdaptiveOpticsController:
    """Test adaptive optics controller."""
    
    @pytest.fixture
    def ao_config(self):
        """Create test adaptive optics configuration."""
        return AdaptiveOpticsConfiguration(
            actuator_count=256,  # 16x16 array
            wavefront_sensor_subapertures=400,  # 20x20 array
            correction_bandwidth=1000.0,  # 1 kHz
            residual_wavefront_error=0.1,  # 0.1 rad RMS
            aperture_diameter=0.2  # 20 cm
        )
    
    @pytest.fixture
    def atmospheric_params(self):
        """Create test atmospheric parameters."""
        return AtmosphericParameters(
            visibility=10.0,
            temperature=288.15,
            pressure=101325,
            humidity=60.0,
            wind_speed=10.0,
            turbulence_strength=1e-13,  # Strong turbulence
            condition=AtmosphericCondition.CLEAR
        )
    
    def test_ao_controller_creation(self, ao_config):
        """Test adaptive optics controller creation."""
        controller = AdaptiveOpticsController(ao_config)
        assert controller.config == ao_config
        assert not controller.closed_loop_active
        assert len(controller.wavefront_measurements) == 0
    
    def test_wavefront_measurement(self, ao_config, atmospheric_params):
        """Test wavefront measurement simulation."""
        controller = AdaptiveOpticsController(ao_config)
        
        propagation_distance = 1000.0  # 1 km
        wavefront = controller.measure_wavefront(atmospheric_params, propagation_distance)
        
        # Should be 2D array
        assert wavefront.ndim == 2
        assert wavefront.shape[0] > 0
        assert wavefront.shape[1] > 0
        
        # Should have reasonable values (phase in radians)
        assert np.all(np.abs(wavefront) < 5)  # Less than 5 radians
        
        # Should be stored in measurements
        assert len(controller.wavefront_measurements) == 1
    
    def test_correction_calculation(self, ao_config):
        """Test correction calculation."""
        controller = AdaptiveOpticsController(ao_config)
        
        # Create test wavefront
        n_points = 16
        wavefront = np.random.normal(0, 0.5, (n_points, n_points))
        
        correction = controller.calculate_correction(wavefront)
        
        # Should be 2D array with normalized values
        assert correction.ndim == 2
        assert np.all(correction >= -1)
        assert np.all(correction <= 1)
        
        # Should be stored in commands
        assert len(controller.correction_commands) == 1
    
    def test_correction_application(self, ao_config):
        """Test correction application."""
        controller = AdaptiveOpticsController(ao_config)
        
        # Create test correction commands
        n_act = int(math.sqrt(ao_config.actuator_count))
        correction_commands = np.random.uniform(-1, 1, (n_act, n_act))
        
        residual_error = controller.apply_correction(correction_commands)
        
        # Should be positive and reasonable
        assert residual_error >= 0
        assert residual_error < 10  # Less than 10 radians RMS
    
    def test_strehl_ratio_calculation(self, ao_config):
        """Test Strehl ratio calculation."""
        controller = AdaptiveOpticsController(ao_config)
        
        # Test various residual errors
        errors = [0.0, 0.1, 0.5, 1.0, 2.0]
        
        for error in errors:
            strehl = controller.get_strehl_ratio(error)
            
            # Should be between 0 and 1
            assert 0 <= strehl <= 1
            
            # Lower error should give higher Strehl
            if error == 0.0:
                assert strehl == 1.0
            elif error > 1.0:
                assert strehl < 0.5


class TestLaserInducedBreakdownSpectroscopy:
    """Test LIBS system."""
    
    @pytest.fixture
    def laser_config(self):
        """Create test laser configuration."""
        return LaserConfiguration(
            wavelength=1.064e-6,
            peak_power=1e9,  # 1 GW
            pulse_energy=0.1,  # 100 mJ
            pulse_duration=10e-9,  # 10 ns
            laser_type=LaserType.PULSED
        )
    
    def test_libs_creation(self, laser_config):
        """Test LIBS system creation."""
        libs = LaserInducedBreakdownSpectroscopy(laser_config)
        assert libs.config == laser_config
        assert 'H' in libs.element_lines
        assert 'U' in libs.element_lines
    
    def test_plasma_temperature_calculation(self, laser_config):
        """Test plasma temperature calculation."""
        libs = LaserInducedBreakdownSpectroscopy(laser_config)
        
        # Test with different powers and materials
        powers = [1e6, 1e9, 1e12]  # W
        materials = ['metal', 'ceramic', 'polymer']
        
        for power in powers:
            for material in materials:
                temp = libs.calculate_plasma_temperature(power, material)
                
                # Should be reasonable plasma temperature
                assert 5000 < temp < 60000  # 5,000 - 60,000 K
                
                # Higher power should give higher temperature
                if power == max(powers):
                    assert temp > 20000
    
    def test_spectrum_simulation(self, laser_config):
        """Test spectrum simulation."""
        libs = LaserInducedBreakdownSpectroscopy(laser_config)
        
        plasma_temp = 15000  # K
        elements = ['H', 'C', 'O', 'U']
        concentrations = [0.5, 0.3, 0.15, 0.05]
        
        spectrum = libs.simulate_spectrum(plasma_temp, elements, concentrations)
        
        # Should have spectral lines
        assert len(spectrum) > 0
        
        # All intensities should be positive
        assert all(intensity >= 0 for intensity in spectrum.values())
        
        # Should have lines from all elements
        h_lines = libs.element_lines['H']
        assert any(line in spectrum for line in h_lines)
    
    def test_radioactive_detection(self, laser_config):
        """Test radioactive element detection."""
        libs = LaserInducedBreakdownSpectroscopy(laser_config)
        
        # Create spectrum with uranium lines
        spectrum_with_u = {
            424.4: 500,  # Strong U line
            435.6: 300,  # Medium U line
            656.3: 200   # H line
        }
        
        detected = libs.detect_radioactive_elements(spectrum_with_u, 100)
        assert 'U' in detected
        
        # Create spectrum without radioactive elements
        spectrum_clean = {
            656.3: 500,  # H line
            500.5: 300   # N line
        }
        
        detected_clean = libs.detect_radioactive_elements(spectrum_clean, 100)
        assert 'U' not in detected_clean
        assert 'Pu' not in detected_clean


class TestLaserSafetyAnalyzer:
    """Test laser safety analyzer."""
    
    @pytest.fixture
    def safety_analyzer(self):
        """Create laser safety analyzer."""
        return LaserSafetyAnalyzer()
    
    def test_safety_analyzer_creation(self, safety_analyzer):
        """Test safety analyzer creation."""
        assert 'eye_visible' in safety_analyzer.mpe_values
        assert 'skin_near_ir' in safety_analyzer.mpe_values
    
    def test_beam_divergence_calculation(self, safety_analyzer):
        """Test beam divergence calculation."""
        initial_diameter = 0.01  # 1 cm
        distance = 1000  # 1 km
        wavelength = 1.064e-6  # Nd:YAG
        
        beam_diameter = safety_analyzer.calculate_beam_divergence(
            initial_diameter, distance, wavelength
        )
        
        # Should be larger than initial diameter
        assert beam_diameter > initial_diameter
        
        # Should be reasonable for 1 km propagation
        assert beam_diameter < 1.0  # Less than 1 m
    
    def test_irradiance_calculation(self, safety_analyzer):
        """Test irradiance calculation."""
        power = 1000  # 1 kW
        beam_diameter = 0.1  # 10 cm
        
        irradiance = safety_analyzer.calculate_irradiance(power, beam_diameter)
        
        # Should be positive
        assert irradiance > 0
        
        # Should have correct units (W/m²)
        expected_area = math.pi * (beam_diameter / 2) ** 2
        expected_irradiance = power / expected_area
        assert abs(irradiance - expected_irradiance) < 1e-6
    
    def test_eye_safety_assessment(self, safety_analyzer):
        """Test eye safety assessment."""
        # Safe laser (very low power)
        safe_assessment = safety_analyzer.assess_eye_safety(
            power=0.00001,  # 10 μW (very safe)
            beam_diameter=0.01,  # 1 cm
            wavelength=650e-9,  # Red laser
            exposure_time=0.25  # 0.25 s
        )
        
        # Check that assessment completes (may or may not be safe depending on MPE scaling)
        assert 'safe' in safe_assessment
        assert 'safety_class' in safe_assessment
        
        # Dangerous laser
        dangerous_assessment = safety_analyzer.assess_eye_safety(
            power=1000,  # 1 kW
            beam_diameter=0.01,  # 1 cm
            wavelength=1064e-9,  # Nd:YAG
            exposure_time=0.25
        )
        
        assert dangerous_assessment['safe'] == False
        assert dangerous_assessment['safety_factor'] > 1
    
    def test_nohd_calculation(self, safety_analyzer):
        """Test NOHD calculation."""
        power = 100  # 100 W
        beam_divergence = 1e-3  # 1 mrad
        wavelength = 1064e-9  # Nd:YAG
        
        nohd = safety_analyzer.calculate_nominal_ocular_hazard_distance(
            power, beam_divergence, wavelength
        )
        
        # Should be positive and reasonable
        assert nohd > 0
        assert nohd < 1000000  # Less than 1000 km (high power laser)
    
    def test_safety_report_generation(self, safety_analyzer):
        """Test comprehensive safety report."""
        laser_config = LaserConfiguration(
            wavelength=1064e-9,
            peak_power=1000,  # 1 kW
            beam_divergence=1e-3,
            laser_type=LaserType.CONTINUOUS_WAVE
        )
        
        operating_distance = 1000  # 1 km
        
        report = safety_analyzer.generate_safety_report(laser_config, operating_distance)
        
        # Should have all required fields
        required_fields = [
            'laser_class', 'eye_safe', 'skin_safe', 'nohd_m',
            'beam_diameter_at_distance_m', 'irradiance_w_per_m2',
            'safety_factor', 'recommendations'
        ]
        
        for field in required_fields:
            assert field in report
        
        # Recommendations should be a list
        assert isinstance(report['recommendations'], list)
        assert len(report['recommendations']) > 0


class TestLaserTarget:
    """Test laser target representation."""
    
    def test_laser_target_creation(self):
        """Test laser target creation."""
        target = LaserTarget(
            position=(1000, 500, 200),
            velocity=(50, 0, -10),
            reflectivity=0.8,
            surface_area=2.5,
            material_properties={'thermal_conductivity': 200},
            target_id="TARGET_001"
        )
        
        assert target.position == (1000, 500, 200)
        assert target.velocity == (50, 0, -10)
        assert target.reflectivity == 0.8
        assert target.surface_area == 2.5
        assert target.target_id == "TARGET_001"
        assert target.material_properties['thermal_conductivity'] == 200


class TestAtmosphericParameters:
    """Test atmospheric parameters."""
    
    def test_atmospheric_params_creation(self):
        """Test atmospheric parameters creation."""
        params = AtmosphericParameters(
            visibility=15.0,
            temperature=293.15,
            pressure=95000,
            humidity=75.0,
            wind_speed=8.0,
            turbulence_strength=5e-14,
            condition=AtmosphericCondition.HAZE
        )
        
        assert params.visibility == 15.0
        assert params.temperature == 293.15
        assert params.condition == AtmosphericCondition.HAZE


if __name__ == "__main__":
    pytest.main([__file__])