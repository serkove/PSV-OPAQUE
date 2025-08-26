"""Integration tests for the complete Sensors Engine."""

import pytest
from fighter_jet_sdk.engines.sensors.engine import SensorsEngine


class TestSensorsEngineIntegration:
    """Test complete sensors engine integration."""
    
    @pytest.fixture
    def sensors_engine(self):
        """Create sensors engine with sample configuration."""
        config = {
            'aesa_radars': {
                'main_radar': {
                    'frequency': 10e9,
                    'peak_power': 100e3,
                    'array_elements': 1024,
                    'element_spacing': 0.015,
                    'pulse_width': 1e-6,
                    'pulse_repetition_frequency': 1000,
                    'noise_figure': 3.0,
                    'system_losses': 6.0,
                    'antenna_gain': 35.0
                }
            },
            'laser_systems': {
                'main_laser': {
                    'wavelength': 1.064e-6,
                    'peak_power': 1e6,
                    'pulse_energy': 10.0,
                    'pulse_duration': 10e-9,
                    'beam_divergence': 1e-6,
                    'beam_quality': 1.2,
                    'laser_type': 'SOLID_STATE'
                }
            },
            'plasma_systems': {
                'main_plasma': {
                    'plasma_type': 'RF_SUSTAINED',
                    'frequency': 2.45e9,
                    'power_input': 50000,
                    'gas_composition': {'Ar': 1.0},
                    'pressure': 1000,
                    'temperature_target': 8000,
                    'volume': 0.5,
                    'communication_range': 10000.0
                }
            }
        }
        
        engine = SensorsEngine(config)
        engine.initialize()
        return engine
    
    def test_engine_initialization(self, sensors_engine):
        """Test engine initialization with all systems."""
        assert sensors_engine.initialized
        assert len(sensors_engine.aesa_radars) == 1
        assert len(sensors_engine.laser_systems) == 1
        assert len(sensors_engine.plasma_systems) == 1
    
    def test_system_status(self, sensors_engine):
        """Test comprehensive system status."""
        status = sensors_engine.get_system_status()
        
        # Check main structure
        assert 'aesa_radars' in status
        assert 'laser_systems' in status
        assert 'plasma_systems' in status
        assert 'atmospheric_conditions' in status
        
        # Check AESA radar status
        assert 'main_radar' in status['aesa_radars']
        radar_status = status['aesa_radars']['main_radar']
        assert 'frequency_ghz' in radar_status
        assert 'peak_power_kw' in radar_status
        
        # Check laser system status
        assert 'main_laser' in status['laser_systems']
        laser_status = status['laser_systems']['main_laser']
        assert 'wavelength_nm' in laser_status
        assert 'peak_power_w' in laser_status
        
        # Check plasma system status
        assert 'main_plasma' in status['plasma_systems']
        plasma_status = status['plasma_systems']['main_plasma']
        assert 'active_orbs' in plasma_status
        assert 'plasma_type' in plasma_status
    
    def test_target_detection_workflow(self, sensors_engine):
        """Test integrated target detection workflow."""
        # Deploy plasma network first
        orb_ids = sensors_engine.deploy_plasma_network(
            'main_plasma',
            center_position=(0, 0, 1000),
            num_orbs=3,
            spacing=1000,
            pattern='line'
        )
        
        assert len(orb_ids) == 3
        
        # Define test targets
        targets = [
            {
                'id': 'T001',
                'position': [1000, 500, 1000],
                'velocity': [100, 0, 0],
                'rcs': 5.0
            },
            {
                'id': 'T002',
                'position': [2000, 0, 1000],
                'velocity': [-50, 25, 0],
                'rcs': 2.0
            }
        ]
        
        # Process target detection
        detection_data = {
            'operation': 'detect_targets',
            'targets': targets,
            'beam_azimuth': 0.0,
            'beam_elevation': 0.0
        }
        
        results = sensors_engine.process(detection_data)
        
        # Check results structure
        assert 'aesa_detections' in results
        assert 'plasma_detections' in results
        assert 'fused_results' in results
        
        # Should have radar detections
        assert 'main_radar' in results['aesa_detections']
        
        # Should have plasma detections
        assert 'main_plasma' in results['plasma_detections']
    
    def test_libs_analysis(self, sensors_engine):
        """Test LIBS material analysis."""
        results = sensors_engine.analyze_material_libs(
            'main_laser',
            target_material='metal',
            laser_power=1e6
        )
        
        # Check analysis results
        assert 'plasma_temperature_k' in results
        assert 'spectrum' in results
        assert 'radioactive_elements' in results
        assert 'analysis_successful' in results
        
        assert results['plasma_temperature_k'] > 0
        assert isinstance(results['spectrum'], dict)
        assert isinstance(results['radioactive_elements'], list)
    
    def test_atmospheric_updates(self, sensors_engine):
        """Test atmospheric condition updates."""
        # Update conditions
        sensors_engine.update_atmospheric_conditions(
            visibility=15.0,
            temperature=293.15,
            pressure=95000,
            humidity=75.0,
            wind_speed=8.0,
            turbulence_strength=5e-14
        )
        
        # Check updated conditions in status
        status = sensors_engine.get_system_status()
        atmo = status['atmospheric_conditions']
        
        assert atmo['visibility_km'] == 15.0
        assert atmo['temperature_k'] == 293.15
        assert atmo['pressure_pa'] == 95000
        assert atmo['humidity_percent'] == 75.0
    
    def test_invalid_operations(self, sensors_engine):
        """Test handling of invalid operations."""
        # Test invalid radar ID
        targets = []
        detected = sensors_engine.detect_targets_aesa(
            'nonexistent_radar', targets, 0.0, 0.0
        )
        assert detected == []
        
        # Test invalid laser ID
        results = sensors_engine.analyze_material_libs(
            'nonexistent_laser', 'metal', 1e6
        )
        assert results == {}
        
        # Test invalid plasma ID
        orb_ids = sensors_engine.deploy_plasma_network(
            'nonexistent_plasma', (0, 0, 0), 3, 1000
        )
        assert orb_ids == []
    
    def test_process_invalid_data(self, sensors_engine):
        """Test processing with invalid data."""
        # Test with non-dict data
        result = sensors_engine.process("invalid_data")
        assert result is None
        
        # Test with unknown operation
        result = sensors_engine.process({'operation': 'unknown_op'})
        assert result is None
        
        # Test status operation
        result = sensors_engine.process({'operation': 'status'})
        assert result is not None
        assert 'aesa_radars' in result


if __name__ == "__main__":
    pytest.main([__file__])