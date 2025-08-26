"""Tests for Propulsion Engine integration."""

import pytest
from fighter_jet_sdk.engines.propulsion.engine import PropulsionEngine


class TestPropulsionEngine:
    """Test cases for PropulsionEngine."""
    
    @pytest.fixture
    def propulsion_engine(self):
        """Create propulsion engine for testing."""
        engine = PropulsionEngine()
        engine.initialize()
        return engine
    
    def test_initialization(self, propulsion_engine):
        """Test propulsion engine initialization."""
        assert propulsion_engine.initialized
        assert len(propulsion_engine.performance_models) > 0
        assert len(propulsion_engine.engine_specifications) > 0
        
        # Check that default engines are loaded
        expected_engines = ['f119_pw_100', 'f135_pw_100', 'advanced_ramjet_001', 'variable_cycle_001']
        for engine_id in expected_engines:
            assert engine_id in propulsion_engine.engine_specifications
            assert engine_id in propulsion_engine.performance_models
    
    def test_get_available_engines(self, propulsion_engine):
        """Test getting available engine list."""
        engines = propulsion_engine.get_available_engines()
        
        assert len(engines) >= 4  # At least the default engines
        
        for engine in engines:
            assert 'engine_id' in engine
            assert 'name' in engine
            assert 'type' in engine
            assert 'max_thrust_sl' in engine
            assert 'mass' in engine
            assert 'afterburner_capable' in engine
            
            # Check data types
            assert isinstance(engine['max_thrust_sl'], (int, float))
            assert isinstance(engine['mass'], (int, float))
            assert isinstance(engine['afterburner_capable'], bool)
    
    def test_process_basic_operation(self, propulsion_engine):
        """Test basic engine operation processing."""
        input_data = {
            'engine_id': 'f119_pw_100',
            'operating_conditions': {
                'altitude': 10000.0,
                'mach_number': 1.2,
                'throttle_setting': 0.8,
                'afterburner_engaged': False
            },
            'aircraft_mass': 15000.0,
            'num_engines': 2
        }
        
        result = propulsion_engine.process(input_data)
        
        # Check result structure
        assert 'engine_id' in result
        assert 'thrust' in result
        assert 'fuel_consumption' in result
        assert 'thrust_to_weight_ratio' in result
        assert 'operating_point' in result
        
        # Check values are reasonable
        assert result['thrust'] > 0
        assert result['fuel_consumption'] > 0
        assert result['thrust_to_weight_ratio'] > 0
        assert result['engine_id'] == 'f119_pw_100'
    
    def test_process_with_afterburner(self, propulsion_engine):
        """Test engine operation with afterburner."""
        base_data = {
            'engine_id': 'f135_pw_100',
            'operating_conditions': {
                'altitude': 5000.0,
                'mach_number': 1.5,
                'throttle_setting': 1.0,
                'afterburner_engaged': False
            }
        }
        
        # Test without afterburner
        result_dry = propulsion_engine.process(base_data)
        
        # Test with afterburner
        base_data['operating_conditions']['afterburner_engaged'] = True
        result_wet = propulsion_engine.process(base_data)
        
        # Afterburner should increase thrust and fuel consumption
        assert result_wet['thrust'] > result_dry['thrust']
        assert result_wet['fuel_consumption'] > result_dry['fuel_consumption']
    
    def test_mission_fuel_consumption(self, propulsion_engine):
        """Test mission fuel consumption calculation."""
        flight_profile = [
            {
                'altitude': 0.0,
                'mach_number': 0.0,
                'throttle_setting': 1.0,
                'afterburner_engaged': True,
                'duration': 60.0  # Takeoff - 1 minute
            },
            {
                'altitude': 8000.0,
                'mach_number': 0.7,
                'throttle_setting': 0.9,
                'afterburner_engaged': False,
                'duration': 600.0  # Climb - 10 minutes
            },
            {
                'altitude': 12000.0,
                'mach_number': 0.85,
                'throttle_setting': 0.7,
                'afterburner_engaged': False,
                'duration': 3600.0  # Cruise - 1 hour
            },
            {
                'altitude': 1000.0,
                'mach_number': 0.4,
                'throttle_setting': 0.6,
                'afterburner_engaged': False,
                'duration': 300.0  # Descent/Landing - 5 minutes
            }
        ]
        
        result = propulsion_engine.calculate_mission_fuel_consumption('f119_pw_100', flight_profile)
        
        # Check result structure
        assert 'engine_id' in result
        assert 'total_fuel_consumption' in result
        assert 'mission_duration' in result
        assert 'segment_breakdown' in result
        
        # Check values
        assert result['total_fuel_consumption'] > 0
        assert result['mission_duration'] == 4560.0  # Total duration
        assert len(result['segment_breakdown']) == 4
        
        # Check segment breakdown
        for i, segment in enumerate(result['segment_breakdown']):
            assert segment['segment'] == i + 1
            assert segment['fuel_flow_rate'] > 0
            assert segment['segment_fuel'] > 0
            assert segment['duration'] == flight_profile[i]['duration']
    
    def test_cruise_optimization(self, propulsion_engine):
        """Test cruise performance optimization."""
        result = propulsion_engine.optimize_cruise_performance(
            'f135_pw_100',
            aircraft_mass=12000.0,
            altitude_range=(8000.0, 15000.0),
            mach_range=(0.7, 1.1)
        )
        
        # Check result structure
        assert 'engine_id' in result
        assert 'optimal_altitude' in result
        assert 'optimal_mach' in result
        assert 'optimal_sfc' in result
        assert 'cruise_thrust' in result
        assert 'cruise_fuel_flow' in result
        
        # Check values are within ranges
        assert 8000.0 <= result['optimal_altitude'] <= 15000.0
        assert 0.7 <= result['optimal_mach'] <= 1.1
        assert result['optimal_sfc'] > 0
        assert result['cruise_thrust'] > 0
        assert result['cruise_fuel_flow'] > 0
    
    def test_performance_envelope(self, propulsion_engine):
        """Test performance envelope retrieval."""
        envelope = propulsion_engine.get_engine_performance_envelope('f119_pw_100')
        
        # Check structure
        assert 'engine_spec' in envelope
        assert 'operating_limits' in envelope
        assert 'performance_data' in envelope
        
        # Check engine spec
        engine_spec = envelope['engine_spec']
        assert engine_spec['name'] == 'F119-PW-100'
        assert engine_spec['type'] == 'afterburning_turbofan'
        
        # Check operating limits
        limits = envelope['operating_limits']
        assert limits['max_altitude'] > 0
        assert limits['max_mach'] > 0
        assert limits['min_mach'] >= 0
    
    def test_mission_validation(self, propulsion_engine):
        """Test engine validation for mission requirements."""
        # Valid mission requirements
        valid_mission = {
            'max_altitude': 15000.0,
            'max_mach': 2.0,
            'min_thrust': 80000.0
        }
        
        result = propulsion_engine.validate_engine_for_mission('f119_pw_100', valid_mission)
        
        assert result['meets_requirements'] == True
        assert len(result['issues']) == 0
        
        # Invalid mission requirements
        invalid_mission = {
            'max_altitude': 30000.0,  # Too high
            'max_mach': 5.0,          # Too fast for turbofan
            'min_thrust': 200000.0    # Too much thrust
        }
        
        result_invalid = propulsion_engine.validate_engine_for_mission('f119_pw_100', invalid_mission)
        
        assert result_invalid['meets_requirements'] == False
        assert len(result_invalid['issues']) > 0
    
    def test_ramjet_validation(self, propulsion_engine):
        """Test ramjet engine validation and recommendations."""
        # Low-speed mission (not suitable for ramjet)
        low_speed_mission = {
            'max_altitude': 15000.0,
            'max_mach': 1.5,
            'min_thrust': 50000.0
        }
        
        result = propulsion_engine.validate_engine_for_mission('advanced_ramjet_001', low_speed_mission)
        
        # Should have recommendations about using turbofan instead
        assert len(result['recommendations']) > 0
        assert any('turbofan' in rec.lower() for rec in result['recommendations'])
        
        # High-speed mission (suitable for ramjet)
        high_speed_mission = {
            'max_altitude': 18000.0,
            'max_mach': 4.0,
            'min_thrust': 100000.0
        }
        
        result_hs = propulsion_engine.validate_engine_for_mission('advanced_ramjet_001', high_speed_mission)
        assert result_hs['meets_requirements'] == True
    
    def test_input_validation(self, propulsion_engine):
        """Test input validation."""
        # Invalid input - not a dictionary
        assert not propulsion_engine.validate_input("invalid")
        assert not propulsion_engine.validate_input(123)
        assert not propulsion_engine.validate_input([])
        
        # Invalid input - missing required fields
        assert not propulsion_engine.validate_input({})
        assert not propulsion_engine.validate_input({'engine_id': 'test'})
        assert not propulsion_engine.validate_input({'operating_conditions': {}})
        
        # Valid input
        valid_input = {
            'engine_id': 'f119_pw_100',
            'operating_conditions': {
                'altitude': 10000.0,
                'mach_number': 1.0,
                'throttle_setting': 0.8
            }
        }
        assert propulsion_engine.validate_input(valid_input)
    
    def test_error_handling(self, propulsion_engine):
        """Test error handling for invalid operations."""
        # Unknown engine ID
        with pytest.raises(ValueError, match="Unknown engine ID"):
            propulsion_engine.process({
                'engine_id': 'nonexistent_engine',
                'operating_conditions': {
                    'altitude': 10000.0,
                    'mach_number': 1.0,
                    'throttle_setting': 0.8
                }
            })
        
        # Invalid input data
        with pytest.raises(ValueError, match="Invalid input data"):
            propulsion_engine.process("invalid_data")
        
        # Unknown engine for mission fuel calculation
        with pytest.raises(ValueError, match="Unknown engine ID"):
            propulsion_engine.calculate_mission_fuel_consumption('nonexistent', [])
        
        # Unknown engine for optimization
        with pytest.raises(ValueError, match="Unknown engine ID"):
            propulsion_engine.optimize_cruise_performance('nonexistent', 10000.0)
    
    def test_variable_cycle_engine(self, propulsion_engine):
        """Test variable cycle engine capabilities."""
        # Test variable cycle engine across different conditions
        subsonic_data = {
            'engine_id': 'variable_cycle_001',
            'operating_conditions': {
                'altitude': 10000.0,
                'mach_number': 0.8,
                'throttle_setting': 0.8
            }
        }
        
        supersonic_data = {
            'engine_id': 'variable_cycle_001',
            'operating_conditions': {
                'altitude': 15000.0,
                'mach_number': 2.0,
                'throttle_setting': 1.0
            }
        }
        
        result_subsonic = propulsion_engine.process(subsonic_data)
        result_supersonic = propulsion_engine.process(supersonic_data)
        
        # Both should work and produce reasonable results
        assert result_subsonic['thrust'] > 0
        assert result_supersonic['thrust'] > 0
        assert result_subsonic['fuel_consumption'] > 0
        assert result_supersonic['fuel_consumption'] > 0
        
        # Variable cycle should handle both regimes efficiently
        # (specific performance characteristics would depend on detailed modeling)


if __name__ == "__main__":
    pytest.main([__file__])