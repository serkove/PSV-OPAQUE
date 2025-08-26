"""Tests for Engine Performance Model."""

import pytest
import math
from fighter_jet_sdk.engines.propulsion.engine_performance_model import (
    EnginePerformanceModel, EngineSpecification, EngineOperatingPoint,
    EngineType, EnginePerformanceData
)


class TestEnginePerformanceModel:
    """Test cases for EnginePerformanceModel."""
    
    @pytest.fixture
    def turbofan_spec(self):
        """Create a turbofan engine specification for testing."""
        return EngineSpecification(
            engine_id="test_turbofan_001",
            name="Test Turbofan Engine",
            engine_type=EngineType.AFTERBURNING_TURBOFAN,
            max_thrust_sea_level=100000.0,  # 100 kN
            max_thrust_altitude=80000.0,    # 80 kN at altitude
            design_altitude=11000.0,        # 11 km
            design_mach=1.6,
            bypass_ratio=0.3,
            pressure_ratio=25.0,
            turbine_inlet_temperature=1800.0,
            mass=1500.0,  # kg
            length=4.5,   # m
            diameter=1.2, # m
            afterburner_thrust_multiplier=1.6
        )
    
    @pytest.fixture
    def ramjet_spec(self):
        """Create a ramjet engine specification for testing."""
        return EngineSpecification(
            engine_id="test_ramjet_001",
            name="Test Ramjet Engine",
            engine_type=EngineType.RAMJET,
            max_thrust_sea_level=0.0,       # Ramjets don't work at sea level
            max_thrust_altitude=150000.0,   # 150 kN at design point
            design_altitude=15000.0,        # 15 km
            design_mach=3.0,
            pressure_ratio=1.0,             # No compressor
            turbine_inlet_temperature=2000.0,
            mass=800.0,   # kg
            length=3.0,   # m
            diameter=0.8  # m
        )
    
    @pytest.fixture
    def performance_model(self, turbofan_spec):
        """Create performance model for testing."""
        return EnginePerformanceModel(turbofan_spec)
    
    def test_initialization(self, performance_model, turbofan_spec):
        """Test engine performance model initialization."""
        assert performance_model.engine_spec == turbofan_spec
        assert len(performance_model.thrust_map) > 0
        assert len(performance_model.sfc_map) > 0
        
        # Check atmosphere model constants
        assert performance_model.sea_level_pressure == 101325.0
        assert performance_model.sea_level_temperature == 288.15
        assert performance_model.sea_level_density == 1.225
    
    def test_atmospheric_conditions(self, performance_model):
        """Test atmospheric conditions calculation."""
        # Sea level conditions
        temp, pressure, density = performance_model.get_atmospheric_conditions(0)
        assert abs(temp - 288.15) < 0.1
        assert abs(pressure - 101325.0) < 1.0
        assert abs(density - 1.225) < 0.001
        
        # Altitude conditions (10 km)
        temp_10k, pressure_10k, density_10k = performance_model.get_atmospheric_conditions(10000)
        assert temp_10k < temp  # Temperature decreases with altitude
        assert pressure_10k < pressure  # Pressure decreases with altitude
        assert density_10k < density  # Density decreases with altitude
        
        # Check reasonable values at 10 km
        assert 200 < temp_10k < 250  # K
        assert 20000 < pressure_10k < 30000  # Pa
        assert 0.3 < density_10k < 0.5  # kg/m³
    
    def test_thrust_calculation_sea_level(self, performance_model):
        """Test thrust calculation at sea level."""
        operating_point = EngineOperatingPoint(
            altitude=0.0,
            mach_number=0.0,
            throttle_setting=1.0,
            afterburner_engaged=False
        )
        
        thrust = performance_model.calculate_thrust(operating_point)
        
        # Should be close to max thrust at sea level
        assert thrust > 0
        assert thrust <= performance_model.engine_spec.max_thrust_sea_level
        
        # Test with reduced throttle
        operating_point.throttle_setting = 0.5
        thrust_half = performance_model.calculate_thrust(operating_point)
        assert thrust_half < thrust
        assert thrust_half > 0.4 * thrust  # Should be roughly proportional
    
    def test_thrust_calculation_with_afterburner(self, performance_model):
        """Test thrust calculation with afterburner engaged."""
        operating_point = EngineOperatingPoint(
            altitude=0.0,
            mach_number=0.9,
            throttle_setting=1.0,
            afterburner_engaged=False
        )
        
        thrust_dry = performance_model.calculate_thrust(operating_point)
        
        # Enable afterburner
        operating_point.afterburner_engaged = True
        thrust_wet = performance_model.calculate_thrust(operating_point)
        
        # Afterburner should increase thrust
        assert thrust_wet > thrust_dry
        multiplier = thrust_wet / thrust_dry
        assert 1.4 < multiplier < 2.0  # Reasonable afterburner multiplier
    
    def test_fuel_consumption_calculation(self, performance_model):
        """Test fuel consumption calculation."""
        operating_point = EngineOperatingPoint(
            altitude=10000.0,
            mach_number=0.8,
            throttle_setting=0.8,
            afterburner_engaged=False
        )
        
        fuel_flow = performance_model.calculate_fuel_consumption(operating_point)
        
        # Fuel flow should be positive
        assert fuel_flow > 0
        
        # Test with afterburner - should increase fuel consumption significantly
        operating_point.afterburner_engaged = True
        fuel_flow_ab = performance_model.calculate_fuel_consumption(operating_point)
        
        assert fuel_flow_ab > fuel_flow
        assert fuel_flow_ab > 2.0 * fuel_flow  # Afterburner penalty
    
    def test_thrust_to_weight_ratio(self, performance_model):
        """Test thrust-to-weight ratio calculation."""
        aircraft_mass = 15000.0  # kg (typical fighter jet)
        
        operating_point = EngineOperatingPoint(
            altitude=0.0,
            mach_number=0.0,
            throttle_setting=1.0,
            afterburner_engaged=True
        )
        
        # Single engine
        twr_single = performance_model.calculate_thrust_to_weight_ratio(
            aircraft_mass, operating_point, num_engines=1
        )
        
        # Twin engine
        twr_twin = performance_model.calculate_thrust_to_weight_ratio(
            aircraft_mass, operating_point, num_engines=2
        )
        
        assert twr_single > 0
        assert twr_twin > twr_single
        assert abs(twr_twin - 2 * twr_single) < 0.01  # Should be roughly double
        
        # Reasonable values for fighter jet
        assert 0.5 < twr_single < 2.0
        assert 1.0 < twr_twin < 4.0
    
    def test_range_fuel_consumption(self, performance_model):
        """Test fuel consumption calculation for flight profile."""
        # Create simple flight profile
        flight_profile = [
            EngineOperatingPoint(0.0, 0.0, 1.0, True),      # Takeoff with AB
            EngineOperatingPoint(5000.0, 0.6, 0.9, False),  # Climb
            EngineOperatingPoint(10000.0, 0.8, 0.7, False), # Cruise
            EngineOperatingPoint(0.0, 0.3, 0.5, False)      # Landing
        ]
        
        flight_times = [60.0, 300.0, 3600.0, 180.0]  # seconds
        
        total_fuel = performance_model.calculate_range_fuel_consumption(
            flight_profile, flight_times
        )
        
        assert total_fuel > 0
        
        # Should be reasonable for a 1+ hour flight
        assert 500 < total_fuel < 5000  # kg
    
    def test_cruise_optimization(self, performance_model):
        """Test cruise condition optimization."""
        aircraft_mass = 12000.0  # kg
        
        best_alt, best_mach, best_sfc = performance_model.optimize_cruise_conditions(
            altitude_range=(8000.0, 15000.0),
            mach_range=(0.7, 1.2),
            aircraft_mass=aircraft_mass
        )
        
        # Results should be within specified ranges
        assert 8000.0 <= best_alt <= 15000.0
        assert 0.7 <= best_mach <= 1.2
        assert best_sfc > 0
        
        # Optimal conditions should be reasonable for turbofan
        assert best_alt > 8000.0  # Higher altitude generally better
        assert 0.7 <= best_mach <= 1.2  # Within specified range
    
    def test_performance_envelope(self, performance_model):
        """Test performance envelope generation."""
        envelope = performance_model.get_performance_envelope()
        
        # Check structure
        assert 'engine_spec' in envelope
        assert 'operating_limits' in envelope
        assert 'performance_data' in envelope
        
        # Check engine spec data
        engine_data = envelope['engine_spec']
        assert engine_data['name'] == "Test Turbofan Engine"
        assert engine_data['type'] == "afterburning_turbofan"
        assert engine_data['max_thrust_sl'] == 100000.0
        
        # Check operating limits
        limits = envelope['operating_limits']
        assert limits['max_altitude'] > 0
        assert limits['max_mach'] > 0
        assert limits['min_mach'] >= 0
        
        # Check performance data points
        perf_data = envelope['performance_data']
        assert len(perf_data) > 0
        
        for point_name, point_data in perf_data.items():
            assert 'thrust' in point_data
            assert 'fuel_flow' in point_data
            assert 'sfc' in point_data
            assert point_data['thrust'] > 0
            assert point_data['fuel_flow'] > 0
    
    def test_ramjet_performance(self, ramjet_spec):
        """Test ramjet-specific performance characteristics."""
        ramjet_model = EnginePerformanceModel(ramjet_spec)
        
        # Ramjet should not work at low Mach numbers
        low_mach_point = EngineOperatingPoint(
            altitude=10000.0,
            mach_number=0.5,
            throttle_setting=1.0
        )
        
        assert not ramjet_model._is_valid_operating_point(10000.0, 0.5)
        
        # Should work at high Mach numbers
        high_mach_point = EngineOperatingPoint(
            altitude=15000.0,
            mach_number=3.0,
            throttle_setting=1.0
        )
        
        assert ramjet_model._is_valid_operating_point(15000.0, 3.0)
        
        thrust = ramjet_model.calculate_thrust(high_mach_point)
        assert thrust > 0
    
    def test_engine_validation(self, turbofan_spec):
        """Test engine specification validation."""
        model = EnginePerformanceModel(turbofan_spec)
        errors = model.validate_engine_specification()
        
        # Valid spec should have no errors
        assert len(errors) == 0
        
        # Test invalid specifications
        invalid_spec = EngineSpecification(
            engine_id="invalid",
            name="Invalid Engine",
            engine_type=EngineType.TURBOFAN,
            max_thrust_sea_level=-1000.0,  # Invalid: negative thrust
            max_thrust_altitude=0.0,
            design_altitude=0.0,
            design_mach=0.0,
            bypass_ratio=-1.0,  # Invalid: negative bypass ratio
            pressure_ratio=0.5,  # Invalid: < 1.0
            mass=-100.0  # Invalid: negative mass
        )
        
        invalid_model = EnginePerformanceModel(invalid_spec)
        errors = invalid_model.validate_engine_specification()
        
        assert len(errors) > 0
        assert any("thrust" in error.lower() for error in errors)
        assert any("mass" in error.lower() for error in errors)
        assert any("pressure ratio" in error.lower() for error in errors)
    
    def test_variable_cycle_engine(self):
        """Test variable cycle engine capabilities."""
        vc_spec = EngineSpecification(
            engine_id="test_vc_001",
            name="Test Variable Cycle Engine",
            engine_type=EngineType.VARIABLE_CYCLE,
            max_thrust_sea_level=120000.0,
            max_thrust_altitude=100000.0,
            design_altitude=12000.0,
            design_mach=2.0,
            bypass_ratio=0.5,  # Can vary in operation
            pressure_ratio=30.0,
            variable_cycle_modes=["low_bypass", "high_bypass", "turbojet"]
        )
        
        vc_model = EnginePerformanceModel(vc_spec)
        
        # Should handle different operating conditions well
        subsonic_point = EngineOperatingPoint(10000.0, 0.8, 0.8)
        supersonic_point = EngineOperatingPoint(15000.0, 1.8, 1.0)
        
        thrust_subsonic = vc_model.calculate_thrust(subsonic_point)
        thrust_supersonic = vc_model.calculate_thrust(supersonic_point)
        
        assert thrust_subsonic > 0
        assert thrust_supersonic > 0
        
        # Variable cycle should be efficient across range
        fuel_subsonic = vc_model.calculate_fuel_consumption(subsonic_point)
        fuel_supersonic = vc_model.calculate_fuel_consumption(supersonic_point)
        
        assert fuel_subsonic > 0
        assert fuel_supersonic > 0
    
    def test_edge_cases(self, performance_model):
        """Test edge cases and boundary conditions."""
        # Zero throttle
        zero_throttle = EngineOperatingPoint(
            altitude=10000.0,
            mach_number=0.8,
            throttle_setting=0.0
        )
        
        thrust_zero = performance_model.calculate_thrust(zero_throttle)
        assert thrust_zero == 0.0
        
        # Very high altitude
        high_alt = EngineOperatingPoint(
            altitude=20000.0,
            mach_number=0.8,
            throttle_setting=1.0
        )
        
        thrust_high_alt = performance_model.calculate_thrust(high_alt)
        assert thrust_high_alt >= 0  # Should not be negative
        
        # Invalid operating points should be handled gracefully
        invalid_point = EngineOperatingPoint(
            altitude=-1000.0,  # Negative altitude
            mach_number=-0.5,  # Negative Mach
            throttle_setting=2.0  # > 1.0 throttle
        )
        
        # Should not crash, may return zero or handle gracefully
        thrust_invalid = performance_model.calculate_thrust(invalid_point)
        assert thrust_invalid >= 0
    
    def test_performance_consistency(self, performance_model):
        """Test consistency of performance calculations."""
        base_point = EngineOperatingPoint(
            altitude=10000.0,
            mach_number=0.8,
            throttle_setting=0.8
        )
        
        # Calculate thrust and fuel consumption
        thrust = performance_model.calculate_thrust(base_point)
        fuel_flow = performance_model.calculate_fuel_consumption(base_point)
        
        # Calculate fuel consumption with provided thrust
        fuel_flow_2 = performance_model.calculate_fuel_consumption(base_point, thrust)
        
        # Should be the same
        assert abs(fuel_flow - fuel_flow_2) < 0.001
        
        # SFC should be reasonable
        sfc = fuel_flow / thrust if thrust > 0 else 0
        assert 0.00001 < sfc < 0.0001  # Reasonable SFC range for turbofan


if __name__ == "__main__":
    pytest.main([__file__])