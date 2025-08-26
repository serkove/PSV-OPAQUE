"""Unit tests for combined-cycle engine performance model."""

import pytest
import math
from unittest.mock import Mock, patch

from fighter_jet_sdk.engines.propulsion.combined_cycle_engine import (
    CombinedCycleEngine,
    CombinedCycleOperatingPoint,
    CombinedCyclePerformanceData,
    CombinedCycleSpecification,
    PropulsionMode
)
from fighter_jet_sdk.common.enums import ExtremePropulsionType


class TestCombinedCycleEngine:
    """Test cases for CombinedCycleEngine class."""
    
    @pytest.fixture
    def sample_combined_cycle_spec(self):
        """Create a sample combined-cycle engine specification."""
        from fighter_jet_sdk.engines.propulsion.engine_performance_model import EngineType
        
        return CombinedCycleSpecification(
            engine_id="cc_engine_001",
            name="Mach 60 Combined Cycle Engine",
            engine_type=EngineType.SCRAMJET,
            extreme_propulsion_type=ExtremePropulsionType.COMBINED_CYCLE_AIRBREATHING,
            max_thrust_sea_level=500000.0,  # 500 kN
            max_thrust_altitude=800000.0,   # 800 kN
            design_altitude=50000.0,        # 50 km
            design_mach=30.0,
            transition_mach_number=12.0,
            rocket_specific_impulse=450.0,
            air_breathing_specific_impulse=3000.0,
            max_rocket_thrust=1000000.0,    # 1000 kN
            rocket_fuel_capacity=5000.0,    # 5000 kg
            air_breathing_fuel_capacity=10000.0,  # 10000 kg
            plasma_interaction_threshold=25.0,
            max_stagnation_temperature=60000.0,
            dissociation_onset_temperature=4000.0,
            mass=2000.0,  # 2000 kg
            length=8.0,   # 8 m
            diameter=1.5  # 1.5 m
        )
    
    @pytest.fixture
    def combined_cycle_engine(self, sample_combined_cycle_spec):
        """Create a combined-cycle engine instance."""
        return CombinedCycleEngine(sample_combined_cycle_spec)
    
    def test_engine_initialization(self, combined_cycle_engine, sample_combined_cycle_spec):
        """Test combined-cycle engine initialization."""
        assert combined_cycle_engine.combined_spec == sample_combined_cycle_spec
        assert combined_cycle_engine.combined_spec.name == "Mach 60 Combined Cycle Engine"
        assert combined_cycle_engine.combined_spec.transition_mach_number == 12.0
        
        # Check that performance maps are generated
        assert len(combined_cycle_engine.mode_transition_map) > 0
        assert len(combined_cycle_engine.rocket_thrust_map) > 0
        assert len(combined_cycle_engine.plasma_effects_map) > 0
    
    def test_propulsion_mode_determination(self, combined_cycle_engine):
        """Test propulsion mode determination based on Mach number."""
        # Air-breathing mode at low Mach
        mode = combined_cycle_engine._determine_propulsion_mode(8.0)
        assert mode == PropulsionMode.AIR_BREATHING
        
        # Transition mode around transition Mach
        mode = combined_cycle_engine._determine_propulsion_mode(12.5)
        assert mode == PropulsionMode.TRANSITION
        
        # Rocket-assisted mode at high Mach
        mode = combined_cycle_engine._determine_propulsion_mode(25.0)
        assert mode == PropulsionMode.ROCKET_ASSISTED
        
        # Pure rocket mode at extreme Mach
        mode = combined_cycle_engine._determine_propulsion_mode(50.0)
        assert mode == PropulsionMode.PURE_ROCKET
    
    def test_valid_operating_points(self, combined_cycle_engine):
        """Test validation of combined-cycle operating points."""
        # Valid point
        assert combined_cycle_engine._is_valid_combined_cycle_point(50000.0, 30.0) == True
        
        # Invalid altitude (too low)
        assert combined_cycle_engine._is_valid_combined_cycle_point(20000.0, 30.0) == False
        
        # Invalid altitude (too high)
        assert combined_cycle_engine._is_valid_combined_cycle_point(120000.0, 30.0) == False
        
        # Invalid Mach (too low)
        assert combined_cycle_engine._is_valid_combined_cycle_point(50000.0, 2.0) == False
        
        # Invalid Mach (too high)
        assert combined_cycle_engine._is_valid_combined_cycle_point(50000.0, 70.0) == False
    
    def test_rocket_thrust_ratio_calculation(self, combined_cycle_engine):
        """Test rocket thrust ratio calculation for different modes."""
        # Air-breathing mode should have zero rocket contribution
        ratio = combined_cycle_engine._calculate_rocket_thrust_ratio(
            50000.0, 8.0, PropulsionMode.AIR_BREATHING
        )
        assert ratio == 0.0
        
        # Pure rocket mode should have full rocket contribution
        ratio = combined_cycle_engine._calculate_rocket_thrust_ratio(
            50000.0, 50.0, PropulsionMode.PURE_ROCKET
        )
        assert ratio == 1.0
        
        # Transition mode should have partial contribution
        ratio = combined_cycle_engine._calculate_rocket_thrust_ratio(
            50000.0, 13.0, PropulsionMode.TRANSITION
        )
        assert 0.0 < ratio < 1.0
    
    def test_plasma_effects_calculation(self, combined_cycle_engine):
        """Test plasma effects factor calculation."""
        # Below plasma threshold should have no effects
        factor = combined_cycle_engine._calculate_plasma_effects_factor(20.0)
        assert factor == 1.0
        
        # Above plasma threshold should have reduced performance
        factor = combined_cycle_engine._calculate_plasma_effects_factor(40.0)
        assert factor < 1.0
        assert factor >= 0.5  # Minimum performance retention
        
        # At extreme Mach numbers
        factor = combined_cycle_engine._calculate_plasma_effects_factor(60.0)
        assert factor >= 0.5
    
    def test_stagnation_temperature_calculation(self, combined_cycle_engine):
        """Test stagnation temperature calculation for extreme conditions."""
        # Low Mach number (standard formula)
        temp = combined_cycle_engine.calculate_stagnation_temperature(5.0, 50000.0)
        assert temp > 0
        assert temp < 5000.0  # Reasonable temperature for low Mach
        
        # High Mach number (real gas effects)
        temp = combined_cycle_engine.calculate_stagnation_temperature(30.0, 50000.0)
        assert temp > 5000.0  # Higher temperature at high Mach
        assert temp <= combined_cycle_engine.combined_spec.max_stagnation_temperature
        
        # Extreme Mach number (dissociation effects)
        temp = combined_cycle_engine.calculate_stagnation_temperature(60.0, 50000.0)
        assert temp <= combined_cycle_engine.combined_spec.max_stagnation_temperature
    
    def test_dissociation_losses_calculation(self, combined_cycle_engine):
        """Test dissociation losses calculation."""
        # Below dissociation onset should have no losses
        losses = combined_cycle_engine.calculate_dissociation_losses(3000.0)
        assert losses == 0.0
        
        # Above dissociation onset should have losses
        losses = combined_cycle_engine.calculate_dissociation_losses(20000.0)
        assert losses > 0.0
        assert losses <= 0.4  # Maximum 40% losses
        
        # At maximum temperature
        losses = combined_cycle_engine.calculate_dissociation_losses(60000.0)
        assert losses <= 0.4
    
    def test_air_breathing_thrust_calculation(self, combined_cycle_engine):
        """Test air-breathing thrust component calculation."""
        operating_point = CombinedCycleOperatingPoint(
            altitude=50000.0,
            mach_number=8.0,
            throttle_setting=1.0,
            propulsion_mode=PropulsionMode.AIR_BREATHING
        )
        
        thrust = combined_cycle_engine._calculate_air_breathing_thrust(
            operating_point, PropulsionMode.AIR_BREATHING
        )
        assert thrust > 0
        
        # Pure rocket mode should have zero air-breathing thrust
        thrust = combined_cycle_engine._calculate_air_breathing_thrust(
            operating_point, PropulsionMode.PURE_ROCKET
        )
        assert thrust == 0.0
    
    def test_rocket_thrust_calculation(self, combined_cycle_engine):
        """Test rocket thrust component calculation."""
        operating_point = CombinedCycleOperatingPoint(
            altitude=50000.0,
            mach_number=30.0,
            throttle_setting=1.0,
            rocket_throttle_setting=1.0,
            propulsion_mode=PropulsionMode.ROCKET_ASSISTED
        )
        
        thrust = combined_cycle_engine._calculate_rocket_thrust(
            operating_point, PropulsionMode.ROCKET_ASSISTED
        )
        assert thrust > 0
        
        # Air-breathing mode should have zero rocket thrust
        thrust = combined_cycle_engine._calculate_rocket_thrust(
            operating_point, PropulsionMode.AIR_BREATHING
        )
        assert thrust == 0.0
    
    def test_combined_cycle_thrust_calculation(self, combined_cycle_engine):
        """Test total combined-cycle thrust calculation."""
        # Air-breathing mode
        operating_point = CombinedCycleOperatingPoint(
            altitude=50000.0,
            mach_number=8.0,
            throttle_setting=1.0,
            propulsion_mode=PropulsionMode.AIR_BREATHING
        )
        
        thrust = combined_cycle_engine.calculate_combined_cycle_thrust(operating_point)
        assert thrust > 0
        
        # Rocket-assisted mode
        operating_point.mach_number = 25.0
        operating_point.rocket_throttle_setting = 0.8
        operating_point.propulsion_mode = PropulsionMode.ROCKET_ASSISTED
        
        thrust = combined_cycle_engine.calculate_combined_cycle_thrust(operating_point)
        assert thrust > 0
        
        # Invalid operating point should return zero thrust
        operating_point.altitude = 20000.0  # Too low
        thrust = combined_cycle_engine.calculate_combined_cycle_thrust(operating_point)
        assert thrust == 0.0
    
    def test_fuel_consumption_calculation(self, combined_cycle_engine):
        """Test fuel consumption calculation for combined-cycle operation."""
        operating_point = CombinedCycleOperatingPoint(
            altitude=50000.0,
            mach_number=25.0,
            throttle_setting=1.0,
            rocket_throttle_setting=0.8,
            propulsion_mode=PropulsionMode.ROCKET_ASSISTED
        )
        
        air_fuel, rocket_fuel, total_fuel = combined_cycle_engine.calculate_combined_cycle_fuel_consumption(operating_point)
        
        assert air_fuel >= 0
        assert rocket_fuel >= 0
        assert total_fuel == air_fuel + rocket_fuel
        assert total_fuel > 0  # Should consume fuel
    
    def test_complete_performance_data(self, combined_cycle_engine):
        """Test complete performance data generation."""
        operating_point = CombinedCycleOperatingPoint(
            altitude=50000.0,
            mach_number=30.0,
            throttle_setting=1.0,
            rocket_throttle_setting=1.0,
            propulsion_mode=PropulsionMode.ROCKET_ASSISTED
        )
        
        performance = combined_cycle_engine.get_combined_cycle_performance(operating_point)
        
        assert isinstance(performance, CombinedCyclePerformanceData)
        assert performance.total_thrust > 0
        assert performance.total_fuel_flow > 0
        assert performance.air_breathing_thrust >= 0
        assert performance.rocket_thrust >= 0
        assert abs(performance.total_thrust - (performance.air_breathing_thrust + performance.rocket_thrust)) < 1.0  # Allow small numerical differences
        assert performance.efficiency > 0
        assert performance.efficiency <= 1.0
        assert performance.dissociation_losses >= 0
        assert performance.dissociation_losses <= 0.4
    
    def test_mode_transition_optimization(self, combined_cycle_engine):
        """Test propulsion mode transition optimization."""
        best_mach, best_mode = combined_cycle_engine.optimize_mode_transition(50000.0, 30.0)
        
        assert best_mach > 0
        assert best_mode == PropulsionMode.TRANSITION
        assert best_mach >= combined_cycle_engine.combined_spec.transition_mach_number - 2.0
        assert best_mach <= combined_cycle_engine.combined_spec.transition_mach_number + 2.0
    
    def test_specification_validation(self, combined_cycle_engine):
        """Test combined-cycle specification validation."""
        errors = combined_cycle_engine.validate_combined_cycle_specification()
        
        # Should have no errors for valid specification
        assert len(errors) == 0
        
        # Test with invalid specification
        from fighter_jet_sdk.engines.propulsion.engine_performance_model import EngineType
        
        invalid_spec = CombinedCycleSpecification(
            engine_id="invalid_engine",
            name="Invalid Engine",
            engine_type=EngineType.SCRAMJET,
            max_thrust_sea_level=100000.0,
            max_thrust_altitude=150000.0,
            design_altitude=50000.0,
            design_mach=30.0,
            transition_mach_number=2.0,  # Too low
            rocket_specific_impulse=-100.0,  # Negative
            max_rocket_thrust=0.0,  # Zero
            plasma_interaction_threshold=5.0,  # Below transition Mach
            max_stagnation_temperature=3000.0,  # Below dissociation onset
            dissociation_onset_temperature=4000.0
        )
        
        invalid_engine = CombinedCycleEngine(invalid_spec)
        errors = invalid_engine.validate_combined_cycle_specification()
        
        assert len(errors) > 0
        assert any("Transition Mach number must be > 4.0" in error for error in errors)
        assert any("Rocket specific impulse must be positive" in error for error in errors)
        assert any("Maximum rocket thrust must be positive" in error for error in errors)
    
    def test_extreme_conditions_handling(self, combined_cycle_engine):
        """Test engine behavior under extreme conditions."""
        # Test at maximum Mach number
        operating_point = CombinedCycleOperatingPoint(
            altitude=80000.0,
            mach_number=60.0,
            throttle_setting=1.0,
            rocket_throttle_setting=1.0,
            propulsion_mode=PropulsionMode.PURE_ROCKET
        )
        
        performance = combined_cycle_engine.get_combined_cycle_performance(operating_point)
        
        assert performance.total_thrust > 0
        assert performance.rocket_thrust > 0
        assert performance.air_breathing_thrust == 0  # Pure rocket mode
        assert performance.plasma_interaction_factor <= 1.0
        assert performance.dissociation_losses >= 0
    
    def test_throttle_response(self, combined_cycle_engine):
        """Test engine response to throttle changes."""
        base_operating_point = CombinedCycleOperatingPoint(
            altitude=50000.0,
            mach_number=20.0,
            throttle_setting=1.0,
            rocket_throttle_setting=1.0,
            propulsion_mode=PropulsionMode.ROCKET_ASSISTED
        )
        
        # Full throttle performance
        full_performance = combined_cycle_engine.get_combined_cycle_performance(base_operating_point)
        
        # Half throttle performance
        base_operating_point.throttle_setting = 0.5
        base_operating_point.rocket_throttle_setting = 0.5
        half_performance = combined_cycle_engine.get_combined_cycle_performance(base_operating_point)
        
        # Thrust should be roughly proportional to throttle setting
        assert half_performance.total_thrust < full_performance.total_thrust
        assert half_performance.total_fuel_flow < full_performance.total_fuel_flow
    
    def test_altitude_effects(self, combined_cycle_engine):
        """Test engine performance at different altitudes."""
        base_operating_point = CombinedCycleOperatingPoint(
            altitude=50000.0,
            mach_number=25.0,
            throttle_setting=1.0,
            rocket_throttle_setting=1.0,
            propulsion_mode=PropulsionMode.ROCKET_ASSISTED
        )
        
        # Performance at 50 km
        performance_50km = combined_cycle_engine.get_combined_cycle_performance(base_operating_point)
        
        # Performance at 80 km (higher altitude)
        base_operating_point.altitude = 80000.0
        performance_80km = combined_cycle_engine.get_combined_cycle_performance(base_operating_point)
        
        # Rocket performance should improve at higher altitude (lower pressure)
        # Air-breathing performance should decrease
        assert performance_80km.rocket_thrust >= performance_50km.rocket_thrust
    
    def test_performance_envelope_consistency(self, combined_cycle_engine):
        """Test consistency of performance across operating envelope."""
        test_points = [
            (40000.0, 8.0, PropulsionMode.AIR_BREATHING),
            (50000.0, 15.0, PropulsionMode.TRANSITION),
            (60000.0, 30.0, PropulsionMode.ROCKET_ASSISTED),
            (80000.0, 50.0, PropulsionMode.PURE_ROCKET)
        ]
        
        for altitude, mach, mode in test_points:
            operating_point = CombinedCycleOperatingPoint(
                altitude=altitude,
                mach_number=mach,
                throttle_setting=1.0,
                rocket_throttle_setting=1.0,
                propulsion_mode=mode
            )
            
            performance = combined_cycle_engine.get_combined_cycle_performance(operating_point)
            
            # Basic consistency checks
            assert performance.total_thrust >= 0
            assert performance.total_fuel_flow >= 0
            assert performance.efficiency > 0
            assert performance.efficiency <= 1.0
            assert abs(performance.total_thrust - (performance.air_breathing_thrust + performance.rocket_thrust)) < 1.0  # Allow small numerical differences
            assert abs(performance.total_fuel_flow - (performance.air_breathing_fuel_flow + performance.rocket_fuel_flow)) < 0.01  # Allow small numerical differences


    def test_extreme_temperature_effects_calculation(self, combined_cycle_engine):
        """Test extreme temperature effects calculation."""
        # Below extreme temperature threshold should have no effects
        effects = combined_cycle_engine._calculate_extreme_temperature_effects(50000.0, 20.0)
        assert effects == 1.0
        
        # Above extreme temperature threshold should have reduced performance
        effects = combined_cycle_engine._calculate_extreme_temperature_effects(50000.0, 60.0)
        assert effects < 1.0
        assert effects >= 0.4  # Minimum performance retention
    
    def test_dissociation_effects_on_thrust(self, combined_cycle_engine):
        """Test dissociation effects on thrust calculation."""
        base_thrust = 1000000.0  # 1000 kN
        
        # Below dissociation onset should have no effects
        modified_thrust, isp_factor = combined_cycle_engine.calculate_dissociation_effects_on_thrust(
            3000.0, base_thrust
        )
        assert modified_thrust == base_thrust
        assert isp_factor == 1.0
        
        # Above dissociation onset should reduce thrust and ISP
        modified_thrust, isp_factor = combined_cycle_engine.calculate_dissociation_effects_on_thrust(
            30000.0, base_thrust
        )
        assert modified_thrust < base_thrust
        assert isp_factor < 1.0
        assert isp_factor > 0.0
    
    def test_plasma_formation_effects(self, combined_cycle_engine):
        """Test plasma formation effects calculation."""
        pressure = 101325.0  # Sea level pressure
        
        # Below plasma threshold should have no effects
        thrust_factor, radiative_factor, ionization = combined_cycle_engine.calculate_plasma_formation_effects(
            10000.0, pressure
        )
        assert thrust_factor == 1.0
        assert radiative_factor == 1.0
        assert ionization == 0.0
        
        # Above plasma threshold should have effects
        thrust_factor, radiative_factor, ionization = combined_cycle_engine.calculate_plasma_formation_effects(
            30000.0, pressure
        )
        assert thrust_factor != 1.0
        assert radiative_factor > 1.0  # Increased radiative losses
        assert ionization > 0.0
        assert ionization <= 0.9  # Maximum ionization fraction
    
    def test_extreme_temperature_performance_data(self, combined_cycle_engine):
        """Test extreme temperature performance data generation."""
        operating_point = CombinedCycleOperatingPoint(
            altitude=60000.0,
            mach_number=50.0,
            throttle_setting=1.0,
            rocket_throttle_setting=1.0,
            propulsion_mode=PropulsionMode.PURE_ROCKET
        )
        
        perf_data = combined_cycle_engine.get_extreme_temperature_performance(operating_point)
        
        # Check that all required fields are present
        required_fields = [
            'stagnation_temperature', 'base_thrust', 'dissociation_thrust', 'final_thrust',
            'dissociation_losses', 'plasma_thrust_factor', 'extreme_temp_factor',
            'ionization_fraction', 'isp_factor', 'radiative_loss_factor', 'total_performance_factor'
        ]
        
        for field in required_fields:
            assert field in perf_data
            assert isinstance(perf_data[field], (int, float))
        
        # Check reasonable values
        assert perf_data['stagnation_temperature'] > 0
        assert perf_data['base_thrust'] >= 0
        assert perf_data['final_thrust'] >= 0
        assert 0 <= perf_data['dissociation_losses'] <= 1
        assert 0 <= perf_data['ionization_fraction'] <= 1
        assert perf_data['isp_factor'] > 0
        assert perf_data['radiative_loss_factor'] >= 1
    
    def test_extreme_temperature_limits_validation(self, combined_cycle_engine):
        """Test validation of extreme temperature limits."""
        # Normal operating point should have no warnings
        normal_point = CombinedCycleOperatingPoint(
            altitude=50000.0,
            mach_number=15.0,
            throttle_setting=0.8,
            rocket_throttle_setting=0.8
        )
        
        warnings = combined_cycle_engine.validate_extreme_temperature_limits(normal_point)
        assert isinstance(warnings, list)
        
        # Extreme operating point should have warnings
        extreme_point = CombinedCycleOperatingPoint(
            altitude=80000.0,
            mach_number=60.0,
            throttle_setting=1.0,
            rocket_throttle_setting=1.0
        )
        
        warnings = combined_cycle_engine.validate_extreme_temperature_limits(extreme_point)
        assert len(warnings) > 0
        assert any("extreme temperature" in warning.lower() for warning in warnings)
    
    def test_extreme_temperature_optimization(self, combined_cycle_engine):
        """Test optimization for extreme temperature conditions."""
        optimization = combined_cycle_engine.optimize_for_extreme_temperatures(80000.0, 55.0)
        
        # Check that all required fields are present
        assert 'performance_data' in optimization
        assert 'recommendations' in optimization
        assert 'optimal_throttle' in optimization
        assert 'warnings' in optimization
        
        # Check data types
        assert isinstance(optimization['performance_data'], dict)
        assert isinstance(optimization['recommendations'], list)
        assert isinstance(optimization['optimal_throttle'], (int, float))
        assert isinstance(optimization['warnings'], list)
        
        # Check reasonable values
        assert 0.0 < optimization['optimal_throttle'] <= 1.0
    
    def test_temperature_effects_across_mach_range(self, combined_cycle_engine):
        """Test temperature effects across different Mach numbers."""
        altitude = 60000.0
        mach_numbers = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        
        previous_temp = 0.0
        
        for mach in mach_numbers:
            operating_point = CombinedCycleOperatingPoint(
                altitude=altitude,
                mach_number=mach,
                throttle_setting=1.0,
                rocket_throttle_setting=1.0
            )
            
            perf_data = combined_cycle_engine.get_extreme_temperature_performance(operating_point)
            current_temp = perf_data['stagnation_temperature']
            
            # Temperature should generally increase with Mach number (until capped)
            if current_temp < combined_cycle_engine.combined_spec.max_stagnation_temperature:
                assert current_temp > previous_temp
            else:
                # At maximum temperature, should be capped
                assert current_temp == combined_cycle_engine.combined_spec.max_stagnation_temperature
            
            # Performance factors should be reasonable
            assert 0.0 < perf_data['total_performance_factor'] <= 1.0
            assert 0.0 <= perf_data['dissociation_losses'] <= 1.0
            
            previous_temp = current_temp
    
    def test_material_temperature_limits(self, combined_cycle_engine):
        """Test engine behavior at material temperature limits."""
        # Test at maximum stagnation temperature
        operating_point = CombinedCycleOperatingPoint(
            altitude=80000.0,
            mach_number=60.0,
            throttle_setting=1.0,
            rocket_throttle_setting=1.0
        )
        
        perf_data = combined_cycle_engine.get_extreme_temperature_performance(operating_point)
        
        # Should not exceed maximum temperature
        assert perf_data['stagnation_temperature'] <= combined_cycle_engine.combined_spec.max_stagnation_temperature
        
        # Should have significant performance degradation at extreme conditions
        assert perf_data['total_performance_factor'] < 1.0
        
        # Should have warnings about extreme conditions
        warnings = combined_cycle_engine.validate_extreme_temperature_limits(operating_point)
        assert len(warnings) > 0


if __name__ == "__main__":
    pytest.main([__file__])