"""Tests for hypersonic mission planner."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from fighter_jet_sdk.core.hypersonic_mission_planner import (
    HypersonicMissionPlanner, TrajectoryOptimizer, AltitudeOptimizer, FuelOptimizer,
    HypersonicMissionProfile, HypersonicWaypoint, ThermalConstraint, PropulsionConstraint,
    HypersonicFlightPhase, OptimizationObjective
)
from fighter_jet_sdk.engines.propulsion.combined_cycle_engine import (
    CombinedCycleEngine, CombinedCycleSpecification, PropulsionMode
)
from fighter_jet_sdk.common.enums import ExtremePropulsionType


class TestAltitudeOptimizer:
    """Test altitude optimization for hypersonic flight."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = AltitudeOptimizer()
    
    def test_atmospheric_conditions_troposphere(self):
        """Test atmospheric conditions calculation in troposphere."""
        temp, pressure, density = self.optimizer.get_atmospheric_conditions(5000.0)
        
        assert temp > 0
        assert pressure > 0
        assert density > 0
        assert temp < 288.15  # Should be cooler than sea level
        assert pressure < 101325.0  # Should be lower than sea level
    
    def test_atmospheric_conditions_stratosphere(self):
        """Test atmospheric conditions calculation in stratosphere."""
        temp, pressure, density = self.optimizer.get_atmospheric_conditions(15000.0)
        
        assert temp > 0
        assert pressure > 0
        assert density > 0
        assert temp == 216.65  # Isothermal stratosphere
    
    def test_atmospheric_conditions_extreme_altitude(self):
        """Test atmospheric conditions at extreme altitudes."""
        temp, pressure, density = self.optimizer.get_atmospheric_conditions(80000.0)
        
        assert temp > 0
        assert pressure > 0
        assert density > 0
        assert pressure < 1000.0  # Very low pressure at high altitude
    
    def test_dynamic_pressure_calculation(self):
        """Test dynamic pressure calculation."""
        q = self.optimizer.calculate_dynamic_pressure(50000.0, 30.0)
        
        assert q > 0
        assert q < 1e6  # Reasonable dynamic pressure range
    
    def test_minimum_altitude_for_mach(self):
        """Test minimum altitude requirements for different Mach numbers."""
        # Low Mach
        min_alt_low = self.optimizer._get_minimum_altitude_for_mach(3.0)
        assert min_alt_low == 10000.0
        
        # High Mach
        min_alt_high = self.optimizer._get_minimum_altitude_for_mach(50.0)
        assert min_alt_high == 55000.0
        assert min_alt_high > min_alt_low
    
    def test_altitude_optimization_for_mach_range(self):
        """Test altitude profile optimization for Mach range."""
        constraints = {'max_altitude': 80000.0}
        
        altitude_profile = self.optimizer.optimize_altitude_for_mach_range(
            10.0, 40.0, 1000000.0, constraints
        )
        
        assert len(altitude_profile) > 0
        assert np.all(altitude_profile > 0)
        assert np.all(altitude_profile <= 80000.0)
        
        # Should generally increase with Mach number
        assert altitude_profile[-1] >= altitude_profile[0]


class TestTrajectoryOptimizer:
    """Test trajectory optimization for hypersonic missions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock combined cycle engine
        engine_spec = CombinedCycleSpecification(
            engine_id="test_hypersonic_engine",
            name="Test Hypersonic Engine",
            engine_type=None,  # Will be set in __post_init__
            max_thrust_sea_level=500000.0,
            max_thrust_altitude=400000.0,
            design_altitude=60000.0,
            design_mach=30.0,
            extreme_propulsion_type=ExtremePropulsionType.COMBINED_CYCLE_AIRBREATHING,
            transition_mach_number=12.0,
            max_rocket_thrust=300000.0
        )
        self.engine = CombinedCycleEngine(engine_spec)
        self.optimizer = TrajectoryOptimizer(self.engine)
    
    def test_distance_calculation(self):
        """Test distance calculation between points."""
        start = (0.0, 0.0, 0.0)
        end = (100000.0, 0.0, 0.0)
        
        distance = self.optimizer._calculate_distance(start, end)
        assert distance == 100000.0
    
    def test_trajectory_segments_creation(self):
        """Test creation of trajectory segments."""
        start = (0.0, 0.0, 15000.0)
        end = (1000000.0, 0.0, 15000.0)
        
        segments = self.optimizer._create_trajectory_segments(start, end, 40.0, 1000000.0)
        
        assert len(segments) == 3  # Acceleration, cruise, descent
        assert segments[0]['type'] == HypersonicFlightPhase.ACCELERATION
        assert segments[1]['type'] == HypersonicFlightPhase.CRUISE
        assert segments[2]['type'] == HypersonicFlightPhase.DESCENT
    
    def test_propulsion_mode_determination(self):
        """Test optimal propulsion mode determination."""
        constraints = PropulsionConstraint(max_air_breathing_mach=15.0)
        
        # Low Mach - air breathing
        mode_low = self.optimizer._determine_optimal_propulsion_mode(5.0, 30000.0, constraints)
        assert mode_low == PropulsionMode.AIR_BREATHING
        
        # High Mach - rocket
        mode_high = self.optimizer._determine_optimal_propulsion_mode(50.0, 70000.0, constraints)
        assert mode_high == PropulsionMode.PURE_ROCKET
    
    def test_thermal_load_calculation(self):
        """Test thermal load calculation."""
        thermal_load = self.optimizer._calculate_thermal_load(30.0, 50000.0)
        
        assert thermal_load > 0
        assert thermal_load < 1e9  # Reasonable thermal load range
        
        # Higher Mach should give higher thermal load
        thermal_load_high = self.optimizer._calculate_thermal_load(50.0, 50000.0)
        assert thermal_load_high > thermal_load
    
    def test_complete_trajectory_optimization(self):
        """Test complete trajectory optimization."""
        start_point = (0.0, 0.0, 15000.0)
        end_point = (2000000.0, 0.0, 15000.0)
        
        thermal_constraints = ThermalConstraint(
            max_heat_flux=100e6,  # 100 MW/m²
            max_temperature=3000.0,
            max_duration_at_peak=300.0,
            cooling_system_capacity=50e6,
            recovery_time_required=120.0
        )
        
        propulsion_constraints = PropulsionConstraint(
            max_air_breathing_mach=15.0,
            min_rocket_altitude=40000.0,
            fuel_capacity_air_breathing=10000.0,
            fuel_capacity_rocket=5000.0
        )
        
        profile = self.optimizer.optimize_trajectory(
            start_point, end_point, 40.0, thermal_constraints, 
            propulsion_constraints, OptimizationObjective.MINIMIZE_FUEL
        )
        
        assert isinstance(profile, HypersonicMissionProfile)
        assert len(profile.waypoints) > 0
        assert len(profile.altitude_profile) > 0
        assert len(profile.mach_profile) > 0
        assert profile.total_duration > 0
        assert profile.total_fuel_consumption > 0


class TestFuelOptimizer:
    """Test fuel consumption optimization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        engine_spec = CombinedCycleSpecification(
            engine_id="test_fuel_optimizer_engine",
            name="Test Hypersonic Engine",
            engine_type=None,  # Will be set in __post_init__
            max_thrust_sea_level=500000.0,
            max_thrust_altitude=400000.0,
            design_altitude=60000.0,
            design_mach=30.0,
            extreme_propulsion_type=ExtremePropulsionType.COMBINED_CYCLE_AIRBREATHING,
            transition_mach_number=12.0,
            max_rocket_thrust=300000.0
        )
        self.engine = CombinedCycleEngine(engine_spec)
        self.optimizer = FuelOptimizer(self.engine)
    
    def test_propulsion_mode_validation(self):
        """Test propulsion mode validation."""
        constraints = PropulsionConstraint(
            max_air_breathing_mach=15.0,
            min_rocket_altitude=40000.0
        )
        
        # Valid air breathing
        valid_ab = self.optimizer._is_valid_propulsion_mode(
            10.0, 30000.0, PropulsionMode.AIR_BREATHING, constraints
        )
        assert valid_ab
        
        # Invalid air breathing (too high Mach)
        invalid_ab = self.optimizer._is_valid_propulsion_mode(
            20.0, 30000.0, PropulsionMode.AIR_BREATHING, constraints
        )
        assert not invalid_ab
        
        # Invalid rocket (too low altitude)
        invalid_rocket = self.optimizer._is_valid_propulsion_mode(
            30.0, 30000.0, PropulsionMode.PURE_ROCKET, constraints
        )
        assert not invalid_rocket
    
    def test_fuel_optimization(self):
        """Test fuel consumption optimization."""
        # Create test profile
        waypoints = [
            HypersonicWaypoint(
                waypoint_id="test_wp_1",
                position=np.array([0.0, 0.0, 50000.0]),
                altitude=50000.0,
                mach_number=20.0,
                heading=0.0,
                flight_phase=HypersonicFlightPhase.CRUISE,
                propulsion_mode=PropulsionMode.AIR_BREATHING,
                thermal_load=50e6,
                fuel_consumption_rate=20.0
            )
        ]
        
        profile = HypersonicMissionProfile(
            profile_id="test_profile",
            name="Test Profile",
            waypoints=waypoints
        )
        
        constraints = PropulsionConstraint(
            max_air_breathing_mach=25.0,
            min_rocket_altitude=40000.0
        )
        
        optimized_profile = self.optimizer.optimize_fuel_consumption(profile, constraints)
        
        assert isinstance(optimized_profile, HypersonicMissionProfile)
        assert len(optimized_profile.waypoints) == len(waypoints)


class TestHypersonicMissionPlanner:
    """Test complete hypersonic mission planning."""
    
    def setup_method(self):
        """Set up test fixtures."""
        engine_spec = CombinedCycleSpecification(
            engine_id="test_mission_planner_engine",
            name="Test Hypersonic Engine",
            engine_type=None,  # Will be set in __post_init__
            max_thrust_sea_level=500000.0,
            max_thrust_altitude=400000.0,
            design_altitude=60000.0,
            design_mach=30.0,
            extreme_propulsion_type=ExtremePropulsionType.COMBINED_CYCLE_AIRBREATHING,
            transition_mach_number=12.0,
            max_rocket_thrust=300000.0,
            rocket_specific_impulse=450.0,
            air_breathing_specific_impulse=3000.0
        )
        self.engine = CombinedCycleEngine(engine_spec)
        self.planner = HypersonicMissionPlanner(self.engine)
    
    def test_mission_planning_initialization(self):
        """Test mission planner initialization."""
        assert self.planner.engine is not None
        assert self.planner.trajectory_optimizer is not None
        assert self.planner.fuel_optimizer is not None
    
    def test_mission_planning_validation(self):
        """Test mission planning input validation."""
        start_point = (0.0, 0.0, 15000.0)
        end_point = (1000000.0, 0.0, 15000.0)
        
        thermal_constraints = ThermalConstraint(
            max_heat_flux=100e6,
            max_temperature=3000.0,
            max_duration_at_peak=300.0,
            cooling_system_capacity=50e6,
            recovery_time_required=120.0
        )
        
        propulsion_constraints = PropulsionConstraint()
        
        # Invalid Mach number - too low
        with pytest.raises(Exception):
            self.planner.plan_mission(
                start_point, end_point, 2.0, thermal_constraints, 
                propulsion_constraints, [OptimizationObjective.MINIMIZE_FUEL]
            )
        
        # Invalid Mach number - too high
        with pytest.raises(Exception):
            self.planner.plan_mission(
                start_point, end_point, 70.0, thermal_constraints, 
                propulsion_constraints, [OptimizationObjective.MINIMIZE_FUEL]
            )
    
    def test_complete_mission_planning(self):
        """Test complete mission planning process."""
        start_point = (0.0, 0.0, 15000.0)
        end_point = (1500000.0, 0.0, 15000.0)
        
        thermal_constraints = ThermalConstraint(
            max_heat_flux=100e6,  # 100 MW/m²
            max_temperature=3000.0,
            max_duration_at_peak=300.0,
            cooling_system_capacity=50e6,
            recovery_time_required=120.0
        )
        
        propulsion_constraints = PropulsionConstraint(
            max_air_breathing_mach=15.0,
            min_rocket_altitude=40000.0,
            fuel_capacity_air_breathing=12000.0,
            fuel_capacity_rocket=8000.0
        )
        
        objectives = [
            OptimizationObjective.MINIMIZE_FUEL,
            OptimizationObjective.MINIMIZE_THERMAL_LOAD
        ]
        
        profile = self.planner.plan_mission(
            start_point, end_point, 30.0, thermal_constraints, 
            propulsion_constraints, objectives
        )
        
        # Validate mission profile
        assert isinstance(profile, HypersonicMissionProfile)
        assert profile.profile_id is not None
        assert len(profile.waypoints) > 0
        assert len(profile.altitude_profile) > 0
        assert len(profile.mach_profile) > 0
        assert profile.total_duration > 0
        assert profile.total_fuel_consumption > 0
        
        # Check that waypoints have reasonable values
        for waypoint in profile.waypoints:
            assert waypoint.altitude > 0
            assert waypoint.mach_number >= 4.0
            assert waypoint.mach_number <= 30.0
            assert waypoint.thermal_load >= 0
    
    def test_thermal_management_optimization(self):
        """Test thermal management optimization."""
        # Create profile with high thermal loads
        waypoints = [
            HypersonicWaypoint(
                waypoint_id="high_thermal_1",
                position=np.array([100000.0, 0.0, 60000.0]),
                altitude=60000.0,
                mach_number=40.0,
                heading=0.0,
                flight_phase=HypersonicFlightPhase.CRUISE,
                thermal_load=120e6  # Above threshold
            ),
            HypersonicWaypoint(
                waypoint_id="high_thermal_2",
                position=np.array([200000.0, 0.0, 60000.0]),
                altitude=60000.0,
                mach_number=40.0,
                heading=0.0,
                flight_phase=HypersonicFlightPhase.CRUISE,
                thermal_load=110e6  # Above threshold
            )
        ]
        
        profile = HypersonicMissionProfile(
            profile_id="thermal_test",
            name="Thermal Test Profile",
            waypoints=waypoints
        )
        
        thermal_constraints = ThermalConstraint(
            max_heat_flux=100e6,
            max_temperature=3000.0,
            max_duration_at_peak=300.0,
            cooling_system_capacity=50e6,
            recovery_time_required=120.0
        )
        
        optimized_profile = self.planner._optimize_for_thermal_management(profile, thermal_constraints)
        
        # Should have added thermal recovery waypoints
        assert len(optimized_profile.waypoints) > len(waypoints)
        
        # Check for thermal recovery phases
        recovery_waypoints = [wp for wp in optimized_profile.waypoints 
                            if wp.flight_phase == HypersonicFlightPhase.THERMAL_RECOVERY]
        assert len(recovery_waypoints) > 0
    
    def test_minimum_time_optimization(self):
        """Test minimum time optimization."""
        waypoints = [
            HypersonicWaypoint(
                waypoint_id="cruise_1",
                position=np.array([100000.0, 0.0, 60000.0]),
                altitude=60000.0,
                mach_number=20.0,
                heading=0.0,
                flight_phase=HypersonicFlightPhase.CRUISE
            )
        ]
        
        profile = HypersonicMissionProfile(
            profile_id="time_test",
            name="Time Test Profile",
            waypoints=waypoints
        )
        
        propulsion_constraints = PropulsionConstraint(max_air_breathing_mach=25.0)
        
        optimized_profile = self.planner._optimize_for_minimum_time(profile, propulsion_constraints)
        
        # Should have increased Mach numbers for cruise segments
        cruise_waypoints = [wp for wp in optimized_profile.waypoints 
                          if wp.flight_phase == HypersonicFlightPhase.CRUISE]
        
        if cruise_waypoints:
            assert cruise_waypoints[0].mach_number >= 20.0  # Should be same or higher
    
    def test_mission_feasibility_analysis(self):
        """Test mission feasibility analysis."""
        waypoints = [
            HypersonicWaypoint(
                waypoint_id="feasible_wp",
                position=np.array([100000.0, 0.0, 60000.0]),
                altitude=60000.0,
                mach_number=30.0,
                heading=0.0,
                flight_phase=HypersonicFlightPhase.CRUISE,
                thermal_load=80e6  # Within limits
            )
        ]
        
        profile = HypersonicMissionProfile(
            profile_id="feasibility_test",
            name="Feasibility Test",
            waypoints=waypoints,
            total_fuel_consumption=8000.0,  # Within limits
            thermal_load_profile=np.array([80e6])
        )
        
        thermal_constraints = ThermalConstraint(
            max_heat_flux=100e6,
            max_temperature=3000.0,
            max_duration_at_peak=300.0,
            cooling_system_capacity=50e6,
            recovery_time_required=120.0
        )
        
        propulsion_constraints = PropulsionConstraint(
            fuel_capacity_air_breathing=10000.0,
            fuel_capacity_rocket=5000.0
        )
        
        profile.thermal_constraints = thermal_constraints
        profile.propulsion_constraints = propulsion_constraints
        
        analysis = self.planner.analyze_mission_feasibility(profile)
        
        assert 'feasible' in analysis
        assert 'critical_issues' in analysis
        assert 'warnings' in analysis
        assert 'performance_metrics' in analysis
        assert 'recommendations' in analysis
        
        # Should be feasible with reasonable constraints
        assert analysis['feasible'] is True
    
    def test_infeasible_mission_analysis(self):
        """Test analysis of infeasible mission."""
        waypoints = [
            HypersonicWaypoint(
                waypoint_id="infeasible_wp",
                position=np.array([100000.0, 0.0, 60000.0]),
                altitude=60000.0,
                mach_number=50.0,
                heading=0.0,
                flight_phase=HypersonicFlightPhase.CRUISE,
                thermal_load=150e6  # Exceeds limits
            )
        ]
        
        profile = HypersonicMissionProfile(
            profile_id="infeasible_test",
            name="Infeasible Test",
            waypoints=waypoints,
            total_fuel_consumption=20000.0,  # Exceeds capacity
            thermal_load_profile=np.array([150e6])
        )
        
        thermal_constraints = ThermalConstraint(
            max_heat_flux=100e6,  # Lower than thermal load
            max_temperature=3000.0,
            max_duration_at_peak=300.0,
            cooling_system_capacity=50e6,
            recovery_time_required=120.0
        )
        
        propulsion_constraints = PropulsionConstraint(
            fuel_capacity_air_breathing=10000.0,
            fuel_capacity_rocket=5000.0  # Total 15000, less than consumption
        )
        
        profile.thermal_constraints = thermal_constraints
        profile.propulsion_constraints = propulsion_constraints
        
        analysis = self.planner.analyze_mission_feasibility(profile)
        
        # Should be infeasible
        assert analysis['feasible'] is False
        assert len(analysis['critical_issues']) > 0


class TestHypersonicMissionProfile:
    """Test hypersonic mission profile data structure."""
    
    def test_profile_validation_empty(self):
        """Test validation of empty profile."""
        profile = HypersonicMissionProfile(
            profile_id="empty_test",
            name="Empty Profile"
        )
        
        errors = profile.validate_profile()
        assert len(errors) > 0
        assert "waypoints" in errors[0].lower()
    
    def test_profile_validation_altitude_mach_correlation(self):
        """Test validation of altitude-Mach correlation."""
        waypoints = [
            HypersonicWaypoint(
                waypoint_id="invalid_wp",
                position=np.array([0.0, 0.0, 30000.0]),
                altitude=30000.0,  # Too low for Mach 60
                mach_number=60.0,
                heading=0.0,
                flight_phase=HypersonicFlightPhase.CRUISE
            )
        ]
        
        profile = HypersonicMissionProfile(
            profile_id="validation_test",
            name="Validation Test",
            waypoints=waypoints
        )
        
        errors = profile.validate_profile()
        assert len(errors) > 0
        assert "mach 60" in errors[0].lower()
    
    def test_profile_validation_thermal_constraints(self):
        """Test validation against thermal constraints."""
        waypoints = [
            HypersonicWaypoint(
                waypoint_id="thermal_wp",
                position=np.array([0.0, 0.0, 60000.0]),
                altitude=60000.0,
                mach_number=40.0,
                heading=0.0,
                flight_phase=HypersonicFlightPhase.CRUISE,
                thermal_load=150e6  # Exceeds constraint
            )
        ]
        
        thermal_constraints = ThermalConstraint(
            max_heat_flux=100e6,
            max_temperature=3000.0,
            max_duration_at_peak=300.0,
            cooling_system_capacity=50e6,
            recovery_time_required=120.0
        )
        
        profile = HypersonicMissionProfile(
            profile_id="thermal_validation_test",
            name="Thermal Validation Test",
            waypoints=waypoints,
            thermal_constraints=thermal_constraints
        )
        
        errors = profile.validate_profile()
        assert len(errors) > 0
        assert "thermal load" in errors[0].lower()
    
    def test_profile_validation_propulsion_transitions(self):
        """Test validation of propulsion mode transitions."""
        waypoints = [
            HypersonicWaypoint(
                waypoint_id="transition_wp_1",
                position=np.array([0.0, 0.0, 60000.0]),
                altitude=60000.0,
                mach_number=10.0,
                heading=0.0,
                flight_phase=HypersonicFlightPhase.CRUISE,
                propulsion_mode=PropulsionMode.AIR_BREATHING
            ),
            HypersonicWaypoint(
                waypoint_id="transition_wp_2",
                position=np.array([100000.0, 0.0, 60000.0]),
                altitude=60000.0,
                mach_number=50.0,  # Abrupt transition
                heading=0.0,
                flight_phase=HypersonicFlightPhase.CRUISE,
                propulsion_mode=PropulsionMode.PURE_ROCKET
            )
        ]
        
        profile = HypersonicMissionProfile(
            profile_id="transition_validation_test",
            name="Transition Validation Test",
            waypoints=waypoints
        )
        
        errors = profile.validate_profile()
        assert len(errors) > 0
        assert "transition" in errors[0].lower()


if __name__ == "__main__":
    pytest.main([__file__])