"""Tests for thermal constraint integration in hypersonic missions."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from fighter_jet_sdk.core.thermal_constraint_manager import (
    ThermalConstraintManager, TrajectoryThermalModifier, ThermalState, ThermalStatus,
    CoolingSystemMode, CoolingSystemPerformance, ThermalEvent,
    integrate_thermal_constraints_with_mission_planner
)
from fighter_jet_sdk.core.hypersonic_mission_planner import (
    HypersonicMissionPlanner, HypersonicMissionProfile, HypersonicWaypoint,
    ThermalConstraint, HypersonicFlightPhase
)
from fighter_jet_sdk.engines.propulsion.combined_cycle_engine import (
    CombinedCycleEngine, CombinedCycleSpecification, PropulsionMode
)
from fighter_jet_sdk.common.enums import ExtremePropulsionType


class TestThermalConstraintManager:
    """Test thermal constraint management functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.thermal_constraints = ThermalConstraint(
            max_heat_flux=100e6,  # 100 MW/m²
            max_temperature=3000.0,  # K
            max_duration_at_peak=300.0,  # seconds
            cooling_system_capacity=50e6,  # W
            recovery_time_required=120.0  # seconds
        )
        
        self.cooling_performance = CoolingSystemPerformance(
            max_cooling_capacity=60e6,  # W
            power_consumption=5000.0,  # W
            coolant_flow_rate=10.0,  # kg/s
            effectiveness=0.8,
            response_time=5.0,  # seconds
            operating_altitude_range=(20000.0, 100000.0),  # m
            operating_mach_range=(5.0, 60.0)
        )
        
        self.thermal_manager = ThermalConstraintManager(
            self.thermal_constraints, self.cooling_performance
        )
    
    def test_thermal_manager_initialization(self):
        """Test thermal constraint manager initialization."""
        assert self.thermal_manager.thermal_constraints == self.thermal_constraints
        assert self.thermal_manager.cooling_performance == self.cooling_performance
        assert isinstance(self.thermal_manager.current_thermal_state, ThermalState)
        assert len(self.thermal_manager.thermal_events) == 0
    
    def test_surface_temperature_calculation(self):
        """Test surface temperature calculation."""
        # Low heat flux
        temp_low = self.thermal_manager._calculate_surface_temperature(1e6, 50000.0, 10.0)
        assert temp_low > 300.0  # Above ambient
        assert temp_low < 5000.0  # Reasonable for hypersonic conditions
        
        # High heat flux
        temp_high = self.thermal_manager._calculate_surface_temperature(100e6, 50000.0, 40.0)
        assert temp_high > temp_low
        assert temp_high <= 4000.0  # Material limit
    
    def test_thermal_stress_calculation(self):
        """Test thermal stress level calculation."""
        # Normal conditions
        stress_normal = self.thermal_manager._calculate_thermal_stress_level(
            50e6, 2000.0, 100.0
        )
        assert 0.0 <= stress_normal <= 1.0
        assert stress_normal < 0.8  # Should be manageable
        
        # Critical conditions
        stress_critical = self.thermal_manager._calculate_thermal_stress_level(
            120e6, 3500.0, 400.0
        )
        assert stress_critical > stress_normal
        assert stress_critical > 0.8  # Should be high stress
    
    def test_material_degradation_calculation(self):
        """Test material degradation calculation."""
        # Start with clean state
        self.thermal_manager.current_thermal_state.material_degradation = 0.0
        
        # Low temperature - minimal degradation
        degradation_low = self.thermal_manager._calculate_material_degradation(
            800.0, 1e10, 0.3
        )
        assert degradation_low < 0.1
        
        # High temperature - significant degradation
        degradation_high = self.thermal_manager._calculate_material_degradation(
            2500.0, 1e11, 0.9
        )
        assert degradation_high > degradation_low
    
    def test_thermal_state_update(self):
        """Test thermal state update."""
        waypoint = HypersonicWaypoint(
            waypoint_id="test_wp",
            position=np.array([100000.0, 0.0, 60000.0]),
            altitude=60000.0,
            mach_number=30.0,
            heading=0.0,
            flight_phase=HypersonicFlightPhase.CRUISE,
            thermal_load=80e6
        )
        
        thermal_state = self.thermal_manager.update_thermal_state(waypoint, 10.0, 100.0)
        
        assert thermal_state.heat_flux == waypoint.thermal_load
        assert thermal_state.surface_temperature > 300.0
        assert thermal_state.thermal_load_integral > 0
        assert 0.0 <= thermal_state.thermal_stress_level <= 1.0
    
    def test_thermal_status_determination(self):
        """Test thermal status determination."""
        # Normal state
        normal_state = ThermalState(
            surface_temperature=1500.0,
            heat_flux=40e6,
            thermal_load_integral=1e10,
            time_at_current_load=100.0
        )
        status_normal = self.thermal_manager.get_thermal_status(normal_state)
        assert status_normal == ThermalStatus.NORMAL
        
        # Critical state
        critical_state = ThermalState(
            surface_temperature=2900.0,
            heat_flux=95e6,
            thermal_load_integral=1e12,
            time_at_current_load=350.0
        )
        status_critical = self.thermal_manager.get_thermal_status(critical_state)
        assert status_critical in [ThermalStatus.CRITICAL, ThermalStatus.EMERGENCY]
    
    def test_cooling_mode_determination(self):
        """Test cooling system mode determination."""
        # Normal status - minimal cooling
        normal_state = ThermalState(surface_temperature=1000.0, heat_flux=20e6, thermal_load_integral=1e9)
        cooling_normal = self.thermal_manager.determine_required_cooling_mode(
            ThermalStatus.NORMAL, normal_state
        )
        assert cooling_normal in [CoolingSystemMode.OFF, CoolingSystemMode.PASSIVE]
        
        # Critical status - active cooling
        critical_state = ThermalState(surface_temperature=2800.0, heat_flux=95e6, thermal_load_integral=1e12)
        cooling_critical = self.thermal_manager.determine_required_cooling_mode(
            ThermalStatus.CRITICAL, critical_state
        )
        assert cooling_critical == CoolingSystemMode.ACTIVE_HIGH
    
    def test_cooling_system_activation(self):
        """Test cooling system activation."""
        waypoint = HypersonicWaypoint(
            waypoint_id="cooling_test_wp",
            position=np.array([100000.0, 0.0, 60000.0]),
            altitude=60000.0,
            mach_number=25.0,
            heading=0.0,
            flight_phase=HypersonicFlightPhase.CRUISE
        )
        
        success, message = self.thermal_manager.activate_cooling_system(
            CoolingSystemMode.ACTIVE_MEDIUM, waypoint, 200.0
        )
        
        assert success
        assert "activated" in message.lower()
        assert self.thermal_manager.current_thermal_state.cooling_system_mode == CoolingSystemMode.ACTIVE_MEDIUM
        assert len(self.thermal_manager.thermal_events) == 1
    
    def test_cooling_system_activation_out_of_range(self):
        """Test cooling system activation outside operating range."""
        # Altitude too low
        waypoint_low_alt = HypersonicWaypoint(
            waypoint_id="low_alt_wp",
            position=np.array([100000.0, 0.0, 10000.0]),
            altitude=10000.0,  # Below operating range
            mach_number=25.0,
            heading=0.0,
            flight_phase=HypersonicFlightPhase.CRUISE
        )
        
        success, message = self.thermal_manager.activate_cooling_system(
            CoolingSystemMode.ACTIVE_MEDIUM, waypoint_low_alt, 200.0
        )
        
        assert not success
        assert "cannot operate" in message.lower()
    
    def test_cooling_effectiveness_calculation(self):
        """Test cooling effectiveness calculation."""
        # Normal heat flux
        effectiveness_normal = self.thermal_manager.calculate_cooling_effectiveness(
            CoolingSystemMode.ACTIVE_MEDIUM, 40e6
        )
        assert 0.0 < effectiveness_normal <= 1.0
        
        # Heat flux exceeding cooling capacity
        effectiveness_overwhelmed = self.thermal_manager.calculate_cooling_effectiveness(
            CoolingSystemMode.ACTIVE_HIGH, 100e6
        )
        assert effectiveness_overwhelmed < effectiveness_normal
    
    def test_cooling_effects_application(self):
        """Test application of cooling effects."""
        initial_state = ThermalState(
            surface_temperature=2500.0,
            heat_flux=80e6,
            thermal_load_integral=1e11,
            thermal_stress_level=0.8
        )
        
        cooled_state = self.thermal_manager.apply_cooling_effects(initial_state, 0.6)
        
        assert cooled_state.heat_flux < initial_state.heat_flux
        assert cooled_state.surface_temperature < initial_state.surface_temperature
        assert cooled_state.thermal_stress_level < initial_state.thermal_stress_level
    
    def test_thermal_recovery_need_detection(self):
        """Test thermal recovery need detection."""
        # State requiring recovery
        recovery_needed_state = ThermalState(
            surface_temperature=2800.0,
            heat_flux=90e6,
            thermal_load_integral=1e12,
            thermal_stress_level=0.9,
            time_at_current_load=400.0,
            time_since_last_recovery=2000.0,
            material_degradation=0.15
        )
        
        needs_recovery = self.thermal_manager.check_thermal_recovery_needed(recovery_needed_state)
        assert needs_recovery
        
        # Normal state
        normal_state = ThermalState(
            surface_temperature=1500.0,
            heat_flux=40e6,
            thermal_load_integral=1e10,
            thermal_stress_level=0.4,
            time_at_current_load=100.0,
            time_since_last_recovery=600.0,
            material_degradation=0.02
        )
        
        no_recovery_needed = self.thermal_manager.check_thermal_recovery_needed(normal_state)
        assert not no_recovery_needed
    
    def test_thermal_recovery_waypoint_generation(self):
        """Test thermal recovery waypoint generation."""
        current_waypoint = HypersonicWaypoint(
            waypoint_id="high_thermal_wp",
            position=np.array([200000.0, 0.0, 50000.0]),
            altitude=50000.0,
            mach_number=40.0,
            heading=0.0,
            flight_phase=HypersonicFlightPhase.CRUISE,
            thermal_load=100e6
        )
        
        recovery_wp = self.thermal_manager.generate_thermal_recovery_waypoint(
            current_waypoint, 300.0
        )
        
        assert recovery_wp.altitude > current_waypoint.altitude
        assert recovery_wp.mach_number < current_waypoint.mach_number
        assert recovery_wp.thermal_load < current_waypoint.thermal_load
        assert recovery_wp.flight_phase == HypersonicFlightPhase.THERMAL_RECOVERY
        assert recovery_wp.cooling_system_active
        assert recovery_wp.duration == self.thermal_constraints.recovery_time_required
    
    def test_mission_thermal_monitoring(self):
        """Test thermal monitoring throughout mission profile."""
        waypoints = [
            HypersonicWaypoint(
                waypoint_id="wp_1",
                position=np.array([100000.0, 0.0, 50000.0]),
                altitude=50000.0,
                mach_number=20.0,
                heading=0.0,
                flight_phase=HypersonicFlightPhase.ACCELERATION,
                thermal_load=60e6
            ),
            HypersonicWaypoint(
                waypoint_id="wp_2",
                position=np.array([200000.0, 0.0, 60000.0]),
                altitude=60000.0,
                mach_number=40.0,
                heading=0.0,
                flight_phase=HypersonicFlightPhase.CRUISE,
                thermal_load=110e6  # Exceeds limit
            )
        ]
        
        profile = HypersonicMissionProfile(
            profile_id="thermal_monitoring_test",
            name="Thermal Monitoring Test",
            waypoints=waypoints
        )
        
        monitoring_results = self.thermal_manager.monitor_thermal_constraints(profile, time_step=5.0)
        
        assert 'thermal_violations' in monitoring_results
        assert 'cooling_activations' in monitoring_results
        assert 'thermal_timeline' in monitoring_results
        assert len(monitoring_results['thermal_timeline']) == len(waypoints)
        
        # Should detect thermal violation from second waypoint
        assert len(monitoring_results['thermal_violations']) > 0


class TestTrajectoryThermalModifier:
    """Test trajectory modification for thermal constraints."""
    
    def setup_method(self):
        """Set up test fixtures."""
        thermal_constraints = ThermalConstraint(
            max_heat_flux=80e6,  # Lower limit for testing
            max_temperature=2500.0,
            max_duration_at_peak=300.0,
            cooling_system_capacity=50e6,
            recovery_time_required=120.0
        )
        
        cooling_performance = CoolingSystemPerformance(
            max_cooling_capacity=60e6,
            power_consumption=5000.0,
            coolant_flow_rate=10.0,
            effectiveness=0.8,
            response_time=5.0,
            operating_altitude_range=(20000.0, 100000.0),
            operating_mach_range=(5.0, 60.0)
        )
        
        thermal_manager = ThermalConstraintManager(thermal_constraints, cooling_performance)
        self.trajectory_modifier = TrajectoryThermalModifier(thermal_manager)
    
    def test_thermal_violation_detection(self):
        """Test detection of thermal limit violations."""
        # Waypoint within limits
        safe_waypoint = HypersonicWaypoint(
            waypoint_id="safe_wp",
            position=np.array([100000.0, 0.0, 60000.0]),
            altitude=60000.0,
            mach_number=25.0,
            heading=0.0,
            flight_phase=HypersonicFlightPhase.CRUISE,
            thermal_load=60e6  # Within limit
        )
        
        violates_safe = self.trajectory_modifier._waypoint_violates_thermal_limits(safe_waypoint)
        assert not violates_safe
        
        # Waypoint exceeding limits
        unsafe_waypoint = HypersonicWaypoint(
            waypoint_id="unsafe_wp",
            position=np.array([100000.0, 0.0, 50000.0]),
            altitude=50000.0,
            mach_number=45.0,
            heading=0.0,
            flight_phase=HypersonicFlightPhase.CRUISE,
            thermal_load=100e6  # Exceeds limit
        )
        
        violates_unsafe = self.trajectory_modifier._waypoint_violates_thermal_limits(unsafe_waypoint)
        assert violates_unsafe
    
    def test_thermally_safe_waypoint_generation(self):
        """Test generation of thermally safe waypoints."""
        unsafe_waypoint = HypersonicWaypoint(
            waypoint_id="unsafe_original",
            position=np.array([100000.0, 0.0, 50000.0]),
            altitude=50000.0,
            mach_number=45.0,
            heading=0.0,
            flight_phase=HypersonicFlightPhase.CRUISE,
            thermal_load=100e6
        )
        
        safe_waypoint = self.trajectory_modifier._generate_thermally_safe_waypoint(unsafe_waypoint)
        
        assert safe_waypoint.altitude > unsafe_waypoint.altitude
        assert safe_waypoint.mach_number < unsafe_waypoint.mach_number
        assert safe_waypoint.thermal_load < unsafe_waypoint.thermal_load
        assert safe_waypoint.cooling_system_active
        assert "thermal_safe" in safe_waypoint.waypoint_id
    
    def test_trajectory_modification_for_thermal_limits(self):
        """Test complete trajectory modification for thermal limits."""
        waypoints = [
            HypersonicWaypoint(
                waypoint_id="wp_safe",
                position=np.array([100000.0, 0.0, 60000.0]),
                altitude=60000.0,
                mach_number=20.0,
                heading=0.0,
                flight_phase=HypersonicFlightPhase.CRUISE,
                thermal_load=50e6  # Safe
            ),
            HypersonicWaypoint(
                waypoint_id="wp_unsafe",
                position=np.array([200000.0, 0.0, 50000.0]),
                altitude=50000.0,
                mach_number=50.0,
                heading=0.0,
                flight_phase=HypersonicFlightPhase.CRUISE,
                thermal_load=120e6  # Unsafe
            )
        ]
        
        profile = HypersonicMissionProfile(
            profile_id="modification_test",
            name="Modification Test",
            waypoints=waypoints
        )
        
        modified_profile = self.trajectory_modifier.modify_trajectory_for_thermal_limits(profile)
        
        assert len(modified_profile.waypoints) == 2
        
        # First waypoint should be unchanged
        assert modified_profile.waypoints[0].waypoint_id == "wp_safe"
        
        # Second waypoint should be modified
        modified_wp = modified_profile.waypoints[1]
        assert "thermal_safe" in modified_wp.waypoint_id
        assert modified_wp.altitude > waypoints[1].altitude
        assert modified_wp.mach_number < waypoints[1].mach_number


class TestThermalIntegration:
    """Test integration of thermal constraints with mission planner."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create combined cycle engine
        engine_spec = CombinedCycleSpecification(
            engine_id="thermal_integration_engine",
            name="Thermal Integration Test Engine",
            engine_type=None,
            max_thrust_sea_level=500000.0,
            max_thrust_altitude=400000.0,
            design_altitude=60000.0,
            design_mach=30.0,
            extreme_propulsion_type=ExtremePropulsionType.COMBINED_CYCLE_AIRBREATHING,
            transition_mach_number=12.0,
            max_rocket_thrust=300000.0
        )
        self.engine = CombinedCycleEngine(engine_spec)
        self.planner = HypersonicMissionPlanner(self.engine)
        
        # Thermal constraints
        self.thermal_constraints = ThermalConstraint(
            max_heat_flux=100e6,
            max_temperature=3000.0,
            max_duration_at_peak=300.0,
            cooling_system_capacity=50e6,
            recovery_time_required=120.0
        )
        
        self.cooling_performance = CoolingSystemPerformance(
            max_cooling_capacity=60e6,
            power_consumption=5000.0,
            coolant_flow_rate=10.0,
            effectiveness=0.8,
            response_time=5.0,
            operating_altitude_range=(20000.0, 100000.0),
            operating_mach_range=(5.0, 60.0)
        )
    
    def test_thermal_integration_setup(self):
        """Test thermal constraint integration setup."""
        integrated_planner = integrate_thermal_constraints_with_mission_planner(
            self.planner, self.thermal_constraints, self.cooling_performance
        )
        
        assert hasattr(integrated_planner, 'thermal_manager')
        assert hasattr(integrated_planner, 'trajectory_modifier')
        assert hasattr(integrated_planner, 'monitor_mission_thermal_constraints')
        assert hasattr(integrated_planner, 'modify_trajectory_for_thermal_safety')
        
        assert integrated_planner.thermal_manager.thermal_constraints == self.thermal_constraints
        assert integrated_planner.thermal_manager.cooling_performance == self.cooling_performance
    
    def test_integrated_thermal_monitoring(self):
        """Test thermal monitoring through integrated planner."""
        integrated_planner = integrate_thermal_constraints_with_mission_planner(
            self.planner, self.thermal_constraints, self.cooling_performance
        )
        
        # Create test profile
        waypoints = [
            HypersonicWaypoint(
                waypoint_id="integrated_wp_1",
                position=np.array([100000.0, 0.0, 60000.0]),
                altitude=60000.0,
                mach_number=30.0,
                heading=0.0,
                flight_phase=HypersonicFlightPhase.CRUISE,
                thermal_load=80e6
            )
        ]
        
        profile = HypersonicMissionProfile(
            profile_id="integrated_test",
            name="Integrated Test",
            waypoints=waypoints
        )
        
        monitoring_results = integrated_planner.monitor_mission_thermal_constraints(profile)
        
        assert isinstance(monitoring_results, dict)
        assert 'thermal_timeline' in monitoring_results
        assert 'thermal_violations' in monitoring_results
        assert 'cooling_activations' in monitoring_results
    
    def test_integrated_trajectory_modification(self):
        """Test trajectory modification through integrated planner."""
        integrated_planner = integrate_thermal_constraints_with_mission_planner(
            self.planner, self.thermal_constraints, self.cooling_performance
        )
        
        # Create profile with thermal violations
        waypoints = [
            HypersonicWaypoint(
                waypoint_id="violation_wp",
                position=np.array([100000.0, 0.0, 40000.0]),
                altitude=40000.0,
                mach_number=50.0,
                heading=0.0,
                flight_phase=HypersonicFlightPhase.CRUISE,
                thermal_load=150e6  # Exceeds limits
            )
        ]
        
        profile = HypersonicMissionProfile(
            profile_id="violation_test",
            name="Violation Test",
            waypoints=waypoints
        )
        
        modified_profile = integrated_planner.modify_trajectory_for_thermal_safety(profile)
        
        assert len(modified_profile.waypoints) == 1
        modified_wp = modified_profile.waypoints[0]
        
        # Should be modified for thermal safety
        assert modified_wp.altitude > waypoints[0].altitude
        assert modified_wp.mach_number < waypoints[0].mach_number
        assert modified_wp.cooling_system_active


if __name__ == "__main__":
    pytest.main([__file__])