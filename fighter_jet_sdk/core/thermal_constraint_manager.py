"""Thermal constraint management for hypersonic missions."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import numpy as np
import math

from .hypersonic_mission_planner import (
    HypersonicMissionProfile, HypersonicWaypoint, ThermalConstraint,
    HypersonicFlightPhase
)
from ..engines.propulsion.combined_cycle_engine import PropulsionMode
from .errors import SimulationError, ValidationError
from .logging import get_logger


class ThermalStatus(Enum):
    """Thermal status levels."""
    NORMAL = "normal"
    ELEVATED = "elevated"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class CoolingSystemMode(Enum):
    """Cooling system operating modes."""
    OFF = "off"
    PASSIVE = "passive"
    ACTIVE_LOW = "active_low"
    ACTIVE_MEDIUM = "active_medium"
    ACTIVE_HIGH = "active_high"
    EMERGENCY = "emergency"


@dataclass
class ThermalState:
    """Current thermal state of the vehicle."""
    surface_temperature: float  # K
    heat_flux: float  # W/m²
    thermal_load_integral: float  # J/m² (accumulated thermal energy)
    cooling_system_mode: CoolingSystemMode = CoolingSystemMode.OFF
    time_at_current_load: float = 0.0  # seconds
    time_since_last_recovery: float = 0.0  # seconds
    thermal_stress_level: float = 0.0  # 0.0 to 1.0
    material_degradation: float = 0.0  # 0.0 to 1.0


@dataclass
class ThermalEvent:
    """Thermal event during mission."""
    event_id: str
    time: float  # seconds from mission start
    event_type: str  # "threshold_exceeded", "cooling_activated", "recovery_started", etc.
    thermal_state: ThermalState
    action_taken: str
    waypoint_id: Optional[str] = None


@dataclass
class CoolingSystemPerformance:
    """Cooling system performance characteristics."""
    max_cooling_capacity: float  # W
    power_consumption: float  # W
    coolant_flow_rate: float  # kg/s
    effectiveness: float  # 0.0 to 1.0
    response_time: float  # seconds
    operating_altitude_range: Tuple[float, float]  # m
    operating_mach_range: Tuple[float, float]


class ThermalConstraintManager:
    """Manages thermal constraints during hypersonic missions."""
    
    def __init__(self, thermal_constraints: ThermalConstraint,
                 cooling_system_performance: CoolingSystemPerformance):
        """Initialize thermal constraint manager."""
        self.thermal_constraints = thermal_constraints
        self.cooling_performance = cooling_system_performance
        self.logger = get_logger("thermal_constraint_manager")
        
        # Thermal monitoring state
        self.current_thermal_state = ThermalState(
            surface_temperature=300.0,  # K
            heat_flux=0.0,
            thermal_load_integral=0.0
        )
        
        # Event history
        self.thermal_events: List[ThermalEvent] = []
        
        # Thermal thresholds (as fractions of maximum)
        self.thermal_thresholds = {
            ThermalStatus.NORMAL: 0.6,
            ThermalStatus.ELEVATED: 0.75,
            ThermalStatus.WARNING: 0.85,
            ThermalStatus.CRITICAL: 0.95,
            ThermalStatus.EMERGENCY: 1.0
        }
        
        self.logger.info("Thermal constraint manager initialized")
    
    def update_thermal_state(self, waypoint: HypersonicWaypoint, 
                           time_step: float, mission_time: float) -> ThermalState:
        """Update thermal state based on current waypoint conditions."""
        # Calculate surface temperature from thermal load
        surface_temp = self._calculate_surface_temperature(
            waypoint.thermal_load, waypoint.altitude, waypoint.mach_number
        )
        
        # Update thermal load integral (accumulated thermal energy)
        thermal_load_integral = (self.current_thermal_state.thermal_load_integral + 
                               waypoint.thermal_load * time_step)
        
        # Update time at current load
        if abs(waypoint.thermal_load - self.current_thermal_state.heat_flux) < 1e6:
            time_at_current_load = self.current_thermal_state.time_at_current_load + time_step
        else:
            time_at_current_load = time_step
        
        # Calculate thermal stress level
        thermal_stress = self._calculate_thermal_stress_level(
            waypoint.thermal_load, surface_temp, time_at_current_load
        )
        
        # Calculate material degradation
        material_degradation = self._calculate_material_degradation(
            surface_temp, thermal_load_integral, thermal_stress
        )
        
        # Update thermal state
        self.current_thermal_state = ThermalState(
            surface_temperature=surface_temp,
            heat_flux=waypoint.thermal_load,
            thermal_load_integral=thermal_load_integral,
            cooling_system_mode=self.current_thermal_state.cooling_system_mode,
            time_at_current_load=time_at_current_load,
            time_since_last_recovery=self.current_thermal_state.time_since_last_recovery + time_step,
            thermal_stress_level=thermal_stress,
            material_degradation=material_degradation
        )
        
        return self.current_thermal_state
    
    def _calculate_surface_temperature(self, heat_flux: float, altitude: float, 
                                     mach: float) -> float:
        """Calculate surface temperature from heat flux and conditions."""
        # Simplified surface temperature calculation
        # Real implementation would use detailed heat transfer models
        
        # Base temperature from heat flux (Stefan-Boltzmann approximation)
        stefan_boltzmann = 5.67e-8  # W/(m²⋅K⁴)
        emissivity = 0.8  # Typical for aerospace materials
        
        # Radiative equilibrium temperature
        if heat_flux > 0:
            radiative_temp = (heat_flux / (emissivity * stefan_boltzmann)) ** 0.25
        else:
            radiative_temp = 300.0  # Ambient temperature
        
        # Add convective heating effects
        if mach > 5.0:
            # Stagnation temperature contribution
            gamma = 1.4
            # Simplified atmospheric temperature at altitude
            if altitude < 11000:
                atm_temp = 288.15 - 0.0065 * altitude
            else:
                atm_temp = 216.65
            
            stagnation_temp = atm_temp * (1 + (gamma - 1) / 2 * mach**2)
            
            # Surface temperature is between radiative and stagnation
            surface_temp = max(radiative_temp, 0.7 * stagnation_temp)
        else:
            surface_temp = radiative_temp
        
        return min(surface_temp, 4000.0)  # Material limit
    
    def _calculate_thermal_stress_level(self, heat_flux: float, temperature: float,
                                      duration: float) -> float:
        """Calculate thermal stress level (0.0 to 1.0)."""
        # Heat flux stress component
        flux_stress = min(1.0, heat_flux / self.thermal_constraints.max_heat_flux)
        
        # Temperature stress component
        temp_stress = min(1.0, temperature / self.thermal_constraints.max_temperature)
        
        # Duration stress component
        duration_stress = min(1.0, duration / self.thermal_constraints.max_duration_at_peak)
        
        # Combined stress (weighted average)
        total_stress = (0.4 * flux_stress + 0.4 * temp_stress + 0.2 * duration_stress)
        
        return min(1.0, total_stress)
    
    def _calculate_material_degradation(self, temperature: float, 
                                      thermal_integral: float,
                                      stress_level: float) -> float:
        """Calculate cumulative material degradation (0.0 to 1.0)."""
        # Temperature-based degradation (Arrhenius-type)
        if temperature > 1000.0:  # K
            temp_degradation = 0.001 * np.exp((temperature - 1000.0) / 500.0)
        else:
            temp_degradation = 0.0
        
        # Thermal cycling degradation
        cycling_degradation = thermal_integral / 1e12  # Normalized
        
        # Stress-based degradation
        stress_degradation = stress_level * 0.01
        
        # Total degradation (cumulative)
        total_degradation = (self.current_thermal_state.material_degradation + 
                           temp_degradation + cycling_degradation + stress_degradation)
        
        return min(1.0, total_degradation)
    
    def get_thermal_status(self, thermal_state: ThermalState) -> ThermalStatus:
        """Determine thermal status based on current state."""
        # Check against heat flux threshold
        flux_ratio = thermal_state.heat_flux / self.thermal_constraints.max_heat_flux
        
        # Check against temperature threshold
        temp_ratio = thermal_state.surface_temperature / self.thermal_constraints.max_temperature
        
        # Check against duration threshold
        duration_ratio = thermal_state.time_at_current_load / self.thermal_constraints.max_duration_at_peak
        
        # Use the highest ratio to determine status
        max_ratio = max(flux_ratio, temp_ratio, duration_ratio)
        
        if max_ratio >= self.thermal_thresholds[ThermalStatus.EMERGENCY]:
            return ThermalStatus.EMERGENCY
        elif max_ratio >= self.thermal_thresholds[ThermalStatus.CRITICAL]:
            return ThermalStatus.CRITICAL
        elif max_ratio >= self.thermal_thresholds[ThermalStatus.WARNING]:
            return ThermalStatus.WARNING
        elif max_ratio >= self.thermal_thresholds[ThermalStatus.ELEVATED]:
            return ThermalStatus.ELEVATED
        else:
            return ThermalStatus.NORMAL
    
    def determine_required_cooling_mode(self, thermal_status: ThermalStatus,
                                      thermal_state: ThermalState) -> CoolingSystemMode:
        """Determine required cooling system mode based on thermal status."""
        if thermal_status == ThermalStatus.EMERGENCY:
            return CoolingSystemMode.EMERGENCY
        elif thermal_status == ThermalStatus.CRITICAL:
            return CoolingSystemMode.ACTIVE_HIGH
        elif thermal_status == ThermalStatus.WARNING:
            return CoolingSystemMode.ACTIVE_MEDIUM
        elif thermal_status == ThermalStatus.ELEVATED:
            return CoolingSystemMode.ACTIVE_LOW
        else:
            # Check if passive cooling is sufficient
            if thermal_state.heat_flux > self.thermal_constraints.max_heat_flux * 0.3:
                return CoolingSystemMode.PASSIVE
            else:
                return CoolingSystemMode.OFF
    
    def activate_cooling_system(self, required_mode: CoolingSystemMode,
                              current_waypoint: HypersonicWaypoint,
                              mission_time: float) -> Tuple[bool, str]:
        """Activate cooling system with specified mode."""
        # Check if cooling system can operate at current conditions
        altitude_ok = (self.cooling_performance.operating_altitude_range[0] <= 
                      current_waypoint.altitude <= 
                      self.cooling_performance.operating_altitude_range[1])
        
        mach_ok = (self.cooling_performance.operating_mach_range[0] <= 
                  current_waypoint.mach_number <= 
                  self.cooling_performance.operating_mach_range[1])
        
        if not (altitude_ok and mach_ok):
            return False, f"Cooling system cannot operate at current conditions (Alt: {current_waypoint.altitude/1000:.1f}km, Mach: {current_waypoint.mach_number:.1f})"
        
        # Activate cooling system
        self.current_thermal_state.cooling_system_mode = required_mode
        
        # Log thermal event
        event = ThermalEvent(
            event_id=f"cooling_activation_{len(self.thermal_events)}",
            time=mission_time,
            event_type="cooling_activated",
            thermal_state=self.current_thermal_state,
            action_taken=f"Activated cooling system in {required_mode.value} mode",
            waypoint_id=current_waypoint.waypoint_id
        )
        self.thermal_events.append(event)
        
        self.logger.info(f"Cooling system activated in {required_mode.value} mode at {mission_time:.1f}s")
        
        return True, f"Cooling system activated in {required_mode.value} mode"
    
    def calculate_cooling_effectiveness(self, cooling_mode: CoolingSystemMode,
                                      heat_flux: float) -> float:
        """Calculate cooling system effectiveness for given conditions."""
        if cooling_mode == CoolingSystemMode.OFF:
            return 0.0
        
        # Base effectiveness by mode
        mode_effectiveness = {
            CoolingSystemMode.PASSIVE: 0.1,
            CoolingSystemMode.ACTIVE_LOW: 0.3,
            CoolingSystemMode.ACTIVE_MEDIUM: 0.6,
            CoolingSystemMode.ACTIVE_HIGH: 0.8,
            CoolingSystemMode.EMERGENCY: 1.0
        }
        
        base_effectiveness = mode_effectiveness.get(cooling_mode, 0.0)
        
        # Adjust for heat flux level
        if heat_flux > self.cooling_performance.max_cooling_capacity:
            # Cooling system overwhelmed
            effectiveness = base_effectiveness * (self.cooling_performance.max_cooling_capacity / heat_flux)
        else:
            effectiveness = base_effectiveness
        
        return min(1.0, effectiveness * self.cooling_performance.effectiveness)
    
    def apply_cooling_effects(self, thermal_state: ThermalState,
                            cooling_effectiveness: float) -> ThermalState:
        """Apply cooling system effects to thermal state."""
        if cooling_effectiveness <= 0:
            return thermal_state
        
        # Reduce heat flux
        reduced_heat_flux = thermal_state.heat_flux * (1.0 - cooling_effectiveness)
        
        # Reduce surface temperature
        temp_reduction = (thermal_state.surface_temperature - 300.0) * cooling_effectiveness
        reduced_temperature = thermal_state.surface_temperature - temp_reduction
        
        # Reduce thermal stress
        reduced_stress = thermal_state.thermal_stress_level * (1.0 - cooling_effectiveness * 0.5)
        
        return ThermalState(
            surface_temperature=reduced_temperature,
            heat_flux=reduced_heat_flux,
            thermal_load_integral=thermal_state.thermal_load_integral,
            cooling_system_mode=thermal_state.cooling_system_mode,
            time_at_current_load=thermal_state.time_at_current_load,
            time_since_last_recovery=thermal_state.time_since_last_recovery,
            thermal_stress_level=reduced_stress,
            material_degradation=thermal_state.material_degradation
        )
    
    def check_thermal_recovery_needed(self, thermal_state: ThermalState) -> bool:
        """Check if thermal recovery maneuver is needed."""
        # Recovery needed if:
        # 1. Thermal stress is high
        # 2. Material degradation is significant
        # 3. Been at high thermal load for too long
        # 4. Haven't had recovery in a while
        
        stress_recovery_needed = thermal_state.thermal_stress_level > 0.8
        degradation_recovery_needed = thermal_state.material_degradation > 0.1
        duration_recovery_needed = thermal_state.time_at_current_load > self.thermal_constraints.max_duration_at_peak
        time_recovery_needed = thermal_state.time_since_last_recovery > 1800.0  # 30 minutes
        
        return (stress_recovery_needed or degradation_recovery_needed or 
                duration_recovery_needed or time_recovery_needed)
    
    def generate_thermal_recovery_waypoint(self, current_waypoint: HypersonicWaypoint,
                                         mission_time: float) -> HypersonicWaypoint:
        """Generate thermal recovery waypoint."""
        # Recovery strategy: higher altitude, lower Mach, activate cooling
        recovery_altitude = min(100000.0, current_waypoint.altitude + 10000.0)
        recovery_mach = max(5.0, current_waypoint.mach_number * 0.7)
        
        # Calculate reduced thermal load
        recovery_thermal_load = current_waypoint.thermal_load * 0.3
        
        recovery_waypoint = HypersonicWaypoint(
            waypoint_id=f"thermal_recovery_{mission_time:.0f}",
            position=current_waypoint.position.copy(),
            altitude=recovery_altitude,
            mach_number=recovery_mach,
            heading=current_waypoint.heading,
            flight_phase=HypersonicFlightPhase.THERMAL_RECOVERY,
            propulsion_mode=PropulsionMode.AIR_BREATHING,  # More efficient for recovery
            thermal_load=recovery_thermal_load,
            cooling_system_active=True,
            duration=self.thermal_constraints.recovery_time_required
        )
        
        # Log thermal event
        event = ThermalEvent(
            event_id=f"recovery_initiated_{len(self.thermal_events)}",
            time=mission_time,
            event_type="recovery_started",
            thermal_state=self.current_thermal_state,
            action_taken="Initiated thermal recovery maneuver",
            waypoint_id=recovery_waypoint.waypoint_id
        )
        self.thermal_events.append(event)
        
        self.logger.info(f"Thermal recovery waypoint generated at {mission_time:.1f}s")
        
        return recovery_waypoint
    
    def monitor_thermal_constraints(self, profile: HypersonicMissionProfile,
                                  time_step: float = 10.0) -> Dict[str, Any]:
        """Monitor thermal constraints throughout mission profile."""
        monitoring_results = {
            'thermal_violations': [],
            'cooling_activations': [],
            'recovery_maneuvers': [],
            'max_thermal_stress': 0.0,
            'max_material_degradation': 0.0,
            'total_cooling_time': 0.0,
            'thermal_timeline': []
        }
        
        mission_time = 0.0
        
        for i, waypoint in enumerate(profile.waypoints):
            # Update thermal state
            thermal_state = self.update_thermal_state(waypoint, time_step, mission_time)
            
            # Get thermal status
            thermal_status = self.get_thermal_status(thermal_state)
            
            # Check for violations
            if thermal_status in [ThermalStatus.CRITICAL, ThermalStatus.EMERGENCY]:
                monitoring_results['thermal_violations'].append({
                    'time': mission_time,
                    'waypoint_id': waypoint.waypoint_id,
                    'status': thermal_status.value,
                    'heat_flux': thermal_state.heat_flux,
                    'temperature': thermal_state.surface_temperature
                })
            
            # Determine cooling requirements
            required_cooling = self.determine_required_cooling_mode(thermal_status, thermal_state)
            
            # Activate cooling if needed
            if required_cooling != CoolingSystemMode.OFF:
                success, message = self.activate_cooling_system(required_cooling, waypoint, mission_time)
                if success:
                    monitoring_results['cooling_activations'].append({
                        'time': mission_time,
                        'waypoint_id': waypoint.waypoint_id,
                        'mode': required_cooling.value,
                        'message': message
                    })
                    monitoring_results['total_cooling_time'] += time_step
            
            # Check for recovery needs
            if self.check_thermal_recovery_needed(thermal_state):
                recovery_wp = self.generate_thermal_recovery_waypoint(waypoint, mission_time)
                monitoring_results['recovery_maneuvers'].append({
                    'time': mission_time,
                    'original_waypoint_id': waypoint.waypoint_id,
                    'recovery_waypoint_id': recovery_wp.waypoint_id,
                    'recovery_duration': recovery_wp.duration
                })
            
            # Update maximums
            monitoring_results['max_thermal_stress'] = max(
                monitoring_results['max_thermal_stress'], thermal_state.thermal_stress_level
            )
            monitoring_results['max_material_degradation'] = max(
                monitoring_results['max_material_degradation'], thermal_state.material_degradation
            )
            
            # Add to timeline
            monitoring_results['thermal_timeline'].append({
                'time': mission_time,
                'waypoint_id': waypoint.waypoint_id,
                'thermal_status': thermal_status.value,
                'heat_flux': thermal_state.heat_flux,
                'surface_temperature': thermal_state.surface_temperature,
                'thermal_stress': thermal_state.thermal_stress_level,
                'cooling_mode': thermal_state.cooling_system_mode.value
            })
            
            mission_time += time_step
        
        return monitoring_results


class TrajectoryThermalModifier:
    """Modifies trajectories based on thermal constraints."""
    
    def __init__(self, thermal_manager: ThermalConstraintManager):
        """Initialize trajectory thermal modifier."""
        self.thermal_manager = thermal_manager
        self.logger = get_logger("trajectory_thermal_modifier")
    
    def modify_trajectory_for_thermal_limits(self, profile: HypersonicMissionProfile) -> HypersonicMissionProfile:
        """Modify trajectory to respect thermal limits."""
        self.logger.info("Modifying trajectory for thermal limits")
        
        modified_waypoints = []
        
        for waypoint in profile.waypoints:
            # Check if waypoint violates thermal constraints
            if self._waypoint_violates_thermal_limits(waypoint):
                # Generate alternative waypoint
                modified_wp = self._generate_thermally_safe_waypoint(waypoint)
                modified_waypoints.append(modified_wp)
                
                self.logger.info(f"Modified waypoint {waypoint.waypoint_id} for thermal safety")
            else:
                modified_waypoints.append(waypoint)
        
        # Update profile
        profile.waypoints = modified_waypoints
        
        # Regenerate profile arrays
        self._regenerate_profile_arrays(profile)
        
        return profile
    
    def _waypoint_violates_thermal_limits(self, waypoint: HypersonicWaypoint) -> bool:
        """Check if waypoint violates thermal limits."""
        constraints = self.thermal_manager.thermal_constraints
        
        # Check heat flux limit
        if waypoint.thermal_load > constraints.max_heat_flux:
            return True
        
        # Check if thermal load would cause excessive temperature
        surface_temp = self.thermal_manager._calculate_surface_temperature(
            waypoint.thermal_load, waypoint.altitude, waypoint.mach_number
        )
        
        if surface_temp > constraints.max_temperature:
            return True
        
        return False
    
    def _generate_thermally_safe_waypoint(self, original_waypoint: HypersonicWaypoint) -> HypersonicWaypoint:
        """Generate thermally safe alternative to waypoint."""
        # Strategy: increase altitude and/or reduce Mach number
        safe_altitude = min(100000.0, original_waypoint.altitude + 5000.0)
        safe_mach = max(5.0, original_waypoint.mach_number * 0.9)
        
        # Recalculate thermal load (simplified calculation)
        # Real implementation would use the trajectory optimizer's thermal load calculation
        safe_thermal_load = original_waypoint.thermal_load * 0.7  # Reduced by altitude/Mach changes
        
        return HypersonicWaypoint(
            waypoint_id=f"{original_waypoint.waypoint_id}_thermal_safe",
            position=original_waypoint.position,
            altitude=safe_altitude,
            mach_number=safe_mach,
            heading=original_waypoint.heading,
            flight_phase=original_waypoint.flight_phase,
            propulsion_mode=original_waypoint.propulsion_mode,
            thermal_load=safe_thermal_load,
            cooling_system_active=True,
            duration=original_waypoint.duration
        )
    
    def _regenerate_profile_arrays(self, profile: HypersonicMissionProfile) -> None:
        """Regenerate profile arrays after waypoint modifications."""
        if not profile.waypoints:
            return
        
        profile.altitude_profile = np.array([wp.altitude for wp in profile.waypoints])
        profile.mach_profile = np.array([wp.mach_number for wp in profile.waypoints])
        profile.thermal_load_profile = np.array([wp.thermal_load for wp in profile.waypoints])
        profile.propulsion_mode_schedule = [wp.propulsion_mode for wp in profile.waypoints]
        profile.cooling_system_schedule = [wp.cooling_system_active for wp in profile.waypoints]
        
        # Update maximum thermal load
        profile.max_thermal_load = np.max(profile.thermal_load_profile)


def integrate_thermal_constraints_with_mission_planner(planner, thermal_constraints: ThermalConstraint,
                                                     cooling_performance: CoolingSystemPerformance):
    """Integrate thermal constraint management with mission planner."""
    # Create thermal constraint manager
    thermal_manager = ThermalConstraintManager(thermal_constraints, cooling_performance)
    
    # Create trajectory modifier
    trajectory_modifier = TrajectoryThermalModifier(thermal_manager)
    
    # Extend mission planner with thermal capabilities
    planner.thermal_manager = thermal_manager
    planner.trajectory_modifier = trajectory_modifier
    
    # Add thermal monitoring method
    def monitor_mission_thermal_constraints(profile: HypersonicMissionProfile) -> Dict[str, Any]:
        return thermal_manager.monitor_thermal_constraints(profile)
    
    # Add trajectory modification method
    def modify_trajectory_for_thermal_safety(profile: HypersonicMissionProfile) -> HypersonicMissionProfile:
        return trajectory_modifier.modify_trajectory_for_thermal_limits(profile)
    
    planner.monitor_mission_thermal_constraints = monitor_mission_thermal_constraints
    planner.modify_trajectory_for_thermal_safety = modify_trajectory_for_thermal_safety
    
    return planner