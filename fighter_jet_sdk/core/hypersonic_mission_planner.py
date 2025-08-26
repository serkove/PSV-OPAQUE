"""Hypersonic mission profile optimization for Mach 60 flight."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import numpy as np
import math
from abc import ABC, abstractmethod

from .mission_simulation import MissionScenario, MissionWaypoint, MissionPhase
from ..common.data_models import FlowConditions, AircraftConfiguration
from ..common.enums import ExtremePropulsionType, PlasmaRegime, ThermalProtectionType
from ..engines.propulsion.combined_cycle_engine import CombinedCycleEngine, CombinedCycleOperatingPoint, PropulsionMode
from .errors import SimulationError, ValidationError
from .logging import get_logger


class HypersonicFlightPhase(Enum):
    """Flight phases for hypersonic missions."""
    ACCELERATION = "acceleration"
    CRUISE = "cruise"
    MANEUVER = "maneuver"
    DESCENT = "descent"
    THERMAL_RECOVERY = "thermal_recovery"


class OptimizationObjective(Enum):
    """Optimization objectives for mission planning."""
    MINIMIZE_FUEL = "minimize_fuel"
    MINIMIZE_TIME = "minimize_time"
    MINIMIZE_THERMAL_LOAD = "minimize_thermal_load"
    MAXIMIZE_RANGE = "maximize_range"
    MAXIMIZE_SURVIVABILITY = "maximize_survivability"


@dataclass
class HypersonicWaypoint:
    """Extended waypoint for hypersonic missions."""
    waypoint_id: str
    position: np.ndarray  # [x, y, z] in meters
    altitude: float  # meters
    mach_number: float
    heading: float  # radians
    flight_phase: HypersonicFlightPhase
    propulsion_mode: PropulsionMode = PropulsionMode.AIR_BREATHING
    thermal_load: float = 0.0  # W/m²
    cooling_system_active: bool = False
    duration: float = 0.0  # seconds
    fuel_consumption_rate: float = 0.0  # kg/s


@dataclass
class ThermalConstraint:
    """Thermal constraint for mission planning."""
    max_heat_flux: float  # W/m²
    max_temperature: float  # K
    max_duration_at_peak: float  # seconds
    cooling_system_capacity: float  # W
    recovery_time_required: float  # seconds


@dataclass
class PropulsionConstraint:
    """Propulsion system constraints."""
    max_air_breathing_mach: float = 15.0
    min_rocket_altitude: float = 40000.0  # m
    fuel_capacity_air_breathing: float = 10000.0  # kg
    fuel_capacity_rocket: float = 5000.0  # kg
    max_continuous_operation_time: float = 1800.0  # seconds


@dataclass
class HypersonicMissionProfile:
    """Complete hypersonic mission profile."""
    profile_id: str
    name: str
    waypoints: List[HypersonicWaypoint] = field(default_factory=list)
    altitude_profile: np.ndarray = field(default_factory=lambda: np.array([]))
    mach_profile: np.ndarray = field(default_factory=lambda: np.array([]))
    thermal_load_profile: np.ndarray = field(default_factory=lambda: np.array([]))
    propulsion_mode_schedule: List[PropulsionMode] = field(default_factory=list)
    cooling_system_schedule: List[bool] = field(default_factory=list)
    total_duration: float = 0.0  # seconds
    total_fuel_consumption: float = 0.0  # kg
    max_thermal_load: float = 0.0  # W/m²
    thermal_constraints: Optional[ThermalConstraint] = None
    propulsion_constraints: Optional[PropulsionConstraint] = None

    def validate_profile(self) -> List[str]:
        """Validate hypersonic mission profile."""
        errors = []
        
        if not self.waypoints:
            errors.append("Mission profile must have waypoints")
            return errors
        
        # Validate altitude-Mach correlation
        for i, waypoint in enumerate(self.waypoints):
            if waypoint.mach_number >= 60.0 and waypoint.altitude < 50000:
                errors.append(f"Waypoint {i}: Mach 60+ flight requires altitude > 50 km")
            
            if waypoint.mach_number >= 40.0 and waypoint.altitude < 40000:
                errors.append(f"Waypoint {i}: Mach 40+ flight requires altitude > 40 km")
        
        # Validate thermal constraints
        if self.thermal_constraints:
            for i, waypoint in enumerate(self.waypoints):
                if waypoint.thermal_load > self.thermal_constraints.max_heat_flux:
                    errors.append(f"Waypoint {i}: Thermal load exceeds maximum limit")
        
        # Validate propulsion mode transitions
        for i in range(1, len(self.waypoints)):
            prev_wp = self.waypoints[i-1]
            curr_wp = self.waypoints[i]
            
            # Check for valid propulsion mode transitions
            if (prev_wp.propulsion_mode == PropulsionMode.AIR_BREATHING and 
                curr_wp.propulsion_mode == PropulsionMode.PURE_ROCKET and
                curr_wp.mach_number - prev_wp.mach_number > 10.0):
                errors.append(f"Waypoint {i}: Abrupt propulsion mode transition")
        
        return errors


class AltitudeOptimizer:
    """Optimizer for altitude profiles in hypersonic flight."""
    
    def __init__(self):
        """Initialize altitude optimizer."""
        self.logger = get_logger("hypersonic_altitude_optimizer")
        
        # Atmospheric model parameters
        self.sea_level_pressure = 101325.0  # Pa
        self.sea_level_temperature = 288.15  # K
        self.sea_level_density = 1.225  # kg/m³
    
    def get_atmospheric_conditions(self, altitude: float) -> Tuple[float, float, float]:
        """Get atmospheric conditions at given altitude."""
        if altitude <= 11000:  # Troposphere
            temperature = 288.15 - 0.0065 * altitude
            pressure = 101325 * (temperature / 288.15) ** 5.256
        elif altitude <= 20000:  # Lower stratosphere
            temperature = 216.65
            pressure = 22632 * np.exp(-0.0001577 * (altitude - 11000))
        elif altitude <= 32000:  # Upper stratosphere
            temperature = 216.65 + 0.001 * (altitude - 20000)
            pressure = 5474.9 * (temperature / 216.65) ** (-34.163)
        else:  # Mesosphere and above
            temperature = max(180.0, 228.65 - 0.0028 * (altitude - 32000))
            pressure = max(0.1, 868.02 * np.exp(-0.0001262 * (altitude - 32000)))
        
        density = pressure / (287.0 * temperature)
        return temperature, pressure, density
    
    def calculate_dynamic_pressure(self, altitude: float, mach: float) -> float:
        """Calculate dynamic pressure at given conditions."""
        temperature, pressure, density = self.get_atmospheric_conditions(altitude)
        
        # Speed of sound
        gamma = 1.4
        speed_of_sound = math.sqrt(gamma * 287.0 * temperature)
        
        # Velocity
        velocity = mach * speed_of_sound
        
        # Dynamic pressure
        q = 0.5 * density * velocity**2
        return q
    
    def optimize_altitude_for_mach_range(self, mach_start: float, mach_end: float,
                                       distance: float, constraints: Dict[str, Any]) -> np.ndarray:
        """Optimize altitude profile for given Mach range and distance."""
        # Create Mach profile
        num_points = max(10, int(distance / 10000))  # Point every 10 km
        mach_profile = np.linspace(mach_start, mach_end, num_points)
        
        # Initialize altitude profile
        altitude_profile = np.zeros(num_points)
        
        # Set initial altitude based on starting Mach number
        if mach_start < 5.0:
            altitude_profile[0] = 15000.0  # 15 km
        elif mach_start < 15.0:
            altitude_profile[0] = 25000.0  # 25 km
        elif mach_start < 30.0:
            altitude_profile[0] = 40000.0  # 40 km
        else:
            altitude_profile[0] = 60000.0  # 60 km
        
        # Optimize each segment
        for i in range(1, num_points):
            mach = mach_profile[i]
            
            # Altitude constraints based on Mach number
            min_altitude = self._get_minimum_altitude_for_mach(mach)
            max_altitude = self._get_maximum_altitude_for_mach(mach, constraints)
            
            # Optimize altitude for current Mach number
            optimal_altitude = self._optimize_single_altitude_point(
                mach, min_altitude, max_altitude, constraints
            )
            
            # Ensure smooth altitude transition
            max_altitude_change = 5000.0  # 5 km max change per segment
            prev_altitude = altitude_profile[i-1]
            
            if abs(optimal_altitude - prev_altitude) > max_altitude_change:
                if optimal_altitude > prev_altitude:
                    altitude_profile[i] = prev_altitude + max_altitude_change
                else:
                    altitude_profile[i] = prev_altitude - max_altitude_change
            else:
                altitude_profile[i] = optimal_altitude
        
        return altitude_profile
    
    def _get_minimum_altitude_for_mach(self, mach: float) -> float:
        """Get minimum altitude for given Mach number."""
        if mach < 5.0:
            return 10000.0  # 10 km
        elif mach < 15.0:
            return 20000.0  # 20 km
        elif mach < 30.0:
            return 35000.0  # 35 km
        elif mach < 45.0:
            return 45000.0  # 45 km
        else:
            return 55000.0  # 55 km
    
    def _get_maximum_altitude_for_mach(self, mach: float, constraints: Dict[str, Any]) -> float:
        """Get maximum altitude for given Mach number."""
        # Base maximum altitude
        if mach < 10.0:
            max_alt = 30000.0  # 30 km
        elif mach < 25.0:
            max_alt = 60000.0  # 60 km
        elif mach < 45.0:
            max_alt = 80000.0  # 80 km
        else:
            max_alt = 100000.0  # 100 km
        
        # Apply mission-specific constraints
        if 'max_altitude' in constraints:
            max_alt = min(max_alt, constraints['max_altitude'])
        
        return max_alt
    
    def _optimize_single_altitude_point(self, mach: float, min_alt: float, 
                                      max_alt: float, constraints: Dict[str, Any]) -> float:
        """Optimize altitude for a single Mach number point."""
        # Test altitudes in the valid range
        test_altitudes = np.linspace(min_alt, max_alt, 20)
        best_altitude = min_alt
        best_score = float('-inf')
        
        for altitude in test_altitudes:
            score = self._evaluate_altitude_point(mach, altitude, constraints)
            if score > best_score:
                best_score = score
                best_altitude = altitude
        
        return best_altitude
    
    def _evaluate_altitude_point(self, mach: float, altitude: float, 
                               constraints: Dict[str, Any]) -> float:
        """Evaluate altitude point based on optimization criteria."""
        score = 0.0
        
        # Dynamic pressure penalty (prefer lower dynamic pressure)
        q = self.calculate_dynamic_pressure(altitude, mach)
        q_penalty = -q / 100000.0  # Normalize
        score += q_penalty
        
        # Thermal load penalty (higher altitude generally better for cooling)
        thermal_benefit = altitude / 100000.0  # Normalize to 0-1
        score += thermal_benefit * 2.0
        
        # Propulsion efficiency (depends on specific constraints)
        if 'propulsion_efficiency_map' in constraints:
            efficiency = constraints['propulsion_efficiency_map'].get((altitude, mach), 0.8)
            score += efficiency * 3.0
        
        # Fuel consumption penalty
        _, pressure, density = self.get_atmospheric_conditions(altitude)
        # Lower density generally means better fuel efficiency for air-breathing
        if mach < 15.0:  # Air-breathing regime
            fuel_benefit = (1.225 - density) / 1.225  # Normalized
            score += fuel_benefit * 1.5
        
        return score


class TrajectoryOptimizer:
    """Optimizer for hypersonic flight trajectories."""
    
    def __init__(self, combined_cycle_engine: CombinedCycleEngine):
        """Initialize trajectory optimizer."""
        self.engine = combined_cycle_engine
        self.altitude_optimizer = AltitudeOptimizer()
        self.logger = get_logger("hypersonic_trajectory_optimizer")
    
    def optimize_trajectory(self, start_point: Tuple[float, float, float],
                          end_point: Tuple[float, float, float],
                          max_mach: float,
                          thermal_constraints: ThermalConstraint,
                          propulsion_constraints: PropulsionConstraint,
                          objective: OptimizationObjective) -> HypersonicMissionProfile:
        """Optimize complete trajectory for hypersonic mission."""
        self.logger.info(f"Optimizing trajectory from {start_point} to {end_point}, max Mach {max_mach}")
        
        # Calculate mission parameters
        distance = self._calculate_distance(start_point, end_point)
        
        # Create initial trajectory segments
        segments = self._create_trajectory_segments(
            start_point, end_point, max_mach, distance
        )
        
        # Optimize each segment
        optimized_waypoints = []
        for segment in segments:
            segment_waypoints = self._optimize_trajectory_segment(
                segment, thermal_constraints, propulsion_constraints, objective
            )
            optimized_waypoints.extend(segment_waypoints)
        
        # Create mission profile
        profile = HypersonicMissionProfile(
            profile_id=f"hypersonic_mission_{int(max_mach)}",
            name=f"Mach {max_mach} Hypersonic Mission",
            waypoints=optimized_waypoints,
            thermal_constraints=thermal_constraints,
            propulsion_constraints=propulsion_constraints
        )
        
        # Generate profile arrays
        self._generate_profile_arrays(profile)
        
        # Validate and refine
        validation_errors = profile.validate_profile()
        if validation_errors:
            self.logger.warning(f"Profile validation issues: {validation_errors}")
            profile = self._refine_profile_for_constraints(profile, validation_errors)
        
        return profile
    
    def _calculate_distance(self, start: Tuple[float, float, float], 
                          end: Tuple[float, float, float]) -> float:
        """Calculate great circle distance between two points."""
        # Simplified distance calculation (assuming flat earth for now)
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        return math.sqrt(dx**2 + dy**2)
    
    def _create_trajectory_segments(self, start_point: Tuple[float, float, float],
                                  end_point: Tuple[float, float, float],
                                  max_mach: float, distance: float) -> List[Dict[str, Any]]:
        """Create trajectory segments for optimization."""
        segments = []
        
        # Acceleration segment
        accel_distance = distance * 0.3  # 30% of distance for acceleration
        segments.append({
            'type': HypersonicFlightPhase.ACCELERATION,
            'start_mach': 4.0,
            'end_mach': max_mach,
            'distance': accel_distance,
            'start_position': start_point,
            'priority': 'fuel_efficiency'
        })
        
        # Cruise segment
        cruise_distance = distance * 0.5  # 50% for cruise
        cruise_start = (
            start_point[0] + (end_point[0] - start_point[0]) * 0.3,
            start_point[1] + (end_point[1] - start_point[1]) * 0.3,
            start_point[2]
        )
        segments.append({
            'type': HypersonicFlightPhase.CRUISE,
            'start_mach': max_mach,
            'end_mach': max_mach,
            'distance': cruise_distance,
            'start_position': cruise_start,
            'priority': 'thermal_management'
        })
        
        # Descent segment
        descent_distance = distance * 0.2  # 20% for descent
        descent_start = (
            start_point[0] + (end_point[0] - start_point[0]) * 0.8,
            start_point[1] + (end_point[1] - start_point[1]) * 0.8,
            start_point[2]
        )
        segments.append({
            'type': HypersonicFlightPhase.DESCENT,
            'start_mach': max_mach,
            'end_mach': 5.0,
            'distance': descent_distance,
            'start_position': descent_start,
            'priority': 'thermal_recovery'
        })
        
        return segments
    
    def _optimize_trajectory_segment(self, segment: Dict[str, Any],
                                   thermal_constraints: ThermalConstraint,
                                   propulsion_constraints: PropulsionConstraint,
                                   objective: OptimizationObjective) -> List[HypersonicWaypoint]:
        """Optimize a single trajectory segment."""
        # Create Mach profile for segment
        num_points = max(5, int(segment['distance'] / 50000))  # Point every 50 km
        mach_profile = np.linspace(segment['start_mach'], segment['end_mach'], num_points)
        
        # Optimize altitude profile
        constraints = {
            'thermal_max_heat_flux': thermal_constraints.max_heat_flux,
            'propulsion_max_mach': propulsion_constraints.max_air_breathing_mach,
            'objective': objective
        }
        
        altitude_profile = self.altitude_optimizer.optimize_altitude_for_mach_range(
            segment['start_mach'], segment['end_mach'], segment['distance'], constraints
        )
        
        # Create waypoints
        waypoints = []
        for i, (mach, altitude) in enumerate(zip(mach_profile, altitude_profile)):
            # Calculate position along segment
            progress = i / (len(mach_profile) - 1)
            position = np.array([
                segment['start_position'][0] + progress * segment['distance'],
                segment['start_position'][1],
                altitude
            ])
            
            # Determine propulsion mode
            propulsion_mode = self._determine_optimal_propulsion_mode(
                mach, altitude, propulsion_constraints
            )
            
            # Calculate thermal load
            thermal_load = self._calculate_thermal_load(mach, altitude)
            
            # Determine cooling system activation
            cooling_active = thermal_load > thermal_constraints.max_heat_flux * 0.8
            
            waypoint = HypersonicWaypoint(
                waypoint_id=f"wp_{segment['type'].value}_{i}",
                position=position,
                altitude=altitude,
                mach_number=mach,
                heading=0.0,  # Simplified
                flight_phase=segment['type'],
                propulsion_mode=propulsion_mode,
                thermal_load=thermal_load,
                cooling_system_active=cooling_active
            )
            
            waypoints.append(waypoint)
        
        return waypoints
    
    def _determine_optimal_propulsion_mode(self, mach: float, altitude: float,
                                         constraints: PropulsionConstraint) -> PropulsionMode:
        """Determine optimal propulsion mode for given conditions."""
        if mach < 8.0:
            return PropulsionMode.AIR_BREATHING
        elif mach < constraints.max_air_breathing_mach:
            return PropulsionMode.ROCKET_ASSISTED
        elif mach < 25.0:
            return PropulsionMode.TRANSITION
        else:
            return PropulsionMode.PURE_ROCKET
    
    def _calculate_thermal_load(self, mach: float, altitude: float) -> float:
        """Calculate thermal load for given flight conditions."""
        # Simplified thermal load calculation
        # Real implementation would use detailed heat transfer models
        
        temperature, pressure, density = self.altitude_optimizer.get_atmospheric_conditions(altitude)
        
        # Stagnation temperature
        gamma = 1.4
        stagnation_temp = temperature * (1 + (gamma - 1) / 2 * mach**2)
        
        # Heat flux approximation (very simplified)
        # Real calculation would include boundary layer effects, catalytic heating, etc.
        heat_flux = 1000.0 * (mach / 10.0)**3 * (density / 1.225)**0.5
        
        # Apply altitude correction
        altitude_factor = max(0.1, 1.0 - altitude / 100000.0)
        heat_flux *= altitude_factor
        
        return heat_flux
    
    def _generate_profile_arrays(self, profile: HypersonicMissionProfile) -> None:
        """Generate profile arrays from waypoints."""
        if not profile.waypoints:
            return
        
        n_points = len(profile.waypoints)
        profile.altitude_profile = np.array([wp.altitude for wp in profile.waypoints])
        profile.mach_profile = np.array([wp.mach_number for wp in profile.waypoints])
        profile.thermal_load_profile = np.array([wp.thermal_load for wp in profile.waypoints])
        profile.propulsion_mode_schedule = [wp.propulsion_mode for wp in profile.waypoints]
        profile.cooling_system_schedule = [wp.cooling_system_active for wp in profile.waypoints]
        
        # Calculate derived properties
        profile.max_thermal_load = np.max(profile.thermal_load_profile)
        
        # Estimate total duration and fuel consumption
        total_duration = 0.0
        total_fuel = 0.0
        
        for i, waypoint in enumerate(profile.waypoints):
            if i > 0:
                prev_wp = profile.waypoints[i-1]
                distance = np.linalg.norm(waypoint.position - prev_wp.position)
                
                # Average conditions for segment
                avg_mach = (waypoint.mach_number + prev_wp.mach_number) / 2
                avg_altitude = (waypoint.altitude + prev_wp.altitude) / 2
                
                # Calculate segment time
                temp, _, _ = self.altitude_optimizer.get_atmospheric_conditions(avg_altitude)
                speed_of_sound = math.sqrt(1.4 * 287.0 * temp)
                avg_velocity = avg_mach * speed_of_sound
                segment_time = distance / avg_velocity if avg_velocity > 0 else 0.0
                
                total_duration += segment_time
                waypoint.duration = segment_time
                
                # Estimate fuel consumption
                operating_point = CombinedCycleOperatingPoint(
                    altitude=avg_altitude,
                    mach_number=avg_mach,
                    throttle_setting=0.8,
                    rocket_throttle_setting=0.5 if waypoint.propulsion_mode != PropulsionMode.AIR_BREATHING else 0.0,
                    propulsion_mode=waypoint.propulsion_mode
                )
                
                try:
                    _, _, fuel_flow = self.engine.calculate_combined_cycle_fuel_consumption(operating_point)
                    segment_fuel = fuel_flow * segment_time
                    total_fuel += segment_fuel
                    waypoint.fuel_consumption_rate = fuel_flow
                except:
                    # Fallback fuel consumption estimate
                    waypoint.fuel_consumption_rate = 10.0  # kg/s
                    total_fuel += 10.0 * segment_time
        
        profile.total_duration = total_duration
        profile.total_fuel_consumption = total_fuel
    
    def _refine_profile_for_constraints(self, profile: HypersonicMissionProfile,
                                      validation_errors: List[str]) -> HypersonicMissionProfile:
        """Refine profile to address validation errors."""
        self.logger.info("Refining profile to address constraint violations")
        
        # Create refined waypoints
        refined_waypoints = []
        
        for waypoint in profile.waypoints:
            refined_wp = waypoint
            
            # Address thermal constraint violations
            if (profile.thermal_constraints and 
                waypoint.thermal_load > profile.thermal_constraints.max_heat_flux):
                
                # Increase altitude to reduce thermal load
                new_altitude = min(100000.0, waypoint.altitude + 5000.0)
                refined_wp.altitude = new_altitude
                refined_wp.thermal_load = self._calculate_thermal_load(waypoint.mach_number, new_altitude)
                refined_wp.cooling_system_active = True
            
            # Address propulsion constraint violations
            if (profile.propulsion_constraints and
                waypoint.mach_number > profile.propulsion_constraints.max_air_breathing_mach and
                waypoint.propulsion_mode == PropulsionMode.AIR_BREATHING):
                
                refined_wp.propulsion_mode = PropulsionMode.ROCKET_ASSISTED
            
            refined_waypoints.append(refined_wp)
        
        # Update profile
        profile.waypoints = refined_waypoints
        self._generate_profile_arrays(profile)
        
        return profile


class FuelOptimizer:
    """Optimizer for fuel consumption in hypersonic missions."""
    
    def __init__(self, combined_cycle_engine: CombinedCycleEngine):
        """Initialize fuel optimizer."""
        self.engine = combined_cycle_engine
        self.logger = get_logger("hypersonic_fuel_optimizer")
    
    def optimize_fuel_consumption(self, profile: HypersonicMissionProfile,
                                propulsion_constraints: PropulsionConstraint) -> HypersonicMissionProfile:
        """Optimize mission profile for minimum fuel consumption."""
        self.logger.info("Optimizing mission profile for fuel consumption")
        
        optimized_waypoints = []
        
        for waypoint in profile.waypoints:
            # Test different propulsion modes and throttle settings
            best_waypoint = waypoint
            best_fuel_rate = float('inf')
            
            # Test propulsion modes
            test_modes = [PropulsionMode.AIR_BREATHING, PropulsionMode.ROCKET_ASSISTED, PropulsionMode.PURE_ROCKET]
            
            for mode in test_modes:
                if not self._is_valid_propulsion_mode(waypoint.mach_number, waypoint.altitude, mode, propulsion_constraints):
                    continue
                
                # Test throttle settings
                for throttle in [0.6, 0.7, 0.8, 0.9, 1.0]:
                    rocket_throttle = 0.5 if mode != PropulsionMode.AIR_BREATHING else 0.0
                    
                    operating_point = CombinedCycleOperatingPoint(
                        altitude=waypoint.altitude,
                        mach_number=waypoint.mach_number,
                        throttle_setting=throttle,
                        rocket_throttle_setting=rocket_throttle,
                        propulsion_mode=mode
                    )
                    
                    try:
                        _, _, fuel_flow = self.engine.calculate_combined_cycle_fuel_consumption(operating_point)
                        
                        if fuel_flow < best_fuel_rate:
                            best_fuel_rate = fuel_flow
                            best_waypoint = HypersonicWaypoint(
                                waypoint_id=waypoint.waypoint_id,
                                position=waypoint.position,
                                altitude=waypoint.altitude,
                                mach_number=waypoint.mach_number,
                                heading=waypoint.heading,
                                flight_phase=waypoint.flight_phase,
                                propulsion_mode=mode,
                                thermal_load=waypoint.thermal_load,
                                cooling_system_active=waypoint.cooling_system_active,
                                duration=waypoint.duration,
                                fuel_consumption_rate=fuel_flow
                            )
                    except:
                        continue
            
            optimized_waypoints.append(best_waypoint)
        
        # Update profile
        profile.waypoints = optimized_waypoints
        
        # Recalculate profile arrays
        trajectory_optimizer = TrajectoryOptimizer(self.engine)
        trajectory_optimizer._generate_profile_arrays(profile)
        
        return profile
    
    def _is_valid_propulsion_mode(self, mach: float, altitude: float, mode: PropulsionMode,
                                constraints: PropulsionConstraint) -> bool:
        """Check if propulsion mode is valid for given conditions."""
        if mode == PropulsionMode.AIR_BREATHING and mach > constraints.max_air_breathing_mach:
            return False
        
        if mode == PropulsionMode.PURE_ROCKET and altitude < constraints.min_rocket_altitude:
            return False
        
        return True


class HypersonicMissionPlanner:
    """Main hypersonic mission planning system."""
    
    def __init__(self, combined_cycle_engine: CombinedCycleEngine):
        """Initialize hypersonic mission planner."""
        self.engine = combined_cycle_engine
        self.trajectory_optimizer = TrajectoryOptimizer(combined_cycle_engine)
        self.fuel_optimizer = FuelOptimizer(combined_cycle_engine)
        self.logger = get_logger("hypersonic_mission_planner")
        
        self.logger.info("Hypersonic mission planner initialized")
    
    def plan_mission(self, start_point: Tuple[float, float, float],
                    end_point: Tuple[float, float, float],
                    max_mach: float,
                    thermal_constraints: ThermalConstraint,
                    propulsion_constraints: PropulsionConstraint,
                    optimization_objectives: List[OptimizationObjective]) -> HypersonicMissionProfile:
        """Plan complete hypersonic mission."""
        self.logger.info(f"Planning hypersonic mission: Mach {max_mach}")
        
        # Validate inputs
        if max_mach < 4.0 or max_mach > 60.0:
            raise ValidationError(f"Invalid Mach number: {max_mach}. Must be between 4.0 and 60.0")
        
        # Primary optimization objective
        primary_objective = optimization_objectives[0] if optimization_objectives else OptimizationObjective.MINIMIZE_FUEL
        
        # Generate initial trajectory
        profile = self.trajectory_optimizer.optimize_trajectory(
            start_point, end_point, max_mach, thermal_constraints, 
            propulsion_constraints, primary_objective
        )
        
        # Apply secondary optimizations
        for objective in optimization_objectives:
            if objective == OptimizationObjective.MINIMIZE_FUEL:
                profile = self.fuel_optimizer.optimize_fuel_consumption(profile, propulsion_constraints)
            elif objective == OptimizationObjective.MINIMIZE_THERMAL_LOAD:
                profile = self._optimize_for_thermal_management(profile, thermal_constraints)
            elif objective == OptimizationObjective.MINIMIZE_TIME:
                profile = self._optimize_for_minimum_time(profile, propulsion_constraints)
        
        # Final validation
        validation_errors = profile.validate_profile()
        if validation_errors:
            self.logger.warning(f"Final profile validation issues: {validation_errors}")
        
        self.logger.info(f"Mission planning complete. Duration: {profile.total_duration:.0f}s, Fuel: {profile.total_fuel_consumption:.0f}kg")
        
        return profile
    
    def _optimize_for_thermal_management(self, profile: HypersonicMissionProfile,
                                       thermal_constraints: ThermalConstraint) -> HypersonicMissionProfile:
        """Optimize profile for thermal management."""
        self.logger.info("Optimizing for thermal management")
        
        # Identify high thermal load segments
        high_thermal_indices = np.where(profile.thermal_load_profile > thermal_constraints.max_heat_flux * 0.9)[0]
        
        # Add thermal recovery waypoints
        modified_waypoints = list(profile.waypoints)
        
        for idx in reversed(high_thermal_indices):  # Reverse to maintain indices
            if idx < len(modified_waypoints) - 1:
                current_wp = modified_waypoints[idx]
                next_wp = modified_waypoints[idx + 1]
                
                # Insert thermal recovery waypoint
                recovery_wp = HypersonicWaypoint(
                    waypoint_id=f"thermal_recovery_{idx}",
                    position=(current_wp.position + next_wp.position) / 2,
                    altitude=min(100000.0, current_wp.altitude + 10000.0),  # Higher altitude for cooling
                    mach_number=max(5.0, current_wp.mach_number * 0.8),  # Reduced Mach for cooling
                    heading=current_wp.heading,
                    flight_phase=HypersonicFlightPhase.THERMAL_RECOVERY,
                    propulsion_mode=PropulsionMode.AIR_BREATHING,
                    thermal_load=current_wp.thermal_load * 0.5,  # Reduced thermal load
                    cooling_system_active=True,
                    duration=thermal_constraints.recovery_time_required
                )
                
                modified_waypoints.insert(idx + 1, recovery_wp)
        
        profile.waypoints = modified_waypoints
        self.trajectory_optimizer._generate_profile_arrays(profile)
        
        return profile
    
    def _optimize_for_minimum_time(self, profile: HypersonicMissionProfile,
                                 propulsion_constraints: PropulsionConstraint) -> HypersonicMissionProfile:
        """Optimize profile for minimum mission time."""
        self.logger.info("Optimizing for minimum time")
        
        # Increase Mach numbers where possible
        for waypoint in profile.waypoints:
            if waypoint.flight_phase == HypersonicFlightPhase.CRUISE:
                # Try to increase cruise Mach number
                max_sustainable_mach = min(60.0, waypoint.mach_number * 1.2)
                
                # Check if higher Mach is feasible
                if (waypoint.propulsion_mode != PropulsionMode.AIR_BREATHING or 
                    max_sustainable_mach <= propulsion_constraints.max_air_breathing_mach):
                    waypoint.mach_number = max_sustainable_mach
                    
                    # Recalculate thermal load
                    waypoint.thermal_load = self.trajectory_optimizer._calculate_thermal_load(
                        waypoint.mach_number, waypoint.altitude
                    )
        
        # Recalculate profile arrays
        self.trajectory_optimizer._generate_profile_arrays(profile)
        
        return profile
    
    def analyze_mission_feasibility(self, profile: HypersonicMissionProfile) -> Dict[str, Any]:
        """Analyze feasibility of hypersonic mission profile."""
        analysis = {
            'feasible': True,
            'critical_issues': [],
            'warnings': [],
            'performance_metrics': {},
            'recommendations': []
        }
        
        # Check thermal feasibility
        if profile.thermal_constraints:
            max_thermal_load = np.max(profile.thermal_load_profile)
            if max_thermal_load > profile.thermal_constraints.max_heat_flux:
                analysis['feasible'] = False
                analysis['critical_issues'].append(f"Maximum thermal load ({max_thermal_load:.0f} W/m²) exceeds limit ({profile.thermal_constraints.max_heat_flux:.0f} W/m²)")
        
        # Check fuel feasibility
        if profile.propulsion_constraints:
            if profile.total_fuel_consumption > (profile.propulsion_constraints.fuel_capacity_air_breathing + 
                                               profile.propulsion_constraints.fuel_capacity_rocket):
                analysis['feasible'] = False
                analysis['critical_issues'].append(f"Fuel consumption ({profile.total_fuel_consumption:.0f} kg) exceeds capacity")
        
        # Performance metrics
        analysis['performance_metrics'] = {
            'total_duration_hours': profile.total_duration / 3600.0,
            'total_fuel_consumption_kg': profile.total_fuel_consumption,
            'max_thermal_load_MW_per_m2': profile.max_thermal_load / 1e6,
            'max_mach_number': np.max(profile.mach_profile),
            'max_altitude_km': np.max(profile.altitude_profile) / 1000.0,
            'average_mach_number': np.mean(profile.mach_profile),
            'thermal_recovery_time_fraction': len([wp for wp in profile.waypoints if wp.flight_phase == HypersonicFlightPhase.THERMAL_RECOVERY]) / len(profile.waypoints)
        }
        
        # Generate recommendations
        if analysis['performance_metrics']['thermal_recovery_time_fraction'] > 0.3:
            analysis['recommendations'].append("Consider reducing maximum Mach number to decrease thermal recovery requirements")
        
        if analysis['performance_metrics']['total_fuel_consumption_kg'] > 15000:
            analysis['recommendations'].append("High fuel consumption - consider optimizing trajectory for fuel efficiency")
        
        return analysis