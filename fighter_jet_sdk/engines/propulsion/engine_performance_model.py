"""Engine Performance Model for thrust-to-weight calculations and fuel consumption modeling."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math
from enum import Enum

from ...common.enums import FlightRegime
from ...core.logging import get_engine_logger


class EngineType(Enum):
    """Engine type enumeration."""
    TURBOJET = "turbojet"
    TURBOFAN = "turbofan"
    AFTERBURNING_TURBOJET = "afterburning_turbojet"
    AFTERBURNING_TURBOFAN = "afterburning_turbofan"
    VARIABLE_CYCLE = "variable_cycle"
    RAMJET = "ramjet"
    SCRAMJET = "scramjet"


@dataclass
class EngineOperatingPoint:
    """Engine operating point definition."""
    altitude: float  # m
    mach_number: float
    throttle_setting: float  # 0.0 to 1.0
    afterburner_engaged: bool = False
    ambient_temperature: Optional[float] = None  # K
    ambient_pressure: Optional[float] = None  # Pa


@dataclass
class EnginePerformanceData:
    """Engine performance data at a specific operating point."""
    thrust: float  # N
    specific_fuel_consumption: float  # kg/s/N
    fuel_flow_rate: float  # kg/s
    exhaust_gas_temperature: float  # K
    pressure_ratio: float
    bypass_ratio: Optional[float] = None
    efficiency: float = 0.0


@dataclass
class EngineSpecification:
    """Engine specification and design parameters."""
    engine_id: str
    name: str
    engine_type: EngineType
    max_thrust_sea_level: float  # N
    max_thrust_altitude: float  # N at design altitude
    design_altitude: float  # m
    design_mach: float
    bypass_ratio: Optional[float] = None
    pressure_ratio: float = 1.0
    turbine_inlet_temperature: float = 1500.0  # K
    mass: float = 1000.0  # kg
    length: float = 4.0  # m
    diameter: float = 1.0  # m
    afterburner_thrust_multiplier: float = 1.5
    variable_cycle_modes: List[str] = field(default_factory=list)


class EnginePerformanceModel:
    """Advanced engine performance modeling with thrust-to-weight calculations."""
    
    def __init__(self, engine_spec: EngineSpecification):
        """Initialize engine performance model."""
        self.engine_spec = engine_spec
        self.logger = get_engine_logger('propulsion.performance')
        
        # Performance maps and lookup tables
        self.thrust_map: Dict[Tuple[float, float], float] = {}
        self.sfc_map: Dict[Tuple[float, float], float] = {}
        
        # Initialize standard atmosphere model
        self._initialize_atmosphere_model()
        
        # Generate performance maps
        self._generate_performance_maps()
    
    def _initialize_atmosphere_model(self) -> None:
        """Initialize standard atmosphere model for performance calculations."""
        # Standard atmosphere constants
        self.sea_level_pressure = 101325.0  # Pa
        self.sea_level_temperature = 288.15  # K
        self.sea_level_density = 1.225  # kg/m³
        self.temperature_lapse_rate = 0.0065  # K/m
        self.gas_constant = 287.0  # J/(kg·K)
        self.gamma = 1.4  # Specific heat ratio for air
    
    def _generate_performance_maps(self) -> None:
        """Generate engine performance maps across flight envelope."""
        self.logger.info(f"Generating performance maps for {self.engine_spec.name}")
        
        # Define altitude and Mach number ranges
        altitudes = [0, 3000, 6000, 9000, 12000, 15000, 18000]  # m
        mach_numbers = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0, 2.5]
        
        for altitude in altitudes:
            for mach in mach_numbers:
                # Skip invalid combinations
                if self._is_valid_operating_point(altitude, mach):
                    thrust_ratio = self._calculate_thrust_ratio(altitude, mach)
                    sfc_ratio = self._calculate_sfc_ratio(altitude, mach)
                    
                    self.thrust_map[(altitude, mach)] = thrust_ratio
                    self.sfc_map[(altitude, mach)] = sfc_ratio
    
    def _is_valid_operating_point(self, altitude: float, mach: float) -> bool:
        """Check if operating point is valid for engine type."""
        # Basic validity checks
        if altitude < 0 or altitude > 25000:  # Reasonable altitude limits
            return False
        
        if mach < 0:
            return False
        
        # Engine-specific limits
        if self.engine_spec.engine_type in [EngineType.TURBOJET, EngineType.TURBOFAN]:
            return mach <= 2.5
        elif self.engine_spec.engine_type == EngineType.RAMJET:
            return mach >= 1.5 and mach <= 5.0
        elif self.engine_spec.engine_type == EngineType.SCRAMJET:
            return mach >= 4.0 and mach <= 15.0
        
        return True
    
    def get_atmospheric_conditions(self, altitude: float) -> Tuple[float, float, float]:
        """Get atmospheric conditions at specified altitude."""
        if altitude <= 11000:  # Troposphere
            temperature = self.sea_level_temperature - self.temperature_lapse_rate * altitude
            pressure = self.sea_level_pressure * (temperature / self.sea_level_temperature) ** (9.80665 / (self.gas_constant * self.temperature_lapse_rate))
        else:  # Stratosphere (simplified)
            temperature = 216.65  # K (constant in lower stratosphere)
            pressure = 22632 * math.exp(-9.80665 * (altitude - 11000) / (self.gas_constant * temperature))
        
        density = pressure / (self.gas_constant * temperature)
        return temperature, pressure, density
    
    def _calculate_thrust_ratio(self, altitude: float, mach: float) -> float:
        """Calculate thrust ratio relative to sea level static conditions."""
        temperature, pressure, density = self.get_atmospheric_conditions(altitude)
        
        # Temperature ratio
        theta = temperature / self.sea_level_temperature
        
        # Pressure ratio
        delta = pressure / self.sea_level_pressure
        
        # Basic thrust lapse calculation
        if self.engine_spec.engine_type in [EngineType.TURBOJET, EngineType.AFTERBURNING_TURBOJET]:
            # Turbojet thrust lapse
            thrust_ratio = delta * (1 + 0.2 * mach**2)**0.5
        elif self.engine_spec.engine_type in [EngineType.TURBOFAN, EngineType.AFTERBURNING_TURBOFAN]:
            # Turbofan thrust lapse (simplified)
            thrust_ratio = delta * (1 + 0.15 * mach**2)**0.5
        elif self.engine_spec.engine_type == EngineType.RAMJET:
            # Ramjet thrust increases with Mach number
            thrust_ratio = delta * mach**2 * (1 - 1/(1 + 0.2 * mach**2)**3.5)
        elif self.engine_spec.engine_type == EngineType.SCRAMJET:
            # Scramjet performance (simplified)
            thrust_ratio = delta * mach * (1 - 1/mach**2)
        else:
            thrust_ratio = delta
        
        return max(0.0, thrust_ratio)
    
    def _calculate_sfc_ratio(self, altitude: float, mach: float) -> float:
        """Calculate specific fuel consumption ratio."""
        temperature, _, _ = self.get_atmospheric_conditions(altitude)
        
        # Temperature ratio
        theta = temperature / self.sea_level_temperature
        
        # SFC generally increases with altitude and Mach number
        if self.engine_spec.engine_type in [EngineType.TURBOJET, EngineType.AFTERBURNING_TURBOJET]:
            sfc_ratio = theta**0.5 * (1 + 0.1 * mach)
        elif self.engine_spec.engine_type in [EngineType.TURBOFAN, EngineType.AFTERBURNING_TURBOFAN]:
            sfc_ratio = theta**0.5 * (1 + 0.05 * mach)
        elif self.engine_spec.engine_type == EngineType.RAMJET:
            sfc_ratio = 1.0 + 0.2 * mach  # Ramjet SFC increases with Mach
        elif self.engine_spec.engine_type == EngineType.SCRAMJET:
            sfc_ratio = 1.0 + 0.1 * mach  # Better SFC than ramjet
        else:
            sfc_ratio = 1.0
        
        return max(0.1, sfc_ratio)
    
    def calculate_thrust(self, operating_point: EngineOperatingPoint) -> float:
        """Calculate thrust at specified operating point."""
        # Check if operating point is valid
        if not self._is_valid_operating_point(operating_point.altitude, operating_point.mach_number):
            return 0.0
        
        # Get base thrust ratio from performance map
        thrust_ratio = self._interpolate_performance_map(
            self.thrust_map, operating_point.altitude, operating_point.mach_number
        )
        
        # Apply throttle setting
        thrust_ratio *= operating_point.throttle_setting
        
        # Apply afterburner if engaged
        if operating_point.afterburner_engaged and self._has_afterburner():
            thrust_ratio *= self.engine_spec.afterburner_thrust_multiplier
        
        # Calculate absolute thrust - use altitude thrust for ramjets
        if self.engine_spec.engine_type in [EngineType.RAMJET, EngineType.SCRAMJET]:
            base_thrust = self.engine_spec.max_thrust_altitude
        else:
            base_thrust = self.engine_spec.max_thrust_sea_level
        
        return max(0.0, base_thrust * thrust_ratio)
    
    def calculate_fuel_consumption(self, operating_point: EngineOperatingPoint, thrust: Optional[float] = None) -> float:
        """Calculate fuel consumption rate at operating point."""
        if thrust is None:
            thrust = self.calculate_thrust(operating_point)
        
        # Get SFC ratio from performance map
        sfc_ratio = self._interpolate_performance_map(
            self.sfc_map, operating_point.altitude, operating_point.mach_number
        )
        
        # Base SFC depends on engine type
        base_sfc = self._get_base_sfc()
        
        # Apply afterburner penalty if engaged
        if operating_point.afterburner_engaged and self._has_afterburner():
            sfc_ratio *= 2.5  # Afterburner significantly increases fuel consumption
        
        # Calculate fuel flow rate
        sfc = base_sfc * sfc_ratio
        fuel_flow = thrust * sfc
        
        return fuel_flow
    
    def _get_base_sfc(self) -> float:
        """Get base specific fuel consumption for engine type."""
        sfc_values = {
            EngineType.TURBOJET: 0.00002,  # kg/s/N
            EngineType.TURBOFAN: 0.000015,
            EngineType.AFTERBURNING_TURBOJET: 0.000025,
            EngineType.AFTERBURNING_TURBOFAN: 0.00002,
            EngineType.VARIABLE_CYCLE: 0.000018,
            EngineType.RAMJET: 0.00003,
            EngineType.SCRAMJET: 0.000025
        }
        return sfc_values.get(self.engine_spec.engine_type, 0.00002)
    
    def _has_afterburner(self) -> bool:
        """Check if engine has afterburner capability."""
        return self.engine_spec.engine_type in [
            EngineType.AFTERBURNING_TURBOJET,
            EngineType.AFTERBURNING_TURBOFAN
        ]
    
    def _interpolate_performance_map(self, performance_map: Dict[Tuple[float, float], float], 
                                   altitude: float, mach: float) -> float:
        """Interpolate performance data from map."""
        # Find closest points in performance map
        closest_points = []
        min_distance = float('inf')
        
        for (alt, mach_key), value in performance_map.items():
            distance = ((altitude - alt)**2 + (mach - mach_key)**2)**0.5
            if distance < min_distance:
                min_distance = distance
                closest_points = [(alt, mach_key, value)]
            elif abs(distance - min_distance) < 1e-6:
                closest_points.append((alt, mach_key, value))
        
        if not closest_points:
            return 1.0  # Default ratio
        
        # Simple interpolation (could be improved with bilinear interpolation)
        if len(closest_points) == 1:
            return closest_points[0][2]
        
        # Average of closest points
        return sum(point[2] for point in closest_points) / len(closest_points)
    
    def calculate_thrust_to_weight_ratio(self, aircraft_mass: float, 
                                       operating_point: EngineOperatingPoint,
                                       num_engines: int = 1) -> float:
        """Calculate thrust-to-weight ratio for aircraft configuration."""
        total_thrust = self.calculate_thrust(operating_point) * num_engines
        weight = aircraft_mass * 9.80665  # Convert mass to weight (N)
        
        return total_thrust / weight if weight > 0 else 0.0
    
    def calculate_range_fuel_consumption(self, flight_profile: List[EngineOperatingPoint],
                                       flight_times: List[float]) -> float:
        """Calculate total fuel consumption for a flight profile."""
        if len(flight_profile) != len(flight_times):
            raise ValueError("Flight profile and time arrays must have same length")
        
        total_fuel = 0.0
        
        for operating_point, time_segment in zip(flight_profile, flight_times):
            fuel_flow = self.calculate_fuel_consumption(operating_point)
            total_fuel += fuel_flow * time_segment
        
        return total_fuel
    
    def optimize_cruise_conditions(self, altitude_range: Tuple[float, float],
                                 mach_range: Tuple[float, float],
                                 aircraft_mass: float) -> Tuple[float, float, float]:
        """Optimize cruise altitude and Mach number for minimum fuel consumption."""
        best_altitude = altitude_range[0]
        best_mach = mach_range[0]
        best_sfc = float('inf')
        
        # Grid search for optimal conditions
        alt_step = (altitude_range[1] - altitude_range[0]) / 10
        mach_step = (mach_range[1] - mach_range[0]) / 10
        
        for i in range(11):
            altitude = altitude_range[0] + i * alt_step
            for j in range(11):
                mach = mach_range[0] + j * mach_step
                
                operating_point = EngineOperatingPoint(
                    altitude=altitude,
                    mach_number=mach,
                    throttle_setting=0.8  # Typical cruise setting
                )
                
                if self._is_valid_operating_point(altitude, mach):
                    fuel_flow = self.calculate_fuel_consumption(operating_point)
                    thrust = self.calculate_thrust(operating_point)
                    
                    # Calculate specific fuel consumption per unit thrust
                    if thrust > 0:
                        sfc = fuel_flow / thrust
                        if sfc < best_sfc:
                            best_sfc = sfc
                            best_altitude = altitude
                            best_mach = mach
        
        return best_altitude, best_mach, best_sfc
    
    def get_performance_envelope(self) -> Dict[str, Any]:
        """Get complete performance envelope data."""
        envelope = {
            'engine_spec': {
                'name': self.engine_spec.name,
                'type': self.engine_spec.engine_type.value,
                'max_thrust_sl': self.engine_spec.max_thrust_sea_level,
                'mass': self.engine_spec.mass
            },
            'operating_limits': {
                'max_altitude': 25000,  # m
                'max_mach': self._get_max_mach(),
                'min_mach': self._get_min_mach()
            },
            'performance_data': {}
        }
        
        # Add sample performance points
        test_points = [
            (0, 0.0), (0, 0.9), (0, 1.2),
            (10000, 0.8), (10000, 1.5), (10000, 2.0),
            (15000, 0.9), (15000, 1.8), (15000, 2.2)
        ]
        
        for altitude, mach in test_points:
            if self._is_valid_operating_point(altitude, mach):
                op_point = EngineOperatingPoint(
                    altitude=altitude,
                    mach_number=mach,
                    throttle_setting=1.0
                )
                
                thrust = self.calculate_thrust(op_point)
                fuel_flow = self.calculate_fuel_consumption(op_point)
                
                envelope['performance_data'][f'alt_{altitude}_mach_{mach}'] = {
                    'thrust': thrust,
                    'fuel_flow': fuel_flow,
                    'sfc': fuel_flow / thrust if thrust > 0 else 0
                }
        
        return envelope
    
    def _get_max_mach(self) -> float:
        """Get maximum Mach number for engine type."""
        max_mach_values = {
            EngineType.TURBOJET: 2.5,
            EngineType.TURBOFAN: 2.2,
            EngineType.AFTERBURNING_TURBOJET: 3.0,
            EngineType.AFTERBURNING_TURBOFAN: 2.8,
            EngineType.VARIABLE_CYCLE: 3.5,
            EngineType.RAMJET: 5.0,
            EngineType.SCRAMJET: 15.0
        }
        return max_mach_values.get(self.engine_spec.engine_type, 2.5)
    
    def _get_min_mach(self) -> float:
        """Get minimum Mach number for engine type."""
        min_mach_values = {
            EngineType.TURBOJET: 0.0,
            EngineType.TURBOFAN: 0.0,
            EngineType.AFTERBURNING_TURBOJET: 0.0,
            EngineType.AFTERBURNING_TURBOFAN: 0.0,
            EngineType.VARIABLE_CYCLE: 0.0,
            EngineType.RAMJET: 1.5,
            EngineType.SCRAMJET: 4.0
        }
        return min_mach_values.get(self.engine_spec.engine_type, 0.0)
    
    def validate_engine_specification(self) -> List[str]:
        """Validate engine specification and return list of errors."""
        errors = []
        
        if self.engine_spec.max_thrust_sea_level <= 0:
            errors.append("Maximum thrust must be positive")
        
        if self.engine_spec.mass <= 0:
            errors.append("Engine mass must be positive")
        
        if self.engine_spec.pressure_ratio < 1.0:
            errors.append("Pressure ratio must be >= 1.0")
        
        if self.engine_spec.turbine_inlet_temperature <= 0:
            errors.append("Turbine inlet temperature must be positive")
        
        if self.engine_spec.afterburner_thrust_multiplier < 1.0:
            errors.append("Afterburner thrust multiplier must be >= 1.0")
        
        # Engine type specific validation
        if self.engine_spec.engine_type in [EngineType.TURBOFAN, EngineType.AFTERBURNING_TURBOFAN]:
            if self.engine_spec.bypass_ratio is None or self.engine_spec.bypass_ratio < 0:
                errors.append("Turbofan engines must have valid bypass ratio")
        
        return errors