"""Combined-cycle engine performance model for extreme hypersonic flight."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math
from enum import Enum

from .engine_performance_model import (
    EnginePerformanceModel, 
    EngineOperatingPoint, 
    EnginePerformanceData, 
    EngineSpecification,
    EngineType
)
from ...common.enums import ExtremePropulsionType, PlasmaRegime
from ...core.logging import get_engine_logger


class PropulsionMode(Enum):
    """Propulsion mode for combined-cycle engines."""
    AIR_BREATHING = "air_breathing"
    ROCKET_ASSISTED = "rocket_assisted"
    PURE_ROCKET = "pure_rocket"
    TRANSITION = "transition"


@dataclass
class CombinedCycleOperatingPoint(EngineOperatingPoint):
    """Extended operating point for combined-cycle engines."""
    propulsion_mode: PropulsionMode = PropulsionMode.AIR_BREATHING
    rocket_throttle_setting: float = 0.0  # 0.0 to 1.0
    fuel_mixture_ratio: float = 1.0  # Air-breathing to rocket fuel ratio
    plasma_effects_enabled: bool = False
    stagnation_temperature: Optional[float] = None  # K


@dataclass
class CombinedCyclePerformanceData(EnginePerformanceData):
    """Extended performance data for combined-cycle engines."""
    air_breathing_thrust: float = 0.0  # N
    rocket_thrust: float = 0.0  # N
    total_thrust: float = 0.0  # N
    air_breathing_fuel_flow: float = 0.0  # kg/s
    rocket_fuel_flow: float = 0.0  # kg/s
    total_fuel_flow: float = 0.0  # kg/s
    propulsion_mode: PropulsionMode = PropulsionMode.AIR_BREATHING
    transition_efficiency: float = 1.0
    plasma_interaction_factor: float = 1.0
    dissociation_losses: float = 0.0  # Fraction of thrust lost to dissociation


@dataclass
class CombinedCycleSpecification(EngineSpecification):
    """Extended specification for combined-cycle engines."""
    extreme_propulsion_type: ExtremePropulsionType = ExtremePropulsionType.COMBINED_CYCLE_AIRBREATHING
    transition_mach_number: float = 12.0  # Mach number for mode transition
    rocket_specific_impulse: float = 450.0  # s
    air_breathing_specific_impulse: float = 3000.0  # s
    max_rocket_thrust: float = 0.0  # N
    rocket_fuel_capacity: float = 0.0  # kg
    air_breathing_fuel_capacity: float = 0.0  # kg
    plasma_interaction_threshold: float = 25.0  # Mach number for plasma effects
    max_stagnation_temperature: float = 60000.0  # K
    dissociation_onset_temperature: float = 4000.0  # K
    
    def __post_init__(self):
        """Set default engine type for combined-cycle engines."""
        if not hasattr(self, 'engine_type') or self.engine_type is None:
            self.engine_type = EngineType.SCRAMJET


class CombinedCycleEngine(EnginePerformanceModel):
    """Combined-cycle engine performance model for Mach 60 flight."""
    
    def __init__(self, engine_spec: CombinedCycleSpecification):
        """Initialize combined-cycle engine performance model."""
        # Initialize base class with converted specification
        base_spec = self._convert_to_base_spec(engine_spec)
        super().__init__(base_spec)
        
        self.combined_spec = engine_spec
        self.logger = get_engine_logger('propulsion.combined_cycle')
        
        # Combined-cycle specific performance maps
        self.mode_transition_map: Dict[Tuple[float, float], PropulsionMode] = {}
        self.rocket_thrust_map: Dict[Tuple[float, float], float] = {}
        self.plasma_effects_map: Dict[Tuple[float, float], float] = {}
        self.extreme_temp_effects_map: Dict[Tuple[float, float], float] = {}
        
        # Generate combined-cycle performance maps
        self._generate_combined_cycle_maps()
        
        self.logger.info(f"Initialized combined-cycle engine: {engine_spec.name}")
    
    def _convert_to_base_spec(self, combined_spec: CombinedCycleSpecification) -> EngineSpecification:
        """Convert combined-cycle spec to base engine spec."""
        return EngineSpecification(
            engine_id=combined_spec.engine_id,
            name=combined_spec.name,
            engine_type=EngineType.SCRAMJET,  # Base type for air-breathing mode
            max_thrust_sea_level=combined_spec.max_thrust_sea_level,
            max_thrust_altitude=combined_spec.max_thrust_altitude,
            design_altitude=combined_spec.design_altitude,
            design_mach=combined_spec.design_mach,
            pressure_ratio=combined_spec.pressure_ratio,
            turbine_inlet_temperature=combined_spec.turbine_inlet_temperature,
            mass=combined_spec.mass,
            length=combined_spec.length,
            diameter=combined_spec.diameter
        )
    
    def _generate_combined_cycle_maps(self) -> None:
        """Generate performance maps for combined-cycle operation."""
        self.logger.info("Generating combined-cycle performance maps")
        
        # Extended altitude and Mach ranges for extreme hypersonic flight
        altitudes = [30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]  # m
        mach_numbers = [4.0, 8.0, 12.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        
        for altitude in altitudes:
            for mach in mach_numbers:
                if self._is_valid_combined_cycle_point(altitude, mach):
                    # Determine propulsion mode
                    mode = self._determine_propulsion_mode(mach)
                    self.mode_transition_map[(altitude, mach)] = mode
                    
                    # Calculate rocket thrust ratio
                    rocket_ratio = self._calculate_rocket_thrust_ratio(altitude, mach, mode)
                    self.rocket_thrust_map[(altitude, mach)] = rocket_ratio
                    
                    # Calculate plasma effects
                    plasma_factor = self._calculate_plasma_effects_factor(mach)
                    self.plasma_effects_map[(altitude, mach)] = plasma_factor
                    
                    # Calculate extreme temperature effects
                    temp_effects = self._calculate_extreme_temperature_effects(altitude, mach)
                    self.extreme_temp_effects_map[(altitude, mach)] = temp_effects
    
    def _is_valid_combined_cycle_point(self, altitude: float, mach: float) -> bool:
        """Check if operating point is valid for combined-cycle engine."""
        # Combined-cycle engines operate at high altitudes and Mach numbers
        if altitude < 30000 or altitude > 100000:  # 30-100 km altitude range
            return False
        
        if mach < 4.0 or mach > 60.0:  # Mach 4-60 range
            return False
        
        return True
    
    def _is_valid_operating_point(self, altitude: float, mach: float) -> bool:
        """Override base class validation for combined-cycle engines."""
        return self._is_valid_combined_cycle_point(altitude, mach)
    
    def _determine_propulsion_mode(self, mach: float) -> PropulsionMode:
        """Determine propulsion mode based on Mach number."""
        if mach < self.combined_spec.transition_mach_number:
            return PropulsionMode.AIR_BREATHING
        elif mach < self.combined_spec.transition_mach_number + 2.0:
            return PropulsionMode.TRANSITION
        elif mach < 40.0:
            return PropulsionMode.ROCKET_ASSISTED
        else:
            return PropulsionMode.PURE_ROCKET
    
    def _calculate_rocket_thrust_ratio(self, altitude: float, mach: float, mode: PropulsionMode) -> float:
        """Calculate rocket thrust contribution ratio."""
        if mode == PropulsionMode.AIR_BREATHING:
            return 0.0
        elif mode == PropulsionMode.TRANSITION:
            # Linear transition between air-breathing and rocket
            transition_progress = (mach - self.combined_spec.transition_mach_number) / 2.0
            return min(1.0, max(0.0, transition_progress))
        elif mode == PropulsionMode.ROCKET_ASSISTED:
            # Rocket provides increasing contribution at higher Mach
            return min(1.0, (mach - self.combined_spec.transition_mach_number) / 20.0)
        else:  # PURE_ROCKET
            return 1.0
    
    def _calculate_plasma_effects_factor(self, mach: float) -> float:
        """Calculate plasma interaction effects on performance."""
        if mach < self.combined_spec.plasma_interaction_threshold:
            return 1.0
        
        # Plasma effects reduce performance at extreme Mach numbers
        plasma_factor = 1.0 - 0.1 * (mach - self.combined_spec.plasma_interaction_threshold) / 35.0
        return max(0.5, plasma_factor)  # Minimum 50% performance retention
    
    def calculate_combined_cycle_thrust(self, operating_point: CombinedCycleOperatingPoint) -> float:
        """Calculate total thrust for combined-cycle operation."""
        if not self._is_valid_combined_cycle_point(operating_point.altitude, operating_point.mach_number):
            return 0.0
        
        # Determine propulsion mode if not specified
        if operating_point.propulsion_mode == PropulsionMode.AIR_BREATHING:
            mode = self._determine_propulsion_mode(operating_point.mach_number)
        else:
            mode = operating_point.propulsion_mode
        
        # Calculate air-breathing thrust component
        air_breathing_thrust = self._calculate_air_breathing_thrust(operating_point, mode)
        
        # Calculate rocket thrust component
        rocket_thrust = self._calculate_rocket_thrust(operating_point, mode)
        
        # Apply plasma effects
        plasma_factor = self._interpolate_performance_map(
            self.plasma_effects_map, operating_point.altitude, operating_point.mach_number
        )
        
        # Calculate total thrust
        total_thrust = (air_breathing_thrust + rocket_thrust) * plasma_factor
        
        return max(0.0, total_thrust)
    
    def _calculate_air_breathing_thrust(self, operating_point: CombinedCycleOperatingPoint, 
                                     mode: PropulsionMode) -> float:
        """Calculate air-breathing thrust component."""
        if mode == PropulsionMode.PURE_ROCKET:
            return 0.0
        
        # Use base class thrust calculation for air-breathing mode
        base_operating_point = EngineOperatingPoint(
            altitude=operating_point.altitude,
            mach_number=operating_point.mach_number,
            throttle_setting=operating_point.throttle_setting,
            afterburner_engaged=operating_point.afterburner_engaged
        )
        
        base_thrust = super().calculate_thrust(base_operating_point)
        
        # Apply mode-specific efficiency
        if mode == PropulsionMode.TRANSITION:
            # Reduced efficiency during transition
            transition_factor = 1.0 - self._calculate_rocket_thrust_ratio(
                operating_point.altitude, operating_point.mach_number, mode
            )
            base_thrust *= transition_factor * 0.8  # 20% efficiency loss during transition
        elif mode == PropulsionMode.ROCKET_ASSISTED:
            # Air-breathing contribution decreases with Mach number
            air_breathing_factor = max(0.1, 1.0 - (operating_point.mach_number - 12.0) / 28.0)
            base_thrust *= air_breathing_factor
        
        return base_thrust
    
    def _calculate_rocket_thrust(self, operating_point: CombinedCycleOperatingPoint, 
                               mode: PropulsionMode) -> float:
        """Calculate rocket thrust component."""
        if mode == PropulsionMode.AIR_BREATHING:
            return 0.0
        
        # Get rocket thrust ratio from performance map
        rocket_ratio = self._interpolate_performance_map(
            self.rocket_thrust_map, operating_point.altitude, operating_point.mach_number
        )
        
        # Calculate base rocket thrust
        base_rocket_thrust = self.combined_spec.max_rocket_thrust * rocket_ratio
        
        # Apply rocket throttle setting
        rocket_thrust = base_rocket_thrust * operating_point.rocket_throttle_setting
        
        # Apply atmospheric effects (rocket performance less affected by atmosphere)
        _, pressure, _ = self.get_atmospheric_conditions(operating_point.altitude)
        pressure_ratio = pressure / self.sea_level_pressure
        
        # Rocket thrust increases slightly in vacuum
        vacuum_factor = 1.0 + 0.2 * (1.0 - pressure_ratio)
        rocket_thrust *= vacuum_factor
        
        return rocket_thrust
    
    def calculate_combined_cycle_fuel_consumption(self, operating_point: CombinedCycleOperatingPoint) -> Tuple[float, float, float]:
        """Calculate fuel consumption for both propulsion modes."""
        # Calculate thrust components
        air_breathing_thrust = self._calculate_air_breathing_thrust(
            operating_point, operating_point.propulsion_mode
        )
        rocket_thrust = self._calculate_rocket_thrust(
            operating_point, operating_point.propulsion_mode
        )
        
        # Calculate air-breathing fuel consumption
        if air_breathing_thrust > 0:
            base_operating_point = EngineOperatingPoint(
                altitude=operating_point.altitude,
                mach_number=operating_point.mach_number,
                throttle_setting=operating_point.throttle_setting
            )
            air_breathing_fuel_flow = super().calculate_fuel_consumption(
                base_operating_point, air_breathing_thrust
            )
        else:
            air_breathing_fuel_flow = 0.0
        
        # Calculate rocket fuel consumption
        if rocket_thrust > 0:
            # Rocket specific fuel consumption
            rocket_sfc = 1.0 / self.combined_spec.rocket_specific_impulse / 9.80665  # kg/s/N
            rocket_fuel_flow = rocket_thrust * rocket_sfc
        else:
            rocket_fuel_flow = 0.0
        
        total_fuel_flow = air_breathing_fuel_flow + rocket_fuel_flow
        
        return air_breathing_fuel_flow, rocket_fuel_flow, total_fuel_flow
    
    def calculate_stagnation_temperature(self, mach: float, altitude: float) -> float:
        """Calculate stagnation temperature for extreme hypersonic conditions."""
        temperature, _, _ = self.get_atmospheric_conditions(altitude)
        
        # Ensure temperature is positive (atmospheric model might have issues at extreme altitudes)
        if temperature <= 0:
            temperature = 216.65  # Use stratospheric temperature as fallback
        
        # Stagnation temperature calculation with real gas effects
        gamma = 1.4  # Specific heat ratio (simplified)
        
        # For extreme hypersonic conditions, use more accurate formula
        if mach > 10.0:
            # Account for real gas effects and dissociation - much higher temperature rise
            stagnation_temp = temperature * (1 + 0.2 * mach**2) * (1 + 0.1 * mach)  # Enhanced formula
            
            # Apply dissociation effects at very high temperatures
            if stagnation_temp > self.combined_spec.dissociation_onset_temperature:
                # Dissociation absorbs energy, limiting temperature rise
                excess_temp = stagnation_temp - self.combined_spec.dissociation_onset_temperature
                dissociation_factor = max(0.1, 1.0 - 0.3 * (excess_temp / 20000.0))  # Ensure positive factor
                stagnation_temp = self.combined_spec.dissociation_onset_temperature + excess_temp * dissociation_factor
        else:
            # Standard stagnation temperature formula
            stagnation_temp = temperature * (1 + (gamma - 1) / 2 * mach**2)
        
        return max(temperature, min(stagnation_temp, self.combined_spec.max_stagnation_temperature))
    
    def calculate_dissociation_losses(self, stagnation_temperature: float) -> float:
        """Calculate thrust losses due to gas dissociation."""
        if stagnation_temperature < self.combined_spec.dissociation_onset_temperature:
            return 0.0
        
        # Dissociation losses increase with temperature
        temp_excess = stagnation_temperature - self.combined_spec.dissociation_onset_temperature
        max_temp_excess = self.combined_spec.max_stagnation_temperature - self.combined_spec.dissociation_onset_temperature
        
        if max_temp_excess <= 0:
            return 0.0
        
        # Losses range from 0% to 40% at maximum temperature
        dissociation_fraction = 0.4 * (temp_excess / max_temp_excess)
        return min(0.4, dissociation_fraction)
    
    def get_combined_cycle_performance(self, operating_point: CombinedCycleOperatingPoint) -> CombinedCyclePerformanceData:
        """Get complete performance data for combined-cycle operation."""
        # Calculate thrust components
        air_breathing_thrust = self._calculate_air_breathing_thrust(
            operating_point, operating_point.propulsion_mode
        )
        rocket_thrust = self._calculate_rocket_thrust(
            operating_point, operating_point.propulsion_mode
        )
        
        # Apply plasma effects
        plasma_factor = self._interpolate_performance_map(
            self.plasma_effects_map, operating_point.altitude, operating_point.mach_number
        )
        
        total_thrust = (air_breathing_thrust + rocket_thrust) * plasma_factor
        
        # Calculate fuel consumption
        air_fuel_flow, rocket_fuel_flow, total_fuel_flow = self.calculate_combined_cycle_fuel_consumption(operating_point)
        
        # Calculate stagnation temperature
        stagnation_temp = self.calculate_stagnation_temperature(
            operating_point.mach_number, operating_point.altitude
        )
        
        # Calculate dissociation losses
        dissociation_losses = self.calculate_dissociation_losses(stagnation_temp)
        
        # Calculate specific fuel consumption
        sfc = total_fuel_flow / total_thrust if total_thrust > 0 else 0.0
        
        # Get atmospheric conditions
        temperature, pressure, _ = self.get_atmospheric_conditions(operating_point.altitude)
        
        return CombinedCyclePerformanceData(
            thrust=total_thrust,
            specific_fuel_consumption=sfc,
            fuel_flow_rate=total_fuel_flow,
            exhaust_gas_temperature=stagnation_temp,
            pressure_ratio=self.combined_spec.pressure_ratio,
            efficiency=plasma_factor * (1.0 - dissociation_losses),
            air_breathing_thrust=air_breathing_thrust * plasma_factor,
            rocket_thrust=rocket_thrust * plasma_factor,
            total_thrust=total_thrust,
            air_breathing_fuel_flow=air_fuel_flow,
            rocket_fuel_flow=rocket_fuel_flow,
            total_fuel_flow=total_fuel_flow,
            propulsion_mode=operating_point.propulsion_mode,
            transition_efficiency=plasma_factor,
            plasma_interaction_factor=plasma_factor,
            dissociation_losses=dissociation_losses
        )
    
    def optimize_mode_transition(self, altitude: float, target_mach: float) -> Tuple[float, PropulsionMode]:
        """Optimize propulsion mode transition for given conditions."""
        best_mach = self.combined_spec.transition_mach_number
        best_mode = PropulsionMode.TRANSITION
        best_efficiency = 0.0
        
        # Test different transition points
        test_machs = [
            self.combined_spec.transition_mach_number - 2.0,
            self.combined_spec.transition_mach_number,
            self.combined_spec.transition_mach_number + 2.0
        ]
        
        for test_mach in test_machs:
            if test_mach > 0 and test_mach < target_mach:
                operating_point = CombinedCycleOperatingPoint(
                    altitude=altitude,
                    mach_number=test_mach,
                    throttle_setting=1.0,
                    rocket_throttle_setting=0.5,
                    propulsion_mode=PropulsionMode.TRANSITION
                )
                
                performance = self.get_combined_cycle_performance(operating_point)
                efficiency = performance.efficiency
                
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_mach = test_mach
                    best_mode = PropulsionMode.TRANSITION
        
        return best_mach, best_mode
    
    def _calculate_extreme_temperature_effects(self, altitude: float, mach: float) -> float:
        """Calculate performance effects due to extreme temperatures > 50,000K."""
        stagnation_temp = self.calculate_stagnation_temperature(mach, altitude)
        
        if stagnation_temp < 50000.0:
            return 1.0  # No extreme temperature effects
        
        # Calculate temperature-dependent performance degradation
        temp_excess = stagnation_temp - 50000.0
        max_temp_excess = self.combined_spec.max_stagnation_temperature - 50000.0
        
        if max_temp_excess <= 0:
            return 1.0
        
        # Performance degradation due to extreme temperatures
        # Includes effects of gas dissociation, ionization, and material limits
        degradation_factor = 0.3 * (temp_excess / max_temp_excess)  # Up to 30% degradation
        performance_factor = 1.0 - degradation_factor
        
        return max(0.4, performance_factor)  # Minimum 40% performance retention
    
    def calculate_dissociation_effects_on_thrust(self, stagnation_temperature: float, 
                                                base_thrust: float) -> Tuple[float, float]:
        """Calculate thrust reduction and specific impulse effects due to gas dissociation."""
        if stagnation_temperature < self.combined_spec.dissociation_onset_temperature:
            return base_thrust, 1.0  # No dissociation effects
        
        # Calculate dissociation fraction
        temp_excess = stagnation_temperature - self.combined_spec.dissociation_onset_temperature
        max_temp_excess = self.combined_spec.max_stagnation_temperature - self.combined_spec.dissociation_onset_temperature
        
        if max_temp_excess <= 0:
            return base_thrust, 1.0
        
        # Dissociation reduces effective molecular weight and specific heat ratio
        dissociation_fraction = min(0.8, temp_excess / max_temp_excess)  # Up to 80% dissociation
        
        # Thrust reduction due to dissociation
        # Dissociated gases have lower molecular weight but also lower specific heat ratio
        molecular_weight_factor = 1.0 - 0.5 * dissociation_fraction  # Lighter gases
        specific_heat_factor = 1.0 - 0.3 * dissociation_fraction     # Lower gamma
        
        # Net effect on thrust (complex interaction)
        thrust_factor = molecular_weight_factor * specific_heat_factor * (1.0 - 0.2 * dissociation_fraction)
        
        # Specific impulse is generally reduced due to dissociation
        isp_factor = 1.0 - 0.4 * dissociation_fraction
        
        modified_thrust = base_thrust * thrust_factor
        
        return modified_thrust, isp_factor
    
    def calculate_plasma_formation_effects(self, stagnation_temperature: float, 
                                         pressure: float) -> Tuple[float, float, float]:
        """Calculate plasma formation effects on engine performance."""
        # Plasma formation threshold (Saha equation approximation)
        plasma_threshold_temp = 15000.0  # K (simplified threshold)
        
        if stagnation_temperature < plasma_threshold_temp:
            return 1.0, 1.0, 0.0  # No plasma effects
        
        # Calculate ionization fraction using simplified Saha equation
        temp_ratio = stagnation_temperature / plasma_threshold_temp
        pressure_factor = pressure / 101325.0  # Normalized pressure
        
        # Simplified ionization fraction
        ionization_fraction = min(0.9, (temp_ratio - 1.0) / 3.0 * (1.0 / pressure_factor**0.5))
        
        # Plasma effects on performance
        # 1. Electromagnetic interactions can enhance or degrade performance
        # 2. Radiative losses increase significantly
        # 3. Electrical conductivity affects flow properties
        
        # Thrust modification due to electromagnetic effects
        em_thrust_factor = 1.0 + 0.1 * ionization_fraction - 0.2 * ionization_fraction**2
        
        # Heat transfer modification due to radiative losses
        radiative_loss_factor = 1.0 + 2.0 * ionization_fraction  # Increased heat transfer
        
        # Overall performance factor
        performance_factor = em_thrust_factor / (1.0 + 0.5 * ionization_fraction)
        
        return max(0.3, performance_factor), radiative_loss_factor, ionization_fraction
    
    def get_extreme_temperature_performance(self, operating_point: CombinedCycleOperatingPoint) -> Dict[str, float]:
        """Get detailed performance data for extreme temperature conditions."""
        # Calculate stagnation temperature
        stagnation_temp = self.calculate_stagnation_temperature(
            operating_point.mach_number, operating_point.altitude
        )
        
        # Get atmospheric conditions
        temperature, pressure, density = self.get_atmospheric_conditions(operating_point.altitude)
        
        # Calculate base performance without extreme temperature effects
        base_performance = self.get_combined_cycle_performance(operating_point)
        
        # Calculate dissociation effects
        dissociation_thrust, isp_factor = self.calculate_dissociation_effects_on_thrust(
            stagnation_temp, base_performance.total_thrust
        )
        
        # Calculate plasma formation effects
        plasma_thrust_factor, radiative_factor, ionization_fraction = self.calculate_plasma_formation_effects(
            stagnation_temp, pressure
        )
        
        # Calculate extreme temperature effects
        extreme_temp_factor = self._interpolate_performance_map(
            self.extreme_temp_effects_map, operating_point.altitude, operating_point.mach_number
        )
        
        # Combine all effects
        total_thrust_factor = plasma_thrust_factor * extreme_temp_factor
        final_thrust = dissociation_thrust * total_thrust_factor
        
        return {
            'stagnation_temperature': stagnation_temp,
            'base_thrust': base_performance.total_thrust,
            'dissociation_thrust': dissociation_thrust,
            'final_thrust': final_thrust,
            'dissociation_losses': (base_performance.total_thrust - dissociation_thrust) / base_performance.total_thrust if base_performance.total_thrust > 0 else 0,
            'plasma_thrust_factor': plasma_thrust_factor,
            'extreme_temp_factor': extreme_temp_factor,
            'ionization_fraction': ionization_fraction,
            'isp_factor': isp_factor,
            'radiative_loss_factor': radiative_factor,
            'total_performance_factor': total_thrust_factor
        }
    
    def validate_extreme_temperature_limits(self, operating_point: CombinedCycleOperatingPoint) -> List[str]:
        """Validate operating point against extreme temperature limits."""
        warnings = []
        
        stagnation_temp = self.calculate_stagnation_temperature(
            operating_point.mach_number, operating_point.altitude
        )
        
        # Check against maximum temperature limit
        if stagnation_temp > self.combined_spec.max_stagnation_temperature:
            warnings.append(f"Stagnation temperature ({stagnation_temp:.0f}K) exceeds maximum limit ({self.combined_spec.max_stagnation_temperature:.0f}K)")
        
        # Check for significant dissociation
        if stagnation_temp > self.combined_spec.dissociation_onset_temperature:
            dissociation_losses = self.calculate_dissociation_losses(stagnation_temp)
            if dissociation_losses > 0.2:
                warnings.append(f"Significant dissociation losses ({dissociation_losses*100:.1f}%) at current conditions")
        
        # Check for plasma formation
        _, pressure, _ = self.get_atmospheric_conditions(operating_point.altitude)
        _, _, ionization_fraction = self.calculate_plasma_formation_effects(stagnation_temp, pressure)
        
        if ionization_fraction > 0.1:
            warnings.append(f"Significant plasma formation ({ionization_fraction*100:.1f}% ionization) affecting performance")
        
        # Check for extreme temperature effects
        if stagnation_temp > 50000.0:
            warnings.append("Operating in extreme temperature regime (>50,000K) with potential material limitations")
        
        return warnings
    
    def optimize_for_extreme_temperatures(self, target_altitude: float, 
                                        target_mach: float) -> Dict[str, Any]:
        """Optimize engine operation for extreme temperature conditions."""
        # Create operating point
        operating_point = CombinedCycleOperatingPoint(
            altitude=target_altitude,
            mach_number=target_mach,
            throttle_setting=1.0,
            rocket_throttle_setting=1.0
        )
        
        # Get extreme temperature performance
        extreme_perf = self.get_extreme_temperature_performance(operating_point)
        
        # Optimization recommendations
        recommendations = []
        
        if extreme_perf['stagnation_temperature'] > 50000.0:
            recommendations.append("Consider reducing throttle setting to limit stagnation temperature")
            
            # Test reduced throttle settings
            best_throttle = 1.0
            best_efficiency = 0.0
            
            for throttle in [0.9, 0.8, 0.7, 0.6]:
                test_point = CombinedCycleOperatingPoint(
                    altitude=target_altitude,
                    mach_number=target_mach,
                    throttle_setting=throttle,
                    rocket_throttle_setting=throttle
                )
                
                test_perf = self.get_extreme_temperature_performance(test_point)
                efficiency = test_perf['final_thrust'] / test_perf['base_thrust'] if test_perf['base_thrust'] > 0 else 0
                
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_throttle = throttle
            
            if best_throttle < 1.0:
                recommendations.append(f"Optimal throttle setting: {best_throttle*100:.0f}%")
        
        if extreme_perf['ionization_fraction'] > 0.2:
            recommendations.append("High ionization levels detected - consider electromagnetic field management")
        
        if extreme_perf['dissociation_losses'] > 0.3:
            recommendations.append("Significant dissociation losses - consider fuel mixture optimization")
        
        return {
            'performance_data': extreme_perf,
            'recommendations': recommendations,
            'optimal_throttle': best_throttle if 'best_throttle' in locals() else 1.0,
            'warnings': self.validate_extreme_temperature_limits(operating_point)
        }
    
    def validate_combined_cycle_specification(self) -> List[str]:
        """Validate combined-cycle engine specification."""
        errors = super().validate_engine_specification()
        
        # Combined-cycle specific validation
        if self.combined_spec.transition_mach_number <= 4.0:
            errors.append("Transition Mach number must be > 4.0 for combined-cycle operation")
        
        if self.combined_spec.rocket_specific_impulse <= 0:
            errors.append("Rocket specific impulse must be positive")
        
        if self.combined_spec.air_breathing_specific_impulse <= 0:
            errors.append("Air-breathing specific impulse must be positive")
        
        if self.combined_spec.max_rocket_thrust <= 0:
            errors.append("Maximum rocket thrust must be positive")
        
        if self.combined_spec.plasma_interaction_threshold < self.combined_spec.transition_mach_number:
            errors.append("Plasma interaction threshold should be >= transition Mach number")
        
        if self.combined_spec.max_stagnation_temperature <= self.combined_spec.dissociation_onset_temperature:
            errors.append("Maximum stagnation temperature must exceed dissociation onset temperature")
        
        return errors