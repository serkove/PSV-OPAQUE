"""Validation methods for extreme hypersonic data structures."""

from typing import List, Dict, Any
import numpy as np
from .data_models import (
    PlasmaConditions, CombinedCyclePerformance, ThermalProtectionSystem,
    HypersonicMissionProfile, AblativeLayer, CoolingChannel, InsulationLayer
)
from .enums import PlasmaRegime, ExtremePropulsionType, ThermalProtectionType


class HypersonicDataValidator:
    """Comprehensive validation for extreme hypersonic data structures."""
    
    # Physical constants
    BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
    ELECTRON_CHARGE = 1.602176634e-19  # C
    ELECTRON_MASS = 9.1093837015e-31  # kg
    VACUUM_PERMITTIVITY = 8.8541878128e-12  # F/m
    
    @classmethod
    def validate_plasma_conditions(cls, plasma: PlasmaConditions) -> List[str]:
        """Comprehensive validation of plasma conditions."""
        errors = []
        
        # Basic validation from the dataclass method
        basic_errors = plasma.validate_plasma_conditions()
        errors.extend(basic_errors)
        
        # Advanced physics-based validation
        physics_errors = cls._validate_plasma_physics(plasma)
        errors.extend(physics_errors)
        
        return errors
    
    @classmethod
    def _validate_plasma_physics(cls, plasma: PlasmaConditions) -> List[str]:
        """Validate plasma conditions against physical laws."""
        errors = []
        
        # Calculate theoretical plasma frequency and compare
        theoretical_plasma_freq = np.sqrt(
            plasma.electron_density * cls.ELECTRON_CHARGE**2 / 
            (cls.VACUUM_PERMITTIVITY * cls.ELECTRON_MASS)
        )
        
        freq_ratio = abs(plasma.plasma_frequency - theoretical_plasma_freq) / theoretical_plasma_freq
        if freq_ratio > 0.1:  # 10% tolerance
            errors.append(f"Plasma frequency inconsistent with electron density (error: {freq_ratio:.1%})")
        
        # Calculate theoretical Debye length and compare
        theoretical_debye_length = np.sqrt(
            cls.VACUUM_PERMITTIVITY * cls.BOLTZMANN_CONSTANT * plasma.electron_temperature /
            (plasma.electron_density * cls.ELECTRON_CHARGE**2)
        )
        
        debye_ratio = abs(plasma.debye_length - theoretical_debye_length) / theoretical_debye_length
        if debye_ratio > 0.1:  # 10% tolerance
            errors.append(f"Debye length inconsistent with plasma parameters (error: {debye_ratio:.1%})")
        
        # Validate plasma regime classification
        regime_errors = cls._validate_plasma_regime(plasma)
        errors.extend(regime_errors)
        
        return errors
    
    @classmethod
    def _validate_plasma_regime(cls, plasma: PlasmaConditions) -> List[str]:
        """Validate plasma regime classification."""
        errors = []
        
        # Calculate ionization fraction (simplified)
        # This is a rough estimate - real calculation would need more parameters
        saha_constant = 2.4e21  # Simplified Saha constant
        ionization_energy = 13.6  # eV for hydrogen (simplified)
        
        ionization_fraction = saha_constant * plasma.electron_temperature**1.5 * np.exp(
            -ionization_energy * cls.ELECTRON_CHARGE / (cls.BOLTZMANN_CONSTANT * plasma.electron_temperature)
        ) / plasma.electron_density
        
        # Check regime consistency
        if plasma.plasma_regime == PlasmaRegime.WEAKLY_IONIZED and ionization_fraction > 0.1:
            errors.append("Plasma classified as weakly ionized but ionization fraction > 10%")
        elif plasma.plasma_regime == PlasmaRegime.FULLY_IONIZED and ionization_fraction < 0.9:
            errors.append("Plasma classified as fully ionized but ionization fraction < 90%")
        
        return errors
    
    @classmethod
    def validate_combined_cycle_performance(cls, performance: CombinedCyclePerformance) -> List[str]:
        """Comprehensive validation of combined-cycle performance."""
        errors = []
        
        # Basic validation from the dataclass method
        basic_errors = performance.validate_performance()
        errors.extend(basic_errors)
        
        # Advanced propulsion-specific validation
        propulsion_errors = cls._validate_propulsion_physics(performance)
        errors.extend(propulsion_errors)
        
        return errors
    
    @classmethod
    def _validate_propulsion_physics(cls, performance: CombinedCyclePerformance) -> List[str]:
        """Validate propulsion performance against physical constraints."""
        errors = []
        
        # Check thrust consistency with fuel flow and specific impulse
        total_fuel_flow = performance.fuel_flow_air_breathing + performance.fuel_flow_rocket
        if total_fuel_flow > 0:
            theoretical_thrust = total_fuel_flow * performance.specific_impulse * 9.80665
            actual_thrust = performance.air_breathing_thrust + performance.rocket_thrust
            
            thrust_ratio = abs(theoretical_thrust - actual_thrust) / max(theoretical_thrust, actual_thrust)
            if thrust_ratio > 0.2:  # 20% tolerance
                errors.append(f"Thrust inconsistent with fuel flow and specific impulse (error: {thrust_ratio:.1%})")
        
        # Validate transition Mach number for propulsion type
        if performance.propulsion_type == ExtremePropulsionType.DUAL_MODE_SCRAMJET:
            if performance.transition_mach < 4 or performance.transition_mach > 8:
                errors.append("Dual-mode scramjet transition Mach should be between 4-8")
        elif performance.propulsion_type == ExtremePropulsionType.COMBINED_CYCLE_AIRBREATHING:
            if performance.transition_mach < 6 or performance.transition_mach > 15:
                errors.append("Combined-cycle transition Mach should be between 6-15")
        
        # Check specific impulse ranges for different propulsion types
        if performance.propulsion_type in [ExtremePropulsionType.DUAL_MODE_SCRAMJET, 
                                         ExtremePropulsionType.COMBINED_CYCLE_AIRBREATHING]:
            if performance.specific_impulse < 1000 or performance.specific_impulse > 4000:
                errors.append("Air-breathing specific impulse should be between 1000-4000 s")
        elif performance.propulsion_type == ExtremePropulsionType.NUCLEAR_THERMAL:
            if performance.specific_impulse < 800 or performance.specific_impulse > 1000:
                errors.append("Nuclear thermal specific impulse should be between 800-1000 s")
        
        return errors
    
    @classmethod
    def validate_thermal_protection_system(cls, tps: ThermalProtectionSystem) -> List[str]:
        """Comprehensive validation of thermal protection system."""
        errors = []
        
        # Basic validation from the dataclass method
        basic_errors = tps.validate_system()
        errors.extend(basic_errors)
        
        # Advanced thermal analysis validation
        thermal_errors = cls._validate_thermal_design(tps)
        errors.extend(thermal_errors)
        
        return errors
    
    @classmethod
    def _validate_thermal_design(cls, tps: ThermalProtectionSystem) -> List[str]:
        """Validate thermal protection system design."""
        errors = []
        
        # Check system configuration consistency
        if tps.protection_type == ThermalProtectionType.PASSIVE_ABLATIVE:
            if not tps.ablative_layers:
                errors.append("Passive ablative TPS must have ablative layers")
            if tps.active_cooling_channels:
                errors.append("Passive ablative TPS should not have active cooling channels")
        
        elif tps.protection_type == ThermalProtectionType.ACTIVE_TRANSPIRATION:
            if not tps.active_cooling_channels:
                errors.append("Active transpiration TPS must have cooling channels")
        
        elif tps.protection_type == ThermalProtectionType.HYBRID_SYSTEM:
            if not tps.ablative_layers and not tps.active_cooling_channels:
                errors.append("Hybrid TPS must have both ablative and active cooling components")
        
        # Validate heat flux capacity
        if tps.max_heat_flux_capacity > 0:
            # For Mach 60 flight, heat flux can exceed 100 MW/m²
            if tps.max_heat_flux_capacity < 1e8:  # 100 MW/m²
                errors.append("Heat flux capacity may be insufficient for Mach 60 flight (< 100 MW/m²)")
            
            if tps.max_heat_flux_capacity > 1e9:  # 1 GW/m²
                errors.append("Heat flux capacity exceeds realistic material limits (> 1 GW/m²)")
        
        # Check cooling effectiveness
        if tps.active_cooling_channels and tps.cooling_effectiveness == 0:
            errors.append("Active cooling system should have non-zero cooling effectiveness")
        
        if not tps.active_cooling_channels and tps.cooling_effectiveness > 0.1:
            errors.append("Passive system should not have significant cooling effectiveness")
        
        return errors
    
    @classmethod
    def validate_hypersonic_mission_profile(cls, profile: HypersonicMissionProfile) -> List[str]:
        """Comprehensive validation of hypersonic mission profile."""
        errors = []
        
        # Basic validation from the dataclass method
        basic_errors = profile.validate_profile()
        errors.extend(basic_errors)
        
        # Advanced mission analysis validation
        mission_errors = cls._validate_mission_physics(profile)
        errors.extend(mission_errors)
        
        return errors
    
    @classmethod
    def _validate_mission_physics(cls, profile: HypersonicMissionProfile) -> List[str]:
        """Validate mission profile against flight physics."""
        errors = []
        
        if len(profile.altitude_profile) == 0 or len(profile.mach_profile) == 0:
            return errors
        
        # Check altitude-Mach correlation for Mach 60 flight
        for i, (alt, mach) in enumerate(zip(profile.altitude_profile, profile.mach_profile)):
            if mach >= 60:
                if alt < 50000:  # 50 km minimum for Mach 60
                    errors.append(f"Mach 60+ flight at altitude {alt/1000:.1f} km is not feasible (point {i})")
                if alt > 120000:  # 120 km practical upper limit
                    errors.append(f"Mach 60 flight at altitude {alt/1000:.1f} km exceeds practical limits (point {i})")
        
        # Validate thermal load correlation with Mach number
        if len(profile.thermal_load_profile) > 0:
            for i, (mach, thermal_load) in enumerate(zip(profile.mach_profile, profile.thermal_load_profile)):
                # Rough correlation: thermal load ~ Mach^3
                expected_thermal_load = 1e6 * (mach / 10)**3  # Simplified scaling
                
                if thermal_load > 0 and abs(thermal_load - expected_thermal_load) / expected_thermal_load > 2:
                    errors.append(f"Thermal load at point {i} inconsistent with Mach number")
        
        # Check plasma conditions correlation
        for i, plasma_cond in enumerate(profile.plasma_conditions_profile):
            if i < len(profile.mach_profile):
                mach = profile.mach_profile[i]
                if mach < 25 and plasma_cond.plasma_regime != PlasmaRegime.WEAKLY_IONIZED:
                    errors.append(f"Plasma regime at point {i} inconsistent with Mach number {mach}")
        
        return errors
    
    @classmethod
    def validate_system_integration(cls, 
                                  plasma: PlasmaConditions,
                                  performance: CombinedCyclePerformance,
                                  tps: ThermalProtectionSystem,
                                  profile: HypersonicMissionProfile) -> List[str]:
        """Validate integration between all hypersonic systems."""
        errors = []
        
        # Check thermal load compatibility
        if profile.max_thermal_load > tps.max_heat_flux_capacity:
            errors.append("Mission thermal loads exceed TPS capacity")
        
        # Check propulsion-altitude compatibility
        if len(profile.altitude_profile) > 0:
            min_alt = np.min(profile.altitude_profile)
            max_alt = np.max(profile.altitude_profile)
            
            if min_alt < performance.operating_altitude_range[0]:
                errors.append("Mission altitude below propulsion system operating range")
            if max_alt > performance.operating_altitude_range[1]:
                errors.append("Mission altitude above propulsion system operating range")
        
        # Check plasma effects on propulsion
        if plasma.plasma_regime in [PlasmaRegime.PARTIALLY_IONIZED, PlasmaRegime.FULLY_IONIZED]:
            if performance.propulsion_type not in [ExtremePropulsionType.MAGNETOPLASMADYNAMIC]:
                errors.append("High ionization plasma may interfere with conventional propulsion")
        
        return errors


def validate_all_hypersonic_data(data_dict: Dict[str, Any]) -> Dict[str, List[str]]:
    """Validate all hypersonic data structures in a dictionary."""
    validation_results = {}
    
    validator = HypersonicDataValidator()
    
    # Validate each data type
    for key, value in data_dict.items():
        if isinstance(value, PlasmaConditions):
            validation_results[key] = validator.validate_plasma_conditions(value)
        elif isinstance(value, CombinedCyclePerformance):
            validation_results[key] = validator.validate_combined_cycle_performance(value)
        elif isinstance(value, ThermalProtectionSystem):
            validation_results[key] = validator.validate_thermal_protection_system(value)
        elif isinstance(value, HypersonicMissionProfile):
            validation_results[key] = validator.validate_hypersonic_mission_profile(value)
    
    return validation_results