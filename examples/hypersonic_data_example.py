#!/usr/bin/env python3
"""
Example demonstrating the new extreme hypersonic data structures for Mach 60+ flight.

This example shows how to create and validate the new data models for:
- Plasma conditions
- Combined-cycle propulsion performance
- Thermal protection systems
- Hypersonic mission profiles
"""

import numpy as np
from fighter_jet_sdk.common.data_models import (
    PlasmaConditions, CombinedCyclePerformance, ThermalProtectionSystem,
    HypersonicMissionProfile, AblativeLayer, CoolingChannel, InsulationLayer
)
from fighter_jet_sdk.common.enums import (
    PlasmaRegime, ExtremePropulsionType, ThermalProtectionType
)
from fighter_jet_sdk.common.hypersonic_validation import HypersonicDataValidator


def create_plasma_conditions_example():
    """Create example plasma conditions for Mach 60 flight."""
    print("=== Plasma Conditions Example ===")
    
    # Calculate physically consistent plasma parameters
    electron_density = 1e20  # m⁻³
    electron_temperature = 15000  # K
    
    # Physical constants
    e_charge = 1.602176634e-19  # C
    e_mass = 9.1093837015e-31  # kg
    epsilon_0 = 8.8541878128e-12  # F/m
    k_b = 1.380649e-23  # J/K
    
    # Calculate consistent plasma frequency
    plasma_frequency = np.sqrt(electron_density * e_charge**2 / (epsilon_0 * e_mass))
    
    # Calculate consistent Debye length
    debye_length = np.sqrt(epsilon_0 * k_b * electron_temperature / (electron_density * e_charge**2))
    
    plasma = PlasmaConditions(
        electron_density=electron_density,
        electron_temperature=electron_temperature,
        ion_temperature=12000,  # K
        magnetic_field=np.array([0.05, 0.02, 0.0]),  # Tesla
        plasma_frequency=plasma_frequency,
        debye_length=debye_length,
        plasma_regime=PlasmaRegime.PARTIALLY_IONIZED
    )
    
    print(f"Electron density: {plasma.electron_density:.2e} m⁻³")
    print(f"Electron temperature: {plasma.electron_temperature} K")
    print(f"Plasma frequency: {plasma.plasma_frequency:.2e} Hz")
    print(f"Debye length: {plasma.debye_length:.2e} m")
    print(f"Plasma beta: {plasma.calculate_plasma_beta():.3f}")
    
    # Validate plasma conditions
    errors = plasma.validate_plasma_conditions()
    if errors:
        print(f"Validation errors: {errors}")
    else:
        print("✓ Plasma conditions are valid")
    
    return plasma


def create_combined_cycle_performance_example():
    """Create example combined-cycle propulsion performance."""
    print("\n=== Combined-Cycle Propulsion Performance Example ===")
    
    # Create realistic combined-cycle performance
    fuel_flow_total = 80.0  # kg/s
    specific_impulse = 2800  # s
    theoretical_thrust = fuel_flow_total * specific_impulse * 9.80665  # N
    
    performance = CombinedCyclePerformance(
        air_breathing_thrust=theoretical_thrust * 0.65,  # 65% air-breathing
        rocket_thrust=theoretical_thrust * 0.35,  # 35% rocket
        transition_mach=12.0,
        fuel_flow_air_breathing=52.0,  # kg/s
        fuel_flow_rocket=28.0,  # kg/s
        specific_impulse=specific_impulse,
        propulsion_type=ExtremePropulsionType.COMBINED_CYCLE_AIRBREATHING,
        operating_altitude_range=(40000, 100000)  # 40-100 km
    )
    
    print(f"Air-breathing thrust: {performance.air_breathing_thrust/1000:.0f} kN")
    print(f"Rocket thrust: {performance.rocket_thrust/1000:.0f} kN")
    print(f"Total thrust: {performance.calculate_total_thrust()/1000:.0f} kN")
    print(f"Transition Mach: {performance.transition_mach}")
    print(f"Specific impulse: {performance.specific_impulse} s")
    
    # Calculate thrust-to-weight ratio for a 100-ton vehicle
    vehicle_mass = 100000  # kg
    twr = performance.calculate_thrust_to_weight_ratio(vehicle_mass)
    print(f"Thrust-to-weight ratio (100t vehicle): {twr:.2f}")
    
    # Validate performance
    errors = performance.validate_performance()
    if errors:
        print(f"Validation errors: {errors}")
    else:
        print("✓ Combined-cycle performance is valid")
    
    return performance


def create_thermal_protection_system_example():
    """Create example thermal protection system for Mach 60."""
    print("\n=== Thermal Protection System Example ===")
    
    # Create ablative layer
    ablative_layer = AblativeLayer(
        material_id="ultra_high_temp_carbon_carbon",
        thickness=0.08,  # 8 cm
        ablation_rate=5e-7,  # m/s per MW/m²
        heat_of_ablation=3e6,  # J/kg
        char_layer_conductivity=0.8  # W/(m⋅K)
    )
    
    # Create active cooling channels
    cooling_channels = [
        CoolingChannel(
            channel_id=f"lh2_channel_{i}",
            diameter=0.003,  # 3 mm
            length=2.0,  # 2 m
            coolant_type="liquid_hydrogen",
            mass_flow_rate=0.05,  # kg/s per channel
            inlet_temperature=20,  # K
            pressure_drop=2e5  # Pa
        ) for i in range(10)
    ]
    
    # Create insulation layer
    insulation_layer = InsulationLayer(
        material_id="ultra_low_conductivity_aerogel",
        thickness=0.05,  # 5 cm
        thermal_conductivity=0.005,  # W/(m⋅K)
        max_operating_temperature=2500  # K
    )
    
    # Create complete TPS
    tps = ThermalProtectionSystem(
        ablative_layers=[ablative_layer],
        active_cooling_channels=cooling_channels,
        insulation_layers=[insulation_layer],
        protection_type=ThermalProtectionType.HYBRID_SYSTEM,
        max_heat_flux_capacity=2e8,  # 200 MW/m²
        cooling_effectiveness=0.85
    )
    
    # Calculate system properties
    total_thickness = tps.calculate_total_thickness()
    mass_per_area = tps.estimate_mass_per_area(1.0)  # per m²
    
    print(f"Total TPS thickness: {total_thickness*100:.1f} cm")
    print(f"Mass per unit area: {mass_per_area:.0f} kg/m²")
    print(f"Max heat flux capacity: {tps.max_heat_flux_capacity/1e6:.0f} MW/m²")
    print(f"Cooling effectiveness: {tps.cooling_effectiveness:.1%}")
    print(f"Number of cooling channels: {len(tps.active_cooling_channels)}")
    
    # Validate TPS
    errors = tps.validate_system()
    if errors:
        print(f"Validation errors: {errors}")
    else:
        print("✓ Thermal protection system is valid")
    
    return tps


def create_hypersonic_mission_profile_example():
    """Create example hypersonic mission profile for Mach 60."""
    print("\n=== Hypersonic Mission Profile Example ===")
    
    # Create mission profile for Mach 60 test flight
    time_points = 50
    mission_time = np.linspace(0, 3600, time_points)  # 1 hour mission
    
    # Altitude profile: climb to 70 km, cruise, descend
    altitude_profile = np.array([
        50000 + 20000 * np.sin(np.pi * t / 3600) for t in mission_time
    ])
    
    # Mach profile: accelerate to Mach 60, cruise, decelerate
    mach_profile = np.array([
        30 + 30 * np.sin(np.pi * t / 3600) for t in mission_time
    ])
    
    # Thermal load profile: correlates with Mach number
    thermal_load_profile = np.array([
        1e7 * (mach / 30)**2.5 for mach in mach_profile
    ])
    
    # Propulsion mode schedule
    propulsion_modes = []
    cooling_schedule = []
    for i, mach in enumerate(mach_profile):
        if mach < 40:
            propulsion_modes.append("air_breathing")
            cooling_schedule.append(False)
        elif mach < 55:
            propulsion_modes.append("combined_cycle")
            cooling_schedule.append(True)
        else:
            propulsion_modes.append("rocket_assisted")
            cooling_schedule.append(True)
    
    # Create plasma conditions for high Mach portions
    plasma_conditions = []
    for mach in mach_profile:
        if mach >= 50:
            # High Mach conditions with significant plasma effects
            electron_density = 1e21 * (mach / 60)**2
            electron_temp = 20000 * (mach / 60)
            
            # Calculate consistent parameters
            e_charge = 1.602176634e-19
            e_mass = 9.1093837015e-31
            epsilon_0 = 8.8541878128e-12
            k_b = 1.380649e-23
            
            plasma_freq = np.sqrt(electron_density * e_charge**2 / (epsilon_0 * e_mass))
            debye_len = np.sqrt(epsilon_0 * k_b * electron_temp / (electron_density * e_charge**2))
            
            plasma_conditions.append(PlasmaConditions(
                electron_density=electron_density,
                electron_temperature=electron_temp,
                ion_temperature=electron_temp * 0.8,
                magnetic_field=np.array([0.1, 0.05, 0.0]),
                plasma_frequency=plasma_freq,
                debye_length=debye_len,
                plasma_regime=PlasmaRegime.PARTIALLY_IONIZED
            ))
        else:
            plasma_conditions.append(None)
    
    # Filter out None values for the profile
    plasma_conditions_profile = [pc for pc in plasma_conditions if pc is not None]
    
    profile = HypersonicMissionProfile(
        mission_name="Mach_60_Demonstration_Flight",
        altitude_profile=altitude_profile,
        mach_profile=mach_profile,
        thermal_load_profile=thermal_load_profile,
        propulsion_mode_schedule=propulsion_modes,
        cooling_system_schedule=cooling_schedule,
        plasma_conditions_profile=plasma_conditions_profile,
        mission_duration=3600,  # 1 hour
        max_thermal_load=np.max(thermal_load_profile)
    )
    
    # Calculate and display statistics
    stats = profile.calculate_profile_statistics()
    print(f"Mission duration: {profile.mission_duration/60:.0f} minutes")
    print(f"Max altitude: {stats['max_altitude']/1000:.1f} km")
    print(f"Max Mach: {stats['max_mach']:.1f}")
    print(f"Peak thermal load: {stats['peak_thermal_load']/1e6:.0f} MW/m²")
    print(f"Time at Mach 60+: {stats['mach_60_plus_time_fraction']:.1%}")
    print(f"Requires plasma modeling: {profile.requires_plasma_modeling()}")
    print(f"Requires active cooling: {profile.requires_active_cooling()}")
    
    # Validate mission profile
    errors = profile.validate_profile()
    if errors:
        print(f"Validation errors: {errors}")
    else:
        print("✓ Hypersonic mission profile is valid")
    
    return profile


def demonstrate_system_integration():
    """Demonstrate integration validation between all systems."""
    print("\n=== System Integration Validation ===")
    
    # Create all system components
    plasma = create_plasma_conditions_example()
    performance = create_combined_cycle_performance_example()
    tps = create_thermal_protection_system_example()
    profile = create_hypersonic_mission_profile_example()
    
    # Validate system integration
    integration_errors = HypersonicDataValidator.validate_system_integration(
        plasma, performance, tps, profile
    )
    
    if integration_errors:
        print(f"Integration errors: {integration_errors}")
    else:
        print("✓ All systems are compatible and properly integrated")
    
    return {
        'plasma': plasma,
        'performance': performance,
        'tps': tps,
        'profile': profile
    }


def main():
    """Main demonstration function."""
    print("Extreme Hypersonic Data Structures Demonstration")
    print("=" * 50)
    
    # Create and validate individual components
    plasma = create_plasma_conditions_example()
    performance = create_combined_cycle_performance_example()
    tps = create_thermal_protection_system_example()
    profile = create_hypersonic_mission_profile_example()
    
    # Demonstrate system integration
    systems = demonstrate_system_integration()
    
    print("\n=== Summary ===")
    print("Successfully created and validated all extreme hypersonic data structures:")
    print("✓ Plasma conditions for ionized flow modeling")
    print("✓ Combined-cycle propulsion performance")
    print("✓ Advanced thermal protection system")
    print("✓ Hypersonic mission profile for Mach 60+ flight")
    print("✓ System integration validation")
    
    print(f"\nAll systems are ready for Mach 60 hypersonic analysis!")


if __name__ == "__main__":
    main()