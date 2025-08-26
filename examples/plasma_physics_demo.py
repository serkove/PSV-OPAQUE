#!/usr/bin/env python3
"""
Demonstration of plasma physics foundation module for Mach 60 hypersonic flight.

This example shows how to use the plasma physics and electromagnetic effects
modules to analyze conditions during extreme hypersonic flight.
"""

import numpy as np
import matplotlib.pyplot as plt
from fighter_jet_sdk.common.plasma_physics import (
    PlasmaPropertiesCalculator, PlasmaPropertyInterpolator, GasMixture
)
from fighter_jet_sdk.common.electromagnetic_effects import (
    ElectromagneticEffectsCalculator, MagneticFieldGenerator
)


def demonstrate_mach_60_analysis():
    """Demonstrate plasma physics analysis for Mach 60 flight."""
    print("=== Mach 60 Hypersonic Plasma Physics Analysis ===\n")
    
    # Initialize calculators
    plasma_calc = PlasmaPropertiesCalculator()
    em_calc = ElectromagneticEffectsCalculator()
    field_gen = MagneticFieldGenerator()
    
    # Mach 60 flight conditions
    print("Flight Conditions:")
    print("- Mach number: 60")
    print("- Velocity: ~20,000 m/s")
    print("- Altitude: 50 km")
    print("- Shock temperature: ~50,000 K")
    print()
    
    # Define gas mixture (air at high altitude)
    air_mixture = GasMixture(
        species={'N2': 0.78, 'O2': 0.21, 'Ar': 0.01},
        temperature=50000.0,  # K - extreme shock heating
        pressure=100.0,       # Pa - high altitude
        total_density=100.0 / (1.380649e-23 * 50000.0)  # kg/m³
    )
    
    print("Gas Mixture Properties:")
    print(f"- Temperature: {air_mixture.temperature:,.0f} K")
    print(f"- Pressure: {air_mixture.pressure:.1f} Pa")
    print(f"- Density: {air_mixture.total_density:.2e} kg/m³")
    print()
    
    # Calculate plasma conditions
    plasma_conditions = plasma_calc.calculate_complete_plasma_conditions(air_mixture)
    
    print("Plasma Properties:")
    print(f"- Electron density: {plasma_conditions.electron_density:.2e} m⁻³")
    print(f"- Plasma frequency: {plasma_conditions.plasma_frequency:.2e} Hz")
    print(f"- Debye length: {plasma_conditions.debye_length:.2e} m")
    print(f"- Ionization fraction: {plasma_conditions.ionization_fraction:.3f}")
    print(f"- Plasma regime: {plasma_conditions.regime.name}")
    print()
    
    # Flight parameters
    velocity = np.array([20000.0, 0.0, 0.0])  # m/s (Mach 60)
    earth_magnetic_field = np.array([0.0, 0.0, 5e-5])  # Tesla
    
    # Calculate electromagnetic effects
    em_properties = em_calc.calculate_complete_electromagnetic_properties(
        plasma_conditions, velocity, earth_magnetic_field, characteristic_length=10.0
    )
    
    print("Electromagnetic Effects:")
    print(f"- Plasma conductivity: {em_properties.conductivity:.2e} S/m")
    print(f"- Hall parameter: {em_properties.hall_parameter:.3f}")
    print(f"- Magnetic Reynolds number: {em_properties.magnetic_reynolds_number:.2e}")
    print(f"- Induced electric field: {np.linalg.norm(em_properties.electric_field):.2f} V/m")
    print(f"- Current density magnitude: {np.linalg.norm(em_properties.current_density):.2e} A/m²")
    print(f"- Lorentz force density: {np.linalg.norm(em_properties.lorentz_force_density):.2e} N/m³")
    print()


def demonstrate_temperature_effects():
    """Demonstrate how plasma properties change with temperature."""
    print("=== Temperature Effects on Plasma Properties ===\n")
    
    plasma_calc = PlasmaPropertiesCalculator()
    
    # Temperature range from 5,000 K to 100,000 K
    temperatures = np.logspace(np.log10(5000), np.log10(100000), 20)
    pressure = 1000.0  # Pa
    
    electron_densities = []
    ionization_fractions = []
    plasma_frequencies = []
    
    print("Calculating plasma properties across temperature range...")
    
    for temp in temperatures:
        gas_mixture = GasMixture(
            species={'N2': 0.78, 'O2': 0.22},
            temperature=temp,
            pressure=pressure,
            total_density=pressure / (1.380649e-23 * temp)
        )
        
        plasma_conditions = plasma_calc.calculate_complete_plasma_conditions(gas_mixture)
        
        electron_densities.append(plasma_conditions.electron_density)
        ionization_fractions.append(plasma_conditions.ionization_fraction)
        plasma_frequencies.append(plasma_conditions.plasma_frequency)
    
    # Display some key results
    print(f"\nAt 10,000 K:")
    idx_10k = np.argmin(np.abs(temperatures - 10000))
    print(f"- Ionization fraction: {ionization_fractions[idx_10k]:.4f}")
    print(f"- Electron density: {electron_densities[idx_10k]:.2e} m⁻³")
    
    print(f"\nAt 50,000 K (Mach 60 conditions):")
    idx_50k = np.argmin(np.abs(temperatures - 50000))
    print(f"- Ionization fraction: {ionization_fractions[idx_50k]:.4f}")
    print(f"- Electron density: {electron_densities[idx_50k]:.2e} m⁻³")
    
    print(f"\nAt 100,000 K:")
    idx_100k = np.argmin(np.abs(temperatures - 100000))
    print(f"- Ionization fraction: {ionization_fractions[idx_100k]:.4f}")
    print(f"- Electron density: {electron_densities[idx_100k]:.2e} m⁻³")
    print()


def demonstrate_magnetic_field_effects():
    """Demonstrate electromagnetic effects with different magnetic field strengths."""
    print("=== Magnetic Field Effects ===\n")
    
    em_calc = ElectromagneticEffectsCalculator()
    
    # High-temperature plasma conditions (typical for Mach 60)
    from fighter_jet_sdk.common.plasma_physics import PlasmaConditions, PlasmaRegime
    
    plasma_conditions = PlasmaConditions(
        electron_density=1e19,
        electron_temperature=50000.0,
        ion_temperature=50000.0,
        magnetic_field=np.array([0.0, 0.0, 0.0]),
        plasma_frequency=1e11,
        debye_length=1e-5,
        ionization_fraction=0.8,
        regime=PlasmaRegime.FULLY_IONIZED
    )
    
    velocity = np.array([20000.0, 0.0, 0.0])  # m/s
    
    # Test different magnetic field strengths
    B_strengths = [0.0, 1e-5, 5e-5, 1e-4, 1e-3, 1e-2]  # Tesla
    
    print("Magnetic Field Strength Effects:")
    print("B-field (T)  | Hall Parameter | Mag. Reynolds | Induced E (V/m)")
    print("-" * 65)
    
    for B_strength in B_strengths:
        magnetic_field = np.array([0.0, B_strength, 0.0])
        
        em_properties = em_calc.calculate_complete_electromagnetic_properties(
            plasma_conditions, velocity, magnetic_field, characteristic_length=10.0
        )
        
        induced_E_magnitude = np.linalg.norm(em_properties.electric_field)
        
        print(f"{B_strength:8.1e}   | {em_properties.hall_parameter:12.3f} | "
              f"{em_properties.magnetic_reynolds_number:11.2e} | {induced_E_magnitude:12.2f}")
    
    print()


def demonstrate_property_interpolation():
    """Demonstrate plasma property interpolation tables."""
    print("=== Plasma Property Interpolation ===\n")
    
    interpolator = PlasmaPropertyInterpolator()
    
    # Create property table for air
    print("Creating interpolation table for air mixture...")
    air_composition = {'N2': 0.78, 'O2': 0.21, 'Ar': 0.01}
    
    property_table = interpolator.create_property_table(
        air_composition,
        temperature_range=(5000.0, 100000.0),
        pressure_range=(10.0, 10000.0),
        num_temp_points=20,
        num_pressure_points=15
    )
    
    print("Table created successfully!")
    print("\nTesting interpolation at various conditions:")
    
    test_conditions = [
        (10000.0, 1000.0),   # 10,000 K, 1000 Pa
        (25000.0, 500.0),    # 25,000 K, 500 Pa
        (50000.0, 100.0),    # 50,000 K, 100 Pa (Mach 60 conditions)
        (75000.0, 50.0),     # 75,000 K, 50 Pa
    ]
    
    print("Temp (K) | Press (Pa) | Electron Density (m⁻³) | Ionization Fraction")
    print("-" * 70)
    
    for temp, pressure in test_conditions:
        properties = interpolator.get_interpolated_properties(
            property_table, temp, pressure
        )
        
        print(f"{temp:8.0f} | {pressure:10.0f} | {properties['electron_density']:20.2e} | "
              f"{properties['ionization_fraction']:17.4f}")
    
    print()


def main():
    """Run all demonstrations."""
    print("Plasma Physics Foundation Module Demonstration")
    print("=" * 50)
    print()
    
    try:
        demonstrate_mach_60_analysis()
        demonstrate_temperature_effects()
        demonstrate_magnetic_field_effects()
        demonstrate_property_interpolation()
        
        print("=== Summary ===")
        print("The plasma physics foundation module successfully demonstrates:")
        print("✓ Plasma property calculations using Saha equation")
        print("✓ Electromagnetic effects modeling")
        print("✓ Magnetic field interaction analysis")
        print("✓ Property interpolation for efficient computation")
        print("✓ Integration with existing Fighter Jet SDK architecture")
        print()
        print("This foundation enables advanced hypersonic analysis for Mach 60+ flight.")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()