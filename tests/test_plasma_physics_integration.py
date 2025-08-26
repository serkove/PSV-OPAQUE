"""Integration tests for plasma physics foundation module."""

import pytest
import numpy as np

from fighter_jet_sdk.common.plasma_physics import (
    PlasmaPropertiesCalculator, GasMixture, PlasmaConditions
)
from fighter_jet_sdk.common.electromagnetic_effects import (
    ElectromagneticEffectsCalculator, MagneticFieldGenerator
)


class TestPlasmaPhysicsIntegration:
    """Integration tests for plasma physics modules."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plasma_calculator = PlasmaPropertiesCalculator()
        self.em_calculator = ElectromagneticEffectsCalculator()
        self.field_generator = MagneticFieldGenerator()
    
    def test_mach_60_hypersonic_scenario(self):
        """Test complete plasma physics calculation for Mach 60 scenario."""
        # Mach 60 flight conditions at 50 km altitude
        # Temperature: ~50,000 K due to shock heating
        # Pressure: ~100 Pa (typical at 50 km)
        
        # Air composition at high altitude
        air_mixture = GasMixture(
            species={'N2': 0.78, 'O2': 0.21, 'Ar': 0.01},
            temperature=50000.0,  # K - extreme shock heating
            pressure=100.0,       # Pa - high altitude
            total_density=100.0 / (1.380649e-23 * 50000.0)  # kg/m³
        )
        
        # Calculate plasma conditions
        plasma_conditions = self.plasma_calculator.calculate_complete_plasma_conditions(
            air_mixture
        )
        
        # Verify plasma properties are reasonable for Mach 60 conditions
        assert plasma_conditions.electron_density > 1e15  # Significant ionization expected
        assert plasma_conditions.plasma_frequency > 1e9   # High plasma frequency
        assert plasma_conditions.debye_length < 1e-3      # Small Debye length
        assert plasma_conditions.ionization_fraction > 0.01  # Some ionization
        
        # Flight velocity (Mach 60 ≈ 20,000 m/s)
        velocity = np.array([20000.0, 0.0, 0.0])  # m/s
        
        # Earth's magnetic field (approximate)
        earth_B_field = np.array([0.0, 0.0, 5e-5])  # Tesla
        
        # Calculate electromagnetic effects
        em_properties = self.em_calculator.calculate_complete_electromagnetic_properties(
            plasma_conditions, velocity, earth_B_field, characteristic_length=10.0
        )
        
        # Verify electromagnetic properties
        assert em_properties.conductivity > 0
        assert em_properties.hall_parameter >= 0
        assert em_properties.magnetic_reynolds_number > 0
        assert np.all(np.isfinite(em_properties.electric_field))
        assert np.all(np.isfinite(em_properties.current_density))
        assert np.all(np.isfinite(em_properties.lorentz_force_density))
        
        # For Mach 60, should have significant induced electric field
        induced_E_magnitude = np.linalg.norm(em_properties.electric_field)
        assert induced_E_magnitude > 0.1  # V/m
    
    def test_plasma_regime_transitions(self):
        """Test plasma regime transitions with temperature."""
        air_composition = {'N2': 0.78, 'O2': 0.22}
        
        # Test different temperature regimes at constant density
        temperatures = [5000.0, 15000.0, 30000.0, 60000.0]  # K
        constant_density = 1e20  # m^-3 - keep density constant
        
        previous_ionization_fraction = 0.0
        
        for temp in temperatures:
            # Calculate pressure for constant density
            pressure = constant_density * 1.380649e-23 * temp
            
            gas_mixture = GasMixture(
                species=air_composition,
                temperature=temp,
                pressure=pressure,
                total_density=constant_density
            )
            
            plasma_conditions = self.plasma_calculator.calculate_complete_plasma_conditions(
                gas_mixture
            )
            
            # Ionization fraction should increase with temperature at constant density
            assert plasma_conditions.ionization_fraction > previous_ionization_fraction
            previous_ionization_fraction = plasma_conditions.ionization_fraction
            
            # Ionization fraction should be between 0 and 1
            assert 0 <= plasma_conditions.ionization_fraction <= 1
    
    def test_magnetic_field_effects(self):
        """Test effects of different magnetic field strengths."""
        # High-temperature plasma conditions
        plasma_conditions = PlasmaConditions(
            electron_density=1e19,
            electron_temperature=30000.0,
            ion_temperature=30000.0,
            magnetic_field=np.array([0.0, 0.0, 0.0]),
            plasma_frequency=1e11,
            debye_length=1e-5,
            ionization_fraction=0.5
        )
        
        velocity = np.array([5000.0, 0.0, 0.0])  # m/s
        
        # Test different magnetic field strengths
        B_strengths = [0.0, 0.001, 0.01, 0.1]  # Tesla
        
        for B_strength in B_strengths:
            magnetic_field = np.array([0.0, B_strength, 0.0])
            
            em_properties = self.em_calculator.calculate_complete_electromagnetic_properties(
                plasma_conditions, velocity, magnetic_field, characteristic_length=1.0
            )
            
            # Hall parameter should increase with magnetic field strength
            if B_strength > 0:
                assert em_properties.hall_parameter > 0
                
                # Stronger magnetic field should give larger Hall parameter
                if B_strength >= 0.01:
                    assert em_properties.hall_parameter > 0.1
    
    def test_property_table_consistency(self):
        """Test consistency between direct calculation and property tables."""
        # Test conditions
        temperature = 20000.0  # K
        pressure = 500.0       # Pa
        
        # Direct calculation
        air_mixture = GasMixture(
            species={'N2': 0.78, 'O2': 0.21, 'Ar': 0.01},
            temperature=temperature,
            pressure=pressure,
            total_density=pressure / (1.380649e-23 * temperature)
        )
        
        direct_electron_density = self.plasma_calculator.calculate_electron_density(air_mixture)
        
        # Table lookup
        table_electron_density, table_ionization_fraction = (
            self.plasma_calculator.get_plasma_properties_from_table(
                'air', temperature, pressure
            )
        )
        
        # Should be reasonably close (within order of magnitude)
        # Some difference expected due to interpolation and different calculation methods
        ratio = direct_electron_density / table_electron_density if table_electron_density > 0 else float('inf')
        assert 0.1 < ratio < 10.0  # Within one order of magnitude
    
    def test_extreme_conditions_stability(self):
        """Test numerical stability under extreme conditions."""
        # Very high temperature and low pressure (edge of space)
        extreme_mixture = GasMixture(
            species={'N2': 0.78, 'O2': 0.22},
            temperature=100000.0,  # 100,000 K
            pressure=1.0,          # 1 Pa
            total_density=1.0 / (1.380649e-23 * 100000.0)
        )
        
        # Should not crash and should return finite values
        plasma_conditions = self.plasma_calculator.calculate_complete_plasma_conditions(
            extreme_mixture
        )
        
        assert np.isfinite(plasma_conditions.electron_density)
        assert np.isfinite(plasma_conditions.plasma_frequency)
        assert np.isfinite(plasma_conditions.debye_length)
        assert 0 <= plasma_conditions.ionization_fraction <= 1
        
        # Very high velocity
        extreme_velocity = np.array([50000.0, 0.0, 0.0])  # 50 km/s
        weak_B_field = np.array([0.0, 0.0, 1e-6])  # Very weak field
        
        em_properties = self.em_calculator.calculate_complete_electromagnetic_properties(
            plasma_conditions, extreme_velocity, weak_B_field, characteristic_length=1.0
        )
        
        # Should return finite values
        assert np.all(np.isfinite(em_properties.electric_field))
        assert np.all(np.isfinite(em_properties.current_density))
        assert np.all(np.isfinite(em_properties.lorentz_force_density))


if __name__ == '__main__':
    pytest.main([__file__])