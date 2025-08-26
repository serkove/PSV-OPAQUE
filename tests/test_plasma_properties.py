"""Unit tests for plasma properties calculation module."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from fighter_jet_sdk.common.plasma_physics import (
    PlasmaPropertiesCalculator, PlasmaPropertyInterpolator,
    PlasmaConditions, GasMixture, PlasmaRegime,
    BOLTZMANN_CONSTANT, ELEMENTARY_CHARGE, ELECTRON_MASS, VACUUM_PERMITTIVITY
)


class TestPlasmaPropertiesCalculator:
    """Test cases for PlasmaPropertiesCalculator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = PlasmaPropertiesCalculator()
        
        # Standard test conditions
        self.test_gas_mixture = GasMixture(
            species={'N2': 0.78, 'O2': 0.21, 'Ar': 0.01},
            temperature=10000.0,  # 10,000 K
            pressure=1000.0,      # 1000 Pa
            total_density=1000.0 / (BOLTZMANN_CONSTANT * 10000.0)
        )
    
    def test_initialization(self):
        """Test calculator initialization."""
        assert self.calculator is not None
        assert 'N2' in self.calculator.ionization_potentials
        assert 'O2' in self.calculator.ionization_potentials
        assert 'air' in self.calculator.property_tables
        
    def test_calculate_electron_density_basic(self):
        """Test basic electron density calculation."""
        electron_density = self.calculator.calculate_electron_density(self.test_gas_mixture)
        
        # Should be positive
        assert electron_density > 0
        
        # Should be less than total density
        total_density = (self.test_gas_mixture.pressure / 
                        (BOLTZMANN_CONSTANT * self.test_gas_mixture.temperature))
        assert electron_density <= total_density
    
    def test_calculate_electron_density_temperature_dependence(self):
        """Test electron density temperature dependence."""
        # Low temperature case
        low_temp_mixture = GasMixture(
            species={'N2': 1.0},
            temperature=1000.0,
            pressure=1000.0,
            total_density=1000.0 / (BOLTZMANN_CONSTANT * 1000.0)
        )
        
        # High temperature case
        high_temp_mixture = GasMixture(
            species={'N2': 1.0},
            temperature=50000.0,
            pressure=1000.0,
            total_density=1000.0 / (BOLTZMANN_CONSTANT * 50000.0)
        )
        
        ne_low = self.calculator.calculate_electron_density(low_temp_mixture)
        ne_high = self.calculator.calculate_electron_density(high_temp_mixture)
        
        # Higher temperature should give higher electron density
        assert ne_high > ne_low
    
    def test_saha_equation_implementation(self):
        """Test Saha equation implementation."""
        # Test with known conditions
        species = {'H': 1.0}  # Pure hydrogen
        temperature = 10000.0  # K
        total_density = 1e20  # m^-3
        
        alpha = self.calculator._calculate_ionization_fraction_saha(
            species, temperature, total_density
        )
        
        # Should be between 0 and 1
        assert 0 <= alpha <= 1
        
        # For hydrogen at 10,000K, should have significant ionization
        assert alpha > 0.01
    
    def test_plasma_frequency_calculation(self):
        """Test plasma frequency calculation."""
        electron_density = 1e18  # m^-3
        
        plasma_freq = self.calculator.calculate_plasma_frequency(electron_density)
        
        # Should be positive
        assert plasma_freq > 0
        
        # Check against analytical formula
        expected_freq = np.sqrt(
            electron_density * ELEMENTARY_CHARGE**2 / 
            (VACUUM_PERMITTIVITY * ELECTRON_MASS)
        ) / (2 * np.pi)
        
        assert abs(plasma_freq - expected_freq) / expected_freq < 1e-10
    
    def test_debye_length_calculation(self):
        """Test Debye length calculation."""
        electron_density = 1e18  # m^-3
        electron_temperature = 10000.0  # K
        
        debye_length = self.calculator.calculate_debye_length(
            electron_density, electron_temperature
        )
        
        # Should be positive
        assert debye_length > 0
        
        # Check against analytical formula
        expected_length = np.sqrt(
            VACUUM_PERMITTIVITY * BOLTZMANN_CONSTANT * electron_temperature / 
            (electron_density * ELEMENTARY_CHARGE**2)
        )
        
        assert abs(debye_length - expected_length) / expected_length < 1e-10
    
    def test_cyclotron_frequency_calculation(self):
        """Test cyclotron frequency calculation."""
        magnetic_field = 1.0  # Tesla
        particle_mass = ELECTRON_MASS
        charge = ELEMENTARY_CHARGE
        
        cyclotron_freq = self.calculator.calculate_cyclotron_frequency(
            magnetic_field, particle_mass, charge
        )
        
        # Should be positive
        assert cyclotron_freq > 0
        
        # Check against analytical formula
        expected_freq = (charge * magnetic_field / particle_mass) / (2 * np.pi)
        
        assert abs(cyclotron_freq - expected_freq) / expected_freq < 1e-10
    
    def test_plasma_regime_determination(self):
        """Test plasma regime classification."""
        # Weakly ionized plasma
        weak_plasma = PlasmaConditions(
            electron_density=1e15,
            electron_temperature=5000.0,
            ion_temperature=5000.0,
            magnetic_field=np.array([0.0, 0.0, 0.0]),
            plasma_frequency=1e9,
            debye_length=1e-4,
            ionization_fraction=0.005
        )
        
        regime = self.calculator.determine_plasma_regime(weak_plasma)
        assert regime == PlasmaRegime.WEAKLY_IONIZED
        
        # Fully ionized plasma
        full_plasma = PlasmaConditions(
            electron_density=1e20,
            electron_temperature=50000.0,
            ion_temperature=50000.0,
            magnetic_field=np.array([0.0, 0.0, 0.0]),
            plasma_frequency=1e12,
            debye_length=1e-6,
            ionization_fraction=0.95
        )
        
        regime = self.calculator.determine_plasma_regime(full_plasma)
        assert regime == PlasmaRegime.FULLY_IONIZED
        
        # Magnetized plasma
        magnetized_plasma = PlasmaConditions(
            electron_density=1e20,
            electron_temperature=50000.0,
            ion_temperature=50000.0,
            magnetic_field=np.array([0.1, 0.0, 0.0]),
            plasma_frequency=1e12,
            debye_length=1e-6,
            ionization_fraction=0.95
        )
        
        regime = self.calculator.determine_plasma_regime(magnetized_plasma)
        assert regime == PlasmaRegime.MAGNETIZED_PLASMA
    
    def test_complete_plasma_conditions_calculation(self):
        """Test complete plasma conditions calculation."""
        magnetic_field = np.array([0.01, 0.0, 0.0])  # 0.01 Tesla
        
        plasma_conditions = self.calculator.calculate_complete_plasma_conditions(
            self.test_gas_mixture, magnetic_field
        )
        
        # Check all properties are calculated
        assert plasma_conditions.electron_density > 0
        assert plasma_conditions.plasma_frequency > 0
        assert plasma_conditions.debye_length > 0
        assert 0 <= plasma_conditions.ionization_fraction <= 1
        assert np.array_equal(plasma_conditions.magnetic_field, magnetic_field)
        assert plasma_conditions.regime in PlasmaRegime
    
    def test_property_table_interpolation(self):
        """Test property table interpolation."""
        # Test with air mixture
        temperature = 15000.0  # K
        pressure = 500.0       # Pa
        
        electron_density, ionization_fraction = self.calculator.get_plasma_properties_from_table(
            'air', temperature, pressure
        )
        
        # Should return reasonable values
        assert electron_density > 0
        assert 0 <= ionization_fraction <= 1
    
    def test_property_table_bounds_checking(self):
        """Test property table bounds checking."""
        # Test with extreme values
        very_high_temp = 1e8  # K
        very_low_pressure = 1e-5  # Pa
        
        electron_density, ionization_fraction = self.calculator.get_plasma_properties_from_table(
            'air', very_high_temp, very_low_pressure
        )
        
        # Should not crash and return reasonable values
        assert electron_density >= 0
        assert 0 <= ionization_fraction <= 1
    
    def test_invalid_mixture_name(self):
        """Test handling of invalid mixture names."""
        with pytest.raises(ValueError, match="No property table available"):
            self.calculator.get_plasma_properties_from_table(
                'invalid_mixture', 10000.0, 1000.0
            )


class TestPlasmaPropertyInterpolator:
    """Test cases for PlasmaPropertyInterpolator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.interpolator = PlasmaPropertyInterpolator()
    
    def test_initialization(self):
        """Test interpolator initialization."""
        assert self.interpolator is not None
        assert self.interpolator.calculator is not None
    
    def test_create_property_table(self):
        """Test property table creation."""
        species_composition = {'N2': 0.8, 'O2': 0.2}
        temperature_range = (5000.0, 50000.0)
        pressure_range = (100.0, 10000.0)
        
        table = self.interpolator.create_property_table(
            species_composition, temperature_range, pressure_range,
            num_temp_points=10, num_pressure_points=10
        )
        
        # Check that all expected properties are in table
        expected_properties = [
            'electron_density', 'plasma_frequency', 
            'debye_length', 'ionization_fraction'
        ]
        
        for prop in expected_properties:
            assert prop in table
            assert callable(table[prop])
    
    def test_interpolated_properties_retrieval(self):
        """Test retrieval of interpolated properties."""
        # Create a simple table
        species_composition = {'N2': 1.0}
        temperature_range = (5000.0, 20000.0)
        pressure_range = (500.0, 2000.0)
        
        table = self.interpolator.create_property_table(
            species_composition, temperature_range, pressure_range,
            num_temp_points=5, num_pressure_points=5
        )
        
        # Get properties at intermediate point
        temperature = 10000.0
        pressure = 1000.0
        
        properties = self.interpolator.get_interpolated_properties(
            table, temperature, pressure
        )
        
        # Check all properties are returned
        expected_properties = [
            'electron_density', 'plasma_frequency', 
            'debye_length', 'ionization_fraction'
        ]
        
        for prop in expected_properties:
            assert prop in properties
            assert isinstance(properties[prop], float)
            assert properties[prop] >= 0


class TestPlasmaConditionsDataClass:
    """Test cases for PlasmaConditions data class."""
    
    def test_plasma_conditions_creation(self):
        """Test PlasmaConditions creation."""
        conditions = PlasmaConditions(
            electron_density=1e18,
            electron_temperature=10000.0,
            ion_temperature=10000.0,
            magnetic_field=np.array([0.01, 0.0, 0.0]),
            plasma_frequency=1e10,
            debye_length=1e-5,
            ionization_fraction=0.1
        )
        
        assert conditions.electron_density == 1e18
        assert conditions.electron_temperature == 10000.0
        assert conditions.regime == PlasmaRegime.WEAKLY_IONIZED  # Default value


class TestGasMixtureDataClass:
    """Test cases for GasMixture data class."""
    
    def test_gas_mixture_creation(self):
        """Test GasMixture creation."""
        mixture = GasMixture(
            species={'N2': 0.78, 'O2': 0.22},
            temperature=10000.0,
            pressure=1000.0,
            total_density=1e20
        )
        
        assert mixture.species['N2'] == 0.78
        assert mixture.species['O2'] == 0.22
        assert mixture.temperature == 10000.0
        assert mixture.pressure == 1000.0
        assert mixture.total_density == 1e20


class TestPhysicalConstants:
    """Test physical constants used in calculations."""
    
    def test_constants_values(self):
        """Test that physical constants have reasonable values."""
        assert BOLTZMANN_CONSTANT > 0
        assert ELEMENTARY_CHARGE > 0
        assert ELECTRON_MASS > 0
        assert VACUUM_PERMITTIVITY > 0
        
        # Check approximate values
        assert abs(BOLTZMANN_CONSTANT - 1.38e-23) / 1.38e-23 < 0.01
        assert abs(ELEMENTARY_CHARGE - 1.60e-19) / 1.60e-19 < 0.01
        assert abs(ELECTRON_MASS - 9.11e-31) / 9.11e-31 < 0.01


if __name__ == '__main__':
    pytest.main([__file__])