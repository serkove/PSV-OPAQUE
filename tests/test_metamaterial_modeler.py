"""Tests for metamaterial electromagnetic simulation."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from fighter_jet_sdk.engines.materials.metamaterial_modeler import (
    MetamaterialModeler, FrequencyResponse, FSSSurface
)
from fighter_jet_sdk.common.data_models import MaterialDefinition, EMProperties
from fighter_jet_sdk.common.enums import MaterialType


class TestMetamaterialModeler:
    """Test suite for MetamaterialModeler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.modeler = MetamaterialModeler()
        
        # Create test metamaterial
        self.test_metamaterial = MaterialDefinition(
            name="Test Metamaterial",
            base_material_type=MaterialType.METAMATERIAL,
            electromagnetic_properties=EMProperties(
                permittivity=complex(2.5, -0.1),
                permeability=complex(-1.0, -0.05),
                conductivity=1e6,
                frequency_range=(8e9, 12e9),
                loss_tangent=0.02
            )
        )
        
        # Create test RAM material
        self.test_ram = MaterialDefinition(
            name="Test RAM",
            base_material_type=MaterialType.STEALTH_COATING,
            electromagnetic_properties=EMProperties(
                permittivity=complex(12.0, -3.0),
                permeability=complex(2.5, -1.2),
                conductivity=1e3,
                frequency_range=(1e9, 18e9),
                loss_tangent=0.25
            )
        )
        
        # Test frequencies
        self.test_frequencies = np.linspace(8e9, 12e9, 10)
    
    def test_initialization(self):
        """Test MetamaterialModeler initialization."""
        assert self.modeler.c0 == 299792458.0
        assert self.modeler.mu0 == 4 * np.pi * 1e-7
        assert self.modeler.eps0 == 8.854187817e-12
        assert hasattr(self.modeler, '_benchmark_data')
        assert isinstance(self.modeler._benchmark_data, dict)
    
    def test_calculate_frequency_response_basic(self):
        """Test basic frequency response calculation."""
        response = self.modeler.calculate_frequency_response(
            self.test_metamaterial, self.test_frequencies
        )
        
        assert isinstance(response, FrequencyResponse)
        assert len(response.frequencies) == len(self.test_frequencies)
        assert len(response.permittivity) == len(self.test_frequencies)
        assert len(response.permeability) == len(self.test_frequencies)
        assert len(response.transmission) == len(self.test_frequencies)
        assert len(response.reflection) == len(self.test_frequencies)
        assert len(response.absorption) == len(self.test_frequencies)
    
    def test_calculate_frequency_response_no_em_properties(self):
        """Test frequency response calculation with no EM properties."""
        material_no_em = MaterialDefinition(
            name="No EM Properties",
            base_material_type=MaterialType.METAMATERIAL
        )
        
        with pytest.raises(ValueError, match="Material must have electromagnetic properties"):
            self.modeler.calculate_frequency_response(material_no_em, self.test_frequencies)
    
    def test_frequency_response_energy_conservation(self):
        """Test that frequency response conserves energy."""
        response = self.modeler.calculate_frequency_response(
            self.test_metamaterial, self.test_frequencies
        )
        
        # Check energy conservation: |T|² + |R|² + A = 1
        total_power = (np.abs(response.transmission)**2 + 
                      np.abs(response.reflection)**2 + 
                      response.absorption)
        
        # Allow small numerical errors
        np.testing.assert_allclose(total_power, 1.0, rtol=1e-2)
    
    def test_frequency_response_physical_bounds(self):
        """Test that frequency response values are within physical bounds."""
        response = self.modeler.calculate_frequency_response(
            self.test_metamaterial, self.test_frequencies
        )
        
        # Absorption should be between 0 and 1
        assert np.all(response.absorption >= 0)
        assert np.all(response.absorption <= 1)
        
        # Transmission and reflection coefficients should be reasonable
        assert np.all(np.abs(response.transmission) <= 1.1)  # Allow small numerical errors
        assert np.all(np.abs(response.reflection) <= 1.1)
    
    def test_model_frequency_selective_surface_patch(self):
        """Test FSS modeling with patch elements."""
        fss_config = FSSSurface(
            unit_cell_size=5e-3,
            element_type='patch',
            element_dimensions={'length': 3e-3, 'width': 2e-3},
            substrate_thickness=1e-3,
            substrate_permittivity=complex(4.0, -0.1),
            periodicity=(5e-3, 5e-3)
        )
        
        response = self.modeler.model_frequency_selective_surface(
            fss_config, self.test_frequencies
        )
        
        assert isinstance(response, FrequencyResponse)
        assert len(response.frequencies) == len(self.test_frequencies)
        
        # Check that response is reasonable
        assert np.all(response.absorption >= 0)
        assert np.all(response.absorption <= 1)
    
    def test_model_frequency_selective_surface_slot(self):
        """Test FSS modeling with slot elements."""
        fss_config = FSSSurface(
            unit_cell_size=5e-3,
            element_type='slot',
            element_dimensions={'length': 3e-3, 'width': 0.5e-3},
            substrate_thickness=1e-3,
            substrate_permittivity=complex(4.0, -0.1),
            periodicity=(5e-3, 5e-3)
        )
        
        response = self.modeler.model_frequency_selective_surface(
            fss_config, self.test_frequencies
        )
        
        assert isinstance(response, FrequencyResponse)
        # Slot FSS should show complementary behavior to patch
        assert np.any(np.abs(response.transmission) > 0.5)
    
    def test_model_frequency_selective_surface_dipole(self):
        """Test FSS modeling with dipole elements."""
        fss_config = FSSSurface(
            unit_cell_size=5e-3,
            element_type='dipole',
            element_dimensions={'length': 4e-3, 'width': 0.2e-3},
            substrate_thickness=1e-3,
            substrate_permittivity=complex(4.0, -0.1),
            periodicity=(5e-3, 5e-3)
        )
        
        response = self.modeler.model_frequency_selective_surface(
            fss_config, self.test_frequencies
        )
        
        assert isinstance(response, FrequencyResponse)
        assert len(response.frequencies) == len(self.test_frequencies)
    
    def test_model_frequency_selective_surface_loop(self):
        """Test FSS modeling with loop elements."""
        fss_config = FSSSurface(
            unit_cell_size=5e-3,
            element_type='loop',
            element_dimensions={'radius': 2e-3, 'width': 0.2e-3},
            substrate_thickness=1e-3,
            substrate_permittivity=complex(4.0, -0.1),
            periodicity=(5e-3, 5e-3)
        )
        
        response = self.modeler.model_frequency_selective_surface(
            fss_config, self.test_frequencies
        )
        
        assert isinstance(response, FrequencyResponse)
        assert len(response.frequencies) == len(self.test_frequencies)
    
    def test_model_frequency_selective_surface_invalid_element(self):
        """Test FSS modeling with invalid element type."""
        fss_config = FSSSurface(
            unit_cell_size=5e-3,
            element_type='invalid_element',
            element_dimensions={'length': 3e-3},
            substrate_thickness=1e-3,
            substrate_permittivity=complex(4.0, -0.1),
            periodicity=(5e-3, 5e-3)
        )
        
        with pytest.raises(ValueError, match="Unsupported FSS element type"):
            self.modeler.model_frequency_selective_surface(fss_config, self.test_frequencies)
    
    def test_calculate_ram_effectiveness_basic(self):
        """Test basic RAM effectiveness calculation."""
        thickness = 2e-3  # 2mm
        
        ram_data = self.modeler.calculate_ram_effectiveness(
            self.test_ram, thickness, self.test_frequencies
        )
        
        assert 'absorption' in ram_data
        assert 'reflection_loss_db' in ram_data
        assert 'insertion_loss_db' in ram_data
        assert 'return_loss_db' in ram_data
        assert 'frequencies' in ram_data
        
        # Check array lengths
        assert len(ram_data['absorption']) == len(self.test_frequencies)
        assert len(ram_data['reflection_loss_db']) == len(self.test_frequencies)
        
        # Check physical bounds
        assert np.all(ram_data['absorption'] >= 0)
        assert np.all(ram_data['absorption'] <= 1)
        assert np.all(ram_data['reflection_loss_db'] >= 0)
    
    def test_calculate_ram_effectiveness_no_em_properties(self):
        """Test RAM effectiveness with no EM properties."""
        material_no_em = MaterialDefinition(
            name="No EM Properties",
            base_material_type=MaterialType.STEALTH_COATING
        )
        
        with pytest.raises(ValueError, match="Material must have electromagnetic properties"):
            self.modeler.calculate_ram_effectiveness(
                material_no_em, 1e-3, self.test_frequencies
            )
    
    def test_calculate_ram_effectiveness_with_angle(self):
        """Test RAM effectiveness calculation with incident angle."""
        thickness = 2e-3
        incident_angle = 30.0  # degrees
        
        ram_data = self.modeler.calculate_ram_effectiveness(
            self.test_ram, thickness, self.test_frequencies, incident_angle
        )
        
        assert 'absorption' in ram_data
        assert len(ram_data['absorption']) == len(self.test_frequencies)
        
        # Absorption should generally decrease with incident angle
        ram_data_normal = self.modeler.calculate_ram_effectiveness(
            self.test_ram, thickness, self.test_frequencies, 0.0
        )
        
        # At least some frequencies should show reduced absorption at angle
        mean_absorption_normal = np.mean(ram_data_normal['absorption'])
        mean_absorption_angle = np.mean(ram_data['absorption'])
        
        # This is a general trend, not strict for all materials
        assert mean_absorption_angle <= mean_absorption_normal + 0.1
    
    def test_optimize_ram_thickness(self):
        """Test RAM thickness optimization."""
        target_frequency = 10e9
        target_absorption = 0.9
        
        optimal_thickness = self.modeler.optimize_ram_thickness(
            self.test_ram, target_frequency, target_absorption
        )
        
        assert isinstance(optimal_thickness, float)
        assert optimal_thickness > 0
        assert optimal_thickness < 0.1  # Should be reasonable thickness
        
        # Verify that optimal thickness gives close to target absorption
        ram_data = self.modeler.calculate_ram_effectiveness(
            self.test_ram, optimal_thickness, np.array([target_frequency])
        )
        
        achieved_absorption = ram_data['absorption'][0]
        assert abs(achieved_absorption - target_absorption) < 0.1
    
    def test_optimize_ram_thickness_custom_range(self):
        """Test RAM thickness optimization with custom range."""
        target_frequency = 10e9
        target_absorption = 0.8
        thickness_range = (0.5e-3, 10e-3)
        
        optimal_thickness = self.modeler.optimize_ram_thickness(
            self.test_ram, target_frequency, target_absorption, thickness_range
        )
        
        assert thickness_range[0] <= optimal_thickness <= thickness_range[1]
    
    def test_validate_against_benchmarks_metamaterial(self):
        """Test benchmark validation for metamaterial."""
        validation_results = self.modeler.validate_against_benchmarks(self.test_metamaterial)
        
        assert isinstance(validation_results, dict)
        assert 'srr_resonance_error' in validation_results
        assert 'frequency_continuity_error' in validation_results
        assert 'energy_conservation_error' in validation_results
        
        # All errors should be finite numbers
        for key, value in validation_results.items():
            assert isinstance(value, (int, float))
            assert np.isfinite(value)
    
    def test_validate_against_benchmarks_ram(self):
        """Test benchmark validation for RAM material."""
        validation_results = self.modeler.validate_against_benchmarks(self.test_ram)
        
        assert isinstance(validation_results, dict)
        assert 'salisbury_absorption_error' in validation_results
        assert 'frequency_continuity_error' in validation_results
        assert 'energy_conservation_error' in validation_results
        
        # All errors should be reasonable
        for key, value in validation_results.items():
            assert isinstance(value, (int, float))
            assert np.isfinite(value)
            assert value >= 0
    
    def test_dispersive_properties_calculation(self):
        """Test dispersive properties calculation."""
        em_props = self.test_metamaterial.electromagnetic_properties
        test_freq = 10e9
        
        eps_r, mu_r = self.modeler._calculate_dispersive_properties(em_props, test_freq)
        
        assert isinstance(eps_r, complex)
        assert isinstance(mu_r, complex)
        
        # Should be close to original values at center frequency
        f_center = np.sqrt(em_props.frequency_range[0] * em_props.frequency_range[1])
        eps_center, mu_center = self.modeler._calculate_dispersive_properties(em_props, f_center)
        
        # At center frequency, should be reasonably close to specified values
        assert abs(eps_center.real - em_props.permittivity.real) < 2.0
        assert abs(mu_center.real - em_props.permeability.real) < 2.0
    
    def test_transmission_reflection_calculation(self):
        """Test transmission and reflection coefficient calculation."""
        eps_r = complex(2.5, -0.1)
        mu_r = complex(-1.0, -0.05)
        frequency = 10e9
        thickness = 1e-3
        
        t_coeff, r_coeff = self.modeler._calculate_transmission_reflection(
            eps_r, mu_r, frequency, thickness
        )
        
        assert isinstance(t_coeff, complex)
        assert isinstance(r_coeff, complex)
        
        # Coefficients should be reasonable
        assert abs(t_coeff) <= 1.1  # Allow small numerical errors
        assert abs(r_coeff) <= 1.1
    
    def test_energy_conservation_validation(self):
        """Test energy conservation validation method."""
        error = self.modeler._validate_energy_conservation(self.test_metamaterial)
        
        assert isinstance(error, float)
        assert error >= 0
        assert error < 0.1  # Should be small for well-behaved materials
    
    def test_frequency_continuity_validation(self):
        """Test frequency continuity validation method."""
        error = self.modeler._validate_frequency_continuity(self.test_metamaterial)
        
        assert isinstance(error, float)
        assert error >= 0
        # Should be reasonable for dispersive materials
        assert error < 10.0
    
    def test_srr_benchmark_validation(self):
        """Test split-ring resonator benchmark validation."""
        error = self.modeler._validate_srr_benchmark(self.test_metamaterial)
        
        assert isinstance(error, float)
        assert error >= 0
        # Error should be finite
        assert np.isfinite(error)
    
    def test_salisbury_benchmark_validation(self):
        """Test Salisbury screen benchmark validation."""
        error = self.modeler._validate_salisbury_benchmark(self.test_ram)
        
        assert isinstance(error, float)
        assert error >= 0
        assert np.isfinite(error)
    
    def test_extract_effective_parameters(self):
        """Test effective parameter extraction from S-parameters."""
        n_freq = 5
        transmission = np.random.random(n_freq) * 0.5 + 0.1j * np.random.random(n_freq)
        reflection = np.random.random(n_freq) * 0.3 + 0.05j * np.random.random(n_freq)
        frequencies = np.linspace(8e9, 12e9, n_freq)
        thickness = 1e-3
        
        permittivity, permeability = self.modeler._extract_effective_parameters(
            transmission, reflection, frequencies, thickness
        )
        
        assert len(permittivity) == n_freq
        assert len(permeability) == n_freq
        assert all(isinstance(eps, complex) for eps in permittivity)
        assert all(isinstance(mu, complex) for mu in permeability)
    
    def test_benchmark_data_loading(self):
        """Test benchmark data loading."""
        benchmark_data = self.modeler._load_benchmark_data()
        
        assert isinstance(benchmark_data, dict)
        assert 'srr_10ghz' in benchmark_data
        assert 'salisbury_screen' in benchmark_data
        
        # Check SRR benchmark structure
        srr_data = benchmark_data['srr_10ghz']
        assert 'frequency' in srr_data
        assert 'expected_mu_real' in srr_data
        assert 'expected_mu_imag' in srr_data
        
        # Check Salisbury screen benchmark structure
        salisbury_data = benchmark_data['salisbury_screen']
        assert 'frequency' in salisbury_data
        assert 'expected_absorption' in salisbury_data
        assert 'thickness_ratio' in salisbury_data


class TestFrequencyResponse:
    """Test suite for FrequencyResponse dataclass."""
    
    def test_frequency_response_creation(self):
        """Test FrequencyResponse creation."""
        frequencies = np.linspace(8e9, 12e9, 10)
        n_freq = len(frequencies)
        
        response = FrequencyResponse(
            frequencies=frequencies,
            permittivity=np.ones(n_freq, dtype=complex),
            permeability=np.ones(n_freq, dtype=complex),
            transmission=np.ones(n_freq, dtype=complex) * 0.5,
            reflection=np.ones(n_freq, dtype=complex) * 0.3,
            absorption=np.ones(n_freq) * 0.2
        )
        
        assert len(response.frequencies) == n_freq
        assert len(response.permittivity) == n_freq
        assert len(response.permeability) == n_freq
        assert len(response.transmission) == n_freq
        assert len(response.reflection) == n_freq
        assert len(response.absorption) == n_freq


class TestFSSSurface:
    """Test suite for FSSSurface dataclass."""
    
    def test_fss_surface_creation(self):
        """Test FSSSurface creation."""
        fss = FSSSurface(
            unit_cell_size=5e-3,
            element_type='patch',
            element_dimensions={'length': 3e-3, 'width': 2e-3},
            substrate_thickness=1e-3,
            substrate_permittivity=complex(4.0, -0.1),
            periodicity=(5e-3, 5e-3)
        )
        
        assert fss.unit_cell_size == 5e-3
        assert fss.element_type == 'patch'
        assert fss.element_dimensions['length'] == 3e-3
        assert fss.element_dimensions['width'] == 2e-3
        assert fss.substrate_thickness == 1e-3
        assert fss.substrate_permittivity == complex(4.0, -0.1)
        assert fss.periodicity == (5e-3, 5e-3)


if __name__ == '__main__':
    pytest.main([__file__])