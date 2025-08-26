"""Tests for stealth analysis and RCS calculation system."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from fighter_jet_sdk.engines.materials.stealth_analyzer import (
    StealthAnalyzer, RCSData, GeometryModel, StealthConfiguration
)
from fighter_jet_sdk.common.data_models import MaterialDefinition, EMProperties
from fighter_jet_sdk.common.enums import MaterialType


class TestStealthAnalyzer:
    """Test suite for StealthAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = StealthAnalyzer()
        
        # Create test geometry
        self.test_geometry = GeometryModel(
            fuselage_length=15.0,
            fuselage_diameter=1.5,
            wing_span=12.0,
            wing_chord=3.0,
            wing_thickness=0.3,
            tail_area=8.0,
            engine_inlet_area=1.2,
            surface_materials={
                'fuselage': 'aluminum',
                'wing': 'aluminum',
                'tail': 'aluminum'
            }
        )
        
        # Create test materials database
        self.materials_db = {
            'aluminum': MaterialDefinition(
                name="Aluminum Alloy",
                base_material_type=MaterialType.CONVENTIONAL_METAL,
                electromagnetic_properties=EMProperties(
                    permittivity=complex(1.0, 0.0),
                    permeability=complex(1.0, 0.0),
                    conductivity=3.5e7,
                    frequency_range=(1e9, 40e9),
                    loss_tangent=0.0
                )
            ),
            'ram_coating': MaterialDefinition(
                name="RAM Coating",
                base_material_type=MaterialType.STEALTH_COATING,
                electromagnetic_properties=EMProperties(
                    permittivity=complex(12.0, -3.0),
                    permeability=complex(2.5, -1.2),
                    conductivity=1e3,
                    frequency_range=(1e9, 18e9),
                    loss_tangent=0.25
                )
            ),
            'metamaterial': MaterialDefinition(
                name="Metamaterial",
                base_material_type=MaterialType.METAMATERIAL,
                electromagnetic_properties=EMProperties(
                    permittivity=complex(2.5, -0.1),
                    permeability=complex(-1.0, -0.05),
                    conductivity=1e6,
                    frequency_range=(8e9, 12e9),
                    loss_tangent=0.02
                )
            )
        }
        
        # Test parameters
        self.test_frequencies = np.array([8e9, 10e9, 12e9])
        self.test_angles = np.array([-90, 0, 90])
    
    def test_initialization(self):
        """Test StealthAnalyzer initialization."""
        assert self.analyzer.c0 == 299792458.0
        assert hasattr(self.analyzer, '_rcs_methods')
        assert hasattr(self.analyzer, 'radar_bands')
        
        # Check radar bands
        assert 'X' in self.analyzer.radar_bands
        assert self.analyzer.radar_bands['X'] == (8e9, 12e9)
    
    def test_calculate_aircraft_rcs_basic(self):
        """Test basic aircraft RCS calculation."""
        rcs_data = self.analyzer.calculate_aircraft_rcs(
            self.test_geometry,
            self.materials_db,
            self.test_frequencies,
            self.test_angles
        )
        
        assert isinstance(rcs_data, RCSData)
        assert np.array_equal(rcs_data.frequencies, self.test_frequencies)
        assert np.array_equal(rcs_data.angles, self.test_angles)
        assert rcs_data.rcs_matrix.shape == (len(self.test_frequencies), len(self.test_angles))
        assert rcs_data.polarization == 'VV'
        assert rcs_data.incident_type == 'monostatic'
        
        # RCS values should be positive
        assert np.all(rcs_data.rcs_matrix > 0)
    
    def test_calculate_aircraft_rcs_different_methods(self):
        """Test RCS calculation with different methods."""
        methods = ['physical_optics', 'method_of_moments', 'geometric_theory', 'hybrid']
        
        for method in methods:
            rcs_data = self.analyzer.calculate_aircraft_rcs(
                self.test_geometry,
                self.materials_db,
                self.test_frequencies,
                self.test_angles,
                method=method
            )
            
            assert isinstance(rcs_data, RCSData)
            assert np.all(rcs_data.rcs_matrix > 0)
    
    def test_calculate_aircraft_rcs_invalid_method(self):
        """Test RCS calculation with invalid method."""
        with pytest.raises(ValueError, match="Unknown RCS calculation method"):
            self.analyzer.calculate_aircraft_rcs(
                self.test_geometry,
                self.materials_db,
                self.test_frequencies,
                self.test_angles,
                method='invalid_method'
            )
    
    def test_calculate_aircraft_rcs_different_polarizations(self):
        """Test RCS calculation with different polarizations."""
        polarizations = ['VV', 'HH', 'VH', 'HV']
        
        for pol in polarizations:
            rcs_data = self.analyzer.calculate_aircraft_rcs(
                self.test_geometry,
                self.materials_db,
                self.test_frequencies,
                self.test_angles,
                polarization=pol
            )
            
            assert rcs_data.polarization == pol
            assert np.all(rcs_data.rcs_matrix > 0)
    
    def test_analyze_multi_frequency_rcs(self):
        """Test multi-frequency RCS analysis."""
        results = self.analyzer.analyze_multi_frequency_rcs(
            self.test_geometry,
            self.materials_db,
            radar_bands=['X', 'Ku']
        )
        
        assert isinstance(results, dict)
        assert 'X' in results
        assert 'Ku' in results
        
        for band_name, rcs_data in results.items():
            assert isinstance(rcs_data, RCSData)
            assert len(rcs_data.frequencies) == 21  # Default frequency points
            assert len(rcs_data.angles) == 73      # Default angle points
            assert np.all(rcs_data.rcs_matrix > 0)
    
    def test_analyze_multi_frequency_rcs_default_bands(self):
        """Test multi-frequency RCS analysis with default bands."""
        results = self.analyzer.analyze_multi_frequency_rcs(
            self.test_geometry,
            self.materials_db
        )
        
        assert isinstance(results, dict)
        assert len(results) == 5  # Default bands: L, S, C, X, Ku
        
        expected_bands = ['L', 'S', 'C', 'X', 'Ku']
        for band in expected_bands:
            assert band in results
    
    def test_analyze_multi_frequency_rcs_invalid_band(self):
        """Test multi-frequency RCS analysis with invalid band."""
        # Should not raise error, just log warning and skip invalid band
        results = self.analyzer.analyze_multi_frequency_rcs(
            self.test_geometry,
            self.materials_db,
            radar_bands=['X', 'invalid_band']
        )
        
        assert 'X' in results
        assert 'invalid_band' not in results
    
    def test_optimize_stealth_configuration(self):
        """Test stealth configuration optimization."""
        stealth_config = StealthConfiguration(
            target_rcs_reduction=20.0,  # 20 dB reduction
            priority_frequencies=[10e9],
            priority_angles=[0.0],
            material_constraints={
                'fuselage': ['aluminum', 'ram_coating'],
                'wing': ['aluminum', 'metamaterial'],
                'tail': ['aluminum', 'ram_coating']
            },
            weight_penalty=10.0,  # kg/m²
            cost_penalty=1000.0   # $/m²
        )
        
        optimal_materials = self.analyzer.optimize_stealth_configuration(
            self.test_geometry,
            self.materials_db,
            stealth_config
        )
        
        assert isinstance(optimal_materials, dict)
        assert 'fuselage' in optimal_materials
        assert 'wing' in optimal_materials
        assert 'tail' in optimal_materials
        
        # Check that selected materials are from allowed lists
        assert optimal_materials['fuselage'] in ['aluminum', 'ram_coating']
        assert optimal_materials['wing'] in ['aluminum', 'metamaterial']
        assert optimal_materials['tail'] in ['aluminum', 'ram_coating']
    
    def test_optimize_stealth_configuration_no_materials(self):
        """Test stealth optimization with no available materials."""
        stealth_config = StealthConfiguration(
            target_rcs_reduction=20.0,
            priority_frequencies=[10e9],
            priority_angles=[0.0],
            material_constraints={
                'fuselage': ['nonexistent_material']
            },
            weight_penalty=10.0,
            cost_penalty=1000.0
        )
        
        with pytest.raises(ValueError, match="No available materials for surface"):
            self.analyzer.optimize_stealth_configuration(
                self.test_geometry,
                self.materials_db,
                stealth_config
            )
    
    def test_calculate_signature_management_effectiveness(self):
        """Test signature management effectiveness calculation."""
        # Create baseline and stealth RCS data
        baseline_rcs = RCSData(
            frequencies=self.test_frequencies,
            angles=self.test_angles,
            rcs_matrix=np.ones((3, 3)) * 10.0,  # 10 m² baseline
            polarization='VV',
            incident_type='monostatic'
        )
        
        stealth_rcs = RCSData(
            frequencies=self.test_frequencies,
            angles=self.test_angles,
            rcs_matrix=np.ones((3, 3)) * 1.0,   # 1 m² with stealth
            polarization='VV',
            incident_type='monostatic'
        )
        
        metrics = self.analyzer.calculate_signature_management_effectiveness(
            baseline_rcs, stealth_rcs
        )
        
        assert isinstance(metrics, dict)
        assert 'mean_rcs_reduction_db' in metrics
        assert 'max_rcs_reduction_db' in metrics
        assert 'min_rcs_reduction_db' in metrics
        assert 'std_rcs_reduction_db' in metrics
        assert 'frontal_rcs_reduction_db' in metrics
        assert 'side_rcs_reduction_db' in metrics
        assert 'rear_rcs_reduction_db' in metrics
        assert 'mean_detection_range_factor' in metrics
        
        # 10 dB reduction expected (10 m² -> 1 m²)
        expected_reduction = 10.0  # 10 * log10(10/1)
        assert abs(metrics['mean_rcs_reduction_db'] - expected_reduction) < 0.1
        
        # Detection range factor should be around 0.56 (10^(-1/4))
        expected_range_factor = 0.56
        assert abs(metrics['mean_detection_range_factor'] - expected_range_factor) < 0.1
    
    def test_calculate_signature_management_effectiveness_mismatched_data(self):
        """Test effectiveness calculation with mismatched data."""
        baseline_rcs = RCSData(
            frequencies=np.array([8e9, 10e9]),
            angles=self.test_angles,
            rcs_matrix=np.ones((2, 3)),
            polarization='VV',
            incident_type='monostatic'
        )
        
        stealth_rcs = RCSData(
            frequencies=self.test_frequencies,  # Different frequencies
            angles=self.test_angles,
            rcs_matrix=np.ones((3, 3)),
            polarization='VV',
            incident_type='monostatic'
        )
        
        with pytest.raises(ValueError, match="Frequency arrays must match"):
            self.analyzer.calculate_signature_management_effectiveness(
                baseline_rcs, stealth_rcs
            )
    
    def test_physical_optics_rcs_calculation(self):
        """Test physical optics RCS calculation method."""
        rcs = self.analyzer._calculate_rcs_physical_optics(
            self.test_geometry,
            self.materials_db,
            10e9,  # 10 GHz
            0.0,   # 0 degrees
            'VV'
        )
        
        assert isinstance(rcs, float)
        assert rcs > 0
        assert rcs >= 1e-10  # Minimum RCS floor
    
    def test_method_of_moments_rcs_calculation(self):
        """Test method of moments RCS calculation."""
        rcs = self.analyzer._calculate_rcs_mom(
            self.test_geometry,
            self.materials_db,
            10e9,
            0.0,
            'VV'
        )
        
        assert isinstance(rcs, float)
        assert rcs > 0
        assert rcs >= 1e-10
    
    def test_gtd_rcs_calculation(self):
        """Test geometric theory of diffraction RCS calculation."""
        rcs = self.analyzer._calculate_rcs_gtd(
            self.test_geometry,
            self.materials_db,
            10e9,
            0.0,
            'VV'
        )
        
        assert isinstance(rcs, float)
        assert rcs > 0
        assert rcs >= 1e-10
    
    def test_hybrid_rcs_calculation(self):
        """Test hybrid RCS calculation method."""
        # Test different frequency regimes
        frequencies = [1e9, 5e9, 20e9]  # Low, medium, high frequency
        
        for freq in frequencies:
            rcs = self.analyzer._calculate_rcs_hybrid(
                self.test_geometry,
                self.materials_db,
                freq,
                0.0,
                'VV'
            )
            
            assert isinstance(rcs, float)
            assert rcs > 0
            assert rcs >= 1e-10
    
    def test_rayleigh_rcs_calculation(self):
        """Test Rayleigh scattering RCS calculation."""
        rcs = self.analyzer._calculate_rcs_rayleigh(
            self.test_geometry,
            self.materials_db,
            1e9,  # Low frequency for Rayleigh regime
            0.0
        )
        
        assert isinstance(rcs, float)
        assert rcs > 0
        assert rcs >= 1e-10
    
    def test_gtd_correction_calculation(self):
        """Test GTD correction calculation."""
        correction = self.analyzer._calculate_gtd_correction(
            self.test_geometry,
            10e9,
            45.0
        )
        
        assert isinstance(correction, float)
        assert correction >= 0  # Correction should be additive
    
    def test_material_reflection_coefficient(self):
        """Test material reflection coefficient calculation."""
        material = self.materials_db['ram_coating']
        
        r_coeff = self.analyzer._get_material_reflection_coefficient(
            material, 10e9, 0.0
        )
        
        assert isinstance(r_coeff, complex)
        assert abs(r_coeff) <= 1.0  # Reflection coefficient magnitude should be <= 1
    
    def test_material_reflection_coefficient_no_em_properties(self):
        """Test reflection coefficient for material without EM properties."""
        material = MaterialDefinition(
            name="No EM Properties",
            base_material_type=MaterialType.CONVENTIONAL_METAL
        )
        
        r_coeff = self.analyzer._get_material_reflection_coefficient(
            material, 10e9, 0.0
        )
        
        assert isinstance(r_coeff, complex)
        assert r_coeff == complex(0.9, 0.0)  # Default metallic reflection
    
    def test_material_reflection_coefficient_out_of_range(self):
        """Test reflection coefficient for frequency out of material range."""
        material = self.materials_db['metamaterial']  # Valid range: 8-12 GHz
        
        r_coeff = self.analyzer._get_material_reflection_coefficient(
            material, 20e9, 0.0  # Outside valid range
        )
        
        assert isinstance(r_coeff, complex)
        assert r_coeff == complex(0.9, 0.0)  # Default outside range
    
    def test_rcs_angle_dependence(self):
        """Test that RCS varies with angle as expected."""
        angles = np.array([0, 45, 90])
        rcs_values = []
        
        for angle in angles:
            rcs = self.analyzer._calculate_rcs_physical_optics(
                self.test_geometry,
                self.materials_db,
                10e9,
                angle,
                'VV'
            )
            rcs_values.append(rcs)
        
        # Check that there's some variation (relative difference > 1%)
        max_val = max(rcs_values)
        min_val = min(rcs_values)
        relative_variation = (max_val - min_val) / max_val
        assert relative_variation > 0.01  # At least 1% variation
        
        # All values should be positive
        assert all(rcs > 0 for rcs in rcs_values)
    
    def test_rcs_frequency_dependence(self):
        """Test that RCS varies with frequency as expected."""
        frequencies = np.array([8e9, 10e9, 12e9])
        rcs_values = []
        
        for freq in frequencies:
            rcs = self.analyzer._calculate_rcs_physical_optics(
                self.test_geometry,
                self.materials_db,
                freq,
                0.0,
                'VV'
            )
            rcs_values.append(rcs)
        
        # RCS should vary with frequency
        assert not np.allclose(rcs_values, rcs_values[0])
        
        # All values should be positive
        assert all(rcs > 0 for rcs in rcs_values)


class TestRCSData:
    """Test suite for RCSData dataclass."""
    
    def test_rcs_data_creation(self):
        """Test RCSData creation."""
        frequencies = np.array([8e9, 10e9, 12e9])
        angles = np.array([-90, 0, 90])
        rcs_matrix = np.random.random((3, 3)) * 10
        
        rcs_data = RCSData(
            frequencies=frequencies,
            angles=angles,
            rcs_matrix=rcs_matrix,
            polarization='VV',
            incident_type='monostatic'
        )
        
        assert np.array_equal(rcs_data.frequencies, frequencies)
        assert np.array_equal(rcs_data.angles, angles)
        assert np.array_equal(rcs_data.rcs_matrix, rcs_matrix)
        assert rcs_data.polarization == 'VV'
        assert rcs_data.incident_type == 'monostatic'


class TestGeometryModel:
    """Test suite for GeometryModel dataclass."""
    
    def test_geometry_model_creation(self):
        """Test GeometryModel creation."""
        geometry = GeometryModel(
            fuselage_length=15.0,
            fuselage_diameter=1.5,
            wing_span=12.0,
            wing_chord=3.0,
            wing_thickness=0.3,
            tail_area=8.0,
            engine_inlet_area=1.2,
            surface_materials={'fuselage': 'aluminum'}
        )
        
        assert geometry.fuselage_length == 15.0
        assert geometry.fuselage_diameter == 1.5
        assert geometry.wing_span == 12.0
        assert geometry.wing_chord == 3.0
        assert geometry.wing_thickness == 0.3
        assert geometry.tail_area == 8.0
        assert geometry.engine_inlet_area == 1.2
        assert geometry.surface_materials['fuselage'] == 'aluminum'


class TestStealthConfiguration:
    """Test suite for StealthConfiguration dataclass."""
    
    def test_stealth_configuration_creation(self):
        """Test StealthConfiguration creation."""
        config = StealthConfiguration(
            target_rcs_reduction=20.0,
            priority_frequencies=[10e9, 12e9],
            priority_angles=[0.0, 45.0],
            material_constraints={'fuselage': ['aluminum', 'ram']},
            weight_penalty=10.0,
            cost_penalty=1000.0
        )
        
        assert config.target_rcs_reduction == 20.0
        assert config.priority_frequencies == [10e9, 12e9]
        assert config.priority_angles == [0.0, 45.0]
        assert config.material_constraints['fuselage'] == ['aluminum', 'ram']
        assert config.weight_penalty == 10.0
        assert config.cost_penalty == 1000.0


if __name__ == '__main__':
    pytest.main([__file__])