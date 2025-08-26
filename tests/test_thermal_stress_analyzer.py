"""Tests for thermal stress analyzer."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from fighter_jet_sdk.engines.structural.thermal_stress_analyzer import (
    ThermalStressAnalyzer, ThermalLoadConditions, ThermalStressResults, StructuralGeometry
)
from fighter_jet_sdk.common.data_models import (
    MaterialDefinition, ThermalProperties, MechanicalProperties, MaterialType
)


class TestThermalStressAnalyzer:
    """Test cases for ThermalStressAnalyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = ThermalStressAnalyzer()
        
        # Create test geometry
        self.geometry = StructuralGeometry(
            nodes=np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]),
            elements=np.array([[0, 1], [1, 2], [2, 3]]),
            element_type='beam',
            thickness=np.array([0.01, 0.01, 0.01])
        )
        
        # Create test materials
        thermal_props = ThermalProperties(
            thermal_conductivity=50.0,
            specific_heat=500.0,
            density=7800.0,
            melting_point=1800.0,
            operating_temp_range=(200.0, 1500.0)
        )
        
        mechanical_props = MechanicalProperties(
            youngs_modulus=200e9,
            poissons_ratio=0.3,
            yield_strength=350e6,
            ultimate_strength=500e6,
            fatigue_limit=200e6,
            density=7800.0
        )
        
        self.materials = {
            'steel': MaterialDefinition(
                name='High-strength steel',
                base_material_type=MaterialType.CONVENTIONAL_METAL,
                thermal_properties=thermal_props,
                mechanical_properties=mechanical_props
            )
        }
        
        # Create test thermal loads
        self.thermal_loads = ThermalLoadConditions(
            temperature_distribution=np.array([300.0, 800.0, 1200.0, 1000.0]),
            temperature_gradient=np.array([500.0, 400.0, -200.0, -300.0]),
            heat_flux=np.array([1e6, 2e6, 1.5e6, 1e6])
        )
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = ThermalStressAnalyzer()
        assert analyzer.logger is not None
        assert analyzer.material_properties_cache == {}
    
    def test_analyze_thermal_stress_steady_state(self):
        """Test steady-state thermal stress analysis."""
        results = self.analyzer.analyze_thermal_stress(
            self.geometry, self.materials, self.thermal_loads, 'steady_state'
        )
        
        assert isinstance(results, ThermalStressResults)
        assert results.thermal_stress.shape == (4, 6)  # 4 nodes, 6 stress components
        assert results.mechanical_stress.shape == (4, 6)
        assert results.total_stress.shape == (4, 6)
        assert results.thermal_strain.shape == (4, 6)
        assert results.mechanical_strain.shape == (4, 6)
        assert results.displacement.shape == (4, 3)  # 4 nodes, 3 DOF
        assert results.safety_factor.shape == (4,)
        assert results.max_temperature == 1200.0
        assert results.max_stress > 0
        assert len(results.critical_regions) >= 0
    
    def test_analyze_thermal_stress_transient(self):
        """Test transient thermal stress analysis."""
        # Add time history to thermal loads
        thermal_loads_transient = ThermalLoadConditions(
            temperature_distribution=self.thermal_loads.temperature_distribution,
            temperature_gradient=self.thermal_loads.temperature_gradient,
            heat_flux=self.thermal_loads.heat_flux,
            time_history=np.linspace(0, 100, 11)  # 0 to 100 seconds
        )
        
        results = self.analyzer.analyze_thermal_stress(
            self.geometry, self.materials, thermal_loads_transient, 'transient'
        )
        
        assert isinstance(results, ThermalStressResults)
        assert results.thermal_stress.shape == (4, 6)
        assert results.max_temperature == 1200.0
    
    def test_calculate_thermal_expansion_effects(self):
        """Test thermal expansion effects calculation."""
        temperature_change = np.array([100.0, 500.0, 900.0, 700.0])
        
        results = self.analyzer.calculate_thermal_expansion_effects(
            self.geometry, self.materials, temperature_change
        )
        
        assert 'thermal_strain' in results
        assert 'thermal_displacement' in results
        assert 'constrained_stress' in results
        assert 'expansion_coefficients' in results
        
        assert results['thermal_strain'].shape == (4, 6)
        assert results['thermal_displacement'].shape == (4, 3)
        assert results['constrained_stress'].shape == (4, 6)
        
        # Check that thermal strain increases with temperature
        thermal_strain = results['thermal_strain']
        assert thermal_strain[1, 0] > thermal_strain[0, 0]  # Higher temp -> higher strain
        assert thermal_strain[2, 0] > thermal_strain[1, 0]
    
    def test_coupled_thermal_structural_analysis(self):
        """Test coupled thermal-structural analysis."""
        mechanical_loads = {
            'applied_forces': np.array([[0, 0, 1000], [0, 0, 0], [0, 0, 0], [0, 0, -1000]])
        }
        
        results = self.analyzer.perform_coupled_thermal_structural_analysis(
            self.geometry, self.materials, self.thermal_loads, mechanical_loads,
            coupling_iterations=3, convergence_tolerance=1e-4
        )
        
        assert isinstance(results, ThermalStressResults)
        assert results.thermal_stress.shape == (4, 6)
        assert results.mechanical_stress.shape == (4, 6)
        assert results.total_stress.shape == (4, 6)
    
    def test_temperature_dependent_properties(self):
        """Test temperature-dependent material properties calculation."""
        temperatures = np.array([300.0, 800.0, 1200.0, 1000.0])
        
        temp_props = self.analyzer._calculate_temperature_dependent_properties(
            self.materials, temperatures
        )
        
        assert 'steel' in temp_props
        steel_props = temp_props['steel']
        
        assert 'youngs_modulus' in steel_props
        assert 'thermal_expansion' in steel_props
        assert 'yield_strength' in steel_props
        
        # Check that properties degrade with temperature
        E_values = steel_props['youngs_modulus']
        assert E_values[0] > E_values[1]  # Lower temp -> higher modulus
        assert E_values[1] > E_values[2]
        
        yield_values = steel_props['yield_strength']
        assert yield_values[0] > yield_values[1]  # Lower temp -> higher strength
    
    def test_thermal_strain_calculation(self):
        """Test thermal strain calculation."""
        temperatures = np.array([300.0, 800.0, 1200.0, 1000.0])
        temp_props = self.analyzer._calculate_temperature_dependent_properties(
            self.materials, temperatures
        )
        
        thermal_strain = self.analyzer._calculate_thermal_strain(temperatures, temp_props)
        
        assert thermal_strain.shape == (4, 6)
        
        # Check that thermal strain increases with temperature
        assert thermal_strain[1, 0] > thermal_strain[0, 0]  # εxx increases with temp
        assert thermal_strain[2, 0] > thermal_strain[1, 0]
        
        # Check isotropic expansion (εxx = εyy = εzz)
        for i in range(4):
            assert abs(thermal_strain[i, 0] - thermal_strain[i, 1]) < 1e-10
            assert abs(thermal_strain[i, 1] - thermal_strain[i, 2]) < 1e-10
            
            # Shear strains should be zero for isotropic expansion
            assert abs(thermal_strain[i, 3]) < 1e-10  # γxy
            assert abs(thermal_strain[i, 4]) < 1e-10  # γxz
            assert abs(thermal_strain[i, 5]) < 1e-10  # γyz
    
    def test_safety_factor_calculation(self):
        """Test safety factor calculation."""
        # Create test stress field
        total_stress = np.array([
            [100e6, 50e6, 30e6, 10e6, 5e6, 2e6],    # Low stress
            [200e6, 100e6, 80e6, 20e6, 15e6, 10e6], # Medium stress
            [400e6, 200e6, 150e6, 50e6, 30e6, 20e6], # High stress
            [300e6, 150e6, 100e6, 30e6, 20e6, 15e6]  # Medium-high stress
        ])
        
        temperatures = np.array([300.0, 800.0, 1200.0, 1000.0])
        temp_props = self.analyzer._calculate_temperature_dependent_properties(
            self.materials, temperatures
        )
        
        safety_factors = self.analyzer._calculate_safety_factors(
            total_stress, temp_props, temperatures
        )
        
        assert safety_factors.shape == (4,)
        assert all(sf > 0 for sf in safety_factors)
        
        # Higher stress should give lower safety factor
        assert safety_factors[0] > safety_factors[2]  # Low stress > high stress
    
    def test_failure_location_identification(self):
        """Test failure location identification."""
        # Create stress field with some failures (SF < 1.0)
        total_stress = np.array([
            [100e6, 50e6, 30e6, 10e6, 5e6, 2e6],     # Safe
            [600e6, 300e6, 200e6, 50e6, 30e6, 20e6], # Failure
            [800e6, 400e6, 300e6, 100e6, 50e6, 30e6], # Failure
            [200e6, 100e6, 80e6, 20e6, 15e6, 10e6]   # Safe
        ])
        
        safety_factors = np.array([2.0, 0.8, 0.5, 1.5])  # Two failures
        
        temperatures = np.array([300.0, 800.0, 1200.0, 1000.0])
        temp_props = self.analyzer._calculate_temperature_dependent_properties(
            self.materials, temperatures
        )
        
        failure_locations = self.analyzer._identify_failure_locations(
            total_stress, safety_factors, temp_props
        )
        
        assert len(failure_locations) == 2  # Two failure locations
        assert (1, 'yielding') in failure_locations or (1, 'tensile') in failure_locations
        assert (2, 'yielding') in failure_locations or (2, 'tensile') in failure_locations
    
    def test_critical_regions_identification(self):
        """Test critical regions identification."""
        total_stress = np.array([
            [100e6, 50e6, 30e6, 10e6, 5e6, 2e6],
            [200e6, 100e6, 80e6, 20e6, 15e6, 10e6],
            [400e6, 200e6, 150e6, 50e6, 30e6, 20e6],  # High stress
            [300e6, 150e6, 100e6, 30e6, 20e6, 15e6]
        ])
        
        temperatures = np.array([300.0, 800.0, 1200.0, 1000.0])  # High temp at node 2
        safety_factors = np.array([3.0, 2.0, 1.2, 1.8])  # Low SF at node 2
        
        critical_regions = self.analyzer._identify_critical_regions(
            self.geometry, total_stress, temperatures, safety_factors
        )
        
        assert len(critical_regions) >= 1
        
        # Check for high stress region
        high_stress_region = next((r for r in critical_regions if r['type'] == 'high_stress'), None)
        assert high_stress_region is not None
        assert 2 in high_stress_region['nodes']  # Node 2 has highest stress
        
        # Check for high temperature region
        high_temp_region = next((r for r in critical_regions if r['type'] == 'high_temperature'), None)
        assert high_temp_region is not None
        assert 2 in high_temp_region['nodes']  # Node 2 has highest temperature
    
    def test_von_mises_stress_calculation(self):
        """Test von Mises stress calculation."""
        # Test case: pure tension
        stress_tension = np.array([100e6, 0, 0, 0, 0, 0])
        von_mises_tension = self.analyzer._calculate_von_mises_stress(stress_tension)
        assert abs(von_mises_tension - 100e6) < 1e-6
        
        # Test case: pure shear
        stress_shear = np.array([0, 0, 0, 100e6, 0, 0])
        von_mises_shear = self.analyzer._calculate_von_mises_stress(stress_shear)
        expected_shear = np.sqrt(3) * 100e6
        assert abs(von_mises_shear - expected_shear) < 1e-6
        
        # Test case: biaxial tension
        stress_biaxial = np.array([100e6, 50e6, 0, 0, 0, 0])
        von_mises_biaxial = self.analyzer._calculate_von_mises_stress(stress_biaxial)
        expected_biaxial = np.sqrt(100e6**2 - 100e6*50e6 + 50e6**2)
        assert abs(von_mises_biaxial - expected_biaxial) < 1e-6
    
    def test_constitutive_matrix(self):
        """Test constitutive matrix calculation."""
        E = 200e9  # Pa
        nu = 0.3
        
        D = self.analyzer._get_constitutive_matrix(E, nu)
        
        assert D.shape == (6, 6)
        
        # Check diagonal terms
        expected_normal = E * (1 - nu) / ((1 + nu) * (1 - 2*nu))
        assert abs(D[0, 0] - expected_normal) < 1e-3  # Relaxed tolerance
        assert abs(D[1, 1] - expected_normal) < 1e-3
        assert abs(D[2, 2] - expected_normal) < 1e-3
        
        # Check off-diagonal terms
        expected_off_diag = E * nu / ((1 + nu) * (1 - 2*nu))
        assert abs(D[0, 1] - expected_off_diag) < 1e-3
        assert abs(D[1, 2] - expected_off_diag) < 1e-3
        
        # Check shear terms
        expected_shear = E * (1 - 2*nu) / (2 * (1 + nu) * (1 - 2*nu))
        assert abs(D[3, 3] - expected_shear) < 1e-3
        assert abs(D[4, 4] - expected_shear) < 1e-3
        assert abs(D[5, 5] - expected_shear) < 1e-3
    
    def test_input_validation(self):
        """Test input validation."""
        # Test empty geometry
        empty_geometry = StructuralGeometry(
            nodes=np.array([]),
            elements=np.array([]),
            element_type='beam'
        )
        
        with pytest.raises(ValueError, match="Geometry must have nodes"):
            self.analyzer._validate_thermal_inputs(empty_geometry, self.materials, self.thermal_loads)
        
        # Test empty materials
        with pytest.raises(ValueError, match="Materials dictionary cannot be empty"):
            self.analyzer._validate_thermal_inputs(self.geometry, {}, self.thermal_loads)
        
        # Test mismatched temperature distribution
        bad_thermal_loads = ThermalLoadConditions(
            temperature_distribution=np.array([300.0, 800.0]),  # Only 2 values for 4 nodes
            temperature_gradient=np.array([500.0, 400.0]),
            heat_flux=np.array([1e6, 2e6])
        )
        
        with pytest.raises(ValueError, match="Temperature distribution must match number of nodes"):
            self.analyzer._validate_thermal_inputs(self.geometry, self.materials, bad_thermal_loads)
        
        # Test material without thermal properties
        bad_material = MaterialDefinition(
            name='Bad material',
            base_material_type=MaterialType.CONVENTIONAL_METAL,
            mechanical_properties=self.materials['steel'].mechanical_properties
            # Missing thermal_properties
        )
        
        with pytest.raises(ValueError, match="missing thermal properties"):
            self.analyzer._validate_thermal_inputs(self.geometry, {'bad': bad_material}, self.thermal_loads)
    
    def test_extreme_temperature_conditions(self):
        """Test analysis with extreme temperature conditions."""
        # Extreme temperature loads (up to 6000K as per requirements)
        extreme_thermal_loads = ThermalLoadConditions(
            temperature_distribution=np.array([300.0, 2000.0, 5000.0, 6000.0]),
            temperature_gradient=np.array([1000.0, 3000.0, 1000.0, -2000.0]),
            heat_flux=np.array([10e6, 100e6, 150e6, 80e6])  # Up to 150 MW/m²
        )
        
        results = self.analyzer.analyze_thermal_stress(
            self.geometry, self.materials, extreme_thermal_loads, 'steady_state'
        )
        
        assert isinstance(results, ThermalStressResults)
        assert results.max_temperature == 6000.0
        assert results.max_stress > 0
        
        # Should have failure locations at extreme temperatures
        assert len(results.failure_locations) > 0
        
        # Should have critical regions identified
        assert len(results.critical_regions) > 0
    
    def test_large_temperature_gradients(self):
        """Test thermal expansion with large temperature differences."""
        # Large temperature changes (up to 5700K above reference)
        large_temp_change = np.array([0.0, 1700.0, 4700.0, 5700.0])
        
        results = self.analyzer.calculate_thermal_expansion_effects(
            self.geometry, self.materials, large_temp_change
        )
        
        assert 'thermal_strain' in results
        thermal_strain = results['thermal_strain']
        
        # Check that strain increases significantly with large temperature changes
        assert thermal_strain[3, 0] > thermal_strain[2, 0]  # Highest temp -> highest strain
        assert thermal_strain[2, 0] > thermal_strain[1, 0]
        assert thermal_strain[1, 0] > thermal_strain[0, 0]
        
        # Check that displacement is significant
        displacement = results['thermal_displacement']
        assert np.max(np.abs(displacement)) > 0.01  # At least 1 cm displacement
    
    @patch('fighter_jet_sdk.engines.structural.thermal_stress_analyzer.get_engine_logger')
    def test_logging(self, mock_logger):
        """Test logging functionality."""
        mock_log = Mock()
        mock_logger.return_value = mock_log
        
        analyzer = ThermalStressAnalyzer()
        
        # Test successful analysis logging
        results = analyzer.analyze_thermal_stress(
            self.geometry, self.materials, self.thermal_loads, 'steady_state'
        )
        
        # Check that info logs were called
        mock_log.info.assert_called()
        
        # Check that the completion log includes max stress
        completion_calls = [call for call in mock_log.info.call_args_list 
                          if 'complete' in str(call)]
        assert len(completion_calls) > 0