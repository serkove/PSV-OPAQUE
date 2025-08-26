"""Unit tests for extreme thermal materials database functionality."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from fighter_jet_sdk.engines.materials.thermal_materials_db import (
    ThermalMaterialsDB, HypersonicConditions, ThermalAnalysisResult
)
from fighter_jet_sdk.common.data_models import MaterialDefinition, ThermalProperties
from fighter_jet_sdk.common.enums import MaterialType


class TestExtremeThermalMaterials:
    """Test extreme thermal materials database functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.thermal_db = ThermalMaterialsDB()
    
    def test_extreme_temperature_materials_initialization(self):
        """Test that extreme temperature materials are properly initialized."""
        # Check that materials with operating temperatures > 5000K exist
        extreme_materials = self.thermal_db.get_extreme_temperature_materials(5000.0)
        
        assert len(extreme_materials) > 0, "Should have materials for extreme temperatures"
        
        # Verify specific extreme materials
        expected_materials = ['hftac', 'w', 'hfc', 'tac', 'cc']
        for mat_id in expected_materials:
            material = self.thermal_db.get_material(mat_id)
            assert material is not None, f"Material {mat_id} should exist"
            assert material.thermal_properties.operating_temp_range[1] >= 5000.0, \
                f"Material {mat_id} should operate above 5000K"
    
    def test_materials_for_6000k_operation(self):
        """Test materials suitable for 6000K operation."""
        materials_6000k = self.thermal_db.get_materials_for_temperature(6000.0)
        
        # Should have at least Hf-Ta carbide
        assert 'hftac' in materials_6000k, "Hf-Ta carbide should operate at 6000K"
        
        # Verify temperature range
        hftac_material = self.thermal_db.get_material('hftac')
        assert hftac_material.thermal_properties.operating_temp_range[1] >= 6000.0
    
    def test_thermal_conductivity_interpolation_extreme_temps(self):
        """Test thermal conductivity interpolation up to 6000K."""
        temperatures = np.array([293, 1000, 3000, 5000, 6000])
        
        # Test different material types
        test_materials = ['hfc', 'w', 'hftac', 'cc']
        
        for mat_id in test_materials:
            k_values = []
            for temp in temperatures:
                k = self.thermal_db.calculate_thermal_conductivity(mat_id, temp)
                k_values.append(k)
                
                # Verify positive values
                assert k > 0, f"Thermal conductivity must be positive for {mat_id} at {temp}K"
            
            k_values = np.array(k_values)
            
            # Generally, thermal conductivity should decrease with temperature for ceramics
            if mat_id in ['hfc', 'hftac']:
                assert k_values[0] > k_values[-1], \
                    f"Thermal conductivity should decrease with temperature for {mat_id}"
    
    def test_specific_heat_interpolation_extreme_temps(self):
        """Test specific heat interpolation up to 6000K."""
        temperatures = np.array([293, 1000, 3000, 5000, 6000])
        
        test_materials = ['hfc', 'w', 'hftac', 're']
        
        for mat_id in test_materials:
            cp_values = []
            for temp in temperatures:
                cp = self.thermal_db.calculate_specific_heat(mat_id, temp)
                cp_values.append(cp)
                
                # Verify positive values
                assert cp > 0, f"Specific heat must be positive for {mat_id} at {temp}K"
            
            cp_values = np.array(cp_values)
            
            # Specific heat generally increases with temperature
            assert cp_values[-1] > cp_values[0], \
                f"Specific heat should increase with temperature for {mat_id}"
    
    def test_plasma_environment_material_selection(self):
        """Test material selection for plasma environments."""
        # Test conditions for Mach 60 plasma environment
        temperature = 5500.0  # K
        plasma_density = 1e19  # m^-3
        
        plasma_materials = self.thermal_db.get_materials_for_plasma_environment(
            temperature, plasma_density
        )
        
        assert len(plasma_materials) > 0, "Should have materials for plasma environment"
        
        # Verify materials have high thermal conductivity
        for mat_id in plasma_materials:
            material = self.thermal_db.get_material(mat_id)
            assert material.thermal_properties.thermal_conductivity > 15.0, \
                f"Plasma materials should have high thermal conductivity: {mat_id}"
    
    def test_extreme_thermal_material_optimization(self):
        """Test advanced material selection for extreme thermal conditions."""
        # Mach 60 conditions
        max_temperature = 5500.0  # K
        heat_flux = 150e6  # W/m² (150 MW/m²)
        thermal_gradient = 1e6  # K/m
        mission_duration = 600.0  # 10 minutes
        
        result = self.thermal_db.optimize_extreme_thermal_material_selection(
            max_temperature=max_temperature,
            heat_flux=heat_flux,
            thermal_gradient=thermal_gradient,
            mission_duration=mission_duration,
            weight_factor=0.3,
            cost_factor=0.1
        )
        
        # Verify result structure
        assert 'optimal_material' in result
        assert 'material_rankings' in result
        assert 'selection_criteria' in result
        
        # Verify optimal material can handle the conditions
        optimal_material = self.thermal_db.get_material(result['optimal_material'])
        assert optimal_material.thermal_properties.operating_temp_range[1] >= max_temperature
        
        # Verify rankings are sorted
        rankings = result['material_rankings']
        assert len(rankings) > 0
        for i in range(len(rankings) - 1):
            assert rankings[i]['total_score'] <= rankings[i + 1]['total_score']
    
    def test_temperature_dependent_properties_physical_bounds(self):
        """Test that temperature-dependent properties stay within physical bounds."""
        # Test extreme temperature range
        temperatures = np.linspace(293, 6000, 100)
        
        for mat_id in ['hfc', 'w', 'hftac']:
            material = self.thermal_db.get_material(mat_id)
            k_ref = material.thermal_properties.thermal_conductivity
            cp_ref = material.thermal_properties.specific_heat
            
            for temp in temperatures:
                k = self.thermal_db.calculate_thermal_conductivity(mat_id, temp)
                cp = self.thermal_db.calculate_specific_heat(mat_id, temp)
                
                # Thermal conductivity bounds
                assert k >= k_ref * 0.1, f"Thermal conductivity too low for {mat_id} at {temp}K"
                assert k <= k_ref * 3.0, f"Thermal conductivity too high for {mat_id} at {temp}K"
                
                # Specific heat bounds
                assert cp >= cp_ref * 0.5, f"Specific heat too low for {mat_id} at {temp}K"
                assert cp <= cp_ref * 5.0, f"Specific heat too high for {mat_id} at {temp}K"
    
    def test_material_property_interpolation_continuity(self):
        """Test that material property interpolation is continuous."""
        # Test temperature points with smaller increments for better continuity
        test_temps = np.array([2990, 3000, 3010, 4990, 5000, 5010])
        
        for mat_id in ['hfc', 'w', 'hftac']:
            k_values = [self.thermal_db.calculate_thermal_conductivity(mat_id, t) for t in test_temps]
            cp_values = [self.thermal_db.calculate_specific_heat(mat_id, t) for t in test_temps]
            
            # Check for continuity (allow for reasonable changes due to temperature dependence)
            for i in range(len(k_values) - 1):
                k_change = abs(k_values[i + 1] - k_values[i]) / k_values[i]
                cp_change = abs(cp_values[i + 1] - cp_values[i]) / cp_values[i]
                
                # Allow for larger changes due to complex temperature dependence at extreme temps
                assert k_change < 0.3, f"Large discontinuity in thermal conductivity for {mat_id}: {k_change:.3f}"
                assert cp_change < 0.3, f"Large discontinuity in specific heat for {mat_id}: {cp_change:.3f}"
    
    def test_extreme_material_database_completeness(self):
        """Test that extreme materials database has required materials."""
        required_materials = {
            'hftac': 'Hafnium-Tantalum Carbide',  # Highest melting point
            'w': 'Pure Tungsten',  # High thermal conductivity
            'hfc': 'Hafnium Carbide',  # Good all-around UHTC
            'cc': 'Carbon-Carbon Composite',  # Thermal shock resistance
            're': 'Rhenium'  # Metallic high-temp
        }
        
        for mat_id, expected_name in required_materials.items():
            material = self.thermal_db.get_material(mat_id)
            assert material is not None, f"Required material {mat_id} missing"
            assert expected_name.lower() in material.name.lower(), \
                f"Material name mismatch for {mat_id}"
            
            # Verify extreme temperature capability
            assert material.thermal_properties.operating_temp_range[1] >= 5000.0, \
                f"Material {mat_id} should operate above 5000K"
    
    def test_thermal_performance_calculation(self):
        """Test thermal performance calculation for extreme conditions."""
        mat_id = 'hftac'
        max_temperature = 5500.0
        heat_flux = 100e6  # W/m²
        thermal_gradient = 5e5  # K/m
        mission_duration = 300.0  # s
        
        performance = self.thermal_db._calculate_extreme_thermal_performance(
            mat_id, max_temperature, heat_flux, thermal_gradient, mission_duration
        )
        
        # Verify performance metrics structure
        required_keys = [
            'thermal_performance_score',
            'thermal_shock_resistance',
            'thermal_failure_risk',
            'thermal_diffusivity',
            'time_factor',
            'temperature_margin'
        ]
        
        for key in required_keys:
            assert key in performance, f"Missing performance metric: {key}"
            assert isinstance(performance[key], (int, float)), \
                f"Performance metric {key} should be numeric"
        
        # Verify reasonable values
        assert performance['thermal_diffusivity'] > 0, "Thermal diffusivity must be positive"
        assert 0 <= performance['thermal_failure_risk'] <= 1, "Failure risk should be 0-1"
        assert performance['temperature_margin'] > 0, "Should have positive temperature margin"
    
    def test_material_selection_error_handling(self):
        """Test error handling in material selection."""
        # Test with impossible temperature (higher than any material can handle)
        materials_10k = self.thermal_db.get_materials_for_temperature(10000.0)
        assert len(materials_10k) == 0, "Should have no materials for 10000K"
        
        # Test optimization with impossible conditions
        with pytest.raises(ValueError, match="No materials suitable"):
            self.thermal_db.optimize_extreme_thermal_material_selection(
                max_temperature=10000.0,
                heat_flux=1e9,
                thermal_gradient=1e8,
                mission_duration=3600.0
            )
    
    def test_material_rankings_consistency(self):
        """Test that material rankings are consistent and meaningful."""
        # Test with moderate extreme conditions
        result = self.thermal_db.optimize_extreme_thermal_material_selection(
            max_temperature=5200.0,
            heat_flux=80e6,
            thermal_gradient=3e5,
            mission_duration=300.0,
            weight_factor=0.5
        )
        
        rankings = result['material_rankings']
        
        # Should have multiple candidates
        assert len(rankings) >= 2, "Should have multiple material candidates"
        
        # Verify each ranking has required fields
        for ranking in rankings:
            assert 'material_id' in ranking
            assert 'total_score' in ranking
            assert 'performance_metrics' in ranking
            
            # Verify material exists
            material = self.thermal_db.get_material(ranking['material_id'])
            assert material is not None
            
            # Verify can handle the temperature
            assert material.thermal_properties.operating_temp_range[1] >= 5200.0


if __name__ == '__main__':
    pytest.main([__file__])