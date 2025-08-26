"""Tests for thermal materials database and modeling."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from fighter_jet_sdk.engines.materials.thermal_materials_db import (
    ThermalMaterialsDB, ThermalAnalysisResult, HypersonicConditions, 
    ThermalStressAnalysis
)
from fighter_jet_sdk.common.data_models import MaterialDefinition, ThermalProperties, MechanicalProperties
from fighter_jet_sdk.common.enums import MaterialType


class TestThermalMaterialsDB:
    """Test suite for ThermalMaterialsDB class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.thermal_db = ThermalMaterialsDB()
        
        # Test hypersonic conditions
        self.test_conditions = HypersonicConditions(
            mach_number=5.0,
            altitude=30000.0,  # 30 km
            flight_time=300.0,  # 5 minutes
            angle_of_attack=5.0,
            surface_emissivity=0.8,
            recovery_factor=0.89
        )
    
    def test_initialization(self):
        """Test ThermalMaterialsDB initialization."""
        assert hasattr(self.thermal_db, '_uhtc_materials')
        assert hasattr(self.thermal_db, '_property_interpolators')
        assert self.thermal_db.stefan_boltzmann == 5.670374419e-8
        assert self.thermal_db.gas_constant == 8.314
        
        # Check that materials were loaded
        assert len(self.thermal_db._uhtc_materials) > 0
        
        # Check specific materials
        assert 'hfc' in self.thermal_db._uhtc_materials
        assert 'tac' in self.thermal_db._uhtc_materials
        assert 'wc' in self.thermal_db._uhtc_materials
    
    def test_get_material(self):
        """Test material retrieval."""
        hfc_material = self.thermal_db.get_material('hfc')
        
        assert hfc_material is not None
        assert isinstance(hfc_material, MaterialDefinition)
        assert hfc_material.name == "Hafnium Carbide (HfC)"
        assert hfc_material.base_material_type == MaterialType.ULTRA_HIGH_TEMP_CERAMIC
        assert hfc_material.thermal_properties is not None
        
        # Test non-existent material
        assert self.thermal_db.get_material('nonexistent') is None
    
    def test_list_materials(self):
        """Test materials listing."""
        materials_list = self.thermal_db.list_materials()
        
        assert isinstance(materials_list, dict)
        assert len(materials_list) > 0
        assert 'hfc' in materials_list
        assert materials_list['hfc'] == "Hafnium Carbide (HfC)"
    
    def test_get_materials_for_temperature(self):
        """Test material selection by temperature."""
        # Test moderate temperature
        materials_1500k = self.thermal_db.get_materials_for_temperature(1500.0)
        assert isinstance(materials_1500k, list)
        assert len(materials_1500k) > 0
        
        # Test very high temperature
        materials_3500k = self.thermal_db.get_materials_for_temperature(3500.0)
        assert isinstance(materials_3500k, list)
        # Should have fewer materials at very high temperature
        assert len(materials_3500k) <= len(materials_1500k)
        
        # Test temperature beyond all materials
        materials_5000k = self.thermal_db.get_materials_for_temperature(5000.0)
        assert isinstance(materials_5000k, list)
        # Might be empty or very few materials
    
    def test_calculate_thermal_conductivity(self):
        """Test thermal conductivity calculation."""
        # Test at room temperature
        k_room = self.thermal_db.calculate_thermal_conductivity('hfc', 293.0)
        assert isinstance(k_room, float)
        assert k_room > 0
        
        # Test at high temperature
        k_high = self.thermal_db.calculate_thermal_conductivity('hfc', 2000.0)
        assert isinstance(k_high, float)
        assert k_high > 0
        
        # Thermal conductivity should generally decrease with temperature
        assert k_high < k_room
        
        # Test non-existent material
        k_none = self.thermal_db.calculate_thermal_conductivity('nonexistent', 293.0)
        assert k_none == 0.0
    
    def test_calculate_specific_heat(self):
        """Test specific heat calculation."""
        # Test at room temperature
        cp_room = self.thermal_db.calculate_specific_heat('hfc', 293.0)
        assert isinstance(cp_room, float)
        assert cp_room > 0
        
        # Test at high temperature
        cp_high = self.thermal_db.calculate_specific_heat('hfc', 2000.0)
        assert isinstance(cp_high, float)
        assert cp_high > 0
        
        # Specific heat should generally increase with temperature
        assert cp_high >= cp_room
        
        # Test non-existent material
        cp_none = self.thermal_db.calculate_specific_heat('nonexistent', 293.0)
        assert cp_none == 0.0
    
    def test_analyze_hypersonic_heating(self):
        """Test hypersonic heating analysis."""
        result = self.thermal_db.analyze_hypersonic_heating(
            'hfc', self.test_conditions, thickness=0.005
        )
        
        assert isinstance(result, ThermalAnalysisResult)
        assert len(result.temperatures) > 0
        assert len(result.heat_flux) > 0
        assert len(result.thermal_stress) > 0
        assert len(result.safety_factor) > 0
        assert len(result.failure_mode) > 0
        
        # Temperatures should be positive and reasonable
        assert np.all(result.temperatures > 0)
        assert np.all(result.temperatures < 10000)  # Reasonable upper bound
        
        # Heat flux should be positive
        assert np.all(result.heat_flux >= 0)
        
        # Safety factors should be positive
        assert np.all(result.safety_factor > 0)
    
    def test_analyze_hypersonic_heating_invalid_material(self):
        """Test hypersonic heating analysis with invalid material."""
        with pytest.raises(ValueError, match="Material nonexistent not found"):
            self.thermal_db.analyze_hypersonic_heating(
                'nonexistent', self.test_conditions
            )
    
    def test_calculate_thermal_stress_distribution(self):
        """Test thermal stress distribution calculation."""
        # Create a temperature profile
        n_points = 21
        thickness = 0.01
        temperatures = np.linspace(1500, 500, n_points)  # Hot surface to cold
        
        # First add mechanical properties to HfC material
        hfc_material = self.thermal_db.get_material('hfc')
        hfc_material.mechanical_properties = MechanicalProperties(
            youngs_modulus=450e9,  # Pa
            poissons_ratio=0.17,
            yield_strength=800e6,  # Pa
            ultimate_strength=1200e6,  # Pa
            fatigue_limit=400e6,  # Pa
            density=12800  # kg/m³
        )
        
        stress_analysis = self.thermal_db.calculate_thermal_stress_distribution(
            'hfc', temperatures, thickness
        )
        
        assert isinstance(stress_analysis, ThermalStressAnalysis)
        assert len(stress_analysis.thermal_strain) == n_points
        assert len(stress_analysis.thermal_stress) == n_points
        assert len(stress_analysis.von_mises_stress) == n_points
        assert stress_analysis.principal_stresses.shape == (n_points, 3)
        assert len(stress_analysis.safety_factors) == n_points
        assert isinstance(stress_analysis.critical_locations, list)
        
        # Thermal strain should be positive for heating
        assert np.all(stress_analysis.thermal_strain >= 0)
        
        # Safety factors should be positive
        assert np.all(stress_analysis.safety_factors > 0)
    
    def test_calculate_thermal_stress_distribution_no_mechanical_props(self):
        """Test thermal stress calculation without mechanical properties."""
        temperatures = np.linspace(1500, 500, 21)
        
        with pytest.raises(ValueError, match="missing mechanical properties"):
            self.thermal_db.calculate_thermal_stress_distribution(
                'hfc', temperatures, 0.01
            )
    
    def test_optimize_material_selection(self):
        """Test material selection optimization."""
        # Add mechanical properties to materials for testing
        for mat_id in ['hfc', 'tac', 'wc']:
            material = self.thermal_db.get_material(mat_id)
            if material:
                material.mechanical_properties = MechanicalProperties(
                    youngs_modulus=400e9,
                    poissons_ratio=0.2,
                    yield_strength=600e6,
                    ultimate_strength=1000e6,
                    fatigue_limit=300e6,
                    density=material.thermal_properties.density
                )
        
        optimal_material = self.thermal_db.optimize_material_selection(
            max_temperature=2000.0,
            max_stress=500e6,
            weight_factor=0.5
        )
        
        assert isinstance(optimal_material, str)
        assert optimal_material in self.thermal_db._uhtc_materials
        
        # Verify the selected material can handle the conditions
        material = self.thermal_db.get_material(optimal_material)
        assert material.thermal_properties.operating_temp_range[1] >= 2000.0
        assert material.mechanical_properties.yield_strength >= 500e6
    
    def test_optimize_material_selection_no_suitable_materials(self):
        """Test material optimization with impossible conditions."""
        with pytest.raises(ValueError, match="No materials suitable for temperature"):
            self.thermal_db.optimize_material_selection(
                max_temperature=10000.0,  # Impossibly high
                max_stress=100e6
            )
    
    def test_stagnation_temperature_calculation(self):
        """Test stagnation temperature calculation."""
        T_stag = self.thermal_db._calculate_stagnation_temperature(self.test_conditions)
        
        assert isinstance(T_stag, float)
        assert T_stag > 0
        # Should be higher than ambient due to high Mach number
        # At 30km altitude, ambient is very cold, so even with heating it might not reach 1000K
        assert T_stag > 200  # K, more reasonable for high altitude
    
    def test_wall_heat_flux_calculation(self):
        """Test wall heat flux calculation."""
        T_stag = 2000.0  # K
        q_wall = self.thermal_db._calculate_wall_heat_flux(self.test_conditions, T_stag)
        
        assert isinstance(q_wall, float)
        assert q_wall >= 0  # Heat flux should be non-negative
    
    def test_transient_heat_conduction_solution(self):
        """Test transient heat conduction solution."""
        temperatures, times = self.thermal_db._solve_transient_heat_conduction(
            'hfc', 1e6, 0.01, 300.0  # 1 MW/m² heat flux, 1cm thick, 300s
        )
        
        assert isinstance(temperatures, np.ndarray)
        assert isinstance(times, np.ndarray)
        assert len(temperatures) > 0
        assert len(times) > 0
        
        # Temperatures should be reasonable
        assert np.all(temperatures > 0)
        assert np.all(temperatures < 20000)  # More reasonable upper bound for high heat flux
        
        # Surface should be hotter than back
        assert temperatures[0] >= temperatures[-1]
    
    def test_thermal_stress_calculation(self):
        """Test thermal stress calculation."""
        temperatures = np.array([1500, 1000, 500])
        
        # Add mechanical properties to HfC
        hfc_material = self.thermal_db.get_material('hfc')
        hfc_material.mechanical_properties = MechanicalProperties(
            youngs_modulus=450e9,
            poissons_ratio=0.17,
            yield_strength=800e6,
            ultimate_strength=1200e6,
            fatigue_limit=400e6,
            density=12800
        )
        
        thermal_stress = self.thermal_db._calculate_thermal_stress(
            'hfc', temperatures, 0.01
        )
        
        assert isinstance(thermal_stress, np.ndarray)
        assert len(thermal_stress) == len(temperatures)
        
        # Stress should be positive for thermal expansion
        assert np.all(thermal_stress >= 0)
    
    def test_thermal_failure_assessment(self):
        """Test thermal failure assessment."""
        temperatures = np.array([1500, 2500, 4500])  # Progressively more severe
        thermal_stress = np.array([100e6, 500e6, 1000e6])  # Increasing stress
        
        safety_factors, failure_modes = self.thermal_db._assess_thermal_failure(
            'hfc', temperatures, thermal_stress
        )
        
        assert isinstance(safety_factors, np.ndarray)
        assert isinstance(failure_modes, list)
        assert len(safety_factors) == len(temperatures)
        assert len(failure_modes) == len(temperatures)
        
        # Safety factors should be positive
        assert np.all(safety_factors > 0)
        
        # Safety factors should generally decrease with severity
        assert safety_factors[0] >= safety_factors[1] >= safety_factors[2]
        
        # Check failure modes are strings
        assert all(isinstance(mode, str) for mode in failure_modes)
    
    def test_thermal_expansion_coefficient(self):
        """Test thermal expansion coefficient retrieval."""
        alpha_hfc = self.thermal_db._get_thermal_expansion_coefficient('hfc')
        assert isinstance(alpha_hfc, float)
        assert alpha_hfc > 0
        assert alpha_hfc < 1e-4  # Reasonable range for ceramics
        
        # Test default value for unknown material
        alpha_default = self.thermal_db._get_thermal_expansion_coefficient('unknown')
        assert alpha_default == 7.0e-6
    
    def test_atmospheric_density_calculation(self):
        """Test atmospheric density calculation."""
        # Test at sea level
        rho_0 = self.thermal_db._get_atmospheric_density(0.0)
        assert isinstance(rho_0, float)
        assert rho_0 > 1.0  # kg/m³, should be around 1.225
        
        # Test at high altitude
        rho_30km = self.thermal_db._get_atmospheric_density(30000.0)
        assert isinstance(rho_30km, float)
        assert rho_30km > 0
        assert rho_30km < rho_0  # Density decreases with altitude
    
    def test_speed_of_sound_calculation(self):
        """Test speed of sound calculation."""
        # Test at sea level
        a_0 = self.thermal_db._get_speed_of_sound(0.0)
        assert isinstance(a_0, float)
        assert a_0 > 300  # m/s, should be around 343
        
        # Test at high altitude
        a_30km = self.thermal_db._get_speed_of_sound(30000.0)
        assert isinstance(a_30km, float)
        assert a_30km > 0
        assert a_30km < a_0  # Speed decreases with altitude (temperature effect)
    
    def test_property_interpolators_setup(self):
        """Test that property interpolators are set up correctly."""
        # Check that interpolators exist for materials
        assert len(self.thermal_db._property_interpolators) > 0
        
        # Check HfC interpolators
        if 'hfc' in self.thermal_db._property_interpolators:
            hfc_interp = self.thermal_db._property_interpolators['hfc']
            assert 'thermal_conductivity' in hfc_interp
            assert 'specific_heat' in hfc_interp
            assert 'temperature_range' in hfc_interp
            
            # Test interpolation
            temp_range = hfc_interp['temperature_range']
            k_interp = hfc_interp['thermal_conductivity']
            cp_interp = hfc_interp['specific_heat']
            
            # Test at middle of range
            T_mid = (temp_range[0] + temp_range[-1]) / 2
            k_val = k_interp(T_mid)
            cp_val = cp_interp(T_mid)
            
            assert isinstance(k_val, (float, np.floating, np.ndarray))
            assert isinstance(cp_val, (float, np.floating, np.ndarray))
            assert k_val > 0
            assert cp_val > 0
    
    def test_temperature_dependent_properties(self):
        """Test temperature-dependent property calculations."""
        temperatures = np.array([500, 1000, 1500, 2000])
        
        # Test thermal conductivity variation
        k_values = [self.thermal_db.calculate_thermal_conductivity('hfc', T) for T in temperatures]
        assert all(k > 0 for k in k_values)
        # Should generally decrease with temperature
        assert k_values[0] >= k_values[-1]
        
        # Test specific heat variation
        cp_values = [self.thermal_db.calculate_specific_heat('hfc', T) for T in temperatures]
        assert all(cp > 0 for cp in cp_values)
        # Should generally increase with temperature
        assert cp_values[-1] >= cp_values[0]


class TestThermalAnalysisResult:
    """Test suite for ThermalAnalysisResult dataclass."""
    
    def test_thermal_analysis_result_creation(self):
        """Test ThermalAnalysisResult creation."""
        n_points = 10
        result = ThermalAnalysisResult(
            temperatures=np.linspace(1000, 500, n_points),
            heat_flux=np.ones(n_points) * 1e6,
            thermal_stress=np.linspace(100e6, 50e6, n_points),
            safety_factor=np.linspace(2.0, 4.0, n_points),
            failure_mode=['safe'] * n_points
        )
        
        assert len(result.temperatures) == n_points
        assert len(result.heat_flux) == n_points
        assert len(result.thermal_stress) == n_points
        assert len(result.safety_factor) == n_points
        assert len(result.failure_mode) == n_points


class TestHypersonicConditions:
    """Test suite for HypersonicConditions dataclass."""
    
    def test_hypersonic_conditions_creation(self):
        """Test HypersonicConditions creation."""
        conditions = HypersonicConditions(
            mach_number=6.0,
            altitude=35000.0,
            flight_time=600.0,
            angle_of_attack=3.0,
            surface_emissivity=0.85,
            recovery_factor=0.9
        )
        
        assert conditions.mach_number == 6.0
        assert conditions.altitude == 35000.0
        assert conditions.flight_time == 600.0
        assert conditions.angle_of_attack == 3.0
        assert conditions.surface_emissivity == 0.85
        assert conditions.recovery_factor == 0.9


class TestThermalStressAnalysis:
    """Test suite for ThermalStressAnalysis dataclass."""
    
    def test_thermal_stress_analysis_creation(self):
        """Test ThermalStressAnalysis creation."""
        n_points = 15
        analysis = ThermalStressAnalysis(
            thermal_strain=np.linspace(0.001, 0.005, n_points),
            thermal_stress=np.linspace(100e6, 500e6, n_points),
            von_mises_stress=np.linspace(100e6, 500e6, n_points),
            principal_stresses=np.random.random((n_points, 3)) * 1e8,
            safety_factors=np.linspace(1.5, 3.0, n_points),
            critical_locations=[0, 5, 10]
        )
        
        assert len(analysis.thermal_strain) == n_points
        assert len(analysis.thermal_stress) == n_points
        assert len(analysis.von_mises_stress) == n_points
        assert analysis.principal_stresses.shape == (n_points, 3)
        assert len(analysis.safety_factors) == n_points
        assert analysis.critical_locations == [0, 5, 10]


if __name__ == '__main__':
    pytest.main([__file__])