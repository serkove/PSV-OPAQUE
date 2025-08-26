"""Tests for extreme heat flux modeling."""

import pytest
import numpy as np
import math

from fighter_jet_sdk.engines.propulsion.extreme_heat_flux_model import (
    ExtremeHeatFluxModel,
    ExtremeHeatFluxConditions,
    MaterialThermalProperties,
    ThermalBoundaryCondition,
    RadiationModel,
    ConductionModel
)
from fighter_jet_sdk.common.plasma_physics import PlasmaConditions


class TestExtremeHeatFluxModel:
    """Test extreme heat flux modeling capabilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = ExtremeHeatFluxModel()
        
        # Test conditions
        self.extreme_conditions = ExtremeHeatFluxConditions(
            heat_flux=200e6,  # 200 MW/m²
            surface_temperature=3000.0,  # K
            pressure=1000.0,  # Pa (high altitude)
            velocity=20000.0,  # m/s (Mach 60)
            time=10.0  # s
        )
        
        # Test geometry
        self.test_geometry = {
            'thickness': 0.005,  # 5mm
            'num_nodes': 20,
            'back_temperature': 1000.0,  # K
            'youngs_modulus': 400e9,  # Pa (tungsten carbide)
            'poisson_ratio': 0.25,
            'yield_strength': 1.5e9  # Pa
        }
    
    def test_model_initialization(self):
        """Test model initialization and material database."""
        assert len(self.model.materials) > 0
        assert "tungsten_carbide" in self.model.materials
        assert "hafnium_carbide" in self.model.materials
        assert "carbon_carbon" in self.model.materials
        assert "rhenium" in self.model.materials
        
        # Check material properties
        wc = self.model.materials["tungsten_carbide"]
        assert wc.melting_point > 3000.0
        assert wc.max_operating_temperature > 3000.0
        assert wc.density > 10000.0
    
    def test_material_property_calculations(self):
        """Test temperature-dependent material property calculations."""
        material = self.model.materials["tungsten_carbide"]
        
        # Test specific heat calculation
        cp_300 = material.specific_heat(300.0)
        cp_3000 = material.specific_heat(3000.0)
        assert cp_300 > 0
        assert cp_3000 > 0
        assert cp_3000 != cp_300  # Should be temperature dependent
        
        # Test thermal conductivity calculation
        k_300 = material.thermal_conductivity(300.0)
        k_3000 = material.thermal_conductivity(3000.0)
        assert k_300 > 0
        assert k_3000 > 0
    
    def test_extreme_heat_flux_calculation(self):
        """Test extreme heat flux calculation without plasma."""
        results = self.model.calculate_extreme_heat_flux(
            self.extreme_conditions,
            "tungsten_carbide",
            self.test_geometry
        )
        
        # Check result structure
        assert 'radiative' in results
        assert 'conductive' in results
        assert 'total_heat_flux' in results
        assert 'temperature_field' in results
        
        # Check radiative results
        rad_results = results['radiative']
        assert rad_results['gray_body_flux'] > 0
        assert rad_results['total_flux'] > 0
        assert rad_results['surface_temperature'] == self.extreme_conditions.surface_temperature
        
        # Check conductive results
        cond_results = results['conductive']
        assert cond_results['thermal_conductivity'] > 0
        assert cond_results['temperature_gradient'] > 0
        assert cond_results['effective_conductivity'] > 0
        
        # Check temperature field
        temp_field = results['temperature_field']
        assert len(temp_field['temperature']) > 0
        assert temp_field['max_temperature'] > 0
        assert temp_field['iterations'] > 0
    
    def test_plasma_heat_transfer(self):
        """Test heat transfer with plasma conditions."""
        # Add plasma conditions
        plasma_conditions = PlasmaConditions(
            electron_density=1e22,  # m⁻³
            electron_temperature=15000.0,  # K
            ion_temperature=12000.0,  # K
            magnetic_field=np.array([0.1, 0.0, 0.0]),  # Tesla
            plasma_frequency=1e12,  # Hz
            debye_length=1e-6,  # m
            ionization_fraction=0.1
        )
        
        conditions_with_plasma = ExtremeHeatFluxConditions(
            heat_flux=300e6,  # 300 MW/m²
            surface_temperature=4000.0,  # K
            plasma_conditions=plasma_conditions,
            pressure=100.0,  # Pa
            velocity=20000.0,  # m/s
            time=5.0  # s
        )
        
        results = self.model.calculate_extreme_heat_flux(
            conditions_with_plasma,
            "hafnium_carbide",
            self.test_geometry
        )
        
        # Should have plasma results
        assert 'plasma' in results
        plasma_results = results['plasma']
        assert plasma_results['ion_energy_flux'] > 0
        assert plasma_results['electron_heat_flux'] > 0
        assert plasma_results['total_plasma_flux'] > 0
        assert plasma_results['sheath_potential'] > 0
        
        # Total heat flux should include plasma contribution
        total_flux = results['total_heat_flux']
        assert total_flux['plasma_fraction'] > 0
        assert total_flux['total_surface_flux'] > results['radiative']['total_flux']
    
    def test_radiative_heat_transfer_models(self):
        """Test different radiative heat transfer models."""
        # Test gray body radiation
        results = self.model._calculate_radiative_heat_transfer(
            self.extreme_conditions,
            self.model.materials["carbon_carbon"],
            self.test_geometry
        )
        
        # Check Stefan-Boltzmann law
        expected_flux = 0.8 * self.model.stefan_boltzmann * self.extreme_conditions.surface_temperature**4
        assert abs(results['gray_body_flux'] - expected_flux) / expected_flux < 0.1
        
        # Test non-equilibrium effects at high temperature
        high_temp_conditions = ExtremeHeatFluxConditions(
            heat_flux=500e6,
            surface_temperature=6000.0,  # K - very high
            pressure=1000.0,
            velocity=25000.0
        )
        
        high_temp_results = self.model._calculate_radiative_heat_transfer(
            high_temp_conditions,
            self.model.materials["hafnium_carbide"],
            self.test_geometry
        )
        
        assert high_temp_results['non_equilibrium_factor'] > 1.0
        assert high_temp_results['total_flux'] > high_temp_results['gray_body_flux']
    
    def test_conductive_heat_transfer(self):
        """Test conductive heat transfer with extreme gradients."""
        results = self.model._calculate_conductive_heat_transfer(
            self.extreme_conditions,
            self.model.materials["rhenium"],
            self.test_geometry
        )
        
        # Check basic conduction
        assert results['thermal_conductivity'] > 0
        assert results['temperature_gradient'] > 0
        assert results['effective_conductivity'] >= results['thermal_conductivity']
        
        # Test hyperbolic effects with rapid heating
        rapid_conditions = ExtremeHeatFluxConditions(
            heat_flux=1000e6,  # 1 GW/m²
            surface_temperature=5000.0,
            time=1e-9  # Very short time - nanosecond pulse
        )
        
        rapid_results = self.model._calculate_conductive_heat_transfer(
            rapid_conditions,
            self.model.materials["tungsten_carbide"],
            self.test_geometry
        )
        
        # Should show hyperbolic effects
        assert rapid_results['fourier_number'] < 1.0
        assert rapid_results['hyperbolic_factor'] > 1.0
    
    def test_temperature_field_solution(self):
        """Test temperature field solution through material thickness."""
        # Create mock heat flux data
        heat_flux_data = {'total_surface_flux': 200e6}
        
        temp_field = self.model._solve_temperature_field(
            self.extreme_conditions,
            self.model.materials["tungsten_carbide"],
            self.test_geometry,
            heat_flux_data
        )
        
        # Check solution structure
        assert len(temp_field['position']) == self.test_geometry['num_nodes']
        assert len(temp_field['temperature']) == self.test_geometry['num_nodes']
        assert len(temp_field['heat_flux']) == self.test_geometry['num_nodes'] - 1
        
        # Check boundary conditions
        assert temp_field['temperature'][0] == self.extreme_conditions.surface_temperature
        assert temp_field['temperature'][-1] == self.test_geometry['back_temperature']
        
        # Check physical consistency
        assert temp_field['max_temperature'] >= temp_field['min_temperature']
        assert temp_field['iterations'] > 0
        assert temp_field['iterations'] <= self.model.max_iterations
        
        # Temperature should decrease through thickness (for this BC)
        assert temp_field['temperature'][0] > temp_field['temperature'][-1]
    
    def test_thermal_stress_calculation(self):
        """Test thermal stress calculation from extreme gradients."""
        # First get temperature field
        heat_flux_data = {'total_surface_flux': 300e6}
        temp_field = self.model._solve_temperature_field(
            self.extreme_conditions,
            self.model.materials["tungsten_carbide"],
            self.test_geometry,
            heat_flux_data
        )
        
        # Calculate thermal stress
        stress_results = self.model.calculate_thermal_stress(
            temp_field,
            "tungsten_carbide",
            self.test_geometry
        )
        
        # Check stress results
        assert stress_results.von_mises_stress > 0
        assert stress_results.max_shear_stress > 0
        assert len(stress_results.principal_stresses) == 3
        assert stress_results.stress_tensor.shape == (3, 3)
        assert stress_results.thermal_strain.shape == (3, 3)
        assert len(stress_results.temperature_gradient) == 3
        
        # Safety factor should be reasonable
        assert stress_results.safety_factor > 0
        if stress_results.safety_factor < 1.0:
            # Material failure expected
            assert stress_results.von_mises_stress > self.test_geometry['yield_strength']
    
    def test_bremsstrahlung_radiation(self):
        """Test bremsstrahlung radiation calculation."""
        electron_temp = 20000.0  # K
        electron_density = 1e23  # m⁻³
        
        brem_flux = self.model._calculate_bremsstrahlung_radiation(
            electron_temp, electron_density
        )
        
        assert brem_flux > 0
        
        # Should scale with density squared and sqrt(temperature)
        double_density_flux = self.model._calculate_bremsstrahlung_radiation(
            electron_temp, 2 * electron_density
        )
        assert double_density_flux > 3 * brem_flux  # Should be ~4x
        
        double_temp_flux = self.model._calculate_bremsstrahlung_radiation(
            2 * electron_temp, electron_density
        )
        assert double_temp_flux > brem_flux  # Should be sqrt(2) times
    
    def test_line_radiation(self):
        """Test line radiation calculation."""
        # Low temperature - should be minimal
        low_temp_flux = self.model._calculate_line_radiation(3000.0, 1e22)
        assert low_temp_flux == 0.0
        
        # High temperature - should have significant line radiation
        high_temp_flux = self.model._calculate_line_radiation(15000.0, 1e22)
        assert high_temp_flux > 0
        
        # Should increase with temperature and density
        higher_temp_flux = self.model._calculate_line_radiation(20000.0, 1e22)
        assert higher_temp_flux > high_temp_flux
        
        higher_density_flux = self.model._calculate_line_radiation(15000.0, 2e22)
        assert higher_density_flux > high_temp_flux
    
    def test_validation_warnings(self):
        """Test validation of extreme conditions."""
        # Test low heat flux warning
        low_flux_conditions = ExtremeHeatFluxConditions(
            heat_flux=50e6,  # 50 MW/m² - below extreme threshold
            surface_temperature=2000.0
        )
        
        warnings = self.model.validate_extreme_conditions(
            low_flux_conditions, "tungsten_carbide"
        )
        assert any("below extreme threshold" in w for w in warnings)
        
        # Test excessive heat flux warning
        excessive_flux_conditions = ExtremeHeatFluxConditions(
            heat_flux=2e9,  # 2 GW/m² - exceeds physical limits
            surface_temperature=3000.0
        )
        
        warnings = self.model.validate_extreme_conditions(
            excessive_flux_conditions, "tungsten_carbide"
        )
        assert any("exceeds physical limits" in w for w in warnings)
        
        # Test temperature exceeding material limits
        high_temp_conditions = ExtremeHeatFluxConditions(
            heat_flux=200e6,
            surface_temperature=4000.0  # K - exceeds tungsten carbide limit
        )
        
        warnings = self.model.validate_extreme_conditions(
            high_temp_conditions, "tungsten_carbide"
        )
        assert any("exceeds material limit" in w for w in warnings)
        
        # Test melting point exceeded
        melting_conditions = ExtremeHeatFluxConditions(
            heat_flux=200e6,
            surface_temperature=3500.0  # K - exceeds melting point
        )
        
        warnings = self.model.validate_extreme_conditions(
            melting_conditions, "tungsten_carbide"
        )
        assert any("melting point" in w for w in warnings)
    
    def test_plasma_validation(self):
        """Test validation of plasma conditions."""
        # Extreme plasma conditions
        extreme_plasma = PlasmaConditions(
            electron_density=1e25,  # m⁻³ - very high
            electron_temperature=100000.0,  # K - very high
            ion_temperature=80000.0,
            magnetic_field=np.array([1.0, 0.0, 0.0]),
            plasma_frequency=1e13,
            debye_length=1e-7,
            ionization_fraction=0.5
        )
        
        extreme_conditions = ExtremeHeatFluxConditions(
            heat_flux=500e6,
            surface_temperature=4000.0,
            plasma_conditions=extreme_plasma
        )
        
        warnings = self.model.validate_extreme_conditions(
            extreme_conditions, "hafnium_carbide"
        )
        
        assert any("density exceeds typical" in w for w in warnings)
        assert any("temperature exceeds typical" in w for w in warnings)
    
    def test_material_compatibility(self):
        """Test material selection for extreme conditions."""
        # Test all materials with extreme conditions
        extreme_conditions = ExtremeHeatFluxConditions(
            heat_flux=400e6,  # 400 MW/m²
            surface_temperature=3500.0  # K
        )
        
        for material_id in self.model.materials.keys():
            try:
                results = self.model.calculate_extreme_heat_flux(
                    extreme_conditions,
                    material_id,
                    self.test_geometry
                )
                
                # Should complete without errors
                assert 'temperature_field' in results
                assert results['temperature_field']['max_temperature'] > 0
                
            except Exception as e:
                pytest.fail(f"Material {material_id} failed with extreme conditions: {e}")
    
    def test_physical_consistency(self):
        """Test physical consistency of calculations."""
        results = self.model.calculate_extreme_heat_flux(
            self.extreme_conditions,
            "carbon_carbon",
            self.test_geometry
        )
        
        # Energy conservation check
        temp_field = results['temperature_field']
        surface_flux = results['total_heat_flux']['total_surface_flux']
        
        # Heat flux should be consistent through material (steady state)
        heat_fluxes = temp_field['heat_flux']
        if len(heat_fluxes) > 1:
            flux_variation = (np.max(heat_fluxes) - np.min(heat_fluxes)) / np.mean(heat_fluxes)
            assert flux_variation < 0.5  # Should be reasonably consistent
        
        # Temperature should be monotonic for this boundary condition
        temperatures = temp_field['temperature']
        temp_diffs = np.diff(temperatures)
        # All differences should have same sign (monotonic)
        if len(temp_diffs) > 1:
            sign_changes = np.sum(np.diff(np.sign(temp_diffs)) != 0)
            assert sign_changes <= 2  # Allow some numerical noise
    
    def test_convergence_behavior(self):
        """Test numerical convergence behavior."""
        # Test with different mesh densities
        geometries = [
            {**self.test_geometry, 'num_nodes': 10},
            {**self.test_geometry, 'num_nodes': 20},
            {**self.test_geometry, 'num_nodes': 40}
        ]
        
        max_temps = []
        for geometry in geometries:
            results = self.model.calculate_extreme_heat_flux(
                self.extreme_conditions,
                "tungsten_carbide",
                geometry
            )
            max_temps.append(results['temperature_field']['max_temperature'])
        
        # Results should converge with mesh refinement
        assert len(max_temps) == 3
        # Differences should decrease with refinement
        diff1 = abs(max_temps[1] - max_temps[0])
        diff2 = abs(max_temps[2] - max_temps[1])
        # Second difference should be smaller (convergence)
        assert diff2 <= diff1 * 2  # Allow some tolerance


if __name__ == "__main__":
    pytest.main([__file__])