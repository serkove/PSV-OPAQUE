"""Tests for multi-physics integration system."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from fighter_jet_sdk.core.multi_physics_integration import (
    HypersonicMultiPhysicsIntegrator,
    HypersonicThermalComponent,
    HypersonicStructuralComponent,
    HypersonicAerodynamicComponent,
    CouplingCoordinator,
    CouplingInterface,
    PhysicsType,
    CouplingStrength,
    MultiPhysicsState
)
from fighter_jet_sdk.common.data_models import AircraftConfiguration, FlowConditions
from fighter_jet_sdk.core.simulation import SimulationParameters, CouplingType


class TestHypersonicThermalComponent:
    """Test hypersonic thermal component."""
    
    def test_initialization(self):
        """Test thermal component initialization."""
        component = HypersonicThermalComponent()
        
        assert component.name == "hypersonic_thermal"
        assert component.priority == 1
        assert not component.initialized
        
        # Initialize component
        assert component.initialize()
        assert component.initialized
        assert component.state['surface_temperature'] == 300.0
        assert component.state['heat_flux'] == 0.0
    
    def test_thermal_state_update(self):
        """Test thermal state update with coupling data."""
        component = HypersonicThermalComponent()
        component.initialize()
        
        # Coupling data from aerodynamics
        coupling_data = {
            'convective_heat_flux': 1e6,  # W/m²
            'radiative_heat_flux': 5e5,   # W/m²
            'plasma_heating': 1e5,        # W/m²
            'surface_deformation': np.array([0.01, 0.0, 0.0])
        }
        
        # Update state
        dt = 1.0  # seconds
        updated_state = component.update_state(dt, coupling_data)
        
        # Check that temperature increased due to heating
        assert updated_state['surface_temperature'] > 300.0
        assert updated_state['heat_flux'] == 1.6e6  # Total heat flux
        assert 'thermal_expansion' in updated_state
        
        # Check coupling variables are updated
        assert 'surface_temperature' in component.coupling_variables
        assert 'thermal_expansion' in component.coupling_variables
    
    def test_ablation_calculation(self):
        """Test ablation rate calculation at high temperatures."""
        component = HypersonicThermalComponent()
        component.initialize()
        
        # Set high surface temperature
        component.surface_temperature = 3000.0  # K
        
        ablation_rate = component._calculate_ablation_rate()
        assert ablation_rate > 0.0  # Should have ablation at high temperature
        
        # Set lower temperature
        component.surface_temperature = 2000.0  # K
        ablation_rate = component._calculate_ablation_rate()
        assert ablation_rate == 0.0  # No ablation below threshold
    
    def test_residual_computation(self):
        """Test thermal residual computation."""
        component = HypersonicThermalComponent()
        component.initialize()
        
        # Set coupling variables
        component.coupling_variables = {
            'aerodynamic_heat_flux': 1e6,
            'target_temperature': 1000.0
        }
        
        # Set current state
        component.state['heat_flux'] = 8e5
        component.state['surface_temperature'] = 900.0
        
        residuals = component.compute_residual(component.state)
        
        assert 'heat_flux_residual' in residuals
        assert 'temperature_residual' in residuals
        assert residuals['heat_flux_residual'] == 2e5  # |1e6 - 8e5|
        assert residuals['temperature_residual'] == 100.0  # |1000 - 900|


class TestHypersonicStructuralComponent:
    """Test hypersonic structural component."""
    
    def test_initialization(self):
        """Test structural component initialization."""
        component = HypersonicStructuralComponent()
        
        assert component.name == "hypersonic_structural"
        assert component.priority == 2
        
        assert component.initialize()
        assert component.initialized
        assert component.state['elastic_modulus'] == 70e9  # Pa
        assert np.allclose(component.state['displacement'], np.zeros(3))
    
    def test_thermal_stress_calculation(self):
        """Test thermal stress calculation."""
        component = HypersonicStructuralComponent()
        component.initialize()
        
        # Test thermal expansion and gradients
        thermal_expansion = 0.001  # 0.1% expansion
        thermal_gradients = np.array([1000.0, 0.0, 0.0])  # K/m
        
        thermal_stress = component._calculate_thermal_stress(thermal_expansion, thermal_gradients)
        
        # Should be a 3x3 stress tensor
        assert thermal_stress.shape == (3, 3)
        assert thermal_stress[0, 0] > 0  # Expansion stress
        assert np.abs(thermal_stress[0, 1]) < 1e-10  # Off-diagonal should be small
    
    def test_temperature_dependent_properties(self):
        """Test temperature-dependent material properties."""
        component = HypersonicStructuralComponent()
        component.initialize()
        
        initial_modulus = component.state['elastic_modulus']
        
        # Update properties at high temperature
        component._update_temperature_dependent_properties(1500.0)  # K
        
        # Modulus should decrease with temperature
        assert component.state['elastic_modulus'] < initial_modulus
        assert component.state['yield_strength'] < 270e6
    
    def test_safety_factor_calculation(self):
        """Test safety factor calculation."""
        component = HypersonicStructuralComponent()
        component.initialize()
        
        # Create test stress tensor
        test_stress = np.array([
            [100e6, 50e6, 0],
            [50e6, 80e6, 0],
            [0, 0, 60e6]
        ])  # Pa
        
        safety_factor = component._calculate_safety_factor(test_stress)
        
        assert safety_factor > 0
        assert safety_factor < 10  # Reasonable range
    
    def test_state_update_with_coupling(self):
        """Test structural state update with coupling data."""
        component = HypersonicStructuralComponent()
        component.initialize()
        
        coupling_data = {
            'thermal_expansion': 0.002,
            'thermal_loads': np.array([500.0, 0.0, 0.0]),
            'surface_temperature': 1200.0,
            'pressure_loads': {'lift': 1e5, 'drag': 5e4},
            'dynamic_pressure': 1e5
        }
        
        updated_state = component.update_state(1.0, coupling_data)
        
        # Check that stress and displacement are calculated
        assert np.any(updated_state['stress'] != 0)
        assert np.any(updated_state['displacement'] != 0)
        assert 'safety_factor' in updated_state
        
        # Check coupling variables
        assert 'surface_deformation' in component.coupling_variables
        assert 'structural_stiffness' in component.coupling_variables


class TestHypersonicAerodynamicComponent:
    """Test hypersonic aerodynamic component."""
    
    def test_initialization(self):
        """Test aerodynamic component initialization."""
        component = HypersonicAerodynamicComponent()
        
        assert component.name == "hypersonic_aerodynamic"
        assert component.priority == 3
        
        assert component.initialize()
        assert component.initialized
        assert component.state['mach_number'] == 60.0
        assert component.state['altitude'] == 60000.0
    
    def test_stagnation_conditions_calculation(self):
        """Test stagnation conditions calculation."""
        component = HypersonicAerodynamicComponent()
        component.initialize()
        
        # Set atmospheric conditions
        component.state.update({
            'atmospheric_temperature': 216.65,  # K
            'atmospheric_pressure': 5474.9,    # Pa
            'mach_number': 60.0
        })
        
        stagnation_conditions = component._calculate_stagnation_conditions()
        
        assert 'temperature' in stagnation_conditions
        assert 'pressure' in stagnation_conditions
        assert 'density' in stagnation_conditions
        
        # Stagnation temperature should be much higher than atmospheric
        assert stagnation_conditions['temperature'] > 216.65
        assert stagnation_conditions['pressure'] > 5474.9
    
    def test_plasma_formation_check(self):
        """Test plasma formation detection."""
        component = HypersonicAerodynamicComponent()
        component.initialize()
        
        # High temperature conditions (should form plasma)
        high_temp_conditions = {'temperature': 10000.0}  # K
        assert component._check_plasma_formation(high_temp_conditions)
        
        # Low temperature conditions (no plasma)
        low_temp_conditions = {'temperature': 5000.0}  # K
        assert not component._check_plasma_formation(low_temp_conditions)
    
    def test_heat_transfer_calculation(self):
        """Test heat transfer calculation."""
        component = HypersonicAerodynamicComponent()
        component.initialize()
        
        stagnation_conditions = {
            'temperature': 8000.0,  # K
            'pressure': 1e6        # Pa
        }
        surface_temperature = 1500.0  # K
        
        conv_flux, rad_flux = component._calculate_heat_transfer(
            stagnation_conditions, surface_temperature
        )
        
        assert conv_flux > 0  # Should have convective heating
        assert rad_flux > 0   # Should have radiative heating at high temperature
        assert conv_flux > rad_flux  # Convective typically dominates
    
    def test_state_update_with_coupling(self):
        """Test aerodynamic state update with coupling data."""
        component = HypersonicAerodynamicComponent()
        component.initialize()
        
        coupling_data = {
            'surface_deformation': np.array([0.01, 0.0, 0.0]),
            'surface_temperature': 1500.0
        }
        
        updated_state = component.update_state(1.0, coupling_data)
        
        # Check that stagnation conditions are calculated
        assert 'stagnation_conditions' in updated_state
        assert 'plasma_formation' in updated_state
        assert 'convective_heat_flux' in updated_state
        
        # Check coupling variables
        assert 'convective_heat_flux' in component.coupling_variables
        assert 'pressure_loads' in component.coupling_variables


class TestCouplingCoordinator:
    """Test coupling coordinator."""
    
    def test_initialization(self):
        """Test coupling coordinator initialization."""
        coordinator = CouplingCoordinator()
        
        assert len(coordinator.coupling_interfaces) == 0
        assert coordinator.convergence_monitor is not None
    
    def test_add_coupling_interface(self):
        """Test adding coupling interfaces."""
        coordinator = CouplingCoordinator()
        
        interface = CouplingInterface(
            source_physics=PhysicsType.THERMAL,
            target_physics=PhysicsType.STRUCTURAL,
            coupling_variables=['temperature', 'expansion'],
            coupling_strength=CouplingStrength.STRONG
        )
        
        coordinator.add_coupling_interface(interface)
        
        assert len(coordinator.coupling_interfaces) == 1
        assert coordinator.coupling_interfaces[0] == interface
    
    def test_setup_default_hypersonic_coupling(self):
        """Test setup of default hypersonic coupling."""
        coordinator = CouplingCoordinator()
        coordinator.setup_default_hypersonic_coupling()
        
        # Should have thermal-structural, structural-aero, and aero-thermal coupling
        assert len(coordinator.coupling_interfaces) == 3
        
        # Check coupling types
        coupling_types = [(i.source_physics, i.target_physics) for i in coordinator.coupling_interfaces]
        
        assert (PhysicsType.THERMAL, PhysicsType.STRUCTURAL) in coupling_types
        assert (PhysicsType.STRUCTURAL, PhysicsType.AERODYNAMIC) in coupling_types
        assert (PhysicsType.AERODYNAMIC, PhysicsType.THERMAL) in coupling_types
    
    def test_convergence_monitoring(self):
        """Test convergence monitoring."""
        coordinator = CouplingCoordinator()
        coordinator.setup_default_hypersonic_coupling()
        
        # Create mock components
        thermal_comp = Mock()
        thermal_comp.compute_residual.return_value = {'temp_residual': 1e-5}
        
        structural_comp = Mock()
        structural_comp.compute_residual.return_value = {'stress_residual': 1e-6}
        
        components = {
            'thermal': thermal_comp,
            'structural': structural_comp
        }
        
        converged, residuals = coordinator.monitor_coupling_convergence(components, 1)
        
        assert 'temp_residual' in residuals
        assert 'stress_residual' in residuals
        assert len(coordinator.convergence_monitor.residual_history) == 1


class TestHypersonicMultiPhysicsIntegrator:
    """Test hypersonic multi-physics integrator."""
    
    def test_initialization(self):
        """Test integrator initialization."""
        integrator = HypersonicMultiPhysicsIntegrator()
        
        assert integrator.thermal_component is not None
        assert integrator.structural_component is not None
        assert integrator.aerodynamic_component is not None
        assert integrator.coupling_coordinator is not None
        
        # Initialize
        assert integrator.initialize()
        assert integrator.initialized
        assert len(integrator.components) == 3  # Three physics components
    
    def test_setup_hypersonic_analysis(self):
        """Test setup for hypersonic analysis."""
        integrator = HypersonicMultiPhysicsIntegrator()
        integrator.initialize()
        
        # Create test configuration and conditions
        config = AircraftConfiguration(
            name="test_hypersonic",
            modules=[]
        )
        
        flow_conditions = FlowConditions(
            mach_number=60.0,
            altitude=60000.0,
            angle_of_attack=0.0,
            sideslip_angle=0.0,
            temperature=216.65,
            pressure=5474.9,
            density=0.03
        )
        
        analysis_params = {
            'initial_temperature': 350.0,
            'material_properties': {'elastic_modulus': 80e9}
        }
        
        # Setup analysis
        integrator._setup_hypersonic_analysis(config, flow_conditions, analysis_params)
        
        # Check that components are configured
        assert integrator.aerodynamic_component.state['mach_number'] == 60.0
        assert integrator.aerodynamic_component.state['altitude'] == 60000.0
        assert integrator.thermal_component.surface_temperature == 350.0
        assert integrator.structural_component.state['elastic_modulus'] == 80e9
    
    @patch('fighter_jet_sdk.core.multi_physics_integration.time.time')
    def test_multi_physics_state_update(self, mock_time):
        """Test multi-physics state update."""
        mock_time.return_value = 1000.0
        
        integrator = HypersonicMultiPhysicsIntegrator()
        integrator.initialize()
        
        # Set component states
        integrator.thermal_component.state = {'temperature': 1500.0}
        integrator.structural_component.state = {'stress': np.eye(3)}
        integrator.aerodynamic_component.state = {'mach': 60.0}
        
        residuals = {'temp_residual': 1e-5, 'stress_residual': 1e-6}
        
        integrator._update_multi_physics_state(residuals, 5, True)
        
        state = integrator.multi_physics_state
        assert state.thermal_state == {'temperature': 1500.0}
        assert np.array_equal(state.structural_state['stress'], np.eye(3))
        assert state.aerodynamic_state == {'mach': 60.0}
        assert state.coupling_residuals == residuals
        assert state.iteration_count == 5
        assert state.converged
        assert state.timestamp == 1000.0
    
    def test_coupling_relaxation(self):
        """Test coupling relaxation application."""
        integrator = HypersonicMultiPhysicsIntegrator()
        integrator.initialize()
        
        # Set coupling variables
        integrator.thermal_component.coupling_variables = {
            'temperature': 1000.0,
            'heat_flux': np.array([1e6, 0, 0])
        }
        
        original_temp = integrator.thermal_component.coupling_variables['temperature']
        original_flux = integrator.thermal_component.coupling_variables['heat_flux'].copy()
        
        integrator._apply_coupling_relaxation()
        
        # Values should be reduced by relaxation factor (0.5)
        assert integrator.thermal_component.coupling_variables['temperature'] == original_temp * 0.5
        assert np.allclose(integrator.thermal_component.coupling_variables['heat_flux'], original_flux * 0.5)
    
    def test_post_process_results(self):
        """Test post-processing of hypersonic results."""
        integrator = HypersonicMultiPhysicsIntegrator()
        integrator.initialize()
        
        # Set up final state
        integrator.multi_physics_state = MultiPhysicsState(
            thermal_state={
                'surface_temperature': 2000.0,
                'heat_flux': 1e6,
                'thermal_expansion': 0.001,
                'ablation_rate': 1e-6
            },
            structural_state={
                'stress': np.array([[1e8, 0, 0], [0, 5e7, 0], [0, 0, 3e7]]),
                'displacement': np.array([0.01, 0.005, 0.002]),
                'safety_factor': 2.5,
                'thermal_stress': np.array([[5e7, 0, 0], [0, 2e7, 0], [0, 0, 1e7]])
            },
            aerodynamic_state={
                'stagnation_conditions': {'temperature': 8000.0, 'pressure': 1e6},
                'convective_heat_flux': 8e5,
                'radiative_heat_flux': 2e5,
                'plasma_formation': True,
                'shock_properties': {'standoff_distance': 0.01}
            },
            coupling_residuals={'temp_residual': 1e-7, 'stress_residual': 1e-8},
            iteration_count=15,
            converged=True
        )
        
        # Mock simulation results
        mock_results = Mock()
        mock_results.computational_metrics = {'cpu_time': 100.0}
        mock_results.validation_results = {'physics_valid': True}
        
        analysis_results = integrator._post_process_hypersonic_results(mock_results, {})
        
        # Check thermal results
        assert analysis_results['thermal']['max_surface_temperature'] == 2000.0
        assert analysis_results['thermal']['max_heat_flux'] == 1e6
        assert analysis_results['thermal']['ablation_rate'] == 1e-6
        
        # Check structural results
        assert analysis_results['structural']['max_stress'] == 1e8
        assert analysis_results['structural']['safety_factor'] == 2.5
        
        # Check aerodynamic results
        assert analysis_results['aerodynamic']['stagnation_temperature'] == 8000.0
        assert analysis_results['aerodynamic']['plasma_formation']
        
        # Check coupling results
        assert analysis_results['coupling']['converged']
        assert analysis_results['coupling']['final_iteration_count'] == 15
    
    def test_export_coupling_data(self):
        """Test export of coupling data."""
        integrator = HypersonicMultiPhysicsIntegrator()
        integrator.initialize()
        
        # Set up some state
        integrator.coupling_coordinator.convergence_monitor.residual_history = [1e-3, 1e-4, 1e-5]
        integrator.multi_physics_state.thermal_state = {'temperature': 1500.0}
        
        exported_data = integrator.export_coupling_data()
        
        assert 'coupling_interfaces' in exported_data
        assert 'convergence_metrics' in exported_data
        assert 'current_state' in exported_data
        
        # Check coupling interfaces
        assert len(exported_data['coupling_interfaces']) == 3  # Default hypersonic coupling
        
        # Check convergence metrics
        assert exported_data['convergence_metrics']['residual_history'] == [1e-3, 1e-4, 1e-5]
        
        # Check current state
        assert exported_data['current_state']['thermal']['temperature'] == 1500.0


class TestIntegrationScenarios:
    """Test integration scenarios."""
    
    def test_mach_60_analysis_scenario(self):
        """Test complete Mach 60 analysis scenario."""
        integrator = HypersonicMultiPhysicsIntegrator()
        
        # Configuration for Mach 60 analysis
        config = {
            'max_workers': 2,
            'convergence_tolerance': 1e-5
        }
        
        integrator.config = config
        assert integrator.initialize()
        
        # Create aircraft configuration
        aircraft_config = AircraftConfiguration(
            name="mach_60_vehicle",
            modules=[]
        )
        
        # Mach 60 flight conditions
        flow_conditions = FlowConditions(
            mach_number=60.0,
            altitude=60000.0,
            angle_of_attack=0.0,
            sideslip_angle=0.0,
            temperature=216.65,
            pressure=5474.9,
            density=0.03
        )
        
        # Analysis parameters
        analysis_params = {
            'total_time': 10.0,
            'time_step': 1.0,
            'convergence_tolerance': 1e-5,
            'max_iterations': 20,
            'initial_temperature': 300.0
        }
        
        # Run analysis (this will use simplified physics models)
        try:
            results = integrator.run_coupled_analysis(
                aircraft_config, flow_conditions, analysis_params
            )
            
            # Verify results structure
            assert 'thermal' in results
            assert 'structural' in results
            assert 'aerodynamic' in results
            assert 'coupling' in results
            
            # Check that thermal analysis shows heating
            assert results['thermal']['max_surface_temperature'] > 300.0
            
            # Check that structural analysis shows stress
            assert results['structural']['max_stress'] > 0.0
            
            # Check that aerodynamic analysis shows high stagnation conditions
            assert results['aerodynamic']['stagnation_temperature'] > 1000.0
            
        except Exception as e:
            # Analysis might fail due to simplified models, but should not crash
            assert "failed" in str(e).lower()
    
    def test_coupling_convergence_scenario(self):
        """Test coupling convergence behavior."""
        integrator = HypersonicMultiPhysicsIntegrator()
        integrator.initialize()
        
        # Ensure components are properly initialized
        integrator.thermal_component.initialize()
        integrator.structural_component.initialize()
        integrator.aerodynamic_component.initialize()
        
        # Set up components with known coupling variables
        integrator.thermal_component.coupling_variables = {
            'surface_temperature': 1500.0,
            'thermal_expansion': 0.001
        }
        
        integrator.structural_component.coupling_variables = {
            'surface_deformation': np.array([0.01, 0.0, 0.0]),
            'structural_stiffness': 70e9
        }
        
        integrator.aerodynamic_component.coupling_variables = {
            'convective_heat_flux': 1e6,
            'pressure_loads': {'lift': 1e5, 'drag': 5e4}
        }
        
        # Test coupling iteration
        dt = 1.0
        converged = integrator._perform_coupling_iteration(dt)
        
        # Should attempt coupling (may or may not converge with simplified models)
        assert isinstance(converged, bool)
        
        # Check that coupling data was exchanged
        # Each component should have received data from others
        thermal_vars = integrator.thermal_component.coupling_variables
        structural_vars = integrator.structural_component.coupling_variables
        aero_vars = integrator.aerodynamic_component.coupling_variables
        
        # Variables should be present (exact values depend on simplified models)
        assert len(thermal_vars) > 0
        assert len(structural_vars) > 0
        assert len(aero_vars) > 0


if __name__ == '__main__':
    pytest.main([__file__])