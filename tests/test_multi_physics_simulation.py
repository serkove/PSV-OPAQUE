"""Tests for multi-physics simulation orchestration."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import time

from fighter_jet_sdk.core.simulation import (
    MultiPhysicsOrchestrator,
    AerodynamicsComponent,
    ThermalComponent,
    StructuralComponent,
    SimulationParameters,
    SimulationState,
    CouplingType,
    ResourceMonitor,
    SimulationValidator,
    DynamicManeuverSimulator
)
from fighter_jet_sdk.common.data_models import AircraftConfiguration
from fighter_jet_sdk.core.errors import SimulationError


class TestPhysicsComponents:
    """Test physics components."""
    
    def test_aerodynamics_component_initialization(self):
        """Test aerodynamics component initialization."""
        aero = AerodynamicsComponent()
        assert aero.name == "aerodynamics"
        assert aero.priority == 1
        assert not aero.initialized
        
        success = aero.initialize()
        assert success
        assert aero.initialized
        assert 'velocity' in aero.state
        assert 'angular_velocity' in aero.state
    
    def test_aerodynamics_component_update(self):
        """Test aerodynamics component state update."""
        aero = AerodynamicsComponent()
        aero.initialize()
        
        coupling_data = {
            'structural_deformation': np.array([0.1, 0.0, 0.0]),
            'temperature_field': {'average': 350.0}
        }
        
        initial_time = aero.simulation_time
        state = aero.update_state(0.01, coupling_data)
        
        assert aero.simulation_time > initial_time
        assert 'velocity' in state
        assert 'pressure_loads' in aero.coupling_variables
        assert 'heat_flux' in aero.coupling_variables
    
    def test_thermal_component_initialization(self):
        """Test thermal component initialization."""
        thermal = ThermalComponent()
        assert thermal.name == "thermal"
        assert thermal.priority == 2
        
        success = thermal.initialize()
        assert success
        assert thermal.state['temperature'] == 300.0
    
    def test_thermal_component_update(self):
        """Test thermal component state update."""
        thermal = ThermalComponent()
        thermal.initialize()
        
        coupling_data = {
            'heat_flux': {'convective': 1000.0},
            'electronics_heat': 500.0
        }
        
        initial_temp = thermal.state['temperature']
        state = thermal.update_state(0.01, coupling_data)
        
        # Temperature should increase due to heat input
        assert state['temperature'] > initial_temp
        assert 'temperature_field' in thermal.coupling_variables
    
    def test_structural_component_initialization(self):
        """Test structural component initialization."""
        structural = StructuralComponent()
        assert structural.name == "structural"
        assert structural.priority == 3
        
        success = structural.initialize()
        assert success
        assert np.allclose(structural.state['displacement'], np.zeros(3))
    
    def test_structural_component_update(self):
        """Test structural component state update."""
        structural = StructuralComponent()
        structural.initialize()
        
        coupling_data = {
            'pressure_loads': {'lift': 10000.0, 'drag': 2000.0},
            'thermal_expansion': 1e-4
        }
        
        state = structural.update_state(0.01, coupling_data)
        
        # Should have some displacement due to loads
        assert not np.allclose(state['displacement'], np.zeros(3))
        assert 'structural_deformation' in structural.coupling_variables


class TestResourceMonitor:
    """Test resource monitoring."""
    
    def test_resource_monitor_initialization(self):
        """Test resource monitor initialization."""
        monitor = ResourceMonitor()
        assert monitor.start_time is None
        assert len(monitor.memory_usage) == 0
        assert len(monitor.cpu_usage) == 0
    
    def test_resource_monitor_start(self):
        """Test starting resource monitoring."""
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        assert monitor.start_time is not None
        assert isinstance(monitor.start_time, float)
    
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    def test_resource_monitor_record(self, mock_cpu, mock_memory):
        """Test recording resource usage."""
        mock_memory.return_value.used = 1024 * 1024 * 1024  # 1GB
        mock_cpu.return_value = 50.0
        
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        usage = monitor.record_usage()
        
        assert usage['memory_mb'] == 1024.0
        assert usage['cpu_percent'] == 50.0
        assert 'elapsed_time' in usage
    
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    def test_resource_monitor_summary(self, mock_cpu, mock_memory):
        """Test resource usage summary."""
        mock_memory.return_value.used = 1024 * 1024 * 1024  # 1GB
        mock_cpu.return_value = 50.0
        
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        # Record some usage
        monitor.record_usage()
        monitor.record_usage()
        
        summary = monitor.get_summary()
        
        assert 'peak_memory_mb' in summary
        assert 'avg_memory_mb' in summary
        assert 'peak_cpu_percent' in summary
        assert 'avg_cpu_percent' in summary
        assert 'total_time' in summary


class TestSimulationValidator:
    """Test simulation validation."""
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = SimulationValidator()
        assert validator.logger is not None
    
    def test_convergence_validation_converged(self):
        """Test convergence validation for converged case."""
        validator = SimulationValidator()
        
        residuals = [
            {'residual1': 1e-3, 'residual2': 5e-4},
            {'residual1': 1e-6, 'residual2': 5e-7}
        ]
        
        converged, results = validator.validate_convergence(residuals, 1e-5)
        
        assert converged
        assert results['converged']
        assert results['max_residual'] < 1e-5
    
    def test_convergence_validation_not_converged(self):
        """Test convergence validation for non-converged case."""
        validator = SimulationValidator()
        
        residuals = [
            {'residual1': 1e-3, 'residual2': 5e-4},
            {'residual1': 1e-2, 'residual2': 5e-3}
        ]
        
        converged, results = validator.validate_convergence(residuals, 1e-5)
        
        assert not converged
        assert not results['converged']
        assert results['max_residual'] > 1e-5
    
    def test_physics_validation(self):
        """Test physics validation."""
        validator = SimulationValidator()
        
        # Mock results object
        results = Mock()
        
        validation = validator.validate_physics(results)
        
        assert 'energy_conservation' in validation
        assert 'momentum_conservation' in validation
        assert 'mass_conservation' in validation
        assert 'stability_check' in validation
    
    def test_numerical_stability_validation_stable(self):
        """Test numerical stability validation for stable case."""
        validator = SimulationValidator()
        
        # Create stable state history
        state_history = []
        for i in range(20):
            state_history.append({
                'temperature': 300.0 + 0.1 * i,
                'pressure': 101325.0 + 10 * i
            })
        
        validation = validator.validate_numerical_stability(state_history)
        
        assert validation['stable']
        assert len(validation['issues']) == 0
    
    def test_numerical_stability_validation_unstable(self):
        """Test numerical stability validation for unstable case."""
        validator = SimulationValidator()
        
        # Create unstable state history with exponential growth
        state_history = []
        for i in range(20):
            state_history.append({
                'temperature': 300.0 * (2.0 ** i),  # More aggressive exponential growth
                'pressure': 101325.0
            })
        
        validation = validator.validate_numerical_stability(state_history)
        
        assert not validation['stable']
        assert len(validation['issues']) > 0


class TestMultiPhysicsOrchestrator:
    """Test multi-physics orchestrator."""
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        orchestrator = MultiPhysicsOrchestrator()
        assert orchestrator.state == SimulationState.IDLE
        assert len(orchestrator.components) == 0
        assert not orchestrator.initialized
    
    def test_add_component(self):
        """Test adding physics components."""
        orchestrator = MultiPhysicsOrchestrator()
        
        aero = AerodynamicsComponent()
        thermal = ThermalComponent()
        
        orchestrator.add_component(aero)
        orchestrator.add_component(thermal)
        
        assert len(orchestrator.components) == 2
        # Components should be sorted by priority
        assert orchestrator.components[0].priority <= orchestrator.components[1].priority
    
    def test_orchestrator_initialize_success(self):
        """Test successful orchestrator initialization."""
        orchestrator = MultiPhysicsOrchestrator()
        
        # Add components
        aero = AerodynamicsComponent()
        thermal = ThermalComponent()
        orchestrator.add_component(aero)
        orchestrator.add_component(thermal)
        
        success = orchestrator.initialize()
        
        assert success
        assert orchestrator.initialized
        assert orchestrator.state == SimulationState.IDLE
        assert all(comp.initialized for comp in orchestrator.components)
    
    def test_orchestrator_initialize_failure(self):
        """Test orchestrator initialization failure."""
        orchestrator = MultiPhysicsOrchestrator()
        
        # Add a mock component that fails to initialize
        mock_component = Mock()
        mock_component.initialize.return_value = False
        mock_component.name = "mock_component"
        mock_component.priority = 1
        
        orchestrator.add_component(mock_component)
        
        success = orchestrator.initialize()
        
        assert not success
        assert not orchestrator.initialized
        assert orchestrator.state == SimulationState.ERROR
    
    def test_validate_input(self):
        """Test input validation."""
        orchestrator = MultiPhysicsOrchestrator()
        
        # Valid inputs
        config = AircraftConfiguration()
        assert orchestrator.validate_input(config)
        assert orchestrator.validate_input({'parameters': {}})
        
        # Invalid inputs
        assert not orchestrator.validate_input("invalid")
        assert not orchestrator.validate_input(123)
    
    def test_setup_simulation(self):
        """Test simulation setup."""
        orchestrator = MultiPhysicsOrchestrator()
        orchestrator.initialize()
        
        config = AircraftConfiguration()
        parameters = SimulationParameters(total_time=5.0)
        
        success = orchestrator.setup_simulation(config, parameters)
        
        assert success
        assert orchestrator.parameters.total_time == 5.0
    
    def test_loose_coupling_simulation(self):
        """Test loose coupling simulation."""
        orchestrator = MultiPhysicsOrchestrator()
        
        # Add components
        aero = AerodynamicsComponent()
        thermal = ThermalComponent()
        orchestrator.add_component(aero)
        orchestrator.add_component(thermal)
        
        orchestrator.initialize()
        
        # Set up short simulation
        parameters = {
            'total_time': 0.1,
            'time_step': 0.01,
            'coupling_type': CouplingType.LOOSE,
            'output_frequency': 1
        }
        
        results = orchestrator.run_simulation(parameters)
        
        assert orchestrator.state == SimulationState.COMPLETED
        assert len(results.time_history) > 0
        assert len(results.state_history) > 0
        assert 'total_timesteps' in results.performance_metrics
    
    def test_tight_coupling_simulation(self):
        """Test tight coupling simulation."""
        orchestrator = MultiPhysicsOrchestrator()
        
        # Add components
        aero = AerodynamicsComponent()
        thermal = ThermalComponent()
        structural = StructuralComponent()
        orchestrator.add_component(aero)
        orchestrator.add_component(thermal)
        orchestrator.add_component(structural)
        
        orchestrator.initialize()
        
        # Set up short simulation with tight coupling
        parameters = {
            'total_time': 0.05,
            'time_step': 0.01,
            'coupling_type': CouplingType.TIGHT,
            'convergence_tolerance': 1e-3,
            'output_frequency': 1
        }
        
        results = orchestrator.run_simulation(parameters)
        
        assert orchestrator.state == SimulationState.COMPLETED
        assert len(results.time_history) > 0
        assert 'convergence' in results.validation_results
    
    def test_adaptive_time_stepping(self):
        """Test adaptive time stepping."""
        orchestrator = MultiPhysicsOrchestrator()
        
        aero = AerodynamicsComponent()
        orchestrator.add_component(aero)
        orchestrator.initialize()
        
        # Test adaptive time step calculation
        orchestrator.results.convergence_history = [1e-3, 1e-6]  # Improving convergence
        
        new_dt = orchestrator._calculate_adaptive_timestep(0.01)
        
        # Should increase time step for good convergence
        assert new_dt >= 0.01
    
    def test_simulation_control(self):
        """Test simulation control methods."""
        orchestrator = MultiPhysicsOrchestrator()
        orchestrator.state = SimulationState.RUNNING
        
        # Test pause
        orchestrator.pause_simulation()
        assert orchestrator.state == SimulationState.PAUSED
        
        # Test resume
        orchestrator.resume_simulation()
        assert orchestrator.state == SimulationState.RUNNING
        
        # Test stop
        orchestrator.stop_simulation()
        assert orchestrator._stop_requested
    
    def test_get_simulation_status(self):
        """Test getting simulation status."""
        orchestrator = MultiPhysicsOrchestrator()
        
        aero = AerodynamicsComponent()
        orchestrator.add_component(aero)
        
        status = orchestrator.get_simulation_status()
        
        assert 'state' in status
        assert 'components' in status
        assert 'current_time' in status
        assert 'iterations' in status
        assert status['components'] == ['aerodynamics']


class TestDynamicManeuverSimulator:
    """Test dynamic maneuver simulator."""
    
    def test_maneuver_simulator_initialization(self):
        """Test maneuver simulator initialization."""
        orchestrator = MultiPhysicsOrchestrator()
        simulator = DynamicManeuverSimulator(orchestrator)
        
        assert simulator.orchestrator is orchestrator
        assert len(simulator.maneuver_profiles) == 0
    
    def test_define_maneuver(self):
        """Test defining a maneuver profile."""
        orchestrator = MultiPhysicsOrchestrator()
        simulator = DynamicManeuverSimulator(orchestrator)
        
        profile = {
            'duration': 10.0,
            'time_step': 0.01,
            'max_g_force': 9.0,
            'control_inputs': {'elevator': [0, 5, 0], 'aileron': [0, 0, 0]}
        }
        
        simulator.define_maneuver('barrel_roll', profile)
        
        assert 'barrel_roll' in simulator.maneuver_profiles
        assert simulator.maneuver_profiles['barrel_roll'] == profile
    
    def test_simulate_maneuver_unknown(self):
        """Test simulating unknown maneuver."""
        orchestrator = MultiPhysicsOrchestrator()
        simulator = DynamicManeuverSimulator(orchestrator)
        
        config = AircraftConfiguration()
        
        with pytest.raises(ValueError, match="Unknown maneuver"):
            simulator.simulate_maneuver('unknown_maneuver', config)
    
    def test_simulate_maneuver_success(self):
        """Test successful maneuver simulation."""
        orchestrator = MultiPhysicsOrchestrator()
        
        # Add components and initialize
        aero = AerodynamicsComponent()
        orchestrator.add_component(aero)
        orchestrator.initialize()
        
        simulator = DynamicManeuverSimulator(orchestrator)
        
        # Define a simple maneuver
        profile = {
            'duration': 0.1,
            'time_step': 0.01
        }
        simulator.define_maneuver('test_maneuver', profile)
        
        config = AircraftConfiguration()
        
        # Mock the setup_simulation method to avoid complex setup
        orchestrator.setup_simulation = Mock(return_value=True)
        
        results = simulator.simulate_maneuver('test_maneuver', config)
        
        assert results is not None
        assert orchestrator.setup_simulation.called


class TestSimulationParameters:
    """Test simulation parameters."""
    
    def test_default_parameters(self):
        """Test default simulation parameters."""
        params = SimulationParameters()
        
        assert params.total_time == 10.0
        assert params.time_step == 0.01
        assert params.max_iterations == 100
        assert params.convergence_tolerance == 1e-6
        assert params.coupling_type == CouplingType.LOOSE
        assert params.output_frequency == 10
        assert params.enable_adaptive_stepping
        assert params.max_time_step == 0.1
        assert params.min_time_step == 1e-6
    
    def test_custom_parameters(self):
        """Test custom simulation parameters."""
        params = SimulationParameters(
            total_time=20.0,
            time_step=0.005,
            coupling_type=CouplingType.TIGHT,
            enable_adaptive_stepping=False
        )
        
        assert params.total_time == 20.0
        assert params.time_step == 0.005
        assert params.coupling_type == CouplingType.TIGHT
        assert not params.enable_adaptive_stepping


if __name__ == '__main__':
    pytest.main([__file__])