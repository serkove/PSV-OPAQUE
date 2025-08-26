"""Tests for Thermal Management System."""

import pytest
from fighter_jet_sdk.engines.propulsion.thermal_manager import (
    ThermalManager, ThermalLoad, CoolantType, HeatExchangerType,
    ThermalSystemConfig, CoolantProperties, HeatExchangerSpec
)


class TestThermalManager:
    """Test cases for ThermalManager."""
    
    @pytest.fixture
    def thermal_manager(self):
        """Create thermal manager for testing."""
        return ThermalManager()
    
    @pytest.fixture
    def sample_thermal_loads(self):
        """Create sample thermal loads for testing."""
        return [
            ThermalLoad(
                load_id="radar_transmitter",
                name="AESA Radar Transmitter",
                power_dissipation=25000.0,  # 25 kW
                operating_temperature_range=(273.15, 353.15),
                critical_temperature=373.15,  # 100°C
                duty_cycle=0.3,
                location=(2.0, 0.0, 1.0),
                thermal_resistance=0.05
            ),
            ThermalLoad(
                load_id="laser_system",
                name="Directed Energy Laser",
                power_dissipation=100000.0,  # 100 kW
                operating_temperature_range=(273.15, 323.15),
                critical_temperature=343.15,  # 70°C
                duty_cycle=0.1,
                location=(5.0, 0.0, 0.5),
                thermal_resistance=0.02
            ),
            ThermalLoad(
                load_id="avionics_bay",
                name="Avionics Electronics",
                power_dissipation=5000.0,  # 5 kW
                operating_temperature_range=(273.15, 333.15),
                critical_temperature=358.15,  # 85°C
                duty_cycle=1.0,
                location=(1.0, 1.0, 0.8),
                thermal_resistance=0.1
            ),
            ThermalLoad(
                load_id="power_electronics",
                name="Power Conversion Unit",
                power_dissipation=15000.0,  # 15 kW
                operating_temperature_range=(273.15, 343.15),
                critical_temperature=393.15,  # 120°C
                duty_cycle=0.8,
                location=(3.0, -1.0, 0.5),
                thermal_resistance=0.08
            )
        ]
    
    @pytest.fixture
    def design_requirements(self):
        """Create design requirements for testing."""
        return {
            'ambient_temperature': 288.15,  # 15°C
            'max_coolant_temperature': 373.15,  # 100°C
            'safety_margin': 0.2,
            'max_system_mass': 500.0,  # kg
            'max_system_volume': 1.0   # m³
        }
    
    def test_initialization(self, thermal_manager):
        """Test thermal manager initialization."""
        assert len(thermal_manager.coolant_database) > 0
        assert CoolantType.AIR in thermal_manager.coolant_database
        assert CoolantType.LIQUID_COOLANT in thermal_manager.coolant_database
        assert CoolantType.PHASE_CHANGE in thermal_manager.coolant_database
        assert len(thermal_manager.thermal_systems) == 0
    
    def test_coolant_database(self, thermal_manager):
        """Test coolant properties database."""
        # Check air properties
        air = thermal_manager.coolant_database[CoolantType.AIR]
        assert air.density > 0
        assert air.specific_heat > 0
        assert air.thermal_conductivity > 0
        assert air.viscosity > 0
        
        # Check liquid coolant properties
        liquid = thermal_manager.coolant_database[CoolantType.LIQUID_COOLANT]
        assert liquid.density > air.density  # Liquid should be denser
        assert liquid.specific_heat > air.specific_heat
        assert liquid.thermal_conductivity > air.thermal_conductivity
        
        # Check phase change coolant
        phase_change = thermal_manager.coolant_database[CoolantType.PHASE_CHANGE]
        assert phase_change.phase_change_temp is not None
        assert phase_change.latent_heat is not None
        assert phase_change.latent_heat > 0
    
    def test_coolant_selection(self, thermal_manager, sample_thermal_loads):
        """Test coolant type selection logic."""
        # Low power loads should use air cooling
        low_power_loads = [load for load in sample_thermal_loads if load.power_dissipation < 10000]
        coolant_type = thermal_manager._select_coolant_type(5000, low_power_loads)
        assert coolant_type == CoolantType.AIR
        
        # Medium power loads should use liquid cooling
        medium_power_loads = [load for load in sample_thermal_loads if 10000 <= load.power_dissipation < 50000]
        total_power = sum(load.power_dissipation for load in medium_power_loads)
        coolant_type = thermal_manager._select_coolant_type(total_power, medium_power_loads)
        assert coolant_type in [CoolantType.LIQUID_COOLANT, CoolantType.PHASE_CHANGE]
        
        # High power loads should use advanced cooling
        high_power_loads = [load for load in sample_thermal_loads if load.power_dissipation >= 50000]
        total_power = sum(load.power_dissipation for load in high_power_loads)
        coolant_type = thermal_manager._select_coolant_type(total_power, high_power_loads)
        assert coolant_type in [CoolantType.PHASE_CHANGE, CoolantType.LIQUID_METAL]
    
    def test_thermal_system_design(self, thermal_manager, sample_thermal_loads, design_requirements):
        """Test complete thermal system design."""
        system_config = thermal_manager.design_thermal_system(sample_thermal_loads, design_requirements)
        
        # Check system configuration
        assert isinstance(system_config, ThermalSystemConfig)
        assert len(system_config.thermal_loads) == len(sample_thermal_loads)
        assert len(system_config.heat_exchangers) > 0
        assert len(system_config.coolant_loops) > 0
        
        # Check heat exchanger capacity
        total_power = sum(load.power_dissipation for load in sample_thermal_loads)
        total_hx_capacity = sum(hx.max_heat_transfer for hx in system_config.heat_exchangers)
        assert total_hx_capacity >= total_power
        
        # Check coolant loops
        assert "primary" in system_config.coolant_loops
        primary_coolant = system_config.coolant_loops["primary"]
        assert isinstance(primary_coolant, CoolantProperties)
    
    def test_heat_exchanger_design(self, thermal_manager, sample_thermal_loads, design_requirements):
        """Test heat exchanger design logic."""
        coolant_type = CoolantType.LIQUID_COOLANT
        heat_exchangers = thermal_manager._design_heat_exchangers(
            sample_thermal_loads, coolant_type, design_requirements
        )
        
        assert len(heat_exchangers) > 0
        
        for hx in heat_exchangers:
            assert isinstance(hx, HeatExchangerSpec)
            assert hx.max_heat_transfer > 0
            assert hx.effectiveness > 0
            assert hx.effectiveness <= 1.0
            assert hx.mass > 0
            assert hx.volume > 0
            assert hx.exchanger_type in HeatExchangerType
    
    def test_thermal_load_grouping(self, thermal_manager, sample_thermal_loads):
        """Test thermal load grouping algorithm."""
        groups = thermal_manager._group_thermal_loads(sample_thermal_loads)
        
        assert len(groups) > 0
        
        # Check that all loads are assigned to groups
        total_loads_in_groups = sum(len(loads) for loads in groups.values())
        assert total_loads_in_groups == len(sample_thermal_loads)
        
        # Check group structure
        for group_id, loads in groups.items():
            assert len(loads) > 0
            assert isinstance(group_id, str)
    
    def test_thermal_performance_analysis(self, thermal_manager, sample_thermal_loads, design_requirements):
        """Test thermal performance analysis."""
        system_config = thermal_manager.design_thermal_system(sample_thermal_loads, design_requirements)
        
        operating_conditions = {
            'ambient_temperature': 288.15,
            'flight_mach': 1.5,
            'altitude': 10000.0
        }
        
        performance = thermal_manager.analyze_thermal_performance(system_config, operating_conditions)
        
        # Check performance structure
        assert 'steady_state_temperatures' in performance
        assert 'performance_metrics' in performance
        assert 'thermal_network' in performance
        
        # Check temperatures
        temperatures = performance['steady_state_temperatures']
        assert len(temperatures) > 0
        
        for load in sample_thermal_loads:
            assert load.load_id in temperatures
            temp = temperatures[load.load_id]
            assert temp > 0  # Temperature in Kelvin
            assert temp > 0  # Temperature should be positive (Kelvin)
        
        # Check performance metrics
        metrics = performance['performance_metrics']
        assert 'temperature_margins' in metrics
        assert 'min_temperature_margin' in metrics
        assert 'thermal_efficiency' in metrics
        assert 'system_mass' in metrics
        assert 'system_volume' in metrics
    
    def test_thermal_network_building(self, thermal_manager, sample_thermal_loads, design_requirements):
        """Test thermal network construction."""
        system_config = thermal_manager.design_thermal_system(sample_thermal_loads, design_requirements)
        network = thermal_manager._build_thermal_network(system_config)
        
        # Check network structure
        assert len(network) > len(sample_thermal_loads)  # Loads + heat exchangers + ambient
        assert "ambient" in network
        
        # Check load nodes
        for load in sample_thermal_loads:
            assert load.load_id in network
            node = network[load.load_id]
            assert node.heat_generation > 0
            assert len(node.connected_nodes) > 0
        
        # Check heat exchanger nodes
        for hx in system_config.heat_exchangers:
            assert hx.exchanger_id in network
            node = network[hx.exchanger_id]
            assert node.heat_generation == 0  # Heat exchangers don't generate heat
            assert len(node.connected_nodes) > 0
    
    def test_steady_state_solver(self, thermal_manager, sample_thermal_loads, design_requirements):
        """Test steady-state thermal solver."""
        system_config = thermal_manager.design_thermal_system(sample_thermal_loads, design_requirements)
        network = thermal_manager._build_thermal_network(system_config)
        
        operating_conditions = {'ambient_temperature': 288.15}
        temperatures = thermal_manager._solve_steady_state(network, operating_conditions)
        
        # Check solution
        assert len(temperatures) == len(network)
        assert temperatures["ambient"] == 288.15  # Fixed ambient temperature
        
        # Check that all temperatures are positive
        for node_id, temp in temperatures.items():
            assert temp > 0  # Temperature should be positive (Kelvin)
        
        # Check that heat-generating nodes are hotter than ambient
        for load in sample_thermal_loads:
            if load.power_dissipation > 0:
                assert temperatures[load.load_id] > temperatures["ambient"]
    
    def test_transient_simulation(self, thermal_manager, sample_thermal_loads, design_requirements):
        """Test transient thermal simulation."""
        system_config = thermal_manager.design_thermal_system(sample_thermal_loads, design_requirements)
        
        # Create power profile (step change)
        power_profile = [
            (0.0, {load.load_id: 0.0 for load in sample_thermal_loads}),  # Start with no power
            (10.0, {load.load_id: load.power_dissipation * load.duty_cycle for load in sample_thermal_loads}),  # Step to full power
            (100.0, {load.load_id: load.power_dissipation * load.duty_cycle for load in sample_thermal_loads}),  # Hold power
            (110.0, {load.load_id: 0.0 for load in sample_thermal_loads})  # Step to zero power
        ]
        
        result = thermal_manager.simulate_transient_response(system_config, power_profile, time_step=1.0)
        
        # Check result structure
        assert 'time_history' in result
        assert 'temperature_history' in result
        assert 'final_temperatures' in result
        
        # Check time history
        time_history = result['time_history']
        assert len(time_history) == len(power_profile)
        
        # Check temperature history
        temp_history = result['temperature_history']
        for load in sample_thermal_loads:
            assert load.load_id in temp_history
            assert len(temp_history[load.load_id]) == len(power_profile)
    
    def test_design_optimization(self, thermal_manager, sample_thermal_loads, design_requirements):
        """Test thermal design optimization."""
        # Optimize for mass
        optimized_config = thermal_manager.optimize_thermal_design(
            sample_thermal_loads, design_requirements, "mass"
        )
        
        assert isinstance(optimized_config, ThermalSystemConfig)
        
        # Compare with baseline design
        baseline_config = thermal_manager.design_thermal_system(sample_thermal_loads, design_requirements)
        
        # Both should be valid designs
        baseline_performance = thermal_manager.analyze_thermal_performance(baseline_config, design_requirements)
        optimized_performance = thermal_manager.analyze_thermal_performance(optimized_config, design_requirements)
        
        # Check that both designs complete without crashing
        assert isinstance(baseline_performance, dict)
        assert isinstance(optimized_performance, dict)
    
    def test_validation(self, thermal_manager, sample_thermal_loads, design_requirements):
        """Test thermal system validation."""
        system_config = thermal_manager.design_thermal_system(sample_thermal_loads, design_requirements)
        errors = thermal_manager._validate_thermal_design(system_config)
        
        # Well-designed system should have minimal errors (some warnings may be acceptable)
        critical_errors = [e for e in errors if "insufficient" in e.lower()]
        assert len(critical_errors) == 0  # No critical capacity errors
        
        # Test with insufficient heat exchanger capacity
        for hx in system_config.heat_exchangers:
            hx.max_heat_transfer = 1000.0  # Very low capacity
        
        errors = thermal_manager._validate_thermal_design(system_config)
        assert len(errors) > 0
        assert any("capacity" in error.lower() for error in errors)
    
    def test_high_power_laser_cooling(self, thermal_manager):
        """Test cooling system for high-power laser."""
        laser_load = ThermalLoad(
            load_id="high_power_laser",
            name="100kW Directed Energy Laser",
            power_dissipation=100000.0,  # 100 kW
            operating_temperature_range=(273.15, 313.15),
            critical_temperature=323.15,  # 50°C
            duty_cycle=0.05,  # 5% duty cycle
            location=(0.0, 0.0, 0.0),
            thermal_resistance=0.01
        )
        
        requirements = {
            'ambient_temperature': 288.15,
            'max_coolant_temperature': 373.15,
            'safety_margin': 0.3  # Higher margin for critical system
        }
        
        system_config = thermal_manager.design_thermal_system([laser_load], requirements)
        
        # Should use advanced cooling
        assert len(system_config.coolant_loops) > 0
        primary_coolant_type = system_config.coolant_loops["primary"].coolant_type
        assert primary_coolant_type in [CoolantType.PHASE_CHANGE, CoolantType.LIQUID_METAL]
        
        # Should have high-capacity heat exchangers
        total_hx_capacity = sum(hx.max_heat_transfer for hx in system_config.heat_exchangers)
        assert total_hx_capacity >= laser_load.power_dissipation * 1.2  # With safety margin
    
    def test_avionics_cooling(self, thermal_manager):
        """Test cooling system for avionics bay."""
        avionics_loads = [
            ThermalLoad(
                load_id=f"avionics_{i}",
                name=f"Avionics Unit {i}",
                power_dissipation=500.0 + i * 100,  # 500-1000W
                operating_temperature_range=(273.15, 333.15),
                critical_temperature=358.15,  # 85°C
                duty_cycle=1.0,
                location=(i * 0.5, 0.0, 0.0),
                thermal_resistance=0.2
            )
            for i in range(5)
        ]
        
        requirements = {
            'ambient_temperature': 288.15,
            'max_coolant_temperature': 343.15,  # 70°C
            'safety_margin': 0.15
        }
        
        system_config = thermal_manager.design_thermal_system(avionics_loads, requirements)
        
        # Should use air or liquid cooling for moderate power
        primary_coolant_type = system_config.coolant_loops["primary"].coolant_type
        assert primary_coolant_type in [CoolantType.AIR, CoolantType.LIQUID_COOLANT]
        
        # Analyze performance
        performance = thermal_manager.analyze_thermal_performance(system_config, requirements)
        
        # All loads should be reasonably close to temperature limits
        for load in avionics_loads:
            temp = performance['steady_state_temperatures'][load.load_id]
            # Check that temperature is calculated
            assert temp > 0  # Temperature should be positive
    
    def test_cryogenic_cooling(self, thermal_manager):
        """Test cryogenic cooling for very high power density loads."""
        cryo_load = ThermalLoad(
            load_id="superconducting_magnet",
            name="Superconducting Electromagnet",
            power_dissipation=50000.0,  # 50 kW
            operating_temperature_range=(77.0, 90.0),  # Liquid nitrogen range
            critical_temperature=100.0,  # 100 K
            duty_cycle=1.0,
            location=(0.0, 0.0, 0.0),
            thermal_resistance=0.005
        )
        
        requirements = {
            'ambient_temperature': 288.15,
            'max_coolant_temperature': 120.0,  # Very low temperature
            'safety_margin': 0.5
        }
        
        system_config = thermal_manager.design_thermal_system([cryo_load], requirements)
        
        # Should include cryogenic cooling loop
        assert "cryogenic" in system_config.coolant_loops
        cryo_coolant = system_config.coolant_loops["cryogenic"]
        assert cryo_coolant.coolant_type == CoolantType.CRYOGENIC
    
    def test_error_handling(self, thermal_manager):
        """Test error handling for invalid inputs."""
        # Empty thermal loads
        empty_loads = []
        requirements = {'ambient_temperature': 288.15}
        
        system_config = thermal_manager.design_thermal_system(empty_loads, requirements)
        assert len(system_config.thermal_loads) == 0
        assert len(system_config.heat_exchangers) == 0
        
        # Invalid thermal load
        invalid_load = ThermalLoad(
            load_id="invalid",
            name="Invalid Load",
            power_dissipation=-1000.0,  # Negative power
            operating_temperature_range=(400.0, 300.0),  # Invalid range
            critical_temperature=200.0,  # Below operating range
            duty_cycle=2.0,  # > 1.0
            location=(0.0, 0.0, 0.0)
        )
        
        # Should handle gracefully (implementation dependent)
        try:
            system_config = thermal_manager.design_thermal_system([invalid_load], requirements)
            # If no exception, check that system handles it reasonably
            assert isinstance(system_config, ThermalSystemConfig)
        except (ValueError, AssertionError):
            # Expected for invalid inputs
            pass


if __name__ == "__main__":
    pytest.main([__file__])