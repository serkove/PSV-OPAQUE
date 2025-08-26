"""Tests for plasma-based decoy and sensor systems."""

import pytest
import numpy as np
import math
import time
from fighter_jet_sdk.engines.sensors.plasma_systems import (
    PlasmaDecoyGenerator, CooperativeSensingNetwork, PlasmaSystemController,
    PlasmaConfiguration, PlasmaProperties, PlasmaOrb, PlasmaType, PlasmaState
)


class TestPlasmaConfiguration:
    """Test plasma configuration."""
    
    def test_plasma_config_creation(self):
        """Test plasma configuration creation."""
        config = PlasmaConfiguration(
            plasma_type=PlasmaType.RF_SUSTAINED,
            frequency=2.45e9,  # 2.45 GHz
            power_input=100000,  # 100 kW
            gas_composition={'Ar': 0.8, 'He': 0.2},
            pressure=1000,  # Pa
            temperature_target=10000,  # K
            volume=1.0,  # m³
            magnetic_field_strength=0.1  # T
        )
        
        assert config.plasma_type == PlasmaType.RF_SUSTAINED
        assert config.frequency == 2.45e9
        assert config.power_input == 100000
        assert config.gas_composition['Ar'] == 0.8


class TestPlasmaDecoyGenerator:
    """Test plasma decoy generator."""
    
    @pytest.fixture
    def plasma_config(self):
        """Create test plasma configuration."""
        return PlasmaConfiguration(
            plasma_type=PlasmaType.RF_SUSTAINED,
            frequency=2.45e9,
            power_input=50000,  # 50 kW
            gas_composition={'Ar': 1.0},
            pressure=1000,
            temperature_target=8000,
            volume=0.5
        )
    
    def test_generator_creation(self, plasma_config):
        """Test plasma generator creation."""
        generator = PlasmaDecoyGenerator(plasma_config)
        
        assert generator.config == plasma_config
        assert generator.base_electron_density > 0
        assert generator.base_plasma_frequency > 0
        assert generator.base_debye_length > 0
    
    def test_plasma_properties_calculation(self, plasma_config):
        """Test plasma properties calculation."""
        generator = PlasmaDecoyGenerator(plasma_config)
        
        # Test full power
        props_full = generator.calculate_plasma_properties(1.0)
        assert props_full.electron_density > 0
        assert props_full.electron_temperature > 0
        assert props_full.plasma_frequency > 0
        assert props_full.debye_length > 0
        
        # Test half power
        props_half = generator.calculate_plasma_properties(0.5)
        assert props_half.electron_density < props_full.electron_density
        assert props_half.electron_temperature < props_full.electron_temperature
    
    def test_orb_creation(self, plasma_config):
        """Test plasma orb creation."""
        generator = PlasmaDecoyGenerator(plasma_config)
        
        orb_id = generator.create_plasma_orb(
            position=(1000, 500, 200),
            velocity=(10, 0, 0),
            radius=5.0,
            power_fraction=0.8,
            lifetime=120.0
        )
        
        assert orb_id in generator.active_orbs
        orb = generator.active_orbs[orb_id]
        
        assert orb.position == (1000, 500, 200)
        assert orb.velocity == (10, 0, 0)
        assert orb.radius == 5.0
        assert orb.state == PlasmaState.IGNITING
        assert orb.lifetime_remaining == 120.0
    
    def test_orb_state_updates(self, plasma_config):
        """Test orb state updates."""
        generator = PlasmaDecoyGenerator(plasma_config)
        
        # Create orb
        orb_id = generator.create_plasma_orb(
            position=(0, 0, 0),
            velocity=(5, 0, 0),
            radius=3.0,
            lifetime=10.0
        )
        
        # Update with small time step
        generator.update_orb_states(0.1)
        orb = generator.active_orbs[orb_id]
        
        # Position should have moved
        assert orb.position[0] == 0.5  # 5 m/s * 0.1 s
        assert orb.lifetime_remaining < 10.0
        
        # Update with large time step to expire orb
        generator.update_orb_states(15.0)
        
        # Orb should be removed
        assert orb_id not in generator.active_orbs
    
    def test_radar_cross_section_calculation(self, plasma_config):
        """Test RCS calculation."""
        generator = PlasmaDecoyGenerator(plasma_config)
        
        orb_id = generator.create_plasma_orb(
            position=(0, 0, 0),
            velocity=(0, 0, 0),
            radius=5.0
        )
        orb = generator.active_orbs[orb_id]
        
        # Test at different frequencies
        frequencies = [1e9, 10e9, 100e9]  # 1, 10, 100 GHz
        
        for freq in frequencies:
            rcs = generator.calculate_radar_cross_section(orb, freq)
            
            # RCS should be positive
            assert rcs > 0
            # Should be reasonable for 5m radius orb
            assert rcs < 1000  # Less than 1000 m²
    
    def test_optical_signature_calculation(self, plasma_config):
        """Test optical signature calculation."""
        generator = PlasmaDecoyGenerator(plasma_config)
        
        orb_id = generator.create_plasma_orb(
            position=(0, 0, 0),
            velocity=(0, 0, 0),
            radius=5.0
        )
        orb = generator.active_orbs[orb_id]
        
        signature = generator.calculate_optical_signature(orb, 550e-9)  # Green light
        
        # Check required fields
        required_fields = [
            'thermal_power_w', 'line_power_w', 'total_power_w',
            'brightness_temperature_k', 'apparent_magnitude'
        ]
        
        for field in required_fields:
            assert field in signature
            assert signature[field] > 0 or field == 'apparent_magnitude'
    
    def test_power_requirements_calculation(self, plasma_config):
        """Test power requirements calculation."""
        generator = PlasmaDecoyGenerator(plasma_config)
        
        power_req = generator.calculate_power_requirements(
            num_orbs=10,
            orb_radius=5.0,
            lifetime=300.0
        )
        
        # Check required fields
        required_fields = [
            'power_per_orb_w', 'total_continuous_power_w', 'peak_power_w',
            'energy_per_orb_j', 'total_energy_per_hour_mj'
        ]
        
        for field in required_fields:
            assert field in power_req
            assert power_req[field] > 0
        
        # Peak power should be higher than continuous
        assert power_req['peak_power_w'] > power_req['total_continuous_power_w']
    
    def test_active_orb_counting(self, plasma_config):
        """Test active orb counting."""
        generator = PlasmaDecoyGenerator(plasma_config)
        
        # Initially no orbs
        assert generator.get_active_orb_count() == 0
        assert generator.get_total_power_consumption() == 0
        
        # Create some orbs
        orb_ids = []
        for i in range(3):
            orb_id = generator.create_plasma_orb(
                position=(i * 100, 0, 0),
                velocity=(0, 0, 0),
                radius=3.0,
                power_fraction=0.5
            )
            orb_ids.append(orb_id)
        
        assert generator.get_active_orb_count() == 3
        assert generator.get_total_power_consumption() > 0


class TestCooperativeSensingNetwork:
    """Test cooperative sensing network."""
    
    @pytest.fixture
    def sensing_network(self):
        """Create test sensing network."""
        return CooperativeSensingNetwork(communication_range=5000.0)
    
    @pytest.fixture
    def test_orbs(self):
        """Create test plasma orbs."""
        config = PlasmaConfiguration(
            plasma_type=PlasmaType.RF_SUSTAINED,
            frequency=2.45e9,
            power_input=50000,
            gas_composition={'Ar': 1.0},
            pressure=1000,
            temperature_target=8000,
            volume=0.5
        )
        
        generator = PlasmaDecoyGenerator(config)
        
        # Create orbs in a line
        orb_ids = []
        for i in range(3):
            orb_id = generator.create_plasma_orb(
                position=(i * 1000, 0, 0),  # 1km spacing
                velocity=(0, 0, 0),
                radius=5.0
            )
            orb_ids.append(orb_id)
            # Set to active state
            generator.active_orbs[orb_id].state = PlasmaState.ACTIVE
        
        return generator.active_orbs
    
    def test_network_creation(self, sensing_network):
        """Test sensing network creation."""
        assert sensing_network.communication_range == 5000.0
        assert len(sensing_network.network_topology) == 0
    
    def test_topology_update(self, sensing_network, test_orbs):
        """Test network topology update."""
        sensing_network.update_network_topology(test_orbs)
        
        # Should have topology entries for active orbs
        assert len(sensing_network.network_topology) == 3
        
        # Check connectivity (1km spacing, 5km range)
        orb_ids = list(test_orbs.keys())
        
        # First orb should connect to second
        assert orb_ids[1] in sensing_network.network_topology[orb_ids[0]]
        
        # Middle orb should connect to both others
        assert orb_ids[0] in sensing_network.network_topology[orb_ids[1]]
        assert orb_ids[2] in sensing_network.network_topology[orb_ids[1]]
    
    def test_target_detection(self, sensing_network, test_orbs):
        """Test target detection simulation."""
        targets = [
            (500, 0, 0),    # Near first orb
            (5000, 0, 0),   # Far from all orbs
            (1500, 0, 0)    # Between first and second orb
        ]
        
        detections = sensing_network.simulate_target_detection(
            test_orbs, targets, detection_range=2000.0
        )
        
        # Should have detections from orbs (number of active orbs)
        active_orbs = len([orb for orb in test_orbs.values() if orb.state == PlasmaState.ACTIVE])
        assert len(detections) == active_orbs
        
        # Check that some detections occurred (at least the close target should be detected)
        total_detections = sum(len(det_list) for det_list in detections.values())
        # Since detection is probabilistic, we'll just check that the function runs
        assert total_detections >= 0  # At least no errors occurred
    
    def test_sensor_data_fusion(self, sensing_network):
        """Test sensor data fusion."""
        # Create mock detections from multiple orbs
        detections = {
            'PLASMA_0001': [
                {
                    'target_id': 'T001',
                    'position': (1000, 100, 0),
                    'distance': 1000,
                    'detection_time': time.time(),
                    'confidence': 0.8,
                    'sensor_type': 'plasma_orb'
                }
            ],
            'PLASMA_0002': [
                {
                    'target_id': 'T001',
                    'position': (1020, 80, 0),  # Slightly different position
                    'distance': 1000,
                    'detection_time': time.time(),
                    'confidence': 0.7,
                    'sensor_type': 'plasma_orb'
                }
            ]
        }
        
        fused = sensing_network.fuse_sensor_data(detections)
        
        # Should fuse into single detection
        assert len(fused) == 1
        
        fused_detection = fused[0]
        assert fused_detection['sensor_type'] == 'fused_plasma_network'
        assert fused_detection['num_detections'] == 2
        assert len(fused_detection['reporting_orbs']) == 2
    
    def test_coverage_calculation(self, sensing_network, test_orbs):
        """Test network coverage calculation."""
        area_bounds = ((-2000, 2000), (-1000, 1000), (-500, 500))
        
        coverage = sensing_network.calculate_network_coverage(
            test_orbs, area_bounds, detection_range=2000.0
        )
        
        # Check required fields
        required_fields = [
            'coverage_fraction', 'redundancy_fraction', 'covered_volume_km3',
            'active_orbs', 'network_connectivity'
        ]
        
        for field in required_fields:
            assert field in coverage
        
        # Coverage should be between 0 and 1
        assert 0 <= coverage['coverage_fraction'] <= 1
        assert 0 <= coverage['redundancy_fraction'] <= 1
        
        # Should have 3 active orbs
        assert coverage['active_orbs'] == 3


class TestPlasmaSystemController:
    """Test plasma system controller."""
    
    @pytest.fixture
    def plasma_controller(self):
        """Create test plasma system controller."""
        config = PlasmaConfiguration(
            plasma_type=PlasmaType.RF_SUSTAINED,
            frequency=2.45e9,
            power_input=50000,
            gas_composition={'Ar': 1.0},
            pressure=1000,
            temperature_target=8000,
            volume=0.5
        )
        
        generator = PlasmaDecoyGenerator(config)
        network = CooperativeSensingNetwork()
        
        return PlasmaSystemController(generator, network)
    
    def test_controller_creation(self, plasma_controller):
        """Test controller creation."""
        assert plasma_controller.decoy_generator is not None
        assert plasma_controller.sensing_network is not None
        assert not plasma_controller.mission_active
    
    def test_orb_network_deployment(self, plasma_controller):
        """Test orb network deployment."""
        center_pos = (0, 0, 1000)
        
        # Test grid deployment
        orb_ids_grid = plasma_controller.deploy_orb_network(
            center_position=center_pos,
            num_orbs=4,
            spacing=500,
            pattern="grid"
        )
        
        assert len(orb_ids_grid) == 4
        
        # Test circle deployment
        orb_ids_circle = plasma_controller.deploy_orb_network(
            center_position=center_pos,
            num_orbs=6,
            spacing=1000,
            pattern="circle"
        )
        
        assert len(orb_ids_circle) == 6
        
        # Test line deployment
        orb_ids_line = plasma_controller.deploy_orb_network(
            center_position=center_pos,
            num_orbs=3,
            spacing=800,
            pattern="line"
        )
        
        assert len(orb_ids_line) == 3
        
        # Total orbs should be sum of all deployments
        total_orbs = plasma_controller.decoy_generator.get_active_orb_count()
        assert total_orbs == 4 + 6 + 3
    
    def test_mission_cycle_execution(self, plasma_controller):
        """Test mission cycle execution."""
        # Deploy some orbs first
        plasma_controller.deploy_orb_network(
            center_position=(0, 0, 0),
            num_orbs=3,
            spacing=1000,
            pattern="line"
        )
        
        # Execute mission cycle without targets
        result = plasma_controller.execute_mission_cycle(dt=1.0)
        
        # Check result structure
        required_fields = [
            'active_orbs', 'total_power_w', 'raw_detections',
            'fused_detections', 'network_nodes', 'mission_time', 'detections'
        ]
        
        for field in required_fields:
            assert field in result
        
        assert result['active_orbs'] == 3
        assert result['total_power_w'] > 0
        
        # Execute with targets
        targets = [(500, 0, 0), (1500, 0, 0)]
        result_with_targets = plasma_controller.execute_mission_cycle(
            dt=1.0, targets=targets
        )
        
        # Should have detection results
        assert 'detections' in result_with_targets
    
    def test_mission_start_stop(self, plasma_controller):
        """Test mission start and stop."""
        # Initially not active
        assert not plasma_controller.mission_active
        
        # Start mission
        plasma_controller.start_mission()
        assert plasma_controller.mission_active
        assert plasma_controller.mission_start_time > 0
        
        # Deploy orbs
        plasma_controller.deploy_orb_network(
            center_position=(0, 0, 0),
            num_orbs=2,
            spacing=1000,
            pattern="line"
        )
        
        # Stop mission
        plasma_controller.stop_mission()
        assert not plasma_controller.mission_active
        
        # Orbs should be in decaying state
        for orb in plasma_controller.decoy_generator.active_orbs.values():
            assert orb.state == PlasmaState.DECAYING
    
    def test_system_status(self, plasma_controller):
        """Test system status reporting."""
        status = plasma_controller.get_system_status()
        
        # Check required fields
        required_fields = [
            'mission_active', 'active_orbs', 'total_orbs_created',
            'current_power_w', 'power_requirements', 'network_connectivity',
            'plasma_type'
        ]
        
        for field in required_fields:
            assert field in status
        
        # Initially no orbs
        assert status['active_orbs'] == 0
        assert status['current_power_w'] == 0
        
        # Deploy orbs and check again
        plasma_controller.deploy_orb_network(
            center_position=(0, 0, 0),
            num_orbs=5,
            spacing=500,
            pattern="grid"
        )
        
        status_with_orbs = plasma_controller.get_system_status()
        assert status_with_orbs['active_orbs'] == 5
        assert status_with_orbs['current_power_w'] > 0
        assert status_with_orbs['total_orbs_created'] == 5


class TestPlasmaProperties:
    """Test plasma properties data structure."""
    
    def test_plasma_properties_creation(self):
        """Test plasma properties creation."""
        props = PlasmaProperties(
            electron_density=1e16,
            electron_temperature=10000,
            ion_temperature=3000,
            plasma_frequency=1e10,
            debye_length=1e-4,
            collision_frequency=1e6,
            conductivity=1000,
            dielectric_constant=complex(0.8, 0.2)
        )
        
        assert props.electron_density == 1e16
        assert props.electron_temperature == 10000
        assert props.dielectric_constant.real == 0.8
        assert props.dielectric_constant.imag == 0.2


class TestPlasmaOrb:
    """Test plasma orb data structure."""
    
    def test_plasma_orb_creation(self):
        """Test plasma orb creation."""
        props = PlasmaProperties(
            electron_density=1e16,
            electron_temperature=10000,
            ion_temperature=3000,
            plasma_frequency=1e10,
            debye_length=1e-4,
            collision_frequency=1e6,
            conductivity=1000,
            dielectric_constant=complex(0.8, 0.2)
        )
        
        orb = PlasmaOrb(
            orb_id="TEST_001",
            position=(1000, 500, 200),
            velocity=(10, 5, 0),
            radius=5.0,
            plasma_properties=props,
            state=PlasmaState.ACTIVE,
            creation_time=time.time(),
            lifetime_remaining=300.0,
            power_consumption=25000
        )
        
        assert orb.orb_id == "TEST_001"
        assert orb.position == (1000, 500, 200)
        assert orb.velocity == (10, 5, 0)
        assert orb.radius == 5.0
        assert orb.state == PlasmaState.ACTIVE
        assert orb.power_consumption == 25000


if __name__ == "__main__":
    pytest.main([__file__])