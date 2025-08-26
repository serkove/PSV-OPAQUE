"""
Tests for UAV Swarm Modeling and Simulation

This module tests the UAV swarm simulation capabilities including
autonomous navigation, cooperative sensing, and swarm coordination.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from fighter_jet_sdk.engines.deployable.uav_swarm import (
    UAVSwarmSimulator, UAVSpecifications, SwarmMission, UAVState,
    AutonomousNavigator, CooperativeSensing, SwarmCoordinator,
    FormationController, TaskAllocator, MissionPlanner,
    NavigationMode, SwarmFormation
)
from fighter_jet_sdk.common.data_models import Position3D, Velocity3D


class TestUAVSpecifications:
    """Test UAV specifications data structure"""
    
    def test_uav_specifications_creation(self):
        """Test creating UAV specifications"""
        specs = UAVSpecifications(
            max_speed=50.0,
            max_acceleration=10.0,
            max_range=100.0,
            sensor_range=5.0,
            communication_range=10.0,
            endurance=4.0,
            payload_capacity=5.0,
            stealth_signature=0.01
        )
        
        assert specs.max_speed == 50.0
        assert specs.sensor_range == 5.0
        assert specs.stealth_signature == 0.01


class TestSwarmMission:
    """Test swarm mission data structure"""
    
    def test_swarm_mission_creation(self):
        """Test creating swarm mission"""
        mission = SwarmMission(
            mission_type="reconnaissance",
            target_area=(40.0, 41.0, -74.0, -73.0),
            altitude_range=(200, 800),
            search_pattern="raster",
            duration=2.0,
            priority_targets=["target1", "target2"]
        )
        
        assert mission.mission_type == "reconnaissance"
        assert mission.target_area == (40.0, 41.0, -74.0, -73.0)
        assert len(mission.priority_targets) == 2


class TestAutonomousNavigator:
    """Test autonomous navigation system"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.uav_specs = UAVSpecifications(
            max_speed=50.0, max_acceleration=10.0, max_range=100.0,
            sensor_range=5.0, communication_range=10.0, endurance=4.0,
            payload_capacity=5.0, stealth_signature=0.01
        )
        self.navigator = AutonomousNavigator(self.uav_specs)
    
    def test_navigator_initialization(self):
        """Test navigator initialization"""
        assert self.navigator.uav_specs == self.uav_specs
        assert self.navigator.navigation_accuracy == 0.95
        assert self.navigator.terrain_map is None
    
    def test_vision_aided_ins_initialization(self):
        """Test vision-aided INS initialization"""
        terrain_data = {"elevation_map": np.random.rand(100, 100)}
        
        result = self.navigator.initialize_vision_aided_ins(terrain_data)
        
        assert result is True
        assert self.navigator.terrain_map == terrain_data
        assert len(self.navigator.landmark_database) == 10
    
    def test_gps_navigation(self):
        """Test GPS navigation mode"""
        uav_state = UAVState(
            uav_id="test_uav",
            position=Position3D(0, 0, 300),
            velocity=Velocity3D(10, 0, 0),
            heading=0.0,
            fuel_remaining=100.0,
            sensor_status={'gps': True},
            communication_status=True,
            mission_status='active'
        )
        
        sensor_data = {
            'gps_accuracy': 0.9,
            'gps_lat': 40.7128,
            'gps_lon': -74.0060,
            'gps_alt': 300
        }
        
        position, accuracy = self.navigator.calculate_navigation_solution(
            uav_state, sensor_data, NavigationMode.GPS
        )
        
        assert accuracy == 0.9
        assert position.x == 40.7128
        assert position.y == -74.0060
    
    def test_vision_aided_navigation(self):
        """Test vision-aided navigation"""
        # Initialize with terrain data
        terrain_data = {"elevation_map": np.random.rand(100, 100)}
        self.navigator.initialize_vision_aided_ins(terrain_data)
        
        uav_state = UAVState(
            uav_id="test_uav",
            position=Position3D(0, 0, 300),
            velocity=Velocity3D(10, 0, 0),
            heading=0.0,
            fuel_remaining=100.0,
            sensor_status={'visual': True},
            communication_status=True,
            mission_status='active'
        )
        
        sensor_data = {'gps_accuracy': 0.1}  # Poor GPS
        
        position, accuracy = self.navigator.calculate_navigation_solution(
            uav_state, sensor_data, NavigationMode.VISION_AIDED_INS
        )
        
        assert 0.5 <= accuracy <= 0.9
        assert isinstance(position, Position3D)
    
    def test_cooperative_navigation(self):
        """Test cooperative navigation with other UAVs"""
        uav_state = UAVState(
            uav_id="test_uav",
            position=Position3D(0, 0, 300),
            velocity=Velocity3D(10, 0, 0),
            heading=0.0,
            fuel_remaining=100.0,
            sensor_status={'communication': True},
            communication_status=True,
            mission_status='active'
        )
        
        sensor_data = {
            'nearby_uavs': ['UAV_001', 'UAV_002'],
            'gps_accuracy': 0.1
        }
        
        position, accuracy = self.navigator.calculate_navigation_solution(
            uav_state, sensor_data, NavigationMode.COOPERATIVE
        )
        
        assert accuracy == 0.8
        assert isinstance(position, Position3D)


class TestCooperativeSensing:
    """Test cooperative sensing and communication"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.cooperative_sensing = CooperativeSensing(communication_range=1000.0)
    
    def test_communication_network_establishment(self):
        """Test establishing communication network"""
        uav_states = [
            UAVState("UAV_001", Position3D(0, 0, 300), Velocity3D(0, 0, 0), 
                    0.0, 100.0, {}, True, 'active'),
            UAVState("UAV_002", Position3D(500, 0, 300), Velocity3D(0, 0, 0), 
                    0.0, 100.0, {}, True, 'active'),
            UAVState("UAV_003", Position3D(2000, 0, 300), Velocity3D(0, 0, 0), 
                    0.0, 100.0, {}, True, 'active')
        ]
        
        network = self.cooperative_sensing.establish_communication_network(uav_states)
        
        assert "UAV_001" in network
        assert "UAV_002" in network["UAV_001"]  # Within range
        assert "UAV_003" not in network["UAV_001"]  # Out of range
    
    def test_sensor_data_fusion(self):
        """Test sensor data fusion from multiple UAVs"""
        sensor_reports = [
            {
                'uav_id': 'UAV_001',
                'detected_targets': [
                    {
                        'position': (100, 200, 50),
                        'confidence': 0.8,
                        'sensor_type': 'visual',
                        'timestamp': 1.0
                    }
                ]
            },
            {
                'uav_id': 'UAV_002',
                'detected_targets': [
                    {
                        'position': (105, 195, 45),  # Same target, slightly different position
                        'confidence': 0.7,
                        'sensor_type': 'radar',
                        'timestamp': 1.1
                    }
                ]
            }
        ]
        
        fused_targets = self.cooperative_sensing.fuse_sensor_data(sensor_reports)
        
        assert len(fused_targets) == 1  # Should fuse into one target
        target_id = list(fused_targets.keys())[0]
        target = fused_targets[target_id]
        
        assert target['detection_count'] == 2
        assert 'visual' in target['sensor_types']
        assert 'radar' in target['sensor_types']
    
    def test_confidence_update(self):
        """Test Bayesian confidence update"""
        existing_conf = 0.7
        new_conf = 0.8
        
        updated_conf = self.cooperative_sensing._update_confidence(existing_conf, new_conf)
        
        assert 0.7 < updated_conf < 1.0  # Should be higher than both inputs


class TestFormationController:
    """Test swarm formation control"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.formation_controller = FormationController()
        self.uav_states = [
            UAVState(f"UAV_{i:03d}", Position3D(0, 0, 300), Velocity3D(0, 0, 0),
                    0.0, 100.0, {}, True, 'active')
            for i in range(4)
        ]
    
    def test_line_abreast_formation(self):
        """Test line abreast formation calculation"""
        center = Position3D(0, 0, 300)
        
        positions = self.formation_controller.calculate_formation_positions(
            self.uav_states, SwarmFormation.LINE_ABREAST, center
        )
        
        assert len(positions) == 4
        # Check that UAVs are spread along x-axis
        x_positions = [positions[uav.uav_id].x for uav in self.uav_states]
        assert len(set(x_positions)) == 4  # All different x positions
    
    def test_wedge_formation(self):
        """Test wedge formation calculation"""
        center = Position3D(0, 0, 300)
        
        positions = self.formation_controller.calculate_formation_positions(
            self.uav_states, SwarmFormation.WEDGE, center
        )
        
        assert len(positions) == 4
        # Lead UAV should be at center
        lead_uav = self.uav_states[0]
        assert positions[lead_uav.uav_id] == center
    
    def test_diamond_formation(self):
        """Test diamond formation calculation"""
        center = Position3D(0, 0, 300)
        
        positions = self.formation_controller.calculate_formation_positions(
            self.uav_states, SwarmFormation.DIAMOND, center
        )
        
        assert len(positions) == 4
        # Check that positions form a diamond pattern
        for uav in self.uav_states:
            pos = positions[uav.uav_id]
            assert pos.z == center.z  # Same altitude


class TestTaskAllocator:
    """Test task allocation algorithms"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.task_allocator = TaskAllocator()
        self.uav_states = [
            UAVState(f"UAV_{i:03d}", Position3D(0, 0, 300), Velocity3D(0, 0, 0),
                    0.0, 100.0, {}, True, 'active')
            for i in range(3)
        ]
        self.mission = SwarmMission(
            mission_type="reconnaissance",
            target_area=(40.0, 41.0, -74.0, -73.0),
            altitude_range=(200, 800),
            search_pattern="raster",
            duration=2.0
        )
    
    def test_reconnaissance_task_allocation(self):
        """Test reconnaissance task allocation"""
        tasks = self.task_allocator.allocate_tasks(self.uav_states, self.mission)
        
        assert len(tasks) == 3
        for uav in self.uav_states:
            assert uav.uav_id in tasks
            task = tasks[uav.uav_id]
            assert task['task_type'] == 'sector_reconnaissance'
            assert 'assigned_sector' in task
            assert 'waypoints' in task
    
    def test_area_division_into_sectors(self):
        """Test dividing target area into sectors"""
        area = (40.0, 41.0, -74.0, -73.0)
        sectors = self.task_allocator._divide_area_into_sectors(area, 4)
        
        assert len(sectors) == 4
        for sector in sectors:
            assert 'id' in sector
            assert 'bounds' in sector
            assert 'area_km2' in sector
    
    def test_search_waypoint_generation(self):
        """Test search waypoint generation"""
        sector = {
            'id': 'test_sector',
            'bounds': (40.0, 40.5, -74.0, -73.5),
            'area_km2': 100.0
        }
        
        waypoints = self.task_allocator._generate_search_waypoints(sector, "raster")
        
        assert len(waypoints) > 0
        assert all(isinstance(wp, Position3D) for wp in waypoints)
    
    def test_task_reallocation_after_failure(self):
        """Test task reallocation when UAV fails"""
        failed_uav_id = "UAV_001"
        # Pass all UAVs, the reallocate method should filter out the failed one
        remaining_uavs = self.uav_states
        
        new_tasks = self.task_allocator.reallocate_after_failure(
            failed_uav_id, remaining_uavs, self.mission
        )
        
        assert failed_uav_id not in new_tasks
        # Should have tasks for 2 UAVs (3 total - 1 failed)
        assert len(new_tasks) == 2


class TestMissionPlanner:
    """Test mission planning capabilities"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mission_planner = MissionPlanner()
        self.uav_specs = UAVSpecifications(
            max_speed=50.0, max_acceleration=10.0, max_range=100.0,
            sensor_range=5.0, communication_range=10.0, endurance=4.0,
            payload_capacity=5.0, stealth_signature=0.01
        )
    
    def test_reconnaissance_mission_planning(self):
        """Test reconnaissance mission planning"""
        target_area = (40.0, 41.0, -74.0, -73.0)
        swarm_size = 4
        mission_requirements = {
            'priority_targets': ['target1', 'target2'],
            'high_resolution': True
        }
        
        mission = self.mission_planner.plan_reconnaissance_mission(
            target_area, self.uav_specs, swarm_size, mission_requirements
        )
        
        assert mission.mission_type == "reconnaissance"
        assert mission.target_area == target_area
        assert mission.duration > 0
        assert len(mission.priority_targets) == 2
    
    def test_mission_duration_calculation(self):
        """Test mission duration calculation"""
        target_area = (40.0, 41.0, -74.0, -73.0)
        swarm_size = 4
        
        duration = self.mission_planner._calculate_mission_duration(
            target_area, self.uav_specs, swarm_size
        )
        
        assert duration > 0
        assert isinstance(duration, float)
    
    def test_search_pattern_selection(self):
        """Test optimal search pattern selection"""
        # Elongated area should prefer raster
        elongated_area = (40.0, 40.1, -74.0, -72.0)
        pattern = self.mission_planner._select_optimal_search_pattern(
            elongated_area, {}
        )
        assert pattern == "raster"
        
        # Square area should prefer spiral
        square_area = (40.0, 41.0, -74.0, -73.0)
        pattern = self.mission_planner._select_optimal_search_pattern(
            square_area, {}
        )
        assert pattern == "spiral"


class TestUAVSwarmSimulator:
    """Test main UAV swarm simulator"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.uav_specs = UAVSpecifications(
            max_speed=50.0, max_acceleration=10.0, max_range=100.0,
            sensor_range=5.0, communication_range=10.0, endurance=4.0,
            payload_capacity=5.0, stealth_signature=0.01
        )
        self.simulator = UAVSwarmSimulator(swarm_size=4, uav_specs=self.uav_specs)
    
    def test_simulator_initialization(self):
        """Test simulator initialization"""
        assert self.simulator.swarm_size == 4
        assert len(self.simulator.uav_states) == 4
        assert self.simulator.simulation_time == 0.0
        
        # Check UAV states are properly initialized
        for i, uav in enumerate(self.simulator.uav_states):
            assert uav.uav_id == f"UAV_{i:03d}"
            assert uav.mission_status == 'ready'
            assert uav.fuel_remaining == 100.0
    
    def test_mission_start(self):
        """Test starting a mission"""
        mission = SwarmMission(
            mission_type="reconnaissance",
            target_area=(40.0, 41.0, -74.0, -73.0),
            altitude_range=(200, 800),
            search_pattern="raster",
            duration=2.0
        )
        
        result = self.simulator.start_mission(mission)
        
        assert result is True
        assert self.simulator.current_mission == mission
        # Check that UAVs are activated
        active_uavs = sum(1 for uav in self.simulator.uav_states 
                         if uav.mission_status == 'active')
        assert active_uavs > 0
    
    def test_simulation_update(self):
        """Test simulation state update"""
        # Start a mission first
        mission = SwarmMission(
            mission_type="reconnaissance",
            target_area=(40.0, 41.0, -74.0, -73.0),
            altitude_range=(200, 800),
            search_pattern="raster",
            duration=2.0
        )
        self.simulator.start_mission(mission)
        
        # Update simulation
        dt = 1.0  # 1 second
        result = self.simulator.update_simulation(dt)
        
        assert 'simulation_time' in result
        assert result['simulation_time'] == dt
        assert 'uav_states' in result
        assert len(result['uav_states']) == 4
        assert 'communication_network' in result
        assert 'detected_targets' in result
        assert 'mission_progress' in result
    
    def test_swarm_status(self):
        """Test getting swarm status"""
        status = self.simulator.get_swarm_status()
        
        assert 'total_uavs' in status
        assert status['total_uavs'] == 4
        assert 'active_uavs' in status
        assert 'failed_uavs' in status
        assert 'mission_progress' in status
        assert 'simulation_time' in status
    
    def test_uav_state_update(self):
        """Test individual UAV state update"""
        uav = self.simulator.uav_states[0]
        initial_fuel = uav.fuel_remaining
        
        # Update UAV state
        dt = 3600.0  # 1 hour
        self.simulator._update_uav_state(uav, dt)
        
        # Fuel should have decreased
        assert uav.fuel_remaining < initial_fuel
    
    def test_sensor_data_generation(self):
        """Test sensor data generation"""
        uav = self.simulator.uav_states[0]
        sensor_data = self.simulator._generate_sensor_data(uav)
        
        assert 'gps_accuracy' in sensor_data
        assert 'gps_lat' in sensor_data
        assert 'gps_lon' in sensor_data
        assert 'radar_altitude' in sensor_data
        assert 'nearby_uavs' in sensor_data
    
    def test_mission_progress_calculation(self):
        """Test mission progress calculation"""
        # Without mission
        progress = self.simulator._calculate_mission_progress()
        assert progress == 0.0
        
        # With mission
        mission = SwarmMission(
            mission_type="reconnaissance",
            target_area=(40.0, 41.0, -74.0, -73.0),
            altitude_range=(200, 800),
            search_pattern="raster",
            duration=2.0
        )
        self.simulator.start_mission(mission)
        
        progress = self.simulator._calculate_mission_progress()
        assert 0.0 <= progress <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])