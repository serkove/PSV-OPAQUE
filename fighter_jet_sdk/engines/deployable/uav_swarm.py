"""
UAV Swarm Modeling and Simulation

This module implements small UAV swarm modeling with autonomous navigation
for GPS-denied environments, cooperative sensing protocols, and swarm
coordination algorithms for reconnaissance missions.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from fighter_jet_sdk.common.data_models import Position3D, Velocity3D
from fighter_jet_sdk.common.interfaces import SimulationComponent

logger = logging.getLogger(__name__)


class NavigationMode(Enum):
    """Navigation modes for UAV operation"""
    GPS = "gps"
    VISION_AIDED_INS = "vision_aided_ins"
    COOPERATIVE = "cooperative"
    TERRAIN_FOLLOWING = "terrain_following"


class SwarmFormation(Enum):
    """Swarm formation patterns"""
    LINE_ABREAST = "line_abreast"
    WEDGE = "wedge"
    DIAMOND = "diamond"
    SEARCH_GRID = "search_grid"
    ADAPTIVE = "adaptive"


@dataclass
class UAVSpecifications:
    """Specifications for individual UAV"""
    max_speed: float  # m/s
    max_acceleration: float  # m/s²
    max_range: float  # km
    sensor_range: float  # km
    communication_range: float  # km
    endurance: float  # hours
    payload_capacity: float  # kg
    stealth_signature: float  # RCS in m²


@dataclass
class SwarmMission:
    """Mission parameters for UAV swarm"""
    mission_type: str
    target_area: Tuple[float, float, float, float]  # lat_min, lat_max, lon_min, lon_max
    altitude_range: Tuple[float, float]  # min_alt, max_alt in meters
    search_pattern: str
    duration: float  # hours
    priority_targets: List[str] = field(default_factory=list)
    threat_zones: List[Dict] = field(default_factory=list)


@dataclass
class UAVState:
    """Current state of individual UAV"""
    uav_id: str
    position: Position3D
    velocity: Velocity3D
    heading: float  # radians
    fuel_remaining: float  # percentage
    sensor_status: Dict[str, bool]
    communication_status: bool
    mission_status: str
    detected_targets: List[Dict] = field(default_factory=list)


class AutonomousNavigator:
    """Autonomous navigation system for GPS-denied environments"""
    
    def __init__(self, uav_specs: UAVSpecifications):
        self.uav_specs = uav_specs
        self.terrain_map = None
        self.landmark_database = {}
        self.navigation_accuracy = 0.95
        
    def initialize_vision_aided_ins(self, terrain_data: Dict) -> bool:
        """Initialize vision-aided inertial navigation system"""
        try:
            self.terrain_map = terrain_data
            self.landmark_database = self._extract_landmarks(terrain_data)
            logger.info("Vision-aided INS initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize vision-aided INS: {e}")
            return False
    
    def _extract_landmarks(self, terrain_data: Dict) -> Dict:
        """Extract navigation landmarks from terrain data"""
        landmarks = {}
        for i in range(10):
            landmark_id = f"landmark_{i}"
            landmarks[landmark_id] = {
                'position': (
                    np.random.uniform(-1000, 1000),
                    np.random.uniform(-1000, 1000),
                    np.random.uniform(100, 500)
                ),
                'confidence': np.random.uniform(0.7, 0.95),
                'feature_type': np.random.choice(['building', 'ridge', 'river', 'road'])
            }
        return landmarks
    
    def calculate_navigation_solution(self, 
                                    current_state: UAVState,
                                    sensor_data: Dict,
                                    mode: NavigationMode) -> Tuple[Position3D, float]:
        """Calculate navigation solution based on available sensors"""
        if mode == NavigationMode.GPS:
            return self._gps_navigation(current_state, sensor_data)
        elif mode == NavigationMode.VISION_AIDED_INS:
            return self._vision_aided_navigation(current_state, sensor_data)
        elif mode == NavigationMode.COOPERATIVE:
            return self._cooperative_navigation(current_state, sensor_data)
        else:
            return self._terrain_following_navigation(current_state, sensor_data)
    
    def _gps_navigation(self, state: UAVState, sensor_data: Dict) -> Tuple[Position3D, float]:
        """GPS-based navigation (when available)"""
        gps_accuracy = sensor_data.get('gps_accuracy', 0.0)
        if gps_accuracy > 0.8:
            position = Position3D(
                sensor_data['gps_lat'],
                sensor_data['gps_lon'], 
                sensor_data['gps_alt']
            )
            return position, gps_accuracy
        else:
            return self._vision_aided_navigation(state, sensor_data)
    
    def _vision_aided_navigation(self, state: UAVState, sensor_data: Dict) -> Tuple[Position3D, float]:
        """Vision-aided inertial navigation for GPS-denied environments"""
        matched_landmarks = []
        for landmark_id, landmark in self.landmark_database.items():
            detection_prob = np.random.uniform(0.3, 0.9)
            if detection_prob > 0.6:
                matched_landmarks.append(landmark)
        
        if len(matched_landmarks) >= 3:
            accuracy = min(0.9, 0.6 + 0.1 * len(matched_landmarks))
            position = Position3D(
                state.position.x + np.random.normal(0, 5),
                state.position.y + np.random.normal(0, 5),
                state.position.z + np.random.normal(0, 2)
            )
            return position, accuracy
        else:
            return self._dead_reckoning(state), 0.5
    
    def _cooperative_navigation(self, state: UAVState, sensor_data: Dict) -> Tuple[Position3D, float]:
        """Cooperative navigation using other UAVs as references"""
        nearby_uavs = sensor_data.get('nearby_uavs', [])
        if len(nearby_uavs) >= 2:
            accuracy = 0.8
            position = Position3D(
                state.position.x + np.random.normal(0, 3),
                state.position.y + np.random.normal(0, 3),
                state.position.z + np.random.normal(0, 1)
            )
            return position, accuracy
        else:
            return self._vision_aided_navigation(state, sensor_data)
    
    def _terrain_following_navigation(self, state: UAVState, sensor_data: Dict) -> Tuple[Position3D, float]:
        """Terrain-following navigation using radar altimeter"""
        terrain_clearance = sensor_data.get('radar_altitude', 100)
        if terrain_clearance > 0:
            position = Position3D(
                state.position.x,
                state.position.y,
                sensor_data.get('terrain_elevation', 0) + terrain_clearance
            )
            return position, 0.7
        else:
            return self._dead_reckoning(state), 0.3
    
    def _dead_reckoning(self, state: UAVState) -> Position3D:
        """Dead reckoning navigation using last known position and velocity"""
        dt = 1.0
        return Position3D(
            state.position.x + state.velocity.vx * dt,
            state.position.y + state.velocity.vy * dt,
            state.position.z + state.velocity.vz * dt
        )


class CooperativeSensing:
    """Cooperative sensing and communication protocols"""
    
    def __init__(self, communication_range: float):
        self.communication_range = communication_range
        self.sensor_fusion_weights = {
            'visual': 0.4,
            'radar': 0.3,
            'infrared': 0.2,
            'acoustic': 0.1
        }
    
    def establish_communication_network(self, uav_states: List[UAVState]) -> Dict[str, List[str]]:
        """Establish communication links between UAVs within range"""
        network = {}
        
        for i, uav1 in enumerate(uav_states):
            network[uav1.uav_id] = []
            for j, uav2 in enumerate(uav_states):
                if i != j:
                    distance = self._calculate_distance(uav1.position, uav2.position)
                    if distance <= self.communication_range:
                        network[uav1.uav_id].append(uav2.uav_id)
        
        return network
    
    def _calculate_distance(self, pos1: Position3D, pos2: Position3D) -> float:
        """Calculate 3D distance between two positions"""
        return np.sqrt(
            (pos1.x - pos2.x)**2 + 
            (pos1.y - pos2.y)**2 + 
            (pos1.z - pos2.z)**2
        )
    
    def fuse_sensor_data(self, sensor_reports: List[Dict]) -> Dict:
        """Fuse sensor data from multiple UAVs using distance-based clustering"""
        all_targets = []
        
        # Collect all targets from all reports
        for report in sensor_reports:
            for target in report.get('detected_targets', []):
                all_targets.append(target)
        
        fused_targets = {}
        fusion_distance = 50.0  # meters - targets within this distance are considered the same
        
        for target in all_targets:
            target_pos = np.array(target['position'])
            merged = False
            
            # Check if this target should be merged with an existing one
            for existing_id, existing_target in fused_targets.items():
                existing_pos = np.array(existing_target['position'])
                distance = np.linalg.norm(target_pos - existing_pos)
                
                if distance <= fusion_distance:
                    # Merge with existing target
                    existing_target['confidence'] = self._update_confidence(
                        existing_target['confidence'], 
                        target['confidence']
                    )
                    existing_target['sensor_types'].append(target['sensor_type'])
                    existing_target['detection_count'] += 1
                    existing_target['last_updated'] = max(existing_target['last_updated'], target['timestamp'])
                    merged = True
                    break
            
            if not merged:
                # Create new target
                target_id = f"target_{len(fused_targets)}"
                fused_targets[target_id] = {
                    'position': target['position'],
                    'confidence': target['confidence'],
                    'sensor_types': [target['sensor_type']],
                    'detection_count': 1,
                    'last_updated': target['timestamp']
                }
        
        return fused_targets
    
    def _generate_target_id(self, target: Dict) -> str:
        """Generate unique target ID based on position with clustering tolerance"""
        pos = target['position']
        # Use larger grid size for clustering nearby detections (100m grid)
        return f"target_{int(pos[0]/100)}_{int(pos[1]/100)}_{int(pos[2]/100)}"
    
    def _update_confidence(self, existing_conf: float, new_conf: float) -> float:
        """Update target confidence using Bayesian fusion"""
        combined = existing_conf * new_conf
        normalization = combined + (1 - existing_conf) * (1 - new_conf)
        return combined / normalization if normalization > 0 else existing_conf


class SwarmCoordinator:
    """Swarm coordination and task allocation algorithms"""
    
    def __init__(self, swarm_size: int):
        self.swarm_size = swarm_size
        self.formation_controller = FormationController()
        self.task_allocator = TaskAllocator()
    
    def coordinate_swarm_movement(self, 
                                uav_states: List[UAVState],
                                formation: SwarmFormation,
                                target_position: Position3D) -> Dict[str, Position3D]:
        """Coordinate movement of entire swarm"""
        return self.formation_controller.calculate_formation_positions(
            uav_states, formation, target_position
        )
    
    def allocate_reconnaissance_tasks(self, 
                                   uav_states: List[UAVState],
                                   mission: SwarmMission) -> Dict[str, Dict]:
        """Allocate reconnaissance tasks to individual UAVs"""
        return self.task_allocator.allocate_tasks(uav_states, mission)
    
    def handle_uav_failure(self, 
                          failed_uav_id: str,
                          remaining_uavs: List[UAVState],
                          mission: SwarmMission) -> Dict[str, Dict]:
        """Reallocate tasks when a UAV fails"""
        logger.warning(f"UAV {failed_uav_id} failed, reallocating tasks")
        return self.task_allocator.reallocate_after_failure(
            failed_uav_id, remaining_uavs, mission
        )


class FormationController:
    """Controls swarm formation patterns"""
    
    def calculate_formation_positions(self,
                                    uav_states: List[UAVState],
                                    formation: SwarmFormation,
                                    center_position: Position3D) -> Dict[str, Position3D]:
        """Calculate target positions for each UAV in formation"""
        positions = {}
        
        if formation == SwarmFormation.LINE_ABREAST:
            positions = self._line_abreast_formation(uav_states, center_position)
        elif formation == SwarmFormation.WEDGE:
            positions = self._wedge_formation(uav_states, center_position)
        elif formation == SwarmFormation.DIAMOND:
            positions = self._diamond_formation(uav_states, center_position)
        elif formation == SwarmFormation.SEARCH_GRID:
            positions = self._search_grid_formation(uav_states, center_position)
        else:  # ADAPTIVE
            positions = self._adaptive_formation(uav_states, center_position)
        
        return positions
    
    def _line_abreast_formation(self, uav_states: List[UAVState], center: Position3D) -> Dict[str, Position3D]:
        """Line abreast formation for wide area coverage"""
        positions = {}
        spacing = 200
        
        for i, uav in enumerate(uav_states):
            offset = (i - len(uav_states) / 2) * spacing
            positions[uav.uav_id] = Position3D(
                center.x + offset,
                center.y,
                center.z
            )
        
        return positions
    
    def _wedge_formation(self, uav_states: List[UAVState], center: Position3D) -> Dict[str, Position3D]:
        """Wedge formation for forward reconnaissance"""
        positions = {}
        
        for i, uav in enumerate(uav_states):
            if i == 0:
                positions[uav.uav_id] = center
            else:
                side = 1 if i % 2 == 1 else -1
                row = (i + 1) // 2
                positions[uav.uav_id] = Position3D(
                    center.x + side * row * 150,
                    center.y - row * 200,
                    center.z
                )
        
        return positions
    
    def _diamond_formation(self, uav_states: List[UAVState], center: Position3D) -> Dict[str, Position3D]:
        """Diamond formation for balanced coverage"""
        positions = {}
        
        if len(uav_states) >= 4:
            offsets = [(0, 200), (200, 0), (0, -200), (-200, 0)]
            for i, uav in enumerate(uav_states[:4]):
                offset_x, offset_y = offsets[i]
                positions[uav.uav_id] = Position3D(
                    center.x + offset_x,
                    center.y + offset_y,
                    center.z
                )
            
            for i, uav in enumerate(uav_states[4:]):
                angle = i * 2 * np.pi / max(1, len(uav_states) - 4)
                positions[uav.uav_id] = Position3D(
                    center.x + 100 * np.cos(angle),
                    center.y + 100 * np.sin(angle),
                    center.z
                )
        
        return positions
    
    def _search_grid_formation(self, uav_states: List[UAVState], center: Position3D) -> Dict[str, Position3D]:
        """Grid formation for systematic area search"""
        positions = {}
        grid_size = int(np.ceil(np.sqrt(len(uav_states))))
        spacing = 300
        
        for i, uav in enumerate(uav_states):
            row = i // grid_size
            col = i % grid_size
            positions[uav.uav_id] = Position3D(
                center.x + (col - grid_size/2) * spacing,
                center.y + (row - grid_size/2) * spacing,
                center.z
            )
        
        return positions
    
    def _adaptive_formation(self, uav_states: List[UAVState], center: Position3D) -> Dict[str, Position3D]:
        """Adaptive formation based on mission requirements and threats"""
        if len(uav_states) <= 4:
            return self._diamond_formation(uav_states, center)
        else:
            return self._search_grid_formation(uav_states, center)


class TaskAllocator:
    """Task allocation algorithms for swarm missions"""
    
    def allocate_tasks(self, uav_states: List[UAVState], mission: SwarmMission) -> Dict[str, Dict]:
        """Allocate tasks to UAVs based on mission requirements"""
        if mission.mission_type == "reconnaissance":
            return self._allocate_reconnaissance_tasks(uav_states, mission)
        elif mission.mission_type == "surveillance":
            return self._allocate_surveillance_tasks(uav_states, mission)
        else:
            return self._allocate_reconnaissance_tasks(uav_states, mission)
    
    def _allocate_reconnaissance_tasks(self, uav_states: List[UAVState], mission: SwarmMission) -> Dict[str, Dict]:
        """Allocate reconnaissance tasks with area coverage optimization"""
        tasks = {}
        
        lat_min, lat_max, lon_min, lon_max = mission.target_area
        sectors = self._divide_area_into_sectors(
            (lat_min, lat_max, lon_min, lon_max), 
            len(uav_states)
        )
        
        for i, uav in enumerate(uav_states):
            if i < len(sectors):
                sector = sectors[i]
                tasks[uav.uav_id] = {
                    'task_type': 'sector_reconnaissance',
                    'assigned_sector': sector,
                    'search_pattern': mission.search_pattern,
                    'priority_targets': mission.priority_targets,
                    'estimated_duration': mission.duration / len(uav_states),
                    'waypoints': self._generate_search_waypoints(sector, mission.search_pattern)
                }
        
        return tasks
    
    def _divide_area_into_sectors(self, area: Tuple[float, float, float, float], num_sectors: int) -> List[Dict]:
        """Divide target area into sectors for individual UAVs"""
        lat_min, lat_max, lon_min, lon_max = area
        sectors = []
        
        grid_size = int(np.ceil(np.sqrt(num_sectors)))
        lat_step = (lat_max - lat_min) / grid_size
        lon_step = (lon_max - lon_min) / grid_size
        
        for i in range(num_sectors):
            row = i // grid_size
            col = i % grid_size
            
            sector_lat_min = lat_min + row * lat_step
            sector_lat_max = min(lat_max, sector_lat_min + lat_step)
            sector_lon_min = lon_min + col * lon_step
            sector_lon_max = min(lon_max, sector_lon_min + lon_step)
            
            sectors.append({
                'id': f'sector_{i}',
                'bounds': (sector_lat_min, sector_lat_max, sector_lon_min, sector_lon_max),
                'area_km2': (sector_lat_max - sector_lat_min) * (sector_lon_max - sector_lon_min) * 111.32**2
            })
        
        return sectors
    
    def _generate_search_waypoints(self, sector: Dict, pattern: str) -> List[Position3D]:
        """Generate waypoints for search pattern within sector"""
        lat_min, lat_max, lon_min, lon_max = sector['bounds']
        waypoints = []
        
        if pattern == "raster":
            num_lines = 5
            lat_step = (lat_max - lat_min) / num_lines
            
            for i in range(num_lines):
                lat = lat_min + i * lat_step
                if i % 2 == 0:
                    waypoints.extend([
                        Position3D(lat, lon_min, 300),
                        Position3D(lat, lon_max, 300)
                    ])
                else:
                    waypoints.extend([
                        Position3D(lat, lon_max, 300),
                        Position3D(lat, lon_min, 300)
                    ])
        
        elif pattern == "spiral":
            center_lat = (lat_min + lat_max) / 2
            center_lon = (lon_min + lon_max) / 2
            max_radius = min(lat_max - lat_min, lon_max - lon_min) / 2
            
            for angle in np.linspace(0, 4 * np.pi, 20):
                radius = max_radius * (1 - angle / (4 * np.pi))
                lat = center_lat + radius * np.cos(angle)
                lon = center_lon + radius * np.sin(angle)
                waypoints.append(Position3D(lat, lon, 300))
        
        return waypoints
    
    def _allocate_surveillance_tasks(self, uav_states: List[UAVState], mission: SwarmMission) -> Dict[str, Dict]:
        """Allocate surveillance tasks for persistent monitoring"""
        tasks = {}
        
        for i, uav in enumerate(uav_states):
            if i < len(mission.priority_targets):
                target = mission.priority_targets[i]
                tasks[uav.uav_id] = {
                    'task_type': 'target_surveillance',
                    'target_id': target,
                    'orbit_radius': 1000,
                    'orbit_altitude': 500,
                    'orbit_duration': mission.duration
                }
        
        return tasks
    
    def reallocate_after_failure(self, 
                               failed_uav_id: str,
                               remaining_uavs: List[UAVState],
                               mission: SwarmMission) -> Dict[str, Dict]:
        """Reallocate tasks after UAV failure"""
        # Filter out the failed UAV and reallocate tasks to remaining UAVs
        active_uavs = [uav for uav in remaining_uavs if uav.uav_id != failed_uav_id]
        return self.allocate_tasks(active_uavs, mission)


class MissionPlanner:
    """Mission planning tools for reconnaissance transects"""
    
    def __init__(self):
        self.weather_model = WeatherModel()
        self.threat_assessment = ThreatAssessment()
    
    def plan_reconnaissance_mission(self, 
                                 target_area: Tuple[float, float, float, float],
                                 uav_specs: UAVSpecifications,
                                 swarm_size: int,
                                 mission_requirements: Dict) -> SwarmMission:
        """Plan comprehensive reconnaissance mission"""
        
        duration = self._calculate_mission_duration(target_area, uav_specs, swarm_size)
        search_pattern = self._select_optimal_search_pattern(target_area, mission_requirements)
        
        threat_zones = self.threat_assessment.identify_threat_zones(target_area)
        weather_constraints = self.weather_model.get_weather_constraints(target_area)
        
        mission = SwarmMission(
            mission_type="reconnaissance",
            target_area=target_area,
            altitude_range=self._calculate_optimal_altitude(weather_constraints, threat_zones),
            search_pattern=search_pattern,
            duration=duration,
            priority_targets=mission_requirements.get('priority_targets', []),
            threat_zones=threat_zones
        )
        
        return mission
    
    def _calculate_mission_duration(self, 
                                  target_area: Tuple[float, float, float, float],
                                  uav_specs: UAVSpecifications,
                                  swarm_size: int) -> float:
        """Calculate estimated mission duration"""
        lat_min, lat_max, lon_min, lon_max = target_area
        area_km2 = (lat_max - lat_min) * (lon_max - lon_min) * 111.32**2
        
        coverage_rate_km2_per_hour = uav_specs.max_speed * 3.6 * uav_specs.sensor_range / 1000
        total_time = area_km2 / (coverage_rate_km2_per_hour * swarm_size)
        
        return total_time * 1.3
    
    def _select_optimal_search_pattern(self, 
                                     target_area: Tuple[float, float, float, float],
                                     requirements: Dict) -> str:
        """Select optimal search pattern based on area and requirements"""
        lat_min, lat_max, lon_min, lon_max = target_area
        aspect_ratio = (lat_max - lat_min) / (lon_max - lon_min)
        
        if requirements.get('high_resolution', False):
            return "raster"
        elif aspect_ratio > 2 or aspect_ratio < 0.5:
            return "raster"
        else:
            return "spiral"
    
    def _calculate_optimal_altitude(self, 
                                  weather_constraints: Dict,
                                  threat_zones: List[Dict]) -> Tuple[float, float]:
        """Calculate optimal altitude range considering weather and threats"""
        min_alt = 200
        max_alt = 1000
        
        if weather_constraints.get('cloud_ceiling', 2000) < max_alt:
            max_alt = weather_constraints['cloud_ceiling'] - 100
        
        for threat in threat_zones:
            if threat.get('type') == 'air_defense':
                threat_ceiling = threat.get('engagement_ceiling', 500)
                if threat_ceiling < max_alt:
                    max_alt = min(max_alt, threat_ceiling - 50)
        
        return (min_alt, max_alt)


class WeatherModel:
    """Simple weather modeling for mission planning"""
    
    def get_weather_constraints(self, area: Tuple[float, float, float, float]) -> Dict:
        """Get weather constraints for mission area"""
        return {
            'wind_speed': np.random.uniform(5, 15),
            'wind_direction': np.random.uniform(0, 360),
            'cloud_ceiling': np.random.uniform(500, 2000),
            'visibility': np.random.uniform(5, 15),
            'precipitation': np.random.choice([True, False], p=[0.2, 0.8])
        }


class ThreatAssessment:
    """Threat assessment for mission planning"""
    
    def identify_threat_zones(self, area: Tuple[float, float, float, float]) -> List[Dict]:
        """Identify potential threat zones in mission area"""
        threats = []
        
        num_threats = np.random.randint(0, 3)
        lat_min, lat_max, lon_min, lon_max = area
        
        for i in range(num_threats):
            threat = {
                'id': f'threat_{i}',
                'type': np.random.choice(['air_defense', 'radar', 'electronic_warfare']),
                'position': (
                    np.random.uniform(lat_min, lat_max),
                    np.random.uniform(lon_min, lon_max)
                ),
                'range': np.random.uniform(5, 20),
                'engagement_ceiling': np.random.uniform(300, 800),
                'threat_level': np.random.choice(['low', 'medium', 'high'])
            }
            threats.append(threat)
        
        return threats


class UAVSwarmSimulator(SimulationComponent):
    """Main UAV swarm simulation class"""
    
    def __init__(self, swarm_size: int, uav_specs: UAVSpecifications):
        super().__init__()
        self.swarm_size = swarm_size
        self.uav_specs = uav_specs
        self.uav_states = []
        self.navigator = AutonomousNavigator(uav_specs)
        self.cooperative_sensing = CooperativeSensing(uav_specs.communication_range)
        self.swarm_coordinator = SwarmCoordinator(swarm_size)
        self.mission_planner = MissionPlanner()
        self.current_mission = None
        
        self._initialize_swarm()
    
    def initialize(self) -> bool:
        """Initialize the UAV swarm simulator."""
        try:
            self._initialize_swarm()
            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize UAV swarm simulator: {e}")
            return False
    
    def _initialize_swarm(self):
        """Initialize individual UAV states"""
        for i in range(self.swarm_size):
            uav_state = UAVState(
                uav_id=f"UAV_{i:03d}",
                position=Position3D(0, i * 100, 300),
                velocity=Velocity3D(0, 0, 0),
                heading=0.0,
                fuel_remaining=100.0,
                sensor_status={'visual': True, 'radar': True, 'infrared': True},
                communication_status=True,
                mission_status='ready'
            )
            self.uav_states.append(uav_state)
    
    def start_mission(self, mission: SwarmMission) -> bool:
        """Start a new swarm mission"""
        try:
            self.current_mission = mission
            
            task_allocation = self.swarm_coordinator.allocate_reconnaissance_tasks(
                self.uav_states, mission
            )
            
            for uav in self.uav_states:
                if uav.uav_id in task_allocation:
                    uav.mission_status = 'active'
            
            logger.info(f"Started mission with {len(self.uav_states)} UAVs")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start mission: {e}")
            return False
    
    def update_simulation(self, dt: float) -> Dict[str, Any]:
        """Update simulation state"""
        self.simulation_time += dt
        
        for uav in self.uav_states:
            if uav.mission_status == 'active':
                self._update_uav_state(uav, dt)
        
        communication_network = self.cooperative_sensing.establish_communication_network(self.uav_states)
        
        sensor_reports = []
        for uav in self.uav_states:
            if uav.sensor_status.get('visual', False):
                sensor_reports.append(self._generate_sensor_report(uav))
        
        fused_targets = self.cooperative_sensing.fuse_sensor_data(sensor_reports)
        
        return {
            'simulation_time': self.simulation_time,
            'uav_states': [self._uav_state_to_dict(uav) for uav in self.uav_states],
            'communication_network': communication_network,
            'detected_targets': fused_targets,
            'mission_progress': self._calculate_mission_progress()
        }
    
    def _update_uav_state(self, uav: UAVState, dt: float):
        """Update individual UAV state"""
        sensor_data = self._generate_sensor_data(uav)
        new_position, accuracy = self.navigator.calculate_navigation_solution(
            uav, sensor_data, NavigationMode.VISION_AIDED_INS
        )
        
        uav.position = new_position
        
        fuel_consumption_rate = 2.0
        uav.fuel_remaining -= fuel_consumption_rate * dt / 3600
        
        if np.random.random() < 0.0001 * dt:
            uav.mission_status = 'failed'
            logger.warning(f"UAV {uav.uav_id} experienced failure")
    
    def _generate_sensor_data(self, uav: UAVState) -> Dict:
        """Generate simulated sensor data for UAV"""
        return {
            'gps_accuracy': np.random.uniform(0.1, 0.9),
            'gps_lat': uav.position.x + np.random.normal(0, 10),
            'gps_lon': uav.position.y + np.random.normal(0, 10),
            'gps_alt': uav.position.z + np.random.normal(0, 5),
            'radar_altitude': uav.position.z + np.random.normal(0, 2),
            'terrain_elevation': 0,
            'nearby_uavs': [other.uav_id for other in self.uav_states 
                          if other.uav_id != uav.uav_id and 
                          self.cooperative_sensing._calculate_distance(uav.position, other.position) < 500]
        }
    
    def _generate_sensor_report(self, uav: UAVState) -> Dict:
        """Generate sensor report for cooperative sensing"""
        detected_targets = []
        if np.random.random() < 0.1:
            target = {
                'position': (
                    uav.position.x + np.random.uniform(-1000, 1000),
                    uav.position.y + np.random.uniform(-1000, 1000),
                    np.random.uniform(0, 100)
                ),
                'confidence': np.random.uniform(0.6, 0.95),
                'sensor_type': 'visual',
                'timestamp': self.simulation_time
            }
            detected_targets.append(target)
        
        return {
            'uav_id': uav.uav_id,
            'timestamp': self.simulation_time,
            'detected_targets': detected_targets
        }
    
    def _uav_state_to_dict(self, uav: UAVState) -> Dict:
        """Convert UAV state to dictionary for output"""
        return {
            'uav_id': uav.uav_id,
            'position': {'x': uav.position.x, 'y': uav.position.y, 'z': uav.position.z},
            'velocity': {'vx': uav.velocity.vx, 'vy': uav.velocity.vy, 'vz': uav.velocity.vz},
            'heading': uav.heading,
            'fuel_remaining': uav.fuel_remaining,
            'sensor_status': uav.sensor_status,
            'communication_status': uav.communication_status,
            'mission_status': uav.mission_status
        }
    
    def _calculate_mission_progress(self) -> float:
        """Calculate overall mission progress"""
        if not self.current_mission:
            return 0.0
        
        active_uavs = sum(1 for uav in self.uav_states if uav.mission_status == 'active')
        total_uavs = len(self.uav_states)
        
        time_progress = min(1.0, self.simulation_time / (self.current_mission.duration * 3600))
        uav_progress = active_uavs / total_uavs if total_uavs > 0 else 0
        
        return (time_progress + uav_progress) / 2
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """Get current swarm status"""
        active_uavs = sum(1 for uav in self.uav_states if uav.mission_status == 'active')
        failed_uavs = sum(1 for uav in self.uav_states if uav.mission_status == 'failed')
        
        return {
            'total_uavs': len(self.uav_states),
            'active_uavs': active_uavs,
            'failed_uavs': failed_uavs,
            'mission_progress': self._calculate_mission_progress(),
            'simulation_time': self.simulation_time,
            'current_mission': self.current_mission.mission_type if self.current_mission else None
        }