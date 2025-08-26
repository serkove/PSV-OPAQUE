"""Plasma-based decoy and sensor systems for advanced detection and countermeasures."""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum, auto
import time
from ...common.enums import SensorType
from ...core.logging import get_engine_logger


class PlasmaType(Enum):
    """Types of plasma generation methods."""
    RF_SUSTAINED = auto()
    MICROWAVE_INDUCED = auto()
    LASER_INDUCED = auto()
    CHEMICAL = auto()
    ELECTROMAGNETIC = auto()


class PlasmaState(Enum):
    """Plasma operational states."""
    INACTIVE = auto()
    IGNITING = auto()
    ACTIVE = auto()
    DECAYING = auto()
    FAILED = auto()


@dataclass
class PlasmaConfiguration:
    """Plasma system configuration parameters."""
    plasma_type: PlasmaType
    frequency: float  # Hz (for RF/microwave systems)
    power_input: float  # W
    gas_composition: Dict[str, float]  # Gas mixture ratios
    pressure: float  # Pa
    temperature_target: float  # K
    volume: float  # m³
    magnetic_field_strength: Optional[float] = None  # T (for magnetized plasmas)


@dataclass
class PlasmaProperties:
    """Physical properties of plasma."""
    electron_density: float  # m^-3
    electron_temperature: float  # K
    ion_temperature: float  # K
    plasma_frequency: float  # Hz
    debye_length: float  # m
    collision_frequency: float  # Hz
    conductivity: float  # S/m
    dielectric_constant: complex


@dataclass
class PlasmaOrb:
    """Individual plasma orb representation."""
    orb_id: str
    position: Tuple[float, float, float]  # x, y, z coordinates (m)
    velocity: Tuple[float, float, float]  # vx, vy, vz (m/s)
    radius: float  # m
    plasma_properties: PlasmaProperties
    state: PlasmaState
    creation_time: float  # s
    lifetime_remaining: float  # s
    power_consumption: float  # W


class PlasmaDecoyGenerator:
    """Generator for RF-sustained plasma structures and decoys."""
    
    def __init__(self, config: PlasmaConfiguration):
        """Initialize plasma decoy generator."""
        self.config = config
        self.logger = get_engine_logger('sensors.plasma.decoy')
        self.active_orbs: Dict[str, PlasmaOrb] = {}
        self.orb_counter = 0
        
        # Physical constants
        self.e = 1.602176634e-19  # Elementary charge (C)
        self.m_e = 9.1093837015e-31  # Electron mass (kg)
        self.epsilon_0 = 8.8541878128e-12  # Vacuum permittivity (F/m)
        self.k_b = 1.380649e-23  # Boltzmann constant (J/K)
        
        # Calculate base plasma parameters
        self._calculate_base_parameters()
    
    def _calculate_base_parameters(self):
        """Calculate base plasma parameters from configuration."""
        # Estimate electron density from power and volume
        # Simplified model: higher power = higher density
        power_density = self.config.power_input / self.config.volume  # W/m³
        
        # Empirical relationship (would be more complex in reality)
        self.base_electron_density = 1e15 * (power_density / 1e6) ** 0.5  # m^-3
        
        # Plasma frequency
        self.base_plasma_frequency = math.sqrt(
            self.base_electron_density * self.e**2 / (self.m_e * self.epsilon_0)
        )
        
        # Debye length
        self.base_debye_length = math.sqrt(
            self.epsilon_0 * self.k_b * self.config.temperature_target / 
            (self.base_electron_density * self.e**2)
        )
        
        self.logger.info(f"Plasma generator initialized: {self.config.plasma_type.name}, "
                        f"base density: {self.base_electron_density:.2e} m^-3, "
                        f"plasma frequency: {self.base_plasma_frequency/1e9:.2f} GHz")
    
    def calculate_plasma_properties(self, power_fraction: float = 1.0) -> PlasmaProperties:
        """
        Calculate plasma properties for given power fraction.
        
        Args:
            power_fraction: Fraction of maximum power (0-1)
            
        Returns:
            PlasmaProperties object
        """
        # Scale properties with power
        electron_density = self.base_electron_density * power_fraction
        
        # Electron temperature (simplified model)
        electron_temp = self.config.temperature_target * (0.5 + 0.5 * power_fraction)
        
        # Ion temperature (typically lower than electron temperature)
        ion_temp = electron_temp * 0.3
        
        # Plasma frequency
        plasma_freq = math.sqrt(
            electron_density * self.e**2 / (self.m_e * self.epsilon_0)
        )
        
        # Debye length
        debye_length = math.sqrt(
            self.epsilon_0 * self.k_b * electron_temp / 
            (electron_density * self.e**2)
        )
        
        # Collision frequency (simplified)
        collision_freq = 1e6 * math.sqrt(electron_temp / 11600)  # Hz (rough approximation)
        
        # Conductivity (Spitzer conductivity)
        conductivity = (electron_density * self.e**2) / (self.m_e * collision_freq)
        
        # Dielectric constant
        omega = 2 * math.pi * self.config.frequency
        omega_p = 2 * math.pi * plasma_freq
        nu = collision_freq
        
        # Complex dielectric constant
        real_part = 1 - (omega_p**2) / (omega**2 + nu**2)
        imag_part = (omega_p**2 * nu) / (omega * (omega**2 + nu**2))
        dielectric_constant = complex(real_part, imag_part)
        
        return PlasmaProperties(
            electron_density=electron_density,
            electron_temperature=electron_temp,
            ion_temperature=ion_temp,
            plasma_frequency=plasma_freq,
            debye_length=debye_length,
            collision_frequency=collision_freq,
            conductivity=conductivity,
            dielectric_constant=dielectric_constant
        )
    
    def create_plasma_orb(self, position: Tuple[float, float, float],
                         velocity: Tuple[float, float, float],
                         radius: float, power_fraction: float = 1.0,
                         lifetime: float = 60.0) -> str:
        """
        Create a new plasma orb.
        
        Args:
            position: Initial position (x, y, z) in meters
            velocity: Initial velocity (vx, vy, vz) in m/s
            radius: Orb radius in meters
            power_fraction: Power fraction (0-1)
            lifetime: Expected lifetime in seconds
            
        Returns:
            Orb ID string
        """
        self.orb_counter += 1
        orb_id = f"PLASMA_{self.orb_counter:04d}"
        
        # Calculate plasma properties
        plasma_props = self.calculate_plasma_properties(power_fraction)
        
        # Calculate power consumption
        power_consumption = self.config.power_input * power_fraction
        
        # Create orb
        orb = PlasmaOrb(
            orb_id=orb_id,
            position=position,
            velocity=velocity,
            radius=radius,
            plasma_properties=plasma_props,
            state=PlasmaState.IGNITING,
            creation_time=time.time(),
            lifetime_remaining=lifetime,
            power_consumption=power_consumption
        )
        
        self.active_orbs[orb_id] = orb
        
        self.logger.info(f"Created plasma orb {orb_id} at {position} with "
                        f"radius {radius:.2f}m, power {power_consumption:.0f}W")
        
        return orb_id
    
    def update_orb_states(self, dt: float) -> None:
        """
        Update states of all active plasma orbs.
        
        Args:
            dt: Time step in seconds
        """
        orbs_to_remove = []
        
        for orb_id, orb in self.active_orbs.items():
            # Update position
            new_position = (
                orb.position[0] + orb.velocity[0] * dt,
                orb.position[1] + orb.velocity[1] * dt,
                orb.position[2] + orb.velocity[2] * dt
            )
            orb.position = new_position
            
            # Update lifetime
            orb.lifetime_remaining -= dt
            
            # Update state based on lifetime
            if orb.state == PlasmaState.IGNITING:
                if time.time() - orb.creation_time > 1.0:  # 1 second ignition time
                    orb.state = PlasmaState.ACTIVE
            
            elif orb.state == PlasmaState.ACTIVE:
                if orb.lifetime_remaining <= 5.0:  # Start decaying 5 seconds before end
                    orb.state = PlasmaState.DECAYING
            
            elif orb.state == PlasmaState.DECAYING:
                if orb.lifetime_remaining <= 0:
                    orb.state = PlasmaState.INACTIVE
                    orbs_to_remove.append(orb_id)
            
            # Also remove orbs that have expired while in other states
            if orb.lifetime_remaining <= 0 and orb.state != PlasmaState.INACTIVE:
                orb.state = PlasmaState.INACTIVE
                orbs_to_remove.append(orb_id)
        
        # Remove inactive orbs
        for orb_id in orbs_to_remove:
            del self.active_orbs[orb_id]
            self.logger.debug(f"Removed inactive plasma orb {orb_id}")
    
    def calculate_radar_cross_section(self, orb: PlasmaOrb, frequency: float) -> float:
        """
        Calculate radar cross-section of plasma orb.
        
        Args:
            orb: Plasma orb
            frequency: Radar frequency (Hz)
            
        Returns:
            RCS in m²
        """
        # Plasma RCS depends on dielectric properties and size
        props = orb.plasma_properties
        
        # Wave number
        k = 2 * math.pi * frequency / 299792458  # c = speed of light
        
        # Size parameter
        size_parameter = k * orb.radius
        
        # Dielectric constant at radar frequency
        omega = 2 * math.pi * frequency
        omega_p = 2 * math.pi * props.plasma_frequency
        nu = props.collision_frequency
        
        epsilon_r = 1 - (omega_p**2) / (omega**2 + nu**2)
        epsilon_i = (omega_p**2 * nu) / (omega * (omega**2 + nu**2))
        epsilon = complex(epsilon_r, epsilon_i)
        
        # Mie scattering approximation for small spheres
        if size_parameter < 1:
            # Rayleigh scattering
            alpha = 4 * math.pi * orb.radius**3 * (epsilon - 1) / (epsilon + 2)
            rcs = (k**4 / (6 * math.pi)) * abs(alpha)**2
        else:
            # Geometric optics approximation
            reflectivity = abs((epsilon - 1) / (epsilon + 1))**2
            rcs = math.pi * orb.radius**2 * reflectivity
        
        return rcs
    
    def calculate_optical_signature(self, orb: PlasmaOrb, wavelength: float) -> Dict[str, float]:
        """
        Calculate optical signature of plasma orb.
        
        Args:
            orb: Plasma orb
            wavelength: Optical wavelength (m)
            
        Returns:
            Dictionary with optical properties
        """
        props = orb.plasma_properties
        
        # Plasma emission (simplified blackbody + line emission)
        # Stefan-Boltzmann law for thermal emission
        sigma_sb = 5.670374419e-8  # Stefan-Boltzmann constant
        thermal_power = 4 * math.pi * orb.radius**2 * sigma_sb * props.electron_temperature**4
        
        # Line emission (simplified)
        line_power = props.electron_density * orb.radius**3 * 1e-40  # Rough approximation
        
        # Total optical power
        total_power = thermal_power + line_power
        
        # Brightness temperature
        brightness_temp = (total_power / (4 * math.pi * orb.radius**2 * sigma_sb))**0.25
        
        return {
            'thermal_power_w': thermal_power,
            'line_power_w': line_power,
            'total_power_w': total_power,
            'brightness_temperature_k': brightness_temp,
            'apparent_magnitude': -2.5 * math.log10(total_power / 1e-12)  # Rough magnitude
        }
    
    def calculate_power_requirements(self, num_orbs: int, orb_radius: float,
                                   lifetime: float) -> Dict[str, float]:
        """
        Calculate power requirements for plasma orb network.
        
        Args:
            num_orbs: Number of simultaneous orbs
            orb_radius: Average orb radius (m)
            lifetime: Average orb lifetime (s)
            
        Returns:
            Power requirement breakdown
        """
        # Power per orb (scales with volume)
        orb_volume = (4/3) * math.pi * orb_radius**3
        power_per_orb = self.config.power_input * (orb_volume / self.config.volume)
        
        # Total continuous power
        total_power = num_orbs * power_per_orb
        
        # Energy per orb
        energy_per_orb = power_per_orb * lifetime
        
        # Peak power (during ignition)
        ignition_power_factor = 3.0  # 3x power during ignition
        peak_power = total_power * ignition_power_factor
        
        return {
            'power_per_orb_w': power_per_orb,
            'total_continuous_power_w': total_power,
            'peak_power_w': peak_power,
            'energy_per_orb_j': energy_per_orb,
            'total_energy_per_hour_mj': (total_power * 3600) / 1e6
        }
    
    def get_active_orb_count(self) -> int:
        """Get number of currently active orbs."""
        return len([orb for orb in self.active_orbs.values() 
                   if orb.state in [PlasmaState.ACTIVE, PlasmaState.IGNITING]])
    
    def get_total_power_consumption(self) -> float:
        """Get total power consumption of all active orbs."""
        return sum(orb.power_consumption for orb in self.active_orbs.values()
                  if orb.state in [PlasmaState.ACTIVE, PlasmaState.IGNITING])


class CooperativeSensingNetwork:
    """Cooperative sensing algorithms for plasma orb networks."""
    
    def __init__(self, communication_range: float = 10000.0):
        """
        Initialize cooperative sensing network.
        
        Args:
            communication_range: Maximum communication range between orbs (m)
        """
        self.communication_range = communication_range
        self.logger = get_engine_logger('sensors.plasma.cooperative')
        self.sensor_data: Dict[str, Dict[str, Any]] = {}
        self.network_topology: Dict[str, List[str]] = {}
    
    def update_network_topology(self, orbs: Dict[str, PlasmaOrb]) -> None:
        """
        Update network topology based on orb positions.
        
        Args:
            orbs: Dictionary of active plasma orbs
        """
        self.network_topology.clear()
        
        for orb_id, orb in orbs.items():
            if orb.state != PlasmaState.ACTIVE:
                continue
                
            neighbors = []
            
            for other_id, other_orb in orbs.items():
                if other_id == orb_id or other_orb.state != PlasmaState.ACTIVE:
                    continue
                
                # Calculate distance
                distance = math.sqrt(
                    sum((orb.position[i] - other_orb.position[i])**2 for i in range(3))
                )
                
                if distance <= self.communication_range:
                    neighbors.append(other_id)
            
            self.network_topology[orb_id] = neighbors
    
    def simulate_target_detection(self, orbs: Dict[str, PlasmaOrb],
                                 targets: List[Tuple[float, float, float]],
                                 detection_range: float = 5000.0) -> Dict[str, List[Dict[str, Any]]]:
        """
        Simulate cooperative target detection.
        
        Args:
            orbs: Dictionary of active plasma orbs
            targets: List of target positions
            detection_range: Detection range per orb (m)
            
        Returns:
            Dictionary of detections per orb
        """
        detections = {}
        
        for orb_id, orb in orbs.items():
            if orb.state != PlasmaState.ACTIVE:
                continue
            
            orb_detections = []
            
            for i, target_pos in enumerate(targets):
                # Calculate distance to target
                distance = math.sqrt(
                    sum((orb.position[j] - target_pos[j])**2 for j in range(3))
                )
                
                if distance <= detection_range:
                    # Calculate detection probability based on distance and plasma properties
                    detection_prob = self._calculate_detection_probability(
                        orb, distance, target_pos
                    )
                    
                    if np.random.random() < detection_prob:
                        detection = {
                            'target_id': f"T{i:03d}",
                            'position': target_pos,
                            'distance': distance,
                            'detection_time': time.time(),
                            'confidence': detection_prob,
                            'sensor_type': 'plasma_orb'
                        }
                        orb_detections.append(detection)
            
            detections[orb_id] = orb_detections
        
        return detections
    
    def _calculate_detection_probability(self, orb: PlasmaOrb, distance: float,
                                       target_pos: Tuple[float, float, float]) -> float:
        """Calculate detection probability for a target."""
        # Simple model based on distance and plasma properties
        max_range = 5000.0  # m
        
        # Range factor
        range_factor = max(0, 1 - distance / max_range)
        
        # Plasma quality factor (based on electron density)
        density_factor = min(1.0, orb.plasma_properties.electron_density / 1e16)
        
        # Base detection probability
        base_prob = 0.9
        
        detection_prob = base_prob * range_factor * density_factor
        
        return max(0, min(1, detection_prob))
    
    def fuse_sensor_data(self, detections: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Fuse sensor data from multiple orbs using cooperative algorithms.
        
        Args:
            detections: Raw detections from each orb
            
        Returns:
            Fused detection results
        """
        # Collect all detections
        all_detections = []
        for orb_id, orb_detections in detections.items():
            for detection in orb_detections:
                detection['reporting_orb'] = orb_id
                all_detections.append(detection)
        
        if not all_detections:
            return []
        
        # Group detections by proximity (simple clustering)
        fused_detections = []
        processed = set()
        
        for i, detection in enumerate(all_detections):
            if i in processed:
                continue
            
            # Find nearby detections
            cluster = [detection]
            cluster_indices = {i}
            
            for j, other_detection in enumerate(all_detections):
                if j in processed or j == i:
                    continue
                
                # Calculate distance between detections
                pos1 = detection['position']
                pos2 = other_detection['position']
                distance = math.sqrt(sum((pos1[k] - pos2[k])**2 for k in range(3)))
                
                if distance < 100:  # 100m clustering threshold
                    cluster.append(other_detection)
                    cluster_indices.add(j)
            
            processed.update(cluster_indices)
            
            # Fuse cluster into single detection
            fused_detection = self._fuse_detection_cluster(cluster)
            fused_detections.append(fused_detection)
        
        return fused_detections
    
    def _fuse_detection_cluster(self, cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fuse a cluster of detections into a single result."""
        if len(cluster) == 1:
            return cluster[0]
        
        # Calculate weighted average position
        total_weight = sum(det['confidence'] for det in cluster)
        
        if total_weight == 0:
            # Fallback to simple average
            avg_pos = tuple(
                sum(det['position'][i] for det in cluster) / len(cluster)
                for i in range(3)
            )
            avg_confidence = sum(det['confidence'] for det in cluster) / len(cluster)
        else:
            avg_pos = tuple(
                sum(det['position'][i] * det['confidence'] for det in cluster) / total_weight
                for i in range(3)
            )
            # Confidence increases with multiple detections
            avg_confidence = min(1.0, total_weight / len(cluster) * 1.2)
        
        # Use most recent detection time
        latest_time = max(det['detection_time'] for det in cluster)
        
        # Combine reporting orbs
        reporting_orbs = [det['reporting_orb'] for det in cluster]
        
        return {
            'target_id': cluster[0]['target_id'],
            'position': avg_pos,
            'distance': math.sqrt(sum(coord**2 for coord in avg_pos)),
            'detection_time': latest_time,
            'confidence': avg_confidence,
            'sensor_type': 'fused_plasma_network',
            'reporting_orbs': reporting_orbs,
            'num_detections': len(cluster)
        }
    
    def calculate_network_coverage(self, orbs: Dict[str, PlasmaOrb],
                                 area_bounds: Tuple[Tuple[float, float], 
                                                   Tuple[float, float],
                                                   Tuple[float, float]],
                                 detection_range: float = 5000.0) -> Dict[str, float]:
        """
        Calculate network coverage statistics.
        
        Args:
            orbs: Dictionary of active plasma orbs
            area_bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
            detection_range: Detection range per orb (m)
            
        Returns:
            Coverage statistics
        """
        # Sample points in the area
        n_samples = 1000
        x_range, y_range, z_range = area_bounds
        
        sample_points = [
            (
                np.random.uniform(x_range[0], x_range[1]),
                np.random.uniform(y_range[0], y_range[1]),
                np.random.uniform(z_range[0], z_range[1])
            )
            for _ in range(n_samples)
        ]
        
        covered_points = 0
        multi_covered_points = 0
        
        for point in sample_points:
            covering_orbs = 0
            
            for orb in orbs.values():
                if orb.state != PlasmaState.ACTIVE:
                    continue
                
                distance = math.sqrt(
                    sum((orb.position[i] - point[i])**2 for i in range(3))
                )
                
                if distance <= detection_range:
                    covering_orbs += 1
            
            if covering_orbs > 0:
                covered_points += 1
                if covering_orbs > 1:
                    multi_covered_points += 1
        
        coverage_fraction = covered_points / n_samples
        redundancy_fraction = multi_covered_points / max(covered_points, 1)
        
        # Calculate area
        area_volume = ((x_range[1] - x_range[0]) * 
                      (y_range[1] - y_range[0]) * 
                      (z_range[1] - z_range[0]))
        
        return {
            'coverage_fraction': coverage_fraction,
            'redundancy_fraction': redundancy_fraction,
            'covered_volume_km3': area_volume * coverage_fraction / 1e9,
            'active_orbs': len([orb for orb in orbs.values() 
                              if orb.state == PlasmaState.ACTIVE]),
            'network_connectivity': len(self.network_topology)
        }


class PlasmaSystemController:
    """High-level controller for plasma-based systems."""
    
    def __init__(self, decoy_generator: PlasmaDecoyGenerator,
                 sensing_network: CooperativeSensingNetwork):
        """
        Initialize plasma system controller.
        
        Args:
            decoy_generator: Plasma decoy generator
            sensing_network: Cooperative sensing network
        """
        self.decoy_generator = decoy_generator
        self.sensing_network = sensing_network
        self.logger = get_engine_logger('sensors.plasma.controller')
        
        # Mission parameters
        self.mission_active = False
        self.mission_start_time = 0.0
        self.deployment_pattern = "grid"
        
    def deploy_orb_network(self, center_position: Tuple[float, float, float],
                          num_orbs: int, spacing: float,
                          pattern: str = "grid") -> List[str]:
        """
        Deploy a network of plasma orbs in specified pattern.
        
        Args:
            center_position: Center position for deployment
            num_orbs: Number of orbs to deploy
            spacing: Spacing between orbs (m)
            pattern: Deployment pattern ("grid", "circle", "line")
            
        Returns:
            List of created orb IDs
        """
        orb_ids = []
        
        if pattern == "grid":
            # Square grid pattern
            grid_size = int(math.ceil(math.sqrt(num_orbs)))
            
            for i in range(num_orbs):
                row = i // grid_size
                col = i % grid_size
                
                # Center the grid
                offset_x = (col - grid_size/2 + 0.5) * spacing
                offset_y = (row - grid_size/2 + 0.5) * spacing
                
                position = (
                    center_position[0] + offset_x,
                    center_position[1] + offset_y,
                    center_position[2]
                )
                
                orb_id = self.decoy_generator.create_plasma_orb(
                    position=position,
                    velocity=(0, 0, 0),
                    radius=5.0,  # 5m radius
                    power_fraction=0.8,
                    lifetime=300.0  # 5 minutes
                )
                orb_ids.append(orb_id)
        
        elif pattern == "circle":
            # Circular pattern
            for i in range(num_orbs):
                angle = 2 * math.pi * i / num_orbs
                
                position = (
                    center_position[0] + spacing * math.cos(angle),
                    center_position[1] + spacing * math.sin(angle),
                    center_position[2]
                )
                
                orb_id = self.decoy_generator.create_plasma_orb(
                    position=position,
                    velocity=(0, 0, 0),
                    radius=5.0,
                    power_fraction=0.8,
                    lifetime=300.0
                )
                orb_ids.append(orb_id)
        
        elif pattern == "line":
            # Linear pattern
            for i in range(num_orbs):
                offset = (i - num_orbs/2 + 0.5) * spacing
                
                position = (
                    center_position[0] + offset,
                    center_position[1],
                    center_position[2]
                )
                
                orb_id = self.decoy_generator.create_plasma_orb(
                    position=position,
                    velocity=(0, 0, 0),
                    radius=5.0,
                    power_fraction=0.8,
                    lifetime=300.0
                )
                orb_ids.append(orb_id)
        
        self.logger.info(f"Deployed {len(orb_ids)} plasma orbs in {pattern} pattern "
                        f"centered at {center_position}")
        
        return orb_ids
    
    def execute_mission_cycle(self, dt: float, targets: List[Tuple[float, float, float]] = None) -> Dict[str, Any]:
        """
        Execute one mission cycle update.
        
        Args:
            dt: Time step (s)
            targets: Optional list of target positions for detection
            
        Returns:
            Mission status and results
        """
        # Update orb states
        self.decoy_generator.update_orb_states(dt)
        
        # Update network topology
        self.sensing_network.update_network_topology(self.decoy_generator.active_orbs)
        
        # Perform target detection if targets provided
        detections = {}
        fused_detections = []
        
        if targets:
            detections = self.sensing_network.simulate_target_detection(
                self.decoy_generator.active_orbs, targets
            )
            fused_detections = self.sensing_network.fuse_sensor_data(detections)
        
        # Calculate system status
        active_orbs = self.decoy_generator.get_active_orb_count()
        total_power = self.decoy_generator.get_total_power_consumption()
        
        return {
            'active_orbs': active_orbs,
            'total_power_w': total_power,
            'raw_detections': len(sum(detections.values(), [])),
            'fused_detections': len(fused_detections),
            'network_nodes': len(self.sensing_network.network_topology),
            'mission_time': time.time() - self.mission_start_time if self.mission_active else 0,
            'detections': fused_detections
        }
    
    def start_mission(self) -> None:
        """Start mission operations."""
        self.mission_active = True
        self.mission_start_time = time.time()
        self.logger.info("Plasma system mission started")
    
    def stop_mission(self) -> None:
        """Stop mission operations."""
        self.mission_active = False
        # Deactivate all orbs
        for orb in self.decoy_generator.active_orbs.values():
            orb.state = PlasmaState.DECAYING
            orb.lifetime_remaining = 5.0  # 5 second decay
        
        self.logger.info("Plasma system mission stopped")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        power_req = self.decoy_generator.calculate_power_requirements(
            num_orbs=self.decoy_generator.get_active_orb_count(),
            orb_radius=5.0,
            lifetime=300.0
        )
        
        return {
            'mission_active': self.mission_active,
            'active_orbs': self.decoy_generator.get_active_orb_count(),
            'total_orbs_created': self.decoy_generator.orb_counter,
            'current_power_w': self.decoy_generator.get_total_power_consumption(),
            'power_requirements': power_req,
            'network_connectivity': len(self.sensing_network.network_topology),
            'plasma_type': self.decoy_generator.config.plasma_type.name
        }