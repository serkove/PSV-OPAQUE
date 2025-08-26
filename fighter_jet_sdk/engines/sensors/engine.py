"""Sensors Engine for advanced detection systems."""

from typing import Any, Dict, Optional, List, Tuple
from ...common.interfaces import BaseEngine
from ...core.logging import get_engine_logger
from .aesa_radar_model import AESARadarModel, RadarConfiguration, RadarTarget
from .laser_systems import (
    LaserFilamentationSim, AdaptiveOpticsController, LaserInducedBreakdownSpectroscopy,
    LaserSafetyAnalyzer, LaserConfiguration, AdaptiveOpticsConfiguration,
    AtmosphericParameters
)
from .plasma_systems import (
    PlasmaDecoyGenerator, CooperativeSensingNetwork, PlasmaSystemController,
    PlasmaConfiguration, PlasmaType
)


class SensorsEngine(BaseEngine):
    """Engine for advanced sensor system modeling and analysis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Sensors Engine."""
        super().__init__(config)
        self.logger = get_engine_logger('sensors')
        
        # Sensor system components
        self.aesa_radars: Dict[str, AESARadarModel] = {}
        self.laser_systems: Dict[str, Dict[str, Any]] = {}
        self.plasma_systems: Dict[str, PlasmaSystemController] = {}
        
        # Atmospheric conditions
        self.atmospheric_params = AtmosphericParameters(
            visibility=20.0,  # km
            temperature=288.15,  # K
            pressure=101325,  # Pa
            humidity=50.0,  # %
            wind_speed=5.0,  # m/s
            turbulence_strength=1e-14,  # Cn²
        )
    
    def initialize(self) -> bool:
        """Initialize the Sensors Engine."""
        try:
            self.logger.info("Initializing Sensors Engine with advanced sensor systems")
            
            # Initialize default systems if config provided
            if self.config:
                self._initialize_from_config()
            
            self.initialized = True
            self.logger.info("Sensors Engine initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Sensors Engine: {e}")
            return False
    
    def _initialize_from_config(self):
        """Initialize sensor systems from configuration."""
        if 'aesa_radars' in self.config:
            for radar_id, radar_config in self.config['aesa_radars'].items():
                self.create_aesa_radar(radar_id, radar_config)
        
        if 'laser_systems' in self.config:
            for laser_id, laser_config in self.config['laser_systems'].items():
                self.create_laser_system(laser_id, laser_config)
        
        if 'plasma_systems' in self.config:
            for plasma_id, plasma_config in self.config['plasma_systems'].items():
                self.create_plasma_system(plasma_id, plasma_config)
    
    def create_aesa_radar(self, radar_id: str, config_dict: Dict[str, Any]) -> bool:
        """
        Create AESA radar system.
        
        Args:
            radar_id: Unique identifier for radar
            config_dict: Radar configuration parameters
            
        Returns:
            True if successful
        """
        try:
            radar_config = RadarConfiguration(**config_dict)
            radar = AESARadarModel(radar_config)
            
            # Validate configuration
            errors = radar.validate_configuration()
            if errors:
                self.logger.error(f"AESA radar {radar_id} configuration errors: {errors}")
                return False
            
            self.aesa_radars[radar_id] = radar
            self.logger.info(f"Created AESA radar system: {radar_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create AESA radar {radar_id}: {e}")
            return False
    
    def create_laser_system(self, laser_id: str, config_dict: Dict[str, Any]) -> bool:
        """
        Create laser-based sensor system.
        
        Args:
            laser_id: Unique identifier for laser system
            config_dict: Laser configuration parameters
            
        Returns:
            True if successful
        """
        try:
            # Handle laser_type enum conversion
            if 'laser_type' in config_dict and isinstance(config_dict['laser_type'], str):
                from .laser_systems import LaserType
                config_dict['laser_type'] = LaserType[config_dict['laser_type']]
            
            laser_config = LaserConfiguration(**config_dict)
            
            # Create laser subsystems
            filamentation_sim = LaserFilamentationSim(laser_config)
            libs_system = LaserInducedBreakdownSpectroscopy(laser_config)
            safety_analyzer = LaserSafetyAnalyzer()
            
            # Create adaptive optics if specified
            adaptive_optics = None
            if 'adaptive_optics' in config_dict:
                ao_config = AdaptiveOpticsConfiguration(**config_dict['adaptive_optics'])
                adaptive_optics = AdaptiveOpticsController(ao_config)
            
            self.laser_systems[laser_id] = {
                'config': laser_config,
                'filamentation': filamentation_sim,
                'libs': libs_system,
                'safety': safety_analyzer,
                'adaptive_optics': adaptive_optics
            }
            
            self.logger.info(f"Created laser system: {laser_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create laser system {laser_id}: {e}")
            return False
    
    def create_plasma_system(self, plasma_id: str, config_dict: Dict[str, Any]) -> bool:
        """
        Create plasma-based sensor/decoy system.
        
        Args:
            plasma_id: Unique identifier for plasma system
            config_dict: Plasma configuration parameters
            
        Returns:
            True if successful
        """
        try:
            # Extract communication_range before creating plasma config
            communication_range = config_dict.pop('communication_range', 10000.0)
            
            # Handle plasma_type enum conversion
            if 'plasma_type' in config_dict and isinstance(config_dict['plasma_type'], str):
                config_dict['plasma_type'] = PlasmaType[config_dict['plasma_type']]
            
            plasma_config = PlasmaConfiguration(**config_dict)
            
            # Create plasma subsystems
            decoy_generator = PlasmaDecoyGenerator(plasma_config)
            sensing_network = CooperativeSensingNetwork(
                communication_range=communication_range
            )
            
            controller = PlasmaSystemController(decoy_generator, sensing_network)
            
            self.plasma_systems[plasma_id] = controller
            self.logger.info(f"Created plasma system: {plasma_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create plasma system {plasma_id}: {e}")
            return False
    
    def detect_targets_aesa(self, radar_id: str, targets: List[RadarTarget],
                           beam_azimuth: float, beam_elevation: float) -> List[RadarTarget]:
        """
        Detect targets using AESA radar.
        
        Args:
            radar_id: AESA radar identifier
            targets: List of potential targets
            beam_azimuth: Beam azimuth angle (degrees)
            beam_elevation: Beam elevation angle (degrees)
            
        Returns:
            List of detected targets
        """
        if radar_id not in self.aesa_radars:
            self.logger.error(f"AESA radar {radar_id} not found")
            return []
        
        radar = self.aesa_radars[radar_id]
        return radar.detect_targets(targets, beam_azimuth, beam_elevation)
    
    def track_targets_aesa(self, radar_id: str, detected_targets: List[RadarTarget],
                          current_time: float) -> Dict[str, RadarTarget]:
        """
        Track targets using AESA radar.
        
        Args:
            radar_id: AESA radar identifier
            detected_targets: List of detected targets
            current_time: Current simulation time
            
        Returns:
            Dictionary of tracked targets
        """
        if radar_id not in self.aesa_radars:
            self.logger.error(f"AESA radar {radar_id} not found")
            return {}
        
        radar = self.aesa_radars[radar_id]
        return radar.track_targets(detected_targets, current_time)
    
    def analyze_material_libs(self, laser_id: str, target_material: str,
                             laser_power: float) -> Dict[str, Any]:
        """
        Analyze material composition using LIBS.
        
        Args:
            laser_id: Laser system identifier
            target_material: Target material type
            laser_power: Laser power for analysis
            
        Returns:
            Analysis results
        """
        if laser_id not in self.laser_systems:
            self.logger.error(f"Laser system {laser_id} not found")
            return {}
        
        laser_system = self.laser_systems[laser_id]
        libs = laser_system['libs']
        
        # Calculate plasma temperature
        plasma_temp = libs.calculate_plasma_temperature(laser_power, target_material)
        
        # Simulate spectrum for common elements
        elements = ['H', 'C', 'N', 'O', 'Fe']
        concentrations = [0.4, 0.2, 0.1, 0.2, 0.1]  # Example concentrations
        
        spectrum = libs.simulate_spectrum(plasma_temp, elements, concentrations)
        
        # Detect radioactive elements
        radioactive_elements = libs.detect_radioactive_elements(spectrum)
        
        return {
            'plasma_temperature_k': plasma_temp,
            'spectrum': spectrum,
            'radioactive_elements': radioactive_elements,
            'analysis_successful': len(spectrum) > 0
        }
    
    def deploy_plasma_network(self, plasma_id: str, center_position: Tuple[float, float, float],
                             num_orbs: int, spacing: float, pattern: str = "grid") -> List[str]:
        """
        Deploy plasma orb network.
        
        Args:
            plasma_id: Plasma system identifier
            center_position: Center position for deployment
            num_orbs: Number of orbs to deploy
            spacing: Spacing between orbs
            pattern: Deployment pattern
            
        Returns:
            List of created orb IDs
        """
        if plasma_id not in self.plasma_systems:
            self.logger.error(f"Plasma system {plasma_id} not found")
            return []
        
        controller = self.plasma_systems[plasma_id]
        return controller.deploy_orb_network(center_position, num_orbs, spacing, pattern)
    
    def update_atmospheric_conditions(self, visibility: float, temperature: float,
                                    pressure: float, humidity: float,
                                    wind_speed: float, turbulence_strength: float):
        """Update atmospheric conditions for all sensor systems."""
        self.atmospheric_params.visibility = visibility
        self.atmospheric_params.temperature = temperature
        self.atmospheric_params.pressure = pressure
        self.atmospheric_params.humidity = humidity
        self.atmospheric_params.wind_speed = wind_speed
        self.atmospheric_params.turbulence_strength = turbulence_strength
        
        self.logger.info(f"Updated atmospheric conditions: visibility={visibility}km, "
                        f"temp={temperature}K, pressure={pressure}Pa")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive sensor system status."""
        status = {
            'aesa_radars': {},
            'laser_systems': {},
            'plasma_systems': {},
            'atmospheric_conditions': {
                'visibility_km': self.atmospheric_params.visibility,
                'temperature_k': self.atmospheric_params.temperature,
                'pressure_pa': self.atmospheric_params.pressure,
                'humidity_percent': self.atmospheric_params.humidity,
                'wind_speed_ms': self.atmospheric_params.wind_speed,
                'turbulence_strength': self.atmospheric_params.turbulence_strength
            }
        }
        
        # AESA radar status
        for radar_id, radar in self.aesa_radars.items():
            status['aesa_radars'][radar_id] = radar.get_performance_metrics()
        
        # Laser system status
        for laser_id, laser_system in self.laser_systems.items():
            status['laser_systems'][laser_id] = {
                'wavelength_nm': laser_system['config'].wavelength * 1e9,
                'peak_power_w': laser_system['config'].peak_power,
                'laser_type': laser_system['config'].laser_type.name if hasattr(laser_system['config'].laser_type, 'name') else str(laser_system['config'].laser_type),
                'has_adaptive_optics': laser_system['adaptive_optics'] is not None
            }
        
        # Plasma system status
        for plasma_id, controller in self.plasma_systems.items():
            status['plasma_systems'][plasma_id] = controller.get_system_status()
        
        return status
    
    def validate_input(self, data: Any) -> bool:
        """Validate input data for Sensors Engine."""
        if not isinstance(data, dict):
            return False
        
        # Basic validation - could be expanded
        return True
    
    def process(self, data: Any) -> Any:
        """Process sensor operations."""
        if not self.validate_input(data):
            self.logger.error("Invalid input data for sensor processing")
            return None
        
        # Process based on operation type
        operation = data.get('operation', 'status')
        
        if operation == 'status':
            return self.get_system_status()
        elif operation == 'detect_targets':
            # Example target detection workflow
            return self._process_target_detection(data)
        else:
            self.logger.warning(f"Unknown operation: {operation}")
            return None
    
    def _process_target_detection(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process target detection across all sensor systems."""
        results = {
            'aesa_detections': {},
            'plasma_detections': {},
            'fused_results': []
        }
        
        targets = data.get('targets', [])
        if not targets:
            return results
        
        # Convert targets to RadarTarget objects for AESA
        radar_targets = []
        for target in targets:
            radar_target = RadarTarget(
                position=tuple(target['position']),
                velocity=tuple(target.get('velocity', [0, 0, 0])),
                rcs=target.get('rcs', 1.0),
                target_id=target.get('id', f"T{len(radar_targets):03d}")
            )
            radar_targets.append(radar_target)
        
        # AESA radar detection
        for radar_id, radar in self.aesa_radars.items():
            beam_az = data.get('beam_azimuth', 0.0)
            beam_el = data.get('beam_elevation', 0.0)
            
            detected = radar.detect_targets(radar_targets, beam_az, beam_el)
            results['aesa_detections'][radar_id] = [
                {
                    'target_id': t.target_id,
                    'position': t.position,
                    'rcs': t.rcs
                } for t in detected
            ]
        
        # Plasma system detection
        for plasma_id, controller in self.plasma_systems.items():
            target_positions = [target['position'] for target in targets]
            mission_result = controller.execute_mission_cycle(1.0, target_positions)
            results['plasma_detections'][plasma_id] = mission_result['detections']
        
        return results