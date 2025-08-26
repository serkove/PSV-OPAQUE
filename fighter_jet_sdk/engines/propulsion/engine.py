"""Propulsion Engine for powerplant integration."""

from typing import Any, Dict, Optional, List
from ...common.interfaces import BaseEngine
from ...core.logging import get_engine_logger
from .engine_performance_model import (
    EnginePerformanceModel, EngineSpecification, EngineOperatingPoint, EngineType
)
from .intake_designer import IntakeDesigner, IntakeType, FlowConditions
from .thermal_manager import ThermalManager, ThermalLoad, CoolantType


class PropulsionEngine(BaseEngine):
    """Engine for propulsion system design and analysis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Propulsion Engine."""
        super().__init__(config)
        self.logger = get_engine_logger('propulsion')
        
        # Engine performance models
        self.performance_models: Dict[str, EnginePerformanceModel] = {}
        self.engine_specifications: Dict[str, EngineSpecification] = {}
        
        # Intake designer
        self.intake_designer = IntakeDesigner()
        
        # Thermal manager
        self.thermal_manager = ThermalManager()
        
        # Load default engine configurations
        self._load_default_engines()
    
    def initialize(self) -> bool:
        """Initialize the Propulsion Engine."""
        try:
            self.logger.info("Initializing Propulsion Engine")
            
            # Initialize performance models for all engines
            for engine_id, spec in self.engine_specifications.items():
                self.performance_models[engine_id] = EnginePerformanceModel(spec)
                self.logger.info(f"Initialized performance model for {spec.name}")
            
            self.initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Propulsion Engine: {e}")
            return False
    
    def _load_default_engines(self) -> None:
        """Load default engine specifications."""
        # F119-PW-100 (F-22 Raptor engine)
        f119_spec = EngineSpecification(
            engine_id="f119_pw_100",
            name="F119-PW-100",
            engine_type=EngineType.AFTERBURNING_TURBOFAN,
            max_thrust_sea_level=116000.0,  # 116 kN
            max_thrust_altitude=156000.0,   # 156 kN with afterburner
            design_altitude=11000.0,
            design_mach=1.8,
            bypass_ratio=0.3,
            pressure_ratio=35.0,
            turbine_inlet_temperature=1977.0,  # K
            mass=1800.0,  # kg
            length=5.2,   # m
            diameter=1.17, # m
            afterburner_thrust_multiplier=1.34
        )
        
        # F135-PW-100 (F-35 Lightning II engine)
        f135_spec = EngineSpecification(
            engine_id="f135_pw_100",
            name="F135-PW-100",
            engine_type=EngineType.AFTERBURNING_TURBOFAN,
            max_thrust_sea_level=125000.0,  # 125 kN
            max_thrust_altitude=191000.0,   # 191 kN with afterburner
            design_altitude=10000.0,
            design_mach=1.6,
            bypass_ratio=0.57,
            pressure_ratio=28.0,
            turbine_inlet_temperature=2000.0,  # K
            mass=1700.0,  # kg
            length=5.6,   # m
            diameter=1.17, # m
            afterburner_thrust_multiplier=1.53
        )
        
        # Hypothetical advanced ramjet for high-speed applications
        advanced_ramjet_spec = EngineSpecification(
            engine_id="advanced_ramjet_001",
            name="Advanced Ramjet Engine",
            engine_type=EngineType.RAMJET,
            max_thrust_sea_level=0.0,
            max_thrust_altitude=200000.0,   # 200 kN at design point
            design_altitude=18000.0,
            design_mach=4.0,
            pressure_ratio=1.0,
            turbine_inlet_temperature=2200.0,  # K
            mass=1200.0,  # kg
            length=4.0,   # m
            diameter=1.0  # m
        )
        
        # Variable cycle engine concept
        variable_cycle_spec = EngineSpecification(
            engine_id="variable_cycle_001",
            name="Advanced Variable Cycle Engine",
            engine_type=EngineType.VARIABLE_CYCLE,
            max_thrust_sea_level=140000.0,  # 140 kN
            max_thrust_altitude=200000.0,   # 200 kN optimized
            design_altitude=12000.0,
            design_mach=2.2,
            bypass_ratio=0.4,  # Variable in operation
            pressure_ratio=40.0,
            turbine_inlet_temperature=2100.0,  # K
            mass=1900.0,  # kg
            length=5.8,   # m
            diameter=1.2,  # m
            afterburner_thrust_multiplier=1.4,
            variable_cycle_modes=["efficient_cruise", "high_speed", "supercruise"]
        )
        
        self.engine_specifications = {
            f119_spec.engine_id: f119_spec,
            f135_spec.engine_id: f135_spec,
            advanced_ramjet_spec.engine_id: advanced_ramjet_spec,
            variable_cycle_spec.engine_id: variable_cycle_spec
        }
    
    def validate_input(self, data: Any) -> bool:
        """Validate input data for Propulsion Engine."""
        if not isinstance(data, dict):
            return False
        
        required_fields = ['engine_id', 'operating_conditions']
        return all(field in data for field in required_fields)
    
    def process(self, data: Any) -> Any:
        """Process propulsion operations."""
        if not self.validate_input(data):
            raise ValueError("Invalid input data for propulsion engine")
        
        engine_id = data['engine_id']
        if engine_id not in self.performance_models:
            raise ValueError(f"Unknown engine ID: {engine_id}")
        
        model = self.performance_models[engine_id]
        operating_conditions = data['operating_conditions']
        
        # Create operating point
        operating_point = EngineOperatingPoint(
            altitude=operating_conditions.get('altitude', 0.0),
            mach_number=operating_conditions.get('mach_number', 0.0),
            throttle_setting=operating_conditions.get('throttle_setting', 1.0),
            afterburner_engaged=operating_conditions.get('afterburner_engaged', False)
        )
        
        # Calculate performance
        thrust = model.calculate_thrust(operating_point)
        fuel_consumption = model.calculate_fuel_consumption(operating_point)
        
        # Calculate thrust-to-weight ratio if aircraft mass provided
        twr = None
        if 'aircraft_mass' in data:
            num_engines = data.get('num_engines', 1)
            twr = model.calculate_thrust_to_weight_ratio(
                data['aircraft_mass'], operating_point, num_engines
            )
        
        return {
            'engine_id': engine_id,
            'thrust': thrust,
            'fuel_consumption': fuel_consumption,
            'thrust_to_weight_ratio': twr,
            'operating_point': {
                'altitude': operating_point.altitude,
                'mach_number': operating_point.mach_number,
                'throttle_setting': operating_point.throttle_setting,
                'afterburner_engaged': operating_point.afterburner_engaged
            }
        }
    
    def get_available_engines(self) -> List[Dict[str, Any]]:
        """Get list of available engine specifications."""
        engines = []
        for engine_id, spec in self.engine_specifications.items():
            engines.append({
                'engine_id': engine_id,
                'name': spec.name,
                'type': spec.engine_type.value,
                'max_thrust_sl': spec.max_thrust_sea_level,
                'mass': spec.mass,
                'afterburner_capable': spec.engine_type in [
                    EngineType.AFTERBURNING_TURBOJET,
                    EngineType.AFTERBURNING_TURBOFAN
                ]
            })
        return engines
    
    def calculate_mission_fuel_consumption(self, engine_id: str, flight_profile: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate fuel consumption for a complete mission profile."""
        if engine_id not in self.performance_models:
            raise ValueError(f"Unknown engine ID: {engine_id}")
        
        model = self.performance_models[engine_id]
        
        # Convert flight profile to operating points
        operating_points = []
        flight_times = []
        
        for segment in flight_profile:
            op_point = EngineOperatingPoint(
                altitude=segment['altitude'],
                mach_number=segment['mach_number'],
                throttle_setting=segment['throttle_setting'],
                afterburner_engaged=segment.get('afterburner_engaged', False)
            )
            operating_points.append(op_point)
            flight_times.append(segment['duration'])  # seconds
        
        # Calculate total fuel consumption
        total_fuel = model.calculate_range_fuel_consumption(operating_points, flight_times)
        
        # Calculate segment-by-segment breakdown
        segment_breakdown = []
        for i, (op_point, duration) in enumerate(zip(operating_points, flight_times)):
            fuel_flow = model.calculate_fuel_consumption(op_point)
            segment_fuel = fuel_flow * duration
            
            segment_breakdown.append({
                'segment': i + 1,
                'altitude': op_point.altitude,
                'mach_number': op_point.mach_number,
                'duration': duration,
                'fuel_flow_rate': fuel_flow,
                'segment_fuel': segment_fuel
            })
        
        return {
            'engine_id': engine_id,
            'total_fuel_consumption': total_fuel,
            'mission_duration': sum(flight_times),
            'segment_breakdown': segment_breakdown
        }
    
    def optimize_cruise_performance(self, engine_id: str, aircraft_mass: float,
                                  altitude_range: tuple = (8000.0, 15000.0),
                                  mach_range: tuple = (0.7, 1.2)) -> Dict[str, Any]:
        """Optimize cruise conditions for minimum fuel consumption."""
        if engine_id not in self.performance_models:
            raise ValueError(f"Unknown engine ID: {engine_id}")
        
        model = self.performance_models[engine_id]
        
        best_altitude, best_mach, best_sfc = model.optimize_cruise_conditions(
            altitude_range, mach_range, aircraft_mass
        )
        
        # Calculate performance at optimal conditions
        optimal_point = EngineOperatingPoint(
            altitude=best_altitude,
            mach_number=best_mach,
            throttle_setting=0.8  # Typical cruise setting
        )
        
        thrust = model.calculate_thrust(optimal_point)
        fuel_flow = model.calculate_fuel_consumption(optimal_point)
        
        return {
            'engine_id': engine_id,
            'optimal_altitude': best_altitude,
            'optimal_mach': best_mach,
            'optimal_sfc': best_sfc,
            'cruise_thrust': thrust,
            'cruise_fuel_flow': fuel_flow
        }
    
    def get_engine_performance_envelope(self, engine_id: str) -> Dict[str, Any]:
        """Get complete performance envelope for specified engine."""
        if engine_id not in self.performance_models:
            raise ValueError(f"Unknown engine ID: {engine_id}")
        
        model = self.performance_models[engine_id]
        return model.get_performance_envelope()
    
    def validate_engine_for_mission(self, engine_id: str, mission_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Validate if engine meets mission requirements."""
        if engine_id not in self.performance_models:
            raise ValueError(f"Unknown engine ID: {engine_id}")
        
        model = self.performance_models[engine_id]
        spec = self.engine_specifications[engine_id]
        
        validation_results = {
            'engine_id': engine_id,
            'meets_requirements': True,
            'issues': [],
            'recommendations': []
        }
        
        # Check altitude requirements
        max_altitude = mission_requirements.get('max_altitude', 0)
        if max_altitude > 25000:  # Engine operational limit
            validation_results['meets_requirements'] = False
            validation_results['issues'].append(f"Required altitude {max_altitude}m exceeds engine limit")
        
        # Check Mach number requirements
        max_mach = mission_requirements.get('max_mach', 0)
        engine_max_mach = model._get_max_mach()
        if max_mach > engine_max_mach:
            validation_results['meets_requirements'] = False
            validation_results['issues'].append(f"Required Mach {max_mach} exceeds engine limit {engine_max_mach}")
        
        # Check thrust requirements
        min_thrust = mission_requirements.get('min_thrust', 0)
        # For ramjets/scramjets, use altitude thrust instead of sea level
        if spec.engine_type in [EngineType.RAMJET, EngineType.SCRAMJET]:
            max_available_thrust = spec.max_thrust_altitude
        else:
            max_available_thrust = spec.max_thrust_sea_level
            
        if min_thrust > max_available_thrust:
            validation_results['meets_requirements'] = False
            validation_results['issues'].append(f"Required thrust {min_thrust}N exceeds engine capability {max_available_thrust}N")
        
        # Add recommendations
        if spec.engine_type == EngineType.RAMJET and max_mach < 2.0:
            validation_results['recommendations'].append("Consider turbofan for subsonic/low supersonic missions")
        
        if spec.engine_type in [EngineType.TURBOJET, EngineType.TURBOFAN] and max_mach > 3.0:
            validation_results['recommendations'].append("Consider ramjet/scramjet for hypersonic missions")
        
        return validation_results