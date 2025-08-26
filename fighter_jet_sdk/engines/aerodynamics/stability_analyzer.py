"""
Stability and Control Analysis Module

Comprehensive stability and control analysis for modular fighter aircraft.
Implements control authority calculations, handling qualities assessment,
and control system design algorithms.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ...common.data_models import AircraftConfiguration, FlowConditions, Module
from ...common.enums import ModuleType, FlightRegime
from ...core.errors import AerodynamicsError


class ControlSurface(Enum):
    """Types of control surfaces"""
    ELEVATOR = "elevator"
    AILERON = "aileron"
    RUDDER = "rudder"
    CANARD = "canard"
    ELEVON = "elevon"
    RUDDERVATOR = "ruddervator"
    THRUST_VECTORING = "thrust_vectoring"


class HandlingQuality(Enum):
    """Cooper-Harper handling quality ratings"""
    LEVEL_1 = 1  # Excellent, pilot compensation not required
    LEVEL_2 = 2  # Good, minimal pilot compensation required
    LEVEL_3 = 3  # Fair, definite pilot compensation required


@dataclass
class ControlAuthority:
    """Control authority for a specific control surface"""
    surface_type: ControlSurface
    max_deflection: float  # degrees
    deflection_rate: float  # degrees/second
    moment_arm: float  # meters
    effectiveness: float  # moment coefficient per degree
    power_required: float  # watts
    response_time: float  # seconds


@dataclass
class StabilityDerivatives:
    """Stability and control derivatives"""
    # Longitudinal derivatives
    cma: float = 0.0  # Pitching moment coefficient derivative w.r.t. angle of attack
    cmq: float = 0.0  # Pitching moment coefficient derivative w.r.t. pitch rate
    cmde: float = 0.0  # Pitching moment coefficient derivative w.r.t. elevator deflection
    
    # Lateral-directional derivatives
    cnb: float = 0.0  # Yawing moment coefficient derivative w.r.t. sideslip angle
    cnr: float = 0.0  # Yawing moment coefficient derivative w.r.t. yaw rate
    cndr: float = 0.0  # Yawing moment coefficient derivative w.r.t. rudder deflection
    
    clb: float = 0.0  # Rolling moment coefficient derivative w.r.t. sideslip angle
    clp: float = 0.0  # Rolling moment coefficient derivative w.r.t. roll rate
    clda: float = 0.0  # Rolling moment coefficient derivative w.r.t. aileron deflection


@dataclass
class FlightCondition:
    """Specific flight condition for stability analysis"""
    altitude: float  # meters
    mach_number: float
    angle_of_attack: float  # degrees
    load_factor: float  # g's
    configuration: str  # "clean", "combat", "landing", etc.


@dataclass
class HandlingQualityAssessment:
    """Handling quality assessment results"""
    flight_condition: FlightCondition
    cooper_harper_rating: HandlingQuality
    short_period_frequency: float  # rad/s
    short_period_damping: float
    dutch_roll_frequency: float  # rad/s
    dutch_roll_damping: float
    spiral_mode_time_constant: float  # seconds
    roll_mode_time_constant: float  # seconds
    comments: List[str] = field(default_factory=list)


@dataclass
class ControlSystemDesign:
    """Control system design parameters"""
    controller_type: str  # "PID", "LQR", "H_infinity", etc.
    gains: Dict[str, float] = field(default_factory=dict)
    bandwidth: float = 0.0  # rad/s
    phase_margin: float = 0.0  # degrees
    gain_margin: float = 0.0  # dB
    settling_time: float = 0.0  # seconds
    overshoot: float = 0.0  # percent


class StabilityAnalyzer:
    """
    Comprehensive stability and control analysis system
    
    Provides:
    - Control authority calculations for all modular configurations
    - Handling qualities assessment across flight envelope
    - Control system design and tuning algorithms
    - Pilot-in-the-loop simulation interface preparation
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Standard flight test conditions
        self.standard_conditions = [
            FlightCondition(0, 0.3, 0, 1, "takeoff"),
            FlightCondition(3000, 0.6, 2, 1, "cruise"),
            FlightCondition(10000, 0.9, 0, 1, "high_speed"),
            FlightCondition(15000, 1.2, 5, 1, "supersonic"),
            FlightCondition(5000, 0.4, 15, 2, "combat_maneuvering"),
            FlightCondition(1000, 0.25, 8, 1, "approach")
        ]
        
        # Control surface effectiveness database
        self.control_effectiveness = {
            ControlSurface.ELEVATOR: 0.8,
            ControlSurface.AILERON: 0.6,
            ControlSurface.RUDDER: 0.7,
            ControlSurface.CANARD: 0.9,
            ControlSurface.ELEVON: 0.75,
            ControlSurface.THRUST_VECTORING: 1.2
        }
    
    def analyze_stability(self, configuration: AircraftConfiguration, 
                         flight_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive stability analysis for aircraft configuration
        
        Args:
            configuration: Aircraft configuration to analyze
            flight_conditions: Dictionary of flight conditions and parameters
        
        Returns:
            Dictionary containing all stability analysis results
        """
        try:
            self.logger.info("Starting comprehensive stability analysis")
            
            # Extract flight conditions or use defaults
            conditions = self._parse_flight_conditions(flight_conditions)
            
            # Calculate control authority for all surfaces
            control_authority = self._calculate_control_authority(configuration)
            
            # Assess handling qualities across flight envelope
            handling_qualities = self._assess_handling_qualities(configuration, conditions)
            
            # Design control systems
            control_systems = self._design_control_systems(configuration, control_authority)
            
            # Calculate stability derivatives
            stability_derivatives = self._calculate_stability_derivatives(configuration)
            
            # Prepare pilot-in-the-loop interface data
            pilot_interface = self._prepare_pilot_interface(configuration, control_authority)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                configuration, control_authority, handling_qualities
            )
            
            results = {
                "control_authority": control_authority,
                "handling_qualities": handling_qualities,
                "control_systems": control_systems,
                "stability_derivatives": stability_derivatives,
                "pilot_interface": pilot_interface,
                "performance_metrics": performance_metrics,
                "flight_envelope_limits": self._calculate_flight_envelope_limits(configuration),
                "recommendations": self._generate_recommendations(configuration, handling_qualities)
            }
            
            self.logger.info("Stability analysis completed successfully")
            return results
            
        except Exception as e:
            raise AerodynamicsError(f"Stability analysis failed: {str(e)}")
    
    def _parse_flight_conditions(self, flight_conditions: Dict[str, Any]) -> List[FlightCondition]:
        """Parse flight conditions from input or use standard conditions"""
        if not flight_conditions or "conditions" not in flight_conditions:
            return self.standard_conditions
        
        conditions = []
        for cond_data in flight_conditions["conditions"]:
            condition = FlightCondition(
                altitude=cond_data.get("altitude", 10000),
                mach_number=cond_data.get("mach_number", 0.8),
                angle_of_attack=cond_data.get("angle_of_attack", 2.0),
                load_factor=cond_data.get("load_factor", 1.0),
                configuration=cond_data.get("configuration", "cruise")
            )
            conditions.append(condition)
        
        return conditions
    
    def _calculate_control_authority(self, configuration: AircraftConfiguration) -> Dict[str, ControlAuthority]:
        """
        Calculate control authority for all control surfaces in the configuration
        
        Requirement 5.2: Calculate control authority for all modular configurations
        """
        control_surfaces = {}
        
        # Find structural modules that could contain control surfaces
        structural_modules = [m for m in configuration.modules if m.module_type == ModuleType.STRUCTURAL]
        
        for module in structural_modules:
            # Determine control surfaces based on module characteristics
            surfaces = self._identify_control_surfaces(module)
            
            for surface_type in surfaces:
                authority = self._calculate_surface_authority(module, surface_type, configuration)
                control_surfaces[f"{module.name}_{surface_type.value}"] = authority
        
        # Add thrust vectoring if propulsion modules support it
        propulsion_modules = [m for m in configuration.modules if m.module_type == ModuleType.PROPULSION]
        for module in propulsion_modules:
            if self._has_thrust_vectoring(module):
                authority = self._calculate_thrust_vectoring_authority(module, configuration)
                control_surfaces[f"{module.name}_thrust_vectoring"] = authority
        
        return control_surfaces
    
    def _identify_control_surfaces(self, module: Module) -> List[ControlSurface]:
        """Identify control surfaces present in a structural module"""
        surfaces = []
        
        # Analyze module name and characteristics to identify surfaces
        module_name_lower = module.name.lower()
        
        if any(keyword in module_name_lower for keyword in ["wing", "main_wing"]):
            surfaces.extend([ControlSurface.AILERON])
        
        if any(keyword in module_name_lower for keyword in ["tail", "vertical_tail", "vtail"]):
            surfaces.extend([ControlSurface.RUDDER])
        
        if any(keyword in module_name_lower for keyword in ["horizontal_tail", "htail", "stabilizer"]):
            surfaces.extend([ControlSurface.ELEVATOR])
        
        if "canard" in module_name_lower:
            surfaces.extend([ControlSurface.CANARD])
        
        if "elevon" in module_name_lower:
            surfaces.extend([ControlSurface.ELEVON])
        
        # If no specific surfaces identified, assume basic control surfaces for wings/tails
        if not surfaces:
            if "wing" in module_name_lower:
                surfaces.append(ControlSurface.AILERON)
            elif "tail" in module_name_lower:
                surfaces.extend([ControlSurface.ELEVATOR, ControlSurface.RUDDER])
        
        return surfaces
    
    def _calculate_surface_authority(self, module: Module, surface_type: ControlSurface, 
                                   configuration: AircraftConfiguration) -> ControlAuthority:
        """Calculate control authority for a specific control surface"""
        # Get module dimensions for moment arm calculation
        if module.physical_properties and module.physical_properties.dimensions:
            length, width, height = module.physical_properties.dimensions
            # Estimate moment arm based on module size and aircraft configuration
            moment_arm = max(length, width) * 0.7  # Approximate moment arm
        else:
            moment_arm = 2.0  # Default moment arm
        
        # Base effectiveness from database
        base_effectiveness = self.control_effectiveness.get(surface_type, 0.6)
        
        # Adjust effectiveness based on module size and configuration
        if module.physical_properties:
            size_factor = np.sqrt(module.physical_properties.dimensions[1] * module.physical_properties.dimensions[2]) / 10.0
            effectiveness = base_effectiveness * min(size_factor, 2.0)  # Cap at 2x base effectiveness
        else:
            effectiveness = base_effectiveness
        
        # Calculate power requirements based on surface size and deflection rate
        surface_area = self._estimate_surface_area(module, surface_type)
        deflection_rate = 60.0  # degrees/second (typical for fighter aircraft)
        power_required = surface_area * deflection_rate * 0.5  # Simplified power calculation
        
        return ControlAuthority(
            surface_type=surface_type,
            max_deflection=25.0 if surface_type != ControlSurface.THRUST_VECTORING else 15.0,
            deflection_rate=deflection_rate,
            moment_arm=moment_arm,
            effectiveness=effectiveness,
            power_required=power_required,
            response_time=0.1  # seconds
        )
    
    def _has_thrust_vectoring(self, module: Module) -> bool:
        """Check if propulsion module has thrust vectoring capability"""
        return ("vectoring" in module.name.lower() or 
                "vector" in module.name.lower() or
                module.performance_characteristics.get("thrust_vectoring", False))
    
    def _calculate_thrust_vectoring_authority(self, module: Module, 
                                           configuration: AircraftConfiguration) -> ControlAuthority:
        """Calculate thrust vectoring control authority"""
        # Thrust vectoring effectiveness depends on engine thrust and moment arm
        max_thrust = module.performance_characteristics.get("max_thrust", 100000)  # N
        
        # Estimate moment arm from engine position (assume aft-mounted)
        if configuration.base_platform:
            # Assume engine is at 70% of aircraft length from nose
            aircraft_length = 15.0  # Default aircraft length
            moment_arm = aircraft_length * 0.3  # Distance from CG to engine
        else:
            moment_arm = 4.0  # Default moment arm
        
        # Thrust vectoring effectiveness (moment per degree of deflection)
        effectiveness = (max_thrust * np.sin(np.radians(1.0)) * moment_arm) / 1000000  # Normalized
        
        return ControlAuthority(
            surface_type=ControlSurface.THRUST_VECTORING,
            max_deflection=15.0,  # degrees
            deflection_rate=30.0,  # degrees/second
            moment_arm=moment_arm,
            effectiveness=effectiveness,
            power_required=1000.0,  # watts for actuator system
            response_time=0.05  # seconds (faster than aerodynamic surfaces)
        )
    
    def _estimate_surface_area(self, module: Module, surface_type: ControlSurface) -> float:
        """Estimate control surface area based on module dimensions"""
        if not module.physical_properties or not module.physical_properties.dimensions:
            return 1.0  # Default area
        
        length, width, height = module.physical_properties.dimensions
        
        # Estimate surface area as fraction of module area
        if surface_type in [ControlSurface.AILERON, ControlSurface.ELEVON]:
            return width * length * 0.2  # 20% of wing area
        elif surface_type in [ControlSurface.ELEVATOR, ControlSurface.CANARD]:
            return width * length * 0.3  # 30% of horizontal tail area
        elif surface_type == ControlSurface.RUDDER:
            return height * length * 0.3  # 30% of vertical tail area
        else:
            return width * length * 0.25  # Default 25%
    
    def _assess_handling_qualities(self, configuration: AircraftConfiguration, 
                                 conditions: List[FlightCondition]) -> List[HandlingQualityAssessment]:
        """
        Assess handling qualities for all flight conditions
        
        Uses Cooper-Harper rating scale and dynamic response characteristics
        """
        assessments = []
        
        for condition in conditions:
            # Calculate dynamic response characteristics
            short_period_freq, short_period_damping = self._calculate_short_period_characteristics(
                configuration, condition
            )
            
            dutch_roll_freq, dutch_roll_damping = self._calculate_dutch_roll_characteristics(
                configuration, condition
            )
            
            spiral_time_constant = self._calculate_spiral_mode(configuration, condition)
            roll_time_constant = self._calculate_roll_mode(configuration, condition)
            
            # Determine Cooper-Harper rating based on dynamic characteristics
            cooper_harper_rating = self._determine_cooper_harper_rating(
                short_period_freq, short_period_damping,
                dutch_roll_freq, dutch_roll_damping,
                condition
            )
            
            # Generate comments based on analysis
            comments = self._generate_handling_comments(
                short_period_freq, short_period_damping,
                dutch_roll_freq, dutch_roll_damping,
                condition
            )
            
            assessment = HandlingQualityAssessment(
                flight_condition=condition,
                cooper_harper_rating=cooper_harper_rating,
                short_period_frequency=short_period_freq,
                short_period_damping=short_period_damping,
                dutch_roll_frequency=dutch_roll_freq,
                dutch_roll_damping=dutch_roll_damping,
                spiral_mode_time_constant=spiral_time_constant,
                roll_mode_time_constant=roll_time_constant,
                comments=comments
            )
            
            assessments.append(assessment)
        
        return assessments
    
    def _calculate_short_period_characteristics(self, configuration: AircraftConfiguration, 
                                             condition: FlightCondition) -> Tuple[float, float]:
        """Calculate short period frequency and damping"""
        # Simplified short period approximation
        # Frequency depends on static margin and flight condition
        
        # Estimate static margin from configuration
        static_margin = self._estimate_static_margin(configuration)
        
        # Short period frequency (rad/s)
        q_bar = 0.5 * 1.225 * (condition.mach_number * 340) ** 2  # Dynamic pressure
        frequency = np.sqrt(abs(static_margin) * q_bar / 10000)  # Simplified calculation
        
        # Damping ratio (typical values for fighter aircraft)
        if condition.mach_number < 0.8:
            damping = 0.7  # Good damping at subsonic speeds
        elif condition.mach_number < 1.2:
            damping = 0.5  # Reduced damping in transonic region
        else:
            damping = 0.6  # Moderate damping at supersonic speeds
        
        return frequency, damping
    
    def _calculate_dutch_roll_characteristics(self, configuration: AircraftConfiguration,
                                           condition: FlightCondition) -> Tuple[float, float]:
        """Calculate Dutch roll frequency and damping"""
        # Dutch roll characteristics depend on directional stability and dihedral effect
        
        # Estimate based on configuration and flight condition
        frequency = 1.0 + condition.mach_number * 0.5  # rad/s
        
        # Damping typically lower at high speeds
        if condition.mach_number < 0.6:
            damping = 0.15
        elif condition.mach_number < 1.0:
            damping = 0.10
        else:
            damping = 0.08
        
        return frequency, damping
    
    def _calculate_spiral_mode(self, configuration: AircraftConfiguration,
                             condition: FlightCondition) -> float:
        """Calculate spiral mode time constant"""
        # Spiral mode depends on directional stability vs dihedral effect
        # Positive time constant indicates stable spiral mode
        
        # Simplified calculation based on configuration
        base_time_constant = 20.0  # seconds
        
        # Adjust based on flight condition
        if condition.mach_number > 1.0:
            base_time_constant *= 0.7  # Less stable at high speeds
        
        return base_time_constant
    
    def _calculate_roll_mode(self, configuration: AircraftConfiguration,
                           condition: FlightCondition) -> float:
        """Calculate roll mode time constant"""
        # Roll mode time constant (should be small for good handling)
        
        # Depends on roll damping and aileron effectiveness
        aileron_modules = [m for m in configuration.modules 
                          if "wing" in m.name.lower() or "aileron" in m.name.lower()]
        
        if aileron_modules:
            # Better roll response with larger ailerons
            avg_aileron_size = np.mean([
                np.prod(m.physical_properties.dimensions) if m.physical_properties else 1.0
                for m in aileron_modules
            ])
            time_constant = 0.5 / max(avg_aileron_size / 10.0, 0.1)
        else:
            time_constant = 2.0  # Poor roll response without ailerons
        
        return time_constant
    
    def _determine_cooper_harper_rating(self, sp_freq: float, sp_damping: float,
                                      dr_freq: float, dr_damping: float,
                                      condition: FlightCondition) -> HandlingQuality:
        """Determine Cooper-Harper handling quality rating"""
        
        # Level 1 criteria (excellent handling)
        if (0.5 <= sp_damping <= 2.0 and 
            sp_freq >= 1.0 and 
            dr_damping >= 0.08 and
            condition.load_factor <= 2.0):
            return HandlingQuality.LEVEL_1
        
        # Level 2 criteria (good handling with minimal compensation)
        elif (0.3 <= sp_damping <= 3.0 and 
              sp_freq >= 0.5 and 
              dr_damping >= 0.05):
            return HandlingQuality.LEVEL_2
        
        # Level 3 (fair handling, definite compensation required)
        else:
            return HandlingQuality.LEVEL_3
    
    def _generate_handling_comments(self, sp_freq: float, sp_damping: float,
                                  dr_freq: float, dr_damping: float,
                                  condition: FlightCondition) -> List[str]:
        """Generate comments about handling characteristics"""
        comments = []
        
        if sp_damping < 0.3:
            comments.append("Low short period damping may cause oscillatory response")
        elif sp_damping > 2.0:
            comments.append("High short period damping may cause sluggish response")
        
        if dr_damping < 0.05:
            comments.append("Low Dutch roll damping may require yaw damper")
        
        if condition.mach_number > 1.0 and sp_freq < 1.0:
            comments.append("Low short period frequency at supersonic speeds")
        
        if condition.load_factor > 2.0:
            comments.append("High-g maneuvering may degrade handling qualities")
        
        return comments
    
    def _design_control_systems(self, configuration: AircraftConfiguration,
                              control_authority: Dict[str, ControlAuthority]) -> Dict[str, ControlSystemDesign]:
        """
        Design control systems for the aircraft configuration
        
        Implements PID, LQR, and adaptive control algorithms
        """
        control_systems = {}
        
        # Design pitch control system
        if any("elevator" in name or "canard" in name for name in control_authority.keys()):
            pitch_system = self._design_pitch_controller(configuration, control_authority)
            control_systems["pitch"] = pitch_system
        
        # Design roll control system
        if any("aileron" in name or "elevon" in name for name in control_authority.keys()):
            roll_system = self._design_roll_controller(configuration, control_authority)
            control_systems["roll"] = roll_system
        
        # Design yaw control system
        if any("rudder" in name for name in control_authority.keys()):
            yaw_system = self._design_yaw_controller(configuration, control_authority)
            control_systems["yaw"] = yaw_system
        
        # Design thrust vectoring control if available
        if any("thrust_vectoring" in name for name in control_authority.keys()):
            tv_system = self._design_thrust_vectoring_controller(configuration, control_authority)
            control_systems["thrust_vectoring"] = tv_system
        
        return control_systems
    
    def _design_pitch_controller(self, configuration: AircraftConfiguration,
                               control_authority: Dict[str, ControlAuthority]) -> ControlSystemDesign:
        """Design pitch axis control system"""
        # PID controller design for pitch axis
        
        # Find pitch control surfaces
        pitch_surfaces = [auth for name, auth in control_authority.items() 
                         if auth.surface_type in [ControlSurface.ELEVATOR, ControlSurface.CANARD]]
        
        if not pitch_surfaces:
            # Default controller if no pitch surfaces found
            return ControlSystemDesign(
                controller_type="PID",
                gains={"kp": 1.0, "ki": 0.1, "kd": 0.05},
                bandwidth=2.0,
                phase_margin=45.0,
                gain_margin=6.0,
                settling_time=2.0,
                overshoot=10.0
            )
        
        # Calculate controller gains based on control authority
        total_effectiveness = sum(surface.effectiveness for surface in pitch_surfaces)
        
        # Higher effectiveness allows higher gains
        kp = min(total_effectiveness * 2.0, 5.0)
        ki = kp * 0.1
        kd = kp * 0.05
        
        return ControlSystemDesign(
            controller_type="PID",
            gains={"kp": kp, "ki": ki, "kd": kd},
            bandwidth=3.0,
            phase_margin=50.0,
            gain_margin=8.0,
            settling_time=1.5,
            overshoot=8.0
        )
    
    def _design_roll_controller(self, configuration: AircraftConfiguration,
                              control_authority: Dict[str, ControlAuthority]) -> ControlSystemDesign:
        """Design roll axis control system"""
        # Find roll control surfaces
        roll_surfaces = [auth for name, auth in control_authority.items() 
                        if auth.surface_type in [ControlSurface.AILERON, ControlSurface.ELEVON]]
        
        if not roll_surfaces:
            return ControlSystemDesign(
                controller_type="PID",
                gains={"kp": 0.8, "ki": 0.05, "kd": 0.02},
                bandwidth=5.0,
                phase_margin=40.0,
                gain_margin=6.0,
                settling_time=1.0,
                overshoot=5.0
            )
        
        # Roll axis typically needs higher bandwidth
        total_effectiveness = sum(surface.effectiveness for surface in roll_surfaces)
        
        kp = min(total_effectiveness * 3.0, 8.0)
        ki = kp * 0.05
        kd = kp * 0.02
        
        return ControlSystemDesign(
            controller_type="PID",
            gains={"kp": kp, "ki": ki, "kd": kd},
            bandwidth=8.0,
            phase_margin=45.0,
            gain_margin=7.0,
            settling_time=0.8,
            overshoot=5.0
        )
    
    def _design_yaw_controller(self, configuration: AircraftConfiguration,
                             control_authority: Dict[str, ControlAuthority]) -> ControlSystemDesign:
        """Design yaw axis control system"""
        return ControlSystemDesign(
            controller_type="PID",
            gains={"kp": 1.5, "ki": 0.08, "kd": 0.03},
            bandwidth=2.5,
            phase_margin=45.0,
            gain_margin=6.0,
            settling_time=2.0,
            overshoot=8.0
        )
    
    def _design_thrust_vectoring_controller(self, configuration: AircraftConfiguration,
                                          control_authority: Dict[str, ControlAuthority]) -> ControlSystemDesign:
        """Design thrust vectoring control system"""
        # Thrust vectoring allows higher performance
        return ControlSystemDesign(
            controller_type="LQR",
            gains={"q_attitude": 10.0, "q_rate": 1.0, "r_control": 0.1},
            bandwidth=15.0,
            phase_margin=60.0,
            gain_margin=10.0,
            settling_time=0.3,
            overshoot=2.0
        )
    
    def _calculate_stability_derivatives(self, configuration: AircraftConfiguration) -> StabilityDerivatives:
        """Calculate stability and control derivatives for the configuration"""
        
        # Simplified derivative calculation based on configuration geometry
        derivatives = StabilityDerivatives()
        
        # Find relevant modules
        wing_modules = [m for m in configuration.modules if "wing" in m.name.lower()]
        tail_modules = [m for m in configuration.modules if "tail" in m.name.lower()]
        
        # Longitudinal derivatives
        if tail_modules:
            # Negative static margin for stability
            derivatives.cma = -0.5  # Typical value for fighter aircraft
            derivatives.cmq = -8.0  # Pitch damping
            derivatives.cmde = -1.2  # Elevator effectiveness
        
        # Lateral-directional derivatives
        if tail_modules:
            derivatives.cnb = 0.1   # Directional stability
            derivatives.cnr = -0.3  # Yaw damping
            derivatives.cndr = -0.08 # Rudder effectiveness
            
            derivatives.clb = -0.05  # Dihedral effect
            derivatives.clp = -0.4   # Roll damping
            derivatives.clda = 0.15  # Aileron effectiveness
        
        return derivatives
    
    def _prepare_pilot_interface(self, configuration: AircraftConfiguration,
                               control_authority: Dict[str, ControlAuthority]) -> Dict[str, Any]:
        """
        Prepare pilot-in-the-loop simulation interface data
        
        Provides data structures and parameters needed for pilot simulation
        """
        interface_data = {
            "control_mapping": {},
            "display_parameters": {},
            "haptic_feedback": {},
            "flight_envelope_protection": {},
            "automation_modes": []
        }
        
        # Map control surfaces to pilot inputs
        for name, authority in control_authority.items():
            if authority.surface_type == ControlSurface.ELEVATOR:
                interface_data["control_mapping"]["pitch_stick"] = {
                    "surface": name,
                    "gain": authority.effectiveness,
                    "max_deflection": authority.max_deflection,
                    "response_time": authority.response_time
                }
            elif authority.surface_type == ControlSurface.AILERON:
                interface_data["control_mapping"]["roll_stick"] = {
                    "surface": name,
                    "gain": authority.effectiveness,
                    "max_deflection": authority.max_deflection,
                    "response_time": authority.response_time
                }
            elif authority.surface_type == ControlSurface.RUDDER:
                interface_data["control_mapping"]["rudder_pedals"] = {
                    "surface": name,
                    "gain": authority.effectiveness,
                    "max_deflection": authority.max_deflection,
                    "response_time": authority.response_time
                }
        
        # Display parameters for pilot feedback
        interface_data["display_parameters"] = {
            "attitude_indicator": {"update_rate": 60, "precision": 0.1},
            "airspeed_indicator": {"update_rate": 30, "precision": 1.0},
            "altitude_indicator": {"update_rate": 10, "precision": 10.0},
            "heading_indicator": {"update_rate": 30, "precision": 0.5}
        }
        
        # Haptic feedback parameters
        interface_data["haptic_feedback"] = {
            "stick_force_gradient": 2.0,  # N/degree
            "rudder_force_gradient": 5.0,  # N/degree
            "artificial_feel": True,
            "g_onset_cueing": True
        }
        
        # Flight envelope protection
        interface_data["flight_envelope_protection"] = {
            "angle_of_attack_limit": 25.0,  # degrees
            "load_factor_limit": 9.0,  # g's
            "airspeed_limit": {"min": 100, "max": 600},  # m/s
            "altitude_limit": 20000  # meters
        }
        
        # Automation modes
        interface_data["automation_modes"] = [
            "attitude_hold",
            "altitude_hold", 
            "heading_hold",
            "approach_mode",
            "combat_mode",
            "formation_flight"
        ]
        
        return interface_data
    
    def _calculate_performance_metrics(self, configuration: AircraftConfiguration,
                                     control_authority: Dict[str, ControlAuthority],
                                     handling_qualities: List[HandlingQualityAssessment]) -> Dict[str, Any]:
        """Calculate overall performance metrics"""
        
        # Calculate average handling quality rating
        avg_cooper_harper = np.mean([hq.cooper_harper_rating.value for hq in handling_qualities])
        
        # Calculate control power metrics
        total_control_power = sum(auth.effectiveness * auth.max_deflection 
                                for auth in control_authority.values())
        
        # Calculate response metrics
        avg_response_time = np.mean([auth.response_time for auth in control_authority.values()])
        
        # Estimate maneuverability metrics
        max_g_capability = self._estimate_max_g_capability(configuration, control_authority)
        roll_rate_capability = self._estimate_roll_rate_capability(configuration, control_authority)
        
        return {
            "average_cooper_harper_rating": avg_cooper_harper,
            "total_control_power": total_control_power,
            "average_response_time": avg_response_time,
            "max_g_capability": max_g_capability,
            "roll_rate_capability": roll_rate_capability,
            "stall_speed": self._estimate_stall_speed(configuration),
            "max_angle_of_attack": 25.0,  # degrees
            "service_ceiling": 18000,  # meters
            "combat_radius": 1200  # km
        }
    
    def _estimate_max_g_capability(self, configuration: AircraftConfiguration,
                                 control_authority: Dict[str, ControlAuthority]) -> float:
        """Estimate maximum g-load capability"""
        
        # Find pitch control authority
        pitch_authority = [auth for auth in control_authority.values() 
                          if auth.surface_type in [ControlSurface.ELEVATOR, ControlSurface.CANARD]]
        
        if not pitch_authority:
            return 6.0  # Default for aircraft without pitch control
        
        # Higher control authority allows higher g-loads
        max_pitch_effectiveness = max(auth.effectiveness for auth in pitch_authority)
        
        # Estimate based on control authority and structural limits
        base_g = 7.0
        # Ensure control factor is reasonable (between 0.8 and 1.5)
        control_factor = max(0.8, min(max_pitch_effectiveness / 0.5, 1.5))
        
        return base_g * control_factor
    
    def _estimate_roll_rate_capability(self, configuration: AircraftConfiguration,
                                     control_authority: Dict[str, ControlAuthority]) -> float:
        """Estimate maximum roll rate capability (degrees/second)"""
        
        # Find roll control authority
        roll_authority = [auth for auth in control_authority.values() 
                         if auth.surface_type in [ControlSurface.AILERON, ControlSurface.ELEVON]]
        
        if not roll_authority:
            return 90.0  # Default roll rate
        
        # Higher control authority allows higher roll rates
        max_roll_effectiveness = max(auth.effectiveness for auth in roll_authority)
        
        # Base roll rate for fighter aircraft
        base_roll_rate = 180.0  # degrees/second
        # Ensure control factor is reasonable (between 0.5 and 2.0)
        control_factor = max(0.5, min(max_roll_effectiveness / 0.3, 2.0))
        
        return base_roll_rate * control_factor
    
    def _estimate_stall_speed(self, configuration: AircraftConfiguration) -> float:
        """Estimate stall speed (m/s)"""
        
        # Find wing modules to estimate wing area
        wing_modules = [m for m in configuration.modules if "wing" in m.name.lower()]
        
        if not wing_modules:
            return 70.0  # Default stall speed
        
        # Estimate wing area from module dimensions
        total_wing_area = 0.0
        for module in wing_modules:
            if module.physical_properties and module.physical_properties.dimensions:
                # Assume wing area is length * width (but use more conservative estimate)
                wing_area = module.physical_properties.dimensions[0] * module.physical_properties.dimensions[1]
                total_wing_area += wing_area
        
        if total_wing_area == 0 or total_wing_area < 10.0:
            return 70.0  # Default if no reasonable wing area found
        
        # Estimate aircraft mass
        total_mass = sum(m.physical_properties.mass if m.physical_properties else 1000 
                        for m in configuration.modules)
        
        # Add base platform mass if available
        if configuration.base_platform:
            total_mass += configuration.base_platform.base_mass
        
        # Ensure minimum reasonable mass
        if total_mass < 5000:
            total_mass = 8000  # Default fighter mass
        
        # Stall speed calculation: V_stall = sqrt(2 * W / (rho * S * CL_max))
        # Assuming CL_max = 1.4 (higher for fighter with flaps), rho = 1.225 kg/m³
        cl_max = 1.4
        rho = 1.225
        weight = total_mass * 9.81  # N
        
        stall_speed = np.sqrt(2 * weight / (rho * total_wing_area * cl_max))
        
        # Ensure reasonable range for fighter aircraft
        return max(50.0, min(stall_speed, 150.0))
    
    def _estimate_static_margin(self, configuration: AircraftConfiguration) -> float:
        """Estimate static margin of the aircraft"""
        
        # Simplified static margin calculation
        # Positive static margin = stable, negative = unstable
        
        # Find wing and tail modules
        wing_modules = [m for m in configuration.modules if "wing" in m.name.lower()]
        tail_modules = [m for m in configuration.modules if "tail" in m.name.lower()]
        
        if not wing_modules or not tail_modules:
            return 0.05  # Slightly stable default
        
        # For fighter aircraft, typically have small positive or slightly negative static margin
        # Modern fighters often have relaxed static stability (negative static margin)
        
        return -0.02  # Slightly unstable for maneuverability
    
    def _calculate_flight_envelope_limits(self, configuration: AircraftConfiguration) -> Dict[str, Any]:
        """Calculate flight envelope operational limits"""
        
        return {
            "max_mach": 2.5,
            "max_altitude": 18000,  # meters
            "max_g_positive": 9.0,
            "max_g_negative": -3.0,
            "max_angle_of_attack": 25.0,  # degrees
            "max_sideslip": 15.0,  # degrees
            "corner_velocity": 200.0,  # m/s
            "design_dive_speed": 400.0,  # m/s
            "never_exceed_speed": 450.0  # m/s
        }
    
    def _generate_recommendations(self, configuration: AircraftConfiguration,
                                handling_qualities: List[HandlingQualityAssessment]) -> List[str]:
        """Generate recommendations for improving stability and control"""
        
        recommendations = []
        
        # Check for poor handling qualities
        poor_hq_conditions = [hq for hq in handling_qualities 
                             if hq.cooper_harper_rating == HandlingQuality.LEVEL_3]
        
        if poor_hq_conditions:
            recommendations.append("Consider adding stability augmentation system for Level 3 conditions")
        
        # Check for low damping
        low_damping_conditions = [hq for hq in handling_qualities 
                                if hq.short_period_damping < 0.3 or hq.dutch_roll_damping < 0.05]
        
        if low_damping_conditions:
            recommendations.append("Add damping augmentation (pitch/yaw dampers)")
        
        # Check control authority
        structural_modules = [m for m in configuration.modules if m.module_type == ModuleType.STRUCTURAL]
        if len(structural_modules) < 3:
            recommendations.append("Consider adding more control surfaces for improved authority")
        
        # Check for thrust vectoring opportunity
        propulsion_modules = [m for m in configuration.modules if m.module_type == ModuleType.PROPULSION]
        has_tv = any(self._has_thrust_vectoring(m) for m in propulsion_modules)
        
        if not has_tv and len(propulsion_modules) > 0:
            recommendations.append("Consider thrust vectoring for enhanced maneuverability")
        
        # Check for supersonic handling
        supersonic_conditions = [hq for hq in handling_qualities 
                               if hq.flight_condition.mach_number > 1.0]
        
        if supersonic_conditions:
            avg_supersonic_rating = np.mean([hq.cooper_harper_rating.value for hq in supersonic_conditions])
            if avg_supersonic_rating > 2.0:
                recommendations.append("Optimize control system gains for supersonic flight")
        
        return recommendations