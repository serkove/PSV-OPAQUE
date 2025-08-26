"""Supersonic Intake Designer with shock wave analysis and optimization."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math
from enum import Enum

from ...core.logging import get_engine_logger


class IntakeType(Enum):
    """Intake type enumeration."""
    PITOT = "pitot"
    EXTERNAL_COMPRESSION = "external_compression"
    MIXED_COMPRESSION = "mixed_compression"
    INTERNAL_COMPRESSION = "internal_compression"
    VARIABLE_GEOMETRY = "variable_geometry"
    DIVERTERLESS = "diverterless"


@dataclass
class IntakeGeometry:
    """Intake geometry definition."""
    intake_type: IntakeType
    capture_area: float  # m²
    throat_area: float  # m²
    diffuser_area_ratio: float  # Exit area / throat area
    compression_ratio: float  # Total pressure ratio
    ramp_angles: List[float] = field(default_factory=list)  # degrees
    cowl_angle: float = 0.0  # degrees
    throat_mach: float = 1.0
    variable_geometry_range: Optional[Tuple[float, float]] = None  # min, max angles


@dataclass
class FlowConditions:
    """Flow conditions at intake."""
    mach_number: float
    altitude: float  # m
    angle_of_attack: float = 0.0  # degrees
    sideslip_angle: float = 0.0  # degrees
    atmospheric_pressure: float = 101325.0  # Pa
    atmospheric_temperature: float = 288.15  # K


@dataclass
class ShockWaveData:
    """Shock wave analysis results."""
    shock_angles: List[float]  # degrees
    pressure_ratios: List[float]
    temperature_ratios: List[float]
    mach_numbers: List[float]
    total_pressure_recovery: float
    entropy_increase: float


@dataclass
class IntakePerformance:
    """Intake performance metrics."""
    pressure_recovery: float  # Total pressure ratio
    mass_flow_ratio: float  # Actual / ideal mass flow
    distortion_coefficient: float  # Pressure distortion at engine face
    drag_coefficient: float
    spillage_drag: float  # N
    additive_drag: float  # N
    efficiency: float


class IntakeDesigner:
    """Advanced supersonic intake design and analysis system."""
    
    def __init__(self):
        """Initialize intake designer."""
        self.logger = get_engine_logger('propulsion.intake')
        
        # Gas properties (air)
        self.gamma = 1.4  # Specific heat ratio
        self.gas_constant = 287.0  # J/(kg·K)
        
        # Design constraints
        self.max_compression_ratio = 50.0
        self.min_throat_mach = 0.3
        self.max_throat_mach = 1.0
        
        # Performance targets
        self.target_pressure_recovery = 0.95
        self.max_distortion = 0.1
    
    def design_intake(self, design_conditions: FlowConditions, 
                     engine_mass_flow: float, intake_type: IntakeType) -> IntakeGeometry:
        """Design intake for specified conditions and requirements."""
        self.logger.info(f"Designing {intake_type.value} intake for Mach {design_conditions.mach_number}")
        
        # Calculate required capture area
        capture_area = self._calculate_capture_area(design_conditions, engine_mass_flow)
        
        # Design compression system
        if intake_type == IntakeType.PITOT:
            geometry = self._design_pitot_intake(design_conditions, capture_area)
        elif intake_type == IntakeType.EXTERNAL_COMPRESSION:
            geometry = self._design_external_compression_intake(design_conditions, capture_area)
        elif intake_type == IntakeType.MIXED_COMPRESSION:
            geometry = self._design_mixed_compression_intake(design_conditions, capture_area)
        elif intake_type == IntakeType.VARIABLE_GEOMETRY:
            geometry = self._design_variable_geometry_intake(design_conditions, capture_area)
        elif intake_type == IntakeType.DIVERTERLESS:
            geometry = self._design_diverterless_intake(design_conditions, capture_area)
        else:
            geometry = self._design_internal_compression_intake(design_conditions, capture_area)
        
        # Validate design
        validation_errors = self._validate_intake_geometry(geometry, design_conditions)
        if validation_errors:
            self.logger.warning(f"Design validation issues: {validation_errors}")
        
        return geometry
    
    def _calculate_capture_area(self, conditions: FlowConditions, mass_flow: float) -> float:
        """Calculate required capture area for mass flow."""
        # Get atmospheric conditions
        rho = self._get_density(conditions.atmospheric_pressure, conditions.atmospheric_temperature)
        
        # Calculate velocity
        a = math.sqrt(self.gamma * self.gas_constant * conditions.atmospheric_temperature)
        velocity = conditions.mach_number * a
        
        # Required capture area based on mass flow continuity
        # mass_flow = rho * velocity * area
        capture_area = mass_flow / (rho * velocity)
        
        # Add margin for spillage and losses (10-20% typical)
        capture_area *= 1.15
        
        return capture_area
    
    def _design_pitot_intake(self, conditions: FlowConditions, capture_area: float) -> IntakeGeometry:
        """Design simple pitot intake."""
        # Pitot intake: simple normal shock at inlet
        throat_area = capture_area * 0.8  # Contraction ratio
        diffuser_area_ratio = 1.5  # Moderate diffusion
        
        # Normal shock pressure ratio
        M1 = conditions.mach_number
        compression_ratio = self._normal_shock_pressure_ratio(M1)
        
        return IntakeGeometry(
            intake_type=IntakeType.PITOT,
            capture_area=capture_area,
            throat_area=throat_area,
            diffuser_area_ratio=diffuser_area_ratio,
            compression_ratio=compression_ratio,
            throat_mach=0.6
        )
    
    def _design_external_compression_intake(self, conditions: FlowConditions, capture_area: float) -> IntakeGeometry:
        """Design external compression intake with oblique shocks."""
        M1 = conditions.mach_number
        
        # Design two-shock system for efficiency
        if M1 <= 2.0:
            # Single oblique shock + normal shock
            shock_angle = self._optimize_single_shock_angle(M1)
            ramp_angles = [shock_angle]
        else:
            # Two oblique shocks + normal shock
            shock1_angle, shock2_angle = self._optimize_two_shock_system(M1)
            ramp_angles = [shock1_angle, shock2_angle]
        
        # Calculate compression ratio
        compression_ratio = self._calculate_compression_ratio(M1, ramp_angles)
        
        # Size throat and diffuser
        throat_area = capture_area * 0.85
        diffuser_area_ratio = 2.0  # Higher diffusion for external compression
        
        return IntakeGeometry(
            intake_type=IntakeType.EXTERNAL_COMPRESSION,
            capture_area=capture_area,
            throat_area=throat_area,
            diffuser_area_ratio=diffuser_area_ratio,
            compression_ratio=compression_ratio,
            ramp_angles=ramp_angles,
            throat_mach=0.7
        )
    
    def _design_mixed_compression_intake(self, conditions: FlowConditions, capture_area: float) -> IntakeGeometry:
        """Design mixed compression intake."""
        M1 = conditions.mach_number
        
        # External compression (2 shocks) + internal compression
        external_shocks = self._optimize_two_shock_system(M1)
        
        # Internal compression in diffuser
        throat_area = capture_area * 0.75  # Higher contraction
        diffuser_area_ratio = 3.0  # Aggressive diffusion
        
        # Total compression ratio
        external_compression = self._calculate_compression_ratio(M1, external_shocks)
        internal_compression = 1.5  # Additional internal compression
        compression_ratio = external_compression * internal_compression
        
        return IntakeGeometry(
            intake_type=IntakeType.MIXED_COMPRESSION,
            capture_area=capture_area,
            throat_area=throat_area,
            diffuser_area_ratio=diffuser_area_ratio,
            compression_ratio=compression_ratio,
            ramp_angles=list(external_shocks),
            throat_mach=0.8
        )
    
    def _design_variable_geometry_intake(self, conditions: FlowConditions, capture_area: float) -> IntakeGeometry:
        """Design variable geometry intake."""
        M1 = conditions.mach_number
        
        # Design for range of Mach numbers
        min_mach = max(1.2, M1 * 0.6)
        max_mach = min(3.5, M1 * 1.4)
        
        # Variable ramp angles
        min_angle = self._optimize_single_shock_angle(min_mach)
        max_angle = self._optimize_single_shock_angle(max_mach)
        
        # Ensure range is valid
        if min_angle >= max_angle:
            min_angle = max(5.0, max_angle - 3.0)
            max_angle = min(25.0, min_angle + 6.0)
        
        # Current design point
        current_angle = self._optimize_single_shock_angle(M1)
        
        # Throat area varies with geometry
        throat_area = capture_area * 0.8
        diffuser_area_ratio = 2.5
        
        compression_ratio = self._calculate_compression_ratio(M1, [current_angle])
        
        return IntakeGeometry(
            intake_type=IntakeType.VARIABLE_GEOMETRY,
            capture_area=capture_area,
            throat_area=throat_area,
            diffuser_area_ratio=diffuser_area_ratio,
            compression_ratio=compression_ratio,
            ramp_angles=[current_angle],
            variable_geometry_range=(min_angle, max_angle),
            throat_mach=0.75
        )
    
    def _design_diverterless_intake(self, conditions: FlowConditions, capture_area: float) -> IntakeGeometry:
        """Design diverterless supersonic intake (DSI)."""
        M1 = conditions.mach_number
        
        # DSI uses compression surface without sharp edges
        # Simplified as single equivalent shock
        equivalent_angle = self._optimize_single_shock_angle(M1) * 0.8  # Softer compression
        
        throat_area = capture_area * 0.82  # Good flow capture
        diffuser_area_ratio = 2.2
        
        compression_ratio = self._calculate_compression_ratio(M1, [equivalent_angle])
        
        return IntakeGeometry(
            intake_type=IntakeType.DIVERTERLESS,
            capture_area=capture_area,
            throat_area=throat_area,
            diffuser_area_ratio=diffuser_area_ratio,
            compression_ratio=compression_ratio,
            ramp_angles=[equivalent_angle],
            throat_mach=0.65
        )
    
    def _design_internal_compression_intake(self, conditions: FlowConditions, capture_area: float) -> IntakeGeometry:
        """Design internal compression intake."""
        # All compression occurs inside the intake
        throat_area = capture_area * 0.7  # High contraction
        diffuser_area_ratio = 4.0  # Very aggressive diffusion
        
        # Estimate compression from area ratio
        area_ratio = capture_area / throat_area
        compression_ratio = area_ratio ** (self.gamma / (self.gamma - 1)) * 0.8  # Efficiency factor
        
        return IntakeGeometry(
            intake_type=IntakeType.INTERNAL_COMPRESSION,
            capture_area=capture_area,
            throat_area=throat_area,
            diffuser_area_ratio=diffuser_area_ratio,
            compression_ratio=compression_ratio,
            throat_mach=0.9
        )
    
    def analyze_shock_system(self, geometry: IntakeGeometry, conditions: FlowConditions) -> ShockWaveData:
        """Analyze shock wave system for intake geometry."""
        M1 = conditions.mach_number
        
        shock_angles = []
        pressure_ratios = []
        temperature_ratios = []
        mach_numbers = [M1]
        
        current_mach = M1
        total_pressure_recovery = 1.0
        entropy_increase = 0.0
        
        # Analyze each shock in sequence
        for ramp_angle in geometry.ramp_angles:
            # Calculate oblique shock properties
            shock_angle = self._calculate_shock_angle(current_mach, ramp_angle)
            
            if shock_angle is not None:
                shock_angles.append(shock_angle)
                
                # Post-shock conditions
                p_ratio = self._oblique_shock_pressure_ratio(current_mach, shock_angle)
                t_ratio = self._oblique_shock_temperature_ratio(current_mach, shock_angle)
                m_post = self._post_shock_mach(current_mach, shock_angle, ramp_angle)
                
                pressure_ratios.append(p_ratio)
                temperature_ratios.append(t_ratio)
                mach_numbers.append(m_post)
                
                # Update total pressure recovery
                total_pressure_recovery *= self._shock_total_pressure_ratio(current_mach, shock_angle)
                
                # Update entropy
                entropy_increase += self._shock_entropy_increase(current_mach, shock_angle)
                
                current_mach = m_post
        
        # Final normal shock at throat (if supersonic)
        if current_mach > 1.0:
            normal_shock_pr = self._normal_shock_pressure_ratio(current_mach)
            pressure_ratios.append(normal_shock_pr)
            
            normal_shock_tr = self._normal_shock_temperature_ratio(current_mach)
            temperature_ratios.append(normal_shock_tr)
            
            final_mach = self._post_normal_shock_mach(current_mach)
            mach_numbers.append(final_mach)
            
            # Update total pressure recovery
            total_pressure_recovery *= self._normal_shock_total_pressure_ratio(current_mach)
            
            shock_angles.append(90.0)  # Normal shock
        
        return ShockWaveData(
            shock_angles=shock_angles,
            pressure_ratios=pressure_ratios,
            temperature_ratios=temperature_ratios,
            mach_numbers=mach_numbers,
            total_pressure_recovery=total_pressure_recovery,
            entropy_increase=entropy_increase
        )
    
    def calculate_performance(self, geometry: IntakeGeometry, 
                            conditions: FlowConditions) -> IntakePerformance:
        """Calculate intake performance metrics."""
        # Analyze shock system
        shock_data = self.analyze_shock_system(geometry, conditions)
        
        # Pressure recovery
        pressure_recovery = shock_data.total_pressure_recovery
        
        # Apply viscous losses
        pressure_recovery *= self._viscous_loss_factor(geometry, conditions)
        
        # Mass flow ratio
        mass_flow_ratio = self._calculate_mass_flow_ratio(geometry, conditions)
        
        # Distortion coefficient
        distortion_coefficient = self._calculate_distortion(geometry, conditions)
        
        # Drag calculations
        drag_coefficient = self._calculate_drag_coefficient(geometry, conditions)
        spillage_drag = self._calculate_spillage_drag(geometry, conditions)
        additive_drag = self._calculate_additive_drag(geometry, conditions)
        
        # Overall efficiency
        efficiency = pressure_recovery * mass_flow_ratio * (1 - distortion_coefficient)
        
        return IntakePerformance(
            pressure_recovery=pressure_recovery,
            mass_flow_ratio=mass_flow_ratio,
            distortion_coefficient=distortion_coefficient,
            drag_coefficient=drag_coefficient,
            spillage_drag=spillage_drag,
            additive_drag=additive_drag,
            efficiency=efficiency
        )
    
    def optimize_intake_geometry(self, conditions: FlowConditions, 
                               engine_mass_flow: float,
                               intake_type: IntakeType,
                               optimization_target: str = "efficiency") -> IntakeGeometry:
        """Optimize intake geometry for specified target."""
        self.logger.info(f"Optimizing {intake_type.value} intake for {optimization_target}")
        
        # Initial design
        best_geometry = self.design_intake(conditions, engine_mass_flow, intake_type)
        best_performance = self.calculate_performance(best_geometry, conditions)
        
        if optimization_target == "efficiency":
            best_metric = best_performance.efficiency
        elif optimization_target == "pressure_recovery":
            best_metric = best_performance.pressure_recovery
        elif optimization_target == "low_distortion":
            best_metric = 1.0 - best_performance.distortion_coefficient
        else:
            best_metric = best_performance.efficiency
        
        # Optimization iterations
        for iteration in range(10):
            # Vary geometry parameters
            test_geometry = self._perturb_geometry(best_geometry, iteration)
            test_performance = self.calculate_performance(test_geometry, conditions)
            
            if optimization_target == "efficiency":
                test_metric = test_performance.efficiency
            elif optimization_target == "pressure_recovery":
                test_metric = test_performance.pressure_recovery
            elif optimization_target == "low_distortion":
                test_metric = 1.0 - test_performance.distortion_coefficient
            else:
                test_metric = test_performance.efficiency
            
            if test_metric > best_metric:
                best_geometry = test_geometry
                best_metric = test_metric
                self.logger.debug(f"Optimization iteration {iteration}: improved {optimization_target} to {best_metric:.4f}")
        
        return best_geometry
    
    def _perturb_geometry(self, geometry: IntakeGeometry, iteration: int) -> IntakeGeometry:
        """Create perturbed geometry for optimization."""
        import copy
        new_geometry = copy.deepcopy(geometry)
        
        # Perturbation magnitude decreases with iterations
        perturbation = 0.1 * (1.0 - iteration / 10.0)
        
        # Vary ramp angles
        if new_geometry.ramp_angles:
            for i in range(len(new_geometry.ramp_angles)):
                delta = perturbation * (2 * (iteration % 2) - 1) * 2.0  # ±2 degrees
                new_geometry.ramp_angles[i] = max(5.0, min(25.0, new_geometry.ramp_angles[i] + delta))
        
        # Vary area ratios
        if iteration % 3 == 0:
            new_geometry.throat_area *= (1.0 + perturbation * 0.1)
        elif iteration % 3 == 1:
            new_geometry.diffuser_area_ratio *= (1.0 + perturbation * 0.2)
        
        return new_geometry
    
    # Shock wave calculation methods
    def _calculate_shock_angle(self, mach: float, ramp_angle: float) -> Optional[float]:
        """Calculate oblique shock angle for given Mach number and ramp angle."""
        if mach <= 1.0:
            return None
        
        # Use theta-beta-M relation for oblique shocks
        theta_rad = math.radians(ramp_angle)
        
        # Check if solution exists (detachment criterion)
        max_theta = self._calculate_max_deflection_angle(mach)
        if ramp_angle > max_theta:
            return None  # Detached shock
        
        # Initial guess - between Mach angle and 90 degrees
        mach_angle = math.asin(1.0 / mach)
        beta_min = mach_angle
        beta_max = math.pi / 2
        
        # Bisection method for robust solution
        for _ in range(50):
            beta_rad = (beta_min + beta_max) / 2
            
            # Theta-beta-M relation
            tan_theta_calc = 2 * (mach**2 * math.sin(beta_rad)**2 - 1) / (mach**2 * (self.gamma + math.cos(2*beta_rad)) + 2) / math.tan(beta_rad)
            theta_calc = math.atan(tan_theta_calc)
            
            if abs(theta_calc - theta_rad) < 1e-8:
                break
            
            if theta_calc < theta_rad:
                beta_min = beta_rad
            else:
                beta_max = beta_rad
        
        return math.degrees(beta_rad)
    
    def _calculate_max_deflection_angle(self, mach: float) -> float:
        """Calculate maximum deflection angle for given Mach number."""
        if mach <= 1.0:
            return 0.0
        
        # Approximate formula for maximum deflection angle
        max_theta_rad = math.atan(math.sqrt((mach**2 - 1) / (2 + (self.gamma - 1) * mach**2)))
        return math.degrees(max_theta_rad)
    
    def _optimize_single_shock_angle(self, mach: float) -> float:
        """Optimize single shock angle for maximum pressure recovery."""
        if mach <= 1.0:
            return 0.0
        
        # For single shock, optimal angle is typically around 10-15 degrees
        best_angle = 10.0
        best_recovery = 0.0
        
        for angle in range(5, 26):  # 5 to 25 degrees
            shock_angle = self._calculate_shock_angle(mach, angle)
            if shock_angle:
                recovery = self._shock_total_pressure_ratio(mach, shock_angle)
                if recovery > best_recovery:
                    best_recovery = recovery
                    best_angle = angle
        
        return best_angle
    
    def _optimize_two_shock_system(self, mach: float) -> Tuple[float, float]:
        """Optimize two-shock system for maximum pressure recovery."""
        if mach <= 1.5:
            return self._optimize_single_shock_angle(mach), 0.0
        
        best_angles = (10.0, 8.0)
        best_recovery = 0.0
        
        # Grid search for optimal angles
        for angle1 in range(8, 20, 2):
            for angle2 in range(6, 15, 2):
                # Calculate intermediate Mach number
                shock1_angle = self._calculate_shock_angle(mach, angle1)
                if shock1_angle:
                    mach_intermediate = self._post_shock_mach(mach, shock1_angle, angle1)
                    
                    shock2_angle = self._calculate_shock_angle(mach_intermediate, angle2)
                    if shock2_angle:
                        # Total pressure recovery
                        recovery1 = self._shock_total_pressure_ratio(mach, shock1_angle)
                        recovery2 = self._shock_total_pressure_ratio(mach_intermediate, shock2_angle)
                        total_recovery = recovery1 * recovery2
                        
                        if total_recovery > best_recovery:
                            best_recovery = total_recovery
                            best_angles = (angle1, angle2)
        
        return best_angles
    
    # Gas dynamics utility methods
    def _get_density(self, pressure: float, temperature: float) -> float:
        """Calculate air density."""
        return pressure / (self.gas_constant * temperature)
    
    def _normal_shock_pressure_ratio(self, mach: float) -> float:
        """Calculate pressure ratio across normal shock."""
        if mach <= 1.0:
            return 1.0
        return (2 * self.gamma * mach**2 - (self.gamma - 1)) / (self.gamma + 1)
    
    def _normal_shock_temperature_ratio(self, mach: float) -> float:
        """Calculate temperature ratio across normal shock."""
        if mach <= 1.0:
            return 1.0
        return ((2 * self.gamma * mach**2 - (self.gamma - 1)) * ((self.gamma - 1) * mach**2 + 2)) / ((self.gamma + 1)**2 * mach**2)
    
    def _normal_shock_total_pressure_ratio(self, mach: float) -> float:
        """Calculate total pressure ratio across normal shock."""
        if mach <= 1.0:
            return 1.0
        
        numerator = ((self.gamma + 1) * mach**2 / (2 + (self.gamma - 1) * mach**2))**(self.gamma / (self.gamma - 1))
        denominator = (2 * self.gamma * mach**2 - (self.gamma - 1)) / (self.gamma + 1)
        
        return numerator / denominator**(1 / (self.gamma - 1))
    
    def _post_normal_shock_mach(self, mach: float) -> float:
        """Calculate Mach number after normal shock."""
        if mach <= 1.0:
            return mach
        return math.sqrt(((self.gamma - 1) * mach**2 + 2) / (2 * self.gamma * mach**2 - (self.gamma - 1)))
    
    def _oblique_shock_pressure_ratio(self, mach: float, shock_angle: float) -> float:
        """Calculate pressure ratio across oblique shock."""
        beta_rad = math.radians(shock_angle)
        mach_n = mach * math.sin(beta_rad)
        return self._normal_shock_pressure_ratio(mach_n)
    
    def _oblique_shock_temperature_ratio(self, mach: float, shock_angle: float) -> float:
        """Calculate temperature ratio across oblique shock."""
        beta_rad = math.radians(shock_angle)
        mach_n = mach * math.sin(beta_rad)
        return self._normal_shock_temperature_ratio(mach_n)
    
    def _shock_total_pressure_ratio(self, mach: float, shock_angle: float) -> float:
        """Calculate total pressure ratio across oblique shock."""
        beta_rad = math.radians(shock_angle)
        mach_n = mach * math.sin(beta_rad)
        return self._normal_shock_total_pressure_ratio(mach_n)
    
    def _post_shock_mach(self, mach: float, shock_angle: float, ramp_angle: float) -> float:
        """Calculate Mach number after oblique shock."""
        beta_rad = math.radians(shock_angle)
        theta_rad = math.radians(ramp_angle)
        
        mach_n1 = mach * math.sin(beta_rad)
        mach_n2 = self._post_normal_shock_mach(mach_n1)
        
        return mach_n2 / math.sin(beta_rad - theta_rad)
    
    def _shock_entropy_increase(self, mach: float, shock_angle: float) -> float:
        """Calculate entropy increase across shock."""
        beta_rad = math.radians(shock_angle)
        mach_n = mach * math.sin(beta_rad)
        
        if mach_n <= 1.0:
            return 0.0
        
        # Entropy increase for normal shock component
        # s2 - s1 = cv * ln(T2/T1) - R * ln(rho2/rho1)
        # For perfect gas: s2 - s1 = cv * ln(T2/T1) + R * ln(p1/p2)
        
        p_ratio = self._normal_shock_pressure_ratio(mach_n)
        t_ratio = self._normal_shock_temperature_ratio(mach_n)
        
        # Entropy increase (always positive for irreversible process)
        entropy_increase = self.gas_constant * (
            (self.gamma / (self.gamma - 1)) * math.log(t_ratio) - math.log(p_ratio)
        )
        
        return max(0.0, entropy_increase)  # Ensure non-negative
    
    def _calculate_compression_ratio(self, mach: float, ramp_angles: List[float]) -> float:
        """Calculate total compression ratio for shock system."""
        current_mach = mach
        total_compression = 1.0
        
        for ramp_angle in ramp_angles:
            shock_angle = self._calculate_shock_angle(current_mach, ramp_angle)
            if shock_angle:
                p_ratio = self._oblique_shock_pressure_ratio(current_mach, shock_angle)
                total_compression *= p_ratio
                current_mach = self._post_shock_mach(current_mach, shock_angle, ramp_angle)
        
        # Final normal shock if still supersonic
        if current_mach > 1.0:
            total_compression *= self._normal_shock_pressure_ratio(current_mach)
        
        return total_compression
    
    # Performance calculation methods
    def _viscous_loss_factor(self, geometry: IntakeGeometry, conditions: FlowConditions) -> float:
        """Calculate viscous loss factor."""
        # Simplified viscous loss model
        reynolds_number = self._estimate_reynolds_number(geometry, conditions)
        
        # Friction factor
        cf = 0.074 / reynolds_number**0.2  # Turbulent flow approximation
        
        # Wetted area to capture area ratio (estimated)
        wetted_area_ratio = 2.5 + geometry.diffuser_area_ratio * 0.5
        
        # Pressure loss
        loss_factor = 1.0 - cf * wetted_area_ratio * 0.1
        
        return max(0.8, loss_factor)
    
    def _estimate_reynolds_number(self, geometry: IntakeGeometry, conditions: FlowConditions) -> float:
        """Estimate Reynolds number for intake flow."""
        # Characteristic length (hydraulic diameter)
        char_length = 2 * math.sqrt(geometry.capture_area / math.pi)
        
        # Velocity
        a = math.sqrt(self.gamma * self.gas_constant * conditions.atmospheric_temperature)
        velocity = conditions.mach_number * a
        
        # Kinematic viscosity (approximate for air)
        mu = 1.8e-5 * (conditions.atmospheric_temperature / 288.15)**0.7  # Dynamic viscosity
        rho = self._get_density(conditions.atmospheric_pressure, conditions.atmospheric_temperature)
        nu = mu / rho  # Kinematic viscosity
        
        return velocity * char_length / nu
    
    def _calculate_mass_flow_ratio(self, geometry: IntakeGeometry, conditions: FlowConditions) -> float:
        """Calculate mass flow ratio (actual/ideal)."""
        # Spillage losses
        spillage_factor = 0.95  # Typical for well-designed intakes
        
        # Boundary layer losses
        bl_factor = 0.98
        
        # Shock losses
        shock_factor = 0.99 if len(geometry.ramp_angles) <= 2 else 0.97
        
        return spillage_factor * bl_factor * shock_factor
    
    def _calculate_distortion(self, geometry: IntakeGeometry, conditions: FlowConditions) -> float:
        """Calculate pressure distortion coefficient."""
        # Simplified distortion model
        base_distortion = 0.02  # Well-designed intake
        
        # Increase with Mach number
        mach_factor = conditions.mach_number * 0.01
        
        # Increase with compression ratio
        compression_factor = (geometry.compression_ratio - 1) * 0.005
        
        # Angle of attack effects
        aoa_factor = abs(conditions.angle_of_attack) * 0.001
        
        total_distortion = base_distortion + mach_factor + compression_factor + aoa_factor
        
        return min(0.2, total_distortion)  # Cap at 20%
    
    def _calculate_drag_coefficient(self, geometry: IntakeGeometry, conditions: FlowConditions) -> float:
        """Calculate intake drag coefficient."""
        # Base drag from pressure forces
        base_cd = 0.02
        
        # Increase with ramp angles
        ramp_drag = sum(math.sin(math.radians(angle)) for angle in geometry.ramp_angles) * 0.01
        
        # Cowl drag
        cowl_drag = math.sin(math.radians(geometry.cowl_angle)) * 0.005
        
        return base_cd + ramp_drag + cowl_drag
    
    def _calculate_spillage_drag(self, geometry: IntakeGeometry, conditions: FlowConditions) -> float:
        """Calculate spillage drag force."""
        # Dynamic pressure
        rho = self._get_density(conditions.atmospheric_pressure, conditions.atmospheric_temperature)
        a = math.sqrt(self.gamma * self.gas_constant * conditions.atmospheric_temperature)
        velocity = conditions.mach_number * a
        q = 0.5 * rho * velocity**2
        
        # Spillage area (estimated)
        spillage_area = geometry.capture_area * 0.05  # 5% spillage
        
        return q * spillage_area * 0.8  # Spillage drag coefficient
    
    def _calculate_additive_drag(self, geometry: IntakeGeometry, conditions: FlowConditions) -> float:
        """Calculate additive drag from intake installation."""
        # Dynamic pressure
        rho = self._get_density(conditions.atmospheric_pressure, conditions.atmospheric_temperature)
        a = math.sqrt(self.gamma * self.gas_constant * conditions.atmospheric_temperature)
        velocity = conditions.mach_number * a
        q = 0.5 * rho * velocity**2
        
        # Additive drag area
        additive_area = geometry.capture_area * 0.1  # Installation effects
        
        return q * additive_area * 0.3  # Additive drag coefficient
    
    def _validate_intake_geometry(self, geometry: IntakeGeometry, conditions: FlowConditions) -> List[str]:
        """Validate intake geometry design."""
        errors = []
        
        # Area ratios
        if geometry.throat_area >= geometry.capture_area:
            errors.append("Throat area must be less than capture area")
        
        if geometry.diffuser_area_ratio < 1.0:
            errors.append("Diffuser area ratio must be >= 1.0")
        
        # Compression ratio
        if geometry.compression_ratio > self.max_compression_ratio:
            errors.append(f"Compression ratio {geometry.compression_ratio:.1f} exceeds limit {self.max_compression_ratio}")
        
        # Throat Mach number
        if geometry.throat_mach < self.min_throat_mach or geometry.throat_mach > self.max_throat_mach:
            errors.append(f"Throat Mach number {geometry.throat_mach:.2f} outside valid range")
        
        # Ramp angles
        for angle in geometry.ramp_angles:
            if angle < 0 or angle > 30:
                errors.append(f"Ramp angle {angle:.1f}° outside reasonable range (0-30°)")
        
        return errors