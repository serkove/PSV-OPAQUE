"""Design validation and reporting system for hypersonic Mach 60 vehicles."""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import json

from ..common.data_models import AircraftConfiguration, FlowConditions
from ..common.enums import ExtremePropulsionType, PlasmaRegime, ThermalProtectionType
from .multi_physics_integration import HypersonicMultiPhysicsIntegrator, MultiPhysicsState
from .hypersonic_mission_planner import HypersonicMissionProfile, ThermalConstraint, PropulsionConstraint
from .thermal_constraint_manager import ThermalConstraintManager, ThermalState
from .errors import ValidationError
from .logging import get_logger


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(Enum):
    """Categories of validation checks."""
    THERMAL = "thermal"
    STRUCTURAL = "structural"
    AERODYNAMIC = "aerodynamic"
    PROPULSION = "propulsion"
    MATERIALS = "materials"
    MISSION = "mission"
    SAFETY = "safety"
    PERFORMANCE = "performance"


@dataclass
class ValidationIssue:
    """Individual validation issue."""
    issue_id: str
    category: ValidationCategory
    severity: ValidationSeverity
    title: str
    description: str
    affected_components: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class SafetyMargin:
    """Safety margin calculation."""
    parameter_name: str
    current_value: float
    limit_value: float
    safety_factor: float
    margin_percentage: float
    acceptable: bool
    critical_threshold: float = 0.1  # 10% margin minimum


@dataclass
class PerformanceMetric:
    """Performance metric evaluation."""
    metric_name: str
    current_value: float
    target_value: Optional[float] = None
    baseline_value: Optional[float] = None
    improvement_percentage: Optional[float] = None
    meets_requirement: bool = True
    units: str = ""


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    report_id: str
    aircraft_config: AircraftConfiguration
    validation_timestamp: float
    issues: List[ValidationIssue] = field(default_factory=list)
    safety_margins: List[SafetyMargin] = field(default_factory=list)
    performance_metrics: List[PerformanceMetric] = field(default_factory=list)
    overall_status: str = "UNKNOWN"
    recommendations: List[str] = field(default_factory=list)
    design_score: float = 0.0
    validation_summary: Dict[str, Any] = field(default_factory=dict)


class BaseValidator(ABC):
    """Base class for design validators."""
    
    def __init__(self, name: str):
        """Initialize base validator."""
        self.name = name
        self.logger = get_logger(f"validator_{name}")
    
    @abstractmethod
    def validate(self, config: AircraftConfiguration, 
                analysis_results: Dict[str, Any]) -> List[ValidationIssue]:
        """Perform validation and return issues."""
        pass
    
    @abstractmethod
    def calculate_safety_margins(self, config: AircraftConfiguration,
                               analysis_results: Dict[str, Any]) -> List[SafetyMargin]:
        """Calculate safety margins."""
        pass
    
    def create_issue(self, issue_id: str, category: ValidationCategory,
                    severity: ValidationSeverity, title: str, description: str,
                    components: List[str] = None, recommendations: List[str] = None,
                    parameters: Dict[str, Any] = None) -> ValidationIssue:
        """Create a validation issue."""
        return ValidationIssue(
            issue_id=issue_id,
            category=category,
            severity=severity,
            title=title,
            description=description,
            affected_components=components or [],
            recommendations=recommendations or [],
            parameters=parameters or {}
        )


class ThermalValidator(BaseValidator):
    """Validator for thermal design aspects."""
    
    def __init__(self):
        """Initialize thermal validator."""
        super().__init__("thermal")
        
        # Thermal limits for Mach 60 flight
        self.max_surface_temperature = 3000.0  # K
        self.max_heat_flux = 100e6  # W/m²
        self.max_thermal_gradient = 1000.0  # K/m
        self.max_ablation_rate = 1e-5  # m/s
    
    def validate(self, config: AircraftConfiguration, 
                analysis_results: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate thermal design."""
        issues = []
        thermal_results = analysis_results.get('thermal', {})
        
        # Check surface temperature
        max_temp = thermal_results.get('max_surface_temperature', 0.0)
        if max_temp > self.max_surface_temperature:
            issues.append(self.create_issue(
                issue_id="THERMAL_001",
                category=ValidationCategory.THERMAL,
                severity=ValidationSeverity.CRITICAL,
                title="Surface Temperature Exceeds Limit",
                description=f"Maximum surface temperature ({max_temp:.0f} K) exceeds material limit ({self.max_surface_temperature:.0f} K)",
                components=["thermal_protection_system", "structure"],
                recommendations=[
                    "Increase thermal protection system thickness",
                    "Consider active cooling systems",
                    "Modify flight profile to reduce thermal loads",
                    "Use higher temperature materials"
                ],
                parameters={'max_temperature': max_temp, 'limit': self.max_surface_temperature}
            ))
        
        # Check heat flux
        max_heat_flux = thermal_results.get('max_heat_flux', 0.0)
        if max_heat_flux > self.max_heat_flux:
            issues.append(self.create_issue(
                issue_id="THERMAL_002",
                category=ValidationCategory.THERMAL,
                severity=ValidationSeverity.ERROR,
                title="Heat Flux Exceeds Design Limit",
                description=f"Maximum heat flux ({max_heat_flux/1e6:.1f} MW/m²) exceeds design limit ({self.max_heat_flux/1e6:.1f} MW/m²)",
                components=["thermal_protection_system"],
                recommendations=[
                    "Implement active cooling system",
                    "Increase nose radius to reduce stagnation heating",
                    "Optimize flight trajectory for lower heat flux"
                ],
                parameters={'max_heat_flux': max_heat_flux, 'limit': self.max_heat_flux}
            ))
        
        # Check ablation rate
        ablation_rate = thermal_results.get('ablation_rate', 0.0)
        if ablation_rate > self.max_ablation_rate:
            issues.append(self.create_issue(
                issue_id="THERMAL_003",
                category=ValidationCategory.THERMAL,
                severity=ValidationSeverity.WARNING,
                title="High Material Ablation Rate",
                description=f"Ablation rate ({ablation_rate*1e6:.2f} μm/s) may cause significant material loss",
                components=["thermal_protection_system"],
                recommendations=[
                    "Use ablative materials with higher recession resistance",
                    "Implement regenerative cooling",
                    "Reduce mission duration at peak heating"
                ],
                parameters={'ablation_rate': ablation_rate, 'limit': self.max_ablation_rate}
            ))
        
        # Check cooling system effectiveness
        cooling_effectiveness = thermal_results.get('cooling_effectiveness', 0.0)
        if cooling_effectiveness < 0.5 and max_heat_flux > self.max_heat_flux * 0.5:
            issues.append(self.create_issue(
                issue_id="THERMAL_004",
                category=ValidationCategory.THERMAL,
                severity=ValidationSeverity.WARNING,
                title="Insufficient Cooling System Effectiveness",
                description=f"Cooling system effectiveness ({cooling_effectiveness:.2f}) may be insufficient for high heat flux conditions",
                components=["cooling_system"],
                recommendations=[
                    "Increase coolant flow rate",
                    "Optimize cooling channel design",
                    "Consider transpiration cooling"
                ],
                parameters={'effectiveness': cooling_effectiveness, 'heat_flux': max_heat_flux}
            ))
        
        return issues
    
    def calculate_safety_margins(self, config: AircraftConfiguration,
                               analysis_results: Dict[str, Any]) -> List[SafetyMargin]:
        """Calculate thermal safety margins."""
        margins = []
        thermal_results = analysis_results.get('thermal', {})
        
        # Temperature margin
        max_temp = thermal_results.get('max_surface_temperature', 0.0)
        temp_margin = SafetyMargin(
            parameter_name="Surface Temperature",
            current_value=max_temp,
            limit_value=self.max_surface_temperature,
            safety_factor=self.max_surface_temperature / max(max_temp, 1.0),
            margin_percentage=(self.max_surface_temperature - max_temp) / self.max_surface_temperature * 100,
            acceptable=max_temp < self.max_surface_temperature * 0.9
        )
        margins.append(temp_margin)
        
        # Heat flux margin
        max_heat_flux = thermal_results.get('max_heat_flux', 0.0)
        flux_margin = SafetyMargin(
            parameter_name="Heat Flux",
            current_value=max_heat_flux,
            limit_value=self.max_heat_flux,
            safety_factor=self.max_heat_flux / max(max_heat_flux, 1.0),
            margin_percentage=(self.max_heat_flux - max_heat_flux) / self.max_heat_flux * 100,
            acceptable=max_heat_flux < self.max_heat_flux * 0.8
        )
        margins.append(flux_margin)
        
        return margins


class StructuralValidator(BaseValidator):
    """Validator for structural design aspects."""
    
    def __init__(self):
        """Initialize structural validator."""
        super().__init__("structural")
        
        # Structural limits
        self.min_safety_factor = 1.5
        self.max_stress_ratio = 0.8  # Fraction of yield strength
        self.max_displacement = 0.1  # m
        self.max_thermal_stress_ratio = 0.6
    
    def validate(self, config: AircraftConfiguration, 
                analysis_results: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate structural design."""
        issues = []
        structural_results = analysis_results.get('structural', {})
        
        # Check safety factor
        safety_factor = structural_results.get('safety_factor', 0.0)
        if safety_factor < self.min_safety_factor:
            issues.append(self.create_issue(
                issue_id="STRUCT_001",
                category=ValidationCategory.STRUCTURAL,
                severity=ValidationSeverity.CRITICAL,
                title="Insufficient Structural Safety Factor",
                description=f"Safety factor ({safety_factor:.2f}) is below minimum requirement ({self.min_safety_factor})",
                components=["structure", "materials"],
                recommendations=[
                    "Increase structural thickness",
                    "Use higher strength materials",
                    "Reduce operational loads",
                    "Add structural reinforcement"
                ],
                parameters={'safety_factor': safety_factor, 'minimum': self.min_safety_factor}
            ))
        
        # Check maximum stress
        max_stress = structural_results.get('max_stress', 0.0)
        yield_strength = 270e6  # Pa (typical aluminum)
        stress_ratio = max_stress / yield_strength
        if stress_ratio > self.max_stress_ratio:
            issues.append(self.create_issue(
                issue_id="STRUCT_002",
                category=ValidationCategory.STRUCTURAL,
                severity=ValidationSeverity.ERROR,
                title="High Structural Stress",
                description=f"Maximum stress ratio ({stress_ratio:.2f}) exceeds design limit ({self.max_stress_ratio})",
                components=["structure"],
                recommendations=[
                    "Redistribute loads through design optimization",
                    "Increase cross-sectional area in high-stress regions",
                    "Use materials with higher yield strength"
                ],
                parameters={'stress_ratio': stress_ratio, 'limit': self.max_stress_ratio}
            ))
        
        # Check displacement
        max_displacement = structural_results.get('max_displacement', 0.0)
        if max_displacement > self.max_displacement:
            issues.append(self.create_issue(
                issue_id="STRUCT_003",
                category=ValidationCategory.STRUCTURAL,
                severity=ValidationSeverity.WARNING,
                title="Excessive Structural Displacement",
                description=f"Maximum displacement ({max_displacement:.3f} m) may affect aerodynamic performance",
                components=["structure"],
                recommendations=[
                    "Increase structural stiffness",
                    "Add structural bracing",
                    "Consider active shape control"
                ],
                parameters={'displacement': max_displacement, 'limit': self.max_displacement}
            ))
        
        # Check thermal stress contribution
        thermal_stress = structural_results.get('thermal_stress_contribution', 0.0)
        thermal_stress_ratio = thermal_stress / yield_strength
        if thermal_stress_ratio > self.max_thermal_stress_ratio:
            issues.append(self.create_issue(
                issue_id="STRUCT_004",
                category=ValidationCategory.STRUCTURAL,
                severity=ValidationSeverity.WARNING,
                title="High Thermal Stress",
                description=f"Thermal stress ratio ({thermal_stress_ratio:.2f}) is significant",
                components=["structure", "thermal_protection_system"],
                recommendations=[
                    "Improve thermal isolation",
                    "Use materials with lower thermal expansion",
                    "Implement thermal stress relief features"
                ],
                parameters={'thermal_stress_ratio': thermal_stress_ratio, 'limit': self.max_thermal_stress_ratio}
            ))
        
        return issues
    
    def calculate_safety_margins(self, config: AircraftConfiguration,
                               analysis_results: Dict[str, Any]) -> List[SafetyMargin]:
        """Calculate structural safety margins."""
        margins = []
        structural_results = analysis_results.get('structural', {})
        
        # Safety factor margin
        safety_factor = structural_results.get('safety_factor', 0.0)
        sf_margin = SafetyMargin(
            parameter_name="Safety Factor",
            current_value=safety_factor,
            limit_value=self.min_safety_factor,
            safety_factor=safety_factor / self.min_safety_factor,
            margin_percentage=(safety_factor - self.min_safety_factor) / self.min_safety_factor * 100,
            acceptable=safety_factor > self.min_safety_factor * 1.1
        )
        margins.append(sf_margin)
        
        # Stress margin
        max_stress = structural_results.get('max_stress', 0.0)
        yield_strength = 270e6  # Pa
        allowable_stress = yield_strength * self.max_stress_ratio
        stress_margin = SafetyMargin(
            parameter_name="Structural Stress",
            current_value=max_stress,
            limit_value=allowable_stress,
            safety_factor=allowable_stress / max(max_stress, 1.0),
            margin_percentage=(allowable_stress - max_stress) / allowable_stress * 100,
            acceptable=max_stress < allowable_stress * 0.9
        )
        margins.append(stress_margin)
        
        return margins


class AerodynamicValidator(BaseValidator):
    """Validator for aerodynamic design aspects."""
    
    def __init__(self):
        """Initialize aerodynamic validator."""
        super().__init__("aerodynamic")
        
        # Aerodynamic limits
        self.max_stagnation_temperature = 50000.0  # K
        self.plasma_formation_threshold = 8000.0  # K
        self.max_shock_standoff_ratio = 0.1  # Fraction of nose radius
    
    def validate(self, config: AircraftConfiguration, 
                analysis_results: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate aerodynamic design."""
        issues = []
        aero_results = analysis_results.get('aerodynamic', {})
        
        # Check stagnation temperature
        stag_temp = aero_results.get('stagnation_temperature', 0.0)
        if stag_temp > self.max_stagnation_temperature:
            issues.append(self.create_issue(
                issue_id="AERO_001",
                category=ValidationCategory.AERODYNAMIC,
                severity=ValidationSeverity.ERROR,
                title="Excessive Stagnation Temperature",
                description=f"Stagnation temperature ({stag_temp:.0f} K) exceeds design limit ({self.max_stagnation_temperature:.0f} K)",
                components=["aerodynamic_design"],
                recommendations=[
                    "Increase flight altitude",
                    "Reduce flight Mach number",
                    "Optimize vehicle shape for lower stagnation heating"
                ],
                parameters={'stagnation_temperature': stag_temp, 'limit': self.max_stagnation_temperature}
            ))
        
        # Check plasma formation
        plasma_formation = aero_results.get('plasma_formation', False)
        if plasma_formation:
            issues.append(self.create_issue(
                issue_id="AERO_002",
                category=ValidationCategory.AERODYNAMIC,
                severity=ValidationSeverity.WARNING,
                title="Plasma Formation Detected",
                description=f"Plasma formation occurs at stagnation temperature {stag_temp:.0f} K",
                components=["aerodynamic_design", "communication_systems"],
                recommendations=[
                    "Design for radio blackout conditions",
                    "Implement plasma mitigation techniques",
                    "Consider alternative communication methods"
                ],
                parameters={'stagnation_temperature': stag_temp, 'plasma_threshold': self.plasma_formation_threshold}
            ))
        
        # Check shock standoff distance
        shock_standoff = aero_results.get('shock_standoff_distance', 0.0)
        nose_radius = 0.5  # m (assumed)
        standoff_ratio = shock_standoff / nose_radius
        if standoff_ratio < self.max_shock_standoff_ratio:
            issues.append(self.create_issue(
                issue_id="AERO_003",
                category=ValidationCategory.AERODYNAMIC,
                severity=ValidationSeverity.INFO,
                title="Small Shock Standoff Distance",
                description=f"Shock standoff distance ({shock_standoff:.3f} m) is small relative to nose radius",
                components=["aerodynamic_design"],
                recommendations=[
                    "Consider blunter nose design for increased standoff",
                    "Verify heat transfer calculations account for thin shock layer"
                ],
                parameters={'standoff_distance': shock_standoff, 'nose_radius': nose_radius}
            ))
        
        return issues
    
    def calculate_safety_margins(self, config: AircraftConfiguration,
                               analysis_results: Dict[str, Any]) -> List[SafetyMargin]:
        """Calculate aerodynamic safety margins."""
        margins = []
        aero_results = analysis_results.get('aerodynamic', {})
        
        # Stagnation temperature margin
        stag_temp = aero_results.get('stagnation_temperature', 0.0)
        temp_margin = SafetyMargin(
            parameter_name="Stagnation Temperature",
            current_value=stag_temp,
            limit_value=self.max_stagnation_temperature,
            safety_factor=self.max_stagnation_temperature / max(stag_temp, 1.0),
            margin_percentage=(self.max_stagnation_temperature - stag_temp) / self.max_stagnation_temperature * 100,
            acceptable=stag_temp < self.max_stagnation_temperature * 0.8
        )
        margins.append(temp_margin)
        
        return margins


class PropulsionValidator(BaseValidator):
    """Validator for propulsion system design."""
    
    def __init__(self):
        """Initialize propulsion validator."""
        super().__init__("propulsion")
        
        # Propulsion limits
        self.max_air_breathing_mach = 15.0
        self.min_rocket_altitude = 40000.0  # m
        self.max_fuel_consumption_rate = 100.0  # kg/s
    
    def validate(self, config: AircraftConfiguration, 
                analysis_results: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate propulsion system design."""
        issues = []
        
        # Check for propulsion mode transitions
        # This would require mission profile data
        # For now, implement basic checks
        
        # Check if air-breathing operation is feasible at Mach 60
        issues.append(self.create_issue(
            issue_id="PROP_001",
            category=ValidationCategory.PROPULSION,
            severity=ValidationSeverity.INFO,
            title="Mach 60 Requires Rocket Propulsion",
            description="Mach 60 flight requires rocket or combined-cycle propulsion",
            components=["propulsion_system"],
            recommendations=[
                "Implement combined-cycle propulsion system",
                "Ensure adequate rocket fuel capacity",
                "Design for propulsion mode transitions"
            ],
            parameters={'max_mach': 60.0, 'air_breathing_limit': self.max_air_breathing_mach}
        ))
        
        return issues
    
    def calculate_safety_margins(self, config: AircraftConfiguration,
                               analysis_results: Dict[str, Any]) -> List[SafetyMargin]:
        """Calculate propulsion safety margins."""
        margins = []
        
        # Fuel capacity margin (placeholder)
        fuel_margin = SafetyMargin(
            parameter_name="Fuel Capacity",
            current_value=10000.0,  # kg (placeholder)
            limit_value=15000.0,    # kg (placeholder)
            safety_factor=1.5,
            margin_percentage=33.3,
            acceptable=True
        )
        margins.append(fuel_margin)
        
        return margins


class PerformanceAnalyzer:
    """Analyzer for design performance metrics."""
    
    def __init__(self):
        """Initialize performance analyzer."""
        self.logger = get_logger("performance_analyzer")
    
    def analyze_performance(self, config: AircraftConfiguration,
                          analysis_results: Dict[str, Any],
                          mission_profile: Optional[HypersonicMissionProfile] = None) -> List[PerformanceMetric]:
        """Analyze design performance metrics."""
        metrics = []
        
        # Thermal performance
        thermal_results = analysis_results.get('thermal', {})
        max_temp = thermal_results.get('max_surface_temperature', 0.0)
        
        thermal_metric = PerformanceMetric(
            metric_name="Maximum Surface Temperature",
            current_value=max_temp,
            target_value=2500.0,  # K
            meets_requirement=max_temp < 2500.0,
            units="K"
        )
        metrics.append(thermal_metric)
        
        # Structural performance
        structural_results = analysis_results.get('structural', {})
        safety_factor = structural_results.get('safety_factor', 0.0)
        
        structural_metric = PerformanceMetric(
            metric_name="Structural Safety Factor",
            current_value=safety_factor,
            target_value=1.5,
            meets_requirement=safety_factor >= 1.5,
            units=""
        )
        metrics.append(structural_metric)
        
        # Aerodynamic performance
        aero_results = analysis_results.get('aerodynamic', {})
        stag_temp = aero_results.get('stagnation_temperature', 0.0)
        
        aero_metric = PerformanceMetric(
            metric_name="Stagnation Temperature",
            current_value=stag_temp,
            target_value=30000.0,  # K
            meets_requirement=stag_temp < 30000.0,
            units="K"
        )
        metrics.append(aero_metric)
        
        # Mission performance (if profile available)
        if mission_profile:
            mission_metric = PerformanceMetric(
                metric_name="Mission Duration",
                current_value=mission_profile.total_duration,
                target_value=3600.0,  # seconds
                meets_requirement=mission_profile.total_duration <= 3600.0,
                units="s"
            )
            metrics.append(mission_metric)
        
        return metrics


class DesignOptimizationRecommender:
    """Recommender for design optimization."""
    
    def __init__(self):
        """Initialize design optimization recommender."""
        self.logger = get_logger("design_optimizer")
    
    def generate_recommendations(self, validation_report: ValidationReport) -> List[str]:
        """Generate design optimization recommendations."""
        recommendations = []
        
        # Analyze issues by category and severity
        critical_issues = [issue for issue in validation_report.issues 
                          if issue.severity == ValidationSeverity.CRITICAL]
        error_issues = [issue for issue in validation_report.issues 
                       if issue.severity == ValidationSeverity.ERROR]
        
        # Critical thermal issues
        thermal_critical = [issue for issue in critical_issues 
                           if issue.category == ValidationCategory.THERMAL]
        if thermal_critical:
            recommendations.append(
                "PRIORITY 1: Address critical thermal issues - implement active cooling system and optimize thermal protection"
            )
        
        # Critical structural issues
        structural_critical = [issue for issue in critical_issues 
                              if issue.category == ValidationCategory.STRUCTURAL]
        if structural_critical:
            recommendations.append(
                "PRIORITY 1: Address critical structural issues - increase safety factors through material or design changes"
            )
        
        # Performance optimization
        low_margins = [margin for margin in validation_report.safety_margins 
                      if margin.margin_percentage < 20.0]
        if low_margins:
            recommendations.append(
                "PRIORITY 2: Improve safety margins - consider design modifications for better performance margins"
            )
        
        # Multi-physics coupling issues
        coupling_results = validation_report.validation_summary.get('coupling', {})
        if not coupling_results.get('converged', True):
            recommendations.append(
                "PRIORITY 2: Improve multi-physics coupling convergence - review coupling parameters and time stepping"
            )
        
        # Overall design score improvement
        if validation_report.design_score < 0.8:
            recommendations.append(
                "PRIORITY 3: Overall design optimization - conduct parametric studies to improve design score"
            )
        
        return recommendations


class HypersonicDesignValidator:
    """Main design validation system for hypersonic Mach 60 vehicles."""
    
    def __init__(self):
        """Initialize hypersonic design validator."""
        self.validators = {
            'thermal': ThermalValidator(),
            'structural': StructuralValidator(),
            'aerodynamic': AerodynamicValidator(),
            'propulsion': PropulsionValidator()
        }
        
        self.performance_analyzer = PerformanceAnalyzer()
        self.optimization_recommender = DesignOptimizationRecommender()
        self.logger = get_logger("hypersonic_design_validator")
    
    def validate_design(self, config: AircraftConfiguration,
                       analysis_results: Dict[str, Any],
                       mission_profile: Optional[HypersonicMissionProfile] = None) -> ValidationReport:
        """Perform comprehensive design validation."""
        self.logger.info(f"Validating hypersonic design: {config.name}")
        
        # Create validation report
        report = ValidationReport(
            report_id=f"validation_{config.name}_{int(time.time())}",
            aircraft_config=config,
            validation_timestamp=time.time()
        )
        
        # Run all validators
        all_issues = []
        all_margins = []
        
        for validator_name, validator in self.validators.items():
            try:
                # Get validation issues
                issues = validator.validate(config, analysis_results)
                all_issues.extend(issues)
                
                # Get safety margins
                margins = validator.calculate_safety_margins(config, analysis_results)
                all_margins.extend(margins)
                
                self.logger.info(f"{validator_name} validation: {len(issues)} issues, {len(margins)} margins")
                
            except Exception as e:
                self.logger.error(f"Error in {validator_name} validation: {e}")
                error_issue = ValidationIssue(
                    issue_id=f"VAL_ERROR_{validator_name.upper()}",
                    category=ValidationCategory.SAFETY,
                    severity=ValidationSeverity.ERROR,
                    title=f"Validation Error in {validator_name.title()}",
                    description=f"Error during {validator_name} validation: {str(e)}",
                    affected_components=[validator_name]
                )
                all_issues.append(error_issue)
        
        # Analyze performance metrics
        performance_metrics = self.performance_analyzer.analyze_performance(
            config, analysis_results, mission_profile
        )
        
        # Calculate overall design score
        design_score = self._calculate_design_score(all_issues, all_margins, performance_metrics)
        
        # Determine overall status
        overall_status = self._determine_overall_status(all_issues, all_margins)
        
        # Generate optimization recommendations
        report.issues = all_issues
        report.safety_margins = all_margins
        report.performance_metrics = performance_metrics
        report.design_score = design_score
        report.overall_status = overall_status
        
        optimization_recommendations = self.optimization_recommender.generate_recommendations(report)
        report.recommendations = optimization_recommendations
        
        # Create validation summary
        report.validation_summary = self._create_validation_summary(
            all_issues, all_margins, performance_metrics, analysis_results
        )
        
        self.logger.info(f"Validation complete: {overall_status}, Score: {design_score:.2f}")
        
        return report
    
    def _calculate_design_score(self, issues: List[ValidationIssue],
                              margins: List[SafetyMargin],
                              metrics: List[PerformanceMetric]) -> float:
        """Calculate overall design score (0.0 to 1.0)."""
        score = 1.0
        
        # Penalty for issues
        for issue in issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                score -= 0.3
            elif issue.severity == ValidationSeverity.ERROR:
                score -= 0.2
            elif issue.severity == ValidationSeverity.WARNING:
                score -= 0.1
            elif issue.severity == ValidationSeverity.INFO:
                score -= 0.05
        
        # Bonus for good safety margins
        acceptable_margins = [m for m in margins if m.acceptable]
        margin_bonus = len(acceptable_margins) / max(len(margins), 1) * 0.2
        score += margin_bonus
        
        # Bonus for meeting performance requirements
        met_requirements = [m for m in metrics if m.meets_requirement]
        performance_bonus = len(met_requirements) / max(len(metrics), 1) * 0.2
        score += performance_bonus
        
        return max(0.0, min(1.0, score))
    
    def _determine_overall_status(self, issues: List[ValidationIssue],
                                margins: List[SafetyMargin]) -> str:
        """Determine overall validation status."""
        critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        error_issues = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        unacceptable_margins = [m for m in margins if not m.acceptable]
        
        if critical_issues:
            return "CRITICAL_ISSUES"
        elif error_issues or unacceptable_margins:
            return "DESIGN_ISSUES"
        elif any(i.severity == ValidationSeverity.WARNING for i in issues):
            return "ACCEPTABLE_WITH_WARNINGS"
        else:
            return "DESIGN_ACCEPTABLE"
    
    def _create_validation_summary(self, issues: List[ValidationIssue],
                                 margins: List[SafetyMargin],
                                 metrics: List[PerformanceMetric],
                                 analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create validation summary."""
        # Count issues by category and severity
        issue_counts = {}
        for category in ValidationCategory:
            issue_counts[category.value] = {
                'critical': len([i for i in issues if i.category == category and i.severity == ValidationSeverity.CRITICAL]),
                'error': len([i for i in issues if i.category == category and i.severity == ValidationSeverity.ERROR]),
                'warning': len([i for i in issues if i.category == category and i.severity == ValidationSeverity.WARNING]),
                'info': len([i for i in issues if i.category == category and i.severity == ValidationSeverity.INFO])
            }
        
        # Safety margin summary
        margin_summary = {
            'total_margins': len(margins),
            'acceptable_margins': len([m for m in margins if m.acceptable]),
            'average_margin_percentage': np.mean([m.margin_percentage for m in margins]) if margins else 0.0,
            'minimum_margin': min([m.margin_percentage for m in margins]) if margins else 0.0
        }
        
        # Performance summary
        performance_summary = {
            'total_metrics': len(metrics),
            'requirements_met': len([m for m in metrics if m.meets_requirement]),
            'performance_score': len([m for m in metrics if m.meets_requirement]) / max(len(metrics), 1)
        }
        
        # Multi-physics coupling summary
        coupling_summary = analysis_results.get('coupling', {})
        
        return {
            'issue_counts': issue_counts,
            'margin_summary': margin_summary,
            'performance_summary': performance_summary,
            'coupling': coupling_summary,
            'total_issues': len(issues),
            'critical_issues': len([i for i in issues if i.severity == ValidationSeverity.CRITICAL]),
            'validation_timestamp': time.time()
        }
    
    def export_report(self, report: ValidationReport, format: str = 'json') -> str:
        """Export validation report in specified format."""
        if format.lower() == 'json':
            return self._export_json_report(report)
        elif format.lower() == 'html':
            return self._export_html_report(report)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json_report(self, report: ValidationReport) -> str:
        """Export report as JSON."""
        # Convert report to dictionary
        report_dict = {
            'report_id': report.report_id,
            'aircraft_config': report.aircraft_config.name,
            'validation_timestamp': report.validation_timestamp,
            'overall_status': report.overall_status,
            'design_score': report.design_score,
            'issues': [
                {
                    'issue_id': issue.issue_id,
                    'category': issue.category.value,
                    'severity': issue.severity.value,
                    'title': issue.title,
                    'description': issue.description,
                    'affected_components': issue.affected_components,
                    'recommendations': issue.recommendations,
                    'parameters': issue.parameters
                }
                for issue in report.issues
            ],
            'safety_margins': [
                {
                    'parameter_name': margin.parameter_name,
                    'current_value': margin.current_value,
                    'limit_value': margin.limit_value,
                    'safety_factor': margin.safety_factor,
                    'margin_percentage': margin.margin_percentage,
                    'acceptable': margin.acceptable
                }
                for margin in report.safety_margins
            ],
            'performance_metrics': [
                {
                    'metric_name': metric.metric_name,
                    'current_value': metric.current_value,
                    'target_value': metric.target_value,
                    'meets_requirement': metric.meets_requirement,
                    'units': metric.units
                }
                for metric in report.performance_metrics
            ],
            'recommendations': report.recommendations,
            'validation_summary': report.validation_summary
        }
        
        return json.dumps(report_dict, indent=2, default=str)
    
    def _export_html_report(self, report: ValidationReport) -> str:
        """Export report as HTML."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Hypersonic Design Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .issue {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }}
                .critical {{ border-left-color: #ff0000; }}
                .error {{ border-left-color: #ff8800; }}
                .warning {{ border-left-color: #ffaa00; }}
                .info {{ border-left-color: #0088ff; }}
                .metric {{ margin: 5px 0; }}
                .acceptable {{ color: green; }}
                .unacceptable {{ color: red; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Hypersonic Design Validation Report</h1>
                <p><strong>Aircraft:</strong> {report.aircraft_config.name}</p>
                <p><strong>Status:</strong> {report.overall_status}</p>
                <p><strong>Design Score:</strong> {report.design_score:.2f}</p>
                <p><strong>Validation Date:</strong> {time.ctime(report.validation_timestamp)}</p>
            </div>
            
            <div class="section">
                <h2>Validation Issues</h2>
        """
        
        for issue in report.issues:
            severity_class = issue.severity.value
            html += f"""
                <div class="issue {severity_class}">
                    <h3>{issue.title} ({issue.severity.value.upper()})</h3>
                    <p>{issue.description}</p>
                    <p><strong>Affected Components:</strong> {', '.join(issue.affected_components)}</p>
                    <p><strong>Recommendations:</strong></p>
                    <ul>
            """
            for rec in issue.recommendations:
                html += f"<li>{rec}</li>"
            html += "</ul></div>"
        
        html += """
            </div>
            
            <div class="section">
                <h2>Safety Margins</h2>
        """
        
        for margin in report.safety_margins:
            acceptable_class = "acceptable" if margin.acceptable else "unacceptable"
            html += f"""
                <div class="metric {acceptable_class}">
                    <strong>{margin.parameter_name}:</strong> 
                    {margin.current_value:.2f} / {margin.limit_value:.2f} 
                    (Margin: {margin.margin_percentage:.1f}%)
                </div>
            """
        
        html += """
            </div>
            
            <div class="section">
                <h2>Performance Metrics</h2>
        """
        
        for metric in report.performance_metrics:
            meets_class = "acceptable" if metric.meets_requirement else "unacceptable"
            html += f"""
                <div class="metric {meets_class}">
                    <strong>{metric.metric_name}:</strong> 
                    {metric.current_value:.2f} {metric.units}
                    {f"(Target: {metric.target_value:.2f})" if metric.target_value else ""}
                </div>
            """
        
        html += """
            </div>
            
            <div class="section">
                <h2>Optimization Recommendations</h2>
                <ul>
        """
        
        for rec in report.recommendations:
            html += f"<li>{rec}</li>"
        
        html += """
                </ul>
            </div>
        </body>
        </html>
        """
        
        return html