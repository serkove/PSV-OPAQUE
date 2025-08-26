"""Modular assembly sequence optimization module."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import numpy as np
from datetime import datetime, timedelta

from ...common.data_models import Module, AircraftConfiguration
from ...core.logging import get_engine_logger


class AssemblyConstraintType(Enum):
    """Types of assembly constraints."""
    PRECEDENCE = "precedence"  # Must be assembled before another
    EXCLUSION = "exclusion"    # Cannot be assembled simultaneously
    RESOURCE = "resource"      # Requires specific resource
    ACCESS = "access"          # Requires access to specific area
    TOOLING = "tooling"        # Requires specific tooling
    SKILL = "skill"            # Requires specific skill level


class QualityCheckType(Enum):
    """Types of quality checkpoints."""
    DIMENSIONAL = "dimensional"
    ELECTRICAL = "electrical"
    MECHANICAL = "mechanical"
    FUNCTIONAL = "functional"
    VISUAL = "visual"
    LEAK_TEST = "leak_test"
    TORQUE_CHECK = "torque_check"


class ResourceType(Enum):
    """Types of assembly resources."""
    TECHNICIAN = "technician"
    CRANE = "crane"
    FIXTURE = "fixture"
    TOOL = "tool"
    WORKSPACE = "workspace"
    INSPECTION_EQUIPMENT = "inspection_equipment"


@dataclass
class AssemblyConstraint:
    """Assembly constraint specification."""
    constraint_id: str
    constraint_type: AssemblyConstraintType
    source_step: str
    target_step: Optional[str] = None
    resource_required: Optional[str] = None
    description: str = ""
    priority: int = 1  # 1=highest, 5=lowest


@dataclass
class QualityCheckpoint:
    """Quality control checkpoint."""
    checkpoint_id: str
    checkpoint_type: QualityCheckType
    step_id: str
    description: str
    inspection_time: float  # minutes
    required_equipment: List[str] = field(default_factory=list)
    acceptance_criteria: Dict[str, Any] = field(default_factory=dict)
    failure_actions: List[str] = field(default_factory=list)


@dataclass
class AssemblyResource:
    """Assembly resource specification."""
    resource_id: str
    resource_type: ResourceType
    capacity: int  # Number of simultaneous operations
    availability_schedule: List[Tuple[datetime, datetime]] = field(default_factory=list)
    hourly_cost: float = 0.0  # USD per hour
    setup_time: float = 0.0  # minutes
    skill_requirements: List[str] = field(default_factory=list)


@dataclass
class AssemblyStep:
    """Individual assembly step."""
    step_id: str
    step_name: str
    module_id: str
    estimated_time: float  # minutes
    labor_hours: float
    required_resources: List[str] = field(default_factory=list)
    skill_level_required: int = 1  # 1=basic, 5=expert
    setup_time: float = 0.0  # minutes
    cleanup_time: float = 0.0  # minutes
    quality_checkpoints: List[str] = field(default_factory=list)
    risk_factors: Dict[str, float] = field(default_factory=dict)
    
    # Scheduling information
    earliest_start: Optional[datetime] = None
    latest_finish: Optional[datetime] = None
    scheduled_start: Optional[datetime] = None
    scheduled_finish: Optional[datetime] = None


@dataclass
class AssemblySequence:
    """Complete assembly sequence."""
    sequence_id: str
    configuration_id: str
    steps: List[AssemblyStep]
    constraints: List[AssemblyConstraint]
    quality_checkpoints: List[QualityCheckpoint]
    total_time: float  # minutes
    total_cost: float  # USD
    resource_utilization: Dict[str, float]  # resource_id -> utilization %
    critical_path: List[str]  # step_ids on critical path
    risk_score: float  # Overall risk assessment


@dataclass
class ConflictDetection:
    """Assembly conflict detection result."""
    conflict_id: str
    conflict_type: str
    affected_steps: List[str]
    description: str
    severity: int  # 1=critical, 5=minor
    suggested_resolution: str


class ModularAssembly:
    """Modular assembly sequence optimization system."""
    
    def __init__(self):
        """Initialize modular assembly system."""
        self.logger = get_engine_logger('modular_assembly')
        self.resources = {}
        self.standard_checkpoints = {}
        self.constraint_templates = {}
        self._initialize_standard_resources()
        self._initialize_quality_templates()
    
    def _initialize_standard_resources(self):
        """Initialize standard assembly resources."""
        # Standard technician resources
        self.resources['technician_basic'] = AssemblyResource(
            resource_id='technician_basic',
            resource_type=ResourceType.TECHNICIAN,
            capacity=4,  # 4 technicians available
            hourly_cost=35.0,
            setup_time=15.0,
            skill_requirements=['basic_assembly', 'safety_certified']
        )
        
        self.resources['technician_advanced'] = AssemblyResource(
            resource_id='technician_advanced',
            resource_type=ResourceType.TECHNICIAN,
            capacity=2,  # 2 advanced technicians
            hourly_cost=55.0,
            setup_time=10.0,
            skill_requirements=['advanced_assembly', 'electrical_systems', 'safety_certified']
        )
        
        # Assembly fixtures and tooling
        self.resources['main_assembly_fixture'] = AssemblyResource(
            resource_id='main_assembly_fixture',
            resource_type=ResourceType.FIXTURE,
            capacity=1,
            hourly_cost=25.0,
            setup_time=30.0
        )
        
        self.resources['overhead_crane'] = AssemblyResource(
            resource_id='overhead_crane',
            resource_type=ResourceType.CRANE,
            capacity=1,
            hourly_cost=40.0,
            setup_time=20.0,
            skill_requirements=['crane_operator']
        )
        
        # Inspection equipment
        self.resources['cmm_machine'] = AssemblyResource(
            resource_id='cmm_machine',
            resource_type=ResourceType.INSPECTION_EQUIPMENT,
            capacity=1,
            hourly_cost=75.0,
            setup_time=45.0,
            skill_requirements=['metrology_certified']
        )
    
    def _initialize_quality_templates(self):
        """Initialize standard quality checkpoint templates."""
        self.standard_checkpoints['dimensional_check'] = {
            'type': QualityCheckType.DIMENSIONAL,
            'time': 30.0,  # minutes
            'equipment': ['cmm_machine', 'calipers'],
            'criteria': {'tolerance': '±0.1mm', 'surface_finish': 'Ra 3.2'}
        }
        
        self.standard_checkpoints['electrical_continuity'] = {
            'type': QualityCheckType.ELECTRICAL,
            'time': 15.0,
            'equipment': ['multimeter', 'insulation_tester'],
            'criteria': {'resistance': '<1Ω', 'insulation': '>10MΩ'}
        }
        
        self.standard_checkpoints['torque_verification'] = {
            'type': QualityCheckType.TORQUE_CHECK,
            'time': 10.0,
            'equipment': ['torque_wrench', 'torque_analyzer'],
            'criteria': {'torque_spec': 'per_drawing', 'sequence': 'star_pattern'}
        }
    
    def optimize_assembly_sequence(
        self,
        configuration: AircraftConfiguration,
        production_schedule: Dict[str, Any],
        resource_constraints: Dict[str, Any]
    ) -> AssemblySequence:
        """Optimize assembly sequence using constraint-based scheduling."""
        self.logger.info(f"Optimizing assembly sequence for configuration {configuration.config_id}")
        
        # Generate assembly steps from configuration
        assembly_steps = self._generate_assembly_steps(configuration)
        
        # Generate constraints
        constraints = self._generate_assembly_constraints(assembly_steps, configuration)
        
        # Generate quality checkpoints
        quality_checkpoints = self._generate_quality_checkpoints(assembly_steps)
        
        # Perform constraint-based scheduling
        scheduled_steps = self._schedule_assembly_steps(
            assembly_steps, constraints, resource_constraints
        )
        
        # Calculate metrics
        total_time = self._calculate_total_time(scheduled_steps)
        total_cost = self._calculate_total_cost(scheduled_steps, quality_checkpoints)
        resource_utilization = self._calculate_resource_utilization(scheduled_steps)
        critical_path = self._find_critical_path(scheduled_steps, constraints)
        risk_score = self._assess_risk(scheduled_steps, constraints)
        
        return AssemblySequence(
            sequence_id=f"seq_{configuration.config_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            configuration_id=configuration.config_id,
            steps=scheduled_steps,
            constraints=constraints,
            quality_checkpoints=quality_checkpoints,
            total_time=total_time,
            total_cost=total_cost,
            resource_utilization=resource_utilization,
            critical_path=critical_path,
            risk_score=risk_score
        )
    
    def _generate_assembly_steps(self, configuration: AircraftConfiguration) -> List[AssemblyStep]:
        """Generate assembly steps from aircraft configuration."""
        steps = []
        
        # Base platform assembly (always first)
        if configuration.base_platform:
            steps.append(AssemblyStep(
                step_id="base_platform_setup",
                step_name="Base Platform Setup",
                module_id="base_platform",
                estimated_time=120.0,  # 2 hours
                labor_hours=4.0,  # 2 technicians
                required_resources=['main_assembly_fixture', 'technician_basic'],
                skill_level_required=2,
                setup_time=30.0,
                cleanup_time=15.0,
                quality_checkpoints=['dimensional_check'],
                risk_factors={'complexity': 0.3, 'criticality': 0.9}
            ))
        
        # Module assembly steps
        for i, module in enumerate(configuration.modules):
            # Pre-assembly preparation
            prep_step = AssemblyStep(
                step_id=f"prep_{module.module_id}",
                step_name=f"Prepare {module.name}",
                module_id=module.module_id,
                estimated_time=self._estimate_prep_time(module),
                labor_hours=1.0,
                required_resources=['technician_basic'],
                skill_level_required=1,
                setup_time=10.0,
                cleanup_time=5.0,
                quality_checkpoints=['visual'],
                risk_factors={'complexity': 0.2, 'criticality': 0.5}
            )
            steps.append(prep_step)
            
            # Main assembly step
            main_step = AssemblyStep(
                step_id=f"install_{module.module_id}",
                step_name=f"Install {module.name}",
                module_id=module.module_id,
                estimated_time=self._estimate_assembly_time(module),
                labor_hours=self._estimate_labor_hours(module),
                required_resources=self._determine_required_resources(module),
                skill_level_required=self._determine_skill_level(module),
                setup_time=self._estimate_setup_time(module),
                cleanup_time=10.0,
                quality_checkpoints=self._determine_quality_checks(module),
                risk_factors=self._assess_module_risk(module)
            )
            steps.append(main_step)
            
            # Interface connections
            if module.electrical_interfaces or module.mechanical_interfaces:
                connect_step = AssemblyStep(
                    step_id=f"connect_{module.module_id}",
                    step_name=f"Connect {module.name} Interfaces",
                    module_id=module.module_id,
                    estimated_time=self._estimate_connection_time(module),
                    labor_hours=1.5,
                    required_resources=['technician_advanced'],
                    skill_level_required=3,
                    setup_time=5.0,
                    cleanup_time=5.0,
                    quality_checkpoints=['electrical_continuity', 'torque_verification'],
                    risk_factors={'complexity': 0.4, 'criticality': 0.8}
                )
                steps.append(connect_step)
        
        # Final system integration
        steps.append(AssemblyStep(
            step_id="system_integration",
            step_name="System Integration and Test",
            module_id="system",
            estimated_time=180.0,  # 3 hours
            labor_hours=6.0,  # 2 advanced technicians
            required_resources=['technician_advanced', 'cmm_machine'],
            skill_level_required=4,
            setup_time=45.0,
            cleanup_time=30.0,
            quality_checkpoints=['functional', 'dimensional_check', 'electrical_continuity'],
            risk_factors={'complexity': 0.7, 'criticality': 1.0}
        ))
        
        return steps
    
    def _generate_assembly_constraints(
        self,
        steps: List[AssemblyStep],
        configuration: AircraftConfiguration
    ) -> List[AssemblyConstraint]:
        """Generate assembly constraints based on steps and configuration."""
        constraints = []
        
        # Base platform must be first
        base_step = next((s for s in steps if s.step_id == "base_platform_setup"), None)
        if base_step:
            for step in steps:
                if step.step_id != "base_platform_setup":
                    constraints.append(AssemblyConstraint(
                        constraint_id=f"base_before_{step.step_id}",
                        constraint_type=AssemblyConstraintType.PRECEDENCE,
                        source_step="base_platform_setup",
                        target_step=step.step_id,
                        description="Base platform must be assembled first",
                        priority=1
                    ))
        
        # Module preparation must precede installation
        for step in steps:
            if step.step_id.startswith("prep_"):
                module_id = step.module_id
                install_step = next((s for s in steps if s.step_id == f"install_{module_id}"), None)
                if install_step:
                    constraints.append(AssemblyConstraint(
                        constraint_id=f"prep_before_install_{module_id}",
                        constraint_type=AssemblyConstraintType.PRECEDENCE,
                        source_step=step.step_id,
                        target_step=install_step.step_id,
                        description=f"Module {module_id} preparation before installation",
                        priority=1
                    ))
        
        # Installation must precede connection
        for step in steps:
            if step.step_id.startswith("install_"):
                module_id = step.module_id
                connect_step = next((s for s in steps if s.step_id == f"connect_{module_id}"), None)
                if connect_step:
                    constraints.append(AssemblyConstraint(
                        constraint_id=f"install_before_connect_{module_id}",
                        constraint_type=AssemblyConstraintType.PRECEDENCE,
                        source_step=step.step_id,
                        target_step=connect_step.step_id,
                        description=f"Module {module_id} installation before connection",
                        priority=1
                    ))
        
        # System integration must be last
        system_step = next((s for s in steps if s.step_id == "system_integration"), None)
        if system_step:
            for step in steps:
                if step.step_id != "system_integration" and not step.step_id.startswith("prep_"):
                    constraints.append(AssemblyConstraint(
                        constraint_id=f"{step.step_id}_before_integration",
                        constraint_type=AssemblyConstraintType.PRECEDENCE,
                        source_step=step.step_id,
                        target_step="system_integration",
                        description="All assembly before system integration",
                        priority=2
                    ))
        
        # Resource exclusions (crane cannot be used simultaneously)
        crane_steps = [s for s in steps if 'overhead_crane' in s.required_resources]
        for i, step1 in enumerate(crane_steps):
            for step2 in crane_steps[i+1:]:
                constraints.append(AssemblyConstraint(
                    constraint_id=f"crane_exclusion_{step1.step_id}_{step2.step_id}",
                    constraint_type=AssemblyConstraintType.EXCLUSION,
                    source_step=step1.step_id,
                    target_step=step2.step_id,
                    resource_required='overhead_crane',
                    description="Crane cannot be used simultaneously",
                    priority=1
                ))
        
        return constraints
    
    def _generate_quality_checkpoints(self, steps: List[AssemblyStep]) -> List[QualityCheckpoint]:
        """Generate quality checkpoints for assembly steps."""
        checkpoints = []
        
        for step in steps:
            for checkpoint_name in step.quality_checkpoints:
                if checkpoint_name in self.standard_checkpoints:
                    template = self.standard_checkpoints[checkpoint_name]
                    checkpoint = QualityCheckpoint(
                        checkpoint_id=f"{step.step_id}_{checkpoint_name}",
                        checkpoint_type=template['type'],
                        step_id=step.step_id,
                        description=f"{checkpoint_name} for {step.step_name}",
                        inspection_time=template['time'],
                        required_equipment=template['equipment'],
                        acceptance_criteria=template['criteria'],
                        failure_actions=['rework', 'escalate', 'document']
                    )
                    checkpoints.append(checkpoint)
        
        return checkpoints
    
    def _schedule_assembly_steps(
        self,
        steps: List[AssemblyStep],
        constraints: List[AssemblyConstraint],
        resource_constraints: Dict[str, Any]
    ) -> List[AssemblyStep]:
        """Schedule assembly steps using constraint-based scheduling."""
        # Simple priority-based scheduling (can be enhanced with more sophisticated algorithms)
        scheduled_steps = steps.copy()
        
        # Sort steps by priority (base platform first, then by estimated time)
        def step_priority(step):
            if step.step_id == "base_platform_setup":
                return 0
            elif step.step_id.startswith("prep_"):
                return 1
            elif step.step_id.startswith("install_"):
                return 2
            elif step.step_id.startswith("connect_"):
                return 3
            elif step.step_id == "system_integration":
                return 4
            else:
                return 5
        
        scheduled_steps.sort(key=step_priority)
        
        # Assign start and finish times
        current_time = datetime.now()
        
        for step in scheduled_steps:
            # Find earliest possible start time based on constraints
            earliest_start = current_time
            
            # Check precedence constraints
            for constraint in constraints:
                if (constraint.constraint_type == AssemblyConstraintType.PRECEDENCE and 
                    constraint.target_step == step.step_id):
                    # Find the source step
                    source_step = next((s for s in scheduled_steps if s.step_id == constraint.source_step), None)
                    if source_step and source_step.scheduled_finish:
                        earliest_start = max(earliest_start, source_step.scheduled_finish)
            
            step.scheduled_start = earliest_start
            step.scheduled_finish = earliest_start + timedelta(minutes=step.estimated_time + step.setup_time + step.cleanup_time)
            
            # Update current time for next step
            current_time = step.scheduled_finish
        
        return scheduled_steps
    
    def _calculate_total_time(self, steps: List[AssemblyStep]) -> float:
        """Calculate total assembly time."""
        if not steps:
            return 0.0
        
        earliest_start = min(step.scheduled_start for step in steps if step.scheduled_start)
        latest_finish = max(step.scheduled_finish for step in steps if step.scheduled_finish)
        
        return (latest_finish - earliest_start).total_seconds() / 60.0  # minutes
    
    def _calculate_total_cost(self, steps: List[AssemblyStep], checkpoints: List[QualityCheckpoint]) -> float:
        """Calculate total assembly cost."""
        total_cost = 0.0
        
        # Labor costs
        for step in steps:
            total_cost += step.labor_hours * 50.0  # Average $50/hour
        
        # Quality checkpoint costs
        for checkpoint in checkpoints:
            total_cost += checkpoint.inspection_time * 75.0 / 60.0  # $75/hour for inspection
        
        return total_cost
    
    def _calculate_resource_utilization(self, steps: List[AssemblyStep]) -> Dict[str, float]:
        """Calculate resource utilization percentages."""
        utilization = {}
        
        # Calculate total time span
        if not steps:
            return utilization
        
        total_span = self._calculate_total_time(steps)
        
        # Calculate utilization for each resource
        for resource_id in self.resources:
            resource_time = 0.0
            for step in steps:
                if resource_id in step.required_resources:
                    resource_time += step.estimated_time + step.setup_time + step.cleanup_time
            
            utilization[resource_id] = (resource_time / total_span) * 100.0 if total_span > 0 else 0.0
        
        return utilization
    
    def _find_critical_path(self, steps: List[AssemblyStep], constraints: List[AssemblyConstraint]) -> List[str]:
        """Find critical path through assembly sequence."""
        # Simplified critical path - longest sequence of dependent steps
        critical_path = []
        
        # Start with base platform if it exists
        current_step = next((s for s in steps if s.step_id == "base_platform_setup"), None)
        if current_step:
            critical_path.append(current_step.step_id)
        
        # Follow the longest chain of dependencies
        while current_step:
            next_step = None
            max_time = 0.0
            
            # Find the longest dependent step
            for constraint in constraints:
                if (constraint.constraint_type == AssemblyConstraintType.PRECEDENCE and 
                    constraint.source_step == current_step.step_id):
                    candidate = next((s for s in steps if s.step_id == constraint.target_step), None)
                    if candidate and candidate.estimated_time > max_time:
                        next_step = candidate
                        max_time = candidate.estimated_time
            
            if next_step and next_step.step_id not in critical_path:
                critical_path.append(next_step.step_id)
                current_step = next_step
            else:
                break
        
        return critical_path
    
    def _assess_risk(self, steps: List[AssemblyStep], constraints: List[AssemblyConstraint]) -> float:
        """Assess overall assembly risk score."""
        total_risk = 0.0
        total_weight = 0.0
        
        for step in steps:
            step_risk = 0.0
            for factor, value in step.risk_factors.items():
                step_risk += value
            
            # Weight by step duration and criticality
            weight = step.estimated_time * step.risk_factors.get('criticality', 0.5)
            total_risk += step_risk * weight
            total_weight += weight
        
        return total_risk / total_weight if total_weight > 0 else 0.0
    
    def detect_conflicts(self, sequence: AssemblySequence) -> List[ConflictDetection]:
        """Detect conflicts in assembly sequence."""
        conflicts = []
        
        # Resource conflicts
        resource_usage = {}
        for step in sequence.steps:
            if not step.scheduled_start or not step.scheduled_finish:
                continue
                
            for resource in step.required_resources:
                if resource not in resource_usage:
                    resource_usage[resource] = []
                resource_usage[resource].append((step.scheduled_start, step.scheduled_finish, step.step_id))
        
        # Check for overlapping resource usage
        for resource, usage_list in resource_usage.items():
            usage_list.sort(key=lambda x: x[0])  # Sort by start time
            
            for i in range(len(usage_list) - 1):
                current_end = usage_list[i][1]
                next_start = usage_list[i + 1][0]
                
                if current_end > next_start:
                    conflicts.append(ConflictDetection(
                        conflict_id=f"resource_conflict_{resource}_{i}",
                        conflict_type="resource_overlap",
                        affected_steps=[usage_list[i][2], usage_list[i + 1][2]],
                        description=f"Resource {resource} double-booked",
                        severity=2,
                        suggested_resolution="Reschedule one of the conflicting steps"
                    ))
        
        # Constraint violations
        for constraint in sequence.constraints:
            if constraint.constraint_type == AssemblyConstraintType.PRECEDENCE:
                source_step = next((s for s in sequence.steps if s.step_id == constraint.source_step), None)
                target_step = next((s for s in sequence.steps if s.step_id == constraint.target_step), None)
                
                if (source_step and target_step and 
                    source_step.scheduled_finish and target_step.scheduled_start and
                    source_step.scheduled_finish > target_step.scheduled_start):
                    
                    conflicts.append(ConflictDetection(
                        conflict_id=f"precedence_violation_{constraint.constraint_id}",
                        conflict_type="precedence_violation",
                        affected_steps=[constraint.source_step, constraint.target_step],
                        description=f"Precedence constraint violated: {constraint.description}",
                        severity=1,
                        suggested_resolution="Adjust scheduling to respect precedence"
                    ))
        
        return conflicts
    
    # Helper methods for step generation
    def _estimate_prep_time(self, module: Module) -> float:
        """Estimate preparation time for a module."""
        base_time = 30.0  # 30 minutes base
        complexity_factor = len(module.electrical_interfaces) + len(module.mechanical_interfaces)
        return base_time + (complexity_factor * 10.0)
    
    def _estimate_assembly_time(self, module: Module) -> float:
        """Estimate assembly time for a module."""
        base_time = 60.0  # 1 hour base
        
        # Adjust based on module type and complexity
        if module.physical_properties:
            mass_factor = min(module.physical_properties.mass / 100.0, 3.0)  # Max 3x for heavy modules
            base_time *= (1.0 + mass_factor)
        
        interface_factor = (len(module.electrical_interfaces) + len(module.mechanical_interfaces)) * 15.0
        return base_time + interface_factor
    
    def _estimate_labor_hours(self, module: Module) -> float:
        """Estimate labor hours for module assembly."""
        base_hours = 2.0
        complexity = len(module.electrical_interfaces) + len(module.mechanical_interfaces)
        return base_hours + (complexity * 0.5)
    
    def _determine_required_resources(self, module: Module) -> List[str]:
        """Determine required resources for module assembly."""
        resources = ['technician_basic']
        
        # Heavy modules need crane
        if module.physical_properties and module.physical_properties.mass > 50.0:  # 50kg threshold
            resources.append('overhead_crane')
        
        # Complex modules need advanced technicians
        if len(module.electrical_interfaces) > 2 or len(module.mechanical_interfaces) > 2:
            resources.append('technician_advanced')
        
        # All modules need fixture
        resources.append('main_assembly_fixture')
        
        return resources
    
    def _determine_skill_level(self, module: Module) -> int:
        """Determine required skill level for module assembly."""
        base_level = 2
        
        # Increase for complex interfaces
        if len(module.electrical_interfaces) > 3:
            base_level += 1
        
        if len(module.mechanical_interfaces) > 3:
            base_level += 1
        
        return min(base_level, 5)  # Max level 5
    
    def _estimate_setup_time(self, module: Module) -> float:
        """Estimate setup time for module assembly."""
        base_time = 15.0  # 15 minutes base
        
        if module.physical_properties and module.physical_properties.mass > 50.0:
            base_time += 20.0  # Extra time for crane setup
        
        return base_time
    
    def _determine_quality_checks(self, module: Module) -> List[str]:
        """Determine quality checkpoints for module."""
        checks = ['dimensional_check']
        
        if module.electrical_interfaces:
            checks.append('electrical_continuity')
        
        if module.mechanical_interfaces:
            checks.append('torque_verification')
        
        return checks
    
    def _estimate_connection_time(self, module: Module) -> float:
        """Estimate time for connecting module interfaces."""
        base_time = 20.0
        electrical_time = len(module.electrical_interfaces) * 10.0
        mechanical_time = len(module.mechanical_interfaces) * 15.0
        return base_time + electrical_time + mechanical_time
    
    def _assess_module_risk(self, module: Module) -> Dict[str, float]:
        """Assess risk factors for module assembly."""
        risk_factors = {
            'complexity': 0.3,  # Base complexity
            'criticality': 0.5  # Base criticality
        }
        
        # Increase complexity based on interfaces
        interface_count = len(module.electrical_interfaces) + len(module.mechanical_interfaces)
        risk_factors['complexity'] = min(0.3 + (interface_count * 0.1), 1.0)
        
        # Increase criticality for heavy or complex modules
        if module.physical_properties and module.physical_properties.mass > 100.0:
            risk_factors['criticality'] = min(risk_factors['criticality'] + 0.3, 1.0)
        
        return risk_factors