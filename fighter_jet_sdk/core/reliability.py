"""Failure mode analysis and reliability assessment system."""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import logging
from collections import defaultdict

from ..common.interfaces import BaseEngine
from ..common.data_models import AircraftConfiguration, Module
from .errors import ReliabilityError, ValidationError
from .logging import get_logger


class FailureMode(Enum):
    """Failure mode types."""
    CATASTROPHIC = "catastrophic"
    HAZARDOUS = "hazardous"
    MAJOR = "major"
    MINOR = "minor"
    NO_EFFECT = "no_effect"


class FailureCause(Enum):
    """Failure cause categories."""
    MECHANICAL = "mechanical"
    ELECTRICAL = "electrical"
    SOFTWARE = "software"
    HUMAN_ERROR = "human_error"
    ENVIRONMENTAL = "environmental"
    WEAR_OUT = "wear_out"
    RANDOM = "random"


class MaintenanceType(Enum):
    """Maintenance type categories."""
    PREVENTIVE = "preventive"
    CORRECTIVE = "corrective"
    PREDICTIVE = "predictive"
    CONDITION_BASED = "condition_based"


@dataclass
class FailureEvent:
    """Definition of a failure event."""
    event_id: str
    component_id: str
    failure_mode: FailureMode
    failure_cause: FailureCause
    failure_rate: float  # failures per hour
    detection_probability: float  # probability of detecting failure
    repair_time: float  # hours to repair
    cost_impact: float  # cost in dollars
    safety_impact: str
    mission_impact: str
    description: str = ""
    mitigation_strategies: List[str] = field(default_factory=list)


@dataclass
class FaultTreeNode:
    """Node in a fault tree analysis."""
    node_id: str
    description: str
    node_type: str  # 'event', 'and_gate', 'or_gate', 'basic_event'
    probability: float = 0.0
    children: List['FaultTreeNode'] = field(default_factory=list)
    parent: Optional['FaultTreeNode'] = None
    failure_events: List[FailureEvent] = field(default_factory=list)


@dataclass
class ReliabilityMetrics:
    """Reliability metrics for a system or component."""
    component_id: str
    mean_time_between_failures: float  # hours
    mean_time_to_repair: float  # hours
    availability: float  # percentage
    reliability_at_time: Dict[float, float]  # time -> reliability
    failure_rate: float  # failures per hour
    repair_rate: float  # repairs per hour
    confidence_interval: Tuple[float, float] = (0.0, 0.0)


@dataclass
class MaintenanceTask:
    """Maintenance task definition."""
    task_id: str
    component_id: str
    maintenance_type: MaintenanceType
    interval: float  # hours between maintenance
    duration: float  # hours to complete
    cost: float  # cost in dollars
    required_skills: List[str]
    required_tools: List[str]
    description: str = ""
    effectiveness: float = 1.0  # reduction in failure rate


@dataclass
class ReliabilityAssessment:
    """Complete reliability assessment results."""
    assessment_id: str
    aircraft_config_id: str
    system_reliability: ReliabilityMetrics
    component_reliabilities: Dict[str, ReliabilityMetrics]
    fault_trees: Dict[str, FaultTreeNode]
    critical_components: List[str]
    maintenance_plan: List[MaintenanceTask]
    risk_assessment: Dict[str, Any]
    recommendations: List[str]
    assessment_date: str = ""


class FaultTreeAnalyzer:
    """Performs fault tree analysis for system reliability."""
    
    def __init__(self):
        """Initialize fault tree analyzer."""
        self.logger = get_logger("fault_tree_analyzer")
        self.fault_trees: Dict[str, FaultTreeNode] = {}
        self.basic_events: Dict[str, FailureEvent] = {}
    
    def create_fault_tree(self, tree_id: str, top_event: str) -> FaultTreeNode:
        """Create a new fault tree with top event."""
        root_node = FaultTreeNode(
            node_id=f"{tree_id}_root",
            description=top_event,
            node_type="event",
            probability=0.0
        )
        
        self.fault_trees[tree_id] = root_node
        self.logger.info(f"Created fault tree: {tree_id}")
        return root_node
    
    def add_gate(self, parent_node: FaultTreeNode, gate_type: str, 
                 description: str) -> FaultTreeNode:
        """Add a logic gate to the fault tree."""
        if gate_type not in ['and_gate', 'or_gate']:
            raise ValueError(f"Invalid gate type: {gate_type}")
        
        gate_node = FaultTreeNode(
            node_id=f"{parent_node.node_id}_{gate_type}_{len(parent_node.children)}",
            description=description,
            node_type=gate_type,
            parent=parent_node
        )
        
        parent_node.children.append(gate_node)
        return gate_node
    
    def add_basic_event(self, parent_node: FaultTreeNode, 
                       failure_event: FailureEvent) -> FaultTreeNode:
        """Add a basic event to the fault tree."""
        event_node = FaultTreeNode(
            node_id=f"{parent_node.node_id}_event_{failure_event.event_id}",
            description=failure_event.description,
            node_type="basic_event",
            probability=failure_event.failure_rate,
            parent=parent_node,
            failure_events=[failure_event]
        )
        
        parent_node.children.append(event_node)
        self.basic_events[failure_event.event_id] = failure_event
        return event_node
    
    def calculate_tree_probability(self, node: FaultTreeNode, 
                                 mission_time: float = 1.0) -> float:
        """Calculate probability of top event occurring."""
        if node.node_type == "basic_event":
            # Convert failure rate to probability over mission time
            # P(t) = 1 - exp(-λt) for exponential distribution
            failure_rate = node.probability
            probability = 1.0 - np.exp(-failure_rate * mission_time)
            node.probability = probability
            return probability
        
        elif node.node_type == "and_gate":
            # AND gate: all children must occur
            probability = 1.0
            for child in node.children:
                child_prob = self.calculate_tree_probability(child, mission_time)
                probability *= child_prob
            node.probability = probability
            return probability
        
        elif node.node_type == "or_gate":
            # OR gate: at least one child must occur
            probability = 1.0
            for child in node.children:
                child_prob = self.calculate_tree_probability(child, mission_time)
                probability *= (1.0 - child_prob)
            probability = 1.0 - probability
            node.probability = probability
            return probability
        
        else:
            # For event nodes, calculate based on children
            if node.children:
                return self.calculate_tree_probability(node.children[0], mission_time)
            return 0.0
    
    def find_minimal_cut_sets(self, tree_id: str) -> List[List[str]]:
        """Find minimal cut sets for the fault tree."""
        if tree_id not in self.fault_trees:
            raise ValueError(f"Fault tree {tree_id} not found")
        
        root = self.fault_trees[tree_id]
        cut_sets = self._find_cut_sets_recursive(root)
        
        # Remove non-minimal cut sets
        minimal_cut_sets = []
        for cut_set in cut_sets:
            is_minimal = True
            for other_cut_set in cut_sets:
                if cut_set != other_cut_set and set(other_cut_set).issubset(set(cut_set)):
                    is_minimal = False
                    break
            if is_minimal:
                minimal_cut_sets.append(cut_set)
        
        return minimal_cut_sets
    
    def _find_cut_sets_recursive(self, node: FaultTreeNode) -> List[List[str]]:
        """Recursively find cut sets."""
        if node.node_type == "basic_event":
            return [[node.node_id]]
        
        if not node.children:
            return [[]]
        
        if node.node_type == "and_gate":
            # AND gate: Cartesian product of children's cut sets
            result = [[]]
            for child in node.children:
                child_cut_sets = self._find_cut_sets_recursive(child)
                new_result = []
                for existing_set in result:
                    for child_set in child_cut_sets:
                        new_result.append(existing_set + child_set)
                result = new_result
            return result
        
        elif node.node_type == "or_gate":
            # OR gate: Union of children's cut sets
            result = []
            for child in node.children:
                child_cut_sets = self._find_cut_sets_recursive(child)
                result.extend(child_cut_sets)
            return result
        
        else:
            # For other node types, process first child
            if node.children:
                return self._find_cut_sets_recursive(node.children[0])
            return [[]]
    
    def analyze_importance(self, tree_id: str, mission_time: float = 1.0) -> Dict[str, float]:
        """Analyze importance of basic events."""
        if tree_id not in self.fault_trees:
            raise ValueError(f"Fault tree {tree_id} not found")
        
        root = self.fault_trees[tree_id]
        base_probability = self.calculate_tree_probability(root, mission_time)
        
        importance_measures = {}
        
        for event_id, event in self.basic_events.items():
            # Calculate Birnbaum importance
            # Set event probability to 1 and recalculate
            original_rate = event.failure_rate
            event.failure_rate = float('inf')  # Probability = 1
            
            # Find the node with this event
            event_node = self._find_event_node(root, event_id)
            if event_node:
                event_node.probability = 1.0
                prob_with_event = self.calculate_tree_probability(root, mission_time)
                
                # Set event probability to 0 and recalculate
                event.failure_rate = 0.0
                event_node.probability = 0.0
                prob_without_event = self.calculate_tree_probability(root, mission_time)
                
                # Birnbaum importance
                birnbaum_importance = prob_with_event - prob_without_event
                importance_measures[event_id] = birnbaum_importance
                
                # Restore original failure rate
                event.failure_rate = original_rate
        
        # Recalculate with original values
        self.calculate_tree_probability(root, mission_time)
        
        return importance_measures
    
    def _find_event_node(self, node: FaultTreeNode, event_id: str) -> Optional[FaultTreeNode]:
        """Find node containing specific event."""
        if node.node_type == "basic_event" and node.failure_events:
            if node.failure_events[0].event_id == event_id:
                return node
        
        for child in node.children:
            result = self._find_event_node(child, event_id)
            if result:
                return result
        
        return None


class ReliabilityCalculator:
    """Calculates reliability metrics for components and systems."""
    
    def __init__(self):
        """Initialize reliability calculator."""
        self.logger = get_logger("reliability_calculator")
    
    def calculate_component_reliability(self, failure_events: List[FailureEvent],
                                     mission_time: float = 1000.0) -> ReliabilityMetrics:
        """Calculate reliability metrics for a component."""
        if not failure_events:
            raise ValueError("No failure events provided")
        
        component_id = failure_events[0].component_id
        
        # Calculate combined failure rate
        total_failure_rate = sum(event.failure_rate for event in failure_events)
        
        # Calculate MTBF
        mtbf = 1.0 / total_failure_rate if total_failure_rate > 0 else float('inf')
        
        # Calculate MTTR (weighted average)
        total_repair_time = sum(event.repair_time * event.failure_rate for event in failure_events)
        mttr = total_repair_time / total_failure_rate if total_failure_rate > 0 else 0.0
        
        # Calculate availability
        availability = mtbf / (mtbf + mttr) if (mtbf + mttr) > 0 else 1.0
        
        # Calculate reliability over time
        reliability_at_time = {}
        time_points = np.logspace(0, np.log10(mission_time), 50)
        for t in time_points:
            reliability = np.exp(-total_failure_rate * t)
            reliability_at_time[float(t)] = reliability
        
        return ReliabilityMetrics(
            component_id=component_id,
            mean_time_between_failures=mtbf,
            mean_time_to_repair=mttr,
            availability=availability,
            reliability_at_time=reliability_at_time,
            failure_rate=total_failure_rate,
            repair_rate=1.0 / mttr if mttr > 0 else float('inf')
        )
    
    def calculate_system_reliability(self, component_reliabilities: Dict[str, ReliabilityMetrics],
                                   system_architecture: str = "series") -> ReliabilityMetrics:
        """Calculate system reliability from component reliabilities."""
        if not component_reliabilities:
            raise ValueError("No component reliabilities provided")
        
        if system_architecture == "series":
            # Series system: all components must work
            system_failure_rate = sum(comp.failure_rate for comp in component_reliabilities.values())
            system_mtbf = 1.0 / system_failure_rate if system_failure_rate > 0 else float('inf')
            
            # Weighted average MTTR
            total_repair_rate = sum(comp.repair_rate for comp in component_reliabilities.values())
            system_mttr = len(component_reliabilities) / total_repair_rate if total_repair_rate > 0 else 0.0
            
        elif system_architecture == "parallel":
            # Parallel system: at least one component must work
            # This is more complex and requires specific redundancy analysis
            system_failure_rate = 0.0  # Simplified
            system_mtbf = float('inf')  # Simplified
            system_mttr = min(comp.mean_time_to_repair for comp in component_reliabilities.values())
            
        else:
            raise ValueError(f"Unsupported system architecture: {system_architecture}")
        
        # Calculate system availability
        system_availability = system_mtbf / (system_mtbf + system_mttr) if (system_mtbf + system_mttr) > 0 else 1.0
        
        # Calculate system reliability over time
        reliability_at_time = {}
        if component_reliabilities:
            # Get time points from first component
            first_comp = next(iter(component_reliabilities.values()))
            for t, _ in first_comp.reliability_at_time.items():
                if system_architecture == "series":
                    # Series: multiply all component reliabilities
                    system_reliability = 1.0
                    for comp in component_reliabilities.values():
                        system_reliability *= comp.reliability_at_time.get(t, 1.0)
                else:
                    # Parallel: 1 - product of all failure probabilities
                    system_unreliability = 1.0
                    for comp in component_reliabilities.values():
                        comp_reliability = comp.reliability_at_time.get(t, 1.0)
                        system_unreliability *= (1.0 - comp_reliability)
                    system_reliability = 1.0 - system_unreliability
                
                reliability_at_time[t] = system_reliability
        
        return ReliabilityMetrics(
            component_id="system",
            mean_time_between_failures=system_mtbf,
            mean_time_to_repair=system_mttr,
            availability=system_availability,
            reliability_at_time=reliability_at_time,
            failure_rate=system_failure_rate,
            repair_rate=1.0 / system_mttr if system_mttr > 0 else float('inf')
        )
    
    def calculate_confidence_intervals(self, reliability_metrics: ReliabilityMetrics,
                                     confidence_level: float = 0.95,
                                     sample_size: int = 100) -> ReliabilityMetrics:
        """Calculate confidence intervals for reliability metrics."""
        # Simplified confidence interval calculation using normal approximation
        z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
        
        # Standard error approximation
        failure_rate = reliability_metrics.failure_rate
        std_error = np.sqrt(failure_rate / sample_size) if sample_size > 0 else 0.0
        
        lower_bound = max(0.0, failure_rate - z_score * std_error)
        upper_bound = failure_rate + z_score * std_error
        
        reliability_metrics.confidence_interval = (lower_bound, upper_bound)
        return reliability_metrics


class MaintenancePlanner:
    """Plans maintenance schedules for optimal reliability."""
    
    def __init__(self):
        """Initialize maintenance planner."""
        self.logger = get_logger("maintenance_planner")
        self.maintenance_tasks: Dict[str, List[MaintenanceTask]] = defaultdict(list)
    
    def add_maintenance_task(self, task: MaintenanceTask) -> None:
        """Add a maintenance task for a component."""
        self.maintenance_tasks[task.component_id].append(task)
        self.logger.info(f"Added maintenance task {task.task_id} for {task.component_id}")
    
    def optimize_maintenance_schedule(self, component_reliabilities: Dict[str, ReliabilityMetrics],
                                    mission_duration: float = 8760.0,  # 1 year in hours
                                    cost_constraint: float = 1000000.0) -> List[MaintenanceTask]:
        """Optimize maintenance schedule for cost and reliability."""
        optimized_schedule = []
        total_cost = 0.0
        
        # Sort components by criticality (inverse of MTBF)
        sorted_components = sorted(component_reliabilities.items(),
                                 key=lambda x: x[1].mean_time_between_failures)
        
        for component_id, reliability in sorted_components:
            if component_id not in self.maintenance_tasks:
                continue
            
            # Select best maintenance tasks for this component
            component_tasks = self.maintenance_tasks[component_id]
            
            # Simple greedy selection based on effectiveness/cost ratio
            for task in sorted(component_tasks, key=lambda t: t.effectiveness / t.cost, reverse=True):
                # Calculate number of maintenance cycles in mission duration
                cycles = int(mission_duration / task.interval) if task.interval > 0 else 1
                task_total_cost = task.cost * cycles
                
                if total_cost + task_total_cost <= cost_constraint:
                    optimized_schedule.append(task)
                    total_cost += task_total_cost
                    self.logger.info(f"Added {task.task_id} to schedule, cost: ${task_total_cost}")
                else:
                    self.logger.warning(f"Skipped {task.task_id} due to cost constraint")
        
        return optimized_schedule
    
    def calculate_maintenance_effectiveness(self, schedule: List[MaintenanceTask],
                                         original_reliability: ReliabilityMetrics) -> Dict[str, float]:
        """Calculate effectiveness of maintenance schedule."""
        # Group tasks by component
        component_tasks = defaultdict(list)
        for task in schedule:
            component_tasks[task.component_id].append(task)
        
        effectiveness = {}
        
        for component_id, tasks in component_tasks.items():
            # Calculate combined effectiveness
            combined_effectiveness = 1.0
            for task in tasks:
                # Assume multiplicative effect of maintenance tasks
                combined_effectiveness *= (1.0 - task.effectiveness)
            
            # Final effectiveness is reduction in failure rate
            failure_rate_reduction = 1.0 - combined_effectiveness
            effectiveness[component_id] = failure_rate_reduction
        
        return effectiveness
    
    def generate_maintenance_calendar(self, schedule: List[MaintenanceTask],
                                    start_date: str = "2024-01-01") -> Dict[str, List[Dict[str, Any]]]:
        """Generate maintenance calendar from schedule."""
        calendar = defaultdict(list)
        
        # Simple calendar generation (would need proper date handling in production)
        for task in schedule:
            # Calculate maintenance dates based on interval
            current_hour = 0.0
            while current_hour < 8760.0:  # One year
                maintenance_date = f"Hour {current_hour:.0f}"
                calendar[task.component_id].append({
                    'date': maintenance_date,
                    'task_id': task.task_id,
                    'duration': task.duration,
                    'cost': task.cost,
                    'description': task.description
                })
                current_hour += task.interval
        
        return dict(calendar)


class RiskAssessmentEngine:
    """Performs risk assessment based on failure modes and effects."""
    
    def __init__(self):
        """Initialize risk assessment engine."""
        self.logger = get_logger("risk_assessment_engine")
        self.risk_matrix = self._create_default_risk_matrix()
    
    def _create_default_risk_matrix(self) -> Dict[Tuple[str, str], str]:
        """Create default risk assessment matrix."""
        # Risk = Probability × Severity
        matrix = {}
        
        probabilities = ['very_low', 'low', 'medium', 'high', 'very_high']
        severities = ['negligible', 'minor', 'major', 'hazardous', 'catastrophic']
        
        for i, prob in enumerate(probabilities):
            for j, sev in enumerate(severities):
                risk_score = i + j
                if risk_score <= 2:
                    risk_level = 'low'
                elif risk_score <= 4:
                    risk_level = 'medium'
                elif risk_score <= 6:
                    risk_level = 'high'
                else:
                    risk_level = 'critical'
                
                matrix[(prob, sev)] = risk_level
        
        return matrix
    
    def assess_failure_risk(self, failure_event: FailureEvent,
                          mission_time: float = 1.0) -> Dict[str, Any]:
        """Assess risk for a single failure event."""
        # Convert failure rate to probability category
        probability = 1.0 - np.exp(-failure_event.failure_rate * mission_time)
        
        if probability < 1e-6:
            prob_category = 'very_low'
        elif probability < 1e-4:
            prob_category = 'low'
        elif probability < 1e-2:
            prob_category = 'medium'
        elif probability < 1e-1:
            prob_category = 'high'
        else:
            prob_category = 'very_high'
        
        # Map failure mode to severity
        severity_map = {
            FailureMode.NO_EFFECT: 'negligible',
            FailureMode.MINOR: 'minor',
            FailureMode.MAJOR: 'major',
            FailureMode.HAZARDOUS: 'hazardous',
            FailureMode.CATASTROPHIC: 'catastrophic'
        }
        
        severity = severity_map.get(failure_event.failure_mode, 'minor')
        risk_level = self.risk_matrix.get((prob_category, severity), 'medium')
        
        return {
            'event_id': failure_event.event_id,
            'probability': probability,
            'probability_category': prob_category,
            'severity': severity,
            'risk_level': risk_level,
            'cost_impact': failure_event.cost_impact,
            'safety_impact': failure_event.safety_impact,
            'mission_impact': failure_event.mission_impact,
            'mitigation_strategies': failure_event.mitigation_strategies
        }
    
    def assess_system_risk(self, failure_events: List[FailureEvent],
                          mission_time: float = 1.0) -> Dict[str, Any]:
        """Assess overall system risk."""
        individual_risks = [self.assess_failure_risk(event, mission_time) 
                          for event in failure_events]
        
        # Categorize risks by level
        risk_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        total_cost_impact = 0.0
        critical_events = []
        
        for risk in individual_risks:
            risk_counts[risk['risk_level']] += 1
            total_cost_impact += risk['cost_impact']
            
            if risk['risk_level'] in ['high', 'critical']:
                critical_events.append(risk['event_id'])
        
        # Determine overall system risk level
        if risk_counts['critical'] > 0:
            overall_risk = 'critical'
        elif risk_counts['high'] > 2:
            overall_risk = 'high'
        elif risk_counts['high'] > 0 or risk_counts['medium'] > 5:
            overall_risk = 'medium'
        else:
            overall_risk = 'low'
        
        return {
            'overall_risk_level': overall_risk,
            'risk_distribution': risk_counts,
            'total_cost_impact': total_cost_impact,
            'critical_events': critical_events,
            'individual_risks': individual_risks,
            'recommendations': self._generate_risk_recommendations(individual_risks)
        }
    
    def _generate_risk_recommendations(self, risks: List[Dict[str, Any]]) -> List[str]:
        """Generate risk mitigation recommendations."""
        recommendations = []
        
        critical_risks = [r for r in risks if r['risk_level'] == 'critical']
        high_risks = [r for r in risks if r['risk_level'] == 'high']
        
        if critical_risks:
            recommendations.append("Immediate action required for critical risk events")
            recommendations.append("Consider design changes to eliminate critical failure modes")
        
        if high_risks:
            recommendations.append("Implement additional monitoring for high-risk components")
            recommendations.append("Increase maintenance frequency for high-risk items")
        
        if len([r for r in risks if r['cost_impact'] > 100000]) > 0:
            recommendations.append("Consider redundancy for high-cost impact failures")
        
        return recommendations


class ReliabilityAssessmentEngine(BaseEngine):
    """Main engine for reliability assessment and failure mode analysis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize reliability assessment engine."""
        super().__init__(config)
        self.fault_tree_analyzer = FaultTreeAnalyzer()
        self.reliability_calculator = ReliabilityCalculator()
        self.maintenance_planner = MaintenancePlanner()
        self.risk_assessor = RiskAssessmentEngine()
        self.logger = get_logger("reliability_assessment_engine")
    
    def initialize(self) -> bool:
        """Initialize the reliability assessment engine."""
        try:
            self.initialized = True
            self.logger.info("Reliability assessment engine initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    def validate_input(self, data: Any) -> bool:
        """Validate input data for reliability assessment."""
        if isinstance(data, AircraftConfiguration):
            return True
        if isinstance(data, dict) and 'failure_events' in data:
            return True
        return False
    
    def process(self, data: Any) -> Any:
        """Process reliability assessment."""
        if isinstance(data, AircraftConfiguration):
            return self.assess_aircraft_reliability(data)
        elif isinstance(data, dict) and 'failure_events' in data:
            return self.analyze_failure_modes(data['failure_events'])
        return None
    
    def assess_aircraft_reliability(self, config: AircraftConfiguration,
                                  mission_time: float = 1000.0) -> ReliabilityAssessment:
        """Perform complete reliability assessment for aircraft configuration."""
        try:
            self.logger.info(f"Starting reliability assessment for aircraft configuration")
            
            # Generate failure events for aircraft components
            failure_events = self._generate_failure_events(config)
            
            # Calculate component reliabilities
            component_reliabilities = {}
            component_events = defaultdict(list)
            
            for event in failure_events:
                component_events[event.component_id].append(event)
            
            for component_id, events in component_events.items():
                reliability = self.reliability_calculator.calculate_component_reliability(
                    events, mission_time)
                component_reliabilities[component_id] = reliability
            
            # Calculate system reliability
            system_reliability = self.reliability_calculator.calculate_system_reliability(
                component_reliabilities, "series")
            
            # Create fault trees for critical systems
            fault_trees = self._create_fault_trees(failure_events)
            
            # Identify critical components
            critical_components = self._identify_critical_components(component_reliabilities)
            
            # Generate maintenance plan
            maintenance_plan = self._generate_maintenance_plan(component_reliabilities)
            
            # Perform risk assessment
            risk_assessment = self.risk_assessor.assess_system_risk(failure_events, mission_time)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                component_reliabilities, risk_assessment, critical_components)
            
            assessment = ReliabilityAssessment(
                assessment_id=f"assessment_{int(time.time())}",
                aircraft_config_id=getattr(config, 'config_id', 'unknown'),
                system_reliability=system_reliability,
                component_reliabilities=component_reliabilities,
                fault_trees=fault_trees,
                critical_components=critical_components,
                maintenance_plan=maintenance_plan,
                risk_assessment=risk_assessment,
                recommendations=recommendations,
                assessment_date=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            self.logger.info("Reliability assessment completed successfully")
            return assessment
            
        except Exception as e:
            self.logger.error(f"Reliability assessment failed: {e}")
            raise ReliabilityError(f"Reliability assessment failed: {e}")
    
    def analyze_failure_modes(self, failure_events: List[FailureEvent]) -> Dict[str, Any]:
        """Analyze failure modes and effects."""
        analysis = {
            'failure_mode_analysis': {},
            'criticality_analysis': {},
            'risk_assessment': {},
            'recommendations': []
        }
        
        # Group by failure mode
        mode_groups = defaultdict(list)
        for event in failure_events:
            mode_groups[event.failure_mode].append(event)
        
        # Analyze each failure mode
        for mode, events in mode_groups.items():
            total_rate = sum(event.failure_rate for event in events)
            total_cost = sum(event.cost_impact for event in events)
            
            analysis['failure_mode_analysis'][mode.value] = {
                'event_count': len(events),
                'total_failure_rate': total_rate,
                'total_cost_impact': total_cost,
                'events': [event.event_id for event in events]
            }
        
        # Criticality analysis
        sorted_events = sorted(failure_events, 
                             key=lambda e: e.failure_rate * e.cost_impact, 
                             reverse=True)
        
        analysis['criticality_analysis'] = {
            'most_critical': sorted_events[0].event_id if sorted_events else None,
            'top_10_critical': [e.event_id for e in sorted_events[:10]],
            'criticality_scores': {e.event_id: e.failure_rate * e.cost_impact 
                                 for e in sorted_events}
        }
        
        # Risk assessment
        analysis['risk_assessment'] = self.risk_assessor.assess_system_risk(failure_events)
        
        return analysis
    
    def _generate_failure_events(self, config: AircraftConfiguration) -> List[FailureEvent]:
        """Generate failure events for aircraft components."""
        failure_events = []
        
        # Get modules from configuration
        modules = getattr(config, 'modules', [])
        if not modules:
            # Create default modules for demonstration
            modules = [
                type('Module', (), {'module_id': 'engine', 'module_type': 'propulsion'})(),
                type('Module', (), {'module_id': 'avionics', 'module_type': 'electronics'})(),
                type('Module', (), {'module_id': 'flight_controls', 'module_type': 'control'})(),
                type('Module', (), {'module_id': 'landing_gear', 'module_type': 'mechanical'})()
            ]
        
        # Generate failure events for each module
        for module in modules:
            module_id = getattr(module, 'module_id', f'module_{id(module)}')
            module_type = getattr(module, 'module_type', 'unknown')
            
            # Generate typical failure events based on module type
            if module_type in ['propulsion', 'engine']:
                failure_events.extend([
                    FailureEvent(
                        event_id=f"{module_id}_engine_failure",
                        component_id=module_id,
                        failure_mode=FailureMode.CATASTROPHIC,
                        failure_cause=FailureCause.MECHANICAL,
                        failure_rate=1e-5,  # per hour
                        detection_probability=0.9,
                        repair_time=24.0,
                        cost_impact=500000.0,
                        safety_impact="Loss of aircraft",
                        mission_impact="Mission abort",
                        description="Complete engine failure"
                    ),
                    FailureEvent(
                        event_id=f"{module_id}_performance_degradation",
                        component_id=module_id,
                        failure_mode=FailureMode.MAJOR,
                        failure_cause=FailureCause.WEAR_OUT,
                        failure_rate=1e-4,
                        detection_probability=0.8,
                        repair_time=8.0,
                        cost_impact=50000.0,
                        safety_impact="Reduced performance",
                        mission_impact="Mission degradation",
                        description="Engine performance degradation"
                    )
                ])
            
            elif module_type in ['electronics', 'avionics']:
                failure_events.extend([
                    FailureEvent(
                        event_id=f"{module_id}_software_failure",
                        component_id=module_id,
                        failure_mode=FailureMode.HAZARDOUS,
                        failure_cause=FailureCause.SOFTWARE,
                        failure_rate=5e-5,
                        detection_probability=0.7,
                        repair_time=2.0,
                        cost_impact=10000.0,
                        safety_impact="System malfunction",
                        mission_impact="Partial capability loss",
                        description="Software system failure"
                    ),
                    FailureEvent(
                        event_id=f"{module_id}_hardware_failure",
                        component_id=module_id,
                        failure_mode=FailureMode.MAJOR,
                        failure_cause=FailureCause.ELECTRICAL,
                        failure_rate=2e-5,
                        detection_probability=0.9,
                        repair_time=4.0,
                        cost_impact=25000.0,
                        safety_impact="System unavailable",
                        mission_impact="Capability loss",
                        description="Hardware component failure"
                    )
                ])
            
            else:
                # Generic failure events
                failure_events.append(
                    FailureEvent(
                        event_id=f"{module_id}_generic_failure",
                        component_id=module_id,
                        failure_mode=FailureMode.MINOR,
                        failure_cause=FailureCause.RANDOM,
                        failure_rate=1e-4,
                        detection_probability=0.8,
                        repair_time=4.0,
                        cost_impact=5000.0,
                        safety_impact="Minor impact",
                        mission_impact="Minimal impact",
                        description="Generic component failure"
                    )
                )
        
        return failure_events
    
    def _create_fault_trees(self, failure_events: List[FailureEvent]) -> Dict[str, FaultTreeNode]:
        """Create fault trees for critical failure modes."""
        fault_trees = {}
        
        # Create fault tree for aircraft loss
        aircraft_loss_tree = self.fault_tree_analyzer.create_fault_tree(
            "aircraft_loss", "Loss of Aircraft")
        
        # Add OR gate for different loss scenarios
        loss_scenarios = self.fault_tree_analyzer.add_gate(
            aircraft_loss_tree, "or_gate", "Aircraft Loss Scenarios")
        
        # Add catastrophic failures
        catastrophic_events = [e for e in failure_events 
                             if e.failure_mode == FailureMode.CATASTROPHIC]
        
        for event in catastrophic_events:
            self.fault_tree_analyzer.add_basic_event(loss_scenarios, event)
        
        fault_trees["aircraft_loss"] = aircraft_loss_tree
        
        return fault_trees
    
    def _identify_critical_components(self, reliabilities: Dict[str, ReliabilityMetrics]) -> List[str]:
        """Identify critical components based on reliability metrics."""
        # Sort by MTBF (ascending) and failure rate (descending)
        sorted_components = sorted(reliabilities.items(),
                                 key=lambda x: (x[1].mean_time_between_failures, -x[1].failure_rate))
        
        # Take bottom 20% as critical
        critical_count = max(1, len(sorted_components) // 5)
        critical_components = [comp_id for comp_id, _ in sorted_components[:critical_count]]
        
        return critical_components
    
    def _generate_maintenance_plan(self, reliabilities: Dict[str, ReliabilityMetrics]) -> List[MaintenanceTask]:
        """Generate maintenance plan based on reliability analysis."""
        maintenance_tasks = []
        
        for component_id, reliability in reliabilities.items():
            # Generate preventive maintenance task
            interval = reliability.mean_time_between_failures * 0.1  # 10% of MTBF
            
            task = MaintenanceTask(
                task_id=f"{component_id}_preventive",
                component_id=component_id,
                maintenance_type=MaintenanceType.PREVENTIVE,
                interval=interval,
                duration=2.0,  # 2 hours
                cost=1000.0,
                required_skills=["technician"],
                required_tools=["basic_tools"],
                description=f"Preventive maintenance for {component_id}",
                effectiveness=0.3  # 30% reduction in failure rate
            )
            
            self.maintenance_planner.add_maintenance_task(task)
            maintenance_tasks.append(task)
        
        return maintenance_tasks
    
    def _generate_recommendations(self, reliabilities: Dict[str, ReliabilityMetrics],
                                risk_assessment: Dict[str, Any],
                                critical_components: List[str]) -> List[str]:
        """Generate reliability improvement recommendations."""
        recommendations = []
        
        # Critical component recommendations
        if critical_components:
            recommendations.append(f"Focus reliability improvement efforts on critical components: {', '.join(critical_components)}")
        
        # Low reliability components
        low_reliability_components = [comp_id for comp_id, rel in reliabilities.items()
                                    if rel.availability < 0.9]
        if low_reliability_components:
            recommendations.append(f"Consider design improvements for low-availability components: {', '.join(low_reliability_components)}")
        
        # Risk-based recommendations
        if risk_assessment['overall_risk_level'] in ['high', 'critical']:
            recommendations.append("Overall system risk is high - consider additional redundancy")
        
        recommendations.extend(risk_assessment.get('recommendations', []))
        
        # Maintenance recommendations
        high_mttr_components = [comp_id for comp_id, rel in reliabilities.items()
                              if rel.mean_time_to_repair > 8.0]
        if high_mttr_components:
            recommendations.append(f"Improve maintainability for components with high repair times: {', '.join(high_mttr_components)}")
        
        return recommendations