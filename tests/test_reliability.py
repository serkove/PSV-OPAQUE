"""Tests for failure mode analysis and reliability assessment."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from fighter_jet_sdk.core.reliability import (
    ReliabilityAssessmentEngine,
    FaultTreeAnalyzer,
    ReliabilityCalculator,
    MaintenancePlanner,
    RiskAssessmentEngine,
    FailureEvent,
    FaultTreeNode,
    ReliabilityMetrics,
    MaintenanceTask,
    FailureMode,
    FailureCause,
    MaintenanceType
)
from fighter_jet_sdk.common.data_models import AircraftConfiguration
from fighter_jet_sdk.core.errors import ReliabilityError


class TestFailureEvent:
    """Test failure event data structure."""
    
    def test_failure_event_creation(self):
        """Test failure event creation."""
        event = FailureEvent(
            event_id="engine_failure_001",
            component_id="engine_1",
            failure_mode=FailureMode.CATASTROPHIC,
            failure_cause=FailureCause.MECHANICAL,
            failure_rate=1e-5,
            detection_probability=0.9,
            repair_time=24.0,
            cost_impact=500000.0,
            safety_impact="Loss of aircraft",
            mission_impact="Mission abort",
            description="Complete engine failure"
        )
        
        assert event.event_id == "engine_failure_001"
        assert event.failure_mode == FailureMode.CATASTROPHIC
        assert event.failure_cause == FailureCause.MECHANICAL
        assert event.failure_rate == 1e-5
        assert event.cost_impact == 500000.0


class TestFaultTreeAnalyzer:
    """Test fault tree analyzer."""
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = FaultTreeAnalyzer()
        assert len(analyzer.fault_trees) == 0
        assert len(analyzer.basic_events) == 0
    
    def test_create_fault_tree(self):
        """Test fault tree creation."""
        analyzer = FaultTreeAnalyzer()
        
        root = analyzer.create_fault_tree("test_tree", "System Failure")
        
        assert "test_tree" in analyzer.fault_trees
        assert root.description == "System Failure"
        assert root.node_type == "event"
    
    def test_add_gate(self):
        """Test adding logic gates."""
        analyzer = FaultTreeAnalyzer()
        root = analyzer.create_fault_tree("test_tree", "System Failure")
        
        and_gate = analyzer.add_gate(root, "and_gate", "Both components fail")
        or_gate = analyzer.add_gate(root, "or_gate", "Either component fails")
        
        assert and_gate.node_type == "and_gate"
        assert or_gate.node_type == "or_gate"
        assert len(root.children) == 2
        assert and_gate.parent == root
    
    def test_add_gate_invalid_type(self):
        """Test adding invalid gate type."""
        analyzer = FaultTreeAnalyzer()
        root = analyzer.create_fault_tree("test_tree", "System Failure")
        
        with pytest.raises(ValueError, match="Invalid gate type"):
            analyzer.add_gate(root, "invalid_gate", "Invalid")
    
    def test_add_basic_event(self):
        """Test adding basic events."""
        analyzer = FaultTreeAnalyzer()
        root = analyzer.create_fault_tree("test_tree", "System Failure")
        
        failure_event = FailureEvent(
            event_id="component_failure",
            component_id="component_1",
            failure_mode=FailureMode.MAJOR,
            failure_cause=FailureCause.ELECTRICAL,
            failure_rate=1e-4,
            detection_probability=0.8,
            repair_time=4.0,
            cost_impact=10000.0,
            safety_impact="System degradation",
            mission_impact="Reduced capability"
        )
        
        event_node = analyzer.add_basic_event(root, failure_event)
        
        assert event_node.node_type == "basic_event"
        assert event_node.probability == 1e-4
        assert len(event_node.failure_events) == 1
        assert "component_failure" in analyzer.basic_events
    
    def test_calculate_tree_probability_basic_event(self):
        """Test probability calculation for basic event."""
        analyzer = FaultTreeAnalyzer()
        root = analyzer.create_fault_tree("test_tree", "System Failure")
        
        failure_event = FailureEvent(
            event_id="test_failure",
            component_id="test_component",
            failure_mode=FailureMode.MINOR,
            failure_cause=FailureCause.RANDOM,
            failure_rate=1e-3,
            detection_probability=0.9,
            repair_time=2.0,
            cost_impact=1000.0,
            safety_impact="Minor",
            mission_impact="Minor"
        )
        
        event_node = analyzer.add_basic_event(root, failure_event)
        
        # Calculate probability for 100 hour mission
        prob = analyzer.calculate_tree_probability(event_node, 100.0)
        
        # Should be approximately 1 - exp(-0.001 * 100) ≈ 0.095
        expected_prob = 1.0 - np.exp(-1e-3 * 100.0)
        assert abs(prob - expected_prob) < 1e-6
    
    def test_calculate_tree_probability_and_gate(self):
        """Test probability calculation for AND gate."""
        analyzer = FaultTreeAnalyzer()
        root = analyzer.create_fault_tree("test_tree", "System Failure")
        and_gate = analyzer.add_gate(root, "and_gate", "Both fail")
        
        # Add two basic events
        event1 = FailureEvent(
            event_id="failure_1", component_id="comp_1", failure_mode=FailureMode.MINOR,
            failure_cause=FailureCause.RANDOM, failure_rate=1e-3, detection_probability=0.9,
            repair_time=2.0, cost_impact=1000.0, safety_impact="Minor", mission_impact="Minor"
        )
        event2 = FailureEvent(
            event_id="failure_2", component_id="comp_2", failure_mode=FailureMode.MINOR,
            failure_cause=FailureCause.RANDOM, failure_rate=2e-3, detection_probability=0.9,
            repair_time=2.0, cost_impact=1000.0, safety_impact="Minor", mission_impact="Minor"
        )
        
        analyzer.add_basic_event(and_gate, event1)
        analyzer.add_basic_event(and_gate, event2)
        
        prob = analyzer.calculate_tree_probability(and_gate, 100.0)
        
        # AND gate: P1 * P2
        p1 = 1.0 - np.exp(-1e-3 * 100.0)
        p2 = 1.0 - np.exp(-2e-3 * 100.0)
        expected_prob = p1 * p2
        
        assert abs(prob - expected_prob) < 1e-6
    
    def test_calculate_tree_probability_or_gate(self):
        """Test probability calculation for OR gate."""
        analyzer = FaultTreeAnalyzer()
        root = analyzer.create_fault_tree("test_tree", "System Failure")
        or_gate = analyzer.add_gate(root, "or_gate", "Either fails")
        
        # Add two basic events
        event1 = FailureEvent(
            event_id="failure_1", component_id="comp_1", failure_mode=FailureMode.MINOR,
            failure_cause=FailureCause.RANDOM, failure_rate=1e-3, detection_probability=0.9,
            repair_time=2.0, cost_impact=1000.0, safety_impact="Minor", mission_impact="Minor"
        )
        event2 = FailureEvent(
            event_id="failure_2", component_id="comp_2", failure_mode=FailureMode.MINOR,
            failure_cause=FailureCause.RANDOM, failure_rate=2e-3, detection_probability=0.9,
            repair_time=2.0, cost_impact=1000.0, safety_impact="Minor", mission_impact="Minor"
        )
        
        analyzer.add_basic_event(or_gate, event1)
        analyzer.add_basic_event(or_gate, event2)
        
        prob = analyzer.calculate_tree_probability(or_gate, 100.0)
        
        # OR gate: 1 - (1-P1) * (1-P2)
        p1 = 1.0 - np.exp(-1e-3 * 100.0)
        p2 = 1.0 - np.exp(-2e-3 * 100.0)
        expected_prob = 1.0 - (1.0 - p1) * (1.0 - p2)
        
        assert abs(prob - expected_prob) < 1e-6
    
    def test_find_minimal_cut_sets(self):
        """Test finding minimal cut sets."""
        analyzer = FaultTreeAnalyzer()
        root = analyzer.create_fault_tree("test_tree", "System Failure")
        or_gate = analyzer.add_gate(root, "or_gate", "Either path fails")
        
        # Add basic events
        event1 = FailureEvent(
            event_id="failure_1", component_id="comp_1", failure_mode=FailureMode.MINOR,
            failure_cause=FailureCause.RANDOM, failure_rate=1e-3, detection_probability=0.9,
            repair_time=2.0, cost_impact=1000.0, safety_impact="Minor", mission_impact="Minor"
        )
        event2 = FailureEvent(
            event_id="failure_2", component_id="comp_2", failure_mode=FailureMode.MINOR,
            failure_cause=FailureCause.RANDOM, failure_rate=2e-3, detection_probability=0.9,
            repair_time=2.0, cost_impact=1000.0, safety_impact="Minor", mission_impact="Minor"
        )
        
        node1 = analyzer.add_basic_event(or_gate, event1)
        node2 = analyzer.add_basic_event(or_gate, event2)
        
        cut_sets = analyzer.find_minimal_cut_sets("test_tree")
        
        # Should have two minimal cut sets, each with one event
        assert len(cut_sets) == 2
        assert len(cut_sets[0]) == 1
        assert len(cut_sets[1]) == 1


class TestReliabilityCalculator:
    """Test reliability calculator."""
    
    def test_calculator_initialization(self):
        """Test calculator initialization."""
        calculator = ReliabilityCalculator()
        assert calculator.logger is not None
    
    def test_calculate_component_reliability(self):
        """Test component reliability calculation."""
        calculator = ReliabilityCalculator()
        
        failure_events = [
            FailureEvent(
                event_id="failure_1", component_id="test_component", failure_mode=FailureMode.MINOR,
                failure_cause=FailureCause.RANDOM, failure_rate=1e-4, detection_probability=0.9,
                repair_time=4.0, cost_impact=5000.0, safety_impact="Minor", mission_impact="Minor"
            ),
            FailureEvent(
                event_id="failure_2", component_id="test_component", failure_mode=FailureMode.MAJOR,
                failure_cause=FailureCause.WEAR_OUT, failure_rate=2e-5, detection_probability=0.8,
                repair_time=8.0, cost_impact=15000.0, safety_impact="Major", mission_impact="Major"
            )
        ]
        
        reliability = calculator.calculate_component_reliability(failure_events, 1000.0)
        
        assert reliability.component_id == "test_component"
        assert reliability.failure_rate == 1.2e-4  # Sum of failure rates
        assert reliability.mean_time_between_failures == 1.0 / 1.2e-4
        assert 0.0 <= reliability.availability <= 1.0
        assert len(reliability.reliability_at_time) > 0
    
    def test_calculate_component_reliability_no_events(self):
        """Test component reliability calculation with no events."""
        calculator = ReliabilityCalculator()
        
        with pytest.raises(ValueError, match="No failure events provided"):
            calculator.calculate_component_reliability([], 1000.0)
    
    def test_calculate_system_reliability_series(self):
        """Test system reliability calculation for series system."""
        calculator = ReliabilityCalculator()
        
        component_reliabilities = {
            "comp_1": ReliabilityMetrics(
                component_id="comp_1",
                mean_time_between_failures=10000.0,
                mean_time_to_repair=4.0,
                availability=0.9996,
                reliability_at_time={100.0: 0.99, 1000.0: 0.90},
                failure_rate=1e-4,
                repair_rate=0.25
            ),
            "comp_2": ReliabilityMetrics(
                component_id="comp_2",
                mean_time_between_failures=5000.0,
                mean_time_to_repair=2.0,
                availability=0.9996,
                reliability_at_time={100.0: 0.98, 1000.0: 0.82},
                failure_rate=2e-4,
                repair_rate=0.5
            )
        }
        
        system_reliability = calculator.calculate_system_reliability(
            component_reliabilities, "series")
        
        assert system_reliability.component_id == "system"
        assert abs(system_reliability.failure_rate - 3e-4) < 1e-10  # Sum for series
        assert abs(system_reliability.mean_time_between_failures - 1.0 / 3e-4) < 1e-6
        
        # Series system reliability should be product of component reliabilities
        assert abs(system_reliability.reliability_at_time[100.0] - 0.99 * 0.98) < 1e-6
    
    def test_calculate_system_reliability_parallel(self):
        """Test system reliability calculation for parallel system."""
        calculator = ReliabilityCalculator()
        
        component_reliabilities = {
            "comp_1": ReliabilityMetrics(
                component_id="comp_1",
                mean_time_between_failures=1000.0,
                mean_time_to_repair=4.0,
                availability=0.996,
                reliability_at_time={100.0: 0.90},
                failure_rate=1e-3,
                repair_rate=0.25
            ),
            "comp_2": ReliabilityMetrics(
                component_id="comp_2",
                mean_time_between_failures=1000.0,
                mean_time_to_repair=2.0,
                availability=0.998,
                reliability_at_time={100.0: 0.90},
                failure_rate=1e-3,
                repair_rate=0.5
            )
        }
        
        system_reliability = calculator.calculate_system_reliability(
            component_reliabilities, "parallel")
        
        assert system_reliability.component_id == "system"
        # Parallel system should have higher reliability than individual components
        assert system_reliability.reliability_at_time[100.0] > 0.90
    
    def test_calculate_system_reliability_invalid_architecture(self):
        """Test system reliability calculation with invalid architecture."""
        calculator = ReliabilityCalculator()
        
        component_reliabilities = {"comp_1": Mock()}
        
        with pytest.raises(ValueError, match="Unsupported system architecture"):
            calculator.calculate_system_reliability(component_reliabilities, "invalid")
    
    def test_calculate_confidence_intervals(self):
        """Test confidence interval calculation."""
        calculator = ReliabilityCalculator()
        
        reliability = ReliabilityMetrics(
            component_id="test_component",
            mean_time_between_failures=1000.0,
            mean_time_to_repair=4.0,
            availability=0.996,
            reliability_at_time={},
            failure_rate=1e-3,
            repair_rate=0.25
        )
        
        updated_reliability = calculator.calculate_confidence_intervals(
            reliability, confidence_level=0.95, sample_size=100)
        
        assert updated_reliability.confidence_interval != (0.0, 0.0)
        assert updated_reliability.confidence_interval[0] <= updated_reliability.failure_rate
        assert updated_reliability.confidence_interval[1] >= updated_reliability.failure_rate


class TestMaintenancePlanner:
    """Test maintenance planner."""
    
    def test_planner_initialization(self):
        """Test planner initialization."""
        planner = MaintenancePlanner()
        assert len(planner.maintenance_tasks) == 0
    
    def test_add_maintenance_task(self):
        """Test adding maintenance task."""
        planner = MaintenancePlanner()
        
        task = MaintenanceTask(
            task_id="preventive_001",
            component_id="engine_1",
            maintenance_type=MaintenanceType.PREVENTIVE,
            interval=100.0,
            duration=4.0,
            cost=2000.0,
            required_skills=["mechanic"],
            required_tools=["wrench", "scanner"],
            description="Engine preventive maintenance"
        )
        
        planner.add_maintenance_task(task)
        
        assert "engine_1" in planner.maintenance_tasks
        assert len(planner.maintenance_tasks["engine_1"]) == 1
        assert planner.maintenance_tasks["engine_1"][0] == task
    
    def test_optimize_maintenance_schedule(self):
        """Test maintenance schedule optimization."""
        planner = MaintenancePlanner()
        
        # Add maintenance tasks
        task1 = MaintenanceTask(
            task_id="task_1", component_id="comp_1", maintenance_type=MaintenanceType.PREVENTIVE,
            interval=100.0, duration=2.0, cost=1000.0, required_skills=[], required_tools=[],
            effectiveness=0.3
        )
        task2 = MaintenanceTask(
            task_id="task_2", component_id="comp_1", maintenance_type=MaintenanceType.CORRECTIVE,
            interval=200.0, duration=4.0, cost=2000.0, required_skills=[], required_tools=[],
            effectiveness=0.5
        )
        
        planner.add_maintenance_task(task1)
        planner.add_maintenance_task(task2)
        
        # Component reliabilities
        reliabilities = {
            "comp_1": ReliabilityMetrics(
                component_id="comp_1", mean_time_between_failures=500.0,
                mean_time_to_repair=4.0, availability=0.992, reliability_at_time={},
                failure_rate=2e-3, repair_rate=0.25
            )
        }
        
        schedule = planner.optimize_maintenance_schedule(
            reliabilities, mission_duration=8760.0, cost_constraint=500000.0)
        
        assert len(schedule) > 0
        assert all(isinstance(task, MaintenanceTask) for task in schedule)
    
    def test_calculate_maintenance_effectiveness(self):
        """Test maintenance effectiveness calculation."""
        planner = MaintenancePlanner()
        
        tasks = [
            MaintenanceTask(
                task_id="task_1", component_id="comp_1", maintenance_type=MaintenanceType.PREVENTIVE,
                interval=100.0, duration=2.0, cost=1000.0, required_skills=[], required_tools=[],
                effectiveness=0.3
            ),
            MaintenanceTask(
                task_id="task_2", component_id="comp_1", maintenance_type=MaintenanceType.PREDICTIVE,
                interval=50.0, duration=1.0, cost=500.0, required_skills=[], required_tools=[],
                effectiveness=0.2
            )
        ]
        
        original_reliability = Mock()
        
        effectiveness = planner.calculate_maintenance_effectiveness(tasks, original_reliability)
        
        assert "comp_1" in effectiveness
        # Combined effectiveness should be 1 - (1-0.3)*(1-0.2) = 0.44
        expected_effectiveness = 1.0 - (1.0 - 0.3) * (1.0 - 0.2)
        assert abs(effectiveness["comp_1"] - expected_effectiveness) < 1e-6
    
    def test_generate_maintenance_calendar(self):
        """Test maintenance calendar generation."""
        planner = MaintenancePlanner()
        
        tasks = [
            MaintenanceTask(
                task_id="monthly_check", component_id="engine_1", maintenance_type=MaintenanceType.PREVENTIVE,
                interval=720.0, duration=2.0, cost=1000.0, required_skills=[], required_tools=[],
                description="Monthly engine check"
            )
        ]
        
        calendar = planner.generate_maintenance_calendar(tasks)
        
        assert "engine_1" in calendar
        assert len(calendar["engine_1"]) > 0
        
        first_entry = calendar["engine_1"][0]
        assert "date" in first_entry
        assert "task_id" in first_entry
        assert first_entry["task_id"] == "monthly_check"


class TestRiskAssessmentEngine:
    """Test risk assessment engine."""
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        engine = RiskAssessmentEngine()
        assert len(engine.risk_matrix) > 0
    
    def test_assess_failure_risk(self):
        """Test single failure risk assessment."""
        engine = RiskAssessmentEngine()
        
        failure_event = FailureEvent(
            event_id="critical_failure",
            component_id="engine_1",
            failure_mode=FailureMode.CATASTROPHIC,
            failure_cause=FailureCause.MECHANICAL,
            failure_rate=1e-4,
            detection_probability=0.9,
            repair_time=24.0,
            cost_impact=1000000.0,
            safety_impact="Loss of aircraft",
            mission_impact="Mission abort"
        )
        
        risk = engine.assess_failure_risk(failure_event, mission_time=100.0)
        
        assert risk["event_id"] == "critical_failure"
        assert "probability" in risk
        assert "probability_category" in risk
        assert "severity" in risk
        assert "risk_level" in risk
        assert risk["severity"] == "catastrophic"
    
    def test_assess_system_risk(self):
        """Test system risk assessment."""
        engine = RiskAssessmentEngine()
        
        failure_events = [
            FailureEvent(
                event_id="failure_1", component_id="comp_1", failure_mode=FailureMode.MAJOR,
                failure_cause=FailureCause.ELECTRICAL, failure_rate=1e-4, detection_probability=0.8,
                repair_time=4.0, cost_impact=50000.0, safety_impact="System degradation", mission_impact="Reduced capability"
            ),
            FailureEvent(
                event_id="failure_2", component_id="comp_2", failure_mode=FailureMode.CATASTROPHIC,
                failure_cause=FailureCause.MECHANICAL, failure_rate=1e-5, detection_probability=0.9,
                repair_time=24.0, cost_impact=500000.0, safety_impact="Loss of aircraft", mission_impact="Mission abort"
            )
        ]
        
        system_risk = engine.assess_system_risk(failure_events, mission_time=100.0)
        
        assert "overall_risk_level" in system_risk
        assert "risk_distribution" in system_risk
        assert "total_cost_impact" in system_risk
        assert "critical_events" in system_risk
        assert "individual_risks" in system_risk
        assert "recommendations" in system_risk
        
        assert len(system_risk["individual_risks"]) == 2
        assert system_risk["total_cost_impact"] == 550000.0


class TestReliabilityAssessmentEngine:
    """Test main reliability assessment engine."""
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        engine = ReliabilityAssessmentEngine()
        assert engine.fault_tree_analyzer is not None
        assert engine.reliability_calculator is not None
        assert engine.maintenance_planner is not None
        assert engine.risk_assessor is not None
    
    def test_engine_initialize(self):
        """Test engine initialization method."""
        engine = ReliabilityAssessmentEngine()
        
        success = engine.initialize()
        assert success
        assert engine.initialized
    
    def test_validate_input_aircraft_config(self):
        """Test input validation with aircraft configuration."""
        engine = ReliabilityAssessmentEngine()
        
        config = AircraftConfiguration()
        assert engine.validate_input(config)
    
    def test_validate_input_failure_events(self):
        """Test input validation with failure events."""
        engine = ReliabilityAssessmentEngine()
        
        data = {"failure_events": []}
        assert engine.validate_input(data)
    
    def test_validate_input_invalid(self):
        """Test input validation with invalid data."""
        engine = ReliabilityAssessmentEngine()
        
        assert not engine.validate_input("invalid")
        assert not engine.validate_input(123)
    
    def test_assess_aircraft_reliability(self):
        """Test aircraft reliability assessment."""
        engine = ReliabilityAssessmentEngine()
        engine.initialize()
        
        config = AircraftConfiguration()
        
        assessment = engine.assess_aircraft_reliability(config, mission_time=1000.0)
        
        assert assessment.aircraft_config_id is not None
        assert assessment.system_reliability is not None
        assert len(assessment.component_reliabilities) > 0
        assert len(assessment.fault_trees) > 0
        assert len(assessment.critical_components) >= 0
        assert len(assessment.maintenance_plan) > 0
        assert assessment.risk_assessment is not None
        assert len(assessment.recommendations) > 0
    
    def test_analyze_failure_modes(self):
        """Test failure mode analysis."""
        engine = ReliabilityAssessmentEngine()
        engine.initialize()
        
        failure_events = [
            FailureEvent(
                event_id="failure_1", component_id="comp_1", failure_mode=FailureMode.MAJOR,
                failure_cause=FailureCause.ELECTRICAL, failure_rate=1e-4, detection_probability=0.8,
                repair_time=4.0, cost_impact=50000.0, safety_impact="System degradation", mission_impact="Reduced capability"
            ),
            FailureEvent(
                event_id="failure_2", component_id="comp_2", failure_mode=FailureMode.MINOR,
                failure_cause=FailureCause.SOFTWARE, failure_rate=2e-4, detection_probability=0.7,
                repair_time=2.0, cost_impact=10000.0, safety_impact="Minor impact", mission_impact="Minimal impact"
            )
        ]
        
        analysis = engine.analyze_failure_modes(failure_events)
        
        assert "failure_mode_analysis" in analysis
        assert "criticality_analysis" in analysis
        assert "risk_assessment" in analysis
        assert "recommendations" in analysis
        
        # Check failure mode analysis
        fma = analysis["failure_mode_analysis"]
        assert "major" in fma
        assert "minor" in fma
        assert fma["major"]["event_count"] == 1
        assert fma["minor"]["event_count"] == 1
        
        # Check criticality analysis
        ca = analysis["criticality_analysis"]
        assert "most_critical" in ca
        assert "top_10_critical" in ca
        assert "criticality_scores" in ca


if __name__ == '__main__':
    pytest.main([__file__])