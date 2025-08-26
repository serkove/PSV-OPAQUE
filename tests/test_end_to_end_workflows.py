"""
Tests for end-to-end workflow validation system.

This module tests the complete aircraft design workflows from concept to manufacturing,
mission-specific optimization scenarios, and user acceptance testing procedures.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock

from fighter_jet_sdk.core.workflow_validator import (
    EndToEndWorkflowValidator, WorkflowStep, WorkflowResult, 
    WorkflowValidationReport
)
from fighter_jet_sdk.core.errors import WorkflowError
from fighter_jet_sdk.common.interfaces import BaseEngine


class MockEngine(BaseEngine):
    """Mock engine for testing workflow validation."""
    
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.operations = {}
        self.initialized = True
    
    def initialize(self) -> bool:
        """Initialize the mock engine."""
        return True
    
    def process(self, operation: str, **kwargs) -> dict:
        """Process an operation."""
        if hasattr(self, operation):
            return getattr(self, operation)(**kwargs)
        return {}
    
    def validate_input(self, operation: str, **kwargs) -> bool:
        """Validate input for an operation."""
        return True
    
    def add_operation(self, operation_name: str, mock_result: dict):
        """Add a mock operation with expected result."""
        def mock_operation(**kwargs):
            return mock_result
        
        setattr(self, operation_name, mock_operation)
        self.operations[operation_name] = mock_result


@pytest.fixture
def mock_engines():
    """Create mock engines for testing."""
    engines = {}
    
    # Design engine
    design_engine = MockEngine("design")
    design_engine.add_operation("create_configuration", {
        "configuration_id": "test_config_001",
        "base_platform": {"type": "fighter", "modules": []},
        "module_count": 3
    })
    design_engine.add_operation("analyze_mission", {
        "mission_profile": {"type": "deep_strike", "duration": 4.5},
        "optimization_targets": ["range", "stealth", "payload"],
        "feasibility_score": 0.85
    })
    design_engine.add_operation("optimize_configuration", {
        "optimized_config": {"id": "optimized_001", "score": 0.92},
        "pareto_frontier": [{"point": i, "values": [i*0.1, (10-i)*0.1]} for i in range(11)],
        "optimization_convergence": True,
        "pareto_points": 11
    })
    design_engine.add_operation("validate_module_compatibility", {
        "compatibility_results": {"total_combinations": 100, "valid": 96},
        "invalid_combinations": ["sensor_a + payload_x", "cockpit_b + engine_y"],
        "compatibility_rate": 0.96,
        "interface_validation": True
    })
    design_engine.add_operation("generate_variants", {
        "variant_configs": [{"id": f"variant_{i}", "config": {}} for i in range(20)],
        "performance_matrix": [[0.8, 0.9, 0.7] for _ in range(20)],
        "variant_diversity": 0.85,
        "all_valid": True
    })
    engines["design"] = design_engine
    
    # Materials engine
    materials_engine = MockEngine("materials")
    materials_engine.add_operation("optimize_materials", {
        "material_assignments": {"fuselage": "carbon_fiber", "coating": "metamaterial_ram"},
        "stealth_properties": {"rcs_reduction": 0.75, "frequency_range": [1, 18]},
        "rcs_reduction": 0.75,
        "weight_penalty": 0.12
    })
    materials_engine.add_operation("define_stealth_requirements", {
        "stealth_spec": {"rcs_target": -42, "frequency_bands": ["x", "ku"]},
        "material_constraints": {"temperature_max": 1200, "thickness_max": 5},
        "rcs_target": -42,
        "frequency_coverage": 0.95
    })
    materials_engine.add_operation("integrate_stealth_materials", {
        "material_config": {"coating_type": "metamaterial_ram", "thickness": 3.2},
        "manufacturing_constraints": {"cure_temp": 180, "pressure": 6},
        "coating_effectiveness": 0.92,
        "durability": 1200
    })
    engines["materials"] = materials_engine
    
    # Propulsion engine
    propulsion_engine = MockEngine("propulsion")
    propulsion_engine.add_operation("design_propulsion", {
        "engine_config": {"type": "turbofan", "thrust": 155000, "bypass_ratio": 0.3},
        "thermal_management": {"cooling_capacity": 50000, "heat_exchangers": 3},
        "thrust_to_weight": 1.25,
        "fuel_efficiency": 0.82
    })
    engines["propulsion"] = propulsion_engine
    
    # Sensors engine
    sensors_engine = MockEngine("sensors")
    sensors_engine.add_operation("integrate_sensors", {
        "sensor_config": {"radar": "aesa", "eo_ir": "advanced", "ew": "integrated"},
        "power_requirements": {"total": 45000, "peak": 60000},
        "detection_range": 220,
        "power_consumption": 45000
    })
    engines["sensors"] = sensors_engine
    
    # Aerodynamics engine
    aero_engine = MockEngine("aerodynamics")
    aero_engine.add_operation("validate_performance", {
        "aero_performance": {"max_speed": 2.6, "cruise_speed": 1.8, "range": 1800},
        "stability_margins": {"pitch": 0.15, "yaw": 0.12, "roll": 0.18},
        "max_speed": 2.6,
        "stability_margin": 0.15
    })
    aero_engine.add_operation("validate_mission_performance", {
        "mission_performance": {"success_rate": 0.96, "fuel_usage": 0.85},
        "success_probability": 0.96,
        "mission_success_rate": 0.96,
        "fuel_margin": 0.15
    })
    aero_engine.add_operation("optimize_stealth_shape", {
        "optimized_shape": {"rcs_reduction": 0.82, "aero_penalty": 0.15},
        "rcs_analysis": {"frontal": -38, "side": -35, "rear": -32},
        "rcs_reduction": 0.82,
        "aero_penalty": 0.15
    })
    aero_engine.add_operation("compare_variants", {
        "performance_comparison": {"best": "variant_5", "worst": "variant_12"},
        "ranking": [f"variant_{i}" for i in [5, 3, 8, 1, 7]],
        "performance_spread": 0.35,
        "ranking_consistency": True
    })
    engines["aerodynamics"] = aero_engine
    
    # Manufacturing engine
    manufacturing_engine = MockEngine("manufacturing")
    manufacturing_engine.add_operation("plan_manufacturing", {
        "manufacturing_plan": {"stages": 8, "critical_path": 22, "resources": 45},
        "cost_estimate": {"unit_cost": 145000000, "tooling": 50000000},
        "unit_cost": 145000000,
        "production_time": 22
    })
    engines["manufacturing"] = manufacturing_engine
    
    return engines


@pytest.fixture
def workflow_validator(mock_engines):
    """Create workflow validator with mock engines."""
    return EndToEndWorkflowValidator(mock_engines)


class TestEndToEndWorkflowValidator:
    """Test the end-to-end workflow validation system."""
    
    def test_initialization(self, workflow_validator):
        """Test workflow validator initialization."""
        assert len(workflow_validator.workflows) == 4
        assert "concept_to_manufacturing" in workflow_validator.workflows
        assert "mission_optimization" in workflow_validator.workflows
        assert "stealth_fighter_design" in workflow_validator.workflows
        assert "modular_configuration" in workflow_validator.workflows
        
        assert len(workflow_validator.benchmark_data) > 0
        assert "thrust_to_weight" in workflow_validator.benchmark_data
    
    def test_concept_to_manufacturing_workflow(self, workflow_validator):
        """Test complete concept-to-manufacturing workflow."""
        report = workflow_validator.execute_workflow("concept_to_manufacturing")
        
        assert isinstance(report, WorkflowValidationReport)
        assert report.workflow_name == "concept_to_manufacturing"
        assert report.total_steps == 6
        assert report.overall_success
        assert report.successful_steps == 6
        assert report.failed_steps == 0
        
        # Check that all expected steps were executed
        step_names = [result.step_name for result in report.step_results]
        expected_steps = [
            "initial_configuration", "materials_selection", "propulsion_integration",
            "sensor_integration", "aerodynamic_validation", "manufacturing_planning"
        ]
        assert all(step in step_names for step in expected_steps)
        
        # Check performance benchmarks
        assert len(report.performance_benchmarks) > 0
        assert "propulsion_integration_thrust_to_weight" in report.performance_benchmarks
        
        # Check requirements validation
        assert report.requirements_validation["complete_workflow"]
        assert report.requirements_validation["manufacturing_ready"]
    
    def test_mission_optimization_workflow(self, workflow_validator):
        """Test mission-specific optimization workflow."""
        report = workflow_validator.execute_workflow("mission_optimization")
        
        assert report.workflow_name == "mission_optimization"
        assert report.total_steps == 3
        assert report.overall_success
        
        # Check mission-specific validations
        assert report.requirements_validation["mission_optimized"]
        assert report.requirements_validation["performance_validated"]
        
        # Verify optimization results
        optimization_step = next(
            result for result in report.step_results 
            if result.step_name == "configuration_optimization"
        )
        assert optimization_step.success
        assert optimization_step.validation_results["optimization_convergence"]
    
    def test_stealth_fighter_workflow(self, workflow_validator):
        """Test stealth-focused fighter design workflow."""
        report = workflow_validator.execute_workflow("stealth_fighter_design")
        
        assert report.workflow_name == "stealth_fighter_design"
        assert report.total_steps == 3
        assert report.overall_success
        
        # Check stealth-specific validations
        assert report.requirements_validation["stealth_optimized"]
        assert report.requirements_validation["materials_integrated"]
        
        # Verify stealth performance
        shape_step = next(
            result for result in report.step_results 
            if result.step_name == "shape_optimization"
        )
        assert shape_step.success
        assert shape_step.validation_results["rcs_reduction_min"]
    
    def test_modular_configuration_workflow(self, workflow_validator):
        """Test modular configuration validation workflow."""
        report = workflow_validator.execute_workflow("modular_configuration")
        
        assert report.workflow_name == "modular_configuration"
        assert report.total_steps == 3
        assert report.overall_success
        
        # Check modular-specific validations
        compatibility_step = next(
            result for result in report.step_results 
            if result.step_name == "module_compatibility"
        )
        assert compatibility_step.success
        assert compatibility_step.validation_results["compatibility_rate_min"]
        assert compatibility_step.validation_results["interface_validation"]
    
    def test_workflow_with_configuration_overrides(self, workflow_validator):
        """Test workflow execution with configuration overrides."""
        overrides = {
            "initial_configuration": {"platform_type": "stealth_fighter"},
            "materials_selection": {"stealth_priority": 0.95}
        }
        
        report = workflow_validator.execute_workflow(
            "concept_to_manufacturing", 
            overrides
        )
        
        assert report.overall_success
        # Overrides should not break the workflow
        assert report.successful_steps == report.total_steps
    
    def test_workflow_step_failure_handling(self, workflow_validator):
        """Test handling of workflow step failures."""
        # Create a mock engine that will fail
        failing_engine = MockEngine("failing")
        failing_engine.add_operation("fail_operation", None)
        
        def failing_operation(**kwargs):
            raise Exception("Simulated failure")
        
        failing_engine.fail_operation = failing_operation
        workflow_validator.engines["failing"] = failing_engine
        
        # Create a workflow with a failing step
        failing_workflow = [
            WorkflowStep(
                name="failing_step",
                description="This step will fail",
                engine="failing",
                operation="fail_operation",
                inputs={},
                expected_outputs=["result"],
                validation_criteria={}
            )
        ]
        workflow_validator.workflows["failing_workflow"] = failing_workflow
        
        report = workflow_validator.execute_workflow("failing_workflow")
        
        assert not report.overall_success
        assert report.failed_steps == 1
        assert len(report.step_results[0].errors) > 0
    
    def test_user_acceptance_scenarios_generation(self, workflow_validator):
        """Test generation of user acceptance testing scenarios."""
        scenarios = workflow_validator.generate_user_acceptance_scenarios()
        
        assert len(scenarios) == 3
        assert "air_superiority_fighter" in scenarios
        assert "multi_role_strike" in scenarios
        assert "modular_platform" in scenarios
        
        # Check scenario structure
        for scenario_name, scenario in scenarios.items():
            assert "description" in scenario
            assert "workflow" in scenario
            assert "acceptance_criteria" in scenario
            assert "test_data" in scenario
    
    def test_user_acceptance_tests_execution(self, workflow_validator):
        """Test execution of user acceptance tests."""
        results = workflow_validator.run_user_acceptance_tests()
        
        assert len(results) == 3
        
        for scenario_name, report in results.items():
            assert isinstance(report, WorkflowValidationReport)
            # All scenarios should pass with mock engines
            assert report.overall_success
    
    def test_performance_benchmarking(self, workflow_validator):
        """Test performance benchmarking against reference aircraft."""
        report = workflow_validator.execute_workflow("concept_to_manufacturing")
        
        benchmarks = report.performance_benchmarks
        
        # Check for derived benchmarks
        if "propulsion_integration_thrust_to_weight" in benchmarks:
            assert "ttw_vs_f22" in benchmarks
            # Should be close to F-22 performance
            assert 0.8 <= benchmarks["ttw_vs_f22"] <= 1.2
        
        if "aerodynamic_validation_max_speed" in benchmarks:
            assert "speed_vs_f22" in benchmarks
    
    def test_validation_report_export(self, workflow_validator):
        """Test export of validation reports to JSON."""
        report = workflow_validator.execute_workflow("mission_optimization")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = Path(f.name)
        
        try:
            workflow_validator.export_validation_report(report, output_path)
            
            # Verify the file was created and contains valid JSON
            assert output_path.exists()
            
            with open(output_path, 'r') as f:
                exported_data = json.load(f)
            
            assert exported_data["workflow_name"] == "mission_optimization"
            assert "step_results" in exported_data
            assert "performance_benchmarks" in exported_data
            
        finally:
            if output_path.exists():
                output_path.unlink()
    
    def test_workflow_summary_generation(self, workflow_validator):
        """Test generation of workflow summaries."""
        summary = workflow_validator.get_workflow_summary()
        
        assert len(summary) == 4
        
        for workflow_name, info in summary.items():
            assert "description" in info
            assert "total_steps" in info
            assert "engines_involved" in info
            assert "estimated_duration" in info
            assert "key_outputs" in info
            
            assert info["total_steps"] > 0
            assert len(info["engines_involved"]) > 0
    
    def test_input_reference_resolution(self, workflow_validator):
        """Test resolution of input references between workflow steps."""
        # This is tested implicitly in the workflow execution tests
        # but we can add specific tests for edge cases
        
        context = {
            "step1": {"output_a": "value_a", "output_b": 42},
            "step2": {"output_c": [1, 2, 3]}
        }
        
        inputs = {
            "direct_input": "direct_value",
            "reference_input": "step1_ref",
            "another_ref": "step2_ref"
        }
        
        resolved = workflow_validator._resolve_step_inputs(inputs, context)
        
        assert resolved["direct_input"] == "direct_value"
        assert resolved["reference_input"] == context["step1"]
        assert resolved["another_ref"] == context["step2"]
    
    def test_step_output_validation(self, workflow_validator):
        """Test validation of step outputs against criteria."""
        step = WorkflowStep(
            name="test_step",
            description="Test step",
            engine="test",
            operation="test_op",
            inputs={},
            expected_outputs=["result_a", "result_b"],
            validation_criteria={
                "performance_metric": {"min": 0.8, "max": 1.2},
                "boolean_check": True,
                "count_metric": {"min": 5}
            }
        )
        
        outputs = {
            "result_a": "present",
            "result_b": "also_present",
            "performance_metric": 0.95,
            "boolean_check": True,
            "count_metric": 7,
            "extra_output": "ignored"
        }
        
        validation_results = workflow_validator._validate_step_outputs(step, outputs)
        
        assert validation_results["has_result_a"]
        assert validation_results["has_result_b"]
        assert validation_results["performance_metric_min"]
        assert validation_results["performance_metric_max"]
        assert validation_results["boolean_check"]
        assert validation_results["count_metric_min"]
    
    def test_unknown_workflow_error(self, workflow_validator):
        """Test error handling for unknown workflows."""
        with pytest.raises(WorkflowError, match="Unknown workflow"):
            workflow_validator.execute_workflow("nonexistent_workflow")
    
    def test_missing_engine_error(self, workflow_validator):
        """Test error handling for missing engines."""
        # Remove an engine
        del workflow_validator.engines["design"]
        
        report = workflow_validator.execute_workflow("concept_to_manufacturing")
        
        # Should fail on the first step that requires the design engine
        assert not report.overall_success
        assert report.failed_steps > 0
        assert "Engine not available" in report.step_results[0].errors[0]


if __name__ == "__main__":
    pytest.main([__file__])