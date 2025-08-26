"""
Integration tests for end-to-end workflow validation system.

This module tests the complete integration of all engines through
the workflow validation system, ensuring that the entire SDK works
together as intended.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from fighter_jet_sdk.core.workflow_validator import EndToEndWorkflowValidator
from fighter_jet_sdk.cli.main import get_engine_registry
from fighter_jet_sdk.core.errors import WorkflowError


class TestWorkflowIntegration:
    """Test complete workflow integration across all engines."""
    
    @pytest.fixture
    def real_engines(self):
        """Get real engine instances for integration testing."""
        return get_engine_registry()
    
    @pytest.fixture
    def workflow_validator(self, real_engines):
        """Create workflow validator with real engines."""
        return EndToEndWorkflowValidator(real_engines)
    
    def test_engine_registry_initialization(self, real_engines):
        """Test that all engines can be initialized from registry."""
        expected_engines = ['design', 'materials', 'propulsion', 'sensors', 'aerodynamics', 'manufacturing']
        
        for engine_name in expected_engines:
            assert engine_name in real_engines, f"Missing engine: {engine_name}"
            assert real_engines[engine_name] is not None
    
    def test_workflow_validator_initialization(self, workflow_validator):
        """Test workflow validator initialization with real engines."""
        assert len(workflow_validator.workflows) == 4
        assert len(workflow_validator.engines) >= 6  # At least 6 engines
        assert len(workflow_validator.benchmark_data) > 0
    
    def test_concept_to_manufacturing_workflow_structure(self, workflow_validator):
        """Test the structure of the concept-to-manufacturing workflow."""
        workflow = workflow_validator.workflows["concept_to_manufacturing"]
        
        assert len(workflow) == 6  # Should have 6 main steps
        
        # Check that all required engines are covered
        engines_used = set(step.engine for step in workflow)
        expected_engines = {'design', 'materials', 'propulsion', 'sensors', 'aerodynamics', 'manufacturing'}
        assert engines_used == expected_engines
        
        # Check step dependencies (each step should reference previous steps appropriately)
        step_names = [step.name for step in workflow]
        assert "initial_configuration" in step_names
        assert "manufacturing_planning" in step_names  # Final step
    
    def test_mission_optimization_workflow_structure(self, workflow_validator):
        """Test the structure of the mission optimization workflow."""
        workflow = workflow_validator.workflows["mission_optimization"]
        
        assert len(workflow) == 3  # Should have 3 main steps
        
        # Check logical flow
        step_names = [step.name for step in workflow]
        assert step_names[0] == "mission_analysis"
        assert step_names[1] == "configuration_optimization"
        assert step_names[2] == "performance_validation"
    
    def test_stealth_fighter_workflow_structure(self, workflow_validator):
        """Test the structure of the stealth fighter design workflow."""
        workflow = workflow_validator.workflows["stealth_fighter_design"]
        
        assert len(workflow) == 3  # Should have 3 main steps
        
        # Check stealth-specific steps
        step_names = [step.name for step in workflow]
        assert "stealth_requirements" in step_names
        assert "shape_optimization" in step_names
        assert "materials_integration" in step_names
    
    def test_modular_configuration_workflow_structure(self, workflow_validator):
        """Test the structure of the modular configuration workflow."""
        workflow = workflow_validator.workflows["modular_configuration"]
        
        assert len(workflow) == 3  # Should have 3 main steps
        
        # Check modular-specific steps
        step_names = [step.name for step in workflow]
        assert "module_compatibility" in step_names
        assert "configuration_variants" in step_names
        assert "performance_comparison" in step_names
    
    def test_workflow_step_validation_criteria(self, workflow_validator):
        """Test that all workflow steps have proper validation criteria."""
        for workflow_name, workflow_steps in workflow_validator.workflows.items():
            for step in workflow_steps:
                # Each step should have validation criteria
                assert len(step.validation_criteria) > 0, f"Step {step.name} has no validation criteria"
                
                # Each step should have expected outputs
                assert len(step.expected_outputs) > 0, f"Step {step.name} has no expected outputs"
                
                # Timeout should be reasonable
                assert 0 < step.timeout_seconds <= 600, f"Step {step.name} has unreasonable timeout"
    
    def test_user_acceptance_scenarios_completeness(self, workflow_validator):
        """Test that user acceptance scenarios cover all major use cases."""
        scenarios = workflow_validator.generate_user_acceptance_scenarios()
        
        # Should have scenarios for different aircraft types
        scenario_names = list(scenarios.keys())
        assert "air_superiority_fighter" in scenario_names
        assert "multi_role_strike" in scenario_names
        assert "modular_platform" in scenario_names
        
        # Each scenario should have complete structure
        for scenario_name, scenario in scenarios.items():
            assert "description" in scenario
            assert "workflow" in scenario
            assert "acceptance_criteria" in scenario
            assert "test_data" in scenario
            
            # Workflow should exist
            assert scenario["workflow"] in workflow_validator.workflows
            
            # Acceptance criteria should be measurable
            for criterion, constraint in scenario["acceptance_criteria"].items():
                assert isinstance(constraint, dict)
                assert "min" in constraint or "max" in constraint
    
    def test_benchmark_data_completeness(self, workflow_validator):
        """Test that benchmark data covers all necessary reference aircraft."""
        benchmark_data = workflow_validator.benchmark_data
        
        # Should have data for major reference aircraft
        assert "f22" in benchmark_data["thrust_to_weight"]
        assert "f35" in benchmark_data["thrust_to_weight"]
        assert "su57" in benchmark_data["thrust_to_weight"]
        
        # Should have all key performance metrics
        required_metrics = ["thrust_to_weight", "max_speed_mach", "combat_radius_km", "rcs_m2", "unit_cost_million"]
        for metric in required_metrics:
            assert metric in benchmark_data
            assert len(benchmark_data[metric]) >= 3  # At least 3 reference aircraft
    
    def test_workflow_summary_generation(self, workflow_validator):
        """Test workflow summary generation for documentation."""
        summary = workflow_validator.get_workflow_summary()
        
        assert len(summary) == 4  # Should match number of workflows
        
        for workflow_name, info in summary.items():
            # Each summary should have complete information
            required_fields = ["description", "total_steps", "engines_involved", "estimated_duration", "key_outputs"]
            for field in required_fields:
                assert field in info, f"Missing field {field} in {workflow_name} summary"
            
            # Sanity checks
            assert info["total_steps"] > 0
            assert len(info["engines_involved"]) > 0
            assert info["estimated_duration"] > 0
    
    def test_input_reference_resolution_edge_cases(self, workflow_validator):
        """Test edge cases in input reference resolution."""
        context = {
            "step1": {"output_a": "value_a"},
            "step2": {"nested": {"value": 42}}
        }
        
        # Test missing reference
        inputs_missing = {"ref": "nonexistent_ref"}
        with pytest.raises(WorkflowError, match="Reference not found"):
            workflow_validator._resolve_step_inputs(inputs_missing, context)
        
        # Test valid references
        inputs_valid = {
            "direct": "direct_value",
            "ref1": "step1_ref",
            "ref2": "step2_ref"
        }
        resolved = workflow_validator._resolve_step_inputs(inputs_valid, context)
        
        assert resolved["direct"] == "direct_value"
        assert resolved["ref1"] == context["step1"]
        assert resolved["ref2"] == context["step2"]
    
    def test_step_output_validation_edge_cases(self, workflow_validator):
        """Test edge cases in step output validation."""
        from fighter_jet_sdk.core.workflow_validator import WorkflowStep
        
        step = WorkflowStep(
            name="test_step",
            description="Test",
            engine="test",
            operation="test_op",
            inputs={},
            expected_outputs=["required_output"],
            validation_criteria={
                "numeric_value": {"min": 0.5, "max": 1.5},
                "boolean_value": True,
                "missing_value": {"min": 10}
            }
        )
        
        # Test with missing required output
        outputs_missing = {"other_output": "present"}
        validation = workflow_validator._validate_step_outputs(step, outputs_missing)
        assert not validation["has_required_output"]
        
        # Test with invalid numeric values
        outputs_invalid = {
            "required_output": "present",
            "numeric_value": 2.0,  # Above max
            "boolean_value": False  # Wrong value
        }
        validation = workflow_validator._validate_step_outputs(step, outputs_invalid)
        assert validation["has_required_output"]
        assert not validation["numeric_value_max"]
        assert not validation["boolean_value"]
        assert not validation["missing_value"]  # Missing entirely
    
    def test_performance_benchmark_calculation(self, workflow_validator):
        """Test performance benchmark calculation logic."""
        # Create mock workflow context
        context = {
            "propulsion_integration": {
                "thrust_to_weight": 1.3,
                "fuel_efficiency": 0.85
            },
            "aerodynamic_validation": {
                "max_speed": 2.4,
                "stability_margin": 0.12
            }
        }
        
        benchmarks = workflow_validator._generate_performance_benchmarks(context)
        
        # Should extract all numeric values
        assert "propulsion_integration_thrust_to_weight" in benchmarks
        assert "propulsion_integration_fuel_efficiency" in benchmarks
        assert "aerodynamic_validation_max_speed" in benchmarks
        assert "aerodynamic_validation_stability_margin" in benchmarks
        
        # Should calculate derived benchmarks
        assert "ttw_vs_f22" in benchmarks
        assert "speed_vs_f22" in benchmarks
        
        # Values should be reasonable
        assert 0.5 <= benchmarks["ttw_vs_f22"] <= 2.0
        assert 0.5 <= benchmarks["speed_vs_f22"] <= 2.0
    
    def test_requirements_validation_logic(self, workflow_validator):
        """Test requirements validation logic."""
        # Test concept-to-manufacturing workflow validation
        context_complete = {
            "initial_configuration": {},
            "materials_selection": {},
            "propulsion_integration": {},
            "sensor_integration": {},
            "aerodynamic_validation": {},
            "manufacturing_planning": {}
        }
        
        validation = workflow_validator._validate_requirements("concept_to_manufacturing", context_complete)
        
        assert validation["req_9_4_performance_validation"]
        assert validation["req_10_4_progress_reports"]
        assert validation["complete_workflow"]
        assert validation["manufacturing_ready"]
        
        # Test incomplete workflow
        context_incomplete = {
            "initial_configuration": {},
            "materials_selection": {}
        }
        
        validation_incomplete = workflow_validator._validate_requirements("concept_to_manufacturing", context_incomplete)
        
        assert not validation_incomplete["complete_workflow"]
        assert not validation_incomplete["manufacturing_ready"]
    
    def test_workflow_error_handling(self, workflow_validator):
        """Test error handling in workflow execution."""
        # Test unknown workflow
        with pytest.raises(WorkflowError, match="Unknown workflow"):
            workflow_validator.execute_workflow("nonexistent_workflow")
        
        # Test workflow with missing engine (temporarily remove an engine)
        original_engines = workflow_validator.engines.copy()
        del workflow_validator.engines["design"]
        
        try:
            report = workflow_validator.execute_workflow("concept_to_manufacturing")
            assert not report.overall_success
            assert report.failed_steps > 0
        finally:
            # Restore engines
            workflow_validator.engines = original_engines
    
    def test_validation_report_export_import(self, workflow_validator):
        """Test export and import of validation reports."""
        # Create a simple mock report
        from fighter_jet_sdk.core.workflow_validator import WorkflowValidationReport, WorkflowResult
        
        report = WorkflowValidationReport(
            workflow_name="test_workflow",
            total_steps=2,
            successful_steps=2,
            failed_steps=0,
            total_execution_time=10.5,
            step_results=[
                WorkflowResult(
                    step_name="step1",
                    success=True,
                    execution_time=5.0,
                    outputs={"result": "success"},
                    validation_results={"test": True},
                    errors=[],
                    warnings=[]
                )
            ],
            performance_benchmarks={"metric1": 1.5},
            requirements_validation={"req1": True},
            overall_success=True
        )
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = Path(f.name)
        
        try:
            workflow_validator.export_validation_report(report, output_path)
            
            # Verify file exists and contains valid JSON
            assert output_path.exists()
            
            with open(output_path, 'r') as f:
                exported_data = json.load(f)
            
            # Verify key fields
            assert exported_data["workflow_name"] == "test_workflow"
            assert exported_data["total_steps"] == 2
            assert exported_data["overall_success"] is True
            assert len(exported_data["step_results"]) == 1
            
        finally:
            if output_path.exists():
                output_path.unlink()
    
    @pytest.mark.slow
    def test_full_workflow_execution_smoke_test(self, workflow_validator):
        """Smoke test for full workflow execution (may be slow)."""
        # This test verifies that workflows can be executed without crashing
        # It doesn't validate correctness, just that the system doesn't break
        
        try:
            # Try to execute a simple workflow
            # Note: This might fail due to missing data or uninitialized engines
            # but it should not crash with unhandled exceptions
            report = workflow_validator.execute_workflow("mission_optimization")
            
            # Basic structure validation
            assert hasattr(report, 'workflow_name')
            assert hasattr(report, 'overall_success')
            assert hasattr(report, 'step_results')
            
        except Exception as e:
            # If execution fails, it should be a controlled failure
            assert isinstance(e, (WorkflowError, ValueError, TypeError))
            # Should not be an unhandled exception like AttributeError, KeyError, etc.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])