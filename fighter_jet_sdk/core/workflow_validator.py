"""
End-to-end workflow validation system for the Fighter Jet SDK.

This module provides comprehensive validation of complete aircraft design workflows
from concept to manufacturing, including mission-specific optimization scenarios
and performance benchmarking against design requirements.
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

from ..common.data_models import AircraftConfiguration, MissionRequirements, PerformanceEnvelope
from ..common.interfaces import BaseEngine
from ..core.errors import ValidationError, WorkflowError
from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WorkflowStep:
    """Represents a single step in an end-to-end workflow."""
    name: str
    description: str
    engine: str
    operation: str
    inputs: Dict[str, Any]
    expected_outputs: List[str]
    validation_criteria: Dict[str, Any]
    timeout_seconds: float = 300.0


@dataclass
class WorkflowResult:
    """Results from executing a workflow step."""
    step_name: str
    success: bool
    execution_time: float
    outputs: Dict[str, Any]
    validation_results: Dict[str, bool]
    errors: List[str]
    warnings: List[str]


@dataclass
class WorkflowValidationReport:
    """Complete validation report for an end-to-end workflow."""
    workflow_name: str
    total_steps: int
    successful_steps: int
    failed_steps: int
    total_execution_time: float
    step_results: List[WorkflowResult]
    performance_benchmarks: Dict[str, float]
    requirements_validation: Dict[str, bool]
    overall_success: bool


class EndToEndWorkflowValidator:
    """
    Validates complete aircraft design workflows from concept to manufacturing.
    
    This class orchestrates the execution of complex multi-engine workflows
    and validates the results against design requirements and performance criteria.
    """
    
    def __init__(self, engines: Dict[str, BaseEngine]):
        """
        Initialize the workflow validator.
        
        Args:
            engines: Dictionary mapping engine names to engine instances
        """
        self.engines = engines
        self.workflows = {}
        self.benchmark_data = {}
        self._load_predefined_workflows()
        self._load_benchmark_data()
    
    def _load_predefined_workflows(self):
        """Load predefined workflow definitions."""
        self.workflows = {
            "concept_to_manufacturing": self._create_concept_to_manufacturing_workflow(),
            "mission_optimization": self._create_mission_optimization_workflow(),
            "stealth_fighter_design": self._create_stealth_fighter_workflow(),
            "modular_configuration": self._create_modular_configuration_workflow()
        }
    
    def _create_concept_to_manufacturing_workflow(self) -> List[WorkflowStep]:
        """Create the complete concept-to-manufacturing workflow."""
        return [
            WorkflowStep(
                name="initial_configuration",
                description="Create initial aircraft configuration",
                engine="design",
                operation="create_configuration",
                inputs={"platform_type": "fighter", "mission_class": "air_superiority"},
                expected_outputs=["configuration_id", "base_platform"],
                validation_criteria={"base_platform": True, "module_count": {"min": 1}}
            ),
            WorkflowStep(
                name="materials_selection",
                description="Select and validate materials for stealth and performance",
                engine="materials",
                operation="optimize_materials",
                inputs={"configuration_ref": "initial_configuration", "stealth_priority": 0.8},
                expected_outputs=["material_assignments", "stealth_properties"],
                validation_criteria={"rcs_reduction": {"min": 0.7}, "weight_penalty": {"max": 0.15}}
            ),
            WorkflowStep(
                name="propulsion_integration",
                description="Design and integrate propulsion system",
                engine="propulsion",
                operation="design_propulsion",
                inputs={"configuration_ref": "initial_configuration", "thrust_requirement": 150000},
                expected_outputs=["engine_config", "thermal_management"],
                validation_criteria={"thrust_to_weight": {"min": 1.2}, "fuel_efficiency": {"min": 0.8}}
            ),
            WorkflowStep(
                name="sensor_integration",
                description="Integrate advanced sensor systems",
                engine="sensors",
                operation="integrate_sensors",
                inputs={"configuration_ref": "initial_configuration", "sensor_suite": "advanced_aesa"},
                expected_outputs=["sensor_config", "power_requirements"],
                validation_criteria={"detection_range": {"min": 200}, "power_consumption": {"max": 50000}}
            ),
            WorkflowStep(
                name="aerodynamic_validation",
                description="Validate aerodynamic performance across flight envelope",
                engine="aerodynamics",
                operation="validate_performance",
                inputs={"configuration_ref": "initial_configuration", "flight_envelope": "full"},
                expected_outputs=["aero_performance", "stability_margins"],
                validation_criteria={"max_speed": {"min": 2.5}, "stability_margin": {"min": 0.1}}
            ),
            WorkflowStep(
                name="manufacturing_planning",
                description="Generate manufacturing specifications and cost analysis",
                engine="manufacturing",
                operation="plan_manufacturing",
                inputs={"configuration_ref": "initial_configuration", "production_volume": 100},
                expected_outputs=["manufacturing_plan", "cost_estimate"],
                validation_criteria={"unit_cost": {"max": 150000000}, "production_time": {"max": 24}}
            )
        ]
    
    def _create_mission_optimization_workflow(self) -> List[WorkflowStep]:
        """Create mission-specific optimization workflow."""
        return [
            WorkflowStep(
                name="mission_analysis",
                description="Analyze mission requirements and constraints",
                engine="design",
                operation="analyze_mission",
                inputs={"mission_type": "deep_strike", "range": 2000, "payload": 8000},
                expected_outputs=["mission_profile", "optimization_targets"],
                validation_criteria={"feasibility_score": {"min": 0.8}}
            ),
            WorkflowStep(
                name="configuration_optimization",
                description="Optimize aircraft configuration for mission",
                engine="design",
                operation="optimize_configuration",
                inputs={"mission_ref": "mission_analysis", "optimization_method": "multi_objective"},
                expected_outputs=["optimized_config", "pareto_frontier"],
                validation_criteria={"optimization_convergence": True, "pareto_points": {"min": 10}}
            ),
            WorkflowStep(
                name="performance_validation",
                description="Validate optimized configuration performance",
                engine="aerodynamics",
                operation="validate_mission_performance",
                inputs={"config_ref": "configuration_optimization", "mission_ref": "mission_analysis"},
                expected_outputs=["mission_performance", "success_probability"],
                validation_criteria={"mission_success_rate": {"min": 0.95}, "fuel_margin": {"min": 0.1}}
            )
        ]
    
    def _create_stealth_fighter_workflow(self) -> List[WorkflowStep]:
        """Create stealth-focused fighter design workflow."""
        return [
            WorkflowStep(
                name="stealth_requirements",
                description="Define stealth requirements and constraints",
                engine="materials",
                operation="define_stealth_requirements",
                inputs={"threat_environment": "advanced_sam", "detection_threshold": -40},
                expected_outputs=["stealth_spec", "material_constraints"],
                validation_criteria={"rcs_target": {"max": -40}, "frequency_coverage": {"min": 0.9}}
            ),
            WorkflowStep(
                name="shape_optimization",
                description="Optimize aircraft shape for stealth and aerodynamics",
                engine="aerodynamics",
                operation="optimize_stealth_shape",
                inputs={"stealth_ref": "stealth_requirements", "aero_priority": 0.3},
                expected_outputs=["optimized_shape", "rcs_analysis"],
                validation_criteria={"rcs_reduction": {"min": 0.8}, "aero_penalty": {"max": 0.2}}
            ),
            WorkflowStep(
                name="materials_integration",
                description="Integrate stealth materials and coatings",
                engine="materials",
                operation="integrate_stealth_materials",
                inputs={"shape_ref": "shape_optimization", "coating_type": "metamaterial_ram"},
                expected_outputs=["material_config", "manufacturing_constraints"],
                validation_criteria={"coating_effectiveness": {"min": 0.9}, "durability": {"min": 1000}}
            )
        ]
    
    def _create_modular_configuration_workflow(self) -> List[WorkflowStep]:
        """Create modular configuration validation workflow."""
        return [
            WorkflowStep(
                name="module_compatibility",
                description="Validate all module combinations",
                engine="design",
                operation="validate_module_compatibility",
                inputs={"module_library": "full", "compatibility_matrix": True},
                expected_outputs=["compatibility_results", "invalid_combinations"],
                validation_criteria={"compatibility_rate": {"min": 0.95}, "interface_validation": True}
            ),
            WorkflowStep(
                name="configuration_variants",
                description="Generate and validate configuration variants",
                engine="design",
                operation="generate_variants",
                inputs={"base_platform": "su75_inspired", "variant_count": 20},
                expected_outputs=["variant_configs", "performance_matrix"],
                validation_criteria={"variant_diversity": {"min": 0.8}, "all_valid": True}
            ),
            WorkflowStep(
                name="performance_comparison",
                description="Compare performance across all variants",
                engine="aerodynamics",
                operation="compare_variants",
                inputs={"variants_ref": "configuration_variants", "metrics": "comprehensive"},
                expected_outputs=["performance_comparison", "ranking"],
                validation_criteria={"performance_spread": {"min": 0.2}, "ranking_consistency": True}
            )
        ]
    
    def _load_benchmark_data(self):
        """Load performance benchmark data for validation."""
        self.benchmark_data = {
            "thrust_to_weight": {"f22": 1.26, "f35": 1.07, "su57": 1.19},
            "max_speed_mach": {"f22": 2.25, "f35": 1.6, "su57": 2.0},
            "combat_radius_km": {"f22": 760, "f35": 1135, "su57": 1500},
            "rcs_m2": {"f22": 0.0001, "f35": 0.005, "su57": 0.1},
            "unit_cost_million": {"f22": 150, "f35": 80, "su57": 50}
        }    

    def execute_workflow(self, workflow_name: str, 
                        configuration_overrides: Optional[Dict[str, Any]] = None) -> WorkflowValidationReport:
        """
        Execute a complete end-to-end workflow and validate results.
        
        Args:
            workflow_name: Name of the workflow to execute
            configuration_overrides: Optional overrides for workflow parameters
            
        Returns:
            Complete validation report with results and benchmarks
        """
        if workflow_name not in self.workflows:
            raise WorkflowError(f"Unknown workflow: {workflow_name}")
        
        workflow_steps = self.workflows[workflow_name]
        step_results = []
        workflow_context = {}
        start_time = time.time()
        
        logger.info(f"Starting end-to-end workflow: {workflow_name}")
        
        for step in workflow_steps:
            logger.info(f"Executing step: {step.name}")
            
            try:
                step_result = self._execute_workflow_step(step, workflow_context, configuration_overrides)
                step_results.append(step_result)
                
                # Update workflow context with step outputs
                if step_result.success:
                    workflow_context[step.name] = step_result.outputs
                else:
                    logger.error(f"Step {step.name} failed: {step_result.errors}")
                    break
                    
            except Exception as e:
                logger.error(f"Exception in step {step.name}: {str(e)}")
                step_results.append(WorkflowResult(
                    step_name=step.name,
                    success=False,
                    execution_time=0.0,
                    outputs={},
                    validation_results={},
                    errors=[str(e)],
                    warnings=[]
                ))
                break
        
        total_time = time.time() - start_time
        
        # Generate performance benchmarks
        performance_benchmarks = self._generate_performance_benchmarks(workflow_context)
        
        # Validate against requirements
        requirements_validation = self._validate_requirements(workflow_name, workflow_context)
        
        # Create validation report
        successful_steps = sum(1 for result in step_results if result.success)
        report = WorkflowValidationReport(
            workflow_name=workflow_name,
            total_steps=len(workflow_steps),
            successful_steps=successful_steps,
            failed_steps=len(step_results) - successful_steps,
            total_execution_time=total_time,
            step_results=step_results,
            performance_benchmarks=performance_benchmarks,
            requirements_validation=requirements_validation,
            overall_success=successful_steps == len(workflow_steps)
        )
        
        logger.info(f"Workflow {workflow_name} completed. Success: {report.overall_success}")
        return report
    
    def _execute_workflow_step(self, step: WorkflowStep, context: Dict[str, Any],
                              overrides: Optional[Dict[str, Any]] = None) -> WorkflowResult:
        """Execute a single workflow step."""
        start_time = time.time()
        
        # Resolve input references from context
        resolved_inputs = self._resolve_step_inputs(step.inputs, context)
        
        # Apply any configuration overrides
        if overrides and step.name in overrides:
            resolved_inputs.update(overrides[step.name])
        
        try:
            # Get the appropriate engine
            if step.engine not in self.engines:
                raise WorkflowError(f"Engine not available: {step.engine}")
            
            engine = self.engines[step.engine]
            
            # Execute the operation
            if not hasattr(engine, step.operation):
                raise WorkflowError(f"Operation {step.operation} not available on {step.engine} engine")
            
            operation = getattr(engine, step.operation)
            outputs = operation(**resolved_inputs)
            
            execution_time = time.time() - start_time
            
            # Validate outputs
            validation_results = self._validate_step_outputs(step, outputs)
            
            # Check for warnings
            warnings = self._check_step_warnings(step, outputs)
            
            return WorkflowResult(
                step_name=step.name,
                success=all(validation_results.values()),
                execution_time=execution_time,
                outputs=outputs,
                validation_results=validation_results,
                errors=[],
                warnings=warnings
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return WorkflowResult(
                step_name=step.name,
                success=False,
                execution_time=execution_time,
                outputs={},
                validation_results={},
                errors=[str(e)],
                warnings=[]
            )
    
    def _resolve_step_inputs(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve input references from workflow context."""
        resolved = {}
        
        for key, value in inputs.items():
            if isinstance(value, str) and value.endswith("_ref"):
                # This is a reference to a previous step's output
                ref_step = value.replace("_ref", "")
                if ref_step in context:
                    resolved[key] = context[ref_step]
                else:
                    raise WorkflowError(f"Reference not found: {value}")
            else:
                resolved[key] = value
        
        return resolved
    
    def _validate_step_outputs(self, step: WorkflowStep, outputs: Dict[str, Any]) -> Dict[str, bool]:
        """Validate step outputs against criteria."""
        validation_results = {}
        
        # Check that all expected outputs are present
        for expected_output in step.expected_outputs:
            validation_results[f"has_{expected_output}"] = expected_output in outputs
        
        # Check validation criteria
        for criterion, constraint in step.validation_criteria.items():
            if criterion in outputs:
                value = outputs[criterion]
                
                if isinstance(constraint, dict):
                    if "min" in constraint:
                        validation_results[f"{criterion}_min"] = value >= constraint["min"]
                    if "max" in constraint:
                        validation_results[f"{criterion}_max"] = value <= constraint["max"]
                elif isinstance(constraint, bool) and constraint:
                    # For boolean True constraints, just check if the key exists and has a truthy value
                    validation_results[criterion] = bool(value)
                else:
                    validation_results[criterion] = value == constraint
            else:
                validation_results[criterion] = False
        
        return validation_results
    
    def _check_step_warnings(self, step: WorkflowStep, outputs: Dict[str, Any]) -> List[str]:
        """Check for potential warnings in step outputs."""
        warnings = []
        
        # Add specific warning checks based on step type
        if step.engine == "materials" and "weight_penalty" in outputs:
            if outputs["weight_penalty"] > 0.1:
                warnings.append("High weight penalty from material selection")
        
        if step.engine == "propulsion" and "fuel_efficiency" in outputs:
            if outputs["fuel_efficiency"] < 0.7:
                warnings.append("Low fuel efficiency may impact range")
        
        if step.engine == "sensors" and "power_consumption" in outputs:
            if outputs["power_consumption"] > 40000:
                warnings.append("High sensor power consumption")
        
        return warnings
    
    def _generate_performance_benchmarks(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Generate performance benchmarks from workflow results."""
        benchmarks = {}
        
        # Extract key performance metrics from workflow context
        for step_name, step_outputs in context.items():
            if isinstance(step_outputs, dict):
                for key, value in step_outputs.items():
                    if isinstance(value, (int, float)):
                        benchmarks[f"{step_name}_{key}"] = float(value)
        
        # Calculate derived benchmarks
        if "propulsion_integration_thrust_to_weight" in benchmarks:
            ttw = benchmarks["propulsion_integration_thrust_to_weight"]
            benchmarks["ttw_vs_f22"] = ttw / self.benchmark_data["thrust_to_weight"]["f22"]
        
        if "aerodynamic_validation_max_speed" in benchmarks:
            speed = benchmarks["aerodynamic_validation_max_speed"]
            benchmarks["speed_vs_f22"] = speed / self.benchmark_data["max_speed_mach"]["f22"]
        
        return benchmarks
    
    def _validate_requirements(self, workflow_name: str, context: Dict[str, Any]) -> Dict[str, bool]:
        """Validate workflow results against design requirements."""
        validation = {}
        
        # Requirement 9.4: Performance validation completed
        validation["req_9_4_performance_validation"] = "aerodynamic_validation" in context
        
        # Requirement 10.4: Progress reports generated
        validation["req_10_4_progress_reports"] = len(context) > 0
        
        # Workflow-specific validations
        if workflow_name == "concept_to_manufacturing":
            validation["complete_workflow"] = len(context) >= 6
            validation["manufacturing_ready"] = "manufacturing_planning" in context
        
        elif workflow_name == "mission_optimization":
            validation["mission_optimized"] = "configuration_optimization" in context
            validation["performance_validated"] = "performance_validation" in context
        
        elif workflow_name == "stealth_fighter_design":
            validation["stealth_optimized"] = "shape_optimization" in context
            validation["materials_integrated"] = "materials_integration" in context
        
        return validation
    
    def generate_user_acceptance_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Generate user acceptance testing scenarios."""
        scenarios = {
            "air_superiority_fighter": {
                "description": "Design an air superiority fighter with advanced stealth",
                "workflow": "stealth_fighter_design",
                "acceptance_criteria": {
                    "rcs_reduction": {"min": 0.8},
                    "max_speed": {"min": 2.0},
                    "thrust_to_weight": {"min": 1.2}
                },
                "test_data": {
                    "mission_type": "air_superiority",
                    "stealth_priority": 0.9,
                    "performance_priority": 0.8
                }
            },
            "multi_role_strike": {
                "description": "Design a multi-role strike fighter with modular payload",
                "workflow": "mission_optimization",
                "acceptance_criteria": {
                    "payload_capacity": {"min": 8000},
                    "combat_radius": {"min": 1000},
                    "mission_success_rate": {"min": 0.9}
                },
                "test_data": {
                    "mission_type": "deep_strike",
                    "range": 2000,
                    "payload": 8000
                }
            },
            "modular_platform": {
                "description": "Validate modular platform with multiple configurations",
                "workflow": "modular_configuration",
                "acceptance_criteria": {
                    "configuration_variants": {"min": 15},
                    "compatibility_rate": {"min": 0.95},
                    "performance_diversity": {"min": 0.3}
                },
                "test_data": {
                    "platform_type": "modular",
                    "variant_count": 20
                }
            }
        }
        
        return scenarios
    
    def run_user_acceptance_tests(self) -> Dict[str, WorkflowValidationReport]:
        """Run all user acceptance testing scenarios."""
        scenarios = self.generate_user_acceptance_scenarios()
        results = {}
        
        for scenario_name, scenario_config in scenarios.items():
            logger.info(f"Running user acceptance test: {scenario_name}")
            
            try:
                # Execute the workflow with test data
                report = self.execute_workflow(
                    scenario_config["workflow"],
                    {"test_scenario": scenario_config["test_data"]}
                )
                
                # Validate against acceptance criteria
                acceptance_validation = {}
                for criterion, constraint in scenario_config["acceptance_criteria"].items():
                    if criterion in report.performance_benchmarks:
                        value = report.performance_benchmarks[criterion]
                        if "min" in constraint:
                            acceptance_validation[criterion] = value >= constraint["min"]
                        elif "max" in constraint:
                            acceptance_validation[criterion] = value <= constraint["max"]
                    else:
                        acceptance_validation[criterion] = False
                
                # Update report with acceptance validation
                report.requirements_validation.update(acceptance_validation)
                results[scenario_name] = report
                
            except Exception as e:
                logger.error(f"User acceptance test {scenario_name} failed: {str(e)}")
                # Create a failed report
                results[scenario_name] = WorkflowValidationReport(
                    workflow_name=scenario_config["workflow"],
                    total_steps=0,
                    successful_steps=0,
                    failed_steps=1,
                    total_execution_time=0.0,
                    step_results=[],
                    performance_benchmarks={},
                    requirements_validation={"test_execution": False},
                    overall_success=False
                )
        
        return results
    
    def export_validation_report(self, report: WorkflowValidationReport, 
                                output_path: Path) -> None:
        """Export validation report to JSON file."""
        report_data = asdict(report)
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Validation report exported to {output_path}")
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get summary of available workflows and their capabilities."""
        summary = {}
        
        for workflow_name, steps in self.workflows.items():
            summary[workflow_name] = {
                "description": self._get_workflow_description(workflow_name),
                "total_steps": len(steps),
                "engines_involved": list(set(step.engine for step in steps)),
                "estimated_duration": sum(step.timeout_seconds for step in steps),
                "key_outputs": [step.expected_outputs for step in steps]
            }
        
        return summary
    
    def _get_workflow_description(self, workflow_name: str) -> str:
        """Get description for a workflow."""
        descriptions = {
            "concept_to_manufacturing": "Complete aircraft design from initial concept through manufacturing planning",
            "mission_optimization": "Mission-specific aircraft optimization and performance validation",
            "stealth_fighter_design": "Stealth-focused fighter design with shape and materials optimization",
            "modular_configuration": "Validation of modular platform configurations and compatibility"
        }
        
        return descriptions.get(workflow_name, "Custom workflow")