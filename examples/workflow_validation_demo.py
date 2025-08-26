#!/usr/bin/env python3
"""
Demonstration of end-to-end workflow validation capabilities.

This script shows how to use the Fighter Jet SDK's workflow validation system
to execute complete aircraft design workflows and validate results against
design requirements and performance benchmarks.
"""

import json
import sys
from pathlib import Path

# Add the SDK to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fighter_jet_sdk.core.workflow_validator import EndToEndWorkflowValidator
from fighter_jet_sdk.cli.main import get_engine_registry


def main():
    """Run workflow validation demonstration."""
    print("=" * 60)
    print("Fighter Jet SDK - End-to-End Workflow Validation Demo")
    print("=" * 60)
    
    try:
        # Initialize the workflow validator with real engines
        print("\n1. Initializing workflow validator...")
        engines = get_engine_registry()
        validator = EndToEndWorkflowValidator(engines)
        
        print(f"   ✓ Loaded {len(engines)} engines")
        print(f"   ✓ Configured {len(validator.workflows)} workflows")
        
        # Show available workflows
        print("\n2. Available workflows:")
        summary = validator.get_workflow_summary()
        for workflow_name, info in summary.items():
            print(f"   • {workflow_name}")
            print(f"     Description: {info['description']}")
            print(f"     Steps: {info['total_steps']}, Engines: {len(info['engines_involved'])}")
        
        # Generate user acceptance scenarios
        print("\n3. User acceptance testing scenarios:")
        scenarios = validator.generate_user_acceptance_scenarios()
        for scenario_name, scenario in scenarios.items():
            print(f"   • {scenario_name}")
            print(f"     {scenario['description']}")
            print(f"     Workflow: {scenario['workflow']}")
        
        # Demonstrate workflow execution (using mock engines)
        print("\n4. Executing sample workflow...")
        print("   Note: This demo uses mock engines for demonstration purposes.")
        
        # Create mock engines for demonstration
        mock_engines = create_demo_engines()
        demo_validator = EndToEndWorkflowValidator(mock_engines)
        
        # Execute a workflow
        print("   Executing 'mission_optimization' workflow...")
        report = demo_validator.execute_workflow("mission_optimization")
        
        # Display results
        print(f"\n5. Workflow execution results:")
        print(f"   Status: {'SUCCESS' if report.overall_success else 'FAILED'}")
        print(f"   Total steps: {report.total_steps}")
        print(f"   Successful steps: {report.successful_steps}")
        print(f"   Execution time: {report.total_execution_time:.2f} seconds")
        
        if report.step_results:
            print("\n   Step details:")
            for result in report.step_results:
                status = "✓" if result.success else "✗"
                print(f"     {status} {result.step_name} ({result.execution_time:.3f}s)")
                if result.warnings:
                    for warning in result.warnings:
                        print(f"       Warning: {warning}")
        
        # Show performance benchmarks
        if report.performance_benchmarks:
            print("\n6. Performance benchmarks:")
            for metric, value in report.performance_benchmarks.items():
                if isinstance(value, float):
                    print(f"   {metric}: {value:.3f}")
                else:
                    print(f"   {metric}: {value}")
        
        # Show requirements validation
        if report.requirements_validation:
            print("\n7. Requirements validation:")
            for requirement, passed in report.requirements_validation.items():
                status = "✓" if passed else "✗"
                print(f"   {status} {requirement}")
        
        # Run user acceptance tests
        print("\n8. Running user acceptance tests...")
        acceptance_results = demo_validator.run_user_acceptance_tests()
        
        total_tests = len(acceptance_results)
        passed_tests = sum(1 for report in acceptance_results.values() if report.overall_success)
        
        print(f"   Total tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {total_tests - passed_tests}")
        
        for scenario_name, test_report in acceptance_results.items():
            status = "PASS" if test_report.overall_success else "FAIL"
            print(f"   • {scenario_name}: {status}")
        
        print("\n" + "=" * 60)
        print("Workflow validation demonstration completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def create_demo_engines():
    """Create mock engines for demonstration purposes."""
    from tests.test_end_to_end_workflows import MockEngine
    
    engines = {}
    
    # Design engine
    design_engine = MockEngine("design")
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
    design_engine.add_operation("define_stealth_requirements", {
        "stealth_spec": {"rcs_target": -42, "frequency_bands": ["x", "ku"]},
        "material_constraints": {"temperature_max": 1200, "thickness_max": 5},
        "rcs_target": -42,
        "frequency_coverage": 0.95
    })
    design_engine.add_operation("validate_module_compatibility", {
        "compatibility_results": {"total_combinations": 100, "valid": 96},
        "invalid_combinations": ["sensor_a + payload_x"],
        "compatibility_rate": 0.96,
        "interface_validation": True
    })
    design_engine.add_operation("generate_variants", {
        "variant_configs": [{"id": f"variant_{i}"} for i in range(20)],
        "performance_matrix": [[0.8, 0.9, 0.7] for _ in range(20)],
        "variant_diversity": 0.85,
        "all_valid": True
    })
    engines["design"] = design_engine
    
    # Materials engine
    materials_engine = MockEngine("materials")
    materials_engine.add_operation("integrate_stealth_materials", {
        "material_config": {"coating_type": "metamaterial_ram", "thickness": 3.2},
        "manufacturing_constraints": {"cure_temp": 180, "pressure": 6},
        "coating_effectiveness": 0.92,
        "durability": 1200
    })
    engines["materials"] = materials_engine
    
    # Aerodynamics engine
    aero_engine = MockEngine("aerodynamics")
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
    
    # Add other engines with minimal operations
    for engine_name in ["propulsion", "sensors", "manufacturing"]:
        engine = MockEngine(engine_name)
        engine.add_operation("placeholder_operation", {"status": "completed"})
        engines[engine_name] = engine
    
    return engines


if __name__ == "__main__":
    sys.exit(main())