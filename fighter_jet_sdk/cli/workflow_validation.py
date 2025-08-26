"""
CLI commands for end-to-end workflow validation.

This module provides command-line interface for executing and validating
complete aircraft design workflows from concept to manufacturing.
"""

import click
import json
from pathlib import Path
from typing import Dict, Any

from ..core.workflow_validator import EndToEndWorkflowValidator
from ..core.logging import get_logger
from ..core.errors import WorkflowError
from .main import get_engine_registry

logger = get_logger(__name__)


@click.group(name='workflow')
def workflow_cli():
    """End-to-end workflow validation commands."""
    pass


@workflow_cli.command()
@click.option('--list-workflows', '-l', is_flag=True, 
              help='List available workflows')
@click.option('--workflow', '-w', type=str,
              help='Workflow name to execute')
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Configuration overrides file (JSON)')
@click.option('--output', '-o', type=click.Path(),
              help='Output file for validation report')
@click.option('--benchmark', '-b', is_flag=True,
              help='Include performance benchmarking')
@click.option('--verbose', '-v', is_flag=True,
              help='Verbose output')
def validate(list_workflows, workflow, config, output, benchmark, verbose):
    """
    Execute end-to-end workflow validation.
    
    This command runs complete aircraft design workflows and validates
    the results against design requirements and performance benchmarks.
    
    Examples:
        # List available workflows
        fighter-jet workflow validate --list-workflows
        
        # Run concept-to-manufacturing workflow
        fighter-jet workflow validate -w concept_to_manufacturing
        
        # Run with configuration overrides
        fighter-jet workflow validate -w mission_optimization -c config.json
        
        # Save detailed report
        fighter-jet workflow validate -w stealth_fighter_design -o report.json
    """
    try:
        # Get engine registry
        engines = get_engine_registry()
        validator = EndToEndWorkflowValidator(engines)
        
        if list_workflows:
            _list_workflows(validator, verbose)
            return
        
        if not workflow:
            click.echo("Error: Must specify --workflow or use --list-workflows")
            return
        
        # Load configuration overrides if provided
        config_overrides = None
        if config:
            with open(config, 'r') as f:
                config_overrides = json.load(f)
        
        # Execute workflow
        click.echo(f"Executing workflow: {workflow}")
        if verbose:
            click.echo("This may take several minutes...")
        
        report = validator.execute_workflow(workflow, config_overrides)
        
        # Display results
        _display_workflow_results(report, benchmark, verbose)
        
        # Save report if requested
        if output:
            validator.export_validation_report(report, Path(output))
            click.echo(f"Detailed report saved to: {output}")
        
    except WorkflowError as e:
        click.echo(f"Workflow error: {e}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        raise click.Abort()


@workflow_cli.command()
@click.option('--output', '-o', type=click.Path(),
              help='Output directory for test reports')
@click.option('--scenario', '-s', type=str,
              help='Specific scenario to run (default: all)')
@click.option('--verbose', '-v', is_flag=True,
              help='Verbose output')
def acceptance_test(output, scenario, verbose):
    """
    Run user acceptance testing scenarios.
    
    This command executes predefined user acceptance tests that validate
    the system against real-world aircraft design requirements.
    
    Examples:
        # Run all acceptance tests
        fighter-jet workflow acceptance-test
        
        # Run specific scenario
        fighter-jet workflow acceptance-test -s air_superiority_fighter
        
        # Save detailed reports
        fighter-jet workflow acceptance-test -o test_reports/
    """
    try:
        engines = get_engine_registry()
        validator = EndToEndWorkflowValidator(engines)
        
        if scenario:
            # Run specific scenario
            scenarios = validator.generate_user_acceptance_scenarios()
            if scenario not in scenarios:
                click.echo(f"Error: Unknown scenario '{scenario}'")
                click.echo(f"Available scenarios: {', '.join(scenarios.keys())}")
                return
            
            click.echo(f"Running acceptance test: {scenario}")
            # Execute single scenario (would need to modify validator for this)
            results = validator.run_user_acceptance_tests()
            results = {scenario: results[scenario]}
        else:
            # Run all scenarios
            click.echo("Running all user acceptance tests...")
            results = validator.run_user_acceptance_tests()
        
        # Display results
        _display_acceptance_test_results(results, verbose)
        
        # Save reports if requested
        if output:
            output_dir = Path(output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for scenario_name, report in results.items():
                report_file = output_dir / f"{scenario_name}_report.json"
                validator.export_validation_report(report, report_file)
            
            click.echo(f"Test reports saved to: {output_dir}")
        
    except Exception as e:
        click.echo(f"Error running acceptance tests: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        raise click.Abort()


@workflow_cli.command()
@click.option('--workflow', '-w', type=str,
              help='Specific workflow to benchmark (default: all)')
@click.option('--reference', '-r', type=str, default='f22',
              help='Reference aircraft for comparison (f22, f35, su57)')
@click.option('--output', '-o', type=click.Path(),
              help='Output file for benchmark results')
@click.option('--verbose', '-v', is_flag=True,
              help='Verbose output')
def benchmark(workflow, reference, output, verbose):
    """
    Run performance benchmarking against reference aircraft.
    
    This command executes workflows and compares the results against
    known performance characteristics of reference fighter aircraft.
    
    Examples:
        # Benchmark all workflows against F-22
        fighter-jet workflow benchmark
        
        # Benchmark specific workflow against F-35
        fighter-jet workflow benchmark -w concept_to_manufacturing -r f35
        
        # Save benchmark results
        fighter-jet workflow benchmark -o benchmark_results.json
    """
    try:
        engines = get_engine_registry()
        validator = EndToEndWorkflowValidator(engines)
        
        workflows_to_test = [workflow] if workflow else list(validator.workflows.keys())
        benchmark_results = {}
        
        for workflow_name in workflows_to_test:
            click.echo(f"Benchmarking workflow: {workflow_name}")
            
            report = validator.execute_workflow(workflow_name)
            
            # Extract performance metrics
            performance_metrics = _extract_performance_metrics(report, reference, validator)
            benchmark_results[workflow_name] = performance_metrics
            
            if verbose:
                _display_benchmark_details(workflow_name, performance_metrics)
        
        # Display summary
        _display_benchmark_summary(benchmark_results, reference)
        
        # Save results if requested
        if output:
            with open(output, 'w') as f:
                json.dump(benchmark_results, f, indent=2)
            click.echo(f"Benchmark results saved to: {output}")
        
    except Exception as e:
        click.echo(f"Error running benchmarks: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        raise click.Abort()


def _list_workflows(validator: EndToEndWorkflowValidator, verbose: bool):
    """List available workflows with details."""
    summary = validator.get_workflow_summary()
    
    click.echo("Available Workflows:")
    click.echo("=" * 50)
    
    for workflow_name, info in summary.items():
        click.echo(f"\n{workflow_name}:")
        click.echo(f"  Description: {info['description']}")
        click.echo(f"  Steps: {info['total_steps']}")
        click.echo(f"  Engines: {', '.join(info['engines_involved'])}")
        
        if verbose:
            click.echo(f"  Estimated Duration: {info['estimated_duration']:.1f} seconds")
            click.echo(f"  Key Outputs: {len(info['key_outputs'])} categories")


def _display_workflow_results(report, include_benchmark: bool, verbose: bool):
    """Display workflow execution results."""
    click.echo("\nWorkflow Execution Results:")
    click.echo("=" * 50)
    
    # Overall status
    status_color = 'green' if report.overall_success else 'red'
    click.echo(f"Status: ", nl=False)
    click.secho(f"{'SUCCESS' if report.overall_success else 'FAILED'}", 
                fg=status_color, bold=True)
    
    click.echo(f"Total Steps: {report.total_steps}")
    click.echo(f"Successful: {report.successful_steps}")
    click.echo(f"Failed: {report.failed_steps}")
    click.echo(f"Execution Time: {report.total_execution_time:.2f} seconds")
    
    # Step details
    if verbose or report.failed_steps > 0:
        click.echo("\nStep Details:")
        for result in report.step_results:
            status_icon = "✓" if result.success else "✗"
            status_color = 'green' if result.success else 'red'
            
            click.echo(f"  {status_icon} ", nl=False)
            click.secho(f"{result.step_name}", fg=status_color)
            click.echo(f"    Time: {result.execution_time:.2f}s")
            
            if result.errors:
                for error in result.errors:
                    click.echo(f"    Error: {error}", err=True)
            
            if result.warnings and verbose:
                for warning in result.warnings:
                    click.echo(f"    Warning: {warning}")
    
    # Performance benchmarks
    if include_benchmark and report.performance_benchmarks:
        click.echo("\nPerformance Benchmarks:")
        for metric, value in report.performance_benchmarks.items():
            if isinstance(value, float):
                click.echo(f"  {metric}: {value:.3f}")
            else:
                click.echo(f"  {metric}: {value}")
    
    # Requirements validation
    if report.requirements_validation:
        click.echo("\nRequirements Validation:")
        for requirement, passed in report.requirements_validation.items():
            status_icon = "✓" if passed else "✗"
            status_color = 'green' if passed else 'red'
            click.echo(f"  {status_icon} ", nl=False)
            click.secho(f"{requirement}", fg=status_color)


def _display_acceptance_test_results(results: Dict[str, Any], verbose: bool):
    """Display user acceptance test results."""
    click.echo("\nUser Acceptance Test Results:")
    click.echo("=" * 50)
    
    total_tests = len(results)
    passed_tests = sum(1 for report in results.values() if report.overall_success)
    
    click.echo(f"Total Tests: {total_tests}")
    click.echo(f"Passed: {passed_tests}")
    click.echo(f"Failed: {total_tests - passed_tests}")
    
    for scenario_name, report in results.items():
        status_color = 'green' if report.overall_success else 'red'
        status_text = 'PASS' if report.overall_success else 'FAIL'
        
        click.echo(f"\n{scenario_name}: ", nl=False)
        click.secho(status_text, fg=status_color, bold=True)
        
        if verbose or not report.overall_success:
            click.echo(f"  Steps: {report.successful_steps}/{report.total_steps}")
            click.echo(f"  Time: {report.total_execution_time:.2f}s")
            
            if not report.overall_success:
                failed_steps = [r for r in report.step_results if not r.success]
                for failed_step in failed_steps:
                    click.echo(f"  Failed: {failed_step.step_name}")
                    for error in failed_step.errors:
                        click.echo(f"    {error}")


def _extract_performance_metrics(report, reference: str, validator) -> Dict[str, Any]:
    """Extract performance metrics for benchmarking."""
    metrics = {
        'execution_success': report.overall_success,
        'total_time': report.total_execution_time,
        'step_success_rate': report.successful_steps / report.total_steps if report.total_steps > 0 else 0
    }
    
    # Add performance benchmarks
    metrics.update(report.performance_benchmarks)
    
    # Add reference comparisons
    reference_data = validator.benchmark_data
    for metric_category, reference_values in reference_data.items():
        if reference in reference_values:
            reference_value = reference_values[reference]
            
            # Look for corresponding metric in report
            for benchmark_key, benchmark_value in report.performance_benchmarks.items():
                if metric_category.replace('_', '') in benchmark_key.lower():
                    if isinstance(benchmark_value, (int, float)):
                        comparison_ratio = benchmark_value / reference_value
                        metrics[f"{metric_category}_vs_{reference}"] = comparison_ratio
    
    return metrics


def _display_benchmark_details(workflow_name: str, metrics: Dict[str, Any]):
    """Display detailed benchmark results for a workflow."""
    click.echo(f"\n  Detailed Results for {workflow_name}:")
    
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            if 'vs_' in metric_name:
                # This is a comparison ratio
                if value >= 1.0:
                    color = 'green'
                    comparison = f"({value:.1%} of reference)"
                else:
                    color = 'yellow'
                    comparison = f"({value:.1%} of reference)"
                click.echo(f"    {metric_name}: ", nl=False)
                click.secho(f"{value:.3f} {comparison}", fg=color)
            else:
                click.echo(f"    {metric_name}: {value:.3f}")
        else:
            click.echo(f"    {metric_name}: {value}")


def _display_benchmark_summary(results: Dict[str, Dict[str, Any]], reference: str):
    """Display benchmark summary across all workflows."""
    click.echo(f"\nBenchmark Summary (vs {reference.upper()}):")
    click.echo("=" * 50)
    
    # Calculate overall statistics
    total_workflows = len(results)
    successful_workflows = sum(1 for metrics in results.values() if metrics.get('execution_success', False))
    
    click.echo(f"Workflows Tested: {total_workflows}")
    click.echo(f"Successful Executions: {successful_workflows}")
    
    # Find best and worst performing workflows
    performance_scores = {}
    for workflow_name, metrics in results.items():
        # Calculate a composite performance score
        score = metrics.get('step_success_rate', 0)
        if 'thrust_to_weight_vs_' + reference in metrics:
            score += min(metrics['thrust_to_weight_vs_' + reference], 1.0) * 0.5
        performance_scores[workflow_name] = score
    
    if performance_scores:
        best_workflow = max(performance_scores.items(), key=lambda x: x[1])
        worst_workflow = min(performance_scores.items(), key=lambda x: x[1])
        
        click.echo(f"\nBest Performing: {best_workflow[0]} (score: {best_workflow[1]:.3f})")
        click.echo(f"Worst Performing: {worst_workflow[0]} (score: {worst_workflow[1]:.3f})")


# Register the workflow CLI group
def register_workflow_commands(main_cli):
    """Register workflow validation commands with the main CLI."""
    main_cli.add_command(workflow_cli)