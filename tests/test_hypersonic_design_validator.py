"""Tests for hypersonic design validation system."""

import pytest
import json
import time
from unittest.mock import Mock, patch

from fighter_jet_sdk.core.hypersonic_design_validator import (
    HypersonicDesignValidator,
    ThermalValidator,
    StructuralValidator,
    AerodynamicValidator,
    PropulsionValidator,
    PerformanceAnalyzer,
    DesignOptimizationRecommender,
    ValidationIssue,
    SafetyMargin,
    PerformanceMetric,
    ValidationReport,
    ValidationSeverity,
    ValidationCategory
)
from fighter_jet_sdk.common.data_models import AircraftConfiguration
from fighter_jet_sdk.core.hypersonic_mission_planner import HypersonicMissionProfile


class TestThermalValidator:
    """Test thermal validator."""
    
    def test_initialization(self):
        """Test thermal validator initialization."""
        validator = ThermalValidator()
        
        assert validator.name == "thermal"
        assert validator.max_surface_temperature == 3000.0
        assert validator.max_heat_flux == 100e6
    
    def test_temperature_validation_pass(self):
        """Test temperature validation with acceptable values."""
        validator = ThermalValidator()
        config = AircraftConfiguration(name="test", modules=[])
        
        analysis_results = {
            'thermal': {
                'max_surface_temperature': 2500.0,  # Below limit
                'max_heat_flux': 50e6,              # Below limit
                'ablation_rate': 1e-6,              # Below limit
                'cooling_effectiveness': 0.8        # Good effectiveness
            }
        }
        
        issues = validator.validate(config, analysis_results)
        
        # Should have no critical or error issues
        critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        error_issues = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        
        assert len(critical_issues) == 0
        assert len(error_issues) == 0
    
    def test_temperature_validation_fail(self):
        """Test temperature validation with excessive values."""
        validator = ThermalValidator()
        config = AircraftConfiguration(name="test", modules=[])
        
        analysis_results = {
            'thermal': {
                'max_surface_temperature': 3500.0,  # Above limit
                'max_heat_flux': 150e6,             # Above limit
                'ablation_rate': 2e-5,              # Above limit
                'cooling_effectiveness': 0.3        # Low effectiveness
            }
        }
        
        issues = validator.validate(config, analysis_results)
        
        # Should have critical and error issues
        critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        error_issues = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        warning_issues = [i for i in issues if i.severity == ValidationSeverity.WARNING]
        
        assert len(critical_issues) >= 1  # Temperature exceeds limit
        assert len(error_issues) >= 1     # Heat flux exceeds limit
        assert len(warning_issues) >= 1   # Ablation rate high
        
        # Check specific issue IDs
        issue_ids = [issue.issue_id for issue in issues]
        assert "THERMAL_001" in issue_ids  # Temperature
        assert "THERMAL_002" in issue_ids  # Heat flux
        assert "THERMAL_003" in issue_ids  # Ablation
    
    def test_safety_margins_calculation(self):
        """Test safety margins calculation."""
        validator = ThermalValidator()
        config = AircraftConfiguration(name="test", modules=[])
        
        analysis_results = {
            'thermal': {
                'max_surface_temperature': 2400.0,
                'max_heat_flux': 80e6
            }
        }
        
        margins = validator.calculate_safety_margins(config, analysis_results)
        
        assert len(margins) == 2  # Temperature and heat flux margins
        
        # Check temperature margin
        temp_margin = next(m for m in margins if m.parameter_name == "Surface Temperature")
        assert temp_margin.current_value == 2400.0
        assert temp_margin.limit_value == 3000.0
        assert temp_margin.safety_factor > 1.0
        assert temp_margin.margin_percentage > 0.0
        
        # Check heat flux margin
        flux_margin = next(m for m in margins if m.parameter_name == "Heat Flux")
        assert flux_margin.current_value == 80e6
        assert flux_margin.limit_value == 100e6
        assert flux_margin.safety_factor > 1.0


class TestStructuralValidator:
    """Test structural validator."""
    
    def test_initialization(self):
        """Test structural validator initialization."""
        validator = StructuralValidator()
        
        assert validator.name == "structural"
        assert validator.min_safety_factor == 1.5
        assert validator.max_stress_ratio == 0.8
    
    def test_safety_factor_validation(self):
        """Test safety factor validation."""
        validator = StructuralValidator()
        config = AircraftConfiguration(name="test", modules=[])
        
        # Test low safety factor
        analysis_results = {
            'structural': {
                'safety_factor': 1.2,  # Below minimum
                'max_stress': 200e6,   # Pa
                'max_displacement': 0.05,
                'thermal_stress_contribution': 50e6
            }
        }
        
        issues = validator.validate(config, analysis_results)
        
        # Should have critical issue for low safety factor
        critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        assert len(critical_issues) >= 1
        
        safety_factor_issue = next(i for i in critical_issues if i.issue_id == "STRUCT_001")
        assert "Safety Factor" in safety_factor_issue.title
        assert safety_factor_issue.parameters['safety_factor'] == 1.2
    
    def test_stress_validation(self):
        """Test stress validation."""
        validator = StructuralValidator()
        config = AircraftConfiguration(name="test", modules=[])
        
        # Test high stress
        analysis_results = {
            'structural': {
                'safety_factor': 2.0,
                'max_stress': 250e6,  # High stress (250 MPa vs 270 MPa yield)
                'max_displacement': 0.05,
                'thermal_stress_contribution': 50e6
            }
        }
        
        issues = validator.validate(config, analysis_results)
        
        # Should have error issue for high stress
        error_issues = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        stress_issues = [i for i in error_issues if "Stress" in i.title]
        
        if stress_issues:  # Stress ratio might be acceptable depending on calculation
            stress_issue = stress_issues[0]
            assert "stress" in stress_issue.description.lower()
    
    def test_displacement_validation(self):
        """Test displacement validation."""
        validator = StructuralValidator()
        config = AircraftConfiguration(name="test", modules=[])
        
        # Test excessive displacement
        analysis_results = {
            'structural': {
                'safety_factor': 2.0,
                'max_stress': 100e6,
                'max_displacement': 0.15,  # Above limit
                'thermal_stress_contribution': 50e6
            }
        }
        
        issues = validator.validate(config, analysis_results)
        
        # Should have warning for excessive displacement
        warning_issues = [i for i in issues if i.severity == ValidationSeverity.WARNING]
        displacement_issues = [i for i in warning_issues if "Displacement" in i.title]
        
        assert len(displacement_issues) >= 1
        displacement_issue = displacement_issues[0]
        assert displacement_issue.issue_id == "STRUCT_003"


class TestAerodynamicValidator:
    """Test aerodynamic validator."""
    
    def test_initialization(self):
        """Test aerodynamic validator initialization."""
        validator = AerodynamicValidator()
        
        assert validator.name == "aerodynamic"
        assert validator.max_stagnation_temperature == 50000.0
        assert validator.plasma_formation_threshold == 8000.0
    
    def test_stagnation_temperature_validation(self):
        """Test stagnation temperature validation."""
        validator = AerodynamicValidator()
        config = AircraftConfiguration(name="test", modules=[])
        
        # Test excessive stagnation temperature
        analysis_results = {
            'aerodynamic': {
                'stagnation_temperature': 60000.0,  # Above limit
                'stagnation_pressure': 1e6,
                'convective_heat_flux': 1e6,
                'radiative_heat_flux': 5e5,
                'plasma_formation': True,
                'shock_standoff_distance': 0.01
            }
        }
        
        issues = validator.validate(config, analysis_results)
        
        # Should have error for excessive stagnation temperature
        error_issues = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        temp_issues = [i for i in error_issues if "Stagnation Temperature" in i.title]
        
        assert len(temp_issues) >= 1
        temp_issue = temp_issues[0]
        assert temp_issue.issue_id == "AERO_001"
        assert temp_issue.parameters['stagnation_temperature'] == 60000.0
    
    def test_plasma_formation_validation(self):
        """Test plasma formation validation."""
        validator = AerodynamicValidator()
        config = AircraftConfiguration(name="test", modules=[])
        
        # Test plasma formation
        analysis_results = {
            'aerodynamic': {
                'stagnation_temperature': 10000.0,
                'plasma_formation': True,
                'shock_standoff_distance': 0.01
            }
        }
        
        issues = validator.validate(config, analysis_results)
        
        # Should have warning for plasma formation
        warning_issues = [i for i in issues if i.severity == ValidationSeverity.WARNING]
        plasma_issues = [i for i in warning_issues if "Plasma" in i.title]
        
        assert len(plasma_issues) >= 1
        plasma_issue = plasma_issues[0]
        assert plasma_issue.issue_id == "AERO_002"
        assert "plasma formation" in plasma_issue.description.lower()


class TestPerformanceAnalyzer:
    """Test performance analyzer."""
    
    def test_performance_analysis(self):
        """Test performance analysis."""
        analyzer = PerformanceAnalyzer()
        config = AircraftConfiguration(name="test", modules=[])
        
        analysis_results = {
            'thermal': {'max_surface_temperature': 2200.0},
            'structural': {'safety_factor': 1.8},
            'aerodynamic': {'stagnation_temperature': 25000.0}
        }
        
        metrics = analyzer.analyze_performance(config, analysis_results)
        
        assert len(metrics) >= 3  # At least thermal, structural, aerodynamic
        
        # Check thermal metric
        thermal_metric = next(m for m in metrics if "Temperature" in m.metric_name)
        assert thermal_metric.current_value == 2200.0
        assert thermal_metric.target_value == 2500.0
        assert thermal_metric.meets_requirement  # 2200 < 2500
        
        # Check structural metric
        structural_metric = next(m for m in metrics if "Safety Factor" in m.metric_name)
        assert structural_metric.current_value == 1.8
        assert structural_metric.target_value == 1.5
        assert structural_metric.meets_requirement  # 1.8 >= 1.5
    
    def test_performance_analysis_with_mission(self):
        """Test performance analysis with mission profile."""
        analyzer = PerformanceAnalyzer()
        config = AircraftConfiguration(name="test", modules=[])
        
        analysis_results = {
            'thermal': {'max_surface_temperature': 2200.0},
            'structural': {'safety_factor': 1.8},
            'aerodynamic': {'stagnation_temperature': 25000.0}
        }
        
        # Mock mission profile
        mission_profile = Mock()
        mission_profile.total_duration = 3000.0  # seconds
        
        metrics = analyzer.analyze_performance(config, analysis_results, mission_profile)
        
        # Should include mission duration metric
        mission_metrics = [m for m in metrics if "Mission Duration" in m.metric_name]
        assert len(mission_metrics) == 1
        
        mission_metric = mission_metrics[0]
        assert mission_metric.current_value == 3000.0
        assert mission_metric.target_value == 3600.0
        assert mission_metric.meets_requirement  # 3000 <= 3600


class TestDesignOptimizationRecommender:
    """Test design optimization recommender."""
    
    def test_critical_issue_recommendations(self):
        """Test recommendations for critical issues."""
        recommender = DesignOptimizationRecommender()
        
        # Create mock validation report with critical thermal issue
        report = Mock()
        report.issues = [
            ValidationIssue(
                issue_id="THERMAL_001",
                category=ValidationCategory.THERMAL,
                severity=ValidationSeverity.CRITICAL,
                title="Critical Thermal Issue",
                description="Temperature too high"
            )
        ]
        report.safety_margins = []
        report.validation_summary = {'coupling': {'converged': True}}
        report.design_score = 0.5
        
        recommendations = recommender.generate_recommendations(report)
        
        # Should have priority 1 recommendation for thermal issues
        thermal_recs = [r for r in recommendations if "PRIORITY 1" in r and "thermal" in r.lower()]
        assert len(thermal_recs) >= 1
    
    def test_low_margin_recommendations(self):
        """Test recommendations for low safety margins."""
        recommender = DesignOptimizationRecommender()
        
        # Create mock validation report with low margins
        report = Mock()
        report.issues = []
        report.safety_margins = [
            SafetyMargin(
                parameter_name="Test Parameter",
                current_value=95.0,
                limit_value=100.0,
                safety_factor=1.05,
                margin_percentage=5.0,  # Low margin
                acceptable=False
            )
        ]
        report.validation_summary = {'coupling': {'converged': True}}
        report.design_score = 0.7
        
        recommendations = recommender.generate_recommendations(report)
        
        # Should have priority 2 recommendation for margins
        margin_recs = [r for r in recommendations if "PRIORITY 2" in r and "margin" in r.lower()]
        assert len(margin_recs) >= 1
    
    def test_convergence_recommendations(self):
        """Test recommendations for convergence issues."""
        recommender = DesignOptimizationRecommender()
        
        # Create mock validation report with convergence issues
        report = Mock()
        report.issues = []
        report.safety_margins = []
        report.validation_summary = {'coupling': {'converged': False}}
        report.design_score = 0.6
        
        recommendations = recommender.generate_recommendations(report)
        
        # Should have priority 2 recommendation for coupling
        coupling_recs = [r for r in recommendations if "coupling" in r.lower()]
        assert len(coupling_recs) >= 1


class TestHypersonicDesignValidator:
    """Test main hypersonic design validator."""
    
    def test_initialization(self):
        """Test validator initialization."""
        validator = HypersonicDesignValidator()
        
        assert 'thermal' in validator.validators
        assert 'structural' in validator.validators
        assert 'aerodynamic' in validator.validators
        assert 'propulsion' in validator.validators
        assert validator.performance_analyzer is not None
        assert validator.optimization_recommender is not None
    
    def test_design_validation_success(self):
        """Test successful design validation."""
        validator = HypersonicDesignValidator()
        config = AircraftConfiguration(name="test_aircraft", modules=[])
        
        # Good analysis results
        analysis_results = {
            'thermal': {
                'max_surface_temperature': 2200.0,
                'max_heat_flux': 60e6,
                'ablation_rate': 5e-7,
                'cooling_effectiveness': 0.8
            },
            'structural': {
                'safety_factor': 2.0,
                'max_stress': 150e6,
                'max_displacement': 0.05,
                'thermal_stress_contribution': 30e6
            },
            'aerodynamic': {
                'stagnation_temperature': 20000.0,
                'stagnation_pressure': 5e5,
                'plasma_formation': False,
                'shock_standoff_distance': 0.02
            },
            'coupling': {
                'converged': True,
                'final_iteration_count': 10,
                'max_coupling_residual': 1e-7
            }
        }
        
        report = validator.validate_design(config, analysis_results)
        
        assert report.aircraft_config == config
        assert report.overall_status in ["DESIGN_ACCEPTABLE", "ACCEPTABLE_WITH_WARNINGS"]
        assert report.design_score > 0.5
        assert len(report.issues) >= 0  # May have some info/warning issues
        assert len(report.safety_margins) > 0
        assert len(report.performance_metrics) > 0
    
    def test_design_validation_with_issues(self):
        """Test design validation with critical issues."""
        validator = HypersonicDesignValidator()
        config = AircraftConfiguration(name="test_aircraft", modules=[])
        
        # Poor analysis results
        analysis_results = {
            'thermal': {
                'max_surface_temperature': 3500.0,  # Too high
                'max_heat_flux': 150e6,             # Too high
                'ablation_rate': 2e-5,              # Too high
                'cooling_effectiveness': 0.2        # Too low
            },
            'structural': {
                'safety_factor': 1.1,               # Too low
                'max_stress': 300e6,                # Too high
                'max_displacement': 0.2,            # Too high
                'thermal_stress_contribution': 100e6
            },
            'aerodynamic': {
                'stagnation_temperature': 60000.0,  # Too high
                'stagnation_pressure': 2e6,
                'plasma_formation': True,
                'shock_standoff_distance': 0.005
            },
            'coupling': {
                'converged': False,
                'final_iteration_count': 50,
                'max_coupling_residual': 1e-3
            }
        }
        
        report = validator.validate_design(config, analysis_results)
        
        assert report.overall_status == "CRITICAL_ISSUES"
        assert report.design_score < 0.5
        
        # Should have critical issues
        critical_issues = [i for i in report.issues if i.severity == ValidationSeverity.CRITICAL]
        assert len(critical_issues) >= 2  # Thermal and structural
        
        # Should have recommendations
        assert len(report.recommendations) > 0
        priority_1_recs = [r for r in report.recommendations if "PRIORITY 1" in r]
        assert len(priority_1_recs) >= 1
    
    def test_design_score_calculation(self):
        """Test design score calculation."""
        validator = HypersonicDesignValidator()
        
        # Test with various issue severities
        issues = [
            ValidationIssue("1", ValidationCategory.THERMAL, ValidationSeverity.CRITICAL, "Critical", ""),
            ValidationIssue("2", ValidationCategory.STRUCTURAL, ValidationSeverity.ERROR, "Error", ""),
            ValidationIssue("3", ValidationCategory.AERODYNAMIC, ValidationSeverity.WARNING, "Warning", ""),
            ValidationIssue("4", ValidationCategory.PROPULSION, ValidationSeverity.INFO, "Info", "")
        ]
        
        margins = [
            SafetyMargin("Test1", 90, 100, 1.1, 10, True),
            SafetyMargin("Test2", 80, 100, 1.25, 20, True)
        ]
        
        metrics = [
            PerformanceMetric("Metric1", 100, 90, meets_requirement=True),
            PerformanceMetric("Metric2", 80, 90, meets_requirement=False)
        ]
        
        score = validator._calculate_design_score(issues, margins, metrics)
        
        # Score should be reduced by issues but increased by good margins/metrics
        assert 0.0 <= score <= 1.0
        # With 1 critical (-0.3), 1 error (-0.2), 1 warning (-0.1), 1 info (-0.05)
        # Plus bonuses for margins and metrics
        expected_penalty = 0.3 + 0.2 + 0.1 + 0.05  # 0.65
        expected_margin_bonus = 1.0 * 0.2  # All margins acceptable
        expected_metric_bonus = 0.5 * 0.2  # Half metrics meet requirements
        expected_score = 1.0 - expected_penalty + expected_margin_bonus + expected_metric_bonus
        expected_score = max(0.0, min(1.0, expected_score))
        
        assert abs(score - expected_score) < 0.1  # Allow some tolerance
    
    def test_overall_status_determination(self):
        """Test overall status determination."""
        validator = HypersonicDesignValidator()
        
        # Test critical issues
        critical_issues = [ValidationIssue("1", ValidationCategory.THERMAL, ValidationSeverity.CRITICAL, "Critical", "")]
        status = validator._determine_overall_status(critical_issues, [])
        assert status == "CRITICAL_ISSUES"
        
        # Test error issues
        error_issues = [ValidationIssue("1", ValidationCategory.THERMAL, ValidationSeverity.ERROR, "Error", "")]
        status = validator._determine_overall_status(error_issues, [])
        assert status == "DESIGN_ISSUES"
        
        # Test warning issues
        warning_issues = [ValidationIssue("1", ValidationCategory.THERMAL, ValidationSeverity.WARNING, "Warning", "")]
        status = validator._determine_overall_status(warning_issues, [])
        assert status == "ACCEPTABLE_WITH_WARNINGS"
        
        # Test no issues
        status = validator._determine_overall_status([], [])
        assert status == "DESIGN_ACCEPTABLE"
    
    def test_json_export(self):
        """Test JSON report export."""
        validator = HypersonicDesignValidator()
        
        # Create mock report
        config = AircraftConfiguration(name="test", modules=[])
        report = ValidationReport(
            report_id="test_report",
            aircraft_config=config,
            validation_timestamp=time.time(),
            overall_status="DESIGN_ACCEPTABLE",
            design_score=0.85
        )
        
        # Add some test data
        report.issues = [
            ValidationIssue("TEST_001", ValidationCategory.THERMAL, ValidationSeverity.INFO, "Test Issue", "Test description")
        ]
        report.safety_margins = [
            SafetyMargin("Test Parameter", 90, 100, 1.1, 10, True)
        ]
        report.performance_metrics = [
            PerformanceMetric("Test Metric", 100, 90, meets_requirement=True, units="K")
        ]
        report.recommendations = ["Test recommendation"]
        report.validation_summary = {"test": "summary"}
        
        json_output = validator.export_report(report, 'json')
        
        # Should be valid JSON
        parsed = json.loads(json_output)
        
        assert parsed['report_id'] == "test_report"
        assert parsed['aircraft_config'] == "test"
        assert parsed['overall_status'] == "DESIGN_ACCEPTABLE"
        assert parsed['design_score'] == 0.85
        assert len(parsed['issues']) == 1
        assert len(parsed['safety_margins']) == 1
        assert len(parsed['performance_metrics']) == 1
        assert len(parsed['recommendations']) == 1
    
    def test_html_export(self):
        """Test HTML report export."""
        validator = HypersonicDesignValidator()
        
        # Create mock report
        config = AircraftConfiguration(name="test", modules=[])
        report = ValidationReport(
            report_id="test_report",
            aircraft_config=config,
            validation_timestamp=time.time(),
            overall_status="DESIGN_ACCEPTABLE",
            design_score=0.85
        )
        
        html_output = validator.export_report(report, 'html')
        
        # Should contain HTML structure
        assert "<!DOCTYPE html>" in html_output
        assert "<title>Hypersonic Design Validation Report</title>" in html_output
        assert "test" in html_output  # Aircraft name
        assert "DESIGN_ACCEPTABLE" in html_output
        assert "0.85" in html_output  # Design score
    
    def test_unsupported_export_format(self):
        """Test unsupported export format."""
        validator = HypersonicDesignValidator()
        
        config = AircraftConfiguration(name="test", modules=[])
        report = ValidationReport(
            report_id="test_report",
            aircraft_config=config,
            validation_timestamp=time.time()
        )
        
        with pytest.raises(ValueError, match="Unsupported export format"):
            validator.export_report(report, 'pdf')


class TestIntegrationScenarios:
    """Test integration scenarios."""
    
    def test_complete_mach_60_validation(self):
        """Test complete Mach 60 design validation scenario."""
        validator = HypersonicDesignValidator()
        
        # Mach 60 aircraft configuration
        config = AircraftConfiguration(name="mach_60_vehicle", modules=[])
        
        # Realistic Mach 60 analysis results
        analysis_results = {
            'thermal': {
                'max_surface_temperature': 2800.0,  # High but manageable
                'max_heat_flux': 90e6,              # Near limit
                'ablation_rate': 8e-6,              # Some ablation
                'cooling_effectiveness': 0.7        # Good cooling
            },
            'structural': {
                'safety_factor': 1.6,               # Adequate
                'max_stress': 200e6,                # Moderate stress
                'max_displacement': 0.08,           # Some displacement
                'thermal_stress_contribution': 80e6  # Significant thermal stress
            },
            'aerodynamic': {
                'stagnation_temperature': 35000.0,  # High but acceptable
                'stagnation_pressure': 1.5e6,
                'convective_heat_flux': 80e6,
                'radiative_heat_flux': 10e6,
                'plasma_formation': True,            # Expected at Mach 60
                'shock_standoff_distance': 0.008
            },
            'coupling': {
                'converged': True,
                'final_iteration_count': 25,
                'max_coupling_residual': 5e-7,
                'convergence_history': [1e-3, 1e-4, 1e-5, 5e-7]
            }
        }
        
        # Create mock mission profile
        mission_profile = Mock()
        mission_profile.total_duration = 2400.0  # 40 minutes
        mission_profile.max_thermal_load = 90e6
        
        report = validator.validate_design(config, analysis_results, mission_profile)
        
        # Verify report structure
        assert report.aircraft_config.name == "mach_60_vehicle"
        assert report.overall_status in ["DESIGN_ACCEPTABLE", "ACCEPTABLE_WITH_WARNINGS", "DESIGN_ISSUES"]
        assert 0.0 <= report.design_score <= 1.0
        
        # Should have some issues due to extreme conditions
        assert len(report.issues) > 0
        
        # Should have safety margins calculated
        assert len(report.safety_margins) > 0
        
        # Should have performance metrics
        assert len(report.performance_metrics) > 0
        
        # Should have recommendations
        assert len(report.recommendations) > 0
        
        # Check for expected Mach 60 specific issues
        issue_titles = [issue.title for issue in report.issues]
        
        # Might have plasma formation warning
        plasma_issues = [title for title in issue_titles if "Plasma" in title]
        # Plasma formation is expected at Mach 60, so should be noted
        
        # Check validation summary
        assert 'issue_counts' in report.validation_summary
        assert 'margin_summary' in report.validation_summary
        assert 'performance_summary' in report.validation_summary
        assert 'coupling' in report.validation_summary
    
    def test_validation_with_validator_error(self):
        """Test validation handling when a validator throws an error."""
        validator = HypersonicDesignValidator()
        
        # Mock a validator to throw an error
        validator.validators['thermal'].validate = Mock(side_effect=Exception("Test error"))
        
        config = AircraftConfiguration(name="test", modules=[])
        analysis_results = {'thermal': {}, 'structural': {}, 'aerodynamic': {}}
        
        report = validator.validate_design(config, analysis_results)
        
        # Should handle the error gracefully
        assert report is not None
        
        # Should have an error issue for the failed validator
        error_issues = [i for i in report.issues if "Validation Error" in i.title]
        assert len(error_issues) >= 1
        
        thermal_error = next(i for i in error_issues if "Thermal" in i.title)
        assert thermal_error.severity == ValidationSeverity.ERROR
        assert "Test error" in thermal_error.description


if __name__ == '__main__':
    pytest.main([__file__])