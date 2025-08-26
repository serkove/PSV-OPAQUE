"""
Tests for Stability and Control Analysis Module

Comprehensive tests for stability analyzer functionality including
control authority calculations, handling qualities assessment,
and control system design.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from fighter_jet_sdk.engines.aerodynamics.stability_analyzer import (
    StabilityAnalyzer, ControlSurface, HandlingQuality, ControlAuthority,
    StabilityDerivatives, FlightCondition, HandlingQualityAssessment,
    ControlSystemDesign
)
from fighter_jet_sdk.common.data_models import (
    AircraftConfiguration, Module, BasePlatform, PhysicalProperties
)
from fighter_jet_sdk.common.enums import ModuleType
from fighter_jet_sdk.core.errors import AerodynamicsError


class TestStabilityAnalyzer:
    """Test suite for StabilityAnalyzer class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.analyzer = StabilityAnalyzer()
        self.test_configuration = self._create_test_configuration()
    
    def _create_test_configuration(self):
        """Create a test aircraft configuration"""
        # Create base platform
        base_platform = BasePlatform(
            name="Test Fighter Platform",
            base_mass=8000.0,
            power_generation_capacity=500000.0,
            fuel_capacity=3000.0
        )
        
        # Create wing module with ailerons
        wing_module = Module(
            name="main_wing",
            module_type=ModuleType.STRUCTURAL,
            physical_properties=PhysicalProperties(
                mass=1500.0,
                center_of_gravity=(0.0, 0.0, 0.0),
                moments_of_inertia=(1000.0, 2000.0, 2500.0),
                dimensions=(12.0, 8.0, 0.5)  # length, width, height
            ),
            performance_characteristics={"wing_area": 40.0}
        )
        
        # Create horizontal tail with elevator
        htail_module = Module(
            name="horizontal_tail",
            module_type=ModuleType.STRUCTURAL,
            physical_properties=PhysicalProperties(
                mass=300.0,
                center_of_gravity=(8.0, 0.0, 0.0),
                moments_of_inertia=(100.0, 200.0, 250.0),
                dimensions=(4.0, 3.0, 0.3)
            ),
            performance_characteristics={"tail_area": 8.0}
        )
        
        # Create vertical tail with rudder
        vtail_module = Module(
            name="vertical_tail",
            module_type=ModuleType.STRUCTURAL,
            physical_properties=PhysicalProperties(
                mass=200.0,
                center_of_gravity=(8.0, 0.0, 1.5),
                moments_of_inertia=(50.0, 100.0, 120.0),
                dimensions=(3.0, 0.3, 2.5)
            ),
            performance_characteristics={"tail_area": 6.0}
        )
        
        # Create engine module with thrust vectoring
        engine_module = Module(
            name="vectoring_engine",
            module_type=ModuleType.PROPULSION,
            physical_properties=PhysicalProperties(
                mass=1200.0,
                center_of_gravity=(6.0, 0.0, 0.0),
                moments_of_inertia=(200.0, 400.0, 500.0),
                dimensions=(3.0, 1.5, 1.5)
            ),
            performance_characteristics={
                "max_thrust": 150000.0,
                "thrust_vectoring": True
            }
        )
        
        # Create configuration
        configuration = AircraftConfiguration(
            name="Test Fighter Aircraft",
            base_platform=base_platform,
            modules=[wing_module, htail_module, vtail_module, engine_module]
        )
        
        return configuration
    
    def test_analyzer_initialization(self):
        """Test StabilityAnalyzer initialization"""
        analyzer = StabilityAnalyzer()
        
        assert analyzer is not None
        assert len(analyzer.standard_conditions) == 6
        assert ControlSurface.ELEVATOR in analyzer.control_effectiveness
        assert analyzer.control_effectiveness[ControlSurface.ELEVATOR] == 0.8
    
    def test_analyze_stability_comprehensive(self):
        """Test comprehensive stability analysis"""
        flight_conditions = {
            "conditions": [
                {
                    "altitude": 10000,
                    "mach_number": 0.8,
                    "angle_of_attack": 2.0,
                    "load_factor": 1.0,
                    "configuration": "cruise"
                }
            ]
        }
        
        results = self.analyzer.analyze_stability(self.test_configuration, flight_conditions)
        
        # Check that all required sections are present
        assert "control_authority" in results
        assert "handling_qualities" in results
        assert "control_systems" in results
        assert "stability_derivatives" in results
        assert "pilot_interface" in results
        assert "performance_metrics" in results
        assert "flight_envelope_limits" in results
        assert "recommendations" in results
    
    def test_calculate_control_authority(self):
        """Test control authority calculation"""
        control_authority = self.analyzer._calculate_control_authority(self.test_configuration)
        
        # Should have control surfaces for wing, tails, and thrust vectoring
        assert len(control_authority) >= 3
        
        # Check for expected control surfaces
        surface_names = list(control_authority.keys())
        assert any("aileron" in name for name in surface_names)
        assert any("elevator" in name for name in surface_names)
        assert any("rudder" in name for name in surface_names)
        assert any("thrust_vectoring" in name for name in surface_names)
        
        # Validate control authority properties
        for name, authority in control_authority.items():
            assert isinstance(authority, ControlAuthority)
            assert authority.max_deflection > 0
            assert authority.deflection_rate > 0
            assert authority.moment_arm > 0
            assert authority.effectiveness > 0
            assert authority.power_required >= 0
            assert authority.response_time > 0
    
    def test_identify_control_surfaces(self):
        """Test control surface identification"""
        # Test wing module
        wing_module = Module(name="main_wing", module_type=ModuleType.STRUCTURAL)
        surfaces = self.analyzer._identify_control_surfaces(wing_module)
        assert ControlSurface.AILERON in surfaces
        
        # Test horizontal tail module
        htail_module = Module(name="horizontal_tail", module_type=ModuleType.STRUCTURAL)
        surfaces = self.analyzer._identify_control_surfaces(htail_module)
        assert ControlSurface.ELEVATOR in surfaces
        
        # Test vertical tail module
        vtail_module = Module(name="vertical_tail", module_type=ModuleType.STRUCTURAL)
        surfaces = self.analyzer._identify_control_surfaces(vtail_module)
        assert ControlSurface.RUDDER in surfaces
        
        # Test canard module
        canard_module = Module(name="canard", module_type=ModuleType.STRUCTURAL)
        surfaces = self.analyzer._identify_control_surfaces(canard_module)
        assert ControlSurface.CANARD in surfaces
    
    def test_calculate_surface_authority(self):
        """Test individual surface authority calculation"""
        wing_module = self.test_configuration.modules[0]  # main_wing
        
        authority = self.analyzer._calculate_surface_authority(
            wing_module, ControlSurface.AILERON, self.test_configuration
        )
        
        assert isinstance(authority, ControlAuthority)
        assert authority.surface_type == ControlSurface.AILERON
        assert authority.max_deflection == 25.0
        assert authority.deflection_rate == 60.0
        assert authority.moment_arm > 0
        assert authority.effectiveness > 0
        assert authority.power_required > 0
        assert authority.response_time == 0.1
    
    def test_thrust_vectoring_detection(self):
        """Test thrust vectoring capability detection"""
        # Test engine with thrust vectoring
        tv_engine = Module(
            name="vectoring_engine",
            module_type=ModuleType.PROPULSION,
            performance_characteristics={"thrust_vectoring": True}
        )
        assert self.analyzer._has_thrust_vectoring(tv_engine)
        
        # Test engine without thrust vectoring
        normal_engine = Module(
            name="normal_engine",
            module_type=ModuleType.PROPULSION,
            performance_characteristics={"max_thrust": 100000}
        )
        assert not self.analyzer._has_thrust_vectoring(normal_engine)
        
        # Test engine with vectoring in name
        named_tv_engine = Module(
            name="thrust_vector_engine",
            module_type=ModuleType.PROPULSION
        )
        assert self.analyzer._has_thrust_vectoring(named_tv_engine)
    
    def test_calculate_thrust_vectoring_authority(self):
        """Test thrust vectoring authority calculation"""
        engine_module = self.test_configuration.modules[3]  # vectoring_engine
        
        authority = self.analyzer._calculate_thrust_vectoring_authority(
            engine_module, self.test_configuration
        )
        
        assert isinstance(authority, ControlAuthority)
        assert authority.surface_type == ControlSurface.THRUST_VECTORING
        assert authority.max_deflection == 15.0
        assert authority.deflection_rate == 30.0
        assert authority.moment_arm > 0
        assert authority.effectiveness > 0
        assert authority.power_required == 1000.0
        assert authority.response_time == 0.05
    
    def test_assess_handling_qualities(self):
        """Test handling qualities assessment"""
        conditions = [
            FlightCondition(10000, 0.8, 2.0, 1.0, "cruise"),
            FlightCondition(5000, 0.4, 15.0, 2.0, "combat")
        ]
        
        assessments = self.analyzer._assess_handling_qualities(self.test_configuration, conditions)
        
        assert len(assessments) == 2
        
        for assessment in assessments:
            assert isinstance(assessment, HandlingQualityAssessment)
            assert isinstance(assessment.cooper_harper_rating, HandlingQuality)
            assert assessment.short_period_frequency > 0
            assert assessment.short_period_damping > 0
            assert assessment.dutch_roll_frequency > 0
            assert assessment.dutch_roll_damping > 0
            assert assessment.spiral_mode_time_constant != 0
            assert assessment.roll_mode_time_constant > 0
            assert isinstance(assessment.comments, list)
    
    def test_calculate_short_period_characteristics(self):
        """Test short period dynamics calculation"""
        condition = FlightCondition(10000, 0.8, 2.0, 1.0, "cruise")
        
        freq, damping = self.analyzer._calculate_short_period_characteristics(
            self.test_configuration, condition
        )
        
        assert freq > 0
        assert 0 < damping < 2.0
    
    def test_calculate_dutch_roll_characteristics(self):
        """Test Dutch roll dynamics calculation"""
        condition = FlightCondition(10000, 0.8, 2.0, 1.0, "cruise")
        
        freq, damping = self.analyzer._calculate_dutch_roll_characteristics(
            self.test_configuration, condition
        )
        
        assert freq > 0
        assert damping > 0
    
    def test_determine_cooper_harper_rating(self):
        """Test Cooper-Harper rating determination"""
        # Test Level 1 conditions (good handling)
        rating = self.analyzer._determine_cooper_harper_rating(
            sp_freq=2.0, sp_damping=0.7, dr_freq=1.5, dr_damping=0.1,
            condition=FlightCondition(10000, 0.8, 2.0, 1.0, "cruise")
        )
        assert rating == HandlingQuality.LEVEL_1
        
        # Test Level 3 conditions (poor handling)
        rating = self.analyzer._determine_cooper_harper_rating(
            sp_freq=0.3, sp_damping=0.1, dr_freq=0.5, dr_damping=0.02,
            condition=FlightCondition(10000, 0.8, 2.0, 3.0, "combat")
        )
        assert rating == HandlingQuality.LEVEL_3
    
    def test_design_control_systems(self):
        """Test control system design"""
        # First calculate control authority
        control_authority = self.analyzer._calculate_control_authority(self.test_configuration)
        
        # Design control systems
        control_systems = self.analyzer._design_control_systems(
            self.test_configuration, control_authority
        )
        
        # Should have systems for pitch, roll, yaw, and thrust vectoring
        expected_systems = ["pitch", "roll", "yaw", "thrust_vectoring"]
        for system_name in expected_systems:
            if system_name in control_systems:
                system = control_systems[system_name]
                assert isinstance(system, ControlSystemDesign)
                assert system.controller_type in ["PID", "LQR"]
                assert len(system.gains) > 0
                assert system.bandwidth > 0
                assert system.phase_margin > 0
                assert system.gain_margin > 0
                assert system.settling_time > 0
                assert system.overshoot >= 0
    
    def test_design_pitch_controller(self):
        """Test pitch controller design"""
        control_authority = self.analyzer._calculate_control_authority(self.test_configuration)
        
        pitch_system = self.analyzer._design_pitch_controller(
            self.test_configuration, control_authority
        )
        
        assert isinstance(pitch_system, ControlSystemDesign)
        assert pitch_system.controller_type == "PID"
        assert "kp" in pitch_system.gains
        assert "ki" in pitch_system.gains
        assert "kd" in pitch_system.gains
        assert pitch_system.gains["kp"] > 0
        assert pitch_system.gains["ki"] > 0
        assert pitch_system.gains["kd"] > 0
    
    def test_design_thrust_vectoring_controller(self):
        """Test thrust vectoring controller design"""
        control_authority = self.analyzer._calculate_control_authority(self.test_configuration)
        
        tv_system = self.analyzer._design_thrust_vectoring_controller(
            self.test_configuration, control_authority
        )
        
        assert isinstance(tv_system, ControlSystemDesign)
        assert tv_system.controller_type == "LQR"
        assert "q_attitude" in tv_system.gains
        assert "q_rate" in tv_system.gains
        assert "r_control" in tv_system.gains
        assert tv_system.bandwidth > 10.0  # High bandwidth for thrust vectoring
    
    def test_calculate_stability_derivatives(self):
        """Test stability derivatives calculation"""
        derivatives = self.analyzer._calculate_stability_derivatives(self.test_configuration)
        
        assert isinstance(derivatives, StabilityDerivatives)
        
        # Check longitudinal derivatives
        assert derivatives.cma < 0  # Negative for stability
        assert derivatives.cmq < 0  # Negative for damping
        assert derivatives.cmde < 0  # Negative for elevator effectiveness
        
        # Check lateral-directional derivatives
        assert derivatives.cnb > 0  # Positive for directional stability
        assert derivatives.cnr < 0  # Negative for yaw damping
        assert derivatives.cndr < 0  # Negative for rudder effectiveness
        assert derivatives.clb < 0  # Negative for dihedral effect
        assert derivatives.clp < 0  # Negative for roll damping
        assert derivatives.clda > 0  # Positive for aileron effectiveness
    
    def test_prepare_pilot_interface(self):
        """Test pilot-in-the-loop interface preparation"""
        control_authority = self.analyzer._calculate_control_authority(self.test_configuration)
        
        interface_data = self.analyzer._prepare_pilot_interface(
            self.test_configuration, control_authority
        )
        
        # Check required interface sections
        required_sections = [
            "control_mapping", "display_parameters", "haptic_feedback",
            "flight_envelope_protection", "automation_modes"
        ]
        
        for section in required_sections:
            assert section in interface_data
        
        # Check control mapping
        control_mapping = interface_data["control_mapping"]
        if "pitch_stick" in control_mapping:
            assert "surface" in control_mapping["pitch_stick"]
            assert "gain" in control_mapping["pitch_stick"]
            assert "max_deflection" in control_mapping["pitch_stick"]
        
        # Check flight envelope protection
        envelope = interface_data["flight_envelope_protection"]
        assert "angle_of_attack_limit" in envelope
        assert "load_factor_limit" in envelope
        assert "airspeed_limit" in envelope
        assert envelope["angle_of_attack_limit"] > 0
        assert envelope["load_factor_limit"] > 0
    
    def test_calculate_performance_metrics(self):
        """Test performance metrics calculation"""
        control_authority = self.analyzer._calculate_control_authority(self.test_configuration)
        
        # Create mock handling qualities
        handling_qualities = [
            HandlingQualityAssessment(
                flight_condition=FlightCondition(10000, 0.8, 2.0, 1.0, "cruise"),
                cooper_harper_rating=HandlingQuality.LEVEL_1,
                short_period_frequency=2.0,
                short_period_damping=0.7,
                dutch_roll_frequency=1.5,
                dutch_roll_damping=0.1,
                spiral_mode_time_constant=20.0,
                roll_mode_time_constant=0.5
            )
        ]
        
        metrics = self.analyzer._calculate_performance_metrics(
            self.test_configuration, control_authority, handling_qualities
        )
        
        # Check required metrics
        required_metrics = [
            "average_cooper_harper_rating", "total_control_power",
            "average_response_time", "max_g_capability",
            "roll_rate_capability", "stall_speed"
        ]
        
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert metrics[metric] > 0
    
    def test_estimate_max_g_capability(self):
        """Test maximum g-load capability estimation"""
        control_authority = self.analyzer._calculate_control_authority(self.test_configuration)
        
        max_g = self.analyzer._estimate_max_g_capability(
            self.test_configuration, control_authority
        )
        
        assert isinstance(max_g, float)
        assert 5.0 <= max_g <= 12.0  # Reasonable range for fighter aircraft
    
    def test_estimate_roll_rate_capability(self):
        """Test roll rate capability estimation"""
        control_authority = self.analyzer._calculate_control_authority(self.test_configuration)
        
        roll_rate = self.analyzer._estimate_roll_rate_capability(
            self.test_configuration, control_authority
        )
        
        assert isinstance(roll_rate, float)
        assert 90.0 <= roll_rate <= 360.0  # Reasonable range for fighter aircraft
    
    def test_estimate_stall_speed(self):
        """Test stall speed estimation"""
        stall_speed = self.analyzer._estimate_stall_speed(self.test_configuration)
        
        assert isinstance(stall_speed, float)
        assert 50.0 <= stall_speed <= 150.0  # Reasonable range for fighter aircraft
    
    def test_calculate_flight_envelope_limits(self):
        """Test flight envelope limits calculation"""
        limits = self.analyzer._calculate_flight_envelope_limits(self.test_configuration)
        
        required_limits = [
            "max_mach", "max_altitude", "max_g_positive", "max_g_negative",
            "max_angle_of_attack", "max_sideslip", "corner_velocity",
            "design_dive_speed", "never_exceed_speed"
        ]
        
        for limit in required_limits:
            assert limit in limits
            assert isinstance(limits[limit], (int, float))
        
        # Check reasonable values
        assert limits["max_mach"] > 1.0
        assert limits["max_altitude"] > 10000
        assert limits["max_g_positive"] > 6.0
        assert limits["max_g_negative"] < 0
    
    def test_generate_recommendations(self):
        """Test recommendations generation"""
        # Create handling qualities with some poor ratings
        handling_qualities = [
            HandlingQualityAssessment(
                flight_condition=FlightCondition(10000, 0.8, 2.0, 1.0, "cruise"),
                cooper_harper_rating=HandlingQuality.LEVEL_3,  # Poor rating
                short_period_frequency=2.0,
                short_period_damping=0.2,  # Low damping
                dutch_roll_frequency=1.5,
                dutch_roll_damping=0.03,  # Very low damping
                spiral_mode_time_constant=20.0,
                roll_mode_time_constant=0.5
            )
        ]
        
        recommendations = self.analyzer._generate_recommendations(
            self.test_configuration, handling_qualities
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should recommend stability augmentation for Level 3 handling
        assert any("stability augmentation" in rec.lower() for rec in recommendations)
        
        # Should recommend damping augmentation for low damping
        assert any("damping" in rec.lower() for rec in recommendations)
    
    def test_parse_flight_conditions_default(self):
        """Test parsing flight conditions with defaults"""
        conditions = self.analyzer._parse_flight_conditions({})
        
        assert len(conditions) == 6  # Standard conditions
        assert all(isinstance(cond, FlightCondition) for cond in conditions)
    
    def test_parse_flight_conditions_custom(self):
        """Test parsing custom flight conditions"""
        custom_conditions = {
            "conditions": [
                {
                    "altitude": 15000,
                    "mach_number": 1.5,
                    "angle_of_attack": 5.0,
                    "load_factor": 2.0,
                    "configuration": "supersonic"
                }
            ]
        }
        
        conditions = self.analyzer._parse_flight_conditions(custom_conditions)
        
        assert len(conditions) == 1
        condition = conditions[0]
        assert condition.altitude == 15000
        assert condition.mach_number == 1.5
        assert condition.angle_of_attack == 5.0
        assert condition.load_factor == 2.0
        assert condition.configuration == "supersonic"
    
    def test_error_handling(self):
        """Test error handling in stability analysis"""
        # Test with invalid configuration
        invalid_config = AircraftConfiguration(name="Invalid")
        
        # Should not raise exception but may produce warnings
        try:
            results = self.analyzer.analyze_stability(invalid_config, {})
            # Should still return results even with minimal configuration
            assert "control_authority" in results
        except AerodynamicsError:
            # Acceptable to raise error for invalid configuration
            pass
    
    def test_estimate_surface_area(self):
        """Test control surface area estimation"""
        wing_module = self.test_configuration.modules[0]  # main_wing
        
        # Test aileron area estimation
        aileron_area = self.analyzer._estimate_surface_area(wing_module, ControlSurface.AILERON)
        assert aileron_area > 0
        
        # Test elevator area estimation
        htail_module = self.test_configuration.modules[1]  # horizontal_tail
        elevator_area = self.analyzer._estimate_surface_area(htail_module, ControlSurface.ELEVATOR)
        assert elevator_area > 0
        
        # Test rudder area estimation
        vtail_module = self.test_configuration.modules[2]  # vertical_tail
        rudder_area = self.analyzer._estimate_surface_area(vtail_module, ControlSurface.RUDDER)
        assert rudder_area > 0
    
    def test_static_margin_estimation(self):
        """Test static margin estimation"""
        static_margin = self.analyzer._estimate_static_margin(self.test_configuration)
        
        assert isinstance(static_margin, float)
        # Fighter aircraft typically have small positive or negative static margin
        assert -0.1 <= static_margin <= 0.1


class TestControlAuthority:
    """Test suite for ControlAuthority dataclass"""
    
    def test_control_authority_creation(self):
        """Test ControlAuthority object creation"""
        authority = ControlAuthority(
            surface_type=ControlSurface.ELEVATOR,
            max_deflection=25.0,
            deflection_rate=60.0,
            moment_arm=3.0,
            effectiveness=0.8,
            power_required=500.0,
            response_time=0.1
        )
        
        assert authority.surface_type == ControlSurface.ELEVATOR
        assert authority.max_deflection == 25.0
        assert authority.deflection_rate == 60.0
        assert authority.moment_arm == 3.0
        assert authority.effectiveness == 0.8
        assert authority.power_required == 500.0
        assert authority.response_time == 0.1


class TestStabilityDerivatives:
    """Test suite for StabilityDerivatives dataclass"""
    
    def test_stability_derivatives_creation(self):
        """Test StabilityDerivatives object creation"""
        derivatives = StabilityDerivatives(
            cma=-0.5, cmq=-8.0, cmde=-1.2,
            cnb=0.1, cnr=-0.3, cndr=-0.08,
            clb=-0.05, clp=-0.4, clda=0.15
        )
        
        # Check longitudinal derivatives
        assert derivatives.cma == -0.5
        assert derivatives.cmq == -8.0
        assert derivatives.cmde == -1.2
        
        # Check lateral-directional derivatives
        assert derivatives.cnb == 0.1
        assert derivatives.cnr == -0.3
        assert derivatives.cndr == -0.08
        assert derivatives.clb == -0.05
        assert derivatives.clp == -0.4
        assert derivatives.clda == 0.15


class TestFlightCondition:
    """Test suite for FlightCondition dataclass"""
    
    def test_flight_condition_creation(self):
        """Test FlightCondition object creation"""
        condition = FlightCondition(
            altitude=10000,
            mach_number=0.8,
            angle_of_attack=2.0,
            load_factor=1.0,
            configuration="cruise"
        )
        
        assert condition.altitude == 10000
        assert condition.mach_number == 0.8
        assert condition.angle_of_attack == 2.0
        assert condition.load_factor == 1.0
        assert condition.configuration == "cruise"


class TestHandlingQualityAssessment:
    """Test suite for HandlingQualityAssessment dataclass"""
    
    def test_handling_quality_assessment_creation(self):
        """Test HandlingQualityAssessment object creation"""
        condition = FlightCondition(10000, 0.8, 2.0, 1.0, "cruise")
        
        assessment = HandlingQualityAssessment(
            flight_condition=condition,
            cooper_harper_rating=HandlingQuality.LEVEL_1,
            short_period_frequency=2.0,
            short_period_damping=0.7,
            dutch_roll_frequency=1.5,
            dutch_roll_damping=0.1,
            spiral_mode_time_constant=20.0,
            roll_mode_time_constant=0.5,
            comments=["Good handling characteristics"]
        )
        
        assert assessment.flight_condition == condition
        assert assessment.cooper_harper_rating == HandlingQuality.LEVEL_1
        assert assessment.short_period_frequency == 2.0
        assert assessment.short_period_damping == 0.7
        assert assessment.dutch_roll_frequency == 1.5
        assert assessment.dutch_roll_damping == 0.1
        assert assessment.spiral_mode_time_constant == 20.0
        assert assessment.roll_mode_time_constant == 0.5
        assert len(assessment.comments) == 1


class TestControlSystemDesign:
    """Test suite for ControlSystemDesign dataclass"""
    
    def test_control_system_design_creation(self):
        """Test ControlSystemDesign object creation"""
        design = ControlSystemDesign(
            controller_type="PID",
            gains={"kp": 2.0, "ki": 0.2, "kd": 0.1},
            bandwidth=5.0,
            phase_margin=45.0,
            gain_margin=6.0,
            settling_time=1.0,
            overshoot=5.0
        )
        
        assert design.controller_type == "PID"
        assert design.gains["kp"] == 2.0
        assert design.gains["ki"] == 0.2
        assert design.gains["kd"] == 0.1
        assert design.bandwidth == 5.0
        assert design.phase_margin == 45.0
        assert design.gain_margin == 6.0
        assert design.settling_time == 1.0
        assert design.overshoot == 5.0


if __name__ == "__main__":
    pytest.main([__file__])