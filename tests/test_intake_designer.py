"""Tests for Supersonic Intake Designer."""

import pytest
import math
from fighter_jet_sdk.engines.propulsion.intake_designer import (
    IntakeDesigner, IntakeGeometry, FlowConditions, IntakeType,
    ShockWaveData, IntakePerformance
)


class TestIntakeDesigner:
    """Test cases for IntakeDesigner."""
    
    @pytest.fixture
    def intake_designer(self):
        """Create intake designer for testing."""
        return IntakeDesigner()
    
    @pytest.fixture
    def supersonic_conditions(self):
        """Create supersonic flow conditions for testing."""
        return FlowConditions(
            mach_number=2.0,
            altitude=10000.0,
            angle_of_attack=0.0,
            atmospheric_pressure=26500.0,  # Pa at 10km
            atmospheric_temperature=223.15  # K at 10km
        )
    
    @pytest.fixture
    def hypersonic_conditions(self):
        """Create hypersonic flow conditions for testing."""
        return FlowConditions(
            mach_number=4.0,
            altitude=20000.0,
            angle_of_attack=2.0,
            atmospheric_pressure=5474.0,   # Pa at 20km
            atmospheric_temperature=216.65  # K at 20km
        )
    
    def test_initialization(self, intake_designer):
        """Test intake designer initialization."""
        assert intake_designer.gamma == 1.4
        assert intake_designer.gas_constant == 287.0
        assert intake_designer.max_compression_ratio == 50.0
        assert intake_designer.target_pressure_recovery == 0.95
    
    def test_pitot_intake_design(self, intake_designer, supersonic_conditions):
        """Test pitot intake design."""
        engine_mass_flow = 50.0  # kg/s
        
        geometry = intake_designer.design_intake(
            supersonic_conditions, engine_mass_flow, IntakeType.PITOT
        )
        
        assert geometry.intake_type == IntakeType.PITOT
        assert geometry.capture_area > 0
        assert geometry.throat_area > 0
        assert geometry.throat_area < geometry.capture_area
        assert geometry.diffuser_area_ratio >= 1.0
        assert geometry.compression_ratio > 1.0
        assert len(geometry.ramp_angles) == 0  # Pitot has no ramps
    
    def test_external_compression_design(self, intake_designer, supersonic_conditions):
        """Test external compression intake design."""
        engine_mass_flow = 75.0  # kg/s
        
        geometry = intake_designer.design_intake(
            supersonic_conditions, engine_mass_flow, IntakeType.EXTERNAL_COMPRESSION
        )
        
        assert geometry.intake_type == IntakeType.EXTERNAL_COMPRESSION
        assert geometry.capture_area > 0
        assert geometry.throat_area < geometry.capture_area
        assert len(geometry.ramp_angles) > 0
        assert all(0 < angle < 30 for angle in geometry.ramp_angles)
        assert geometry.compression_ratio > 1.0
    
    def test_mixed_compression_design(self, intake_designer, hypersonic_conditions):
        """Test mixed compression intake design."""
        engine_mass_flow = 100.0  # kg/s
        
        geometry = intake_designer.design_intake(
            hypersonic_conditions, engine_mass_flow, IntakeType.MIXED_COMPRESSION
        )
        
        assert geometry.intake_type == IntakeType.MIXED_COMPRESSION
        assert geometry.capture_area > 0
        assert geometry.throat_area < geometry.capture_area
        assert len(geometry.ramp_angles) >= 2  # Multiple shocks
        assert geometry.diffuser_area_ratio > 2.0  # Aggressive diffusion
        assert geometry.compression_ratio > 3.0  # Reasonable compression for Mach 4
    
    def test_variable_geometry_design(self, intake_designer, supersonic_conditions):
        """Test variable geometry intake design."""
        engine_mass_flow = 60.0  # kg/s
        
        geometry = intake_designer.design_intake(
            supersonic_conditions, engine_mass_flow, IntakeType.VARIABLE_GEOMETRY
        )
        
        assert geometry.intake_type == IntakeType.VARIABLE_GEOMETRY
        assert geometry.variable_geometry_range is not None
        assert len(geometry.variable_geometry_range) == 2
        assert geometry.variable_geometry_range[0] < geometry.variable_geometry_range[1]
        assert len(geometry.ramp_angles) > 0
    
    def test_diverterless_design(self, intake_designer, supersonic_conditions):
        """Test diverterless supersonic intake design."""
        engine_mass_flow = 55.0  # kg/s
        
        geometry = intake_designer.design_intake(
            supersonic_conditions, engine_mass_flow, IntakeType.DIVERTERLESS
        )
        
        assert geometry.intake_type == IntakeType.DIVERTERLESS
        assert geometry.capture_area > 0
        assert len(geometry.ramp_angles) == 1  # Single equivalent compression
        assert geometry.throat_mach < 0.7  # Conservative throat Mach
    
    def test_shock_wave_analysis(self, intake_designer, supersonic_conditions):
        """Test shock wave system analysis."""
        # Create test geometry with two oblique shocks
        geometry = IntakeGeometry(
            intake_type=IntakeType.EXTERNAL_COMPRESSION,
            capture_area=1.0,
            throat_area=0.8,
            diffuser_area_ratio=2.0,
            compression_ratio=4.0,
            ramp_angles=[12.0, 8.0],
            throat_mach=0.7
        )
        
        shock_data = intake_designer.analyze_shock_system(geometry, supersonic_conditions)
        
        assert isinstance(shock_data, ShockWaveData)
        assert len(shock_data.shock_angles) >= len(geometry.ramp_angles)
        assert len(shock_data.pressure_ratios) >= len(geometry.ramp_angles)
        assert len(shock_data.mach_numbers) == len(shock_data.pressure_ratios) + 1
        assert shock_data.total_pressure_recovery > 0
        assert shock_data.total_pressure_recovery <= 1.0
        assert shock_data.entropy_increase >= 0
        
        # Check that final Mach number is lower than initial (overall deceleration)
        assert shock_data.mach_numbers[-1] < shock_data.mach_numbers[0]
    
    def test_performance_calculation(self, intake_designer, supersonic_conditions):
        """Test intake performance calculation."""
        geometry = IntakeGeometry(
            intake_type=IntakeType.EXTERNAL_COMPRESSION,
            capture_area=1.2,
            throat_area=1.0,
            diffuser_area_ratio=2.5,
            compression_ratio=6.0,
            ramp_angles=[15.0],
            throat_mach=0.6
        )
        
        performance = intake_designer.calculate_performance(geometry, supersonic_conditions)
        
        assert isinstance(performance, IntakePerformance)
        assert 0 < performance.pressure_recovery <= 1.0
        assert 0 < performance.mass_flow_ratio <= 1.0
        assert 0 <= performance.distortion_coefficient < 1.0
        assert performance.drag_coefficient >= 0
        assert performance.spillage_drag >= 0
        assert performance.additive_drag >= 0
        assert 0 < performance.efficiency <= 1.0
    
    def test_normal_shock_calculations(self, intake_designer):
        """Test normal shock wave calculations."""
        # Test at Mach 2.0
        mach = 2.0
        
        p_ratio = intake_designer._normal_shock_pressure_ratio(mach)
        t_ratio = intake_designer._normal_shock_temperature_ratio(mach)
        pt_ratio = intake_designer._normal_shock_total_pressure_ratio(mach)
        mach_post = intake_designer._post_normal_shock_mach(mach)
        
        # Check physical validity
        assert p_ratio > 1.0  # Pressure increases
        assert t_ratio > 1.0  # Temperature increases
        assert pt_ratio < 1.0  # Total pressure decreases (irreversible)
        assert mach_post < 1.0  # Becomes subsonic
        
        # Check known values for Mach 2.0
        assert abs(p_ratio - 4.5) < 0.1  # Approximately 4.5
        assert abs(mach_post - 0.577) < 0.01  # Approximately 0.577
    
    def test_oblique_shock_calculations(self, intake_designer):
        """Test oblique shock wave calculations."""
        mach = 2.5
        ramp_angle = 10.0
        
        shock_angle = intake_designer._calculate_shock_angle(mach, ramp_angle)
        
        assert shock_angle is not None
        assert shock_angle > ramp_angle  # Shock angle > ramp angle
        assert shock_angle < 90.0  # Less than normal shock
        
        # Calculate post-shock properties
        p_ratio = intake_designer._oblique_shock_pressure_ratio(mach, shock_angle)
        mach_post = intake_designer._post_shock_mach(mach, shock_angle, ramp_angle)
        
        assert p_ratio > 1.0
        assert mach_post < mach  # Mach number decreases
        assert mach_post > 1.0  # Still supersonic for weak shock
    
    def test_optimization(self, intake_designer, supersonic_conditions):
        """Test intake geometry optimization."""
        engine_mass_flow = 70.0  # kg/s
        
        # Optimize for efficiency
        optimized_geometry = intake_designer.optimize_intake_geometry(
            supersonic_conditions, engine_mass_flow, 
            IntakeType.EXTERNAL_COMPRESSION, "efficiency"
        )
        
        assert optimized_geometry.intake_type == IntakeType.EXTERNAL_COMPRESSION
        
        # Calculate performance of optimized design
        performance = intake_designer.calculate_performance(
            optimized_geometry, supersonic_conditions
        )
        
        assert performance.efficiency > 0.7  # Should be reasonably efficient
        
        # Optimize for pressure recovery
        pr_optimized = intake_designer.optimize_intake_geometry(
            supersonic_conditions, engine_mass_flow,
            IntakeType.MIXED_COMPRESSION, "pressure_recovery"
        )
        
        pr_performance = intake_designer.calculate_performance(
            pr_optimized, supersonic_conditions
        )
        
        assert pr_performance.pressure_recovery > 0.8
    
    def test_capture_area_calculation(self, intake_designer, supersonic_conditions):
        """Test capture area calculation."""
        engine_mass_flow = 50.0  # kg/s
        
        capture_area = intake_designer._calculate_capture_area(
            supersonic_conditions, engine_mass_flow
        )
        
        assert capture_area > 0
        
        # Check scaling with mass flow
        double_mass_flow = 100.0  # kg/s
        double_area = intake_designer._calculate_capture_area(
            supersonic_conditions, double_mass_flow
        )
        
        assert abs(double_area / capture_area - 2.0) < 0.1  # Should be roughly double
    
    def test_reynolds_number_estimation(self, intake_designer, supersonic_conditions):
        """Test Reynolds number estimation."""
        geometry = IntakeGeometry(
            intake_type=IntakeType.PITOT,
            capture_area=1.0,
            throat_area=0.8,
            diffuser_area_ratio=1.5,
            compression_ratio=3.0
        )
        
        reynolds = intake_designer._estimate_reynolds_number(geometry, supersonic_conditions)
        
        assert reynolds > 1e6  # Should be high Reynolds number for aircraft
        assert reynolds < 1e9  # But not unreasonably high
    
    def test_validation(self, intake_designer, supersonic_conditions):
        """Test intake geometry validation."""
        # Valid geometry
        valid_geometry = IntakeGeometry(
            intake_type=IntakeType.EXTERNAL_COMPRESSION,
            capture_area=1.0,
            throat_area=0.8,
            diffuser_area_ratio=2.0,
            compression_ratio=5.0,
            ramp_angles=[12.0],
            throat_mach=0.7
        )
        
        errors = intake_designer._validate_intake_geometry(valid_geometry, supersonic_conditions)
        assert len(errors) == 0
        
        # Invalid geometry - throat larger than capture
        invalid_geometry = IntakeGeometry(
            intake_type=IntakeType.PITOT,
            capture_area=1.0,
            throat_area=1.2,  # Invalid: larger than capture
            diffuser_area_ratio=0.8,  # Invalid: < 1.0
            compression_ratio=100.0,  # Invalid: too high
            ramp_angles=[45.0],  # Invalid: too steep
            throat_mach=1.5  # Invalid: > 1.0
        )
        
        errors = intake_designer._validate_intake_geometry(invalid_geometry, supersonic_conditions)
        assert len(errors) > 0
        assert any("throat area" in error.lower() for error in errors)
        assert any("diffuser" in error.lower() for error in errors)
        assert any("compression ratio" in error.lower() for error in errors)
    
    def test_subsonic_conditions(self, intake_designer):
        """Test behavior with subsonic conditions."""
        subsonic_conditions = FlowConditions(
            mach_number=0.8,
            altitude=5000.0,
            atmospheric_pressure=54048.0,
            atmospheric_temperature=255.65
        )
        
        # Should handle subsonic gracefully
        geometry = intake_designer.design_intake(
            subsonic_conditions, 30.0, IntakeType.PITOT
        )
        
        assert geometry.compression_ratio == 1.0  # No compression for subsonic
        
        # Shock calculations should return None or handle gracefully
        shock_angle = intake_designer._calculate_shock_angle(0.8, 10.0)
        assert shock_angle is None
    
    def test_edge_cases(self, intake_designer, supersonic_conditions):
        """Test edge cases and boundary conditions."""
        # Very high Mach number
        high_mach_conditions = FlowConditions(
            mach_number=6.0,
            altitude=25000.0,
            atmospheric_pressure=2549.0,
            atmospheric_temperature=221.65
        )
        
        geometry = intake_designer.design_intake(
            high_mach_conditions, 80.0, IntakeType.MIXED_COMPRESSION
        )
        
        assert geometry.compression_ratio > 5.0  # High compression for hypersonic
        assert len(geometry.ramp_angles) >= 2  # Multiple shocks needed
        
        # Very small mass flow
        small_geometry = intake_designer.design_intake(
            supersonic_conditions, 1.0, IntakeType.PITOT
        )
        
        assert small_geometry.capture_area > 0
        assert small_geometry.capture_area < 0.1  # Should be small
        
        # Large mass flow
        large_geometry = intake_designer.design_intake(
            supersonic_conditions, 200.0, IntakeType.MIXED_COMPRESSION
        )
        
        assert large_geometry.capture_area > 0.5  # Should be larger than small geometry
    
    def test_angle_of_attack_effects(self, intake_designer):
        """Test effects of angle of attack on performance."""
        base_conditions = FlowConditions(
            mach_number=2.0,
            altitude=10000.0,
            angle_of_attack=0.0,
            atmospheric_pressure=26500.0,
            atmospheric_temperature=223.15
        )
        
        aoa_conditions = FlowConditions(
            mach_number=2.0,
            altitude=10000.0,
            angle_of_attack=5.0,  # 5 degrees AoA
            atmospheric_pressure=26500.0,
            atmospheric_temperature=223.15
        )
        
        geometry = IntakeGeometry(
            intake_type=IntakeType.EXTERNAL_COMPRESSION,
            capture_area=1.0,
            throat_area=0.8,
            diffuser_area_ratio=2.0,
            compression_ratio=4.0,
            ramp_angles=[12.0]
        )
        
        base_performance = intake_designer.calculate_performance(geometry, base_conditions)
        aoa_performance = intake_designer.calculate_performance(geometry, aoa_conditions)
        
        # Angle of attack should increase distortion
        assert aoa_performance.distortion_coefficient > base_performance.distortion_coefficient
        
        # Overall efficiency should decrease
        assert aoa_performance.efficiency < base_performance.efficiency
    
    def test_compression_ratio_scaling(self, intake_designer, supersonic_conditions):
        """Test compression ratio scaling with Mach number."""
        mach_numbers = [1.5, 2.0, 2.5, 3.0]
        compression_ratios = []
        
        for mach in mach_numbers:
            conditions = FlowConditions(
                mach_number=mach,
                altitude=supersonic_conditions.altitude,
                atmospheric_pressure=supersonic_conditions.atmospheric_pressure,
                atmospheric_temperature=supersonic_conditions.atmospheric_temperature
            )
            
            geometry = intake_designer.design_intake(
                conditions, 50.0, IntakeType.EXTERNAL_COMPRESSION
            )
            
            compression_ratios.append(geometry.compression_ratio)
        
        # Compression ratio should generally increase with Mach number
        # (allowing for some variation due to optimization differences)
        assert compression_ratios[-1] > compression_ratios[0]  # Last should be higher than first


if __name__ == "__main__":
    pytest.main([__file__])