"""Tests for extreme hypersonic data models."""

import pytest
import numpy as np
from fighter_jet_sdk.common.data_models import (
    PlasmaConditions, CombinedCyclePerformance, ThermalProtectionSystem,
    HypersonicMissionProfile, AblativeLayer, CoolingChannel, InsulationLayer
)
from fighter_jet_sdk.common.enums import (
    PlasmaRegime, ExtremePropulsionType, ThermalProtectionType
)
from fighter_jet_sdk.common.hypersonic_validation import HypersonicDataValidator


class TestPlasmaConditions:
    """Test plasma conditions data structure."""
    
    def test_valid_plasma_conditions(self):
        """Test creation of valid plasma conditions."""
        plasma = PlasmaConditions(
            electron_density=1e20,  # m⁻³
            electron_temperature=10000,  # K
            ion_temperature=8000,  # K
            magnetic_field=np.array([0.1, 0.0, 0.0]),  # Tesla
            plasma_frequency=8.98e12,  # Hz (calculated for given density)
            debye_length=7.4e-6,  # m (calculated for given conditions)
            plasma_regime=PlasmaRegime.WEAKLY_IONIZED
        )
        
        errors = plasma.validate_plasma_conditions()
        assert len(errors) == 0, f"Valid plasma conditions should not have errors: {errors}"
    
    def test_invalid_plasma_conditions(self):
        """Test validation of invalid plasma conditions."""
        # Negative electron density
        plasma = PlasmaConditions(
            electron_density=-1e20,
            electron_temperature=10000,
            ion_temperature=8000,
            magnetic_field=np.array([0.1, 0.0, 0.0]),
            plasma_frequency=8.98e12,
            debye_length=7.4e-6
        )
        
        errors = plasma.validate_plasma_conditions()
        assert len(errors) > 0
        assert any("density must be positive" in error for error in errors)
    
    def test_plasma_beta_calculation(self):
        """Test plasma beta parameter calculation."""
        plasma = PlasmaConditions(
            electron_density=1e20,
            electron_temperature=10000,
            ion_temperature=8000,
            magnetic_field=np.array([0.1, 0.0, 0.0]),
            plasma_frequency=8.98e12,
            debye_length=7.4e-6
        )
        
        beta = plasma.calculate_plasma_beta()
        assert beta > 0, "Plasma beta should be positive"
        assert beta < 100, "Plasma beta should be reasonable"
    
    def test_magnetic_field_validation(self):
        """Test magnetic field vector validation."""
        with pytest.raises(ValueError, match="3D vector"):
            PlasmaConditions(
                electron_density=1e20,
                electron_temperature=10000,
                ion_temperature=8000,
                magnetic_field=np.array([0.1, 0.0]),  # Wrong dimension
                plasma_frequency=8.98e12,
                debye_length=7.4e-6
            )


class TestCombinedCyclePerformance:
    """Test combined-cycle performance data structure."""
    
    def test_valid_performance(self):
        """Test creation of valid combined-cycle performance."""
        performance = CombinedCyclePerformance(
            air_breathing_thrust=500000,  # N
            rocket_thrust=200000,  # N
            transition_mach=8.0,
            fuel_flow_air_breathing=50.0,  # kg/s
            fuel_flow_rocket=20.0,  # kg/s
            specific_impulse=2500,  # s
            propulsion_type=ExtremePropulsionType.COMBINED_CYCLE_AIRBREATHING
        )
        
        errors = performance.validate_performance()
        assert len(errors) == 0, f"Valid performance should not have errors: {errors}"
    
    def test_thrust_calculations(self):
        """Test thrust calculation methods."""
        performance = CombinedCyclePerformance(
            air_breathing_thrust=500000,
            rocket_thrust=200000,
            transition_mach=8.0,
            fuel_flow_air_breathing=50.0,
            fuel_flow_rocket=20.0,
            specific_impulse=2500
        )
        
        total_thrust = performance.calculate_total_thrust()
        assert total_thrust == 700000, "Total thrust calculation incorrect"
        
        twr = performance.calculate_thrust_to_weight_ratio(50000)  # 50 ton vehicle
        assert twr > 0, "Thrust-to-weight ratio should be positive"
        assert twr < 10, "Thrust-to-weight ratio should be reasonable"
    
    def test_invalid_performance(self):
        """Test validation of invalid performance data."""
        performance = CombinedCyclePerformance(
            air_breathing_thrust=-100000,  # Negative thrust
            rocket_thrust=200000,
            transition_mach=1.0,  # Too low for combined cycle
            fuel_flow_air_breathing=50.0,
            fuel_flow_rocket=20.0,
            specific_impulse=2500
        )
        
        errors = performance.validate_performance()
        assert len(errors) > 0
        assert any("cannot be negative" in error for error in errors)
        assert any("too low" in error for error in errors)


class TestThermalProtectionSystem:
    """Test thermal protection system data structure."""
    
    def test_valid_tps(self):
        """Test creation of valid thermal protection system."""
        ablative_layer = AblativeLayer(
            material_id="carbon_carbon",
            thickness=0.05,  # 5 cm
            ablation_rate=1e-6,  # m/s per MW/m²
            heat_of_ablation=2e6,  # J/kg
            char_layer_conductivity=0.5  # W/(m⋅K)
        )
        
        cooling_channel = CoolingChannel(
            channel_id="channel_1",
            diameter=0.005,  # 5 mm
            length=1.0,  # 1 m
            coolant_type="liquid_hydrogen",
            mass_flow_rate=0.1,  # kg/s
            inlet_temperature=20,  # K
            pressure_drop=1e5  # Pa
        )
        
        tps = ThermalProtectionSystem(
            ablative_layers=[ablative_layer],
            active_cooling_channels=[cooling_channel],
            protection_type=ThermalProtectionType.HYBRID_SYSTEM,
            max_heat_flux_capacity=1e8  # 100 MW/m²
        )
        
        errors = tps.validate_system()
        assert len(errors) == 0, f"Valid TPS should not have errors: {errors}"
    
    def test_thickness_calculation(self):
        """Test total thickness calculation."""
        ablative_layer = AblativeLayer(
            material_id="carbon_carbon",
            thickness=0.05,
            ablation_rate=1e-6,
            heat_of_ablation=2e6,
            char_layer_conductivity=0.5
        )
        
        insulation_layer = InsulationLayer(
            material_id="aerogel",
            thickness=0.02,
            thermal_conductivity=0.01,
            max_operating_temperature=2000
        )
        
        tps = ThermalProtectionSystem(
            ablative_layers=[ablative_layer],
            insulation_layers=[insulation_layer]
        )
        
        total_thickness = tps.calculate_total_thickness()
        assert total_thickness == 0.07, "Total thickness calculation incorrect"
    
    def test_mass_estimation(self):
        """Test mass per area estimation."""
        tps = ThermalProtectionSystem(total_thickness=0.1)
        
        mass = tps.estimate_mass_per_area(10.0)  # 10 m²
        assert mass > 0, "Mass estimation should be positive"
        
        with pytest.raises(ValueError, match="Area must be positive"):
            tps.estimate_mass_per_area(-1.0)


class TestHypersonicMissionProfile:
    """Test hypersonic mission profile data structure."""
    
    def test_valid_mission_profile(self):
        """Test creation of valid hypersonic mission profile."""
        profile = HypersonicMissionProfile(
            mission_name="Mach_60_Test",
            altitude_profile=np.array([50000, 60000, 70000, 60000, 50000]),  # m
            mach_profile=np.array([30, 45, 60, 45, 30]),
            thermal_load_profile=np.array([1e7, 5e7, 1e8, 5e7, 1e7]),  # W/m²
            propulsion_mode_schedule=["air_breathing", "combined", "rocket", "combined", "air_breathing"],
            cooling_system_schedule=[False, True, True, True, False],
            mission_duration=3600,  # 1 hour
            max_thermal_load=1e8
        )
        
        errors = profile.validate_profile()
        assert len(errors) == 0, f"Valid mission profile should not have errors: {errors}"
    
    def test_profile_statistics(self):
        """Test mission profile statistics calculation."""
        profile = HypersonicMissionProfile(
            mission_name="Test_Mission",
            altitude_profile=np.array([50000, 60000, 70000]),
            mach_profile=np.array([30, 60, 45]),
            thermal_load_profile=np.array([1e7, 1e8, 5e7])
        )
        
        stats = profile.calculate_profile_statistics()
        
        assert stats['avg_altitude'] == 60000
        assert stats['max_mach'] == 60
        assert stats['peak_thermal_load'] == 1e8
        assert 'mach_60_plus_time_fraction' in stats
    
    def test_plasma_modeling_requirement(self):
        """Test plasma modeling requirement detection."""
        # Profile with Mach > 25 should require plasma modeling
        profile = HypersonicMissionProfile(
            mach_profile=np.array([30, 60, 45])
        )
        assert profile.requires_plasma_modeling()
        
        # Profile with Mach < 25 should not require plasma modeling
        profile_low = HypersonicMissionProfile(
            mach_profile=np.array([10, 15, 20])
        )
        assert not profile_low.requires_plasma_modeling()
    
    def test_active_cooling_requirement(self):
        """Test active cooling requirement detection."""
        # High thermal loads should require active cooling
        profile = HypersonicMissionProfile(
            thermal_load_profile=np.array([1e7, 5e7, 1e8])
        )
        assert profile.requires_active_cooling()
        
        # Low thermal loads should not require active cooling
        profile_low = HypersonicMissionProfile(
            thermal_load_profile=np.array([1e5, 5e5, 1e6])
        )
        assert not profile_low.requires_active_cooling()
    
    def test_conditions_at_time(self):
        """Test getting conditions at specific time."""
        plasma_cond = PlasmaConditions(
            electron_density=1e20,
            electron_temperature=10000,
            ion_temperature=8000,
            magnetic_field=np.array([0.1, 0.0, 0.0]),
            plasma_frequency=8.98e12,
            debye_length=7.4e-6
        )
        
        profile = HypersonicMissionProfile(
            altitude_profile=np.array([50000, 60000]),
            mach_profile=np.array([30, 60]),
            thermal_load_profile=np.array([1e7, 1e8]),
            plasma_conditions_profile=[plasma_cond, plasma_cond]
        )
        
        conditions = profile.get_conditions_at_time(1)
        assert conditions['altitude'] == 60000
        assert conditions['mach'] == 60
        assert conditions['thermal_load'] == 1e8
        assert 'plasma_conditions' in conditions
        
        with pytest.raises(IndexError):
            profile.get_conditions_at_time(10)


class TestHypersonicDataValidator:
    """Test comprehensive hypersonic data validation."""
    
    def test_plasma_physics_validation(self):
        """Test physics-based plasma validation."""
        # Create plasma with physically consistent values
        electron_density = 1e20  # m⁻³
        electron_temperature = 10000  # K
        
        # Calculate consistent plasma frequency
        e_charge = 1.602176634e-19  # C
        e_mass = 9.1093837015e-31  # kg
        epsilon_0 = 8.8541878128e-12  # F/m
        plasma_frequency = np.sqrt(electron_density * e_charge**2 / (epsilon_0 * e_mass))
        
        # Calculate consistent Debye length
        k_b = 1.380649e-23  # J/K
        debye_length = np.sqrt(epsilon_0 * k_b * electron_temperature / (electron_density * e_charge**2))
        
        plasma = PlasmaConditions(
            electron_density=electron_density,
            electron_temperature=electron_temperature,
            ion_temperature=8000,
            magnetic_field=np.array([0.1, 0.0, 0.0]),
            plasma_frequency=plasma_frequency,
            debye_length=debye_length,
            plasma_regime=PlasmaRegime.WEAKLY_IONIZED
        )
        
        errors = HypersonicDataValidator.validate_plasma_conditions(plasma)
        # Should have minimal errors for consistent physics
        physics_errors = [e for e in errors if "inconsistent" in e.lower()]
        assert len(physics_errors) <= 1, f"Too many physics inconsistencies: {physics_errors}"
    
    def test_propulsion_physics_validation(self):
        """Test physics-based propulsion validation."""
        # Create performance with consistent thrust and fuel flow
        fuel_flow_total = 70.0  # kg/s
        specific_impulse = 2500  # s
        theoretical_thrust = fuel_flow_total * specific_impulse * 9.80665  # N
        
        performance = CombinedCyclePerformance(
            air_breathing_thrust=theoretical_thrust * 0.7,  # 70% air-breathing
            rocket_thrust=theoretical_thrust * 0.3,  # 30% rocket
            transition_mach=6.0,  # Within scramjet range
            fuel_flow_air_breathing=49.0,  # kg/s
            fuel_flow_rocket=21.0,  # kg/s
            specific_impulse=specific_impulse,
            propulsion_type=ExtremePropulsionType.DUAL_MODE_SCRAMJET
        )
        
        errors = HypersonicDataValidator.validate_combined_cycle_performance(performance)
        # Should validate transition Mach for scramjet
        thrust_errors = [e for e in errors if "inconsistent" in e.lower()]
        assert len(thrust_errors) <= 1, f"Thrust calculation should be consistent: {thrust_errors}"
    
    def test_system_integration_validation(self):
        """Test integration validation between systems."""
        plasma = PlasmaConditions(
            electron_density=1e20,
            electron_temperature=10000,
            ion_temperature=8000,
            magnetic_field=np.array([0.1, 0.0, 0.0]),
            plasma_frequency=8.98e12,
            debye_length=7.4e-6
        )
        
        performance = CombinedCyclePerformance(
            air_breathing_thrust=500000,
            rocket_thrust=200000,
            transition_mach=8.0,
            fuel_flow_air_breathing=50.0,
            fuel_flow_rocket=20.0,
            specific_impulse=2500,
            operating_altitude_range=(40000, 80000)
        )
        
        tps = ThermalProtectionSystem(
            max_heat_flux_capacity=1e8
        )
        
        profile = HypersonicMissionProfile(
            altitude_profile=np.array([50000, 60000, 70000]),
            mach_profile=np.array([30, 60, 45]),
            max_thermal_load=5e7  # Within TPS capacity
        )
        
        errors = HypersonicDataValidator.validate_system_integration(
            plasma, performance, tps, profile
        )
        
        # Should not have integration errors for compatible systems
        assert len(errors) == 0, f"Compatible systems should not have integration errors: {errors}"


if __name__ == "__main__":
    pytest.main([__file__])