"""Demonstration of hypersonic mission planning for Mach 60 flight."""

import numpy as np
from typing import List

from fighter_jet_sdk.engines.propulsion.combined_cycle_engine import (
    CombinedCycleEngine, CombinedCycleSpecification
)
from fighter_jet_sdk.core.hypersonic_mission_planner import (
    HypersonicMissionPlanner, ThermalConstraint, PropulsionConstraint,
    OptimizationObjective
)
from fighter_jet_sdk.core.thermal_constraint_manager import (
    CoolingSystemPerformance, integrate_thermal_constraints_with_mission_planner
)
from fighter_jet_sdk.common.enums import ExtremePropulsionType


def create_mach60_engine() -> CombinedCycleEngine:
    """Create a combined-cycle engine capable of Mach 60 flight."""
    engine_spec = CombinedCycleSpecification(
        engine_id="mach60_hypersonic_engine",
        name="Mach 60 Hypersonic Combined-Cycle Engine",
        engine_type=None,  # Will be set in __post_init__
        max_thrust_sea_level=800000.0,  # N
        max_thrust_altitude=600000.0,  # N
        design_altitude=70000.0,  # m
        design_mach=40.0,
        extreme_propulsion_type=ExtremePropulsionType.COMBINED_CYCLE_AIRBREATHING,
        transition_mach_number=15.0,
        rocket_specific_impulse=450.0,  # s
        air_breathing_specific_impulse=3500.0,  # s
        max_rocket_thrust=500000.0,  # N
        rocket_fuel_capacity=8000.0,  # kg
        air_breathing_fuel_capacity=15000.0,  # kg
        plasma_interaction_threshold=25.0,
        max_stagnation_temperature=60000.0,  # K
        dissociation_onset_temperature=4000.0  # K
    )
    
    return CombinedCycleEngine(engine_spec)


def define_thermal_constraints() -> ThermalConstraint:
    """Define thermal constraints for Mach 60 flight."""
    return ThermalConstraint(
        max_heat_flux=150e6,  # 150 MW/m² - extreme for Mach 60
        max_temperature=3500.0,  # K
        max_duration_at_peak=180.0,  # seconds
        cooling_system_capacity=80e6,  # W
        recovery_time_required=300.0  # seconds
    )


def define_propulsion_constraints() -> PropulsionConstraint:
    """Define propulsion system constraints."""
    return PropulsionConstraint(
        max_air_breathing_mach=18.0,
        min_rocket_altitude=45000.0,  # m
        fuel_capacity_air_breathing=15000.0,  # kg
        fuel_capacity_rocket=8000.0,  # kg
        max_continuous_operation_time=3600.0  # seconds
    )


def define_cooling_system() -> CoolingSystemPerformance:
    """Define advanced cooling system for Mach 60 flight."""
    return CoolingSystemPerformance(
        max_cooling_capacity=100e6,  # W
        power_consumption=8000.0,  # W
        coolant_flow_rate=20.0,  # kg/s
        effectiveness=0.85,
        response_time=3.0,  # seconds
        operating_altitude_range=(30000.0, 120000.0),  # m
        operating_mach_range=(5.0, 60.0)
    )


def plan_transcontinental_mach60_mission():
    """Plan a transcontinental Mach 60 mission."""
    print("=== Mach 60 Hypersonic Mission Planning Demo ===\n")
    
    # Create engine and mission planner
    print("1. Initializing Mach 60 hypersonic engine...")
    engine = create_mach60_engine()
    planner = HypersonicMissionPlanner(engine)
    
    # Define constraints
    print("2. Defining mission constraints...")
    thermal_constraints = define_thermal_constraints()
    propulsion_constraints = define_propulsion_constraints()
    cooling_system = define_cooling_system()
    
    # Integrate thermal constraint management
    print("3. Integrating thermal constraint management...")
    integrated_planner = integrate_thermal_constraints_with_mission_planner(
        planner, thermal_constraints, cooling_system
    )
    
    # Define mission parameters
    print("4. Defining mission parameters...")
    start_point = (0.0, 0.0, 20000.0)  # Launch point (x, y, altitude in meters)
    end_point = (4000000.0, 0.0, 20000.0)  # 4000 km transcontinental flight
    max_mach = 60.0
    
    optimization_objectives = [
        OptimizationObjective.MINIMIZE_THERMAL_LOAD,
        OptimizationObjective.MINIMIZE_FUEL,
        OptimizationObjective.MINIMIZE_TIME
    ]
    
    print(f"   Start: {start_point}")
    print(f"   End: {end_point}")
    print(f"   Distance: {(end_point[0] - start_point[0])/1000:.0f} km")
    print(f"   Maximum Mach: {max_mach}")
    print(f"   Optimization objectives: {[obj.value for obj in optimization_objectives]}")
    
    # Plan the mission
    print("\n5. Planning hypersonic mission profile...")
    try:
        mission_profile = integrated_planner.plan_mission(
            start_point, end_point, max_mach,
            thermal_constraints, propulsion_constraints,
            optimization_objectives
        )
        
        print("   ✓ Mission planning completed successfully!")
        
        # Display mission summary
        print("\n6. Mission Profile Summary:")
        print(f"   Profile ID: {mission_profile.profile_id}")
        print(f"   Total waypoints: {len(mission_profile.waypoints)}")
        print(f"   Total duration: {mission_profile.total_duration/3600:.2f} hours")
        print(f"   Total fuel consumption: {mission_profile.total_fuel_consumption:.0f} kg")
        print(f"   Maximum thermal load: {mission_profile.max_thermal_load/1e6:.1f} MW/m²")
        print(f"   Maximum Mach number: {np.max(mission_profile.mach_profile):.1f}")
        print(f"   Maximum altitude: {np.max(mission_profile.altitude_profile)/1000:.1f} km")
        
        # Analyze key waypoints
        print("\n7. Key Mission Waypoints:")
        for i, waypoint in enumerate(mission_profile.waypoints[:5]):  # Show first 5
            print(f"   Waypoint {i+1}: Mach {waypoint.mach_number:.1f}, "
                  f"Alt {waypoint.altitude/1000:.1f} km, "
                  f"Phase {waypoint.flight_phase.value}, "
                  f"Thermal {waypoint.thermal_load/1e6:.1f} MW/m²")
        
        if len(mission_profile.waypoints) > 5:
            print(f"   ... and {len(mission_profile.waypoints) - 5} more waypoints")
        
        # Monitor thermal constraints
        print("\n8. Monitoring thermal constraints...")
        thermal_monitoring = integrated_planner.monitor_mission_thermal_constraints(mission_profile)
        
        print(f"   Thermal violations: {len(thermal_monitoring['thermal_violations'])}")
        print(f"   Cooling activations: {len(thermal_monitoring['cooling_activations'])}")
        print(f"   Recovery maneuvers: {len(thermal_monitoring['recovery_maneuvers'])}")
        print(f"   Maximum thermal stress: {thermal_monitoring['max_thermal_stress']:.2f}")
        print(f"   Maximum material degradation: {thermal_monitoring['max_material_degradation']:.3f}")
        print(f"   Total cooling time: {thermal_monitoring['total_cooling_time']/60:.1f} minutes")
        
        # Analyze mission feasibility
        print("\n9. Mission Feasibility Analysis:")
        feasibility = integrated_planner.analyze_mission_feasibility(mission_profile)
        
        print(f"   Mission feasible: {'✓ YES' if feasibility['feasible'] else '✗ NO'}")
        
        if feasibility['critical_issues']:
            print("   Critical issues:")
            for issue in feasibility['critical_issues']:
                print(f"     - {issue}")
        
        if feasibility['warnings']:
            print("   Warnings:")
            for warning in feasibility['warnings']:
                print(f"     - {warning}")
        
        print("\n   Performance Metrics:")
        metrics = feasibility['performance_metrics']
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"     {key.replace('_', ' ').title()}: {value:.2f}")
            else:
                print(f"     {key.replace('_', ' ').title()}: {value}")
        
        if feasibility['recommendations']:
            print("\n   Recommendations:")
            for rec in feasibility['recommendations']:
                print(f"     - {rec}")
        
        # Demonstrate trajectory modification for thermal safety
        print("\n10. Demonstrating thermal safety modifications...")
        modified_profile = integrated_planner.modify_trajectory_for_thermal_safety(mission_profile)
        
        if len(modified_profile.waypoints) != len(mission_profile.waypoints):
            print("    ✓ Trajectory modified for thermal safety")
        else:
            print("    ✓ Original trajectory already thermally safe")
        
        print(f"    Modified profile thermal load: {modified_profile.max_thermal_load/1e6:.1f} MW/m²")
        
        return mission_profile, thermal_monitoring, feasibility
        
    except Exception as e:
        print(f"   ✗ Mission planning failed: {e}")
        return None, None, None


def demonstrate_altitude_optimization():
    """Demonstrate altitude optimization for different Mach numbers."""
    print("\n=== Altitude Optimization Demo ===\n")
    
    from fighter_jet_sdk.core.hypersonic_mission_planner import AltitudeOptimizer
    
    optimizer = AltitudeOptimizer()
    
    print("Optimizing altitude profiles for different Mach ranges:")
    
    test_cases = [
        (10.0, 25.0, "Moderate Hypersonic"),
        (25.0, 45.0, "High Hypersonic"),
        (45.0, 60.0, "Extreme Hypersonic")
    ]
    
    for mach_start, mach_end, description in test_cases:
        print(f"\n{description} ({mach_start:.0f} - {mach_end:.0f} Mach):")
        
        constraints = {
            'max_altitude': 100000.0,
            'thermal_max_heat_flux': 120e6
        }
        
        altitude_profile = optimizer.optimize_altitude_for_mach_range(
            mach_start, mach_end, 1000000.0, constraints  # 1000 km distance
        )
        
        print(f"  Altitude range: {np.min(altitude_profile)/1000:.1f} - {np.max(altitude_profile)/1000:.1f} km")
        print(f"  Profile points: {len(altitude_profile)}")
        
        # Show atmospheric conditions at key points
        temp_start, pressure_start, density_start = optimizer.get_atmospheric_conditions(altitude_profile[0])
        temp_end, pressure_end, density_end = optimizer.get_atmospheric_conditions(altitude_profile[-1])
        
        print(f"  Start conditions: T={temp_start:.1f}K, P={pressure_start:.1f}Pa, ρ={density_start:.4f}kg/m³")
        print(f"  End conditions: T={temp_end:.1f}K, P={pressure_end:.1f}Pa, ρ={density_end:.4f}kg/m³")


def demonstrate_cooling_system_performance():
    """Demonstrate cooling system performance analysis."""
    print("\n=== Cooling System Performance Demo ===\n")
    
    from fighter_jet_sdk.core.thermal_constraint_manager import (
        ThermalConstraintManager, CoolingSystemMode, ThermalState
    )
    
    # Create thermal manager
    thermal_constraints = define_thermal_constraints()
    cooling_system = define_cooling_system()
    thermal_manager = ThermalConstraintManager(thermal_constraints, cooling_system)
    
    print("Testing cooling system performance at different thermal loads:")
    
    test_thermal_loads = [50e6, 80e6, 120e6, 150e6, 200e6]  # W/m²
    cooling_modes = [
        CoolingSystemMode.PASSIVE,
        CoolingSystemMode.ACTIVE_LOW,
        CoolingSystemMode.ACTIVE_MEDIUM,
        CoolingSystemMode.ACTIVE_HIGH,
        CoolingSystemMode.EMERGENCY
    ]
    
    print(f"{'Thermal Load (MW/m²)':<20} {'Cooling Mode':<15} {'Effectiveness':<12} {'Heat Reduction':<15}")
    print("-" * 65)
    
    for thermal_load in test_thermal_loads:
        for mode in cooling_modes:
            effectiveness = thermal_manager.calculate_cooling_effectiveness(mode, thermal_load)
            heat_reduction = thermal_load * effectiveness
            
            print(f"{thermal_load/1e6:<20.1f} {mode.value:<15} {effectiveness:<12.2f} {heat_reduction/1e6:<15.1f}")
    
    print("\nTesting thermal state evolution:")
    
    # Simulate thermal state evolution
    initial_state = ThermalState(
        surface_temperature=1500.0,
        heat_flux=100e6,
        thermal_load_integral=0.0
    )
    
    print(f"Initial state: T={initial_state.surface_temperature:.0f}K, "
          f"Heat flux={initial_state.heat_flux/1e6:.1f}MW/m²")
    
    # Apply cooling
    cooled_state = thermal_manager.apply_cooling_effects(initial_state, 0.7)
    
    print(f"After cooling: T={cooled_state.surface_temperature:.0f}K, "
          f"Heat flux={cooled_state.heat_flux/1e6:.1f}MW/m²")
    print(f"Temperature reduction: {initial_state.surface_temperature - cooled_state.surface_temperature:.0f}K")
    print(f"Heat flux reduction: {(initial_state.heat_flux - cooled_state.heat_flux)/1e6:.1f}MW/m²")


def main():
    """Run the complete hypersonic mission planning demonstration."""
    print("Hypersonic Mission Planning for Mach 60 Flight")
    print("=" * 50)
    
    # Main mission planning demo
    mission_profile, thermal_monitoring, feasibility = plan_transcontinental_mach60_mission()
    
    if mission_profile:
        # Additional demonstrations
        demonstrate_altitude_optimization()
        demonstrate_cooling_system_performance()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("\nKey achievements demonstrated:")
        print("✓ Mach 60 hypersonic mission planning")
        print("✓ Thermal constraint management")
        print("✓ Real-time cooling system activation")
        print("✓ Trajectory optimization for thermal safety")
        print("✓ Mission feasibility analysis")
        print("✓ Altitude profile optimization")
        print("✓ Cooling system performance analysis")
        
        return True
    else:
        print("\nDemo failed - check engine and constraint configurations")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)