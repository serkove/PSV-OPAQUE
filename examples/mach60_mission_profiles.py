#!/usr/bin/env python3
"""
Mach 60 Mission Profiles and Optimization Examples

This module provides various mission profiles and optimization scenarios
for Mach 60 hypersonic vehicles, demonstrating different operational
concepts and performance trade-offs.

Requirements: 7.5
"""

import json
import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path

from fighter_jet_sdk.core.hypersonic_mission_planner import HypersonicMissionPlanner
from fighter_jet_sdk.core.config import get_config_manager
from fighter_jet_sdk.common.data_models import MissionProfile, FlightConditions


class MissionProfileLibrary:
    """Library of predefined mission profiles for Mach 60 vehicles."""
    
    @staticmethod
    def global_strike_mission() -> Dict[str, Any]:
        """Global strike mission: 12,000 km range in 45 minutes."""
        return {
            'name': 'Global Strike Mission',
            'description': 'Long-range precision strike capability',
            'mission_type': 'strike',
            'range': 12000000,  # 12,000 km
            'duration': 2700,   # 45 minutes
            'payload': {
                'mass': 2000,   # kg
                'type': 'precision_munitions',
                'delivery_accuracy': 10  # meters CEP
            },
            'flight_profile': {
                'cruise_altitude': 85000,  # m
                'cruise_mach': 60.0,
                'ingress_altitude': 30000,  # m for terminal phase
                'egress_required': False
            },
            'constraints': {
                'max_thermal_load': 150e6,  # W/m²
                'max_g_load': 8.0,
                'stealth_required': True,
                'communication_blackout_acceptable': True
            },
            'success_criteria': {
                'range_achievement': 0.95,  # 95% of target range
                'time_on_target': 300,      # ±5 minutes
                'payload_delivery': True
            }
        }
    
    @staticmethod
    def reconnaissance_mission() -> Dict[str, Any]:
        """High-speed reconnaissance mission with return capability."""
        return {
            'name': 'Strategic Reconnaissance',
            'description': 'High-speed intelligence gathering with return',
            'mission_type': 'reconnaissance',
            'range': 8000000,   # 8,000 km (4,000 km each way)
            'duration': 3600,   # 1 hour
            'payload': {
                'mass': 1500,   # kg
                'type': 'sensor_package',
                'sensors': ['hyperspectral_imaging', 'synthetic_aperture_radar', 'signals_intelligence']
            },
            'flight_profile': {
                'cruise_altitude': 90000,  # m
                'cruise_mach': 60.0,
                'loiter_altitude': 70000,  # m for data collection
                'loiter_duration': 600,    # 10 minutes
                'return_required': True
            },
            'constraints': {
                'max_thermal_load': 120e6,  # W/m² (reduced for return flight)
                'max_g_load': 6.0,
                'stealth_required': True,
                'communication_required': True,  # For data transmission
                'fuel_reserve': 0.15  # 15% fuel reserve for return
            },
            'success_criteria': {
                'target_area_coverage': 0.90,
                'data_quality': 'high',
                'safe_return': True
            }
        }
    
    @staticmethod
    def space_access_mission() -> Dict[str, Any]:
        """Space access mission using hypersonic vehicle as first stage."""
        return {
            'name': 'Space Access Mission',
            'description': 'Hypersonic first stage for space launch',
            'mission_type': 'space_access',
            'range': 2000000,   # 2,000 km downrange
            'duration': 1800,   # 30 minutes to separation
            'payload': {
                'mass': 5000,   # kg (upper stage + payload)
                'type': 'space_vehicle',
                'target_orbit': 'LEO_400km'
            },
            'flight_profile': {
                'separation_altitude': 100000,  # m (100 km)
                'separation_mach': 60.0,
                'climb_rate': 50,  # m/s average
                'trajectory_angle': 30  # degrees at separation
            },
            'constraints': {
                'max_thermal_load': 200e6,  # W/m² (higher for space mission)
                'max_g_load': 10.0,
                'structural_margin': 2.0,  # Higher safety factor
                'payload_protection': True
            },
            'success_criteria': {
                'separation_conditions_met': True,
                'payload_integrity': True,
                'trajectory_accuracy': 0.99
            }
        }
    
    @staticmethod
    def research_mission() -> Dict[str, Any]:
        """Scientific research mission for hypersonic flight studies."""
        return {
            'name': 'Hypersonic Research Mission',
            'description': 'Scientific data collection at Mach 60',
            'mission_type': 'research',
            'range': 5000000,   # 5,000 km
            'duration': 2400,   # 40 minutes
            'payload': {
                'mass': 1000,   # kg
                'type': 'research_instruments',
                'instruments': [
                    'plasma_diagnostics',
                    'heat_flux_sensors',
                    'pressure_measurements',
                    'atmospheric_sampling'
                ]
            },
            'flight_profile': {
                'test_altitudes': [60000, 80000, 100000],  # m
                'test_duration_per_altitude': 300,  # 5 minutes each
                'data_collection_mach': [50, 55, 60],
                'return_required': True
            },
            'constraints': {
                'max_thermal_load': 180e6,  # W/m²
                'max_g_load': 5.0,  # Gentle for instruments
                'data_transmission_required': True,
                'instrument_protection': True
            },
            'success_criteria': {
                'data_collection_completeness': 0.95,
                'instrument_survival': True,
                'flight_envelope_coverage': True
            }
        }
    
    @staticmethod
    def intercept_mission() -> Dict[str, Any]:
        """High-speed intercept mission for defensive operations."""
        return {
            'name': 'Hypersonic Intercept Mission',
            'description': 'Intercept of high-speed threats',
            'mission_type': 'intercept',
            'range': 3000000,   # 3,000 km
            'duration': 900,    # 15 minutes (rapid response)
            'payload': {
                'mass': 800,    # kg
                'type': 'interceptor_package',
                'intercept_capability': 'kinetic_kill'
            },
            'flight_profile': {
                'intercept_altitude': 75000,  # m
                'intercept_mach': 60.0,
                'approach_angle': 45,  # degrees
                'terminal_guidance': True
            },
            'constraints': {
                'max_thermal_load': 160e6,  # W/m²
                'max_g_load': 12.0,  # High for intercept maneuvers
                'response_time': 600,  # 10 minutes from alert
                'guidance_accuracy': 1.0  # 1 meter accuracy
            },
            'success_criteria': {
                'intercept_probability': 0.90,
                'response_time_met': True,
                'guidance_performance': True
            }
        }


class MissionOptimizer:
    """Mission profile optimization for various objectives."""
    
    def __init__(self):
        """Initialize the mission optimizer."""
        self.config_manager = get_config_manager()
        self.mission_planner = HypersonicMissionPlanner(self.config_manager.get_config())
    
    def optimize_for_range(self, base_mission: Dict[str, Any], 
                          vehicle_config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize mission profile for maximum range."""
        print(f"Optimizing {base_mission['name']} for maximum range")
        
        # Define optimization parameters
        altitude_options = np.linspace(70000, 110000, 9)  # 70-110 km
        mach_options = np.linspace(55, 65, 6)  # Mach 55-65
        
        best_range = 0
        best_config = None
        optimization_results = []
        
        for altitude in altitude_options:
            for mach in mach_options:
                # Create mission variant
                mission_variant = base_mission.copy()
                mission_variant['flight_profile']['cruise_altitude'] = altitude
                mission_variant['flight_profile']['cruise_mach'] = mach
                
                try:
                    # Plan mission with these parameters
                    mission_result = self.mission_planner.plan_hypersonic_mission(
                        target_mach=mach,
                        altitude_range=(altitude-5000, altitude+5000),
                        mission_profile=mission_variant
                    )
                    
                    achieved_range = mission_result.get('achievable_range', 0)
                    fuel_consumption = mission_result.get('fuel_consumption', float('inf'))
                    thermal_load = mission_result.get('max_thermal_load', 0)
                    
                    # Check constraints
                    thermal_ok = thermal_load <= base_mission['constraints']['max_thermal_load']
                    fuel_ok = fuel_consumption <= vehicle_config['mass']['fuel_capacity']
                    
                    if thermal_ok and fuel_ok and achieved_range > best_range:
                        best_range = achieved_range
                        best_config = {
                            'altitude': altitude,
                            'mach': mach,
                            'range': achieved_range,
                            'fuel_consumption': fuel_consumption,
                            'thermal_load': thermal_load
                        }
                    
                    optimization_results.append({
                        'altitude': altitude,
                        'mach': mach,
                        'range': achieved_range,
                        'fuel_consumption': fuel_consumption,
                        'thermal_load': thermal_load,
                        'feasible': thermal_ok and fuel_ok
                    })
                
                except Exception as e:
                    print(f"  Failed at altitude {altitude/1000:.0f}km, Mach {mach:.1f}: {e}")
                    continue
        
        return {
            'optimized_mission': base_mission,
            'optimal_parameters': best_config,
            'optimization_results': optimization_results,
            'range_improvement': (best_range - base_mission['range']) / base_mission['range'] * 100
        }
    
    def optimize_for_fuel_efficiency(self, base_mission: Dict[str, Any], 
                                   vehicle_config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize mission profile for fuel efficiency."""
        print(f"Optimizing {base_mission['name']} for fuel efficiency")
        
        # Multi-point optimization for fuel efficiency
        trajectory_points = 20
        altitudes = np.linspace(60000, 100000, trajectory_points)
        
        best_efficiency = 0
        best_trajectory = None
        
        # Try different altitude profiles
        for profile_type in ['constant', 'climbing', 'diving', 'optimal']:
            if profile_type == 'constant':
                altitude_profile = np.full(trajectory_points, 80000)
            elif profile_type == 'climbing':
                altitude_profile = np.linspace(60000, 100000, trajectory_points)
            elif profile_type == 'diving':
                altitude_profile = np.linspace(100000, 60000, trajectory_points)
            else:  # optimal - parabolic profile
                x = np.linspace(0, 1, trajectory_points)
                altitude_profile = 60000 + 40000 * (1 - 4*(x-0.5)**2)
            
            try:
                # Calculate fuel consumption for this profile
                total_fuel = 0
                thermal_violations = 0
                
                for i, altitude in enumerate(altitude_profile):
                    segment_distance = base_mission['range'] / trajectory_points
                    segment_time = base_mission['duration'] / trajectory_points
                    
                    # Estimate fuel consumption for this segment
                    mach = base_mission['flight_profile']['cruise_mach']
                    fuel_flow = self._estimate_fuel_flow(mach, altitude, vehicle_config)
                    segment_fuel = fuel_flow * segment_time
                    total_fuel += segment_fuel
                    
                    # Check thermal constraints
                    thermal_load = self._estimate_thermal_load(mach, altitude)
                    if thermal_load > base_mission['constraints']['max_thermal_load']:
                        thermal_violations += 1
                
                # Calculate efficiency (range per unit fuel)
                if total_fuel > 0 and thermal_violations == 0:
                    efficiency = base_mission['range'] / total_fuel
                    
                    if efficiency > best_efficiency:
                        best_efficiency = efficiency
                        best_trajectory = {
                            'profile_type': profile_type,
                            'altitude_profile': altitude_profile.tolist(),
                            'total_fuel': total_fuel,
                            'efficiency': efficiency,
                            'thermal_violations': thermal_violations
                        }
            
            except Exception as e:
                print(f"  Failed for {profile_type} profile: {e}")
                continue
        
        return {
            'optimized_mission': base_mission,
            'optimal_trajectory': best_trajectory,
            'fuel_savings': ((vehicle_config['mass']['fuel_capacity'] - best_trajectory['total_fuel']) / 
                           vehicle_config['mass']['fuel_capacity'] * 100) if best_trajectory else 0
        }
    
    def optimize_for_thermal_management(self, base_mission: Dict[str, Any], 
                                      vehicle_config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize mission profile to minimize thermal loads."""
        print(f"Optimizing {base_mission['name']} for thermal management")
        
        # Thermal optimization focuses on altitude and speed management
        base_mach = base_mission['flight_profile']['cruise_mach']
        base_altitude = base_mission['flight_profile']['cruise_altitude']
        
        optimization_strategies = [
            {
                'name': 'high_altitude',
                'altitude_modifier': 1.2,
                'mach_modifier': 0.95,
                'description': 'Higher altitude, slightly reduced speed'
            },
            {
                'name': 'speed_modulation',
                'altitude_modifier': 1.0,
                'mach_modifier': 0.90,
                'description': 'Reduced cruise speed'
            },
            {
                'name': 'skip_trajectory',
                'altitude_modifier': 1.1,
                'mach_modifier': 0.98,
                'description': 'Skip-glide trajectory'
            },
            {
                'name': 'thermal_soaking',
                'altitude_modifier': 1.15,
                'mach_modifier': 1.0,
                'description': 'Thermal soaking periods'
            }
        ]
        
        best_strategy = None
        best_thermal_load = float('inf')
        strategy_results = []
        
        for strategy in optimization_strategies:
            modified_altitude = base_altitude * strategy['altitude_modifier']
            modified_mach = base_mach * strategy['mach_modifier']
            
            try:
                # Calculate thermal loads for this strategy
                max_thermal_load = self._estimate_thermal_load(modified_mach, modified_altitude)
                
                # Calculate mission feasibility
                mission_time = base_mission['duration'] * (base_mach / modified_mach)
                range_penalty = (base_mach / modified_mach) - 1
                
                # Check if strategy meets constraints
                thermal_ok = max_thermal_load <= base_mission['constraints']['max_thermal_load']
                time_ok = mission_time <= base_mission['duration'] * 1.1  # 10% time margin
                
                if thermal_ok and time_ok and max_thermal_load < best_thermal_load:
                    best_thermal_load = max_thermal_load
                    best_strategy = {
                        'strategy': strategy,
                        'altitude': modified_altitude,
                        'mach': modified_mach,
                        'max_thermal_load': max_thermal_load,
                        'mission_time': mission_time,
                        'range_penalty': range_penalty
                    }
                
                strategy_results.append({
                    'name': strategy['name'],
                    'description': strategy['description'],
                    'altitude': modified_altitude,
                    'mach': modified_mach,
                    'thermal_load': max_thermal_load,
                    'feasible': thermal_ok and time_ok
                })
            
            except Exception as e:
                print(f"  Failed for {strategy['name']}: {e}")
                continue
        
        return {
            'optimized_mission': base_mission,
            'optimal_strategy': best_strategy,
            'strategy_comparison': strategy_results,
            'thermal_reduction': ((base_mission['constraints']['max_thermal_load'] - best_thermal_load) / 
                                base_mission['constraints']['max_thermal_load'] * 100) if best_strategy else 0
        }
    
    def multi_objective_optimization(self, base_mission: Dict[str, Any], 
                                   vehicle_config: Dict[str, Any],
                                   objectives: Dict[str, float]) -> Dict[str, Any]:
        """Multi-objective optimization balancing multiple criteria."""
        print(f"Multi-objective optimization for {base_mission['name']}")
        
        # Normalize objective weights
        total_weight = sum(objectives.values())
        normalized_objectives = {k: v/total_weight for k, v in objectives.items()}
        
        print(f"Objective weights: {normalized_objectives}")
        
        # Define parameter space
        altitude_range = np.linspace(70000, 110000, 15)
        mach_range = np.linspace(55, 65, 10)
        
        pareto_solutions = []
        all_solutions = []
        
        for altitude in altitude_range:
            for mach in mach_range:
                try:
                    # Evaluate all objectives
                    solution = self._evaluate_solution(
                        base_mission, vehicle_config, altitude, mach
                    )
                    
                    if solution['feasible']:
                        # Calculate weighted objective score
                        weighted_score = 0
                        for obj_name, weight in normalized_objectives.items():
                            if obj_name in solution:
                                weighted_score += weight * solution[obj_name]
                        
                        solution['weighted_score'] = weighted_score
                        all_solutions.append(solution)
                
                except Exception as e:
                    continue
        
        # Find Pareto frontier
        pareto_solutions = self._find_pareto_frontier(all_solutions, list(normalized_objectives.keys()))
        
        # Select best solution based on weighted score
        if all_solutions:
            best_solution = max(all_solutions, key=lambda x: x['weighted_score'])
        else:
            best_solution = None
        
        return {
            'optimized_mission': base_mission,
            'best_solution': best_solution,
            'pareto_solutions': pareto_solutions,
            'all_solutions': all_solutions,
            'objective_weights': normalized_objectives
        }
    
    def _estimate_fuel_flow(self, mach: float, altitude: float, 
                           vehicle_config: Dict[str, Any]) -> float:
        """Estimate fuel flow rate for given conditions."""
        # Simplified fuel flow estimation
        base_flow = 50  # kg/s base flow
        
        # Mach number effect
        mach_factor = (mach / 60) ** 2
        
        # Altitude effect (lower density = higher fuel flow for same thrust)
        altitude_factor = np.exp((altitude - 80000) / 20000)
        
        # Propulsion mode effect
        if mach < 25:
            mode_factor = 0.6  # Air-breathing more efficient
        else:
            mode_factor = 1.0  # Rocket mode
        
        return base_flow * mach_factor * altitude_factor * mode_factor
    
    def _estimate_thermal_load(self, mach: float, altitude: float) -> float:
        """Estimate thermal load for given flight conditions."""
        # Simplified thermal load estimation
        # Heat flux proportional to density * velocity^3
        
        # Atmospheric density (simplified exponential model)
        density = 1.225 * np.exp(-altitude / 8400)  # kg/m³
        
        # Velocity
        velocity = mach * 343 * np.sqrt(288 / (288 - 0.0065 * altitude))  # m/s (simplified)
        
        # Heat flux (simplified)
        heat_flux = 1e-6 * density * (velocity ** 3)  # W/m²
        
        return heat_flux
    
    def _evaluate_solution(self, base_mission: Dict[str, Any], 
                          vehicle_config: Dict[str, Any],
                          altitude: float, mach: float) -> Dict[str, Any]:
        """Evaluate a solution for multi-objective optimization."""
        solution = {
            'altitude': altitude,
            'mach': mach,
            'feasible': True
        }
        
        # Range objective (normalized 0-1, higher is better)
        estimated_range = self._estimate_range(mach, altitude, vehicle_config)
        solution['range'] = min(estimated_range / base_mission['range'], 1.0)
        
        # Fuel efficiency objective (normalized 0-1, higher is better)
        fuel_consumption = self._estimate_fuel_flow(mach, altitude, vehicle_config) * base_mission['duration']
        fuel_capacity = vehicle_config['mass']['fuel_capacity']
        solution['fuel_efficiency'] = max(0, 1 - fuel_consumption / fuel_capacity)
        
        # Thermal objective (normalized 0-1, higher is better = lower thermal load)
        thermal_load = self._estimate_thermal_load(mach, altitude)
        max_thermal = base_mission['constraints']['max_thermal_load']
        solution['thermal'] = max(0, 1 - thermal_load / max_thermal)
        
        # Speed objective (normalized 0-1, higher is better)
        target_mach = base_mission['flight_profile']['cruise_mach']
        solution['speed'] = min(mach / target_mach, 1.0)
        
        # Check feasibility
        if (thermal_load > max_thermal or 
            fuel_consumption > fuel_capacity or
            estimated_range < base_mission['range'] * 0.9):
            solution['feasible'] = False
        
        return solution
    
    def _estimate_range(self, mach: float, altitude: float, 
                       vehicle_config: Dict[str, Any]) -> float:
        """Estimate achievable range for given conditions."""
        # Simplified range estimation based on fuel capacity and consumption
        fuel_capacity = vehicle_config['mass']['fuel_capacity']
        fuel_flow = self._estimate_fuel_flow(mach, altitude, vehicle_config)
        
        if fuel_flow > 0:
            flight_time = fuel_capacity / fuel_flow
            velocity = mach * 343  # Simplified
            return velocity * flight_time * 0.8  # 80% efficiency factor
        else:
            return 0
    
    def _find_pareto_frontier(self, solutions: List[Dict[str, Any]], 
                             objectives: List[str]) -> List[Dict[str, Any]]:
        """Find Pareto-optimal solutions."""
        pareto_solutions = []
        
        for i, solution in enumerate(solutions):
            if not solution['feasible']:
                continue
                
            is_pareto = True
            
            for j, other_solution in enumerate(solutions):
                if i == j or not other_solution['feasible']:
                    continue
                
                # Check if other solution dominates this one
                dominates = True
                for obj in objectives:
                    if other_solution[obj] <= solution[obj]:
                        dominates = False
                        break
                
                if dominates:
                    is_pareto = False
                    break
            
            if is_pareto:
                pareto_solutions.append(solution)
        
        return pareto_solutions


def run_mission_optimization_examples():
    """Run examples of mission optimization for different scenarios."""
    print("🚀 Mach 60 Mission Optimization Examples")
    print("="*60)
    
    # Initialize components
    mission_library = MissionProfileLibrary()
    optimizer = MissionOptimizer()
    
    # Example vehicle configuration
    vehicle_config = {
        'mass': {
            'fuel_capacity': 15000,  # kg
            'empty_mass': 25000,     # kg
            'max_takeoff_mass': 42000  # kg
        },
        'propulsion': {
            'transition_mach': 25.0,
            'max_thrust': 200000  # N
        }
    }
    
    # Get mission profiles
    missions = [
        mission_library.global_strike_mission(),
        mission_library.reconnaissance_mission(),
        mission_library.space_access_mission()
    ]
    
    optimization_results = {}
    
    for mission in missions:
        print(f"\n📋 Optimizing: {mission['name']}")
        print(f"   Base range: {mission['range']/1000000:.0f},000 km")
        print(f"   Duration: {mission['duration']/60:.0f} minutes")
        
        # Range optimization
        print("   🎯 Optimizing for range...")
        range_opt = optimizer.optimize_for_range(mission, vehicle_config)
        
        # Fuel efficiency optimization
        print("   ⛽ Optimizing for fuel efficiency...")
        fuel_opt = optimizer.optimize_for_fuel_efficiency(mission, vehicle_config)
        
        # Thermal management optimization
        print("   🔥 Optimizing for thermal management...")
        thermal_opt = optimizer.optimize_for_thermal_management(mission, vehicle_config)
        
        # Multi-objective optimization
        print("   🎯 Multi-objective optimization...")
        objectives = {
            'range': 0.3,
            'fuel_efficiency': 0.3,
            'thermal': 0.2,
            'speed': 0.2
        }
        multi_opt = optimizer.multi_objective_optimization(mission, vehicle_config, objectives)
        
        optimization_results[mission['name']] = {
            'base_mission': mission,
            'range_optimization': range_opt,
            'fuel_optimization': fuel_opt,
            'thermal_optimization': thermal_opt,
            'multi_objective_optimization': multi_opt
        }
        
        # Print summary results
        if range_opt['optimal_parameters']:
            print(f"   ✅ Range optimized: +{range_opt['range_improvement']:.1f}% range improvement")
        
        if fuel_opt['optimal_trajectory']:
            print(f"   ✅ Fuel optimized: {fuel_opt['fuel_savings']:.1f}% fuel savings")
        
        if thermal_opt['optimal_strategy']:
            print(f"   ✅ Thermal optimized: {thermal_opt['thermal_reduction']:.1f}% thermal reduction")
        
        if multi_opt['best_solution']:
            print(f"   ✅ Multi-objective: Score {multi_opt['best_solution']['weighted_score']:.3f}")
    
    # Save results
    output_file = "mach60_mission_optimization_results.json"
    with open(output_file, 'w') as f:
        json.dump(optimization_results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Generate summary report
    print("\n📊 OPTIMIZATION SUMMARY")
    print("="*60)
    
    for mission_name, results in optimization_results.items():
        print(f"\n🎯 {mission_name}:")
        
        base_mission = results['base_mission']
        print(f"   Base Performance:")
        print(f"     Range: {base_mission['range']/1000000:.0f},000 km")
        print(f"     Duration: {base_mission['duration']/60:.0f} min")
        print(f"     Payload: {base_mission['payload']['mass']/1000:.1f} tonnes")
        
        # Range optimization results
        range_opt = results['range_optimization']
        if range_opt['optimal_parameters']:
            opt_params = range_opt['optimal_parameters']
            print(f"   Range Optimization:")
            print(f"     Optimal altitude: {opt_params['altitude']/1000:.0f} km")
            print(f"     Optimal Mach: {opt_params['mach']:.1f}")
            print(f"     Range improvement: +{range_opt['range_improvement']:.1f}%")
        
        # Fuel optimization results
        fuel_opt = results['fuel_optimization']
        if fuel_opt['optimal_trajectory']:
            print(f"   Fuel Optimization:")
            print(f"     Profile type: {fuel_opt['optimal_trajectory']['profile_type']}")
            print(f"     Fuel savings: {fuel_opt['fuel_savings']:.1f}%")
        
        # Thermal optimization results
        thermal_opt = results['thermal_optimization']
        if thermal_opt['optimal_strategy']:
            strategy = thermal_opt['optimal_strategy']
            print(f"   Thermal Optimization:")
            print(f"     Strategy: {strategy['strategy']['name']}")
            print(f"     Thermal reduction: {thermal_opt['thermal_reduction']:.1f}%")
        
        # Multi-objective results
        multi_opt = results['multi_objective_optimization']
        if multi_opt['best_solution']:
            best = multi_opt['best_solution']
            print(f"   Multi-Objective Optimization:")
            print(f"     Altitude: {best['altitude']/1000:.0f} km")
            print(f"     Mach: {best['mach']:.1f}")
            print(f"     Overall score: {best['weighted_score']:.3f}")
    
    return optimization_results


def create_mission_comparison_report(optimization_results: Dict[str, Any]):
    """Create a detailed comparison report of optimized missions."""
    print("\n📈 MISSION COMPARISON REPORT")
    print("="*80)
    
    # Performance metrics comparison
    metrics = ['range', 'fuel_efficiency', 'thermal_performance', 'speed']
    
    print("\n🏆 Performance Comparison (Normalized Scores 0-1):")
    print("-" * 80)
    print(f"{'Mission':<25} {'Range':<10} {'Fuel Eff':<10} {'Thermal':<10} {'Speed':<10} {'Overall':<10}")
    print("-" * 80)
    
    for mission_name, results in optimization_results.items():
        multi_opt = results['multi_objective_optimization']
        if multi_opt['best_solution']:
            best = multi_opt['best_solution']
            overall = best['weighted_score']
            
            print(f"{mission_name:<25} "
                  f"{best.get('range', 0):<10.3f} "
                  f"{best.get('fuel_efficiency', 0):<10.3f} "
                  f"{best.get('thermal', 0):<10.3f} "
                  f"{best.get('speed', 0):<10.3f} "
                  f"{overall:<10.3f}")
    
    # Mission capability matrix
    print("\n🎯 Mission Capability Matrix:")
    print("-" * 80)
    
    capabilities = ['Long Range', 'High Speed', 'Fuel Efficient', 'Thermal Robust', 'Payload Capacity']
    
    for mission_name, results in optimization_results.items():
        base_mission = results['base_mission']
        print(f"\n{mission_name}:")
        
        # Assess capabilities based on mission parameters
        long_range = "✅" if base_mission['range'] > 10000000 else "⚠️" if base_mission['range'] > 5000000 else "❌"
        high_speed = "✅" if base_mission['flight_profile']['cruise_mach'] >= 60 else "⚠️"
        fuel_efficient = "✅" if results['fuel_optimization']['fuel_savings'] > 10 else "⚠️"
        thermal_robust = "✅" if results['thermal_optimization']['thermal_reduction'] > 15 else "⚠️"
        payload_cap = "✅" if base_mission['payload']['mass'] > 2000 else "⚠️" if base_mission['payload']['mass'] > 1000 else "❌"
        
        print(f"  Long Range (>10,000km): {long_range}")
        print(f"  High Speed (Mach 60+): {high_speed}")
        print(f"  Fuel Efficient (>10% savings): {fuel_efficient}")
        print(f"  Thermal Robust (>15% reduction): {thermal_robust}")
        print(f"  High Payload (>2 tonnes): {payload_cap}")
    
    # Technology requirements
    print("\n🔧 Technology Requirements Summary:")
    print("-" * 80)
    
    tech_requirements = {
        'Combined-Cycle Propulsion': ['All missions require seamless air-breathing to rocket transition'],
        'Thermal Protection': ['UHTC materials', 'Active cooling systems', 'Thermal management algorithms'],
        'Plasma Management': ['Communication through plasma', 'Electromagnetic shielding', 'Plasma diagnostics'],
        'Advanced Materials': ['Ultra-high temperature ceramics', 'Lightweight structures', 'Thermal barriers'],
        'Flight Control': ['Hypersonic stability', 'Plasma-aware guidance', 'Autonomous systems'],
        'Ground Infrastructure': ['Specialized launch facilities', 'Hypersonic test ranges', 'Recovery systems']
    }
    
    for category, requirements in tech_requirements.items():
        print(f"\n{category}:")
        for req in requirements:
            print(f"  • {req}")


if __name__ == "__main__":
    # Run mission optimization examples
    results = run_mission_optimization_examples()
    
    # Create comparison report
    create_mission_comparison_report(results)
    
    print("\n🎉 Mission optimization examples completed!")
    print("📄 Detailed results saved to JSON file for further analysis")