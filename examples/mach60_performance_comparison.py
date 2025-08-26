#!/usr/bin/env python3
"""
Mach 60 Performance Comparison with Conventional Systems

This script provides comprehensive performance comparisons between
Mach 60 hypersonic vehicles and existing conventional systems,
demonstrating the technological advancement and operational advantages.

Requirements: 7.5
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from pathlib import Path

from examples.mach60_hypersonic_vehicle_demo import Mach60VehicleDesigner


class PerformanceComparator:
    """Compare Mach 60 vehicles with conventional hypersonic systems."""
    
    def __init__(self):
        """Initialize the performance comparator."""
        self.designer = Mach60VehicleDesigner()
        self.conventional_systems = self._load_conventional_systems()
        self.comparison_metrics = self._define_comparison_metrics()
    
    def _load_conventional_systems(self) -> Dict[str, Dict[str, Any]]:
        """Load database of conventional hypersonic systems for comparison."""
        return {
            'sr71_blackbird': {
                'name': 'SR-71 Blackbird',
                'type': 'Strategic Reconnaissance Aircraft',
                'first_flight': 1964,
                'operational_period': '1966-1998',
                'performance': {
                    'max_mach': 3.3,
                    'service_ceiling': 25900,  # m
                    'range': 5400000,  # m
                    'cruise_speed': 980,  # m/s (Mach 3.3 at altitude)
                    'fuel_capacity': 46000,  # kg
                    'empty_weight': 30600,  # kg
                    'max_takeoff_weight': 78000,  # kg
                    'payload': 3200,  # kg (sensors)
                },
                'propulsion': {
                    'type': 'Turbojet with afterburner',
                    'engines': 2,
                    'thrust_per_engine': 145000,  # N with afterburner
                    'fuel_type': 'JP-7',
                    'specific_impulse': 1800,  # s (estimated)
                },
                'thermal_protection': {
                    'type': 'Passive',
                    'materials': ['Titanium alloy', 'Heat-resistant steel'],
                    'max_surface_temperature': 700,  # K
                },
                'operational_capabilities': {
                    'mission_types': ['Strategic reconnaissance', 'Surveillance'],
                    'typical_mission_duration': 10800,  # s (3 hours)
                    'crew': 2,
                    'stealth_features': 'Limited RCS reduction'
                }
            },
            
            'x15_research': {
                'name': 'North American X-15',
                'type': 'Experimental Rocket-Powered Aircraft',
                'first_flight': 1959,
                'operational_period': '1959-1968',
                'performance': {
                    'max_mach': 6.7,
                    'max_altitude': 107960,  # m
                    'range': 450000,  # m (limited by fuel)
                    'max_speed': 2020,  # m/s
                    'fuel_capacity': 6800,  # kg
                    'empty_weight': 6620,  # kg
                    'max_takeoff_weight': 15420,  # kg
                    'payload': 500,  # kg (pilot + instruments)
                },
                'propulsion': {
                    'type': 'Rocket engine',
                    'engines': 1,
                    'thrust': 254000,  # N
                    'fuel_type': 'Anhydrous ammonia + Liquid oxygen',
                    'specific_impulse': 276,  # s
                },
                'thermal_protection': {
                    'type': 'Passive',
                    'materials': ['Inconel X', 'Stainless steel'],
                    'max_surface_temperature': 1200,  # K
                },
                'operational_capabilities': {
                    'mission_types': ['Research', 'Test flights'],
                    'typical_mission_duration': 600,  # s (10 minutes)
                    'crew': 1,
                    'air_launched': True
                }
            },
            
            'x43a_scramjet': {
                'name': 'NASA X-43A',
                'type': 'Scramjet Technology Demonstrator',
                'first_flight': 2004,
                'operational_period': '2004-2004',
                'performance': {
                    'max_mach': 9.6,
                    'test_altitude': 33500,  # m
                    'range': 100000,  # m (test flight)
                    'max_speed': 3200,  # m/s
                    'fuel_capacity': 100,  # kg (estimated)
                    'empty_weight': 1270,  # kg
                    'max_takeoff_weight': 1270,  # kg (unmanned)
                    'payload': 200,  # kg (instruments)
                },
                'propulsion': {
                    'type': 'Scramjet',
                    'engines': 1,
                    'thrust': 1500,  # N (estimated)
                    'fuel_type': 'Hydrogen',
                    'specific_impulse': 3600,  # s
                },
                'thermal_protection': {
                    'type': 'Passive',
                    'materials': ['Carbon-carbon', 'UHTC'],
                    'max_surface_temperature': 2000,  # K
                },
                'operational_capabilities': {
                    'mission_types': ['Technology demonstration'],
                    'typical_mission_duration': 10,  # s (powered flight)
                    'crew': 0,
                    'air_launched': True,
                    'technology_demonstrator': True
                }
            },
            
            'space_shuttle': {
                'name': 'Space Shuttle',
                'type': 'Reusable Space Transportation System',
                'first_flight': 1981,
                'operational_period': '1981-2011',
                'performance': {
                    'max_mach': 25.0,  # During reentry
                    'max_altitude': 600000,  # m (orbital)
                    'range': 40000000,  # m (orbital distance)
                    'max_speed': 7800,  # m/s (orbital velocity)
                    'fuel_capacity': 728000,  # kg (including SRBs)
                    'empty_weight': 78000,  # kg (orbiter only)
                    'max_takeoff_weight': 2030000,  # kg (full stack)
                    'payload': 27500,  # kg to LEO
                },
                'propulsion': {
                    'type': 'Rocket engines + SRBs',
                    'main_engines': 3,
                    'thrust_main_engines': 1860000,  # N (3 engines)
                    'fuel_type': 'Liquid hydrogen + Liquid oxygen',
                    'specific_impulse': 452,  # s (vacuum)
                },
                'thermal_protection': {
                    'type': 'Passive',
                    'materials': ['Silica tiles', 'RCC panels'],
                    'max_surface_temperature': 1900,  # K
                },
                'operational_capabilities': {
                    'mission_types': ['Space access', 'Satellite deployment', 'ISS missions'],
                    'typical_mission_duration': 1209600,  # s (14 days)
                    'crew': 7,
                    'reusable': True
                }
            },
            
            'falcon_heavy': {
                'name': 'SpaceX Falcon Heavy',
                'type': 'Heavy-Lift Launch Vehicle',
                'first_flight': 2018,
                'operational_period': '2018-present',
                'performance': {
                    'max_mach': 25.0,  # During ascent
                    'max_altitude': 200000,  # m (typical)
                    'range': 40000000,  # m (GTO mission)
                    'max_speed': 11000,  # m/s (escape velocity missions)
                    'fuel_capacity': 1420000,  # kg
                    'empty_weight': 140000,  # kg
                    'max_takeoff_weight': 1420000,  # kg
                    'payload': 63800,  # kg to LEO
                },
                'propulsion': {
                    'type': 'Rocket engines',
                    'engines': 27,  # First stage
                    'thrust': 22819000,  # N (all engines)
                    'fuel_type': 'RP-1 + Liquid oxygen',
                    'specific_impulse': 282,  # s (sea level)
                },
                'thermal_protection': {
                    'type': 'Passive',
                    'materials': ['PICA-X', 'Aluminum grid fins'],
                    'max_surface_temperature': 1800,  # K
                },
                'operational_capabilities': {
                    'mission_types': ['Heavy payload launch', 'Interplanetary missions'],
                    'typical_mission_duration': 3600,  # s (to orbit)
                    'crew': 0,
                    'partially_reusable': True
                }
            }
        }
    
    def _define_comparison_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Define metrics for performance comparison."""
        return {
            'speed_performance': {
                'name': 'Speed Performance',
                'metrics': ['max_mach', 'cruise_speed', 'acceleration_capability'],
                'units': ['Mach', 'm/s', 'm/s²'],
                'weight': 0.25
            },
            'range_endurance': {
                'name': 'Range and Endurance',
                'metrics': ['range', 'mission_duration', 'fuel_efficiency'],
                'units': ['km', 'hours', 'km/kg'],
                'weight': 0.20
            },
            'altitude_capability': {
                'name': 'Altitude Capability',
                'metrics': ['service_ceiling', 'operational_envelope', 'climb_rate'],
                'units': ['km', 'km²', 'm/s'],
                'weight': 0.15
            },
            'payload_capacity': {
                'name': 'Payload Capacity',
                'metrics': ['payload_mass', 'payload_volume', 'payload_fraction'],
                'units': ['kg', 'm³', '%'],
                'weight': 0.15
            },
            'thermal_capability': {
                'name': 'Thermal Capability',
                'metrics': ['max_temperature', 'heat_flux_capability', 'thermal_protection'],
                'units': ['K', 'MW/m²', 'score'],
                'weight': 0.15
            },
            'operational_flexibility': {
                'name': 'Operational Flexibility',
                'metrics': ['mission_types', 'turnaround_time', 'weather_independence'],
                'units': ['count', 'hours', 'score'],
                'weight': 0.10
            }
        }
    
    def create_mach60_baseline(self) -> Dict[str, Any]:
        """Create baseline Mach 60 vehicle for comparison."""
        config = self.designer.create_baseline_configuration()
        
        # Run basic analysis to get performance data
        propulsion_results = self.designer.analyze_propulsion_system(config)
        thermal_results = self.designer.analyze_thermal_protection(config)
        mission_results = self.designer.plan_mission_profile(config)
        
        # Extract key performance metrics
        mach60_system = {
            'name': 'Mach 60 Hypersonic Vehicle',
            'type': 'Advanced Hypersonic Vehicle',
            'first_flight': 2045,  # Projected
            'operational_period': '2045-future',
            'performance': {
                'max_mach': config['design_mach'],
                'service_ceiling': config['operational_altitude_range'][1],
                'range': config['mission']['range'],
                'cruise_speed': config['design_mach'] * 343,  # Simplified
                'fuel_capacity': config['mass']['fuel_capacity'],
                'empty_weight': config['mass']['empty_mass'],
                'max_takeoff_weight': config['mass']['max_takeoff_mass'],
                'payload': config['mass']['payload_capacity'],
            },
            'propulsion': {
                'type': 'Combined-cycle (scramjet + rocket)',
                'transition_mach': config['propulsion']['transition_mach'],
                'fuel_type': 'Hydrogen',
                'specific_impulse': 3500,  # Combined average
            },
            'thermal_protection': {
                'type': 'Hybrid (passive + active)',
                'materials': ['UHTC', 'Active cooling'],
                'max_surface_temperature': 3000,  # K
                'max_heat_flux': config['thermal_protection']['design_heat_flux'] / 1e6,  # MW/m²
            },
            'operational_capabilities': {
                'mission_types': ['Global strike', 'Reconnaissance', 'Space access'],
                'typical_mission_duration': config['mission']['mission_duration'],
                'crew': 0,  # Unmanned
                'autonomous': True
            },
            'analysis_results': {
                'propulsion': propulsion_results,
                'thermal': thermal_results,
                'mission': mission_results
            }
        }
        
        return mach60_system
    
    def calculate_performance_ratios(self, mach60_system: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Calculate performance ratios comparing Mach 60 to conventional systems."""
        ratios = {}
        
        for system_name, system_data in self.conventional_systems.items():
            system_ratios = {}
            
            # Speed comparison
            speed_ratio = mach60_system['performance']['max_mach'] / system_data['performance']['max_mach']
            system_ratios['speed_advantage'] = speed_ratio
            
            # Range comparison
            range_ratio = mach60_system['performance']['range'] / system_data['performance']['range']
            system_ratios['range_advantage'] = range_ratio
            
            # Altitude comparison
            altitude_ratio = (mach60_system['performance']['service_ceiling'] / 
                            system_data['performance'].get('service_ceiling', 
                                                         system_data['performance'].get('max_altitude', 1)))
            system_ratios['altitude_advantage'] = altitude_ratio
            
            # Payload comparison (if applicable)
            if system_data['performance']['payload'] > 0:
                payload_ratio = mach60_system['performance']['payload'] / system_data['performance']['payload']
                system_ratios['payload_ratio'] = payload_ratio
            else:
                system_ratios['payload_ratio'] = float('inf')
            
            # Thermal capability comparison
            mach60_temp = mach60_system['thermal_protection']['max_surface_temperature']
            conventional_temp = system_data['thermal_protection']['max_surface_temperature']
            thermal_ratio = mach60_temp / conventional_temp
            system_ratios['thermal_advantage'] = thermal_ratio
            
            # Mission duration comparison
            duration_ratio = (mach60_system['operational_capabilities']['typical_mission_duration'] / 
                            system_data['operational_capabilities']['typical_mission_duration'])
            system_ratios['endurance_ratio'] = duration_ratio
            
            # Calculate overall performance index
            weights = {'speed': 0.3, 'range': 0.2, 'altitude': 0.2, 'thermal': 0.2, 'payload': 0.1}
            overall_index = (weights['speed'] * speed_ratio +
                           weights['range'] * min(range_ratio, 10) +  # Cap at 10x
                           weights['altitude'] * altitude_ratio +
                           weights['thermal'] * thermal_ratio +
                           weights['payload'] * min(system_ratios['payload_ratio'], 10))
            
            system_ratios['overall_performance_index'] = overall_index
            
            ratios[system_name] = system_ratios
        
        return ratios
    
    def analyze_technology_gaps(self, mach60_system: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Analyze technology gaps between Mach 60 and conventional systems."""
        technology_gaps = {}
        
        for system_name, system_data in self.conventional_systems.items():
            gaps = {
                'propulsion_technology': self._assess_propulsion_gap(mach60_system, system_data),
                'thermal_technology': self._assess_thermal_gap(mach60_system, system_data),
                'materials_technology': self._assess_materials_gap(mach60_system, system_data),
                'control_technology': self._assess_control_gap(mach60_system, system_data),
                'overall_technology_leap': self._assess_overall_gap(mach60_system, system_data)
            }
            
            technology_gaps[system_name] = gaps
        
        return technology_gaps
    
    def _assess_propulsion_gap(self, mach60: Dict[str, Any], conventional: Dict[str, Any]) -> Dict[str, Any]:
        """Assess propulsion technology gap."""
        mach60_prop = mach60['propulsion']
        conv_prop = conventional['propulsion']
        
        gap_analysis = {
            'technology_advancement': 'Revolutionary' if 'combined-cycle' in mach60_prop['type'].lower() else 'Evolutionary',
            'complexity_increase': 'Very High',
            'specific_impulse_improvement': mach60_prop['specific_impulse'] / conv_prop.get('specific_impulse', 1000),
            'key_challenges': [
                'Air-breathing to rocket transition',
                'Hypersonic combustion control',
                'Thermal management integration',
                'Fuel system complexity'
            ],
            'development_risk': 'High',
            'estimated_development_time': '15-20 years'
        }
        
        return gap_analysis
    
    def _assess_thermal_gap(self, mach60: Dict[str, Any], conventional: Dict[str, Any]) -> Dict[str, Any]:
        """Assess thermal protection technology gap."""
        mach60_thermal = mach60['thermal_protection']
        conv_thermal = conventional['thermal_protection']
        
        temp_ratio = mach60_thermal['max_surface_temperature'] / conv_thermal['max_surface_temperature']
        
        gap_analysis = {
            'temperature_increase': f"{temp_ratio:.1f}x higher operating temperature",
            'heat_flux_capability': f"{mach60_thermal.get('max_heat_flux', 150)} MW/m² vs conventional ~1 MW/m²",
            'protection_system_complexity': 'Active cooling required vs passive only',
            'material_requirements': [
                'Ultra-high temperature ceramics (UHTC)',
                'Active cooling systems',
                'Thermal barrier coatings',
                'Graded material systems'
            ],
            'key_challenges': [
                'Material development and testing',
                'Active cooling system integration',
                'Thermal stress management',
                'Manufacturing scalability'
            ],
            'development_risk': 'Very High',
            'estimated_development_time': '10-15 years'
        }
        
        return gap_analysis
    
    def _assess_materials_gap(self, mach60: Dict[str, Any], conventional: Dict[str, Any]) -> Dict[str, Any]:
        """Assess materials technology gap."""
        gap_analysis = {
            'material_advancement': 'Revolutionary',
            'new_material_classes': [
                'Ultra-high temperature ceramics (UHTC)',
                'Functionally graded materials',
                'Smart thermal protection materials',
                'Advanced carbon-carbon composites'
            ],
            'property_requirements': {
                'temperature_capability': '3000+ K vs 1200-2000 K conventional',
                'thermal_shock_resistance': 'Extreme cycling capability',
                'oxidation_resistance': 'Long-term high-temperature exposure',
                'mechanical_properties': 'Strength retention at extreme temperatures'
            },
            'manufacturing_challenges': [
                'UHTC processing and shaping',
                'Quality control at extreme conditions',
                'Cost-effective production',
                'Joining and integration techniques'
            ],
            'development_risk': 'Very High',
            'estimated_development_time': '12-18 years'
        }
        
        return gap_analysis
    
    def _assess_control_gap(self, mach60: Dict[str, Any], conventional: Dict[str, Any]) -> Dict[str, Any]:
        """Assess flight control technology gap."""
        gap_analysis = {
            'control_complexity': 'Extreme - plasma effects, multi-physics coupling',
            'guidance_challenges': [
                'Navigation through plasma blackout',
                'Hypersonic flight dynamics',
                'Multi-mode propulsion control',
                'Thermal constraint management'
            ],
            'autonomy_requirements': 'Full autonomy required due to communication blackout',
            'sensor_technology': [
                'Plasma-hardened sensors',
                'High-temperature electronics',
                'Inertial navigation systems',
                'Thermal monitoring systems'
            ],
            'control_algorithms': [
                'Adaptive control for changing dynamics',
                'Multi-physics state estimation',
                'Predictive thermal management',
                'Fault-tolerant control systems'
            ],
            'development_risk': 'High',
            'estimated_development_time': '8-12 years'
        }
        
        return gap_analysis
    
    def _assess_overall_gap(self, mach60: Dict[str, Any], conventional: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall technology gap."""
        # Calculate technology readiness level gap
        conventional_trl = self._estimate_system_trl(conventional)
        mach60_trl = 2  # Current estimated TRL for Mach 60 systems
        
        gap_analysis = {
            'technology_readiness_gap': f"TRL {conventional_trl} to TRL {mach60_trl} - Major development required",
            'system_complexity_increase': 'Order of magnitude increase',
            'integration_challenges': [
                'Multi-physics coupling',
                'Extreme environment operation',
                'System-level thermal management',
                'Autonomous operation requirements'
            ],
            'development_approach': [
                'Component technology development',
                'Subscale demonstration vehicles',
                'Ground testing infrastructure',
                'Flight test progression'
            ],
            'estimated_total_development_time': '20-30 years',
            'estimated_development_cost': '$50-100 billion',
            'key_risk_factors': [
                'Technical feasibility at system level',
                'Manufacturing scalability',
                'Operational safety and reliability',
                'Cost-effectiveness'
            ]
        }
        
        return gap_analysis
    
    def _estimate_system_trl(self, system_data: Dict[str, Any]) -> int:
        """Estimate Technology Readiness Level of conventional system."""
        if system_data['operational_period'].endswith('present'):
            return 9  # Operational
        elif 'demonstrator' in system_data['type'].lower():
            return 6  # Technology demonstration
        else:
            return 9  # Historical operational systems
    
    def generate_comparison_charts(self, mach60_system: Dict[str, Any], 
                                 performance_ratios: Dict[str, Dict[str, float]]) -> Dict[str, str]:
        """Generate comparison charts and save as files."""
        chart_files = {}
        
        # Performance comparison radar chart
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
        
        metrics = ['Speed', 'Range', 'Altitude', 'Payload', 'Thermal', 'Endurance']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Plot each system
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (system_name, ratios) in enumerate(performance_ratios.items()):
            values = [
                ratios['speed_advantage'],
                min(ratios['range_advantage'], 10),  # Cap at 10x for visualization
                ratios['altitude_advantage'],
                min(ratios['payload_ratio'], 10),
                ratios['thermal_advantage'],
                min(ratios['endurance_ratio'], 10)
            ]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=system_name.replace('_', ' ').title(), 
                   color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 10)
        ax.set_title('Mach 60 Performance Advantage\n(Ratio vs Conventional Systems)', 
                    size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        chart_file = 'mach60_performance_comparison.png'
        plt.tight_layout()
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        chart_files['performance_radar'] = chart_file
        
        # Speed vs Range comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot conventional systems
        for system_name, system_data in self.conventional_systems.items():
            perf = system_data['performance']
            ax.scatter(perf['range']/1000000, perf['max_mach'], 
                      s=100, alpha=0.7, label=system_name.replace('_', ' ').title())
        
        # Plot Mach 60 system
        mach60_perf = mach60_system['performance']
        ax.scatter(mach60_perf['range']/1000000, mach60_perf['max_mach'], 
                  s=200, color='red', marker='*', label='Mach 60 Vehicle', 
                  edgecolors='black', linewidth=2)
        
        ax.set_xlabel('Range (1000 km)', fontsize=12)
        ax.set_ylabel('Maximum Mach Number', fontsize=12)
        ax.set_title('Speed vs Range Comparison', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(mach60_perf['range']/1000000 * 1.1, 15))
        ax.set_ylim(0, mach60_perf['max_mach'] * 1.1)
        
        chart_file = 'mach60_speed_range_comparison.png'
        plt.tight_layout()
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        chart_files['speed_range'] = chart_file
        
        # Technology timeline
        fig, ax = plt.subplots(figsize=(14, 8))
        
        systems_timeline = []
        for system_name, system_data in self.conventional_systems.items():
            first_flight = system_data['first_flight']
            max_mach = system_data['performance']['max_mach']
            systems_timeline.append((first_flight, max_mach, system_name))
        
        # Add Mach 60 system
        systems_timeline.append((2045, mach60_system['performance']['max_mach'], 'Mach 60 Vehicle'))
        
        # Sort by first flight year
        systems_timeline.sort(key=lambda x: x[0])
        
        years = [s[0] for s in systems_timeline]
        machs = [s[1] for s in systems_timeline]
        names = [s[2].replace('_', ' ').title() for s in systems_timeline]
        
        colors = ['blue' if year < 2020 else 'red' for year in years]
        
        ax.scatter(years, machs, s=150, c=colors, alpha=0.7, edgecolors='black')
        
        for i, (year, mach, name) in enumerate(systems_timeline):
            ax.annotate(name, (year, mach), xytext=(5, 5), textcoords='offset points',
                       fontsize=10, ha='left')
        
        ax.set_xlabel('First Flight Year', fontsize=12)
        ax.set_ylabel('Maximum Mach Number', fontsize=12)
        ax.set_title('Hypersonic Vehicle Development Timeline', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        historical_years = [s[0] for s in systems_timeline if s[0] < 2020]
        historical_machs = [s[1] for s in systems_timeline if s[0] < 2020]
        
        if len(historical_years) > 1:
            z = np.polyfit(historical_years, historical_machs, 1)
            p = np.poly1d(z)
            trend_years = np.linspace(min(historical_years), 2050, 100)
            ax.plot(trend_years, p(trend_years), "r--", alpha=0.8, label='Historical Trend')
            ax.legend()
        
        chart_file = 'mach60_technology_timeline.png'
        plt.tight_layout()
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        chart_files['timeline'] = chart_file
        
        return chart_files
    
    def run_comprehensive_comparison(self) -> Dict[str, Any]:
        """Run comprehensive performance comparison analysis."""
        print("🚀 Running Comprehensive Mach 60 Performance Comparison")
        print("="*70)
        
        # Create Mach 60 baseline
        print("📊 Creating Mach 60 baseline configuration...")
        mach60_system = self.create_mach60_baseline()
        
        # Calculate performance ratios
        print("📈 Calculating performance ratios...")
        performance_ratios = self.calculate_performance_ratios(mach60_system)
        
        # Analyze technology gaps
        print("🔬 Analyzing technology gaps...")
        technology_gaps = self.analyze_technology_gaps(mach60_system)
        
        # Generate comparison charts
        print("📊 Generating comparison charts...")
        chart_files = self.generate_comparison_charts(mach60_system, performance_ratios)
        
        # Compile comprehensive results
        comparison_results = {
            'mach60_system': mach60_system,
            'conventional_systems': self.conventional_systems,
            'performance_ratios': performance_ratios,
            'technology_gaps': technology_gaps,
            'comparison_charts': chart_files,
            'analysis_summary': self._generate_analysis_summary(
                mach60_system, performance_ratios, technology_gaps
            )
        }
        
        return comparison_results
    
    def _generate_analysis_summary(self, mach60_system: Dict[str, Any],
                                 performance_ratios: Dict[str, Dict[str, float]],
                                 technology_gaps: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate analysis summary."""
        # Find best performance advantages
        best_advantages = {}
        for metric in ['speed_advantage', 'range_advantage', 'altitude_advantage', 'thermal_advantage']:
            best_system = max(performance_ratios.keys(), 
                            key=lambda x: performance_ratios[x].get(metric, 0))
            best_advantages[metric] = {
                'vs_system': best_system,
                'advantage': performance_ratios[best_system][metric]
            }
        
        # Calculate average performance improvement
        avg_improvements = {}
        for metric in ['speed_advantage', 'range_advantage', 'altitude_advantage']:
            values = [ratios[metric] for ratios in performance_ratios.values()]
            avg_improvements[metric] = np.mean(values)
        
        summary = {
            'key_advantages': {
                'speed': f"{best_advantages['speed_advantage']['advantage']:.1f}x faster than {best_advantages['speed_advantage']['vs_system']}",
                'range': f"{best_advantages['range_advantage']['advantage']:.1f}x longer range than {best_advantages['range_advantage']['vs_system']}",
                'altitude': f"{best_advantages['altitude_advantage']['advantage']:.1f}x higher altitude than {best_advantages['altitude_advantage']['vs_system']}",
                'thermal': f"{best_advantages['thermal_advantage']['advantage']:.1f}x higher temperature capability"
            },
            'average_improvements': {
                'speed': f"{avg_improvements['speed_advantage']:.1f}x average speed improvement",
                'range': f"{avg_improvements['range_advantage']:.1f}x average range improvement",
                'altitude': f"{avg_improvements['altitude_advantage']:.1f}x average altitude improvement"
            },
            'technology_readiness': {
                'current_status': 'Conceptual design phase (TRL 2-3)',
                'development_timeline': '20-30 years to operational capability',
                'key_challenges': [
                    'Combined-cycle propulsion integration',
                    'Ultra-high temperature materials',
                    'Active thermal protection systems',
                    'Hypersonic flight control systems'
                ]
            },
            'operational_impact': {
                'mission_capabilities': [
                    'Global strike in <1 hour',
                    'Rapid global reconnaissance',
                    'Space access vehicle',
                    'Hypersonic research platform'
                ],
                'strategic_advantages': [
                    'Unprecedented speed and range',
                    'Reduced vulnerability window',
                    'Global reach capability',
                    'Technology demonstration leadership'
                ]
            }
        }
        
        return summary


def main():
    """Main function to run performance comparison analysis."""
    print("🚀 Mach 60 Performance Comparison Analysis")
    print("="*60)
    
    # Initialize comparator
    comparator = PerformanceComparator()
    
    # Run comprehensive comparison
    results = comparator.run_comprehensive_comparison()
    
    # Save results to file
    output_file = "mach60_performance_comparison_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Print summary
    print("\n📊 PERFORMANCE COMPARISON SUMMARY")
    print("="*60)
    
    summary = results['analysis_summary']
    
    print("\n🏆 KEY ADVANTAGES:")
    for advantage, description in summary['key_advantages'].items():
        print(f"   • {advantage.title()}: {description}")
    
    print("\n📈 AVERAGE IMPROVEMENTS:")
    for improvement, description in summary['average_improvements'].items():
        print(f"   • {improvement.title()}: {description}")
    
    print("\n🔬 TECHNOLOGY STATUS:")
    tech_status = summary['technology_readiness']
    print(f"   • Current Status: {tech_status['current_status']}")
    print(f"   • Development Timeline: {tech_status['development_timeline']}")
    print("   • Key Challenges:")
    for challenge in tech_status['key_challenges']:
        print(f"     - {challenge}")
    
    print("\n🎯 OPERATIONAL IMPACT:")
    operational = summary['operational_impact']
    print("   • Mission Capabilities:")
    for capability in operational['mission_capabilities']:
        print(f"     - {capability}")
    print("   • Strategic Advantages:")
    for advantage in operational['strategic_advantages']:
        print(f"     - {advantage}")
    
    # Print detailed performance ratios
    print("\n📊 DETAILED PERFORMANCE RATIOS:")
    print("-" * 60)
    print(f"{'System':<25} {'Speed':<8} {'Range':<8} {'Altitude':<10} {'Overall':<8}")
    print("-" * 60)
    
    for system_name, ratios in results['performance_ratios'].items():
        system_display = system_name.replace('_', ' ').title()[:24]
        print(f"{system_display:<25} "
              f"{ratios['speed_advantage']:<8.1f} "
              f"{ratios['range_advantage']:<8.1f} "
              f"{ratios['altitude_advantage']:<10.1f} "
              f"{ratios['overall_performance_index']:<8.1f}")
    
    print("\n📈 Charts generated:")
    for chart_type, filename in results['comparison_charts'].items():
        print(f"   • {chart_type}: {filename}")
    
    print("\n🎉 Performance comparison analysis completed!")
    return results


if __name__ == "__main__":
    results = main()