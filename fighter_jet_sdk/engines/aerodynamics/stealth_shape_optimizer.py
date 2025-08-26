"""
Stealth Shape Optimization Module

Implements stealth-aerodynamic optimization framework that balances RCS reduction
with aerodynamic performance using multi-objective optimization techniques.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass
import logging
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist

from ...common.data_models import AircraftConfiguration, MaterialDefinition
from ...common.enums import ModuleType, MaterialType
from ...core.logging import get_engine_logger
from ..materials.stealth_analyzer import StealthAnalyzer, GeometryModel, StealthConfiguration


@dataclass
class ShapeParameter:
    """Represents a shape parameter for optimization"""
    name: str
    current_value: float
    min_value: float
    max_value: float
    parameter_type: str  # 'geometric', 'material', 'surface'
    affected_surfaces: List[str]


@dataclass
class OptimizationObjective:
    """Defines optimization objective with weight and constraints"""
    name: str
    weight: float
    target_value: Optional[float] = None
    constraint_type: str = 'minimize'  # 'minimize', 'maximize', 'target'


@dataclass
class ParetoPoint:
    """Represents a point on the Pareto frontier"""
    parameters: Dict[str, float]
    objectives: Dict[str, float]
    rcs_reduction_db: float
    aerodynamic_efficiency: float
    feasible: bool


@dataclass
class OptimizationResults:
    """Complete optimization results"""
    pareto_frontier: List[ParetoPoint]
    best_compromise: ParetoPoint
    convergence_history: List[Dict[str, float]]
    optimization_statistics: Dict[str, Any]


class StealthShapeOptimizer:
    """
    Stealth-aerodynamic optimization framework that balances RCS reduction
    with aerodynamic performance using multi-objective optimization.
    """
    
    def __init__(self):
        """Initialize the stealth shape optimizer"""
        self.logger = get_engine_logger('aerodynamics.stealth_optimizer')
        self.stealth_analyzer = StealthAnalyzer()
        
        # Optimization parameters
        self.max_iterations = 200
        self.population_size = 50
        self.convergence_tolerance = 1e-6
        
        # Shape parameterization
        self.shape_parameters = {}
        self.optimization_objectives = {}
        
        # Aerodynamic efficiency models (simplified)
        self._aero_models = {
            'lift_to_drag': self._calculate_lift_to_drag_ratio,
            'wave_drag': self._calculate_wave_drag_coefficient,
            'induced_drag': self._calculate_induced_drag_coefficient
        }
    
    def optimize_configuration(self, configuration: AircraftConfiguration,
                              optimization_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform stealth-aerodynamic optimization on aircraft configuration
        
        Args:
            configuration: Aircraft configuration to optimize
            optimization_params: Optimization parameters and constraints
            
        Returns:
            Dictionary containing optimization results
        """
        self.logger.info("Starting stealth-aerodynamic optimization")
        
        try:
            # Setup optimization problem
            shape_params = self._setup_shape_parameters(configuration, optimization_params)
            objectives = self._setup_optimization_objectives(optimization_params)
            
            # Extract geometry model from configuration
            geometry = self._extract_geometry_model(configuration)
            
            # Get materials database
            materials_db = self._get_materials_database(configuration)
            
            # Run multi-objective optimization
            results = self._run_multi_objective_optimization(
                geometry, materials_db, shape_params, objectives, optimization_params
            )
            
            # Format results for return
            return self._format_optimization_results(results)
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            raise
    
    def _setup_shape_parameters(self, configuration: AircraftConfiguration,
                               params: Dict[str, Any]) -> List[ShapeParameter]:
        """Setup shape parameters for optimization"""
        shape_params = []
        
        # Geometric parameters
        if params.get('optimize_geometry', True):
            # Wing parameters
            shape_params.extend([
                ShapeParameter(
                    name='wing_sweep_angle',
                    current_value=params.get('initial_wing_sweep', 35.0),
                    min_value=params.get('min_wing_sweep', 20.0),
                    max_value=params.get('max_wing_sweep', 60.0),
                    parameter_type='geometric',
                    affected_surfaces=['wing_upper', 'wing_lower']
                ),
                ShapeParameter(
                    name='wing_taper_ratio',
                    current_value=params.get('initial_wing_taper', 0.3),
                    min_value=0.1,
                    max_value=0.8,
                    parameter_type='geometric',
                    affected_surfaces=['wing_upper', 'wing_lower']
                ),
                ShapeParameter(
                    name='fuselage_fineness_ratio',
                    current_value=params.get('initial_fineness_ratio', 8.0),
                    min_value=6.0,
                    max_value=12.0,
                    parameter_type='geometric',
                    affected_surfaces=['fuselage']
                )
            ])
        
        # Surface treatment parameters
        if params.get('optimize_materials', True):
            shape_params.extend([
                ShapeParameter(
                    name='ram_coating_thickness',
                    current_value=params.get('initial_ram_thickness', 0.005),
                    min_value=0.001,
                    max_value=0.020,
                    parameter_type='material',
                    affected_surfaces=['fuselage', 'wing_upper', 'wing_lower']
                ),
                ShapeParameter(
                    name='surface_roughness',
                    current_value=params.get('initial_surface_roughness', 1e-6),
                    min_value=1e-7,
                    max_value=1e-4,
                    parameter_type='surface',
                    affected_surfaces=['all']
                )
            ])
        
        # Engine inlet shaping
        if params.get('optimize_inlets', True):
            shape_params.append(
                ShapeParameter(
                    name='inlet_lip_radius',
                    current_value=params.get('initial_inlet_radius', 0.1),
                    min_value=0.05,
                    max_value=0.3,
                    parameter_type='geometric',
                    affected_surfaces=['engine_inlet']
                )
            )
        
        self.logger.info(f"Setup {len(shape_params)} shape parameters for optimization")
        return shape_params
    
    def _setup_optimization_objectives(self, params: Dict[str, Any]) -> List[OptimizationObjective]:
        """Setup optimization objectives"""
        objectives = []
        
        # RCS reduction objective
        objectives.append(OptimizationObjective(
            name='rcs_reduction',
            weight=params.get('rcs_weight', 0.6),
            constraint_type='maximize'
        ))
        
        # Aerodynamic efficiency objective
        objectives.append(OptimizationObjective(
            name='aerodynamic_efficiency',
            weight=params.get('aero_weight', 0.4),
            constraint_type='maximize'
        ))
        
        # Additional objectives if specified
        if params.get('minimize_weight', False):
            objectives.append(OptimizationObjective(
                name='structural_weight',
                weight=params.get('weight_penalty', 0.1),
                constraint_type='minimize'
            ))
        
        if params.get('cost_constraint', False):
            objectives.append(OptimizationObjective(
                name='manufacturing_cost',
                weight=params.get('cost_penalty', 0.05),
                target_value=params.get('max_cost', 1e6),
                constraint_type='target'
            ))
        
        return objectives
    
    def _extract_geometry_model(self, configuration: AircraftConfiguration) -> GeometryModel:
        """Extract geometry model from aircraft configuration"""
        # Extract basic dimensions from configuration modules
        fuselage_modules = [m for m in configuration.modules if 'fuselage' in m.module_id.lower()]
        wing_modules = [m for m in configuration.modules if 'wing' in m.module_id.lower()]
        engine_modules = [m for m in configuration.modules if m.module_type == ModuleType.PROPULSION]
        
        # Default values with extraction from modules if available
        fuselage_length = 15.0  # Default
        fuselage_diameter = 1.5
        wing_span = 12.0
        wing_chord = 3.0
        wing_thickness = 0.3
        tail_area = 8.0
        engine_inlet_area = 1.0
        
        # Extract from modules if available
        if fuselage_modules:
            fuselage_props = fuselage_modules[0].physical_properties
            if fuselage_props:
                fuselage_length = fuselage_props.dimensions[0]  # length
                fuselage_diameter = fuselage_props.dimensions[1]  # width/diameter
        
        if wing_modules:
            wing_props = wing_modules[0].physical_properties
            if wing_props:
                wing_span = wing_props.dimensions[0]  # span
                wing_chord = wing_props.dimensions[1]  # chord
                wing_thickness = wing_props.dimensions[2]  # thickness
        
        if engine_modules:
            engine_props = engine_modules[0].physical_properties
            if engine_props:
                # Estimate inlet area from engine dimensions
                engine_inlet_area = np.pi * (engine_props.dimensions[1] / 2) ** 2
        
        # Default surface materials
        surface_materials = {
            'fuselage': 'conventional_metal',
            'wing_upper': 'conventional_metal',
            'wing_lower': 'conventional_metal',
            'tail': 'conventional_metal',
            'engine_inlet': 'conventional_metal'
        }
        
        return GeometryModel(
            fuselage_length=fuselage_length,
            fuselage_diameter=fuselage_diameter,
            wing_span=wing_span,
            wing_chord=wing_chord,
            wing_thickness=wing_thickness,
            tail_area=tail_area,
            engine_inlet_area=engine_inlet_area,
            surface_materials=surface_materials
        )
    
    def _get_materials_database(self, configuration: AircraftConfiguration) -> Dict[str, MaterialDefinition]:
        """Get materials database for optimization"""
        # Create basic materials database
        materials_db = {}
        
        # Conventional metal
        materials_db['conventional_metal'] = MaterialDefinition(
            material_id='conventional_metal',
            base_material_type=MaterialType.CONVENTIONAL_METAL,
            electromagnetic_properties={
                'permittivity': complex(1.0, -1e6),  # High conductivity
                'permeability': complex(1.0, 0.0),
                'frequency_range': (1e6, 1e12)
            },
            thermal_properties={
                'thermal_conductivity': 200.0,
                'specific_heat': 900.0,
                'density': 2700.0
            },
            mechanical_properties={
                'youngs_modulus': 70e9,
                'poisson_ratio': 0.33,
                'yield_strength': 270e6
            },
            manufacturing_constraints={
                'min_thickness': 0.001,
                'max_thickness': 0.1,
                'formability': 0.8
            }
        )
        
        # RAM coating
        materials_db['ram_coating'] = MaterialDefinition(
            material_id='ram_coating',
            base_material_type=MaterialType.STEALTH_COATING,
            electromagnetic_properties={
                'permittivity': complex(12.0, -3.0),
                'permeability': complex(1.0, 0.0),
                'frequency_range': (1e9, 40e9)
            },
            thermal_properties={
                'thermal_conductivity': 0.5,
                'specific_heat': 1200.0,
                'density': 1800.0
            },
            mechanical_properties={
                'youngs_modulus': 5e9,
                'poisson_ratio': 0.4,
                'yield_strength': 50e6
            },
            manufacturing_constraints={
                'min_thickness': 0.001,
                'max_thickness': 0.02,
                'formability': 0.6
            }
        )
        
        # Metamaterial
        materials_db['metamaterial'] = MaterialDefinition(
            material_id='metamaterial',
            base_material_type=MaterialType.METAMATERIAL,
            electromagnetic_properties={
                'permittivity': complex(-2.0, -0.5),  # Negative real part
                'permeability': complex(1.2, -0.1),
                'frequency_range': (8e9, 12e9)  # X-band
            },
            thermal_properties={
                'thermal_conductivity': 1.0,
                'specific_heat': 800.0,
                'density': 1200.0
            },
            mechanical_properties={
                'youngs_modulus': 2e9,
                'poisson_ratio': 0.35,
                'yield_strength': 30e6
            },
            manufacturing_constraints={
                'min_thickness': 0.005,
                'max_thickness': 0.05,
                'formability': 0.3
            }
        )
        
        return materials_db
    
    def _run_multi_objective_optimization(self, geometry: GeometryModel,
                                        materials_db: Dict[str, MaterialDefinition],
                                        shape_params: List[ShapeParameter],
                                        objectives: List[OptimizationObjective],
                                        params: Dict[str, Any]) -> OptimizationResults:
        """Run multi-objective optimization using NSGA-II inspired approach"""
        self.logger.info("Running multi-objective optimization")
        
        # Initialize population
        population = self._initialize_population(shape_params, self.population_size)
        
        # Evolution loop
        pareto_history = []
        convergence_history = []
        
        for generation in range(self.max_iterations):
            # Evaluate population
            evaluated_pop = []
            for individual in population:
                objectives_values = self._evaluate_individual(
                    individual, geometry, materials_db, shape_params, objectives
                )
                
                pareto_point = ParetoPoint(
                    parameters=individual.copy(),
                    objectives=objectives_values,
                    rcs_reduction_db=objectives_values.get('rcs_reduction', 0.0),
                    aerodynamic_efficiency=objectives_values.get('aerodynamic_efficiency', 0.0),
                    feasible=self._check_feasibility(individual, shape_params)
                )
                evaluated_pop.append(pareto_point)
            
            # Find Pareto frontier
            pareto_frontier = self._find_pareto_frontier(evaluated_pop)
            pareto_history.append(pareto_frontier)
            
            # Record convergence metrics
            if pareto_frontier:
                avg_rcs = np.mean([p.rcs_reduction_db for p in pareto_frontier])
                avg_aero = np.mean([p.aerodynamic_efficiency for p in pareto_frontier])
                convergence_history.append({
                    'generation': generation,
                    'pareto_size': len(pareto_frontier),
                    'avg_rcs_reduction': avg_rcs,
                    'avg_aero_efficiency': avg_aero
                })
            
            # Check convergence
            if generation > 10 and self._check_convergence(pareto_history[-10:]):
                self.logger.info(f"Optimization converged at generation {generation}")
                break
            
            # Generate next population
            population = self._generate_next_population(
                evaluated_pop, pareto_frontier, shape_params
            )
            
            if generation % 20 == 0:
                self.logger.info(f"Generation {generation}: {len(pareto_frontier)} Pareto points")
        
        # Select best compromise solution
        final_pareto = pareto_history[-1] if pareto_history else []
        best_compromise = self._select_best_compromise(final_pareto, objectives)
        
        # Compile statistics
        statistics = {
            'total_generations': len(convergence_history),
            'final_pareto_size': len(final_pareto),
            'convergence_achieved': generation < self.max_iterations - 1,
            'best_rcs_reduction': best_compromise.rcs_reduction_db if best_compromise else 0.0,
            'best_aero_efficiency': best_compromise.aerodynamic_efficiency if best_compromise else 0.0
        }
        
        return OptimizationResults(
            pareto_frontier=final_pareto,
            best_compromise=best_compromise,
            convergence_history=convergence_history,
            optimization_statistics=statistics
        )
    
    def _initialize_population(self, shape_params: List[ShapeParameter], 
                              pop_size: int) -> List[Dict[str, float]]:
        """Initialize random population within parameter bounds"""
        population = []
        
        for _ in range(pop_size):
            individual = {}
            for param in shape_params:
                # Random value within bounds
                value = np.random.uniform(param.min_value, param.max_value)
                individual[param.name] = value
            population.append(individual)
        
        return population
    
    def _evaluate_individual(self, individual: Dict[str, float],
                           geometry: GeometryModel,
                           materials_db: Dict[str, MaterialDefinition],
                           shape_params: List[ShapeParameter],
                           objectives: List[OptimizationObjective]) -> Dict[str, float]:
        """Evaluate objectives for an individual"""
        # Update geometry with individual parameters
        updated_geometry = self._apply_shape_parameters(geometry, individual, shape_params)
        
        # Calculate RCS reduction
        rcs_reduction = self._calculate_rcs_reduction(updated_geometry, materials_db)
        
        # Calculate aerodynamic efficiency
        aero_efficiency = self._calculate_aerodynamic_efficiency(updated_geometry, individual)
        
        # Calculate additional objectives
        objectives_values = {
            'rcs_reduction': rcs_reduction,
            'aerodynamic_efficiency': aero_efficiency
        }
        
        # Add weight and cost if requested
        for obj in objectives:
            if obj.name == 'structural_weight':
                objectives_values['structural_weight'] = self._calculate_structural_weight(
                    updated_geometry, individual
                )
            elif obj.name == 'manufacturing_cost':
                objectives_values['manufacturing_cost'] = self._calculate_manufacturing_cost(
                    updated_geometry, individual
                )
        
        return objectives_values
    
    def _apply_shape_parameters(self, geometry: GeometryModel,
                               individual: Dict[str, float],
                               shape_params: List[ShapeParameter]) -> GeometryModel:
        """Apply shape parameters to geometry"""
        # Create copy of geometry
        updated_geometry = GeometryModel(
            fuselage_length=geometry.fuselage_length,
            fuselage_diameter=geometry.fuselage_diameter,
            wing_span=geometry.wing_span,
            wing_chord=geometry.wing_chord,
            wing_thickness=geometry.wing_thickness,
            tail_area=geometry.tail_area,
            engine_inlet_area=geometry.engine_inlet_area,
            surface_materials=geometry.surface_materials.copy()
        )
        
        # Apply geometric parameters
        if 'wing_sweep_angle' in individual:
            # Wing sweep affects effective span and chord
            sweep_rad = np.radians(individual['wing_sweep_angle'])
            updated_geometry.wing_span *= np.cos(sweep_rad)
            updated_geometry.wing_chord *= 1.0 + 0.1 * np.sin(sweep_rad)
        
        if 'wing_taper_ratio' in individual:
            # Taper ratio affects effective wing area
            taper = individual['wing_taper_ratio']
            updated_geometry.wing_chord *= (1.0 + taper) / 2.0
        
        if 'fuselage_fineness_ratio' in individual:
            # Fineness ratio affects length-to-diameter ratio
            fineness = individual['fuselage_fineness_ratio']
            # Keep diameter constant, adjust length based on fineness ratio
            updated_geometry.fuselage_length = fineness * updated_geometry.fuselage_diameter
        
        if 'inlet_lip_radius' in individual:
            # Inlet radius affects effective inlet area
            radius_factor = individual['inlet_lip_radius'] / 0.1  # Normalized to default
            updated_geometry.engine_inlet_area *= radius_factor
        
        # Apply material parameters
        if 'ram_coating_thickness' in individual:
            # Apply RAM coating to appropriate surfaces
            for surface in ['fuselage', 'wing_upper', 'wing_lower']:
                if surface in updated_geometry.surface_materials:
                    updated_geometry.surface_materials[surface] = 'ram_coating'
        
        return updated_geometry
    
    def _calculate_rcs_reduction(self, geometry: GeometryModel,
                               materials_db: Dict[str, MaterialDefinition]) -> float:
        """Calculate RCS reduction compared to baseline"""
        # Define baseline geometry with conventional materials
        baseline_geometry = GeometryModel(
            fuselage_length=geometry.fuselage_length,
            fuselage_diameter=geometry.fuselage_diameter,
            wing_span=geometry.wing_span,
            wing_chord=geometry.wing_chord,
            wing_thickness=geometry.wing_thickness,
            tail_area=geometry.tail_area,
            engine_inlet_area=geometry.engine_inlet_area,
            surface_materials={k: 'conventional_metal' for k in geometry.surface_materials.keys()}
        )
        
        # Calculate RCS for key threat frequencies and angles
        threat_frequencies = [3e9, 10e9, 16e9]  # S, X, Ku bands
        threat_angles = [0, 30, 60, 90]  # degrees
        
        baseline_rcs_total = 0.0
        optimized_rcs_total = 0.0
        
        for freq in threat_frequencies:
            for angle in threat_angles:
                baseline_rcs = self.stealth_analyzer._calculate_rcs_hybrid(
                    baseline_geometry, materials_db, freq, angle, 'VV'
                )
                optimized_rcs = self.stealth_analyzer._calculate_rcs_hybrid(
                    geometry, materials_db, freq, angle, 'VV'
                )
                
                baseline_rcs_total += baseline_rcs
                optimized_rcs_total += optimized_rcs
        
        # Calculate RCS reduction in dB
        if optimized_rcs_total > 0:
            rcs_reduction_db = 10 * np.log10(baseline_rcs_total / optimized_rcs_total)
        else:
            rcs_reduction_db = 50.0  # Maximum theoretical reduction
        
        return max(0.0, rcs_reduction_db)
    
    def _calculate_aerodynamic_efficiency(self, geometry: GeometryModel,
                                        individual: Dict[str, float]) -> float:
        """Calculate aerodynamic efficiency metric"""
        # Simplified aerodynamic efficiency calculation
        # In practice, this would interface with CFD solver
        
        # Base efficiency
        efficiency = 0.8
        
        # Wing sweep penalty/benefit
        if 'wing_sweep_angle' in individual:
            sweep = individual['wing_sweep_angle']
            if 30 <= sweep <= 45:  # Optimal range for transonic flight
                efficiency += 0.1
            elif sweep > 60:  # High sweep penalty
                efficiency -= 0.2
        
        # Taper ratio effect
        if 'wing_taper_ratio' in individual:
            taper = individual['wing_taper_ratio']
            if 0.2 <= taper <= 0.4:  # Optimal taper range
                efficiency += 0.05
            else:
                efficiency -= 0.1
        
        # Fineness ratio effect
        if 'fuselage_fineness_ratio' in individual:
            fineness = individual['fuselage_fineness_ratio']
            if 8 <= fineness <= 10:  # Optimal fineness
                efficiency += 0.05
            elif fineness > 12:  # Too slender
                efficiency -= 0.1
        
        # Surface roughness penalty
        if 'surface_roughness' in individual:
            roughness = individual['surface_roughness']
            if roughness > 1e-5:  # Rough surface
                efficiency -= 0.05
        
        # RAM coating penalty
        if 'ram_coating_thickness' in individual:
            coating_thickness = individual['ram_coating_thickness']
            # Thicker coatings add weight and may affect boundary layer
            efficiency -= coating_thickness * 2.0  # Penalty factor
        
        return max(0.1, min(1.0, efficiency))
    
    def _calculate_structural_weight(self, geometry: GeometryModel,
                                   individual: Dict[str, float]) -> float:
        """Calculate structural weight penalty"""
        base_weight = 5000.0  # kg
        
        # Wing sweep structural penalty
        if 'wing_sweep_angle' in individual:
            sweep = individual['wing_sweep_angle']
            if sweep > 45:
                base_weight += (sweep - 45) * 20  # kg per degree
        
        # RAM coating weight
        if 'ram_coating_thickness' in individual:
            coating_thickness = individual['ram_coating_thickness']
            wing_area = geometry.wing_span * geometry.wing_chord * 2
            fuselage_area = np.pi * geometry.fuselage_diameter * geometry.fuselage_length
            total_area = wing_area + fuselage_area
            
            coating_weight = total_area * coating_thickness * 1800  # kg (density of RAM)
            base_weight += coating_weight
        
        return base_weight
    
    def _calculate_manufacturing_cost(self, geometry: GeometryModel,
                                    individual: Dict[str, float]) -> float:
        """Calculate manufacturing cost estimate"""
        base_cost = 50e6  # USD
        
        # Complex geometry cost
        if 'wing_sweep_angle' in individual:
            sweep = individual['wing_sweep_angle']
            if sweep > 50:
                base_cost += (sweep - 50) * 1e5  # USD per degree
        
        # RAM coating cost
        if 'ram_coating_thickness' in individual:
            coating_thickness = individual['ram_coating_thickness']
            wing_area = geometry.wing_span * geometry.wing_chord * 2
            fuselage_area = np.pi * geometry.fuselage_diameter * geometry.fuselage_length
            total_area = wing_area + fuselage_area
            
            coating_cost = total_area * 5000  # USD per m²
            base_cost += coating_cost
        
        return base_cost
    
    def _check_feasibility(self, individual: Dict[str, float],
                          shape_params: List[ShapeParameter]) -> bool:
        """Check if individual satisfies all constraints"""
        for param in shape_params:
            value = individual.get(param.name, param.current_value)
            if not (param.min_value <= value <= param.max_value):
                return False
        
        # Additional feasibility checks
        if 'wing_sweep_angle' in individual and 'wing_taper_ratio' in individual:
            # High sweep with low taper can cause structural issues
            if individual['wing_sweep_angle'] > 55 and individual['wing_taper_ratio'] < 0.15:
                return False
        
        return True
    
    def _find_pareto_frontier(self, population: List[ParetoPoint]) -> List[ParetoPoint]:
        """Find Pareto frontier from population"""
        pareto_frontier = []
        
        for i, point1 in enumerate(population):
            if not point1.feasible:
                continue
                
            is_dominated = False
            
            for j, point2 in enumerate(population):
                if i == j or not point2.feasible:
                    continue
                
                # Check if point1 is dominated by point2
                if (point2.rcs_reduction_db >= point1.rcs_reduction_db and
                    point2.aerodynamic_efficiency >= point1.aerodynamic_efficiency and
                    (point2.rcs_reduction_db > point1.rcs_reduction_db or
                     point2.aerodynamic_efficiency > point1.aerodynamic_efficiency)):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_frontier.append(point1)
        
        return pareto_frontier
    
    def _check_convergence(self, recent_frontiers: List[List[ParetoPoint]]) -> bool:
        """Check if optimization has converged"""
        if len(recent_frontiers) < 5:
            return False
        
        # Check if Pareto frontier size has stabilized
        sizes = [len(frontier) for frontier in recent_frontiers]
        size_variance = np.var(sizes)
        
        if size_variance < 1.0:  # Stable size
            # Check if objective values have stabilized
            recent_rcs = []
            recent_aero = []
            
            for frontier in recent_frontiers[-3:]:
                if frontier:
                    recent_rcs.extend([p.rcs_reduction_db for p in frontier])
                    recent_aero.extend([p.aerodynamic_efficiency for p in frontier])
            
            if recent_rcs and recent_aero:
                rcs_variance = np.var(recent_rcs)
                aero_variance = np.var(recent_aero)
                
                return rcs_variance < 0.1 and aero_variance < 0.001
        
        return False
    
    def _generate_next_population(self, current_pop: List[ParetoPoint],
                                 pareto_frontier: List[ParetoPoint],
                                 shape_params: List[ShapeParameter]) -> List[Dict[str, float]]:
        """Generate next population using selection and mutation"""
        next_population = []
        
        # Elite preservation - keep Pareto frontier
        for point in pareto_frontier[:self.population_size // 4]:
            next_population.append(point.parameters.copy())
        
        # Generate offspring through crossover and mutation
        while len(next_population) < self.population_size:
            # Select parents (tournament selection)
            parent1 = self._tournament_selection(current_pop)
            parent2 = self._tournament_selection(current_pop)
            
            # Crossover
            offspring = self._crossover(parent1.parameters, parent2.parameters)
            
            # Mutation
            offspring = self._mutate(offspring, shape_params)
            
            next_population.append(offspring)
        
        return next_population
    
    def _tournament_selection(self, population: List[ParetoPoint]) -> ParetoPoint:
        """Tournament selection for parent selection"""
        tournament_size = 3
        tournament = np.random.choice(population, tournament_size, replace=False)
        
        # Select best based on combined objective
        best = tournament[0]
        best_score = best.rcs_reduction_db * 0.6 + best.aerodynamic_efficiency * 0.4
        
        for candidate in tournament[1:]:
            if not candidate.feasible:
                continue
            score = candidate.rcs_reduction_db * 0.6 + candidate.aerodynamic_efficiency * 0.4
            if score > best_score:
                best = candidate
                best_score = score
        
        return best
    
    def _crossover(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> Dict[str, float]:
        """Uniform crossover between two parents"""
        offspring = {}
        
        for param_name in parent1.keys():
            if np.random.random() < 0.5:
                offspring[param_name] = parent1[param_name]
            else:
                offspring[param_name] = parent2[param_name]
        
        return offspring
    
    def _mutate(self, individual: Dict[str, float],
               shape_params: List[ShapeParameter]) -> Dict[str, float]:
        """Mutate individual with Gaussian noise"""
        mutation_rate = 0.1
        mutation_strength = 0.05
        
        mutated = individual.copy()
        
        for param in shape_params:
            if np.random.random() < mutation_rate:
                current_value = mutated[param.name]
                param_range = param.max_value - param.min_value
                
                # Gaussian mutation
                noise = np.random.normal(0, mutation_strength * param_range)
                new_value = current_value + noise
                
                # Clamp to bounds
                new_value = max(param.min_value, min(param.max_value, new_value))
                mutated[param.name] = new_value
        
        return mutated
    
    def _select_best_compromise(self, pareto_frontier: List[ParetoPoint],
                               objectives: List[OptimizationObjective]) -> Optional[ParetoPoint]:
        """Select best compromise solution from Pareto frontier"""
        if not pareto_frontier:
            return None
        
        # Calculate weighted sum for each point
        best_point = None
        best_score = -np.inf
        
        for point in pareto_frontier:
            score = 0.0
            
            for obj in objectives:
                if obj.name == 'rcs_reduction':
                    score += obj.weight * point.rcs_reduction_db / 50.0  # Normalize
                elif obj.name == 'aerodynamic_efficiency':
                    score += obj.weight * point.aerodynamic_efficiency
            
            if score > best_score:
                best_score = score
                best_point = point
        
        return best_point
    
    def _format_optimization_results(self, results: OptimizationResults) -> Dict[str, Any]:
        """Format optimization results for return"""
        # Convert Pareto frontier to serializable format
        pareto_data = []
        for point in results.pareto_frontier:
            pareto_data.append({
                'parameters': point.parameters,
                'rcs_reduction_db': point.rcs_reduction_db,
                'aerodynamic_efficiency': point.aerodynamic_efficiency,
                'feasible': point.feasible
            })
        
        # Best compromise data
        best_compromise_data = None
        if results.best_compromise:
            best_compromise_data = {
                'parameters': results.best_compromise.parameters,
                'rcs_reduction_db': results.best_compromise.rcs_reduction_db,
                'aerodynamic_efficiency': results.best_compromise.aerodynamic_efficiency
            }
        
        return {
            'pareto_frontier': pareto_data,
            'best_compromise': best_compromise_data,
            'convergence_history': results.convergence_history,
            'optimization_statistics': results.optimization_statistics,
            'pareto_frontier_size': len(results.pareto_frontier),
            'optimization_successful': results.optimization_statistics.get('convergence_achieved', False)
        }
    
    def _calculate_lift_to_drag_ratio(self, geometry: GeometryModel, 
                                     individual: Dict[str, float]) -> float:
        """Calculate lift-to-drag ratio (simplified model)"""
        # Simplified L/D calculation based on geometry parameters
        base_ld = 15.0  # Baseline L/D ratio
        
        # Wing sweep effect
        if 'wing_sweep_angle' in individual:
            sweep = individual['wing_sweep_angle']
            if sweep < 30:
                base_ld *= 1.1  # Low sweep benefit
            elif sweep > 50:
                base_ld *= 0.9  # High sweep penalty
        
        # Aspect ratio effect (simplified)
        aspect_ratio = geometry.wing_span**2 / (geometry.wing_span * geometry.wing_chord)
        if aspect_ratio > 8:
            base_ld *= 1.05
        elif aspect_ratio < 4:
            base_ld *= 0.95
        
        return max(5.0, base_ld)
    
    def _calculate_wave_drag_coefficient(self, geometry: GeometryModel,
                                       individual: Dict[str, float]) -> float:
        """Calculate wave drag coefficient for supersonic flight"""
        base_cd_wave = 0.02
        
        # Fineness ratio effect
        if 'fuselage_fineness_ratio' in individual:
            fineness = individual['fuselage_fineness_ratio']
            if fineness > 10:
                base_cd_wave *= 0.8  # Slender body reduces wave drag
            elif fineness < 6:
                base_cd_wave *= 1.3  # Blunt body increases wave drag
        
        # Wing sweep effect on wave drag
        if 'wing_sweep_angle' in individual:
            sweep = individual['wing_sweep_angle']
            if sweep > 45:
                base_cd_wave *= 0.7  # Swept wing reduces wave drag
        
        return max(0.005, base_cd_wave)
    
    def _calculate_induced_drag_coefficient(self, geometry: GeometryModel,
                                          individual: Dict[str, float]) -> float:
        """Calculate induced drag coefficient"""
        # Simplified induced drag calculation
        aspect_ratio = geometry.wing_span**2 / (geometry.wing_span * geometry.wing_chord)
        
        # Basic induced drag coefficient
        cd_induced = 0.05 / (np.pi * aspect_ratio * 0.8)  # Oswald efficiency factor = 0.8
        
        # Wing taper effect
        if 'wing_taper_ratio' in individual:
            taper = individual['wing_taper_ratio']
            if 0.2 <= taper <= 0.4:
                cd_induced *= 0.95  # Optimal taper reduces induced drag
            else:
                cd_induced *= 1.1
        
        return max(0.001, cd_induced)