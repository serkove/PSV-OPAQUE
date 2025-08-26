"""Configuration optimizer for mission-specific designs."""

from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod

from ...common.data_models import AircraftConfiguration, Module, MissionRequirements
from ...common.enums import ModuleType
from ...core.errors import OptimizationError, ValidationError


class OptimizationObjective(Enum):
    """Optimization objectives."""
    MINIMIZE_WEIGHT = "minimize_weight"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_PERFORMANCE = "maximize_performance"
    MAXIMIZE_STEALTH = "maximize_stealth"
    MINIMIZE_POWER = "minimize_power"
    MAXIMIZE_RANGE = "maximize_range"


@dataclass
class OptimizationConstraint:
    """Represents an optimization constraint."""
    name: str
    constraint_type: str  # 'equality', 'inequality'
    target_value: float
    tolerance: float = 0.01
    weight: float = 1.0


@dataclass
class OptimizationResult:
    """Result of configuration optimization."""
    optimized_configuration: AircraftConfiguration
    objective_value: float
    constraint_violations: List[str]
    optimization_history: List[Dict[str, Any]]
    convergence_achieved: bool
    iterations: int
    computation_time: float


class ObjectiveFunction(ABC):
    """Base class for optimization objective functions."""
    
    @abstractmethod
    def evaluate(self, config: AircraftConfiguration) -> float:
        """Evaluate the objective function for a configuration."""
        pass
    
    @abstractmethod
    def get_gradient(self, config: AircraftConfiguration) -> Dict[str, float]:
        """Get gradient of objective function with respect to design variables."""
        pass


class WeightObjective(ObjectiveFunction):
    """Objective function for minimizing aircraft weight."""
    
    def evaluate(self, config: AircraftConfiguration) -> float:
        """Calculate total aircraft weight."""
        total_weight = 0.0
        
        if config.base_platform:
            total_weight += config.base_platform.base_mass
        
        for module in config.modules:
            if module.physical_properties:
                total_weight += module.physical_properties.mass
        
        return total_weight
    
    def get_gradient(self, config: AircraftConfiguration) -> Dict[str, float]:
        """Get weight gradient (simplified)."""
        gradient = {}
        
        for module in config.modules:
            if module.physical_properties:
                gradient[f"module_{module.module_id}_mass"] = 1.0
        
        return gradient


class PerformanceObjective(ObjectiveFunction):
    """Objective function for maximizing performance."""
    
    def __init__(self, performance_weights: Optional[Dict[str, float]] = None):
        """Initialize with performance metric weights."""
        self.performance_weights = performance_weights or {
            'thrust_to_weight_ratio': 0.4,
            'max_speed': 0.3,
            'range': 0.2,
            'stealth_rating': 0.1
        }
    
    def evaluate(self, config: AircraftConfiguration) -> float:
        """Calculate weighted performance score."""
        if not config.performance_envelope:
            return 0.0
        
        score = 0.0
        
        # Thrust-to-weight ratio contribution
        if config.performance_envelope.thrust_to_weight_ratio > 0:
            twr_score = min(config.performance_envelope.thrust_to_weight_ratio / 2.0, 1.0)
            score += twr_score * self.performance_weights.get('thrust_to_weight_ratio', 0.0)
        
        # Range contribution
        if config.performance_envelope.range > 0:
            range_score = min(config.performance_envelope.range / 3000.0, 1.0)  # Normalize to 3000km
            score += range_score * self.performance_weights.get('range', 0.0)
        
        # Stealth contribution (simplified)
        if config.performance_envelope.radar_cross_section:
            avg_rcs = np.mean(list(config.performance_envelope.radar_cross_section.values()))
            stealth_score = max(0.0, 1.0 - avg_rcs / 10.0)  # Normalize to 10 m²
            score += stealth_score * self.performance_weights.get('stealth_rating', 0.0)
        
        return score
    
    def get_gradient(self, config: AircraftConfiguration) -> Dict[str, float]:
        """Get performance gradient (simplified)."""
        gradient = {}
        
        # This would be more complex in a real implementation
        for module in config.modules:
            gradient[f"module_{module.module_id}_performance"] = 0.1
        
        return gradient


class ConfigurationOptimizer:
    """Optimizer for mission-specific aircraft configurations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the configuration optimizer.
        
        Args:
            config: Optimizer configuration
        """
        self.config = config or {}
        
        # Optimization parameters
        self.max_iterations = self.config.get('max_iterations', 100)
        self.convergence_tolerance = self.config.get('convergence_tolerance', 1e-6)
        self.population_size = self.config.get('population_size', 50)
        self.mutation_rate = self.config.get('mutation_rate', 0.1)
        
        # Optimization state
        self.objectives: List[ObjectiveFunction] = []
        self.constraints: List[OptimizationConstraint] = []
        self.design_variables: Dict[str, Any] = {}
        
        # Results tracking
        self.optimization_history: List[Dict[str, Any]] = []
        self.best_configuration: Optional[AircraftConfiguration] = None
        self.best_objective_value: float = float('inf')
    
    def add_objective(self, objective: ObjectiveFunction, weight: float = 1.0) -> None:
        """Add an objective function to the optimization.
        
        Args:
            objective: Objective function to add
            weight: Weight for multi-objective optimization
        """
        objective.weight = weight
        self.objectives.append(objective)
    
    def add_constraint(self, constraint: OptimizationConstraint) -> None:
        """Add a constraint to the optimization.
        
        Args:
            constraint: Constraint to add
        """
        self.constraints.append(constraint)
    
    def set_design_variables(self, variables: Dict[str, Any]) -> None:
        """Set design variables for optimization.
        
        Args:
            variables: Dictionary of design variables and their bounds
        """
        self.design_variables = variables
    
    def optimize_for_mission(self, base_config: AircraftConfiguration,
                           mission_requirements: MissionRequirements,
                           available_modules: List[Module]) -> OptimizationResult:
        """Optimize configuration for specific mission requirements.
        
        Args:
            base_config: Base aircraft configuration
            mission_requirements: Mission-specific requirements
            available_modules: List of available modules to choose from
            
        Returns:
            OptimizationResult with optimized configuration
        """
        import time
        start_time = time.time()
        
        # Set up mission-specific objectives and constraints
        self._setup_mission_optimization(mission_requirements)
        
        # Validate inputs
        if not available_modules:
            raise OptimizationError("No modules available for optimization")
        
        if self.max_iterations <= 0:
            raise OptimizationError("Invalid max_iterations: must be positive")
        
        # Run optimization algorithm
        try:
            if self.config.get('algorithm', 'genetic') == 'genetic':
                result_config = self._genetic_algorithm_optimization(
                    base_config, available_modules
                )
            else:
                result_config = self._gradient_based_optimization(
                    base_config, available_modules
                )
            
            # Evaluate final configuration
            final_objective = self._evaluate_configuration(result_config)
            constraint_violations = self._check_constraints(result_config)
            
            computation_time = time.time() - start_time
            
            return OptimizationResult(
                optimized_configuration=result_config,
                objective_value=final_objective,
                constraint_violations=constraint_violations,
                optimization_history=self.optimization_history.copy(),
                convergence_achieved=len(constraint_violations) == 0,
                iterations=len(self.optimization_history),
                computation_time=computation_time
            )
            
        except Exception as e:
            raise OptimizationError(f"Optimization failed: {e}")
    
    def multi_objective_optimization(self, base_config: AircraftConfiguration,
                                   objectives: List[Tuple[ObjectiveFunction, float]],
                                   available_modules: List[Module]) -> List[OptimizationResult]:
        """Perform multi-objective optimization using Pareto frontier analysis.
        
        Args:
            base_config: Base aircraft configuration
            objectives: List of (objective_function, weight) tuples
            available_modules: Available modules for optimization
            
        Returns:
            List of Pareto-optimal solutions
        """
        # Clear existing objectives and add new ones
        self.objectives.clear()
        for obj_func, weight in objectives:
            self.add_objective(obj_func, weight)
        
        # Generate multiple optimization runs with different objective weights
        pareto_solutions = []
        
        # Create weight combinations for Pareto frontier
        num_objectives = len(objectives)
        weight_combinations = self._generate_weight_combinations(num_objectives)
        
        for weights in weight_combinations:
            # Update objective weights
            for i, (obj_func, _) in enumerate(objectives):
                obj_func.weight = weights[i]
            
            # Run optimization
            try:
                result = self.optimize_for_mission(
                    base_config, 
                    MissionRequirements(),  # Default requirements
                    available_modules
                )
                pareto_solutions.append(result)
            except OptimizationError:
                continue  # Skip failed optimizations
        
        # Filter for Pareto-optimal solutions
        pareto_optimal = self._filter_pareto_optimal(pareto_solutions)
        
        return pareto_optimal
    
    def constraint_satisfaction_solve(self, base_config: AircraftConfiguration,
                                    constraints: List[OptimizationConstraint],
                                    available_modules: List[Module]) -> Optional[AircraftConfiguration]:
        """Solve constraint satisfaction problem for module placement.
        
        Args:
            base_config: Base configuration
            constraints: List of constraints to satisfy
            available_modules: Available modules
            
        Returns:
            Configuration satisfying all constraints, or None if no solution exists
        """
        # Set constraints
        self.constraints = constraints
        
        # Use backtracking algorithm for constraint satisfaction
        return self._backtrack_constraint_satisfaction(
            base_config, available_modules, 0
        )
    
    def _setup_mission_optimization(self, mission_requirements: MissionRequirements) -> None:
        """Set up optimization objectives and constraints based on mission requirements."""
        # Clear existing objectives and constraints
        self.objectives.clear()
        self.constraints.clear()
        
        # Add mission-specific objectives
        if mission_requirements.mission_type == "air_superiority":
            self.add_objective(PerformanceObjective({
                'thrust_to_weight_ratio': 0.5,
                'max_speed': 0.3,
                'stealth_rating': 0.2
            }), weight=1.0)
        elif mission_requirements.mission_type == "strike":
            self.add_objective(PerformanceObjective({
                'range': 0.4,
                'stealth_rating': 0.4,
                'thrust_to_weight_ratio': 0.2
            }), weight=1.0)
        else:
            # Default balanced optimization
            self.add_objective(WeightObjective(), weight=0.3)
            self.add_objective(PerformanceObjective(), weight=0.7)
        
        # Add mission-specific constraints
        if mission_requirements.range_requirement > 0:
            self.add_constraint(OptimizationConstraint(
                name="range_requirement",
                constraint_type="inequality",
                target_value=mission_requirements.range_requirement,
                tolerance=0.05
            ))
        
        if mission_requirements.payload_requirement > 0:
            self.add_constraint(OptimizationConstraint(
                name="payload_requirement",
                constraint_type="inequality",
                target_value=mission_requirements.payload_requirement,
                tolerance=0.02
            ))
    
    def _genetic_algorithm_optimization(self, base_config: AircraftConfiguration,
                                     available_modules: List[Module]) -> AircraftConfiguration:
        """Run genetic algorithm optimization."""
        # Initialize population
        population = self._initialize_population(base_config, available_modules)
        
        for iteration in range(self.max_iterations):
            # Evaluate population
            fitness_scores = [self._evaluate_configuration(config) for config in population]
            
            # Track best solution
            best_idx = np.argmin(fitness_scores)
            if fitness_scores[best_idx] < self.best_objective_value:
                self.best_objective_value = fitness_scores[best_idx]
                self.best_configuration = population[best_idx]
            
            # Record iteration
            self.optimization_history.append({
                'iteration': iteration,
                'best_objective': self.best_objective_value,
                'population_diversity': self._calculate_diversity(population)
            })
            
            # Check convergence
            if iteration > 10:
                recent_improvements = [
                    self.optimization_history[i]['best_objective'] 
                    for i in range(max(0, iteration-10), iteration)
                ]
                if max(recent_improvements) - min(recent_improvements) < self.convergence_tolerance:
                    break
            
            # Selection, crossover, and mutation
            population = self._evolve_population(population, fitness_scores, available_modules)
        
        return self.best_configuration or base_config
    
    def _gradient_based_optimization(self, base_config: AircraftConfiguration,
                                   available_modules: List[Module]) -> AircraftConfiguration:
        """Run gradient-based optimization (simplified implementation)."""
        current_config = base_config
        
        for iteration in range(self.max_iterations):
            # Calculate objective and gradients
            current_objective = self._evaluate_configuration(current_config)
            
            # Record iteration
            self.optimization_history.append({
                'iteration': iteration,
                'objective_value': current_objective
            })
            
            # Simple hill-climbing approach (placeholder)
            # In a real implementation, this would use proper gradient descent
            improved_config = self._local_search_improvement(current_config, available_modules)
            
            if improved_config:
                improved_objective = self._evaluate_configuration(improved_config)
                if improved_objective < current_objective:
                    current_config = improved_config
                else:
                    break  # No improvement found
            else:
                break
        
        return current_config
    
    def _evaluate_configuration(self, config: AircraftConfiguration) -> float:
        """Evaluate configuration using all objective functions."""
        total_objective = 0.0
        
        for objective in self.objectives:
            obj_value = objective.evaluate(config)
            weight = getattr(objective, 'weight', 1.0)
            total_objective += weight * obj_value
        
        # Add penalty for constraint violations
        violations = self._check_constraints(config)
        penalty = len(violations) * 1000.0  # Large penalty for violations
        
        return total_objective + penalty
    
    def _check_constraints(self, config: AircraftConfiguration) -> List[str]:
        """Check constraint violations for a configuration."""
        violations = []
        
        for constraint in self.constraints:
            if constraint.name == "range_requirement":
                if config.performance_envelope and config.performance_envelope.range > 0:
                    if config.performance_envelope.range < constraint.target_value:
                        violations.append(f"Range requirement not met: {config.performance_envelope.range} < {constraint.target_value}")
            
            elif constraint.name == "payload_requirement":
                total_payload = sum(
                    module.performance_characteristics.get('max_payload_mass', 0.0)
                    for module in config.modules
                    if module.module_type == ModuleType.PAYLOAD
                )
                if total_payload < constraint.target_value:
                    violations.append(f"Payload requirement not met: {total_payload} < {constraint.target_value}")
            
            elif constraint.name == "impossible_weight":
                # Calculate total weight
                total_weight = 0.0
                if config.base_platform:
                    total_weight += config.base_platform.base_mass
                for module in config.modules:
                    if module.physical_properties:
                        total_weight += module.physical_properties.mass
                
                if total_weight > constraint.target_value:
                    violations.append(f"Weight constraint violated: {total_weight} > {constraint.target_value}")
        
        return violations
    
    def _initialize_population(self, base_config: AircraftConfiguration,
                             available_modules: List[Module]) -> List[AircraftConfiguration]:
        """Initialize population for genetic algorithm."""
        population = []
        
        for _ in range(self.population_size):
            # Create random configuration by selecting random modules
            config = AircraftConfiguration(
                name=f"Individual_{len(population)}",
                base_platform=base_config.base_platform,
                modules=[]
            )
            
            # Randomly select modules (simplified)
            num_modules = min(5, len(available_modules))  # Limit to 5 modules
            selected_modules = np.random.choice(available_modules, num_modules, replace=False)
            
            for module in selected_modules:
                config.modules.append(module)
            
            population.append(config)
        
        return population
    
    def _evolve_population(self, population: List[AircraftConfiguration],
                         fitness_scores: List[float],
                         available_modules: List[Module]) -> List[AircraftConfiguration]:
        """Evolve population through selection, crossover, and mutation."""
        new_population = []
        
        # Selection (tournament selection)
        for _ in range(self.population_size):
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)
            
            # Crossover
            child = self._crossover(parent1, parent2)
            
            # Mutation
            if np.random.random() < self.mutation_rate:
                child = self._mutate(child, available_modules)
            
            new_population.append(child)
        
        return new_population
    
    def _tournament_selection(self, population: List[AircraftConfiguration],
                            fitness_scores: List[float]) -> AircraftConfiguration:
        """Tournament selection for genetic algorithm."""
        tournament_size = min(3, len(population))  # Ensure tournament size doesn't exceed population
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx]
    
    def _crossover(self, parent1: AircraftConfiguration,
                  parent2: AircraftConfiguration) -> AircraftConfiguration:
        """Crossover operation for genetic algorithm."""
        child = AircraftConfiguration(
            name="Child",
            base_platform=parent1.base_platform,
            modules=[]
        )
        
        # Simple crossover: randomly select modules from both parents
        all_modules = parent1.modules + parent2.modules
        unique_modules = {m.module_id: m for m in all_modules}.values()
        
        for module in unique_modules:
            if np.random.random() < 0.5:
                child.modules.append(module)
        
        return child
    
    def _mutate(self, config: AircraftConfiguration,
               available_modules: List[Module]) -> AircraftConfiguration:
        """Mutation operation for genetic algorithm."""
        mutated = AircraftConfiguration(
            name=config.name,
            base_platform=config.base_platform,
            modules=config.modules.copy()
        )
        
        # Random mutation: add, remove, or replace a module
        mutation_type = np.random.choice(['add', 'remove', 'replace'])
        
        if mutation_type == 'add' and len(mutated.modules) < 10:
            new_module = np.random.choice(available_modules)
            if new_module not in mutated.modules:
                mutated.modules.append(new_module)
        
        elif mutation_type == 'remove' and len(mutated.modules) > 1:
            remove_idx = np.random.randint(len(mutated.modules))
            mutated.modules.pop(remove_idx)
        
        elif mutation_type == 'replace' and len(mutated.modules) > 0:
            replace_idx = np.random.randint(len(mutated.modules))
            new_module = np.random.choice(available_modules)
            mutated.modules[replace_idx] = new_module
        
        return mutated
    
    def _calculate_diversity(self, population: List[AircraftConfiguration]) -> float:
        """Calculate population diversity metric."""
        if len(population) < 2:
            return 0.0
        
        # Simple diversity metric based on number of unique modules
        all_module_ids = set()
        for config in population:
            for module in config.modules:
                all_module_ids.add(module.module_id)
        
        return len(all_module_ids) / len(population)
    
    def _local_search_improvement(self, config: AircraftConfiguration,
                                available_modules: List[Module]) -> Optional[AircraftConfiguration]:
        """Perform local search for configuration improvement."""
        # Try replacing each module with alternatives
        for i, current_module in enumerate(config.modules):
            # Find modules of the same type
            same_type_modules = [
                m for m in available_modules 
                if m.module_type == current_module.module_type and m.module_id != current_module.module_id
            ]
            
            for alternative in same_type_modules:
                # Create new configuration with replacement
                new_config = AircraftConfiguration(
                    name=config.name,
                    base_platform=config.base_platform,
                    modules=config.modules.copy()
                )
                new_config.modules[i] = alternative
                
                # Check if this is an improvement
                if self._evaluate_configuration(new_config) < self._evaluate_configuration(config):
                    return new_config
        
        return None
    
    def _generate_weight_combinations(self, num_objectives: int) -> List[List[float]]:
        """Generate weight combinations for multi-objective optimization."""
        combinations = []
        
        # Generate uniform weight distributions
        for i in range(11):  # 0.0, 0.1, 0.2, ..., 1.0
            weights = [0.0] * num_objectives
            if num_objectives > 0:
                weights[0] = i / 10.0
                if num_objectives > 1:
                    weights[1] = (10 - i) / 10.0
            combinations.append(weights)
        
        return combinations
    
    def _filter_pareto_optimal(self, solutions: List[OptimizationResult]) -> List[OptimizationResult]:
        """Filter solutions to keep only Pareto-optimal ones."""
        pareto_optimal = []
        
        for i, solution1 in enumerate(solutions):
            is_dominated = False
            
            for j, solution2 in enumerate(solutions):
                if i != j and self._dominates(solution2, solution1):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_optimal.append(solution1)
        
        return pareto_optimal
    
    def _dominates(self, solution1: OptimizationResult, solution2: OptimizationResult) -> bool:
        """Check if solution1 dominates solution2 in Pareto sense."""
        # Simplified domination check based on objective value and constraint violations
        return (solution1.objective_value <= solution2.objective_value and
                len(solution1.constraint_violations) <= len(solution2.constraint_violations) and
                (solution1.objective_value < solution2.objective_value or
                 len(solution1.constraint_violations) < len(solution2.constraint_violations)))
    
    def _backtrack_constraint_satisfaction(self, config: AircraftConfiguration,
                                         available_modules: List[Module],
                                         module_index: int) -> Optional[AircraftConfiguration]:
        """Backtracking algorithm for constraint satisfaction."""
        # Base case: all modules assigned
        if module_index >= len(available_modules):
            violations = self._check_constraints(config)
            return config if len(violations) == 0 else None
        
        # Try assigning current module
        current_module = available_modules[module_index]
        
        # Try adding the module
        test_config = AircraftConfiguration(
            name=config.name,
            base_platform=config.base_platform,
            modules=config.modules + [current_module]
        )
        
        # Check if this assignment is consistent with constraints
        if self._is_consistent(test_config):
            result = self._backtrack_constraint_satisfaction(
                test_config, available_modules, module_index + 1
            )
            if result:
                return result
        
        # Try not adding the module
        result = self._backtrack_constraint_satisfaction(
            config, available_modules, module_index + 1
        )
        
        return result
    
    def _is_consistent(self, config: AircraftConfiguration) -> bool:
        """Check if current configuration is consistent with constraints."""
        # Simplified consistency check
        violations = self._check_constraints(config)
        return len(violations) == 0