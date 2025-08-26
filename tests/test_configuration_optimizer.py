"""Tests for the Configuration Optimizer."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from fighter_jet_sdk.engines.design.configuration_optimizer import (
    ConfigurationOptimizer, OptimizationObjective, OptimizationConstraint,
    OptimizationResult, WeightObjective, PerformanceObjective
)
from fighter_jet_sdk.common.data_models import (
    AircraftConfiguration, BasePlatform, Module, MissionRequirements,
    PhysicalProperties, PerformanceEnvelope
)
from fighter_jet_sdk.common.enums import ModuleType, FlightRegime
from fighter_jet_sdk.core.errors import OptimizationError


class TestObjectiveFunctions:
    """Test cases for objective functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create test configuration
        self.platform = BasePlatform(
            name="Test Platform",
            base_mass=5000.0,
            power_generation_capacity=20000.0
        )
        
        self.test_module = Module(
            name="Test Module",
            module_type=ModuleType.SENSOR,
            physical_properties=PhysicalProperties(
                mass=200.0,
                center_of_gravity=(0.0, 0.0, 0.0),
                moments_of_inertia=(50.0, 50.0, 25.0),
                dimensions=(1.0, 1.0, 0.3)
            )
        )
        
        self.config = AircraftConfiguration(
            name="Test Configuration",
            base_platform=self.platform,
            modules=[self.test_module],
            performance_envelope=PerformanceEnvelope(
                max_speed={FlightRegime.SUPERSONIC: 2.0},
                service_ceiling=15000.0,
                range=2000.0,
                thrust_to_weight_ratio=1.2,
                radar_cross_section={'X-band': 0.5}
            )
        )
    
    def test_weight_objective_evaluation(self):
        """Test weight objective function evaluation."""
        weight_obj = WeightObjective()
        
        total_weight = weight_obj.evaluate(self.config)
        
        # Should be platform mass + module mass
        expected_weight = 5000.0 + 200.0
        assert total_weight == expected_weight
    
    def test_weight_objective_gradient(self):
        """Test weight objective gradient calculation."""
        weight_obj = WeightObjective()
        
        gradient = weight_obj.get_gradient(self.config)
        
        # Should have gradient entry for the module
        assert f"module_{self.test_module.module_id}_mass" in gradient
        assert gradient[f"module_{self.test_module.module_id}_mass"] == 1.0
    
    def test_performance_objective_evaluation(self):
        """Test performance objective function evaluation."""
        perf_obj = PerformanceObjective()
        
        score = perf_obj.evaluate(self.config)
        
        # Should return a score between 0 and 1
        assert 0.0 <= score <= 1.0
    
    def test_performance_objective_no_envelope(self):
        """Test performance objective with no performance envelope."""
        perf_obj = PerformanceObjective()
        
        config_no_perf = AircraftConfiguration(
            name="No Performance",
            base_platform=self.platform,
            modules=[self.test_module]
        )
        
        score = perf_obj.evaluate(config_no_perf)
        assert score == 0.0
    
    def test_performance_objective_gradient(self):
        """Test performance objective gradient calculation."""
        perf_obj = PerformanceObjective()
        
        gradient = perf_obj.get_gradient(self.config)
        
        # Should have gradient entries for modules
        assert f"module_{self.test_module.module_id}_performance" in gradient


class TestConfigurationOptimizer:
    """Test cases for ConfigurationOptimizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = ConfigurationOptimizer({
            'max_iterations': 10,  # Small number for testing
            'population_size': 5,
            'convergence_tolerance': 1e-3
        })
        
        # Create test data
        self.platform = BasePlatform(
            name="Test Platform",
            base_mass=5000.0,
            power_generation_capacity=20000.0
        )
        
        self.modules = [
            Module(
                name="Light Sensor",
                module_type=ModuleType.SENSOR,
                physical_properties=PhysicalProperties(
                    mass=100.0,
                    center_of_gravity=(0.0, 0.0, 0.0),
                    moments_of_inertia=(25.0, 25.0, 12.5),
                    dimensions=(0.5, 0.5, 0.2)
                ),
                performance_characteristics={'detection_range': 100.0}
            ),
            Module(
                name="Heavy Sensor",
                module_type=ModuleType.SENSOR,
                physical_properties=PhysicalProperties(
                    mass=300.0,
                    center_of_gravity=(0.0, 0.0, 0.0),
                    moments_of_inertia=(75.0, 75.0, 37.5),
                    dimensions=(1.5, 1.5, 0.6)
                ),
                performance_characteristics={'detection_range': 200.0}
            ),
            Module(
                name="Light Payload",
                module_type=ModuleType.PAYLOAD,
                physical_properties=PhysicalProperties(
                    mass=200.0,
                    center_of_gravity=(0.0, 0.0, 0.0),
                    moments_of_inertia=(50.0, 50.0, 25.0),
                    dimensions=(1.0, 1.0, 0.5)
                ),
                performance_characteristics={'max_payload_mass': 500.0}
            )
        ]
        
        self.base_config = AircraftConfiguration(
            name="Base Configuration",
            base_platform=self.platform,
            modules=[self.modules[0]]  # Start with light sensor
        )
        
        self.mission_requirements = MissionRequirements(
            mission_type="reconnaissance",
            duration=4.0,
            range_requirement=1500.0,
            payload_requirement=300.0
        )
    
    def test_add_objective(self):
        """Test adding objective functions."""
        weight_obj = WeightObjective()
        
        self.optimizer.add_objective(weight_obj, weight=0.5)
        
        assert len(self.optimizer.objectives) == 1
        assert self.optimizer.objectives[0] == weight_obj
        assert self.optimizer.objectives[0].weight == 0.5
    
    def test_add_constraint(self):
        """Test adding constraints."""
        constraint = OptimizationConstraint(
            name="weight_limit",
            constraint_type="inequality",
            target_value=6000.0
        )
        
        self.optimizer.add_constraint(constraint)
        
        assert len(self.optimizer.constraints) == 1
        assert self.optimizer.constraints[0] == constraint
    
    def test_set_design_variables(self):
        """Test setting design variables."""
        variables = {
            'module_selection': {'type': 'discrete', 'options': [0, 1, 2]},
            'module_count': {'type': 'integer', 'min': 1, 'max': 5}
        }
        
        self.optimizer.set_design_variables(variables)
        
        assert self.optimizer.design_variables == variables
    
    def test_optimize_for_mission_basic(self):
        """Test basic mission optimization."""
        # Add objectives
        self.optimizer.add_objective(WeightObjective(), weight=0.6)
        self.optimizer.add_objective(PerformanceObjective(), weight=0.4)
        
        # Run optimization
        result = self.optimizer.optimize_for_mission(
            self.base_config, self.mission_requirements, self.modules
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.optimized_configuration is not None
        assert result.objective_value >= 0.0
        assert result.iterations > 0
        assert result.computation_time > 0.0
    
    def test_optimize_air_superiority_mission(self):
        """Test optimization for air superiority mission."""
        air_superiority_mission = MissionRequirements(
            mission_type="air_superiority",
            duration=2.0,
            range_requirement=800.0
        )
        
        result = self.optimizer.optimize_for_mission(
            self.base_config, air_superiority_mission, self.modules
        )
        
        # Should have performance-focused objectives
        assert len(self.optimizer.objectives) > 0
        assert isinstance(result, OptimizationResult)
    
    def test_optimize_strike_mission(self):
        """Test optimization for strike mission."""
        strike_mission = MissionRequirements(
            mission_type="strike",
            duration=6.0,
            range_requirement=2000.0,
            payload_requirement=1000.0
        )
        
        result = self.optimizer.optimize_for_mission(
            self.base_config, strike_mission, self.modules
        )
        
        # Should have range and stealth focused objectives
        assert len(self.optimizer.objectives) > 0
        assert isinstance(result, OptimizationResult)
    
    def test_multi_objective_optimization(self):
        """Test multi-objective optimization."""
        objectives = [
            (WeightObjective(), 0.5),
            (PerformanceObjective(), 0.5)
        ]
        
        results = self.optimizer.multi_objective_optimization(
            self.base_config, objectives, self.modules
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        # All results should be OptimizationResult instances
        for result in results:
            assert isinstance(result, OptimizationResult)
    
    def test_constraint_satisfaction_solve(self):
        """Test constraint satisfaction solving."""
        constraints = [
            OptimizationConstraint(
                name="weight_limit",
                constraint_type="inequality",
                target_value=6000.0
            ),
            OptimizationConstraint(
                name="payload_requirement",
                constraint_type="inequality",
                target_value=400.0
            )
        ]
        
        result_config = self.optimizer.constraint_satisfaction_solve(
            self.base_config, constraints, self.modules
        )
        
        # May return None if no solution exists, or a valid configuration
        if result_config is not None:
            assert isinstance(result_config, AircraftConfiguration)
    
    def test_evaluate_configuration(self):
        """Test configuration evaluation."""
        # Add objectives
        self.optimizer.add_objective(WeightObjective(), weight=1.0)
        
        objective_value = self.optimizer._evaluate_configuration(self.base_config)
        
        assert objective_value >= 0.0
    
    def test_check_constraints_range(self):
        """Test range constraint checking."""
        # Add range constraint
        self.optimizer.add_constraint(OptimizationConstraint(
            name="range_requirement",
            constraint_type="inequality",
            target_value=2000.0
        ))
        
        # Create config with performance envelope
        config_with_perf = AircraftConfiguration(
            name="Test Config",
            base_platform=self.platform,
            modules=self.modules,
            performance_envelope=PerformanceEnvelope(range=1500.0)  # Below requirement
        )
        
        violations = self.optimizer._check_constraints(config_with_perf)
        
        # Should have range violation
        assert len(violations) > 0
        assert any("Range requirement not met" in v for v in violations)
    
    def test_check_constraints_payload(self):
        """Test payload constraint checking."""
        # Add payload constraint
        self.optimizer.add_constraint(OptimizationConstraint(
            name="payload_requirement",
            constraint_type="inequality",
            target_value=1000.0  # High requirement
        ))
        
        violations = self.optimizer._check_constraints(self.base_config)
        
        # Should have payload violation (base config has sensor, not payload module)
        assert len(violations) > 0
        assert any("Payload requirement not met" in v for v in violations)
    
    def test_initialize_population(self):
        """Test population initialization for genetic algorithm."""
        population = self.optimizer._initialize_population(self.base_config, self.modules)
        
        assert len(population) == self.optimizer.population_size
        
        # All should be AircraftConfiguration instances
        for individual in population:
            assert isinstance(individual, AircraftConfiguration)
            assert individual.base_platform == self.base_config.base_platform
    
    def test_tournament_selection(self):
        """Test tournament selection."""
        population = [self.base_config] * 5  # Simple population
        fitness_scores = [1.0, 2.0, 0.5, 3.0, 1.5]  # Different fitness values
        
        selected = self.optimizer._tournament_selection(population, fitness_scores)
        
        assert isinstance(selected, AircraftConfiguration)
    
    def test_crossover(self):
        """Test crossover operation."""
        parent1 = AircraftConfiguration(
            name="Parent1",
            base_platform=self.platform,
            modules=[self.modules[0]]
        )
        
        parent2 = AircraftConfiguration(
            name="Parent2", 
            base_platform=self.platform,
            modules=[self.modules[1]]
        )
        
        child = self.optimizer._crossover(parent1, parent2)
        
        assert isinstance(child, AircraftConfiguration)
        assert child.base_platform == self.platform
    
    def test_mutate(self):
        """Test mutation operation."""
        config = AircraftConfiguration(
            name="Original",
            base_platform=self.platform,
            modules=[self.modules[0]]
        )
        
        mutated = self.optimizer._mutate(config, self.modules)
        
        assert isinstance(mutated, AircraftConfiguration)
        assert mutated.base_platform == self.platform
    
    def test_calculate_diversity(self):
        """Test population diversity calculation."""
        population = [
            AircraftConfiguration(name="Config1", base_platform=self.platform, modules=[self.modules[0]]),
            AircraftConfiguration(name="Config2", base_platform=self.platform, modules=[self.modules[1]]),
            AircraftConfiguration(name="Config3", base_platform=self.platform, modules=[self.modules[0], self.modules[1]])
        ]
        
        diversity = self.optimizer._calculate_diversity(population)
        
        assert 0.0 <= diversity <= len(self.modules)
    
    def test_local_search_improvement(self):
        """Test local search improvement."""
        improved = self.optimizer._local_search_improvement(self.base_config, self.modules)
        
        # May return None if no improvement found, or an improved configuration
        if improved is not None:
            assert isinstance(improved, AircraftConfiguration)
    
    def test_generate_weight_combinations(self):
        """Test weight combination generation for multi-objective optimization."""
        combinations = self.optimizer._generate_weight_combinations(2)
        
        assert len(combinations) > 0
        
        # Each combination should have 2 weights
        for combo in combinations:
            assert len(combo) == 2
            # Weights should be non-negative
            assert all(w >= 0.0 for w in combo)
    
    def test_filter_pareto_optimal(self):
        """Test Pareto optimal filtering."""
        # Create mock results with different objective values
        results = [
            OptimizationResult(
                optimized_configuration=self.base_config,
                objective_value=1.0,
                constraint_violations=[],
                optimization_history=[],
                convergence_achieved=True,
                iterations=10,
                computation_time=1.0
            ),
            OptimizationResult(
                optimized_configuration=self.base_config,
                objective_value=2.0,
                constraint_violations=[],
                optimization_history=[],
                convergence_achieved=True,
                iterations=10,
                computation_time=1.0
            ),
            OptimizationResult(
                optimized_configuration=self.base_config,
                objective_value=0.5,
                constraint_violations=[],
                optimization_history=[],
                convergence_achieved=True,
                iterations=10,
                computation_time=1.0
            )
        ]
        
        pareto_optimal = self.optimizer._filter_pareto_optimal(results)
        
        assert len(pareto_optimal) <= len(results)
        
        # All results should be OptimizationResult instances
        for result in pareto_optimal:
            assert isinstance(result, OptimizationResult)
    
    def test_dominates(self):
        """Test Pareto domination check."""
        result1 = OptimizationResult(
            optimized_configuration=self.base_config,
            objective_value=1.0,
            constraint_violations=[],
            optimization_history=[],
            convergence_achieved=True,
            iterations=10,
            computation_time=1.0
        )
        
        result2 = OptimizationResult(
            optimized_configuration=self.base_config,
            objective_value=2.0,
            constraint_violations=["violation"],
            optimization_history=[],
            convergence_achieved=True,
            iterations=10,
            computation_time=1.0
        )
        
        # result1 should dominate result2 (better objective, fewer violations)
        assert self.optimizer._dominates(result1, result2)
        assert not self.optimizer._dominates(result2, result1)
    
    def test_optimization_with_constraints_violations(self):
        """Test constraint checking with impossible constraints."""
        # Add impossible weight constraint
        constraint = OptimizationConstraint(
            name="impossible_weight",
            constraint_type="inequality", 
            target_value=1.0  # Impossible weight limit - even platform alone is 5000kg
        )
        self.optimizer.add_constraint(constraint)
        
        # Test constraint checking directly
        violations = self.optimizer._check_constraints(self.base_config)
        assert len(violations) > 0, "Base configuration should violate weight constraint"
        assert "Weight constraint violated" in violations[0]
        
        # Test with a configuration that has multiple modules
        heavy_config = AircraftConfiguration(
            name="Heavy Config",
            base_platform=self.platform,
            modules=self.modules  # All modules
        )
        
        heavy_violations = self.optimizer._check_constraints(heavy_config)
        assert len(heavy_violations) > 0, "Heavy configuration should also violate weight constraint"
    
    def test_optimization_error_handling(self):
        """Test optimization error handling."""
        # Create optimizer with invalid configuration
        bad_optimizer = ConfigurationOptimizer({'max_iterations': -1})
        
        # Should handle errors gracefully
        with pytest.raises(OptimizationError):
            bad_optimizer.optimize_for_mission(
                self.base_config, self.mission_requirements, []  # Empty modules list
            )


if __name__ == "__main__":
    pytest.main([__file__])