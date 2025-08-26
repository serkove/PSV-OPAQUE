"""Tests for stealth shape optimizer module."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from fighter_jet_sdk.engines.aerodynamics.stealth_shape_optimizer import (
    StealthShapeOptimizer, ShapeParameter, OptimizationObjective, 
    ParetoPoint, OptimizationResults
)
from fighter_jet_sdk.common.data_models import AircraftConfiguration, Module, PhysicalProperties
from fighter_jet_sdk.common.enums import ModuleType, MaterialType


class TestStealthShapeOptimizer:
    """Test cases for StealthShapeOptimizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = StealthShapeOptimizer()
        
        # Create test aircraft configuration
        self.test_configuration = AircraftConfiguration(
            config_id="test_stealth_config",
            modules=[
                Module(
                    module_id="fuselage_main",
                    module_type=ModuleType.STRUCTURAL,
                    physical_properties=PhysicalProperties(
                        mass=2000.0,
                        center_of_gravity=(7.5, 0.0, 0.0),
                        moments_of_inertia=(5000.0, 15000.0, 15000.0),
                        dimensions=(15.0, 1.5, 1.5)
                    ),
                    performance_characteristics={}
                ),
                Module(
                    module_id="wing_main",
                    module_type=ModuleType.STRUCTURAL,
                    physical_properties=PhysicalProperties(
                        mass=1500.0,
                        center_of_gravity=(8.0, 0.0, 0.0),
                        moments_of_inertia=(2000.0, 8000.0, 10000.0),
                        dimensions=(12.0, 3.0, 0.3)
                    ),
                    performance_characteristics={}
                ),
                Module(
                    module_id="engine_main",
                    module_type=ModuleType.PROPULSION,
                    physical_properties=PhysicalProperties(
                        mass=800.0,
                        center_of_gravity=(12.0, 0.0, 0.0),
                        moments_of_inertia=(100.0, 500.0, 500.0),
                        dimensions=(2.0, 1.0, 1.0)
                    ),
                    performance_characteristics={'max_thrust': 120000.0}
                )
            ],
            mission_requirements={
                'max_speed': 'Mach 2.0',
                'stealth_requirement': 'low_observable'
            }
        )
        
        # Test optimization parameters
        self.test_params = {
            'optimize_geometry': True,
            'optimize_materials': True,
            'optimize_inlets': True,
            'rcs_weight': 0.6,
            'aero_weight': 0.4,
            'initial_wing_sweep': 35.0,
            'min_wing_sweep': 20.0,
            'max_wing_sweep': 60.0,
            'initial_wing_taper': 0.3,
            'initial_fineness_ratio': 8.0,
            'initial_ram_thickness': 0.005,
            'initial_surface_roughness': 1e-6,
            'initial_inlet_radius': 0.1
        }
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer.stealth_analyzer is not None
        assert self.optimizer.max_iterations == 200
        assert self.optimizer.population_size == 50
        assert self.optimizer.convergence_tolerance == 1e-6
        assert hasattr(self.optimizer, '_aero_models')
    
    def test_setup_shape_parameters(self):
        """Test shape parameter setup."""
        shape_params = self.optimizer._setup_shape_parameters(
            self.test_configuration, self.test_params
        )
        
        assert len(shape_params) > 0
        
        # Check for expected parameters
        param_names = [p.name for p in shape_params]
        assert 'wing_sweep_angle' in param_names
        assert 'wing_taper_ratio' in param_names
        assert 'fuselage_fineness_ratio' in param_names
        assert 'ram_coating_thickness' in param_names
        assert 'surface_roughness' in param_names
        assert 'inlet_lip_radius' in param_names
        
        # Verify parameter bounds
        for param in shape_params:
            assert param.min_value < param.max_value
            assert param.min_value <= param.current_value <= param.max_value
    
    def test_setup_optimization_objectives(self):
        """Test optimization objectives setup."""
        objectives = self.optimizer._setup_optimization_objectives(self.test_params)
        
        assert len(objectives) >= 2
        
        # Check for required objectives
        obj_names = [obj.name for obj in objectives]
        assert 'rcs_reduction' in obj_names
        assert 'aerodynamic_efficiency' in obj_names
        
        # Verify weights
        rcs_obj = next(obj for obj in objectives if obj.name == 'rcs_reduction')
        aero_obj = next(obj for obj in objectives if obj.name == 'aerodynamic_efficiency')
        
        assert rcs_obj.weight == 0.6
        assert aero_obj.weight == 0.4
        assert rcs_obj.constraint_type == 'maximize'
        assert aero_obj.constraint_type == 'maximize'
    
    def test_extract_geometry_model(self):
        """Test geometry model extraction."""
        geometry = self.optimizer._extract_geometry_model(self.test_configuration)
        
        assert geometry.fuselage_length > 0
        assert geometry.fuselage_diameter > 0
        assert geometry.wing_span > 0
        assert geometry.wing_chord > 0
        assert geometry.wing_thickness > 0
        assert geometry.engine_inlet_area > 0
        
        # Check surface materials
        assert 'fuselage' in geometry.surface_materials
        assert 'wing_upper' in geometry.surface_materials
        assert 'wing_lower' in geometry.surface_materials
    
    def test_get_materials_database(self):
        """Test materials database creation."""
        materials_db = self.optimizer._get_materials_database(self.test_configuration)
        
        assert len(materials_db) >= 3
        assert 'conventional_metal' in materials_db
        assert 'ram_coating' in materials_db
        assert 'metamaterial' in materials_db
        
        # Verify material properties
        metal = materials_db['conventional_metal']
        assert metal.base_material_type == MaterialType.CONVENTIONAL_METAL
        assert metal.electromagnetic_properties is not None
        
        ram = materials_db['ram_coating']
        assert ram.base_material_type == MaterialType.STEALTH_COATING
        
        metamat = materials_db['metamaterial']
        assert metamat.base_material_type == MaterialType.METAMATERIAL
    
    def test_initialize_population(self):
        """Test population initialization."""
        shape_params = [
            ShapeParameter('param1', 5.0, 0.0, 10.0, 'geometric', ['surface1']),
            ShapeParameter('param2', 0.5, 0.1, 1.0, 'material', ['surface2'])
        ]
        
        population = self.optimizer._initialize_population(shape_params, 10)
        
        assert len(population) == 10
        
        for individual in population:
            assert 'param1' in individual
            assert 'param2' in individual
            assert 0.0 <= individual['param1'] <= 10.0
            assert 0.1 <= individual['param2'] <= 1.0
    
    def test_apply_shape_parameters(self):
        """Test shape parameter application to geometry."""
        geometry = self.optimizer._extract_geometry_model(self.test_configuration)
        
        individual = {
            'wing_sweep_angle': 45.0,
            'wing_taper_ratio': 0.4,
            'fuselage_fineness_ratio': 12.0,  # Different from original 10.0
            'inlet_lip_radius': 0.15,
            'ram_coating_thickness': 0.01
        }
        
        shape_params = self.optimizer._setup_shape_parameters(
            self.test_configuration, self.test_params
        )
        
        updated_geometry = self.optimizer._apply_shape_parameters(
            geometry, individual, shape_params
        )
        
        # Verify geometry modifications
        assert updated_geometry.fuselage_length != geometry.fuselage_length
        assert updated_geometry.wing_span != geometry.wing_span
        
        # Check material updates
        assert 'ram_coating' in updated_geometry.surface_materials.values()
    
    def test_calculate_rcs_reduction(self):
        """Test RCS reduction calculation."""
        geometry = self.optimizer._extract_geometry_model(self.test_configuration)
        materials_db = self.optimizer._get_materials_database(self.test_configuration)
        
        # Mock the stealth analyzer to avoid complex calculations
        with patch.object(self.optimizer.stealth_analyzer, '_calculate_rcs_hybrid') as mock_rcs:
            mock_rcs.side_effect = [10.0, 5.0] * 12  # Baseline higher than optimized
            
            rcs_reduction = self.optimizer._calculate_rcs_reduction(geometry, materials_db)
            
            assert rcs_reduction > 0
            assert isinstance(rcs_reduction, float)
    
    def test_calculate_aerodynamic_efficiency(self):
        """Test aerodynamic efficiency calculation."""
        geometry = self.optimizer._extract_geometry_model(self.test_configuration)
        
        # Test with optimal parameters
        optimal_individual = {
            'wing_sweep_angle': 35.0,
            'wing_taper_ratio': 0.3,
            'fuselage_fineness_ratio': 9.0,
            'surface_roughness': 1e-7,
            'ram_coating_thickness': 0.002
        }
        
        efficiency = self.optimizer._calculate_aerodynamic_efficiency(
            geometry, optimal_individual
        )
        
        assert 0.1 <= efficiency <= 1.0
        
        # Test with suboptimal parameters
        suboptimal_individual = {
            'wing_sweep_angle': 70.0,  # Too high
            'wing_taper_ratio': 0.9,   # Too high
            'fuselage_fineness_ratio': 15.0,  # Too high
            'surface_roughness': 1e-4,  # Too rough
            'ram_coating_thickness': 0.02  # Too thick
        }
        
        efficiency_sub = self.optimizer._calculate_aerodynamic_efficiency(
            geometry, suboptimal_individual
        )
        
        assert efficiency_sub < efficiency
    
    def test_check_feasibility(self):
        """Test feasibility checking."""
        shape_params = [
            ShapeParameter('param1', 5.0, 0.0, 10.0, 'geometric', ['surface1']),
            ShapeParameter('param2', 0.5, 0.1, 1.0, 'material', ['surface2'])
        ]
        
        # Feasible individual
        feasible_individual = {'param1': 5.0, 'param2': 0.5}
        assert self.optimizer._check_feasibility(feasible_individual, shape_params)
        
        # Infeasible individual (out of bounds)
        infeasible_individual = {'param1': 15.0, 'param2': 0.5}
        assert not self.optimizer._check_feasibility(infeasible_individual, shape_params)
        
        # Test structural constraint
        structural_individual = {
            'wing_sweep_angle': 60.0,
            'wing_taper_ratio': 0.1  # Low taper with high sweep
        }
        shape_params_struct = [
            ShapeParameter('wing_sweep_angle', 35.0, 20.0, 70.0, 'geometric', ['wing']),
            ShapeParameter('wing_taper_ratio', 0.3, 0.1, 0.8, 'geometric', ['wing'])
        ]
        
        assert not self.optimizer._check_feasibility(structural_individual, shape_params_struct)
    
    def test_find_pareto_frontier(self):
        """Test Pareto frontier identification."""
        population = [
            ParetoPoint({}, {}, 10.0, 0.8, True),  # Good RCS, good aero
            ParetoPoint({}, {}, 15.0, 0.6, True),  # Better RCS, worse aero
            ParetoPoint({}, {}, 5.0, 0.9, True),   # Worse RCS, better aero
            ParetoPoint({}, {}, 8.0, 0.7, True),   # Dominated by first point
            ParetoPoint({}, {}, 12.0, 0.5, False)  # Infeasible
        ]
        
        pareto_frontier = self.optimizer._find_pareto_frontier(population)
        
        # Should have 3 points on frontier (excluding dominated and infeasible)
        assert len(pareto_frontier) == 3
        
        # Check that dominated point is not included
        rcs_values = [p.rcs_reduction_db for p in pareto_frontier]
        aero_values = [p.aerodynamic_efficiency for p in pareto_frontier]
        
        assert 8.0 not in rcs_values or 0.7 not in aero_values
    
    def test_tournament_selection(self):
        """Test tournament selection."""
        population = [
            ParetoPoint({}, {}, 10.0, 0.8, True),
            ParetoPoint({}, {}, 15.0, 0.6, True),
            ParetoPoint({}, {}, 5.0, 0.9, True)
        ]
        
        selected = self.optimizer._tournament_selection(population)
        
        assert selected in population
        assert selected.feasible
    
    def test_crossover(self):
        """Test crossover operation."""
        parent1 = {'param1': 1.0, 'param2': 2.0, 'param3': 3.0}
        parent2 = {'param1': 4.0, 'param2': 5.0, 'param3': 6.0}
        
        offspring = self.optimizer._crossover(parent1, parent2)
        
        assert len(offspring) == 3
        
        for param_name in parent1.keys():
            assert offspring[param_name] in [parent1[param_name], parent2[param_name]]
    
    def test_mutate(self):
        """Test mutation operation."""
        individual = {'param1': 5.0, 'param2': 0.5}
        shape_params = [
            ShapeParameter('param1', 5.0, 0.0, 10.0, 'geometric', ['surface1']),
            ShapeParameter('param2', 0.5, 0.1, 1.0, 'material', ['surface2'])
        ]
        
        # Set random seed for reproducible test
        np.random.seed(42)
        
        mutated = self.optimizer._mutate(individual, shape_params)
        
        assert len(mutated) == len(individual)
        
        # Values should be within bounds
        assert 0.0 <= mutated['param1'] <= 10.0
        assert 0.1 <= mutated['param2'] <= 1.0
    
    def test_select_best_compromise(self):
        """Test best compromise selection."""
        pareto_frontier = [
            ParetoPoint({}, {}, 10.0, 0.8, True),
            ParetoPoint({}, {}, 15.0, 0.6, True),
            ParetoPoint({}, {}, 5.0, 0.9, True)
        ]
        
        objectives = [
            OptimizationObjective('rcs_reduction', 0.6, constraint_type='maximize'),
            OptimizationObjective('aerodynamic_efficiency', 0.4, constraint_type='maximize')
        ]
        
        best = self.optimizer._select_best_compromise(pareto_frontier, objectives)
        
        assert best is not None
        assert best in pareto_frontier
    
    def test_check_convergence(self):
        """Test convergence checking."""
        # Create stable frontiers (converged)
        stable_frontiers = [
            [ParetoPoint({}, {}, 10.0, 0.8, True)] * 3,
            [ParetoPoint({}, {}, 10.1, 0.81, True)] * 3,
            [ParetoPoint({}, {}, 10.0, 0.8, True)] * 3,
            [ParetoPoint({}, {}, 10.1, 0.79, True)] * 3,
            [ParetoPoint({}, {}, 10.0, 0.8, True)] * 3
        ]
        
        assert self.optimizer._check_convergence(stable_frontiers)
        
        # Create unstable frontiers (not converged)
        unstable_frontiers = [
            [ParetoPoint({}, {}, 5.0, 0.5, True)] * 2,
            [ParetoPoint({}, {}, 15.0, 0.9, True)] * 5,
            [ParetoPoint({}, {}, 8.0, 0.7, True)] * 3
        ]
        
        assert not self.optimizer._check_convergence(unstable_frontiers)
    
    @patch('fighter_jet_sdk.engines.aerodynamics.stealth_shape_optimizer.StealthShapeOptimizer._run_multi_objective_optimization')
    def test_optimize_configuration(self, mock_optimization):
        """Test main optimization method."""
        # Mock the optimization results
        mock_results = OptimizationResults(
            pareto_frontier=[
                ParetoPoint({'param1': 1.0}, {'rcs_reduction': 10.0}, 10.0, 0.8, True)
            ],
            best_compromise=ParetoPoint({'param1': 1.0}, {'rcs_reduction': 10.0}, 10.0, 0.8, True),
            convergence_history=[{'generation': 0, 'pareto_size': 1}],
            optimization_statistics={'convergence_achieved': True}
        )
        mock_optimization.return_value = mock_results
        
        result = self.optimizer.optimize_configuration(
            self.test_configuration, self.test_params
        )
        
        assert 'pareto_frontier' in result
        assert 'best_compromise' in result
        assert 'convergence_history' in result
        assert 'optimization_statistics' in result
        assert 'optimization_successful' in result
        
        mock_optimization.assert_called_once()
    
    def test_calculate_structural_weight(self):
        """Test structural weight calculation."""
        geometry = self.optimizer._extract_geometry_model(self.test_configuration)
        
        # Test with minimal configuration
        minimal_individual = {
            'wing_sweep_angle': 30.0,
            'ram_coating_thickness': 0.001
        }
        
        weight_minimal = self.optimizer._calculate_structural_weight(
            geometry, minimal_individual
        )
        
        # Test with heavy configuration
        heavy_individual = {
            'wing_sweep_angle': 60.0,  # High sweep penalty
            'ram_coating_thickness': 0.02  # Thick coating
        }
        
        weight_heavy = self.optimizer._calculate_structural_weight(
            geometry, heavy_individual
        )
        
        assert weight_heavy > weight_minimal
        assert weight_minimal > 0
    
    def test_calculate_manufacturing_cost(self):
        """Test manufacturing cost calculation."""
        geometry = self.optimizer._extract_geometry_model(self.test_configuration)
        
        # Test with simple configuration
        simple_individual = {
            'wing_sweep_angle': 35.0,
            'ram_coating_thickness': 0.002
        }
        
        cost_simple = self.optimizer._calculate_manufacturing_cost(
            geometry, simple_individual
        )
        
        # Test with complex configuration
        complex_individual = {
            'wing_sweep_angle': 65.0,  # Complex geometry
            'ram_coating_thickness': 0.015  # Expensive coating
        }
        
        cost_complex = self.optimizer._calculate_manufacturing_cost(
            geometry, complex_individual
        )
        
        assert cost_complex > cost_simple
        assert cost_simple > 0


class TestShapeParameter:
    """Test cases for ShapeParameter dataclass."""
    
    def test_shape_parameter_creation(self):
        """Test shape parameter creation."""
        param = ShapeParameter(
            name='test_param',
            current_value=5.0,
            min_value=0.0,
            max_value=10.0,
            parameter_type='geometric',
            affected_surfaces=['surface1', 'surface2']
        )
        
        assert param.name == 'test_param'
        assert param.current_value == 5.0
        assert param.min_value == 0.0
        assert param.max_value == 10.0
        assert param.parameter_type == 'geometric'
        assert len(param.affected_surfaces) == 2


class TestOptimizationObjective:
    """Test cases for OptimizationObjective dataclass."""
    
    def test_optimization_objective_creation(self):
        """Test optimization objective creation."""
        obj = OptimizationObjective(
            name='test_objective',
            weight=0.5,
            target_value=100.0,
            constraint_type='target'
        )
        
        assert obj.name == 'test_objective'
        assert obj.weight == 0.5
        assert obj.target_value == 100.0
        assert obj.constraint_type == 'target'


class TestParetoPoint:
    """Test cases for ParetoPoint dataclass."""
    
    def test_pareto_point_creation(self):
        """Test Pareto point creation."""
        point = ParetoPoint(
            parameters={'param1': 1.0, 'param2': 2.0},
            objectives={'obj1': 10.0, 'obj2': 20.0},
            rcs_reduction_db=15.0,
            aerodynamic_efficiency=0.85,
            feasible=True
        )
        
        assert len(point.parameters) == 2
        assert len(point.objectives) == 2
        assert point.rcs_reduction_db == 15.0
        assert point.aerodynamic_efficiency == 0.85
        assert point.feasible is True


if __name__ == '__main__':
    pytest.main([__file__])