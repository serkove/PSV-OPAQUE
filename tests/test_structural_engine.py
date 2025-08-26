"""Tests for structural analysis engine."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from fighter_jet_sdk.engines.structural.engine import StructuralEngine
from fighter_jet_sdk.engines.structural.thermal_stress_analyzer import (
    ThermalLoadConditions, StructuralGeometry
)
from fighter_jet_sdk.common.data_models import (
    MaterialDefinition, ThermalProperties, MechanicalProperties, MaterialType
)


class TestStructuralEngine:
    """Test cases for StructuralEngine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = StructuralEngine()
        
        # Create test geometry
        self.geometry = StructuralGeometry(
            nodes=np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]),
            elements=np.array([[0, 1], [1, 2], [2, 3]]),
            element_type='beam',
            thickness=np.array([0.01, 0.01, 0.01])
        )
        
        # Create test materials
        thermal_props = ThermalProperties(
            thermal_conductivity=50.0,
            specific_heat=500.0,
            density=7800.0,
            melting_point=1800.0,
            operating_temp_range=(200.0, 1500.0)
        )
        
        mechanical_props = MechanicalProperties(
            youngs_modulus=200e9,
            poissons_ratio=0.3,
            yield_strength=350e6,
            ultimate_strength=500e6,
            fatigue_limit=200e6,
            density=7800.0
        )
        
        self.materials = {
            'steel': MaterialDefinition(
                name='High-strength steel',
                base_material_type=MaterialType.CONVENTIONAL_METAL,
                thermal_properties=thermal_props,
                mechanical_properties=mechanical_props
            )
        }
        
        # Create test thermal loads
        self.thermal_loads = ThermalLoadConditions(
            temperature_distribution=np.array([300.0, 800.0, 1200.0, 1000.0]),
            temperature_gradient=np.array([500.0, 400.0, -200.0, -300.0]),
            heat_flux=np.array([1e6, 2e6, 1.5e6, 1e6])
        )
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        engine = StructuralEngine()
        assert not engine.initialized
        assert engine.thermal_stress_analyzer is None
        assert engine.atmospheric_loads_analyzer is None
        
        # Initialize engine
        success = engine.initialize()
        assert success
        assert engine.initialized
        assert engine.thermal_stress_analyzer is not None
        assert engine.atmospheric_loads_analyzer is not None
    
    def test_engine_initialization_failure(self):
        """Test engine initialization failure handling."""
        engine = StructuralEngine()
        
        # Mock initialization failure
        with patch.object(engine, 'logger') as mock_logger:
            with patch('fighter_jet_sdk.engines.structural.engine.ThermalStressAnalyzer', 
                      side_effect=Exception("Init failed")):
                success = engine.initialize()
                assert not success
                assert not engine.initialized
                mock_logger.error.assert_called()
    
    def test_validate_input(self):
        """Test input validation."""
        engine = StructuralEngine()
        
        # Should fail when not initialized
        assert not engine.validate_input({'test': 'data'})
        
        # Initialize engine
        engine.initialize()
        
        # Should pass with valid data
        assert engine.validate_input({'test': 'data'})
        
        # Should fail with None data
        assert not engine.validate_input(None)
    
    def test_analyze_thermal_stress(self):
        """Test thermal stress analysis through engine."""
        self.engine.initialize()
        
        results = self.engine.analyze_thermal_stress(
            self.geometry, self.materials, self.thermal_loads, 'steady_state'
        )
        
        assert results is not None
        assert hasattr(results, 'thermal_stress')
        assert hasattr(results, 'mechanical_stress')
        assert hasattr(results, 'total_stress')
        assert hasattr(results, 'safety_factor')
        assert results.max_temperature == 1200.0
    
    def test_analyze_thermal_stress_not_initialized(self):
        """Test thermal stress analysis when engine not initialized."""
        with pytest.raises(RuntimeError, match="not properly initialized"):
            self.engine.analyze_thermal_stress(
                self.geometry, self.materials, self.thermal_loads
            )
    
    def test_calculate_thermal_expansion_effects(self):
        """Test thermal expansion effects calculation through engine."""
        self.engine.initialize()
        
        temperature_change = np.array([100.0, 500.0, 900.0, 700.0])
        
        results = self.engine.calculate_thermal_expansion_effects(
            self.geometry, self.materials, temperature_change
        )
        
        assert 'thermal_strain' in results
        assert 'thermal_displacement' in results
        assert 'constrained_stress' in results
        assert 'expansion_coefficients' in results
    
    def test_perform_coupled_thermal_structural_analysis(self):
        """Test coupled thermal-structural analysis through engine."""
        self.engine.initialize()
        
        mechanical_loads = {
            'applied_forces': np.array([[0, 0, 1000], [0, 0, 0], [0, 0, 0], [0, 0, -1000]])
        }
        
        results = self.engine.perform_coupled_thermal_structural_analysis(
            self.geometry, self.materials, self.thermal_loads, mechanical_loads
        )
        
        assert results is not None
        assert hasattr(results, 'thermal_stress')
        assert hasattr(results, 'mechanical_stress')
        assert hasattr(results, 'total_stress')
    
    def test_analyze_hypersonic_loads(self):
        """Test hypersonic loads analysis through engine."""
        self.engine.initialize()
        
        altitude_range = (30000.0, 80000.0)
        mach_number = 60.0
        aircraft_geometry = {
            'reference_area': 150.0,
            'reference_length': 25.0,
            'wing_span': 18.0
        }
        
        results = self.engine.analyze_hypersonic_loads(
            altitude_range, mach_number, aircraft_geometry
        )
        
        assert results is not None
        assert hasattr(results, 'atmospheric_conditions')
        assert hasattr(results, 'structural_loads')
        assert hasattr(results, 'max_dynamic_pressure')
        assert hasattr(results, 'max_load_factor')
        assert results.atmospheric_conditions.mach_number == 60.0
    
    def test_calculate_dynamic_pressure_envelope(self):
        """Test dynamic pressure envelope calculation through engine."""
        self.engine.initialize()
        
        altitude_range = (30000.0, 80000.0)
        mach_range = (10.0, 60.0)
        
        envelope = self.engine.calculate_dynamic_pressure_envelope(
            altitude_range, mach_range, n_points=20
        )
        
        assert 'altitude' in envelope
        assert 'mach_number' in envelope
        assert 'dynamic_pressure' in envelope
        assert 'max_q' in envelope
        assert envelope['max_q'] > 0
    
    def test_analyze_safety_factors(self):
        """Test safety factor analysis through engine."""
        self.engine.initialize()
        
        # Create mock structural loads
        from fighter_jet_sdk.engines.structural.atmospheric_loads_analyzer import StructuralLoads
        
        structural_loads = StructuralLoads(
            dynamic_pressure=50000.0,
            normal_force=np.full(10, 1e6),
            shear_force=np.full(10, 1e5),
            bending_moment=np.full(10, 1e7),
            torsional_moment=np.full(10, 5e6),
            pressure_distribution=np.full(10, 50000.0),
            load_factor=2.0,
            safety_factor=1.5
        )
        
        material_properties = {
            'fuselage': {'ultimate_strength': 600e6, 'yield_strength': 400e6},
            'wings': {'ultimate_strength': 500e6, 'yield_strength': 350e6}
        }
        
        safety_factors = self.engine.analyze_safety_factors(
            structural_loads, material_properties
        )
        
        assert isinstance(safety_factors, dict)
        assert len(safety_factors) > 0
        assert all(sf > 0 for sf in safety_factors.values())
    
    def test_optimize_flight_profile(self):
        """Test flight profile optimization through engine."""
        self.engine.initialize()
        
        target_altitude = 50000.0
        max_dynamic_pressure = 100000.0
        mach_range = (20.0, 60.0)
        
        profile = self.engine.optimize_flight_profile(
            target_altitude, max_dynamic_pressure, mach_range
        )
        
        assert 'optimal_mach' in profile
        assert 'optimal_altitude' in profile
        assert 'dynamic_pressure' in profile
        assert mach_range[0] <= profile['optimal_mach'] <= mach_range[1]
        assert abs(profile['optimal_altitude'] - target_altitude) < 1e-6
    
    def test_validate_structural_design(self):
        """Test comprehensive structural design validation."""
        self.engine.initialize()
        
        flight_conditions = {
            'thermal_loads': self.thermal_loads,
            'altitude_range': (40000.0, 60000.0),
            'mach_number': 30.0
        }
        
        safety_requirements = {
            'thermal_safety_factor': 1.5,
            'structural_margin': 0.5
        }
        
        validation_results = self.engine.validate_structural_design(
            self.geometry, self.materials, flight_conditions, safety_requirements
        )
        
        assert 'overall_status' in validation_results
        assert 'thermal_analysis' in validation_results
        assert 'atmospheric_analysis' in validation_results
        assert 'safety_analysis' in validation_results
        assert 'recommendations' in validation_results
        
        assert validation_results['overall_status'] in ['PASS', 'FAIL']
        assert isinstance(validation_results['recommendations'], list)
    
    def test_process_thermal_stress_analysis(self):
        """Test processing thermal stress analysis request."""
        self.engine.initialize()
        
        data = {
            'operation': 'thermal_stress_analysis',
            'geometry': {
                'nodes': self.geometry.nodes.tolist(),
                'elements': self.geometry.elements.tolist(),
                'element_type': self.geometry.element_type,
                'thickness': self.geometry.thickness.tolist()
            },
            'materials': {
                'steel': {
                    'name': 'High-strength steel',
                    'base_material_type': 'CONVENTIONAL_METAL',
                    'thermal_properties': {
                        'thermal_conductivity': 50.0,
                        'specific_heat': 500.0,
                        'density': 7800.0,
                        'melting_point': 1800.0,
                        'operating_temp_range': [200.0, 1500.0]
                    },
                    'mechanical_properties': {
                        'youngs_modulus': 200e9,
                        'poissons_ratio': 0.3,
                        'yield_strength': 350e6,
                        'ultimate_strength': 500e6,
                        'fatigue_limit': 200e6,
                        'density': 7800.0
                    }
                }
            },
            'thermal_loads': {
                'temperature_distribution': self.thermal_loads.temperature_distribution.tolist(),
                'temperature_gradient': self.thermal_loads.temperature_gradient.tolist(),
                'heat_flux': self.thermal_loads.heat_flux.tolist()
            },
            'analysis_type': 'steady_state'
        }
        
        result = self.engine.process(data)
        
        assert result is not None
        assert 'max_temperature' in result
        assert 'max_stress' in result
        assert 'thermal_stress' in result
        assert 'safety_factor' in result
        assert result['max_temperature'] == 1200.0
    
    def test_process_atmospheric_loads_analysis(self):
        """Test processing atmospheric loads analysis request."""
        self.engine.initialize()
        
        data = {
            'operation': 'atmospheric_loads_analysis',
            'altitude_range': [30000.0, 80000.0],
            'mach_number': 60.0,
            'aircraft_geometry': {
                'reference_area': 150.0,
                'reference_length': 25.0,
                'wing_span': 18.0
            }
        }
        
        result = self.engine.process(data)
        
        assert result is not None
        assert 'max_dynamic_pressure' in result
        assert 'max_load_factor' in result
        assert 'structural_margins' in result
        assert 'atmospheric_conditions' in result
        assert result['atmospheric_conditions']['mach_number'] == 60.0
    
    def test_process_coupled_analysis(self):
        """Test processing coupled analysis request."""
        self.engine.initialize()
        
        data = {
            'operation': 'coupled_analysis',
            'geometry': {
                'nodes': self.geometry.nodes.tolist(),
                'elements': self.geometry.elements.tolist(),
                'element_type': self.geometry.element_type,
                'thickness': self.geometry.thickness.tolist()
            },
            'materials': {
                'steel': {
                    'name': 'High-strength steel',
                    'base_material_type': 'CONVENTIONAL_METAL',
                    'thermal_properties': {
                        'thermal_conductivity': 50.0,
                        'specific_heat': 500.0,
                        'density': 7800.0,
                        'melting_point': 1800.0,
                        'operating_temp_range': [200.0, 1500.0]
                    },
                    'mechanical_properties': {
                        'youngs_modulus': 200e9,
                        'poissons_ratio': 0.3,
                        'yield_strength': 350e6,
                        'ultimate_strength': 500e6,
                        'fatigue_limit': 200e6,
                        'density': 7800.0
                    }
                }
            },
            'thermal_loads': {
                'temperature_distribution': self.thermal_loads.temperature_distribution.tolist(),
                'temperature_gradient': self.thermal_loads.temperature_gradient.tolist(),
                'heat_flux': self.thermal_loads.heat_flux.tolist()
            },
            'mechanical_loads': {
                'applied_forces': [[0, 0, 1000], [0, 0, 0], [0, 0, 0], [0, 0, -1000]]
            }
        }
        
        result = self.engine.process(data)
        
        assert result is not None
        assert 'max_temperature' in result
        assert 'max_stress' in result
        assert 'thermal_stress' in result
        assert 'mechanical_stress' in result
        assert 'total_stress' in result
        assert 'displacement' in result
    
    def test_process_safety_factor_analysis(self):
        """Test processing safety factor analysis request."""
        self.engine.initialize()
        
        data = {
            'operation': 'safety_factor_analysis',
            'material_properties': {
                'fuselage': {'ultimate_strength': 600e6, 'yield_strength': 400e6},
                'wings': {'ultimate_strength': 500e6, 'yield_strength': 350e6}
            }
        }
        
        result = self.engine.process(data)
        
        assert result is not None
        assert 'safety_factors' in result
        assert 'overall_status' in result
        assert result['overall_status'] in ['PASS', 'FAIL']
    
    def test_process_unknown_operation(self):
        """Test processing unknown operation."""
        self.engine.initialize()
        
        data = {
            'operation': 'unknown_operation',
            'some_data': 'test'
        }
        
        result = self.engine.process(data)
        assert result is None
    
    def test_process_invalid_input(self):
        """Test processing with invalid input."""
        self.engine.initialize()
        
        # Test with None input
        result = self.engine.process(None)
        assert result is None
        
        # Test with non-dict input
        result = self.engine.process("invalid")
        assert result == "invalid"  # Should return as-is for non-dict
    
    def test_process_error_handling(self):
        """Test error handling in process methods."""
        self.engine.initialize()
        
        # Test thermal stress analysis with invalid data
        data = {
            'operation': 'thermal_stress_analysis',
            'geometry': {},  # Invalid geometry
            'materials': {},
            'thermal_loads': {}
        }
        
        result = self.engine.process(data)
        
        assert result is not None
        assert 'error' in result
        assert isinstance(result['error'], str)
    
    def test_extreme_conditions_integration(self):
        """Test integration under extreme hypersonic conditions."""
        self.engine.initialize()
        
        # Extreme thermal loads (up to 6000K, 150 MW/m²)
        extreme_thermal_loads = ThermalLoadConditions(
            temperature_distribution=np.array([300.0, 2000.0, 5000.0, 6000.0]),
            temperature_gradient=np.array([2000.0, 3000.0, 1000.0, -2000.0]),
            heat_flux=np.array([10e6, 100e6, 150e6, 80e6])
        )
        
        # Extreme flight conditions (Mach 60, 80 km altitude)
        flight_conditions = {
            'thermal_loads': extreme_thermal_loads,
            'altitude_range': (75000.0, 80000.0),
            'mach_number': 60.0
        }
        
        safety_requirements = {
            'thermal_safety_factor': 1.5,
            'structural_margin': 0.5
        }
        
        validation_results = self.engine.validate_structural_design(
            self.geometry, self.materials, flight_conditions, safety_requirements
        )
        
        assert validation_results is not None
        assert 'overall_status' in validation_results
        
        # Should likely fail under such extreme conditions
        # but the analysis should complete without errors
        assert validation_results['overall_status'] in ['PASS', 'FAIL']
        
        # Should have many recommendations for extreme conditions
        assert len(validation_results['recommendations']) > 0
        
        # Check thermal analysis results
        assert 'thermal_analysis' in validation_results
        assert validation_results['thermal_analysis']['max_temperature'] == 6000.0
        
        # Check atmospheric analysis results
        assert 'atmospheric_analysis' in validation_results
        assert validation_results['atmospheric_analysis']['max_dynamic_pressure'] > 0
    
    @patch('fighter_jet_sdk.engines.structural.engine.get_engine_logger')
    def test_logging(self, mock_logger):
        """Test logging functionality."""
        mock_log = Mock()
        mock_logger.return_value = mock_log
        
        engine = StructuralEngine()
        
        # Test initialization logging
        engine.initialize()
        mock_log.info.assert_called()
        
        # Check that initialization completion was logged
        init_calls = [call for call in mock_log.info.call_args_list 
                     if 'initialization complete' in str(call)]
        assert len(init_calls) > 0