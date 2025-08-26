"""
Tests for Aerodynamics Engine
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from fighter_jet_sdk.engines.aerodynamics.engine import AerodynamicsEngine, AerodynamicResults
from fighter_jet_sdk.engines.aerodynamics.cfd_solver import CFDResults, FlowRegime
from fighter_jet_sdk.common.data_models import (
    AircraftConfiguration, Module, ModuleType, FlowConditions,
    PhysicalProperties, BasePlatform, MechanicalInterface
)


class TestAerodynamicsEngine:
    """Test main aerodynamics engine functionality"""
    
    def setup_method(self):
        self.engine = AerodynamicsEngine()
        self.configuration = self._create_test_configuration()
        self.flow_conditions = FlowConditions(
            mach_number=0.8,
            altitude=10000,
            angle_of_attack=2.0,
            sideslip_angle=0.0
        )
    
    def _create_test_configuration(self):
        """Create test aircraft configuration"""
        fuselage = Module(
            module_id="fuselage_001",
            module_type=ModuleType.STRUCTURAL,
            name="Main Fuselage",
            physical_properties=PhysicalProperties(
                mass=5000.0,
                center_of_gravity=(7.5, 0.0, 0.0),
                moments_of_inertia=(10000.0, 15000.0, 20000.0),
                dimensions=(15.0, 2.0, 2.0)
            ),
            performance_characteristics={}
        )
        
        wing = Module(
            module_id="wing_001", 
            module_type=ModuleType.STRUCTURAL,
            name="Main Wing",
            physical_properties=PhysicalProperties(
                mass=2000.0,
                center_of_gravity=(6.0, 0.0, 0.0),
                moments_of_inertia=(5000.0, 8000.0, 10000.0),
                dimensions=(12.0, 1.0, 0.2)
            ),
            performance_characteristics={"span": 12.0, "area": 40.0, "aspect_ratio": 3.6}
        )
        
        engine_module = Module(
            module_id="engine_001",
            module_type=ModuleType.PROPULSION,
            name="Main Engine",
            physical_properties=PhysicalProperties(
                mass=1500.0,
                center_of_gravity=(8.0, 0.0, 0.0),
                moments_of_inertia=(2000.0, 3000.0, 4000.0),
                dimensions=(3.0, 1.5, 1.5)
            ),
            performance_characteristics={"max_thrust": 120000.0}
        )
        
        base_platform = BasePlatform(
            platform_id="fighter_platform_001",
            name="Fighter Platform",
            base_mass=3000.0,
            attachment_points=[
                MechanicalInterface(
                    interface_id="attach_1",
                    attachment_type="standard",
                    load_capacity=(50000.0, 50000.0, 100000.0),
                    moment_capacity=(10000.0, 10000.0, 5000.0),
                    position=(5.0, 0.0, 0.0)
                )
            ],
            power_generation_capacity=100000.0,
            fuel_capacity=5000.0
        )
        
        return AircraftConfiguration(
            config_id="test_config_001",
            name="Test Fighter Configuration",
            base_platform=base_platform,
            modules=[fuselage, wing, engine_module]
        )
    
    def test_engine_initialization(self):
        """Test aerodynamics engine initialization"""
        assert self.engine.cfd_solver is not None
        assert self.engine.stability_analyzer is not None
        assert self.engine.stealth_optimizer is not None
        
        assert self.engine.capabilities["cfd_analysis"] is True
        assert self.engine.capabilities["stability_analysis"] is True
        assert self.engine.capabilities["stealth_optimization"] is True
        assert self.engine.capabilities["multi_speed_analysis"] is True
        assert self.engine.capabilities["performance_envelope"] is True
    
    @patch('fighter_jet_sdk.engines.aerodynamics.engine.AerodynamicsEngine._perform_cfd_analysis')
    def test_cfd_analysis_type(self, mock_cfd):
        """Test CFD analysis type"""
        mock_cfd_results = CFDResults(
            forces={"drag": 1000.0, "lift": 50000.0, "side_force": 0.0},
            moments={"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
            pressure_distribution=None,
            velocity_field=None,
            convergence_history=[1e-3, 1e-6],
            residuals={"U": [1e-3, 1e-6], "p": [1e-3, 1e-6]},
            flow_regime=FlowRegime.TRANSONIC,
            mach_number=0.8,
            reynolds_number=1e6
        )
        mock_cfd.return_value = mock_cfd_results
        
        results = self.engine.analyze(
            self.configuration, 
            analysis_type="cfd",
            flow_conditions=self.flow_conditions
        )
        
        assert isinstance(results, AerodynamicResults)
        assert results.cfd_results is not None
        assert results.stability_results is None
        assert results.optimization_results is None
        assert results.performance_envelope is None
        
        mock_cfd.assert_called_once()
    
    @patch('fighter_jet_sdk.engines.aerodynamics.engine.AerodynamicsEngine._perform_stability_analysis')
    def test_stability_analysis_type(self, mock_stability):
        """Test stability analysis type"""
        mock_stability_results = {
            "max_g_load": 9.0,
            "stall_speed": 150.0,
            "stability_margin": 0.1
        }
        mock_stability.return_value = mock_stability_results
        
        results = self.engine.analyze(
            self.configuration, 
            analysis_type="stability"
        )
        
        assert isinstance(results, AerodynamicResults)
        assert results.cfd_results is None
        assert results.stability_results is not None
        assert results.optimization_results is None
        assert results.performance_envelope is None
        
        mock_stability.assert_called_once()
    
    @patch('fighter_jet_sdk.engines.aerodynamics.engine.AerodynamicsEngine._perform_optimization')
    def test_optimization_analysis_type(self, mock_optimization):
        """Test optimization analysis type"""
        mock_optimization_results = {
            "rcs_reduction": 0.8,
            "aerodynamic_efficiency": 0.95,
            "pareto_frontier": []
        }
        mock_optimization.return_value = mock_optimization_results
        
        results = self.engine.analyze(
            self.configuration, 
            analysis_type="optimization"
        )
        
        assert isinstance(results, AerodynamicResults)
        assert results.cfd_results is None
        assert results.stability_results is None
        assert results.optimization_results is not None
        assert results.performance_envelope is None
        
        mock_optimization.assert_called_once()
    
    @patch('fighter_jet_sdk.engines.aerodynamics.engine.AerodynamicsEngine._perform_cfd_analysis')
    @patch('fighter_jet_sdk.engines.aerodynamics.engine.AerodynamicsEngine._perform_stability_analysis')
    @patch('fighter_jet_sdk.engines.aerodynamics.engine.AerodynamicsEngine._perform_optimization')
    @patch('fighter_jet_sdk.engines.aerodynamics.engine.AerodynamicsEngine._calculate_performance_envelope')
    def test_comprehensive_analysis(self, mock_envelope, mock_optimization, 
                                   mock_stability, mock_cfd):
        """Test comprehensive analysis"""
        # Setup mocks
        mock_cfd_results = CFDResults(
            forces={"drag": 1000.0, "lift": 50000.0, "side_force": 0.0},
            moments={"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
            pressure_distribution=None,
            velocity_field=None,
            convergence_history=[1e-3, 1e-6],
            residuals={"U": [1e-3, 1e-6], "p": [1e-3, 1e-6]},
            flow_regime=FlowRegime.TRANSONIC,
            mach_number=0.8,
            reynolds_number=1e6
        )
        mock_cfd.return_value = mock_cfd_results
        
        mock_stability_results = {
            "max_g_load": 9.0,
            "stall_speed": 150.0,
            "stability_margin": 0.1
        }
        mock_stability.return_value = mock_stability_results
        
        mock_optimization_results = {
            "rcs_reduction": 0.8,
            "aerodynamic_efficiency": 0.95,
            "pareto_frontier": []
        }
        mock_optimization.return_value = mock_optimization_results
        
        mock_envelope_results = {
            "max_mach": 2.5,
            "service_ceiling": 18000,
            "max_g_load": 9.0,
            "stall_speed": 150.0,
            "max_range": 3000,
            "combat_radius": 1200
        }
        mock_envelope.return_value = mock_envelope_results
        
        # Run comprehensive analysis
        results = self.engine.analyze(
            self.configuration, 
            analysis_type="comprehensive",
            flow_conditions=self.flow_conditions
        )
        
        # Verify all components were called
        assert isinstance(results, AerodynamicResults)
        assert results.cfd_results is not None
        assert results.stability_results is not None
        assert results.optimization_results is not None
        assert results.performance_envelope is not None
        
        mock_cfd.assert_called_once()
        mock_stability.assert_called_once()
        mock_optimization.assert_called_once()
        mock_envelope.assert_called_once()
    
    def test_multi_speed_regime_analysis(self):
        """Test multi-speed regime analysis"""
        mach_range = [0.5, 0.8, 1.2, 2.0]
        
        with patch.object(self.engine.cfd_solver, 'analyze') as mock_analyze:
            mock_cfd_results = CFDResults(
                forces={"drag": 1000.0, "lift": 50000.0, "side_force": 0.0},
                moments={"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                pressure_distribution=None,
                velocity_field=None,
                convergence_history=[1e-3, 1e-6],
                residuals={"U": [1e-3, 1e-6], "p": [1e-3, 1e-6]},
                flow_regime=FlowRegime.TRANSONIC,
                mach_number=0.8,
                reynolds_number=1e6
            )
            mock_analyze.return_value = mock_cfd_results
            
            results = self.engine.analyze_multi_speed_regime(
                self.configuration, mach_range
            )
            
            assert len(results) == len(mach_range)
            assert mock_analyze.call_count == len(mach_range)
            
            # Check that all Mach numbers are in results
            for mach in mach_range:
                assert mach in results
                assert isinstance(results[mach], CFDResults)
    
    def test_configuration_validation(self):
        """Test aircraft configuration validation"""
        validation = self.engine.validate_configuration(self.configuration)
        
        assert isinstance(validation, dict)
        assert "geometry_valid" in validation
        assert "mass_properties_valid" in validation
        assert "control_surfaces_valid" in validation
        assert "engine_integration_valid" in validation
        
        # Should pass basic validation
        assert validation["geometry_valid"] is True
        assert validation["mass_properties_valid"] is True
        assert validation["engine_integration_valid"] is True
    
    def test_analysis_recommendations(self):
        """Test analysis recommendations"""
        recommendations = self.engine.get_analysis_recommendations(self.configuration)
        
        assert isinstance(recommendations, list)
        
        # Should recommend multi-speed analysis due to high-thrust engine
        multi_speed_rec = any("multi-speed" in rec.lower() for rec in recommendations)
        assert multi_speed_rec is True
    
    def test_performance_envelope_calculation(self):
        """Test performance envelope calculation"""
        # Create mock results
        mock_cfd_results = CFDResults(
            forces={"drag": 1000.0, "lift": 50000.0, "side_force": 0.0},
            moments={"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
            pressure_distribution=None,
            velocity_field=None,
            convergence_history=[1e-3, 1e-6],
            residuals={"U": [1e-3, 1e-6], "p": [1e-3, 1e-6]},
            flow_regime=FlowRegime.TRANSONIC,
            mach_number=0.8,
            reynolds_number=1e6
        )
        
        mock_stability_results = {
            "max_g_load": 9.0,
            "stall_speed": 150.0,
            "stability_margin": 0.1
        }
        
        mock_aerodynamic_results = AerodynamicResults(
            configuration_id=self.configuration.config_id,
            analysis_type="test",
            timestamp="2024-01-01T00:00:00",
            cfd_results=mock_cfd_results,
            stability_results=mock_stability_results
        )
        
        envelope = self.engine._calculate_performance_envelope(
            self.configuration, mock_aerodynamic_results
        )
        
        assert isinstance(envelope, dict)
        assert "max_mach" in envelope
        assert "service_ceiling" in envelope
        assert "max_g_load" in envelope
        assert "stall_speed" in envelope
        assert "max_range" in envelope
        assert "combat_radius" in envelope
        
        # Check that values are reasonable
        assert envelope["max_g_load"] == 9.0
        assert envelope["stall_speed"] == 150.0
        assert envelope["max_mach"] > 0
        assert envelope["service_ceiling"] > 0
    
    def test_max_mach_estimation(self):
        """Test maximum Mach number estimation"""
        # Test low drag (supercruise capable)
        low_drag_results = CFDResults(
            forces={"drag": 500.0, "lift": 50000.0, "side_force": 0.0},
            moments={"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
            pressure_distribution=None,
            velocity_field=None,
            convergence_history=[1e-3, 1e-6],
            residuals={"U": [1e-3, 1e-6], "p": [1e-3, 1e-6]},
            flow_regime=FlowRegime.SUPERSONIC,
            mach_number=1.5,
            reynolds_number=1e6
        )
        
        max_mach = self.engine._estimate_max_mach(low_drag_results)
        assert max_mach == 2.5  # Supercruise capable
        
        # Test high drag (limited supersonic)
        high_drag_results = CFDResults(
            forces={"drag": 2000.0, "lift": 50000.0, "side_force": 0.0},
            moments={"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
            pressure_distribution=None,
            velocity_field=None,
            convergence_history=[1e-3, 1e-6],
            residuals={"U": [1e-3, 1e-6], "p": [1e-3, 1e-6]},
            flow_regime=FlowRegime.SUPERSONIC,
            mach_number=1.5,
            reynolds_number=1e6
        )
        
        max_mach = self.engine._estimate_max_mach(high_drag_results)
        assert max_mach == 1.6  # Limited supersonic


if __name__ == "__main__":
    pytest.main([__file__])