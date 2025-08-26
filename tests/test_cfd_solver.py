"""
Tests for CFD Solver Integration Module
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from fighter_jet_sdk.engines.aerodynamics.cfd_solver import (
    CFDSolver, MeshGenerator, ConvergenceMonitor, FlowRegime,
    MeshParameters, SolverSettings, CFDResults
)
from fighter_jet_sdk.common.data_models import AircraftConfiguration, FlowConditions, Module, ModuleType
from fighter_jet_sdk.core.errors import CFDError, ValidationError


class TestMeshGenerator:
    """Test mesh generation capabilities"""
    
    def setup_method(self):
        self.mesh_generator = MeshGenerator()
        self.flow_conditions = FlowConditions(
            mach_number=0.8,
            altitude=10000,
            angle_of_attack=2.0,
            sideslip_angle=0.0
        )
    
    def test_flow_regime_classification(self):
        """Test flow regime classification"""
        # Test subsonic
        assert self.mesh_generator._classify_flow_regime(0.5) == FlowRegime.SUBSONIC
        
        # Test transonic
        assert self.mesh_generator._classify_flow_regime(1.0) == FlowRegime.TRANSONIC
        
        # Test supersonic
        assert self.mesh_generator._classify_flow_regime(2.0) == FlowRegime.SUPERSONIC
        
        # Test hypersonic
        assert self.mesh_generator._classify_flow_regime(6.0) == FlowRegime.HYPERSONIC
    
    def test_mesh_parameters_creation(self):
        """Test mesh parameter generation"""
        mesh_params = MeshParameters(
            base_cell_size=0.1,
            boundary_layer_thickness=0.001,
            refinement_levels=3,
            growth_ratio=1.2,
            surface_refinement={"aircraft": 0.01}
        )
        
        assert mesh_params.base_cell_size == 0.1
        assert mesh_params.refinement_levels == 3
        assert mesh_params.surface_refinement["aircraft"] == 0.01
    
    @patch('subprocess.run')
    def test_base_mesh_generation(self, mock_subprocess):
        """Test base mesh generation"""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stderr = ""
        
        mesh_params = MeshParameters(
            base_cell_size=0.1,
            boundary_layer_thickness=0.001,
            refinement_levels=3,
            growth_ratio=1.2,
            surface_refinement={"aircraft": 0.01}
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            geometry_file = os.path.join(temp_dir, "test.stl")
            with open(geometry_file, 'w') as f:
                f.write("solid test\nendsolid test\n")
            
            # Mock the mesh generation process
            with patch.object(self.mesh_generator, '_create_block_mesh_dict') as mock_dict:
                mock_dict.return_value = "test mesh dict"
                
                mesh_dir = self.mesh_generator._generate_base_mesh(
                    geometry_file, mesh_params, temp_dir
                )
                
                assert mesh_dir == temp_dir
                mock_subprocess.assert_called()
    
    def test_block_mesh_dict_creation(self):
        """Test blockMeshDict creation"""
        mesh_params = MeshParameters(
            base_cell_size=0.1,
            boundary_layer_thickness=0.001,
            refinement_levels=3,
            growth_ratio=1.2,
            surface_refinement={"aircraft": 0.01}
        )
        
        mesh_dict = self.mesh_generator._create_block_mesh_dict(mesh_params)
        
        assert "FoamFile" in mesh_dict
        assert "vertices" in mesh_dict
        assert "blocks" in mesh_dict
        assert "boundary" in mesh_dict


class TestConvergenceMonitor:
    """Test convergence monitoring"""
    
    def setup_method(self):
        self.monitor = ConvergenceMonitor(tolerance=1e-6)
    
    def test_convergence_check_converged(self):
        """Test convergence detection when converged"""
        residuals = {"U": 1e-7, "p": 5e-7}
        
        converged = self.monitor.check_convergence(residuals, 100)
        
        assert converged is True
        assert len(self.monitor.residual_history["U"]) == 1
        assert len(self.monitor.residual_history["p"]) == 1
    
    def test_convergence_check_not_converged(self):
        """Test convergence detection when not converged"""
        residuals = {"U": 1e-4, "p": 1e-5}
        
        converged = self.monitor.check_convergence(residuals, 50)
        
        assert converged is False
    
    def test_residual_history_tracking(self):
        """Test residual history tracking"""
        residuals_1 = {"U": 1e-3, "p": 1e-3}
        residuals_2 = {"U": 1e-4, "p": 1e-4}
        residuals_3 = {"U": 1e-7, "p": 1e-7}
        
        self.monitor.check_convergence(residuals_1, 1)
        self.monitor.check_convergence(residuals_2, 2)
        self.monitor.check_convergence(residuals_3, 3)
        
        history = self.monitor.get_convergence_history()
        
        assert len(history["U"]) == 3
        assert len(history["p"]) == 3
        assert history["U"] == [1e-3, 1e-4, 1e-7]
        assert history["p"] == [1e-3, 1e-4, 1e-7]


class TestCFDSolver:
    """Test main CFD solver functionality"""
    
    def setup_method(self):
        self.solver = CFDSolver()
        self.configuration = self._create_test_configuration()
        self.flow_conditions = FlowConditions(
            mach_number=0.8,
            altitude=10000,
            angle_of_attack=2.0,
            sideslip_angle=0.0
        )
    
    def _create_test_configuration(self):
        """Create test aircraft configuration"""
        from fighter_jet_sdk.common.data_models import PhysicalProperties
        
        fuselage = Module(
            module_id="fuselage_001",
            module_type=ModuleType.STRUCTURAL,
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
            physical_properties=PhysicalProperties(
                mass=2000.0,
                center_of_gravity=(6.0, 0.0, 0.0),
                moments_of_inertia=(5000.0, 8000.0, 10000.0),
                dimensions=(12.0, 1.0, 0.2)
            ),
            performance_characteristics={"span": 12.0, "area": 40.0, "aspect_ratio": 3.6}
        )
        
        from fighter_jet_sdk.common.data_models import BasePlatform, MechanicalInterface
        
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
            modules=[fuselage, wing]
        )
    
    def test_input_validation(self):
        """Test input validation"""
        # Test missing configuration
        with pytest.raises(ValidationError):
            self.solver._validate_inputs(None, self.flow_conditions)
        
        # Test negative Mach number
        bad_conditions = FlowConditions(
            mach_number=-0.5,
            altitude=10000,
            angle_of_attack=2.0,
            sideslip_angle=0.0
        )
        
        with pytest.raises(ValidationError):
            self.solver._validate_inputs(self.configuration, bad_conditions)
        
        # Test negative altitude
        bad_conditions = FlowConditions(
            mach_number=0.8,
            altitude=-1000,
            angle_of_attack=2.0,
            sideslip_angle=0.0
        )
        
        with pytest.raises(ValidationError):
            self.solver._validate_inputs(self.configuration, bad_conditions)
    
    def test_default_solver_settings(self):
        """Test default solver settings generation"""
        # Test subsonic settings
        subsonic_conditions = FlowConditions(
            mach_number=0.5, altitude=5000, angle_of_attack=2.0, sideslip_angle=0.0
        )
        settings = self.solver._get_default_solver_settings(subsonic_conditions)
        
        assert settings.solver_type == "simpleFoam"
        assert settings.turbulence_model == "kOmegaSST"
        assert settings.max_iterations == 1000
        
        # Test supersonic settings
        supersonic_conditions = FlowConditions(
            mach_number=2.0, altitude=15000, angle_of_attack=2.0, sideslip_angle=0.0
        )
        settings = self.solver._get_default_solver_settings(supersonic_conditions)
        
        assert settings.solver_type == "rhoSimpleFoam"
        assert settings.max_iterations == 2000
    
    def test_reynolds_number_calculation(self):
        """Test Reynolds number calculation"""
        reynolds = self.solver._calculate_reynolds_number(self.flow_conditions)
        
        assert reynolds > 0
        assert isinstance(reynolds, float)
        
        # Higher Mach should give higher Reynolds number
        high_mach_conditions = FlowConditions(
            mach_number=2.0, altitude=10000, angle_of_attack=2.0, sideslip_angle=0.0
        )
        high_reynolds = self.solver._calculate_reynolds_number(high_mach_conditions)
        
        assert high_reynolds > reynolds
    
    def test_mesh_parameters_generation(self):
        """Test mesh parameter generation"""
        mesh_params = self.solver._get_mesh_parameters(self.flow_conditions)
        
        assert isinstance(mesh_params, MeshParameters)
        assert mesh_params.base_cell_size > 0
        assert mesh_params.boundary_layer_thickness > 0
        assert mesh_params.refinement_levels > 0
        assert mesh_params.growth_ratio > 1.0
    
    def test_geometry_export(self):
        """Test geometry export"""
        geometry_file = self.solver._export_geometry(self.configuration)
        
        assert os.path.exists(geometry_file)
        assert geometry_file.endswith('.stl')
        
        # Clean up
        os.remove(geometry_file)
    
    @patch('subprocess.run')
    @patch('tempfile.mkdtemp')
    def test_case_setup(self, mock_mkdtemp, mock_subprocess):
        """Test OpenFOAM case setup"""
        mock_mkdtemp.return_value = "/tmp/test_case"
        mock_subprocess.return_value.returncode = 0
        
        solver_settings = SolverSettings(
            solver_type="simpleFoam",
            turbulence_model="kOmegaSST",
            max_iterations=1000,
            convergence_tolerance=1e-6,
            relaxation_factors={"p": 0.3, "U": 0.7}
        )
        
        with patch('os.makedirs'), patch('builtins.open', create=True):
            case_dir = self.solver._setup_case(
                "/tmp/mesh", self.flow_conditions, solver_settings
            )
            
            assert case_dir == "/tmp/test_case"
    
    def test_solver_output_parsing(self):
        """Test solver output parsing"""
        solver_output = """
        Time = 100
        
        SIMPLE solution converged
        
        Forces:
        Drag = 1000.0 N
        Lift = 50000.0 N
        """
        
        results = self.solver._parse_solver_output(solver_output, "/tmp/case")
        
        assert "residuals" in results
        assert "forces" in results
        assert "moments" in results
        assert results["case_dir"] == "/tmp/case"
    
    @patch.object(CFDSolver, '_run_solver')
    @patch.object(CFDSolver, '_setup_case')
    @patch.object(CFDSolver, '_export_geometry')
    @patch.object(MeshGenerator, 'generate_mesh')
    def test_full_analysis(self, mock_mesh, mock_export, mock_setup, mock_run):
        """Test complete CFD analysis workflow"""
        # Setup mocks
        mock_export.return_value = "/tmp/geometry.stl"
        mock_mesh.return_value = "/tmp/mesh"
        mock_setup.return_value = "/tmp/case"
        mock_run.return_value = {
            "residuals": {"U": [1e-3, 1e-6], "p": [1e-3, 1e-6]},
            "forces": {"drag": 1000.0, "lift": 50000.0, "side_force": 0.0},
            "moments": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
            "case_dir": "/tmp/case"
        }
        
        # Run analysis
        results = self.solver.analyze(self.configuration, self.flow_conditions)
        
        # Verify results
        assert isinstance(results, CFDResults)
        assert results.forces["drag"] == 1000.0
        assert results.forces["lift"] == 50000.0
        assert results.mach_number == 0.8
        assert results.flow_regime == FlowRegime.TRANSONIC
        
        # Verify method calls
        mock_export.assert_called_once()
        mock_mesh.assert_called_once()
        mock_setup.assert_called_once()
        mock_run.assert_called_once()


class TestSolverSettings:
    """Test solver settings configuration"""
    
    def test_solver_settings_creation(self):
        """Test solver settings creation"""
        settings = SolverSettings(
            solver_type="rhoSimpleFoam",
            turbulence_model="kOmegaSST",
            max_iterations=2000,
            convergence_tolerance=1e-5,
            relaxation_factors={"p": 0.2, "U": 0.5, "T": 0.7}
        )
        
        assert settings.solver_type == "rhoSimpleFoam"
        assert settings.turbulence_model == "kOmegaSST"
        assert settings.max_iterations == 2000
        assert settings.convergence_tolerance == 1e-5
        assert settings.relaxation_factors["p"] == 0.2


class TestCFDResults:
    """Test CFD results data structure"""
    
    def test_cfd_results_creation(self):
        """Test CFD results creation"""
        forces = {"drag": 1000.0, "lift": 50000.0, "side_force": 0.0}
        moments = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
        pressure_dist = np.zeros((10, 10))
        velocity_field = np.zeros((10, 10, 3))
        
        results = CFDResults(
            forces=forces,
            moments=moments,
            pressure_distribution=pressure_dist,
            velocity_field=velocity_field,
            convergence_history=[1e-3, 1e-4, 1e-5, 1e-6],
            residuals={"U": [1e-3, 1e-6], "p": [1e-3, 1e-6]},
            flow_regime=FlowRegime.SUBSONIC,
            mach_number=0.8,
            reynolds_number=1e6
        )
        
        assert results.forces["drag"] == 1000.0
        assert results.mach_number == 0.8
        assert results.flow_regime == FlowRegime.SUBSONIC
        assert results.pressure_distribution.shape == (10, 10)
        assert results.velocity_field.shape == (10, 10, 3)


if __name__ == "__main__":
    pytest.main([__file__])