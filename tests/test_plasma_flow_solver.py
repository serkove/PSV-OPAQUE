"""
Unit tests for PlasmaFlowSolver

Tests the plasma flow solver foundation including MHD equation setup,
plasma property integration, and electromagnetic source terms.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from fighter_jet_sdk.engines.aerodynamics.plasma_flow_solver import (
    PlasmaFlowSolver, PlasmaFlowConditions, MHDSolverSettings, PlasmaFlowResults
)
from fighter_jet_sdk.engines.aerodynamics.cfd_solver import SolverSettings, CFDResults, FlowRegime
from fighter_jet_sdk.common.data_models import AircraftConfiguration, FlowConditions
from fighter_jet_sdk.common.plasma_physics import PlasmaConditions, GasMixture
from fighter_jet_sdk.common.electromagnetic_effects import ElectromagneticProperties
from fighter_jet_sdk.common.enums import PlasmaRegime
from fighter_jet_sdk.core.errors import CFDError, ValidationError


class TestPlasmaFlowSolver:
    """Test cases for PlasmaFlowSolver class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.solver = PlasmaFlowSolver()
        
        # Create test aircraft configuration
        self.config = AircraftConfiguration(
            config_id="test_mach60",
            name="Test Mach 60 Vehicle",
            modules=[]
        )
        
        # Create test flow conditions for Mach 60
        self.flow_conditions = FlowConditions(
            mach_number=60.0,
            altitude=50000.0,  # 50 km
            angle_of_attack=0.0,
            sideslip_angle=0.0,
            temperature=15000.0,  # High temperature for plasma formation
            pressure=1000.0,      # Low pressure at high altitude
            density=0.001         # Low density
        )
        
        # Test magnetic field
        self.magnetic_field = np.array([0.0, 0.0, 0.1])  # 0.1 Tesla in z-direction
        
        # Create test plasma conditions
        self.plasma_conditions = PlasmaConditions(
            electron_density=1e20,
            electron_temperature=15000.0,
            ion_temperature=15000.0,
            magnetic_field=self.magnetic_field,
            plasma_frequency=1e12,
            debye_length=1e-6,
            ionization_fraction=0.1,
            regime=PlasmaRegime.PARTIALLY_IONIZED
        )
        
        # Create test electromagnetic properties
        self.em_properties = ElectromagneticProperties(
            conductivity=1000.0,
            hall_parameter=0.5,
            magnetic_reynolds_number=100.0,
            electric_field=np.array([1000.0, 0.0, 0.0]),
            current_density=np.array([1e6, 0.0, 0.0]),
            lorentz_force_density=np.array([1e5, 0.0, 0.0])
        )
    
    def test_solver_initialization(self):
        """Test plasma flow solver initialization."""
        assert isinstance(self.solver, PlasmaFlowSolver)
        assert hasattr(self.solver, 'plasma_calculator')
        assert hasattr(self.solver, 'electromagnetic_calculator')
        assert self.solver.plasma_enabled is True
        assert self.solver.mhd_coupling_enabled is True
    
    def test_validate_plasma_inputs_valid(self):
        """Test validation with valid plasma inputs."""
        # Should not raise any exceptions
        self.solver._validate_plasma_inputs(
            self.config, self.flow_conditions, self.magnetic_field
        )
    
    def test_validate_plasma_inputs_invalid_magnetic_field(self):
        """Test validation with invalid magnetic field."""
        invalid_field = np.array([1.0, 2.0])  # Wrong shape
        
        with pytest.raises(ValidationError, match="Magnetic field must be a 3D vector"):
            self.solver._validate_plasma_inputs(
                self.config, self.flow_conditions, invalid_field
            )
    
    def test_validate_plasma_inputs_missing_temperature(self):
        """Test validation with missing temperature."""
        flow_conditions_no_temp = FlowConditions(
            mach_number=60.0,
            altitude=50000.0,
            angle_of_attack=0.0,
            sideslip_angle=0.0,
            temperature=None  # Missing temperature
        )
        
        with pytest.raises(ValidationError, match="Temperature is required"):
            self.solver._validate_plasma_inputs(
                self.config, flow_conditions_no_temp, self.magnetic_field
            )
    
    def test_validate_plasma_inputs_low_mach_warning(self, caplog):
        """Test validation warning for low Mach number."""
        low_mach_conditions = FlowConditions(
            mach_number=10.0,  # Below plasma threshold
            altitude=50000.0,
            angle_of_attack=0.0,
            sideslip_angle=0.0,
            temperature=5000.0
        )
        
        self.solver._validate_plasma_inputs(
            self.config, low_mach_conditions, self.magnetic_field
        )
        
        assert "may be too low for significant plasma effects" in caplog.text
    
    def test_get_default_mhd_settings_high_mach(self):
        """Test default MHD settings for high Mach number."""
        settings = self.solver._get_default_mhd_settings(self.flow_conditions)
        
        assert isinstance(settings, MHDSolverSettings)
        assert settings.plasma_model == "single_fluid"
        assert settings.magnetic_field_coupling is True
        assert settings.electromagnetic_source_terms is True
        assert settings.hall_effect is True
        assert settings.ion_slip is True
        assert settings.plasma_chemistry is True  # Mach 60 > 40
    
    def test_get_default_mhd_settings_medium_mach(self):
        """Test default MHD settings for medium Mach number."""
        medium_mach_conditions = FlowConditions(
            mach_number=35.0,
            altitude=50000.0,
            angle_of_attack=0.0,
            sideslip_angle=0.0,
            temperature=10000.0
        )
        
        settings = self.solver._get_default_mhd_settings(medium_mach_conditions)
        
        assert settings.hall_effect is True
        assert settings.ion_slip is False
        assert settings.plasma_chemistry is False  # Mach 35 < 40
    
    def test_get_default_mhd_settings_low_mach(self):
        """Test default MHD settings for low Mach number."""
        low_mach_conditions = FlowConditions(
            mach_number=20.0,
            altitude=50000.0,
            angle_of_attack=0.0,
            sideslip_angle=0.0,
            temperature=8000.0
        )
        
        settings = self.solver._get_default_mhd_settings(low_mach_conditions)
        
        assert settings.hall_effect is False
        assert settings.ion_slip is False
        assert settings.plasma_chemistry is False
    
    @patch('fighter_jet_sdk.engines.aerodynamics.plasma_flow_solver.PlasmaPropertiesCalculator')
    @patch('fighter_jet_sdk.engines.aerodynamics.plasma_flow_solver.ElectromagneticEffectsCalculator')
    def test_calculate_plasma_flow_conditions(self, mock_em_calc, mock_plasma_calc):
        """Test calculation of plasma flow conditions."""
        # Setup mocks
        mock_plasma_calc.return_value.calculate_complete_plasma_conditions.return_value = self.plasma_conditions
        mock_em_calc.return_value.calculate_complete_electromagnetic_properties.return_value = self.em_properties
        
        # Create solver with mocked calculators
        solver = PlasmaFlowSolver()
        solver.plasma_calculator = mock_plasma_calc.return_value
        solver.electromagnetic_calculator = mock_em_calc.return_value
        
        # Calculate plasma flow conditions
        plasma_flow_conditions = solver._calculate_plasma_flow_conditions(
            self.flow_conditions, self.magnetic_field
        )
        
        # Verify results
        assert isinstance(plasma_flow_conditions, PlasmaFlowConditions)
        assert plasma_flow_conditions.base_conditions == self.flow_conditions
        assert plasma_flow_conditions.plasma_conditions == self.plasma_conditions
        assert plasma_flow_conditions.electromagnetic_properties == self.em_properties
        assert np.array_equal(plasma_flow_conditions.magnetic_field, self.magnetic_field)
        
        # Verify calculator calls
        mock_plasma_calc.return_value.calculate_complete_plasma_conditions.assert_called_once()
        mock_em_calc.return_value.calculate_complete_electromagnetic_properties.assert_called_once()
    
    def test_plasma_flow_conditions_properties(self):
        """Test PlasmaFlowConditions property access."""
        plasma_flow_conditions = PlasmaFlowConditions(
            base_conditions=self.flow_conditions,
            plasma_conditions=self.plasma_conditions,
            electromagnetic_properties=self.em_properties,
            magnetic_field=self.magnetic_field,
            electric_field=np.array([1000.0, 0.0, 0.0])
        )
        
        assert plasma_flow_conditions.mach_number == 60.0
        assert plasma_flow_conditions.temperature == 15000.0
        assert plasma_flow_conditions.pressure == 1000.0
    
    @patch('tempfile.mkdtemp')
    @patch('os.makedirs')
    def test_create_mhd_properties(self, mock_makedirs, mock_mkdtemp):
        """Test creation of MHD properties file."""
        mock_mkdtemp.return_value = "/tmp/test_case"
        
        plasma_flow_conditions = PlasmaFlowConditions(
            base_conditions=self.flow_conditions,
            plasma_conditions=self.plasma_conditions,
            electromagnetic_properties=self.em_properties,
            magnetic_field=self.magnetic_field,
            electric_field=np.array([1000.0, 0.0, 0.0])
        )
        
        with patch('builtins.open', create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            self.solver._create_mhd_properties("/tmp/test_case", plasma_flow_conditions)
            
            # Verify file was opened for writing
            mock_open.assert_called_once_with("/tmp/test_case/constant/MHDProperties", 'w')
            
            # Verify content was written
            mock_file.write.assert_called_once()
            written_content = mock_file.write.call_args[0][0]
            
            # Check key properties are in the file
            assert "electronDensity" in written_content
            assert "plasmaConductivity" in written_content
            assert "hallParameter" in written_content
            assert "magneticField" in written_content
            assert str(self.plasma_conditions.electron_density) in written_content
    
    @patch('tempfile.mkdtemp')
    @patch('os.makedirs')
    def test_create_electromagnetic_fields(self, mock_makedirs, mock_mkdtemp):
        """Test creation of electromagnetic field files."""
        mock_mkdtemp.return_value = "/tmp/test_case"
        
        plasma_flow_conditions = PlasmaFlowConditions(
            base_conditions=self.flow_conditions,
            plasma_conditions=self.plasma_conditions,
            electromagnetic_properties=self.em_properties,
            magnetic_field=self.magnetic_field,
            electric_field=np.array([1000.0, 0.0, 0.0])
        )
        
        with patch('builtins.open', create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            self.solver._create_electromagnetic_fields("/tmp/test_case", plasma_flow_conditions)
            
            # Should create B, E, and J field files
            assert mock_open.call_count == 3
            
            # Check that B, E, J files were created
            call_args = [call[0][0] for call in mock_open.call_args_list]
            assert any("B" in path for path in call_args)
            assert any("E" in path for path in call_args)
            assert any("J" in path for path in call_args)
    
    def test_calculate_plasma_source_terms(self):
        """Test calculation of electromagnetic source terms."""
        velocity_field = np.zeros((10, 10, 3))
        velocity_field[:, :, 0] = 20000.0  # 20 km/s in x-direction
        
        source_terms = self.solver.calculate_plasma_source_terms(
            self.plasma_conditions, velocity_field, self.magnetic_field
        )
        
        # Verify source terms structure
        assert "momentum_x" in source_terms
        assert "momentum_y" in source_terms
        assert "momentum_z" in source_terms
        assert "energy" in source_terms
        
        # Verify array shapes
        assert source_terms["momentum_x"].shape == (10, 10)
        assert source_terms["momentum_y"].shape == (10, 10)
        assert source_terms["momentum_z"].shape == (10, 10)
        assert source_terms["energy"].shape == (10, 10)
    
    @patch('fighter_jet_sdk.engines.aerodynamics.plasma_flow_solver.PlasmaPropertiesCalculator')
    def test_update_plasma_properties(self, mock_plasma_calc):
        """Test updating plasma properties from flow fields."""
        mock_plasma_calc.return_value.calculate_complete_plasma_conditions.return_value = self.plasma_conditions
        
        solver = PlasmaFlowSolver()
        solver.plasma_calculator = mock_plasma_calc.return_value
        
        # Create test fields
        temperature_field = np.full((10, 10), 15000.0)
        pressure_field = np.full((10, 10), 1000.0)
        
        updated_conditions = solver.update_plasma_properties(
            temperature_field, pressure_field, self.magnetic_field
        )
        
        assert isinstance(updated_conditions, PlasmaConditions)
        mock_plasma_calc.return_value.calculate_complete_plasma_conditions.assert_called_once()
    
    @patch('subprocess.run')
    @patch('tempfile.mkdtemp')
    def test_run_mhd_solver_success(self, mock_mkdtemp, mock_subprocess):
        """Test successful MHD solver execution."""
        mock_mkdtemp.return_value = "/tmp/test_case"
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "MHD solver completed successfully"
        
        settings = MHDSolverSettings(
            base_settings=SolverSettings(
                solver_type="rhoSimpleFoam",
                turbulence_model="kOmegaSST",
                max_iterations=1000,
                convergence_tolerance=1e-6,
                relaxation_factors={"p": 0.3, "U": 0.7}
            )
        )
        
        with patch.object(self.solver, '_parse_mhd_solver_output') as mock_parse:
            mock_parse.return_value = {"forces": {"drag": 1000.0}}
            
            result = self.solver._run_mhd_solver("/tmp/test_case", settings)
            
            # Verify solver was called
            mock_subprocess.assert_called_once()
            assert mock_subprocess.call_args[0][0] == ["mhdFoam", "-case", "/tmp/test_case"]
            
            # Verify parsing was called
            mock_parse.assert_called_once()
    
    @patch('subprocess.run')
    def test_run_mhd_solver_fallback(self, mock_subprocess):
        """Test MHD solver fallback to standard CFD."""
        mock_subprocess.return_value.returncode = 1  # MHD solver fails
        mock_subprocess.return_value.stderr = "mhdFoam not found"
        
        settings = MHDSolverSettings(
            base_settings=SolverSettings(
                solver_type="rhoSimpleFoam",
                turbulence_model="kOmegaSST",
                max_iterations=1000,
                convergence_tolerance=1e-6,
                relaxation_factors={"p": 0.3, "U": 0.7}
            )
        )
        
        with patch.object(self.solver, '_run_solver') as mock_run_solver:
            mock_run_solver.return_value = {"forces": {"drag": 500.0}}
            
            result = self.solver._run_mhd_solver("/tmp/test_case", settings)
            
            # Verify fallback was used
            mock_run_solver.assert_called_once_with("/tmp/test_case", settings.base_settings)
    
    def test_parse_mhd_solver_output(self):
        """Test parsing of MHD solver output."""
        solver_output = """
        MHD Solver Output
        Iteration 100: Residuals - U: 1e-6, p: 1e-7, B: 1e-8, E: 1e-8
        Forces: Drag = 1000.0 N, Lift = 500.0 N
        """
        
        with patch.object(self.solver, '_parse_solver_output') as mock_parse_base:
            mock_parse_base.return_value = {
                "forces": {"drag": 1000.0, "lift": 500.0},
                "residuals": {"U": [1e-6], "p": [1e-7]}
            }
            
            result = self.solver._parse_mhd_solver_output(solver_output, "/tmp/test_case")
            
            # Verify base parsing was called
            mock_parse_base.assert_called_once()
            
            # Verify MHD-specific results were added
            assert "electromagnetic_residuals" in result
            assert "plasma_properties" in result
            assert "electromagnetic_fields" in result
            
            # Check structure of added results
            assert "B" in result["electromagnetic_residuals"]
            assert "E" in result["electromagnetic_residuals"]
            assert "J" in result["electromagnetic_residuals"]
    
    @patch.object(PlasmaFlowSolver, '_setup_mhd_case')
    @patch.object(PlasmaFlowSolver, '_run_mhd_solver')
    @patch.object(PlasmaFlowSolver, '_calculate_plasma_flow_conditions')
    @patch.object(PlasmaFlowSolver, '_post_process_plasma_results')
    def test_analyze_plasma_flow_integration(self, mock_post_process, mock_calc_conditions, 
                                           mock_run_solver, mock_setup_case):
        """Test full plasma flow analysis integration."""
        # Setup mocks
        plasma_flow_conditions = PlasmaFlowConditions(
            base_conditions=self.flow_conditions,
            plasma_conditions=self.plasma_conditions,
            electromagnetic_properties=self.em_properties,
            magnetic_field=self.magnetic_field,
            electric_field=np.array([1000.0, 0.0, 0.0])
        )
        
        mock_calc_conditions.return_value = plasma_flow_conditions
        mock_setup_case.return_value = "/tmp/test_case"
        mock_run_solver.return_value = {"forces": {"drag": 1000.0}}
        
        # Create mock results
        base_results = CFDResults(
            forces={"drag": 1000.0, "lift": 500.0, "side_force": 0.0},
            moments={"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
            pressure_distribution=np.zeros((100, 100)),
            velocity_field=np.zeros((100, 100, 3)),
            convergence_history=[1e-3, 1e-4, 1e-5, 1e-6],
            residuals={"U": [1e-6], "p": [1e-7]},
            flow_regime=FlowRegime.HYPERSONIC,
            mach_number=60.0,
            reynolds_number=1e6
        )
        
        plasma_results = PlasmaFlowResults(
            base_results=base_results,
            plasma_conditions_field=np.full((100, 100), self.plasma_conditions),
            electromagnetic_field=np.full((100, 100), self.em_properties),
            current_density_field=np.zeros((100, 100, 3)),
            lorentz_force_field=np.zeros((100, 100, 3)),
            joule_heating_field=np.zeros((100, 100)),
            plasma_regime_field=np.full((100, 100), PlasmaRegime.PARTIALLY_IONIZED),
            electron_density_field=np.full((100, 100), 1e20),
            ionization_fraction_field=np.full((100, 100), 0.1),
            plasma_conductivity_field=np.full((100, 100), 1000.0),
            magnetic_reynolds_number=100.0,
            hall_parameter_field=np.full((100, 100), 0.5)
        )
        
        mock_post_process.return_value = plasma_results
        
        # Run analysis
        result = self.solver.analyze_plasma_flow(
            self.config, self.flow_conditions, self.magnetic_field
        )
        
        # Verify all steps were called
        mock_calc_conditions.assert_called_once()
        mock_setup_case.assert_called_once()
        mock_run_solver.assert_called_once()
        mock_post_process.assert_called_once()
        
        # Verify result type
        assert isinstance(result, PlasmaFlowResults)
        assert result.magnetic_reynolds_number == 100.0
    
    def test_mhd_solver_settings_defaults(self):
        """Test MHDSolverSettings default values."""
        base_settings = SolverSettings(
            solver_type="rhoSimpleFoam",
            turbulence_model="kOmegaSST",
            max_iterations=1000,
            convergence_tolerance=1e-6,
            relaxation_factors={"p": 0.3, "U": 0.7}
        )
        
        mhd_settings = MHDSolverSettings(base_settings=base_settings)
        
        assert mhd_settings.plasma_model == "single_fluid"
        assert mhd_settings.magnetic_field_coupling is True
        assert mhd_settings.electromagnetic_source_terms is True
        assert mhd_settings.plasma_chemistry is False
        assert mhd_settings.hall_effect is True
        assert mhd_settings.ion_slip is False
        assert mhd_settings.plasma_time_step_factor == 0.1
        assert mhd_settings.electromagnetic_iterations == 5
        assert mhd_settings.plasma_convergence_tolerance == 1e-8


if __name__ == "__main__":
    pytest.main([__file__])