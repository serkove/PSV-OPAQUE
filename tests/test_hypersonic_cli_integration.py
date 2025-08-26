"""Tests for hypersonic CLI integration."""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from fighter_jet_sdk.cli.main import (
    handle_hypersonic_command,
    handle_hypersonic_mission,
    handle_hypersonic_plasma,
    handle_hypersonic_thermal,
    handle_hypersonic_propulsion,
    handle_hypersonic_vehicle,
    handle_hypersonic_compare,
    create_cli
)


class TestHypersonicCLI:
    """Test hypersonic CLI commands."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.json')
        self.output_file = os.path.join(self.temp_dir, 'output.json')
        
        # Create test configuration
        test_config = {
            'name': 'Test Hypersonic Vehicle',
            'type': 'hypersonic',
            'mach_capability': 60.0,
            'propulsion': {
                'type': 'combined_cycle',
                'fuel': 'hydrogen'
            },
            'thermal_protection': {
                'type': 'hybrid',
                'max_heat_flux': 150.0
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(test_config, f)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_cli_parser_includes_hypersonic_commands(self):
        """Test that CLI parser includes hypersonic commands."""
        parser = create_cli()
        
        # Test that hypersonic command is available
        help_text = parser.format_help()
        assert 'hypersonic' in help_text
        
        # Test parsing hypersonic mission command
        args = parser.parse_args(['hypersonic', 'mission', '--config', 'test.json'])
        assert args.command == 'hypersonic'
        assert args.hypersonic_action == 'mission'
        assert args.config == 'test.json'
    
    def test_hypersonic_mission_command_parsing(self):
        """Test hypersonic mission command argument parsing."""
        parser = create_cli()
        
        args = parser.parse_args([
            'hypersonic', 'mission',
            '--config', 'config.json',
            '--mach-target', '60',
            '--altitude-range', '40,100',
            '--optimize',
            '--output', 'mission.json'
        ])
        
        assert args.hypersonic_action == 'mission'
        assert args.config == 'config.json'
        assert args.mach_target == 60.0
        assert args.altitude_range == '40,100'
        assert args.optimize is True
        assert args.output == 'mission.json'
    
    def test_hypersonic_plasma_command_parsing(self):
        """Test hypersonic plasma command argument parsing."""
        parser = create_cli()
        
        args = parser.parse_args([
            'hypersonic', 'plasma',
            '--geometry', 'vehicle.stl',
            '--mach', '60',
            '--altitude', '60000',
            '--chemistry',
            '--output', 'plasma_results'
        ])
        
        assert args.hypersonic_action == 'plasma'
        assert args.geometry == 'vehicle.stl'
        assert args.mach == 60.0
        assert args.altitude == 60000.0
        assert args.chemistry is True
        assert args.output == 'plasma_results'
    
    def test_hypersonic_thermal_command_parsing(self):
        """Test hypersonic thermal command argument parsing."""
        parser = create_cli()
        
        args = parser.parse_args([
            'hypersonic', 'thermal',
            '--config', 'config.json',
            '--heat-flux', '150',
            '--cooling-type', 'hybrid',
            '--optimize',
            '--output', 'tps_design.json'
        ])
        
        assert args.hypersonic_action == 'thermal'
        assert args.config == 'config.json'
        assert args.heat_flux == 150.0
        assert args.cooling_type == 'hybrid'
        assert args.optimize is True
        assert args.output == 'tps_design.json'
    
    def test_hypersonic_propulsion_command_parsing(self):
        """Test hypersonic propulsion command argument parsing."""
        parser = create_cli()
        
        args = parser.parse_args([
            'hypersonic', 'propulsion',
            '--engine', 'engine.yaml',
            '--fuel-type', 'hydrogen',
            '--transition-mach', '25',
            '--analyze-performance',
            '--output', 'propulsion.json'
        ])
        
        assert args.hypersonic_action == 'propulsion'
        assert args.engine == 'engine.yaml'
        assert args.fuel_type == 'hydrogen'
        assert args.transition_mach == 25.0
        assert args.analyze_performance is True
        assert args.output == 'propulsion.json'
    
    def test_hypersonic_vehicle_command_parsing(self):
        """Test hypersonic vehicle command argument parsing."""
        parser = create_cli()
        
        args = parser.parse_args([
            'hypersonic', 'vehicle',
            '--config', 'vehicle.json',
            '--validate',
            '--multi-physics',
            '--safety-margins',
            '--output-dir', './analysis'
        ])
        
        assert args.hypersonic_action == 'vehicle'
        assert args.config == 'vehicle.json'
        assert args.validate is True
        assert args.multi_physics is True
        assert args.safety_margins is True
        assert args.output_dir == './analysis'
    
    def test_hypersonic_compare_command_parsing(self):
        """Test hypersonic compare command argument parsing."""
        parser = create_cli()
        
        args = parser.parse_args([
            'hypersonic', 'compare',
            '--configs', 'config1.json', 'config2.json', 'config3.json',
            '--baseline', 'config1.json',
            '--output', 'comparison.json'
        ])
        
        assert args.hypersonic_action == 'compare'
        assert args.configs == ['config1.json', 'config2.json', 'config3.json']
        assert args.baseline == 'config1.json'
        assert args.output == 'comparison.json'
    
    @patch('fighter_jet_sdk.core.hypersonic_mission_planner.HypersonicMissionPlanner')
    @patch('fighter_jet_sdk.core.config.get_config_manager')
    def test_hypersonic_mission_handler(self, mock_config_manager, mock_planner_class):
        """Test hypersonic mission command handler."""
        # Mock the mission planner
        mock_planner = Mock()
        mock_planner_class.return_value = mock_planner
        mock_planner.plan_hypersonic_mission.return_value = {
            'optimal_altitude': 65000,
            'flight_time': 3600,
            'fuel_consumption': 5000,
            'max_thermal_load': 120.0
        }
        
        # Create mock arguments
        args = Mock()
        args.config = self.config_file
        args.mach_target = 60.0
        args.altitude_range = '40,100'
        args.profile = None
        args.optimize = False
        args.output = self.output_file
        
        # Test the handler
        result = handle_hypersonic_mission(args)
        
        assert result == 0
        mock_planner.plan_hypersonic_mission.assert_called_once()
        
        # Check output file was created
        assert os.path.exists(self.output_file)
        with open(self.output_file, 'r') as f:
            output_data = json.load(f)
        assert 'optimal_altitude' in output_data
    
    @patch('fighter_jet_sdk.engines.aerodynamics.plasma_flow_solver.PlasmaFlowSolver')
    def test_hypersonic_plasma_handler(self, mock_solver_class):
        """Test hypersonic plasma command handler."""
        # Mock the plasma solver
        mock_solver = Mock()
        mock_solver_class.return_value = mock_solver
        mock_solver.solve_plasma_flow.return_value = {
            'plasma_density': 1e15,
            'electron_temperature': 8000,
            'blackout_region': 'minimal'
        }
        
        # Create mock arguments
        args = Mock()
        args.geometry = 'vehicle.stl'
        args.mach = 60.0
        args.altitude = 60000.0
        args.magnetic_field = None
        args.chemistry = False
        args.output = None
        
        # Test the handler
        result = handle_hypersonic_plasma(args)
        
        assert result == 0
        mock_solver.solve_plasma_flow.assert_called_once()
    
    @patch('fighter_jet_sdk.engines.propulsion.extreme_heat_flux_model.ExtremeHeatFluxModel')
    @patch('fighter_jet_sdk.engines.materials.thermal_materials_db.ThermalMaterialsDB')
    def test_hypersonic_thermal_handler(self, mock_materials_class, mock_heat_flux_class):
        """Test hypersonic thermal command handler."""
        # Mock the thermal models
        mock_heat_flux = Mock()
        mock_materials = Mock()
        mock_heat_flux_class.return_value = mock_heat_flux
        mock_materials_class.return_value = mock_materials
        mock_materials.get_ultra_high_temp_materials.return_value = ['UHTC-1', 'UHTC-2']
        
        # Create mock arguments
        args = Mock()
        args.config = self.config_file
        args.heat_flux = 150.0
        args.cooling_type = 'hybrid'
        args.materials = None
        args.optimize = False
        args.output = self.output_file
        
        # Test the handler
        result = handle_hypersonic_thermal(args)
        
        assert result == 0
        
        # Check output file was created
        assert os.path.exists(self.output_file)
        with open(self.output_file, 'r') as f:
            output_data = json.load(f)
        assert 'heat_flux' in output_data
        assert 'cooling_type' in output_data
    
    @patch('fighter_jet_sdk.engines.propulsion.combined_cycle_engine.CombinedCycleEngine')
    def test_hypersonic_propulsion_handler(self, mock_engine_class):
        """Test hypersonic propulsion command handler."""
        # Mock the combined cycle engine
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine
        mock_engine.calculate_performance.return_value = {
            'thrust': 100000,
            'specific_impulse': 3500,
            'fuel_flow': 50
        }
        
        # Create mock arguments
        args = Mock()
        args.engine = self.config_file
        args.flight_envelope = None
        args.transition_mach = 25.0
        args.fuel_type = 'hydrogen'
        args.analyze_performance = True
        args.output = self.output_file
        
        # Test the handler
        result = handle_hypersonic_propulsion(args)
        
        assert result == 0
        
        # Check output file was created
        assert os.path.exists(self.output_file)
        with open(self.output_file, 'r') as f:
            output_data = json.load(f)
        assert 'transition_analysis' in output_data
    
    @patch('fighter_jet_sdk.core.hypersonic_design_validator.HypersonicDesignValidator')
    @patch('fighter_jet_sdk.core.multi_physics_integration.MultiPhysicsIntegration')
    def test_hypersonic_vehicle_handler(self, mock_physics_class, mock_validator_class):
        """Test hypersonic vehicle command handler."""
        # Mock the validator and physics integration
        mock_validator = Mock()
        mock_physics = Mock()
        mock_validator_class.return_value = mock_validator
        mock_physics_class.return_value = mock_physics
        
        mock_validator.validate_mach60_design.return_value = {'overall_status': True}
        mock_physics.run_coupled_analysis.return_value = {'convergence': True}
        mock_validator.calculate_safety_margins.return_value = {'min_margin': 1.5}
        
        # Create mock arguments
        args = Mock()
        args.config = self.config_file
        args.mission = None
        args.validate = True
        args.multi_physics = True
        args.safety_margins = True
        args.output_dir = self.temp_dir
        
        # Test the handler
        result = handle_hypersonic_vehicle(args)
        
        assert result == 0
        
        # Check output file was created
        output_file = os.path.join(self.temp_dir, 'vehicle_analysis.json')
        assert os.path.exists(output_file)
    
    def test_hypersonic_compare_handler(self):
        """Test hypersonic compare command handler."""
        # Create additional config files
        config2_file = os.path.join(self.temp_dir, 'config2.json')
        config3_file = os.path.join(self.temp_dir, 'config3.json')
        
        test_config2 = {'name': 'Vehicle 2', 'mach_capability': 55.0}
        test_config3 = {'name': 'Vehicle 3', 'mach_capability': 65.0}
        
        with open(config2_file, 'w') as f:
            json.dump(test_config2, f)
        with open(config3_file, 'w') as f:
            json.dump(test_config3, f)
        
        # Create mock arguments
        args = Mock()
        args.configs = [self.config_file, config2_file, config3_file]
        args.metrics = None
        args.baseline = self.config_file
        args.output = self.output_file
        
        # Test the handler
        result = handle_hypersonic_compare(args)
        
        assert result == 0
        
        # Check output file was created
        assert os.path.exists(self.output_file)
        with open(self.output_file, 'r') as f:
            output_data = json.load(f)
        assert 'configurations' in output_data
        assert len(output_data['configurations']) == 3
    
    def test_hypersonic_command_dispatcher(self):
        """Test hypersonic command dispatcher."""
        # Test valid action
        args = Mock()
        args.hypersonic_action = 'mission'
        
        with patch('fighter_jet_sdk.cli.main.handle_hypersonic_mission', return_value=0) as mock_handler:
            result = handle_hypersonic_command(args)
            assert result == 0
            mock_handler.assert_called_once_with(args)
        
        # Test invalid action
        args.hypersonic_action = 'invalid_action'
        result = handle_hypersonic_command(args)
        assert result == 1
    
    def test_error_handling_missing_config(self):
        """Test error handling for missing configuration files."""
        args = Mock()
        args.config = 'nonexistent_config.json'
        args.mach_target = 60.0
        args.altitude_range = None
        args.profile = None
        args.optimize = False
        args.output = None
        
        result = handle_hypersonic_mission(args)
        assert result == 1
    
    def test_error_handling_invalid_altitude_range(self):
        """Test error handling for invalid altitude range format."""
        args = Mock()
        args.config = self.config_file
        args.mach_target = 60.0
        args.altitude_range = 'invalid_format'
        args.profile = None
        args.optimize = False
        args.output = None
        
        with patch('fighter_jet_sdk.core.config.get_config_manager'):
            with patch('fighter_jet_sdk.core.hypersonic_mission_planner.HypersonicMissionPlanner'):
                result = handle_hypersonic_mission(args)
                assert result == 1


class TestHypersonicCLIIntegration:
    """Integration tests for hypersonic CLI functionality."""
    
    def test_end_to_end_mission_planning(self):
        """Test end-to-end mission planning workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test configuration
            config_file = os.path.join(temp_dir, 'vehicle.json')
            config_data = {
                'name': 'Test Hypersonic Vehicle',
                'type': 'hypersonic',
                'mach_capability': 60.0
            }
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f)
            
            # Mock the mission planner
            with patch('fighter_jet_sdk.core.hypersonic_mission_planner.HypersonicMissionPlanner') as mock_planner_class:
                mock_planner = Mock()
                mock_planner_class.return_value = mock_planner
                mock_planner.plan_hypersonic_mission.return_value = {
                    'status': 'success',
                    'optimal_altitude': 65000
                }
                
                # Create arguments
                args = Mock()
                args.config = config_file
                args.mach_target = 60.0
                args.altitude_range = None
                args.profile = None
                args.optimize = False
                args.output = None
                
                # Test the workflow
                with patch('fighter_jet_sdk.core.config.get_config_manager'):
                    result = handle_hypersonic_mission(args)
                    assert result == 0
    
    def test_cli_help_system_includes_hypersonic(self):
        """Test that CLI help system includes hypersonic commands."""
        parser = create_cli()
        
        # Test main help includes hypersonic
        help_text = parser.format_help()
        assert 'hypersonic' in help_text.lower()
        
        # Test hypersonic subcommand help
        try:
            args = parser.parse_args(['hypersonic', '--help'])
        except SystemExit:
            # argparse calls sys.exit() for help, which is expected
            pass
    
    def test_batch_processing_compatibility(self):
        """Test that hypersonic commands work with batch processing."""
        # This would test integration with the batch processing system
        # For now, just verify the command structure is compatible
        parser = create_cli()
        
        # Test that hypersonic commands can be parsed in batch context
        args = parser.parse_args(['hypersonic', 'mission', '--config', 'test.json'])
        assert hasattr(args, 'hypersonic_action')
        assert hasattr(args, 'config')


if __name__ == '__main__':
    pytest.main([__file__])