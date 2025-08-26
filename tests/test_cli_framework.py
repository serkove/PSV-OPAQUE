"""Tests for the CLI framework and command orchestration."""

import pytest
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

from fighter_jet_sdk.cli.main import (
    create_cli, main, InteractiveCLI, BatchProcessor,
    format_output, handle_config_command, handle_help_command,
    handle_examples_command
)
from fighter_jet_sdk.cli.help_system import HelpSystem, get_help, get_examples


class TestCLIFramework:
    """Test the main CLI framework."""
    
    def test_create_cli_parser(self):
        """Test CLI parser creation."""
        parser = create_cli()
        
        # Test that parser is created
        assert parser is not None
        assert parser.prog == 'fighter-jet-sdk'
        
        # Test that all main commands are available
        help_text = parser.format_help()
        expected_commands = [
            'config', 'help', 'examples', 'interactive', 'batch',
            'design', 'materials', 'propulsion', 'sensors',
            'aerodynamics', 'manufacturing', 'simulate'
        ]
        
        for command in expected_commands:
            assert command in help_text
    
    def test_global_options(self):
        """Test global CLI options."""
        parser = create_cli()
        
        # Test version option
        with pytest.raises(SystemExit):
            parser.parse_args(['--version'])
        
        # Test config option
        args = parser.parse_args(['--config', 'test.yaml', 'config', 'show'])
        assert args.config == 'test.yaml'
        
        # Test log level option
        args = parser.parse_args(['--log-level', 'DEBUG', 'config', 'show'])
        assert args.log_level == 'DEBUG'
        
        # Test output format option
        args = parser.parse_args(['--output-format', 'json', 'config', 'show'])
        assert args.output_format == 'json'
    
    def test_config_commands(self):
        """Test configuration commands parsing."""
        parser = create_cli()
        
        # Test config init
        args = parser.parse_args(['config', 'init'])
        assert args.command == 'config'
        assert args.config_action == 'init'
        
        # Test config show
        args = parser.parse_args(['config', 'show'])
        assert args.command == 'config'
        assert args.config_action == 'show'
        
        # Test config validate
        args = parser.parse_args(['config', 'validate'])
        assert args.command == 'config'
        assert args.config_action == 'validate'
    
    def test_design_commands(self):
        """Test design engine commands parsing."""
        parser = create_cli()
        
        # Test design create
        args = parser.parse_args(['design', 'create', '--name', 'TestJet'])
        assert args.command == 'design'
        assert args.design_action == 'create'
        assert args.name == 'TestJet'
        
        # Test design list
        args = parser.parse_args(['design', 'list', '--type', 'modules'])
        assert args.command == 'design'
        assert args.design_action == 'list'
        assert args.type == 'modules'
        
        # Test design validate
        args = parser.parse_args(['design', 'validate', '--config', 'test.json'])
        assert args.command == 'design'
        assert args.design_action == 'validate'
        assert args.config == 'test.json'
    
    def test_materials_commands(self):
        """Test materials engine commands parsing."""
        parser = create_cli()
        
        # Test materials list
        args = parser.parse_args(['materials', 'list'])
        assert args.command == 'materials'
        assert args.materials_action == 'list'
        
        # Test materials metamaterial
        args = parser.parse_args(['materials', 'metamaterial', '--material', 'META001'])
        assert args.command == 'materials'
        assert args.materials_action == 'metamaterial'
        assert args.material == 'META001'
        
        # Test materials stealth
        args = parser.parse_args(['materials', 'stealth', '--geometry', 'aircraft.stl'])
        assert args.command == 'materials'
        assert args.materials_action == 'stealth'
        assert args.geometry == 'aircraft.stl'
    
    def test_propulsion_commands(self):
        """Test propulsion engine commands parsing."""
        parser = create_cli()
        
        # Test propulsion list
        args = parser.parse_args(['propulsion', 'list'])
        assert args.command == 'propulsion'
        assert args.propulsion_action == 'list'
        
        # Test propulsion analyze
        args = parser.parse_args([
            'propulsion', 'analyze', '--engine', 'f119_pw_100',
            '--altitude', '10000', '--mach', '1.5', '--afterburner'
        ])
        assert args.command == 'propulsion'
        assert args.propulsion_action == 'analyze'
        assert args.engine == 'f119_pw_100'
        assert args.altitude == 10000.0
        assert args.mach == 1.5
        assert args.afterburner is True
        
        # Test propulsion optimize
        args = parser.parse_args([
            'propulsion', 'optimize', '--engine', 'f119_pw_100', '--mass', '19700'
        ])
        assert args.command == 'propulsion'
        assert args.propulsion_action == 'optimize'
        assert args.engine == 'f119_pw_100'
        assert args.mass == 19700.0
    
    def test_batch_commands(self):
        """Test batch processing commands parsing."""
        parser = create_cli()
        
        # Test batch command
        args = parser.parse_args(['batch', '--script', 'test.yaml'])
        assert args.command == 'batch'
        assert args.script == 'test.yaml'
        assert args.parallel is False
        
        # Test batch with parallel
        args = parser.parse_args(['batch', '--script', 'test.yaml', '--parallel'])
        assert args.command == 'batch'
        assert args.script == 'test.yaml'
        assert args.parallel is True
    
    def test_interactive_command(self):
        """Test interactive mode command parsing."""
        parser = create_cli()
        
        args = parser.parse_args(['interactive'])
        assert args.command == 'interactive'
    
    def test_help_commands(self):
        """Test help system commands parsing."""
        parser = create_cli()
        
        # Test help command
        args = parser.parse_args(['help'])
        assert args.command == 'help'
        assert args.help_command is None
        
        # Test help with command
        args = parser.parse_args(['help', 'design'])
        assert args.command == 'help'
        assert args.help_command == 'design'
        
        # Test help with subcommand
        args = parser.parse_args(['help', 'design', 'create'])
        assert args.command == 'help'
        assert args.help_command == 'design'
        assert args.help_subcommand == 'create'
        
        # Test examples command
        args = parser.parse_args(['examples'])
        assert args.command == 'examples'
        assert args.category is None
        
        # Test examples with category
        args = parser.parse_args(['examples', '--category', 'basic'])
        assert args.command == 'examples'
        assert args.category == 'basic'


class TestInteractiveCLI:
    """Test the interactive CLI system."""
    
    def test_interactive_cli_creation(self):
        """Test interactive CLI initialization."""
        with patch('fighter_jet_sdk.cli.main.get_config_manager'), \
             patch('fighter_jet_sdk.cli.main.get_log_manager'):
            cli = InteractiveCLI()
            assert cli is not None
            assert cli.prompt == 'fighter-jet-sdk> '
    
    @patch('fighter_jet_sdk.cli.main.get_config_manager')
    @patch('fighter_jet_sdk.cli.main.get_log_manager')
    def test_interactive_commands(self, mock_log_manager, mock_config_manager):
        """Test interactive command processing."""
        cli = InteractiveCLI()
        
        # Test status command
        with patch('builtins.print') as mock_print:
            cli.do_status('')
            mock_print.assert_called()
        
        # Test config show command
        mock_config = Mock()
        mock_config.log_level = 'INFO'
        mock_config.data_directory = '/data'
        mock_config.parallel_processing = True
        mock_config.cache_enabled = True
        mock_config_manager.return_value.get_config.return_value = mock_config
        
        with patch('builtins.print') as mock_print:
            cli.do_config('show')
            mock_print.assert_called()


class TestBatchProcessor:
    """Test the batch processing system."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_batch_processor_creation(self):
        """Test batch processor initialization."""
        mock_config_manager = Mock()
        processor = BatchProcessor(mock_config_manager)
        assert processor is not None
        assert processor.config_manager == mock_config_manager
    
    def test_yaml_batch_script_loading(self):
        """Test loading YAML batch scripts."""
        # Create test YAML batch script
        batch_script = {
            'operations': [
                {
                    'name': 'Test Operation',
                    'type': 'propulsion',
                    'action': 'analyze',
                    'engine_id': 'f119_pw_100',
                    'conditions': {
                        'altitude': 10000,
                        'mach_number': 1.5
                    }
                }
            ]
        }
        
        script_path = self.temp_path / 'test_batch.yaml'
        with open(script_path, 'w') as f:
            yaml.dump(batch_script, f)
        
        mock_config_manager = Mock()
        processor = BatchProcessor(mock_config_manager)
        
        # Test processing
        results = processor.process_batch_file(str(script_path))
        
        assert results['script_path'] == str(script_path)
        assert results['summary']['total'] == 1
        assert len(results['operations']) == 1
    
    def test_json_batch_script_loading(self):
        """Test loading JSON batch scripts."""
        # Create test JSON batch script
        batch_script = {
            'operations': [
                {
                    'name': 'Test Materials Operation',
                    'type': 'materials',
                    'action': 'metamaterial',
                    'material_id': 'META001',
                    'frequencies': [1e9, 10e9],
                    'thickness': 0.001
                }
            ]
        }
        
        script_path = self.temp_path / 'test_batch.json'
        with open(script_path, 'w') as f:
            json.dump(batch_script, f)
        
        mock_config_manager = Mock()
        processor = BatchProcessor(mock_config_manager)
        
        # Test processing
        results = processor.process_batch_file(str(script_path))
        
        assert results['script_path'] == str(script_path)
        assert results['summary']['total'] == 1
        assert len(results['operations']) == 1
    
    def test_batch_operation_processing(self):
        """Test individual batch operation processing."""
        mock_config_manager = Mock()
        processor = BatchProcessor(mock_config_manager)
        
        # Test design operation
        design_op = {
            'name': 'Test Design',
            'type': 'design',
            'action': 'create',
            'name': 'Test Aircraft',
            'platform': 'su75'
        }
        
        result = processor._process_design_operation(design_op)
        assert result['operation'] == 'Test Design'
        assert result['action'] == 'create'
        assert 'Test Aircraft' in result['result']
        
        # Test materials operation
        materials_op = {
            'name': 'Test Materials',
            'type': 'materials',
            'action': 'metamaterial',
            'material_id': 'META001',
            'frequencies': [1e9, 10e9]
        }
        
        result = processor._process_materials_operation(materials_op)
        assert result['operation'] == 'Test Materials'
        assert result['action'] == 'metamaterial'
        assert 'META001' in result['result']
    
    def test_unsupported_batch_script_format(self):
        """Test handling of unsupported batch script formats."""
        script_path = self.temp_path / 'test_batch.txt'
        script_path.write_text('invalid format')
        
        mock_config_manager = Mock()
        processor = BatchProcessor(mock_config_manager)
        
        with pytest.raises(ValueError, match="Unsupported batch script format"):
            processor.process_batch_file(str(script_path))
    
    def test_missing_batch_script(self):
        """Test handling of missing batch script files."""
        mock_config_manager = Mock()
        processor = BatchProcessor(mock_config_manager)
        
        with pytest.raises(FileNotFoundError):
            processor.process_batch_file('nonexistent_file.yaml')


class TestHelpSystem:
    """Test the help system."""
    
    def test_help_system_creation(self):
        """Test help system initialization."""
        help_system = HelpSystem()
        assert help_system is not None
        assert len(help_system.commands) > 0
        assert len(help_system.examples) > 0
        assert len(help_system.workflows) > 0
    
    def test_command_help(self):
        """Test getting help for commands."""
        help_text = get_help('config')
        assert 'CONFIG' in help_text
        assert 'configuration management' in help_text.lower()
        
        # Test subcommand help
        help_text = get_help('config', 'init')
        assert 'init' in help_text.lower()
        assert 'initialize' in help_text.lower()
    
    def test_general_help(self):
        """Test general help."""
        help_text = get_help()
        assert 'Fighter Jet SDK Help' in help_text
        assert 'Available commands:' in help_text
        assert 'config' in help_text
        assert 'design' in help_text
    
    def test_examples(self):
        """Test examples system."""
        # Test all examples
        examples_text = get_examples()
        assert 'Fighter Jet SDK Examples' in examples_text
        assert 'basic' in examples_text.lower()
        assert 'advanced' in examples_text.lower()
        
        # Test specific category
        basic_examples = get_examples('basic')
        assert 'config init' in basic_examples
        assert 'propulsion list' in basic_examples
    
    def test_unknown_command_help(self):
        """Test help for unknown commands."""
        help_text = get_help('unknown_command')
        assert 'Unknown command' in help_text
    
    def test_unknown_examples_category(self):
        """Test examples for unknown category."""
        examples_text = get_examples('unknown_category')
        assert 'No examples found' in examples_text


class TestOutputFormatting:
    """Test output formatting functions."""
    
    def test_json_formatting(self):
        """Test JSON output formatting."""
        data = {'key': 'value', 'number': 42}
        formatted = format_output(data, 'json')
        
        # Should be valid JSON
        parsed = json.loads(formatted)
        assert parsed == data
    
    def test_yaml_formatting(self):
        """Test YAML output formatting."""
        data = {'key': 'value', 'number': 42}
        formatted = format_output(data, 'yaml')
        
        # Should be valid YAML
        parsed = yaml.safe_load(formatted)
        assert parsed == data
    
    def test_table_formatting(self):
        """Test table output formatting."""
        # Test dictionary formatting
        data = {'key1': 'value1', 'key2': 'value2'}
        formatted = format_output(data, 'table')
        assert 'key1: value1' in formatted
        assert 'key2: value2' in formatted
        
        # Test list formatting
        data = ['item1', 'item2', 'item3']
        formatted = format_output(data, 'table')
        assert 'item1' in formatted
        assert 'item2' in formatted
        assert 'item3' in formatted
    
    def test_unknown_format(self):
        """Test unknown format handling."""
        data = {'key': 'value'}
        formatted = format_output(data, 'unknown')
        assert str(data) == formatted


class TestMainFunction:
    """Test the main CLI entry point."""
    
    @patch('fighter_jet_sdk.cli.main.get_config_manager')
    @patch('fighter_jet_sdk.cli.main.get_log_manager')
    def test_main_help_command(self, mock_log_manager, mock_config_manager):
        """Test main function with help command."""
        result = main(['help'])
        assert result == 0
    
    @patch('fighter_jet_sdk.cli.main.get_config_manager')
    @patch('fighter_jet_sdk.cli.main.get_log_manager')
    def test_main_examples_command(self, mock_log_manager, mock_config_manager):
        """Test main function with examples command."""
        result = main(['examples'])
        assert result == 0
    
    @patch('fighter_jet_sdk.cli.main.get_config_manager')
    @patch('fighter_jet_sdk.cli.main.get_log_manager')
    def test_main_config_command(self, mock_log_manager, mock_config_manager):
        """Test main function with config command."""
        mock_config_manager.return_value.create_default_config.return_value = None
        
        result = main(['config', 'init'])
        assert result == 0
    
    def test_main_no_command(self):
        """Test main function with no command."""
        with patch('builtins.print'):
            result = main([])
            assert result == 1
    
    def test_main_keyboard_interrupt(self):
        """Test main function keyboard interrupt handling."""
        with patch('fighter_jet_sdk.cli.main.get_config_manager', side_effect=KeyboardInterrupt):
            with patch('builtins.print'):
                result = main(['config', 'init'])
                assert result == 130


if __name__ == '__main__':
    pytest.main([__file__])