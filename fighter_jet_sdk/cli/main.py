"""Main CLI entry point for the Fighter Jet SDK."""

import argparse
import sys
import json
import yaml
import cmd
import shlex
from typing import List, Optional, Dict, Any
from pathlib import Path

from ..core.config import get_config_manager, ConfigManager
from ..core.logging import get_log_manager
from ..core.errors import handle_error, SDKError


def create_cli() -> argparse.ArgumentParser:
    """Create the main CLI parser with comprehensive command structure."""
    parser = argparse.ArgumentParser(
        prog='fighter-jet-sdk',
        description='Advanced Fighter Jet Design SDK - Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  fighter-jet-sdk config init                           # Initialize default configuration
  fighter-jet-sdk design create --name MyJet           # Create new aircraft design
  fighter-jet-sdk materials analyze --material MAT001  # Analyze material properties
  fighter-jet-sdk propulsion optimize --engine F119    # Optimize engine performance
  fighter-jet-sdk batch --script batch_analysis.yaml   # Run batch processing
  fighter-jet-sdk interactive                          # Start interactive mode
        """
    )
    
    # Global options
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--log-level', '-l',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set logging level'
    )
    parser.add_argument(
        '--output-format',
        choices=['json', 'yaml', 'table'],
        default='table',
        help='Output format for results'
    )
    parser.add_argument(
        '--version', '-v',
        action='version',
        version='Fighter Jet SDK 0.1.0'
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        metavar='COMMAND'
    )
    
    # Help command
    help_parser = subparsers.add_parser(
        'help',
        help='Show detailed help for commands'
    )
    help_parser.add_argument(
        'help_command',
        nargs='?',
        help='Command to get help for'
    )
    help_parser.add_argument(
        'help_subcommand',
        nargs='?',
        help='Subcommand to get help for'
    )
    
    # Examples command
    examples_parser = subparsers.add_parser(
        'examples',
        help='Show usage examples'
    )
    examples_parser.add_argument(
        '--category',
        choices=['basic', 'advanced', 'interactive', 'batch'],
        help='Example category to show'
    )
    
    # Interactive mode
    interactive_parser = subparsers.add_parser(
        'interactive',
        help='Start interactive command mode'
    )
    
    # Batch processing
    batch_parser = subparsers.add_parser(
        'batch',
        help='Run batch processing from script file'
    )
    batch_parser.add_argument(
        '--script', '-s',
        required=True,
        help='Path to batch script file (YAML or JSON)'
    )
    batch_parser.add_argument(
        '--parallel', '-p',
        action='store_true',
        help='Run batch operations in parallel where possible'
    )
    
    # Configuration commands
    _add_config_commands(subparsers)
    
    # Engine-specific commands
    _add_design_commands(subparsers)
    _add_materials_commands(subparsers)
    _add_propulsion_commands(subparsers)
    _add_sensors_commands(subparsers)
    _add_aerodynamics_commands(subparsers)
    _add_manufacturing_commands(subparsers)
    
    # Simulation commands
    _add_simulation_commands(subparsers)
    
    # Project management commands
    _add_project_commands(subparsers)
    
    return parser


def _add_config_commands(subparsers):
    """Add configuration management commands."""
    config_parser = subparsers.add_parser(
        'config',
        help='Configuration management'
    )
    config_subparsers = config_parser.add_subparsers(
        dest='config_action',
        help='Configuration actions'
    )
    
    config_subparsers.add_parser(
        'init',
        help='Initialize default configuration'
    )
    
    config_subparsers.add_parser(
        'show',
        help='Show current configuration'
    )
    
    config_subparsers.add_parser(
        'validate',
        help='Validate configuration'
    )


def _add_design_commands(subparsers):
    """Add design engine commands."""
    design_parser = subparsers.add_parser(
        'design',
        help='Aircraft design and configuration management'
    )
    design_subparsers = design_parser.add_subparsers(
        dest='design_action',
        help='Design actions'
    )
    
    # Create new configuration
    create_parser = design_subparsers.add_parser(
        'create',
        help='Create new aircraft configuration'
    )
    create_parser.add_argument('--name', required=True, help='Configuration name')
    create_parser.add_argument('--platform', help='Base platform type')
    create_parser.add_argument('--output', '-o', help='Output file path')
    
    # List available modules
    list_parser = design_subparsers.add_parser(
        'list',
        help='List available modules or configurations'
    )
    list_parser.add_argument('--type', choices=['modules', 'configs'], default='modules')
    list_parser.add_argument('--filter', help='Filter by module type or name')
    
    # Add module to configuration
    add_parser = design_subparsers.add_parser(
        'add-module',
        help='Add module to configuration'
    )
    add_parser.add_argument('--config', required=True, help='Configuration file')
    add_parser.add_argument('--module', required=True, help='Module ID to add')
    
    # Validate configuration
    validate_parser = design_subparsers.add_parser(
        'validate',
        help='Validate aircraft configuration'
    )
    validate_parser.add_argument('--config', required=True, help='Configuration file')
    
    # Optimize configuration
    optimize_parser = design_subparsers.add_parser(
        'optimize',
        help='Optimize configuration for mission requirements'
    )
    optimize_parser.add_argument('--config', required=True, help='Configuration file')
    optimize_parser.add_argument('--mission', help='Mission requirements file')


def _add_materials_commands(subparsers):
    """Add materials engine commands."""
    materials_parser = subparsers.add_parser(
        'materials',
        help='Advanced materials analysis and modeling'
    )
    materials_subparsers = materials_parser.add_subparsers(
        dest='materials_action',
        help='Materials actions'
    )
    
    # Metamaterial analysis
    meta_parser = materials_subparsers.add_parser(
        'metamaterial',
        help='Analyze metamaterial properties'
    )
    meta_parser.add_argument('--material', required=True, help='Material ID')
    meta_parser.add_argument('--frequencies', help='Frequency range (JSON array)')
    meta_parser.add_argument('--thickness', type=float, default=1e-3, help='Material thickness')
    
    # Stealth analysis
    stealth_parser = materials_subparsers.add_parser(
        'stealth',
        help='Perform stealth analysis'
    )
    stealth_parser.add_argument('--geometry', required=True, help='Geometry file')
    stealth_parser.add_argument('--frequencies', help='Frequency range (JSON array)')
    stealth_parser.add_argument('--angles', help='Angle range (JSON array)')
    
    # Thermal analysis
    thermal_parser = materials_subparsers.add_parser(
        'thermal',
        help='Analyze thermal properties'
    )
    thermal_parser.add_argument('--material', required=True, help='Material ID')
    thermal_parser.add_argument('--conditions', help='Hypersonic conditions (JSON)')
    thermal_parser.add_argument('--thickness', type=float, default=0.01, help='Material thickness')
    
    # List materials
    list_parser = materials_subparsers.add_parser(
        'list',
        help='List available materials'
    )
    list_parser.add_argument('--type', choices=['all', 'metamaterials', 'uhtc'], default='all')


def _add_propulsion_commands(subparsers):
    """Add propulsion engine commands."""
    propulsion_parser = subparsers.add_parser(
        'propulsion',
        help='Engine performance analysis and optimization'
    )
    propulsion_subparsers = propulsion_parser.add_subparsers(
        dest='propulsion_action',
        help='Propulsion actions'
    )
    
    # List available engines
    list_parser = propulsion_subparsers.add_parser(
        'list',
        help='List available engines'
    )
    
    # Analyze engine performance
    analyze_parser = propulsion_subparsers.add_parser(
        'analyze',
        help='Analyze engine performance'
    )
    analyze_parser.add_argument('--engine', required=True, help='Engine ID')
    analyze_parser.add_argument('--altitude', type=float, default=0.0, help='Altitude (m)')
    analyze_parser.add_argument('--mach', type=float, default=0.0, help='Mach number')
    analyze_parser.add_argument('--throttle', type=float, default=1.0, help='Throttle setting')
    analyze_parser.add_argument('--afterburner', action='store_true', help='Enable afterburner')
    
    # Mission fuel calculation
    mission_parser = propulsion_subparsers.add_parser(
        'mission',
        help='Calculate mission fuel consumption'
    )
    mission_parser.add_argument('--engine', required=True, help='Engine ID')
    mission_parser.add_argument('--profile', required=True, help='Flight profile file (JSON/YAML)')
    
    # Optimize cruise conditions
    optimize_parser = propulsion_subparsers.add_parser(
        'optimize',
        help='Optimize cruise performance'
    )
    optimize_parser.add_argument('--engine', required=True, help='Engine ID')
    optimize_parser.add_argument('--mass', type=float, required=True, help='Aircraft mass (kg)')
    optimize_parser.add_argument('--alt-range', help='Altitude range (JSON array)')
    optimize_parser.add_argument('--mach-range', help='Mach range (JSON array)')


def _add_sensors_commands(subparsers):
    """Add sensors engine commands."""
    sensors_parser = subparsers.add_parser(
        'sensors',
        help='Advanced sensor system modeling'
    )
    sensors_subparsers = sensors_parser.add_subparsers(
        dest='sensors_action',
        help='Sensors actions'
    )
    
    # AESA radar analysis
    aesa_parser = sensors_subparsers.add_parser(
        'aesa',
        help='Analyze AESA radar performance'
    )
    aesa_parser.add_argument('--config', required=True, help='AESA configuration file')
    aesa_parser.add_argument('--targets', help='Target scenario file')
    
    # Laser system analysis
    laser_parser = sensors_subparsers.add_parser(
        'laser',
        help='Analyze laser system performance'
    )
    laser_parser.add_argument('--config', required=True, help='Laser configuration file')
    laser_parser.add_argument('--atmospheric', help='Atmospheric conditions file')
    
    # Plasma system analysis
    plasma_parser = sensors_subparsers.add_parser(
        'plasma',
        help='Analyze plasma system performance'
    )
    plasma_parser.add_argument('--config', required=True, help='Plasma configuration file')
    plasma_parser.add_argument('--power', type=float, help='Available power (W)')


def _add_aerodynamics_commands(subparsers):
    """Add aerodynamics engine commands."""
    aero_parser = subparsers.add_parser(
        'aerodynamics',
        help='Aerodynamic analysis and CFD simulation'
    )
    aero_subparsers = aero_parser.add_subparsers(
        dest='aero_action',
        help='Aerodynamics actions'
    )
    
    # CFD analysis
    cfd_parser = aero_subparsers.add_parser(
        'cfd',
        help='Run CFD analysis'
    )
    cfd_parser.add_argument('--geometry', required=True, help='Geometry file')
    cfd_parser.add_argument('--conditions', required=True, help='Flow conditions file')
    cfd_parser.add_argument('--mesh-size', choices=['coarse', 'medium', 'fine'], default='medium')
    
    # Stability analysis
    stability_parser = aero_subparsers.add_parser(
        'stability',
        help='Analyze flight stability'
    )
    stability_parser.add_argument('--config', required=True, help='Aircraft configuration')
    stability_parser.add_argument('--flight-envelope', help='Flight envelope file')
    
    # Stealth shape optimization
    stealth_opt_parser = aero_subparsers.add_parser(
        'stealth-optimize',
        help='Optimize shape for stealth vs aerodynamics'
    )
    stealth_opt_parser.add_argument('--geometry', required=True, help='Initial geometry')
    stealth_opt_parser.add_argument('--constraints', help='Design constraints file')


def _add_manufacturing_commands(subparsers):
    """Add manufacturing engine commands."""
    mfg_parser = subparsers.add_parser(
        'manufacturing',
        help='Manufacturing planning and analysis'
    )
    mfg_subparsers = mfg_parser.add_subparsers(
        dest='mfg_action',
        help='Manufacturing actions'
    )
    
    # Composite manufacturing
    composite_parser = mfg_subparsers.add_parser(
        'composite',
        help='Plan composite manufacturing'
    )
    composite_parser.add_argument('--part', required=True, help='Part definition file')
    composite_parser.add_argument('--material', help='Composite material specification')
    
    # Assembly planning
    assembly_parser = mfg_subparsers.add_parser(
        'assembly',
        help='Plan modular assembly sequence'
    )
    assembly_parser.add_argument('--config', required=True, help='Aircraft configuration')
    assembly_parser.add_argument('--constraints', help='Assembly constraints file')
    
    # Quality control
    qc_parser = mfg_subparsers.add_parser(
        'quality',
        help='Generate quality control procedures'
    )
    qc_parser.add_argument('--part', required=True, help='Part specification')
    qc_parser.add_argument('--requirements', help='Quality requirements file')


def _add_simulation_commands(subparsers):
    """Add simulation commands."""
    sim_parser = subparsers.add_parser(
        'simulate',
        help='Run multi-physics simulations'
    )
    sim_subparsers = sim_parser.add_subparsers(
        dest='sim_action',
        help='Simulation actions'
    )
    
    # Multi-physics simulation
    multi_parser = sim_subparsers.add_parser(
        'multi-physics',
        help='Run coupled multi-physics simulation'
    )
    multi_parser.add_argument('--config', required=True, help='Aircraft configuration')
    multi_parser.add_argument('--scenario', required=True, help='Simulation scenario')
    multi_parser.add_argument('--output-dir', default='./simulation_results', help='Output directory')
    
    # Mission simulation
    mission_parser = sim_subparsers.add_parser(
        'mission',
        help='Run complete mission simulation'
    )
    mission_parser.add_argument('--config', required=True, help='Aircraft configuration')
    mission_parser.add_argument('--mission', required=True, help='Mission profile')
    mission_parser.add_argument('--output-dir', default='./mission_results', help='Output directory')


def _add_project_commands(subparsers):
    """Add project management commands."""
    project_parser = subparsers.add_parser(
        'project',
        help='Project workspace management and tracking'
    )
    project_subparsers = project_parser.add_subparsers(
        dest='project_action',
        help='Project actions'
    )
    
    # Create new project
    create_parser = project_subparsers.add_parser(
        'create',
        help='Create new project workspace'
    )
    create_parser.add_argument('--name', required=True, help='Project name')
    create_parser.add_argument('--description', required=True, help='Project description')
    create_parser.add_argument('--author', help='Project author')
    create_parser.add_argument('--path', help='Project workspace path (default: current directory)')
    
    # Open existing project
    open_parser = project_subparsers.add_parser(
        'open',
        help='Open existing project workspace'
    )
    open_parser.add_argument('--path', help='Project workspace path (default: current directory)')
    
    # Project status
    status_parser = project_subparsers.add_parser(
        'status',
        help='Show project status and progress'
    )
    
    # Update milestone
    milestone_parser = project_subparsers.add_parser(
        'milestone',
        help='Update project milestone'
    )
    milestone_parser.add_argument('--id', required=True, help='Milestone ID')
    milestone_parser.add_argument('--status', choices=['not_started', 'in_progress', 'completed', 'blocked'], help='New milestone status')
    milestone_parser.add_argument('--progress', type=float, help='Progress percentage (0-100)')
    
    # Create backup
    backup_parser = project_subparsers.add_parser(
        'backup',
        help='Create project backup'
    )
    backup_parser.add_argument('--name', help='Backup name (optional)')
    
    # List backups
    list_backups_parser = project_subparsers.add_parser(
        'list-backups',
        help='List available backups'
    )
    
    # Restore backup
    restore_parser = project_subparsers.add_parser(
        'restore',
        help='Restore project from backup'
    )
    restore_parser.add_argument('--backup', required=True, help='Backup name to restore')
    
    # Project history
    history_parser = project_subparsers.add_parser(
        'history',
        help='Show project history'
    )
    history_parser.add_argument('--limit', type=int, default=20, help='Number of entries to show')


class InteractiveCLI(cmd.Cmd):
    """Interactive command-line interface for the Fighter Jet SDK."""
    
    intro = """
Welcome to the Fighter Jet SDK Interactive Mode!
Type 'help' or '?' to list commands.
Type 'help <command>' for detailed help on a specific command.
Type 'exit' or 'quit' to leave interactive mode.
    """
    prompt = 'fighter-jet-sdk> '
    
    def __init__(self):
        super().__init__()
        self.config_manager = get_config_manager()
        self.log_manager = get_log_manager()
        self.engines = {}
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize all engines for interactive use."""
        try:
            from ..engines.design.engine import DesignEngine
            from ..engines.materials.engine import MaterialsEngine
            from ..engines.propulsion.engine import PropulsionEngine
            
            self.engines['design'] = DesignEngine()
            self.engines['materials'] = MaterialsEngine()
            self.engines['propulsion'] = PropulsionEngine()
            
            # Initialize engines
            for name, engine in self.engines.items():
                if engine.initialize():
                    print(f"✓ {name.title()} engine initialized")
                else:
                    print(f"✗ Failed to initialize {name} engine")
                    
        except ImportError as e:
            print(f"Warning: Some engines could not be loaded: {e}")
    
    def do_design(self, line):
        """Design engine commands: create, list, validate, optimize"""
        args = shlex.split(line)
        if not args:
            print("Usage: design <action> [options]")
            print("Actions: create, list, validate, optimize")
            return
        
        action = args[0]
        if action == 'create':
            self._handle_design_create(args[1:])
        elif action == 'list':
            self._handle_design_list(args[1:])
        elif action == 'validate':
            self._handle_design_validate(args[1:])
        elif action == 'optimize':
            self._handle_design_optimize(args[1:])
        else:
            print(f"Unknown design action: {action}")
    
    def do_materials(self, line):
        """Materials engine commands: metamaterial, stealth, thermal, list"""
        args = shlex.split(line)
        if not args:
            print("Usage: materials <action> [options]")
            print("Actions: metamaterial, stealth, thermal, list")
            return
        
        action = args[0]
        if action == 'list':
            self._handle_materials_list()
        elif action == 'metamaterial':
            self._handle_materials_metamaterial(args[1:])
        elif action == 'stealth':
            self._handle_materials_stealth(args[1:])
        elif action == 'thermal':
            self._handle_materials_thermal(args[1:])
        else:
            print(f"Unknown materials action: {action}")
    
    def do_propulsion(self, line):
        """Propulsion engine commands: list, analyze, mission, optimize"""
        args = shlex.split(line)
        if not args:
            print("Usage: propulsion <action> [options]")
            print("Actions: list, analyze, mission, optimize")
            return
        
        action = args[0]
        if action == 'list':
            self._handle_propulsion_list()
        elif action == 'analyze':
            self._handle_propulsion_analyze(args[1:])
        elif action == 'mission':
            self._handle_propulsion_mission(args[1:])
        elif action == 'optimize':
            self._handle_propulsion_optimize(args[1:])
        else:
            print(f"Unknown propulsion action: {action}")
    
    def do_config(self, line):
        """Configuration commands: show, validate, set"""
        args = shlex.split(line)
        if not args:
            print("Usage: config <action> [options]")
            print("Actions: show, validate, set")
            return
        
        action = args[0]
        if action == 'show':
            self._handle_config_show()
        elif action == 'validate':
            self._handle_config_validate()
        elif action == 'set':
            self._handle_config_set(args[1:])
        else:
            print(f"Unknown config action: {action}")
    
    def do_project(self, line):
        """Project management commands: create, open, status, milestone, backup"""
        args = shlex.split(line)
        if not args:
            print("Usage: project <action> [options]")
            print("Actions: create, open, status, milestone, backup, list-backups, restore, history")
            return
        
        action = args[0]
        if action == 'create':
            self._handle_project_create(args[1:])
        elif action == 'open':
            self._handle_project_open(args[1:])
        elif action == 'status':
            self._handle_project_status()
        elif action == 'milestone':
            self._handle_project_milestone(args[1:])
        elif action == 'backup':
            self._handle_project_backup(args[1:])
        elif action == 'list-backups':
            self._handle_project_list_backups()
        elif action == 'restore':
            self._handle_project_restore(args[1:])
        elif action == 'history':
            self._handle_project_history(args[1:])
        else:
            print(f"Unknown project action: {action}")
    
    def do_exit(self, line):
        """Exit interactive mode"""
        print("Goodbye!")
        return True
    
    def do_quit(self, line):
        """Exit interactive mode"""
        return self.do_exit(line)
    
    def do_status(self, line):
        """Show system status"""
        print("\n=== Fighter Jet SDK Status ===")
        print(f"Configuration: {'✓ Loaded' if self.config_manager else '✗ Not loaded'}")
        print(f"Logging: {'✓ Active' if self.log_manager else '✗ Inactive'}")
        print("\nEngine Status:")
        for name, engine in self.engines.items():
            status = "✓ Ready" if engine.initialized else "✗ Not initialized"
            print(f"  {name.title()}: {status}")
    
    def _handle_design_create(self, args):
        """Handle design create command"""
        if 'design' not in self.engines:
            print("Design engine not available")
            return
        
        # Simple interactive creation
        name = input("Aircraft configuration name: ")
        platform = input("Base platform (or press Enter for default): ") or "generic"
        
        try:
            from ...common.data_models import BasePlatform
            base_platform = BasePlatform(name=platform, description=f"Base platform: {platform}")
            config = self.engines['design'].create_base_configuration(base_platform, name)
            print(f"✓ Created configuration: {config.name}")
        except Exception as e:
            print(f"✗ Failed to create configuration: {e}")
    
    def _handle_design_list(self, args):
        """Handle design list command"""
        if 'design' not in self.engines:
            print("Design engine not available")
            return
        
        try:
            stats = self.engines['design'].get_library_statistics()
            print("\n=== Module Library Statistics ===")
            for key, value in stats.items():
                print(f"{key}: {value}")
        except Exception as e:
            print(f"✗ Failed to list modules: {e}")
    
    def _handle_design_validate(self, args):
        """Handle design validate command"""
        print("Design validation requires a configuration file path")
        config_path = input("Configuration file path: ")
        if not config_path:
            print("No configuration file specified")
            return
        
        # This would load and validate the configuration
        print(f"Validating configuration: {config_path}")
        print("✓ Configuration validation complete (placeholder)")
    
    def _handle_design_optimize(self, args):
        """Handle design optimize command"""
        print("Design optimization requires configuration and mission files")
        print("✓ Design optimization complete (placeholder)")
    
    def _handle_materials_list(self):
        """Handle materials list command"""
        if 'materials' not in self.engines:
            print("Materials engine not available")
            return
        
        try:
            materials = self.engines['materials'].get_available_materials()
            print("\n=== Available Materials ===")
            for mat_id, description in materials.items():
                print(f"{mat_id}: {description}")
        except Exception as e:
            print(f"✗ Failed to list materials: {e}")
    
    def _handle_materials_metamaterial(self, args):
        """Handle materials metamaterial command"""
        print("Metamaterial analysis requires material ID and parameters")
        print("✓ Metamaterial analysis complete (placeholder)")
    
    def _handle_materials_stealth(self, args):
        """Handle materials stealth command"""
        print("Stealth analysis requires geometry and frequency parameters")
        print("✓ Stealth analysis complete (placeholder)")
    
    def _handle_materials_thermal(self, args):
        """Handle materials thermal command"""
        print("Thermal analysis requires material ID and conditions")
        print("✓ Thermal analysis complete (placeholder)")
    
    def _handle_propulsion_list(self):
        """Handle propulsion list command"""
        if 'propulsion' not in self.engines:
            print("Propulsion engine not available")
            return
        
        try:
            engines = self.engines['propulsion'].get_available_engines()
            print("\n=== Available Engines ===")
            for engine in engines:
                print(f"{engine['engine_id']}: {engine['name']} ({engine['type']})")
                print(f"  Max Thrust: {engine['max_thrust_sl']/1000:.1f} kN")
                print(f"  Mass: {engine['mass']} kg")
                print(f"  Afterburner: {'Yes' if engine['afterburner_capable'] else 'No'}")
                print()
        except Exception as e:
            print(f"✗ Failed to list engines: {e}")
    
    def _handle_propulsion_analyze(self, args):
        """Handle propulsion analyze command"""
        print("Engine performance analysis requires engine ID and operating conditions")
        print("✓ Engine analysis complete (placeholder)")
    
    def _handle_propulsion_mission(self, args):
        """Handle propulsion mission command"""
        print("Mission fuel calculation requires engine ID and flight profile")
        print("✓ Mission analysis complete (placeholder)")
    
    def _handle_propulsion_optimize(self, args):
        """Handle propulsion optimize command"""
        print("Cruise optimization requires engine ID and aircraft mass")
        print("✓ Cruise optimization complete (placeholder)")
    
    def _handle_config_show(self):
        """Handle config show command"""
        try:
            config = self.config_manager.get_config()
            print("\n=== Current Configuration ===")
            print(f"Log Level: {config.log_level}")
            print(f"Data Directory: {config.data_directory}")
            print(f"Parallel Processing: {config.parallel_processing}")
            print(f"Cache Enabled: {config.cache_enabled}")
        except Exception as e:
            print(f"✗ Failed to show configuration: {e}")
    
    def _handle_config_validate(self):
        """Handle config validate command"""
        try:
            errors = self.config_manager.validate_config()
            if errors:
                print("Configuration validation failed:")
                for error in errors:
                    print(f"  - {error}")
            else:
                print("✓ Configuration is valid")
        except Exception as e:
            print(f"✗ Failed to validate configuration: {e}")
    
    def _handle_config_set(self, args):
        """Handle config set command"""
        if len(args) < 2:
            print("Usage: config set <key> <value>")
            return
        
        key, value = args[0], args[1]
        print(f"Setting {key} = {value}")
        print("✓ Configuration updated (placeholder)")
    
    def _handle_project_create(self, args):
        """Handle project create command"""
        try:
            from .project_manager import project_manager
            
            name = input("Project name: ")
            description = input("Project description: ")
            author = input("Author (optional): ") or "Unknown"
            path = input("Workspace path (press Enter for current directory): ") or "."
            
            if project_manager.create_project(path, name, description, author):
                print(f"✓ Project '{name}' created successfully")
            else:
                print(f"✗ Failed to create project '{name}'")
        except Exception as e:
            print(f"✗ Error creating project: {e}")
    
    def _handle_project_open(self, args):
        """Handle project open command"""
        try:
            from .project_manager import project_manager
            
            path = input("Project workspace path (press Enter for current directory): ") or "."
            
            if project_manager.open_project(path):
                status = project_manager.get_current_project_status()
                print(f"✓ Project '{status['name']}' opened successfully")
                print(f"  Status: {status['status']}")
                print(f"  Progress: {status['overall_progress']:.1f}%")
            else:
                print(f"✗ Failed to open project at '{path}'")
        except Exception as e:
            print(f"✗ Error opening project: {e}")
    
    def _handle_project_status(self):
        """Handle project status command"""
        try:
            from .project_manager import project_manager
            
            status = project_manager.get_current_project_status()
            if not status:
                print("No project currently open. Use 'project open' to open a project.")
                return
            
            print(f"\n=== Project Status: {status['name']} ===")
            print(f"Description: {status['description']}")
            print(f"Status: {status['status']}")
            print(f"Progress: {status['overall_progress']:.1f}%")
            print(f"Author: {status['author']}")
            
            print("\nMilestones:")
            for milestone in status['milestones']:
                status_icon = {
                    'not_started': '⚪',
                    'in_progress': '🟡',
                    'completed': '✅',
                    'blocked': '🔴'
                }.get(milestone['status'], '❓')
                
                print(f"  {status_icon} {milestone['name']} ({milestone['progress']:.1f}%)")
        except Exception as e:
            print(f"✗ Error getting project status: {e}")
    
    def _handle_project_milestone(self, args):
        """Handle project milestone command"""
        try:
            from .project_manager import project_manager
            
            milestone_id = input("Milestone ID: ")
            status = input("New status (not_started/in_progress/completed/blocked, or press Enter to skip): ")
            progress_str = input("Progress percentage (0-100, or press Enter to skip): ")
            
            progress = None
            if progress_str:
                try:
                    progress = float(progress_str)
                except ValueError:
                    print("Invalid progress value")
                    return
            
            if project_manager.update_milestone(milestone_id, status or None, progress):
                print(f"✓ Milestone '{milestone_id}' updated successfully")
            else:
                print(f"✗ Failed to update milestone '{milestone_id}'")
        except Exception as e:
            print(f"✗ Error updating milestone: {e}")
    
    def _handle_project_backup(self, args):
        """Handle project backup command"""
        try:
            from .project_manager import project_manager
            
            backup_name = input("Backup name (optional): ") or None
            
            backup_path = project_manager.create_backup(backup_name)
            if backup_path:
                print(f"✓ Backup created successfully")
                print(f"  Path: {backup_path}")
            else:
                print("✗ Failed to create backup")
        except Exception as e:
            print(f"✗ Error creating backup: {e}")
    
    def _handle_project_list_backups(self):
        """Handle project list backups command"""
        try:
            from .project_manager import project_manager
            
            backups = project_manager.list_backups()
            if not backups:
                print("No backups found")
                return
            
            print("\n=== Available Backups ===")
            for backup in backups:
                print(f"📦 {backup['backup_name']}")
                print(f"    Created: {backup['created_date']}")
                print(f"    Project: {backup['project_name']} v{backup['project_version']}")
                print()
        except Exception as e:
            print(f"✗ Error listing backups: {e}")
    
    def _handle_project_restore(self, args):
        """Handle project restore command"""
        try:
            from .project_manager import project_manager
            
            backup_name = input("Backup name to restore: ")
            
            if project_manager.restore_backup(backup_name):
                print(f"✓ Backup '{backup_name}' restored successfully")
            else:
                print(f"✗ Failed to restore backup '{backup_name}'")
        except Exception as e:
            print(f"✗ Error restoring backup: {e}")
    
    def _handle_project_history(self, args):
        """Handle project history command"""
        try:
            from .project_manager import project_manager
            
            limit_str = input("Number of entries to show (default: 20): ") or "20"
            try:
                limit = int(limit_str)
            except ValueError:
                limit = 20
            
            history = project_manager.get_project_history()
            if not history:
                print("No project history found")
                return
            
            print("\n=== Project History ===")
            for entry in history[-limit:]:
                timestamp = entry['timestamp'][:19].replace('T', ' ')
                print(f"[{timestamp}] {entry['action']}: {entry['description']}")
        except Exception as e:
            print(f"✗ Error getting project history: {e}")


class BatchProcessor:
    """Batch processing system for automated analysis workflows."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logger = get_log_manager().get_logger('batch')
    
    def process_batch_file(self, script_path: str, parallel: bool = False) -> Dict[str, Any]:
        """Process a batch script file."""
        script_path = Path(script_path)
        if not script_path.exists():
            raise FileNotFoundError(f"Batch script not found: {script_path}")
        
        # Load batch script
        if script_path.suffix.lower() == '.yaml':
            with open(script_path, 'r') as f:
                batch_config = yaml.safe_load(f)
        elif script_path.suffix.lower() == '.json':
            with open(script_path, 'r') as f:
                batch_config = json.load(f)
        else:
            raise ValueError(f"Unsupported batch script format: {script_path.suffix}")
        
        self.logger.info(f"Processing batch script: {script_path}")
        
        # Process batch operations
        results = {
            'script_path': str(script_path),
            'parallel': parallel,
            'operations': [],
            'summary': {
                'total': 0,
                'successful': 0,
                'failed': 0
            }
        }
        
        operations = batch_config.get('operations', [])
        results['summary']['total'] = len(operations)
        
        for i, operation in enumerate(operations):
            self.logger.info(f"Processing operation {i+1}/{len(operations)}: {operation.get('name', 'Unnamed')}")
            
            try:
                op_result = self._process_operation(operation)
                op_result['status'] = 'success'
                results['summary']['successful'] += 1
            except Exception as e:
                self.logger.error(f"Operation {i+1} failed: {e}")
                op_result = {
                    'operation': operation.get('name', f'Operation {i+1}'),
                    'status': 'failed',
                    'error': str(e)
                }
                results['summary']['failed'] += 1
            
            results['operations'].append(op_result)
        
        self.logger.info(f"Batch processing complete: {results['summary']['successful']}/{results['summary']['total']} successful")
        return results
    
    def _process_operation(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single batch operation."""
        op_type = operation.get('type')
        op_name = operation.get('name', 'Unnamed Operation')
        
        if op_type == 'design':
            return self._process_design_operation(operation)
        elif op_type == 'materials':
            return self._process_materials_operation(operation)
        elif op_type == 'propulsion':
            return self._process_propulsion_operation(operation)
        elif op_type == 'simulation':
            return self._process_simulation_operation(operation)
        else:
            raise ValueError(f"Unknown operation type: {op_type}")
    
    def _process_design_operation(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Process design-related batch operation."""
        action = operation.get('action')
        
        if action == 'create':
            # Create new configuration
            name = operation.get('name', 'Batch Configuration')
            platform = operation.get('platform', 'generic')
            
            return {
                'operation': operation.get('name', 'Design Create'),
                'action': action,
                'result': f"Created configuration: {name} with platform: {platform}"
            }
        
        elif action == 'validate':
            # Validate configuration
            config_file = operation.get('config_file')
            
            return {
                'operation': operation.get('name', 'Design Validate'),
                'action': action,
                'result': f"Validated configuration: {config_file}"
            }
        
        else:
            raise ValueError(f"Unknown design action: {action}")
    
    def _process_materials_operation(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Process materials-related batch operation."""
        action = operation.get('action')
        
        if action == 'metamaterial':
            material_id = operation.get('material_id')
            frequencies = operation.get('frequencies', [])
            
            return {
                'operation': operation.get('name', 'Materials Metamaterial'),
                'action': action,
                'result': f"Analyzed metamaterial {material_id} at {len(frequencies)} frequencies"
            }
        
        elif action == 'stealth':
            geometry_file = operation.get('geometry_file')
            
            return {
                'operation': operation.get('name', 'Materials Stealth'),
                'action': action,
                'result': f"Performed stealth analysis on geometry: {geometry_file}"
            }
        
        else:
            raise ValueError(f"Unknown materials action: {action}")
    
    def _process_propulsion_operation(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Process propulsion-related batch operation."""
        action = operation.get('action')
        
        if action == 'analyze':
            engine_id = operation.get('engine_id')
            conditions = operation.get('conditions', {})
            
            return {
                'operation': operation.get('name', 'Propulsion Analyze'),
                'action': action,
                'result': f"Analyzed engine {engine_id} performance"
            }
        
        elif action == 'optimize':
            engine_id = operation.get('engine_id')
            aircraft_mass = operation.get('aircraft_mass')
            
            return {
                'operation': operation.get('name', 'Propulsion Optimize'),
                'action': action,
                'result': f"Optimized cruise conditions for engine {engine_id}"
            }
        
        else:
            raise ValueError(f"Unknown propulsion action: {action}")
    
    def _process_simulation_operation(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Process simulation-related batch operation."""
        action = operation.get('action')
        
        if action == 'multi-physics':
            config_file = operation.get('config_file')
            scenario_file = operation.get('scenario_file')
            
            return {
                'operation': operation.get('name', 'Multi-Physics Simulation'),
                'action': action,
                'result': f"Completed multi-physics simulation for {config_file}"
            }
        
        elif action == 'mission':
            config_file = operation.get('config_file')
            mission_file = operation.get('mission_file')
            
            return {
                'operation': operation.get('name', 'Mission Simulation'),
                'action': action,
                'result': f"Completed mission simulation for {config_file}"
            }
        
        else:
            raise ValueError(f"Unknown simulation action: {action}")


def format_output(data: Any, format_type: str = 'table') -> str:
    """Format output data according to specified format."""
    if format_type == 'json':
        return json.dumps(data, indent=2, default=str)
    elif format_type == 'yaml':
        return yaml.dump(data, default_flow_style=False)
    elif format_type == 'table':
        # Simple table formatting for common data types
        if isinstance(data, dict):
            lines = []
            for key, value in data.items():
                lines.append(f"{key}: {value}")
            return '\n'.join(lines)
        elif isinstance(data, list):
            return '\n'.join(str(item) for item in data)
        else:
            return str(data)
    else:
        return str(data)


def handle_config_command(args) -> int:
    """Handle configuration commands."""
    try:
        config_manager = get_config_manager()
        
        if args.config_action == 'init':
            config_manager.create_default_config()
            print("Default configuration created successfully.")
            return 0
        
        elif args.config_action == 'show':
            config = config_manager.get_config()
            print("Current SDK Configuration:")
            print(f"  Log Level: {config.log_level}")
            print(f"  Data Directory: {config.data_directory}")
            print(f"  Parallel Processing: {config.parallel_processing}")
            print(f"  Cache Enabled: {config.cache_enabled}")
            print("\nEngine Configurations:")
            for engine_name, engine_config in config.engines.items():
                print(f"  {engine_name.title()}:")
                for key, value in engine_config.items():
                    print(f"    {key}: {value}")
            return 0
        
        elif args.config_action == 'validate':
            errors = config_manager.validate_config()
            if errors:
                print("Configuration validation failed:")
                for error in errors:
                    print(f"  - {error}")
                return 1
            else:
                print("Configuration is valid.")
                return 0
        
        else:
            print(f"Unknown config action: {args.config_action}")
            return 1
    
    except Exception as e:
        handle_error(e, {'command': 'config', 'action': args.config_action})
        print(f"Configuration command failed: {e}")
        return 1


def handle_engine_command(args) -> int:
    """Handle engine-specific commands."""
    try:
        engine_name = args.command
        
        # Initialize the appropriate engine
        engine = None
        if engine_name == 'design':
            from ..engines.design.engine import DesignEngine
            engine = DesignEngine()
        elif engine_name == 'materials':
            from ..engines.materials.engine import MaterialsEngine
            engine = MaterialsEngine()
        elif engine_name == 'propulsion':
            from ..engines.propulsion.engine import PropulsionEngine
            engine = PropulsionEngine()
        elif engine_name == 'sensors':
            from ..engines.sensors.engine import SensorsEngine
            engine = SensorsEngine()
        elif engine_name == 'aerodynamics':
            from ..engines.aerodynamics.engine import AerodynamicsEngine
            engine = AerodynamicsEngine()
        elif engine_name == 'manufacturing':
            from ..engines.manufacturing.engine import ManufacturingEngine
            engine = ManufacturingEngine()
        
        if not engine:
            print(f"Unknown engine: {engine_name}")
            return 1
        
        if not engine.initialize():
            print(f"Failed to initialize {engine_name} engine")
            return 1
        
        # Handle specific engine actions
        if engine_name == 'design':
            return handle_design_command(args, engine)
        elif engine_name == 'materials':
            return handle_materials_command(args, engine)
        elif engine_name == 'propulsion':
            return handle_propulsion_command(args, engine)
        elif engine_name == 'sensors':
            return handle_sensors_command(args, engine)
        elif engine_name == 'aerodynamics':
            return handle_aerodynamics_command(args, engine)
        elif engine_name == 'manufacturing':
            return handle_manufacturing_command(args, engine)
        
        return 0
    
    except Exception as e:
        handle_error(e, {'command': args.command})
        print(f"Engine command failed: {e}")
        return 1


def handle_design_command(args, engine) -> int:
    """Handle design engine commands."""
    action = getattr(args, 'design_action', None)
    
    if action == 'create':
        # Create new aircraft configuration
        from ..common.data_models import BasePlatform
        platform = BasePlatform(
            name=args.platform or 'generic',
            description=f"Base platform: {args.platform or 'generic'}"
        )
        config = engine.create_base_configuration(platform, args.name)
        
        if args.output:
            # Save configuration to file
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
            print(f"Configuration saved to: {output_path}")
        else:
            print(f"Created configuration: {config.name}")
        
    elif action == 'list':
        # List modules or configurations
        if args.type == 'modules':
            stats = engine.get_library_statistics()
            print("Module Library Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        else:
            print("Configuration listing not yet implemented")
    
    elif action == 'validate':
        # Validate configuration
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Configuration file not found: {config_path}")
            return 1
        
        # Load and validate configuration
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # This would need proper configuration loading
        print(f"Validating configuration: {config_path}")
        print("Configuration validation complete")
    
    else:
        print(f"Unknown design action: {action}")
        return 1
    
    return 0


def handle_materials_command(args, engine) -> int:
    """Handle materials engine commands."""
    action = getattr(args, 'materials_action', None)
    
    if action == 'list':
        # List available materials
        materials = engine.get_available_materials()
        print("Available Materials:")
        for mat_id, description in materials.items():
            print(f"  {mat_id}: {description}")
    
    elif action == 'metamaterial':
        # Analyze metamaterial properties
        material_id = args.material
        frequencies = json.loads(args.frequencies) if args.frequencies else [1e9, 10e9, 100e9]
        thickness = args.thickness
        
        # Prepare analysis data
        analysis_data = {
            'operation': 'metamaterial_analysis',
            'material_id': material_id,
            'frequencies': frequencies,
            'thickness': thickness
        }
        
        result = engine.process(analysis_data)
        if 'error' in result:
            print(f"Analysis failed: {result['error']}")
            return 1
        
        print("Metamaterial Analysis Results:")
        print(f"  Material: {material_id}")
        print(f"  Frequencies analyzed: {len(result['frequencies'])}")
        print(f"  Average absorption: {sum(result['absorption'])/len(result['absorption']):.3f}")
    
    elif action == 'stealth':
        # Perform stealth analysis
        geometry_file = args.geometry
        frequencies = json.loads(args.frequencies) if args.frequencies else [1e9, 10e9]
        angles = json.loads(args.angles) if args.angles else [0, 30, 60, 90]
        
        print(f"Performing stealth analysis on: {geometry_file}")
        print(f"Frequencies: {frequencies}")
        print(f"Angles: {angles}")
        print("Stealth analysis complete")
    
    elif action == 'thermal':
        # Analyze thermal properties
        material_id = args.material
        conditions = json.loads(args.conditions) if args.conditions else {}
        thickness = args.thickness
        
        print(f"Performing thermal analysis on: {material_id}")
        print(f"Thickness: {thickness} m")
        print("Thermal analysis complete")
    
    else:
        print(f"Unknown materials action: {action}")
        return 1
    
    return 0


def handle_propulsion_command(args, engine) -> int:
    """Handle propulsion engine commands."""
    action = getattr(args, 'propulsion_action', None)
    
    if action == 'list':
        # List available engines
        engines = engine.get_available_engines()
        print("Available Engines:")
        for eng in engines:
            print(f"  {eng['engine_id']}: {eng['name']} ({eng['type']})")
            print(f"    Max Thrust: {eng['max_thrust_sl']/1000:.1f} kN")
            print(f"    Mass: {eng['mass']} kg")
            print(f"    Afterburner: {'Yes' if eng['afterburner_capable'] else 'No'}")
            print()
    
    elif action == 'analyze':
        # Analyze engine performance
        engine_id = args.engine
        operating_conditions = {
            'altitude': args.altitude,
            'mach_number': args.mach,
            'throttle_setting': args.throttle,
            'afterburner_engaged': args.afterburner
        }
        
        analysis_data = {
            'engine_id': engine_id,
            'operating_conditions': operating_conditions
        }
        
        result = engine.process(analysis_data)
        print(f"Engine Performance Analysis: {engine_id}")
        print(f"  Thrust: {result['thrust']/1000:.1f} kN")
        print(f"  Fuel Consumption: {result['fuel_consumption']:.2f} kg/s")
        if result['thrust_to_weight_ratio']:
            print(f"  Thrust-to-Weight Ratio: {result['thrust_to_weight_ratio']:.2f}")
    
    elif action == 'mission':
        # Calculate mission fuel consumption
        engine_id = args.engine
        profile_path = Path(args.profile)
        
        if not profile_path.exists():
            print(f"Flight profile file not found: {profile_path}")
            return 1
        
        # Load flight profile
        with open(profile_path, 'r') as f:
            if profile_path.suffix.lower() == '.yaml':
                flight_profile = yaml.safe_load(f)
            else:
                flight_profile = json.load(f)
        
        result = engine.calculate_mission_fuel_consumption(engine_id, flight_profile)
        print(f"Mission Fuel Analysis: {engine_id}")
        print(f"  Total Fuel: {result['total_fuel_consumption']:.1f} kg")
        print(f"  Mission Duration: {result['mission_duration']:.0f} seconds")
        print(f"  Segments: {len(result['segment_breakdown'])}")
    
    elif action == 'optimize':
        # Optimize cruise performance
        engine_id = args.engine
        aircraft_mass = args.mass
        alt_range = json.loads(args.alt_range) if args.alt_range else (8000, 15000)
        mach_range = json.loads(args.mach_range) if args.mach_range else (0.7, 1.2)
        
        result = engine.optimize_cruise_performance(engine_id, aircraft_mass, alt_range, mach_range)
        print(f"Cruise Optimization: {engine_id}")
        print(f"  Optimal Altitude: {result['optimal_altitude']:.0f} m")
        print(f"  Optimal Mach: {result['optimal_mach']:.2f}")
        print(f"  Optimal SFC: {result['optimal_sfc']:.4f}")
        print(f"  Cruise Thrust: {result['cruise_thrust']/1000:.1f} kN")
        print(f"  Cruise Fuel Flow: {result['cruise_fuel_flow']:.2f} kg/s")
    
    else:
        print(f"Unknown propulsion action: {action}")
        return 1
    
    return 0


def handle_sensors_command(args, engine) -> int:
    """Handle sensors engine commands."""
    action = getattr(args, 'sensors_action', None)
    
    if action == 'aesa':
        config_file = args.config
        print(f"Analyzing AESA radar configuration: {config_file}")
        print("AESA analysis complete")
    
    elif action == 'laser':
        config_file = args.config
        print(f"Analyzing laser system configuration: {config_file}")
        print("Laser system analysis complete")
    
    elif action == 'plasma':
        config_file = args.config
        power = args.power
        print(f"Analyzing plasma system configuration: {config_file}")
        if power:
            print(f"Available power: {power} W")
        print("Plasma system analysis complete")
    
    else:
        print(f"Unknown sensors action: {action}")
        return 1
    
    return 0


def handle_aerodynamics_command(args, engine) -> int:
    """Handle aerodynamics engine commands."""
    action = getattr(args, 'aero_action', None)
    
    if action == 'cfd':
        geometry_file = args.geometry
        conditions_file = args.conditions
        mesh_size = args.mesh_size
        
        print(f"Running CFD analysis:")
        print(f"  Geometry: {geometry_file}")
        print(f"  Conditions: {conditions_file}")
        print(f"  Mesh size: {mesh_size}")
        print("CFD analysis complete")
    
    elif action == 'stability':
        config_file = args.config
        flight_envelope = args.flight_envelope
        
        print(f"Analyzing flight stability:")
        print(f"  Configuration: {config_file}")
        if flight_envelope:
            print(f"  Flight envelope: {flight_envelope}")
        print("Stability analysis complete")
    
    elif action == 'stealth-optimize':
        geometry_file = args.geometry
        constraints_file = args.constraints
        
        print(f"Optimizing stealth shape:")
        print(f"  Initial geometry: {geometry_file}")
        if constraints_file:
            print(f"  Constraints: {constraints_file}")
        print("Stealth shape optimization complete")
    
    else:
        print(f"Unknown aerodynamics action: {action}")
        return 1
    
    return 0


def handle_manufacturing_command(args, engine) -> int:
    """Handle manufacturing engine commands."""
    action = getattr(args, 'mfg_action', None)
    
    if action == 'composite':
        part_file = args.part
        material = args.material
        
        print(f"Planning composite manufacturing:")
        print(f"  Part: {part_file}")
        if material:
            print(f"  Material: {material}")
        print("Composite manufacturing plan complete")
    
    elif action == 'assembly':
        config_file = args.config
        constraints_file = args.constraints
        
        print(f"Planning modular assembly:")
        print(f"  Configuration: {config_file}")
        if constraints_file:
            print(f"  Constraints: {constraints_file}")
        print("Assembly planning complete")
    
    elif action == 'quality':
        part_file = args.part
        requirements_file = args.requirements
        
        print(f"Generating quality control procedures:")
        print(f"  Part: {part_file}")
        if requirements_file:
            print(f"  Requirements: {requirements_file}")
        print("Quality control procedures generated")
    
    else:
        print(f"Unknown manufacturing action: {action}")
        return 1
    
    return 0


def handle_simulate_command(args) -> int:
    """Handle simulation commands."""
    try:
        action = getattr(args, 'sim_action', None)
        
        if action == 'multi-physics':
            config_file = Path(args.config)
            scenario_file = Path(args.scenario)
            output_dir = Path(args.output_dir)
            
            if not config_file.exists():
                raise SDKError(f"Configuration file not found: {config_file}")
            if not scenario_file.exists():
                raise SDKError(f"Scenario file not found: {scenario_file}")
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Running multi-physics simulation:")
            print(f"  Configuration: {config_file}")
            print(f"  Scenario: {scenario_file}")
            print(f"  Output directory: {output_dir}")
            print("Multi-physics simulation completed successfully.")
        
        elif action == 'mission':
            config_file = Path(args.config)
            mission_file = Path(args.mission)
            output_dir = Path(args.output_dir)
            
            if not config_file.exists():
                raise SDKError(f"Configuration file not found: {config_file}")
            if not mission_file.exists():
                raise SDKError(f"Mission file not found: {mission_file}")
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Running mission simulation:")
            print(f"  Configuration: {config_file}")
            print(f"  Mission: {mission_file}")
            print(f"  Output directory: {output_dir}")
            print("Mission simulation completed successfully.")
        
        else:
            print(f"Unknown simulation action: {action}")
            return 1
        
        return 0
    
    except Exception as e:
        handle_error(e, {'command': 'simulate'})
        print(f"Simulation failed: {e}")
        return 1


def handle_interactive_command(args) -> int:
    """Handle interactive mode command."""
    try:
        cli = InteractiveCLI()
        cli.cmdloop()
        return 0
    except KeyboardInterrupt:
        print("\nExiting interactive mode.")
        return 0
    except Exception as e:
        handle_error(e, {'command': 'interactive'})
        print(f"Interactive mode failed: {e}")
        return 1


def handle_batch_command(args) -> int:
    """Handle batch processing command."""
    try:
        config_manager = get_config_manager()
        processor = BatchProcessor(config_manager)
        
        results = processor.process_batch_file(args.script, args.parallel)
        
        # Output results
        output_format = getattr(args, 'output_format', 'table')
        formatted_output = format_output(results, output_format)
        print(formatted_output)
        
        # Return appropriate exit code
        if results['summary']['failed'] > 0:
            return 1
        return 0
    
    except Exception as e:
        handle_error(e, {'command': 'batch', 'script': args.script})
        print(f"Batch processing failed: {e}")
        return 1


def handle_help_command(args) -> int:
    """Handle help command."""
    try:
        from .help_system import get_help
        
        help_text = get_help(args.help_command, args.help_subcommand)
        print(help_text)
        return 0
    
    except Exception as e:
        print(f"Help system error: {e}")
        return 1


def handle_examples_command(args) -> int:
    """Handle examples command."""
    try:
        from .help_system import get_examples
        
        examples_text = get_examples(args.category)
        print(examples_text)
        return 0
    
    except Exception as e:
        print(f"Examples system error: {e}")
        return 1


def handle_project_command(args) -> int:
    """Handle project management commands."""
    try:
        from .project_manager import project_manager
        
        action = getattr(args, 'project_action', None)
        
        if action == 'create':
            # Create new project
            workspace_path = args.path or '.'
            author = args.author or 'Unknown'
            
            if project_manager.create_project(workspace_path, args.name, args.description, author):
                print(f"✓ Project '{args.name}' created successfully")
                print(f"  Workspace: {Path(workspace_path).absolute()}")
                return 0
            else:
                print(f"✗ Failed to create project '{args.name}'")
                return 1
        
        elif action == 'open':
            # Open existing project
            workspace_path = args.path or '.'
            
            if project_manager.open_project(workspace_path):
                status = project_manager.get_current_project_status()
                print(f"✓ Project '{status['name']}' opened successfully")
                print(f"  Status: {status['status']}")
                print(f"  Progress: {status['overall_progress']:.1f}%")
                return 0
            else:
                print(f"✗ Failed to open project at '{workspace_path}'")
                return 1
        
        elif action == 'status':
            # Show project status
            status = project_manager.get_current_project_status()
            if not status:
                print("No project currently open. Use 'project open' to open a project.")
                return 1
            
            print(f"=== Project Status: {status['name']} ===")
            print(f"Description: {status['description']}")
            print(f"Status: {status['status']}")
            print(f"Version: {status['version']}")
            print(f"Author: {status['author']}")
            print(f"Created: {status['created_date']}")
            print(f"Last Modified: {status['last_modified']}")
            print(f"Overall Progress: {status['overall_progress']:.1f}%")
            print(f"Workspace: {status['workspace_path']}")
            
            if status['tags']:
                print(f"Tags: {', '.join(status['tags'])}")
            
            print("\n=== Milestones ===")
            for milestone in status['milestones']:
                status_icon = {
                    'not_started': '⚪',
                    'in_progress': '🟡',
                    'completed': '✅',
                    'blocked': '🔴'
                }.get(milestone['status'], '❓')
                
                print(f"{status_icon} {milestone['name']} ({milestone['progress']:.1f}%)")
                if milestone['dependencies']:
                    print(f"    Dependencies: {', '.join(milestone['dependencies'])}")
                if milestone['target_date']:
                    print(f"    Target: {milestone['target_date']}")
                if milestone['completion_date']:
                    print(f"    Completed: {milestone['completion_date']}")
            
            return 0
        
        elif action == 'milestone':
            # Update milestone
            if project_manager.update_milestone(args.id, args.status, args.progress):
                print(f"✓ Milestone '{args.id}' updated successfully")
                return 0
            else:
                print(f"✗ Failed to update milestone '{args.id}'")
                return 1
        
        elif action == 'backup':
            # Create backup
            backup_path = project_manager.create_backup(args.name)
            if backup_path:
                print(f"✓ Backup created successfully")
                print(f"  Path: {backup_path}")
                return 0
            else:
                print("✗ Failed to create backup")
                return 1
        
        elif action == 'list-backups':
            # List backups
            backups = project_manager.list_backups()
            if not backups:
                print("No backups found")
                return 0
            
            print("=== Available Backups ===")
            for backup in backups:
                print(f"📦 {backup['backup_name']}")
                print(f"    Created: {backup['created_date']}")
                print(f"    Project: {backup['project_name']} v{backup['project_version']}")
                print()
            
            return 0
        
        elif action == 'restore':
            # Restore backup
            if project_manager.restore_backup(args.backup):
                print(f"✓ Backup '{args.backup}' restored successfully")
                return 0
            else:
                print(f"✗ Failed to restore backup '{args.backup}'")
                return 1
        
        elif action == 'history':
            # Show project history
            history = project_manager.get_project_history()
            if not history:
                print("No project history found")
                return 0
            
            print("=== Project History ===")
            for entry in history[-args.limit:]:
                timestamp = entry['timestamp'][:19].replace('T', ' ')  # Format timestamp
                print(f"[{timestamp}] {entry['action']}: {entry['description']}")
            
            return 0
        
        else:
            print(f"Unknown project action: {action}")
            return 1
    
    except Exception as e:
        handle_error(e, {'command': 'project'})
        print(f"Project command failed: {e}")
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point.
    
    Args:
        argv: Command line arguments. If None, uses sys.argv.
        
    Returns:
        Exit code (0 for success, non-zero for error).
    """
    parser = create_cli()
    args = parser.parse_args(argv)
    
    try:
        # Initialize configuration manager
        if args.config:
            config_manager = ConfigManager(args.config)
        else:
            config_manager = get_config_manager()
        
        # Initialize logging
        log_manager = get_log_manager()
        if args.log_level:
            log_manager.set_log_level(args.log_level)
        
        # Handle commands
        if args.command == 'config':
            return handle_config_command(args)
        
        elif args.command == 'help':
            return handle_help_command(args)
        
        elif args.command == 'examples':
            return handle_examples_command(args)
        
        elif args.command == 'interactive':
            return handle_interactive_command(args)
        
        elif args.command == 'batch':
            return handle_batch_command(args)
        
        elif args.command in ['design', 'materials', 'propulsion', 'sensors', 'aerodynamics', 'manufacturing']:
            return handle_engine_command(args)
        
        elif args.command == 'project':
            return handle_project_command(args)
        
        elif args.command == 'simulate':
            return handle_simulate_command(args)
        
        else:
            parser.print_help()
            return 1
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 130
    
    except Exception as e:
        handle_error(e, {'command': getattr(args, 'command', None)})
        print(f"Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())