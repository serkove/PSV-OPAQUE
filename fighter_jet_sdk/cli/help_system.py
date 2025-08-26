"""Comprehensive help system for the Fighter Jet SDK CLI."""

from typing import Dict, List, Optional
import textwrap


class HelpSystem:
    """Comprehensive help and documentation system."""
    
    def __init__(self):
        self.commands = self._build_command_help()
        self.examples = self._build_examples()
        self.workflows = self._build_workflows()
    
    def get_command_help(self, command: str, subcommand: Optional[str] = None) -> str:
        """Get help for a specific command or subcommand."""
        if command not in self.commands:
            return f"Unknown command: {command}"
        
        cmd_help = self.commands[command]
        
        if subcommand:
            if 'subcommands' in cmd_help and subcommand in cmd_help['subcommands']:
                sub_help = cmd_help['subcommands'][subcommand]
                return self._format_help(sub_help, f"{command} {subcommand}")
            else:
                return f"Unknown subcommand: {command} {subcommand}"
        
        return self._format_help(cmd_help, command)
    
    def get_workflow_help(self, workflow: str) -> str:
        """Get help for a specific workflow."""
        if workflow not in self.workflows:
            return f"Unknown workflow: {workflow}"
        
        return self.workflows[workflow]
    
    def list_commands(self) -> List[str]:
        """List all available commands."""
        return list(self.commands.keys())
    
    def list_workflows(self) -> List[str]:
        """List all available workflows."""
        return list(self.workflows.keys())
    
    def get_examples(self, category: Optional[str] = None) -> str:
        """Get examples for a specific category or all examples."""
        if category:
            if category in self.examples:
                return self.examples[category]
            else:
                return f"No examples found for category: {category}"
        
        # Return all examples
        result = "=== Fighter Jet SDK Examples ===\n\n"
        for cat, examples in self.examples.items():
            result += f"## {cat.title()}\n{examples}\n\n"
        
        return result
    
    def _format_help(self, help_data: Dict, command_name: str) -> str:
        """Format help data into readable text."""
        result = f"=== {command_name.upper()} ===\n\n"
        
        if 'description' in help_data:
            result += f"{help_data['description']}\n\n"
        
        if 'usage' in help_data:
            result += f"Usage: {help_data['usage']}\n\n"
        
        if 'options' in help_data:
            result += "Options:\n"
            for option, desc in help_data['options'].items():
                result += f"  {option:<20} {desc}\n"
            result += "\n"
        
        if 'examples' in help_data:
            result += "Examples:\n"
            for example in help_data['examples']:
                result += f"  {example}\n"
            result += "\n"
        
        if 'subcommands' in help_data:
            result += "Subcommands:\n"
            for subcmd, desc in help_data['subcommands'].items():
                if isinstance(desc, dict) and 'description' in desc:
                    result += f"  {subcmd:<15} {desc['description']}\n"
                else:
                    result += f"  {subcmd:<15} {desc}\n"
            result += "\n"
        
        return result
    
    def _build_command_help(self) -> Dict:
        """Build comprehensive command help documentation."""
        return {
            'config': {
                'description': 'Configuration management for the Fighter Jet SDK',
                'usage': 'fighter-jet-sdk config <action> [options]',
                'subcommands': {
                    'init': {
                        'description': 'Initialize default configuration',
                        'usage': 'fighter-jet-sdk config init',
                        'examples': ['fighter-jet-sdk config init']
                    },
                    'show': {
                        'description': 'Display current configuration',
                        'usage': 'fighter-jet-sdk config show',
                        'examples': ['fighter-jet-sdk config show']
                    },
                    'validate': {
                        'description': 'Validate configuration file',
                        'usage': 'fighter-jet-sdk config validate',
                        'examples': ['fighter-jet-sdk config validate']
                    }
                }
            },
            
            'design': {
                'description': 'Aircraft design and configuration management',
                'usage': 'fighter-jet-sdk design <action> [options]',
                'subcommands': {
                    'create': {
                        'description': 'Create new aircraft configuration',
                        'usage': 'fighter-jet-sdk design create --name <name> [--platform <platform>] [--output <file>]',
                        'options': {
                            '--name': 'Configuration name (required)',
                            '--platform': 'Base platform type (optional)',
                            '--output': 'Output file path (optional)'
                        },
                        'examples': [
                            'fighter-jet-sdk design create --name "Stealth Fighter" --platform su75',
                            'fighter-jet-sdk design create --name "Multi-Role" --output config.json'
                        ]
                    },
                    'list': {
                        'description': 'List available modules or configurations',
                        'usage': 'fighter-jet-sdk design list [--type <type>] [--filter <filter>]',
                        'options': {
                            '--type': 'Type to list: modules, configs (default: modules)',
                            '--filter': 'Filter by module type or name'
                        },
                        'examples': [
                            'fighter-jet-sdk design list',
                            'fighter-jet-sdk design list --type modules --filter sensor'
                        ]
                    },
                    'validate': {
                        'description': 'Validate aircraft configuration',
                        'usage': 'fighter-jet-sdk design validate --config <file>',
                        'options': {
                            '--config': 'Configuration file to validate (required)'
                        },
                        'examples': [
                            'fighter-jet-sdk design validate --config my_aircraft.json'
                        ]
                    }
                }
            },
            
            'materials': {
                'description': 'Advanced materials analysis and modeling',
                'usage': 'fighter-jet-sdk materials <action> [options]',
                'subcommands': {
                    'list': {
                        'description': 'List available materials',
                        'usage': 'fighter-jet-sdk materials list [--type <type>]',
                        'options': {
                            '--type': 'Material type: all, metamaterials, uhtc (default: all)'
                        },
                        'examples': [
                            'fighter-jet-sdk materials list',
                            'fighter-jet-sdk materials list --type metamaterials'
                        ]
                    },
                    'metamaterial': {
                        'description': 'Analyze metamaterial properties',
                        'usage': 'fighter-jet-sdk materials metamaterial --material <id> [options]',
                        'options': {
                            '--material': 'Material ID (required)',
                            '--frequencies': 'Frequency range as JSON array',
                            '--thickness': 'Material thickness in meters (default: 1e-3)'
                        },
                        'examples': [
                            'fighter-jet-sdk materials metamaterial --material META001',
                            'fighter-jet-sdk materials metamaterial --material META001 --frequencies "[1e9, 10e9, 100e9]"'
                        ]
                    },
                    'stealth': {
                        'description': 'Perform stealth analysis',
                        'usage': 'fighter-jet-sdk materials stealth --geometry <file> [options]',
                        'options': {
                            '--geometry': 'Geometry file (required)',
                            '--frequencies': 'Frequency range as JSON array',
                            '--angles': 'Angle range as JSON array'
                        },
                        'examples': [
                            'fighter-jet-sdk materials stealth --geometry aircraft.stl',
                            'fighter-jet-sdk materials stealth --geometry aircraft.stl --frequencies "[1e9, 10e9]"'
                        ]
                    }
                }
            },
            
            'propulsion': {
                'description': 'Engine performance analysis and optimization',
                'usage': 'fighter-jet-sdk propulsion <action> [options]',
                'subcommands': {
                    'list': {
                        'description': 'List available engines',
                        'usage': 'fighter-jet-sdk propulsion list',
                        'examples': ['fighter-jet-sdk propulsion list']
                    },
                    'analyze': {
                        'description': 'Analyze engine performance',
                        'usage': 'fighter-jet-sdk propulsion analyze --engine <id> [options]',
                        'options': {
                            '--engine': 'Engine ID (required)',
                            '--altitude': 'Altitude in meters (default: 0)',
                            '--mach': 'Mach number (default: 0)',
                            '--throttle': 'Throttle setting 0-1 (default: 1.0)',
                            '--afterburner': 'Enable afterburner (flag)'
                        },
                        'examples': [
                            'fighter-jet-sdk propulsion analyze --engine f119_pw_100',
                            'fighter-jet-sdk propulsion analyze --engine f119_pw_100 --altitude 10000 --mach 1.5 --afterburner'
                        ]
                    },
                    'optimize': {
                        'description': 'Optimize cruise performance',
                        'usage': 'fighter-jet-sdk propulsion optimize --engine <id> --mass <kg> [options]',
                        'options': {
                            '--engine': 'Engine ID (required)',
                            '--mass': 'Aircraft mass in kg (required)',
                            '--alt-range': 'Altitude range as JSON array (default: [8000, 15000])',
                            '--mach-range': 'Mach range as JSON array (default: [0.7, 1.2])'
                        },
                        'examples': [
                            'fighter-jet-sdk propulsion optimize --engine f119_pw_100 --mass 19700',
                            'fighter-jet-sdk propulsion optimize --engine f119_pw_100 --mass 19700 --alt-range "[10000, 18000]"'
                        ]
                    }
                }
            },
            
            'interactive': {
                'description': 'Start interactive command mode',
                'usage': 'fighter-jet-sdk interactive',
                'examples': ['fighter-jet-sdk interactive']
            },
            
            'batch': {
                'description': 'Run batch processing from script file',
                'usage': 'fighter-jet-sdk batch --script <file> [options]',
                'options': {
                    '--script': 'Path to batch script file (YAML or JSON) (required)',
                    '--parallel': 'Run operations in parallel where possible (flag)'
                },
                'examples': [
                    'fighter-jet-sdk batch --script analysis_batch.yaml',
                    'fighter-jet-sdk batch --script analysis_batch.yaml --parallel'
                ]
            },
            
            'project': {
                'description': 'Project workspace management and tracking',
                'usage': 'fighter-jet-sdk project <action> [options]',
                'subcommands': {
                    'create': {
                        'description': 'Create new project workspace',
                        'usage': 'fighter-jet-sdk project create --name <name> --description <desc> [options]',
                        'options': {
                            '--name': 'Project name (required)',
                            '--description': 'Project description (required)',
                            '--author': 'Project author (optional)',
                            '--path': 'Project workspace path (default: current directory)'
                        },
                        'examples': [
                            'fighter-jet-sdk project create --name "Stealth Fighter" --description "Advanced stealth aircraft design"',
                            'fighter-jet-sdk project create --name "Multi-Role Fighter" --description "Versatile combat aircraft" --author "John Doe"'
                        ]
                    },
                    'open': {
                        'description': 'Open existing project workspace',
                        'usage': 'fighter-jet-sdk project open [--path <path>]',
                        'options': {
                            '--path': 'Project workspace path (default: current directory)'
                        },
                        'examples': [
                            'fighter-jet-sdk project open',
                            'fighter-jet-sdk project open --path /path/to/project'
                        ]
                    },
                    'status': {
                        'description': 'Show project status and progress',
                        'usage': 'fighter-jet-sdk project status',
                        'examples': ['fighter-jet-sdk project status']
                    },
                    'milestone': {
                        'description': 'Update project milestone',
                        'usage': 'fighter-jet-sdk project milestone --id <id> [options]',
                        'options': {
                            '--id': 'Milestone ID (required)',
                            '--status': 'New milestone status (not_started, in_progress, completed, blocked)',
                            '--progress': 'Progress percentage (0-100)'
                        },
                        'examples': [
                            'fighter-jet-sdk project milestone --id requirements --status completed',
                            'fighter-jet-sdk project milestone --id design --progress 75'
                        ]
                    },
                    'backup': {
                        'description': 'Create project backup',
                        'usage': 'fighter-jet-sdk project backup [--name <name>]',
                        'options': {
                            '--name': 'Backup name (optional, auto-generated if not provided)'
                        },
                        'examples': [
                            'fighter-jet-sdk project backup',
                            'fighter-jet-sdk project backup --name milestone_1_complete'
                        ]
                    },
                    'restore': {
                        'description': 'Restore project from backup',
                        'usage': 'fighter-jet-sdk project restore --backup <name>',
                        'options': {
                            '--backup': 'Backup name to restore (required)'
                        },
                        'examples': [
                            'fighter-jet-sdk project restore --backup milestone_1_complete'
                        ]
                    }
                }
            }
        }
    
    def _build_examples(self) -> Dict[str, str]:
        """Build example usage scenarios."""
        return {
            'basic': textwrap.dedent("""
                # Initialize configuration
                fighter-jet-sdk config init
                
                # List available engines
                fighter-jet-sdk propulsion list
                
                # Analyze engine performance
                fighter-jet-sdk propulsion analyze --engine f119_pw_100 --altitude 10000 --mach 1.5
                
                # List available materials
                fighter-jet-sdk materials list
                
                # Create new aircraft design
                fighter-jet-sdk design create --name "Stealth Fighter" --platform su75
            """).strip(),
            
            'advanced': textwrap.dedent("""
                # Optimize engine for cruise conditions
                fighter-jet-sdk propulsion optimize --engine f119_pw_100 --mass 19700 --alt-range "[10000, 18000]"
                
                # Analyze metamaterial properties
                fighter-jet-sdk materials metamaterial --material META001 --frequencies "[1e9, 10e9, 100e9]"
                
                # Perform stealth analysis
                fighter-jet-sdk materials stealth --geometry aircraft.stl --frequencies "[1e9, 10e9]"
                
                # Run batch analysis
                fighter-jet-sdk batch --script comprehensive_analysis.yaml --parallel
            """).strip(),
            
            'interactive': textwrap.dedent("""
                # Start interactive mode
                fighter-jet-sdk interactive
                
                # In interactive mode:
                fighter-jet-sdk> propulsion list
                fighter-jet-sdk> propulsion analyze f119_pw_100 --altitude 10000 --mach 1.5
                fighter-jet-sdk> materials list --type metamaterials
                fighter-jet-sdk> design create "Test Configuration"
                fighter-jet-sdk> status
                fighter-jet-sdk> exit
            """).strip(),
            
            'batch': textwrap.dedent("""
                # Example batch script (YAML format):
                operations:
                  - name: "Engine Performance Analysis"
                    type: "propulsion"
                    action: "analyze"
                    engine_id: "f119_pw_100"
                    conditions:
                      altitude: 10000
                      mach_number: 1.5
                      throttle_setting: 1.0
                      afterburner_engaged: true
                
                  - name: "Metamaterial Analysis"
                    type: "materials"
                    action: "metamaterial"
                    material_id: "META001"
                    frequencies: [1e9, 10e9, 100e9]
                    thickness: 0.001
                
                  - name: "Design Validation"
                    type: "design"
                    action: "validate"
                    config_file: "aircraft_config.json"
            """).strip()
        }
    
    def _build_workflows(self) -> Dict[str, str]:
        """Build workflow documentation."""
        return {
            'aircraft_design': textwrap.dedent("""
                === Aircraft Design Workflow ===
                
                1. Initialize SDK configuration:
                   fighter-jet-sdk config init
                
                2. Create base aircraft configuration:
                   fighter-jet-sdk design create --name "My Aircraft" --platform su75
                
                3. List available modules:
                   fighter-jet-sdk design list --type modules
                
                4. Add modules to configuration:
                   fighter-jet-sdk design add-module --config my_aircraft.json --module SENSOR_001
                
                5. Validate complete configuration:
                   fighter-jet-sdk design validate --config my_aircraft.json
                
                6. Optimize for mission requirements:
                   fighter-jet-sdk design optimize --config my_aircraft.json --mission mission.json
            """).strip(),
            
            'materials_analysis': textwrap.dedent("""
                === Materials Analysis Workflow ===
                
                1. List available materials:
                   fighter-jet-sdk materials list
                
                2. Analyze metamaterial properties:
                   fighter-jet-sdk materials metamaterial --material META001 --frequencies "[1e9, 10e9, 100e9]"
                
                3. Perform stealth analysis:
                   fighter-jet-sdk materials stealth --geometry aircraft.stl --frequencies "[1e9, 10e9]"
                
                4. Analyze thermal properties:
                   fighter-jet-sdk materials thermal --material UHTC001 --conditions '{"mach": 5.0, "altitude": 30000}'
            """).strip(),
            
            'propulsion_optimization': textwrap.dedent("""
                === Propulsion Optimization Workflow ===
                
                1. List available engines:
                   fighter-jet-sdk propulsion list
                
                2. Analyze baseline performance:
                   fighter-jet-sdk propulsion analyze --engine f119_pw_100 --altitude 0 --mach 0
                
                3. Analyze high-altitude performance:
                   fighter-jet-sdk propulsion analyze --engine f119_pw_100 --altitude 15000 --mach 2.0 --afterburner
                
                4. Optimize cruise conditions:
                   fighter-jet-sdk propulsion optimize --engine f119_pw_100 --mass 19700
                
                5. Calculate mission fuel consumption:
                   fighter-jet-sdk propulsion mission --engine f119_pw_100 --profile mission_profile.json
            """).strip(),
            
            'batch_processing': textwrap.dedent("""
                === Batch Processing Workflow ===
                
                1. Create batch script file (YAML or JSON format)
                2. Define operations with required parameters
                3. Run batch processing:
                   fighter-jet-sdk batch --script analysis_batch.yaml
                
                4. For parallel processing:
                   fighter-jet-sdk batch --script analysis_batch.yaml --parallel
                
                5. Review results in specified output format:
                   fighter-jet-sdk batch --script analysis_batch.yaml --output-format json
            """).strip()
        }


# Global help system instance
help_system = HelpSystem()


def get_help(command: Optional[str] = None, subcommand: Optional[str] = None) -> str:
    """Get help for commands or general help."""
    if command:
        return help_system.get_command_help(command, subcommand)
    else:
        # General help
        result = "=== Fighter Jet SDK Help ===\n\n"
        result += "Available commands:\n"
        for cmd in help_system.list_commands():
            result += f"  {cmd}\n"
        result += "\nUse 'fighter-jet-sdk help <command>' for detailed help on a specific command.\n"
        result += "Use 'fighter-jet-sdk interactive' to start interactive mode.\n"
        result += "Use 'fighter-jet-sdk examples' to see usage examples.\n"
        return result


def get_examples(category: Optional[str] = None) -> str:
    """Get usage examples."""
    return help_system.get_examples(category)


def get_workflow_help(workflow: str) -> str:
    """Get workflow documentation."""
    return help_system.get_workflow_help(workflow)