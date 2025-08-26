"""Configuration management system for the Fighter Jet SDK."""

import json
import yaml
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
import logging

from .errors import ConfigurationError


@dataclass
class SDKConfig:
    """Main SDK configuration."""
    # Logging configuration
    log_level: str = "INFO"
    log_file: Optional[str] = None
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Engine configurations
    engines: Dict[str, Dict[str, Any]] = None
    
    # Simulation settings
    simulation_precision: str = "double"
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-6
    
    # Performance settings
    parallel_processing: bool = True
    max_threads: Optional[int] = None
    cache_enabled: bool = True
    cache_size_mb: int = 1024
    
    # Data storage settings
    data_directory: str = "./data"
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    
    # Security settings
    encryption_enabled: bool = False
    access_control_enabled: bool = False
    
    def __post_init__(self):
        """Initialize default engine configurations."""
        if self.engines is None:
            self.engines = {
                "design": {
                    "module_library_path": "./modules",
                    "validation_strict": True,
                    "auto_optimize": False
                },
                "materials": {
                    "database_path": "./materials.db",
                    "simulation_precision": "high",
                    "frequency_points": 1000
                },
                "propulsion": {
                    "cfd_solver": "openfoam",
                    "thermal_analysis": True,
                    "optimization_enabled": True
                },
                "sensors": {
                    "atmospheric_model": "standard",
                    "noise_modeling": True,
                    "multi_target_tracking": True
                },
                "aerodynamics": {
                    "cfd_mesh_density": "medium",
                    "turbulence_model": "k-omega-sst",
                    "compressibility_effects": True
                },
                "manufacturing": {
                    "cost_database_path": "./costs.db",
                    "quality_standards": "aerospace",
                    "optimization_method": "genetic_algorithm"
                }
            }


class ConfigManager:
    """Configuration manager for the Fighter Jet SDK."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default locations.
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = self._resolve_config_path(config_path)
        self.config = SDKConfig()
        self._load_config()
    
    def _resolve_config_path(self, config_path: Optional[Union[str, Path]]) -> Path:
        """Resolve configuration file path."""
        if config_path:
            return Path(config_path)
        
        # Check for config files in order of preference
        possible_paths = [
            Path("./fighter_jet_sdk_config.yaml"),
            Path("./fighter_jet_sdk_config.json"),
            Path("~/.fighter_jet_sdk/config.yaml").expanduser(),
            Path("~/.fighter_jet_sdk/config.json").expanduser(),
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # Return default path if no config file found
        return Path("./fighter_jet_sdk_config.yaml")
    
    def _load_config(self) -> None:
        """Load configuration from file."""
        if not self.config_path.exists():
            self.logger.info(f"Configuration file not found at {self.config_path}. Using defaults.")
            return
        
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                elif self.config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                else:
                    raise ConfigurationError(f"Unsupported config file format: {self.config_path.suffix}")
            
            # Update config with loaded data
            self._update_config_from_dict(config_data)
            self.logger.info(f"Configuration loaded from {self.config_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def _update_config_from_dict(self, config_data: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for key, value in config_data.items():
            if hasattr(self.config, key):
                if key == 'engines' and isinstance(value, dict):
                    # Merge engine configurations
                    for engine_name, engine_config in value.items():
                        if engine_name in self.config.engines:
                            self.config.engines[engine_name].update(engine_config)
                        else:
                            self.config.engines[engine_name] = engine_config
                else:
                    setattr(self.config, key, value)
            else:
                self.logger.warning(f"Unknown configuration key: {key}")
    
    def save_config(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Save configuration to file.
        
        Args:
            config_path: Path to save configuration. If None, uses current config_path.
        """
        save_path = Path(config_path) if config_path else self.config_path
        
        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            config_dict = asdict(self.config)
            
            with open(save_path, 'w') as f:
                if save_path.suffix.lower() == '.json':
                    json.dump(config_dict, f, indent=2)
                elif save_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    raise ConfigurationError(f"Unsupported config file format: {save_path.suffix}")
            
            self.logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}")
    
    def get_config(self) -> SDKConfig:
        """Get current configuration."""
        return self.config
    
    def get_engine_config(self, engine_name: str) -> Dict[str, Any]:
        """Get configuration for specific engine.
        
        Args:
            engine_name: Name of the engine.
            
        Returns:
            Engine configuration dictionary.
        """
        if engine_name not in self.config.engines:
            raise ConfigurationError(f"Unknown engine: {engine_name}")
        
        return self.config.engines[engine_name]
    
    def update_engine_config(self, engine_name: str, config_updates: Dict[str, Any]) -> None:
        """Update configuration for specific engine.
        
        Args:
            engine_name: Name of the engine.
            config_updates: Configuration updates to apply.
        """
        if engine_name not in self.config.engines:
            self.config.engines[engine_name] = {}
        
        self.config.engines[engine_name].update(config_updates)
        self.logger.info(f"Updated configuration for engine: {engine_name}")
    
    def validate_config(self) -> List[str]:
        """Validate current configuration.
        
        Returns:
            List of validation errors (empty if valid).
        """
        errors = []
        
        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.config.log_level not in valid_log_levels:
            errors.append(f"Invalid log_level: {self.config.log_level}")
        
        # Validate simulation precision
        valid_precisions = ['single', 'double', 'extended']
        if self.config.simulation_precision not in valid_precisions:
            errors.append(f"Invalid simulation_precision: {self.config.simulation_precision}")
        
        # Validate numeric ranges
        if self.config.max_iterations <= 0:
            errors.append("max_iterations must be positive")
        
        if self.config.convergence_tolerance <= 0:
            errors.append("convergence_tolerance must be positive")
        
        if self.config.cache_size_mb <= 0:
            errors.append("cache_size_mb must be positive")
        
        if self.config.backup_interval_hours <= 0:
            errors.append("backup_interval_hours must be positive")
        
        # Validate paths exist if specified
        data_dir = Path(self.config.data_directory)
        if not data_dir.exists():
            try:
                data_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create data_directory: {e}")
        
        return errors
    
    def create_default_config(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Create a default configuration file.
        
        Args:
            config_path: Path where to create the config file.
        """
        save_path = Path(config_path) if config_path else self.config_path
        
        # Reset to defaults
        self.config = SDKConfig()
        
        # Save default configuration
        self.save_config(save_path)
        self.logger.info(f"Default configuration created at {save_path}")


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> SDKConfig:
    """Get current SDK configuration."""
    return get_config_manager().get_config()


def get_engine_config(engine_name: str) -> Dict[str, Any]:
    """Get configuration for specific engine."""
    return get_config_manager().get_engine_config(engine_name)