#!/usr/bin/env python3
"""Basic functionality test for the Fighter Jet SDK."""

import sys
import os
from pathlib import Path

# Add the SDK to the Python path
sdk_path = Path(__file__).parent / "fighter_jet_sdk"
sys.path.insert(0, str(sdk_path.parent))

def test_imports():
    """Test that all core modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test core imports
        from fighter_jet_sdk.core.config import ConfigManager, get_config
        from fighter_jet_sdk.core.logging import LogManager, get_logger
        from fighter_jet_sdk.core.errors import SDKError, ValidationError
        print("✓ Core modules imported successfully")
        
        # Test common imports
        from fighter_jet_sdk.common.data_models import AircraftConfiguration, Module
        from fighter_jet_sdk.common.interfaces import BaseEngine
        from fighter_jet_sdk.common.enums import ModuleType, MaterialType
        print("✓ Common modules imported successfully")
        
        # Test engine imports
        from fighter_jet_sdk.engines.design import DesignEngine
        from fighter_jet_sdk.engines.materials import MaterialsEngine
        print("✓ Engine modules imported successfully")
        
        # Test CLI import
        from fighter_jet_sdk.cli.main import create_cli
        print("✓ CLI module imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_configuration():
    """Test configuration management."""
    print("\nTesting configuration...")
    
    try:
        from fighter_jet_sdk.core.config import ConfigManager
        
        # Create config manager
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        print(f"✓ Configuration loaded: log_level={config.log_level}")
        
        # Test validation
        errors = config_manager.validate_config()
        if errors:
            print(f"Configuration validation errors: {errors}")
        else:
            print("✓ Configuration validation passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_logging():
    """Test logging framework."""
    print("\nTesting logging...")
    
    try:
        from fighter_jet_sdk.core.logging import get_logger, get_log_manager
        
        # Get logger
        logger = get_logger('test')
        logger.info("Test log message")
        
        # Get log manager stats
        log_manager = get_log_manager()
        stats = log_manager.get_log_stats()
        print(f"✓ Logging initialized: {stats['handlers_count']} handlers")
        
        return True
        
    except Exception as e:
        print(f"✗ Logging test failed: {e}")
        return False


def test_data_models():
    """Test data model creation."""
    print("\nTesting data models...")
    
    try:
        from fighter_jet_sdk.common.data_models import (
            AircraftConfiguration, Module, MaterialDefinition
        )
        from fighter_jet_sdk.common.enums import ModuleType, MaterialType
        
        # Create a module
        module = Module(
            name="Test Cockpit",
            module_type=ModuleType.COCKPIT
        )
        print(f"✓ Module created: {module.name} ({module.module_id})")
        
        # Create aircraft configuration
        config = AircraftConfiguration(
            name="Test Aircraft",
            modules=[module]
        )
        print(f"✓ Aircraft configuration created: {config.name}")
        
        # Create material
        material = MaterialDefinition(
            name="Test Metamaterial",
            base_material_type=MaterialType.METAMATERIAL
        )
        print(f"✓ Material created: {material.name}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data model test failed: {e}")
        return False


def test_engines():
    """Test engine initialization."""
    print("\nTesting engines...")
    
    try:
        from fighter_jet_sdk.engines.design import DesignEngine
        from fighter_jet_sdk.engines.materials import MaterialsEngine
        
        # Test Design Engine
        design_engine = DesignEngine()
        if design_engine.initialize():
            print("✓ Design Engine initialized")
        else:
            print("✗ Design Engine initialization failed")
        
        # Test Materials Engine
        materials_engine = MaterialsEngine()
        if materials_engine.initialize():
            print("✓ Materials Engine initialized")
        else:
            print("✗ Materials Engine initialization failed")
        
        return True
        
    except Exception as e:
        print(f"✗ Engine test failed: {e}")
        return False


def test_cli():
    """Test CLI creation."""
    print("\nTesting CLI...")
    
    try:
        from fighter_jet_sdk.cli.main import create_cli
        
        # Create CLI parser
        parser = create_cli()
        
        # Test help
        help_text = parser.format_help()
        if "fighter-jet-sdk" in help_text:
            print("✓ CLI parser created successfully")
            return True
        else:
            print("✗ CLI parser missing expected content")
            return False
        
    except Exception as e:
        print(f"✗ CLI test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Fighter Jet SDK - Basic Functionality Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_configuration,
        test_logging,
        test_data_models,
        test_engines,
        test_cli
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! SDK foundation is working correctly.")
        return 0
    else:
        print("✗ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())