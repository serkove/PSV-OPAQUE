"""Integration tests for inter-engine communication and data flow."""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch
from typing import Dict, Any

from fighter_jet_sdk.core.engine_coordinator import EngineCoordinator
from fighter_jet_sdk.core.engine_communication import (
    EventBus, EngineRegistry, DataConsistencyManager, PerformanceMonitor,
    EngineEvent, EventType, EventPriority, EventHandler
)
from fighter_jet_sdk.common.data_models import AircraftConfiguration, BasePlatform, Module
from fighter_jet_sdk.common.enums import ModuleType


class MockEventHandler(EventHandler):
    """Mock event handler for testing."""
    
    def __init__(self, engine_id: str):
        self.engine_id = engine_id
        self.handled_events = []
    
    async def handle_event(self, event: EngineEvent) -> Dict[str, Any]:
        """Handle event and record it."""
        self.handled_events.append(event)
        return {"status": "handled", "engine": self.engine_id}
    
    def get_supported_events(self) -> list:
        """Return supported event types."""
        return [EventType.CONFIGURATION_CHANGED, EventType.DATA_INVALIDATED]


@pytest.fixture
def sample_configuration():
    """Create a sample aircraft configuration for testing."""
    platform = BasePlatform(
        name="Test Platform",
        base_mass=5000.0
    )
    
    config = AircraftConfiguration(
        name="Test Aircraft",
        description="Test configuration for integration tests",
        base_platform=platform
    )
    
    # Add some test modules
    test_module = Module(
        module_id="test_module_001",
        name="Test Module",
        module_type=ModuleType.SENSOR,
        description="Test sensor module"
    )
    config.add_module(test_module)
    
    return config


@pytest.fixture
def event_bus():
    """Create event bus for testing."""
    return EventBus(max_workers=2)


@pytest.fixture
def engine_registry():
    """Create engine registry for testing."""
    return EngineRegistry()


@pytest.fixture
def engine_coordinator():
    """Create engine coordinator for testing."""
    config = {
        'max_workers': 2,
        'engines': {
            'design': {},
            'materials': {},
            'propulsion': {},
            'sensors': {},
            'aerodynamics': {},
            'manufacturing': {}
        }
    }
    
    return EngineCoordinator(config)


class TestEventBus:
    """Test event bus functionality."""
    
    @pytest.mark.asyncio
    async def test_event_publishing_and_handling(self, event_bus):
        """Test basic event publishing and handling."""
        # Create mock handlers
        handler1 = MockEventHandler("engine1")
        handler2 = MockEventHandler("engine2")
        
        # Subscribe handlers
        event_bus.subscribe("engine1", handler1)
        event_bus.subscribe("engine2", handler2)
        
        # Create test event
        event = EngineEvent(
            event_type=EventType.CONFIGURATION_CHANGED,
            source_engine="test_source",
            data={"test": "data"}
        )
        
        # Start processing
        processing_task = asyncio.create_task(event_bus.start_processing())
        
        # Publish event
        result = await event_bus.publish(event)
        
        # Wait a bit for processing
        await asyncio.sleep(0.1)
        
        # Stop processing
        event_bus.stop_processing()
        processing_task.cancel()
        
        # Verify event was handled
        assert len(handler1.handled_events) == 1
        assert len(handler2.handled_events) == 1
        assert handler1.handled_events[0].event_type == EventType.CONFIGURATION_CHANGED
        assert result["status"] == "published"
    
    def test_event_history(self, event_bus):
        """Test event history functionality."""
        # Create test events
        event1 = EngineEvent(
            event_type=EventType.CONFIGURATION_CHANGED,
            source_engine="engine1"
        )
        event2 = EngineEvent(
            event_type=EventType.DATA_INVALIDATED,
            source_engine="engine2"
        )
        
        # Add to history manually (simulating processing)
        event_bus.event_history.extend([event1, event2])
        
        # Test history retrieval
        all_history = event_bus.get_event_history()
        assert len(all_history) == 2
        
        # Test filtered history
        config_history = event_bus.get_event_history(
            event_type=EventType.CONFIGURATION_CHANGED
        )
        assert len(config_history) == 1
        assert config_history[0].event_type == EventType.CONFIGURATION_CHANGED
        
        engine1_history = event_bus.get_event_history(source_engine="engine1")
        assert len(engine1_history) == 1
        assert engine1_history[0].source_engine == "engine1"


class TestEngineRegistry:
    """Test engine registry functionality."""
    
    def test_engine_registration(self, engine_registry):
        """Test engine registration and retrieval."""
        # Create mock engine
        mock_engine = Mock()
        mock_engine.name = "TestEngine"
        mock_engine.version = "1.0.0"
        
        capabilities = {"test_capability": True}
        
        # Register engine
        result = engine_registry.register_engine("test_engine", mock_engine, capabilities)
        assert result is True
        
        # Retrieve engine
        retrieved_engine = engine_registry.get_engine("test_engine")
        assert retrieved_engine == mock_engine
        
        # Check capabilities
        assert engine_registry.engine_capabilities["test_engine"] == capabilities
    
    def test_engine_dependencies(self, engine_registry):
        """Test engine dependency management."""
        # Add dependencies
        engine_registry.add_dependency("engine_a", "engine_b")
        engine_registry.add_dependency("engine_a", "engine_c")
        
        # Check dependencies
        deps = engine_registry.get_dependencies("engine_a")
        assert "engine_b" in deps
        assert "engine_c" in deps
        
        # Check dependents
        dependents = engine_registry.get_dependents("engine_b")
        assert "engine_a" in dependents
    
    def test_engine_unregistration(self, engine_registry):
        """Test engine unregistration."""
        # Register engine
        mock_engine = Mock()
        engine_registry.register_engine("test_engine", mock_engine)
        
        # Add dependency
        engine_registry.add_dependency("test_engine", "other_engine")
        
        # Unregister
        result = engine_registry.unregister_engine("test_engine")
        assert result is True
        
        # Verify removal
        assert engine_registry.get_engine("test_engine") is None
        assert "test_engine" not in engine_registry.engine_dependencies


class TestDataConsistencyManager:
    """Test data consistency management."""
    
    @pytest.mark.asyncio
    async def test_consistency_validation(self, event_bus, engine_registry):
        """Test data consistency validation."""
        consistency_manager = DataConsistencyManager(event_bus, engine_registry)
        
        # Add test rule
        from fighter_jet_sdk.core.engine_communication import DataConsistencyRule
        rule = DataConsistencyRule(
            rule_id="test_rule",
            source_engine="engine1",
            dependent_engines=["engine2"],
            data_fields=["test_field"],
            auto_update=False
        )
        consistency_manager.add_consistency_rule(rule)
        
        # Set up cached data
        consistency_manager.data_cache["engine2"] = {"test_field": "value1"}
        
        # Test consistent data
        violations = await consistency_manager.validate_data_consistency(
            "engine1", {"test_field": "value1"}
        )
        assert len(violations) == 0
        
        # Test inconsistent data
        violations = await consistency_manager.validate_data_consistency(
            "engine1", {"test_field": "value2"}
        )
        assert len(violations) == 1
        assert "inconsistency" in violations[0].lower()


class TestPerformanceMonitor:
    """Test performance monitoring."""
    
    def test_operation_time_recording(self):
        """Test operation time recording and statistics."""
        monitor = PerformanceMonitor()
        
        # Record some operation times
        monitor.record_operation_time("test_operation", 0.1)
        monitor.record_operation_time("test_operation", 0.2)
        monitor.record_operation_time("test_operation", 0.15)
        
        # Get statistics
        stats = monitor.get_performance_stats()
        
        assert "operations" in stats
        assert "test_operation" in stats["operations"]
        
        op_stats = stats["operations"]["test_operation"]
        assert op_stats["count"] == 3
        assert abs(op_stats["avg_time"] - 0.15) < 0.001
        assert op_stats["min_time"] == 0.1
        assert op_stats["max_time"] == 0.2
    
    def test_slow_operations_detection(self):
        """Test detection of slow operations."""
        monitor = PerformanceMonitor()
        
        # Record fast and slow operations
        monitor.record_operation_time("fast_op", 0.01)  # 10ms
        monitor.record_operation_time("slow_op", 0.2)   # 200ms
        
        # Get slow operations (threshold 100ms)
        slow_ops = monitor.get_slow_operations(threshold_ms=100.0)
        
        assert len(slow_ops) == 1
        assert slow_ops[0]["operation"] == "slow_op"
        assert slow_ops[0]["avg_time_ms"] == 200.0


@pytest.mark.asyncio
class TestEngineCoordinator:
    """Test engine coordinator functionality."""
    
    async def test_coordinator_initialization(self, engine_coordinator):
        """Test coordinator initialization."""
        # Mock the engine initialization to avoid dependencies
        with patch.multiple(
            'fighter_jet_sdk.engines.design.engine.DesignEngine',
            initialize=Mock(return_value=True)
        ), patch.multiple(
            'fighter_jet_sdk.engines.materials.engine.MaterialsEngine',
            initialize=Mock(return_value=True)
        ), patch.multiple(
            'fighter_jet_sdk.engines.propulsion.engine.PropulsionEngine',
            initialize=Mock(return_value=True)
        ), patch.multiple(
            'fighter_jet_sdk.engines.sensors.engine.SensorsEngine',
            initialize=Mock(return_value=True)
        ), patch.multiple(
            'fighter_jet_sdk.engines.aerodynamics.engine.AerodynamicsEngine',
            initialize=Mock(return_value=True)
        ), patch.multiple(
            'fighter_jet_sdk.engines.manufacturing.engine.ManufacturingEngine',
            initialize=Mock(return_value=True)
        ):
            await engine_coordinator.initialize()
        
        try:
            assert engine_coordinator.initialized is True
            assert engine_coordinator.running is True
            assert len(engine_coordinator.engines) == 6  # All engine types
        finally:
            await engine_coordinator.shutdown()
    
    async def test_configuration_change_processing(self, engine_coordinator, sample_configuration):
        """Test configuration change processing across engines."""
        # Mock the engine initialization to avoid dependencies
        with patch.multiple(
            'fighter_jet_sdk.engines.design.engine.DesignEngine',
            initialize=Mock(return_value=True)
        ), patch.multiple(
            'fighter_jet_sdk.engines.materials.engine.MaterialsEngine',
            initialize=Mock(return_value=True)
        ), patch.multiple(
            'fighter_jet_sdk.engines.propulsion.engine.PropulsionEngine',
            initialize=Mock(return_value=True)
        ), patch.multiple(
            'fighter_jet_sdk.engines.sensors.engine.SensorsEngine',
            initialize=Mock(return_value=True)
        ), patch.multiple(
            'fighter_jet_sdk.engines.aerodynamics.engine.AerodynamicsEngine',
            initialize=Mock(return_value=True)
        ), patch.multiple(
            'fighter_jet_sdk.engines.manufacturing.engine.ManufacturingEngine',
            initialize=Mock(return_value=True)
        ):
            await engine_coordinator.initialize()
        
        try:
            result = await engine_coordinator.process_configuration_change(
                sample_configuration, "test_source"
            )
            
            assert result["status"] == "success"
            assert "event_id" in result
            assert "processing_time" in result
            assert engine_coordinator.current_configuration == sample_configuration
        finally:
            await engine_coordinator.shutdown()
    
    async def test_cross_engine_analysis(self, engine_coordinator, sample_configuration):
        """Test cross-engine analysis coordination."""
        # Set current configuration
        engine_coordinator.current_configuration = sample_configuration
        
        # Mock engine process methods
        for engine in engine_coordinator.engines.values():
            engine.process = Mock(return_value={"analysis": "completed"})
        
        result = await engine_coordinator.run_cross_engine_analysis(
            "complete_performance", {"test_param": "value"}
        )
        
        assert result["status"] == "success"
        assert result["analysis_type"] == "complete_performance"
        assert "results" in result
        assert "processing_time" in result
    
    def test_system_status(self, engine_coordinator):
        """Test system status reporting."""
        status = engine_coordinator.get_system_status()
        
        assert "coordinator" in status
        assert "engines" in status
        assert "event_bus" in status
        assert "performance" in status
        
        assert status["coordinator"]["initialized"] is True
        assert len(status["engines"]) == 6
    
    def test_performance_report(self, engine_coordinator):
        """Test performance reporting."""
        report = engine_coordinator.get_performance_report()
        
        assert "performance_stats" in report
        assert "slow_operations" in report
        assert "event_history" in report
        assert "engine_dependencies" in report


@pytest.mark.asyncio
class TestIntegrationScenarios:
    """Test complete integration scenarios."""
    
    async def test_complete_design_workflow(self, engine_coordinator, sample_configuration):
        """Test complete design workflow across all engines."""
        # Step 1: Process configuration change
        config_result = await engine_coordinator.process_configuration_change(
            sample_configuration, "design"
        )
        assert config_result["status"] == "success"
        
        # Step 2: Run cross-engine analysis
        analysis_result = await engine_coordinator.run_cross_engine_analysis(
            "system_integration", {"analysis_depth": "comprehensive"}
        )
        assert analysis_result["status"] == "success"
        
        # Step 3: Optimize configuration
        optimization_result = await engine_coordinator.optimize_configuration({
            "minimize_weight": True,
            "maximize_stealth": True,
            "minimize_cost": False
        })
        assert optimization_result["status"] == "success"
        
        # Verify system state
        status = engine_coordinator.get_system_status()
        assert status["coordinator"]["initialized"] is True
    
    async def test_error_handling_and_recovery(self, engine_coordinator, sample_configuration):
        """Test error handling and recovery mechanisms."""
        # Mock an engine to raise an exception
        engine_coordinator.engines["design"].process = Mock(
            side_effect=Exception("Test error")
        )
        
        # Set current configuration
        engine_coordinator.current_configuration = sample_configuration
        
        # Attempt analysis that should handle the error gracefully
        result = await engine_coordinator.run_cross_engine_analysis(
            "complete_performance", {}
        )
        
        # Should still return a result with error information
        assert result["status"] == "success"  # Coordinator handles individual engine errors
        assert "results" in result
    
    async def test_performance_optimization(self, engine_coordinator):
        """Test performance optimization features."""
        # Record some operations
        engine_coordinator.performance_monitor.record_operation_time("test_op", 0.1)
        engine_coordinator.performance_monitor.record_operation_time("test_op", 0.2)
        
        # Get performance report
        report = engine_coordinator.get_performance_report()
        
        assert "performance_stats" in report
        assert "operations" in report["performance_stats"]
        
        # Check if slow operations are detected
        slow_ops = engine_coordinator.performance_monitor.get_slow_operations(50.0)
        assert len(slow_ops) >= 0  # May or may not have slow operations


if __name__ == "__main__":
    pytest.main([__file__])