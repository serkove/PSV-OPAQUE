"""Basic tests for inter-engine communication system."""

import pytest
import asyncio
import time
from unittest.mock import Mock

from fighter_jet_sdk.core.engine_communication import (
    EventBus, EngineRegistry, DataConsistencyManager, PerformanceMonitor,
    EngineEvent, EventType, EventPriority, EventHandler, DataConsistencyRule
)


class MockEventHandler(EventHandler):
    """Mock event handler for testing."""
    
    def __init__(self, engine_id: str):
        self.engine_id = engine_id
        self.handled_events = []
    
    async def handle_event(self, event: EngineEvent) -> dict:
        """Handle event and record it."""
        self.handled_events.append(event)
        return {"status": "handled", "engine": self.engine_id}
    
    def get_supported_events(self) -> list:
        """Return supported event types."""
        return [EventType.CONFIGURATION_CHANGED, EventType.DATA_INVALIDATED]


class TestEventBus:
    """Test event bus functionality."""
    
    @pytest.mark.asyncio
    async def test_event_publishing_and_handling(self):
        """Test basic event publishing and handling."""
        event_bus = EventBus(max_workers=2)
        
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
    
    def test_event_history(self):
        """Test event history functionality."""
        event_bus = EventBus()
        
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


class TestEngineRegistry:
    """Test engine registry functionality."""
    
    def test_engine_registration(self):
        """Test engine registration and retrieval."""
        registry = EngineRegistry()
        
        # Create mock engine
        mock_engine = Mock()
        mock_engine.name = "TestEngine"
        mock_engine.version = "1.0.0"
        
        capabilities = {"test_capability": True}
        
        # Register engine
        result = registry.register_engine("test_engine", mock_engine, capabilities)
        assert result is True
        
        # Retrieve engine
        retrieved_engine = registry.get_engine("test_engine")
        assert retrieved_engine == mock_engine
        
        # Check capabilities
        assert registry.engine_capabilities["test_engine"] == capabilities
    
    def test_engine_dependencies(self):
        """Test engine dependency management."""
        registry = EngineRegistry()
        
        # Add dependencies
        registry.add_dependency("engine_a", "engine_b")
        registry.add_dependency("engine_a", "engine_c")
        
        # Check dependencies
        deps = registry.get_dependencies("engine_a")
        assert "engine_b" in deps
        assert "engine_c" in deps
        
        # Check dependents
        dependents = registry.get_dependents("engine_b")
        assert "engine_a" in dependents


class TestDataConsistencyManager:
    """Test data consistency management."""
    
    @pytest.mark.asyncio
    async def test_consistency_validation(self):
        """Test data consistency validation."""
        event_bus = EventBus()
        registry = EngineRegistry()
        consistency_manager = DataConsistencyManager(event_bus, registry)
        
        # Add test rule
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


class TestEventTypes:
    """Test event type functionality."""
    
    def test_event_creation(self):
        """Test creating events with different types."""
        event = EngineEvent(
            event_type=EventType.CONFIGURATION_CHANGED,
            source_engine="test_engine",
            data={"config": "test"},
            priority=EventPriority.HIGH
        )
        
        assert event.event_type == EventType.CONFIGURATION_CHANGED
        assert event.source_engine == "test_engine"
        assert event.data["config"] == "test"
        assert event.priority == EventPriority.HIGH
        assert event.event_id is not None
        assert event.timestamp > 0
    
    def test_event_serialization(self):
        """Test event serialization and deserialization."""
        original_event = EngineEvent(
            event_type=EventType.ANALYSIS_COMPLETED,
            source_engine="materials_engine",
            target_engines=["design_engine", "manufacturing_engine"],
            data={"analysis_result": "success"},
            requires_response=True
        )
        
        # Serialize to dict
        event_dict = original_event.to_dict()
        
        # Deserialize back
        restored_event = EngineEvent.from_dict(event_dict)
        
        # Verify all fields match
        assert restored_event.event_type == original_event.event_type
        assert restored_event.source_engine == original_event.source_engine
        assert restored_event.target_engines == original_event.target_engines
        assert restored_event.data == original_event.data
        assert restored_event.requires_response == original_event.requires_response


class TestDataConsistencyRules:
    """Test data consistency rule functionality."""
    
    def test_rule_creation(self):
        """Test creating consistency rules."""
        rule = DataConsistencyRule(
            rule_id="test_rule",
            source_engine="design",
            dependent_engines=["materials", "propulsion"],
            data_fields=["configuration", "mass_properties"],
            auto_update=True,
            description="Test rule for configuration consistency"
        )
        
        assert rule.rule_id == "test_rule"
        assert rule.source_engine == "design"
        assert "materials" in rule.dependent_engines
        assert "propulsion" in rule.dependent_engines
        assert "configuration" in rule.data_fields
        assert rule.auto_update is True
    
    @pytest.mark.asyncio
    async def test_custom_validation_function(self):
        """Test custom validation function."""
        event_bus = EventBus()
        registry = EngineRegistry()
        consistency_manager = DataConsistencyManager(event_bus, registry)
        
        def custom_validator(data: dict) -> bool:
            """Custom validation: thrust-to-weight ratio must be > 1.0"""
            thrust = data.get("total_thrust", 0)
            weight = data.get("aircraft_weight", 1)
            return (thrust / weight) > 1.0 if weight > 0 else False
        
        rule = DataConsistencyRule(
            rule_id="twr_consistency",
            source_engine="propulsion",
            dependent_engines=["design"],
            data_fields=["total_thrust", "aircraft_weight"],
            validation_function=custom_validator,
            auto_update=False
        )
        consistency_manager.add_consistency_rule(rule)
        
        # Set up cached data
        consistency_manager.data_cache["design"] = {
            "aircraft_weight": 10000.0
        }
        
        # Test valid thrust-to-weight ratio
        violations = await consistency_manager.validate_data_consistency(
            "propulsion",
            {
                "total_thrust": 150000.0,  # T/W = 1.5
                "aircraft_weight": 10000.0
            }
        )
        assert len(violations) == 0
        
        # Test invalid thrust-to-weight ratio
        violations = await consistency_manager.validate_data_consistency(
            "propulsion",
            {
                "total_thrust": 8000.0,  # T/W = 0.8
                "aircraft_weight": 10000.0
            }
        )
        assert len(violations) == 1
        assert "twr_consistency" in violations[0]


if __name__ == "__main__":
    pytest.main([__file__])