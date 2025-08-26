"""Tests for data consistency validation across engine boundaries."""

import pytest
import asyncio
from unittest.mock import Mock

from fighter_jet_sdk.core.engine_communication import (
    DataConsistencyManager, DataConsistencyRule, EventBus, EngineRegistry,
    EngineEvent, EventType
)
from fighter_jet_sdk.common.data_models import AircraftConfiguration, BasePlatform, Module
from fighter_jet_sdk.common.enums import ModuleType


@pytest.fixture
def event_bus():
    """Create event bus for testing."""
    return EventBus(max_workers=2)


@pytest.fixture
def engine_registry():
    """Create engine registry for testing."""
    registry = EngineRegistry()
    
    # Register mock engines
    for engine_id in ["design", "materials", "propulsion", "aerodynamics"]:
        mock_engine = Mock()
        mock_engine.name = f"{engine_id.title()}Engine"
        registry.register_engine(engine_id, mock_engine)
    
    return registry


@pytest.fixture
def consistency_manager(event_bus, engine_registry):
    """Create data consistency manager for testing."""
    return DataConsistencyManager(event_bus, engine_registry)


@pytest.fixture
def sample_configuration():
    """Create sample configuration for testing."""
    platform = BasePlatform(
        name="Test Platform",
        base_mass=5000.0,
        dimensions=(15.0, 10.0, 4.0)
    )
    
    config = AircraftConfiguration(
        name="Test Aircraft",
        description="Test configuration",
        base_platform=platform
    )
    
    return config


class TestDataConsistencyRules:
    """Test data consistency rule management."""
    
    def test_add_consistency_rule(self, consistency_manager):
        """Test adding consistency rules."""
        rule = DataConsistencyRule(
            rule_id="test_rule",
            source_engine="design",
            dependent_engines=["materials", "propulsion"],
            data_fields=["configuration", "mass_properties"],
            auto_update=True,
            description="Test rule for configuration consistency"
        )
        
        consistency_manager.add_consistency_rule(rule)
        
        assert "test_rule" in consistency_manager.consistency_rules
        assert consistency_manager.consistency_rules["test_rule"] == rule
    
    def test_remove_consistency_rule(self, consistency_manager):
        """Test removing consistency rules."""
        rule = DataConsistencyRule(
            rule_id="test_rule",
            source_engine="design",
            dependent_engines=["materials"],
            data_fields=["configuration"]
        )
        
        consistency_manager.add_consistency_rule(rule)
        assert "test_rule" in consistency_manager.consistency_rules
        
        consistency_manager.remove_consistency_rule("test_rule")
        assert "test_rule" not in consistency_manager.consistency_rules


class TestDataConsistencyValidation:
    """Test data consistency validation logic."""
    
    @pytest.mark.asyncio
    async def test_consistent_data_validation(self, consistency_manager):
        """Test validation with consistent data."""
        # Add consistency rule
        rule = DataConsistencyRule(
            rule_id="config_consistency",
            source_engine="design",
            dependent_engines=["materials", "propulsion"],
            data_fields=["aircraft_mass", "wing_area"],
            auto_update=False
        )
        consistency_manager.add_consistency_rule(rule)
        
        # Set up consistent cached data
        consistency_manager.data_cache["materials"] = {
            "aircraft_mass": 10000.0,
            "wing_area": 50.0
        }
        consistency_manager.data_cache["propulsion"] = {
            "aircraft_mass": 10000.0,
            "wing_area": 50.0
        }
        
        # Validate consistent data
        violations = await consistency_manager.validate_data_consistency(
            "design",
            {
                "aircraft_mass": 10000.0,
                "wing_area": 50.0,
                "other_field": "value"
            }
        )
        
        assert len(violations) == 0
    
    @pytest.mark.asyncio
    async def test_inconsistent_data_validation(self, consistency_manager):
        """Test validation with inconsistent data."""
        # Add consistency rule
        rule = DataConsistencyRule(
            rule_id="config_consistency",
            source_engine="design",
            dependent_engines=["materials"],
            data_fields=["aircraft_mass"],
            auto_update=False
        )
        consistency_manager.add_consistency_rule(rule)
        
        # Set up inconsistent cached data
        consistency_manager.data_cache["materials"] = {
            "aircraft_mass": 10000.0
        }
        
        # Validate inconsistent data
        violations = await consistency_manager.validate_data_consistency(
            "design",
            {"aircraft_mass": 12000.0}  # Different value
        )
        
        assert len(violations) == 1
        assert "aircraft_mass" in violations[0]
        assert "design" in violations[0]
        assert "materials" in violations[0]
    
    @pytest.mark.asyncio
    async def test_custom_validation_function(self, consistency_manager):
        """Test custom validation function."""
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


class TestDataCacheManagement:
    """Test data cache management functionality."""
    
    @pytest.mark.asyncio
    async def test_data_cache_update(self, consistency_manager):
        """Test updating data cache."""
        test_data = {
            "configuration": {"name": "Test Config"},
            "performance": {"max_speed": 2.0}
        }
        
        await consistency_manager.update_data_cache("design", test_data)
        
        cached_data = consistency_manager.get_cached_data("design")
        assert cached_data == test_data
    
    @pytest.mark.asyncio
    async def test_auto_update_notification(self, consistency_manager, event_bus):
        """Test automatic update notifications."""
        # Add auto-update rule
        rule = DataConsistencyRule(
            rule_id="auto_update_rule",
            source_engine="design",
            dependent_engines=["materials", "propulsion"],
            data_fields=["configuration"],
            auto_update=True
        )
        consistency_manager.add_consistency_rule(rule)
        
        # Mock event bus publish method to capture events
        published_events = []
        original_publish = event_bus.publish
        
        async def mock_publish(event):
            published_events.append(event)
            return await original_publish(event)
        
        event_bus.publish = mock_publish
        
        # Update data cache
        test_data = {"configuration": {"updated": True}}
        await consistency_manager.update_data_cache("design", test_data)
        
        # Verify event was published
        assert len(published_events) == 1
        event = published_events[0]
        assert event.event_type == EventType.DATA_INVALIDATED
        assert event.source_engine == "design"
        assert set(event.target_engines) == {"materials", "propulsion"}
        assert event.data["rule_id"] == "auto_update_rule"


class TestComplexConsistencyScenarios:
    """Test complex consistency scenarios."""
    
    @pytest.mark.asyncio
    async def test_multi_engine_consistency_chain(self, consistency_manager):
        """Test consistency validation across multiple engines in a chain."""
        # Set up consistency chain: design -> materials -> manufacturing
        rule1 = DataConsistencyRule(
            rule_id="design_materials",
            source_engine="design",
            dependent_engines=["materials"],
            data_fields=["material_specs"],
            auto_update=False
        )
        
        rule2 = DataConsistencyRule(
            rule_id="materials_manufacturing",
            source_engine="materials",
            dependent_engines=["manufacturing"],
            data_fields=["material_specs", "processing_temp"],
            auto_update=False
        )
        
        consistency_manager.add_consistency_rule(rule1)
        consistency_manager.add_consistency_rule(rule2)
        
        # Set up cached data
        consistency_manager.data_cache["materials"] = {
            "material_specs": {"type": "carbon_fiber", "grade": "T800"},
            "processing_temp": 180.0
        }
        consistency_manager.data_cache["manufacturing"] = {
            "material_specs": {"type": "carbon_fiber", "grade": "T800"},
            "processing_temp": 180.0
        }
        
        # Test consistency at design level
        violations = await consistency_manager.validate_data_consistency(
            "design",
            {"material_specs": {"type": "carbon_fiber", "grade": "T800"}}
        )
        assert len(violations) == 0
        
        # Test inconsistency
        violations = await consistency_manager.validate_data_consistency(
            "design",
            {"material_specs": {"type": "titanium", "grade": "Ti6Al4V"}}
        )
        assert len(violations) == 1
    
    @pytest.mark.asyncio
    async def test_circular_dependency_handling(self, consistency_manager):
        """Test handling of circular dependencies."""
        # Create circular dependency: A -> B -> A
        rule1 = DataConsistencyRule(
            rule_id="a_to_b",
            source_engine="engine_a",
            dependent_engines=["engine_b"],
            data_fields=["shared_field"],
            auto_update=False
        )
        
        rule2 = DataConsistencyRule(
            rule_id="b_to_a",
            source_engine="engine_b",
            dependent_engines=["engine_a"],
            data_fields=["shared_field"],
            auto_update=False
        )
        
        consistency_manager.add_consistency_rule(rule1)
        consistency_manager.add_consistency_rule(rule2)
        
        # Set up cached data
        consistency_manager.data_cache["engine_a"] = {"shared_field": "value1"}
        consistency_manager.data_cache["engine_b"] = {"shared_field": "value1"}
        
        # Validate - should handle circular dependency gracefully
        violations = await consistency_manager.validate_data_consistency(
            "engine_a",
            {"shared_field": "value1"}
        )
        assert len(violations) == 0
    
    @pytest.mark.asyncio
    async def test_partial_field_consistency(self, consistency_manager):
        """Test consistency validation with partial field overlap."""
        rule = DataConsistencyRule(
            rule_id="partial_consistency",
            source_engine="design",
            dependent_engines=["aerodynamics"],
            data_fields=["wing_area", "aspect_ratio"],
            auto_update=False
        )
        consistency_manager.add_consistency_rule(rule)
        
        # Set up cached data with only one field
        consistency_manager.data_cache["aerodynamics"] = {
            "wing_area": 50.0,
            "other_field": "value"
        }
        
        # Validate with both fields - should only check the overlapping field
        violations = await consistency_manager.validate_data_consistency(
            "design",
            {
                "wing_area": 50.0,  # Consistent
                "aspect_ratio": 8.0,  # Not in cache, should not cause violation
                "extra_field": "extra"
            }
        )
        assert len(violations) == 0
        
        # Test inconsistency in the overlapping field
        violations = await consistency_manager.validate_data_consistency(
            "design",
            {
                "wing_area": 60.0,  # Inconsistent
                "aspect_ratio": 8.0
            }
        )
        assert len(violations) == 1


class TestPerformanceAndScalability:
    """Test performance and scalability of consistency validation."""
    
    @pytest.mark.asyncio
    async def test_large_data_validation_performance(self, consistency_manager):
        """Test performance with large data sets."""
        import time
        
        # Create rule with many fields
        large_field_list = [f"field_{i}" for i in range(100)]
        rule = DataConsistencyRule(
            rule_id="large_data_rule",
            source_engine="source",
            dependent_engines=["target"],
            data_fields=large_field_list,
            auto_update=False
        )
        consistency_manager.add_consistency_rule(rule)
        
        # Set up large cached data
        large_cached_data = {f"field_{i}": f"value_{i}" for i in range(100)}
        consistency_manager.data_cache["target"] = large_cached_data
        
        # Validate large consistent data
        start_time = time.time()
        violations = await consistency_manager.validate_data_consistency(
            "source", large_cached_data.copy()
        )
        end_time = time.time()
        
        assert len(violations) == 0
        assert (end_time - start_time) < 1.0  # Should complete within 1 second
    
    @pytest.mark.asyncio
    async def test_many_rules_performance(self, consistency_manager):
        """Test performance with many consistency rules."""
        import time
        
        # Create many rules
        for i in range(50):
            rule = DataConsistencyRule(
                rule_id=f"rule_{i}",
                source_engine="source",
                dependent_engines=[f"target_{i}"],
                data_fields=[f"field_{i}"],
                auto_update=False
            )
            consistency_manager.add_consistency_rule(rule)
            
            # Set up cached data
            consistency_manager.data_cache[f"target_{i}"] = {f"field_{i}": f"value_{i}"}
        
        # Validate data against all rules
        test_data = {f"field_{i}": f"value_{i}" for i in range(50)}
        
        start_time = time.time()
        violations = await consistency_manager.validate_data_consistency("source", test_data)
        end_time = time.time()
        
        assert len(violations) == 0
        assert (end_time - start_time) < 2.0  # Should complete within 2 seconds


if __name__ == "__main__":
    pytest.main([__file__])