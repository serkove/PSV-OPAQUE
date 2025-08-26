"""Inter-engine communication and event system for the Fighter Jet SDK."""

import asyncio
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Set, Union
from collections import defaultdict
import logging
import json
import uuid
from concurrent.futures import ThreadPoolExecutor

from ..common.interfaces import BaseEngine
from ..common.data_models import AircraftConfiguration


class EventType(Enum):
    """Types of events that can be communicated between engines."""
    CONFIGURATION_CHANGED = "configuration_changed"
    MODULE_ADDED = "module_added"
    MODULE_REMOVED = "module_removed"
    MATERIAL_UPDATED = "material_updated"
    ANALYSIS_COMPLETED = "analysis_completed"
    VALIDATION_FAILED = "validation_failed"
    OPTIMIZATION_STARTED = "optimization_started"
    OPTIMIZATION_COMPLETED = "optimization_completed"
    SIMULATION_STARTED = "simulation_started"
    SIMULATION_COMPLETED = "simulation_completed"
    ERROR_OCCURRED = "error_occurred"
    DATA_INVALIDATED = "data_invalidated"
    PERFORMANCE_UPDATED = "performance_updated"


class EventPriority(Enum):
    """Priority levels for events."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class EngineEvent:
    """Event data structure for inter-engine communication."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.CONFIGURATION_CHANGED
    source_engine: str = ""
    target_engines: Optional[List[str]] = None  # None means broadcast to all
    priority: EventPriority = EventPriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    requires_response: bool = False
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'source_engine': self.source_engine,
            'target_engines': self.target_engines,
            'priority': self.priority.value,
            'timestamp': self.timestamp,
            'data': self.data,
            'metadata': self.metadata,
            'requires_response': self.requires_response,
            'correlation_id': self.correlation_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EngineEvent':
        """Create event from dictionary."""
        return cls(
            event_id=data['event_id'],
            event_type=EventType(data['event_type']),
            source_engine=data['source_engine'],
            target_engines=data.get('target_engines'),
            priority=EventPriority(data['priority']),
            timestamp=data['timestamp'],
            data=data.get('data', {}),
            metadata=data.get('metadata', {}),
            requires_response=data.get('requires_response', False),
            correlation_id=data.get('correlation_id')
        )


class EventHandler(ABC):
    """Abstract base class for event handlers."""
    
    @abstractmethod
    async def handle_event(self, event: EngineEvent) -> Optional[Dict[str, Any]]:
        """Handle an incoming event. Return response data if required."""
        pass
    
    @abstractmethod
    def get_supported_events(self) -> List[EventType]:
        """Return list of event types this handler supports."""
        pass


@dataclass
class DataConsistencyRule:
    """Rule for maintaining data consistency across engines."""
    rule_id: str
    source_engine: str
    dependent_engines: List[str]
    data_fields: List[str]
    validation_function: Optional[Callable[[Dict[str, Any]], bool]] = None
    auto_update: bool = True
    description: str = ""


class EngineRegistry:
    """Registry for managing engine instances and their capabilities."""
    
    def __init__(self):
        self.engines: Dict[str, BaseEngine] = {}
        self.engine_capabilities: Dict[str, Dict[str, Any]] = {}
        self.engine_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.logger = logging.getLogger(__name__)
    
    def register_engine(self, engine_id: str, engine: BaseEngine, 
                       capabilities: Optional[Dict[str, Any]] = None) -> bool:
        """Register an engine with the communication system."""
        try:
            self.engines[engine_id] = engine
            self.engine_capabilities[engine_id] = capabilities or {}
            self.logger.info(f"Registered engine: {engine_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register engine {engine_id}: {e}")
            return False
    
    def unregister_engine(self, engine_id: str) -> bool:
        """Unregister an engine from the communication system."""
        try:
            if engine_id in self.engines:
                del self.engines[engine_id]
                del self.engine_capabilities[engine_id]
                # Remove from dependencies
                for deps in self.engine_dependencies.values():
                    deps.discard(engine_id)
                if engine_id in self.engine_dependencies:
                    del self.engine_dependencies[engine_id]
                self.logger.info(f"Unregistered engine: {engine_id}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to unregister engine {engine_id}: {e}")
            return False
    
    def get_engine(self, engine_id: str) -> Optional[BaseEngine]:
        """Get engine instance by ID."""
        return self.engines.get(engine_id)
    
    def get_all_engines(self) -> Dict[str, BaseEngine]:
        """Get all registered engines."""
        return self.engines.copy()
    
    def add_dependency(self, engine_id: str, depends_on: str) -> None:
        """Add dependency relationship between engines."""
        self.engine_dependencies[engine_id].add(depends_on)
    
    def get_dependencies(self, engine_id: str) -> Set[str]:
        """Get engines that the specified engine depends on."""
        return self.engine_dependencies.get(engine_id, set())
    
    def get_dependents(self, engine_id: str) -> Set[str]:
        """Get engines that depend on the specified engine."""
        dependents = set()
        for engine, deps in self.engine_dependencies.items():
            if engine_id in deps:
                dependents.add(engine)
        return dependents


class EventBus:
    """Central event bus for inter-engine communication."""
    
    def __init__(self, max_workers: int = 4):
        self.event_handlers: Dict[str, List[EventHandler]] = defaultdict(list)
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.logger = logging.getLogger(__name__)
        self.event_history: List[EngineEvent] = []
        self.max_history_size = 1000
        self._lock = threading.Lock()
    
    def subscribe(self, engine_id: str, handler: EventHandler) -> None:
        """Subscribe an engine's event handler to the bus."""
        with self._lock:
            self.event_handlers[engine_id].append(handler)
            supported_events = handler.get_supported_events()
            self.logger.info(f"Engine {engine_id} subscribed to events: {[e.value for e in supported_events]}")
    
    def unsubscribe(self, engine_id: str, handler: EventHandler) -> None:
        """Unsubscribe an engine's event handler from the bus."""
        with self._lock:
            if engine_id in self.event_handlers:
                self.event_handlers[engine_id].remove(handler)
                if not self.event_handlers[engine_id]:
                    del self.event_handlers[engine_id]
    
    async def publish(self, event: EngineEvent) -> Dict[str, Any]:
        """Publish an event to the bus."""
        # Add to history
        with self._lock:
            self.event_history.append(event)
            if len(self.event_history) > self.max_history_size:
                # Keep only the most recent half when limit is exceeded
                self.event_history = self.event_history[-self.max_history_size//2:]
        
        self.logger.debug(f"Publishing event {event.event_type.value} from {event.source_engine}")
        
        # Add to queue for processing
        await self.event_queue.put(event)
        
        # If response required, wait for it
        if event.requires_response:
            # This is a simplified implementation - in practice, you'd want
            # a more sophisticated response correlation mechanism
            return {"status": "published", "event_id": event.event_id}
        
        return {"status": "published", "event_id": event.event_id}
    
    async def start_processing(self) -> None:
        """Start processing events from the queue."""
        self.running = True
        self.logger.info("Event bus started processing")
        
        while self.running:
            try:
                # Wait for event with timeout
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                await self._process_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing event: {e}")
    
    def stop_processing(self) -> None:
        """Stop processing events."""
        self.running = False
        self.logger.info("Event bus stopped processing")
    
    async def _process_event(self, event: EngineEvent) -> None:
        """Process a single event."""
        target_engines = event.target_engines or list(self.event_handlers.keys())
        
        # Remove source engine from targets to avoid self-notification
        if event.source_engine in target_engines:
            target_engines.remove(event.source_engine)
        
        # Process event for each target engine
        tasks = []
        for engine_id in target_engines:
            if engine_id in self.event_handlers:
                for handler in self.event_handlers[engine_id]:
                    if event.event_type in handler.get_supported_events():
                        task = asyncio.create_task(
                            self._handle_event_safely(handler, event, engine_id)
                        )
                        tasks.append(task)
        
        # Wait for all handlers to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _handle_event_safely(self, handler: EventHandler, event: EngineEvent, engine_id: str) -> None:
        """Handle event with error protection."""
        try:
            response = await handler.handle_event(event)
            if response:
                self.logger.debug(f"Engine {engine_id} responded to event {event.event_id}")
        except Exception as e:
            self.logger.error(f"Error in event handler for engine {engine_id}: {e}")
            # Publish error event
            error_event = EngineEvent(
                event_type=EventType.ERROR_OCCURRED,
                source_engine="event_bus",
                data={
                    "original_event_id": event.event_id,
                    "error_message": str(e),
                    "failed_engine": engine_id
                },
                priority=EventPriority.HIGH
            )
            await self.event_queue.put(error_event)
    
    def get_event_history(self, event_type: Optional[EventType] = None,
                         source_engine: Optional[str] = None,
                         limit: int = 100) -> List[EngineEvent]:
        """Get event history with optional filtering."""
        with self._lock:
            events = self.event_history.copy()
        
        # Apply filters
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if source_engine:
            events = [e for e in events if e.source_engine == source_engine]
        
        # Return most recent events
        return events[-limit:]


class DataConsistencyManager:
    """Manages data consistency across engines."""
    
    def __init__(self, event_bus: EventBus, engine_registry: EngineRegistry):
        self.event_bus = event_bus
        self.engine_registry = engine_registry
        self.consistency_rules: Dict[str, DataConsistencyRule] = {}
        self.data_cache: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
    
    def add_consistency_rule(self, rule: DataConsistencyRule) -> None:
        """Add a data consistency rule."""
        self.consistency_rules[rule.rule_id] = rule
        self.logger.info(f"Added consistency rule: {rule.rule_id}")
    
    def remove_consistency_rule(self, rule_id: str) -> None:
        """Remove a data consistency rule."""
        if rule_id in self.consistency_rules:
            del self.consistency_rules[rule_id]
            self.logger.info(f"Removed consistency rule: {rule_id}")
    
    async def validate_data_consistency(self, engine_id: str, data: Dict[str, Any]) -> List[str]:
        """Validate data consistency across engines."""
        violations = []
        
        for rule in self.consistency_rules.values():
            if engine_id == rule.source_engine:
                # Check if data fields are consistent with dependent engines
                for dependent_engine in rule.dependent_engines:
                    dependent_data = self.data_cache.get(dependent_engine, {})
                    
                    for field in rule.data_fields:
                        if field in data and field in dependent_data:
                            if data[field] != dependent_data[field]:
                                violations.append(
                                    f"Data inconsistency in field '{field}' between "
                                    f"{engine_id} and {dependent_engine}"
                                )
                    
                    # Apply custom validation function if provided
                    if rule.validation_function:
                        combined_data = {**dependent_data, **data}
                        if not rule.validation_function(combined_data):
                            violations.append(
                                f"Custom validation failed for rule {rule.rule_id}"
                            )
        
        return violations
    
    async def update_data_cache(self, engine_id: str, data: Dict[str, Any]) -> None:
        """Update cached data for an engine."""
        self.data_cache[engine_id] = data
        
        # Check for auto-update rules
        for rule in self.consistency_rules.values():
            if engine_id == rule.source_engine and rule.auto_update:
                # Notify dependent engines of data change
                event = EngineEvent(
                    event_type=EventType.DATA_INVALIDATED,
                    source_engine=engine_id,
                    target_engines=rule.dependent_engines,
                    data={
                        "updated_fields": list(data.keys()),
                        "rule_id": rule.rule_id
                    },
                    priority=EventPriority.HIGH
                )
                await self.event_bus.publish(event)
    
    def get_cached_data(self, engine_id: str) -> Dict[str, Any]:
        """Get cached data for an engine."""
        return self.data_cache.get(engine_id, {})


class PerformanceMonitor:
    """Monitors performance of inter-engine operations."""
    
    def __init__(self):
        self.operation_times: Dict[str, List[float]] = defaultdict(list)
        self.event_processing_times: Dict[str, List[float]] = defaultdict(list)
        self.logger = logging.getLogger(__name__)
    
    def record_operation_time(self, operation: str, duration: float) -> None:
        """Record time taken for an operation."""
        self.operation_times[operation].append(duration)
        
        # Keep only recent measurements
        if len(self.operation_times[operation]) > 1000:
            self.operation_times[operation] = self.operation_times[operation][-500:]
    
    def record_event_processing_time(self, event_type: str, duration: float) -> None:
        """Record time taken to process an event."""
        self.event_processing_times[event_type].append(duration)
        
        # Keep only recent measurements
        if len(self.event_processing_times[event_type]) > 1000:
            self.event_processing_times[event_type] = self.event_processing_times[event_type][-500:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {
            "operations": {},
            "events": {}
        }
        
        # Operation statistics
        for operation, times in self.operation_times.items():
            if times:
                stats["operations"][operation] = {
                    "count": len(times),
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times)
                }
        
        # Event processing statistics
        for event_type, times in self.event_processing_times.items():
            if times:
                stats["events"][event_type] = {
                    "count": len(times),
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times)
                }
        
        return stats
    
    def get_slow_operations(self, threshold_ms: float = 100.0) -> List[Dict[str, Any]]:
        """Get operations that are slower than threshold."""
        slow_ops = []
        
        for operation, times in self.operation_times.items():
            if times:
                avg_time = sum(times) / len(times)
                if avg_time > threshold_ms / 1000.0:  # Convert to seconds
                    slow_ops.append({
                        "operation": operation,
                        "avg_time_ms": avg_time * 1000,
                        "count": len(times)
                    })
        
        return sorted(slow_ops, key=lambda x: x["avg_time_ms"], reverse=True)