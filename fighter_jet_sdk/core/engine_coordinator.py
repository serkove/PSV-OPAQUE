"""Engine coordinator for managing all SDK engines and their interactions."""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Type, Union
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from ..common.interfaces import BaseEngine
from ..common.data_models import AircraftConfiguration
from .engine_communication import (
    EventBus, EngineRegistry, DataConsistencyManager, PerformanceMonitor,
    EngineEvent, EventType, EventPriority, EventHandler, DataConsistencyRule
)

# Import all engine classes
from ..engines.design.engine import DesignEngine
from ..engines.materials.engine import MaterialsEngine
from ..engines.propulsion.engine import PropulsionEngine
from ..engines.sensors.engine import SensorsEngine
from ..engines.aerodynamics.engine import AerodynamicsEngine
from ..engines.manufacturing.engine import ManufacturingEngine


class EngineCoordinator:
    """Central coordinator for all SDK engines and their interactions."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the engine coordinator."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Core communication components
        self.event_bus = EventBus(max_workers=self.config.get('max_workers', 4))
        self.engine_registry = EngineRegistry()
        self.consistency_manager = DataConsistencyManager(self.event_bus, self.engine_registry)
        self.performance_monitor = PerformanceMonitor()
        
        # Engine instances
        self.engines: Dict[str, BaseEngine] = {}
        self.engine_handlers: Dict[str, 'EngineEventHandler'] = {}
        
        # Coordination state
        self.initialized = False
        self.running = False
        self.current_configuration: Optional[AircraftConfiguration] = None
        
        # Performance optimization
        self.executor = ThreadPoolExecutor(max_workers=self.config.get('max_workers', 4))
        
        # Setup default consistency rules
        self._setup_default_consistency_rules()
    
    async def initialize(self) -> bool:
        """Initialize all engines and the coordination system."""
        try:
            self.logger.info("Initializing Engine Coordinator")
            
            # Initialize engines
            engine_configs = self.config.get('engines', {})
            
            # Design Engine
            design_config = engine_configs.get('design', {})
            self.engines['design'] = DesignEngine(design_config)
            
            # Materials Engine
            materials_config = engine_configs.get('materials', {})
            self.engines['materials'] = MaterialsEngine(materials_config)
            
            # Propulsion Engine
            propulsion_config = engine_configs.get('propulsion', {})
            self.engines['propulsion'] = PropulsionEngine(propulsion_config)
            
            # Sensors Engine
            sensors_config = engine_configs.get('sensors', {})
            self.engines['sensors'] = SensorsEngine(sensors_config)
            
            # Aerodynamics Engine
            aerodynamics_config = engine_configs.get('aerodynamics', {})
            self.engines['aerodynamics'] = AerodynamicsEngine(aerodynamics_config)
            
            # Manufacturing Engine
            manufacturing_config = engine_configs.get('manufacturing', {})
            self.engines['manufacturing'] = ManufacturingEngine(manufacturing_config)
            
            # Initialize each engine
            for engine_id, engine in self.engines.items():
                if not engine.initialize():
                    raise RuntimeError(f"Failed to initialize {engine_id} engine")
                
                # Register with registry
                capabilities = self._get_engine_capabilities(engine_id, engine)
                self.engine_registry.register_engine(engine_id, engine, capabilities)
                
                # Create and register event handler
                handler = EngineEventHandler(engine_id, engine, self)
                self.engine_handlers[engine_id] = handler
                self.event_bus.subscribe(engine_id, handler)
                
                self.logger.info(f"Initialized {engine_id} engine")
            
            # Setup engine dependencies
            self._setup_engine_dependencies()
            
            # Start event bus processing
            asyncio.create_task(self.event_bus.start_processing())
            
            self.initialized = True
            self.running = True
            self.logger.info("Engine Coordinator initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Engine Coordinator: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown all engines and the coordination system."""
        try:
            self.logger.info("Shutting down Engine Coordinator")
            self.running = False
            
            # Stop event bus
            self.event_bus.stop_processing()
            
            # Unregister all engines
            for engine_id in list(self.engines.keys()):
                self.engine_registry.unregister_engine(engine_id)
                if engine_id in self.engine_handlers:
                    self.event_bus.unsubscribe(engine_id, self.engine_handlers[engine_id])
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            self.initialized = False
            self.logger.info("Engine Coordinator shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def get_engine(self, engine_id: str) -> Optional[BaseEngine]:
        """Get engine instance by ID."""
        return self.engines.get(engine_id)
    
    def get_all_engines(self) -> Dict[str, BaseEngine]:
        """Get all engine instances."""
        return self.engines.copy()
    
    async def process_configuration_change(self, config: AircraftConfiguration,
                                         source_engine: str = "coordinator") -> Dict[str, Any]:
        """Process a configuration change across all engines."""
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing configuration change from {source_engine}")
            
            # Update current configuration
            self.current_configuration = config
            
            # Validate configuration consistency
            consistency_violations = await self.consistency_manager.validate_data_consistency(
                source_engine, {"configuration": config.to_dict()}
            )
            
            if consistency_violations:
                self.logger.warning(f"Configuration consistency violations: {consistency_violations}")
            
            # Publish configuration change event
            event = EngineEvent(
                event_type=EventType.CONFIGURATION_CHANGED,
                source_engine=source_engine,
                data={
                    "configuration": config.to_dict(),
                    "change_timestamp": time.time()
                },
                priority=EventPriority.HIGH
            )
            
            result = await self.event_bus.publish(event)
            
            # Update data cache
            await self.consistency_manager.update_data_cache(
                source_engine, {"configuration": config.to_dict()}
            )
            
            # Record performance
            duration = time.time() - start_time
            self.performance_monitor.record_operation_time("configuration_change", duration)
            
            return {
                "status": "success",
                "event_id": result.get("event_id"),
                "consistency_violations": consistency_violations,
                "processing_time": duration
            }
            
        except Exception as e:
            self.logger.error(f"Error processing configuration change: {e}")
            return {
                "status": "error",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def run_cross_engine_analysis(self, analysis_type: str,
                                      parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run analysis that requires coordination between multiple engines."""
        start_time = time.time()
        
        try:
            self.logger.info(f"Running cross-engine analysis: {analysis_type}")
            
            if not self.current_configuration:
                raise ValueError("No current configuration available for analysis")
            
            # Determine which engines are needed for this analysis
            required_engines = self._get_required_engines_for_analysis(analysis_type)
            
            # Publish analysis start event
            start_event = EngineEvent(
                event_type=EventType.ANALYSIS_COMPLETED,
                source_engine="coordinator",
                target_engines=required_engines,
                data={
                    "analysis_type": analysis_type,
                    "parameters": parameters,
                    "configuration": self.current_configuration.to_dict()
                },
                priority=EventPriority.HIGH,
                requires_response=True
            )
            
            await self.event_bus.publish(start_event)
            
            # Coordinate the analysis across engines
            results = await self._coordinate_analysis(analysis_type, required_engines, parameters)
            
            # Publish analysis completion event
            completion_event = EngineEvent(
                event_type=EventType.ANALYSIS_COMPLETED,
                source_engine="coordinator",
                data={
                    "analysis_type": analysis_type,
                    "results": results,
                    "completion_timestamp": time.time()
                },
                priority=EventPriority.NORMAL
            )
            
            await self.event_bus.publish(completion_event)
            
            # Record performance
            duration = time.time() - start_time
            self.performance_monitor.record_operation_time(f"analysis_{analysis_type}", duration)
            
            return {
                "status": "success",
                "analysis_type": analysis_type,
                "results": results,
                "processing_time": duration,
                "engines_involved": required_engines
            }
            
        except Exception as e:
            self.logger.error(f"Error in cross-engine analysis: {e}")
            return {
                "status": "error",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def optimize_configuration(self, optimization_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize aircraft configuration using multiple engines."""
        start_time = time.time()
        
        try:
            self.logger.info("Starting configuration optimization")
            
            if not self.current_configuration:
                raise ValueError("No current configuration available for optimization")
            
            # Publish optimization start event
            start_event = EngineEvent(
                event_type=EventType.OPTIMIZATION_STARTED,
                source_engine="coordinator",
                data={
                    "criteria": optimization_criteria,
                    "initial_configuration": self.current_configuration.to_dict()
                },
                priority=EventPriority.HIGH
            )
            
            await self.event_bus.publish(start_event)
            
            # Run multi-objective optimization
            optimized_config = await self._run_multi_objective_optimization(optimization_criteria)
            
            # Validate optimized configuration
            validation_results = await self._validate_optimized_configuration(optimized_config)
            
            # Publish optimization completion event
            completion_event = EngineEvent(
                event_type=EventType.OPTIMIZATION_COMPLETED,
                source_engine="coordinator",
                data={
                    "optimized_configuration": optimized_config.to_dict(),
                    "validation_results": validation_results,
                    "completion_timestamp": time.time()
                },
                priority=EventPriority.NORMAL
            )
            
            await self.event_bus.publish(completion_event)
            
            # Record performance
            duration = time.time() - start_time
            self.performance_monitor.record_operation_time("configuration_optimization", duration)
            
            return {
                "status": "success",
                "optimized_configuration": optimized_config,
                "validation_results": validation_results,
                "processing_time": duration
            }
            
        except Exception as e:
            self.logger.error(f"Error in configuration optimization: {e}")
            return {
                "status": "error",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        engine_statuses = {}
        for engine_id, engine in self.engines.items():
            engine_statuses[engine_id] = engine.get_status()
        
        return {
            "coordinator": {
                "initialized": self.initialized,
                "running": self.running,
                "current_configuration": self.current_configuration.name if self.current_configuration else None
            },
            "engines": engine_statuses,
            "event_bus": {
                "running": self.event_bus.running,
                "queue_size": self.event_bus.event_queue.qsize() if hasattr(self.event_bus.event_queue, 'qsize') else 0
            },
            "performance": self.performance_monitor.get_performance_stats(),
            "consistency_rules": len(self.consistency_manager.consistency_rules)
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report."""
        return {
            "performance_stats": self.performance_monitor.get_performance_stats(),
            "slow_operations": self.performance_monitor.get_slow_operations(),
            "event_history": len(self.event_bus.event_history),
            "engine_dependencies": dict(self.engine_registry.engine_dependencies)
        }
    
    def _setup_default_consistency_rules(self) -> None:
        """Setup default data consistency rules between engines."""
        # Configuration consistency between design and all other engines
        config_rule = DataConsistencyRule(
            rule_id="configuration_consistency",
            source_engine="design",
            dependent_engines=["materials", "propulsion", "sensors", "aerodynamics", "manufacturing"],
            data_fields=["configuration"],
            auto_update=True,
            description="Ensure configuration changes propagate to all engines"
        )
        self.consistency_manager.add_consistency_rule(config_rule)
        
        # Materials consistency between materials and manufacturing
        materials_rule = DataConsistencyRule(
            rule_id="materials_manufacturing_consistency",
            source_engine="materials",
            dependent_engines=["manufacturing"],
            data_fields=["material_properties", "stealth_coatings"],
            auto_update=True,
            description="Ensure material changes update manufacturing processes"
        )
        self.consistency_manager.add_consistency_rule(materials_rule)
        
        # Propulsion-aerodynamics consistency
        propulsion_aero_rule = DataConsistencyRule(
            rule_id="propulsion_aerodynamics_consistency",
            source_engine="propulsion",
            dependent_engines=["aerodynamics"],
            data_fields=["engine_specifications", "intake_geometry"],
            auto_update=True,
            description="Ensure propulsion changes update aerodynamic analysis"
        )
        self.consistency_manager.add_consistency_rule(propulsion_aero_rule)
    
    def _setup_engine_dependencies(self) -> None:
        """Setup dependencies between engines."""
        # Materials engine depends on design for configuration
        self.engine_registry.add_dependency("materials", "design")
        
        # Propulsion engine depends on design and materials
        self.engine_registry.add_dependency("propulsion", "design")
        self.engine_registry.add_dependency("propulsion", "materials")
        
        # Sensors engine depends on design and materials
        self.engine_registry.add_dependency("sensors", "design")
        self.engine_registry.add_dependency("sensors", "materials")
        
        # Aerodynamics engine depends on design, materials, and propulsion
        self.engine_registry.add_dependency("aerodynamics", "design")
        self.engine_registry.add_dependency("aerodynamics", "materials")
        self.engine_registry.add_dependency("aerodynamics", "propulsion")
        
        # Manufacturing engine depends on all other engines
        self.engine_registry.add_dependency("manufacturing", "design")
        self.engine_registry.add_dependency("manufacturing", "materials")
        self.engine_registry.add_dependency("manufacturing", "propulsion")
        self.engine_registry.add_dependency("manufacturing", "sensors")
        self.engine_registry.add_dependency("manufacturing", "aerodynamics")
    
    def _get_engine_capabilities(self, engine_id: str, engine: BaseEngine) -> Dict[str, Any]:
        """Get capabilities for an engine."""
        capabilities = {
            "name": engine.name,
            "version": engine.version,
            "initialized": engine.initialized
        }
        
        # Add engine-specific capabilities
        if engine_id == "design":
            capabilities.update({
                "supports_modular_design": True,
                "supports_configuration_optimization": True,
                "supports_interface_validation": True
            })
        elif engine_id == "materials":
            capabilities.update({
                "supports_metamaterials": True,
                "supports_stealth_analysis": True,
                "supports_thermal_analysis": True
            })
        elif engine_id == "propulsion":
            capabilities.update({
                "supports_engine_performance": True,
                "supports_intake_design": True,
                "supports_thermal_management": True
            })
        elif engine_id == "sensors":
            capabilities.update({
                "supports_radar_modeling": True,
                "supports_laser_systems": True,
                "supports_plasma_systems": True
            })
        elif engine_id == "aerodynamics":
            capabilities.update({
                "supports_cfd_analysis": True,
                "supports_stability_analysis": True,
                "supports_stealth_shaping": True
            })
        elif engine_id == "manufacturing":
            capabilities.update({
                "supports_composite_manufacturing": True,
                "supports_assembly_planning": True,
                "supports_quality_control": True
            })
        
        return capabilities
    
    def _get_required_engines_for_analysis(self, analysis_type: str) -> List[str]:
        """Determine which engines are required for a specific analysis type."""
        analysis_engine_map = {
            "complete_performance": ["design", "materials", "propulsion", "aerodynamics"],
            "stealth_analysis": ["design", "materials", "aerodynamics"],
            "thermal_analysis": ["materials", "propulsion"],
            "manufacturing_feasibility": ["design", "materials", "manufacturing"],
            "mission_analysis": ["design", "propulsion", "aerodynamics", "sensors"],
            "cost_analysis": ["design", "materials", "manufacturing"],
            "system_integration": ["design", "materials", "propulsion", "sensors", "aerodynamics", "manufacturing"]
        }
        
        return analysis_engine_map.get(analysis_type, list(self.engines.keys()))
    
    async def _coordinate_analysis(self, analysis_type: str, required_engines: List[str],
                                 parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate analysis across multiple engines."""
        results = {}
        
        # Run analysis in dependency order
        for engine_id in required_engines:
            if engine_id in self.engines:
                engine = self.engines[engine_id]
                
                # Prepare engine-specific parameters
                engine_params = parameters.copy()
                engine_params["configuration"] = self.current_configuration
                engine_params["analysis_type"] = analysis_type
                
                # Run analysis
                try:
                    result = engine.process(engine_params)
                    results[engine_id] = result
                except Exception as e:
                    self.logger.error(f"Error in {engine_id} analysis: {e}")
                    results[engine_id] = {"error": str(e)}
        
        return results
    
    async def _run_multi_objective_optimization(self, criteria: Dict[str, Any]) -> AircraftConfiguration:
        """Run multi-objective optimization across engines."""
        # This is a simplified implementation - in practice, this would use
        # sophisticated optimization algorithms like NSGA-II or similar
        
        current_config = self.current_configuration
        if not current_config:
            raise ValueError("No configuration to optimize")
        
        # For now, return the current configuration
        # In a full implementation, this would iteratively modify the configuration
        # based on feedback from all engines
        return current_config
    
    async def _validate_optimized_configuration(self, config: AircraftConfiguration) -> Dict[str, Any]:
        """Validate an optimized configuration across all engines."""
        validation_results = {}
        
        for engine_id, engine in self.engines.items():
            try:
                # Each engine validates the configuration
                is_valid = engine.validate_input(config)
                validation_results[engine_id] = {
                    "valid": is_valid,
                    "issues": [] if is_valid else ["Configuration validation failed"]
                }
            except Exception as e:
                validation_results[engine_id] = {
                    "valid": False,
                    "issues": [str(e)]
                }
        
        return validation_results


class EngineEventHandler(EventHandler):
    """Event handler for individual engines."""
    
    def __init__(self, engine_id: str, engine: BaseEngine, coordinator: EngineCoordinator):
        self.engine_id = engine_id
        self.engine = engine
        self.coordinator = coordinator
        self.logger = logging.getLogger(f"{__name__}.{engine_id}")
    
    async def handle_event(self, event: EngineEvent) -> Optional[Dict[str, Any]]:
        """Handle incoming events for this engine."""
        start_time = time.time()
        
        try:
            self.logger.debug(f"Handling event {event.event_type.value} from {event.source_engine}")
            
            response = None
            
            if event.event_type == EventType.CONFIGURATION_CHANGED:
                response = await self._handle_configuration_change(event)
            elif event.event_type == EventType.DATA_INVALIDATED:
                response = await self._handle_data_invalidation(event)
            elif event.event_type == EventType.ANALYSIS_COMPLETED:
                response = await self._handle_analysis_request(event)
            elif event.event_type == EventType.OPTIMIZATION_STARTED:
                response = await self._handle_optimization_request(event)
            
            # Record processing time
            duration = time.time() - start_time
            self.coordinator.performance_monitor.record_event_processing_time(
                event.event_type.value, duration
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error handling event {event.event_type.value}: {e}")
            return {"error": str(e)}
    
    def get_supported_events(self) -> List[EventType]:
        """Return list of supported event types."""
        return [
            EventType.CONFIGURATION_CHANGED,
            EventType.DATA_INVALIDATED,
            EventType.ANALYSIS_COMPLETED,
            EventType.OPTIMIZATION_STARTED
        ]
    
    async def _handle_configuration_change(self, event: EngineEvent) -> Dict[str, Any]:
        """Handle configuration change event."""
        try:
            config_data = event.data.get("configuration", {})
            # In a full implementation, this would update the engine's internal state
            # based on the new configuration
            
            return {
                "status": "configuration_updated",
                "engine": self.engine_id,
                "timestamp": time.time()
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _handle_data_invalidation(self, event: EngineEvent) -> Dict[str, Any]:
        """Handle data invalidation event."""
        try:
            updated_fields = event.data.get("updated_fields", [])
            # In a full implementation, this would invalidate cached data
            # and trigger recalculation as needed
            
            return {
                "status": "data_invalidated",
                "engine": self.engine_id,
                "invalidated_fields": updated_fields,
                "timestamp": time.time()
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _handle_analysis_request(self, event: EngineEvent) -> Dict[str, Any]:
        """Handle analysis request event."""
        try:
            analysis_type = event.data.get("analysis_type")
            parameters = event.data.get("parameters", {})
            
            # Run engine-specific analysis
            result = self.engine.process({
                "operation": "analysis",
                "analysis_type": analysis_type,
                "parameters": parameters
            })
            
            return {
                "status": "analysis_completed",
                "engine": self.engine_id,
                "result": result,
                "timestamp": time.time()
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _handle_optimization_request(self, event: EngineEvent) -> Dict[str, Any]:
        """Handle optimization request event."""
        try:
            criteria = event.data.get("criteria", {})
            
            # Contribute to optimization based on engine capabilities
            contribution = self._calculate_optimization_contribution(criteria)
            
            return {
                "status": "optimization_contribution",
                "engine": self.engine_id,
                "contribution": contribution,
                "timestamp": time.time()
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_optimization_contribution(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate this engine's contribution to optimization."""
        # This would be implemented differently for each engine type
        # For now, return a placeholder
        return {
            "engine": self.engine_id,
            "optimization_factors": criteria,
            "recommendations": []
        }