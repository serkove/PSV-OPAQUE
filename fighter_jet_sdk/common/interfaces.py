"""Base interfaces for the Fighter Jet SDK."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from .data_models import AircraftConfiguration, Module, MaterialDefinition, SensorSystem


class BaseEngine(ABC):
    """Base class for all SDK engines."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the engine with configuration."""
        self.config = config or {}
        self.name = self.__class__.__name__
        self.version = "0.1.0"
        self.initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the engine. Returns True if successful."""
        pass
    
    @abstractmethod
    def validate_input(self, data: Any) -> bool:
        """Validate input data. Returns True if valid."""
        pass
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process input data and return results."""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status information."""
        return {
            'name': self.name,
            'version': self.version,
            'initialized': self.initialized,
            'config': self.config
        }


class ModuleValidator(ABC):
    """Base class for module validation."""
    
    @abstractmethod
    def validate_compatibility(self, module1: Module, module2: Module) -> bool:
        """Check if two modules are compatible."""
        pass
    
    @abstractmethod
    def validate_configuration(self, config: AircraftConfiguration) -> List[str]:
        """Validate aircraft configuration. Returns list of validation errors."""
        pass
    
    @abstractmethod
    def get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for this validator."""
        pass


class DataProcessor(ABC):
    """Base class for data processing operations."""
    
    @abstractmethod
    def serialize(self, data: Any) -> Dict[str, Any]:
        """Serialize data to dictionary format."""
        pass
    
    @abstractmethod
    def deserialize(self, data: Dict[str, Any]) -> Any:
        """Deserialize data from dictionary format."""
        pass
    
    @abstractmethod
    def validate_schema(self, data: Dict[str, Any]) -> bool:
        """Validate data against expected schema."""
        pass


class SimulationEngine(ABC):
    """Base class for simulation engines."""
    
    @abstractmethod
    def setup_simulation(self, config: AircraftConfiguration) -> bool:
        """Set up simulation with aircraft configuration."""
        pass
    
    @abstractmethod
    def run_simulation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run simulation with given parameters."""
        pass
    
    @abstractmethod
    def get_results(self) -> Dict[str, Any]:
        """Get simulation results."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up simulation resources."""
        pass


class OptimizationEngine(ABC):
    """Base class for optimization engines."""
    
    @abstractmethod
    def define_objective(self, objective_function: callable) -> None:
        """Define optimization objective function."""
        pass
    
    @abstractmethod
    def add_constraint(self, constraint_function: callable) -> None:
        """Add optimization constraint."""
        pass
    
    @abstractmethod
    def optimize(self, initial_guess: Dict[str, float]) -> Dict[str, Any]:
        """Run optimization and return results."""
        pass
    
    @abstractmethod
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization iteration history."""
        pass


class MaterialsInterface(ABC):
    """Interface for materials-related operations."""
    
    @abstractmethod
    def get_material_properties(self, material_id: str) -> Optional[MaterialDefinition]:
        """Get material properties by ID."""
        pass
    
    @abstractmethod
    def calculate_electromagnetic_response(self, material: MaterialDefinition, 
                                        frequency: float) -> Dict[str, complex]:
        """Calculate electromagnetic response at given frequency."""
        pass
    
    @abstractmethod
    def calculate_thermal_response(self, material: MaterialDefinition, 
                                 temperature: float) -> Dict[str, float]:
        """Calculate thermal response at given temperature."""
        pass


class SensorsInterface(ABC):
    """Interface for sensor-related operations."""
    
    @abstractmethod
    def get_sensor_capabilities(self, sensor_id: str) -> Optional[SensorSystem]:
        """Get sensor capabilities by ID."""
        pass
    
    @abstractmethod
    def calculate_detection_probability(self, sensor: SensorSystem, 
                                     target_signature: Dict[str, float],
                                     range_km: float) -> float:
        """Calculate detection probability for given target and range."""
        pass
    
    @abstractmethod
    def calculate_power_requirements(self, sensor: SensorSystem,
                                   operating_mode: str) -> Dict[str, float]:
        """Calculate power requirements for sensor in given operating mode."""
        pass


class AerodynamicsInterface(ABC):
    """Interface for aerodynamics-related operations."""
    
    @abstractmethod
    def calculate_lift_drag(self, config: AircraftConfiguration,
                          flight_conditions: Dict[str, float]) -> Dict[str, float]:
        """Calculate lift and drag coefficients."""
        pass
    
    @abstractmethod
    def calculate_stability_derivatives(self, config: AircraftConfiguration) -> Dict[str, float]:
        """Calculate stability and control derivatives."""
        pass
    
    @abstractmethod
    def calculate_performance_envelope(self, config: AircraftConfiguration) -> Dict[str, Any]:
        """Calculate aircraft performance envelope."""
        pass


class ManufacturingInterface(ABC):
    """Interface for manufacturing-related operations."""
    
    @abstractmethod
    def calculate_manufacturing_cost(self, config: AircraftConfiguration) -> Dict[str, float]:
        """Calculate manufacturing cost breakdown."""
        pass
    
    @abstractmethod
    def generate_assembly_sequence(self, config: AircraftConfiguration) -> List[Dict[str, Any]]:
        """Generate optimized assembly sequence."""
        pass
    
    @abstractmethod
    def validate_manufacturability(self, config: AircraftConfiguration) -> List[str]:
        """Validate configuration manufacturability. Returns list of issues."""
        pass


class AnalysisEngine(BaseEngine):
    """Base class for analysis engines."""
    
    @abstractmethod
    def analyze(self, configuration: AircraftConfiguration, **kwargs) -> Any:
        """Perform analysis on aircraft configuration."""
        pass
    
    def initialize(self) -> bool:
        """Initialize the analysis engine."""
        self.initialized = True
        return True
    
    def validate_input(self, data: Any) -> bool:
        """Validate input data for analysis."""
        return isinstance(data, AircraftConfiguration)
    
    def process(self, data: Any) -> Any:
        """Process data using the analyze method."""
        return self.analyze(data)


class SimulationComponent(ABC):
    """Base class for simulation components."""
    
    def __init__(self):
        """Initialize the simulation component."""
        self.simulation_time = 0.0
        self.initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the simulation component."""
        pass
    
    @abstractmethod
    def update_simulation(self, dt: float) -> Dict[str, Any]:
        """Update simulation state with time step dt."""
        pass
    
    def get_simulation_time(self) -> float:
        """Get current simulation time."""
        return self.simulation_time
    
    def reset_simulation(self) -> None:
        """Reset simulation to initial state."""
        self.simulation_time = 0.0