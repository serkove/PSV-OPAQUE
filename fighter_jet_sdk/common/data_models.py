"""Core data models for the Fighter Jet SDK."""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
import uuid
import json
import yaml
import numpy as np
from pathlib import Path

from .enums import ModuleType, MaterialType, SensorType, EngineType, FlightRegime, PlasmaRegime, ThermalProtectionType, ExtremePropulsionType


@dataclass
class PhysicalProperties:
    """Physical properties of components."""
    mass: float  # kg
    center_of_gravity: tuple[float, float, float]  # x, y, z coordinates
    moments_of_inertia: tuple[float, float, float]  # Ixx, Iyy, Izz
    dimensions: tuple[float, float, float]  # length, width, height


@dataclass
class ElectricalInterface:
    """Electrical interface specification."""
    interface_id: str
    voltage: float  # V
    current_capacity: float  # A
    power_consumption: float  # W
    data_rate: Optional[float] = None  # Mbps
    protocol: Optional[str] = None


@dataclass
class MechanicalInterface:
    """Mechanical interface specification."""
    interface_id: str
    attachment_type: str
    load_capacity: tuple[float, float, float]  # Fx, Fy, Fz (N)
    moment_capacity: tuple[float, float, float]  # Mx, My, Mz (N⋅m)
    position: tuple[float, float, float]  # x, y, z coordinates


@dataclass
class ModuleInterface:
    """Interface between modules."""
    interface_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    electrical: Optional[ElectricalInterface] = None
    mechanical: Optional[MechanicalInterface] = None
    thermal: Optional[Dict[str, float]] = None
    data: Optional[Dict[str, Any]] = None

    def validate_interface(self) -> List[str]:
        """Validate module interface configuration."""
        errors = []
        
        # Must have at least one interface type
        if not any([self.electrical, self.mechanical, self.thermal, self.data]):
            errors.append("Interface must have at least one connection type")
        
        # Validate electrical interface
        if self.electrical:
            if self.electrical.voltage <= 0:
                errors.append("Electrical interface voltage must be positive")
            if self.electrical.current_capacity <= 0:
                errors.append("Electrical interface current capacity must be positive")
        
        # Validate thermal interface
        if self.thermal:
            for key, value in self.thermal.items():
                if not isinstance(value, (int, float)):
                    errors.append(f"Thermal property '{key}' must be numeric")
        
        return errors

    def is_compatible_with(self, other_interface: 'ModuleInterface') -> bool:
        """Check if this interface is compatible with another interface."""
        # Check electrical compatibility
        if self.electrical and other_interface.electrical:
            if abs(self.electrical.voltage - other_interface.electrical.voltage) > 0.1:
                return False
            if self.electrical.protocol != other_interface.electrical.protocol:
                return False
        
        # Check mechanical compatibility
        if self.mechanical and other_interface.mechanical:
            if self.mechanical.attachment_type != other_interface.mechanical.attachment_type:
                return False
        
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert interface to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModuleInterface':
        """Create interface from dictionary."""
        # Handle nested objects
        if 'electrical' in data and data['electrical']:
            data['electrical'] = ElectricalInterface(**data['electrical'])
        
        if 'mechanical' in data and data['mechanical']:
            data['mechanical'] = MechanicalInterface(**data['mechanical'])
        
        return cls(**data)


@dataclass
class Module:
    """Modular aircraft component."""
    module_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    module_type: ModuleType = ModuleType.STRUCTURAL
    name: str = ""
    description: str = ""
    physical_properties: Optional[PhysicalProperties] = None
    electrical_interfaces: List[ElectricalInterface] = field(default_factory=list)
    mechanical_interfaces: List[MechanicalInterface] = field(default_factory=list)
    performance_characteristics: Dict[str, float] = field(default_factory=dict)
    compatibility_requirements: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def validate_module(self) -> List[str]:
        """Validate module configuration and return list of errors."""
        errors = []
        
        # Basic validation
        if not self.name.strip():
            errors.append("Module must have a name")
        
        # Physical properties validation
        if self.physical_properties:
            phys_errors = self._validate_physical_properties()
            errors.extend(phys_errors)
        
        # Interface validation
        interface_errors = self._validate_interfaces()
        errors.extend(interface_errors)
        
        # Performance characteristics validation
        perf_errors = self._validate_performance_characteristics()
        errors.extend(perf_errors)
        
        return errors

    def _validate_physical_properties(self) -> List[str]:
        """Validate physical properties."""
        errors = []
        
        if not self.physical_properties:
            return errors
        
        # Mass must be positive
        if self.physical_properties.mass <= 0:
            errors.append("Module mass must be positive")
        
        # Moments of inertia must be positive
        for i, moment in enumerate(self.physical_properties.moments_of_inertia):
            if moment <= 0:
                errors.append(f"Moment of inertia {i} must be positive")
        
        # Dimensions must be positive
        for i, dim in enumerate(self.physical_properties.dimensions):
            if dim <= 0:
                errors.append(f"Dimension {i} must be positive")
        
        return errors

    def _validate_interfaces(self) -> List[str]:
        """Validate electrical and mechanical interfaces."""
        errors = []
        
        # Validate electrical interfaces
        for i, interface in enumerate(self.electrical_interfaces):
            if interface.voltage <= 0:
                errors.append(f"Electrical interface {i} voltage must be positive")
            if interface.current_capacity <= 0:
                errors.append(f"Electrical interface {i} current capacity must be positive")
            if interface.power_consumption < 0:
                errors.append(f"Electrical interface {i} power consumption cannot be negative")
        
        # Validate mechanical interfaces
        for i, interface in enumerate(self.mechanical_interfaces):
            if not interface.attachment_type:
                errors.append(f"Mechanical interface {i} must have attachment type")
        
        return errors

    def _validate_performance_characteristics(self) -> List[str]:
        """Validate performance characteristics."""
        errors = []
        
        # Check for reasonable values in performance characteristics
        for key, value in self.performance_characteristics.items():
            if not isinstance(value, (int, float)):
                errors.append(f"Performance characteristic '{key}' must be numeric")
        
        return errors

    def is_compatible_with(self, other_module: 'Module') -> bool:
        """Check if this module is compatible with another module."""
        # Check compatibility requirements
        for req in self.compatibility_requirements:
            if req.startswith("incompatible_with:"):
                incompatible_type = req.split(":")[1]
                if other_module.module_type.name == incompatible_type:
                    return False
        
        # Check other module's requirements
        for req in other_module.compatibility_requirements:
            if req.startswith("incompatible_with:"):
                incompatible_type = req.split(":")[1]
                if self.module_type.name == incompatible_type:
                    return False
        
        return True

    def calculate_total_power_consumption(self) -> float:
        """Calculate total power consumption of all electrical interfaces."""
        return sum(interface.power_consumption for interface in self.electrical_interfaces)

    def to_dict(self) -> Dict[str, Any]:
        """Convert module to dictionary for serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['module_type'] = self.module_type.name
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Module':
        """Create module from dictionary."""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        if 'module_type' in data and isinstance(data['module_type'], str):
            data['module_type'] = ModuleType[data['module_type']]
        
        # Handle nested objects
        if 'physical_properties' in data and data['physical_properties']:
            data['physical_properties'] = PhysicalProperties(**data['physical_properties'])
        
        if 'electrical_interfaces' in data:
            data['electrical_interfaces'] = [ElectricalInterface(**ei) for ei in data['electrical_interfaces']]
        
        if 'mechanical_interfaces' in data:
            data['mechanical_interfaces'] = [MechanicalInterface(**mi) for mi in data['mechanical_interfaces']]
        
        return cls(**data)


@dataclass
class BasePlatform:
    """Base aircraft platform configuration."""
    platform_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    base_mass: float = 0.0  # kg
    attachment_points: List[MechanicalInterface] = field(default_factory=list)
    power_generation_capacity: float = 0.0  # W
    fuel_capacity: float = 0.0  # kg
    structural_limits: Dict[str, float] = field(default_factory=dict)


@dataclass
class PerformanceEnvelope:
    """Aircraft performance characteristics."""
    max_speed: Dict[FlightRegime, float] = field(default_factory=dict)  # Mach number
    service_ceiling: float = 0.0  # m
    range: float = 0.0  # km
    thrust_to_weight_ratio: float = 0.0
    wing_loading: float = 0.0  # N/m²
    radar_cross_section: Dict[str, float] = field(default_factory=dict)  # m² by frequency


@dataclass
class MissionRequirements:
    """Mission-specific requirements."""
    mission_type: str = ""
    duration: float = 0.0  # hours
    range_requirement: float = 0.0  # km
    payload_requirement: float = 0.0  # kg
    altitude_requirement: tuple[float, float] = (0.0, 0.0)  # min, max altitude (m)
    speed_requirement: tuple[float, float] = (0.0, 0.0)  # min, max speed (Mach)
    stealth_requirements: Dict[str, float] = field(default_factory=dict)


@dataclass
class AircraftConfiguration:
    """Complete aircraft configuration."""
    config_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    base_platform: Optional[BasePlatform] = None
    modules: List[Module] = field(default_factory=list)
    interfaces: List[ModuleInterface] = field(default_factory=list)
    performance_envelope: Optional[PerformanceEnvelope] = None
    mission_requirements: Optional[MissionRequirements] = None
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)

    def validate_configuration(self) -> List[str]:
        """Validate the aircraft configuration and return list of validation errors."""
        errors = []
        
        # Basic validation
        if not self.name.strip():
            errors.append("Aircraft configuration must have a name")
        
        if self.base_platform is None:
            errors.append("Aircraft configuration must have a base platform")
        
        # Module validation
        if not self.modules:
            errors.append("Aircraft configuration must have at least one module")
        
        # Check for duplicate module IDs
        module_ids = [module.module_id for module in self.modules]
        if len(module_ids) != len(set(module_ids)):
            errors.append("Duplicate module IDs found in configuration")
        
        # Validate module compatibility
        compatibility_errors = self._validate_module_compatibility()
        errors.extend(compatibility_errors)
        
        # Validate interfaces
        interface_errors = self._validate_interfaces()
        errors.extend(interface_errors)
        
        # Validate performance envelope consistency
        if self.performance_envelope:
            perf_errors = self._validate_performance_envelope()
            errors.extend(perf_errors)
        
        return errors

    def _validate_module_compatibility(self) -> List[str]:
        """Validate compatibility between modules."""
        errors = []
        
        if not self.base_platform:
            return errors
        
        # Check if modules can be attached to available attachment points
        available_points = len(self.base_platform.attachment_points)
        required_points = len([m for m in self.modules if m.mechanical_interfaces])
        
        if required_points > available_points:
            errors.append(f"Not enough attachment points: need {required_points}, have {available_points}")
        
        # Check power requirements vs generation capacity
        total_power_required = sum(
            sum(ei.power_consumption for ei in module.electrical_interfaces)
            for module in self.modules
        )
        
        if total_power_required > self.base_platform.power_generation_capacity:
            errors.append(f"Power requirements exceed generation capacity: {total_power_required}W > {self.base_platform.power_generation_capacity}W")
        
        # Check module-specific compatibility requirements
        for module in self.modules:
            for req in module.compatibility_requirements:
                if not self._check_compatibility_requirement(req):
                    errors.append(f"Module {module.name} compatibility requirement not met: {req}")
        
        return errors

    def _validate_interfaces(self) -> List[str]:
        """Validate module interfaces."""
        errors = []
        
        # Check that all interfaces reference valid modules
        module_ids = {module.module_id for module in self.modules}
        
        for interface in self.interfaces:
            # Interface validation logic would go here
            # For now, just check basic structure
            if not interface.interface_id:
                errors.append("Interface missing ID")
        
        return errors

    def _validate_performance_envelope(self) -> List[str]:
        """Validate performance envelope consistency."""
        errors = []
        
        if not self.performance_envelope:
            return errors
        
        # Check that thrust-to-weight ratio is reasonable
        if self.performance_envelope.thrust_to_weight_ratio <= 0:
            errors.append("Thrust-to-weight ratio must be positive")
        
        if self.performance_envelope.thrust_to_weight_ratio > 10:
            errors.append("Thrust-to-weight ratio seems unrealistic (>10)")
        
        # Check service ceiling
        if self.performance_envelope.service_ceiling < 0:
            errors.append("Service ceiling cannot be negative")
        
        return errors

    def _check_compatibility_requirement(self, requirement: str) -> bool:
        """Check if a specific compatibility requirement is met."""
        # This would implement specific compatibility checking logic
        # For now, return True as placeholder
        return True

    def add_module(self, module: 'Module') -> bool:
        """Add a module to the configuration with validation."""
        # Check if module can be added
        if not self.base_platform:
            return False
        
        # Check for duplicate ID
        if any(m.module_id == module.module_id for m in self.modules):
            return False
        
        # Add module and update timestamp
        self.modules.append(module)
        self.modified_at = datetime.now()
        return True

    def remove_module(self, module_id: str) -> bool:
        """Remove a module from the configuration."""
        original_count = len(self.modules)
        self.modules = [m for m in self.modules if m.module_id != module_id]
        
        if len(self.modules) < original_count:
            self.modified_at = datetime.now()
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        data = asdict(self)
        
        # Convert datetime objects to ISO format strings
        data['created_at'] = self.created_at.isoformat()
        data['modified_at'] = self.modified_at.isoformat()
        
        # Convert modules to serializable format
        if 'modules' in data and data['modules']:
            data['modules'] = [module.to_dict() if hasattr(module, 'to_dict') else self._convert_module_dict(module) for module in self.modules]
        
        return data

    def _convert_module_dict(self, module_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert module dictionary to serializable format."""
        if isinstance(module_data, dict) and 'module_type' in module_data:
            if hasattr(module_data['module_type'], 'name'):
                module_data['module_type'] = module_data['module_type'].name
        return module_data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AircraftConfiguration':
        """Create configuration from dictionary."""
        # Make a copy to avoid modifying original data
        data = data.copy()
        
        # Convert datetime strings back to datetime objects
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'modified_at' in data and isinstance(data['modified_at'], str):
            data['modified_at'] = datetime.fromisoformat(data['modified_at'])
        
        # Handle nested objects
        if 'base_platform' in data and data['base_platform']:
            data['base_platform'] = BasePlatform(**data['base_platform'])
        
        if 'modules' in data and data['modules']:
            data['modules'] = [Module.from_dict(module_data) if isinstance(module_data, dict) else module_data for module_data in data['modules']]
        
        if 'interfaces' in data and data['interfaces']:
            data['interfaces'] = [ModuleInterface.from_dict(interface_data) if isinstance(interface_data, dict) else interface_data for interface_data in data['interfaces']]
        
        if 'performance_envelope' in data and data['performance_envelope']:
            perf_data = data['performance_envelope']
            # Convert enum keys back to enum objects if needed
            if 'max_speed' in perf_data and isinstance(perf_data['max_speed'], dict):
                new_max_speed = {}
                for key, value in perf_data['max_speed'].items():
                    if isinstance(key, str):
                        try:
                            enum_key = FlightRegime[key]
                            new_max_speed[enum_key] = value
                        except KeyError:
                            new_max_speed[key] = value
                    else:
                        new_max_speed[key] = value
                perf_data['max_speed'] = new_max_speed
            data['performance_envelope'] = PerformanceEnvelope(**perf_data)
        
        if 'mission_requirements' in data and data['mission_requirements']:
            data['mission_requirements'] = MissionRequirements(**data['mission_requirements'])
        
        return cls(**data)

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save configuration to file (JSON or YAML based on extension)."""
        file_path = Path(file_path)
        data = self.to_dict()
        
        if file_path.suffix.lower() == '.yaml' or file_path.suffix.lower() == '.yml':
            # Convert tuples to lists for YAML compatibility
            data = self._convert_tuples_to_lists(data)
            with open(file_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
        else:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)

    def _convert_tuples_to_lists(self, obj):
        """Recursively convert tuples to lists for YAML serialization."""
        if isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_tuples_to_lists(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_tuples_to_lists(item) for item in obj]
        else:
            return obj

    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'AircraftConfiguration':
        """Load configuration from file (JSON or YAML)."""
        file_path = Path(file_path)
        
        with open(file_path, 'r') as f:
            if file_path.suffix.lower() == '.yaml' or file_path.suffix.lower() == '.yml':
                data = yaml.safe_load(f)
                # Convert lists back to tuples where needed
                data = cls._convert_lists_to_tuples(data)
            else:
                data = json.load(f)
        
        return cls.from_dict(data)

    @classmethod
    def _convert_lists_to_tuples(cls, obj):
        """Recursively convert lists back to tuples for specific fields."""
        if isinstance(obj, dict):
            # Convert specific fields that should be tuples
            tuple_fields = [
                'center_of_gravity', 'moments_of_inertia', 'dimensions',
                'load_capacity', 'moment_capacity', 'position',
                'altitude_requirement', 'speed_requirement'
            ]
            
            for key, value in obj.items():
                if key in tuple_fields and isinstance(value, list):
                    obj[key] = tuple(value)
                else:
                    obj[key] = cls._convert_lists_to_tuples(value)
        elif isinstance(obj, list):
            return [cls._convert_lists_to_tuples(item) for item in obj]
        
        return obj


@dataclass
class EMProperties:
    """Electromagnetic properties of materials."""
    permittivity: complex
    permeability: complex
    conductivity: float  # S/m
    frequency_range: tuple[float, float]  # Hz
    loss_tangent: float


@dataclass
class ThermalProperties:
    """Thermal properties of materials."""
    thermal_conductivity: float  # W/(m⋅K)
    specific_heat: float  # J/(kg⋅K)
    density: float  # kg/m³
    melting_point: float  # K
    operating_temp_range: tuple[float, float]  # K


@dataclass
class MechanicalProperties:
    """Mechanical properties of materials."""
    youngs_modulus: float  # Pa
    poissons_ratio: float
    yield_strength: float  # Pa
    ultimate_strength: float  # Pa
    fatigue_limit: float  # Pa
    density: float  # kg/m³


@dataclass
class ManufacturingConstraints:
    """Manufacturing constraints for materials."""
    min_thickness: float  # m
    max_thickness: float  # m
    cure_temperature: Optional[float] = None  # K
    cure_time: Optional[float] = None  # s
    tooling_requirements: List[str] = field(default_factory=list)
    cost_per_kg: float = 0.0  # $/kg


@dataclass
class FlowConditions:
    """Flow conditions for aerodynamic analysis."""
    mach_number: float
    altitude: float  # m
    angle_of_attack: float  # degrees
    sideslip_angle: float  # degrees
    temperature: Optional[float] = None  # K
    pressure: Optional[float] = None  # Pa
    density: Optional[float] = None  # kg/m³
    
    def __post_init__(self):
        """Calculate atmospheric properties if not provided."""
        if self.temperature is None or self.pressure is None or self.density is None:
            # Use standard atmosphere model
            self._calculate_atmospheric_properties()
    
    def _calculate_atmospheric_properties(self):
        """Calculate atmospheric properties using standard atmosphere model."""
        # Simplified standard atmosphere model
        if self.altitude <= 11000:  # Troposphere
            T0 = 288.15  # K
            P0 = 101325  # Pa
            rho0 = 1.225  # kg/m³
            L = -0.0065  # K/m
            
            if self.temperature is None:
                self.temperature = T0 + L * self.altitude
            
            if self.pressure is None:
                self.pressure = P0 * (self.temperature / T0) ** (-9.80665 / (287.0 * L))
            
            if self.density is None:
                self.density = self.pressure / (287.0 * self.temperature)
        else:
            # Simplified stratosphere
            if self.temperature is None:
                self.temperature = 216.65  # K
            if self.pressure is None:
                self.pressure = 22632 * np.exp(-9.80665 * (self.altitude - 11000) / (287.0 * 216.65))
            if self.density is None:
                self.density = self.pressure / (287.0 * self.temperature)


@dataclass
class AnalysisResults:
    """Base class for analysis results."""
    configuration_id: str
    analysis_type: str
    timestamp: str
    success: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class MaterialDefinition:
    """Advanced material definition."""
    material_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    base_material_type: MaterialType = MaterialType.CONVENTIONAL_METAL
    electromagnetic_properties: Optional[EMProperties] = None
    thermal_properties: Optional[ThermalProperties] = None
    mechanical_properties: Optional[MechanicalProperties] = None
    manufacturing_constraints: Optional[ManufacturingConstraints] = None
    created_at: datetime = field(default_factory=datetime.now)

    def validate_material(self) -> List[str]:
        """Validate material definition and return list of errors."""
        errors = []
        
        # Basic validation
        if not self.name.strip():
            errors.append("Material must have a name")
        
        # Validate electromagnetic properties
        if self.electromagnetic_properties:
            em_errors = self._validate_electromagnetic_properties()
            errors.extend(em_errors)
        
        # Validate thermal properties
        if self.thermal_properties:
            thermal_errors = self._validate_thermal_properties()
            errors.extend(thermal_errors)
        
        # Validate mechanical properties
        if self.mechanical_properties:
            mech_errors = self._validate_mechanical_properties()
            errors.extend(mech_errors)
        
        # Validate manufacturing constraints
        if self.manufacturing_constraints:
            mfg_errors = self._validate_manufacturing_constraints()
            errors.extend(mfg_errors)
        
        # Material type specific validation
        type_errors = self._validate_material_type_requirements()
        errors.extend(type_errors)
        
        return errors

    def _validate_electromagnetic_properties(self) -> List[str]:
        """Validate electromagnetic properties."""
        errors = []
        
        if not self.electromagnetic_properties:
            return errors
        
        em = self.electromagnetic_properties
        
        # Conductivity must be non-negative
        if em.conductivity < 0:
            errors.append("Conductivity cannot be negative")
        
        # Loss tangent must be non-negative
        if em.loss_tangent < 0:
            errors.append("Loss tangent cannot be negative")
        
        # Frequency range validation
        if em.frequency_range[0] >= em.frequency_range[1]:
            errors.append("Invalid frequency range: min must be less than max")
        
        if em.frequency_range[0] < 0:
            errors.append("Frequency range cannot contain negative values")
        
        return errors

    def _validate_thermal_properties(self) -> List[str]:
        """Validate thermal properties."""
        errors = []
        
        if not self.thermal_properties:
            return errors
        
        thermal = self.thermal_properties
        
        # All thermal properties must be positive
        if thermal.thermal_conductivity <= 0:
            errors.append("Thermal conductivity must be positive")
        
        if thermal.specific_heat <= 0:
            errors.append("Specific heat must be positive")
        
        if thermal.density <= 0:
            errors.append("Density must be positive")
        
        if thermal.melting_point <= 0:
            errors.append("Melting point must be positive")
        
        # Operating temperature range validation
        if thermal.operating_temp_range[0] >= thermal.operating_temp_range[1]:
            errors.append("Invalid operating temperature range")
        
        if thermal.operating_temp_range[1] > thermal.melting_point:
            errors.append("Operating temperature range exceeds melting point")
        
        return errors

    def _validate_mechanical_properties(self) -> List[str]:
        """Validate mechanical properties."""
        errors = []
        
        if not self.mechanical_properties:
            return errors
        
        mech = self.mechanical_properties
        
        # All mechanical properties must be positive
        if mech.youngs_modulus <= 0:
            errors.append("Young's modulus must be positive")
        
        if mech.yield_strength <= 0:
            errors.append("Yield strength must be positive")
        
        if mech.ultimate_strength <= 0:
            errors.append("Ultimate strength must be positive")
        
        if mech.fatigue_limit <= 0:
            errors.append("Fatigue limit must be positive")
        
        if mech.density <= 0:
            errors.append("Mechanical density must be positive")
        
        # Poisson's ratio must be between -1 and 0.5 for most materials
        if not (-1.0 <= mech.poissons_ratio <= 0.5):
            errors.append("Poisson's ratio must be between -1.0 and 0.5")
        
        # Yield strength should be less than ultimate strength
        if mech.yield_strength >= mech.ultimate_strength:
            errors.append("Yield strength must be less than ultimate strength")
        
        return errors

    def _validate_manufacturing_constraints(self) -> List[str]:
        """Validate manufacturing constraints."""
        errors = []
        
        if not self.manufacturing_constraints:
            return errors
        
        mfg = self.manufacturing_constraints
        
        # Thickness constraints
        if mfg.min_thickness <= 0:
            errors.append("Minimum thickness must be positive")
        
        if mfg.max_thickness <= 0:
            errors.append("Maximum thickness must be positive")
        
        if mfg.min_thickness >= mfg.max_thickness:
            errors.append("Minimum thickness must be less than maximum thickness")
        
        # Cure temperature validation
        if mfg.cure_temperature is not None and mfg.cure_temperature <= 0:
            errors.append("Cure temperature must be positive")
        
        # Cure time validation
        if mfg.cure_time is not None and mfg.cure_time <= 0:
            errors.append("Cure time must be positive")
        
        # Cost validation
        if mfg.cost_per_kg < 0:
            errors.append("Cost per kg cannot be negative")
        
        return errors

    def _validate_material_type_requirements(self) -> List[str]:
        """Validate material type specific requirements."""
        errors = []
        
        # Metamaterials must have electromagnetic properties
        if self.base_material_type == MaterialType.METAMATERIAL:
            if not self.electromagnetic_properties:
                errors.append("Metamaterials must have electromagnetic properties defined")
        
        # Ultra-high temp ceramics must have thermal properties
        if self.base_material_type == MaterialType.ULTRA_HIGH_TEMP_CERAMIC:
            if not self.thermal_properties:
                errors.append("Ultra-high temperature ceramics must have thermal properties defined")
            elif self.thermal_properties.melting_point < 2273:  # 2000°C
                errors.append("Ultra-high temperature ceramics must have melting point > 2000°C")
        
        # Conductive polymers must have electromagnetic properties
        if self.base_material_type == MaterialType.CONDUCTIVE_POLYMER:
            if not self.electromagnetic_properties:
                errors.append("Conductive polymers must have electromagnetic properties defined")
            elif self.electromagnetic_properties.conductivity <= 0:
                errors.append("Conductive polymers must have positive conductivity")
        
        # Stealth coatings must have electromagnetic properties
        if self.base_material_type == MaterialType.STEALTH_COATING:
            if not self.electromagnetic_properties:
                errors.append("Stealth coatings must have electromagnetic properties defined")
        
        return errors

    def calculate_metamaterial_response(self, frequency: float) -> complex:
        """Calculate metamaterial electromagnetic response at given frequency."""
        if not self.electromagnetic_properties:
            raise ValueError("No electromagnetic properties defined")
        
        if self.base_material_type != MaterialType.METAMATERIAL:
            raise ValueError("Material is not a metamaterial")
        
        em = self.electromagnetic_properties
        
        # Check if frequency is in valid range
        if not (em.frequency_range[0] <= frequency <= em.frequency_range[1]):
            raise ValueError(f"Frequency {frequency} Hz outside valid range {em.frequency_range}")
        
        # Simple metamaterial response model
        # In practice, this would be much more complex
        omega = 2 * 3.14159 * frequency
        
        # Frequency-dependent permittivity with resonance
        resonance_freq = (em.frequency_range[0] + em.frequency_range[1]) / 2
        omega_r = 2 * 3.14159 * resonance_freq
        
        # Lorentzian response model
        gamma = omega_r * em.loss_tangent  # Damping
        
        epsilon_response = em.permittivity * (1 - (omega_r**2) / (omega**2 - omega_r**2 + 1j * gamma * omega))
        
        return epsilon_response

    def calculate_stealth_effectiveness(self, frequency: float, thickness: float) -> float:
        """Calculate radar absorption effectiveness for stealth materials."""
        if not self.electromagnetic_properties:
            raise ValueError("No electromagnetic properties defined")
        
        if self.base_material_type not in [MaterialType.STEALTH_COATING, MaterialType.METAMATERIAL]:
            raise ValueError("Material is not suitable for stealth applications")
        
        em = self.electromagnetic_properties
        
        # Simple absorption calculation using complex permittivity
        # Real implementation would use full electromagnetic simulation
        
        # Calculate wave impedance in material
        epsilon_r = em.permittivity
        mu_r = em.permeability
        
        # Intrinsic impedance of material
        eta = (mu_r / epsilon_r) ** 0.5
        
        # Reflection coefficient at air-material interface
        eta_0 = 377.0  # Free space impedance
        gamma = (eta - eta_0) / (eta + eta_0)
        
        # Transmission through material (simplified)
        k = 2 * 3.14159 * frequency / 3e8 * (epsilon_r * mu_r) ** 0.5
        transmission = abs(1 - gamma) ** 2 * abs(1 / (1 + gamma * complex(0, 1) * k * thickness)) ** 2
        
        # Absorption = 1 - reflection - transmission
        reflection = abs(gamma) ** 2
        absorption = 1 - reflection - transmission
        
        return max(0.0, min(1.0, absorption))  # Clamp between 0 and 1

    def calculate_thermal_stress(self, temperature_gradient: float) -> float:
        """Calculate thermal stress in material due to temperature gradient."""
        if not self.thermal_properties or not self.mechanical_properties:
            raise ValueError("Both thermal and mechanical properties required")
        
        # Thermal stress = α * E * ΔT
        # Using simplified linear expansion coefficient estimation
        alpha = 1e-5  # Typical thermal expansion coefficient (1/K)
        
        thermal_stress = alpha * self.mechanical_properties.youngs_modulus * temperature_gradient
        
        return thermal_stress

    def is_suitable_for_temperature(self, temperature: float) -> bool:
        """Check if material is suitable for given operating temperature."""
        if not self.thermal_properties:
            return False
        
        return (self.thermal_properties.operating_temp_range[0] <= 
                temperature <= 
                self.thermal_properties.operating_temp_range[1])

    def calculate_manufacturing_cost(self, volume: float, complexity_factor: float = 1.0) -> float:
        """Calculate manufacturing cost for given volume and complexity."""
        if not self.manufacturing_constraints or not self.mechanical_properties:
            raise ValueError("Manufacturing constraints and mechanical properties required")
        
        # Mass = volume * density
        mass = volume * self.mechanical_properties.density
        
        # Base cost
        base_cost = mass * self.manufacturing_constraints.cost_per_kg
        
        # Apply complexity factor
        total_cost = base_cost * complexity_factor
        
        return total_cost

    def to_dict(self) -> Dict[str, Any]:
        """Convert material to dictionary for serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['base_material_type'] = self.base_material_type.name
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MaterialDefinition':
        """Create material from dictionary."""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        if 'base_material_type' in data and isinstance(data['base_material_type'], str):
            data['base_material_type'] = MaterialType[data['base_material_type']]
        
        # Handle nested objects
        if 'electromagnetic_properties' in data and data['electromagnetic_properties']:
            data['electromagnetic_properties'] = EMProperties(**data['electromagnetic_properties'])
        
        if 'thermal_properties' in data and data['thermal_properties']:
            data['thermal_properties'] = ThermalProperties(**data['thermal_properties'])
        
        if 'mechanical_properties' in data and data['mechanical_properties']:
            data['mechanical_properties'] = MechanicalProperties(**data['mechanical_properties'])
        
        if 'manufacturing_constraints' in data and data['manufacturing_constraints']:
            data['manufacturing_constraints'] = ManufacturingConstraints(**data['manufacturing_constraints'])
        
        return cls(**data)


@dataclass
class DetectionCapabilities:
    """Sensor detection capabilities."""
    detection_range: Dict[str, float] = field(default_factory=dict)  # km by target type
    resolution: Dict[str, float] = field(default_factory=dict)  # various units
    accuracy: Dict[str, float] = field(default_factory=dict)  # various units
    update_rate: float = 0.0  # Hz
    field_of_view: tuple[float, float] = (0.0, 0.0)  # azimuth, elevation (degrees)


@dataclass
class PowerRequirements:
    """Power requirements for systems."""
    peak_power: float = 0.0  # W
    average_power: float = 0.0  # W
    startup_power: float = 0.0  # W
    voltage_requirement: float = 0.0  # V
    power_quality_requirements: Dict[str, float] = field(default_factory=dict)


@dataclass
class AtmosphericConstraints:
    """Atmospheric operation constraints."""
    altitude_range: tuple[float, float] = (0.0, 0.0)  # m
    temperature_range: tuple[float, float] = (0.0, 0.0)  # K
    humidity_limits: tuple[float, float] = (0.0, 100.0)  # %
    pressure_range: tuple[float, float] = (0.0, 0.0)  # Pa
    weather_limitations: List[str] = field(default_factory=list)


@dataclass
class IntegrationRequirements:
    """System integration requirements."""
    cooling_requirements: Dict[str, float] = field(default_factory=dict)
    vibration_isolation: bool = False
    electromagnetic_shielding: bool = False
    data_interfaces: List[str] = field(default_factory=list)
    physical_constraints: Dict[str, float] = field(default_factory=dict)


@dataclass
class SensorSystem:
    """Advanced sensor system definition."""
    sensor_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    sensor_type: SensorType = SensorType.PASSIVE_RF
    detection_capabilities: Optional[DetectionCapabilities] = None
    power_requirements: Optional[PowerRequirements] = None
    atmospheric_limitations: Optional[AtmosphericConstraints] = None
    integration_requirements: Optional[IntegrationRequirements] = None
    created_at: datetime = field(default_factory=datetime.now)

    def validate_sensor(self) -> List[str]:
        """Validate sensor system configuration and return list of errors."""
        errors = []
        
        # Basic validation
        if not self.name.strip():
            errors.append("Sensor system must have a name")
        
        # Validate detection capabilities
        if self.detection_capabilities:
            detection_errors = self._validate_detection_capabilities()
            errors.extend(detection_errors)
        
        # Validate power requirements
        if self.power_requirements:
            power_errors = self._validate_power_requirements()
            errors.extend(power_errors)
        
        # Validate atmospheric constraints
        if self.atmospheric_limitations:
            atmo_errors = self._validate_atmospheric_constraints()
            errors.extend(atmo_errors)
        
        # Validate integration requirements
        if self.integration_requirements:
            integration_errors = self._validate_integration_requirements()
            errors.extend(integration_errors)
        
        # Sensor type specific validation
        type_errors = self._validate_sensor_type_requirements()
        errors.extend(type_errors)
        
        return errors

    def _validate_detection_capabilities(self) -> List[str]:
        """Validate detection capabilities."""
        errors = []
        
        if not self.detection_capabilities:
            return errors
        
        dc = self.detection_capabilities
        
        # Detection ranges must be positive
        for target_type, range_km in dc.detection_range.items():
            if range_km <= 0:
                errors.append(f"Detection range for {target_type} must be positive")
        
        # Update rate must be positive
        if dc.update_rate <= 0:
            errors.append("Update rate must be positive")
        
        # Field of view validation
        azimuth_fov, elevation_fov = dc.field_of_view
        if not (0 < azimuth_fov <= 360):
            errors.append("Azimuth field of view must be between 0 and 360 degrees")
        
        if not (0 < elevation_fov <= 180):
            errors.append("Elevation field of view must be between 0 and 180 degrees")
        
        # Resolution values must be positive
        for param, resolution in dc.resolution.items():
            if resolution <= 0:
                errors.append(f"Resolution for {param} must be positive")
        
        # Accuracy values must be positive
        for param, accuracy in dc.accuracy.items():
            if accuracy <= 0:
                errors.append(f"Accuracy for {param} must be positive")
        
        return errors

    def _validate_power_requirements(self) -> List[str]:
        """Validate power requirements."""
        errors = []
        
        if not self.power_requirements:
            return errors
        
        pr = self.power_requirements
        
        # All power values must be non-negative
        if pr.peak_power < 0:
            errors.append("Peak power cannot be negative")
        
        if pr.average_power < 0:
            errors.append("Average power cannot be negative")
        
        if pr.startup_power < 0:
            errors.append("Startup power cannot be negative")
        
        if pr.voltage_requirement <= 0:
            errors.append("Voltage requirement must be positive")
        
        # Average power should not exceed peak power
        if pr.average_power > pr.peak_power:
            errors.append("Average power cannot exceed peak power")
        
        # Startup power validation
        if pr.startup_power > pr.peak_power * 2:  # Allow 2x peak for startup
            errors.append("Startup power seems unreasonably high (>2x peak power)")
        
        return errors

    def _validate_atmospheric_constraints(self) -> List[str]:
        """Validate atmospheric constraints."""
        errors = []
        
        if not self.atmospheric_limitations:
            return errors
        
        ac = self.atmospheric_limitations
        
        # Altitude range validation
        if ac.altitude_range[0] >= ac.altitude_range[1]:
            errors.append("Invalid altitude range: min must be less than max")
        
        if ac.altitude_range[0] < 0:
            errors.append("Minimum altitude cannot be negative")
        
        # Temperature range validation
        if ac.temperature_range[0] >= ac.temperature_range[1]:
            errors.append("Invalid temperature range: min must be less than max")
        
        # Humidity limits validation
        if not (0 <= ac.humidity_limits[0] <= ac.humidity_limits[1] <= 100):
            errors.append("Humidity limits must be between 0 and 100 percent")
        
        # Pressure range validation
        if ac.pressure_range[0] >= ac.pressure_range[1]:
            errors.append("Invalid pressure range: min must be less than max")
        
        if ac.pressure_range[0] < 0:
            errors.append("Pressure cannot be negative")
        
        return errors

    def _validate_integration_requirements(self) -> List[str]:
        """Validate integration requirements."""
        errors = []
        
        if not self.integration_requirements:
            return errors
        
        ir = self.integration_requirements
        
        # Cooling requirements validation
        for system, cooling_power in ir.cooling_requirements.items():
            if cooling_power < 0:
                errors.append(f"Cooling requirement for {system} cannot be negative")
        
        # Physical constraints validation
        for constraint, value in ir.physical_constraints.items():
            if isinstance(value, (int, float)) and value < 0:
                errors.append(f"Physical constraint {constraint} cannot be negative")
        
        return errors

    def _validate_sensor_type_requirements(self) -> List[str]:
        """Validate sensor type specific requirements."""
        errors = []
        
        # AESA radar specific requirements
        if self.sensor_type == SensorType.AESA_RADAR:
            if not self.power_requirements:
                errors.append("AESA radar must have power requirements defined")
            elif self.power_requirements.peak_power < 1000:  # 1kW minimum
                errors.append("AESA radar peak power should be at least 1kW")
        
        # Laser-based sensor requirements
        if self.sensor_type == SensorType.LASER_BASED:
            if not self.power_requirements:
                errors.append("Laser-based sensors must have power requirements defined")
            if not self.atmospheric_limitations:
                errors.append("Laser-based sensors must have atmospheric limitations defined")
        
        # Plasma-based sensor requirements
        if self.sensor_type == SensorType.PLASMA_BASED:
            if not self.power_requirements:
                errors.append("Plasma-based sensors must have power requirements defined")
            elif self.power_requirements.peak_power < 10000:  # 10kW minimum for plasma
                errors.append("Plasma-based sensors require at least 10kW peak power")
        
        return errors

    def calculate_power_consumption(self, duty_cycle: float = 1.0) -> float:
        """Calculate power consumption based on duty cycle."""
        if not self.power_requirements:
            return 0.0
        
        if not (0.0 <= duty_cycle <= 1.0):
            raise ValueError("Duty cycle must be between 0 and 1")
        
        # For continuous operation, use average power
        # For pulsed operation, scale between average and peak based on duty cycle
        if duty_cycle == 1.0:
            return self.power_requirements.average_power
        else:
            # Linear interpolation between average and peak power
            return (self.power_requirements.average_power + 
                   (self.power_requirements.peak_power - self.power_requirements.average_power) * duty_cycle)

    def calculate_detection_probability(self, target_rcs: float, range_km: float, 
                                     atmospheric_attenuation: float = 1.0) -> float:
        """Calculate detection probability for given target and conditions."""
        if not self.detection_capabilities:
            return 0.0
        
        if target_rcs <= 0 or range_km <= 0:
            return 0.0
        
        # Simplified radar equation for detection probability
        # In practice, this would be much more complex
        
        # Get maximum detection range for similar target
        max_range = 0.0
        for target_type, max_r in self.detection_capabilities.detection_range.items():
            if "fighter" in target_type.lower() or "aircraft" in target_type.lower():
                max_range = max_r
                break
        
        if max_range == 0.0:
            # Use first available range if no aircraft type found
            max_range = next(iter(self.detection_capabilities.detection_range.values()), 100.0)
        
        # Simple range-based probability with RCS scaling
        # Assume max_range is for 1 m² RCS target
        effective_range = max_range * (target_rcs ** 0.25)  # Fourth root scaling
        
        # Apply atmospheric attenuation
        effective_range *= atmospheric_attenuation
        
        if range_km > effective_range:
            return 0.0
        
        # Linear probability decrease with range
        probability = 1.0 - (range_km / effective_range)
        
        return max(0.0, min(1.0, probability))

    def check_atmospheric_compatibility(self, altitude: float, temperature: float, 
                                      humidity: float, pressure: float) -> bool:
        """Check if sensor can operate in given atmospheric conditions."""
        if not self.atmospheric_limitations:
            return True  # No limitations defined, assume compatible
        
        ac = self.atmospheric_limitations
        
        # Check altitude
        if not (ac.altitude_range[0] <= altitude <= ac.altitude_range[1]):
            return False
        
        # Check temperature
        if not (ac.temperature_range[0] <= temperature <= ac.temperature_range[1]):
            return False
        
        # Check humidity
        if not (ac.humidity_limits[0] <= humidity <= ac.humidity_limits[1]):
            return False
        
        # Check pressure
        if not (ac.pressure_range[0] <= pressure <= ac.pressure_range[1]):
            return False
        
        return True

    def calculate_cooling_requirements(self, ambient_temperature: float, 
                                    duty_cycle: float = 1.0) -> Dict[str, float]:
        """Calculate cooling requirements based on operating conditions."""
        if not self.power_requirements or not self.integration_requirements:
            return {}
        
        power_dissipation = self.calculate_power_consumption(duty_cycle)
        
        # Estimate heat generation (assume 70% of power becomes heat)
        heat_generation = power_dissipation * 0.7
        
        # Temperature-dependent cooling scaling
        temp_factor = max(1.0, (ambient_temperature - 273.15) / 25.0)  # Scale from 25°C baseline
        
        cooling_requirements = {}
        
        # Base cooling requirement
        cooling_requirements["primary_cooling"] = heat_generation * temp_factor
        
        # Additional cooling for high-power sensors
        if self.sensor_type in [SensorType.AESA_RADAR, SensorType.LASER_BASED, SensorType.PLASMA_BASED]:
            cooling_requirements["secondary_cooling"] = heat_generation * 0.3 * temp_factor
        
        # Electronics cooling
        cooling_requirements["electronics_cooling"] = power_dissipation * 0.1 * temp_factor
        
        return cooling_requirements

    def estimate_detection_range(self, target_rcs: float, detection_probability: float = 0.9) -> float:
        """Estimate detection range for given target RCS and required probability."""
        if not self.detection_capabilities:
            return 0.0
        
        if target_rcs <= 0 or not (0.0 < detection_probability <= 1.0):
            return 0.0
        
        # Get reference range (assume for 1 m² RCS)
        reference_range = 0.0
        for target_type, range_km in self.detection_capabilities.detection_range.items():
            if "fighter" in target_type.lower() or "aircraft" in target_type.lower():
                reference_range = range_km
                break
        
        if reference_range == 0.0:
            reference_range = next(iter(self.detection_capabilities.detection_range.values()), 100.0)
        
        # Scale range based on RCS (fourth root relationship)
        estimated_range = reference_range * (target_rcs ** 0.25)
        
        # Adjust for required detection probability
        probability_factor = detection_probability ** 0.25  # Approximate scaling
        estimated_range *= probability_factor
        
        return estimated_range

    def to_dict(self) -> Dict[str, Any]:
        """Convert sensor system to dictionary for serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['sensor_type'] = self.sensor_type.name
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SensorSystem':
        """Create sensor system from dictionary."""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        if 'sensor_type' in data and isinstance(data['sensor_type'], str):
            data['sensor_type'] = SensorType[data['sensor_type']]
        
        # Handle nested objects
        if 'detection_capabilities' in data and data['detection_capabilities']:
            data['detection_capabilities'] = DetectionCapabilities(**data['detection_capabilities'])
        
        if 'power_requirements' in data and data['power_requirements']:
            data['power_requirements'] = PowerRequirements(**data['power_requirements'])
        
        if 'atmospheric_limitations' in data and data['atmospheric_limitations']:
            data['atmospheric_limitations'] = AtmosphericConstraints(**data['atmospheric_limitations'])
        
        if 'integration_requirements' in data and data['integration_requirements']:
            data['integration_requirements'] = IntegrationRequirements(**data['integration_requirements'])
        
        return cls(**data)


@dataclass
class Position3D:
    """3D position coordinates."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def distance_to(self, other: 'Position3D') -> float:
        """Calculate distance to another position."""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Position3D':
        """Create from numpy array."""
        return cls(x=arr[0], y=arr[1], z=arr[2])


@dataclass
class Velocity3D:
    """3D velocity vector."""
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    
    def magnitude(self) -> float:
        """Calculate velocity magnitude."""
        return np.sqrt(self.vx**2 + self.vy**2 + self.vz**2)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.vx, self.vy, self.vz])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Velocity3D':
        """Create from numpy array."""
        return cls(vx=arr[0], vy=arr[1], vz=arr[2])


# Extreme Hypersonic Data Models

@dataclass
class PlasmaConditions:
    """Plasma conditions for extreme hypersonic flight analysis."""
    electron_density: float  # m⁻³
    electron_temperature: float  # K
    ion_temperature: float  # K
    magnetic_field: np.ndarray  # Tesla [Bx, By, Bz]
    plasma_frequency: float  # Hz
    debye_length: float  # m
    plasma_regime: 'PlasmaRegime' = field(default_factory=lambda: PlasmaRegime.WEAKLY_IONIZED)
    
    def __post_init__(self):
        """Validate plasma conditions after initialization."""
        if isinstance(self.magnetic_field, (list, tuple)):
            self.magnetic_field = np.array(self.magnetic_field)
        elif not isinstance(self.magnetic_field, np.ndarray):
            self.magnetic_field = np.array([0.0, 0.0, 0.0])
    
    def validate_conditions(self) -> List[str]:
        """Validate plasma conditions and return list of errors."""
        errors = []
        
        if self.electron_density < 0:
            errors.append("Electron density cannot be negative")
        
        if self.electron_temperature <= 0:
            errors.append("Electron temperature must be positive")
        
        if self.ion_temperature <= 0:
            errors.append("Ion temperature must be positive")
        
        if self.plasma_frequency < 0:
            errors.append("Plasma frequency cannot be negative")
        
        if self.debye_length <= 0:
            errors.append("Debye length must be positive")
        
        if len(self.magnetic_field) != 3:
            errors.append("Magnetic field must be 3D vector")
        
        return errors
    
    def calculate_plasma_beta(self, pressure: float) -> float:
        """Calculate plasma beta parameter (ratio of plasma pressure to magnetic pressure)."""
        if np.linalg.norm(self.magnetic_field) == 0:
            return float('inf')
        
        magnetic_pressure = np.linalg.norm(self.magnetic_field)**2 / (2 * 4e-7 * np.pi)  # μ₀ = 4π×10⁻⁷
        return pressure / magnetic_pressure if magnetic_pressure > 0 else float('inf')


@dataclass
class CombinedCyclePerformance:
    """Performance characteristics for combined-cycle propulsion systems."""
    air_breathing_thrust: float  # N
    rocket_thrust: float  # N
    transition_mach: float  # Mach number for mode transition
    fuel_flow_air_breathing: float  # kg/s
    fuel_flow_rocket: float  # kg/s
    specific_impulse: float  # s
    combustion_efficiency: float = 0.95  # dimensionless
    nozzle_efficiency: float = 0.98  # dimensionless
    
    def validate_performance(self) -> List[str]:
        """Validate combined-cycle performance data."""
        errors = []
        
        if self.air_breathing_thrust < 0:
            errors.append("Air-breathing thrust cannot be negative")
        
        if self.rocket_thrust < 0:
            errors.append("Rocket thrust cannot be negative")
        
        if self.transition_mach <= 0:
            errors.append("Transition Mach number must be positive")
        
        if self.fuel_flow_air_breathing < 0:
            errors.append("Air-breathing fuel flow cannot be negative")
        
        if self.fuel_flow_rocket < 0:
            errors.append("Rocket fuel flow cannot be negative")
        
        if self.specific_impulse <= 0:
            errors.append("Specific impulse must be positive")
        
        if not (0 < self.combustion_efficiency <= 1):
            errors.append("Combustion efficiency must be between 0 and 1")
        
        if not (0 < self.nozzle_efficiency <= 1):
            errors.append("Nozzle efficiency must be between 0 and 1")
        
        return errors
    
    def calculate_total_thrust(self, mach_number: float) -> float:
        """Calculate total thrust based on flight Mach number."""
        if mach_number < self.transition_mach:
            return self.air_breathing_thrust
        else:
            # Transition region - blend both modes
            blend_factor = min(1.0, (mach_number - self.transition_mach) / 5.0)
            return (1 - blend_factor) * self.air_breathing_thrust + blend_factor * self.rocket_thrust
    
    def calculate_total_fuel_flow(self, mach_number: float) -> float:
        """Calculate total fuel flow based on flight Mach number."""
        if mach_number < self.transition_mach:
            return self.fuel_flow_air_breathing
        else:
            blend_factor = min(1.0, (mach_number - self.transition_mach) / 5.0)
            return (1 - blend_factor) * self.fuel_flow_air_breathing + blend_factor * self.fuel_flow_rocket


@dataclass
class AblativeLayer:
    """Ablative layer specification for thermal protection."""
    material_id: str
    thickness: float  # m
    density: float  # kg/m³
    heat_of_ablation: float  # J/kg
    char_yield: float  # dimensionless (0-1)
    
    def validate_layer(self) -> List[str]:
        """Validate ablative layer properties."""
        errors = []
        
        if not self.material_id.strip():
            errors.append("Material ID cannot be empty")
        
        if self.thickness <= 0:
            errors.append("Layer thickness must be positive")
        
        if self.density <= 0:
            errors.append("Layer density must be positive")
        
        if self.heat_of_ablation <= 0:
            errors.append("Heat of ablation must be positive")
        
        if not (0 <= self.char_yield <= 1):
            errors.append("Char yield must be between 0 and 1")
        
        return errors


@dataclass
class CoolingChannel:
    """Active cooling channel specification."""
    channel_id: str
    diameter: float  # m
    length: float  # m
    coolant_type: str
    mass_flow_rate: float  # kg/s
    inlet_temperature: float  # K
    inlet_pressure: float  # Pa
    
    def validate_channel(self) -> List[str]:
        """Validate cooling channel properties."""
        errors = []
        
        if not self.channel_id.strip():
            errors.append("Channel ID cannot be empty")
        
        if self.diameter <= 0:
            errors.append("Channel diameter must be positive")
        
        if self.length <= 0:
            errors.append("Channel length must be positive")
        
        if not self.coolant_type.strip():
            errors.append("Coolant type cannot be empty")
        
        if self.mass_flow_rate <= 0:
            errors.append("Mass flow rate must be positive")
        
        if self.inlet_temperature <= 0:
            errors.append("Inlet temperature must be positive")
        
        if self.inlet_pressure <= 0:
            errors.append("Inlet pressure must be positive")
        
        return errors
    
    def calculate_reynolds_number(self, viscosity: float) -> float:
        """Calculate Reynolds number for the cooling channel."""
        if viscosity <= 0:
            return 0.0
        
        velocity = self.mass_flow_rate / (np.pi * (self.diameter/2)**2 * 1000)  # Assuming water density
        return (1000 * velocity * self.diameter) / viscosity


@dataclass
class InsulationLayer:
    """Insulation layer specification."""
    material_id: str
    thickness: float  # m
    thermal_conductivity: float  # W/(m⋅K)
    max_operating_temperature: float  # K
    
    def validate_layer(self) -> List[str]:
        """Validate insulation layer properties."""
        errors = []
        
        if not self.material_id.strip():
            errors.append("Material ID cannot be empty")
        
        if self.thickness <= 0:
            errors.append("Layer thickness must be positive")
        
        if self.thermal_conductivity <= 0:
            errors.append("Thermal conductivity must be positive")
        
        if self.max_operating_temperature <= 0:
            errors.append("Maximum operating temperature must be positive")
        
        return errors


@dataclass
class ThermalProtectionSystem:
    """Complete thermal protection system specification."""
    system_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    protection_type: 'ThermalProtectionType' = ThermalProtectionType.PASSIVE_ABLATIVE
    ablative_layers: List[AblativeLayer] = field(default_factory=list)
    active_cooling_channels: List[CoolingChannel] = field(default_factory=list)
    insulation_layers: List[InsulationLayer] = field(default_factory=list)
    total_thickness: float = 0.0  # m
    total_mass: float = 0.0  # kg
    cooling_effectiveness: float = 0.0  # dimensionless (0-1)
    max_heat_flux_capacity: float = 0.0  # W/m²
    
    def validate_system(self) -> List[str]:
        """Validate thermal protection system."""
        errors = []
        
        if not self.system_id.strip():
            errors.append("System ID cannot be empty")
        
        if self.total_thickness < 0:
            errors.append("Total thickness cannot be negative")
        
        if self.total_mass < 0:
            errors.append("Total mass cannot be negative")
        
        if not (0 <= self.cooling_effectiveness <= 1):
            errors.append("Cooling effectiveness must be between 0 and 1")
        
        if self.max_heat_flux_capacity < 0:
            errors.append("Maximum heat flux capacity cannot be negative")
        
        # Validate individual layers
        for i, layer in enumerate(self.ablative_layers):
            layer_errors = layer.validate_layer()
            errors.extend([f"Ablative layer {i}: {error}" for error in layer_errors])
        
        for i, channel in enumerate(self.active_cooling_channels):
            channel_errors = channel.validate_channel()
            errors.extend([f"Cooling channel {i}: {error}" for error in channel_errors])
        
        for i, layer in enumerate(self.insulation_layers):
            layer_errors = layer.validate_layer()
            errors.extend([f"Insulation layer {i}: {error}" for error in layer_errors])
        
        # System-level validation
        if self.protection_type == ThermalProtectionType.ACTIVE_TRANSPIRATION and not self.active_cooling_channels:
            errors.append("Active transpiration cooling requires cooling channels")
        
        if self.protection_type == ThermalProtectionType.PASSIVE_ABLATIVE and not self.ablative_layers:
            errors.append("Passive ablative protection requires ablative layers")
        
        return errors
    
    def calculate_total_properties(self) -> None:
        """Calculate total system properties from individual components."""
        # Calculate total thickness
        self.total_thickness = (
            sum(layer.thickness for layer in self.ablative_layers) +
            sum(layer.thickness for layer in self.insulation_layers)
        )
        
        # Calculate total mass (simplified - would need area in real implementation)
        area = 1.0  # m² - placeholder, would be provided by calling system
        self.total_mass = (
            sum(layer.thickness * layer.density * area for layer in self.ablative_layers) +
            sum(layer.thickness * 500 * area for layer in self.insulation_layers)  # Assumed insulation density
        )
    
    def estimate_cooling_effectiveness(self, heat_flux: float) -> float:
        """Estimate cooling effectiveness for given heat flux."""
        if not self.active_cooling_channels:
            return 0.0
        
        # Simplified effectiveness calculation
        total_cooling_capacity = sum(
            channel.mass_flow_rate * 4186 * 100  # Simplified: mass_flow * cp * ΔT
            for channel in self.active_cooling_channels
        )
        
        if heat_flux <= 0:
            return 1.0
        
        return min(1.0, total_cooling_capacity / (heat_flux * 1.0))  # Assuming 1 m² area


@dataclass
class HypersonicMissionProfile:
    """Mission profile for extreme hypersonic flight."""
    profile_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    mission_name: str = ""
    altitude_profile: np.ndarray = field(default_factory=lambda: np.array([]))  # m
    mach_profile: np.ndarray = field(default_factory=lambda: np.array([]))
    thermal_load_profile: np.ndarray = field(default_factory=lambda: np.array([]))  # W/m²
    propulsion_mode_schedule: List[str] = field(default_factory=list)
    cooling_system_schedule: List[bool] = field(default_factory=list)
    mission_duration: float = 0.0  # s
    max_altitude: float = 0.0  # m
    max_mach: float = 0.0
    max_thermal_load: float = 0.0  # W/m²
    
    def __post_init__(self):
        """Initialize arrays if provided as lists."""
        if isinstance(self.altitude_profile, (list, tuple)):
            self.altitude_profile = np.array(self.altitude_profile)
        if isinstance(self.mach_profile, (list, tuple)):
            self.mach_profile = np.array(self.mach_profile)
        if isinstance(self.thermal_load_profile, (list, tuple)):
            self.thermal_load_profile = np.array(self.thermal_load_profile)
    
    def validate_profile(self) -> List[str]:
        """Validate hypersonic mission profile."""
        errors = []
        
        if not self.profile_id.strip():
            errors.append("Profile ID cannot be empty")
        
        if not self.mission_name.strip():
            errors.append("Mission name cannot be empty")
        
        if self.mission_duration <= 0:
            errors.append("Mission duration must be positive")
        
        if self.max_altitude < 0:
            errors.append("Maximum altitude cannot be negative")
        
        if self.max_mach <= 0:
            errors.append("Maximum Mach number must be positive")
        
        if self.max_thermal_load < 0:
            errors.append("Maximum thermal load cannot be negative")
        
        # Check array consistency
        profile_lengths = [
            len(self.altitude_profile),
            len(self.mach_profile),
            len(self.thermal_load_profile)
        ]
        
        if len(set(profile_lengths)) > 1:
            errors.append("All profile arrays must have the same length")
        
        if len(self.propulsion_mode_schedule) > 0 and len(self.propulsion_mode_schedule) != profile_lengths[0]:
            errors.append("Propulsion mode schedule length must match profile arrays")
        
        if len(self.cooling_system_schedule) > 0 and len(self.cooling_system_schedule) != profile_lengths[0]:
            errors.append("Cooling system schedule length must match profile arrays")
        
        # Physical constraints
        if len(self.altitude_profile) > 0:
            if np.any(self.altitude_profile < 0):
                errors.append("Altitude profile cannot contain negative values")
            
            if np.any(self.altitude_profile > 200000):  # 200 km reasonable upper limit
                errors.append("Altitude profile contains unrealistic values (>200 km)")
        
        if len(self.mach_profile) > 0:
            if np.any(self.mach_profile <= 0):
                errors.append("Mach profile must contain only positive values")
            
            if np.any(self.mach_profile > 100):  # Mach 100 as extreme upper limit
                errors.append("Mach profile contains unrealistic values (>Mach 100)")
        
        if len(self.thermal_load_profile) > 0:
            if np.any(self.thermal_load_profile < 0):
                errors.append("Thermal load profile cannot contain negative values")
        
        return errors
    
    def calculate_profile_statistics(self) -> Dict[str, float]:
        """Calculate statistics for the mission profile."""
        stats = {}
        
        if len(self.altitude_profile) > 0:
            stats['avg_altitude'] = float(np.mean(self.altitude_profile))
            stats['min_altitude'] = float(np.min(self.altitude_profile))
            stats['max_altitude'] = float(np.max(self.altitude_profile))
        
        if len(self.mach_profile) > 0:
            stats['avg_mach'] = float(np.mean(self.mach_profile))
            stats['min_mach'] = float(np.min(self.mach_profile))
            stats['max_mach'] = float(np.max(self.mach_profile))
        
        if len(self.thermal_load_profile) > 0:
            stats['avg_thermal_load'] = float(np.mean(self.thermal_load_profile))
            stats['min_thermal_load'] = float(np.min(self.thermal_load_profile))
            stats['max_thermal_load'] = float(np.max(self.thermal_load_profile))
        
        return stats
    
    def get_conditions_at_time(self, time_index: int) -> Dict[str, Any]:
        """Get flight conditions at specific time index."""
        if time_index < 0 or time_index >= len(self.altitude_profile):
            raise IndexError("Time index out of range")
        
        conditions = {
            'altitude': float(self.altitude_profile[time_index]),
            'mach': float(self.mach_profile[time_index]),
            'thermal_load': float(self.thermal_load_profile[time_index])
        }
        
        if time_index < len(self.propulsion_mode_schedule):
            conditions['propulsion_mode'] = self.propulsion_mode_schedule[time_index]
        
        if time_index < len(self.cooling_system_schedule):
            conditions['cooling_active'] = self.cooling_system_schedule[time_index]
        
        return conditions



# New data structures for extreme hypersonic conditions (Mach 60+)

@dataclass
class PlasmaConditions:
    """Plasma conditions for extreme hypersonic flight analysis."""
    electron_density: float  # m⁻³
    electron_temperature: float  # K
    ion_temperature: float  # K
    magnetic_field: np.ndarray  # Tesla (3D vector)
    plasma_frequency: float  # Hz
    debye_length: float  # m
    plasma_regime: PlasmaRegime = PlasmaRegime.WEAKLY_IONIZED
    
    def __post_init__(self):
        """Validate plasma conditions after initialization."""
        if self.magnetic_field.shape != (3,):
            raise ValueError("Magnetic field must be a 3D vector")
    
    def validate_plasma_conditions(self) -> List[str]:
        """Validate plasma conditions and return list of errors."""
        errors = []
        
        # Physical constraints
        if self.electron_density <= 0:
            errors.append("Electron density must be positive")
        
        if self.electron_temperature <= 0:
            errors.append("Electron temperature must be positive")
        
        if self.ion_temperature <= 0:
            errors.append("Ion temperature must be positive")
        
        if self.plasma_frequency <= 0:
            errors.append("Plasma frequency must be positive")
        
        if self.debye_length <= 0:
            errors.append("Debye length must be positive")
        
        # Physical consistency checks
        if self.electron_temperature > 100000:  # 100,000 K
            errors.append("Electron temperature exceeds realistic limits")
        
        if self.ion_temperature > 100000:  # 100,000 K
            errors.append("Ion temperature exceeds realistic limits")
        
        if self.electron_density > 1e24:  # m⁻³
            errors.append("Electron density exceeds realistic limits")
        
        # Magnetic field magnitude check
        b_magnitude = np.linalg.norm(self.magnetic_field)
        if b_magnitude > 100:  # Tesla
            errors.append("Magnetic field magnitude exceeds realistic limits")
        
        return errors
    
    def calculate_plasma_beta(self) -> float:
        """Calculate plasma beta parameter (ratio of plasma to magnetic pressure)."""
        k_b = 1.380649e-23  # Boltzmann constant
        mu_0 = 4e-7 * np.pi  # Permeability of free space
        
        plasma_pressure = self.electron_density * k_b * (self.electron_temperature + self.ion_temperature)
        magnetic_pressure = np.linalg.norm(self.magnetic_field)**2 / (2 * mu_0)
        
        if magnetic_pressure == 0:
            return float('inf')
        
        return plasma_pressure / magnetic_pressure


@dataclass
class CombinedCyclePerformance:
    """Performance characteristics for combined-cycle propulsion systems."""
    air_breathing_thrust: float  # N
    rocket_thrust: float  # N
    transition_mach: float
    fuel_flow_air_breathing: float  # kg/s
    fuel_flow_rocket: float  # kg/s
    specific_impulse: float  # s
    propulsion_type: ExtremePropulsionType = ExtremePropulsionType.COMBINED_CYCLE_AIRBREATHING
    operating_altitude_range: tuple[float, float] = (30000.0, 100000.0)  # m
    
    def validate_performance(self) -> List[str]:
        """Validate combined-cycle performance data."""
        errors = []
        
        # Thrust validation
        if self.air_breathing_thrust < 0:
            errors.append("Air-breathing thrust cannot be negative")
        
        if self.rocket_thrust < 0:
            errors.append("Rocket thrust cannot be negative")
        
        if self.air_breathing_thrust == 0 and self.rocket_thrust == 0:
            errors.append("At least one thrust component must be positive")
        
        # Transition Mach validation
        if self.transition_mach <= 0:
            errors.append("Transition Mach number must be positive")
        
        if self.transition_mach < 3:
            errors.append("Transition Mach number seems too low for combined-cycle operation")
        
        if self.transition_mach > 25:
            errors.append("Transition Mach number exceeds typical combined-cycle limits")
        
        # Fuel flow validation
        if self.fuel_flow_air_breathing < 0:
            errors.append("Air-breathing fuel flow cannot be negative")
        
        if self.fuel_flow_rocket < 0:
            errors.append("Rocket fuel flow cannot be negative")
        
        # Specific impulse validation
        if self.specific_impulse <= 0:
            errors.append("Specific impulse must be positive")
        
        if self.specific_impulse > 10000:  # Unrealistic upper limit
            errors.append("Specific impulse exceeds realistic limits")
        
        # Altitude range validation
        if self.operating_altitude_range[0] >= self.operating_altitude_range[1]:
            errors.append("Invalid operating altitude range")
        
        if self.operating_altitude_range[0] < 0:
            errors.append("Operating altitude cannot be negative")
        
        return errors
    
    def calculate_total_thrust(self) -> float:
        """Calculate total thrust from both propulsion modes."""
        return self.air_breathing_thrust + self.rocket_thrust
    
    def calculate_thrust_to_weight_ratio(self, vehicle_mass: float) -> float:
        """Calculate thrust-to-weight ratio."""
        if vehicle_mass <= 0:
            raise ValueError("Vehicle mass must be positive")
        
        total_thrust = self.calculate_total_thrust()
        weight = vehicle_mass * 9.80665  # N
        
        return total_thrust / weight


@dataclass
class AblativeLayer:
    """Definition of an ablative layer in thermal protection system."""
    material_id: str
    thickness: float  # m
    ablation_rate: float  # m/s per MW/m²
    heat_of_ablation: float  # J/kg
    char_layer_conductivity: float  # W/(m⋅K)
    
    def validate_layer(self) -> List[str]:
        """Validate ablative layer properties."""
        errors = []
        
        if not self.material_id.strip():
            errors.append("Material ID cannot be empty")
        
        if self.thickness <= 0:
            errors.append("Layer thickness must be positive")
        
        if self.ablation_rate < 0:
            errors.append("Ablation rate cannot be negative")
        
        if self.heat_of_ablation <= 0:
            errors.append("Heat of ablation must be positive")
        
        if self.char_layer_conductivity <= 0:
            errors.append("Char layer conductivity must be positive")
        
        return errors


@dataclass
class CoolingChannel:
    """Definition of active cooling channel."""
    channel_id: str
    diameter: float  # m
    length: float  # m
    coolant_type: str
    mass_flow_rate: float  # kg/s
    inlet_temperature: float  # K
    pressure_drop: float  # Pa
    
    def validate_channel(self) -> List[str]:
        """Validate cooling channel properties."""
        errors = []
        
        if not self.channel_id.strip():
            errors.append("Channel ID cannot be empty")
        
        if self.diameter <= 0:
            errors.append("Channel diameter must be positive")
        
        if self.length <= 0:
            errors.append("Channel length must be positive")
        
        if not self.coolant_type.strip():
            errors.append("Coolant type cannot be empty")
        
        if self.mass_flow_rate <= 0:
            errors.append("Mass flow rate must be positive")
        
        if self.inlet_temperature <= 0:
            errors.append("Inlet temperature must be positive")
        
        if self.pressure_drop < 0:
            errors.append("Pressure drop cannot be negative")
        
        return errors


@dataclass
class InsulationLayer:
    """Definition of insulation layer in thermal protection system."""
    material_id: str
    thickness: float  # m
    thermal_conductivity: float  # W/(m⋅K)
    max_operating_temperature: float  # K
    
    def validate_layer(self) -> List[str]:
        """Validate insulation layer properties."""
        errors = []
        
        if not self.material_id.strip():
            errors.append("Material ID cannot be empty")
        
        if self.thickness <= 0:
            errors.append("Layer thickness must be positive")
        
        if self.thermal_conductivity <= 0:
            errors.append("Thermal conductivity must be positive")
        
        if self.max_operating_temperature <= 0:
            errors.append("Maximum operating temperature must be positive")
        
        return errors


@dataclass
class ThermalProtectionSystem:
    """Complete thermal protection system definition for extreme hypersonic conditions."""
    system_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ablative_layers: List[AblativeLayer] = field(default_factory=list)
    active_cooling_channels: List[CoolingChannel] = field(default_factory=list)
    insulation_layers: List[InsulationLayer] = field(default_factory=list)
    total_thickness: float = 0.0  # m
    total_mass: float = 0.0  # kg
    cooling_effectiveness: float = 0.0  # dimensionless (0-1)
    protection_type: ThermalProtectionType = ThermalProtectionType.PASSIVE_ABLATIVE
    max_heat_flux_capacity: float = 0.0  # W/m²
    
    def validate_system(self) -> List[str]:
        """Validate thermal protection system configuration."""
        errors = []
        
        if not self.system_id.strip():
            errors.append("System ID cannot be empty")
        
        # Must have at least one protection mechanism
        if not self.ablative_layers and not self.active_cooling_channels and not self.insulation_layers:
            errors.append("TPS must have at least one protection mechanism")
        
        # Validate individual layers
        for i, layer in enumerate(self.ablative_layers):
            layer_errors = layer.validate_layer()
            for error in layer_errors:
                errors.append(f"Ablative layer {i}: {error}")
        
        for i, channel in enumerate(self.active_cooling_channels):
            channel_errors = channel.validate_channel()
            for error in channel_errors:
                errors.append(f"Cooling channel {i}: {error}")
        
        for i, layer in enumerate(self.insulation_layers):
            layer_errors = layer.validate_layer()
            for error in layer_errors:
                errors.append(f"Insulation layer {i}: {error}")
        
        # System-level validation
        if self.total_thickness < 0:
            errors.append("Total thickness cannot be negative")
        
        if self.total_mass < 0:
            errors.append("Total mass cannot be negative")
        
        if self.cooling_effectiveness < 0 or self.cooling_effectiveness > 1:
            errors.append("Cooling effectiveness must be between 0 and 1")
        
        if self.max_heat_flux_capacity < 0:
            errors.append("Maximum heat flux capacity cannot be negative")
        
        return errors
    
    def calculate_total_thickness(self) -> float:
        """Calculate total thickness from all layers."""
        total = 0.0
        
        for layer in self.ablative_layers:
            total += layer.thickness
        
        for layer in self.insulation_layers:
            total += layer.thickness
        
        self.total_thickness = total
        return total
    
    def estimate_mass_per_area(self, area: float) -> float:
        """Estimate mass per unit area for the TPS."""
        if area <= 0:
            raise ValueError("Area must be positive")
        
        # This is a simplified estimation - would need material density data
        # for accurate calculation
        mass_per_area = self.total_thickness * 2000  # Assume 2000 kg/m³ average density
        return mass_per_area * area


@dataclass
class HypersonicMissionProfile:
    """Mission profile definition for extreme hypersonic flight (Mach 60+)."""
    profile_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    mission_name: str = ""
    altitude_profile: np.ndarray = field(default_factory=lambda: np.array([]))  # m
    mach_profile: np.ndarray = field(default_factory=lambda: np.array([]))
    thermal_load_profile: np.ndarray = field(default_factory=lambda: np.array([]))  # W/m²
    propulsion_mode_schedule: List[str] = field(default_factory=list)
    cooling_system_schedule: List[bool] = field(default_factory=list)
    plasma_conditions_profile: List[PlasmaConditions] = field(default_factory=list)
    mission_duration: float = 0.0  # s
    max_thermal_load: float = 0.0  # W/m²
    
    def validate_profile(self) -> List[str]:
        """Validate hypersonic mission profile."""
        errors = []
        
        if not self.mission_name.strip():
            errors.append("Mission name cannot be empty")
        
        # Check array lengths consistency
        profile_lengths = []
        if len(self.altitude_profile) > 0:
            profile_lengths.append(len(self.altitude_profile))
        if len(self.mach_profile) > 0:
            profile_lengths.append(len(self.mach_profile))
        if len(self.thermal_load_profile) > 0:
            profile_lengths.append(len(self.thermal_load_profile))
        
        if len(set(profile_lengths)) > 1:
            errors.append("All profile arrays must have the same length")
        
        if len(self.propulsion_mode_schedule) > 0 and len(self.propulsion_mode_schedule) != profile_lengths[0]:
            errors.append("Propulsion mode schedule length must match profile arrays")
        
        if len(self.cooling_system_schedule) > 0 and len(self.cooling_system_schedule) != profile_lengths[0]:
            errors.append("Cooling system schedule length must match profile arrays")
        
        # Physical constraints for extreme hypersonic flight
        if len(self.altitude_profile) > 0:
            if np.any(self.altitude_profile < 30000):  # Minimum altitude for Mach 60
                errors.append("Altitude profile contains values below minimum for Mach 60 flight (30 km)")
            
            if np.any(self.altitude_profile > 200000):  # 200 km reasonable upper limit
                errors.append("Altitude profile contains unrealistic values (>200 km)")
        
        if len(self.mach_profile) > 0:
            if np.any(self.mach_profile < 25):  # Minimum for extreme hypersonic
                errors.append("Mach profile contains values below extreme hypersonic regime (Mach 25)")
            
            if np.any(self.mach_profile > 100):  # Mach 100 as extreme upper limit
                errors.append("Mach profile contains unrealistic values (>Mach 100)")
        
        if len(self.thermal_load_profile) > 0:
            if np.any(self.thermal_load_profile < 0):
                errors.append("Thermal load profile cannot contain negative values")
            
            if np.any(self.thermal_load_profile > 1e9):  # 1 GW/m² as extreme upper limit
                errors.append("Thermal load profile contains unrealistic values (>1 GW/m²)")
        
        # Mission duration validation
        if self.mission_duration < 0:
            errors.append("Mission duration cannot be negative")
        
        if self.max_thermal_load < 0:
            errors.append("Maximum thermal load cannot be negative")
        
        # Validate plasma conditions if provided
        for i, plasma_cond in enumerate(self.plasma_conditions_profile):
            plasma_errors = plasma_cond.validate_plasma_conditions()
            for error in plasma_errors:
                errors.append(f"Plasma conditions {i}: {error}")
        
        return errors
    
    def calculate_profile_statistics(self) -> Dict[str, float]:
        """Calculate statistics for the hypersonic mission profile."""
        stats = {}
        
        if len(self.altitude_profile) > 0:
            stats['avg_altitude'] = float(np.mean(self.altitude_profile))
            stats['min_altitude'] = float(np.min(self.altitude_profile))
            stats['max_altitude'] = float(np.max(self.altitude_profile))
        
        if len(self.mach_profile) > 0:
            stats['avg_mach'] = float(np.mean(self.mach_profile))
            stats['min_mach'] = float(np.min(self.mach_profile))
            stats['max_mach'] = float(np.max(self.mach_profile))
        
        if len(self.thermal_load_profile) > 0:
            stats['avg_thermal_load'] = float(np.mean(self.thermal_load_profile))
            stats['min_thermal_load'] = float(np.min(self.thermal_load_profile))
            stats['max_thermal_load'] = float(np.max(self.thermal_load_profile))
            stats['peak_thermal_load'] = float(np.max(self.thermal_load_profile))
        
        # Calculate time in different flight regimes
        if len(self.mach_profile) > 0:
            extreme_hypersonic_time = np.sum(self.mach_profile >= 25)
            mach_60_plus_time = np.sum(self.mach_profile >= 60)
            
            stats['extreme_hypersonic_time_fraction'] = float(extreme_hypersonic_time / len(self.mach_profile))
            stats['mach_60_plus_time_fraction'] = float(mach_60_plus_time / len(self.mach_profile))
        
        return stats
    
    def get_conditions_at_time(self, time_index: int) -> Dict[str, Any]:
        """Get flight conditions at specific time index."""
        if time_index < 0 or time_index >= len(self.altitude_profile):
            raise IndexError("Time index out of range")
        
        conditions = {
            'altitude': float(self.altitude_profile[time_index]),
            'mach': float(self.mach_profile[time_index]),
            'thermal_load': float(self.thermal_load_profile[time_index])
        }
        
        if time_index < len(self.propulsion_mode_schedule):
            conditions['propulsion_mode'] = self.propulsion_mode_schedule[time_index]
        
        if time_index < len(self.cooling_system_schedule):
            conditions['cooling_active'] = self.cooling_system_schedule[time_index]
        
        if time_index < len(self.plasma_conditions_profile):
            conditions['plasma_conditions'] = self.plasma_conditions_profile[time_index]
        
        return conditions
    
    def requires_plasma_modeling(self) -> bool:
        """Check if mission profile requires plasma flow modeling."""
        if len(self.mach_profile) == 0:
            return False
        
        # Plasma effects become significant above Mach 25
        return np.any(self.mach_profile >= 25)
    
    def requires_active_cooling(self) -> bool:
        """Check if mission profile requires active cooling systems."""
        if len(self.thermal_load_profile) == 0:
            return False
        
        # Active cooling typically required above 10 MW/m²
        return np.any(self.thermal_load_profile >= 1e7)

@dataclass
class PlasmaConditions:
    """Plasma conditions for extreme hypersonic flight."""
    electron_density: float  # m⁻³
    electron_temperature: float  # K
    ion_temperature: float  # K
    magnetic_field: np.ndarray  # Tesla
    plasma_frequency: float  # Hz
    debye_length: float  # m
    ionization_fraction: float  # dimensionless
    regime: PlasmaRegime = PlasmaRegime.WEAKLY_IONIZED


@dataclass
class GasMixture:
    """Gas mixture composition for plasma calculations."""
    species: Dict[str, float]  # species name -> mole fraction
    temperature: float  # K
    pressure: float  # Pa
    total_density: float  # kg/m³


@dataclass
class ElectromagneticProperties:
    """Electromagnetic properties of plasma flow."""
    conductivity: float  # S/m
    hall_parameter: float  # dimensionless
    magnetic_reynolds_number: float  # dimensionless
    electric_field: np.ndarray  # V/m
    current_density: np.ndarray  # A/m²
    lorentz_force_density: np.ndarray  # N/m³


@dataclass
class MagneticFieldConfiguration:
    """Magnetic field configuration for MHD analysis."""
    field_strength: np.ndarray  # Tesla
    field_gradient: np.ndarray  # Tesla/m
    field_type: str  # 'uniform', 'dipole', 'custom'
    source_location: Optional[np.ndarray] = None  # m


@dataclass
class CombinedCyclePerformance:
    """Combined-cycle propulsion performance data."""
    air_breathing_thrust: float  # N
    rocket_thrust: float  # N
    transition_mach: float
    fuel_flow_air_breathing: float  # kg/s
    fuel_flow_rocket: float  # kg/s
    specific_impulse: float  # s


@dataclass
class ThermalProtectionSystem:
    """Thermal protection system configuration."""
    ablative_layers: List[Dict[str, Any]]  # List of ablative layer specifications
    active_cooling_channels: List[Dict[str, Any]]  # List of cooling channel specifications
    insulation_layers: List[Dict[str, Any]]  # List of insulation layer specifications
    total_thickness: float  # m
    total_mass: float  # kg
    cooling_effectiveness: float


@dataclass
class HypersonicMissionProfile:
    """Mission profile for hypersonic flight."""
    altitude_profile: np.ndarray  # m
    mach_profile: np.ndarray
    thermal_load_profile: np.ndarray  # W/m²
    propulsion_mode_schedule: List[str]
    cooling_system_schedule: List[bool]