"""Interface validator for module compatibility checking."""

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from ...common.data_models import (
    Module, ModuleInterface, AircraftConfiguration, BasePlatform,
    ElectricalInterface, MechanicalInterface
)
from ...common.enums import ModuleType
from ...core.errors import ValidationError, ConfigurationError


class CompatibilityLevel(Enum):
    """Levels of module compatibility."""
    FULLY_COMPATIBLE = "fully_compatible"
    COMPATIBLE_WITH_ADAPTER = "compatible_with_adapter"
    INCOMPATIBLE = "incompatible"
    REQUIRES_MODIFICATION = "requires_modification"


@dataclass
class CompatibilityResult:
    """Result of compatibility check."""
    level: CompatibilityLevel
    issues: List[str]
    warnings: List[str]
    required_adapters: List[str]
    confidence_score: float  # 0.0 to 1.0


@dataclass
class InterfaceConflict:
    """Represents a conflict between interfaces."""
    module1_id: str
    module2_id: str
    interface1_id: str
    interface2_id: str
    conflict_type: str
    description: str
    severity: str  # 'critical', 'warning', 'info'


class InterfaceValidator:
    """Validates module compatibility and interface connections."""
    
    def __init__(self):
        """Initialize the interface validator."""
        self._compatibility_rules: Dict[str, Any] = {}
        self._interface_standards: Dict[str, Dict[str, Any]] = {}
        self._load_default_rules()
    
    def validate_module_compatibility(self, module1: Module, module2: Module) -> CompatibilityResult:
        """Validate compatibility between two modules.
        
        Args:
            module1: First module
            module2: Second module
            
        Returns:
            CompatibilityResult with detailed compatibility information
        """
        issues = []
        warnings = []
        required_adapters = []
        confidence_score = 1.0
        
        # Check basic module type compatibility
        type_compat = self._check_module_type_compatibility(module1, module2)
        if not type_compat['compatible']:
            issues.extend(type_compat['issues'])
            confidence_score *= 0.5
        
        # Check physical compatibility
        physical_compat = self._check_physical_compatibility(module1, module2)
        if not physical_compat['compatible']:
            issues.extend(physical_compat['issues'])
            confidence_score *= 0.7
        else:
            warnings.extend(physical_compat['warnings'])
        
        # Check electrical interface compatibility
        electrical_compat = self._check_electrical_compatibility(module1, module2)
        if not electrical_compat['compatible']:
            issues.extend(electrical_compat['issues'])
            confidence_score *= 0.8
        required_adapters.extend(electrical_compat['adapters'])
        
        # Check mechanical interface compatibility
        mechanical_compat = self._check_mechanical_compatibility(module1, module2)
        if not mechanical_compat['compatible']:
            issues.extend(mechanical_compat['issues'])
            confidence_score *= 0.8
        required_adapters.extend(mechanical_compat['adapters'])
        
        # Check compatibility requirements
        req_compat = self._check_compatibility_requirements(module1, module2)
        if not req_compat['compatible']:
            issues.extend(req_compat['issues'])
            confidence_score *= 0.6
        
        # Determine overall compatibility level
        if not issues:
            if required_adapters:
                level = CompatibilityLevel.COMPATIBLE_WITH_ADAPTER
            else:
                level = CompatibilityLevel.FULLY_COMPATIBLE
        elif confidence_score > 0.5:
            level = CompatibilityLevel.REQUIRES_MODIFICATION
        else:
            level = CompatibilityLevel.INCOMPATIBLE
        
        return CompatibilityResult(
            level=level,
            issues=issues,
            warnings=warnings,
            required_adapters=required_adapters,
            confidence_score=confidence_score
        )
    
    def validate_configuration_interfaces(self, config: AircraftConfiguration) -> List[InterfaceConflict]:
        """Validate all interfaces in an aircraft configuration.
        
        Args:
            config: Aircraft configuration to validate
            
        Returns:
            List of interface conflicts found
        """
        conflicts = []
        
        if not config.modules:
            return conflicts
        
        # Check pairwise module compatibility
        for i, module1 in enumerate(config.modules):
            for j, module2 in enumerate(config.modules[i+1:], i+1):
                module_conflicts = self._check_module_pair_conflicts(module1, module2)
                conflicts.extend(module_conflicts)
        
        # Check platform attachment compatibility
        if config.base_platform:
            platform_conflicts = self._check_platform_attachment_conflicts(config)
            conflicts.extend(platform_conflicts)
        
        # Check power distribution conflicts
        power_conflicts = self._check_power_distribution_conflicts(config)
        conflicts.extend(power_conflicts)
        
        # Check data network conflicts
        data_conflicts = self._check_data_network_conflicts(config)
        conflicts.extend(data_conflicts)
        
        return conflicts
    
    def suggest_interface_solutions(self, conflict: InterfaceConflict) -> List[Dict[str, Any]]:
        """Suggest solutions for an interface conflict.
        
        Args:
            conflict: Interface conflict to resolve
            
        Returns:
            List of suggested solutions
        """
        solutions = []
        
        if conflict.conflict_type == "voltage_mismatch":
            solutions.append({
                'type': 'voltage_converter',
                'description': 'Add voltage converter/regulator',
                'cost_estimate': 'low',
                'complexity': 'low'
            })
        
        elif conflict.conflict_type == "mechanical_incompatible":
            solutions.append({
                'type': 'mechanical_adapter',
                'description': 'Design custom mechanical adapter',
                'cost_estimate': 'medium',
                'complexity': 'medium'
            })
        
        elif conflict.conflict_type == "protocol_mismatch":
            solutions.append({
                'type': 'protocol_bridge',
                'description': 'Add protocol conversion bridge',
                'cost_estimate': 'medium',
                'complexity': 'high'
            })
        
        elif conflict.conflict_type == "power_overload":
            solutions.extend([
                {
                    'type': 'power_upgrade',
                    'description': 'Upgrade power generation capacity',
                    'cost_estimate': 'high',
                    'complexity': 'high'
                },
                {
                    'type': 'load_management',
                    'description': 'Implement intelligent load management',
                    'cost_estimate': 'medium',
                    'complexity': 'medium'
                }
            ])
        
        elif conflict.conflict_type == "thermal_interference":
            solutions.append({
                'type': 'thermal_management',
                'description': 'Add thermal isolation/cooling',
                'cost_estimate': 'medium',
                'complexity': 'medium'
            })
        
        return solutions
    
    def get_compatibility_matrix(self, modules: List[Module]) -> Dict[str, Dict[str, CompatibilityResult]]:
        """Generate compatibility matrix for a list of modules.
        
        Args:
            modules: List of modules to analyze
            
        Returns:
            Matrix of compatibility results
        """
        matrix = {}
        
        for i, module1 in enumerate(modules):
            matrix[module1.module_id] = {}
            
            for j, module2 in enumerate(modules):
                if i != j:
                    result = self.validate_module_compatibility(module1, module2)
                    matrix[module1.module_id][module2.module_id] = result
                else:
                    # Self-compatibility is always full
                    matrix[module1.module_id][module2.module_id] = CompatibilityResult(
                        level=CompatibilityLevel.FULLY_COMPATIBLE,
                        issues=[],
                        warnings=[],
                        required_adapters=[],
                        confidence_score=1.0
                    )
        
        return matrix
    
    def _check_module_type_compatibility(self, module1: Module, module2: Module) -> Dict[str, Any]:
        """Check basic module type compatibility."""
        result = {'compatible': True, 'issues': []}
        
        # Check for explicitly incompatible types
        incompatible_pairs = [
            (ModuleType.COCKPIT, ModuleType.COCKPIT),  # Only one cockpit allowed
            (ModuleType.PROPULSION, ModuleType.PROPULSION)  # Typically only one main propulsion
        ]
        
        type_pair = (module1.module_type, module2.module_type)
        reverse_pair = (module2.module_type, module1.module_type)
        
        if type_pair in incompatible_pairs or reverse_pair in incompatible_pairs:
            result['compatible'] = False
            result['issues'].append(f"Module types {module1.module_type.name} and {module2.module_type.name} are typically incompatible")
        
        return result
    
    def _check_physical_compatibility(self, module1: Module, module2: Module) -> Dict[str, Any]:
        """Check physical compatibility between modules."""
        result = {'compatible': True, 'issues': [], 'warnings': []}
        
        if not module1.physical_properties or not module2.physical_properties:
            result['warnings'].append("Missing physical properties for complete compatibility check")
            return result
        
        # Check for physical interference (simplified check)
        # In a real implementation, this would involve 3D collision detection
        
        # Check mass distribution effects
        total_mass = module1.physical_properties.mass + module2.physical_properties.mass
        if total_mass > 5000.0:  # Example threshold
            result['warnings'].append(f"Combined mass ({total_mass:.1f} kg) may affect aircraft balance")
        
        # Check center of gravity effects
        cg1 = module1.physical_properties.center_of_gravity
        cg2 = module2.physical_properties.center_of_gravity
        
        # Calculate distance between centers of gravity
        cg_distance = ((cg1[0] - cg2[0])**2 + (cg1[1] - cg2[1])**2 + (cg1[2] - cg2[2])**2)**0.5
        
        if cg_distance < 2.0:  # Example minimum separation
            result['warnings'].append("Modules may be too close together - check for physical interference")
        
        return result
    
    def _check_electrical_compatibility(self, module1: Module, module2: Module) -> Dict[str, Any]:
        """Check electrical interface compatibility."""
        result = {'compatible': True, 'issues': [], 'adapters': []}
        
        # Check for voltage compatibility
        voltages1 = {ei.voltage for ei in module1.electrical_interfaces}
        voltages2 = {ei.voltage for ei in module2.electrical_interfaces}
        
        # Check if modules can share power buses
        common_voltages = voltages1.intersection(voltages2)
        if voltages1 and voltages2 and not common_voltages:
            result['adapters'].append("voltage_converter")
        
        # Check for protocol compatibility
        protocols1 = {ei.protocol for ei in module1.electrical_interfaces if ei.protocol}
        protocols2 = {ei.protocol for ei in module2.electrical_interfaces if ei.protocol}
        
        if protocols1 and protocols2:
            common_protocols = protocols1.intersection(protocols2)
            if not common_protocols:
                result['adapters'].append("protocol_bridge")
        
        # Check for electrical interference
        total_power1 = sum(ei.power_consumption for ei in module1.electrical_interfaces)
        total_power2 = sum(ei.power_consumption for ei in module2.electrical_interfaces)
        
        if total_power1 > 10000 and total_power2 > 10000:  # High power modules
            result['issues'].append("High power modules may cause electrical interference")
            result['compatible'] = False
        
        return result
    
    def _check_mechanical_compatibility(self, module1: Module, module2: Module) -> Dict[str, Any]:
        """Check mechanical interface compatibility."""
        result = {'compatible': True, 'issues': [], 'adapters': []}
        
        # Check attachment types
        attach_types1 = {mi.attachment_type for mi in module1.mechanical_interfaces}
        attach_types2 = {mi.attachment_type for mi in module2.mechanical_interfaces}
        
        if attach_types1 and attach_types2:
            common_types = attach_types1.intersection(attach_types2)
            if not common_types:
                result['adapters'].append("mechanical_adapter")
        
        # Check load compatibility
        for mi1 in module1.mechanical_interfaces:
            for mi2 in module2.mechanical_interfaces:
                if mi1.attachment_type == mi2.attachment_type:
                    # Check if load capacities are compatible (allow reasonable tolerance)
                    for i in range(3):  # Check Fx, Fy, Fz
                        # Only flag as incompatible if one is significantly higher than the other
                        max_load = max(abs(mi1.load_capacity[i]), abs(mi2.load_capacity[i]))
                        min_load = min(abs(mi1.load_capacity[i]), abs(mi2.load_capacity[i]))
                        if max_load > min_load * 3.0:  # More tolerant threshold
                            result['issues'].append(f"Load capacity mismatch in {['X', 'Y', 'Z'][i]} direction")
                            result['compatible'] = False
        
        return result
    
    def _check_compatibility_requirements(self, module1: Module, module2: Module) -> Dict[str, Any]:
        """Check module-specific compatibility requirements."""
        result = {'compatible': True, 'issues': []}
        
        # Check module1's requirements against module2
        for req in module1.compatibility_requirements:
            if req.startswith("incompatible_with:"):
                incompatible_type = req.split(":")[1]
                if module2.module_type.name == incompatible_type:
                    result['compatible'] = False
                    result['issues'].append(f"Module {module1.name} is incompatible with {incompatible_type} modules")
            
            elif req.startswith("requires:"):
                required_feature = req.split(":")[1]
                if not self._module_has_feature(module2, required_feature):
                    result['compatible'] = False
                    result['issues'].append(f"Module {module1.name} requires {required_feature} which {module2.name} doesn't provide")
        
        # Check module2's requirements against module1
        for req in module2.compatibility_requirements:
            if req.startswith("incompatible_with:"):
                incompatible_type = req.split(":")[1]
                if module1.module_type.name == incompatible_type:
                    result['compatible'] = False
                    result['issues'].append(f"Module {module2.name} is incompatible with {incompatible_type} modules")
            
            elif req.startswith("requires:"):
                required_feature = req.split(":")[1]
                if not self._module_has_feature(module1, required_feature):
                    result['compatible'] = False
                    result['issues'].append(f"Module {module2.name} requires {required_feature} which {module1.name} doesn't provide")
        
        return result
    
    def _check_module_pair_conflicts(self, module1: Module, module2: Module) -> List[InterfaceConflict]:
        """Check for conflicts between a pair of modules."""
        conflicts = []
        
        # Check electrical interface conflicts
        for ei1 in module1.electrical_interfaces:
            for ei2 in module2.electrical_interfaces:
                if abs(ei1.voltage - ei2.voltage) > 1.0 and ei1.interface_id == ei2.interface_id:
                    conflicts.append(InterfaceConflict(
                        module1_id=module1.module_id,
                        module2_id=module2.module_id,
                        interface1_id=ei1.interface_id,
                        interface2_id=ei2.interface_id,
                        conflict_type="voltage_mismatch",
                        description=f"Voltage mismatch: {ei1.voltage}V vs {ei2.voltage}V",
                        severity="warning"
                    ))
        
        return conflicts
    
    def _check_platform_attachment_conflicts(self, config: AircraftConfiguration) -> List[InterfaceConflict]:
        """Check for platform attachment conflicts."""
        conflicts = []
        
        if not config.base_platform or not config.modules:
            return conflicts
        
        # Check if there are enough attachment points
        modules_needing_attachment = [m for m in config.modules if m.mechanical_interfaces]
        available_points = len(config.base_platform.attachment_points)
        
        if len(modules_needing_attachment) > available_points:
            conflicts.append(InterfaceConflict(
                module1_id="platform",
                module2_id="multiple",
                interface1_id="attachment_points",
                interface2_id="mechanical_interfaces",
                conflict_type="insufficient_attachment_points",
                description=f"Need {len(modules_needing_attachment)} attachment points, only {available_points} available",
                severity="critical"
            ))
        
        return conflicts
    
    def _check_power_distribution_conflicts(self, config: AircraftConfiguration) -> List[InterfaceConflict]:
        """Check for power distribution conflicts."""
        conflicts = []
        
        if not config.base_platform or not config.modules:
            return conflicts
        
        # Calculate total power requirement
        total_power_required = sum(
            sum(ei.power_consumption for ei in module.electrical_interfaces)
            for module in config.modules
        )
        
        if total_power_required > config.base_platform.power_generation_capacity:
            conflicts.append(InterfaceConflict(
                module1_id="platform",
                module2_id="multiple",
                interface1_id="power_generation",
                interface2_id="power_consumption",
                conflict_type="power_overload",
                description=f"Power requirement ({total_power_required:.1f}W) exceeds generation capacity ({config.base_platform.power_generation_capacity:.1f}W)",
                severity="critical"
            ))
        
        return conflicts
    
    def _check_data_network_conflicts(self, config: AircraftConfiguration) -> List[InterfaceConflict]:
        """Check for data network conflicts."""
        conflicts = []
        
        # Check for protocol mismatches
        protocols_used = set()
        for module in config.modules:
            for ei in module.electrical_interfaces:
                if ei.protocol:
                    protocols_used.add(ei.protocol)
        
        # If more than 3 different protocols, suggest standardization
        if len(protocols_used) > 3:
            conflicts.append(InterfaceConflict(
                module1_id="multiple",
                module2_id="multiple",
                interface1_id="data_protocols",
                interface2_id="data_protocols",
                conflict_type="protocol_fragmentation",
                description=f"Too many different protocols ({len(protocols_used)}) may complicate integration",
                severity="warning"
            ))
        
        return conflicts
    
    def _module_has_feature(self, module: Module, feature: str) -> bool:
        """Check if a module has a specific feature."""
        # This is a simplified implementation
        # In practice, this would check module capabilities, interfaces, etc.
        
        feature_map = {
            'cooling': lambda m: any('cooling' in ei.protocol.lower() if ei.protocol else False for ei in m.electrical_interfaces),
            'high_power': lambda m: sum(ei.power_consumption for ei in m.electrical_interfaces) > 1000,
            'avionics': lambda m: m.module_type == ModuleType.AVIONICS,
            'life_support': lambda m: 'life_support' in m.performance_characteristics
        }
        
        if feature in feature_map:
            return feature_map[feature](module)
        
        # Default: check if feature is mentioned in compatibility requirements
        return feature in module.compatibility_requirements
    
    def _load_default_rules(self) -> None:
        """Load default compatibility rules and interface standards."""
        # Define standard electrical interfaces
        self._interface_standards['electrical'] = {
            'MIL-STD-1553': {'voltage': 28.0, 'protocol': 'MIL-STD-1553'},
            'ARINC-429': {'voltage': 5.0, 'protocol': 'ARINC-429'},
            'Ethernet': {'voltage': 48.0, 'protocol': 'Ethernet'},
            'CAN-Bus': {'voltage': 12.0, 'protocol': 'CAN-Bus'}
        }
        
        # Define standard mechanical interfaces
        self._interface_standards['mechanical'] = {
            'NATO-Standard': {'attachment_type': 'NATO-Standard'},
            'MIL-STD-8591': {'attachment_type': 'MIL-STD-8591'},
            'Custom-Hardpoint': {'attachment_type': 'Custom-Hardpoint'}
        }
        
        # Define compatibility rules
        self._compatibility_rules = {
            'module_type_exclusions': [
                (ModuleType.COCKPIT, ModuleType.COCKPIT),
                (ModuleType.PROPULSION, ModuleType.PROPULSION)
            ],
            'power_limits': {
                'max_single_module': 15000.0,  # W
                'max_total_system': 50000.0    # W
            },
            'thermal_limits': {
                'max_heat_density': 1000.0,    # W/m³
                'min_separation': 0.5          # m
            }
        }