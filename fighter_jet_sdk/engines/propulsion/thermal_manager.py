"""Thermal Management System for high-power electronics and directed energy systems."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math
from enum import Enum

from ...core.logging import get_engine_logger


class CoolantType(Enum):
    """Coolant type enumeration."""
    AIR = "air"
    LIQUID_COOLANT = "liquid_coolant"
    PHASE_CHANGE = "phase_change"
    THERMOELECTRIC = "thermoelectric"
    LIQUID_METAL = "liquid_metal"
    CRYOGENIC = "cryogenic"


class HeatExchangerType(Enum):
    """Heat exchanger type enumeration."""
    PLATE_FIN = "plate_fin"
    TUBE_FIN = "tube_fin"
    MICROCHANNEL = "microchannel"
    HEAT_PIPE = "heat_pipe"
    VAPOR_CHAMBER = "vapor_chamber"
    REGENERATIVE = "regenerative"


@dataclass
class ThermalLoad:
    """Thermal load specification."""
    load_id: str
    name: str
    power_dissipation: float  # W
    operating_temperature_range: Tuple[float, float]  # K (min, max)
    critical_temperature: float  # K
    duty_cycle: float  # 0.0 to 1.0
    location: Tuple[float, float, float]  # x, y, z coordinates
    thermal_resistance: float = 0.1  # K/W (junction to case)
    transient_response_time: float = 1.0  # seconds


@dataclass
class CoolantProperties:
    """Coolant properties definition."""
    coolant_type: CoolantType
    density: float  # kg/m³
    specific_heat: float  # J/(kg·K)
    thermal_conductivity: float  # W/(m·K)
    viscosity: float  # Pa·s
    operating_temp_range: Tuple[float, float]  # K
    phase_change_temp: Optional[float] = None  # K
    latent_heat: Optional[float] = None  # J/kg


@dataclass
class HeatExchangerSpec:
    """Heat exchanger specification."""
    exchanger_id: str
    exchanger_type: HeatExchangerType
    effectiveness: float  # 0.0 to 1.0
    heat_capacity: float  # W/K
    pressure_drop: float  # Pa
    mass: float  # kg
    volume: float  # m³
    max_heat_transfer: float  # W
    operating_temp_range: Tuple[float, float]  # K


@dataclass
class ThermalNetworkNode:
    """Node in thermal network."""
    node_id: str
    temperature: float  # K
    thermal_mass: float  # J/K
    heat_generation: float = 0.0  # W
    connected_nodes: List[str] = field(default_factory=list)
    thermal_resistances: Dict[str, float] = field(default_factory=dict)  # K/W to connected nodes


@dataclass
class ThermalSystemConfig:
    """Complete thermal system configuration."""
    system_id: str
    thermal_loads: List[ThermalLoad]
    heat_exchangers: List[HeatExchangerSpec]
    coolant_loops: Dict[str, CoolantProperties]
    ambient_temperature: float = 288.15  # K
    max_coolant_temperature: float = 373.15  # K
    safety_margin: float = 0.2  # 20% margin


class ThermalManager:
    """Advanced thermal management system for high-power electronics."""
    
    def __init__(self):
        """Initialize thermal manager."""
        self.logger = get_engine_logger('propulsion.thermal')
        
        # Thermal network
        self.thermal_nodes: Dict[str, ThermalNetworkNode] = {}
        self.thermal_systems: Dict[str, ThermalSystemConfig] = {}
        
        # Performance tracking
        self.temperature_history: Dict[str, List[Tuple[float, float]]] = {}  # time, temperature
        
        # Load default coolant properties
        self._initialize_coolant_database()
    
    def _initialize_coolant_database(self) -> None:
        """Initialize database of coolant properties."""
        self.coolant_database = {
            CoolantType.AIR: CoolantProperties(
                coolant_type=CoolantType.AIR,
                density=1.225,  # kg/m³ at STP
                specific_heat=1005.0,  # J/(kg·K)
                thermal_conductivity=0.026,  # W/(m·K)
                viscosity=1.8e-5,  # Pa·s
                operating_temp_range=(200.0, 800.0)
            ),
            CoolantType.LIQUID_COOLANT: CoolantProperties(
                coolant_type=CoolantType.LIQUID_COOLANT,
                density=1050.0,  # kg/m³ (ethylene glycol mix)
                specific_heat=3500.0,  # J/(kg·K)
                thermal_conductivity=0.4,  # W/(m·K)
                viscosity=0.002,  # Pa·s
                operating_temp_range=(253.15, 393.15)
            ),
            CoolantType.PHASE_CHANGE: CoolantProperties(
                coolant_type=CoolantType.PHASE_CHANGE,
                density=1000.0,  # kg/m³ (water)
                specific_heat=4186.0,  # J/(kg·K)
                thermal_conductivity=0.6,  # W/(m·K)
                viscosity=0.001,  # Pa·s
                operating_temp_range=(273.15, 373.15),
                phase_change_temp=373.15,  # K (boiling point)
                latent_heat=2.26e6  # J/kg (latent heat of vaporization)
            ),
            CoolantType.LIQUID_METAL: CoolantProperties(
                coolant_type=CoolantType.LIQUID_METAL,
                density=6500.0,  # kg/m³ (liquid sodium)
                specific_heat=1230.0,  # J/(kg·K)
                thermal_conductivity=85.0,  # W/(m·K)
                viscosity=0.0007,  # Pa·s
                operating_temp_range=(371.0, 1156.0)
            ),
            CoolantType.CRYOGENIC: CoolantProperties(
                coolant_type=CoolantType.CRYOGENIC,
                density=808.0,  # kg/m³ (liquid nitrogen)
                specific_heat=2040.0,  # J/(kg·K)
                thermal_conductivity=0.14,  # W/(m·K)
                viscosity=0.00016,  # Pa·s
                operating_temp_range=(63.15, 126.2),
                phase_change_temp=77.35,  # K (boiling point)
                latent_heat=2.0e5  # J/kg
            )
        }
    
    def design_thermal_system(self, thermal_loads: List[ThermalLoad],
                            design_requirements: Dict[str, Any]) -> ThermalSystemConfig:
        """Design thermal management system for given loads and requirements."""
        self.logger.info(f"Designing thermal system for {len(thermal_loads)} thermal loads")
        
        # Calculate total heat load
        total_power = sum(load.power_dissipation * load.duty_cycle for load in thermal_loads)
        peak_power = sum(load.power_dissipation for load in thermal_loads)
        
        self.logger.info(f"Total thermal load: {total_power:.1f}W average, {peak_power:.1f}W peak")
        
        # Select appropriate cooling technology
        coolant_type = self._select_coolant_type(peak_power, thermal_loads)
        
        # Design heat exchangers
        heat_exchangers = self._design_heat_exchangers(thermal_loads, coolant_type, design_requirements)
        
        # Create coolant loops
        coolant_loops = self._design_coolant_loops(thermal_loads, coolant_type, heat_exchangers)
        
        # Create system configuration
        system_config = ThermalSystemConfig(
            system_id=f"thermal_system_{len(self.thermal_systems)}",
            thermal_loads=thermal_loads,
            heat_exchangers=heat_exchangers,
            coolant_loops=coolant_loops,
            ambient_temperature=design_requirements.get('ambient_temperature', 288.15),
            max_coolant_temperature=design_requirements.get('max_coolant_temperature', 373.15),
            safety_margin=design_requirements.get('safety_margin', 0.2)
        )
        
        # Validate design
        validation_results = self._validate_thermal_design(system_config)
        if validation_results:
            self.logger.warning(f"Design validation issues: {validation_results}")
        
        self.thermal_systems[system_config.system_id] = system_config
        return system_config
    
    def _select_coolant_type(self, peak_power: float, thermal_loads: List[ThermalLoad]) -> CoolantType:
        """Select appropriate coolant type based on power and temperature requirements."""
        if not thermal_loads:
            return CoolantType.AIR
        
        max_temp = max(load.critical_temperature for load in thermal_loads)
        
        # Decision logic based on power density and temperature
        if peak_power > 100000:  # > 100 kW
            if max_temp > 500:
                return CoolantType.LIQUID_METAL
            else:
                return CoolantType.PHASE_CHANGE
        elif peak_power > 50000:  # > 50 kW
            if max_temp > 400:
                return CoolantType.LIQUID_COOLANT
            else:
                return CoolantType.PHASE_CHANGE
        elif peak_power > 10000:  # > 10 kW
            return CoolantType.LIQUID_COOLANT
        else:
            return CoolantType.AIR
    
    def _design_heat_exchangers(self, thermal_loads: List[ThermalLoad],
                              coolant_type: CoolantType,
                              requirements: Dict[str, Any]) -> List[HeatExchangerSpec]:
        """Design heat exchangers for thermal loads."""
        heat_exchangers = []
        
        # Group loads by location and power level
        load_groups = self._group_thermal_loads(thermal_loads)
        
        for group_id, loads in load_groups.items():
            total_power = sum(load.power_dissipation for load in loads)
            max_temp = max(load.critical_temperature for load in loads)
            
            # Select heat exchanger type
            if total_power > 50000:  # High power
                exchanger_type = HeatExchangerType.MICROCHANNEL
                effectiveness = 0.9
            elif total_power > 10000:  # Medium power
                exchanger_type = HeatExchangerType.PLATE_FIN
                effectiveness = 0.85
            elif coolant_type == CoolantType.PHASE_CHANGE:
                exchanger_type = HeatExchangerType.HEAT_PIPE
                effectiveness = 0.95
            else:
                exchanger_type = HeatExchangerType.TUBE_FIN
                effectiveness = 0.8
            
            # Size heat exchanger
            temp_diff = max(10.0, max_temp - requirements.get('ambient_temperature', 288.15))  # Minimum 10K difference
            heat_capacity = total_power / temp_diff
            
            # Estimate physical properties
            volume = total_power / 1e6  # Rough estimate: 1 MW/m³
            mass = volume * 2700  # Aluminum density
            pressure_drop = 1000 + total_power * 0.01  # Empirical correlation
            
            heat_exchanger = HeatExchangerSpec(
                exchanger_id=f"hx_{group_id}",
                exchanger_type=exchanger_type,
                effectiveness=effectiveness,
                heat_capacity=heat_capacity,
                pressure_drop=pressure_drop,
                mass=mass,
                volume=volume,
                max_heat_transfer=total_power * 1.5,  # 50% margin for safety
                operating_temp_range=(requirements.get('ambient_temperature', 288.15), max_temp)
            )
            
            heat_exchangers.append(heat_exchanger)
        
        return heat_exchangers
    
    def _group_thermal_loads(self, thermal_loads: List[ThermalLoad]) -> Dict[str, List[ThermalLoad]]:
        """Group thermal loads by proximity and characteristics."""
        groups = {}
        
        for load in thermal_loads:
            # Simple grouping by power level and location
            x, y, z = load.location
            power_class = "high" if load.power_dissipation > 10000 else "medium" if load.power_dissipation > 1000 else "low"
            location_key = f"{int(x/10)}_{int(y/10)}_{int(z/10)}"  # 10m grid
            group_key = f"{power_class}_{location_key}"
            
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(load)
        
        return groups
    
    def _design_coolant_loops(self, thermal_loads: List[ThermalLoad],
                            coolant_type: CoolantType,
                            heat_exchangers: List[HeatExchangerSpec]) -> Dict[str, CoolantProperties]:
        """Design coolant loops for thermal system."""
        coolant_loops = {}
        
        # Primary coolant loop
        primary_coolant = self.coolant_database[coolant_type]
        coolant_loops["primary"] = primary_coolant
        
        # Secondary loops for high-temperature loads
        high_temp_loads = [load for load in thermal_loads if load.critical_temperature > 400]
        if high_temp_loads and coolant_type != CoolantType.LIQUID_METAL:
            # Add high-temperature secondary loop
            secondary_coolant = self.coolant_database[CoolantType.LIQUID_METAL]
            coolant_loops["secondary_high_temp"] = secondary_coolant
        
        # Cryogenic loop for very high power density loads or low temperature requirements
        very_high_power_loads = [load for load in thermal_loads if load.power_dissipation > 50000]
        low_temp_loads = [load for load in thermal_loads if load.critical_temperature < 200.0]
        if very_high_power_loads or low_temp_loads:
            cryo_coolant = self.coolant_database[CoolantType.CRYOGENIC]
            coolant_loops["cryogenic"] = cryo_coolant
        
        return coolant_loops
    
    def analyze_thermal_performance(self, system_config: ThermalSystemConfig,
                                  operating_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze thermal performance of designed system."""
        # Build thermal network
        thermal_network = self._build_thermal_network(system_config)
        
        # Solve steady-state temperatures
        steady_state_temps = self._solve_steady_state(thermal_network, operating_conditions)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            system_config, steady_state_temps, operating_conditions
        )
        
        return {
            'steady_state_temperatures': steady_state_temps,
            'performance_metrics': performance_metrics,
            'thermal_network': thermal_network
        }
    
    def _build_thermal_network(self, system_config: ThermalSystemConfig) -> Dict[str, ThermalNetworkNode]:
        """Build thermal network model from system configuration."""
        network = {}
        
        # Create nodes for thermal loads
        for load in system_config.thermal_loads:
            node = ThermalNetworkNode(
                node_id=load.load_id,
                temperature=load.operating_temperature_range[1],  # Initial guess
                thermal_mass=1000.0,  # J/K (estimated)
                heat_generation=load.power_dissipation * load.duty_cycle
            )
            network[load.load_id] = node
        
        # Create nodes for heat exchangers
        for hx in system_config.heat_exchangers:
            node = ThermalNetworkNode(
                node_id=hx.exchanger_id,
                temperature=system_config.ambient_temperature + 50.0,  # Initial guess
                thermal_mass=hx.heat_capacity * 10.0,  # Estimated thermal mass
                heat_generation=0.0
            )
            network[hx.exchanger_id] = node
        
        # Create ambient node
        ambient_node = ThermalNetworkNode(
            node_id="ambient",
            temperature=system_config.ambient_temperature,
            thermal_mass=1e9,  # Very large (infinite heat sink)
            heat_generation=0.0
        )
        network["ambient"] = ambient_node
        
        # Connect nodes with thermal resistances
        self._connect_thermal_nodes(network, system_config)
        
        return network
    
    def _connect_thermal_nodes(self, network: Dict[str, ThermalNetworkNode],
                             system_config: ThermalSystemConfig) -> None:
        """Connect thermal nodes with appropriate thermal resistances."""
        # Connect loads to nearest heat exchangers
        for load in system_config.thermal_loads:
            load_node = network[load.load_id]
            
            # Find nearest heat exchanger
            nearest_hx = self._find_nearest_heat_exchanger(load, system_config.heat_exchangers)
            if nearest_hx:
                hx_node = network[nearest_hx.exchanger_id]
                
                # Calculate thermal resistance (simplified)
                distance = max(0.1, self._calculate_distance(load.location, (0, 0, 0)))  # Minimum 10cm distance
                conduction_resistance = distance / (50.0 * 0.01)  # Aluminum conductor, 1cm² area
                interface_resistance = load.thermal_resistance
                total_resistance = conduction_resistance + interface_resistance
                
                # Ensure reasonable resistance values
                total_resistance = max(0.01, min(10.0, total_resistance))  # Clamp between 0.01 and 10 K/W
                
                # Connect nodes
                load_node.connected_nodes.append(nearest_hx.exchanger_id)
                load_node.thermal_resistances[nearest_hx.exchanger_id] = total_resistance
                
                hx_node.connected_nodes.append(load.load_id)
                hx_node.thermal_resistances[load.load_id] = total_resistance
        
        # Connect heat exchangers to ambient
        for hx in system_config.heat_exchangers:
            hx_node = network[hx.exchanger_id]
            ambient_node = network["ambient"]
            
            # Thermal resistance based on heat exchanger effectiveness
            thermal_resistance = (1.0 - hx.effectiveness) / max(1.0, abs(hx.heat_capacity))
            thermal_resistance = max(0.001, min(1.0, thermal_resistance))  # Reasonable bounds
            
            hx_node.connected_nodes.append("ambient")
            hx_node.thermal_resistances["ambient"] = thermal_resistance
            
            ambient_node.connected_nodes.append(hx.exchanger_id)
            ambient_node.thermal_resistances[hx.exchanger_id] = thermal_resistance
    
    def _find_nearest_heat_exchanger(self, load: ThermalLoad,
                                   heat_exchangers: List[HeatExchangerSpec]) -> Optional[HeatExchangerSpec]:
        """Find nearest heat exchanger to thermal load."""
        if not heat_exchangers:
            return None
        
        # For simplicity, return first heat exchanger that can handle the load
        for hx in heat_exchangers:
            if hx.max_heat_transfer >= load.power_dissipation:
                return hx
        
        # If no suitable HX found, return the largest one
        return max(heat_exchangers, key=lambda hx: hx.max_heat_transfer)
    
    def _calculate_distance(self, pos1: Tuple[float, float, float],
                          pos2: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt(sum((a - b)**2 for a, b in zip(pos1, pos2)))
    
    def _solve_steady_state(self, network: Dict[str, ThermalNetworkNode],
                          operating_conditions: Dict[str, Any]) -> Dict[str, float]:
        """Solve steady-state thermal network."""
        # Simple iterative solver (Gauss-Seidel method)
        temperatures = {node_id: node.temperature for node_id, node in network.items()}
        
        # Fix ambient temperature
        temperatures["ambient"] = network["ambient"].temperature
        
        for iteration in range(100):  # Maximum iterations
            max_change = 0.0
            
            for node_id, node in network.items():
                if node_id == "ambient":
                    continue  # Skip ambient node (fixed temperature)
                
                # Calculate new temperature based on heat balance
                heat_in = node.heat_generation
                thermal_conductance_sum = 0.0
                
                for connected_id in node.connected_nodes:
                    resistance = node.thermal_resistances[connected_id]
                    conductance = 1.0 / resistance
                    heat_in += conductance * temperatures[connected_id]
                    thermal_conductance_sum += conductance
                
                if thermal_conductance_sum > 0:
                    new_temperature = heat_in / thermal_conductance_sum
                    change = abs(new_temperature - temperatures[node_id])
                    max_change = max(max_change, change)
                    temperatures[node_id] = new_temperature
            
            # Check convergence
            if max_change < 0.1:  # 0.1 K tolerance
                break
        
        return temperatures
    
    def _calculate_performance_metrics(self, system_config: ThermalSystemConfig,
                                     temperatures: Dict[str, float],
                                     operating_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate thermal system performance metrics."""
        metrics = {}
        
        # Temperature margins
        temp_margins = {}
        for load in system_config.thermal_loads:
            load_temp = temperatures.get(load.load_id, load.critical_temperature)
            margin = load.critical_temperature - load_temp
            temp_margins[load.load_id] = margin
        
        metrics['temperature_margins'] = temp_margins
        metrics['min_temperature_margin'] = min(temp_margins.values())
        
        # System efficiency
        total_power = sum(load.power_dissipation * load.duty_cycle for load in system_config.thermal_loads)
        total_hx_power = sum(hx.max_heat_transfer for hx in system_config.heat_exchangers)
        metrics['thermal_efficiency'] = total_power / total_hx_power if total_hx_power > 0 else 0
        
        # Coolant flow requirements
        coolant_flows = {}
        for loop_id, coolant in system_config.coolant_loops.items():
            # Estimate required flow rate
            delta_t = 20.0  # K (assumed temperature rise)
            flow_rate = total_power / (coolant.specific_heat * coolant.density * delta_t)
            coolant_flows[loop_id] = flow_rate
        
        metrics['coolant_flow_rates'] = coolant_flows
        
        # System mass and volume
        total_mass = sum(hx.mass for hx in system_config.heat_exchangers)
        total_volume = sum(hx.volume for hx in system_config.heat_exchangers)
        
        metrics['system_mass'] = total_mass
        metrics['system_volume'] = total_volume
        metrics['power_density'] = total_power / total_volume if total_volume > 0 else 0
        
        return metrics
    
    def simulate_transient_response(self, system_config: ThermalSystemConfig,
                                  power_profile: List[Tuple[float, Dict[str, float]]],
                                  time_step: float = 1.0) -> Dict[str, Any]:
        """Simulate transient thermal response to power profile."""
        # Build thermal network
        network = self._build_thermal_network(system_config)
        
        # Initialize temperatures
        temperatures = {node_id: node.temperature for node_id, node in network.items()}
        
        # Time history
        time_history = []
        temp_history = {node_id: [] for node_id in network.keys()}
        
        for time, power_dict in power_profile:
            # Update heat generation
            for node_id, node in network.items():
                if node_id in power_dict:
                    node.heat_generation = power_dict[node_id]
            
            # Solve for new temperatures (simplified explicit method)
            new_temperatures = temperatures.copy()
            
            for node_id, node in network.items():
                if node_id == "ambient":
                    continue
                
                # Heat balance: C * dT/dt = Q_gen - Q_out
                heat_out = 0.0
                for connected_id in node.connected_nodes:
                    resistance = node.thermal_resistances[connected_id]
                    heat_out += (temperatures[node_id] - temperatures[connected_id]) / resistance
                
                net_heat = node.heat_generation - heat_out
                temp_change = net_heat * time_step / node.thermal_mass
                new_temperatures[node_id] = temperatures[node_id] + temp_change
            
            temperatures = new_temperatures
            
            # Record history
            time_history.append(time)
            for node_id, temp in temperatures.items():
                temp_history[node_id].append(temp)
        
        return {
            'time_history': time_history,
            'temperature_history': temp_history,
            'final_temperatures': temperatures
        }
    
    def optimize_thermal_design(self, thermal_loads: List[ThermalLoad],
                              design_requirements: Dict[str, Any],
                              optimization_target: str = "mass") -> ThermalSystemConfig:
        """Optimize thermal system design for specified target."""
        self.logger.info(f"Optimizing thermal design for {optimization_target}")
        
        # Initial design
        best_config = self.design_thermal_system(thermal_loads, design_requirements)
        best_performance = self.analyze_thermal_performance(best_config, design_requirements)
        
        if optimization_target == "mass":
            best_metric = best_performance['performance_metrics']['system_mass']
        elif optimization_target == "volume":
            best_metric = best_performance['performance_metrics']['system_volume']
        elif optimization_target == "efficiency":
            best_metric = -best_performance['performance_metrics']['thermal_efficiency']  # Minimize negative
        else:
            best_metric = best_performance['performance_metrics']['system_mass']
        
        # Optimization iterations
        for iteration in range(5):
            # Vary design parameters
            modified_requirements = design_requirements.copy()
            
            # Adjust safety margin
            if iteration % 2 == 0:
                modified_requirements['safety_margin'] = design_requirements.get('safety_margin', 0.2) * (0.8 + 0.4 * iteration / 5)
            
            # Adjust temperature limits
            if iteration % 3 == 0:
                modified_requirements['max_coolant_temperature'] = design_requirements.get('max_coolant_temperature', 373.15) + 20 * (iteration - 2)
            
            test_config = self.design_thermal_system(thermal_loads, modified_requirements)
            test_performance = self.analyze_thermal_performance(test_config, modified_requirements)
            
            # Check if design is valid
            if test_performance['performance_metrics']['min_temperature_margin'] < 0:
                continue  # Skip invalid designs
            
            if optimization_target == "mass":
                test_metric = test_performance['performance_metrics']['system_mass']
            elif optimization_target == "volume":
                test_metric = test_performance['performance_metrics']['system_volume']
            elif optimization_target == "efficiency":
                test_metric = -test_performance['performance_metrics']['thermal_efficiency']
            else:
                test_metric = test_performance['performance_metrics']['system_mass']
            
            if test_metric < best_metric:
                best_config = test_config
                best_metric = test_metric
                self.logger.debug(f"Optimization iteration {iteration}: improved {optimization_target}")
        
        return best_config
    
    def _validate_thermal_design(self, system_config: ThermalSystemConfig) -> List[str]:
        """Validate thermal system design."""
        errors = []
        
        # Check heat exchanger capacity
        total_power = sum(load.power_dissipation for load in system_config.thermal_loads)
        total_hx_capacity = sum(hx.max_heat_transfer for hx in system_config.heat_exchangers)
        
        if total_hx_capacity < total_power * (1 + system_config.safety_margin):
            errors.append(f"Insufficient heat exchanger capacity: {total_hx_capacity:.1f}W < {total_power * (1 + system_config.safety_margin):.1f}W required")
        
        # Check temperature compatibility
        for load in system_config.thermal_loads:
            if load.critical_temperature > system_config.max_coolant_temperature + 100:
                errors.append(f"Load {load.load_id} critical temperature {load.critical_temperature:.1f}K exceeds system capability")
        
        # Check coolant compatibility
        for loop_id, coolant in system_config.coolant_loops.items():
            if system_config.max_coolant_temperature > coolant.operating_temp_range[1]:
                errors.append(f"Coolant loop {loop_id} temperature limit exceeded")
        
        return errors