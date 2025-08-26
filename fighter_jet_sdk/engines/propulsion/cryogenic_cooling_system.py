"""Cryogenic Cooling System for Extreme Hypersonic Flight."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math
import numpy as np
from enum import Enum

from ...core.logging import get_engine_logger


class CoolantType(Enum):
    """Cryogenic coolant types."""
    LIQUID_HYDROGEN = "liquid_hydrogen"
    LIQUID_NITROGEN = "liquid_nitrogen"
    LIQUID_HELIUM = "liquid_helium"
    LIQUID_METHANE = "liquid_methane"
    LIQUID_OXYGEN = "liquid_oxygen"


class CoolingMode(Enum):
    """Active cooling modes."""
    REGENERATIVE = "regenerative"
    TRANSPIRATION = "transpiration"
    FILM_COOLING = "film_cooling"
    IMPINGEMENT = "impingement"
    HYBRID = "hybrid"


@dataclass
class CryogenicCoolantProperties:
    """Properties of cryogenic coolants."""
    coolant_type: CoolantType
    boiling_point: float  # K at 1 atm
    critical_temperature: float  # K
    critical_pressure: float  # Pa
    density_liquid: float  # kg/m³ at boiling point
    density_vapor: float  # kg/m³ at boiling point
    specific_heat_liquid: float  # J/(kg⋅K)
    specific_heat_vapor: float  # J/(kg⋅K)
    thermal_conductivity_liquid: float  # W/(m⋅K)
    thermal_conductivity_vapor: float  # W/(m⋅K)
    viscosity_liquid: float  # Pa⋅s
    viscosity_vapor: float  # Pa⋅s
    latent_heat_vaporization: float  # J/kg
    molecular_weight: float  # kg/mol


@dataclass
class CoolingChannel:
    """Cooling channel specification."""
    channel_id: str
    geometry_type: str  # "rectangular", "circular", "serpentine"
    width: float  # m
    height: float  # m
    length: float  # m
    wall_thickness: float  # m
    surface_roughness: float  # m
    inlet_temperature: float  # K
    inlet_pressure: float  # Pa
    mass_flow_rate: float  # kg/s


@dataclass
class TranspirationCoolingSpec:
    """Transpiration cooling specification."""
    porous_material_id: str
    porosity: float  # dimensionless (0-1)
    permeability: float  # m²
    pore_diameter: float  # m
    thickness: float  # m
    coolant_injection_rate: float  # kg/(m²⋅s)
    injection_temperature: float  # K
    injection_pressure: float  # Pa


@dataclass
class FilmCoolingSpec:
    """Film cooling specification."""
    injection_angle: float  # degrees
    hole_diameter: float  # m
    hole_spacing: float  # m
    number_of_holes: int
    blowing_ratio: float  # coolant mass flux / hot gas mass flux
    density_ratio: float  # coolant density / hot gas density
    momentum_ratio: float  # coolant momentum / hot gas momentum


@dataclass
class CoolingSystemPerformance:
    """Cooling system performance results."""
    heat_removal_rate: float  # W
    cooling_effectiveness: float  # dimensionless
    pressure_drop: float  # Pa
    coolant_consumption: float  # kg/s
    surface_temperature_reduction: float  # K
    film_effectiveness: Optional[float] = None  # For film cooling
    transpiration_effectiveness: Optional[float] = None  # For transpiration cooling


class CryogenicCoolingSystem:
    """Advanced cryogenic cooling system for Mach 60 thermal protection."""
    
    def __init__(self):
        """Initialize cryogenic cooling system."""
        self.logger = get_engine_logger('propulsion.cryogenic_cooling')
        
        # Physical constants
        self.gas_constant = 8.314  # J/(mol⋅K)
        self.stefan_boltzmann = 5.67e-8  # W/(m²⋅K⁴)
        
        # Coolant database
        self.coolants: Dict[CoolantType, CryogenicCoolantProperties] = {}
        self._initialize_coolant_database()
        
        # Performance tracking
        self.cooling_history: List[Dict[str, Any]] = []
        
    def _initialize_coolant_database(self) -> None:
        """Initialize database of cryogenic coolant properties."""
        self.coolants[CoolantType.LIQUID_HYDROGEN] = CryogenicCoolantProperties(
            coolant_type=CoolantType.LIQUID_HYDROGEN,
            boiling_point=20.28,  # K
            critical_temperature=33.19,  # K
            critical_pressure=1.315e6,  # Pa
            density_liquid=70.8,  # kg/m³
            density_vapor=1.34,  # kg/m³
            specific_heat_liquid=9420.0,  # J/(kg⋅K)
            specific_heat_vapor=14300.0,  # J/(kg⋅K)
            thermal_conductivity_liquid=0.1,  # W/(m⋅K)
            thermal_conductivity_vapor=0.017,  # W/(m⋅K)
            viscosity_liquid=1.3e-5,  # Pa⋅s
            viscosity_vapor=8.8e-6,  # Pa⋅s
            latent_heat_vaporization=445000.0,  # J/kg
            molecular_weight=0.002016  # kg/mol
        )
        
        self.coolants[CoolantType.LIQUID_NITROGEN] = CryogenicCoolantProperties(
            coolant_type=CoolantType.LIQUID_NITROGEN,
            boiling_point=77.36,  # K
            critical_temperature=126.19,  # K
            critical_pressure=3.396e6,  # Pa
            density_liquid=808.0,  # kg/m³
            density_vapor=4.61,  # kg/m³
            specific_heat_liquid=2040.0,  # J/(kg⋅K)
            specific_heat_vapor=1040.0,  # J/(kg⋅K)
            thermal_conductivity_liquid=0.14,  # W/(m⋅K)
            thermal_conductivity_vapor=0.024,  # W/(m⋅K)
            viscosity_liquid=1.6e-4,  # Pa⋅s
            viscosity_vapor=1.7e-5,  # Pa⋅s
            latent_heat_vaporization=200000.0,  # J/kg
            molecular_weight=0.028014  # kg/mol
        )
        
        self.coolants[CoolantType.LIQUID_HELIUM] = CryogenicCoolantProperties(
            coolant_type=CoolantType.LIQUID_HELIUM,
            boiling_point=4.22,  # K
            critical_temperature=5.19,  # K
            critical_pressure=2.275e5,  # Pa
            density_liquid=125.0,  # kg/m³
            density_vapor=16.9,  # kg/m³
            specific_heat_liquid=4190.0,  # J/(kg⋅K)
            specific_heat_vapor=5193.0,  # J/(kg⋅K)
            thermal_conductivity_liquid=0.025,  # W/(m⋅K)
            thermal_conductivity_vapor=0.015,  # W/(m⋅K)
            viscosity_liquid=3.3e-6,  # Pa⋅s
            viscosity_vapor=1.9e-5,  # Pa⋅s
            latent_heat_vaporization=20900.0,  # J/kg
            molecular_weight=0.004003  # kg/mol
        )
        
        self.coolants[CoolantType.LIQUID_METHANE] = CryogenicCoolantProperties(
            coolant_type=CoolantType.LIQUID_METHANE,
            boiling_point=111.66,  # K
            critical_temperature=190.56,  # K
            critical_pressure=4.599e6,  # Pa
            density_liquid=422.6,  # kg/m³
            density_vapor=1.82,  # kg/m³
            specific_heat_liquid=3480.0,  # J/(kg⋅K)
            specific_heat_vapor=2220.0,  # J/(kg⋅K)
            thermal_conductivity_liquid=0.19,  # W/(m⋅K)
            thermal_conductivity_vapor=0.034,  # W/(m⋅K)
            viscosity_liquid=1.1e-4,  # Pa⋅s
            viscosity_vapor=1.0e-5,  # Pa⋅s
            latent_heat_vaporization=510000.0,  # J/kg
            molecular_weight=0.016043  # kg/mol
        )
    
    def design_regenerative_cooling(self, heat_flux: float, surface_area: float,
                                  coolant_type: CoolantType,
                                  design_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Design regenerative cooling system for extreme heat fluxes."""
        self.logger.info(f"Designing regenerative cooling for {heat_flux/1e6:.1f} MW/m² heat flux")
        
        coolant = self.coolants[coolant_type]
        
        # Calculate required heat removal
        total_heat_load = heat_flux * surface_area  # W
        
        # Design cooling channels
        channel_design = self._design_cooling_channels(
            total_heat_load, coolant, design_constraints
        )
        
        # Calculate cooling performance
        performance = self._calculate_regenerative_performance(
            heat_flux, channel_design, coolant, design_constraints
        )
        
        # Optimize channel configuration
        optimized_design = self._optimize_channel_configuration(
            channel_design, performance, design_constraints
        )
        
        return {
            'channel_design': optimized_design,
            'performance': performance,
            'coolant_properties': coolant,
            'total_heat_removal': total_heat_load,
            'design_margins': self._calculate_design_margins(performance, design_constraints)
        }
    
    def _design_cooling_channels(self, heat_load: float,
                               coolant: CryogenicCoolantProperties,
                               constraints: Dict[str, Any]) -> List[CoolingChannel]:
        """Design cooling channel geometry and configuration."""
        # Channel sizing based on heat load and coolant properties
        max_heat_flux_per_channel = constraints.get('max_heat_flux_per_channel', 50e6)  # W/m²
        channel_length = constraints.get('channel_length', 1.0)  # m
        
        # Number of channels needed
        num_channels = max(1, int(math.ceil(heat_load / (max_heat_flux_per_channel * channel_length * 0.01))))
        
        # Channel dimensions
        channel_width = constraints.get('channel_width', 0.002)  # 2mm
        channel_height = constraints.get('channel_height', 0.001)  # 1mm
        wall_thickness = constraints.get('wall_thickness', 0.0005)  # 0.5mm
        
        # Mass flow rate per channel
        total_mass_flow = self._calculate_required_mass_flow(heat_load, coolant)
        mass_flow_per_channel = total_mass_flow / num_channels
        
        channels = []
        for i in range(num_channels):
            channel = CoolingChannel(
                channel_id=f"channel_{i+1}",
                geometry_type="rectangular",
                width=channel_width,
                height=channel_height,
                length=channel_length,
                wall_thickness=wall_thickness,
                surface_roughness=constraints.get('surface_roughness', 1e-6),
                inlet_temperature=coolant.boiling_point + 5.0,  # Slightly subcooled
                inlet_pressure=constraints.get('inlet_pressure', 2e6),  # 20 bar
                mass_flow_rate=mass_flow_per_channel
            )
            channels.append(channel)
        
        return channels
    
    def _calculate_required_mass_flow(self, heat_load: float,
                                    coolant: CryogenicCoolantProperties) -> float:
        """Calculate required coolant mass flow rate."""
        # Assume coolant heats from inlet to near boiling point
        temperature_rise = 50.0  # K (conservative)
        
        # Heat capacity method
        mass_flow_sensible = heat_load / (coolant.specific_heat_liquid * temperature_rise)
        
        # Phase change method (if boiling occurs)
        mass_flow_latent = heat_load / coolant.latent_heat_vaporization
        
        # Use the larger of the two (more conservative)
        return max(mass_flow_sensible, mass_flow_latent) * 1.5  # 50% safety margin
    
    def _calculate_regenerative_performance(self, heat_flux: float,
                                          channels: List[CoolingChannel],
                                          coolant: CryogenicCoolantProperties,
                                          constraints: Dict[str, Any]) -> CoolingSystemPerformance:
        """Calculate regenerative cooling system performance."""
        total_heat_removal = 0.0
        total_pressure_drop = 0.0
        total_mass_flow = 0.0
        
        for channel in channels:
            # Heat transfer coefficient
            reynolds = self._calculate_reynolds_number(channel, coolant)
            prandtl = self._calculate_prandtl_number(coolant)
            nusselt = self._calculate_nusselt_number(reynolds, prandtl, channel)
            
            heat_transfer_coeff = nusselt * coolant.thermal_conductivity_liquid / channel.height
            
            # Heat removal per channel
            perimeter = 2 * (channel.width + channel.height)
            heat_transfer_area = perimeter * channel.length
            
            # Log mean temperature difference
            wall_temp = constraints.get('wall_temperature', 2000.0)  # K
            inlet_temp = channel.inlet_temperature
            outlet_temp = inlet_temp + 100.0  # Estimated temperature rise
            
            lmtd = ((wall_temp - inlet_temp) - (wall_temp - outlet_temp)) / \
                   math.log((wall_temp - inlet_temp) / (wall_temp - outlet_temp))
            
            channel_heat_removal = heat_transfer_coeff * heat_transfer_area * lmtd
            total_heat_removal += channel_heat_removal
            
            # Pressure drop
            friction_factor = self._calculate_friction_factor(reynolds, channel)
            velocity = channel.mass_flow_rate / (coolant.density_liquid * channel.width * channel.height)
            
            pressure_drop = friction_factor * (channel.length / channel.height) * \
                          0.5 * coolant.density_liquid * velocity**2
            total_pressure_drop = max(total_pressure_drop, pressure_drop)  # Parallel channels
            
            total_mass_flow += channel.mass_flow_rate
        
        # Overall effectiveness
        max_possible_heat_removal = heat_flux * sum(ch.width * ch.length for ch in channels)
        effectiveness = min(1.0, total_heat_removal / max_possible_heat_removal)
        
        # Surface temperature reduction
        surface_temp_reduction = total_heat_removal / (heat_flux * len(channels) * 0.01)  # Simplified
        
        return CoolingSystemPerformance(
            heat_removal_rate=total_heat_removal,
            cooling_effectiveness=effectiveness,
            pressure_drop=total_pressure_drop,
            coolant_consumption=total_mass_flow,
            surface_temperature_reduction=surface_temp_reduction
        )
    
    def calculate_transpiration_cooling(self, heat_flux: float,
                                      transpiration_spec: TranspirationCoolingSpec,
                                      coolant_type: CoolantType,
                                      operating_conditions: Dict[str, Any]) -> CoolingSystemPerformance:
        """Calculate transpiration cooling effectiveness."""
        self.logger.info("Calculating transpiration cooling performance")
        
        coolant = self.coolants[coolant_type]
        
        # Coolant flow through porous material
        coolant_velocity = self._calculate_transpiration_velocity(
            transpiration_spec, coolant, operating_conditions
        )
        
        # Heat transfer effectiveness
        effectiveness = self._calculate_transpiration_effectiveness(
            heat_flux, transpiration_spec, coolant, coolant_velocity, operating_conditions
        )
        
        # Heat removal rate
        surface_area = operating_conditions.get('surface_area', 1.0)  # m²
        heat_removal = effectiveness * heat_flux * surface_area
        
        # Coolant consumption
        coolant_consumption = transpiration_spec.coolant_injection_rate * surface_area
        
        # Pressure drop through porous material
        pressure_drop = self._calculate_transpiration_pressure_drop(
            transpiration_spec, coolant, coolant_velocity
        )
        
        # Surface temperature reduction
        surface_temp_reduction = heat_removal / (surface_area * 1000.0)  # Simplified estimate
        
        return CoolingSystemPerformance(
            heat_removal_rate=heat_removal,
            cooling_effectiveness=effectiveness,
            pressure_drop=pressure_drop,
            coolant_consumption=coolant_consumption,
            surface_temperature_reduction=surface_temp_reduction,
            transpiration_effectiveness=effectiveness
        )
    
    def calculate_film_cooling(self, heat_flux: float,
                             film_spec: FilmCoolingSpec,
                             coolant_type: CoolantType,
                             mainstream_conditions: Dict[str, Any]) -> CoolingSystemPerformance:
        """Calculate film cooling effectiveness."""
        self.logger.info("Calculating film cooling performance")
        
        coolant = self.coolants[coolant_type]
        
        # Film cooling effectiveness correlation
        effectiveness = self._calculate_film_effectiveness(
            film_spec, coolant, mainstream_conditions
        )
        
        # Heat removal calculation
        surface_area = mainstream_conditions.get('surface_area', 1.0)  # m²
        heat_removal = effectiveness * heat_flux * surface_area
        
        # Coolant mass flow rate
        hole_area = math.pi * (film_spec.hole_diameter / 2)**2
        total_hole_area = hole_area * film_spec.number_of_holes
        
        mainstream_density = mainstream_conditions.get('density', 0.1)  # kg/m³ (high altitude)
        mainstream_velocity = mainstream_conditions.get('velocity', 20000.0)  # m/s
        
        coolant_velocity = film_spec.blowing_ratio * mainstream_velocity * \
                         (mainstream_density / coolant.density_vapor)
        
        coolant_consumption = coolant.density_vapor * coolant_velocity * total_hole_area
        
        # Pressure drop (simplified)
        pressure_drop = 0.5 * coolant.density_vapor * coolant_velocity**2
        
        # Surface temperature reduction
        surface_temp_reduction = effectiveness * (mainstream_conditions.get('temperature', 3000.0) - 
                                                coolant.boiling_point)
        
        return CoolingSystemPerformance(
            heat_removal_rate=heat_removal,
            cooling_effectiveness=effectiveness,
            pressure_drop=pressure_drop,
            coolant_consumption=coolant_consumption,
            surface_temperature_reduction=surface_temp_reduction,
            film_effectiveness=effectiveness
        )
    
    def optimize_cooling_system(self, heat_flux: float,
                              cooling_modes: List[CoolingMode],
                              coolant_type: CoolantType,
                              optimization_target: str = "effectiveness") -> Dict[str, Any]:
        """Optimize cooling system configuration for given target."""
        self.logger.info(f"Optimizing cooling system for {optimization_target}")
        
        best_config = None
        best_performance = None
        best_metric = float('inf') if optimization_target == "mass_flow" else 0.0
        
        # Test different cooling configurations
        for mode in cooling_modes:
            try:
                if mode == CoolingMode.REGENERATIVE:
                    config = self.design_regenerative_cooling(
                        heat_flux, 1.0, coolant_type, {'channel_length': 0.5}
                    )
                    performance = config['performance']
                    
                elif mode == CoolingMode.TRANSPIRATION:
                    transpiration_spec = TranspirationCoolingSpec(
                        porous_material_id="porous_tungsten",
                        porosity=0.3,
                        permeability=1e-12,  # m²
                        pore_diameter=50e-6,  # 50 microns
                        thickness=0.005,  # 5mm
                        coolant_injection_rate=0.1,  # kg/(m²⋅s)
                        injection_temperature=coolant_type.value == "liquid_hydrogen" and 25.0 or 80.0,
                        injection_pressure=1e6  # 10 bar
                    )
                    
                    performance = self.calculate_transpiration_cooling(
                        heat_flux, transpiration_spec, coolant_type, {'surface_area': 1.0}
                    )
                    config = {'transpiration_spec': transpiration_spec}
                    
                elif mode == CoolingMode.FILM_COOLING:
                    film_spec = FilmCoolingSpec(
                        injection_angle=30.0,  # degrees
                        hole_diameter=0.001,  # 1mm
                        hole_spacing=0.005,  # 5mm
                        number_of_holes=100,
                        blowing_ratio=1.0,
                        density_ratio=10.0,
                        momentum_ratio=1.0
                    )
                    
                    performance = self.calculate_film_cooling(
                        heat_flux, film_spec, coolant_type,
                        {'surface_area': 1.0, 'temperature': 3000.0, 'density': 0.1, 'velocity': 20000.0}
                    )
                    config = {'film_spec': film_spec}
                
                # Evaluate performance metric
                if optimization_target == "effectiveness":
                    metric = performance.cooling_effectiveness
                    if metric > best_metric:
                        best_metric = metric
                        best_config = config
                        best_performance = performance
                        
                elif optimization_target == "mass_flow":
                    metric = performance.coolant_consumption
                    if metric < best_metric:
                        best_metric = metric
                        best_config = config
                        best_performance = performance
                        
                elif optimization_target == "pressure_drop":
                    metric = performance.pressure_drop
                    if metric < best_metric:
                        best_metric = metric
                        best_config = config
                        best_performance = performance
                
            except Exception as e:
                self.logger.warning(f"Failed to evaluate {mode}: {e}")
                continue
        
        return {
            'best_configuration': best_config,
            'best_performance': best_performance,
            'optimization_metric': best_metric,
            'cooling_mode': cooling_modes[0] if best_config else None
        }
    
    def _calculate_reynolds_number(self, channel: CoolingChannel,
                                 coolant: CryogenicCoolantProperties) -> float:
        """Calculate Reynolds number for channel flow."""
        hydraulic_diameter = 2 * channel.width * channel.height / (channel.width + channel.height)
        velocity = channel.mass_flow_rate / (coolant.density_liquid * channel.width * channel.height)
        return coolant.density_liquid * velocity * hydraulic_diameter / coolant.viscosity_liquid
    
    def _calculate_prandtl_number(self, coolant: CryogenicCoolantProperties) -> float:
        """Calculate Prandtl number for coolant."""
        return coolant.viscosity_liquid * coolant.specific_heat_liquid / coolant.thermal_conductivity_liquid
    
    def _calculate_nusselt_number(self, reynolds: float, prandtl: float,
                                channel: CoolingChannel) -> float:
        """Calculate Nusselt number for heat transfer."""
        # Dittus-Boelter correlation for turbulent flow
        if reynolds > 2300:
            return 0.023 * reynolds**0.8 * prandtl**0.4
        else:
            # Laminar flow
            return 4.36  # Constant for rectangular channels
    
    def _calculate_friction_factor(self, reynolds: float, channel: CoolingChannel) -> float:
        """Calculate friction factor for pressure drop."""
        if reynolds > 2300:
            # Turbulent flow - Blasius correlation
            return 0.316 * reynolds**(-0.25)
        else:
            # Laminar flow
            return 64.0 / reynolds
    
    def _calculate_transpiration_velocity(self, spec: TranspirationCoolingSpec,
                                        coolant: CryogenicCoolantProperties,
                                        conditions: Dict[str, Any]) -> float:
        """Calculate coolant velocity through porous material."""
        # Darcy's law
        pressure_gradient = (spec.injection_pressure - conditions.get('back_pressure', 1e5)) / spec.thickness
        velocity = spec.permeability * pressure_gradient / coolant.viscosity_liquid
        return velocity
    
    def _calculate_transpiration_effectiveness(self, heat_flux: float,
                                             spec: TranspirationCoolingSpec,
                                             coolant: CryogenicCoolantProperties,
                                             velocity: float,
                                             conditions: Dict[str, Any]) -> float:
        """Calculate transpiration cooling effectiveness."""
        # Simplified effectiveness model
        mass_flux = spec.coolant_injection_rate
        heat_capacity_rate = mass_flux * coolant.specific_heat_liquid
        
        # Effectiveness based on mass flux and heat capacity
        dimensionless_mass_flux = mass_flux / 1.0  # Normalized
        effectiveness = 1.0 - math.exp(-2.0 * dimensionless_mass_flux)
        
        return min(0.95, effectiveness)  # Cap at 95%
    
    def _calculate_transpiration_pressure_drop(self, spec: TranspirationCoolingSpec,
                                             coolant: CryogenicCoolantProperties,
                                             velocity: float) -> float:
        """Calculate pressure drop through porous material."""
        # Darcy-Forchheimer equation
        darcy_term = coolant.viscosity_liquid * velocity * spec.thickness / spec.permeability
        forchheimer_term = 0.55 * coolant.density_liquid * velocity**2 * spec.thickness / math.sqrt(spec.permeability)
        
        return darcy_term + forchheimer_term
    
    def _calculate_film_effectiveness(self, spec: FilmCoolingSpec,
                                    coolant: CryogenicCoolantProperties,
                                    mainstream: Dict[str, Any]) -> float:
        """Calculate film cooling effectiveness."""
        # Simplified film effectiveness correlation
        # Based on blowing ratio and momentum ratio
        
        M = spec.blowing_ratio
        I = spec.momentum_ratio
        
        # Effectiveness correlation (simplified)
        if M < 0.5:
            effectiveness = 1.2 * M
        elif M < 2.0:
            effectiveness = 0.6 * (1 + M) / (1 + 0.5 * M)
        else:
            effectiveness = 0.4 / M**0.5  # Jet lift-off reduces effectiveness
        
        # Momentum ratio correction
        effectiveness *= (1.0 + I)**(-0.25)
        
        return min(0.8, max(0.1, effectiveness))  # Reasonable bounds
    
    def _optimize_channel_configuration(self, channels: List[CoolingChannel],
                                      performance: CoolingSystemPerformance,
                                      constraints: Dict[str, Any]) -> List[CoolingChannel]:
        """Optimize cooling channel configuration."""
        # Simple optimization - adjust channel dimensions for better performance
        optimized_channels = []
        
        for channel in channels:
            # Optimize based on heat transfer vs pressure drop trade-off
            if performance.pressure_drop > constraints.get('max_pressure_drop', 1e6):
                # Reduce pressure drop by increasing channel height
                new_height = channel.height * 1.2
                new_width = channel.width * 0.9  # Maintain roughly same area
            elif performance.cooling_effectiveness < constraints.get('min_effectiveness', 0.8):
                # Improve heat transfer by reducing channel height (higher velocity)
                new_height = channel.height * 0.8
                new_width = channel.width * 1.1
            else:
                # Keep current dimensions
                new_height = channel.height
                new_width = channel.width
            
            optimized_channel = CoolingChannel(
                channel_id=channel.channel_id,
                geometry_type=channel.geometry_type,
                width=new_width,
                height=new_height,
                length=channel.length,
                wall_thickness=channel.wall_thickness,
                surface_roughness=channel.surface_roughness,
                inlet_temperature=channel.inlet_temperature,
                inlet_pressure=channel.inlet_pressure,
                mass_flow_rate=channel.mass_flow_rate
            )
            optimized_channels.append(optimized_channel)
        
        return optimized_channels
    
    def _calculate_design_margins(self, performance: CoolingSystemPerformance,
                                constraints: Dict[str, Any]) -> Dict[str, float]:
        """Calculate design margins for cooling system."""
        margins = {}
        
        # Effectiveness margin
        target_effectiveness = constraints.get('target_effectiveness', 0.8)
        margins['effectiveness'] = (performance.cooling_effectiveness - target_effectiveness) / target_effectiveness
        
        # Pressure drop margin
        max_pressure_drop = constraints.get('max_pressure_drop', 1e6)
        margins['pressure_drop'] = (max_pressure_drop - performance.pressure_drop) / max_pressure_drop
        
        # Mass flow margin
        max_mass_flow = constraints.get('max_mass_flow', 10.0)
        margins['mass_flow'] = (max_mass_flow - performance.coolant_consumption) / max_mass_flow
        
        return margins
    
    def validate_cooling_design(self, performance: CoolingSystemPerformance,
                              design_requirements: Dict[str, Any]) -> List[str]:
        """Validate cooling system design against requirements."""
        warnings = []
        
        # Check effectiveness
        min_effectiveness = design_requirements.get('min_effectiveness', 0.7)
        if performance.cooling_effectiveness < min_effectiveness:
            warnings.append(f"Cooling effectiveness {performance.cooling_effectiveness:.3f} "
                          f"below minimum {min_effectiveness:.3f}")
        
        # Check pressure drop
        max_pressure_drop = design_requirements.get('max_pressure_drop', 2e6)
        if performance.pressure_drop > max_pressure_drop:
            warnings.append(f"Pressure drop {performance.pressure_drop/1e6:.1f} MPa "
                          f"exceeds maximum {max_pressure_drop/1e6:.1f} MPa")
        
        # Check mass flow rate
        max_mass_flow = design_requirements.get('max_mass_flow', 50.0)
        if performance.coolant_consumption > max_mass_flow:
            warnings.append(f"Coolant consumption {performance.coolant_consumption:.1f} kg/s "
                          f"exceeds maximum {max_mass_flow:.1f} kg/s")
        
        # Check temperature reduction
        min_temp_reduction = design_requirements.get('min_temperature_reduction', 500.0)
        if performance.surface_temperature_reduction < min_temp_reduction:
            warnings.append(f"Temperature reduction {performance.surface_temperature_reduction:.1f} K "
                          f"below minimum {min_temp_reduction:.1f} K")
        
        return warnings