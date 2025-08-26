"""Thermal materials database and modeling for ultra-high temperature applications."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass
import math
from scipy.interpolate import interp1d, interp2d, griddata
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, fsolve

from ...common.data_models import MaterialDefinition, ThermalProperties
from ...common.enums import MaterialType
from ...core.logging import get_engine_logger


@dataclass
class ThermalAnalysisResult:
    """Results from thermal analysis calculations."""
    temperatures: np.ndarray  # K
    heat_flux: np.ndarray  # W/m²
    thermal_stress: np.ndarray  # Pa
    safety_factor: np.ndarray  # Dimensionless
    failure_mode: List[str]  # Failure modes at each point


@dataclass
class HypersonicConditions:
    """Hypersonic flight conditions for thermal analysis."""
    mach_number: float
    altitude: float  # m
    flight_time: float  # s
    angle_of_attack: float  # degrees
    surface_emissivity: float  # 0-1
    recovery_factor: float  # 0-1, typically 0.89 for turbulent flow


@dataclass
class ThermalStressAnalysis:
    """Thermal stress analysis results."""
    thermal_strain: np.ndarray
    thermal_stress: np.ndarray  # Pa
    von_mises_stress: np.ndarray  # Pa
    principal_stresses: np.ndarray  # Pa, shape (n_points, 3)
    safety_factors: np.ndarray
    critical_locations: List[int]  # Indices of critical stress locations


@dataclass
class AblativeProperties:
    """Ablative material properties."""
    heat_of_ablation: float  # J/kg - energy required to ablate unit mass
    char_yield: float  # Dimensionless - fraction of material forming char layer
    pyrolysis_temperature: float  # K - temperature at which pyrolysis begins
    surface_emissivity: float  # Dimensionless - surface emissivity for radiation
    blowing_parameter: float  # Dimensionless - mass transfer parameter
    recession_rate_coefficient: float  # m/(s⋅Pa) - recession rate coefficient


@dataclass
class AblativeCoolingResult:
    """Results from ablative cooling analysis."""
    recession_rate: np.ndarray  # m/s - surface recession rate
    mass_loss_rate: np.ndarray  # kg/(m²⋅s) - mass loss per unit area
    cooling_effectiveness: np.ndarray  # Dimensionless - cooling effectiveness
    char_thickness: np.ndarray  # m - char layer thickness
    surface_temperature: np.ndarray  # K - surface temperature
    heat_flux_reduction: np.ndarray  # W/m² - heat flux reduction due to ablation
    total_mass_loss: float  # kg/m² - total mass loss over mission
    remaining_thickness: float  # m - remaining material thickness


class ThermalMaterialsDB:
    """Ultra-high temperature ceramics database and thermal modeling."""
    
    def __init__(self):
        """Initialize the thermal materials database."""
        self.logger = get_engine_logger('materials.thermal')
        
        # Physical constants
        self.stefan_boltzmann = 5.670374419e-8  # W/(m²⋅K⁴)
        self.gas_constant = 8.314  # J/(mol⋅K)
        
        # Initialize UHTC materials database
        self._uhtc_materials = {}
        self._initialize_uhtc_database()
        
        # Thermal property interpolation functions
        self._property_interpolators = {}
        self._setup_property_interpolators()
        
        # Initialize ablative cooling model
        self.ablative_model = AblativeCoolingModel()
    
    def _initialize_uhtc_database(self):
        """Initialize database with ultra-high temperature ceramics."""
        
        # Hafnium Carbide (HfC) - Enhanced for extreme temperatures
        hfc_material = MaterialDefinition(
            name="Hafnium Carbide (HfC)",
            base_material_type=MaterialType.ULTRA_HIGH_TEMP_CERAMIC,
            thermal_properties=ThermalProperties(
                thermal_conductivity=22.0,  # W/(m⋅K) at room temperature
                specific_heat=200.0,  # J/(kg⋅K) at room temperature
                density=12800.0,  # kg/m³
                melting_point=4273.0,  # K (4000°C)
                operating_temp_range=(293.0, 5500.0)  # Extended to 5227°C for Mach 60
            )
        )
        self._uhtc_materials['hfc'] = hfc_material
        
        # Tantalum Carbide (TaC) - Enhanced for extreme temperatures
        tac_material = MaterialDefinition(
            name="Tantalum Carbide (TaC)",
            base_material_type=MaterialType.ULTRA_HIGH_TEMP_CERAMIC,
            thermal_properties=ThermalProperties(
                thermal_conductivity=20.0,  # W/(m⋅K)
                specific_heat=180.0,  # J/(kg⋅K)
                density=14500.0,  # kg/m³
                melting_point=4258.0,  # K (3985°C)
                operating_temp_range=(293.0, 5400.0)  # Extended to 5127°C
            )
        )
        self._uhtc_materials['tac'] = tac_material
        
        # Tungsten (W) - Pure tungsten for extreme temperatures
        w_material = MaterialDefinition(
            name="Pure Tungsten (W)",
            base_material_type=MaterialType.ULTRA_HIGH_TEMP_CERAMIC,
            thermal_properties=ThermalProperties(
                thermal_conductivity=173.0,  # W/(m⋅K) - highest thermal conductivity
                specific_heat=132.0,  # J/(kg⋅K)
                density=19300.0,  # kg/m³
                melting_point=3695.0,  # K (3422°C)
                operating_temp_range=(293.0, 5800.0)  # Extended to 5527°C
            )
        )
        self._uhtc_materials['w'] = w_material
        
        # Hafnium-Tantalum Carbide (Hf-Ta)C - Highest melting point material
        hftac_material = MaterialDefinition(
            name="Hafnium-Tantalum Carbide (Hf-Ta)C",
            base_material_type=MaterialType.ULTRA_HIGH_TEMP_CERAMIC,
            thermal_properties=ThermalProperties(
                thermal_conductivity=18.0,  # W/(m⋅K)
                specific_heat=190.0,  # J/(kg⋅K)
                density=13600.0,  # kg/m³
                melting_point=4488.0,  # K (4215°C) - highest known melting point
                operating_temp_range=(293.0, 6000.0)  # Up to 5727°C for extreme conditions
            )
        )
        self._uhtc_materials['hftac'] = hftac_material
        
        # Zirconium Carbide (ZrC) - Enhanced
        zrc_material = MaterialDefinition(
            name="Zirconium Carbide (ZrC)",
            base_material_type=MaterialType.ULTRA_HIGH_TEMP_CERAMIC,
            thermal_properties=ThermalProperties(
                thermal_conductivity=18.0,  # W/(m⋅K)
                specific_heat=280.0,  # J/(kg⋅K)
                density=6730.0,  # kg/m³
                melting_point=3813.0,  # K (3540°C)
                operating_temp_range=(293.0, 5200.0)  # Extended to 4927°C
            )
        )
        self._uhtc_materials['zrc'] = zrc_material
        
        # Hafnium Nitride (HfN) - Enhanced
        hfn_material = MaterialDefinition(
            name="Hafnium Nitride (HfN)",
            base_material_type=MaterialType.ULTRA_HIGH_TEMP_CERAMIC,
            thermal_properties=ThermalProperties(
                thermal_conductivity=25.0,  # W/(m⋅K)
                specific_heat=190.0,  # J/(kg⋅K)
                density=13800.0,  # kg/m³
                melting_point=3583.0,  # K (3310°C)
                operating_temp_range=(293.0, 5100.0)  # Extended to 4827°C
            )
        )
        self._uhtc_materials['hfn'] = hfn_material
        
        # Rhenium (Re) - Ultra-high temperature metal
        re_material = MaterialDefinition(
            name="Rhenium (Re)",
            base_material_type=MaterialType.ULTRA_HIGH_TEMP_CERAMIC,  # Classified as UHTC for this application
            thermal_properties=ThermalProperties(
                thermal_conductivity=48.0,  # W/(m⋅K)
                specific_heat=137.0,  # J/(kg⋅K)
                density=21020.0,  # kg/m³
                melting_point=3459.0,  # K (3186°C)
                operating_temp_range=(293.0, 5000.0)  # Extended to 4727°C
            )
        )
        self._uhtc_materials['re'] = re_material
        
        # Carbon-Carbon Composite - For extreme thermal shock resistance
        cc_material = MaterialDefinition(
            name="Carbon-Carbon Composite (C-C)",
            base_material_type=MaterialType.ULTRA_HIGH_TEMP_CERAMIC,
            thermal_properties=ThermalProperties(
                thermal_conductivity=120.0,  # W/(m⋅K) - high in-plane conductivity
                specific_heat=710.0,  # J/(kg⋅K)
                density=1800.0,  # kg/m³ - lightweight
                melting_point=3773.0,  # K (3500°C) - sublimation point
                operating_temp_range=(293.0, 5500.0)  # Up to 5227°C in inert atmosphere
            )
        )
        self._uhtc_materials['cc'] = cc_material
        
        # Boron Carbide (B4C) - For neutron shielding and extreme conditions
        b4c_material = MaterialDefinition(
            name="Boron Carbide (B4C)",
            base_material_type=MaterialType.ULTRA_HIGH_TEMP_CERAMIC,
            thermal_properties=ThermalProperties(
                thermal_conductivity=30.0,  # W/(m⋅K)
                specific_heat=950.0,  # J/(kg⋅K)
                density=2520.0,  # kg/m³
                melting_point=2723.0,  # K (2450°C)
                operating_temp_range=(293.0, 5300.0)  # Extended for plasma applications
            )
        )
        self._uhtc_materials['b4c'] = b4c_material
        
        self.logger.info(f"Initialized UHTC database with {len(self._uhtc_materials)} materials")
    
    def _setup_property_interpolators(self):
        """Set up temperature-dependent property interpolation functions."""
        
        # Temperature-dependent thermal conductivity data (simplified models)
        for material_id, material in self._uhtc_materials.items():
            if not material.thermal_properties:
                continue
            
            props = material.thermal_properties
            temp_range = np.linspace(props.operating_temp_range[0], 
                                   props.operating_temp_range[1], 50)
            
            # Thermal conductivity typically decreases with temperature for ceramics
            k_values = self._calculate_thermal_conductivity_vs_temperature(
                props.thermal_conductivity, temp_range, material_id
            )
            
            # Specific heat typically increases with temperature
            cp_values = self._calculate_specific_heat_vs_temperature(
                props.specific_heat, temp_range, material_id
            )
            
            # Create interpolation functions
            self._property_interpolators[material_id] = {
                'thermal_conductivity': interp1d(temp_range, k_values, 
                                               bounds_error=False, fill_value='extrapolate'),
                'specific_heat': interp1d(temp_range, cp_values,
                                        bounds_error=False, fill_value='extrapolate'),
                'temperature_range': temp_range
            }
    
    def get_material(self, material_id: str) -> Optional[MaterialDefinition]:
        """Get UHTC material by ID."""
        return self._uhtc_materials.get(material_id)
    
    def list_materials(self) -> Dict[str, str]:
        """List all UHTC materials."""
        return {mat_id: material.name for mat_id, material in self._uhtc_materials.items()}
    
    def get_materials_for_temperature(self, temperature: float) -> List[str]:
        """Get materials suitable for given temperature."""
        suitable_materials = []
        
        for mat_id, material in self._uhtc_materials.items():
            if not material.thermal_properties:
                continue
            
            temp_range = material.thermal_properties.operating_temp_range
            if temp_range[0] <= temperature <= temp_range[1]:
                suitable_materials.append(mat_id)
        
        return suitable_materials
    
    def get_extreme_temperature_materials(self, min_temperature: float = 5000.0) -> List[str]:
        """Get materials suitable for extreme temperatures above 5000K."""
        extreme_materials = []
        
        for mat_id, material in self._uhtc_materials.items():
            if not material.thermal_properties:
                continue
            
            # Check if material can operate above the minimum temperature
            if material.thermal_properties.operating_temp_range[1] >= min_temperature:
                extreme_materials.append(mat_id)
        
        # Sort by maximum operating temperature (descending)
        extreme_materials.sort(
            key=lambda mat_id: self._uhtc_materials[mat_id].thermal_properties.operating_temp_range[1],
            reverse=True
        )
        
        return extreme_materials
    
    def get_materials_for_plasma_environment(self, temperature: float, plasma_density: float) -> List[str]:
        """Get materials suitable for plasma environment conditions."""
        suitable_materials = self.get_materials_for_temperature(temperature)
        
        # Filter materials based on plasma compatibility
        plasma_compatible = []
        for mat_id in suitable_materials:
            material = self._uhtc_materials[mat_id]
            
            # Prefer materials with high thermal conductivity for plasma environments
            if material.thermal_properties.thermal_conductivity > 15.0:
                # Prefer refractory metals and carbides for plasma applications
                if any(element in mat_id for element in ['hfc', 'tac', 'w', 'hftac', 're']):
                    plasma_compatible.append(mat_id)
                elif plasma_density < 1e20:  # Lower density plasma - ceramics acceptable
                    plasma_compatible.append(mat_id)
        
        return plasma_compatible
    
    def calculate_thermal_conductivity(self, material_id: str, temperature: float) -> float:
        """Calculate temperature-dependent thermal conductivity."""
        if material_id not in self._property_interpolators:
            material = self.get_material(material_id)
            if material and material.thermal_properties:
                return material.thermal_properties.thermal_conductivity
            return 0.0
        
        interpolator = self._property_interpolators[material_id]['thermal_conductivity']
        return float(interpolator(temperature))
    
    def calculate_specific_heat(self, material_id: str, temperature: float) -> float:
        """Calculate temperature-dependent specific heat."""
        if material_id not in self._property_interpolators:
            material = self.get_material(material_id)
            if material and material.thermal_properties:
                return material.thermal_properties.specific_heat
            return 0.0
        
        interpolator = self._property_interpolators[material_id]['specific_heat']
        return float(interpolator(temperature))
    
    def analyze_hypersonic_heating(self, material_id: str,
                                  conditions: HypersonicConditions,
                                  thickness: float = 0.01) -> ThermalAnalysisResult:
        """
        Analyze thermal response under hypersonic conditions.
        
        Args:
            material_id: Material identifier
            conditions: Hypersonic flight conditions
            thickness: Material thickness (m)
            
        Returns:
            ThermalAnalysisResult with temperature and stress distributions
        """
        self.logger.info(f"Analyzing hypersonic heating for {material_id} at Mach {conditions.mach_number}")
        
        material = self.get_material(material_id)
        if not material or not material.thermal_properties:
            raise ValueError(f"Material {material_id} not found or missing thermal properties")
        
        # Calculate stagnation temperature
        T_stag = self._calculate_stagnation_temperature(conditions)
        
        # Calculate heat flux
        q_wall = self._calculate_wall_heat_flux(conditions, T_stag)
        
        # Solve 1D transient heat conduction
        temperatures, times = self._solve_transient_heat_conduction(
            material_id, q_wall, thickness, conditions.flight_time
        )
        
        # Calculate thermal stress
        thermal_stress = self._calculate_thermal_stress(
            material_id, temperatures, thickness
        )
        
        # Determine safety factors and failure modes
        safety_factors, failure_modes = self._assess_thermal_failure(
            material_id, temperatures, thermal_stress
        )
        
        return ThermalAnalysisResult(
            temperatures=temperatures,
            heat_flux=np.full_like(temperatures, q_wall),
            thermal_stress=thermal_stress,
            safety_factor=safety_factors,
            failure_mode=failure_modes
        )
    
    def calculate_thermal_stress_distribution(self, material_id: str,
                                            temperature_profile: np.ndarray,
                                            thickness: float) -> ThermalStressAnalysis:
        """
        Calculate detailed thermal stress distribution.
        
        Args:
            material_id: Material identifier
            temperature_profile: Temperature distribution through thickness
            thickness: Total thickness (m)
            
        Returns:
            ThermalStressAnalysis with detailed stress information
        """
        material = self.get_material(material_id)
        if not material or not material.mechanical_properties:
            raise ValueError(f"Material {material_id} missing mechanical properties")
        
        mech_props = material.mechanical_properties
        
        # Calculate thermal strain
        alpha = self._get_thermal_expansion_coefficient(material_id)
        T_ref = 293.0  # Reference temperature (K)
        thermal_strain = alpha * (temperature_profile - T_ref)
        
        # Calculate thermal stress (assuming constrained expansion)
        E = mech_props.youngs_modulus
        nu = mech_props.poissons_ratio
        
        # Plane stress assumption
        thermal_stress = E * thermal_strain / (1 - nu)
        
        # Calculate von Mises stress (simplified for 1D case)
        von_mises_stress = np.abs(thermal_stress)
        
        # Principal stresses (1D case)
        principal_stresses = np.column_stack([
            thermal_stress,
            np.zeros_like(thermal_stress),
            np.zeros_like(thermal_stress)
        ])
        
        # Safety factors
        yield_strength = mech_props.yield_strength
        safety_factors = yield_strength / (von_mises_stress + 1e-10)
        
        # Critical locations (safety factor < 2.0)
        critical_locations = np.where(safety_factors < 2.0)[0].tolist()
        
        return ThermalStressAnalysis(
            thermal_strain=thermal_strain,
            thermal_stress=thermal_stress,
            von_mises_stress=von_mises_stress,
            principal_stresses=principal_stresses,
            safety_factors=safety_factors,
            critical_locations=critical_locations
        )
    
    def optimize_material_selection(self, max_temperature: float,
                                   max_stress: float,
                                   weight_factor: float = 1.0) -> str:
        """
        Optimize material selection for given thermal conditions.
        
        Args:
            max_temperature: Maximum operating temperature (K)
            max_stress: Maximum allowable stress (Pa)
            weight_factor: Weight importance factor (0-1)
            
        Returns:
            Optimal material ID
        """
        self.logger.info(f"Optimizing material selection for T_max={max_temperature}K")
        
        suitable_materials = self.get_materials_for_temperature(max_temperature)
        
        if not suitable_materials:
            raise ValueError(f"No materials suitable for temperature {max_temperature}K")
        
        best_material = None
        best_score = float('inf')
        
        for mat_id in suitable_materials:
            material = self.get_material(mat_id)
            if not material or not material.mechanical_properties:
                continue
            
            mech_props = material.mechanical_properties
            thermal_props = material.thermal_properties
            
            # Check stress capability
            if mech_props.yield_strength < max_stress:
                continue
            
            # Calculate performance score
            # Lower density is better (weight factor)
            # Higher thermal conductivity is better (heat dissipation)
            # Higher operating temperature is better (margin)
            
            density_score = thermal_props.density * weight_factor
            conductivity_score = 1000.0 / thermal_props.thermal_conductivity  # Inverse
            temp_margin_score = 1000.0 / (thermal_props.operating_temp_range[1] - max_temperature)
            
            total_score = density_score + conductivity_score + temp_margin_score
            
            if total_score < best_score:
                best_score = total_score
                best_material = mat_id
        
        if best_material is None:
            raise ValueError("No suitable materials found for given conditions")
        
        self.logger.info(f"Selected optimal material: {best_material}")
        return best_material
    
    def optimize_extreme_thermal_material_selection(self, 
                                                   max_temperature: float,
                                                   heat_flux: float,
                                                   thermal_gradient: float,
                                                   mission_duration: float,
                                                   weight_factor: float = 1.0,
                                                   cost_factor: float = 0.1) -> Dict[str, Any]:
        """
        Advanced material selection for extreme thermal environments (>5000K).
        
        Args:
            max_temperature: Maximum operating temperature (K)
            heat_flux: Maximum heat flux (W/m²)
            thermal_gradient: Maximum thermal gradient (K/m)
            mission_duration: Mission duration (s)
            weight_factor: Weight importance (0-1)
            cost_factor: Cost importance (0-1)
            
        Returns:
            Dictionary with optimal material and performance metrics
        """
        self.logger.info(f"Optimizing extreme thermal material selection for T_max={max_temperature}K")
        
        # Get materials suitable for extreme temperatures
        if max_temperature >= 5000.0:
            suitable_materials = self.get_extreme_temperature_materials(max_temperature)
        else:
            suitable_materials = self.get_materials_for_temperature(max_temperature)
        
        if not suitable_materials:
            raise ValueError(f"No materials suitable for extreme temperature {max_temperature}K")
        
        best_material = None
        best_score = float('inf')
        material_rankings = []
        
        for mat_id in suitable_materials:
            material = self.get_material(mat_id)
            if not material or not material.thermal_properties:
                continue
            
            thermal_props = material.thermal_properties
            
            # Calculate performance metrics
            performance_metrics = self._calculate_extreme_thermal_performance(
                mat_id, max_temperature, heat_flux, thermal_gradient, mission_duration
            )
            
            if performance_metrics['thermal_failure_risk'] > 0.1:  # 10% failure risk threshold
                continue
            
            # Multi-objective optimization score
            score_components = {
                'thermal_performance': performance_metrics['thermal_performance_score'],
                'weight': thermal_props.density * weight_factor,
                'thermal_margin': 1000.0 / (thermal_props.operating_temp_range[1] - max_temperature + 1),
                'thermal_shock_resistance': performance_metrics['thermal_shock_resistance'],
                'cost': cost_factor * 1000  # Simplified cost model
            }
            
            # Weighted total score (lower is better)
            total_score = (
                score_components['thermal_performance'] * 0.4 +
                score_components['weight'] * 0.2 +
                score_components['thermal_margin'] * 0.2 +
                (1.0 / score_components['thermal_shock_resistance']) * 0.1 +
                score_components['cost'] * 0.1
            )
            
            material_rankings.append({
                'material_id': mat_id,
                'material_name': material.name,
                'total_score': total_score,
                'performance_metrics': performance_metrics,
                'score_components': score_components
            })
            
            if total_score < best_score:
                best_score = total_score
                best_material = mat_id
        
        if best_material is None:
            raise ValueError("No suitable materials found for extreme thermal conditions")
        
        # Sort rankings by score
        material_rankings.sort(key=lambda x: x['total_score'])
        
        self.logger.info(f"Selected optimal extreme thermal material: {best_material}")
        
        return {
            'optimal_material': best_material,
            'optimal_material_name': self.get_material(best_material).name,
            'performance_score': best_score,
            'material_rankings': material_rankings[:5],  # Top 5 materials
            'selection_criteria': {
                'max_temperature': max_temperature,
                'heat_flux': heat_flux,
                'thermal_gradient': thermal_gradient,
                'mission_duration': mission_duration,
                'weight_factor': weight_factor,
                'cost_factor': cost_factor
            }
        }
    
    def _calculate_extreme_thermal_performance(self, material_id: str, 
                                             max_temperature: float,
                                             heat_flux: float,
                                             thermal_gradient: float,
                                             mission_duration: float) -> Dict[str, float]:
        """Calculate performance metrics for extreme thermal conditions."""
        material = self.get_material(material_id)
        thermal_props = material.thermal_properties
        
        # Thermal performance score (lower is better)
        k_at_temp = self.calculate_thermal_conductivity(material_id, max_temperature)
        cp_at_temp = self.calculate_specific_heat(material_id, max_temperature)
        
        # Thermal diffusivity
        alpha = k_at_temp / (thermal_props.density * cp_at_temp)
        
        # Thermal performance score based on heat dissipation capability
        thermal_performance_score = heat_flux / (k_at_temp * thermal_gradient + 1e-6)
        
        # Thermal shock resistance (higher is better)
        # R = k * σ_f / (α * E) where σ_f is fracture strength, α is expansion coefficient
        alpha_expansion = self._get_thermal_expansion_coefficient(material_id)
        if material.mechanical_properties:
            E = material.mechanical_properties.youngs_modulus
            sigma_f = material.mechanical_properties.ultimate_strength
            thermal_shock_resistance = (k_at_temp * sigma_f) / (alpha_expansion * E + 1e-6)
        else:
            thermal_shock_resistance = k_at_temp / alpha_expansion  # Simplified
        
        # Thermal failure risk assessment
        temp_margin = thermal_props.operating_temp_range[1] - max_temperature
        thermal_failure_risk = max(0.0, 1.0 - temp_margin / 500.0)  # Risk increases as margin decreases
        
        # Mission duration factor
        time_factor = min(1.0, mission_duration / 3600.0)  # Normalize to 1 hour
        
        return {
            'thermal_performance_score': thermal_performance_score,
            'thermal_shock_resistance': thermal_shock_resistance,
            'thermal_failure_risk': thermal_failure_risk,
            'thermal_diffusivity': alpha,
            'time_factor': time_factor,
            'temperature_margin': temp_margin
        }
    
    def analyze_ablative_thermal_protection(self, base_material_id: str, 
                                          ablative_material_id: str,
                                          conditions: HypersonicConditions,
                                          ablative_thickness: float,
                                          base_thickness: float = 0.01) -> Dict[str, Any]:
        """
        Analyze thermal protection system with ablative cooling.
        
        Args:
            base_material_id: Base structural material ID
            ablative_material_id: Ablative material ID
            conditions: Hypersonic flight conditions
            ablative_thickness: Ablative layer thickness (m)
            base_thickness: Base material thickness (m)
            
        Returns:
            Combined thermal and ablative analysis results
        """
        self.logger.info(f"Analyzing ablative thermal protection system")
        
        # Get materials
        base_material = self.get_material(base_material_id)
        if not base_material:
            raise ValueError(f"Base material {base_material_id} not found")
        
        # Calculate heat flux without ablation
        T_stag = self._calculate_stagnation_temperature(conditions)
        q_wall_initial = self._calculate_wall_heat_flux(conditions, T_stag)
        
        # Create time and heat flux profiles for ablative analysis
        n_points = 50
        time_profile = np.linspace(0, conditions.flight_time, n_points)
        heat_flux_profile = np.full(n_points, q_wall_initial)
        pressure_profile = np.full(n_points, 1000.0)  # Simplified pressure
        
        # Analyze ablative cooling
        ablative_density = 1500.0  # kg/m³ (typical for ablatives)
        ablative_result = self.ablative_model.analyze_ablative_cooling(
            ablative_material_id, heat_flux_profile, pressure_profile, 
            time_profile, ablative_thickness, ablative_density
        )
        
        # Calculate reduced heat flux to base material
        reduced_heat_flux = q_wall_initial - np.mean(ablative_result.heat_flux_reduction)
        reduced_heat_flux = max(reduced_heat_flux, 0.1 * q_wall_initial)  # Minimum 10% of original
        
        # Analyze base material with reduced heat flux
        base_conditions = HypersonicConditions(
            mach_number=conditions.mach_number,
            altitude=conditions.altitude,
            flight_time=conditions.flight_time,
            angle_of_attack=conditions.angle_of_attack,
            surface_emissivity=conditions.surface_emissivity,
            recovery_factor=conditions.recovery_factor
        )
        
        # Modify the heat flux calculation for base material analysis
        original_method = self._calculate_wall_heat_flux
        def modified_heat_flux(cond, T_stag):
            return reduced_heat_flux
        
        self._calculate_wall_heat_flux = modified_heat_flux
        
        try:
            base_thermal_result = self.analyze_hypersonic_heating(
                base_material_id, base_conditions, base_thickness
            )
        finally:
            # Restore original method
            self._calculate_wall_heat_flux = original_method
        
        # Calculate overall system performance
        total_thickness = ablative_thickness + base_thickness
        mass_per_area = (ablative_density * ablative_thickness + 
                        base_material.thermal_properties.density * base_thickness)
        
        # System effectiveness
        if q_wall_initial > 0:
            heat_flux_reduction_percent = (np.mean(ablative_result.heat_flux_reduction) / 
                                         q_wall_initial * 100)
        else:
            heat_flux_reduction_percent = 0.0
        
        return {
            'ablative_analysis': {
                'material_id': ablative_material_id,
                'initial_thickness': ablative_thickness,
                'remaining_thickness': ablative_result.remaining_thickness,
                'total_mass_loss': ablative_result.total_mass_loss,
                'average_cooling_effectiveness': np.mean(ablative_result.cooling_effectiveness),
                'max_recession_rate': np.max(ablative_result.recession_rate),
                'final_char_thickness': ablative_result.char_thickness[-1]
            },
            'base_material_analysis': {
                'material_id': base_material_id,
                'max_temperature': np.max(base_thermal_result.temperatures),
                'max_thermal_stress': np.max(base_thermal_result.thermal_stress),
                'min_safety_factor': np.min(base_thermal_result.safety_factor)
            },
            'system_performance': {
                'total_thickness': total_thickness,
                'total_mass_per_area': mass_per_area,
                'heat_flux_reduction_percent': heat_flux_reduction_percent,
                'original_heat_flux': q_wall_initial,
                'reduced_heat_flux': reduced_heat_flux,
                'system_survival': (ablative_result.remaining_thickness > 0 and 
                                  np.min(base_thermal_result.safety_factor) > 1.0)
            },
            'time_profiles': {
                'time': time_profile,
                'ablative_recession_rate': ablative_result.recession_rate,
                'ablative_cooling_effectiveness': ablative_result.cooling_effectiveness,
                'base_material_temperatures': base_thermal_result.temperatures
            }
        }
    
    def optimize_ablative_thermal_protection_system(self, base_material_id: str,
                                                   conditions: HypersonicConditions,
                                                   max_total_thickness: float = 0.1,
                                                   max_mass_per_area: float = 100.0) -> Dict[str, Any]:
        """
        Optimize ablative thermal protection system design.
        
        Args:
            base_material_id: Base structural material ID
            conditions: Hypersonic flight conditions
            max_total_thickness: Maximum total thickness constraint (m)
            max_mass_per_area: Maximum mass per area constraint (kg/m²)
            
        Returns:
            Optimized TPS design
        """
        self.logger.info("Optimizing ablative thermal protection system")
        
        best_design = None
        best_score = float('inf')
        design_options = []
        
        # Test different ablative materials
        ablative_materials = self.ablative_model.list_ablative_materials()
        
        for ablative_id in ablative_materials.keys():
            # Estimate required ablative thickness
            T_stag = self._calculate_stagnation_temperature(conditions)
            q_wall = self._calculate_wall_heat_flux(conditions, T_stag)
            
            required_thickness = self.ablative_model.optimize_ablative_thickness(
                ablative_id, q_wall, conditions.flight_time, safety_factor=1.5
            )
            
            # Test different thickness ratios
            for thickness_ratio in [0.3, 0.5, 0.7]:  # Ablative fraction of total thickness
                ablative_thickness = min(required_thickness, 
                                       max_total_thickness * thickness_ratio)
                base_thickness = min(0.02, max_total_thickness - ablative_thickness)
                
                if ablative_thickness <= 0 or base_thickness <= 0:
                    continue
                
                try:
                    # Analyze this configuration
                    analysis = self.analyze_ablative_thermal_protection(
                        base_material_id, ablative_id, conditions,
                        ablative_thickness, base_thickness
                    )
                    
                    # Check constraints
                    if (analysis['system_performance']['total_thickness'] > max_total_thickness or
                        analysis['system_performance']['total_mass_per_area'] > max_mass_per_area):
                        continue
                    
                    # Check survival
                    if not analysis['system_performance']['system_survival']:
                        continue
                    
                    # Calculate design score (lower is better)
                    score = (
                        analysis['system_performance']['total_mass_per_area'] * 0.4 +
                        analysis['system_performance']['total_thickness'] * 1000 * 0.3 +
                        (1.0 / analysis['base_material_analysis']['min_safety_factor']) * 0.3
                    )
                    
                    design_option = {
                        'ablative_material': ablative_id,
                        'ablative_thickness': ablative_thickness,
                        'base_thickness': base_thickness,
                        'score': score,
                        'analysis': analysis
                    }
                    
                    design_options.append(design_option)
                    
                    if score < best_score:
                        best_score = score
                        best_design = design_option
                
                except Exception as e:
                    self.logger.warning(f"Failed to analyze design {ablative_id}: {e}")
                    continue
        
        if best_design is None:
            raise ValueError("No viable ablative TPS design found within constraints")
        
        # Sort design options by score
        design_options.sort(key=lambda x: x['score'])
        
        return {
            'optimal_design': best_design,
            'design_alternatives': design_options[:5],  # Top 5 alternatives
            'constraints': {
                'max_total_thickness': max_total_thickness,
                'max_mass_per_area': max_mass_per_area
            },
            'conditions': {
                'mach_number': conditions.mach_number,
                'altitude': conditions.altitude,
                'flight_time': conditions.flight_time
            }
        }
    
    def _calculate_thermal_conductivity_vs_temperature(self, k_ref: float, 
                                                     temperatures: np.ndarray,
                                                     material_id: str) -> np.ndarray:
        """Calculate temperature-dependent thermal conductivity up to 6000K."""
        T_ref = 293.0  # Reference temperature (K)
        T_norm = temperatures / T_ref
        
        # Enhanced models for extreme temperatures
        if 'hfc' in material_id or 'tac' in material_id or 'hftac' in material_id:
            # Carbides: k(T) = k_ref * (T_ref/T)^n with temperature-dependent n
            n_base = 0.3
            # Adjust exponent for extreme temperatures
            n = n_base * (1 + 0.1 * np.log(T_norm))
            k_values = k_ref * (T_ref / temperatures) ** n
            
        elif 'w' in material_id:
            # Pure tungsten: more complex temperature dependence
            # k(T) = k_ref * (T_ref/T)^0.1 * exp(-T/10000) for T > 3000K
            n = 0.1
            k_values = k_ref * (T_ref / temperatures) ** n
            # Add exponential decay at very high temperatures
            high_temp_mask = temperatures > 3000
            k_values[high_temp_mask] *= np.exp(-(temperatures[high_temp_mask] - 3000) / 10000)
            
        elif 'hfn' in material_id:
            # Nitrides: moderate temperature dependence
            n = 0.25 * (1 + 0.05 * np.log(T_norm))
            k_values = k_ref * (T_ref / temperatures) ** n
            
        elif 're' in material_id:
            # Rhenium: metallic behavior with phonon scattering
            n = 0.1 + 0.05 * np.log(T_norm)
            k_values = k_ref * (T_ref / temperatures) ** n
            
        elif 'cc' in material_id:
            # Carbon-Carbon: anisotropic, use in-plane values
            # Thermal conductivity increases with temperature up to ~2000K, then decreases
            k_values = np.where(
                temperatures < 2000,
                k_ref * (1 + 0.0002 * (temperatures - T_ref)),
                k_ref * 1.34 * (2000 / temperatures) ** 0.2
            )
            
        elif 'b4c' in material_id:
            # Boron Carbide: ceramic behavior
            n = 0.35
            k_values = k_ref * (T_ref / temperatures) ** n
            
        else:
            # Default ceramic behavior
            n = 0.2 + 0.1 * np.log(T_norm)
            k_values = k_ref * (T_ref / temperatures) ** n
        
        # Ensure reasonable physical bounds for extreme temperatures
        # Minimum: 10% of reference value
        # Maximum: 3x reference value (for some materials at low temps)
        k_values = np.clip(k_values, k_ref * 0.1, k_ref * 3.0)
        
        return k_values
    
    def _calculate_specific_heat_vs_temperature(self, cp_ref: float,
                                              temperatures: np.ndarray,
                                              material_id: str) -> np.ndarray:
        """Calculate temperature-dependent specific heat up to 6000K."""
        T_ref = 293.0
        T_norm = temperatures / T_ref
        
        # Enhanced models for extreme temperatures
        if 'hfc' in material_id or 'tac' in material_id or 'hftac' in material_id:
            # Carbides: Debye model with electronic contribution
            # cp(T) = cp_ref * (1 + a*T + b*T^2 + c*T^3) for high temperatures
            a, b, c = 0.0003, -2e-8, 1e-12
            cp_values = cp_ref * (1 + a * (temperatures - T_ref) + 
                                b * (temperatures - T_ref)**2 + 
                                c * (temperatures - T_ref)**3)
            
        elif 'w' in material_id:
            # Pure tungsten: metallic specific heat with electronic contribution
            # cp = cp_lattice + cp_electronic
            # Electronic contribution becomes significant at high T
            cp_lattice = cp_ref * (1 + 0.0001 * (temperatures - T_ref))
            cp_electronic = 0.01 * temperatures  # Linear electronic contribution
            cp_values = cp_lattice + cp_electronic
            
        elif 'hfn' in material_id:
            # Nitrides: similar to carbides but different coefficients
            a, b, c = 0.0004, -2.5e-8, 1.2e-12
            cp_values = cp_ref * (1 + a * (temperatures - T_ref) + 
                                b * (temperatures - T_ref)**2 + 
                                c * (temperatures - T_ref)**3)
            
        elif 're' in material_id:
            # Rhenium: metallic behavior
            cp_lattice = cp_ref * (1 + 0.0001 * (temperatures - T_ref))
            cp_electronic = 0.008 * temperatures
            cp_values = cp_lattice + cp_electronic
            
        elif 'cc' in material_id:
            # Carbon-Carbon: graphitic behavior
            # Specific heat increases significantly with temperature
            cp_values = cp_ref * (1 + 0.0008 * (temperatures - T_ref) - 
                                3e-8 * (temperatures - T_ref)**2)
            
        elif 'b4c' in material_id:
            # Boron Carbide: ceramic with high temperature stability
            a, b = 0.0005, -3e-8
            cp_values = cp_ref * (1 + a * (temperatures - T_ref) + 
                                b * (temperatures - T_ref)**2)
            
        else:
            # Default ceramic behavior
            a, b, c = 0.0003, -1.5e-8, 8e-13
            cp_values = cp_ref * (1 + a * (temperatures - T_ref) + 
                                b * (temperatures - T_ref)**2 + 
                                c * (temperatures - T_ref)**3)
        
        # Physical bounds for extreme temperatures
        # Minimum: 50% of reference (some materials decrease at very high T)
        # Maximum: 5x reference (significant increase possible for some materials)
        cp_values = np.clip(cp_values, cp_ref * 0.5, cp_ref * 5.0)
        
        return cp_values
    
    def _calculate_stagnation_temperature(self, conditions: HypersonicConditions) -> float:
        """Calculate stagnation temperature for hypersonic flow."""
        # Standard atmosphere model
        if conditions.altitude <= 11000:  # Troposphere
            T_ambient = 288.15 - 0.0065 * conditions.altitude
        elif conditions.altitude <= 20000:  # Lower stratosphere
            T_ambient = 216.65
        else:  # Simplified for higher altitudes
            T_ambient = 216.65 * np.exp(-(conditions.altitude - 20000) / 6000)
        
        # Stagnation temperature
        gamma = 1.4  # Specific heat ratio for air
        M = conditions.mach_number
        
        T_stag = T_ambient * (1 + (gamma - 1) / 2 * M**2)
        
        return T_stag
    
    def _calculate_wall_heat_flux(self, conditions: HypersonicConditions, T_stag: float) -> float:
        """Calculate wall heat flux for hypersonic conditions."""
        # Simplified heat transfer correlation
        # q = h * (T_recovery - T_wall)
        
        # Recovery temperature
        T_recovery = T_stag * conditions.recovery_factor
        
        # Assume wall temperature is much lower initially
        T_wall = 300.0  # K (initial guess)
        
        # Heat transfer coefficient (simplified correlation)
        # Based on Fay-Riddell stagnation point heating
        rho_inf = self._get_atmospheric_density(conditions.altitude)
        V_inf = conditions.mach_number * self._get_speed_of_sound(conditions.altitude)
        
        # Simplified heat transfer coefficient
        h = 0.76 * (rho_inf * V_inf)**0.5 * (T_recovery / T_wall)**0.1
        
        # Heat flux
        q_wall = h * (T_recovery - T_wall)
        
        return max(q_wall, 0.0)
    
    def _solve_transient_heat_conduction(self, material_id: str, q_wall: float,
                                       thickness: float, flight_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """Solve 1D transient heat conduction equation."""
        # Simplified finite difference solution
        
        # Spatial discretization
        n_nodes = 21
        dx = thickness / (n_nodes - 1)
        x = np.linspace(0, thickness, n_nodes)
        
        # Time discretization
        n_time = 101
        dt = flight_time / (n_time - 1)
        times = np.linspace(0, flight_time, n_time)
        
        # Initialize temperature field
        T_initial = 293.0  # K
        T = np.full((n_time, n_nodes), T_initial)
        
        # Material properties (simplified - use average values)
        material = self.get_material(material_id)
        thermal_props = material.thermal_properties
        
        k = thermal_props.thermal_conductivity
        rho = thermal_props.density
        cp = thermal_props.specific_heat
        alpha_thermal = k / (rho * cp)  # Thermal diffusivity
        
        # Stability criterion - adjust time step if needed
        r = alpha_thermal * dt / dx**2
        if r > 0.5:
            # Reduce time step to ensure stability
            dt_stable = 0.4 * dx**2 / alpha_thermal
            n_time = max(int(flight_time / dt_stable) + 1, 101)
            dt = flight_time / (n_time - 1)
            times = np.linspace(0, flight_time, n_time)
            T = np.full((n_time, n_nodes), T_initial)
            r = alpha_thermal * dt / dx**2
            self.logger.info(f"Adjusted time step for stability: r = {r:.3f}")
        
        # Time stepping
        for t_idx in range(1, n_time):
            T_old = T[t_idx - 1, :].copy()
            
            # Interior nodes (explicit finite difference)
            for i in range(1, n_nodes - 1):
                T[t_idx, i] = T_old[i] + r * (T_old[i+1] - 2*T_old[i] + T_old[i-1])
            
            # Boundary conditions
            # Left boundary (heated surface): heat flux BC
            T[t_idx, 0] = T[t_idx, 1] + q_wall * dx / k
            
            # Right boundary (insulated): zero gradient
            T[t_idx, -1] = T[t_idx, -2]
        
        # Return final temperature distribution
        return T[-1, :], times
    
    def _calculate_thermal_stress(self, material_id: str, temperatures: np.ndarray,
                                thickness: float) -> np.ndarray:
        """Calculate thermal stress distribution."""
        material = self.get_material(material_id)
        if not material or not material.mechanical_properties:
            return np.zeros_like(temperatures)
        
        mech_props = material.mechanical_properties
        
        # Thermal expansion coefficient (simplified)
        alpha = self._get_thermal_expansion_coefficient(material_id)
        
        # Reference temperature
        T_ref = 293.0  # K
        
        # Thermal strain
        thermal_strain = alpha * (temperatures - T_ref)
        
        # Thermal stress (assuming constrained expansion)
        E = mech_props.youngs_modulus
        nu = mech_props.poissons_ratio
        
        thermal_stress = E * thermal_strain / (1 - nu)
        
        return thermal_stress
    
    def _assess_thermal_failure(self, material_id: str, temperatures: np.ndarray,
                              thermal_stress: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Assess thermal failure modes and safety factors."""
        material = self.get_material(material_id)
        if not material:
            return np.ones_like(temperatures), ['unknown'] * len(temperatures)
        
        thermal_props = material.thermal_properties
        mech_props = material.mechanical_properties
        
        safety_factors = np.ones_like(temperatures, dtype=float)
        failure_modes = []
        
        for i, (T, stress) in enumerate(zip(temperatures, thermal_stress)):
            failure_mode = 'safe'
            sf = 100.0  # Use large finite value instead of infinity
            
            # Temperature limit check
            if thermal_props and T > thermal_props.operating_temp_range[1]:
                sf_temp = thermal_props.operating_temp_range[1] / T
                if sf_temp < sf:
                    sf = sf_temp
                    failure_mode = 'temperature_limit'
            
            # Stress limit check
            if mech_props and abs(stress) > 0:
                sf_stress = mech_props.yield_strength / abs(stress)
                if sf_stress < sf:
                    sf = sf_stress
                    failure_mode = 'thermal_stress'
            
            # Melting point check
            if thermal_props and T > thermal_props.melting_point:
                sf_melt = thermal_props.melting_point / T
                if sf_melt < sf:
                    sf = sf_melt
                    failure_mode = 'melting'
            
            safety_factors[i] = sf
            failure_modes.append(failure_mode)
        
        return safety_factors, failure_modes
    
    def _get_thermal_expansion_coefficient(self, material_id: str) -> float:
        """Get thermal expansion coefficient for material."""
        # Simplified values for UHTC materials
        expansion_coefficients = {
            'hfc': 6.6e-6,   # /K
            'tac': 6.3e-6,   # /K
            'wc': 5.2e-6,    # /K
            'zrc': 7.1e-6,   # /K
            'hfn': 7.8e-6,   # /K
            're': 6.2e-6     # /K
        }
        
        return expansion_coefficients.get(material_id, 7.0e-6)  # Default value
    
    def _get_atmospheric_density(self, altitude: float) -> float:
        """Get atmospheric density at altitude."""
        # Standard atmosphere model (simplified)
        if altitude <= 11000:  # Troposphere
            T = 288.15 - 0.0065 * altitude
            p = 101325 * (T / 288.15)**5.256
        elif altitude <= 20000:  # Lower stratosphere
            T = 216.65
            p = 22632 * np.exp(-(altitude - 11000) / 6341.6)
        else:  # Simplified for higher altitudes
            T = 216.65
            p = 5474.9 * np.exp(-(altitude - 20000) / 7922.3)
        
        # Density from ideal gas law
        R_specific = 287.0  # J/(kg⋅K) for air
        rho = p / (R_specific * T)
        
        return rho
    
    def _get_speed_of_sound(self, altitude: float) -> float:
        """Get speed of sound at altitude."""
        # Temperature from standard atmosphere
        if altitude <= 11000:
            T = 288.15 - 0.0065 * altitude
        else:
            T = 216.65
        
        gamma = 1.4
        R_specific = 287.0  # J/(kg⋅K)
        
        a = np.sqrt(gamma * R_specific * T)
        
        return a


class AblativeCoolingModel:
    """Model for ablative cooling analysis and mass loss calculations."""
    
    def __init__(self):
        """Initialize the ablative cooling model."""
        self.logger = get_engine_logger('materials.ablative')
        
        # Initialize ablative materials database
        self._ablative_materials = {}
        self._initialize_ablative_materials()
    
    def _initialize_ablative_materials(self):
        """Initialize database with ablative materials and their properties."""
        
        # Carbon-Carbon Ablative
        cc_ablative = AblativeProperties(
            heat_of_ablation=32e6,  # J/kg - high for carbon materials
            char_yield=0.85,  # High char yield for carbon-carbon
            pyrolysis_temperature=1273.0,  # K (1000°C)
            surface_emissivity=0.9,  # High emissivity for carbon
            blowing_parameter=0.3,  # Moderate mass transfer
            recession_rate_coefficient=1e-8  # m/(s⋅Pa)
        )
        self._ablative_materials['cc_ablative'] = cc_ablative
        
        # Phenolic Resin Ablative
        phenolic_ablative = AblativeProperties(
            heat_of_ablation=25e6,  # J/kg
            char_yield=0.6,  # Moderate char yield
            pyrolysis_temperature=573.0,  # K (300°C)
            surface_emissivity=0.85,
            blowing_parameter=0.5,  # Higher mass transfer
            recession_rate_coefficient=2e-8  # m/(s⋅Pa)
        )
        self._ablative_materials['phenolic'] = phenolic_ablative
        
        # PICA (Phenolic Impregnated Carbon Ablator)
        pica_ablative = AblativeProperties(
            heat_of_ablation=28e6,  # J/kg
            char_yield=0.75,  # Good char yield
            pyrolysis_temperature=673.0,  # K (400°C)
            surface_emissivity=0.88,
            blowing_parameter=0.4,
            recession_rate_coefficient=1.5e-8  # m/(s⋅Pa)
        )
        self._ablative_materials['pica'] = pica_ablative
        
        # Ultra-High Temperature Ablative (Hafnium Carbide based)
        uht_ablative = AblativeProperties(
            heat_of_ablation=45e6,  # J/kg - very high for refractory materials
            char_yield=0.95,  # Very high char yield
            pyrolysis_temperature=2273.0,  # K (2000°C)
            surface_emissivity=0.7,  # Lower emissivity for metals
            blowing_parameter=0.2,  # Lower mass transfer
            recession_rate_coefficient=5e-9  # m/(s⋅Pa) - very resistant
        )
        self._ablative_materials['uht_ablative'] = uht_ablative
        
        self.logger.info(f"Initialized ablative materials database with {len(self._ablative_materials)} materials")
    
    def get_ablative_material(self, material_id: str) -> Optional[AblativeProperties]:
        """Get ablative material properties by ID."""
        return self._ablative_materials.get(material_id)
    
    def list_ablative_materials(self) -> Dict[str, str]:
        """List all ablative materials."""
        material_names = {
            'cc_ablative': 'Carbon-Carbon Ablative',
            'phenolic': 'Phenolic Resin Ablative',
            'pica': 'PICA (Phenolic Impregnated Carbon Ablator)',
            'uht_ablative': 'Ultra-High Temperature Ablative'
        }
        return {mat_id: material_names.get(mat_id, mat_id) for mat_id in self._ablative_materials.keys()}
    
    def calculate_recession_rate(self, material_id: str, heat_flux: float, 
                               surface_pressure: float, surface_temperature: float) -> float:
        """
        Calculate material recession rate.
        
        Args:
            material_id: Ablative material identifier
            heat_flux: Heat flux at surface (W/m²)
            surface_pressure: Surface pressure (Pa)
            surface_temperature: Surface temperature (K)
            
        Returns:
            Recession rate (m/s)
        """
        ablative_props = self.get_ablative_material(material_id)
        if not ablative_props:
            raise ValueError(f"Ablative material {material_id} not found")
        
        # Recession rate model: ṡ = C * P^n * (q/H_abl)^m
        # where C is recession coefficient, P is pressure, q is heat flux, H_abl is heat of ablation
        
        # Pressure exponent (typically 0.5-0.8)
        n = 0.6
        # Heat flux exponent (typically 0.8-1.0)
        m = 0.9
        
        # Calculate recession rate
        recession_rate = (ablative_props.recession_rate_coefficient * 
                         (surface_pressure ** n) * 
                         ((heat_flux / ablative_props.heat_of_ablation) ** m))
        
        # Temperature correction factor
        if surface_temperature > ablative_props.pyrolysis_temperature:
            temp_factor = (surface_temperature / ablative_props.pyrolysis_temperature) ** 0.5
            recession_rate *= temp_factor
        
        return max(recession_rate, 0.0)
    
    def calculate_mass_loss_rate(self, material_id: str, recession_rate: float, 
                               material_density: float) -> float:
        """
        Calculate mass loss rate per unit area.
        
        Args:
            material_id: Ablative material identifier
            recession_rate: Surface recession rate (m/s)
            material_density: Material density (kg/m³)
            
        Returns:
            Mass loss rate (kg/(m²⋅s))
        """
        ablative_props = self.get_ablative_material(material_id)
        if not ablative_props:
            raise ValueError(f"Ablative material {material_id} not found")
        
        # Mass loss rate = recession rate * density * (1 - char yield)
        # Char yield represents the fraction that remains as char layer
        effective_density = material_density * (1.0 - ablative_props.char_yield)
        mass_loss_rate = recession_rate * effective_density
        
        return mass_loss_rate
    
    def calculate_cooling_effectiveness(self, material_id: str, heat_flux: float,
                                      mass_loss_rate: float) -> float:
        """
        Calculate ablative cooling effectiveness.
        
        Args:
            material_id: Ablative material identifier
            heat_flux: Incident heat flux (W/m²)
            mass_loss_rate: Mass loss rate (kg/(m²⋅s))
            
        Returns:
            Cooling effectiveness (dimensionless, 0-1)
        """
        ablative_props = self.get_ablative_material(material_id)
        if not ablative_props:
            raise ValueError(f"Ablative material {material_id} not found")
        
        # Cooling effectiveness = (heat absorbed by ablation) / (incident heat flux)
        heat_absorbed = mass_loss_rate * ablative_props.heat_of_ablation
        
        if heat_flux <= 0:
            return 0.0
        
        effectiveness = min(heat_absorbed / heat_flux, 1.0)
        
        # Account for blowing effect (mass transfer reduces heat transfer)
        blowing_factor = 1.0 - ablative_props.blowing_parameter * (mass_loss_rate / 10.0)  # Normalized
        effectiveness *= max(blowing_factor, 0.1)  # Minimum 10% effectiveness
        
        return effectiveness
    
    def calculate_char_layer_growth(self, material_id: str, recession_rate: float,
                                  char_thickness: float, time_step: float) -> float:
        """
        Calculate char layer thickness growth.
        
        Args:
            material_id: Ablative material identifier
            recession_rate: Surface recession rate (m/s)
            char_thickness: Current char thickness (m)
            time_step: Time step (s)
            
        Returns:
            New char thickness (m)
        """
        ablative_props = self.get_ablative_material(material_id)
        if not ablative_props:
            raise ValueError(f"Ablative material {material_id} not found")
        
        # Char layer grows as material ablates
        char_growth_rate = recession_rate * ablative_props.char_yield
        
        # Char layer also recedes due to oxidation (simplified model)
        char_recession_rate = recession_rate * 0.1  # 10% of surface recession
        
        net_char_growth = char_growth_rate - char_recession_rate
        new_char_thickness = char_thickness + net_char_growth * time_step
        
        return max(new_char_thickness, 0.0)
    
    def analyze_ablative_cooling(self, material_id: str, heat_flux_profile: np.ndarray,
                               pressure_profile: np.ndarray, time_profile: np.ndarray,
                               initial_thickness: float, material_density: float) -> AblativeCoolingResult:
        """
        Comprehensive ablative cooling analysis over time.
        
        Args:
            material_id: Ablative material identifier
            heat_flux_profile: Heat flux vs time (W/m²)
            pressure_profile: Pressure vs time (Pa)
            time_profile: Time points (s)
            initial_thickness: Initial material thickness (m)
            material_density: Material density (kg/m³)
            
        Returns:
            AblativeCoolingResult with complete analysis
        """
        self.logger.info(f"Analyzing ablative cooling for {material_id}")
        
        ablative_props = self.get_ablative_material(material_id)
        if not ablative_props:
            raise ValueError(f"Ablative material {material_id} not found")
        
        n_points = len(time_profile)
        
        # Initialize arrays
        recession_rate = np.zeros(n_points)
        mass_loss_rate = np.zeros(n_points)
        cooling_effectiveness = np.zeros(n_points)
        char_thickness = np.zeros(n_points)
        surface_temperature = np.zeros(n_points)
        heat_flux_reduction = np.zeros(n_points)
        
        # Initial conditions
        current_thickness = initial_thickness
        current_char_thickness = 0.0
        total_mass_loss = 0.0
        
        # Time stepping analysis
        for i in range(n_points):
            if i == 0:
                dt = 0.0
            else:
                dt = time_profile[i] - time_profile[i-1]
            
            # Estimate surface temperature (simplified energy balance)
            surface_temperature[i] = self._estimate_surface_temperature(
                heat_flux_profile[i], ablative_props
            )
            
            # Calculate recession rate
            recession_rate[i] = self.calculate_recession_rate(
                material_id, heat_flux_profile[i], pressure_profile[i], surface_temperature[i]
            )
            
            # Calculate mass loss rate
            mass_loss_rate[i] = self.calculate_mass_loss_rate(
                material_id, recession_rate[i], material_density
            )
            
            # Calculate cooling effectiveness
            cooling_effectiveness[i] = self.calculate_cooling_effectiveness(
                material_id, heat_flux_profile[i], mass_loss_rate[i]
            )
            
            # Update char thickness
            if dt > 0:
                current_char_thickness = self.calculate_char_layer_growth(
                    material_id, recession_rate[i], current_char_thickness, dt
                )
                
                # Update material thickness
                current_thickness -= recession_rate[i] * dt
                current_thickness = max(current_thickness, 0.0)
                
                # Accumulate mass loss
                total_mass_loss += mass_loss_rate[i] * dt
            
            char_thickness[i] = current_char_thickness
            
            # Calculate heat flux reduction
            heat_flux_reduction[i] = heat_flux_profile[i] * cooling_effectiveness[i]
        
        return AblativeCoolingResult(
            recession_rate=recession_rate,
            mass_loss_rate=mass_loss_rate,
            cooling_effectiveness=cooling_effectiveness,
            char_thickness=char_thickness,
            surface_temperature=surface_temperature,
            heat_flux_reduction=heat_flux_reduction,
            total_mass_loss=total_mass_loss,
            remaining_thickness=current_thickness
        )
    
    def optimize_ablative_thickness(self, material_id: str, max_heat_flux: float,
                                  mission_duration: float, safety_factor: float = 2.0) -> float:
        """
        Optimize ablative material thickness for given conditions.
        
        Args:
            material_id: Ablative material identifier
            max_heat_flux: Maximum heat flux (W/m²)
            mission_duration: Mission duration (s)
            safety_factor: Safety factor for thickness
            
        Returns:
            Recommended thickness (m)
        """
        ablative_props = self.get_ablative_material(material_id)
        if not ablative_props:
            raise ValueError(f"Ablative material {material_id} not found")
        
        # Estimate average recession rate with more realistic conditions
        avg_pressure = 5000.0  # Pa (higher pressure for hypersonic conditions)
        avg_temperature = max(3000.0, ablative_props.pyrolysis_temperature * 1.5)  # Higher temperature
        
        avg_recession_rate = self.calculate_recession_rate(
            material_id, max_heat_flux, avg_pressure, avg_temperature
        )
        
        # Ensure minimum recession rate for extreme conditions
        min_recession_rate = 1e-6  # 1 μm/s minimum
        avg_recession_rate = max(avg_recession_rate, min_recession_rate)
        
        # Calculate total recession over mission
        total_recession = avg_recession_rate * mission_duration
        
        # Add safety factor
        required_thickness = total_recession * safety_factor
        
        # Ensure minimum thickness for structural integrity
        min_thickness = 0.001  # 1 mm minimum
        required_thickness = max(required_thickness, min_thickness)
        
        self.logger.info(f"Recommended ablative thickness for {material_id}: {required_thickness:.4f} m")
        
        return required_thickness
    
    def _estimate_surface_temperature(self, heat_flux: float, 
                                    ablative_props: AblativeProperties) -> float:
        """Estimate surface temperature from energy balance."""
        # Simplified energy balance: q_in = q_ablation + q_radiation
        # q_radiation = ε * σ * T^4
        
        stefan_boltzmann = 5.670374419e-8  # W/(m²⋅K⁴)
        
        # Assume some fraction of heat flux goes to ablation
        ablation_fraction = 0.7  # 70% of heat flux absorbed by ablation
        radiation_heat_flux = heat_flux * (1.0 - ablation_fraction)
        
        # Solve for temperature: T = (q_rad / (ε * σ))^(1/4)
        if radiation_heat_flux > 0:
            T_surface = (radiation_heat_flux / 
                        (ablative_props.surface_emissivity * stefan_boltzmann)) ** 0.25
        else:
            T_surface = ablative_props.pyrolysis_temperature
        
        # Ensure temperature is at least pyrolysis temperature
        T_surface = max(T_surface, ablative_props.pyrolysis_temperature)
        
        return T_surface