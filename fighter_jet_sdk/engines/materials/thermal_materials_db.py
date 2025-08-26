"""Thermal materials database and modeling for ultra-high temperature applications."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
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
    
    def _initialize_uhtc_database(self):
        """Initialize database with ultra-high temperature ceramics."""
        
        # Hafnium Carbide (HfC)
        hfc_material = MaterialDefinition(
            name="Hafnium Carbide (HfC)",
            base_material_type=MaterialType.ULTRA_HIGH_TEMP_CERAMIC,
            thermal_properties=ThermalProperties(
                thermal_conductivity=22.0,  # W/(m⋅K) at room temperature
                specific_heat=200.0,  # J/(kg⋅K) at room temperature
                density=12800.0,  # kg/m³
                melting_point=4273.0,  # K (4000°C)
                operating_temp_range=(293.0, 3773.0)  # 20°C to 3500°C
            )
        )
        self._uhtc_materials['hfc'] = hfc_material
        
        # Tantalum Carbide (TaC)
        tac_material = MaterialDefinition(
            name="Tantalum Carbide (TaC)",
            base_material_type=MaterialType.ULTRA_HIGH_TEMP_CERAMIC,
            thermal_properties=ThermalProperties(
                thermal_conductivity=20.0,  # W/(m⋅K)
                specific_heat=180.0,  # J/(kg⋅K)
                density=14500.0,  # kg/m³
                melting_point=4258.0,  # K (3985°C)
                operating_temp_range=(293.0, 3758.0)
            )
        )
        self._uhtc_materials['tac'] = tac_material
        
        # Tungsten Carbide (WC)
        wc_material = MaterialDefinition(
            name="Tungsten Carbide (WC)",
            base_material_type=MaterialType.ULTRA_HIGH_TEMP_CERAMIC,
            thermal_properties=ThermalProperties(
                thermal_conductivity=85.0,  # W/(m⋅K)
                specific_heat=200.0,  # J/(kg⋅K)
                density=15600.0,  # kg/m³
                melting_point=3058.0,  # K (2785°C)
                operating_temp_range=(293.0, 2758.0)
            )
        )
        self._uhtc_materials['wc'] = wc_material
        
        # Zirconium Carbide (ZrC)
        zrc_material = MaterialDefinition(
            name="Zirconium Carbide (ZrC)",
            base_material_type=MaterialType.ULTRA_HIGH_TEMP_CERAMIC,
            thermal_properties=ThermalProperties(
                thermal_conductivity=18.0,  # W/(m⋅K)
                specific_heat=280.0,  # J/(kg⋅K)
                density=6730.0,  # kg/m³
                melting_point=3813.0,  # K (3540°C)
                operating_temp_range=(293.0, 3513.0)
            )
        )
        self._uhtc_materials['zrc'] = zrc_material
        
        # Hafnium Nitride (HfN)
        hfn_material = MaterialDefinition(
            name="Hafnium Nitride (HfN)",
            base_material_type=MaterialType.ULTRA_HIGH_TEMP_CERAMIC,
            thermal_properties=ThermalProperties(
                thermal_conductivity=25.0,  # W/(m⋅K)
                specific_heat=190.0,  # J/(kg⋅K)
                density=13800.0,  # kg/m³
                melting_point=3583.0,  # K (3310°C)
                operating_temp_range=(293.0, 3283.0)
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
                operating_temp_range=(293.0, 3159.0)
            )
        )
        self._uhtc_materials['re'] = re_material
        
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
    
    def _calculate_thermal_conductivity_vs_temperature(self, k_ref: float, 
                                                     temperatures: np.ndarray,
                                                     material_id: str) -> np.ndarray:
        """Calculate temperature-dependent thermal conductivity."""
        # Simplified model: k(T) = k_ref * (T_ref/T)^n
        # where n varies by material type
        
        T_ref = 293.0  # Reference temperature (K)
        
        if 'hfc' in material_id or 'tac' in material_id:
            n = 0.3  # Carbides
        elif 'hfn' in material_id:
            n = 0.25  # Nitrides
        elif 're' in material_id:
            n = 0.1  # Metals
        else:
            n = 0.2  # Default
        
        k_values = k_ref * (T_ref / temperatures) ** n
        
        # Ensure reasonable bounds
        k_values = np.clip(k_values, k_ref * 0.1, k_ref * 2.0)
        
        return k_values
    
    def _calculate_specific_heat_vs_temperature(self, cp_ref: float,
                                              temperatures: np.ndarray,
                                              material_id: str) -> np.ndarray:
        """Calculate temperature-dependent specific heat."""
        # Simplified model: cp(T) = cp_ref * (1 + a*T + b*T^2)
        
        T_ref = 293.0
        T_norm = temperatures / T_ref
        
        if 'hfc' in material_id or 'tac' in material_id:
            a, b = 0.0002, -1e-8  # Carbides
        elif 'hfn' in material_id:
            a, b = 0.0003, -1.5e-8  # Nitrides
        elif 're' in material_id:
            a, b = 0.0001, -5e-9  # Metals
        else:
            a, b = 0.0002, -1e-8  # Default
        
        cp_values = cp_ref * (1 + a * (temperatures - T_ref) + b * (temperatures - T_ref)**2)
        
        # Ensure reasonable bounds
        cp_values = np.clip(cp_values, cp_ref * 0.5, cp_ref * 3.0)
        
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