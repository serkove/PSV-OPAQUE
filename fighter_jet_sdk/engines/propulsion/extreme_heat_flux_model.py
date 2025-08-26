"""Extreme Heat Flux Modeling for Mach 60 Hypersonic Flight."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math
import numpy as np
from enum import Enum

from ...core.logging import get_engine_logger
from ...common.plasma_physics import PlasmaConditions


class RadiationModel(Enum):
    """Radiation heat transfer models."""
    GRAY_BODY = "gray_body"
    SPECTRAL = "spectral"
    PLASMA_EMISSION = "plasma_emission"
    NON_EQUILIBRIUM = "non_equilibrium"


class ConductionModel(Enum):
    """Conduction heat transfer models."""
    FOURIER = "fourier"
    HYPERBOLIC = "hyperbolic"
    PHONON_TRANSPORT = "phonon_transport"


@dataclass
class ThermalBoundaryCondition:
    """Thermal boundary condition specification."""
    surface_id: str
    temperature: Optional[float] = None  # K (Dirichlet BC)
    heat_flux: Optional[float] = None  # W/m² (Neumann BC)
    convection_coefficient: Optional[float] = None  # W/(m²⋅K)
    ambient_temperature: Optional[float] = None  # K
    emissivity: float = 0.8
    absorptivity: float = 0.8
    surface_roughness: float = 1e-6  # m


@dataclass
class MaterialThermalProperties:
    """Temperature-dependent material thermal properties."""
    material_id: str
    density: float  # kg/m³
    specific_heat_coeffs: List[float]  # Polynomial coefficients for Cp(T)
    thermal_conductivity_coeffs: List[float]  # Polynomial coefficients for k(T)
    melting_point: float  # K
    vaporization_point: float  # K
    max_operating_temperature: float  # K
    thermal_expansion_coeff: float = 1e-5  # 1/K
    
    def specific_heat(self, temperature: float) -> float:
        """Calculate specific heat at given temperature."""
        return sum(coeff * temperature**i for i, coeff in enumerate(self.specific_heat_coeffs))
    
    def thermal_conductivity(self, temperature: float) -> float:
        """Calculate thermal conductivity at given temperature."""
        return sum(coeff * temperature**i for i, coeff in enumerate(self.thermal_conductivity_coeffs))


@dataclass
class ExtremeHeatFluxConditions:
    """Extreme heat flux operating conditions."""
    heat_flux: float  # W/m²
    plasma_conditions: Optional[PlasmaConditions] = None
    surface_temperature: float = 300.0  # K
    pressure: float = 101325.0  # Pa
    velocity: float = 0.0  # m/s
    magnetic_field: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Tesla
    time: float = 0.0  # s


@dataclass
class ThermalStressResults:
    """Thermal stress analysis results."""
    stress_tensor: np.ndarray  # Pa (3x3 tensor)
    principal_stresses: np.ndarray  # Pa (3 components)
    von_mises_stress: float  # Pa
    max_shear_stress: float  # Pa
    thermal_strain: np.ndarray  # Dimensionless (3x3 tensor)
    temperature_gradient: np.ndarray  # K/m (3 components)
    safety_factor: float


class ExtremeHeatFluxModel:
    """Advanced heat flux modeling for extreme hypersonic conditions (>100 MW/m²)."""
    
    def __init__(self):
        """Initialize extreme heat flux model."""
        self.logger = get_engine_logger('propulsion.extreme_heat_flux')
        
        # Physical constants
        self.stefan_boltzmann = 5.67e-8  # W/(m²⋅K⁴)
        self.planck_constant = 6.626e-34  # J⋅s
        self.boltzmann_constant = 1.381e-23  # J/K
        self.speed_of_light = 2.998e8  # m/s
        
        # Material database
        self.materials: Dict[str, MaterialThermalProperties] = {}
        self._initialize_extreme_materials()
        
        # Numerical parameters
        self.convergence_tolerance = 1e-6
        self.max_iterations = 1000
        
    def _initialize_extreme_materials(self) -> None:
        """Initialize database of extreme temperature materials."""
        # Ultra-High Temperature Ceramics (UHTCs)
        self.materials["tungsten_carbide"] = MaterialThermalProperties(
            material_id="tungsten_carbide",
            density=15600.0,  # kg/m³
            specific_heat_coeffs=[200.0, 0.05, -1e-5],  # J/(kg⋅K)
            thermal_conductivity_coeffs=[120.0, -0.02, 1e-6],  # W/(m⋅K)
            melting_point=3058.0,  # K
            vaporization_point=6273.0,  # K
            max_operating_temperature=3500.0,  # K
            thermal_expansion_coeff=5.2e-6
        )
        
        self.materials["hafnium_carbide"] = MaterialThermalProperties(
            material_id="hafnium_carbide",
            density=12200.0,
            specific_heat_coeffs=[180.0, 0.08, -2e-5],
            thermal_conductivity_coeffs=[22.0, 0.01, -5e-7],
            melting_point=4201.0,  # Highest known melting point
            vaporization_point=7000.0,
            max_operating_temperature=4500.0,
            thermal_expansion_coeff=6.8e-6
        )
        
        self.materials["carbon_carbon"] = MaterialThermalProperties(
            material_id="carbon_carbon",
            density=1800.0,
            specific_heat_coeffs=[700.0, 0.5, -1e-4],
            thermal_conductivity_coeffs=[80.0, 0.1, -2e-5],
            melting_point=3773.0,  # Sublimation point
            vaporization_point=3773.0,
            max_operating_temperature=3000.0,
            thermal_expansion_coeff=1e-6
        )
        
        self.materials["rhenium"] = MaterialThermalProperties(
            material_id="rhenium",
            density=21020.0,
            specific_heat_coeffs=[137.0, 0.02, -3e-6],
            thermal_conductivity_coeffs=[48.0, -0.01, 2e-6],
            melting_point=3459.0,
            vaporization_point=5869.0,
            max_operating_temperature=3200.0,
            thermal_expansion_coeff=6.2e-6
        )
    
    def calculate_extreme_heat_flux(self, conditions: ExtremeHeatFluxConditions,
                                  material_id: str,
                                  geometry: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate heat flux for extreme hypersonic conditions."""
        self.logger.info(f"Calculating extreme heat flux: {conditions.heat_flux/1e6:.1f} MW/m²")
        
        if conditions.heat_flux < 100e6:  # 100 MW/m²
            self.logger.warning("Heat flux below extreme threshold (100 MW/m²)")
        
        material = self.materials.get(material_id)
        if not material:
            raise ValueError(f"Unknown material: {material_id}")
        
        results = {}
        
        # Radiative heat transfer
        radiative_results = self._calculate_radiative_heat_transfer(
            conditions, material, geometry
        )
        results['radiative'] = radiative_results
        
        # Conductive heat transfer
        conductive_results = self._calculate_conductive_heat_transfer(
            conditions, material, geometry
        )
        results['conductive'] = conductive_results
        
        # Plasma-surface interactions
        if conditions.plasma_conditions:
            plasma_results = self._calculate_plasma_heat_transfer(
                conditions, material, geometry
            )
            results['plasma'] = plasma_results
        
        # Total heat flux distribution
        results['total_heat_flux'] = self._combine_heat_transfer_modes(results)
        
        # Temperature distribution
        results['temperature_field'] = self._solve_temperature_field(
            conditions, material, geometry, results['total_heat_flux']
        )
        
        return results
    
    def _calculate_radiative_heat_transfer(self, conditions: ExtremeHeatFluxConditions,
                                         material: MaterialThermalProperties,
                                         geometry: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate radiative heat transfer for extreme temperatures."""
        surface_temp = conditions.surface_temperature
        
        # Gray body radiation
        emissivity = 0.8  # Typical for high-temperature materials
        gray_body_flux = emissivity * self.stefan_boltzmann * surface_temp**4
        
        # Spectral radiation for plasma environments
        spectral_flux = 0.0
        if conditions.plasma_conditions:
            # Plasma emission contribution
            plasma_temp = conditions.plasma_conditions.electron_temperature
            plasma_density = conditions.plasma_conditions.electron_density
            
            # Bremsstrahlung radiation
            bremsstrahlung_flux = self._calculate_bremsstrahlung_radiation(
                plasma_temp, plasma_density
            )
            
            # Line radiation
            line_radiation_flux = self._calculate_line_radiation(
                plasma_temp, plasma_density
            )
            
            spectral_flux = bremsstrahlung_flux + line_radiation_flux
        
        # Non-equilibrium radiation effects
        non_eq_factor = 1.0
        if surface_temp > 5000.0:  # K
            # Non-equilibrium effects become significant
            non_eq_factor = 1.2 + 0.1 * (surface_temp - 5000.0) / 1000.0
        
        total_radiative_flux = (gray_body_flux + spectral_flux) * non_eq_factor
        
        return {
            'gray_body_flux': gray_body_flux,
            'spectral_flux': spectral_flux,
            'non_equilibrium_factor': non_eq_factor,
            'total_flux': total_radiative_flux,
            'surface_temperature': surface_temp
        }
    
    def _calculate_bremsstrahlung_radiation(self, electron_temp: float,
                                          electron_density: float) -> float:
        """Calculate bremsstrahlung radiation from plasma."""
        # Simplified bremsstrahlung formula
        # P = 1.69e-32 * Z² * ne * ni * Te^0.5 * g_ff  [W/m³]
        # Assuming Z=1 (hydrogen), ni ≈ ne, g_ff ≈ 1
        
        power_density = 1.69e-32 * electron_density**2 * math.sqrt(electron_temp)
        
        # Convert to surface flux (assuming 1m path length)
        return power_density
    
    def _calculate_line_radiation(self, electron_temp: float,
                                electron_density: float) -> float:
        """Calculate line radiation from plasma."""
        # Simplified line radiation model
        # Depends on species present and excitation/ionization states
        
        if electron_temp < 5000.0:
            return 0.0
        
        # Approximate line radiation for air plasma
        line_power = 1e-30 * electron_density * electron_temp * math.exp(-50000.0 / electron_temp)
        
        return line_power
    
    def _calculate_conductive_heat_transfer(self, conditions: ExtremeHeatFluxConditions,
                                          material: MaterialThermalProperties,
                                          geometry: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate conductive heat transfer with extreme gradients."""
        # Temperature-dependent thermal conductivity
        k_thermal = material.thermal_conductivity(conditions.surface_temperature)
        
        # Estimate temperature gradient
        thickness = geometry.get('thickness', 0.01)  # m
        temp_gradient = conditions.heat_flux / k_thermal
        
        # Check for hyperbolic heat conduction effects
        # Becomes important when heat flux changes rapidly
        relaxation_time = 1e-12  # s (typical for metals)
        fourier_number = k_thermal * conditions.time / (material.density * 
                        material.specific_heat(conditions.surface_temperature) * thickness**2)
        
        hyperbolic_factor = 1.0
        if fourier_number < 0.1:
            # Hyperbolic effects significant
            hyperbolic_factor = 1.0 + relaxation_time * temp_gradient / thickness
        
        effective_conductivity = k_thermal * hyperbolic_factor
        
        return {
            'thermal_conductivity': k_thermal,
            'temperature_gradient': temp_gradient,
            'hyperbolic_factor': hyperbolic_factor,
            'effective_conductivity': effective_conductivity,
            'fourier_number': fourier_number
        }
    
    def _calculate_plasma_heat_transfer(self, conditions: ExtremeHeatFluxConditions,
                                      material: MaterialThermalProperties,
                                      geometry: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate plasma-surface heat transfer interactions."""
        plasma = conditions.plasma_conditions
        
        # Plasma sheath heat transfer
        sheath_potential = 3.0 * self.boltzmann_constant * plasma.electron_temperature / 1.602e-19  # V
        # Estimate ion density from electron density and ionization fraction
        ion_density = plasma.electron_density * plasma.ionization_fraction
        ion_flux = 0.25 * ion_density * math.sqrt(
            8 * self.boltzmann_constant * plasma.ion_temperature / (math.pi * 1.67e-27)  # Proton mass
        )
        
        # Energy flux from ions
        ion_energy_flux = ion_flux * sheath_potential * 1.602e-19  # W/m²
        
        # Electron heat flux
        electron_heat_flux = 2.0 * self.boltzmann_constant * plasma.electron_temperature * \
                           0.25 * plasma.electron_density * math.sqrt(
                               8 * self.boltzmann_constant * plasma.electron_temperature / 
                               (math.pi * 9.109e-31)  # Electron mass
                           )
        
        # Recombination heat release
        recombination_flux = 0.0
        if plasma.electron_temperature > 10000.0:  # K
            # Significant recombination
            recombination_rate = 1e-12 * plasma.electron_density * ion_density / plasma.electron_temperature**0.5
            ionization_energy = 13.6 * 1.602e-19  # J (hydrogen)
            recombination_flux = recombination_rate * ionization_energy
        
        total_plasma_flux = ion_energy_flux + electron_heat_flux + recombination_flux
        
        return {
            'ion_energy_flux': ion_energy_flux,
            'electron_heat_flux': electron_heat_flux,
            'recombination_flux': recombination_flux,
            'total_plasma_flux': total_plasma_flux,
            'sheath_potential': sheath_potential
        }
    
    def _combine_heat_transfer_modes(self, heat_transfer_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine different heat transfer modes."""
        total_flux = 0.0
        
        # Radiative contribution
        if 'radiative' in heat_transfer_results:
            total_flux += heat_transfer_results['radiative']['total_flux']
        
        # Plasma contribution
        if 'plasma' in heat_transfer_results:
            total_flux += heat_transfer_results['plasma']['total_plasma_flux']
        
        # Conductive flux is internal - doesn't add to surface flux
        
        return {
            'total_surface_flux': total_flux,
            'radiative_fraction': heat_transfer_results.get('radiative', {}).get('total_flux', 0) / max(total_flux, 1e-10),
            'plasma_fraction': heat_transfer_results.get('plasma', {}).get('total_plasma_flux', 0) / max(total_flux, 1e-10)
        }
    
    def _solve_temperature_field(self, conditions: ExtremeHeatFluxConditions,
                               material: MaterialThermalProperties,
                               geometry: Dict[str, Any],
                               heat_flux_data: Dict[str, Any]) -> Dict[str, Any]:
        """Solve temperature field through material thickness."""
        thickness = geometry.get('thickness', 0.01)  # m
        num_nodes = geometry.get('num_nodes', 50)
        
        # Create 1D mesh
        dx = thickness / (num_nodes - 1)
        x = np.linspace(0, thickness, num_nodes)
        
        # Initialize temperature field
        T = np.full(num_nodes, conditions.surface_temperature)
        
        # Boundary conditions
        T[0] = conditions.surface_temperature  # Surface temperature
        # Back face: adiabatic or specified temperature
        back_temp = geometry.get('back_temperature', conditions.surface_temperature - 1000.0)
        T[-1] = back_temp
        
        # Iterative solution for nonlinear thermal conductivity
        for iteration in range(self.max_iterations):
            T_old = T.copy()
            
            # Interior nodes
            for i in range(1, num_nodes - 1):
                # Temperature-dependent properties
                k_left = material.thermal_conductivity((T[i-1] + T[i]) / 2)
                k_right = material.thermal_conductivity((T[i] + T[i+1]) / 2)
                
                # Finite difference equation
                T[i] = (k_left * T[i-1] + k_right * T[i+1]) / (k_left + k_right)
            
            # Check convergence
            max_change = np.max(np.abs(T - T_old))
            if max_change < self.convergence_tolerance:
                break
        
        # Calculate heat flux distribution
        heat_flux = np.zeros(num_nodes - 1)
        for i in range(num_nodes - 1):
            k_avg = material.thermal_conductivity((T[i] + T[i+1]) / 2)
            heat_flux[i] = -k_avg * (T[i+1] - T[i]) / dx
        
        return {
            'position': x,
            'temperature': T,
            'heat_flux': heat_flux,
            'max_temperature': np.max(T),
            'min_temperature': np.min(T),
            'temperature_gradient': np.gradient(T, dx),
            'iterations': iteration + 1
        }
    
    def calculate_thermal_stress(self, temperature_field: Dict[str, Any],
                               material_id: str,
                               geometry: Dict[str, Any]) -> ThermalStressResults:
        """Calculate thermal stress from extreme temperature gradients."""
        material = self.materials[material_id]
        
        # Temperature gradient
        temp_gradient = temperature_field['temperature_gradient']
        max_gradient = np.max(np.abs(temp_gradient))
        
        # Material properties at average temperature
        avg_temp = np.mean(temperature_field['temperature'])
        youngs_modulus = geometry.get('youngs_modulus', 200e9)  # Pa
        poisson_ratio = geometry.get('poisson_ratio', 0.3)
        
        # Thermal strain
        thermal_strain_magnitude = material.thermal_expansion_coeff * \
                                 (temperature_field['max_temperature'] - 300.0)  # Reference temp
        
        # Thermal stress (simplified 1D analysis)
        # σ = E * α * ΔT / (1 - ν) for constrained thermal expansion
        thermal_stress = youngs_modulus * thermal_strain_magnitude / (1 - poisson_ratio)
        
        # Stress concentration due to gradient
        gradient_factor = 1.0 + max_gradient / 1000.0  # Empirical factor
        effective_stress = thermal_stress * gradient_factor
        
        # Principal stresses (assuming uniaxial stress state)
        principal_stresses = np.array([effective_stress, 0.0, 0.0])
        
        # Von Mises stress
        von_mises = effective_stress  # For uniaxial stress
        
        # Safety factor
        yield_strength = geometry.get('yield_strength', 500e6)  # Pa
        safety_factor = yield_strength / max(effective_stress, 1.0)
        
        # Create stress tensor (simplified)
        stress_tensor = np.array([
            [effective_stress, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
        
        # Thermal strain tensor
        strain_tensor = np.array([
            [thermal_strain_magnitude, 0, 0],
            [0, thermal_strain_magnitude, 0],
            [0, 0, thermal_strain_magnitude]
        ])
        
        return ThermalStressResults(
            stress_tensor=stress_tensor,
            principal_stresses=principal_stresses,
            von_mises_stress=von_mises,
            max_shear_stress=effective_stress / 2,
            thermal_strain=strain_tensor,
            temperature_gradient=np.array([max_gradient, 0, 0]),
            safety_factor=safety_factor
        )
    
    def validate_extreme_conditions(self, conditions: ExtremeHeatFluxConditions,
                                  material_id: str) -> List[str]:
        """Validate extreme heat flux conditions and material compatibility."""
        warnings = []
        
        # Check heat flux magnitude
        if conditions.heat_flux < 100e6:  # 100 MW/m²
            warnings.append(f"Heat flux {conditions.heat_flux/1e6:.1f} MW/m² below extreme threshold")
        
        if conditions.heat_flux > 1e9:  # 1 GW/m²
            warnings.append(f"Heat flux {conditions.heat_flux/1e6:.1f} MW/m² exceeds physical limits")
        
        # Check material compatibility
        material = self.materials.get(material_id)
        if material:
            if conditions.surface_temperature > material.max_operating_temperature:
                warnings.append(f"Surface temperature {conditions.surface_temperature:.1f}K exceeds "
                              f"material limit {material.max_operating_temperature:.1f}K")
            
            if conditions.surface_temperature > material.melting_point:
                warnings.append(f"Surface temperature {conditions.surface_temperature:.1f}K exceeds "
                              f"melting point {material.melting_point:.1f}K - ablation expected")
        
        # Check plasma conditions
        if conditions.plasma_conditions:
            if conditions.plasma_conditions.electron_density > 1e24:  # m⁻³
                warnings.append("Electron density exceeds typical atmospheric plasma values")
            
            if conditions.plasma_conditions.electron_temperature > 50000.0:  # K
                warnings.append("Electron temperature exceeds typical shock layer values")
        
        return warnings