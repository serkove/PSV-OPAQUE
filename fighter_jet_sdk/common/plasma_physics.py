"""Plasma physics calculations for extreme hypersonic flight conditions.

This module provides fundamental plasma physics calculations needed for Mach 60
hypersonic flight analysis, including plasma properties, ionization equilibrium,
and electromagnetic effects.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import math
from scipy.optimize import fsolve
from scipy.interpolate import interp1d, interp2d

from .enums import PlasmaRegime


# Physical constants
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
ELEMENTARY_CHARGE = 1.602176634e-19  # C
ELECTRON_MASS = 9.1093837015e-31  # kg
PROTON_MASS = 1.67262192369e-27  # kg
VACUUM_PERMITTIVITY = 8.8541878128e-12  # F/m
PLANCK_CONSTANT = 6.62607015e-34  # J⋅s


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


class PlasmaPropertiesCalculator:
    """Calculator for fundamental plasma properties."""
    
    def __init__(self):
        """Initialize plasma properties calculator."""
        # Ionization potentials (eV) for common atmospheric species
        self.ionization_potentials = {
            'N2': 15.58,
            'O2': 12.07,
            'N': 14.53,
            'O': 13.62,
            'Ar': 15.76,
            'He': 24.59,
            'H2': 15.43,
            'H': 13.60
        }
        
        # Molecular masses (kg/mol)
        self.molecular_masses = {
            'N2': 0.028014,
            'O2': 0.031998,
            'N': 0.014007,
            'O': 0.015999,
            'Ar': 0.039948,
            'He': 0.004003,
            'H2': 0.002016,
            'H': 0.001008
        }
        
        # Initialize interpolation tables
        self._initialize_property_tables()
    
    def _initialize_property_tables(self):
        """Initialize interpolation tables for plasma properties."""
        # Temperature range for property tables (K)
        self.temp_range = np.logspace(3, 6, 100)  # 1000K to 1,000,000K
        
        # Pressure range for property tables (Pa)
        self.pressure_range = np.logspace(2, 7, 50)  # 100 Pa to 10 MPa
        
        # Pre-calculate property tables for common gas mixtures
        self.property_tables = {}
        
        # Standard air composition
        air_composition = {'N2': 0.78, 'O2': 0.21, 'Ar': 0.01}
        self._build_property_table('air', air_composition)
        
        # Pure nitrogen
        n2_composition = {'N2': 1.0}
        self._build_property_table('N2', n2_composition)
        
        # Pure oxygen
        o2_composition = {'O2': 1.0}
        self._build_property_table('O2', o2_composition)
    
    def _build_property_table(self, mixture_name: str, composition: Dict[str, float]):
        """Build interpolation table for a specific gas mixture."""
        temp_grid, pressure_grid = np.meshgrid(self.temp_range, self.pressure_range)
        
        # Calculate properties for each temperature-pressure combination
        electron_density_grid = np.zeros_like(temp_grid)
        ionization_fraction_grid = np.zeros_like(temp_grid)
        
        for i, temp in enumerate(self.temp_range):
            for j, pressure in enumerate(self.pressure_range):
                # Calculate total number density
                total_density = pressure / (BOLTZMANN_CONSTANT * temp)
                
                # Calculate ionization fraction using Saha equation
                alpha = self._calculate_ionization_fraction_saha(
                    composition, temp, total_density
                )
                
                # Calculate electron density
                ne = alpha * total_density
                
                electron_density_grid[j, i] = ne
                ionization_fraction_grid[j, i] = alpha
        
        # Create interpolation functions
        self.property_tables[mixture_name] = {
            'electron_density': interp2d(
                self.temp_range, self.pressure_range, electron_density_grid,
                kind='linear', bounds_error=False, fill_value=0
            ),
            'ionization_fraction': interp2d(
                self.temp_range, self.pressure_range, ionization_fraction_grid,
                kind='linear', bounds_error=False, fill_value=0
            )
        }
    
    def calculate_electron_density(self, gas_mixture: GasMixture) -> float:
        """Calculate electron density using Saha equation.
        
        Args:
            gas_mixture: Gas mixture composition and conditions
            
        Returns:
            Electron density in m⁻³
        """
        # Calculate total number density
        total_density = (gas_mixture.pressure / 
                        (BOLTZMANN_CONSTANT * gas_mixture.temperature))
        
        # Calculate ionization fraction
        alpha = self._calculate_ionization_fraction_saha(
            gas_mixture.species, gas_mixture.temperature, total_density
        )
        
        # Electron density equals ion density for singly ionized plasma
        electron_density = alpha * total_density
        
        return electron_density
    
    def _calculate_ionization_fraction_saha(self, species: Dict[str, float], 
                                          temperature: float, 
                                          total_density: float) -> float:
        """Calculate ionization fraction using Saha equation.
        
        Args:
            species: Dictionary of species and their mole fractions
            temperature: Temperature in K
            total_density: Total number density in m⁻³
            
        Returns:
            Ionization fraction (0 to 1)
        """
        # For mixture, use weighted average ionization potential
        avg_ionization_potential = 0.0
        for species_name, fraction in species.items():
            if species_name in self.ionization_potentials:
                avg_ionization_potential += (fraction * 
                                           self.ionization_potentials[species_name])
        
        # Convert eV to Joules
        ionization_energy = avg_ionization_potential * ELEMENTARY_CHARGE
        
        # Saha equation for singly ionized plasma
        # K_saha = (2 * pi * m_e * k_B * T / h^2)^(3/2) * exp(-E_i / k_B * T)
        thermal_de_broglie = (PLANCK_CONSTANT / 
                             np.sqrt(2 * np.pi * ELECTRON_MASS * 
                                   BOLTZMANN_CONSTANT * temperature))
        
        saha_constant = (1 / thermal_de_broglie**3 * 
                        np.exp(-ionization_energy / (BOLTZMANN_CONSTANT * temperature)))
        
        # Solve quadratic equation: alpha^2 / (1 - alpha) = K_saha / n_total
        # This gives: alpha^2 + K_saha/n_total * alpha - K_saha/n_total = 0
        a = 1.0
        b = saha_constant / total_density
        c = -saha_constant / total_density
        
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return 0.0
        
        # Take positive root
        alpha = (-b + np.sqrt(discriminant)) / (2*a)
        
        # Ensure alpha is between 0 and 1
        return max(0.0, min(1.0, alpha))
    
    def calculate_plasma_frequency(self, electron_density: float) -> float:
        """Calculate plasma frequency.
        
        Args:
            electron_density: Electron density in m⁻³
            
        Returns:
            Plasma frequency in Hz
        """
        # ω_p = sqrt(n_e * e^2 / (ε_0 * m_e))
        plasma_frequency = np.sqrt(
            electron_density * ELEMENTARY_CHARGE**2 / 
            (VACUUM_PERMITTIVITY * ELECTRON_MASS)
        )
        
        return plasma_frequency / (2 * np.pi)  # Convert to Hz
    
    def calculate_debye_length(self, electron_density: float, 
                              electron_temperature: float) -> float:
        """Calculate Debye length.
        
        Args:
            electron_density: Electron density in m⁻³
            electron_temperature: Electron temperature in K
            
        Returns:
            Debye length in m
        """
        # λ_D = sqrt(ε_0 * k_B * T_e / (n_e * e^2))
        debye_length = np.sqrt(
            VACUUM_PERMITTIVITY * BOLTZMANN_CONSTANT * electron_temperature / 
            (electron_density * ELEMENTARY_CHARGE**2)
        )
        
        return debye_length
    
    def calculate_cyclotron_frequency(self, magnetic_field_strength: float, 
                                    particle_mass: float, 
                                    charge: float) -> float:
        """Calculate cyclotron frequency for charged particles.
        
        Args:
            magnetic_field_strength: Magnetic field strength in Tesla
            particle_mass: Particle mass in kg
            charge: Particle charge in Coulombs
            
        Returns:
            Cyclotron frequency in Hz
        """
        # ω_c = q * B / m
        cyclotron_frequency = (charge * magnetic_field_strength / particle_mass)
        
        return cyclotron_frequency / (2 * np.pi)  # Convert to Hz
    
    def determine_plasma_regime(self, plasma_conditions: PlasmaConditions) -> PlasmaRegime:
        """Determine plasma regime based on conditions.
        
        Args:
            plasma_conditions: Plasma conditions
            
        Returns:
            Plasma regime classification
        """
        alpha = plasma_conditions.ionization_fraction
        
        # Classify based on ionization fraction
        if alpha < 0.01:
            return PlasmaRegime.WEAKLY_IONIZED
        elif alpha < 0.1:
            return PlasmaRegime.PARTIALLY_IONIZED
        elif alpha < 0.9:
            return PlasmaRegime.FULLY_IONIZED
        else:
            # Check if magnetized
            if np.linalg.norm(plasma_conditions.magnetic_field) > 0.01:  # > 0.01 T
                return PlasmaRegime.MAGNETIZED_PLASMA
            else:
                return PlasmaRegime.FULLY_IONIZED
    
    def get_plasma_properties_from_table(self, mixture_name: str, 
                                       temperature: float, 
                                       pressure: float) -> Tuple[float, float]:
        """Get plasma properties from pre-calculated interpolation tables.
        
        Args:
            mixture_name: Name of gas mixture ('air', 'N2', 'O2')
            temperature: Temperature in K
            pressure: Pressure in Pa
            
        Returns:
            Tuple of (electron_density, ionization_fraction)
        """
        if mixture_name not in self.property_tables:
            raise ValueError(f"No property table available for mixture: {mixture_name}")
        
        table = self.property_tables[mixture_name]
        
        # Clamp values to table bounds
        temp_clamped = np.clip(temperature, self.temp_range[0], self.temp_range[-1])
        pressure_clamped = np.clip(pressure, self.pressure_range[0], self.pressure_range[-1])
        
        electron_density = float(table['electron_density'](temp_clamped, pressure_clamped))
        ionization_fraction = float(table['ionization_fraction'](temp_clamped, pressure_clamped))
        
        return electron_density, ionization_fraction
    
    def calculate_complete_plasma_conditions(self, gas_mixture: GasMixture,
                                           magnetic_field: Optional[np.ndarray] = None) -> PlasmaConditions:
        """Calculate complete plasma conditions for given gas mixture.
        
        Args:
            gas_mixture: Gas mixture composition and conditions
            magnetic_field: Optional magnetic field vector in Tesla
            
        Returns:
            Complete plasma conditions
        """
        if magnetic_field is None:
            magnetic_field = np.array([0.0, 0.0, 0.0])
        
        # Calculate electron density
        electron_density = self.calculate_electron_density(gas_mixture)
        
        # Calculate plasma frequency
        plasma_frequency = self.calculate_plasma_frequency(electron_density)
        
        # Calculate Debye length (assume electron temperature equals gas temperature)
        debye_length = self.calculate_debye_length(electron_density, gas_mixture.temperature)
        
        # Calculate ionization fraction
        total_density = (gas_mixture.pressure / 
                        (BOLTZMANN_CONSTANT * gas_mixture.temperature))
        ionization_fraction = electron_density / total_density if total_density > 0 else 0.0
        
        # Create plasma conditions object
        plasma_conditions = PlasmaConditions(
            electron_density=electron_density,
            electron_temperature=gas_mixture.temperature,  # Assume thermal equilibrium
            ion_temperature=gas_mixture.temperature,
            magnetic_field=magnetic_field,
            plasma_frequency=plasma_frequency,
            debye_length=debye_length,
            ionization_fraction=ionization_fraction
        )
        
        # Determine plasma regime
        plasma_conditions.regime = self.determine_plasma_regime(plasma_conditions)
        
        return plasma_conditions


class PlasmaPropertyInterpolator:
    """Interpolator for plasma properties across different conditions."""
    
    def __init__(self):
        """Initialize plasma property interpolator."""
        self.calculator = PlasmaPropertiesCalculator()
        self.cached_tables = {}
    
    def create_property_table(self, species_composition: Dict[str, float],
                            temperature_range: Tuple[float, float],
                            pressure_range: Tuple[float, float],
                            num_temp_points: int = 50,
                            num_pressure_points: int = 50) -> Dict[str, interp2d]:
        """Create interpolation table for plasma properties.
        
        Args:
            species_composition: Dictionary of species and mole fractions
            temperature_range: (min_temp, max_temp) in K
            pressure_range: (min_pressure, max_pressure) in Pa
            num_temp_points: Number of temperature points
            num_pressure_points: Number of pressure points
            
        Returns:
            Dictionary of interpolation functions
        """
        # Create temperature and pressure grids
        temp_points = np.logspace(
            np.log10(temperature_range[0]), 
            np.log10(temperature_range[1]), 
            num_temp_points
        )
        pressure_points = np.logspace(
            np.log10(pressure_range[0]), 
            np.log10(pressure_range[1]), 
            num_pressure_points
        )
        
        temp_grid, pressure_grid = np.meshgrid(temp_points, pressure_points)
        
        # Initialize result grids
        electron_density_grid = np.zeros_like(temp_grid)
        plasma_frequency_grid = np.zeros_like(temp_grid)
        debye_length_grid = np.zeros_like(temp_grid)
        ionization_fraction_grid = np.zeros_like(temp_grid)
        
        # Calculate properties for each point
        for i, temp in enumerate(temp_points):
            for j, pressure in enumerate(pressure_points):
                gas_mixture = GasMixture(
                    species=species_composition,
                    temperature=temp,
                    pressure=pressure,
                    total_density=pressure / (BOLTZMANN_CONSTANT * temp)
                )
                
                plasma_conditions = self.calculator.calculate_complete_plasma_conditions(gas_mixture)
                
                electron_density_grid[j, i] = plasma_conditions.electron_density
                plasma_frequency_grid[j, i] = plasma_conditions.plasma_frequency
                debye_length_grid[j, i] = plasma_conditions.debye_length
                ionization_fraction_grid[j, i] = plasma_conditions.ionization_fraction
        
        # Create interpolation functions
        interpolation_table = {
            'electron_density': interp2d(
                temp_points, pressure_points, electron_density_grid,
                kind='linear', bounds_error=False, fill_value=0
            ),
            'plasma_frequency': interp2d(
                temp_points, pressure_points, plasma_frequency_grid,
                kind='linear', bounds_error=False, fill_value=0
            ),
            'debye_length': interp2d(
                temp_points, pressure_points, debye_length_grid,
                kind='linear', bounds_error=False, fill_value=1e-6
            ),
            'ionization_fraction': interp2d(
                temp_points, pressure_points, ionization_fraction_grid,
                kind='linear', bounds_error=False, fill_value=0
            )
        }
        
        return interpolation_table
    
    def get_interpolated_properties(self, table: Dict[str, interp2d],
                                  temperature: float, 
                                  pressure: float) -> Dict[str, float]:
        """Get interpolated plasma properties from table.
        
        Args:
            table: Interpolation table from create_property_table
            temperature: Temperature in K
            pressure: Pressure in Pa
            
        Returns:
            Dictionary of plasma properties
        """
        properties = {}
        
        for prop_name, interpolator in table.items():
            properties[prop_name] = float(interpolator(temperature, pressure))
        
        return properties