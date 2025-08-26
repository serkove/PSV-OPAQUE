"""Electromagnetic effects modeling for plasma flows in hypersonic flight.

This module provides calculations for electromagnetic body forces, magnetic field
interactions, and plasma conductivity for Mach 60 hypersonic flight analysis.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import math

from .plasma_physics import PlasmaConditions, ELEMENTARY_CHARGE, ELECTRON_MASS, VACUUM_PERMITTIVITY, BOLTZMANN_CONSTANT


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


class ElectromagneticEffectsCalculator:
    """Calculator for electromagnetic effects in plasma flows."""
    
    def __init__(self):
        """Initialize electromagnetic effects calculator."""
        # Collision frequencies for different species (approximate values)
        self.collision_frequencies = {
            'electron_neutral': 1e12,  # Hz
            'electron_ion': 1e11,      # Hz
            'ion_neutral': 1e10        # Hz
        }
    
    def calculate_plasma_conductivity(self, plasma_conditions: PlasmaConditions,
                                    collision_frequency: Optional[float] = None) -> float:
        """Calculate plasma electrical conductivity.
        
        Args:
            plasma_conditions: Plasma conditions
            collision_frequency: Optional collision frequency (Hz)
            
        Returns:
            Plasma conductivity in S/m
        """
        if collision_frequency is None:
            # Use temperature-dependent collision frequency
            collision_frequency = self._calculate_collision_frequency(
                plasma_conditions.electron_temperature,
                plasma_conditions.electron_density
            )
        
        # Spitzer conductivity for fully ionized plasma
        # σ = n_e * e^2 / (m_e * ν_collision)
        conductivity = (plasma_conditions.electron_density * ELEMENTARY_CHARGE**2 / 
                       (ELECTRON_MASS * collision_frequency))
        
        return conductivity
    
    def _calculate_collision_frequency(self, electron_temperature: float,
                                     electron_density: float) -> float:
        """Calculate electron-ion collision frequency.
        
        Args:
            electron_temperature: Electron temperature in K
            electron_density: Electron density in m⁻³
            
        Returns:
            Collision frequency in Hz
        """
        # Coulomb logarithm (approximate)
        coulomb_log = 15.0  # Typical value for atmospheric plasmas
        
        # Electron-ion collision frequency
        # ν_ei = (4/3) * sqrt(2π) * n_e * e^4 * ln(Λ) / (4πε_0)^2 * (k_B * T_e)^(3/2) * m_e^(1/2)
        
        thermal_velocity = np.sqrt(BOLTZMANN_CONSTANT * electron_temperature / ELECTRON_MASS)
        
        collision_frequency = (4.0/3.0 * np.sqrt(2*np.pi) * electron_density * 
                             ELEMENTARY_CHARGE**4 * coulomb_log / 
                             ((4*np.pi*VACUUM_PERMITTIVITY)**2 * 
                              (BOLTZMANN_CONSTANT * electron_temperature)**(3/2) * 
                              np.sqrt(ELECTRON_MASS)))
        
        return collision_frequency
    
    def calculate_hall_parameter(self, plasma_conditions: PlasmaConditions,
                               magnetic_field_strength: float,
                               collision_frequency: Optional[float] = None) -> float:
        """Calculate Hall parameter (ω_c * τ).
        
        Args:
            plasma_conditions: Plasma conditions
            magnetic_field_strength: Magnetic field strength in Tesla
            collision_frequency: Optional collision frequency (Hz)
            
        Returns:
            Hall parameter (dimensionless)
        """
        if collision_frequency is None:
            collision_frequency = self._calculate_collision_frequency(
                plasma_conditions.electron_temperature,
                plasma_conditions.electron_density
            )
        
        # Cyclotron frequency for electrons
        cyclotron_frequency = (ELEMENTARY_CHARGE * magnetic_field_strength / ELECTRON_MASS)
        
        # Hall parameter = ω_c / ν_collision
        hall_parameter = cyclotron_frequency / collision_frequency
        
        return hall_parameter
    
    def calculate_magnetic_reynolds_number(self, conductivity: float,
                                         characteristic_velocity: float,
                                         characteristic_length: float,
                                         magnetic_permeability: float = 4*np.pi*1e-7) -> float:
        """Calculate magnetic Reynolds number.
        
        Args:
            conductivity: Plasma conductivity in S/m
            characteristic_velocity: Characteristic flow velocity in m/s
            characteristic_length: Characteristic length scale in m
            magnetic_permeability: Magnetic permeability in H/m
            
        Returns:
            Magnetic Reynolds number (dimensionless)
        """
        # R_m = σ * μ * U * L
        magnetic_reynolds = (conductivity * magnetic_permeability * 
                           characteristic_velocity * characteristic_length)
        
        return magnetic_reynolds
    
    def calculate_lorentz_force_density(self, current_density: np.ndarray,
                                      magnetic_field: np.ndarray) -> np.ndarray:
        """Calculate Lorentz force density (J × B).
        
        Args:
            current_density: Current density vector in A/m²
            magnetic_field: Magnetic field vector in Tesla
            
        Returns:
            Lorentz force density in N/m³
        """
        # F = J × B
        lorentz_force = np.cross(current_density, magnetic_field)
        
        return lorentz_force
    
    def calculate_current_density(self, conductivity: float,
                                electric_field: np.ndarray,
                                velocity: np.ndarray,
                                magnetic_field: np.ndarray,
                                hall_parameter: float) -> np.ndarray:
        """Calculate current density using generalized Ohm's law.
        
        Args:
            conductivity: Plasma conductivity in S/m
            electric_field: Electric field vector in V/m
            velocity: Flow velocity vector in m/s
            magnetic_field: Magnetic field vector in Tesla
            hall_parameter: Hall parameter (dimensionless)
            
        Returns:
            Current density vector in A/m²
        """
        # Generalized Ohm's law: J = σ * (E + v × B) / (1 + β²)
        # where β is the Hall parameter
        
        # Calculate v × B
        velocity_cross_B = np.cross(velocity, magnetic_field)
        
        # Total electric field including motional EMF
        total_E_field = electric_field + velocity_cross_B
        
        # Current density without Hall effect
        j_parallel = conductivity * total_E_field
        
        # Hall current (perpendicular component)
        if hall_parameter != 0:
            b_unit = magnetic_field / np.linalg.norm(magnetic_field) if np.linalg.norm(magnetic_field) > 0 else np.zeros(3)
            j_hall = conductivity * hall_parameter * np.cross(total_E_field, b_unit)
            
            # Total current density
            current_density = (j_parallel + j_hall) / (1 + hall_parameter**2)
        else:
            current_density = j_parallel
        
        return current_density
    
    def calculate_induced_electric_field(self, velocity: np.ndarray,
                                       magnetic_field: np.ndarray) -> np.ndarray:
        """Calculate induced electric field from moving conductor.
        
        Args:
            velocity: Flow velocity vector in m/s
            magnetic_field: Magnetic field vector in Tesla
            
        Returns:
            Induced electric field in V/m
        """
        # E_induced = -v × B
        induced_E_field = -np.cross(velocity, magnetic_field)
        
        return induced_E_field
    
    def calculate_joule_heating_rate(self, current_density: np.ndarray,
                                   electric_field: np.ndarray) -> float:
        """Calculate Joule heating rate per unit volume.
        
        Args:
            current_density: Current density vector in A/m²
            electric_field: Electric field vector in V/m
            
        Returns:
            Joule heating rate in W/m³
        """
        # Q_joule = J · E
        joule_heating = np.dot(current_density, electric_field)
        
        return joule_heating
    
    def calculate_electromagnetic_body_force(self, plasma_conditions: PlasmaConditions,
                                           velocity: np.ndarray,
                                           magnetic_field: np.ndarray,
                                           electric_field: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate electromagnetic body force on plasma flow.
        
        Args:
            plasma_conditions: Plasma conditions
            velocity: Flow velocity vector in m/s
            magnetic_field: Magnetic field vector in Tesla
            electric_field: Optional external electric field in V/m
            
        Returns:
            Electromagnetic body force per unit volume in N/m³
        """
        if electric_field is None:
            electric_field = np.zeros(3)
        
        # Calculate plasma conductivity
        conductivity = self.calculate_plasma_conductivity(plasma_conditions)
        
        # Calculate Hall parameter
        B_magnitude = np.linalg.norm(magnetic_field)
        hall_parameter = self.calculate_hall_parameter(plasma_conditions, B_magnitude)
        
        # Calculate current density
        current_density = self.calculate_current_density(
            conductivity, electric_field, velocity, magnetic_field, hall_parameter
        )
        
        # Calculate Lorentz force
        lorentz_force = self.calculate_lorentz_force_density(current_density, magnetic_field)
        
        return lorentz_force
    
    def calculate_complete_electromagnetic_properties(self, 
                                                    plasma_conditions: PlasmaConditions,
                                                    velocity: np.ndarray,
                                                    magnetic_field: np.ndarray,
                                                    characteristic_length: float,
                                                    external_electric_field: Optional[np.ndarray] = None) -> ElectromagneticProperties:
        """Calculate complete electromagnetic properties for plasma flow.
        
        Args:
            plasma_conditions: Plasma conditions
            velocity: Flow velocity vector in m/s
            magnetic_field: Magnetic field vector in Tesla
            characteristic_length: Characteristic length scale in m
            external_electric_field: Optional external electric field in V/m
            
        Returns:
            Complete electromagnetic properties
        """
        if external_electric_field is None:
            external_electric_field = np.zeros(3)
        
        # Calculate conductivity
        conductivity = self.calculate_plasma_conductivity(plasma_conditions)
        
        # Calculate Hall parameter
        B_magnitude = np.linalg.norm(magnetic_field)
        hall_parameter = self.calculate_hall_parameter(plasma_conditions, B_magnitude)
        
        # Calculate magnetic Reynolds number
        velocity_magnitude = np.linalg.norm(velocity)
        magnetic_reynolds = self.calculate_magnetic_reynolds_number(
            conductivity, velocity_magnitude, characteristic_length
        )
        
        # Calculate induced electric field
        induced_E_field = self.calculate_induced_electric_field(velocity, magnetic_field)
        total_E_field = external_electric_field + induced_E_field
        
        # Calculate current density
        current_density = self.calculate_current_density(
            conductivity, total_E_field, velocity, magnetic_field, hall_parameter
        )
        
        # Calculate Lorentz force density
        lorentz_force = self.calculate_lorentz_force_density(current_density, magnetic_field)
        
        return ElectromagneticProperties(
            conductivity=conductivity,
            hall_parameter=hall_parameter,
            magnetic_reynolds_number=magnetic_reynolds,
            electric_field=total_E_field,
            current_density=current_density,
            lorentz_force_density=lorentz_force
        )


class MagneticFieldGenerator:
    """Generator for various magnetic field configurations."""
    
    def __init__(self):
        """Initialize magnetic field generator."""
        pass
    
    def generate_uniform_field(self, field_strength: float,
                             direction: np.ndarray) -> MagneticFieldConfiguration:
        """Generate uniform magnetic field.
        
        Args:
            field_strength: Magnetic field strength in Tesla
            direction: Unit vector for field direction
            
        Returns:
            Magnetic field configuration
        """
        # Normalize direction vector
        direction_normalized = direction / np.linalg.norm(direction)
        
        # Create uniform field
        field_vector = field_strength * direction_normalized
        
        return MagneticFieldConfiguration(
            field_strength=field_vector,
            field_gradient=np.zeros((3, 3)),  # Zero gradient for uniform field
            field_type='uniform'
        )
    
    def generate_dipole_field(self, dipole_moment: float,
                            dipole_location: np.ndarray,
                            evaluation_point: np.ndarray) -> MagneticFieldConfiguration:
        """Generate magnetic dipole field.
        
        Args:
            dipole_moment: Magnetic dipole moment in A⋅m²
            dipole_location: Location of dipole in m
            evaluation_point: Point where field is evaluated in m
            
        Returns:
            Magnetic field configuration
        """
        # Vector from dipole to evaluation point
        r_vector = evaluation_point - dipole_location
        r_magnitude = np.linalg.norm(r_vector)
        r_unit = r_vector / r_magnitude
        
        # Magnetic permeability of free space
        mu_0 = 4 * np.pi * 1e-7  # H/m
        
        # Dipole field (assuming dipole aligned with z-axis)
        dipole_direction = np.array([0, 0, 1])
        
        # B = (μ₀/4π) * (1/r³) * [3(m⋅r̂)r̂ - m]
        m_dot_r = np.dot(dipole_direction, r_unit)
        
        field_vector = (mu_0 / (4 * np.pi)) * dipole_moment / r_magnitude**3 * (
            3 * m_dot_r * r_unit - dipole_direction
        )
        
        # Calculate field gradient (simplified)
        gradient = np.zeros((3, 3))
        # This would require more complex calculation for full gradient tensor
        
        return MagneticFieldConfiguration(
            field_strength=field_vector,
            field_gradient=gradient,
            field_type='dipole',
            source_location=dipole_location
        )
    
    def calculate_field_at_points(self, config: MagneticFieldConfiguration,
                                points: np.ndarray) -> np.ndarray:
        """Calculate magnetic field at multiple points.
        
        Args:
            config: Magnetic field configuration
            points: Array of points where field is evaluated (N x 3)
            
        Returns:
            Magnetic field vectors at each point (N x 3)
        """
        num_points = points.shape[0]
        field_vectors = np.zeros((num_points, 3))
        
        if config.field_type == 'uniform':
            # Uniform field is same everywhere
            for i in range(num_points):
                field_vectors[i] = config.field_strength
        
        elif config.field_type == 'dipole' and config.source_location is not None:
            # Calculate dipole field at each point
            for i, point in enumerate(points):
                dipole_config = self.generate_dipole_field(
                    np.linalg.norm(config.field_strength),  # Use field strength as dipole moment
                    config.source_location,
                    point
                )
                field_vectors[i] = dipole_config.field_strength
        
        else:
            # Custom field - would need specific implementation
            field_vectors = np.tile(config.field_strength, (num_points, 1))
        
        return field_vectors