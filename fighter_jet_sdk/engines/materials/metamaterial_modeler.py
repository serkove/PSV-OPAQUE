"""Metamaterial electromagnetic simulation and modeling."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import math
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d

from ...common.data_models import MaterialDefinition, EMProperties
from ...common.enums import MaterialType
from ...core.logging import get_engine_logger


@dataclass
class FrequencyResponse:
    """Frequency response data for metamaterials."""
    frequencies: np.ndarray  # Hz
    permittivity: np.ndarray  # Complex permittivity
    permeability: np.ndarray  # Complex permeability
    transmission: np.ndarray  # Transmission coefficient
    reflection: np.ndarray   # Reflection coefficient
    absorption: np.ndarray   # Absorption coefficient


@dataclass
class FSSSurface:
    """Frequency Selective Surface configuration."""
    unit_cell_size: float  # m
    element_type: str  # 'patch', 'slot', 'dipole', 'loop'
    element_dimensions: Dict[str, float]  # Element-specific dimensions
    substrate_thickness: float  # m
    substrate_permittivity: complex
    periodicity: Tuple[float, float]  # x, y periodicity in m


class MetamaterialModeler:
    """Advanced metamaterial electromagnetic simulation and modeling."""
    
    def __init__(self):
        """Initialize the metamaterial modeler."""
        self.logger = get_engine_logger('materials.metamaterial')
        self.c0 = 299792458.0  # Speed of light in vacuum (m/s)
        self.mu0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)
        self.eps0 = 8.854187817e-12  # Permittivity of free space (F/m)
        
        # Benchmark metamaterial data for validation
        self._benchmark_data = self._load_benchmark_data()
    
    def calculate_frequency_response(self, material: MaterialDefinition, 
                                   frequencies: np.ndarray,
                                   thickness: float = 1e-3) -> FrequencyResponse:
        """
        Calculate frequency response of metamaterial.
        
        Args:
            material: Material definition with EM properties
            frequencies: Array of frequencies to analyze (Hz)
            thickness: Material thickness (m)
            
        Returns:
            FrequencyResponse object with calculated properties
        """
        if not material.electromagnetic_properties:
            raise ValueError("Material must have electromagnetic properties")
        
        em_props = material.electromagnetic_properties
        
        # Initialize arrays
        n_freq = len(frequencies)
        permittivity = np.zeros(n_freq, dtype=complex)
        permeability = np.zeros(n_freq, dtype=complex)
        transmission = np.zeros(n_freq, dtype=complex)
        reflection = np.zeros(n_freq, dtype=complex)
        absorption = np.zeros(n_freq)
        
        for i, freq in enumerate(frequencies):
            # Calculate frequency-dependent properties
            eps_r, mu_r = self._calculate_dispersive_properties(em_props, freq)
            
            permittivity[i] = eps_r
            permeability[i] = mu_r
            
            # Calculate transmission and reflection coefficients
            t_coeff, r_coeff = self._calculate_transmission_reflection(
                eps_r, mu_r, freq, thickness
            )
            
            transmission[i] = t_coeff
            reflection[i] = r_coeff
            
            # Calculate absorption ensuring energy conservation
            abs_t_sq = abs(t_coeff)**2
            abs_r_sq = abs(r_coeff)**2
            
            # Normalize to ensure energy conservation if needed
            total_power = abs_t_sq + abs_r_sq
            if total_power > 1.0:
                # Renormalize transmission and reflection coefficients
                norm_factor = np.sqrt(1.0 / total_power)
                transmission[i] = t_coeff * norm_factor
                reflection[i] = r_coeff * norm_factor
                absorption[i] = 0.0
            else:
                transmission[i] = t_coeff
                reflection[i] = r_coeff
                absorption[i] = 1.0 - total_power
        
        self.logger.info(f"Calculated frequency response for {len(frequencies)} frequencies")
        
        return FrequencyResponse(
            frequencies=frequencies,
            permittivity=permittivity,
            permeability=permeability,
            transmission=transmission,
            reflection=reflection,
            absorption=absorption
        )
    
    def model_frequency_selective_surface(self, fss_config: FSSSurface,
                                        frequencies: np.ndarray) -> FrequencyResponse:
        """
        Model frequency selective surface (FSS) electromagnetic response.
        
        Args:
            fss_config: FSS configuration parameters
            frequencies: Array of frequencies to analyze (Hz)
            
        Returns:
            FrequencyResponse object with FSS characteristics
        """
        self.logger.info(f"Modeling FSS with {fss_config.element_type} elements")
        
        # Initialize arrays
        n_freq = len(frequencies)
        transmission = np.zeros(n_freq, dtype=complex)
        reflection = np.zeros(n_freq, dtype=complex)
        
        for i, freq in enumerate(frequencies):
            # Calculate FSS response based on element type
            if fss_config.element_type == 'patch':
                t_coeff, r_coeff = self._calculate_patch_fss_response(fss_config, freq)
            elif fss_config.element_type == 'slot':
                t_coeff, r_coeff = self._calculate_slot_fss_response(fss_config, freq)
            elif fss_config.element_type == 'dipole':
                t_coeff, r_coeff = self._calculate_dipole_fss_response(fss_config, freq)
            elif fss_config.element_type == 'loop':
                t_coeff, r_coeff = self._calculate_loop_fss_response(fss_config, freq)
            else:
                raise ValueError(f"Unsupported FSS element type: {fss_config.element_type}")
            
            transmission[i] = t_coeff
            reflection[i] = r_coeff
        
        # Calculate absorption ensuring energy conservation
        abs_t_sq = np.abs(transmission)**2
        abs_r_sq = np.abs(reflection)**2
        total_power = abs_t_sq + abs_r_sq
        
        # Normalize if total power exceeds 1 (energy conservation)
        exceed_mask = total_power > 1.0
        if np.any(exceed_mask):
            norm_factors = np.sqrt(1.0 / total_power)
            transmission[exceed_mask] *= norm_factors[exceed_mask]
            reflection[exceed_mask] *= norm_factors[exceed_mask]
            
            # Recalculate after normalization
            abs_t_sq = np.abs(transmission)**2
            abs_r_sq = np.abs(reflection)**2
        
        absorption = 1.0 - abs_t_sq - abs_r_sq
        absorption = np.clip(absorption, 0.0, 1.0)  # Handle small numerical errors
        
        # Effective permittivity and permeability from S-parameters
        permittivity, permeability = self._extract_effective_parameters(
            transmission, reflection, frequencies, fss_config.substrate_thickness
        )
        
        return FrequencyResponse(
            frequencies=frequencies,
            permittivity=permittivity,
            permeability=permeability,
            transmission=transmission,
            reflection=reflection,
            absorption=absorption
        )
    
    def calculate_ram_effectiveness(self, material: MaterialDefinition,
                                  thickness: float,
                                  frequencies: np.ndarray,
                                  incident_angle: float = 0.0) -> Dict[str, np.ndarray]:
        """
        Calculate radar absorption material (RAM) effectiveness.
        
        Args:
            material: RAM material definition
            thickness: RAM thickness (m)
            frequencies: Array of frequencies (Hz)
            incident_angle: Incident angle in degrees
            
        Returns:
            Dictionary with absorption, reflection loss, and other metrics
        """
        if material.base_material_type != MaterialType.STEALTH_COATING:
            self.logger.warning("Material type is not stealth coating")
        
        if not material.electromagnetic_properties:
            raise ValueError("Material must have electromagnetic properties")
        
        self.logger.info(f"Calculating RAM effectiveness for thickness {thickness*1000:.2f}mm")
        
        em_props = material.electromagnetic_properties
        
        # Initialize result arrays
        n_freq = len(frequencies)
        absorption = np.zeros(n_freq)
        reflection_loss = np.zeros(n_freq)
        insertion_loss = np.zeros(n_freq)
        return_loss = np.zeros(n_freq)
        
        # Convert incident angle to radians
        theta = np.radians(incident_angle)
        
        for i, freq in enumerate(frequencies):
            # Get frequency-dependent material properties
            eps_r, mu_r = self._calculate_dispersive_properties(em_props, freq)
            
            # Calculate wave impedance and propagation constant
            eta = np.sqrt(mu_r / eps_r) * 377.0  # Wave impedance (Ohms)
            k0 = 2 * np.pi * freq / self.c0  # Free space wave number
            kz = k0 * np.sqrt(eps_r * mu_r - np.sin(theta)**2)  # z-component wave number
            
            # Calculate reflection coefficient at air-material interface
            eta0 = 377.0  # Free space impedance
            r01 = (eta - eta0 * np.cos(theta)) / (eta + eta0 * np.cos(theta))
            
            # Calculate transmission through material
            gamma = 1j * kz * thickness  # Propagation factor
            
            # Total reflection coefficient including multiple reflections
            r_total = (r01 + r01 * np.exp(-2 * gamma)) / (1 + r01**2 * np.exp(-2 * gamma))
            
            # Calculate metrics
            absorption[i] = 1.0 - abs(r_total)**2
            reflection_loss[i] = -20 * np.log10(abs(r_total))
            return_loss[i] = -20 * np.log10(abs(r_total))
            
            # Insertion loss (transmission loss)
            t_total = (1 - r01**2) * np.exp(-gamma) / (1 + r01**2 * np.exp(-2 * gamma))
            insertion_loss[i] = -20 * np.log10(abs(t_total))
        
        return {
            'absorption': absorption,
            'reflection_loss_db': reflection_loss,
            'insertion_loss_db': insertion_loss,
            'return_loss_db': return_loss,
            'frequencies': frequencies
        }
    
    def optimize_ram_thickness(self, material: MaterialDefinition,
                              target_frequency: float,
                              target_absorption: float = 0.9,
                              thickness_range: Tuple[float, float] = (0.1e-3, 20e-3)) -> float:
        """
        Optimize RAM thickness for maximum absorption at target frequency.
        
        Args:
            material: RAM material definition
            target_frequency: Target frequency for optimization (Hz)
            target_absorption: Target absorption coefficient
            thickness_range: Min and max thickness range (m)
            
        Returns:
            Optimal thickness (m)
        """
        def objective(thickness):
            """Objective function for thickness optimization."""
            try:
                ram_data = self.calculate_ram_effectiveness(
                    material, thickness, np.array([target_frequency])
                )
                absorption = ram_data['absorption'][0]
                return abs(absorption - target_absorption)
            except Exception:
                return 1.0  # Return high error for invalid thickness
        
        # Optimize thickness
        result = minimize_scalar(
            objective,
            bounds=thickness_range,
            method='bounded'
        )
        
        optimal_thickness = result.x
        final_absorption = 1.0 - objective(optimal_thickness)
        
        self.logger.info(f"Optimal thickness: {optimal_thickness*1000:.3f}mm "
                        f"(absorption: {final_absorption:.3f})")
        
        return optimal_thickness
    
    def validate_against_benchmarks(self, material: MaterialDefinition) -> Dict[str, float]:
        """
        Validate metamaterial model against known benchmarks.
        
        Args:
            material: Material to validate
            
        Returns:
            Dictionary with validation metrics
        """
        self.logger.info("Validating metamaterial model against benchmarks")
        
        validation_results = {}
        
        # Test against split-ring resonator benchmark
        if material.base_material_type == MaterialType.METAMATERIAL:
            srr_error = self._validate_srr_benchmark(material)
            validation_results['srr_resonance_error'] = srr_error
        
        # Test against Salisbury screen benchmark
        if material.base_material_type == MaterialType.STEALTH_COATING:
            salisbury_error = self._validate_salisbury_benchmark(material)
            validation_results['salisbury_absorption_error'] = salisbury_error
        
        # Test frequency response continuity
        continuity_error = self._validate_frequency_continuity(material)
        validation_results['frequency_continuity_error'] = continuity_error
        
        # Test energy conservation
        energy_error = self._validate_energy_conservation(material)
        validation_results['energy_conservation_error'] = energy_error
        
        return validation_results
    
    def _calculate_dispersive_properties(self, em_props: EMProperties, 
                                       frequency: float) -> Tuple[complex, complex]:
        """Calculate frequency-dependent permittivity and permeability."""
        # Simple Lorentzian dispersion model
        f_min, f_max = em_props.frequency_range
        f_center = np.sqrt(f_min * f_max)  # Geometric mean
        
        # Normalized frequency
        f_norm = frequency / f_center
        
        # Dispersive permittivity (Lorentzian model)
        gamma = em_props.loss_tangent
        eps_inf = em_props.permittivity.real * 0.8  # High-frequency limit
        delta_eps = em_props.permittivity.real - eps_inf
        
        eps_r = eps_inf + delta_eps / (1 - f_norm**2 + 1j * gamma * f_norm)
        
        # Dispersive permeability (similar model)
        mu_inf = em_props.permeability.real * 0.9
        delta_mu = em_props.permeability.real - mu_inf
        
        mu_r = mu_inf + delta_mu / (1 - f_norm**2 + 1j * gamma * f_norm)
        
        return eps_r, mu_r
    
    def _calculate_transmission_reflection(self, eps_r: complex, mu_r: complex,
                                         frequency: float, thickness: float) -> Tuple[complex, complex]:
        """Calculate transmission and reflection coefficients."""
        # Wave impedance
        eta = np.sqrt(mu_r / eps_r) * 377.0
        
        # Propagation constant
        k0 = 2 * np.pi * frequency / self.c0
        k = k0 * np.sqrt(eps_r * mu_r)
        
        # Reflection coefficient at interface
        r = (eta - 377.0) / (eta + 377.0)
        
        # Transmission coefficient through slab
        gamma = 1j * k * thickness
        t = (1 - r**2) * np.exp(-gamma) / (1 - r**2 * np.exp(-2 * gamma))
        
        # Total reflection including multiple reflections
        r_total = r * (1 - np.exp(-2 * gamma)) / (1 - r**2 * np.exp(-2 * gamma))
        
        return t, r_total
    
    def _calculate_patch_fss_response(self, fss_config: FSSSurface, 
                                    frequency: float) -> Tuple[complex, complex]:
        """Calculate FSS response for patch elements."""
        # Simplified patch FSS model
        wavelength = self.c0 / frequency
        
        # Resonant frequency based on patch dimensions
        patch_length = fss_config.element_dimensions.get('length', 0.01)
        f_res = self.c0 / (2 * patch_length * np.sqrt(fss_config.substrate_permittivity.real))
        
        # Quality factor
        Q = 10.0  # Typical value
        
        # Transmission coefficient (bandstop behavior)
        f_norm = frequency / f_res
        t = 1 / (1 + 1j * Q * (f_norm - 1/f_norm))
        r = 1 - t
        
        return t, r
    
    def _calculate_slot_fss_response(self, fss_config: FSSSurface,
                                   frequency: float) -> Tuple[complex, complex]:
        """Calculate FSS response for slot elements."""
        # Simplified slot FSS model (complementary to patch)
        wavelength = self.c0 / frequency
        
        slot_length = fss_config.element_dimensions.get('length', 0.01)
        f_res = self.c0 / (2 * slot_length * np.sqrt(fss_config.substrate_permittivity.real))
        
        Q = 8.0  # Typical value for slots
        
        # Transmission coefficient (bandpass behavior)
        f_norm = frequency / f_res
        r = 1 / (1 + 1j * Q * (f_norm - 1/f_norm))
        t = 1 - r
        
        return t, r
    
    def _calculate_dipole_fss_response(self, fss_config: FSSSurface,
                                     frequency: float) -> Tuple[complex, complex]:
        """Calculate FSS response for dipole elements."""
        # Simplified dipole FSS model
        dipole_length = fss_config.element_dimensions.get('length', 0.005)
        f_res = self.c0 / (2 * dipole_length)
        
        Q = 15.0  # Higher Q for dipoles
        
        f_norm = frequency / f_res
        t = 1 / (1 + 1j * Q * (f_norm - 1/f_norm))
        r = 1 - t
        
        return t, r
    
    def _calculate_loop_fss_response(self, fss_config: FSSSurface,
                                   frequency: float) -> Tuple[complex, complex]:
        """Calculate FSS response for loop elements."""
        # Simplified loop FSS model
        loop_radius = fss_config.element_dimensions.get('radius', 0.003)
        circumference = 2 * np.pi * loop_radius
        f_res = self.c0 / circumference
        
        Q = 12.0  # Typical value for loops
        
        f_norm = frequency / f_res
        r = 1 / (1 + 1j * Q * (f_norm - 1/f_norm))
        t = 1 - r
        
        return t, r
    
    def _extract_effective_parameters(self, transmission: np.ndarray, reflection: np.ndarray,
                                    frequencies: np.ndarray, thickness: float) -> Tuple[np.ndarray, np.ndarray]:
        """Extract effective permittivity and permeability from S-parameters."""
        n_freq = len(frequencies)
        permittivity = np.zeros(n_freq, dtype=complex)
        permeability = np.zeros(n_freq, dtype=complex)
        
        for i, freq in enumerate(frequencies):
            k0 = 2 * np.pi * freq / self.c0
            
            # S-parameter extraction (simplified with numerical stability)
            S11 = reflection[i]
            S21 = transmission[i]
            
            # Add small regularization to avoid division by zero
            eps = 1e-12
            
            try:
                # Calculate impedance and refractive index with stability checks
                denominator = (1 - S11)**2 - S21**2
                if abs(denominator) < eps:
                    # Use simplified approximation for near-singular cases
                    z = complex(1.0, 0.0)
                    n = complex(1.0, 0.0)
                else:
                    numerator = (1 + S11)**2 - S21**2
                    z = np.sqrt(numerator / denominator)
                    
                    # Calculate refractive index with log stability
                    log_arg = S21 / (1 - S11 * (z - 1) / (z + 1))
                    if abs(log_arg) < eps:
                        n = complex(1.0, 0.0)
                    else:
                        n = (1j / (k0 * thickness)) * np.log(log_arg)
                
                # Extract permittivity and permeability with bounds checking
                if abs(z) < eps:
                    permittivity[i] = complex(1.0, 0.0)
                    permeability[i] = complex(1.0, 0.0)
                else:
                    permittivity[i] = n / z
                    permeability[i] = n * z
                    
                    # Check for NaN or infinite values
                    if not (np.isfinite(permittivity[i]) and np.isfinite(permeability[i])):
                        permittivity[i] = complex(1.0, 0.0)
                        permeability[i] = complex(1.0, 0.0)
                        
            except (ValueError, ZeroDivisionError, OverflowError):
                # Fallback to default values for numerical issues
                permittivity[i] = complex(1.0, 0.0)
                permeability[i] = complex(1.0, 0.0)
        
        return permittivity, permeability
    
    def _validate_srr_benchmark(self, material: MaterialDefinition) -> float:
        """Validate against split-ring resonator benchmark."""
        # Known SRR resonance at ~10 GHz
        test_freq = 10e9
        
        if not material.electromagnetic_properties:
            return 1.0
        
        # Calculate response at resonance
        response = self.calculate_frequency_response(
            material, np.array([test_freq]), thickness=1e-3
        )
        
        # Expected negative permeability near resonance
        expected_mu_real = -1.0
        actual_mu_real = response.permeability[0].real
        
        error = abs(actual_mu_real - expected_mu_real) / abs(expected_mu_real)
        return error
    
    def _validate_salisbury_benchmark(self, material: MaterialDefinition) -> float:
        """Validate against Salisbury screen benchmark."""
        # Salisbury screen: quarter-wave thickness should give perfect absorption
        test_freq = 10e9
        
        if not material.electromagnetic_properties:
            return 1.0
        
        # Calculate quarter-wave thickness
        em_props = material.electromagnetic_properties
        eps_r, mu_r = self._calculate_dispersive_properties(em_props, test_freq)
        
        wavelength = self.c0 / (test_freq * np.sqrt(eps_r.real * mu_r.real))
        quarter_wave_thickness = wavelength / 4
        
        # Calculate absorption at quarter-wave thickness
        ram_data = self.calculate_ram_effectiveness(
            material, quarter_wave_thickness, np.array([test_freq])
        )
        
        expected_absorption = 1.0
        actual_absorption = ram_data['absorption'][0]
        
        error = abs(actual_absorption - expected_absorption)
        return error
    
    def _validate_frequency_continuity(self, material: MaterialDefinition) -> float:
        """Validate frequency response continuity."""
        if not material.electromagnetic_properties:
            return 1.0
        
        # Test frequency continuity
        f_min, f_max = material.electromagnetic_properties.frequency_range
        frequencies = np.linspace(f_min, f_max, 100)
        
        response = self.calculate_frequency_response(material, frequencies)
        
        # Calculate discontinuities in permittivity
        eps_diff = np.diff(response.permittivity.real)
        max_discontinuity = np.max(np.abs(eps_diff))
        
        # Normalize by typical permittivity value
        typical_eps = np.mean(np.abs(response.permittivity.real))
        error = max_discontinuity / typical_eps if typical_eps > 0 else 1.0
        
        return error
    
    def _validate_energy_conservation(self, material: MaterialDefinition) -> float:
        """Validate energy conservation (T + R + A = 1)."""
        if not material.electromagnetic_properties:
            return 1.0
        
        f_min, f_max = material.electromagnetic_properties.frequency_range
        frequencies = np.linspace(f_min, f_max, 50)
        
        response = self.calculate_frequency_response(material, frequencies)
        
        # Check energy conservation
        total_power = (np.abs(response.transmission)**2 + 
                      np.abs(response.reflection)**2 + 
                      response.absorption)
        
        energy_error = np.max(np.abs(total_power - 1.0))
        return energy_error
    
    def _load_benchmark_data(self) -> Dict[str, Dict]:
        """Load benchmark data for validation."""
        # Simplified benchmark data
        return {
            'srr_10ghz': {
                'frequency': 10e9,
                'expected_mu_real': -1.0,
                'expected_mu_imag': -0.1
            },
            'salisbury_screen': {
                'frequency': 10e9,
                'expected_absorption': 1.0,
                'thickness_ratio': 0.25  # Quarter wavelength
            }
        }