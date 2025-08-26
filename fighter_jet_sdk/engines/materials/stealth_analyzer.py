"""Stealth analysis and radar cross-section (RCS) calculation system."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import math
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp2d, griddata

from ...common.data_models import MaterialDefinition, AircraftConfiguration
from ...common.enums import MaterialType
from ...core.logging import get_engine_logger


@dataclass
class RCSData:
    """Radar cross-section analysis results."""
    frequencies: np.ndarray  # Hz
    angles: np.ndarray  # degrees (azimuth or elevation)
    rcs_matrix: np.ndarray  # RCS values (m²) - shape: (n_freq, n_angles)
    polarization: str  # 'VV', 'HH', 'VH', 'HV'
    incident_type: str  # 'monostatic' or 'bistatic'


@dataclass
class GeometryModel:
    """Simplified aircraft geometry model for RCS calculation."""
    fuselage_length: float  # m
    fuselage_diameter: float  # m
    wing_span: float  # m
    wing_chord: float  # m
    wing_thickness: float  # m
    tail_area: float  # m²
    engine_inlet_area: float  # m²
    surface_materials: Dict[str, str]  # surface_name -> material_id mapping


@dataclass
class StealthConfiguration:
    """Stealth configuration parameters."""
    target_rcs_reduction: float  # dB
    priority_frequencies: List[float]  # Hz - frequencies of highest importance
    priority_angles: List[float]  # degrees - angles of highest importance
    material_constraints: Dict[str, List[str]]  # surface -> allowed materials
    weight_penalty: float  # kg/m² - weight penalty for stealth materials
    cost_penalty: float  # $/m² - cost penalty for stealth materials


class StealthAnalyzer:
    """Advanced stealth analysis and RCS calculation system."""
    
    def __init__(self):
        """Initialize the stealth analyzer."""
        self.logger = get_engine_logger('materials.stealth')
        self.c0 = 299792458.0  # Speed of light (m/s)
        
        # RCS calculation methods
        self._rcs_methods = {
            'physical_optics': self._calculate_rcs_physical_optics,
            'method_of_moments': self._calculate_rcs_mom,
            'geometric_theory': self._calculate_rcs_gtd,
            'hybrid': self._calculate_rcs_hybrid
        }
        
        # Radar band definitions
        self.radar_bands = {
            'L': (1e9, 2e9),      # L-band
            'S': (2e9, 4e9),      # S-band  
            'C': (4e9, 8e9),      # C-band
            'X': (8e9, 12e9),     # X-band
            'Ku': (12e9, 18e9),   # Ku-band
            'K': (18e9, 27e9),    # K-band
            'Ka': (27e9, 40e9),   # Ka-band
        }
    
    def calculate_aircraft_rcs(self, geometry: GeometryModel, 
                              materials_db: Dict[str, MaterialDefinition],
                              frequencies: np.ndarray,
                              angles: np.ndarray,
                              polarization: str = 'VV',
                              method: str = 'hybrid') -> RCSData:
        """
        Calculate radar cross-section for aircraft configuration.
        
        Args:
            geometry: Aircraft geometry model
            materials_db: Database of materials by ID
            frequencies: Array of frequencies to analyze (Hz)
            angles: Array of angles to analyze (degrees)
            polarization: Radar polarization ('VV', 'HH', 'VH', 'HV')
            method: RCS calculation method
            
        Returns:
            RCSData object with calculated RCS values
        """
        self.logger.info(f"Calculating aircraft RCS using {method} method")
        
        if method not in self._rcs_methods:
            raise ValueError(f"Unknown RCS calculation method: {method}")
        
        # Initialize RCS matrix
        rcs_matrix = np.zeros((len(frequencies), len(angles)))
        
        # Calculate RCS for each frequency and angle combination
        for i, freq in enumerate(frequencies):
            for j, angle in enumerate(angles):
                rcs_value = self._rcs_methods[method](
                    geometry, materials_db, freq, angle, polarization
                )
                rcs_matrix[i, j] = rcs_value
        
        self.logger.info(f"RCS calculation complete. Max RCS: {np.max(rcs_matrix):.2e} m²")
        
        return RCSData(
            frequencies=frequencies,
            angles=angles,
            rcs_matrix=rcs_matrix,
            polarization=polarization,
            incident_type='monostatic'
        )
    
    def analyze_multi_frequency_rcs(self, geometry: GeometryModel,
                                   materials_db: Dict[str, MaterialDefinition],
                                   radar_bands: Optional[List[str]] = None) -> Dict[str, RCSData]:
        """
        Analyze RCS across multiple radar bands.
        
        Args:
            geometry: Aircraft geometry model
            materials_db: Database of materials
            radar_bands: List of radar band names to analyze
            
        Returns:
            Dictionary mapping band names to RCS data
        """
        if radar_bands is None:
            radar_bands = ['L', 'S', 'C', 'X', 'Ku']
        
        self.logger.info(f"Multi-frequency RCS analysis for bands: {radar_bands}")
        
        results = {}
        angles = np.linspace(-180, 180, 73)  # 5-degree resolution
        
        for band_name in radar_bands:
            if band_name not in self.radar_bands:
                self.logger.warning(f"Unknown radar band: {band_name}")
                continue
            
            f_min, f_max = self.radar_bands[band_name]
            frequencies = np.linspace(f_min, f_max, 21)  # 21 frequency points
            
            rcs_data = self.calculate_aircraft_rcs(
                geometry, materials_db, frequencies, angles
            )
            
            results[band_name] = rcs_data
        
        return results
    
    def optimize_stealth_configuration(self, geometry: GeometryModel,
                                     materials_db: Dict[str, MaterialDefinition],
                                     stealth_config: StealthConfiguration,
                                     method: str = 'differential_evolution') -> Dict[str, str]:
        """
        Optimize material selection for stealth performance.
        
        Args:
            geometry: Aircraft geometry model
            materials_db: Available materials database
            stealth_config: Stealth optimization configuration
            method: Optimization method ('differential_evolution', 'minimize')
            
        Returns:
            Dictionary mapping surface names to optimal material IDs
        """
        self.logger.info("Starting stealth configuration optimization")
        
        # Get available materials for each surface
        surface_materials = {}
        for surface, allowed_materials in stealth_config.material_constraints.items():
            available = [mat_id for mat_id in allowed_materials if mat_id in materials_db]
            if not available:
                raise ValueError(f"No available materials for surface: {surface}")
            surface_materials[surface] = available
        
        # Define optimization objective function
        def objective_function(material_indices):
            """Objective function for stealth optimization."""
            # Map indices to material IDs
            current_materials = {}
            idx = 0
            for surface, available_mats in surface_materials.items():
                mat_idx = int(material_indices[idx])
                current_materials[surface] = available_mats[mat_idx]
                idx += 1
            
            # Update geometry with current materials
            test_geometry = GeometryModel(
                fuselage_length=geometry.fuselage_length,
                fuselage_diameter=geometry.fuselage_diameter,
                wing_span=geometry.wing_span,
                wing_chord=geometry.wing_chord,
                wing_thickness=geometry.wing_thickness,
                tail_area=geometry.tail_area,
                engine_inlet_area=geometry.engine_inlet_area,
                surface_materials=current_materials
            )
            
            # Calculate RCS at priority frequencies and angles
            total_penalty = 0.0
            
            for freq in stealth_config.priority_frequencies:
                for angle in stealth_config.priority_angles:
                    rcs = self._calculate_rcs_hybrid(
                        test_geometry, materials_db, freq, angle, 'VV'
                    )
                    
                    # Convert to dBsm (dB relative to 1 m²)
                    rcs_dbsm = 10 * np.log10(max(rcs, 1e-10))
                    total_penalty += rcs_dbsm
            
            # Add weight and cost penalties
            weight_penalty = 0.0
            cost_penalty = 0.0
            
            for surface, mat_id in current_materials.items():
                material = materials_db[mat_id]
                if material.base_material_type in [MaterialType.STEALTH_COATING, MaterialType.METAMATERIAL]:
                    # Estimate surface area (simplified)
                    if 'fuselage' in surface:
                        area = np.pi * geometry.fuselage_diameter * geometry.fuselage_length
                    elif 'wing' in surface:
                        area = geometry.wing_span * geometry.wing_chord * 2  # Both wings
                    else:
                        area = geometry.tail_area
                    
                    weight_penalty += area * stealth_config.weight_penalty
                    cost_penalty += area * stealth_config.cost_penalty
            
            return total_penalty + weight_penalty * 1e-6 + cost_penalty * 1e-9
        
        # Set up optimization bounds
        bounds = []
        for surface, available_mats in surface_materials.items():
            bounds.append((0, len(available_mats) - 1))
        
        # Run optimization
        if method == 'differential_evolution':
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=100,
                popsize=15,
                seed=42
            )
        else:
            # Use scipy.optimize.minimize with integer constraints
            x0 = [0] * len(bounds)  # Start with first material for each surface
            result = minimize(
                objective_function,
                x0,
                method='L-BFGS-B',
                bounds=bounds
            )
        
        # Extract optimal material assignment
        optimal_materials = {}
        idx = 0
        for surface, available_mats in surface_materials.items():
            mat_idx = int(round(result.x[idx]))
            mat_idx = max(0, min(mat_idx, len(available_mats) - 1))  # Clamp to valid range
            optimal_materials[surface] = available_mats[mat_idx]
            idx += 1
        
        self.logger.info(f"Optimization complete. Final objective: {result.fun:.2f}")
        
        return optimal_materials
    
    def calculate_signature_management_effectiveness(self, 
                                                   baseline_rcs: RCSData,
                                                   stealth_rcs: RCSData) -> Dict[str, float]:
        """
        Calculate signature management effectiveness metrics.
        
        Args:
            baseline_rcs: Baseline RCS data (without stealth measures)
            stealth_rcs: RCS data with stealth measures applied
            
        Returns:
            Dictionary with effectiveness metrics
        """
        self.logger.info("Calculating signature management effectiveness")
        
        # Ensure compatible data
        if not np.array_equal(baseline_rcs.frequencies, stealth_rcs.frequencies):
            raise ValueError("Frequency arrays must match")
        if not np.array_equal(baseline_rcs.angles, stealth_rcs.angles):
            raise ValueError("Angle arrays must match")
        
        # Calculate RCS reduction in dB
        rcs_reduction_linear = baseline_rcs.rcs_matrix / (stealth_rcs.rcs_matrix + 1e-10)
        rcs_reduction_db = 10 * np.log10(rcs_reduction_linear)
        
        # Calculate various effectiveness metrics
        metrics = {
            'mean_rcs_reduction_db': np.mean(rcs_reduction_db),
            'max_rcs_reduction_db': np.max(rcs_reduction_db),
            'min_rcs_reduction_db': np.min(rcs_reduction_db),
            'std_rcs_reduction_db': np.std(rcs_reduction_db),
            'frontal_rcs_reduction_db': np.mean(rcs_reduction_db[:, len(baseline_rcs.angles)//2]),
            'side_rcs_reduction_db': np.mean([
                np.mean(rcs_reduction_db[:, len(baseline_rcs.angles)//4]),
                np.mean(rcs_reduction_db[:, 3*len(baseline_rcs.angles)//4])
            ]),
            'rear_rcs_reduction_db': np.mean(rcs_reduction_db[:, 0])  # 180 degrees
        }
        
        # Calculate detection range reduction
        # Detection range scales as RCS^(1/4)
        range_reduction_factor = np.power(stealth_rcs.rcs_matrix / baseline_rcs.rcs_matrix, 0.25)
        metrics['mean_detection_range_factor'] = np.mean(range_reduction_factor)
        
        return metrics
    
    def _calculate_rcs_physical_optics(self, geometry: GeometryModel,
                                     materials_db: Dict[str, MaterialDefinition],
                                     frequency: float, angle: float,
                                     polarization: str) -> float:
        """Calculate RCS using physical optics approximation."""
        wavelength = self.c0 / frequency
        k = 2 * np.pi / wavelength
        
        # Convert angle to radians
        theta = np.radians(angle)
        
        # Calculate contributions from major surfaces
        rcs_total = 0.0
        
        # Fuselage contribution (cylinder approximation)
        fuselage_material_id = geometry.surface_materials.get('fuselage', 'conventional_metal')
        if fuselage_material_id in materials_db:
            material = materials_db[fuselage_material_id]
            reflection_coeff = self._get_material_reflection_coefficient(material, frequency, theta)
        else:
            reflection_coeff = 0.9  # Default metallic reflection
        
        # Cylinder RCS (simplified with angle dependence)
        cylinder_length = geometry.fuselage_length
        cylinder_radius = geometry.fuselage_diameter / 2
        
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        if abs(cos_theta) > 0.7:  # Near nose-on incidence
            rcs_fuselage = np.pi * cylinder_radius**2 * abs(reflection_coeff)**2
        elif abs(sin_theta) > 0.7:  # Near broadside incidence
            rcs_fuselage = (2 * cylinder_radius * cylinder_length * 
                           abs(reflection_coeff)**2) / wavelength
        else:  # Intermediate angles
            rcs_fuselage = (np.pi * cylinder_radius**2 * cylinder_length * 
                           abs(reflection_coeff)**2 * abs(cos_theta)) / (2 * wavelength)
        
        rcs_total += rcs_fuselage
        
        # Wing contribution (flat plate approximation)
        wing_material_id = geometry.surface_materials.get('wing', 'conventional_metal')
        if wing_material_id in materials_db:
            material = materials_db[wing_material_id]
            reflection_coeff = self._get_material_reflection_coefficient(material, frequency, theta)
        else:
            reflection_coeff = 0.9
        
        wing_area = geometry.wing_span * geometry.wing_chord
        
        # Strong angle dependence for wing RCS
        cos_theta = np.cos(theta)
        if abs(cos_theta) < 0.1:  # Near grazing incidence
            rcs_wing = wing_area * abs(reflection_coeff)**2 * 0.1
        else:
            rcs_wing = (4 * np.pi * wing_area**2 * abs(reflection_coeff)**2 * 
                       cos_theta**4) / wavelength**2  # Stronger angle dependence
        
        rcs_total += rcs_wing
        
        # Engine inlet contribution (cavity scattering with angle dependence)
        inlet_area = geometry.engine_inlet_area
        
        # Inlet RCS depends strongly on viewing angle
        cos_theta = np.cos(theta)
        if abs(cos_theta) > 0.8:  # Near frontal aspect
            rcs_inlet = 4 * np.pi * inlet_area / wavelength**2  # Strong cavity scattering
        elif abs(cos_theta) > 0.3:  # Intermediate angles
            rcs_inlet = np.pi * inlet_area * abs(cos_theta) / wavelength**2
        else:  # Side/rear aspects - inlet not visible
            rcs_inlet = 0.1 * inlet_area / wavelength**2
        
        rcs_total += rcs_inlet
        
        return max(rcs_total, 1e-10)  # Minimum RCS floor
    
    def _calculate_rcs_mom(self, geometry: GeometryModel,
                          materials_db: Dict[str, MaterialDefinition],
                          frequency: float, angle: float,
                          polarization: str) -> float:
        """Calculate RCS using method of moments (simplified)."""
        # Simplified MoM calculation - in practice would use full electromagnetic solver
        # This is a placeholder that applies corrections to physical optics
        
        po_rcs = self._calculate_rcs_physical_optics(
            geometry, materials_db, frequency, angle, polarization
        )
        
        # Apply MoM corrections for edge effects and multiple scattering
        wavelength = self.c0 / frequency
        
        # Edge diffraction correction
        edge_length = 2 * (geometry.wing_span + geometry.fuselage_length)
        edge_correction = (edge_length / wavelength)**2 * 0.1
        
        # Multiple scattering between surfaces
        interaction_correction = po_rcs * 0.05
        
        mom_rcs = po_rcs + edge_correction + interaction_correction
        
        return max(mom_rcs, 1e-10)
    
    def _calculate_rcs_gtd(self, geometry: GeometryModel,
                          materials_db: Dict[str, MaterialDefinition],
                          frequency: float, angle: float,
                          polarization: str) -> float:
        """Calculate RCS using geometric theory of diffraction."""
        # Simplified GTD calculation
        wavelength = self.c0 / frequency
        
        # Start with physical optics
        po_rcs = self._calculate_rcs_physical_optics(
            geometry, materials_db, frequency, angle, polarization
        )
        
        # Add diffraction contributions from edges and corners
        theta = np.radians(angle)
        
        # Wing tip diffraction
        wing_tip_rcs = (geometry.wing_span / wavelength) * np.sin(theta)**2
        
        # Fuselage nose/tail diffraction
        nose_tail_rcs = (geometry.fuselage_diameter / wavelength) * 0.5
        
        gtd_rcs = po_rcs + wing_tip_rcs + nose_tail_rcs
        
        return max(gtd_rcs, 1e-10)
    
    def _calculate_rcs_hybrid(self, geometry: GeometryModel,
                             materials_db: Dict[str, MaterialDefinition],
                             frequency: float, angle: float,
                             polarization: str) -> float:
        """Calculate RCS using hybrid method combining multiple approaches."""
        wavelength = self.c0 / frequency
        
        # Use different methods based on frequency and geometry size
        electrical_size = geometry.fuselage_length / wavelength
        
        if electrical_size < 1.0:
            # Low frequency - use Rayleigh scattering
            rcs = self._calculate_rcs_rayleigh(geometry, materials_db, frequency, angle)
        elif electrical_size < 10.0:
            # Medium frequency - use method of moments
            rcs = self._calculate_rcs_mom(geometry, materials_db, frequency, angle, polarization)
        else:
            # High frequency - use physical optics with GTD corrections
            po_rcs = self._calculate_rcs_physical_optics(
                geometry, materials_db, frequency, angle, polarization
            )
            gtd_correction = self._calculate_gtd_correction(geometry, frequency, angle)
            rcs = po_rcs + gtd_correction
        
        return max(rcs, 1e-10)
    
    def _calculate_rcs_rayleigh(self, geometry: GeometryModel,
                               materials_db: Dict[str, MaterialDefinition],
                               frequency: float, angle: float) -> float:
        """Calculate RCS using Rayleigh scattering for small objects."""
        wavelength = self.c0 / frequency
        k = 2 * np.pi / wavelength
        
        # Approximate aircraft as ellipsoid
        a = geometry.fuselage_length / 2  # Semi-major axis
        b = geometry.wing_span / 2        # Semi-minor axis
        c = geometry.fuselage_diameter / 2 # Semi-minor axis
        
        # Rayleigh scattering cross-section
        volume = (4/3) * np.pi * a * b * c
        
        # Simplified polarizability (metallic object)
        alpha = 3 * volume / (4 * np.pi)
        
        rcs = (k**4 / (6 * np.pi)) * abs(alpha)**2
        
        return max(rcs, 1e-10)
    
    def _calculate_gtd_correction(self, geometry: GeometryModel,
                                 frequency: float, angle: float) -> float:
        """Calculate GTD correction terms."""
        wavelength = self.c0 / frequency
        theta = np.radians(angle)
        
        # Edge diffraction contributions
        edge_contributions = 0.0
        
        # Wing leading/trailing edges
        wing_edges = 4 * geometry.wing_chord  # 4 edges per wing
        edge_contributions += (wing_edges / wavelength) * abs(np.sin(theta)) * 0.1
        
        # Fuselage discontinuities
        fuselage_edges = 2 * np.pi * geometry.fuselage_diameter  # Circumferential edges
        edge_contributions += (fuselage_edges / wavelength) * 0.05
        
        return edge_contributions
    
    def _get_material_reflection_coefficient(self, material: MaterialDefinition,
                                           frequency: float, incident_angle: float) -> complex:
        """Get material reflection coefficient at given frequency and angle."""
        if not material.electromagnetic_properties:
            return complex(0.9, 0.0)  # Default metallic reflection
        
        em_props = material.electromagnetic_properties
        
        # Check if frequency is in valid range
        f_min, f_max = em_props.frequency_range
        if not (f_min <= frequency <= f_max):
            return complex(0.9, 0.0)  # Default outside valid range
        
        # Calculate reflection coefficient using Fresnel equations
        eps_r = em_props.permittivity
        mu_r = em_props.permeability
        
        # Wave impedance
        eta = np.sqrt(mu_r / eps_r) * 377.0  # 377 ohms = free space impedance
        eta0 = 377.0
        
        # Reflection coefficient for normal incidence (simplified)
        theta = np.radians(incident_angle)
        cos_theta = np.cos(theta)
        
        # For oblique incidence (simplified - assumes TE polarization)
        r = (eta0 * cos_theta - eta) / (eta0 * cos_theta + eta)
        
        return r