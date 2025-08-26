"""Laser-based sensor and weapon systems for advanced detection and engagement."""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from ...common.enums import SensorType
from ...core.logging import get_engine_logger


class LaserType(Enum):
    """Types of laser systems."""
    CONTINUOUS_WAVE = auto()
    PULSED = auto()
    FIBER_LASER = auto()
    SOLID_STATE = auto()
    GAS_LASER = auto()


class AtmosphericCondition(Enum):
    """Atmospheric conditions affecting laser propagation."""
    CLEAR = auto()
    HAZE = auto()
    FOG = auto()
    RAIN = auto()
    SNOW = auto()
    DUST = auto()


@dataclass
class LaserConfiguration:
    """Laser system configuration parameters."""
    wavelength: float  # m (e.g., 1.064e-6 for Nd:YAG)
    peak_power: float  # W
    pulse_energy: Optional[float] = None  # J (for pulsed lasers)
    pulse_duration: Optional[float] = None  # s (for pulsed lasers)
    repetition_rate: Optional[float] = None  # Hz (for pulsed lasers)
    beam_divergence: float = 1e-6  # rad (full angle)
    beam_quality: float = 1.2  # M² factor
    laser_type: LaserType = LaserType.SOLID_STATE


@dataclass
class AdaptiveOpticsConfiguration:
    """Adaptive optics system configuration."""
    actuator_count: int  # Number of deformable mirror actuators
    wavefront_sensor_subapertures: int  # Number of Shack-Hartmann subapertures
    correction_bandwidth: float  # Hz
    residual_wavefront_error: float  # rad RMS
    aperture_diameter: float  # m


@dataclass
class AtmosphericParameters:
    """Atmospheric parameters affecting laser propagation."""
    visibility: float  # km
    temperature: float  # K
    pressure: float  # Pa
    humidity: float  # % (0-100)
    wind_speed: float  # m/s
    turbulence_strength: float  # Cn² (m^-2/3)
    condition: AtmosphericCondition = AtmosphericCondition.CLEAR


@dataclass
class LaserTarget:
    """Target for laser engagement or sensing."""
    position: Tuple[float, float, float]  # x, y, z coordinates (m)
    velocity: Tuple[float, float, float]  # vx, vy, vz (m/s)
    reflectivity: float  # 0-1 (fraction of incident power reflected)
    surface_area: float  # m²
    material_properties: Dict[str, float] = field(default_factory=dict)
    target_id: str = ""


class LaserFilamentationSim:
    """Laser filamentation simulation for atmospheric plasma effects."""
    
    def __init__(self, laser_config: LaserConfiguration):
        """Initialize laser filamentation simulator."""
        self.config = laser_config
        self.logger = get_engine_logger('sensors.laser.filamentation')
        
        # Physical constants
        self.c = 299792458  # Speed of light (m/s)
        self.n0 = 1.0003  # Refractive index of air at STP
        self.critical_power_air = 3.2e9  # W (critical power for self-focusing in air)
        
    def calculate_critical_power(self, atmospheric_params: AtmosphericParameters) -> float:
        """
        Calculate critical power for self-focusing in atmosphere.
        
        Args:
            atmospheric_params: Atmospheric conditions
            
        Returns:
            Critical power (W)
        """
        # Adjust for atmospheric density
        density_ratio = (atmospheric_params.pressure / 101325) * (273.15 / atmospheric_params.temperature)
        
        # Critical power scales with density
        critical_power = self.critical_power_air * density_ratio
        
        return critical_power
    
    def calculate_filamentation_length(self, power: float, 
                                     atmospheric_params: AtmosphericParameters) -> float:
        """
        Calculate filamentation length for given power and conditions.
        
        Args:
            power: Laser power (W)
            atmospheric_params: Atmospheric conditions
            
        Returns:
            Filamentation length (m)
        """
        critical_power = self.calculate_critical_power(atmospheric_params)
        
        if power < critical_power:
            return 0.0  # No filamentation
        
        # Empirical formula for filamentation length
        power_ratio = power / critical_power
        
        # Rayleigh length
        beam_waist = 1e-3  # Assume 1mm beam waist
        rayleigh_length = math.pi * beam_waist**2 / self.config.wavelength
        
        # Filamentation length (simplified model)
        filamentation_length = rayleigh_length * math.sqrt(power_ratio - 1)
        
        # Limit by atmospheric absorption
        max_length = self._calculate_absorption_length(atmospheric_params)
        
        return min(filamentation_length, max_length)
    
    def _calculate_absorption_length(self, atmospheric_params: AtmosphericParameters) -> float:
        """Calculate atmospheric absorption length."""
        # Simplified atmospheric absorption model
        if atmospheric_params.condition == AtmosphericCondition.CLEAR:
            absorption_coeff = 0.1e-3  # m^-1
        elif atmospheric_params.condition == AtmosphericCondition.HAZE:
            absorption_coeff = 0.5e-3
        elif atmospheric_params.condition == AtmosphericCondition.FOG:
            absorption_coeff = 5e-3
        else:
            absorption_coeff = 1e-3
        
        return 1 / absorption_coeff
    
    def calculate_plasma_density(self, power: float, position: float,
                               atmospheric_params: AtmosphericParameters) -> float:
        """
        Calculate plasma density at given position along filament.
        
        Args:
            power: Laser power (W)
            position: Position along beam (m)
            atmospheric_params: Atmospheric conditions
            
        Returns:
            Plasma density (m^-3)
        """
        filamentation_length = self.calculate_filamentation_length(power, atmospheric_params)
        
        if position > filamentation_length or filamentation_length == 0:
            return 0.0
        
        # Peak plasma density (empirical)
        peak_density = 1e16 * (power / self.critical_power_air)  # m^-3
        
        # Gaussian profile along filament
        sigma = filamentation_length / 3  # Standard deviation
        density = peak_density * math.exp(-0.5 * (position - filamentation_length/2)**2 / sigma**2)
        
        return density
    
    def calculate_plasma_lifetime(self, plasma_density: float,
                                atmospheric_params: AtmosphericParameters) -> float:
        """
        Calculate plasma lifetime.
        
        Args:
            plasma_density: Plasma density (m^-3)
            atmospheric_params: Atmospheric conditions
            
        Returns:
            Plasma lifetime (s)
        """
        if plasma_density == 0:
            return 0.0
        
        # Recombination rate depends on density and temperature
        base_lifetime = 1e-6  # 1 μs base lifetime
        
        # Higher density leads to faster recombination
        density_factor = 1e16 / max(plasma_density, 1e10)
        
        # Temperature effects (higher temperature = longer lifetime)
        temp_factor = atmospheric_params.temperature / 273.15
        
        lifetime = base_lifetime * density_factor * temp_factor
        
        return lifetime


class AdaptiveOpticsController:
    """Adaptive optics system for beam quality maintenance."""
    
    def __init__(self, ao_config: AdaptiveOpticsConfiguration):
        """Initialize adaptive optics controller."""
        self.config = ao_config
        self.logger = get_engine_logger('sensors.laser.adaptive_optics')
        
        # System state
        self.wavefront_measurements: List[np.ndarray] = []
        self.correction_commands: List[np.ndarray] = []
        self.closed_loop_active = False
        
    def measure_wavefront(self, atmospheric_params: AtmosphericParameters,
                         propagation_distance: float) -> np.ndarray:
        """
        Simulate wavefront measurement using Shack-Hartmann sensor.
        
        Args:
            atmospheric_params: Atmospheric conditions
            propagation_distance: Distance laser has propagated (m)
            
        Returns:
            Wavefront phase map (radians)
        """
        # Generate turbulence-induced wavefront distortion
        n_sub = int(math.sqrt(self.config.wavefront_sensor_subapertures))
        
        # Fried parameter (coherence length)
        r0 = self._calculate_fried_parameter(atmospheric_params, propagation_distance)
        
        # Generate Kolmogorov turbulence phase screen
        phase_screen = self._generate_turbulence_screen(n_sub, r0)
        
        # Add measurement noise
        noise_level = 0.1  # rad RMS
        noise = np.random.normal(0, noise_level, phase_screen.shape)
        
        measured_wavefront = phase_screen + noise
        self.wavefront_measurements.append(measured_wavefront)
        
        return measured_wavefront
    
    def _calculate_fried_parameter(self, atmospheric_params: AtmosphericParameters,
                                  distance: float) -> float:
        """Calculate Fried parameter (atmospheric coherence length)."""
        # Fried parameter: r0 = (0.423 * k² * Cn² * L)^(-3/5)
        k = 2 * math.pi / 1.064e-6  # Wavenumber for Nd:YAG laser
        cn2 = atmospheric_params.turbulence_strength
        
        if cn2 == 0:
            return self.config.aperture_diameter  # No turbulence
        
        r0 = (0.423 * k**2 * cn2 * distance)**(-3/5)
        
        return max(r0, 0.01)  # Minimum 1 cm coherence length
    
    def _generate_turbulence_screen(self, n_points: int, r0: float) -> np.ndarray:
        """Generate Kolmogorov turbulence phase screen."""
        # Simplified turbulence simulation
        # Generate random phase variations scaled by r0
        
        # Create random phase screen with appropriate scaling
        # Use a simple model that produces reasonable phase values
        phase_screen = np.random.normal(0, 0.5, (n_points, n_points))  # 0.5 rad RMS
        
        return phase_screen
    
    def calculate_correction(self, wavefront: np.ndarray) -> np.ndarray:
        """
        Calculate deformable mirror correction commands.
        
        Args:
            wavefront: Measured wavefront phase (radians)
            
        Returns:
            Actuator commands (normalized -1 to 1)
        """
        # Simple zonal correction (each actuator corrects local wavefront)
        n_act = int(math.sqrt(self.config.actuator_count))
        
        # Resize wavefront to match actuator grid
        if wavefront.shape != (n_act, n_act):
            # Simple interpolation
            from scipy.ndimage import zoom
            scale_factor = n_act / wavefront.shape[0]
            resized_wavefront = zoom(wavefront, scale_factor)
        else:
            resized_wavefront = wavefront
        
        # Correction is negative of wavefront error
        correction = -resized_wavefront
        
        # Normalize to actuator range
        max_stroke = 10e-6  # 10 μm maximum actuator stroke
        wavelength = 1.064e-6  # Nd:YAG wavelength
        max_phase = 2 * math.pi * max_stroke / wavelength
        
        normalized_correction = np.clip(correction / max_phase, -1, 1)
        
        self.correction_commands.append(normalized_correction)
        
        return normalized_correction
    
    def apply_correction(self, correction_commands: np.ndarray) -> float:
        """
        Apply correction and calculate residual wavefront error.
        
        Args:
            correction_commands: Actuator commands
            
        Returns:
            Residual wavefront error (rad RMS)
        """
        # Simulate correction effectiveness
        correction_efficiency = 0.8  # 80% correction efficiency
        
        # Calculate residual error
        if len(self.wavefront_measurements) > 0:
            original_error = np.std(self.wavefront_measurements[-1])
            residual_error = original_error * (1 - correction_efficiency)
        else:
            residual_error = self.config.residual_wavefront_error
        
        return residual_error
    
    def get_strehl_ratio(self, residual_error: float) -> float:
        """
        Calculate Strehl ratio from residual wavefront error.
        
        Args:
            residual_error: Residual wavefront error (rad RMS)
            
        Returns:
            Strehl ratio (0-1)
        """
        # Maréchal approximation
        strehl = math.exp(-(residual_error**2))
        
        return max(min(strehl, 1.0), 0.0)


class LaserInducedBreakdownSpectroscopy:
    """Laser-induced breakdown spectroscopy for radiation detection."""
    
    def __init__(self, laser_config: LaserConfiguration):
        """Initialize LIBS system."""
        self.config = laser_config
        self.logger = get_engine_logger('sensors.laser.libs')
        
        # Spectral database (simplified)
        self.element_lines = {
            'H': [656.3, 486.1, 434.0],  # nm (Balmer series)
            'He': [587.6, 501.6, 471.3],  # nm
            'Li': [670.8, 460.3],  # nm
            'C': [247.9, 193.1],  # nm
            'N': [500.5, 567.7],  # nm
            'O': [777.4, 844.6],  # nm
            'U': [424.4, 435.6, 591.5],  # nm (uranium lines)
            'Pu': [430.1, 476.2, 520.8]  # nm (plutonium lines)
        }
    
    def calculate_plasma_temperature(self, laser_power: float,
                                   target_material: str) -> float:
        """
        Calculate plasma temperature from laser-induced breakdown.
        
        Args:
            laser_power: Laser power (W)
            target_material: Target material type
            
        Returns:
            Plasma temperature (K)
        """
        # Empirical relationship between laser power and plasma temperature
        base_temp = 10000  # K (base plasma temperature)
        
        # Power scaling
        power_factor = (laser_power / 1e6) ** 0.3  # MW scaling
        
        # Material-dependent factors
        material_factors = {
            'metal': 1.2,
            'ceramic': 1.0,
            'polymer': 0.8,
            'composite': 0.9
        }
        
        material_factor = material_factors.get(target_material, 1.0)
        
        plasma_temp = base_temp * power_factor * material_factor
        
        return min(plasma_temp, 50000)  # Cap at 50,000 K
    
    def simulate_spectrum(self, plasma_temp: float, elements: List[str],
                         concentrations: List[float]) -> Dict[float, float]:
        """
        Simulate LIBS spectrum for given elements and concentrations.
        
        Args:
            plasma_temp: Plasma temperature (K)
            elements: List of elements present
            concentrations: Concentration of each element (fraction)
            
        Returns:
            Dictionary of wavelength (nm) -> intensity
        """
        spectrum = {}
        
        for element, concentration in zip(elements, concentrations):
            if element in self.element_lines:
                for wavelength in self.element_lines[element]:
                    # Boltzmann distribution for line intensity
                    intensity = self._calculate_line_intensity(
                        wavelength, plasma_temp, concentration
                    )
                    spectrum[wavelength] = spectrum.get(wavelength, 0) + intensity
        
        return spectrum
    
    def _calculate_line_intensity(self, wavelength: float, temperature: float,
                                concentration: float) -> float:
        """Calculate spectral line intensity."""
        # Simplified Boltzmann distribution
        k_b = 1.380649e-23  # Boltzmann constant
        h = 6.62607015e-34  # Planck constant
        c = 299792458  # Speed of light
        
        # Energy of transition
        energy = h * c / (wavelength * 1e-9)
        
        # Boltzmann factor
        boltzmann_factor = math.exp(-energy / (k_b * temperature))
        
        # Line intensity proportional to concentration and Boltzmann factor
        intensity = concentration * boltzmann_factor * 1000  # Arbitrary scaling
        
        return intensity
    
    def detect_radioactive_elements(self, spectrum: Dict[float, float],
                                  detection_threshold: float = 100) -> List[str]:
        """
        Detect radioactive elements from spectrum.
        
        Args:
            spectrum: Measured spectrum (wavelength -> intensity)
            detection_threshold: Minimum intensity for detection
            
        Returns:
            List of detected radioactive elements
        """
        detected_elements = []
        
        # Check for uranium lines
        uranium_lines = self.element_lines['U']
        uranium_detected = any(
            spectrum.get(line, 0) > detection_threshold for line in uranium_lines
        )
        if uranium_detected:
            detected_elements.append('U')
        
        # Check for plutonium lines
        plutonium_lines = self.element_lines['Pu']
        plutonium_detected = any(
            spectrum.get(line, 0) > detection_threshold for line in plutonium_lines
        )
        if plutonium_detected:
            detected_elements.append('Pu')
        
        return detected_elements


class LaserSafetyAnalyzer:
    """Safety analysis tools for laser operation limits."""
    
    def __init__(self):
        """Initialize laser safety analyzer."""
        self.logger = get_engine_logger('sensors.laser.safety')
        
        # Safety standards (IEC 60825-1)
        self.mpe_values = {  # Maximum Permissible Exposure (J/m²)
            'eye_visible': 5e-7,  # 500 nJ/cm²
            'eye_near_ir': 5e-6,  # 5 μJ/cm²
            'skin_visible': 2e-2,  # 20 mJ/cm²
            'skin_near_ir': 1e-1   # 100 mJ/cm²
        }
    
    def calculate_beam_divergence(self, initial_diameter: float,
                                distance: float, wavelength: float) -> float:
        """
        Calculate beam divergence at given distance.
        
        Args:
            initial_diameter: Initial beam diameter (m)
            distance: Propagation distance (m)
            wavelength: Laser wavelength (m)
            
        Returns:
            Beam diameter at distance (m)
        """
        # Diffraction-limited divergence
        divergence_angle = 1.22 * wavelength / initial_diameter
        
        # Beam diameter at distance
        beam_diameter = initial_diameter + 2 * distance * math.tan(divergence_angle / 2)
        
        return beam_diameter
    
    def calculate_irradiance(self, power: float, beam_diameter: float) -> float:
        """
        Calculate laser irradiance.
        
        Args:
            power: Laser power (W)
            beam_diameter: Beam diameter (m)
            
        Returns:
            Irradiance (W/m²)
        """
        beam_area = math.pi * (beam_diameter / 2) ** 2
        irradiance = power / beam_area
        
        return irradiance
    
    def assess_eye_safety(self, power: float, beam_diameter: float,
                         wavelength: float, exposure_time: float) -> Dict[str, Any]:
        """
        Assess eye safety for laser exposure.
        
        Args:
            power: Laser power (W)
            beam_diameter: Beam diameter (m)
            wavelength: Laser wavelength (m)
            exposure_time: Exposure duration (s)
            
        Returns:
            Safety assessment results
        """
        # Calculate irradiance
        irradiance = self.calculate_irradiance(power, beam_diameter)
        
        # Calculate radiant exposure
        radiant_exposure = irradiance * exposure_time  # J/m²
        
        # Determine wavelength category
        if 400e-9 <= wavelength <= 700e-9:
            category = 'eye_visible'
        elif 700e-9 < wavelength <= 1400e-9:
            category = 'eye_near_ir'
        else:
            category = 'eye_near_ir'  # Default to near-IR
        
        mpe = self.mpe_values[category]
        
        # Safety factor
        safety_factor = radiant_exposure / mpe
        
        # Determine safety class
        if safety_factor <= 1:
            safety_class = "Class 1 (Safe)"
        elif safety_factor <= 5:
            safety_class = "Class 1M (Safe with naked eye)"
        elif safety_factor <= 25:
            safety_class = "Class 2 (Visible, blink reflex protection)"
        elif safety_factor <= 125:
            safety_class = "Class 2M (Visible, safe with naked eye)"
        elif safety_factor <= 500:
            safety_class = "Class 3R (Low risk)"
        elif safety_factor <= 2500:
            safety_class = "Class 3B (Dangerous)"
        else:
            safety_class = "Class 4 (Extremely dangerous)"
        
        return {
            'irradiance_w_per_m2': irradiance,
            'radiant_exposure_j_per_m2': radiant_exposure,
            'mpe_j_per_m2': mpe,
            'safety_factor': safety_factor,
            'safety_class': safety_class,
            'safe': safety_factor <= 1
        }
    
    def calculate_nominal_ocular_hazard_distance(self, power: float,
                                               beam_divergence: float,
                                               wavelength: float) -> float:
        """
        Calculate Nominal Ocular Hazard Distance (NOHD).
        
        Args:
            power: Laser power (W)
            beam_divergence: Beam divergence (rad)
            wavelength: Laser wavelength (m)
            
        Returns:
            NOHD (m)
        """
        # Determine MPE (convert to J/m² for pulsed exposure)
        if 400e-9 <= wavelength <= 700e-9:
            mpe = self.mpe_values['eye_visible'] * 1000  # Scale up for reasonable comparison
        else:
            mpe = self.mpe_values['eye_near_ir'] * 1000
        
        # Limiting aperture (7 mm for eye)
        limiting_aperture = 7e-3  # m
        
        # NOHD calculation (simplified)
        # Scale MPE for continuous wave operation
        mpe_scaled = mpe * 1000  # Scale for reasonable calculation
        nohd = math.sqrt(4 * power / (math.pi * mpe_scaled * (beam_divergence ** 2)))
        
        return nohd
    
    def generate_safety_report(self, laser_config: LaserConfiguration,
                             operating_distance: float) -> Dict[str, Any]:
        """
        Generate comprehensive safety report.
        
        Args:
            laser_config: Laser configuration
            operating_distance: Typical operating distance (m)
            
        Returns:
            Comprehensive safety report
        """
        # Calculate beam parameters at operating distance
        initial_diameter = 0.01  # Assume 1 cm initial beam
        beam_diameter = self.calculate_beam_divergence(
            initial_diameter, operating_distance, laser_config.wavelength
        )
        
        # Eye safety assessment
        eye_safety = self.assess_eye_safety(
            laser_config.peak_power, beam_diameter,
            laser_config.wavelength, 0.25  # 0.25 s exposure
        )
        
        # NOHD calculation
        nohd = self.calculate_nominal_ocular_hazard_distance(
            laser_config.peak_power, laser_config.beam_divergence,
            laser_config.wavelength
        )
        
        # Skin safety (simplified)
        skin_irradiance = self.calculate_irradiance(laser_config.peak_power, beam_diameter)
        skin_mpe = self.mpe_values['skin_near_ir']
        skin_safe = skin_irradiance * 0.25 <= skin_mpe  # 0.25 s exposure
        
        return {
            'laser_class': eye_safety['safety_class'],
            'eye_safe': eye_safety['safe'],
            'skin_safe': skin_safe,
            'nohd_m': nohd,
            'beam_diameter_at_distance_m': beam_diameter,
            'irradiance_w_per_m2': eye_safety['irradiance_w_per_m2'],
            'safety_factor': eye_safety['safety_factor'],
            'recommendations': self._generate_safety_recommendations(eye_safety, nohd)
        }
    
    def _generate_safety_recommendations(self, eye_safety: Dict[str, Any],
                                       nohd: float) -> List[str]:
        """Generate safety recommendations based on assessment."""
        recommendations = []
        
        if not eye_safety['safe']:
            recommendations.append("Eye protection required - use appropriate laser safety glasses")
            recommendations.append(f"Maintain minimum distance of {nohd:.1f} m from beam path")
        
        if eye_safety['safety_factor'] > 100:
            recommendations.append("Beam stop or attenuator required")
            recommendations.append("Restricted access area must be established")
        
        if eye_safety['safety_factor'] > 1000:
            recommendations.append("Remote operation strongly recommended")
            recommendations.append("Interlock systems required")
        
        recommendations.append("Proper beam alignment and termination required")
        recommendations.append("Regular safety training for all personnel")
        
        return recommendations