"""AESA Radar modeling and simulation for advanced detection systems."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import math
from ...common.enums import SensorType
from ...core.logging import get_engine_logger


@dataclass
class RadarTarget:
    """Radar target representation."""
    position: Tuple[float, float, float]  # x, y, z coordinates (m)
    velocity: Tuple[float, float, float]  # vx, vy, vz (m/s)
    rcs: float  # Radar cross-section (m²)
    target_id: str = ""
    last_detection_time: float = 0.0


@dataclass
class BeamPattern:
    """Radar beam pattern characteristics."""
    azimuth_angle: float  # degrees
    elevation_angle: float  # degrees
    beam_width_az: float  # degrees
    beam_width_el: float  # degrees
    gain: float  # dB
    sidelobe_level: float  # dB


@dataclass
class RadarConfiguration:
    """AESA radar system configuration."""
    frequency: float  # Hz
    peak_power: float  # W
    array_elements: int
    element_spacing: float  # m
    pulse_width: float  # s
    pulse_repetition_frequency: float  # Hz
    noise_figure: float  # dB
    system_losses: float  # dB
    antenna_gain: float  # dB


class AESARadarModel:
    """Active Electronically Scanned Array radar modeling and simulation."""
    
    def __init__(self, config: RadarConfiguration):
        """Initialize AESA radar model."""
        self.config = config
        self.logger = get_engine_logger('sensors.aesa')
        self.tracked_targets: Dict[str, RadarTarget] = {}
        self.beam_patterns: List[BeamPattern] = []
        self.jamming_sources: List[Dict[str, Any]] = []
        
        # Physical constants
        self.c = 299792458  # Speed of light (m/s)
        self.k_boltzmann = 1.380649e-23  # Boltzmann constant (J/K)
        self.temp_noise = 290  # System noise temperature (K)
        
        # Calculate derived parameters
        self._calculate_derived_parameters()
    
    def _calculate_derived_parameters(self):
        """Calculate derived radar parameters."""
        self.wavelength = self.c / self.config.frequency
        self.max_unambiguous_range = (self.c / (2 * self.config.pulse_repetition_frequency))
        self.range_resolution = (self.c * self.config.pulse_width) / 2
        
        # Array aperture size
        self.aperture_size = (self.config.array_elements ** 0.5) * self.config.element_spacing
        
        # Theoretical beam width (3dB)
        self.theoretical_beam_width = math.degrees(0.886 * self.wavelength / self.aperture_size)
        
        self.logger.info(f"AESA radar initialized: {self.config.frequency/1e9:.2f} GHz, "
                        f"{self.config.array_elements} elements, "
                        f"beam width: {self.theoretical_beam_width:.2f}°")
    
    def calculate_radar_equation(self, target_range: float, target_rcs: float) -> float:
        """
        Calculate received power using radar equation.
        
        Args:
            target_range: Distance to target (m)
            target_rcs: Target radar cross-section (m²)
            
        Returns:
            Received power (W)
        """
        # Radar equation: Pr = (Pt * G² * λ² * σ) / ((4π)³ * R⁴ * L)
        pt = self.config.peak_power
        g_linear = 10 ** (self.config.antenna_gain / 10)
        lambda_sq = self.wavelength ** 2
        sigma = target_rcs
        range_fourth = target_range ** 4
        losses_linear = 10 ** (self.config.system_losses / 10)
        
        numerator = pt * (g_linear ** 2) * lambda_sq * sigma
        denominator = ((4 * math.pi) ** 3) * range_fourth * losses_linear
        
        received_power = numerator / denominator
        
        return received_power
    
    def calculate_snr(self, received_power: float) -> float:
        """
        Calculate signal-to-noise ratio.
        
        Args:
            received_power: Received signal power (W)
            
        Returns:
            SNR in dB
        """
        # Noise power calculation
        bandwidth = 1 / self.config.pulse_width
        noise_figure_linear = 10 ** (self.config.noise_figure / 10)
        noise_power = self.k_boltzmann * self.temp_noise * bandwidth * noise_figure_linear
        
        snr_linear = received_power / noise_power
        snr_db = 10 * math.log10(max(snr_linear, 1e-10))  # Avoid log(0)
        
        return snr_db
    
    def calculate_detection_probability(self, snr_db: float, false_alarm_rate: float = 1e-6) -> float:
        """
        Calculate probability of detection using Swerling case I model.
        
        Args:
            snr_db: Signal-to-noise ratio (dB)
            false_alarm_rate: Desired false alarm rate
            
        Returns:
            Probability of detection (0-1)
        """
        # Convert SNR to linear
        snr_linear = 10 ** (snr_db / 10)
        
        # Threshold for given false alarm rate (simplified)
        threshold_snr = -math.log(false_alarm_rate)
        
        # Swerling case I detection probability (approximation)
        if snr_linear > threshold_snr:
            pd = 1 - math.exp(-snr_linear + threshold_snr)
        else:
            pd = snr_linear / (threshold_snr + 1)
        
        return min(max(pd, 0.0), 1.0)
    
    def generate_beam_pattern(self, azimuth: float, elevation: float) -> BeamPattern:
        """
        Generate beam pattern for given steering angles.
        
        Args:
            azimuth: Azimuth steering angle (degrees)
            elevation: Elevation steering angle (degrees)
            
        Returns:
            BeamPattern object
        """
        # Calculate beam width (increases with steering angle)
        steering_factor = 1 / math.cos(math.radians(max(abs(azimuth), abs(elevation))))
        beam_width_az = self.theoretical_beam_width * steering_factor
        beam_width_el = self.theoretical_beam_width * steering_factor
        
        # Calculate gain reduction due to steering
        steering_loss = 20 * math.log10(math.cos(math.radians(azimuth)) * 
                                       math.cos(math.radians(elevation)))
        gain = self.config.antenna_gain + steering_loss
        
        # Sidelobe level (typical for AESA)
        sidelobe_level = -25.0  # dB below main beam
        
        return BeamPattern(
            azimuth_angle=azimuth,
            elevation_angle=elevation,
            beam_width_az=beam_width_az,
            beam_width_el=beam_width_el,
            gain=gain,
            sidelobe_level=sidelobe_level
        )
    
    def detect_targets(self, targets: List[RadarTarget], 
                      beam_azimuth: float, beam_elevation: float,
                      detection_threshold: float = 13.0) -> List[RadarTarget]:
        """
        Detect targets within the radar beam.
        
        Args:
            targets: List of potential targets
            beam_azimuth: Beam azimuth angle (degrees)
            beam_elevation: Beam elevation angle (degrees)
            detection_threshold: Minimum SNR for detection (dB)
            
        Returns:
            List of detected targets
        """
        detected_targets = []
        beam_pattern = self.generate_beam_pattern(beam_azimuth, beam_elevation)
        
        for target in targets:
            # Calculate target angle relative to beam center
            target_range = math.sqrt(sum(coord**2 for coord in target.position))
            
            if target_range == 0:
                continue
                
            # Calculate angles to target
            target_az = math.degrees(math.atan2(target.position[1], target.position[0]))
            target_el = math.degrees(math.asin(target.position[2] / target_range))
            
            # Check if target is within beam
            az_diff = abs(target_az - beam_azimuth)
            el_diff = abs(target_el - beam_elevation)
            
            if (az_diff <= beam_pattern.beam_width_az / 2 and 
                el_diff <= beam_pattern.beam_width_el / 2):
                
                # Calculate received power and SNR
                received_power = self.calculate_radar_equation(target_range, target.rcs)
                snr = self.calculate_snr(received_power)
                
                # Apply jamming effects if present
                snr_with_jamming = self._apply_jamming_effects(snr, target_range, target_az, target_el)
                
                # Check detection threshold
                if snr_with_jamming >= detection_threshold:
                    detection_prob = self.calculate_detection_probability(snr_with_jamming)
                    
                    # Monte Carlo detection decision
                    if np.random.random() < detection_prob:
                        detected_targets.append(target)
                        self.logger.debug(f"Target detected: range={target_range:.0f}m, "
                                        f"SNR={snr_with_jamming:.1f}dB, "
                                        f"Pd={detection_prob:.3f}")
        
        return detected_targets
    
    def track_targets(self, detected_targets: List[RadarTarget], 
                     current_time: float) -> Dict[str, RadarTarget]:
        """
        Multi-target tracking algorithm.
        
        Args:
            detected_targets: List of currently detected targets
            current_time: Current simulation time (s)
            
        Returns:
            Dictionary of tracked targets with updated states
        """
        # Simple nearest-neighbor tracking (could be enhanced with Kalman filtering)
        updated_tracks = {}
        
        for detected_target in detected_targets:
            best_match = None
            min_distance = float('inf')
            
            # Find closest existing track
            for track_id, existing_track in self.tracked_targets.items():
                # Predict target position based on velocity
                dt = current_time - existing_track.last_detection_time
                predicted_pos = (
                    existing_track.position[0] + existing_track.velocity[0] * dt,
                    existing_track.position[1] + existing_track.velocity[1] * dt,
                    existing_track.position[2] + existing_track.velocity[2] * dt
                )
                
                # Calculate distance between predicted and detected position
                distance = math.sqrt(sum((detected_target.position[i] - predicted_pos[i])**2 
                                       for i in range(3)))
                
                if distance < min_distance and distance < 1000:  # 1km gate
                    min_distance = distance
                    best_match = track_id
            
            if best_match:
                # Update existing track
                existing_track = self.tracked_targets[best_match]
                dt = current_time - existing_track.last_detection_time
                
                if dt > 0:
                    # Update velocity estimate
                    new_velocity = tuple(
                        (detected_target.position[i] - existing_track.position[i]) / dt
                        for i in range(3)
                    )
                    
                    # Simple filtering (could use Kalman filter)
                    alpha = 0.7  # Filter coefficient
                    filtered_velocity = tuple(
                        alpha * new_velocity[i] + (1 - alpha) * existing_track.velocity[i]
                        for i in range(3)
                    )
                    
                    updated_track = RadarTarget(
                        position=detected_target.position,
                        velocity=filtered_velocity,
                        rcs=detected_target.rcs,
                        target_id=best_match,
                        last_detection_time=current_time
                    )
                else:
                    updated_track = existing_track
                    updated_track.position = detected_target.position
                    updated_track.last_detection_time = current_time
                
                updated_tracks[best_match] = updated_track
            else:
                # Create new track
                new_track_id = f"T{len(self.tracked_targets) + len(updated_tracks):03d}"
                new_track = RadarTarget(
                    position=detected_target.position,
                    velocity=(0.0, 0.0, 0.0),  # Initial velocity unknown
                    rcs=detected_target.rcs,
                    target_id=new_track_id,
                    last_detection_time=current_time
                )
                updated_tracks[new_track_id] = new_track
        
        # Remove old tracks (not detected for too long)
        track_timeout = 10.0  # seconds
        for track_id, track in self.tracked_targets.items():
            if (current_time - track.last_detection_time < track_timeout and 
                track_id not in updated_tracks):
                updated_tracks[track_id] = track
        
        self.tracked_targets = updated_tracks
        return self.tracked_targets
    
    def add_jamming_source(self, position: Tuple[float, float, float], 
                          power: float, frequency: float, 
                          jamming_type: str = "noise") -> None:
        """
        Add electronic warfare jamming source.
        
        Args:
            position: Jammer position (x, y, z) in meters
            power: Jammer power (W)
            frequency: Jammer frequency (Hz)
            jamming_type: Type of jamming ("noise", "deception", "barrage")
        """
        jammer = {
            'position': position,
            'power': power,
            'frequency': frequency,
            'type': jamming_type
        }
        self.jamming_sources.append(jammer)
        self.logger.info(f"Added {jamming_type} jammer at {position} with {power}W power")
    
    def _apply_jamming_effects(self, snr_db: float, target_range: float,
                              target_azimuth: float, target_elevation: float) -> float:
        """
        Apply electronic warfare jamming effects to SNR.
        
        Args:
            snr_db: Original SNR (dB)
            target_range: Range to target (m)
            target_azimuth: Target azimuth (degrees)
            target_elevation: Target elevation (degrees)
            
        Returns:
            SNR after jamming effects (dB)
        """
        if not self.jamming_sources:
            return snr_db
        
        total_jamming_power = 0.0
        
        for jammer in self.jamming_sources:
            # Calculate range to jammer
            jammer_range = math.sqrt(sum(coord**2 for coord in jammer['position']))
            
            if jammer_range == 0:
                continue
            
            # Calculate jammer angle
            jammer_az = math.degrees(math.atan2(jammer['position'][1], jammer['position'][0]))
            jammer_el = math.degrees(math.asin(jammer['position'][2] / jammer_range))
            
            # Calculate angular separation from target
            angular_separation = math.sqrt((target_azimuth - jammer_az)**2 + 
                                         (target_elevation - jammer_el)**2)
            
            # Antenna pattern suppression of jammer
            if angular_separation < self.theoretical_beam_width:
                # Jammer in main beam
                pattern_suppression = 0.0  # dB
            else:
                # Jammer in sidelobe
                pattern_suppression = -25.0  # dB (typical sidelobe level)
            
            # Calculate received jamming power
            jammer_power_received = (jammer['power'] * 
                                   10**(self.config.antenna_gain/10) * 
                                   10**(pattern_suppression/10) * 
                                   (self.wavelength/(4*math.pi*jammer_range))**2)
            
            # Frequency separation effects
            freq_separation = abs(jammer['frequency'] - self.config.frequency)
            if freq_separation > 100e6:  # 100 MHz
                # Reduced effectiveness for frequency separation
                jammer_power_received *= 0.1
            
            total_jamming_power += jammer_power_received
        
        if total_jamming_power > 0:
            # Calculate signal power
            signal_power = 10**(snr_db/10) * self.k_boltzmann * self.temp_noise / self.config.pulse_width
            
            # Calculate jammer-to-signal ratio
            jsr_linear = total_jamming_power / signal_power
            
            # Degraded SNR
            degraded_snr_linear = 1 / (1/10**(snr_db/10) + jsr_linear)
            degraded_snr_db = 10 * math.log10(max(degraded_snr_linear, 1e-10))
            
            self.logger.debug(f"Jamming applied: JSR={10*math.log10(jsr_linear):.1f}dB, "
                            f"SNR degraded from {snr_db:.1f} to {degraded_snr_db:.1f}dB")
            
            return degraded_snr_db
        
        return snr_db
    
    def calculate_engagement_timeline(self, target: RadarTarget) -> Dict[str, float]:
        """
        Calculate sensor-to-shooter timeline for target engagement.
        
        Args:
            target: Target to engage
            
        Returns:
            Dictionary with timeline phases and durations
        """
        target_range = math.sqrt(sum(coord**2 for coord in target.position))
        
        # Timeline phases (typical values)
        timeline = {
            'detection_time': 0.1,  # Initial detection (s)
            'classification_time': 2.0,  # Target classification (s)
            'track_establishment': 5.0,  # Stable track (s)
            'threat_assessment': 1.0,  # Threat evaluation (s)
            'weapon_assignment': 0.5,  # Weapon selection (s)
            'engagement_authorization': 2.0,  # Authorization (s)
            'weapon_launch': 0.1,  # Launch command (s)
        }
        
        # Adjust based on target characteristics
        if target.rcs < 0.1:  # Stealth target
            timeline['detection_time'] *= 3
            timeline['classification_time'] *= 2
            timeline['track_establishment'] *= 1.5
        
        if target_range > 100000:  # Long range target
            timeline['classification_time'] *= 1.5
            timeline['track_establishment'] *= 1.2
        
        # Calculate total timeline
        timeline['total_time'] = sum(timeline.values()) - timeline['total_time'] if 'total_time' in timeline else sum(timeline.values())
        
        return timeline
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get radar performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        return {
            'frequency_ghz': self.config.frequency / 1e9,
            'peak_power_kw': self.config.peak_power / 1000,
            'array_elements': self.config.array_elements,
            'max_range_km': self.max_unambiguous_range / 1000,
            'range_resolution_m': self.range_resolution,
            'beam_width_deg': self.theoretical_beam_width,
            'antenna_gain_db': self.config.antenna_gain,
            'active_tracks': len(self.tracked_targets),
            'jamming_sources': len(self.jamming_sources)
        }
    
    def validate_configuration(self) -> List[str]:
        """
        Validate radar configuration parameters.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Frequency validation
        if self.config.frequency < 1e9 or self.config.frequency > 100e9:
            errors.append("Frequency must be between 1-100 GHz")
        
        # Power validation
        if self.config.peak_power <= 0:
            errors.append("Peak power must be positive")
        
        if self.config.peak_power > 1e6:  # 1 MW
            errors.append("Peak power exceeds reasonable limits (>1MW)")
        
        # Array validation
        if self.config.array_elements < 4:
            errors.append("Array must have at least 4 elements")
        
        if self.config.array_elements > 10000:
            errors.append("Array size exceeds practical limits (>10000 elements)")
        
        # Element spacing validation
        if self.config.element_spacing < self.wavelength / 4:
            errors.append("Element spacing too small (causes grating lobes)")
        
        if self.config.element_spacing > self.wavelength:
            errors.append("Element spacing too large (reduces efficiency)")
        
        # Pulse parameters
        if self.config.pulse_width <= 0:
            errors.append("Pulse width must be positive")
        
        if self.config.pulse_repetition_frequency <= 0:
            errors.append("PRF must be positive")
        
        # Check for range ambiguity
        if self.max_unambiguous_range < 10000:  # 10 km minimum
            errors.append("PRF too high - causes range ambiguity at short ranges")
        
        return errors