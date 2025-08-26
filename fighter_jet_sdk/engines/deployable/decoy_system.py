"""
Decoy System Simulation

This module implements decoy system simulation with visual and radar signature
modeling, deployment sequence optimization, and effectiveness assessment against
threat systems.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from fighter_jet_sdk.common.data_models import Position3D, Velocity3D
from fighter_jet_sdk.common.interfaces import SimulationComponent

logger = logging.getLogger(__name__)


class DecoyType(Enum):
    """Types of decoy systems"""
    CHAFF = "chaff"
    FLARE = "flare"
    TOWED_DECOY = "towed_decoy"
    EXPENDABLE_JAMMER = "expendable_jammer"
    PLASMA_DECOY = "plasma_decoy"
    CORNER_REFLECTOR = "corner_reflector"


class ThreatType(Enum):
    """Types of threat systems"""
    RADAR_GUIDED_MISSILE = "radar_guided_missile"
    INFRARED_MISSILE = "infrared_missile"
    VISUAL_TRACKING = "visual_tracking"
    ELECTRONIC_WARFARE = "electronic_warfare"
    MULTI_MODE = "multi_mode"


@dataclass
class DecoySpecifications:
    """Specifications for individual decoy"""
    decoy_type: DecoyType
    radar_cross_section: float  # m²
    infrared_signature: float  # W/sr
    visual_signature: float  # relative brightness
    deployment_time: float  # seconds
    effective_duration: float  # seconds
    deployment_velocity: float  # m/s
    mass: float  # kg
    cost: float  # $


@dataclass
class ThreatSystem:
    """Threat system characteristics"""
    threat_id: str
    threat_type: ThreatType
    detection_range: float  # km
    tracking_accuracy: float  # degrees
    seeker_frequency: Optional[float] = None  # Hz for radar threats
    seeker_bandwidth: Optional[float] = None  # Hz
    countermeasure_resistance: float = 0.5  # 0-1 scale


@dataclass
class DecoyState:
    """Current state of deployed decoy"""
    decoy_id: str
    decoy_type: DecoyType
    position: Position3D
    velocity: Velocity3D
    deployment_time: float
    remaining_effectiveness: float  # 0-1 scale
    active: bool = True


class SignatureModeler:
    """Models visual and radar signatures of decoys"""
    
    def __init__(self):
        self.atmospheric_attenuation = 0.1  # dB/km
        self.weather_factor = 1.0
    
    def calculate_radar_signature(self, 
                                decoy: DecoyState,
                                observer_position: Position3D,
                                frequency: float) -> float:
        """Calculate radar cross-section as seen by observer"""
        base_rcs = self._get_base_rcs(decoy.decoy_type)
        
        # Distance-based attenuation
        distance = decoy.position.distance_to(observer_position) / 1000  # km
        attenuation = np.exp(-self.atmospheric_attenuation * distance)
        
        # Frequency-dependent effects
        frequency_factor = self._calculate_frequency_factor(decoy.decoy_type, frequency)
        
        # Time-based degradation
        time_factor = decoy.remaining_effectiveness
        
        # Aspect angle effects
        aspect_factor = self._calculate_aspect_factor(decoy, observer_position)
        
        effective_rcs = (base_rcs * attenuation * frequency_factor * 
                        time_factor * aspect_factor * self.weather_factor)
        
        return max(0.0, effective_rcs)
    
    def calculate_infrared_signature(self,
                                   decoy: DecoyState,
                                   observer_position: Position3D) -> float:
        """Calculate infrared signature as seen by observer"""
        base_ir = self._get_base_ir_signature(decoy.decoy_type)
        
        # Distance-based attenuation (inverse square law)
        distance = decoy.position.distance_to(observer_position)
        distance_factor = 1.0 / (distance**2 + 1.0)  # +1 to avoid division by zero
        
        # Time-based degradation
        time_factor = decoy.remaining_effectiveness
        
        # Atmospheric effects
        atmospheric_factor = np.exp(-0.05 * distance / 1000)  # km
        
        effective_ir = (base_ir * distance_factor * time_factor * 
                       atmospheric_factor * self.weather_factor)
        
        return max(0.0, effective_ir)
    
    def calculate_visual_signature(self,
                                 decoy: DecoyState,
                                 observer_position: Position3D,
                                 lighting_conditions: str = "daylight") -> float:
        """Calculate visual signature as seen by observer"""
        base_visual = self._get_base_visual_signature(decoy.decoy_type)
        
        # Distance-based visibility
        distance = decoy.position.distance_to(observer_position) / 1000  # km
        visibility_factor = np.exp(-0.2 * distance)  # Atmospheric scattering
        
        # Lighting conditions
        lighting_factor = self._get_lighting_factor(lighting_conditions)
        
        # Time-based effects
        time_factor = decoy.remaining_effectiveness
        
        effective_visual = (base_visual * visibility_factor * 
                          lighting_factor * time_factor)
        
        return max(0.0, effective_visual)
    
    def _get_base_rcs(self, decoy_type: DecoyType) -> float:
        """Get base radar cross-section for decoy type"""
        rcs_values = {
            DecoyType.CHAFF: 10.0,
            DecoyType.FLARE: 0.1,
            DecoyType.TOWED_DECOY: 5.0,
            DecoyType.EXPENDABLE_JAMMER: 1.0,
            DecoyType.PLASMA_DECOY: 15.0,
            DecoyType.CORNER_REFLECTOR: 100.0
        }
        return rcs_values.get(decoy_type, 1.0)
    
    def _get_base_ir_signature(self, decoy_type: DecoyType) -> float:
        """Get base infrared signature for decoy type"""
        ir_values = {
            DecoyType.CHAFF: 0.1,
            DecoyType.FLARE: 1000.0,
            DecoyType.TOWED_DECOY: 10.0,
            DecoyType.EXPENDABLE_JAMMER: 5.0,
            DecoyType.PLASMA_DECOY: 50.0,
            DecoyType.CORNER_REFLECTOR: 1.0
        }
        return ir_values.get(decoy_type, 1.0)
    
    def _get_base_visual_signature(self, decoy_type: DecoyType) -> float:
        """Get base visual signature for decoy type"""
        visual_values = {
            DecoyType.CHAFF: 0.5,
            DecoyType.FLARE: 10.0,
            DecoyType.TOWED_DECOY: 2.0,
            DecoyType.EXPENDABLE_JAMMER: 1.0,
            DecoyType.PLASMA_DECOY: 8.0,
            DecoyType.CORNER_REFLECTOR: 3.0
        }
        return visual_values.get(decoy_type, 1.0)
    
    def _calculate_frequency_factor(self, decoy_type: DecoyType, frequency: float) -> float:
        """Calculate frequency-dependent effectiveness factor"""
        if decoy_type == DecoyType.CHAFF:
            # Chaff effectiveness depends on frequency matching
            optimal_freq = 10e9  # 10 GHz
            freq_deviation = abs(frequency - optimal_freq) / optimal_freq
            return np.exp(-freq_deviation)
        elif decoy_type == DecoyType.PLASMA_DECOY:
            # Plasma decoys are broadband but less effective at very high frequencies
            return max(0.1, 1.0 - frequency / 100e9)
        else:
            return 1.0
    
    def _calculate_aspect_factor(self, decoy: DecoyState, observer_position: Position3D) -> float:
        """Calculate aspect angle-dependent effectiveness"""
        # Simplified aspect angle calculation
        return np.random.uniform(0.7, 1.0)  # Most decoys are somewhat omnidirectional
    
    def _get_lighting_factor(self, lighting_conditions: str) -> float:
        """Get lighting condition factor for visual signatures"""
        factors = {
            "daylight": 1.0,
            "twilight": 0.5,
            "night": 0.1,
            "overcast": 0.7,
            "clear": 1.2
        }
        return factors.get(lighting_conditions, 1.0)


class DeploymentSequencer:
    """Optimizes decoy deployment sequence and timing"""
    
    def __init__(self):
        self.deployment_patterns = {
            "burst": self._burst_deployment,
            "sequential": self._sequential_deployment,
            "adaptive": self._adaptive_deployment,
            "coordinated": self._coordinated_deployment
        }
    
    def optimize_deployment_sequence(self,
                                   available_decoys: List[DecoySpecifications],
                                   threat_systems: List[ThreatSystem],
                                   aircraft_position: Position3D,
                                   aircraft_velocity: Velocity3D,
                                   pattern: str = "adaptive") -> List[Dict]:
        """Optimize deployment sequence based on threat assessment"""
        
        if pattern not in self.deployment_patterns:
            pattern = "adaptive"
        
        return self.deployment_patterns[pattern](
            available_decoys, threat_systems, aircraft_position, aircraft_velocity
        )
    
    def _burst_deployment(self,
                         decoys: List[DecoySpecifications],
                         threats: List[ThreatSystem],
                         position: Position3D,
                         velocity: Velocity3D) -> List[Dict]:
        """Deploy all decoys simultaneously"""
        deployment_sequence = []
        
        for i, decoy in enumerate(decoys):
            deployment_sequence.append({
                'decoy_index': i,
                'deployment_time': 0.0,
                'deployment_position': position,
                'deployment_velocity': self._calculate_deployment_velocity(decoy, velocity),
                'priority': self._calculate_decoy_priority(decoy, threats)
            })
        
        return sorted(deployment_sequence, key=lambda x: x['priority'], reverse=True)
    
    def _sequential_deployment(self,
                             decoys: List[DecoySpecifications],
                             threats: List[ThreatSystem],
                             position: Position3D,
                             velocity: Velocity3D) -> List[Dict]:
        """Deploy decoys in sequence with optimal timing"""
        deployment_sequence = []
        current_time = 0.0
        
        # Sort decoys by effectiveness against current threats
        sorted_decoys = sorted(enumerate(decoys), 
                             key=lambda x: self._calculate_decoy_priority(x[1], threats),
                             reverse=True)
        
        for i, (decoy_index, decoy) in enumerate(sorted_decoys):
            deployment_sequence.append({
                'decoy_index': decoy_index,
                'deployment_time': current_time,
                'deployment_position': self._calculate_future_position(position, velocity, current_time),
                'deployment_velocity': self._calculate_deployment_velocity(decoy, velocity),
                'priority': self._calculate_decoy_priority(decoy, threats)
            })
            current_time += 0.5  # 0.5 second intervals
        
        return deployment_sequence
    
    def _adaptive_deployment(self,
                           decoys: List[DecoySpecifications],
                           threats: List[ThreatSystem],
                           position: Position3D,
                           velocity: Velocity3D) -> List[Dict]:
        """Adaptively deploy decoys based on threat characteristics"""
        deployment_sequence = []
        
        # Group threats by type
        threat_groups = {}
        for threat in threats:
            if threat.threat_type not in threat_groups:
                threat_groups[threat.threat_type] = []
            threat_groups[threat.threat_type].append(threat)
        
        # Deploy specific decoys for each threat type
        current_time = 0.0
        for threat_type, threat_list in threat_groups.items():
            best_decoys = self._select_best_decoys_for_threat(decoys, threat_type)
            
            for decoy_index, decoy in best_decoys:
                deployment_sequence.append({
                    'decoy_index': decoy_index,
                    'deployment_time': current_time,
                    'deployment_position': self._calculate_future_position(position, velocity, current_time),
                    'deployment_velocity': self._calculate_deployment_velocity(decoy, velocity),
                    'priority': self._calculate_decoy_priority(decoy, threat_list),
                    'target_threats': [t.threat_id for t in threat_list]
                })
                current_time += 0.2
        
        return sorted(deployment_sequence, key=lambda x: x['deployment_time'])
    
    def _coordinated_deployment(self,
                              decoys: List[DecoySpecifications],
                              threats: List[ThreatSystem],
                              position: Position3D,
                              velocity: Velocity3D) -> List[Dict]:
        """Deploy decoys in coordinated pattern for maximum effectiveness"""
        deployment_sequence = []
        
        # Create a coordinated deployment pattern
        num_decoys = len(decoys)
        if num_decoys == 0:
            return deployment_sequence
        
        # Deploy in waves
        wave_size = min(3, num_decoys)
        waves = [decoys[i:i+wave_size] for i in range(0, num_decoys, wave_size)]
        
        current_time = 0.0
        for wave_index, wave in enumerate(waves):
            wave_time = current_time + wave_index * 1.0  # 1 second between waves
            
            for decoy_index_in_wave, decoy in enumerate(wave):
                global_decoy_index = wave_index * wave_size + decoy_index_in_wave
                
                # Spread decoys spatially within the wave
                spatial_offset = self._calculate_spatial_offset(decoy_index_in_wave, len(wave))
                deployment_pos = Position3D(
                    position.x + spatial_offset[0],
                    position.y + spatial_offset[1],
                    position.z + spatial_offset[2]
                )
                
                deployment_sequence.append({
                    'decoy_index': global_decoy_index,
                    'deployment_time': wave_time,
                    'deployment_position': deployment_pos,
                    'deployment_velocity': self._calculate_deployment_velocity(decoy, velocity),
                    'priority': self._calculate_decoy_priority(decoy, threats),
                    'wave': wave_index
                })
        
        return deployment_sequence
    
    def _calculate_decoy_priority(self, decoy: DecoySpecifications, threats: List[ThreatSystem]) -> float:
        """Calculate priority score for decoy against threats"""
        if not threats:
            return 0.5
        
        effectiveness_scores = []
        for threat in threats:
            score = self._calculate_effectiveness_against_threat(decoy, threat)
            effectiveness_scores.append(score)
        
        return np.mean(effectiveness_scores)
    
    def _calculate_effectiveness_against_threat(self, decoy: DecoySpecifications, threat: ThreatSystem) -> float:
        """Calculate how effective a decoy is against a specific threat"""
        effectiveness_map = {
            (DecoyType.CHAFF, ThreatType.RADAR_GUIDED_MISSILE): 0.9,
            (DecoyType.FLARE, ThreatType.INFRARED_MISSILE): 0.9,
            (DecoyType.TOWED_DECOY, ThreatType.RADAR_GUIDED_MISSILE): 0.8,
            (DecoyType.EXPENDABLE_JAMMER, ThreatType.ELECTRONIC_WARFARE): 0.8,
            (DecoyType.PLASMA_DECOY, ThreatType.RADAR_GUIDED_MISSILE): 0.7,
            (DecoyType.CORNER_REFLECTOR, ThreatType.RADAR_GUIDED_MISSILE): 0.6,
        }
        
        base_effectiveness = effectiveness_map.get((decoy.decoy_type, threat.threat_type), 0.3)
        
        # Adjust for threat countermeasure resistance
        adjusted_effectiveness = base_effectiveness * (1.0 - threat.countermeasure_resistance * 0.5)
        
        return max(0.0, min(1.0, adjusted_effectiveness))
    
    def _select_best_decoys_for_threat(self, decoys: List[DecoySpecifications], threat_type: ThreatType) -> List[Tuple[int, DecoySpecifications]]:
        """Select best decoys for a specific threat type"""
        decoy_scores = []
        
        for i, decoy in enumerate(decoys):
            # Create a dummy threat for scoring
            dummy_threat = ThreatSystem("dummy", threat_type, 10.0, 1.0)
            score = self._calculate_effectiveness_against_threat(decoy, dummy_threat)
            decoy_scores.append((i, decoy, score))
        
        # Sort by effectiveness and return top candidates
        decoy_scores.sort(key=lambda x: x[2], reverse=True)
        return [(i, decoy) for i, decoy, score in decoy_scores[:3]]  # Top 3
    
    def _calculate_deployment_velocity(self, decoy: DecoySpecifications, aircraft_velocity: Velocity3D) -> Velocity3D:
        """Calculate deployment velocity for decoy"""
        # Add some randomization to deployment velocity
        base_velocity = decoy.deployment_velocity
        
        return Velocity3D(
            aircraft_velocity.vx + np.random.uniform(-base_velocity, base_velocity),
            aircraft_velocity.vy + np.random.uniform(-base_velocity, base_velocity),
            aircraft_velocity.vz + np.random.uniform(-base_velocity/2, base_velocity/2)
        )
    
    def _calculate_future_position(self, position: Position3D, velocity: Velocity3D, time: float) -> Position3D:
        """Calculate future position based on current velocity"""
        return Position3D(
            position.x + velocity.vx * time,
            position.y + velocity.vy * time,
            position.z + velocity.vz * time
        )
    
    def _calculate_spatial_offset(self, index: int, total: int) -> Tuple[float, float, float]:
        """Calculate spatial offset for coordinated deployment"""
        if total == 1:
            return (0.0, 0.0, 0.0)
        
        # Arrange in a circle pattern
        angle = 2 * np.pi * index / total
        radius = 50.0  # 50 meter radius
        
        return (
            radius * np.cos(angle),
            radius * np.sin(angle),
            np.random.uniform(-10, 10)  # Small vertical spread
        )


class EffectivenessAssessor:
    """Assesses decoy effectiveness against threat systems"""
    
    def __init__(self):
        self.signature_modeler = SignatureModeler()
    
    def assess_decoy_effectiveness(self,
                                 decoy_states: List[DecoyState],
                                 threat_systems: List[ThreatSystem],
                                 aircraft_position: Position3D,
                                 simulation_time: float) -> Dict[str, Any]:
        """Assess overall effectiveness of deployed decoys"""
        
        assessment = {
            'overall_effectiveness': 0.0,
            'threat_assessments': {},
            'decoy_assessments': {},
            'recommendations': []
        }
        
        if not decoy_states or not threat_systems:
            return assessment
        
        threat_effectiveness = []
        
        for threat in threat_systems:
            threat_assessment = self._assess_against_threat(
                decoy_states, threat, aircraft_position, simulation_time
            )
            assessment['threat_assessments'][threat.threat_id] = threat_assessment
            threat_effectiveness.append(threat_assessment['effectiveness'])
        
        # Calculate overall effectiveness
        assessment['overall_effectiveness'] = np.mean(threat_effectiveness)
        
        # Assess individual decoys
        for decoy in decoy_states:
            decoy_assessment = self._assess_individual_decoy(
                decoy, threat_systems, aircraft_position, simulation_time
            )
            assessment['decoy_assessments'][decoy.decoy_id] = decoy_assessment
        
        # Generate recommendations
        assessment['recommendations'] = self._generate_recommendations(
            decoy_states, threat_systems, assessment
        )
        
        return assessment
    
    def _assess_against_threat(self,
                             decoy_states: List[DecoyState],
                             threat: ThreatSystem,
                             aircraft_position: Position3D,
                             simulation_time: float) -> Dict[str, Any]:
        """Assess decoy effectiveness against a specific threat"""
        
        threat_assessment = {
            'threat_id': threat.threat_id,
            'threat_type': threat.threat_type.value,
            'effectiveness': 0.0,
            'active_decoys': 0,
            'best_decoy': None,
            'confusion_factor': 0.0
        }
        
        active_decoys = [d for d in decoy_states if d.active and d.remaining_effectiveness > 0.1]
        threat_assessment['active_decoys'] = len(active_decoys)
        
        if not active_decoys:
            return threat_assessment
        
        # Calculate signatures for each decoy as seen by the threat
        decoy_signatures = []
        aircraft_signature = self._calculate_aircraft_signature(threat, aircraft_position)
        
        for decoy in active_decoys:
            if threat.threat_type == ThreatType.RADAR_GUIDED_MISSILE:
                signature = self.signature_modeler.calculate_radar_signature(
                    decoy, aircraft_position, threat.seeker_frequency or 10e9
                )
            elif threat.threat_type == ThreatType.INFRARED_MISSILE:
                signature = self.signature_modeler.calculate_infrared_signature(
                    decoy, aircraft_position
                )
            else:
                signature = self.signature_modeler.calculate_visual_signature(
                    decoy, aircraft_position
                )
            
            decoy_signatures.append((decoy, signature))
        
        # Find best decoy
        if decoy_signatures:
            best_decoy, best_signature = max(decoy_signatures, key=lambda x: x[1])
            threat_assessment['best_decoy'] = best_decoy.decoy_id
            
            # Calculate confusion factor (how much decoys confuse the threat)
            total_signature = sum(sig for _, sig in decoy_signatures)
            if aircraft_signature > 0:
                confusion_ratio = total_signature / aircraft_signature
                threat_assessment['confusion_factor'] = min(1.0, confusion_ratio)
            
            # Calculate overall effectiveness
            base_effectiveness = self._calculate_base_effectiveness(threat, active_decoys)
            signature_factor = min(1.0, total_signature / max(aircraft_signature, 0.1))
            time_factor = np.mean([d.remaining_effectiveness for d in active_decoys])
            
            threat_assessment['effectiveness'] = (
                base_effectiveness * signature_factor * time_factor * 
                (1.0 - threat.countermeasure_resistance * 0.3)
            )
        
        return threat_assessment
    
    def _assess_individual_decoy(self,
                               decoy: DecoyState,
                               threats: List[ThreatSystem],
                               aircraft_position: Position3D,
                               simulation_time: float) -> Dict[str, Any]:
        """Assess individual decoy performance"""
        
        decoy_assessment = {
            'decoy_id': decoy.decoy_id,
            'decoy_type': decoy.decoy_type.value,
            'active': decoy.active,
            'remaining_effectiveness': decoy.remaining_effectiveness,
            'age': simulation_time - decoy.deployment_time,
            'threat_effectiveness': {},
            'overall_score': 0.0
        }
        
        if not decoy.active:
            return decoy_assessment
        
        effectiveness_scores = []
        
        for threat in threats:
            if threat.threat_type == ThreatType.RADAR_GUIDED_MISSILE:
                signature = self.signature_modeler.calculate_radar_signature(
                    decoy, aircraft_position, threat.seeker_frequency or 10e9
                )
            elif threat.threat_type == ThreatType.INFRARED_MISSILE:
                signature = self.signature_modeler.calculate_infrared_signature(
                    decoy, aircraft_position
                )
            else:
                signature = self.signature_modeler.calculate_visual_signature(
                    decoy, aircraft_position
                )
            
            # Normalize signature to effectiveness score
            effectiveness = min(1.0, signature / 10.0)  # Assuming max useful signature of 10
            decoy_assessment['threat_effectiveness'][threat.threat_id] = effectiveness
            effectiveness_scores.append(effectiveness)
        
        decoy_assessment['overall_score'] = np.mean(effectiveness_scores) if effectiveness_scores else 0.0
        
        return decoy_assessment
    
    def _calculate_aircraft_signature(self, threat: ThreatSystem, aircraft_position: Position3D) -> float:
        """Calculate aircraft signature as baseline"""
        # Simplified aircraft signature calculation
        if threat.threat_type == ThreatType.RADAR_GUIDED_MISSILE:
            return 1.0  # 1 m² RCS
        elif threat.threat_type == ThreatType.INFRARED_MISSILE:
            return 100.0  # 100 W/sr IR signature
        else:
            return 5.0  # Visual signature
    
    def _calculate_base_effectiveness(self, threat: ThreatSystem, decoys: List[DecoyState]) -> float:
        """Calculate base effectiveness based on decoy types and threat"""
        if not decoys:
            return 0.0
        
        # Count decoys by type
        decoy_counts = {}
        for decoy in decoys:
            decoy_counts[decoy.decoy_type] = decoy_counts.get(decoy.decoy_type, 0) + 1
        
        # Calculate effectiveness based on optimal decoy types for threat
        effectiveness = 0.0
        
        if threat.threat_type == ThreatType.RADAR_GUIDED_MISSILE:
            effectiveness += decoy_counts.get(DecoyType.CHAFF, 0) * 0.3
            effectiveness += decoy_counts.get(DecoyType.TOWED_DECOY, 0) * 0.25
            effectiveness += decoy_counts.get(DecoyType.PLASMA_DECOY, 0) * 0.2
        elif threat.threat_type == ThreatType.INFRARED_MISSILE:
            effectiveness += decoy_counts.get(DecoyType.FLARE, 0) * 0.4
        
        return min(1.0, effectiveness)
    
    def _generate_recommendations(self,
                                decoy_states: List[DecoyState],
                                threats: List[ThreatSystem],
                                assessment: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving decoy effectiveness"""
        recommendations = []
        
        overall_effectiveness = assessment['overall_effectiveness']
        
        if overall_effectiveness < 0.3:
            recommendations.append("Overall decoy effectiveness is low - consider deploying additional decoys")
        
        # Check for specific threat types that are poorly covered
        for threat_id, threat_assessment in assessment['threat_assessments'].items():
            if threat_assessment['effectiveness'] < 0.4:
                threat_type = threat_assessment['threat_type']
                if threat_type == 'radar_guided_missile':
                    recommendations.append(f"Deploy more chaff or towed decoys for {threat_id}")
                elif threat_type == 'infrared_missile':
                    recommendations.append(f"Deploy more flares for {threat_id}")
        
        # Check for expired decoys
        expired_decoys = [d for d in decoy_states if d.remaining_effectiveness < 0.2]
        if len(expired_decoys) > len(decoy_states) * 0.5:
            recommendations.append("Many decoys are losing effectiveness - consider fresh deployment")
        
        return recommendations


class DecoySystemSimulator(SimulationComponent):
    """Main decoy system simulation class"""
    
    def __init__(self, available_decoys: List[DecoySpecifications]):
        super().__init__()
        self.available_decoys = available_decoys
        self.deployed_decoys = []
        self.signature_modeler = SignatureModeler()
        self.deployment_sequencer = DeploymentSequencer()
        self.effectiveness_assessor = EffectivenessAssessor()
        self.aircraft_position = Position3D(0, 0, 1000)
        self.aircraft_velocity = Velocity3D(200, 0, 0)  # 200 m/s forward
        self.threat_systems = []
        
    def initialize(self) -> bool:
        """Initialize the decoy system simulator"""
        try:
            self.deployed_decoys = []
            self.simulation_time = 0.0
            self.initialized = True
            logger.info("Decoy system simulator initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize decoy system simulator: {e}")
            return False
    
    def add_threat_system(self, threat: ThreatSystem) -> None:
        """Add a threat system to track"""
        self.threat_systems.append(threat)
        logger.info(f"Added threat system: {threat.threat_id}")
    
    def deploy_decoys(self,
                     pattern: str = "adaptive",
                     aircraft_position: Optional[Position3D] = None,
                     aircraft_velocity: Optional[Velocity3D] = None) -> bool:
        """Deploy decoys using specified pattern"""
        try:
            if aircraft_position:
                self.aircraft_position = aircraft_position
            if aircraft_velocity:
                self.aircraft_velocity = aircraft_velocity
            
            # Get optimal deployment sequence
            deployment_sequence = self.deployment_sequencer.optimize_deployment_sequence(
                self.available_decoys,
                self.threat_systems,
                self.aircraft_position,
                self.aircraft_velocity,
                pattern
            )
            
            # Deploy decoys according to sequence
            for deployment in deployment_sequence:
                decoy_spec = self.available_decoys[deployment['decoy_index']]
                
                decoy_state = DecoyState(
                    decoy_id=f"decoy_{len(self.deployed_decoys):03d}",
                    decoy_type=decoy_spec.decoy_type,
                    position=deployment['deployment_position'],
                    velocity=deployment['deployment_velocity'],
                    deployment_time=self.simulation_time + deployment['deployment_time'],
                    remaining_effectiveness=1.0
                )
                
                self.deployed_decoys.append(decoy_state)
            
            logger.info(f"Deployed {len(deployment_sequence)} decoys using {pattern} pattern")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy decoys: {e}")
            return False
    
    def update_simulation(self, dt: float) -> Dict[str, Any]:
        """Update simulation state"""
        self.simulation_time += dt
        
        # Update aircraft position
        self.aircraft_position = Position3D(
            self.aircraft_position.x + self.aircraft_velocity.vx * dt,
            self.aircraft_position.y + self.aircraft_velocity.vy * dt,
            self.aircraft_position.z + self.aircraft_velocity.vz * dt
        )
        
        # Update decoy states
        for decoy in self.deployed_decoys:
            if decoy.active:
                # Update position
                decoy.position = Position3D(
                    decoy.position.x + decoy.velocity.vx * dt,
                    decoy.position.y + decoy.velocity.vy * dt,
                    decoy.position.z + decoy.velocity.vz * dt
                )
                
                # Update effectiveness based on age
                age = self.simulation_time - decoy.deployment_time
                decoy.remaining_effectiveness = self._calculate_time_degradation(decoy.decoy_type, age)
                
                # Deactivate if effectiveness is too low
                if decoy.remaining_effectiveness < 0.05:
                    decoy.active = False
        
        # Assess current effectiveness
        effectiveness_assessment = self.effectiveness_assessor.assess_decoy_effectiveness(
            [d for d in self.deployed_decoys if d.active],
            self.threat_systems,
            self.aircraft_position,
            self.simulation_time
        )
        
        return {
            'simulation_time': self.simulation_time,
            'aircraft_position': {
                'x': self.aircraft_position.x,
                'y': self.aircraft_position.y,
                'z': self.aircraft_position.z
            },
            'active_decoys': len([d for d in self.deployed_decoys if d.active]),
            'total_decoys': len(self.deployed_decoys),
            'effectiveness_assessment': effectiveness_assessment,
            'decoy_states': [self._decoy_state_to_dict(d) for d in self.deployed_decoys if d.active]
        }
    
    def _calculate_time_degradation(self, decoy_type: DecoyType, age: float) -> float:
        """Calculate effectiveness degradation over time"""
        degradation_rates = {
            DecoyType.CHAFF: 30.0,  # 30 second effective duration
            DecoyType.FLARE: 10.0,  # 10 second effective duration
            DecoyType.TOWED_DECOY: 300.0,  # 5 minute effective duration
            DecoyType.EXPENDABLE_JAMMER: 60.0,  # 1 minute effective duration
            DecoyType.PLASMA_DECOY: 20.0,  # 20 second effective duration
            DecoyType.CORNER_REFLECTOR: 600.0  # 10 minute effective duration
        }
        
        effective_duration = degradation_rates.get(decoy_type, 30.0)
        return max(0.0, 1.0 - age / effective_duration)
    
    def _decoy_state_to_dict(self, decoy: DecoyState) -> Dict[str, Any]:
        """Convert decoy state to dictionary"""
        return {
            'decoy_id': decoy.decoy_id,
            'decoy_type': decoy.decoy_type.value,
            'position': {
                'x': decoy.position.x,
                'y': decoy.position.y,
                'z': decoy.position.z
            },
            'velocity': {
                'vx': decoy.velocity.vx,
                'vy': decoy.velocity.vy,
                'vz': decoy.velocity.vz
            },
            'deployment_time': decoy.deployment_time,
            'remaining_effectiveness': decoy.remaining_effectiveness,
            'active': decoy.active
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        active_decoys = [d for d in self.deployed_decoys if d.active]
        
        return {
            'simulation_time': self.simulation_time,
            'available_decoys': len(self.available_decoys),
            'deployed_decoys': len(self.deployed_decoys),
            'active_decoys': len(active_decoys),
            'threat_systems': len(self.threat_systems),
            'aircraft_position': {
                'x': self.aircraft_position.x,
                'y': self.aircraft_position.y,
                'z': self.aircraft_position.z
            }
        }