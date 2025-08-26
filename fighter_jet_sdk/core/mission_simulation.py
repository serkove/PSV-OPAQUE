"""Mission scenario simulation system."""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import logging

from ..common.interfaces import SimulationComponent, BaseEngine
from ..common.data_models import AircraftConfiguration, SensorSystem
from .simulation import MultiPhysicsOrchestrator, SimulationResults, SimulationParameters
from .errors import SimulationError, ValidationError
from .logging import get_logger


class MissionPhase(Enum):
    """Mission phase enumeration."""
    TAKEOFF = "takeoff"
    TRANSIT = "transit"
    APPROACH = "approach"
    ENGAGEMENT = "engagement"
    EGRESS = "egress"
    RETURN = "return"
    LANDING = "landing"


class ThreatType(Enum):
    """Threat type enumeration."""
    SURFACE_TO_AIR = "surface_to_air"
    AIR_TO_AIR = "air_to_air"
    ELECTRONIC_WARFARE = "electronic_warfare"
    CYBER = "cyber"


class EngagementOutcome(Enum):
    """Engagement outcome enumeration."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL_SUCCESS = "partial_success"
    ABORTED = "aborted"


@dataclass
class ThreatDefinition:
    """Definition of a threat in the mission scenario."""
    threat_id: str
    threat_type: ThreatType
    position: np.ndarray  # [x, y, z] in meters
    detection_range: float  # meters
    engagement_range: float  # meters
    radar_cross_section: float  # m²
    electronic_signature: Dict[str, float]
    countermeasures: List[str] = field(default_factory=list)
    active: bool = True


@dataclass
class MissionWaypoint:
    """Mission waypoint definition."""
    waypoint_id: str
    position: np.ndarray  # [x, y, z] in meters
    altitude: float  # meters
    speed: float  # m/s
    heading: float  # radians
    phase: MissionPhase
    duration: float = 0.0  # seconds to spend at waypoint
    actions: List[str] = field(default_factory=list)


@dataclass
class MissionScenario:
    """Complete mission scenario definition."""
    scenario_id: str
    description: str
    waypoints: List[MissionWaypoint]
    threats: List[ThreatDefinition]
    environmental_conditions: Dict[str, Any]
    success_criteria: Dict[str, Any]
    failure_conditions: Dict[str, Any]
    duration: float = 3600.0  # seconds
    rules_of_engagement: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EngagementEvent:
    """Single engagement event in mission timeline."""
    event_id: str
    time: float  # seconds from mission start
    phase: MissionPhase
    threat_id: str
    aircraft_position: np.ndarray
    threat_position: np.ndarray
    detection_probability: float
    engagement_probability: float
    outcome: Optional[EngagementOutcome] = None
    sensor_data: Dict[str, Any] = field(default_factory=dict)
    countermeasures_used: List[str] = field(default_factory=list)


@dataclass
class MissionResults:
    """Results from mission simulation."""
    scenario_id: str
    mission_success: bool
    total_duration: float
    phases_completed: List[MissionPhase]
    engagement_events: List[EngagementEvent]
    sensor_timeline: Dict[str, List[Dict[str, Any]]]
    effectiveness_metrics: Dict[str, float]
    resource_consumption: Dict[str, float]
    survivability_assessment: Dict[str, Any]
    lessons_learned: List[str] = field(default_factory=list)


class SensorToShooterAnalyzer:
    """Analyzes sensor-to-shooter timeline and effectiveness."""
    
    def __init__(self):
        """Initialize sensor-to-shooter analyzer."""
        self.logger = get_logger("sensor_to_shooter_analyzer")
        self.detection_models = {}
        self.engagement_models = {}
    
    def add_detection_model(self, sensor_type: str, model: Callable) -> None:
        """Add detection probability model for sensor type."""
        self.detection_models[sensor_type] = model
        self.logger.info(f"Added detection model for {sensor_type}")
    
    def add_engagement_model(self, weapon_type: str, model: Callable) -> None:
        """Add engagement probability model for weapon type."""
        self.engagement_models[weapon_type] = model
        self.logger.info(f"Added engagement model for {weapon_type}")
    
    def calculate_detection_probability(self, sensor: SensorSystem, 
                                      target_signature: Dict[str, float],
                                      range_km: float,
                                      environmental_factors: Dict[str, float]) -> float:
        """Calculate detection probability for given conditions."""
        sensor_type = sensor.sensor_type.value if hasattr(sensor.sensor_type, 'value') else str(sensor.sensor_type)
        
        if sensor_type in self.detection_models:
            return self.detection_models[sensor_type](sensor, target_signature, range_km, environmental_factors)
        
        # Default simplified detection model
        base_detection_range = getattr(sensor, 'detection_range', 100.0)  # km
        rcs = target_signature.get('radar_cross_section', 1.0)  # m²
        weather_factor = environmental_factors.get('weather_degradation', 1.0)
        
        # Radar equation approximation
        detection_prob = min(1.0, (base_detection_range / range_km) ** 2 * 
                           np.sqrt(rcs) * weather_factor)
        
        return max(0.0, detection_prob)
    
    def calculate_engagement_timeline(self, engagement: EngagementEvent,
                                    aircraft_sensors: List[SensorSystem],
                                    weapon_systems: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate detailed engagement timeline."""
        timeline = {
            'detection_time': 0.0,
            'classification_time': 0.0,
            'targeting_time': 0.0,
            'weapon_release_time': 0.0,
            'time_of_flight': 0.0,
            'total_engagement_time': 0.0,
            'phases': []
        }
        
        # Detection phase
        detection_range = np.linalg.norm(engagement.aircraft_position - engagement.threat_position)
        detection_time = self._calculate_detection_time(aircraft_sensors, detection_range)
        timeline['detection_time'] = detection_time
        timeline['phases'].append({
            'phase': 'detection',
            'start_time': 0.0,
            'duration': detection_time,
            'success_probability': engagement.detection_probability
        })
        
        # Classification phase
        classification_time = self._calculate_classification_time(aircraft_sensors)
        timeline['classification_time'] = classification_time
        timeline['phases'].append({
            'phase': 'classification',
            'start_time': detection_time,
            'duration': classification_time,
            'success_probability': 0.9  # Simplified
        })
        
        # Targeting phase
        targeting_time = self._calculate_targeting_time(weapon_systems)
        timeline['targeting_time'] = targeting_time
        timeline['phases'].append({
            'phase': 'targeting',
            'start_time': detection_time + classification_time,
            'duration': targeting_time,
            'success_probability': 0.95  # Simplified
        })
        
        # Weapon release and time of flight
        weapon_release_time = detection_time + classification_time + targeting_time
        time_of_flight = self._calculate_time_of_flight(weapon_systems, detection_range)
        
        timeline['weapon_release_time'] = weapon_release_time
        timeline['time_of_flight'] = time_of_flight
        timeline['total_engagement_time'] = weapon_release_time + time_of_flight
        
        timeline['phases'].append({
            'phase': 'weapon_flight',
            'start_time': weapon_release_time,
            'duration': time_of_flight,
            'success_probability': engagement.engagement_probability
        })
        
        return timeline
    
    def _calculate_detection_time(self, sensors: List[SensorSystem], range_km: float) -> float:
        """Calculate time required for detection."""
        # Simplified model based on sensor scan rate and range
        base_scan_time = 2.0  # seconds
        range_factor = min(1.0, 100.0 / range_km)  # Closer targets detected faster
        return base_scan_time / range_factor
    
    def _calculate_classification_time(self, sensors: List[SensorSystem]) -> float:
        """Calculate time required for target classification."""
        # Simplified model
        return 1.5  # seconds
    
    def _calculate_targeting_time(self, weapon_systems: List[Dict[str, Any]]) -> float:
        """Calculate time required for weapon targeting."""
        # Simplified model
        return 2.0  # seconds
    
    def _calculate_time_of_flight(self, weapon_systems: List[Dict[str, Any]], range_km: float) -> float:
        """Calculate weapon time of flight."""
        if not weapon_systems:
            return 0.0
        
        # Use first weapon system for simplification
        weapon = weapon_systems[0]
        weapon_speed = weapon.get('speed', 1000.0)  # m/s
        
        return (range_km * 1000.0) / weapon_speed


class ThreatEnvironmentSimulator:
    """Simulates threat environment and countermeasures."""
    
    def __init__(self):
        """Initialize threat environment simulator."""
        self.logger = get_logger("threat_environment_simulator")
        self.threat_models = {}
        self.countermeasure_effectiveness = {}
    
    def add_threat_model(self, threat_type: ThreatType, model: Callable) -> None:
        """Add threat behavior model."""
        self.threat_models[threat_type] = model
        self.logger.info(f"Added threat model for {threat_type.value}")
    
    def set_countermeasure_effectiveness(self, countermeasure: str, 
                                       effectiveness: Dict[ThreatType, float]) -> None:
        """Set countermeasure effectiveness against different threats."""
        self.countermeasure_effectiveness[countermeasure] = effectiveness
        self.logger.info(f"Set effectiveness for countermeasure: {countermeasure}")
    
    def simulate_threat_behavior(self, threat: ThreatDefinition, 
                               aircraft_position: np.ndarray,
                               time: float) -> Dict[str, Any]:
        """Simulate threat behavior at given time."""
        if threat.threat_type in self.threat_models:
            return self.threat_models[threat.threat_type](threat, aircraft_position, time)
        
        # Default threat behavior
        distance = np.linalg.norm(aircraft_position - threat.position)
        
        return {
            'active': threat.active and distance <= threat.detection_range,
            'detection_probability': max(0.0, 1.0 - distance / threat.detection_range),
            'engagement_probability': max(0.0, 1.0 - distance / threat.engagement_range) if distance <= threat.engagement_range else 0.0,
            'threat_level': 'high' if distance <= threat.engagement_range else 'medium' if distance <= threat.detection_range else 'low'
        }
    
    def evaluate_countermeasures(self, threats: List[ThreatDefinition],
                               countermeasures: List[str]) -> Dict[str, float]:
        """Evaluate effectiveness of countermeasures against threats."""
        effectiveness = {}
        
        for threat in threats:
            threat_effectiveness = 1.0  # No countermeasures
            
            for countermeasure in countermeasures:
                if countermeasure in self.countermeasure_effectiveness:
                    cm_effectiveness = self.countermeasure_effectiveness[countermeasure].get(
                        threat.threat_type, 0.0)
                    threat_effectiveness *= (1.0 - cm_effectiveness)
            
            effectiveness[threat.threat_id] = 1.0 - threat_effectiveness
        
        return effectiveness


class MissionEffectivenessAnalyzer:
    """Analyzes mission effectiveness and optimization opportunities."""
    
    def __init__(self):
        """Initialize mission effectiveness analyzer."""
        self.logger = get_logger("mission_effectiveness_analyzer")
        self.metrics_weights = {
            'mission_success': 0.4,
            'survivability': 0.3,
            'resource_efficiency': 0.2,
            'timeline_performance': 0.1
        }
    
    def set_metrics_weights(self, weights: Dict[str, float]) -> None:
        """Set weights for different effectiveness metrics."""
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError("Metrics weights must sum to 1.0")
        
        self.metrics_weights = weights.copy()
        self.logger.info("Updated metrics weights")
    
    def analyze_mission_effectiveness(self, results: MissionResults) -> Dict[str, Any]:
        """Analyze overall mission effectiveness."""
        analysis = {
            'overall_score': 0.0,
            'component_scores': {},
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        # Mission success score
        success_score = 1.0 if results.mission_success else 0.0
        analysis['component_scores']['mission_success'] = success_score
        
        # Survivability score
        survivability_score = self._calculate_survivability_score(results)
        analysis['component_scores']['survivability'] = survivability_score
        
        # Resource efficiency score
        efficiency_score = self._calculate_efficiency_score(results)
        analysis['component_scores']['resource_efficiency'] = efficiency_score
        
        # Timeline performance score
        timeline_score = self._calculate_timeline_score(results)
        analysis['component_scores']['timeline_performance'] = timeline_score
        
        # Calculate overall score
        overall_score = (
            success_score * self.metrics_weights['mission_success'] +
            survivability_score * self.metrics_weights['survivability'] +
            efficiency_score * self.metrics_weights['resource_efficiency'] +
            timeline_score * self.metrics_weights['timeline_performance']
        )
        analysis['overall_score'] = overall_score
        
        # Generate insights
        analysis['strengths'] = self._identify_strengths(analysis['component_scores'])
        analysis['weaknesses'] = self._identify_weaknesses(analysis['component_scores'])
        analysis['recommendations'] = self._generate_recommendations(results, analysis)
        
        return analysis
    
    def _calculate_survivability_score(self, results: MissionResults) -> float:
        """Calculate survivability score based on threat encounters."""
        if not results.engagement_events:
            return 1.0  # No threats encountered
        
        successful_survivals = sum(1 for event in results.engagement_events 
                                 if event.outcome in [EngagementOutcome.SUCCESS, None])
        total_engagements = len(results.engagement_events)
        
        return successful_survivals / total_engagements
    
    def _calculate_efficiency_score(self, results: MissionResults) -> float:
        """Calculate resource efficiency score."""
        # Simplified efficiency calculation
        fuel_efficiency = results.resource_consumption.get('fuel_efficiency', 0.8)
        time_efficiency = min(1.0, 3600.0 / results.total_duration)  # Prefer shorter missions
        
        return (fuel_efficiency + time_efficiency) / 2.0
    
    def _calculate_timeline_score(self, results: MissionResults) -> float:
        """Calculate timeline performance score."""
        # Score based on phases completed and engagement response times
        phases_score = len(results.phases_completed) / len(MissionPhase)
        
        # Average engagement response time score
        if results.engagement_events:
            avg_response_time = np.mean([10.0 for _ in results.engagement_events])  # Simplified
            response_score = max(0.0, 1.0 - avg_response_time / 30.0)  # 30s is poor response
        else:
            response_score = 1.0
        
        return (phases_score + response_score) / 2.0
    
    def _identify_strengths(self, scores: Dict[str, float]) -> List[str]:
        """Identify mission strengths based on scores."""
        strengths = []
        for metric, score in scores.items():
            if score >= 0.8:
                strengths.append(f"Excellent {metric.replace('_', ' ')}")
        return strengths
    
    def _identify_weaknesses(self, scores: Dict[str, float]) -> List[str]:
        """Identify mission weaknesses based on scores."""
        weaknesses = []
        for metric, score in scores.items():
            if score < 0.6:
                weaknesses.append(f"Poor {metric.replace('_', ' ')}")
        return weaknesses
    
    def _generate_recommendations(self, results: MissionResults, 
                                analysis: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        if analysis['component_scores']['survivability'] < 0.7:
            recommendations.append("Consider additional countermeasures or route optimization")
        
        if analysis['component_scores']['resource_efficiency'] < 0.7:
            recommendations.append("Optimize flight profile for better fuel efficiency")
        
        if analysis['component_scores']['timeline_performance'] < 0.7:
            recommendations.append("Improve sensor-to-shooter timeline through training or equipment upgrades")
        
        if not results.mission_success:
            recommendations.append("Review mission planning and threat assessment procedures")
        
        return recommendations


class MissionScenarioSimulator(BaseEngine):
    """Main mission scenario simulation engine."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize mission scenario simulator."""
        super().__init__(config)
        self.physics_orchestrator = MultiPhysicsOrchestrator()
        self.sensor_analyzer = SensorToShooterAnalyzer()
        self.threat_simulator = ThreatEnvironmentSimulator()
        self.effectiveness_analyzer = MissionEffectivenessAnalyzer()
        self.logger = get_logger("mission_scenario_simulator")
        
        # Initialize default threat and countermeasure models
        self._initialize_default_models()
    
    def initialize(self) -> bool:
        """Initialize the mission simulator."""
        try:
            success = self.physics_orchestrator.initialize()
            if not success:
                return False
            
            self.initialized = True
            self.logger.info("Mission scenario simulator initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    def validate_input(self, data: Any) -> bool:
        """Validate mission scenario input."""
        if isinstance(data, MissionScenario):
            return self._validate_mission_scenario(data)
        return False
    
    def process(self, data: Any) -> Any:
        """Process mission simulation."""
        if isinstance(data, dict) and 'scenario' in data:
            scenario = data['scenario']
            config = data.get('aircraft_config')
            return self.simulate_mission(scenario, config)
        return None
    
    def _validate_mission_scenario(self, scenario: MissionScenario) -> bool:
        """Validate mission scenario completeness."""
        if not scenario.waypoints:
            self.logger.error("Mission scenario must have waypoints")
            return False
        
        if not scenario.success_criteria:
            self.logger.warning("No success criteria defined")
        
        return True
    
    def simulate_mission(self, scenario: MissionScenario, 
                        aircraft_config: AircraftConfiguration) -> MissionResults:
        """Simulate complete mission scenario."""
        try:
            self.logger.info(f"Starting mission simulation: {scenario.scenario_id}")
            
            # Initialize results
            results = MissionResults(
                scenario_id=scenario.scenario_id,
                mission_success=False,
                total_duration=0.0,
                phases_completed=[],
                engagement_events=[],
                sensor_timeline={},
                effectiveness_metrics={},
                resource_consumption={},
                survivability_assessment={}
            )
            
            # Simulate mission phases
            current_time = 0.0
            current_position = scenario.waypoints[0].position if scenario.waypoints else np.zeros(3)
            
            for waypoint in scenario.waypoints:
                phase_results = self._simulate_mission_phase(
                    waypoint, scenario.threats, aircraft_config, current_time)
                
                # Update results
                results.phases_completed.append(waypoint.phase)
                results.engagement_events.extend(phase_results['engagements'])
                current_time += phase_results['duration']
                current_position = waypoint.position
                
                # Check for mission failure conditions
                if self._check_failure_conditions(scenario.failure_conditions, phase_results):
                    self.logger.warning("Mission failure condition met")
                    break
            
            results.total_duration = current_time
            
            # Evaluate mission success
            results.mission_success = self._evaluate_mission_success(
                scenario.success_criteria, results)
            
            # Calculate effectiveness metrics
            effectiveness_analysis = self.effectiveness_analyzer.analyze_mission_effectiveness(results)
            results.effectiveness_metrics = effectiveness_analysis
            
            # Generate resource consumption estimates
            results.resource_consumption = self._calculate_resource_consumption(
                aircraft_config, results.total_duration, results.engagement_events)
            
            # Assess survivability
            results.survivability_assessment = self._assess_survivability(
                results.engagement_events, scenario.threats)
            
            self.logger.info(f"Mission simulation completed: {scenario.scenario_id}")
            return results
            
        except Exception as e:
            self.logger.error(f"Mission simulation failed: {e}")
            raise SimulationError(f"Mission simulation failed: {e}")
    
    def _simulate_mission_phase(self, waypoint: MissionWaypoint, 
                              threats: List[ThreatDefinition],
                              aircraft_config: AircraftConfiguration,
                              start_time: float) -> Dict[str, Any]:
        """Simulate a single mission phase."""
        phase_results = {
            'duration': waypoint.duration,
            'engagements': [],
            'sensor_data': {},
            'countermeasures_used': []
        }
        
        # Check for threat encounters during this phase
        for threat in threats:
            if not threat.active:
                continue
            
            distance = np.linalg.norm(waypoint.position - threat.position)
            
            if distance <= threat.detection_range:
                # Create engagement event
                engagement = self._create_engagement_event(
                    waypoint, threat, start_time, aircraft_config)
                
                if engagement:
                    phase_results['engagements'].append(engagement)
        
        return phase_results
    
    def _create_engagement_event(self, waypoint: MissionWaypoint,
                               threat: ThreatDefinition,
                               time: float,
                               aircraft_config: AircraftConfiguration) -> Optional[EngagementEvent]:
        """Create engagement event for threat encounter."""
        distance = np.linalg.norm(waypoint.position - threat.position)
        
        # Calculate detection probability
        environmental_factors = {'weather_degradation': 1.0}  # Simplified
        target_signature = {
            'radar_cross_section': threat.radar_cross_section,
            'infrared_signature': 1000.0  # Simplified
        }
        
        # Use first sensor for simplification
        aircraft_sensors = getattr(aircraft_config, 'sensors', [])
        if not aircraft_sensors:
            # Create default sensor for simulation
            aircraft_sensors = [type('DefaultSensor', (), {
                'sensor_type': 'radar',
                'detection_range': 150.0
            })()]
        
        detection_prob = self.sensor_analyzer.calculate_detection_probability(
            aircraft_sensors[0], target_signature, distance / 1000.0, environmental_factors)
        
        # Calculate engagement probability
        engagement_prob = 0.8 if distance <= threat.engagement_range else 0.0
        
        # Determine outcome
        outcome = None
        if detection_prob > 0.5 and engagement_prob > 0.5:
            outcome = EngagementOutcome.SUCCESS
        elif detection_prob > 0.5:
            outcome = EngagementOutcome.PARTIAL_SUCCESS
        else:
            outcome = EngagementOutcome.FAILURE
        
        return EngagementEvent(
            event_id=f"engagement_{threat.threat_id}_{int(time)}",
            time=time,
            phase=waypoint.phase,
            threat_id=threat.threat_id,
            aircraft_position=waypoint.position,
            threat_position=threat.position,
            detection_probability=detection_prob,
            engagement_probability=engagement_prob,
            outcome=outcome
        )
    
    def _check_failure_conditions(self, failure_conditions: Dict[str, Any],
                                phase_results: Dict[str, Any]) -> bool:
        """Check if any failure conditions are met."""
        # Simplified failure condition checking
        if 'max_threat_encounters' in failure_conditions:
            max_encounters = failure_conditions['max_threat_encounters']
            if len(phase_results['engagements']) > max_encounters:
                return True
        
        return False
    
    def _evaluate_mission_success(self, success_criteria: Dict[str, Any],
                                results: MissionResults) -> bool:
        """Evaluate if mission meets success criteria."""
        # Simplified success evaluation
        if 'min_phases_completed' in success_criteria:
            min_phases = success_criteria['min_phases_completed']
            if len(results.phases_completed) < min_phases:
                return False
        
        if 'max_engagement_failures' in success_criteria:
            max_failures = success_criteria['max_engagement_failures']
            failures = sum(1 for event in results.engagement_events 
                         if event.outcome == EngagementOutcome.FAILURE)
            if failures > max_failures:
                return False
        
        return True
    
    def _calculate_resource_consumption(self, aircraft_config: AircraftConfiguration,
                                      duration: float,
                                      engagements: List[EngagementEvent]) -> Dict[str, float]:
        """Calculate resource consumption during mission."""
        # Simplified resource calculation
        base_fuel_rate = 0.5  # kg/s
        engagement_fuel_penalty = 10.0  # kg per engagement
        
        fuel_consumed = base_fuel_rate * duration + len(engagements) * engagement_fuel_penalty
        
        return {
            'fuel_consumed_kg': fuel_consumed,
            'fuel_efficiency': max(0.0, 1.0 - fuel_consumed / 10000.0),  # Simplified
            'ammunition_used': len(engagements),
            'flight_hours': duration / 3600.0
        }
    
    def _assess_survivability(self, engagements: List[EngagementEvent],
                            threats: List[ThreatDefinition]) -> Dict[str, Any]:
        """Assess mission survivability."""
        if not engagements:
            return {'survivability_score': 1.0, 'risk_level': 'low'}
        
        successful_engagements = sum(1 for event in engagements 
                                   if event.outcome == EngagementOutcome.SUCCESS)
        survivability_score = successful_engagements / len(engagements)
        
        risk_level = 'low'
        if survivability_score < 0.5:
            risk_level = 'high'
        elif survivability_score < 0.8:
            risk_level = 'medium'
        
        return {
            'survivability_score': survivability_score,
            'risk_level': risk_level,
            'total_threats_encountered': len(engagements),
            'successful_engagements': successful_engagements
        }
    
    def _initialize_default_models(self) -> None:
        """Initialize default threat and sensor models."""
        # Default SAM threat model
        def sam_threat_model(threat: ThreatDefinition, aircraft_pos: np.ndarray, time: float) -> Dict[str, Any]:
            distance = np.linalg.norm(aircraft_pos - threat.position)
            return {
                'active': distance <= threat.detection_range,
                'detection_probability': max(0.0, 1.0 - distance / threat.detection_range),
                'engagement_probability': max(0.0, 1.0 - distance / threat.engagement_range) if distance <= threat.engagement_range else 0.0,
                'threat_level': 'high' if distance <= threat.engagement_range else 'medium'
            }
        
        self.threat_simulator.add_threat_model(ThreatType.SURFACE_TO_AIR, sam_threat_model)
        
        # Default countermeasure effectiveness
        self.threat_simulator.set_countermeasure_effectiveness('chaff', {
            ThreatType.SURFACE_TO_AIR: 0.6,
            ThreatType.AIR_TO_AIR: 0.4
        })
        
        self.threat_simulator.set_countermeasure_effectiveness('flare', {
            ThreatType.SURFACE_TO_AIR: 0.7,
            ThreatType.AIR_TO_AIR: 0.8
        })