"""Tests for mission scenario simulation system."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from fighter_jet_sdk.core.mission_simulation import (
    MissionScenarioSimulator,
    SensorToShooterAnalyzer,
    ThreatEnvironmentSimulator,
    MissionEffectivenessAnalyzer,
    MissionScenario,
    MissionWaypoint,
    ThreatDefinition,
    EngagementEvent,
    MissionResults,
    MissionPhase,
    ThreatType,
    EngagementOutcome
)
from fighter_jet_sdk.common.data_models import AircraftConfiguration, SensorSystem
from fighter_jet_sdk.core.errors import SimulationError


class TestSensorToShooterAnalyzer:
    """Test sensor-to-shooter analyzer."""
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = SensorToShooterAnalyzer()
        assert len(analyzer.detection_models) == 0
        assert len(analyzer.engagement_models) == 0
    
    def test_add_detection_model(self):
        """Test adding detection model."""
        analyzer = SensorToShooterAnalyzer()
        
        def dummy_model(sensor, signature, range_km, env_factors):
            return 0.8
        
        analyzer.add_detection_model('radar', dummy_model)
        assert 'radar' in analyzer.detection_models
        assert analyzer.detection_models['radar'] == dummy_model
    
    def test_add_engagement_model(self):
        """Test adding engagement model."""
        analyzer = SensorToShooterAnalyzer()
        
        def dummy_model(weapon, target, range_km):
            return 0.9
        
        analyzer.add_engagement_model('missile', dummy_model)
        assert 'missile' in analyzer.engagement_models
        assert analyzer.engagement_models['missile'] == dummy_model
    
    def test_calculate_detection_probability_default(self):
        """Test default detection probability calculation."""
        analyzer = SensorToShooterAnalyzer()
        
        # Mock sensor
        sensor = Mock()
        sensor.sensor_type = Mock()
        sensor.sensor_type.value = 'unknown_sensor'
        sensor.detection_range = 100.0
        
        target_signature = {'radar_cross_section': 1.0}
        environmental_factors = {'weather_degradation': 1.0}
        
        prob = analyzer.calculate_detection_probability(
            sensor, target_signature, 50.0, environmental_factors)
        
        assert 0.0 <= prob <= 1.0
        assert prob > 0.0  # Should detect at 50km range
    
    def test_calculate_detection_probability_custom_model(self):
        """Test detection probability with custom model."""
        analyzer = SensorToShooterAnalyzer()
        
        def custom_model(sensor, signature, range_km, env_factors):
            return 0.75
        
        analyzer.add_detection_model('radar', custom_model)
        
        sensor = Mock()
        sensor.sensor_type = Mock()
        sensor.sensor_type.value = 'radar'
        
        prob = analyzer.calculate_detection_probability(
            sensor, {}, 50.0, {})
        
        assert prob == 0.75
    
    def test_calculate_engagement_timeline(self):
        """Test engagement timeline calculation."""
        analyzer = SensorToShooterAnalyzer()
        
        engagement = EngagementEvent(
            event_id='test_engagement',
            time=100.0,
            phase=MissionPhase.ENGAGEMENT,
            threat_id='threat_1',
            aircraft_position=np.array([0, 0, 1000]),
            threat_position=np.array([50000, 0, 0]),
            detection_probability=0.8,
            engagement_probability=0.9
        )
        
        sensors = [Mock()]
        weapons = [{'speed': 1000.0, 'type': 'missile'}]
        
        timeline = analyzer.calculate_engagement_timeline(engagement, sensors, weapons)
        
        assert 'detection_time' in timeline
        assert 'classification_time' in timeline
        assert 'targeting_time' in timeline
        assert 'weapon_release_time' in timeline
        assert 'time_of_flight' in timeline
        assert 'total_engagement_time' in timeline
        assert 'phases' in timeline
        
        assert len(timeline['phases']) == 4  # detection, classification, targeting, weapon_flight
        assert timeline['total_engagement_time'] > 0


class TestThreatEnvironmentSimulator:
    """Test threat environment simulator."""
    
    def test_simulator_initialization(self):
        """Test simulator initialization."""
        simulator = ThreatEnvironmentSimulator()
        assert len(simulator.threat_models) == 0
        assert len(simulator.countermeasure_effectiveness) == 0
    
    def test_add_threat_model(self):
        """Test adding threat model."""
        simulator = ThreatEnvironmentSimulator()
        
        def dummy_model(threat, aircraft_pos, time):
            return {'active': True}
        
        simulator.add_threat_model(ThreatType.SURFACE_TO_AIR, dummy_model)
        assert ThreatType.SURFACE_TO_AIR in simulator.threat_models
    
    def test_set_countermeasure_effectiveness(self):
        """Test setting countermeasure effectiveness."""
        simulator = ThreatEnvironmentSimulator()
        
        effectiveness = {
            ThreatType.SURFACE_TO_AIR: 0.7,
            ThreatType.AIR_TO_AIR: 0.5
        }
        
        simulator.set_countermeasure_effectiveness('chaff', effectiveness)
        assert 'chaff' in simulator.countermeasure_effectiveness
        assert simulator.countermeasure_effectiveness['chaff'] == effectiveness
    
    def test_simulate_threat_behavior_default(self):
        """Test default threat behavior simulation."""
        simulator = ThreatEnvironmentSimulator()
        
        threat = ThreatDefinition(
            threat_id='sam_1',
            threat_type=ThreatType.SURFACE_TO_AIR,
            position=np.array([10000, 0, 0]),
            detection_range=50000.0,
            engagement_range=30000.0,
            radar_cross_section=1.0,
            electronic_signature={}
        )
        
        aircraft_position = np.array([20000, 0, 1000])
        
        behavior = simulator.simulate_threat_behavior(threat, aircraft_position, 100.0)
        
        assert 'active' in behavior
        assert 'detection_probability' in behavior
        assert 'engagement_probability' in behavior
        assert 'threat_level' in behavior
        
        assert behavior['active']  # Within detection range
        assert behavior['detection_probability'] > 0
    
    def test_simulate_threat_behavior_custom_model(self):
        """Test threat behavior with custom model."""
        simulator = ThreatEnvironmentSimulator()
        
        def custom_model(threat, aircraft_pos, time):
            return {
                'active': True,
                'detection_probability': 0.95,
                'engagement_probability': 0.8,
                'threat_level': 'critical'
            }
        
        simulator.add_threat_model(ThreatType.SURFACE_TO_AIR, custom_model)
        
        threat = ThreatDefinition(
            threat_id='sam_1',
            threat_type=ThreatType.SURFACE_TO_AIR,
            position=np.array([0, 0, 0]),
            detection_range=50000.0,
            engagement_range=30000.0,
            radar_cross_section=1.0,
            electronic_signature={}
        )
        
        behavior = simulator.simulate_threat_behavior(threat, np.array([0, 0, 0]), 0.0)
        
        assert behavior['detection_probability'] == 0.95
        assert behavior['threat_level'] == 'critical'
    
    def test_evaluate_countermeasures(self):
        """Test countermeasure effectiveness evaluation."""
        simulator = ThreatEnvironmentSimulator()
        
        # Set up countermeasure effectiveness
        simulator.set_countermeasure_effectiveness('chaff', {
            ThreatType.SURFACE_TO_AIR: 0.6
        })
        simulator.set_countermeasure_effectiveness('flare', {
            ThreatType.SURFACE_TO_AIR: 0.4
        })
        
        threats = [
            ThreatDefinition(
                threat_id='sam_1',
                threat_type=ThreatType.SURFACE_TO_AIR,
                position=np.array([0, 0, 0]),
                detection_range=50000.0,
                engagement_range=30000.0,
                radar_cross_section=1.0,
                electronic_signature={}
            )
        ]
        
        countermeasures = ['chaff', 'flare']
        
        effectiveness = simulator.evaluate_countermeasures(threats, countermeasures)
        
        assert 'sam_1' in effectiveness
        # Combined effectiveness should be higher than individual
        assert effectiveness['sam_1'] > 0.6


class TestMissionEffectivenessAnalyzer:
    """Test mission effectiveness analyzer."""
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = MissionEffectivenessAnalyzer()
        assert 'mission_success' in analyzer.metrics_weights
        assert 'survivability' in analyzer.metrics_weights
        assert abs(sum(analyzer.metrics_weights.values()) - 1.0) < 1e-6
    
    def test_set_metrics_weights(self):
        """Test setting custom metrics weights."""
        analyzer = MissionEffectivenessAnalyzer()
        
        new_weights = {
            'mission_success': 0.5,
            'survivability': 0.3,
            'resource_efficiency': 0.1,
            'timeline_performance': 0.1
        }
        
        analyzer.set_metrics_weights(new_weights)
        assert analyzer.metrics_weights == new_weights
    
    def test_set_metrics_weights_invalid(self):
        """Test setting invalid metrics weights."""
        analyzer = MissionEffectivenessAnalyzer()
        
        invalid_weights = {
            'mission_success': 0.5,
            'survivability': 0.3,
            'resource_efficiency': 0.1,
            'timeline_performance': 0.2  # Sum > 1.0
        }
        
        with pytest.raises(ValueError, match="must sum to 1.0"):
            analyzer.set_metrics_weights(invalid_weights)
    
    def test_analyze_mission_effectiveness_success(self):
        """Test mission effectiveness analysis for successful mission."""
        analyzer = MissionEffectivenessAnalyzer()
        
        results = MissionResults(
            scenario_id='test_mission',
            mission_success=True,
            total_duration=1800.0,  # 30 minutes
            phases_completed=[MissionPhase.TAKEOFF, MissionPhase.TRANSIT, MissionPhase.ENGAGEMENT],
            engagement_events=[
                EngagementEvent(
                    event_id='eng_1',
                    time=1000.0,
                    phase=MissionPhase.ENGAGEMENT,
                    threat_id='threat_1',
                    aircraft_position=np.array([0, 0, 0]),
                    threat_position=np.array([1000, 0, 0]),
                    detection_probability=0.9,
                    engagement_probability=0.8,
                    outcome=EngagementOutcome.SUCCESS
                )
            ],
            sensor_timeline={},
            effectiveness_metrics={},
            resource_consumption={'fuel_efficiency': 0.8},
            survivability_assessment={}
        )
        
        analysis = analyzer.analyze_mission_effectiveness(results)
        
        assert 'overall_score' in analysis
        assert 'component_scores' in analysis
        assert 'strengths' in analysis
        assert 'weaknesses' in analysis
        assert 'recommendations' in analysis
        
        assert analysis['overall_score'] > 0.5  # Should be reasonably high for successful mission
        assert analysis['component_scores']['mission_success'] == 1.0
    
    def test_analyze_mission_effectiveness_failure(self):
        """Test mission effectiveness analysis for failed mission."""
        analyzer = MissionEffectivenessAnalyzer()
        
        results = MissionResults(
            scenario_id='test_mission',
            mission_success=False,
            total_duration=3600.0,
            phases_completed=[MissionPhase.TAKEOFF],
            engagement_events=[
                EngagementEvent(
                    event_id='eng_1',
                    time=1000.0,
                    phase=MissionPhase.ENGAGEMENT,
                    threat_id='threat_1',
                    aircraft_position=np.array([0, 0, 0]),
                    threat_position=np.array([1000, 0, 0]),
                    detection_probability=0.3,
                    engagement_probability=0.2,
                    outcome=EngagementOutcome.FAILURE
                )
            ],
            sensor_timeline={},
            effectiveness_metrics={},
            resource_consumption={'fuel_efficiency': 0.3},
            survivability_assessment={}
        )
        
        analysis = analyzer.analyze_mission_effectiveness(results)
        
        assert analysis['component_scores']['mission_success'] == 0.0
        assert len(analysis['weaknesses']) > 0
        assert len(analysis['recommendations']) > 0


class TestMissionScenarioSimulator:
    """Test mission scenario simulator."""
    
    def test_simulator_initialization(self):
        """Test simulator initialization."""
        simulator = MissionScenarioSimulator()
        assert simulator.physics_orchestrator is not None
        assert simulator.sensor_analyzer is not None
        assert simulator.threat_simulator is not None
        assert simulator.effectiveness_analyzer is not None
    
    def test_simulator_initialize(self):
        """Test simulator initialization."""
        simulator = MissionScenarioSimulator()
        
        # Mock the physics orchestrator initialization
        simulator.physics_orchestrator.initialize = Mock(return_value=True)
        
        success = simulator.initialize()
        assert success
        assert simulator.initialized
    
    def test_validate_input_valid_scenario(self):
        """Test input validation with valid scenario."""
        simulator = MissionScenarioSimulator()
        
        scenario = MissionScenario(
            scenario_id='test_scenario',
            description='Test mission',
            waypoints=[
                MissionWaypoint(
                    waypoint_id='wp1',
                    position=np.array([0, 0, 1000]),
                    altitude=1000.0,
                    speed=200.0,
                    heading=0.0,
                    phase=MissionPhase.TAKEOFF
                )
            ],
            threats=[],
            environmental_conditions={},
            success_criteria={'min_phases_completed': 1},
            failure_conditions={}
        )
        
        assert simulator.validate_input(scenario)
    
    def test_validate_input_invalid_scenario(self):
        """Test input validation with invalid scenario."""
        simulator = MissionScenarioSimulator()
        
        # Scenario without waypoints
        scenario = MissionScenario(
            scenario_id='test_scenario',
            description='Test mission',
            waypoints=[],  # Empty waypoints
            threats=[],
            environmental_conditions={},
            success_criteria={},
            failure_conditions={}
        )
        
        assert not simulator.validate_input(scenario)
    
    def test_simulate_mission_simple(self):
        """Test simple mission simulation."""
        simulator = MissionScenarioSimulator()
        
        # Mock initialization
        simulator.physics_orchestrator.initialize = Mock(return_value=True)
        simulator.initialize()
        
        # Create simple scenario
        scenario = MissionScenario(
            scenario_id='simple_mission',
            description='Simple test mission',
            waypoints=[
                MissionWaypoint(
                    waypoint_id='takeoff',
                    position=np.array([0, 0, 1000]),
                    altitude=1000.0,
                    speed=200.0,
                    heading=0.0,
                    phase=MissionPhase.TAKEOFF,
                    duration=300.0
                ),
                MissionWaypoint(
                    waypoint_id='target',
                    position=np.array([50000, 0, 5000]),
                    altitude=5000.0,
                    speed=300.0,
                    heading=0.0,
                    phase=MissionPhase.ENGAGEMENT,
                    duration=600.0
                )
            ],
            threats=[
                ThreatDefinition(
                    threat_id='sam_site',
                    threat_type=ThreatType.SURFACE_TO_AIR,
                    position=np.array([40000, 0, 0]),
                    detection_range=60000.0,
                    engagement_range=40000.0,
                    radar_cross_section=1.0,
                    electronic_signature={}
                )
            ],
            environmental_conditions={'weather': 'clear'},
            success_criteria={'min_phases_completed': 2},
            failure_conditions={'max_threat_encounters': 5}
        )
        
        aircraft_config = AircraftConfiguration()
        
        results = simulator.simulate_mission(scenario, aircraft_config)
        
        assert results.scenario_id == 'simple_mission'
        assert results.total_duration > 0
        assert len(results.phases_completed) > 0
        assert 'effectiveness_metrics' in results.__dict__
        assert 'resource_consumption' in results.__dict__
        assert 'survivability_assessment' in results.__dict__
    
    def test_simulate_mission_with_threats(self):
        """Test mission simulation with threat encounters."""
        simulator = MissionScenarioSimulator()
        
        # Mock initialization
        simulator.physics_orchestrator.initialize = Mock(return_value=True)
        simulator.initialize()
        
        # Create scenario with threats
        scenario = MissionScenario(
            scenario_id='threat_mission',
            description='Mission with threats',
            waypoints=[
                MissionWaypoint(
                    waypoint_id='engagement_zone',
                    position=np.array([30000, 0, 5000]),  # Close to threat
                    altitude=5000.0,
                    speed=300.0,
                    heading=0.0,
                    phase=MissionPhase.ENGAGEMENT,
                    duration=300.0
                )
            ],
            threats=[
                ThreatDefinition(
                    threat_id='close_sam',
                    threat_type=ThreatType.SURFACE_TO_AIR,
                    position=np.array([25000, 0, 0]),  # Close to waypoint
                    detection_range=50000.0,
                    engagement_range=35000.0,
                    radar_cross_section=1.0,
                    electronic_signature={}
                )
            ],
            environmental_conditions={},
            success_criteria={'min_phases_completed': 1},
            failure_conditions={}
        )
        
        aircraft_config = AircraftConfiguration()
        
        results = simulator.simulate_mission(scenario, aircraft_config)
        
        # Should have engagement events due to close threat
        assert len(results.engagement_events) > 0
        assert results.engagement_events[0].threat_id == 'close_sam'
    
    def test_create_engagement_event(self):
        """Test engagement event creation."""
        simulator = MissionScenarioSimulator()
        
        waypoint = MissionWaypoint(
            waypoint_id='test_wp',
            position=np.array([10000, 0, 5000]),
            altitude=5000.0,
            speed=300.0,
            heading=0.0,
            phase=MissionPhase.ENGAGEMENT
        )
        
        threat = ThreatDefinition(
            threat_id='test_threat',
            threat_type=ThreatType.SURFACE_TO_AIR,
            position=np.array([15000, 0, 0]),
            detection_range=20000.0,
            engagement_range=15000.0,
            radar_cross_section=1.0,
            electronic_signature={}
        )
        
        aircraft_config = AircraftConfiguration()
        
        engagement = simulator._create_engagement_event(waypoint, threat, 100.0, aircraft_config)
        
        assert engagement is not None
        assert engagement.threat_id == 'test_threat'
        assert engagement.phase == MissionPhase.ENGAGEMENT
        assert engagement.detection_probability >= 0.0
        assert engagement.engagement_probability >= 0.0
        assert engagement.outcome is not None


class TestMissionDataStructures:
    """Test mission data structures."""
    
    def test_mission_waypoint_creation(self):
        """Test mission waypoint creation."""
        waypoint = MissionWaypoint(
            waypoint_id='wp1',
            position=np.array([1000, 2000, 3000]),
            altitude=3000.0,
            speed=250.0,
            heading=1.57,  # 90 degrees
            phase=MissionPhase.TRANSIT
        )
        
        assert waypoint.waypoint_id == 'wp1'
        assert np.array_equal(waypoint.position, np.array([1000, 2000, 3000]))
        assert waypoint.altitude == 3000.0
        assert waypoint.phase == MissionPhase.TRANSIT
    
    def test_threat_definition_creation(self):
        """Test threat definition creation."""
        threat = ThreatDefinition(
            threat_id='sam_1',
            threat_type=ThreatType.SURFACE_TO_AIR,
            position=np.array([5000, 0, 0]),
            detection_range=50000.0,
            engagement_range=30000.0,
            radar_cross_section=2.0,
            electronic_signature={'frequency': 10e9}
        )
        
        assert threat.threat_id == 'sam_1'
        assert threat.threat_type == ThreatType.SURFACE_TO_AIR
        assert threat.detection_range == 50000.0
        assert threat.active  # Default should be True
    
    def test_engagement_event_creation(self):
        """Test engagement event creation."""
        event = EngagementEvent(
            event_id='eng_1',
            time=150.0,
            phase=MissionPhase.ENGAGEMENT,
            threat_id='threat_1',
            aircraft_position=np.array([1000, 0, 2000]),
            threat_position=np.array([5000, 0, 0]),
            detection_probability=0.8,
            engagement_probability=0.6
        )
        
        assert event.event_id == 'eng_1'
        assert event.time == 150.0
        assert event.phase == MissionPhase.ENGAGEMENT
        assert event.detection_probability == 0.8
        assert event.outcome is None  # Default
    
    def test_mission_scenario_creation(self):
        """Test mission scenario creation."""
        waypoints = [
            MissionWaypoint(
                waypoint_id='start',
                position=np.array([0, 0, 1000]),
                altitude=1000.0,
                speed=200.0,
                heading=0.0,
                phase=MissionPhase.TAKEOFF
            )
        ]
        
        threats = [
            ThreatDefinition(
                threat_id='threat_1',
                threat_type=ThreatType.AIR_TO_AIR,
                position=np.array([10000, 0, 5000]),
                detection_range=30000.0,
                engagement_range=20000.0,
                radar_cross_section=1.5,
                electronic_signature={}
            )
        ]
        
        scenario = MissionScenario(
            scenario_id='test_scenario',
            description='Test scenario',
            waypoints=waypoints,
            threats=threats,
            environmental_conditions={'visibility': 'good'},
            success_criteria={'target_destroyed': True},
            failure_conditions={'aircraft_lost': True}
        )
        
        assert scenario.scenario_id == 'test_scenario'
        assert len(scenario.waypoints) == 1
        assert len(scenario.threats) == 1
        assert scenario.duration == 3600.0  # Default


if __name__ == '__main__':
    pytest.main([__file__])