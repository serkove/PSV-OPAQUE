"""
Tests for Decoy System Simulation

This module tests the decoy system simulation capabilities including
signature modeling, deployment sequencing, and effectiveness assessment.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from fighter_jet_sdk.engines.deployable.decoy_system import (
    DecoySystemSimulator, DecoySpecifications, ThreatSystem, DecoyState,
    SignatureModeler, DeploymentSequencer, EffectivenessAssessor,
    DecoyType, ThreatType
)
from fighter_jet_sdk.common.data_models import Position3D, Velocity3D


class TestDecoySpecifications:
    """Test decoy specifications data structure"""
    
    def test_decoy_specifications_creation(self):
        """Test creating decoy specifications"""
        specs = DecoySpecifications(
            decoy_type=DecoyType.CHAFF,
            radar_cross_section=10.0,
            infrared_signature=0.1,
            visual_signature=0.5,
            deployment_time=1.0,
            effective_duration=30.0,
            deployment_velocity=50.0,
            mass=0.5,
            cost=100.0
        )
        
        assert specs.decoy_type == DecoyType.CHAFF
        assert specs.radar_cross_section == 10.0
        assert specs.effective_duration == 30.0


class TestThreatSystem:
    """Test threat system data structure"""
    
    def test_threat_system_creation(self):
        """Test creating threat system"""
        threat = ThreatSystem(
            threat_id="SAM_001",
            threat_type=ThreatType.RADAR_GUIDED_MISSILE,
            detection_range=50.0,
            tracking_accuracy=1.0,
            seeker_frequency=10e9,
            countermeasure_resistance=0.3
        )
        
        assert threat.threat_id == "SAM_001"
        assert threat.threat_type == ThreatType.RADAR_GUIDED_MISSILE
        assert threat.seeker_frequency == 10e9


class TestSignatureModeler:
    """Test signature modeling capabilities"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.signature_modeler = SignatureModeler()
        self.decoy_state = DecoyState(
            decoy_id="test_decoy",
            decoy_type=DecoyType.CHAFF,
            position=Position3D(1000, 0, 500),
            velocity=Velocity3D(10, 0, 0),
            deployment_time=0.0,
            remaining_effectiveness=1.0
        )
        self.observer_position = Position3D(0, 0, 500)
    
    def test_radar_signature_calculation(self):
        """Test radar signature calculation"""
        frequency = 10e9  # 10 GHz
        
        signature = self.signature_modeler.calculate_radar_signature(
            self.decoy_state, self.observer_position, frequency
        )
        
        assert signature > 0
        assert isinstance(signature, float)
    
    def test_infrared_signature_calculation(self):
        """Test infrared signature calculation"""
        signature = self.signature_modeler.calculate_infrared_signature(
            self.decoy_state, self.observer_position
        )
        
        assert signature >= 0
        assert isinstance(signature, float)
    
    def test_visual_signature_calculation(self):
        """Test visual signature calculation"""
        signature = self.signature_modeler.calculate_visual_signature(
            self.decoy_state, self.observer_position, "daylight"
        )
        
        assert signature >= 0
        assert isinstance(signature, float)
    
    def test_distance_attenuation(self):
        """Test that signatures decrease with distance"""
        close_observer = Position3D(100, 0, 500)
        far_observer = Position3D(10000, 0, 500)
        
        close_signature = self.signature_modeler.calculate_radar_signature(
            self.decoy_state, close_observer, 10e9
        )
        far_signature = self.signature_modeler.calculate_radar_signature(
            self.decoy_state, far_observer, 10e9
        )
        
        assert close_signature > far_signature
    
    def test_time_degradation_effect(self):
        """Test that signatures decrease with decoy age"""
        fresh_decoy = DecoyState(
            decoy_id="fresh", decoy_type=DecoyType.CHAFF,
            position=Position3D(1000, 0, 500), velocity=Velocity3D(0, 0, 0),
            deployment_time=0.0, remaining_effectiveness=1.0
        )
        
        aged_decoy = DecoyState(
            decoy_id="aged", decoy_type=DecoyType.CHAFF,
            position=Position3D(1000, 0, 500), velocity=Velocity3D(0, 0, 0),
            deployment_time=0.0, remaining_effectiveness=0.3
        )
        
        fresh_signature = self.signature_modeler.calculate_radar_signature(
            fresh_decoy, self.observer_position, 10e9
        )
        aged_signature = self.signature_modeler.calculate_radar_signature(
            aged_decoy, self.observer_position, 10e9
        )
        
        assert fresh_signature > aged_signature
    
    def test_frequency_dependent_effects(self):
        """Test frequency-dependent effectiveness for chaff"""
        optimal_freq = 10e9
        suboptimal_freq = 20e9
        
        optimal_signature = self.signature_modeler.calculate_radar_signature(
            self.decoy_state, self.observer_position, optimal_freq
        )
        suboptimal_signature = self.signature_modeler.calculate_radar_signature(
            self.decoy_state, self.observer_position, suboptimal_freq
        )
        
        # Chaff should be more effective at its optimal frequency
        assert optimal_signature >= suboptimal_signature
    
    def test_lighting_conditions_effect(self):
        """Test lighting conditions effect on visual signatures"""
        daylight_signature = self.signature_modeler.calculate_visual_signature(
            self.decoy_state, self.observer_position, "daylight"
        )
        night_signature = self.signature_modeler.calculate_visual_signature(
            self.decoy_state, self.observer_position, "night"
        )
        
        assert daylight_signature > night_signature


class TestDeploymentSequencer:
    """Test deployment sequencing optimization"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.deployment_sequencer = DeploymentSequencer()
        self.decoys = [
            DecoySpecifications(
                DecoyType.CHAFF, 10.0, 0.1, 0.5, 1.0, 30.0, 50.0, 0.5, 100.0
            ),
            DecoySpecifications(
                DecoyType.FLARE, 0.1, 1000.0, 10.0, 1.0, 10.0, 30.0, 0.3, 50.0
            ),
            DecoySpecifications(
                DecoyType.TOWED_DECOY, 5.0, 10.0, 2.0, 2.0, 300.0, 20.0, 5.0, 1000.0
            )
        ]
        self.threats = [
            ThreatSystem("SAM_001", ThreatType.RADAR_GUIDED_MISSILE, 50.0, 1.0),
            ThreatSystem("IR_001", ThreatType.INFRARED_MISSILE, 20.0, 2.0)
        ]
        self.position = Position3D(0, 0, 1000)
        self.velocity = Velocity3D(200, 0, 0)
    
    def test_burst_deployment(self):
        """Test burst deployment pattern"""
        sequence = self.deployment_sequencer.optimize_deployment_sequence(
            self.decoys, self.threats, self.position, self.velocity, "burst"
        )
        
        assert len(sequence) == len(self.decoys)
        # All deployments should be at time 0 for burst pattern
        assert all(dep['deployment_time'] == 0.0 for dep in sequence)
        # Should be sorted by priority
        priorities = [dep['priority'] for dep in sequence]
        assert priorities == sorted(priorities, reverse=True)
    
    def test_sequential_deployment(self):
        """Test sequential deployment pattern"""
        sequence = self.deployment_sequencer.optimize_deployment_sequence(
            self.decoys, self.threats, self.position, self.velocity, "sequential"
        )
        
        assert len(sequence) == len(self.decoys)
        # Deployment times should be increasing
        times = [dep['deployment_time'] for dep in sequence]
        assert times == sorted(times)
    
    def test_adaptive_deployment(self):
        """Test adaptive deployment pattern"""
        sequence = self.deployment_sequencer.optimize_deployment_sequence(
            self.decoys, self.threats, self.position, self.velocity, "adaptive"
        )
        
        assert len(sequence) > 0
        # Should have target_threats field for adaptive deployment
        assert all('target_threats' in dep for dep in sequence)
    
    def test_coordinated_deployment(self):
        """Test coordinated deployment pattern"""
        sequence = self.deployment_sequencer.optimize_deployment_sequence(
            self.decoys, self.threats, self.position, self.velocity, "coordinated"
        )
        
        assert len(sequence) == len(self.decoys)
        # Should have wave information
        assert all('wave' in dep for dep in sequence)
    
    def test_decoy_priority_calculation(self):
        """Test decoy priority calculation against threats"""
        chaff_decoy = self.decoys[0]  # CHAFF
        flare_decoy = self.decoys[1]  # FLARE
        
        radar_threat = [ThreatSystem("SAM", ThreatType.RADAR_GUIDED_MISSILE, 50.0, 1.0)]
        ir_threat = [ThreatSystem("IR", ThreatType.INFRARED_MISSILE, 20.0, 1.0)]
        
        chaff_vs_radar = self.deployment_sequencer._calculate_decoy_priority(chaff_decoy, radar_threat)
        chaff_vs_ir = self.deployment_sequencer._calculate_decoy_priority(chaff_decoy, ir_threat)
        flare_vs_radar = self.deployment_sequencer._calculate_decoy_priority(flare_decoy, radar_threat)
        flare_vs_ir = self.deployment_sequencer._calculate_decoy_priority(flare_decoy, ir_threat)
        
        # Chaff should be more effective against radar threats
        assert chaff_vs_radar > chaff_vs_ir
        # Flares should be more effective against IR threats
        assert flare_vs_ir > flare_vs_radar
    
    def test_effectiveness_against_threat(self):
        """Test effectiveness calculation against specific threats"""
        chaff_decoy = self.decoys[0]
        radar_threat = ThreatSystem("SAM", ThreatType.RADAR_GUIDED_MISSILE, 50.0, 1.0)
        ir_threat = ThreatSystem("IR", ThreatType.INFRARED_MISSILE, 20.0, 1.0)
        
        effectiveness_vs_radar = self.deployment_sequencer._calculate_effectiveness_against_threat(
            chaff_decoy, radar_threat
        )
        effectiveness_vs_ir = self.deployment_sequencer._calculate_effectiveness_against_threat(
            chaff_decoy, ir_threat
        )
        
        assert 0.0 <= effectiveness_vs_radar <= 1.0
        assert 0.0 <= effectiveness_vs_ir <= 1.0
        assert effectiveness_vs_radar > effectiveness_vs_ir  # Chaff better vs radar
    
    def test_best_decoys_selection(self):
        """Test selection of best decoys for threat type"""
        best_vs_radar = self.deployment_sequencer._select_best_decoys_for_threat(
            self.decoys, ThreatType.RADAR_GUIDED_MISSILE
        )
        best_vs_ir = self.deployment_sequencer._select_best_decoys_for_threat(
            self.decoys, ThreatType.INFRARED_MISSILE
        )
        
        assert len(best_vs_radar) > 0
        assert len(best_vs_ir) > 0
        
        # Check that appropriate decoys are selected
        radar_decoy_types = [decoy.decoy_type for _, decoy in best_vs_radar]
        ir_decoy_types = [decoy.decoy_type for _, decoy in best_vs_ir]
        
        # Should prefer chaff/towed decoys for radar threats
        assert DecoyType.CHAFF in radar_decoy_types or DecoyType.TOWED_DECOY in radar_decoy_types
        # Should prefer flares for IR threats
        assert DecoyType.FLARE in ir_decoy_types


class TestEffectivenessAssessor:
    """Test effectiveness assessment capabilities"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.effectiveness_assessor = EffectivenessAssessor()
        self.decoy_states = [
            DecoyState(
                "chaff_001", DecoyType.CHAFF, Position3D(100, 0, 500),
                Velocity3D(10, 0, 0), 0.0, 1.0, True
            ),
            DecoyState(
                "flare_001", DecoyType.FLARE, Position3D(200, 0, 500),
                Velocity3D(5, 0, 0), 0.0, 0.8, True
            )
        ]
        self.threats = [
            ThreatSystem("SAM_001", ThreatType.RADAR_GUIDED_MISSILE, 50.0, 1.0, 10e9),
            ThreatSystem("IR_001", ThreatType.INFRARED_MISSILE, 20.0, 2.0)
        ]
        self.aircraft_position = Position3D(0, 0, 500)
    
    def test_overall_effectiveness_assessment(self):
        """Test overall effectiveness assessment"""
        assessment = self.effectiveness_assessor.assess_decoy_effectiveness(
            self.decoy_states, self.threats, self.aircraft_position, 10.0
        )
        
        assert 'overall_effectiveness' in assessment
        assert 'threat_assessments' in assessment
        assert 'decoy_assessments' in assessment
        assert 'recommendations' in assessment
        
        assert 0.0 <= assessment['overall_effectiveness'] <= 1.0
        assert len(assessment['threat_assessments']) == len(self.threats)
        assert len(assessment['decoy_assessments']) == len(self.decoy_states)
    
    def test_threat_specific_assessment(self):
        """Test assessment against specific threats"""
        radar_threat = self.threats[0]
        
        threat_assessment = self.effectiveness_assessor._assess_against_threat(
            self.decoy_states, radar_threat, self.aircraft_position, 10.0
        )
        
        assert 'threat_id' in threat_assessment
        assert 'effectiveness' in threat_assessment
        assert 'active_decoys' in threat_assessment
        assert 'confusion_factor' in threat_assessment
        
        assert threat_assessment['threat_id'] == radar_threat.threat_id
        assert 0.0 <= threat_assessment['effectiveness'] <= 1.0
        assert threat_assessment['active_decoys'] == len(self.decoy_states)
    
    def test_individual_decoy_assessment(self):
        """Test individual decoy assessment"""
        decoy = self.decoy_states[0]
        
        decoy_assessment = self.effectiveness_assessor._assess_individual_decoy(
            decoy, self.threats, self.aircraft_position, 10.0
        )
        
        assert 'decoy_id' in decoy_assessment
        assert 'overall_score' in decoy_assessment
        assert 'threat_effectiveness' in decoy_assessment
        
        assert decoy_assessment['decoy_id'] == decoy.decoy_id
        assert 0.0 <= decoy_assessment['overall_score'] <= 1.0
        assert len(decoy_assessment['threat_effectiveness']) == len(self.threats)
    
    def test_recommendations_generation(self):
        """Test recommendation generation"""
        # Create a scenario with low effectiveness
        poor_assessment = {
            'overall_effectiveness': 0.2,
            'threat_assessments': {
                'SAM_001': {'effectiveness': 0.1, 'threat_type': 'radar_guided_missile'},
                'IR_001': {'effectiveness': 0.3, 'threat_type': 'infrared_missile'}
            }
        }
        
        recommendations = self.effectiveness_assessor._generate_recommendations(
            self.decoy_states, self.threats, poor_assessment
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        # Should recommend improvements for low effectiveness
        assert any("low" in rec.lower() for rec in recommendations)
    
    def test_empty_inputs_handling(self):
        """Test handling of empty inputs"""
        # Test with no decoys
        assessment = self.effectiveness_assessor.assess_decoy_effectiveness(
            [], self.threats, self.aircraft_position, 10.0
        )
        assert assessment['overall_effectiveness'] == 0.0
        
        # Test with no threats
        assessment = self.effectiveness_assessor.assess_decoy_effectiveness(
            self.decoy_states, [], self.aircraft_position, 10.0
        )
        assert assessment['overall_effectiveness'] == 0.0


class TestDecoySystemSimulator:
    """Test main decoy system simulator"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.available_decoys = [
            DecoySpecifications(
                DecoyType.CHAFF, 10.0, 0.1, 0.5, 1.0, 30.0, 50.0, 0.5, 100.0
            ),
            DecoySpecifications(
                DecoyType.FLARE, 0.1, 1000.0, 10.0, 1.0, 10.0, 30.0, 0.3, 50.0
            )
        ]
        self.simulator = DecoySystemSimulator(self.available_decoys)
        
        # Add some threats
        self.simulator.add_threat_system(
            ThreatSystem("SAM_001", ThreatType.RADAR_GUIDED_MISSILE, 50.0, 1.0)
        )
        self.simulator.add_threat_system(
            ThreatSystem("IR_001", ThreatType.INFRARED_MISSILE, 20.0, 2.0)
        )
    
    def test_simulator_initialization(self):
        """Test simulator initialization"""
        assert len(self.simulator.available_decoys) == 2
        assert len(self.simulator.deployed_decoys) == 0
        assert len(self.simulator.threat_systems) == 2
        
        # Test initialize method
        result = self.simulator.initialize()
        assert result is True
        assert self.simulator.initialized is True
    
    def test_threat_system_addition(self):
        """Test adding threat systems"""
        initial_count = len(self.simulator.threat_systems)
        
        new_threat = ThreatSystem("EW_001", ThreatType.ELECTRONIC_WARFARE, 30.0, 1.5)
        self.simulator.add_threat_system(new_threat)
        
        assert len(self.simulator.threat_systems) == initial_count + 1
        assert new_threat in self.simulator.threat_systems
    
    def test_decoy_deployment(self):
        """Test decoy deployment"""
        initial_deployed = len(self.simulator.deployed_decoys)
        
        result = self.simulator.deploy_decoys("burst")
        
        assert result is True
        assert len(self.simulator.deployed_decoys) > initial_deployed
        
        # Check that deployed decoys have proper structure
        for decoy in self.simulator.deployed_decoys:
            assert hasattr(decoy, 'decoy_id')
            assert hasattr(decoy, 'decoy_type')
            assert hasattr(decoy, 'position')
            assert hasattr(decoy, 'remaining_effectiveness')
    
    def test_different_deployment_patterns(self):
        """Test different deployment patterns"""
        patterns = ["burst", "sequential", "adaptive", "coordinated"]
        
        for pattern in patterns:
            # Reset simulator
            self.simulator.deployed_decoys = []
            
            result = self.simulator.deploy_decoys(pattern)
            assert result is True
            assert len(self.simulator.deployed_decoys) > 0
    
    def test_simulation_update(self):
        """Test simulation state update"""
        # Deploy some decoys first
        self.simulator.deploy_decoys("burst")
        initial_time = self.simulator.simulation_time
        
        # Update simulation
        dt = 1.0
        result = self.simulator.update_simulation(dt)
        
        assert 'simulation_time' in result
        assert result['simulation_time'] == initial_time + dt
        assert 'aircraft_position' in result
        assert 'active_decoys' in result
        assert 'effectiveness_assessment' in result
        assert 'decoy_states' in result
    
    def test_time_degradation(self):
        """Test decoy effectiveness degradation over time"""
        # Deploy decoys
        self.simulator.deploy_decoys("burst")
        
        # Get initial effectiveness
        initial_effectiveness = [d.remaining_effectiveness for d in self.simulator.deployed_decoys]
        
        # Advance time significantly
        for _ in range(10):
            self.simulator.update_simulation(5.0)  # 5 seconds each
        
        # Check that effectiveness has degraded
        final_effectiveness = [d.remaining_effectiveness for d in self.simulator.deployed_decoys]
        
        for initial, final in zip(initial_effectiveness, final_effectiveness):
            assert final <= initial  # Should not increase
    
    def test_decoy_deactivation(self):
        """Test that decoys deactivate when effectiveness is too low"""
        # Deploy decoys
        self.simulator.deploy_decoys("burst")
        
        # Manually set very old deployment time to force low effectiveness
        for decoy in self.simulator.deployed_decoys:
            decoy.deployment_time = -1000.0  # Very old deployment
        
        # Update simulation
        self.simulator.update_simulation(1.0)
        
        # Check that decoys are deactivated
        active_decoys = [d for d in self.simulator.deployed_decoys if d.active]
        assert len(active_decoys) == 0
    
    def test_system_status(self):
        """Test getting system status"""
        status = self.simulator.get_system_status()
        
        assert 'simulation_time' in status
        assert 'available_decoys' in status
        assert 'deployed_decoys' in status
        assert 'active_decoys' in status
        assert 'threat_systems' in status
        assert 'aircraft_position' in status
        
        assert status['available_decoys'] == len(self.available_decoys)
        assert status['threat_systems'] == 2
    
    def test_aircraft_position_update(self):
        """Test aircraft position updates during simulation"""
        initial_position = Position3D(
            self.simulator.aircraft_position.x,
            self.simulator.aircraft_position.y,
            self.simulator.aircraft_position.z
        )
        
        # Update simulation
        dt = 10.0
        self.simulator.update_simulation(dt)
        
        # Position should have changed based on velocity
        expected_x = initial_position.x + self.simulator.aircraft_velocity.vx * dt
        assert abs(self.simulator.aircraft_position.x - expected_x) < 0.1
    
    def test_decoy_state_to_dict_conversion(self):
        """Test conversion of decoy state to dictionary"""
        decoy_state = DecoyState(
            "test_decoy", DecoyType.CHAFF, Position3D(100, 200, 300),
            Velocity3D(10, 20, 30), 5.0, 0.8, True
        )
        
        decoy_dict = self.simulator._decoy_state_to_dict(decoy_state)
        
        assert decoy_dict['decoy_id'] == "test_decoy"
        assert decoy_dict['decoy_type'] == "chaff"
        assert decoy_dict['position']['x'] == 100
        assert decoy_dict['velocity']['vx'] == 10
        assert decoy_dict['deployment_time'] == 5.0
        assert decoy_dict['remaining_effectiveness'] == 0.8
        assert decoy_dict['active'] is True


if __name__ == "__main__":
    pytest.main([__file__])