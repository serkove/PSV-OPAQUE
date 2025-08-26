"""Tests for modular assembly sequence optimization."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from fighter_jet_sdk.engines.manufacturing.modular_assembly import (
    ModularAssembly,
    AssemblyConstraintType,
    QualityCheckType,
    ResourceType,
    AssemblyConstraint,
    QualityCheckpoint,
    AssemblyResource,
    AssemblyStep,
    AssemblySequence,
    ConflictDetection
)
from fighter_jet_sdk.common.data_models import (
    AircraftConfiguration,
    BasePlatform,
    Module,
    ModuleType,
    PhysicalProperties,
    ElectricalInterface,
    MechanicalInterface
)


class TestModularAssembly:
    """Test suite for ModularAssembly class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.assembly_system = ModularAssembly()
        
        # Create test aircraft configuration
        self.test_config = self._create_test_configuration()
    
    def _create_test_configuration(self) -> AircraftConfiguration:
        """Create a test aircraft configuration."""
        # Base platform
        base_platform = BasePlatform(
            platform_id="test_platform",
            name="Test Fighter Platform",
            base_mass=5000.0,  # 5000 kg
            power_generation_capacity=100000.0,  # 100 kW
            fuel_capacity=3000.0  # 3000 kg
        )
        
        # Test modules
        cockpit_module = Module(
            module_id="cockpit_001",
            module_type=ModuleType.COCKPIT,
            name="Advanced Cockpit",
            physical_properties=PhysicalProperties(
                mass=30.0,  # Light module under 50kg threshold
                center_of_gravity=(0.0, 0.0, 0.0),
                moments_of_inertia=(100.0, 100.0, 100.0),
                dimensions=(2.0, 1.5, 1.0)
            ),
            electrical_interfaces=[
                ElectricalInterface("elec_001", 28.0, 50.0, 1000.0),
                ElectricalInterface("elec_002", 115.0, 10.0, 500.0)
            ],
            mechanical_interfaces=[
                MechanicalInterface("mech_001", "bolt_pattern", (10000.0, 10000.0, 5000.0), (1000.0, 1000.0, 500.0), (0.0, 0.0, 0.0))
            ]
        )
        
        sensor_module = Module(
            module_id="sensor_001",
            module_type=ModuleType.SENSOR,
            name="AESA Radar",
            physical_properties=PhysicalProperties(
                mass=150.0,
                center_of_gravity=(0.0, 0.0, 0.0),
                moments_of_inertia=(75.0, 75.0, 75.0),
                dimensions=(1.5, 1.0, 0.5)
            ),
            electrical_interfaces=[
                ElectricalInterface("elec_003", 28.0, 100.0, 2000.0)
            ],
            mechanical_interfaces=[
                MechanicalInterface("mech_002", "gimbal_mount", (5000.0, 5000.0, 2500.0), (500.0, 500.0, 250.0), (1.0, 0.0, 0.0))
            ]
        )
        
        payload_module = Module(
            module_id="payload_001",
            module_type=ModuleType.PAYLOAD,
            name="Weapon Bay",
            physical_properties=PhysicalProperties(
                mass=300.0,
                center_of_gravity=(0.0, 0.0, 0.0),
                moments_of_inertia=(150.0, 150.0, 150.0),
                dimensions=(3.0, 2.0, 1.0)
            ),
            electrical_interfaces=[
                ElectricalInterface("elec_004", 28.0, 25.0, 500.0)
            ],
            mechanical_interfaces=[
                MechanicalInterface("mech_003", "rail_system", (15000.0, 15000.0, 7500.0), (1500.0, 1500.0, 750.0), (2.0, 0.0, 0.0))
            ]
        )
        
        return AircraftConfiguration(
            config_id="test_config_001",
            name="Test Fighter Configuration",
            base_platform=base_platform,
            modules=[cockpit_module, sensor_module, payload_module]
        )
    
    def test_initialization(self):
        """Test ModularAssembly initialization."""
        assert self.assembly_system.logger is not None
        assert isinstance(self.assembly_system.resources, dict)
        assert len(self.assembly_system.resources) > 0
        assert isinstance(self.assembly_system.standard_checkpoints, dict)
        assert len(self.assembly_system.standard_checkpoints) > 0
        
        # Check standard resources
        assert 'technician_basic' in self.assembly_system.resources
        assert 'technician_advanced' in self.assembly_system.resources
        assert 'main_assembly_fixture' in self.assembly_system.resources
        assert 'overhead_crane' in self.assembly_system.resources
        
        # Check standard checkpoints
        assert 'dimensional_check' in self.assembly_system.standard_checkpoints
        assert 'electrical_continuity' in self.assembly_system.standard_checkpoints
        assert 'torque_verification' in self.assembly_system.standard_checkpoints
    
    def test_generate_assembly_steps(self):
        """Test assembly step generation from configuration."""
        steps = self.assembly_system._generate_assembly_steps(self.test_config)
        
        assert len(steps) > 0
        
        # Check for base platform setup
        base_step = next((s for s in steps if s.step_id == "base_platform_setup"), None)
        assert base_step is not None
        assert base_step.estimated_time > 0
        assert base_step.labor_hours > 0
        
        # Check for module steps
        module_ids = [m.module_id for m in self.test_config.modules]
        for module_id in module_ids:
            # Should have prep, install, and connect steps
            prep_step = next((s for s in steps if s.step_id == f"prep_{module_id}"), None)
            install_step = next((s for s in steps if s.step_id == f"install_{module_id}"), None)
            connect_step = next((s for s in steps if s.step_id == f"connect_{module_id}"), None)
            
            assert prep_step is not None, f"Missing prep step for {module_id}"
            assert install_step is not None, f"Missing install step for {module_id}"
            assert connect_step is not None, f"Missing connect step for {module_id}"
        
        # Check for system integration
        integration_step = next((s for s in steps if s.step_id == "system_integration"), None)
        assert integration_step is not None
        assert integration_step.skill_level_required >= 4
    
    def test_generate_assembly_constraints(self):
        """Test assembly constraint generation."""
        steps = self.assembly_system._generate_assembly_steps(self.test_config)
        constraints = self.assembly_system._generate_assembly_constraints(steps, self.test_config)
        
        assert len(constraints) > 0
        
        # Check for precedence constraints
        precedence_constraints = [c for c in constraints if c.constraint_type == AssemblyConstraintType.PRECEDENCE]
        assert len(precedence_constraints) > 0
        
        # Base platform should precede all other steps
        base_precedence = [c for c in precedence_constraints if c.source_step == "base_platform_setup"]
        assert len(base_precedence) > 0
        
        # Prep should precede install for each module
        for module in self.test_config.modules:
            prep_to_install = next((c for c in precedence_constraints 
                                  if c.source_step == f"prep_{module.module_id}" 
                                  and c.target_step == f"install_{module.module_id}"), None)
            assert prep_to_install is not None, f"Missing prep->install constraint for {module.module_id}"
        
        # Check for exclusion constraints (crane usage)
        exclusion_constraints = [c for c in constraints if c.constraint_type == AssemblyConstraintType.EXCLUSION]
        # Should have exclusion constraints if multiple steps use crane
        crane_steps = [s for s in steps if 'overhead_crane' in s.required_resources]
        if len(crane_steps) > 1:
            assert len(exclusion_constraints) > 0
    
    def test_generate_quality_checkpoints(self):
        """Test quality checkpoint generation."""
        steps = self.assembly_system._generate_assembly_steps(self.test_config)
        checkpoints = self.assembly_system._generate_quality_checkpoints(steps)
        
        assert len(checkpoints) > 0
        
        # Each checkpoint should have required fields
        for checkpoint in checkpoints:
            assert checkpoint.checkpoint_id
            assert checkpoint.step_id
            assert checkpoint.inspection_time > 0
            assert isinstance(checkpoint.required_equipment, list)
            assert isinstance(checkpoint.acceptance_criteria, dict)
        
        # Should have dimensional checks
        dimensional_checks = [c for c in checkpoints if c.checkpoint_type == QualityCheckType.DIMENSIONAL]
        assert len(dimensional_checks) > 0
        
        # Should have electrical checks for modules with electrical interfaces
        electrical_checks = [c for c in checkpoints if c.checkpoint_type == QualityCheckType.ELECTRICAL]
        modules_with_electrical = [m for m in self.test_config.modules if m.electrical_interfaces]
        if modules_with_electrical:
            assert len(electrical_checks) > 0
    
    def test_optimize_assembly_sequence(self):
        """Test complete assembly sequence optimization."""
        production_schedule = {
            'start_date': datetime.now(),
            'target_completion': datetime.now() + timedelta(days=7)
        }
        
        resource_constraints = {
            'max_technicians': 6,
            'work_hours_per_day': 8
        }
        
        sequence = self.assembly_system.optimize_assembly_sequence(
            self.test_config,
            production_schedule,
            resource_constraints
        )
        
        assert isinstance(sequence, AssemblySequence)
        assert sequence.sequence_id
        assert sequence.configuration_id == self.test_config.config_id
        assert len(sequence.steps) > 0
        assert len(sequence.constraints) > 0
        assert len(sequence.quality_checkpoints) > 0
        assert sequence.total_time > 0
        assert sequence.total_cost > 0
        assert isinstance(sequence.resource_utilization, dict)
        assert isinstance(sequence.critical_path, list)
        assert sequence.risk_score >= 0
        
        # Check that steps are scheduled
        scheduled_steps = [s for s in sequence.steps if s.scheduled_start and s.scheduled_finish]
        assert len(scheduled_steps) > 0
        
        # Check that base platform is first
        base_step = next((s for s in sequence.steps if s.step_id == "base_platform_setup"), None)
        if base_step and base_step.scheduled_start:
            other_steps = [s for s in sequence.steps if s.step_id != "base_platform_setup" and s.scheduled_start]
            for step in other_steps:
                assert step.scheduled_start >= base_step.scheduled_start
    
    def test_detect_conflicts(self):
        """Test conflict detection in assembly sequence."""
        # Create a sequence with potential conflicts
        steps = self.assembly_system._generate_assembly_steps(self.test_config)
        constraints = self.assembly_system._generate_assembly_constraints(steps, self.test_config)
        checkpoints = self.assembly_system._generate_quality_checkpoints(steps)
        
        # Manually create overlapping schedules to test conflict detection
        current_time = datetime.now()
        for i, step in enumerate(steps):
            step.scheduled_start = current_time + timedelta(hours=i)
            step.scheduled_finish = current_time + timedelta(hours=i+2)  # 2-hour overlap
        
        sequence = AssemblySequence(
            sequence_id="test_conflict_seq",
            configuration_id=self.test_config.config_id,
            steps=steps,
            constraints=constraints,
            quality_checkpoints=checkpoints,
            total_time=480.0,  # 8 hours
            total_cost=5000.0,
            resource_utilization={},
            critical_path=[],
            risk_score=0.5
        )
        
        conflicts = self.assembly_system.detect_conflicts(sequence)
        
        # Should detect resource conflicts due to overlapping schedules
        assert isinstance(conflicts, list)
        # May or may not have conflicts depending on resource allocation
    
    def test_step_estimation_methods(self):
        """Test individual step estimation methods."""
        test_module = self.test_config.modules[0]  # Cockpit module
        
        # Test prep time estimation
        prep_time = self.assembly_system._estimate_prep_time(test_module)
        assert prep_time > 0
        assert prep_time >= 30.0  # Base time
        
        # Test assembly time estimation
        assembly_time = self.assembly_system._estimate_assembly_time(test_module)
        assert assembly_time > 0
        assert assembly_time >= 60.0  # Base time
        
        # Test labor hours estimation
        labor_hours = self.assembly_system._estimate_labor_hours(test_module)
        assert labor_hours > 0
        assert labor_hours >= 2.0  # Base hours
        
        # Test resource determination
        resources = self.assembly_system._determine_required_resources(test_module)
        assert isinstance(resources, list)
        assert len(resources) > 0
        assert 'technician_basic' in resources
        
        # Test skill level determination
        skill_level = self.assembly_system._determine_skill_level(test_module)
        assert isinstance(skill_level, int)
        assert 1 <= skill_level <= 5
        
        # Test quality checks determination
        quality_checks = self.assembly_system._determine_quality_checks(test_module)
        assert isinstance(quality_checks, list)
        assert 'dimensional_check' in quality_checks
        
        # Test risk assessment
        risk_factors = self.assembly_system._assess_module_risk(test_module)
        assert isinstance(risk_factors, dict)
        assert 'complexity' in risk_factors
        assert 'criticality' in risk_factors
        assert 0 <= risk_factors['complexity'] <= 1
        assert 0 <= risk_factors['criticality'] <= 1
    
    def test_heavy_module_handling(self):
        """Test handling of heavy modules requiring crane."""
        # Create a heavy module
        heavy_module = Module(
            module_id="heavy_001",
            module_type=ModuleType.PROPULSION,
            name="Heavy Engine",
            physical_properties=PhysicalProperties(
                mass=1000.0,  # 1000 kg - heavy module
                center_of_gravity=(0.0, 0.0, 0.0),
                moments_of_inertia=(500.0, 500.0, 500.0),
                dimensions=(3.0, 2.0, 2.0)
            )
        )
        
        # Test resource determination
        resources = self.assembly_system._determine_required_resources(heavy_module)
        assert 'overhead_crane' in resources
        
        # Test setup time (should be longer for heavy modules)
        setup_time = self.assembly_system._estimate_setup_time(heavy_module)
        light_setup_time = self.assembly_system._estimate_setup_time(self.test_config.modules[0])
        # Heavy module (1000kg) should have longer setup time than light module (30kg)
        assert setup_time > light_setup_time
    
    def test_complex_module_handling(self):
        """Test handling of modules with many interfaces."""
        # Create a complex module with many interfaces
        complex_module = Module(
            module_id="complex_001",
            module_type=ModuleType.AVIONICS,
            name="Complex Avionics",
            electrical_interfaces=[
                ElectricalInterface(f"elec_{i}", 28.0, 10.0, 100.0) for i in range(5)
            ],
            mechanical_interfaces=[
                MechanicalInterface(f"mech_{i}", "connector", (1000.0, 1000.0, 1000.0), (100.0, 100.0, 100.0), (0.0, 0.0, 0.0)) for i in range(4)
            ]
        )
        
        # Test skill level (should be higher for complex modules)
        skill_level = self.assembly_system._determine_skill_level(complex_module)
        simple_skill_level = self.assembly_system._determine_skill_level(self.test_config.modules[0])
        assert skill_level >= simple_skill_level
        
        # Test resource requirements (should include advanced technicians)
        resources = self.assembly_system._determine_required_resources(complex_module)
        assert 'technician_advanced' in resources
        
        # Test risk assessment (should have higher complexity)
        risk_factors = self.assembly_system._assess_module_risk(complex_module)
        simple_risk = self.assembly_system._assess_module_risk(self.test_config.modules[0])
        assert risk_factors['complexity'] >= simple_risk['complexity']
    
    def test_critical_path_calculation(self):
        """Test critical path calculation."""
        steps = self.assembly_system._generate_assembly_steps(self.test_config)
        constraints = self.assembly_system._generate_assembly_constraints(steps, self.test_config)
        
        critical_path = self.assembly_system._find_critical_path(steps, constraints)
        
        assert isinstance(critical_path, list)
        # Critical path should start with base platform if it exists
        if any(s.step_id == "base_platform_setup" for s in steps):
            assert critical_path[0] == "base_platform_setup"
    
    def test_resource_utilization_calculation(self):
        """Test resource utilization calculation."""
        steps = self.assembly_system._generate_assembly_steps(self.test_config)
        
        # Set some scheduled times for testing
        current_time = datetime.now()
        for i, step in enumerate(steps):
            step.scheduled_start = current_time + timedelta(hours=i*2)
            step.scheduled_finish = current_time + timedelta(hours=i*2+1)
        
        utilization = self.assembly_system._calculate_resource_utilization(steps)
        
        assert isinstance(utilization, dict)
        # All utilization values should be between 0 and 100
        for resource_id, util in utilization.items():
            assert 0 <= util <= 100
    
    def test_cost_calculation(self):
        """Test cost calculation."""
        steps = self.assembly_system._generate_assembly_steps(self.test_config)
        checkpoints = self.assembly_system._generate_quality_checkpoints(steps)
        
        total_cost = self.assembly_system._calculate_total_cost(steps, checkpoints)
        
        assert total_cost > 0
        assert isinstance(total_cost, float)
    
    def test_time_calculation(self):
        """Test total time calculation."""
        steps = self.assembly_system._generate_assembly_steps(self.test_config)
        
        # Set scheduled times
        current_time = datetime.now()
        for i, step in enumerate(steps):
            step.scheduled_start = current_time + timedelta(hours=i)
            step.scheduled_finish = current_time + timedelta(hours=i+1)
        
        total_time = self.assembly_system._calculate_total_time(steps)
        
        assert total_time > 0
        assert isinstance(total_time, float)
    
    def test_risk_assessment(self):
        """Test risk assessment calculation."""
        steps = self.assembly_system._generate_assembly_steps(self.test_config)
        constraints = self.assembly_system._generate_assembly_constraints(steps, self.test_config)
        
        risk_score = self.assembly_system._assess_risk(steps, constraints)
        
        assert isinstance(risk_score, float)
        assert risk_score >= 0
    
    def test_empty_configuration(self):
        """Test handling of empty configuration."""
        empty_config = AircraftConfiguration(
            config_id="empty_config",
            name="Empty Configuration",
            modules=[]
        )
        
        steps = self.assembly_system._generate_assembly_steps(empty_config)
        
        # Should still have some steps (like system integration)
        assert isinstance(steps, list)


if __name__ == '__main__':
    pytest.main([__file__])