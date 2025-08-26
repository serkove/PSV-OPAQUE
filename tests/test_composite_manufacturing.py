"""Tests for composite manufacturing process planning."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from fighter_jet_sdk.engines.manufacturing.composite_manufacturing import (
    CompositeManufacturing,
    ProcessType,
    ToolingType,
    FiberPlacementParameters,
    CureParameters,
    ToolingRequirement,
    ManufacturingStep,
    WasteAnalysis,
    CostBreakdown
)
from fighter_jet_sdk.common.data_models import (
    MaterialDefinition,
    MaterialType,
    ThermalProperties,
    ManufacturingConstraints
)


class TestCompositeManufacturing:
    """Test suite for CompositeManufacturing class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.composite_mfg = CompositeManufacturing()
        
        # Create test material
        self.test_material = MaterialDefinition(
            material_id="test_carbon_fiber",
            name="Test Carbon Fiber Composite",
            base_material_type=MaterialType.COMPOSITE,
            thermal_properties=ThermalProperties(
                thermal_conductivity=1.0,
                specific_heat=1000.0,
                density=1600.0,
                melting_point=573.0,
                operating_temp_range=(223.0, 473.0)
            ),
            manufacturing_constraints=ManufacturingConstraints(
                min_thickness=0.001,
                max_thickness=0.1,
                cure_temperature=450.0,
                cure_time=7200.0,
                tooling_requirements=["autoclave", "vacuum_bag"],
                cost_per_kg=75.0
            )
        )
        
        # Test part geometry
        self.test_geometry = {
            'length': 2.0,  # 2m
            'width': 1.0,   # 1m
            'height': 0.1,  # 0.1m
            'thickness': 0.005,  # 5mm
            'complexity_factor': 1.2
        }
    
    def test_initialization(self):
        """Test CompositeManufacturing initialization."""
        assert self.composite_mfg.logger is not None
        assert isinstance(self.composite_mfg.tooling_database, dict)
        assert isinstance(self.composite_mfg.process_templates, dict)
        assert ProcessType.AUTOCLAVE in self.composite_mfg.process_templates
        assert ProcessType.OUT_OF_AUTOCLAVE in self.composite_mfg.process_templates
        assert ProcessType.RESIN_TRANSFER_MOLDING in self.composite_mfg.process_templates
    
    def test_generate_tooling_requirements_autoclave(self):
        """Test tooling requirements generation for autoclave process."""
        tooling_reqs = self.composite_mfg.generate_tooling_requirements(
            self.test_geometry,
            self.test_material,
            ProcessType.AUTOCLAVE,
            production_volume=50
        )
        
        assert len(tooling_reqs) >= 2  # At least mold and vacuum bag
        
        # Check for autoclave mold
        mold_found = False
        vacuum_bag_found = False
        
        for req in tooling_reqs:
            if req.tooling_id == "autoclave_mold_001":
                mold_found = True
                assert req.tooling_type == ToolingType.FEMALE_MOLD
                assert req.material in ["Aluminum", "Invar"]
                assert req.cost_estimate > 0
                assert req.lead_time > 0
            elif req.tooling_id == "vacuum_bag_001":
                vacuum_bag_found = True
                assert req.tooling_type == ToolingType.VACUUM_BAG
                assert req.material == "Nylon film"
        
        assert mold_found, "Autoclave mold not found in tooling requirements"
        assert vacuum_bag_found, "Vacuum bag not found in tooling requirements"
    
    def test_generate_tooling_requirements_rtm(self):
        """Test tooling requirements generation for RTM process."""
        tooling_reqs = self.composite_mfg.generate_tooling_requirements(
            self.test_geometry,
            self.test_material,
            ProcessType.RESIN_TRANSFER_MOLDING,
            production_volume=100
        )
        
        # RTM should have upper and lower molds
        upper_mold_found = False
        lower_mold_found = False
        
        for req in tooling_reqs:
            if req.tooling_id == "rtm_upper_mold_001":
                upper_mold_found = True
                assert req.tooling_type == ToolingType.MALE_MOLD
                assert req.material == "Tool steel"
            elif req.tooling_id == "rtm_lower_mold_001":
                lower_mold_found = True
                assert req.tooling_type == ToolingType.FEMALE_MOLD
                assert req.material == "Tool steel"
        
        assert upper_mold_found, "RTM upper mold not found"
        assert lower_mold_found, "RTM lower mold not found"
    
    def test_generate_tooling_requirements_afp(self):
        """Test tooling requirements generation for AFP process."""
        tooling_reqs = self.composite_mfg.generate_tooling_requirements(
            self.test_geometry,
            self.test_material,
            ProcessType.AUTOMATED_FIBER_PLACEMENT,
            production_volume=25
        )
        
        # AFP should have mandrel
        mandrel_found = False
        
        for req in tooling_reqs:
            if req.tooling_id == "afp_mandrel_001":
                mandrel_found = True
                assert req.tooling_type == ToolingType.MANDREL
                assert req.material == "Aluminum"
        
        assert mandrel_found, "AFP mandrel not found"
    
    def test_mold_cost_calculation(self):
        """Test mold cost calculation logic."""
        # Test different materials
        aluminum_cost = self.composite_mfg._calculate_mold_cost(1.0, 1.0, 0.1, "Aluminum", 50)
        steel_cost = self.composite_mfg._calculate_mold_cost(1.0, 1.0, 0.1, "Steel", 50)
        invar_cost = self.composite_mfg._calculate_mold_cost(1.0, 1.0, 0.1, "Invar", 50)
        
        assert steel_cost > aluminum_cost, "Steel should be more expensive than aluminum"
        assert invar_cost > steel_cost, "Invar should be more expensive than steel"
        
        # Test volume discount
        low_volume_cost = self.composite_mfg._calculate_mold_cost(1.0, 1.0, 0.1, "Aluminum", 10)
        high_volume_cost = self.composite_mfg._calculate_mold_cost(1.0, 1.0, 0.1, "Aluminum", 100)
        
        assert low_volume_cost > high_volume_cost, "Higher volume should have lower per-unit cost"
    
    def test_model_autoclave_process(self):
        """Test autoclave process modeling."""
        cure_params = self.composite_mfg.model_autoclave_process(
            self.test_material,
            part_thickness=0.005,
            fiber_volume_fraction=0.6
        )
        
        assert isinstance(cure_params, CureParameters)
        assert len(cure_params.temperature_profile) >= 4
        assert len(cure_params.pressure_profile) >= 3
        assert cure_params.vacuum_level > 0
        assert cure_params.cure_time > 0
        
        # Check temperature profile makes sense
        temps = [point[1] for point in cure_params.temperature_profile]
        assert max(temps) >= 400.0, "Cure temperature should be reasonable"
        assert min(temps) <= 300.0, "Should start at room temperature"
        
        # Check pressure profile
        pressures = [point[1] for point in cure_params.pressure_profile]
        assert max(pressures) > 500000.0, "Should have significant pressure"
    
    def test_model_oof_process(self):
        """Test out-of-autoclave process modeling."""
        cure_params = self.composite_mfg.model_oof_process(
            self.test_material,
            part_thickness=0.003
        )
        
        assert isinstance(cure_params, CureParameters)
        assert cure_params.post_cure_required is True
        assert cure_params.post_cure_temperature is not None
        assert cure_params.post_cure_time is not None
        
        # OOA should have longer cure times
        autoclave_params = self.composite_mfg.model_autoclave_process(
            self.test_material, 0.003
        )
        assert cure_params.cure_time > autoclave_params.cure_time
    
    def test_model_autoclave_process_no_constraints(self):
        """Test autoclave process modeling with material lacking constraints."""
        material_no_constraints = MaterialDefinition(
            material_id="test_no_constraints",
            name="Test Material No Constraints",
            base_material_type=MaterialType.COMPOSITE
        )
        
        with pytest.raises(ValueError, match="Material must have manufacturing constraints"):
            self.composite_mfg.model_autoclave_process(material_no_constraints, 0.005)
    
    def test_design_fiber_placement(self):
        """Test fiber placement parameter design."""
        load_requirements = {
            'tension_x': 100000.0,  # Primary load in X direction
            'tension_y': 50000.0,
            'shear_xy': 25000.0
        }
        
        placement_params = self.composite_mfg.design_fiber_placement(
            self.test_geometry,
            load_requirements,
            self.test_material
        )
        
        assert isinstance(placement_params, FiberPlacementParameters)
        assert placement_params.fiber_angle == 0.0  # Should align with primary tension
        assert placement_params.tow_width > 0
        assert placement_params.placement_speed > 0
        assert placement_params.compaction_force > 0
        assert placement_params.number_of_plies >= 4
        
        # Test shear-dominated loading
        shear_loads = {
            'shear_xy': 100000.0,
            'tension_x': 25000.0,
            'tension_y': 25000.0
        }
        
        shear_params = self.composite_mfg.design_fiber_placement(
            self.test_geometry,
            shear_loads,
            self.test_material
        )
        
        assert shear_params.fiber_angle == 45.0  # Should be 45° for shear
    
    def test_analyze_material_waste(self):
        """Test material waste analysis."""
        waste_analysis = self.composite_mfg.analyze_material_waste(
            self.test_geometry,
            self.test_material,
            ProcessType.AUTOCLAVE,
            production_volume=50
        )
        
        assert isinstance(waste_analysis, WasteAnalysis)
        assert waste_analysis.material_id == self.test_material.material_id
        assert waste_analysis.total_material_required > waste_analysis.usable_material
        assert waste_analysis.waste_material > 0
        assert 0 < waste_analysis.waste_percentage < 50  # Reasonable waste percentage
        assert waste_analysis.waste_cost > 0
        assert 0 <= waste_analysis.recycling_potential <= 100
        assert waste_analysis.disposal_cost >= 0
        
        # Test different processes have different waste factors
        afp_waste = self.composite_mfg.analyze_material_waste(
            self.test_geometry,
            self.test_material,
            ProcessType.AUTOMATED_FIBER_PLACEMENT,
            production_volume=50
        )
        
        hand_layup_waste = self.composite_mfg.analyze_material_waste(
            self.test_geometry,
            self.test_material,
            ProcessType.HAND_LAYUP,
            production_volume=50
        )
        
        # AFP should have less waste than hand layup
        assert afp_waste.waste_percentage < hand_layup_waste.waste_percentage
    
    def test_estimate_manufacturing_cost(self):
        """Test comprehensive manufacturing cost estimation."""
        # Create test tooling requirements
        tooling_reqs = [
            ToolingRequirement(
                tooling_id="test_mold",
                tooling_type=ToolingType.FEMALE_MOLD,
                material="Aluminum",
                dimensions=(2.0, 1.0, 0.2),
                surface_finish="Ra 1.6 μm",
                thermal_expansion_coefficient=23e-6,
                operating_temperature_range=(293.0, 473.0),
                cost_estimate=25000.0,
                lead_time=30.0
            )
        ]
        
        # Create test manufacturing steps
        mfg_steps = [
            ManufacturingStep(
                step_id="layup",
                step_name="Fiber Layup",
                process_type=ProcessType.AUTOCLAVE,
                duration=120.0,  # minutes
                labor_hours=4.0,
                equipment_required=["autoclave"],
                materials_consumed={"carbon_fiber": 10.0},
                quality_checkpoints=["ply_orientation", "compaction"]
            ),
            ManufacturingStep(
                step_id="cure",
                step_name="Autoclave Cure",
                process_type=ProcessType.AUTOCLAVE,
                duration=180.0,  # minutes
                labor_hours=1.0,
                equipment_required=["autoclave"],
                materials_consumed={},
                quality_checkpoints=["temperature_profile", "pressure_profile"]
            )
        ]
        
        cost_breakdown = self.composite_mfg.estimate_manufacturing_cost(
            self.test_geometry,
            self.test_material,
            ProcessType.AUTOCLAVE,
            production_volume=25,
            tooling_requirements=tooling_reqs,
            manufacturing_steps=mfg_steps
        )
        
        assert isinstance(cost_breakdown, CostBreakdown)
        assert len(cost_breakdown.material_costs) > 0
        assert len(cost_breakdown.labor_costs) > 0
        assert len(cost_breakdown.tooling_costs) > 0
        assert len(cost_breakdown.equipment_costs) > 0
        assert cost_breakdown.overhead_costs > 0
        assert cost_breakdown.total_cost > 0
        assert cost_breakdown.cost_per_unit > 0
        
        # Verify cost components sum correctly
        expected_total = (
            sum(cost_breakdown.material_costs.values()) +
            sum(cost_breakdown.labor_costs.values()) +
            sum(cost_breakdown.tooling_costs.values()) +
            sum(cost_breakdown.equipment_costs.values()) +
            cost_breakdown.overhead_costs +
            sum(cost_breakdown.waste_costs.values())
        )
        
        assert abs(cost_breakdown.total_cost - expected_total) < 0.01
    
    def test_process_templates_initialization(self):
        """Test that process templates are properly initialized."""
        templates = self.composite_mfg.process_templates
        
        # Check autoclave template
        autoclave = templates[ProcessType.AUTOCLAVE]
        assert 'typical_cure_temp' in autoclave
        assert 'typical_pressure' in autoclave
        assert 'typical_cure_time' in autoclave
        assert autoclave['typical_cure_temp'] > 400.0
        assert autoclave['typical_pressure'] > 500000.0
        
        # Check OOA template
        ooa = templates[ProcessType.OUT_OF_AUTOCLAVE]
        assert ooa['typical_cure_temp'] < autoclave['typical_cure_temp']
        assert ooa['typical_cure_time'] > autoclave['typical_cure_time']
        
        # Check RTM template
        rtm = templates[ProcessType.RESIN_TRANSFER_MOLDING]
        assert 'injection_pressure' in rtm
        assert 'mold_temperature' in rtm
        assert 'injection_time' in rtm
    
    def test_thickness_adjustment_in_cure_modeling(self):
        """Test that cure parameters adjust properly for part thickness."""
        thin_part_params = self.composite_mfg.model_autoclave_process(
            self.test_material, 0.002  # 2mm
        )
        
        thick_part_params = self.composite_mfg.model_autoclave_process(
            self.test_material, 0.010  # 10mm
        )
        
        # Thicker parts should have longer cure times
        assert thick_part_params.cure_time > thin_part_params.cure_time
    
    def test_volume_effects_on_waste(self):
        """Test that production volume affects waste calculations."""
        low_volume_waste = self.composite_mfg.analyze_material_waste(
            self.test_geometry,
            self.test_material,
            ProcessType.AUTOCLAVE,
            production_volume=1
        )
        
        high_volume_waste = self.composite_mfg.analyze_material_waste(
            self.test_geometry,
            self.test_material,
            ProcessType.AUTOCLAVE,
            production_volume=100
        )
        
        # Higher volume should have lower waste percentage due to learning curve
        assert high_volume_waste.waste_percentage < low_volume_waste.waste_percentage
    
    def test_fiber_placement_complexity_adjustment(self):
        """Test fiber placement parameter adjustment for part complexity."""
        simple_geometry = self.test_geometry.copy()
        simple_geometry['complexity_factor'] = 1.0
        
        complex_geometry = self.test_geometry.copy()
        complex_geometry['complexity_factor'] = 3.0
        
        load_requirements = {'tension_x': 100000.0}
        
        simple_params = self.composite_mfg.design_fiber_placement(
            simple_geometry, load_requirements, self.test_material
        )
        
        complex_params = self.composite_mfg.design_fiber_placement(
            complex_geometry, load_requirements, self.test_material
        )
        
        # Complex parts should have smaller tow width and slower placement speed
        assert complex_params.tow_width <= simple_params.tow_width
        assert complex_params.placement_speed <= simple_params.placement_speed


if __name__ == '__main__':
    pytest.main([__file__])