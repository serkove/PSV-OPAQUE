"""Composite manufacturing process planning module."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
from datetime import datetime

from ...common.data_models import MaterialDefinition, ManufacturingConstraints
from ...core.logging import get_engine_logger


class ProcessType(Enum):
    """Composite manufacturing process types."""
    AUTOCLAVE = "autoclave"
    OUT_OF_AUTOCLAVE = "out_of_autoclave"
    RESIN_TRANSFER_MOLDING = "resin_transfer_molding"
    AUTOMATED_FIBER_PLACEMENT = "automated_fiber_placement"
    HAND_LAYUP = "hand_layup"
    FILAMENT_WINDING = "filament_winding"


class ToolingType(Enum):
    """Tooling types for composite manufacturing."""
    MALE_MOLD = "male_mold"
    FEMALE_MOLD = "female_mold"
    MANDREL = "mandrel"
    CAUL_PLATE = "caul_plate"
    VACUUM_BAG = "vacuum_bag"
    AUTOCLAVE_TOOLING = "autoclave_tooling"


@dataclass
class FiberPlacementParameters:
    """Parameters for automated fiber placement."""
    fiber_angle: float  # degrees
    tow_width: float  # mm
    placement_speed: float  # mm/min
    compaction_force: float  # N
    heating_temperature: float  # K
    number_of_plies: int


@dataclass
class CureParameters:
    """Curing parameters for composite materials."""
    temperature_profile: List[Tuple[float, float]]  # (time_min, temp_K)
    pressure_profile: List[Tuple[float, float]]  # (time_min, pressure_Pa)
    vacuum_level: float  # Pa
    cure_time: float  # minutes
    post_cure_required: bool = False
    post_cure_temperature: Optional[float] = None  # K
    post_cure_time: Optional[float] = None  # minutes


@dataclass
class ToolingRequirement:
    """Tooling requirement specification."""
    tooling_id: str
    tooling_type: ToolingType
    material: str
    dimensions: Tuple[float, float, float]  # length, width, height (m)
    surface_finish: str  # Ra value or specification
    thermal_expansion_coefficient: float  # 1/K
    operating_temperature_range: Tuple[float, float]  # K
    cost_estimate: float  # USD
    lead_time: float  # days
    maintenance_requirements: List[str] = field(default_factory=list)


@dataclass
class ManufacturingStep:
    """Individual manufacturing step."""
    step_id: str
    step_name: str
    process_type: ProcessType
    duration: float  # minutes
    labor_hours: float
    equipment_required: List[str]
    materials_consumed: Dict[str, float]  # material_id -> quantity
    quality_checkpoints: List[str]
    predecessor_steps: List[str] = field(default_factory=list)


@dataclass
class WasteAnalysis:
    """Material waste analysis."""
    material_id: str
    total_material_required: float  # kg
    usable_material: float  # kg
    waste_material: float  # kg
    waste_percentage: float
    waste_cost: float  # USD
    recycling_potential: float  # percentage that can be recycled
    disposal_cost: float  # USD


@dataclass
class CostBreakdown:
    """Manufacturing cost breakdown."""
    material_costs: Dict[str, float]  # material_id -> cost_USD
    labor_costs: Dict[str, float]  # step_id -> cost_USD
    tooling_costs: Dict[str, float]  # tooling_id -> cost_USD
    equipment_costs: Dict[str, float]  # equipment_id -> cost_USD
    overhead_costs: float  # USD
    waste_costs: Dict[str, float]  # material_id -> waste_cost_USD
    total_cost: float  # USD
    cost_per_unit: float  # USD per part


class CompositeManufacturing:
    """Composite manufacturing process planning system."""
    
    def __init__(self):
        """Initialize composite manufacturing system."""
        self.logger = get_engine_logger('composite_manufacturing')
        self.tooling_database = {}
        self.process_templates = {}
        self.material_properties = {}
        self._initialize_process_templates()
    
    def _initialize_process_templates(self):
        """Initialize standard process templates."""
        # Autoclave process template
        self.process_templates[ProcessType.AUTOCLAVE] = {
            'typical_cure_temp': 450.0,  # K (177°C)
            'typical_pressure': 689475.0,  # Pa (100 psi)
            'typical_cure_time': 120.0,  # minutes
            'vacuum_level': 84656.0,  # Pa (-25 inHg)
            'heating_rate': 2.0,  # K/min
            'cooling_rate': -2.0,  # K/min
        }
        
        # Out-of-autoclave process template
        self.process_templates[ProcessType.OUT_OF_AUTOCLAVE] = {
            'typical_cure_temp': 420.0,  # K (147°C)
            'typical_pressure': 101325.0,  # Pa (atmospheric)
            'typical_cure_time': 180.0,  # minutes
            'vacuum_level': 84656.0,  # Pa (-25 inHg)
            'heating_rate': 1.5,  # K/min
            'cooling_rate': -1.5,  # K/min
        }
        
        # RTM process template
        self.process_templates[ProcessType.RESIN_TRANSFER_MOLDING] = {
            'injection_pressure': 689475.0,  # Pa (100 psi)
            'mold_temperature': 393.0,  # K (120°C)
            'injection_time': 30.0,  # minutes
            'cure_time': 60.0,  # minutes
            'demolding_time': 15.0,  # minutes
        }
    
    def generate_tooling_requirements(
        self,
        part_geometry: Dict[str, Any],
        material: MaterialDefinition,
        process_type: ProcessType,
        production_volume: int
    ) -> List[ToolingRequirement]:
        """Generate tooling requirements for composite structures."""
        self.logger.info(f"Generating tooling requirements for {process_type.value} process")
        
        tooling_requirements = []
        
        # Extract part dimensions
        length = part_geometry.get('length', 1.0)
        width = part_geometry.get('width', 1.0)
        height = part_geometry.get('height', 0.1)
        complexity = part_geometry.get('complexity_factor', 1.0)
        
        # Determine primary tooling based on process type
        if process_type == ProcessType.AUTOCLAVE:
            tooling_requirements.extend(
                self._generate_autoclave_tooling(length, width, height, material, production_volume)
            )
        elif process_type == ProcessType.OUT_OF_AUTOCLAVE:
            tooling_requirements.extend(
                self._generate_oof_tooling(length, width, height, material, production_volume)
            )
        elif process_type == ProcessType.RESIN_TRANSFER_MOLDING:
            tooling_requirements.extend(
                self._generate_rtm_tooling(length, width, height, material, production_volume)
            )
        elif process_type == ProcessType.AUTOMATED_FIBER_PLACEMENT:
            tooling_requirements.extend(
                self._generate_afp_tooling(length, width, height, material, production_volume)
            )
        
        # Add common tooling requirements
        tooling_requirements.extend(
            self._generate_common_tooling(length, width, height, complexity)
        )
        
        return tooling_requirements
    
    def _generate_autoclave_tooling(
        self,
        length: float,
        width: float,
        height: float,
        material: MaterialDefinition,
        volume: int
    ) -> List[ToolingRequirement]:
        """Generate autoclave-specific tooling requirements."""
        tooling = []
        
        # Primary mold tool
        mold_material = "Invar" if volume > 100 else "Aluminum"
        mold_cost = self._calculate_mold_cost(length, width, height, mold_material, volume)
        
        tooling.append(ToolingRequirement(
            tooling_id="autoclave_mold_001",
            tooling_type=ToolingType.FEMALE_MOLD,
            material=mold_material,
            dimensions=(length * 1.1, width * 1.1, height * 2.0),
            surface_finish="Ra 0.8 μm",
            thermal_expansion_coefficient=1.2e-6 if mold_material == "Invar" else 23e-6,
            operating_temperature_range=(293.0, 500.0),
            cost_estimate=mold_cost,
            lead_time=30 + (volume // 50) * 5,
            maintenance_requirements=["Surface inspection", "Dimensional check", "Release agent application"]
        ))
        
        # Autoclave-specific tooling
        tooling.append(ToolingRequirement(
            tooling_id="vacuum_bag_001",
            tooling_type=ToolingType.VACUUM_BAG,
            material="Nylon film",
            dimensions=(length * 1.2, width * 1.2, 0.001),
            surface_finish="Standard",
            thermal_expansion_coefficient=80e-6,
            operating_temperature_range=(293.0, 473.0),
            cost_estimate=500.0 + (length * width * 50.0),
            lead_time=5.0,
            maintenance_requirements=["Replace after each cycle"]
        ))
        
        return tooling
    
    def _generate_oof_tooling(
        self,
        length: float,
        width: float,
        height: float,
        material: MaterialDefinition,
        volume: int
    ) -> List[ToolingRequirement]:
        """Generate out-of-autoclave tooling requirements."""
        tooling = []
        
        # Heated mold for OOA
        mold_cost = self._calculate_mold_cost(length, width, height, "Aluminum", volume) * 1.3
        
        tooling.append(ToolingRequirement(
            tooling_id="oof_heated_mold_001",
            tooling_type=ToolingType.FEMALE_MOLD,
            material="Aluminum with heating elements",
            dimensions=(length * 1.1, width * 1.1, height * 2.0),
            surface_finish="Ra 1.6 μm",
            thermal_expansion_coefficient=23e-6,
            operating_temperature_range=(293.0, 450.0),
            cost_estimate=mold_cost,
            lead_time=35 + (volume // 30) * 5,
            maintenance_requirements=["Heating element check", "Temperature calibration", "Surface maintenance"]
        ))
        
        return tooling
    
    def _generate_rtm_tooling(
        self,
        length: float,
        width: float,
        height: float,
        material: MaterialDefinition,
        volume: int
    ) -> List[ToolingRequirement]:
        """Generate RTM tooling requirements."""
        tooling = []
        
        # Matched die tooling for RTM
        mold_cost = self._calculate_mold_cost(length, width, height, "Steel", volume) * 2.0
        
        # Upper mold
        tooling.append(ToolingRequirement(
            tooling_id="rtm_upper_mold_001",
            tooling_type=ToolingType.MALE_MOLD,
            material="Tool steel",
            dimensions=(length * 1.05, width * 1.05, height * 1.5),
            surface_finish="Ra 0.4 μm",
            thermal_expansion_coefficient=12e-6,
            operating_temperature_range=(293.0, 423.0),
            cost_estimate=mold_cost * 0.6,
            lead_time=45 + (volume // 20) * 7,
            maintenance_requirements=["Precision measurement", "Surface polishing", "Seal inspection"]
        ))
        
        # Lower mold
        tooling.append(ToolingRequirement(
            tooling_id="rtm_lower_mold_001",
            tooling_type=ToolingType.FEMALE_MOLD,
            material="Tool steel",
            dimensions=(length * 1.05, width * 1.05, height * 1.5),
            surface_finish="Ra 0.4 μm",
            thermal_expansion_coefficient=12e-6,
            operating_temperature_range=(293.0, 423.0),
            cost_estimate=mold_cost * 0.4,
            lead_time=45 + (volume // 20) * 7,
            maintenance_requirements=["Flow channel cleaning", "Vent inspection", "Dimensional check"]
        ))
        
        return tooling
    
    def _generate_afp_tooling(
        self,
        length: float,
        width: float,
        height: float,
        material: MaterialDefinition,
        volume: int
    ) -> List[ToolingRequirement]:
        """Generate automated fiber placement tooling requirements."""
        tooling = []
        
        # AFP mandrel
        mandrel_cost = self._calculate_mold_cost(length, width, height, "Aluminum", volume) * 0.8
        
        tooling.append(ToolingRequirement(
            tooling_id="afp_mandrel_001",
            tooling_type=ToolingType.MANDREL,
            material="Aluminum",
            dimensions=(length, width, height),
            surface_finish="Ra 0.8 μm",
            thermal_expansion_coefficient=23e-6,
            operating_temperature_range=(293.0, 473.0),
            cost_estimate=mandrel_cost,
            lead_time=25 + (volume // 100) * 3,
            maintenance_requirements=["Surface inspection", "Dimensional verification", "Coating renewal"]
        ))
        
        return tooling
    
    def _generate_common_tooling(
        self,
        length: float,
        width: float,
        height: float,
        complexity: float
    ) -> List[ToolingRequirement]:
        """Generate common tooling requirements."""
        tooling = []
        
        # Caul plates
        tooling.append(ToolingRequirement(
            tooling_id="caul_plate_001",
            tooling_type=ToolingType.CAUL_PLATE,
            material="Aluminum",
            dimensions=(length * 1.05, width * 1.05, 0.01),
            surface_finish="Ra 3.2 μm",
            thermal_expansion_coefficient=23e-6,
            operating_temperature_range=(293.0, 473.0),
            cost_estimate=1000.0 + (length * width * 100.0),
            lead_time=10.0,
            maintenance_requirements=["Flatness check", "Surface cleaning"]
        ))
        
        return tooling
    
    def _calculate_mold_cost(
        self,
        length: float,
        width: float,
        height: float,
        material: str,
        volume: int
    ) -> float:
        """Calculate mold cost based on size, material, and volume."""
        base_area = length * width
        complexity_factor = 1.0 + (height / max(length, width))
        
        # Material cost factors
        material_factors = {
            "Aluminum": 1.0,
            "Steel": 1.5,
            "Tool steel": 2.0,
            "Invar": 3.0,
            "Aluminum with heating elements": 1.3
        }
        
        material_factor = material_factors.get(material, 1.0)
        
        # Volume discount
        volume_factor = max(0.5, 1.0 - (volume - 10) * 0.01)
        
        base_cost = 10000.0 + (base_area * 5000.0) * complexity_factor
        return base_cost * material_factor * volume_factor
    
    def model_autoclave_process(
        self,
        material: MaterialDefinition,
        part_thickness: float,
        fiber_volume_fraction: float = 0.6
    ) -> CureParameters:
        """Model autoclave curing process parameters."""
        self.logger.info("Modeling autoclave process parameters")
        
        # Get material constraints
        constraints = material.manufacturing_constraints
        if not constraints:
            raise ValueError("Material must have manufacturing constraints defined")
        
        # Base cure temperature from material or default
        cure_temp = constraints.cure_temperature or 450.0  # K
        cure_time = constraints.cure_time or 7200.0  # seconds (2 hours)
        
        # Adjust for part thickness
        thickness_factor = max(1.0, part_thickness / 0.005)  # 5mm reference
        adjusted_cure_time = cure_time * thickness_factor
        
        # Temperature profile
        temp_profile = [
            (0.0, 293.0),  # Room temperature start
            (30.0, cure_temp * 0.7),  # Ramp to 70% cure temp
            (60.0, cure_temp),  # Full cure temperature
            (60.0 + adjusted_cure_time / 60.0, cure_temp),  # Hold at cure temp
            (90.0 + adjusted_cure_time / 60.0, 293.0)  # Cool down
        ]
        
        # Pressure profile
        pressure_profile = [
            (0.0, 101325.0),  # Atmospheric pressure
            (15.0, 689475.0),  # Ramp to 100 psi
            (90.0 + adjusted_cure_time / 60.0, 689475.0),  # Hold pressure
            (95.0 + adjusted_cure_time / 60.0, 101325.0)  # Release pressure
        ]
        
        return CureParameters(
            temperature_profile=temp_profile,
            pressure_profile=pressure_profile,
            vacuum_level=84656.0,  # -25 inHg
            cure_time=adjusted_cure_time / 60.0,  # minutes
            post_cure_required=cure_temp < 450.0,
            post_cure_temperature=cure_temp + 20.0 if cure_temp < 450.0 else None,
            post_cure_time=120.0 if cure_temp < 450.0 else None
        )
    
    def model_oof_process(
        self,
        material: MaterialDefinition,
        part_thickness: float
    ) -> CureParameters:
        """Model out-of-autoclave curing process parameters."""
        self.logger.info("Modeling out-of-autoclave process parameters")
        
        constraints = material.manufacturing_constraints
        if not constraints:
            raise ValueError("Material must have manufacturing constraints defined")
        
        # OOA typically uses lower temperatures and longer times
        cure_temp = (constraints.cure_temperature or 450.0) - 30.0  # K
        cure_time = (constraints.cure_time or 7200.0) * 1.5  # 50% longer
        
        # Adjust for thickness
        thickness_factor = max(1.0, part_thickness / 0.003)  # 3mm reference for OOA
        adjusted_cure_time = cure_time * thickness_factor
        
        # Temperature profile (slower heating for OOA)
        temp_profile = [
            (0.0, 293.0),
            (45.0, cure_temp * 0.6),
            (90.0, cure_temp),
            (90.0 + adjusted_cure_time / 60.0, cure_temp),
            (120.0 + adjusted_cure_time / 60.0, 293.0)
        ]
        
        # Vacuum only (no pressure)
        pressure_profile = [
            (0.0, 101325.0),
            (120.0 + adjusted_cure_time / 60.0, 101325.0)
        ]
        
        return CureParameters(
            temperature_profile=temp_profile,
            pressure_profile=pressure_profile,
            vacuum_level=84656.0,
            cure_time=adjusted_cure_time / 60.0,
            post_cure_required=True,
            post_cure_temperature=cure_temp + 30.0,
            post_cure_time=180.0
        )
    
    def design_fiber_placement(
        self,
        part_geometry: Dict[str, Any],
        load_requirements: Dict[str, float],
        material: MaterialDefinition
    ) -> FiberPlacementParameters:
        """Design automated fiber placement parameters."""
        self.logger.info("Designing fiber placement parameters")
        
        # Analyze load requirements to determine fiber orientation
        primary_load = max(load_requirements.values())
        primary_direction = max(load_requirements, key=load_requirements.get)
        
        # Determine fiber angle based on primary load direction
        fiber_angles = {
            'tension_x': 0.0,
            'tension_y': 90.0,
            'shear_xy': 45.0,
            'compression_x': 0.0,
            'compression_y': 90.0
        }
        
        fiber_angle = fiber_angles.get(primary_direction, 0.0)
        
        # Calculate placement parameters
        part_area = part_geometry.get('length', 1.0) * part_geometry.get('width', 1.0)
        complexity = part_geometry.get('complexity_factor', 1.0)
        
        # Adjust parameters based on part size and complexity
        tow_width = max(3.175, min(12.7, 25.4 / complexity))  # 1/8" to 1/2"
        placement_speed = max(500.0, 2000.0 / complexity)  # mm/min
        compaction_force = 100.0 + (primary_load / 1000.0)  # N
        
        # Number of plies based on thickness requirement
        target_thickness = part_geometry.get('thickness', 0.005)  # 5mm default
        ply_thickness = 0.000125  # 0.125mm typical ply thickness
        number_of_plies = max(4, int(target_thickness / ply_thickness))
        
        return FiberPlacementParameters(
            fiber_angle=fiber_angle,
            tow_width=tow_width,
            placement_speed=placement_speed,
            compaction_force=compaction_force,
            heating_temperature=373.0,  # 100°C for tack
            number_of_plies=number_of_plies
        )
    
    def analyze_material_waste(
        self,
        part_geometry: Dict[str, Any],
        material: MaterialDefinition,
        process_type: ProcessType,
        production_volume: int
    ) -> WasteAnalysis:
        """Analyze material waste for manufacturing process."""
        self.logger.info(f"Analyzing material waste for {process_type.value}")
        
        # Calculate part material requirements
        part_volume = (
            part_geometry.get('length', 1.0) *
            part_geometry.get('width', 1.0) *
            part_geometry.get('thickness', 0.005)
        )
        
        # Material density from thermal properties
        density = 1600.0  # kg/m³ default for carbon fiber composite
        if material.thermal_properties:
            density = material.thermal_properties.density
        
        part_mass = part_volume * density
        
        # Waste factors by process type
        waste_factors = {
            ProcessType.AUTOCLAVE: 0.15,  # 15% waste
            ProcessType.OUT_OF_AUTOCLAVE: 0.12,  # 12% waste
            ProcessType.RESIN_TRANSFER_MOLDING: 0.08,  # 8% waste
            ProcessType.AUTOMATED_FIBER_PLACEMENT: 0.05,  # 5% waste
            ProcessType.HAND_LAYUP: 0.25,  # 25% waste
            ProcessType.FILAMENT_WINDING: 0.10  # 10% waste
        }
        
        waste_factor = waste_factors.get(process_type, 0.15)
        
        # Adjust waste factor for production volume (learning curve)
        volume_factor = max(0.5, 1.0 - (production_volume - 1) * 0.01)
        adjusted_waste_factor = waste_factor * volume_factor
        
        # Calculate waste quantities
        usable_material = part_mass * production_volume
        waste_material = usable_material * adjusted_waste_factor / (1.0 - adjusted_waste_factor)
        total_material = usable_material + waste_material
        waste_percentage = (waste_material / total_material) * 100.0
        
        # Cost calculations
        material_cost_per_kg = 50.0  # Default $50/kg for carbon fiber
        if material.manufacturing_constraints:
            material_cost_per_kg = material.manufacturing_constraints.cost_per_kg
        
        waste_cost = waste_material * material_cost_per_kg
        
        # Recycling potential
        recycling_potential = {
            ProcessType.AUTOCLAVE: 0.3,  # 30% can be recycled
            ProcessType.OUT_OF_AUTOCLAVE: 0.3,
            ProcessType.RESIN_TRANSFER_MOLDING: 0.1,  # Thermoset, limited recycling
            ProcessType.AUTOMATED_FIBER_PLACEMENT: 0.4,  # Better material utilization
            ProcessType.HAND_LAYUP: 0.2,
            ProcessType.FILAMENT_WINDING: 0.3
        }.get(process_type, 0.2)
        
        disposal_cost = waste_material * (1.0 - recycling_potential) * 5.0  # $5/kg disposal
        
        return WasteAnalysis(
            material_id=material.material_id,
            total_material_required=total_material,
            usable_material=usable_material,
            waste_material=waste_material,
            waste_percentage=waste_percentage,
            waste_cost=waste_cost,
            recycling_potential=recycling_potential * 100.0,
            disposal_cost=disposal_cost
        )
    
    def estimate_manufacturing_cost(
        self,
        part_geometry: Dict[str, Any],
        material: MaterialDefinition,
        process_type: ProcessType,
        production_volume: int,
        tooling_requirements: List[ToolingRequirement],
        manufacturing_steps: List[ManufacturingStep]
    ) -> CostBreakdown:
        """Estimate comprehensive manufacturing costs."""
        self.logger.info("Estimating manufacturing costs")
        
        # Material costs
        waste_analysis = self.analyze_material_waste(
            part_geometry, material, process_type, production_volume
        )
        
        material_cost_per_kg = 50.0
        if material.manufacturing_constraints:
            material_cost_per_kg = material.manufacturing_constraints.cost_per_kg
        
        material_costs = {
            material.material_id: waste_analysis.total_material_required * material_cost_per_kg
        }
        
        # Labor costs
        labor_rate = 75.0  # USD per hour
        labor_costs = {}
        
        for step in manufacturing_steps:
            labor_costs[step.step_id] = step.labor_hours * labor_rate * production_volume
        
        # Tooling costs (amortized over production volume)
        tooling_costs = {}
        for tooling in tooling_requirements:
            amortized_cost = tooling.cost_estimate / max(production_volume, 1)
            tooling_costs[tooling.tooling_id] = amortized_cost * production_volume
        
        # Equipment costs (usage-based)
        equipment_costs = {}
        equipment_rates = {
            'autoclave': 200.0,  # USD per hour
            'oven': 50.0,
            'afp_machine': 300.0,
            'rtm_press': 150.0
        }
        
        for step in manufacturing_steps:
            for equipment in step.equipment_required:
                rate = equipment_rates.get(equipment.lower(), 100.0)
                equipment_costs[equipment] = (step.duration / 60.0) * rate * production_volume
        
        # Overhead costs (30% of direct costs)
        direct_costs = (
            sum(material_costs.values()) +
            sum(labor_costs.values()) +
            sum(equipment_costs.values())
        )
        overhead_costs = direct_costs * 0.30
        
        # Waste costs
        waste_costs = {material.material_id: waste_analysis.waste_cost}
        
        # Total cost calculation
        total_cost = (
            sum(material_costs.values()) +
            sum(labor_costs.values()) +
            sum(tooling_costs.values()) +
            sum(equipment_costs.values()) +
            overhead_costs +
            sum(waste_costs.values())
        )
        
        cost_per_unit = total_cost / max(production_volume, 1)
        
        return CostBreakdown(
            material_costs=material_costs,
            labor_costs=labor_costs,
            tooling_costs=tooling_costs,
            equipment_costs=equipment_costs,
            overhead_costs=overhead_costs,
            waste_costs=waste_costs,
            total_cost=total_cost,
            cost_per_unit=cost_per_unit
        )