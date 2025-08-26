"""Materials database with predefined advanced materials."""

from typing import Dict, List, Optional
from .data_models import MaterialDefinition, EMProperties, ThermalProperties, MechanicalProperties, ManufacturingConstraints
from .enums import MaterialType


class MaterialsDatabase:
    """Database of predefined advanced materials."""
    
    def __init__(self):
        """Initialize the materials database."""
        self._materials: Dict[str, MaterialDefinition] = {}
        self._initialize_default_materials()
    
    def _initialize_default_materials(self):
        """Initialize database with default advanced materials."""
        
        # Metamaterial - Frequency Selective Surface
        fss_metamaterial = MaterialDefinition(
            name="FSS Metamaterial - X-Band",
            base_material_type=MaterialType.METAMATERIAL,
            electromagnetic_properties=EMProperties(
                permittivity=complex(2.5, -0.1),
                permeability=complex(1.0, -0.05),
                conductivity=1e6,  # S/m
                frequency_range=(8e9, 12e9),  # X-band
                loss_tangent=0.02
            ),
            mechanical_properties=MechanicalProperties(
                youngs_modulus=70e9,  # Pa
                poissons_ratio=0.33,
                yield_strength=200e6,  # Pa
                ultimate_strength=300e6,  # Pa
                fatigue_limit=100e6,  # Pa
                density=2700  # kg/m³
            ),
            manufacturing_constraints=ManufacturingConstraints(
                min_thickness=0.1e-3,  # 0.1 mm
                max_thickness=5e-3,    # 5 mm
                tooling_requirements=["precision_etching", "clean_room"],
                cost_per_kg=50000  # $/kg
            )
        )
        self._materials["fss_xband"] = fss_metamaterial
        
        # Stealth Coating - RAM (Radar Absorbing Material)
        ram_coating = MaterialDefinition(
            name="Iron Ball RAM Coating",
            base_material_type=MaterialType.STEALTH_COATING,
            electromagnetic_properties=EMProperties(
                permittivity=complex(12.0, -3.0),
                permeability=complex(2.5, -1.2),
                conductivity=1e3,  # S/m
                frequency_range=(1e9, 18e9),  # L to Ku band
                loss_tangent=0.25
            ),
            thermal_properties=ThermalProperties(
                thermal_conductivity=0.5,  # W/(m⋅K)
                specific_heat=800,  # J/(kg⋅K)
                density=1800,  # kg/m³
                melting_point=1773,  # K (1500°C)
                operating_temp_range=(223, 1273)  # -50°C to 1000°C
            ),
            mechanical_properties=MechanicalProperties(
                youngs_modulus=5e9,  # Pa
                poissons_ratio=0.4,
                yield_strength=50e6,  # Pa
                ultimate_strength=80e6,  # Pa
                fatigue_limit=25e6,  # Pa
                density=1800  # kg/m³
            ),
            manufacturing_constraints=ManufacturingConstraints(
                min_thickness=0.5e-3,  # 0.5 mm
                max_thickness=20e-3,   # 20 mm
                cure_temperature=423,  # K (150°C)
                cure_time=7200,  # 2 hours
                tooling_requirements=["spray_booth", "curing_oven"],
                cost_per_kg=1500  # $/kg
            )
        )
        self._materials["ram_iron_ball"] = ram_coating
        
        # Conductive Polymer
        conductive_polymer = MaterialDefinition(
            name="PEDOT:PSS Conductive Polymer",
            base_material_type=MaterialType.CONDUCTIVE_POLYMER,
            electromagnetic_properties=EMProperties(
                permittivity=complex(3.5, -0.2),
                permeability=complex(1.0, 0.0),
                conductivity=1e4,  # S/m
                frequency_range=(1e6, 1e12),  # MHz to THz
                loss_tangent=0.05
            ),
            thermal_properties=ThermalProperties(
                thermal_conductivity=0.2,  # W/(m⋅K)
                specific_heat=1200,  # J/(kg⋅K)
                density=1300,  # kg/m³
                melting_point=473,  # K (200°C)
                operating_temp_range=(233, 423)  # -40°C to 150°C
            ),
            mechanical_properties=MechanicalProperties(
                youngs_modulus=2e9,  # Pa
                poissons_ratio=0.35,
                yield_strength=30e6,  # Pa
                ultimate_strength=50e6,  # Pa
                fatigue_limit=15e6,  # Pa
                density=1300  # kg/m³
            ),
            manufacturing_constraints=ManufacturingConstraints(
                min_thickness=10e-6,  # 10 μm
                max_thickness=1e-3,   # 1 mm
                cure_temperature=353,  # K (80°C)
                cure_time=1800,  # 30 minutes
                tooling_requirements=["spin_coater", "low_temp_oven"],
                cost_per_kg=25000  # $/kg
            )
        )
        self._materials["pedot_pss"] = conductive_polymer
        
        # Ultra-High Temperature Ceramic
        uhtc = MaterialDefinition(
            name="Hafnium Carbide UHTC",
            base_material_type=MaterialType.ULTRA_HIGH_TEMP_CERAMIC,
            thermal_properties=ThermalProperties(
                thermal_conductivity=22,  # W/(m⋅K)
                specific_heat=200,  # J/(kg⋅K)
                density=12800,  # kg/m³
                melting_point=4273,  # K (4000°C)
                operating_temp_range=(293, 3773)  # 20°C to 3500°C
            ),
            mechanical_properties=MechanicalProperties(
                youngs_modulus=450e9,  # Pa
                poissons_ratio=0.17,
                yield_strength=800e6,  # Pa
                ultimate_strength=1200e6,  # Pa
                fatigue_limit=400e6,  # Pa
                density=12800  # kg/m³
            ),
            manufacturing_constraints=ManufacturingConstraints(
                min_thickness=1e-3,  # 1 mm
                max_thickness=50e-3,  # 50 mm
                cure_temperature=2273,  # K (2000°C)
                cure_time=14400,  # 4 hours
                tooling_requirements=["hot_press", "inert_atmosphere", "ultra_high_temp_furnace"],
                cost_per_kg=100000  # $/kg
            )
        )
        self._materials["hfc_uhtc"] = uhtc
        
        # Advanced Composite
        carbon_fiber = MaterialDefinition(
            name="T800 Carbon Fiber Composite",
            base_material_type=MaterialType.COMPOSITE,
            thermal_properties=ThermalProperties(
                thermal_conductivity=7.0,  # W/(m⋅K) - in fiber direction
                specific_heat=700,  # J/(kg⋅K)
                density=1600,  # kg/m³
                melting_point=3773,  # K (3500°C) - decomposition
                operating_temp_range=(173, 473)  # -100°C to 200°C
            ),
            mechanical_properties=MechanicalProperties(
                youngs_modulus=150e9,  # Pa - in fiber direction
                poissons_ratio=0.3,
                yield_strength=2500e6,  # Pa
                ultimate_strength=3000e6,  # Pa
                fatigue_limit=1500e6,  # Pa
                density=1600  # kg/m³
            ),
            manufacturing_constraints=ManufacturingConstraints(
                min_thickness=0.125e-3,  # 0.125 mm (single ply)
                max_thickness=100e-3,    # 100 mm
                cure_temperature=450,  # K (177°C)
                cure_time=7200,  # 2 hours
                tooling_requirements=["autoclave", "vacuum_bag", "tooling_surface"],
                cost_per_kg=200  # $/kg
            )
        )
        self._materials["t800_carbon"] = carbon_fiber
    
    def get_material(self, material_id: str) -> Optional[MaterialDefinition]:
        """Get material by ID."""
        return self._materials.get(material_id)
    
    def get_materials_by_type(self, material_type: MaterialType) -> List[MaterialDefinition]:
        """Get all materials of a specific type."""
        return [material for material in self._materials.values() 
                if material.base_material_type == material_type]
    
    def get_stealth_materials(self) -> List[MaterialDefinition]:
        """Get materials suitable for stealth applications."""
        stealth_types = [MaterialType.STEALTH_COATING, MaterialType.METAMATERIAL]
        return [material for material in self._materials.values() 
                if material.base_material_type in stealth_types]
    
    def get_high_temperature_materials(self, min_temperature: float) -> List[MaterialDefinition]:
        """Get materials suitable for high temperature applications."""
        suitable_materials = []
        for material in self._materials.values():
            if (material.thermal_properties and 
                material.thermal_properties.operating_temp_range[1] >= min_temperature):
                suitable_materials.append(material)
        return suitable_materials
    
    def add_material(self, material_id: str, material: MaterialDefinition) -> bool:
        """Add a new material to the database."""
        if material_id in self._materials:
            return False  # Material already exists
        
        # Validate material before adding
        errors = material.validate_material()
        if errors:
            raise ValueError(f"Invalid material: {errors}")
        
        self._materials[material_id] = material
        return True
    
    def list_all_materials(self) -> Dict[str, str]:
        """List all materials with their names."""
        return {material_id: material.name for material_id, material in self._materials.items()}
    
    def search_materials(self, name_pattern: str) -> List[tuple[str, MaterialDefinition]]:
        """Search materials by name pattern."""
        results = []
        pattern_lower = name_pattern.lower()
        
        for material_id, material in self._materials.items():
            if pattern_lower in material.name.lower():
                results.append((material_id, material))
        
        return results
    
    def calculate_stealth_coating_thickness(self, material_id: str, frequency: float, 
                                         target_absorption: float = 0.9) -> Optional[float]:
        """Calculate optimal thickness for stealth coating at given frequency."""
        material = self.get_material(material_id)
        if not material or material.base_material_type != MaterialType.STEALTH_COATING:
            return None
        
        if not material.electromagnetic_properties:
            return None
        
        # Simple quarter-wave thickness calculation
        # In practice, this would use optimization algorithms
        em = material.electromagnetic_properties
        
        # Check frequency is in valid range
        if not (em.frequency_range[0] <= frequency <= em.frequency_range[1]):
            return None
        
        # Calculate wavelength in material
        c = 3e8  # Speed of light
        epsilon_r = abs(em.permittivity)
        mu_r = abs(em.permeability)
        
        wavelength_material = c / (frequency * (epsilon_r * mu_r) ** 0.5)
        
        # Quarter-wave thickness for maximum absorption
        optimal_thickness = wavelength_material / 4
        
        # Check against manufacturing constraints
        if material.manufacturing_constraints:
            mfg = material.manufacturing_constraints
            optimal_thickness = max(optimal_thickness, mfg.min_thickness)
            optimal_thickness = min(optimal_thickness, mfg.max_thickness)
        
        return optimal_thickness


# Global instance
materials_db = MaterialsDatabase()