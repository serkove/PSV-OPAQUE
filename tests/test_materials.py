"""Unit tests for materials functionality."""

import unittest
import math
from fighter_jet_sdk.common.data_models import (
    MaterialDefinition, EMProperties, ThermalProperties, 
    MechanicalProperties, ManufacturingConstraints
)
from fighter_jet_sdk.common.enums import MaterialType
from fighter_jet_sdk.common.materials_database import MaterialsDatabase, materials_db


class TestMaterialDefinition(unittest.TestCase):
    """Test MaterialDefinition class."""

    def setUp(self):
        """Set up test fixtures."""
        self.em_props = EMProperties(
            permittivity=complex(2.5, -0.1),
            permeability=complex(1.0, -0.05),
            conductivity=1e6,
            frequency_range=(8e9, 12e9),
            loss_tangent=0.02
        )
        
        self.thermal_props = ThermalProperties(
            thermal_conductivity=22.0,
            specific_heat=200.0,
            density=12800.0,
            melting_point=4273.0,
            operating_temp_range=(293.0, 3773.0)
        )
        
        self.mech_props = MechanicalProperties(
            youngs_modulus=450e9,
            poissons_ratio=0.17,
            yield_strength=800e6,
            ultimate_strength=1200e6,
            fatigue_limit=400e6,
            density=12800.0
        )
        
        self.mfg_constraints = ManufacturingConstraints(
            min_thickness=1e-3,
            max_thickness=50e-3,
            cure_temperature=2273.0,
            cure_time=14400.0,
            tooling_requirements=["hot_press", "inert_atmosphere"],
            cost_per_kg=100000.0
        )

    def test_creation(self):
        """Test basic creation of MaterialDefinition."""
        material = MaterialDefinition(
            name="Test Material",
            base_material_type=MaterialType.METAMATERIAL,
            electromagnetic_properties=self.em_props,
            thermal_properties=self.thermal_props,
            mechanical_properties=self.mech_props,
            manufacturing_constraints=self.mfg_constraints
        )
        
        self.assertEqual(material.name, "Test Material")
        self.assertEqual(material.base_material_type, MaterialType.METAMATERIAL)
        self.assertEqual(material.electromagnetic_properties, self.em_props)

    def test_validation_valid_material(self):
        """Test validation of valid material."""
        material = MaterialDefinition(
            name="Valid Material",
            base_material_type=MaterialType.METAMATERIAL,
            electromagnetic_properties=self.em_props,
            thermal_properties=self.thermal_props,
            mechanical_properties=self.mech_props,
            manufacturing_constraints=self.mfg_constraints
        )
        
        errors = material.validate_material()
        self.assertEqual(len(errors), 0)

    def test_validation_empty_name(self):
        """Test validation of material with empty name."""
        material = MaterialDefinition(name="")
        errors = material.validate_material()
        
        self.assertIn("Material must have a name", errors)

    def test_validation_metamaterial_requirements(self):
        """Test validation of metamaterial requirements."""
        material = MaterialDefinition(
            name="Metamaterial",
            base_material_type=MaterialType.METAMATERIAL
            # Missing electromagnetic properties
        )
        
        errors = material.validate_material()
        self.assertIn("Metamaterials must have electromagnetic properties defined", errors)

    def test_validation_uhtc_requirements(self):
        """Test validation of UHTC requirements."""
        # UHTC with low melting point
        low_temp_thermal = ThermalProperties(
            thermal_conductivity=22.0,
            specific_heat=200.0,
            density=12800.0,
            melting_point=1500.0,  # Too low for UHTC
            operating_temp_range=(293.0, 1400.0)
        )
        
        material = MaterialDefinition(
            name="Low Temp UHTC",
            base_material_type=MaterialType.ULTRA_HIGH_TEMP_CERAMIC,
            thermal_properties=low_temp_thermal
        )
        
        errors = material.validate_material()
        self.assertIn("Ultra-high temperature ceramics must have melting point > 2000°C", errors)

    def test_validation_conductive_polymer_requirements(self):
        """Test validation of conductive polymer requirements."""
        # Conductive polymer with zero conductivity
        bad_em_props = EMProperties(
            permittivity=complex(3.5, -0.2),
            permeability=complex(1.0, 0.0),
            conductivity=0.0,  # Invalid for conductive polymer
            frequency_range=(1e6, 1e12),
            loss_tangent=0.05
        )
        
        material = MaterialDefinition(
            name="Non-conductive Polymer",
            base_material_type=MaterialType.CONDUCTIVE_POLYMER,
            electromagnetic_properties=bad_em_props
        )
        
        errors = material.validate_material()
        self.assertIn("Conductive polymers must have positive conductivity", errors)

    def test_validation_invalid_electromagnetic_properties(self):
        """Test validation of invalid electromagnetic properties."""
        bad_em_props = EMProperties(
            permittivity=complex(2.5, -0.1),
            permeability=complex(1.0, -0.05),
            conductivity=-1e6,  # Invalid negative conductivity
            frequency_range=(12e9, 8e9),  # Invalid range (min > max)
            loss_tangent=-0.02  # Invalid negative loss tangent
        )
        
        material = MaterialDefinition(
            name="Bad EM Material",
            electromagnetic_properties=bad_em_props
        )
        
        errors = material.validate_material()
        self.assertIn("Conductivity cannot be negative", errors)
        self.assertIn("Invalid frequency range: min must be less than max", errors)
        self.assertIn("Loss tangent cannot be negative", errors)

    def test_validation_invalid_thermal_properties(self):
        """Test validation of invalid thermal properties."""
        bad_thermal_props = ThermalProperties(
            thermal_conductivity=-22.0,  # Invalid negative
            specific_heat=0.0,  # Invalid zero
            density=-12800.0,  # Invalid negative
            melting_point=0.0,  # Invalid zero
            operating_temp_range=(3773.0, 293.0)  # Invalid range (min > max)
        )
        
        material = MaterialDefinition(
            name="Bad Thermal Material",
            thermal_properties=bad_thermal_props
        )
        
        errors = material.validate_material()
        self.assertIn("Thermal conductivity must be positive", errors)
        self.assertIn("Specific heat must be positive", errors)
        self.assertIn("Density must be positive", errors)
        self.assertIn("Melting point must be positive", errors)
        self.assertIn("Invalid operating temperature range", errors)

    def test_validation_invalid_mechanical_properties(self):
        """Test validation of invalid mechanical properties."""
        bad_mech_props = MechanicalProperties(
            youngs_modulus=-450e9,  # Invalid negative
            poissons_ratio=0.8,  # Invalid > 0.5
            yield_strength=1200e6,  # Greater than ultimate strength
            ultimate_strength=800e6,  # Less than yield strength
            fatigue_limit=0.0,  # Invalid zero
            density=0.0  # Invalid zero
        )
        
        material = MaterialDefinition(
            name="Bad Mechanical Material",
            mechanical_properties=bad_mech_props
        )
        
        errors = material.validate_material()
        self.assertIn("Young's modulus must be positive", errors)
        self.assertIn("Poisson's ratio must be between -1.0 and 0.5", errors)
        self.assertIn("Yield strength must be less than ultimate strength", errors)
        self.assertIn("Fatigue limit must be positive", errors)
        self.assertIn("Mechanical density must be positive", errors)

    def test_metamaterial_response_calculation(self):
        """Test metamaterial electromagnetic response calculation."""
        material = MaterialDefinition(
            name="Test Metamaterial",
            base_material_type=MaterialType.METAMATERIAL,
            electromagnetic_properties=self.em_props
        )
        
        # Test valid frequency
        frequency = 10e9  # 10 GHz (within range)
        response = material.calculate_metamaterial_response(frequency)
        
        self.assertIsInstance(response, complex)
        self.assertNotEqual(response, 0)

    def test_metamaterial_response_invalid_frequency(self):
        """Test metamaterial response with invalid frequency."""
        material = MaterialDefinition(
            name="Test Metamaterial",
            base_material_type=MaterialType.METAMATERIAL,
            electromagnetic_properties=self.em_props
        )
        
        # Test frequency outside range
        with self.assertRaises(ValueError):
            material.calculate_metamaterial_response(20e9)  # Outside range

    def test_metamaterial_response_non_metamaterial(self):
        """Test metamaterial response on non-metamaterial."""
        material = MaterialDefinition(
            name="Regular Material",
            base_material_type=MaterialType.CONVENTIONAL_METAL,
            electromagnetic_properties=self.em_props
        )
        
        with self.assertRaises(ValueError):
            material.calculate_metamaterial_response(10e9)

    def test_stealth_effectiveness_calculation(self):
        """Test stealth effectiveness calculation."""
        material = MaterialDefinition(
            name="Stealth Material",
            base_material_type=MaterialType.STEALTH_COATING,
            electromagnetic_properties=self.em_props
        )
        
        effectiveness = material.calculate_stealth_effectiveness(10e9, 5e-3)  # 5mm thickness
        
        self.assertIsInstance(effectiveness, float)
        self.assertGreaterEqual(effectiveness, 0.0)
        self.assertLessEqual(effectiveness, 1.0)

    def test_thermal_stress_calculation(self):
        """Test thermal stress calculation."""
        material = MaterialDefinition(
            name="Thermal Material",
            thermal_properties=self.thermal_props,
            mechanical_properties=self.mech_props
        )
        
        stress = material.calculate_thermal_stress(100.0)  # 100K gradient
        
        self.assertIsInstance(stress, float)
        self.assertGreater(stress, 0)

    def test_temperature_suitability(self):
        """Test temperature suitability check."""
        material = MaterialDefinition(
            name="High Temp Material",
            thermal_properties=self.thermal_props
        )
        
        # Test temperature within range
        self.assertTrue(material.is_suitable_for_temperature(1000.0))
        
        # Test temperature outside range
        self.assertFalse(material.is_suitable_for_temperature(5000.0))

    def test_manufacturing_cost_calculation(self):
        """Test manufacturing cost calculation."""
        material = MaterialDefinition(
            name="Cost Material",
            mechanical_properties=self.mech_props,
            manufacturing_constraints=self.mfg_constraints
        )
        
        volume = 0.001  # 1 liter
        cost = material.calculate_manufacturing_cost(volume, complexity_factor=1.5)
        
        self.assertIsInstance(cost, float)
        self.assertGreater(cost, 0)

    def test_serialization(self):
        """Test material serialization and deserialization."""
        material = MaterialDefinition(
            name="Serialization Test",
            base_material_type=MaterialType.METAMATERIAL,
            electromagnetic_properties=self.em_props,
            thermal_properties=self.thermal_props
        )
        
        # Test to_dict
        data = material.to_dict()
        self.assertIsInstance(data, dict)
        self.assertEqual(data['name'], "Serialization Test")
        self.assertEqual(data['base_material_type'], "METAMATERIAL")
        
        # Test from_dict
        restored_material = MaterialDefinition.from_dict(data)
        self.assertEqual(restored_material.name, material.name)
        self.assertEqual(restored_material.base_material_type, material.base_material_type)


class TestMaterialsDatabase(unittest.TestCase):
    """Test MaterialsDatabase class."""

    def setUp(self):
        """Set up test fixtures."""
        self.db = MaterialsDatabase()

    def test_database_initialization(self):
        """Test that database initializes with default materials."""
        materials = self.db.list_all_materials()
        
        self.assertGreater(len(materials), 0)
        self.assertIn("fss_xband", materials)
        self.assertIn("ram_iron_ball", materials)

    def test_get_material(self):
        """Test getting material by ID."""
        material = self.db.get_material("fss_xband")
        
        self.assertIsNotNone(material)
        self.assertEqual(material.base_material_type, MaterialType.METAMATERIAL)

    def test_get_nonexistent_material(self):
        """Test getting nonexistent material."""
        material = self.db.get_material("nonexistent")
        
        self.assertIsNone(material)

    def test_get_materials_by_type(self):
        """Test getting materials by type."""
        metamaterials = self.db.get_materials_by_type(MaterialType.METAMATERIAL)
        
        self.assertGreater(len(metamaterials), 0)
        for material in metamaterials:
            self.assertEqual(material.base_material_type, MaterialType.METAMATERIAL)

    def test_get_stealth_materials(self):
        """Test getting stealth materials."""
        stealth_materials = self.db.get_stealth_materials()
        
        self.assertGreater(len(stealth_materials), 0)
        for material in stealth_materials:
            self.assertIn(material.base_material_type, 
                         [MaterialType.STEALTH_COATING, MaterialType.METAMATERIAL])

    def test_get_high_temperature_materials(self):
        """Test getting high temperature materials."""
        high_temp_materials = self.db.get_high_temperature_materials(2000.0)  # 2000K
        
        self.assertGreater(len(high_temp_materials), 0)
        for material in high_temp_materials:
            self.assertIsNotNone(material.thermal_properties)
            self.assertGreaterEqual(material.thermal_properties.operating_temp_range[1], 2000.0)

    def test_add_material(self):
        """Test adding new material to database."""
        new_material = MaterialDefinition(
            name="Test Addition",
            base_material_type=MaterialType.CONVENTIONAL_METAL
        )
        
        success = self.db.add_material("test_addition", new_material)
        self.assertTrue(success)
        
        retrieved = self.db.get_material("test_addition")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "Test Addition")

    def test_add_duplicate_material(self):
        """Test adding duplicate material."""
        new_material = MaterialDefinition(
            name="Duplicate Test",
            base_material_type=MaterialType.CONVENTIONAL_METAL
        )
        
        # Add first time
        success1 = self.db.add_material("duplicate_test", new_material)
        self.assertTrue(success1)
        
        # Try to add again
        success2 = self.db.add_material("duplicate_test", new_material)
        self.assertFalse(success2)

    def test_search_materials(self):
        """Test searching materials by name."""
        results = self.db.search_materials("carbon")
        
        self.assertGreater(len(results), 0)
        for material_id, material in results:
            self.assertIn("carbon", material.name.lower())

    def test_calculate_stealth_coating_thickness(self):
        """Test stealth coating thickness calculation."""
        thickness = self.db.calculate_stealth_coating_thickness(
            "ram_iron_ball", 10e9, target_absorption=0.9
        )
        
        self.assertIsNotNone(thickness)
        self.assertGreater(thickness, 0)

    def test_calculate_thickness_invalid_material(self):
        """Test thickness calculation with invalid material."""
        thickness = self.db.calculate_stealth_coating_thickness(
            "nonexistent", 10e9
        )
        
        self.assertIsNone(thickness)

    def test_global_database_instance(self):
        """Test that global database instance is available."""
        self.assertIsInstance(materials_db, MaterialsDatabase)
        
        # Should have same materials as new instance
        materials = materials_db.list_all_materials()
        self.assertGreater(len(materials), 0)


if __name__ == '__main__':
    unittest.main()