"""Materials Engine for advanced materials modeling."""

from typing import Any, Dict, Optional, List
import numpy as np

from ...common.interfaces import BaseEngine
from ...common.data_models import MaterialDefinition, AircraftConfiguration
from ...common.materials_database import materials_db
from ...core.logging import get_engine_logger

from .metamaterial_modeler import MetamaterialModeler, FrequencyResponse, FSSSurface
from .stealth_analyzer import StealthAnalyzer, RCSData, GeometryModel, StealthConfiguration
from .thermal_materials_db import ThermalMaterialsDB, HypersonicConditions, ThermalAnalysisResult


class MaterialsEngine(BaseEngine):
    """Engine for advanced materials modeling and analysis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Materials Engine."""
        super().__init__(config)
        self.logger = get_engine_logger('materials')
        
        # Initialize specialized components
        self.metamaterial_modeler = None
        self.stealth_analyzer = None
        self.thermal_db = None
    
    def initialize(self) -> bool:
        """Initialize the Materials Engine."""
        try:
            self.logger.info("Initializing Materials Engine")
            
            # Initialize metamaterial modeler
            self.metamaterial_modeler = MetamaterialModeler()
            self.logger.info("Metamaterial modeler initialized")
            
            # Initialize stealth analyzer
            self.stealth_analyzer = StealthAnalyzer()
            self.logger.info("Stealth analyzer initialized")
            
            # Initialize thermal materials database
            self.thermal_db = ThermalMaterialsDB()
            self.logger.info("Thermal materials database initialized")
            
            self.initialized = True
            self.logger.info("Materials Engine initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Materials Engine: {e}")
            return False
    
    def validate_input(self, data: Any) -> bool:
        """Validate input data for Materials Engine."""
        if not self.initialized:
            self.logger.error("Materials Engine not initialized")
            return False
        
        # Basic validation - can be extended based on operation type
        if data is None:
            return False
        
        return True
    
    def process(self, data: Any) -> Any:
        """Process materials operations."""
        if not self.validate_input(data):
            return None
        
        # Process based on data type and operation
        if isinstance(data, dict):
            operation = data.get('operation')
            
            if operation == 'metamaterial_analysis':
                return self._process_metamaterial_analysis(data)
            elif operation == 'stealth_analysis':
                return self._process_stealth_analysis(data)
            elif operation == 'thermal_analysis':
                return self._process_thermal_analysis(data)
            elif operation == 'material_selection':
                return self._process_material_selection(data)
            else:
                self.logger.warning(f"Unknown operation: {operation}")
                return None
        
        return data
    
    def analyze_metamaterial_frequency_response(self, material: MaterialDefinition,
                                              frequencies: np.ndarray,
                                              thickness: float = 1e-3) -> FrequencyResponse:
        """Analyze metamaterial frequency response."""
        if not self.initialized or not self.metamaterial_modeler:
            raise RuntimeError("Materials Engine not properly initialized")
        
        return self.metamaterial_modeler.calculate_frequency_response(
            material, frequencies, thickness
        )
    
    def model_frequency_selective_surface(self, fss_config: FSSSurface,
                                        frequencies: np.ndarray) -> FrequencyResponse:
        """Model frequency selective surface response."""
        if not self.initialized or not self.metamaterial_modeler:
            raise RuntimeError("Materials Engine not properly initialized")
        
        return self.metamaterial_modeler.model_frequency_selective_surface(
            fss_config, frequencies
        )
    
    def calculate_ram_effectiveness(self, material: MaterialDefinition,
                                  thickness: float,
                                  frequencies: np.ndarray,
                                  incident_angle: float = 0.0) -> Dict[str, np.ndarray]:
        """Calculate radar absorption material effectiveness."""
        if not self.initialized or not self.metamaterial_modeler:
            raise RuntimeError("Materials Engine not properly initialized")
        
        return self.metamaterial_modeler.calculate_ram_effectiveness(
            material, thickness, frequencies, incident_angle
        )
    
    def optimize_ram_thickness(self, material: MaterialDefinition,
                              target_frequency: float,
                              target_absorption: float = 0.9) -> float:
        """Optimize RAM thickness for target absorption."""
        if not self.initialized or not self.metamaterial_modeler:
            raise RuntimeError("Materials Engine not properly initialized")
        
        return self.metamaterial_modeler.optimize_ram_thickness(
            material, target_frequency, target_absorption
        )
    
    def calculate_aircraft_rcs(self, geometry: GeometryModel,
                              materials_db: Dict[str, MaterialDefinition],
                              frequencies: np.ndarray,
                              angles: np.ndarray,
                              polarization: str = 'VV',
                              method: str = 'hybrid') -> RCSData:
        """Calculate aircraft radar cross-section."""
        if not self.initialized or not self.stealth_analyzer:
            raise RuntimeError("Materials Engine not properly initialized")
        
        return self.stealth_analyzer.calculate_aircraft_rcs(
            geometry, materials_db, frequencies, angles, polarization, method
        )
    
    def analyze_multi_frequency_rcs(self, geometry: GeometryModel,
                                   materials_db: Dict[str, MaterialDefinition],
                                   radar_bands: Optional[List[str]] = None) -> Dict[str, RCSData]:
        """Analyze RCS across multiple radar bands."""
        if not self.initialized or not self.stealth_analyzer:
            raise RuntimeError("Materials Engine not properly initialized")
        
        return self.stealth_analyzer.analyze_multi_frequency_rcs(
            geometry, materials_db, radar_bands
        )
    
    def optimize_stealth_configuration(self, geometry: GeometryModel,
                                     materials_db: Dict[str, MaterialDefinition],
                                     stealth_config: StealthConfiguration) -> Dict[str, str]:
        """Optimize material selection for stealth performance."""
        if not self.initialized or not self.stealth_analyzer:
            raise RuntimeError("Materials Engine not properly initialized")
        
        return self.stealth_analyzer.optimize_stealth_configuration(
            geometry, materials_db, stealth_config
        )
    
    def analyze_hypersonic_heating(self, material_id: str,
                                  conditions: HypersonicConditions,
                                  thickness: float = 0.01) -> ThermalAnalysisResult:
        """Analyze thermal response under hypersonic conditions."""
        if not self.initialized or not self.thermal_db:
            raise RuntimeError("Materials Engine not properly initialized")
        
        return self.thermal_db.analyze_hypersonic_heating(
            material_id, conditions, thickness
        )
    
    def get_uhtc_materials_for_temperature(self, temperature: float) -> List[str]:
        """Get UHTC materials suitable for given temperature."""
        if not self.initialized or not self.thermal_db:
            raise RuntimeError("Materials Engine not properly initialized")
        
        return self.thermal_db.get_materials_for_temperature(temperature)
    
    def optimize_thermal_material_selection(self, max_temperature: float,
                                          max_stress: float,
                                          weight_factor: float = 1.0) -> str:
        """Optimize material selection for thermal conditions."""
        if not self.initialized or not self.thermal_db:
            raise RuntimeError("Materials Engine not properly initialized")
        
        return self.thermal_db.optimize_material_selection(
            max_temperature, max_stress, weight_factor
        )
    
    def get_available_materials(self) -> Dict[str, str]:
        """Get all available materials from all databases."""
        available_materials = {}
        
        # Add materials from common database
        common_materials = materials_db.list_all_materials()
        available_materials.update(common_materials)
        
        # Add UHTC materials
        if self.thermal_db:
            uhtc_materials = self.thermal_db.list_materials()
            available_materials.update(uhtc_materials)
        
        return available_materials
    
    def validate_metamaterial_model(self, material: MaterialDefinition) -> Dict[str, float]:
        """Validate metamaterial model against benchmarks."""
        if not self.initialized or not self.metamaterial_modeler:
            raise RuntimeError("Materials Engine not properly initialized")
        
        return self.metamaterial_modeler.validate_against_benchmarks(material)
    
    def _process_metamaterial_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process metamaterial analysis request."""
        try:
            material_id = data.get('material_id')
            frequencies = np.array(data.get('frequencies', []))
            thickness = data.get('thickness', 1e-3)
            
            # Get material from database
            material = materials_db.get_material(material_id)
            if not material:
                return {'error': f'Material {material_id} not found'}
            
            # Perform analysis
            response = self.analyze_metamaterial_frequency_response(
                material, frequencies, thickness
            )
            
            return {
                'frequencies': response.frequencies.tolist(),
                'absorption': response.absorption.tolist(),
                'transmission': [complex(t).real for t in response.transmission],
                'reflection': [complex(r).real for r in response.reflection]
            }
            
        except Exception as e:
            self.logger.error(f"Metamaterial analysis failed: {e}")
            return {'error': str(e)}
    
    def _process_stealth_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process stealth analysis request."""
        try:
            # Extract geometry and analysis parameters
            geometry_data = data.get('geometry', {})
            frequencies = np.array(data.get('frequencies', []))
            angles = np.array(data.get('angles', []))
            
            # Create geometry model
            geometry = GeometryModel(**geometry_data)
            
            # Get materials database
            materials_dict = {}
            for mat_id in geometry.surface_materials.values():
                material = materials_db.get_material(mat_id)
                if material:
                    materials_dict[mat_id] = material
            
            # Perform RCS analysis
            rcs_data = self.calculate_aircraft_rcs(
                geometry, materials_dict, frequencies, angles
            )
            
            return {
                'frequencies': rcs_data.frequencies.tolist(),
                'angles': rcs_data.angles.tolist(),
                'rcs_matrix': rcs_data.rcs_matrix.tolist(),
                'polarization': rcs_data.polarization
            }
            
        except Exception as e:
            self.logger.error(f"Stealth analysis failed: {e}")
            return {'error': str(e)}
    
    def _process_thermal_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process thermal analysis request."""
        try:
            material_id = data.get('material_id')
            conditions_data = data.get('conditions', {})
            thickness = data.get('thickness', 0.01)
            
            # Create hypersonic conditions
            conditions = HypersonicConditions(**conditions_data)
            
            # Perform thermal analysis
            result = self.analyze_hypersonic_heating(
                material_id, conditions, thickness
            )
            
            return {
                'temperatures': result.temperatures.tolist(),
                'heat_flux': result.heat_flux.tolist(),
                'thermal_stress': result.thermal_stress.tolist(),
                'safety_factor': result.safety_factor.tolist(),
                'failure_mode': result.failure_mode
            }
            
        except Exception as e:
            self.logger.error(f"Thermal analysis failed: {e}")
            return {'error': str(e)}
    
    def _process_material_selection(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process material selection optimization."""
        try:
            selection_type = data.get('type', 'thermal')
            
            if selection_type == 'thermal':
                max_temp = data.get('max_temperature')
                max_stress = data.get('max_stress')
                weight_factor = data.get('weight_factor', 1.0)
                
                optimal_material = self.optimize_thermal_material_selection(
                    max_temp, max_stress, weight_factor
                )
                
                return {'optimal_material': optimal_material}
            
            elif selection_type == 'stealth':
                geometry_data = data.get('geometry', {})
                stealth_config_data = data.get('stealth_config', {})
                
                geometry = GeometryModel(**geometry_data)
                stealth_config = StealthConfiguration(**stealth_config_data)
                
                # Get materials database
                materials_dict = {}
                for surface, mat_list in stealth_config.material_constraints.items():
                    for mat_id in mat_list:
                        material = materials_db.get_material(mat_id)
                        if material:
                            materials_dict[mat_id] = material
                
                optimal_materials = self.optimize_stealth_configuration(
                    geometry, materials_dict, stealth_config
                )
                
                return {'optimal_materials': optimal_materials}
            
            else:
                return {'error': f'Unknown selection type: {selection_type}'}
                
        except Exception as e:
            self.logger.error(f"Material selection failed: {e}")
            return {'error': str(e)}