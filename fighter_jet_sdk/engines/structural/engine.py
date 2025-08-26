"""Structural Analysis Engine for extreme hypersonic conditions."""

from typing import Any, Dict, Optional, List, Tuple
import numpy as np

from ...common.interfaces import BaseEngine
from ...common.data_models import MaterialDefinition, AircraftConfiguration
from ...core.logging import get_engine_logger

from .thermal_stress_analyzer import (
    ThermalStressAnalyzer, ThermalLoadConditions, ThermalStressResults, StructuralGeometry
)
from .atmospheric_loads_analyzer import (
    AtmosphericLoadsAnalyzer, AtmosphericConditions, HypersonicLoadResults
)


class StructuralEngine(BaseEngine):
    """Engine for structural analysis under extreme hypersonic conditions."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Structural Engine."""
        super().__init__(config)
        self.logger = get_engine_logger('structural')
        
        # Initialize specialized analyzers
        self.thermal_stress_analyzer = None
        self.atmospheric_loads_analyzer = None
    
    def initialize(self) -> bool:
        """Initialize the Structural Engine."""
        try:
            self.logger.info("Initializing Structural Engine")
            
            # Initialize thermal stress analyzer
            self.thermal_stress_analyzer = ThermalStressAnalyzer()
            self.logger.info("Thermal stress analyzer initialized")
            
            # Initialize atmospheric loads analyzer
            self.atmospheric_loads_analyzer = AtmosphericLoadsAnalyzer()
            self.logger.info("Atmospheric loads analyzer initialized")
            
            self.initialized = True
            self.logger.info("Structural Engine initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Structural Engine: {e}")
            return False
    
    def validate_input(self, data: Any) -> bool:
        """Validate input data for Structural Engine."""
        if not self.initialized:
            self.logger.error("Structural Engine not initialized")
            return False
        
        # Basic validation - can be extended based on operation type
        if data is None:
            return False
        
        return True
    
    def process(self, data: Any) -> Any:
        """Process structural analysis operations."""
        if not self.validate_input(data):
            return None
        
        # Process based on data type and operation
        if isinstance(data, dict):
            operation = data.get('operation')
            
            if operation == 'thermal_stress_analysis':
                return self._process_thermal_stress_analysis(data)
            elif operation == 'atmospheric_loads_analysis':
                return self._process_atmospheric_loads_analysis(data)
            elif operation == 'coupled_analysis':
                return self._process_coupled_analysis(data)
            elif operation == 'safety_factor_analysis':
                return self._process_safety_factor_analysis(data)
            else:
                self.logger.warning(f"Unknown operation: {operation}")
                return None
        
        return data
    
    def analyze_thermal_stress(self,
                             geometry: StructuralGeometry,
                             materials: Dict[str, MaterialDefinition],
                             thermal_loads: ThermalLoadConditions,
                             analysis_type: str = 'steady_state') -> ThermalStressResults:
        """
        Analyze thermal stress for extreme temperature gradients.
        
        Args:
            geometry: Structural geometry definition
            materials: Material definitions with temperature-dependent properties
            thermal_loads: Thermal loading conditions
            analysis_type: 'steady_state' or 'transient'
            
        Returns:
            ThermalStressResults with comprehensive thermal stress analysis
        """
        if not self.initialized or not self.thermal_stress_analyzer:
            raise RuntimeError("Structural Engine not properly initialized")
        
        return self.thermal_stress_analyzer.analyze_thermal_stress(
            geometry, materials, thermal_loads, analysis_type
        )
    
    def calculate_thermal_expansion_effects(self,
                                          geometry: StructuralGeometry,
                                          materials: Dict[str, MaterialDefinition],
                                          temperature_change: np.ndarray,
                                          reference_temperature: float = 293.15) -> Dict[str, np.ndarray]:
        """
        Calculate thermal expansion effects for large temperature differences.
        
        Args:
            geometry: Structural geometry
            materials: Material definitions
            temperature_change: Temperature change from reference [K]
            reference_temperature: Reference temperature [K]
            
        Returns:
            Dictionary with thermal expansion results
        """
        if not self.initialized or not self.thermal_stress_analyzer:
            raise RuntimeError("Structural Engine not properly initialized")
        
        return self.thermal_stress_analyzer.calculate_thermal_expansion_effects(
            geometry, materials, temperature_change, reference_temperature
        )
    
    def perform_coupled_thermal_structural_analysis(self,
                                                   geometry: StructuralGeometry,
                                                   materials: Dict[str, MaterialDefinition],
                                                   thermal_loads: ThermalLoadConditions,
                                                   mechanical_loads: Dict[str, np.ndarray],
                                                   coupling_iterations: int = 10,
                                                   convergence_tolerance: float = 1e-6) -> ThermalStressResults:
        """
        Perform coupled thermal-structural analysis with iteration.
        
        Args:
            geometry: Structural geometry
            materials: Material definitions
            thermal_loads: Thermal loading conditions
            mechanical_loads: Mechanical loading conditions
            coupling_iterations: Maximum coupling iterations
            convergence_tolerance: Convergence tolerance for coupling
            
        Returns:
            Coupled thermal-structural analysis results
        """
        if not self.initialized or not self.thermal_stress_analyzer:
            raise RuntimeError("Structural Engine not properly initialized")
        
        return self.thermal_stress_analyzer.perform_coupled_thermal_structural_analysis(
            geometry, materials, thermal_loads, mechanical_loads, 
            coupling_iterations, convergence_tolerance
        )
    
    def analyze_hypersonic_loads(self,
                               altitude_range: Tuple[float, float],
                               mach_number: float,
                               aircraft_geometry: Optional[Dict[str, Any]] = None,
                               load_cases: Optional[List[str]] = None) -> HypersonicLoadResults:
        """
        Analyze structural loads for hypersonic flight conditions.
        
        Args:
            altitude_range: (min_alt, max_alt) in meters (30-80 km range)
            mach_number: Mach number (up to Mach 60)
            aircraft_geometry: Aircraft geometry parameters
            load_cases: List of load cases to analyze
            
        Returns:
            HypersonicLoadResults with comprehensive load analysis
        """
        if not self.initialized or not self.atmospheric_loads_analyzer:
            raise RuntimeError("Structural Engine not properly initialized")
        
        return self.atmospheric_loads_analyzer.analyze_hypersonic_loads(
            altitude_range, mach_number, aircraft_geometry, load_cases
        )
    
    def calculate_dynamic_pressure_envelope(self,
                                          altitude_range: Tuple[float, float],
                                          mach_range: Tuple[float, float],
                                          n_points: int = 50) -> Dict[str, np.ndarray]:
        """
        Calculate dynamic pressure envelope for flight envelope.
        
        Args:
            altitude_range: (min_alt, max_alt) in meters
            mach_range: (min_mach, max_mach)
            n_points: Number of points for each dimension
            
        Returns:
            Dictionary with altitude, mach, and dynamic pressure grids
        """
        if not self.initialized or not self.atmospheric_loads_analyzer:
            raise RuntimeError("Structural Engine not properly initialized")
        
        return self.atmospheric_loads_analyzer.calculate_dynamic_pressure_envelope(
            altitude_range, mach_range, n_points
        )
    
    def analyze_safety_factors(self,
                             structural_loads: Any,  # StructuralLoads from atmospheric analyzer
                             material_properties: Dict[str, Dict[str, float]],
                             design_factors: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Calculate safety factors for extreme conditions.
        
        Args:
            structural_loads: Calculated structural loads
            material_properties: Material strength properties
            design_factors: Design safety factors by component
            
        Returns:
            Dictionary of safety factors by component
        """
        if not self.initialized or not self.atmospheric_loads_analyzer:
            raise RuntimeError("Structural Engine not properly initialized")
        
        return self.atmospheric_loads_analyzer.analyze_safety_factors(
            structural_loads, material_properties, design_factors
        )
    
    def optimize_flight_profile(self,
                              target_altitude: float,
                              max_dynamic_pressure: float,
                              mach_range: Tuple[float, float]) -> Dict[str, Any]:
        """
        Optimize flight profile to minimize structural loads.
        
        Args:
            target_altitude: Target cruise altitude [m]
            max_dynamic_pressure: Maximum allowable dynamic pressure [Pa]
            mach_range: (min_mach, max_mach) range
            
        Returns:
            Optimized flight profile parameters
        """
        if not self.initialized or not self.atmospheric_loads_analyzer:
            raise RuntimeError("Structural Engine not properly initialized")
        
        return self.atmospheric_loads_analyzer.optimize_flight_profile(
            target_altitude, max_dynamic_pressure, mach_range
        )
    
    def validate_structural_design(self,
                                 geometry: StructuralGeometry,
                                 materials: Dict[str, MaterialDefinition],
                                 flight_conditions: Dict[str, Any],
                                 safety_requirements: Dict[str, float]) -> Dict[str, Any]:
        """
        Comprehensive structural design validation for extreme conditions.
        
        Args:
            geometry: Structural geometry
            materials: Material definitions
            flight_conditions: Flight conditions including thermal and atmospheric loads
            safety_requirements: Required safety factors
            
        Returns:
            Validation results with pass/fail status and recommendations
        """
        self.logger.info("Performing comprehensive structural design validation")
        
        try:
            validation_results = {
                'overall_status': 'PASS',
                'thermal_analysis': {},
                'atmospheric_analysis': {},
                'safety_analysis': {},
                'recommendations': []
            }
            
            # Thermal stress analysis
            if 'thermal_loads' in flight_conditions:
                thermal_loads = flight_conditions['thermal_loads']
                thermal_results = self.analyze_thermal_stress(
                    geometry, materials, thermal_loads
                )
                
                validation_results['thermal_analysis'] = {
                    'max_temperature': thermal_results.max_temperature,
                    'max_stress': thermal_results.max_stress,
                    'min_safety_factor': np.min(thermal_results.safety_factor),
                    'failure_locations': thermal_results.failure_locations,
                    'critical_regions': thermal_results.critical_regions
                }
                
                # Check thermal safety requirements
                min_thermal_sf = safety_requirements.get('thermal_safety_factor', 1.5)
                if np.min(thermal_results.safety_factor) < min_thermal_sf:
                    validation_results['overall_status'] = 'FAIL'
                    validation_results['recommendations'].append(
                        f"Thermal safety factor below requirement: {np.min(thermal_results.safety_factor):.2f} < {min_thermal_sf}"
                    )
            
            # Atmospheric loads analysis
            if 'altitude_range' in flight_conditions and 'mach_number' in flight_conditions:
                altitude_range = flight_conditions['altitude_range']
                mach_number = flight_conditions['mach_number']
                
                atmospheric_results = self.analyze_hypersonic_loads(
                    altitude_range, mach_number
                )
                
                validation_results['atmospheric_analysis'] = {
                    'max_dynamic_pressure': atmospheric_results.max_dynamic_pressure,
                    'max_load_factor': atmospheric_results.max_load_factor,
                    'structural_margins': atmospheric_results.structural_margins,
                    'critical_locations': atmospheric_results.critical_load_locations
                }
                
                # Check atmospheric load safety requirements
                min_structural_margin = safety_requirements.get('structural_margin', 0.5)
                for component, margin in atmospheric_results.structural_margins.items():
                    if margin < min_structural_margin:
                        validation_results['overall_status'] = 'FAIL'
                        validation_results['recommendations'].append(
                            f"{component} structural margin below requirement: {margin:.2f} < {min_structural_margin}"
                        )
                
                # Add atmospheric load recommendations
                validation_results['recommendations'].extend(
                    atmospheric_results.recommended_modifications
                )
            
            # Overall safety analysis
            validation_results['safety_analysis'] = {
                'thermal_safety_met': validation_results['thermal_analysis'].get('min_safety_factor', float('inf')) >= safety_requirements.get('thermal_safety_factor', 1.5),
                'structural_safety_met': all(
                    margin >= safety_requirements.get('structural_margin', 0.5)
                    for margin in validation_results['atmospheric_analysis'].get('structural_margins', {}).values()
                ),
                'overall_safety_status': validation_results['overall_status']
            }
            
            self.logger.info(f"Structural validation complete. Status: {validation_results['overall_status']}")
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Structural design validation failed: {e}")
            raise
    
    def _process_thermal_stress_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process thermal stress analysis request."""
        try:
            # Extract geometry data
            geometry_data = data.get('geometry', {})
            # Convert lists to numpy arrays
            for key in ['nodes', 'elements', 'thickness']:
                if key in geometry_data and isinstance(geometry_data[key], list):
                    geometry_data[key] = np.array(geometry_data[key])
            geometry = StructuralGeometry(**geometry_data)
            
            # Extract materials
            materials = {}
            for mat_id, mat_data in data.get('materials', {}).items():
                # Convert nested properties
                if 'thermal_properties' in mat_data and mat_data['thermal_properties']:
                    from ...common.data_models import ThermalProperties
                    thermal_data = mat_data['thermal_properties'].copy()
                    # Convert list to tuple for operating_temp_range
                    if 'operating_temp_range' in thermal_data and isinstance(thermal_data['operating_temp_range'], list):
                        thermal_data['operating_temp_range'] = tuple(thermal_data['operating_temp_range'])
                    mat_data['thermal_properties'] = ThermalProperties(**thermal_data)
                
                if 'mechanical_properties' in mat_data and mat_data['mechanical_properties']:
                    from ...common.data_models import MechanicalProperties
                    mat_data['mechanical_properties'] = MechanicalProperties(**mat_data['mechanical_properties'])
                
                # Convert enum
                if 'base_material_type' in mat_data and isinstance(mat_data['base_material_type'], str):
                    from ...common.enums import MaterialType
                    mat_data['base_material_type'] = MaterialType[mat_data['base_material_type']]
                
                materials[mat_id] = MaterialDefinition(**mat_data)
            
            # Extract thermal loads
            thermal_loads_data = data.get('thermal_loads', {})
            # Convert lists to numpy arrays
            for key in ['temperature_distribution', 'temperature_gradient', 'heat_flux']:
                if key in thermal_loads_data and isinstance(thermal_loads_data[key], list):
                    thermal_loads_data[key] = np.array(thermal_loads_data[key])
            thermal_loads = ThermalLoadConditions(**thermal_loads_data)
            
            analysis_type = data.get('analysis_type', 'steady_state')
            
            # Perform analysis
            results = self.analyze_thermal_stress(
                geometry, materials, thermal_loads, analysis_type
            )
            
            return {
                'max_temperature': results.max_temperature,
                'max_stress': results.max_stress,
                'thermal_stress': results.thermal_stress.tolist(),
                'mechanical_stress': results.mechanical_stress.tolist(),
                'total_stress': results.total_stress.tolist(),
                'safety_factor': results.safety_factor.tolist(),
                'failure_locations': results.failure_locations,
                'critical_regions': results.critical_regions
            }
            
        except Exception as e:
            self.logger.error(f"Thermal stress analysis processing failed: {e}")
            return {'error': str(e)}
    
    def _process_atmospheric_loads_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process atmospheric loads analysis request."""
        try:
            altitude_range = tuple(data.get('altitude_range', [30000, 80000]))
            mach_number = data.get('mach_number', 60.0)
            aircraft_geometry = data.get('aircraft_geometry')
            load_cases = data.get('load_cases')
            
            # Perform analysis
            results = self.analyze_hypersonic_loads(
                altitude_range, mach_number, aircraft_geometry, load_cases
            )
            
            return {
                'max_dynamic_pressure': results.max_dynamic_pressure,
                'max_load_factor': results.max_load_factor,
                'structural_margins': results.structural_margins,
                'critical_locations': results.critical_load_locations,
                'recommendations': results.recommended_modifications,
                'atmospheric_conditions': {
                    'altitude': results.atmospheric_conditions.altitude,
                    'mach_number': results.atmospheric_conditions.mach_number,
                    'temperature': results.atmospheric_conditions.temperature,
                    'pressure': results.atmospheric_conditions.pressure,
                    'density': results.atmospheric_conditions.density
                }
            }
            
        except Exception as e:
            self.logger.error(f"Atmospheric loads analysis processing failed: {e}")
            return {'error': str(e)}
    
    def _process_coupled_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process coupled thermal-structural analysis request."""
        try:
            # Extract geometry and materials
            geometry_data = data.get('geometry', {})
            # Convert lists to numpy arrays
            for key in ['nodes', 'elements', 'thickness']:
                if key in geometry_data and isinstance(geometry_data[key], list):
                    geometry_data[key] = np.array(geometry_data[key])
            geometry = StructuralGeometry(**geometry_data)
            
            materials = {}
            for mat_id, mat_data in data.get('materials', {}).items():
                # Convert nested properties
                if 'thermal_properties' in mat_data and mat_data['thermal_properties']:
                    from ...common.data_models import ThermalProperties
                    thermal_data = mat_data['thermal_properties'].copy()
                    # Convert list to tuple for operating_temp_range
                    if 'operating_temp_range' in thermal_data and isinstance(thermal_data['operating_temp_range'], list):
                        thermal_data['operating_temp_range'] = tuple(thermal_data['operating_temp_range'])
                    mat_data['thermal_properties'] = ThermalProperties(**thermal_data)
                
                if 'mechanical_properties' in mat_data and mat_data['mechanical_properties']:
                    from ...common.data_models import MechanicalProperties
                    mat_data['mechanical_properties'] = MechanicalProperties(**mat_data['mechanical_properties'])
                
                # Convert enum
                if 'base_material_type' in mat_data and isinstance(mat_data['base_material_type'], str):
                    from ...common.enums import MaterialType
                    mat_data['base_material_type'] = MaterialType[mat_data['base_material_type']]
                
                materials[mat_id] = MaterialDefinition(**mat_data)
            
            # Extract loads
            thermal_loads_data = data.get('thermal_loads', {})
            # Convert lists to numpy arrays
            for key in ['temperature_distribution', 'temperature_gradient', 'heat_flux']:
                if key in thermal_loads_data and isinstance(thermal_loads_data[key], list):
                    thermal_loads_data[key] = np.array(thermal_loads_data[key])
            thermal_loads = ThermalLoadConditions(**thermal_loads_data)
            
            mechanical_loads = data.get('mechanical_loads', {})
            
            # Coupling parameters
            coupling_iterations = data.get('coupling_iterations', 10)
            convergence_tolerance = data.get('convergence_tolerance', 1e-6)
            
            # Perform coupled analysis
            results = self.perform_coupled_thermal_structural_analysis(
                geometry, materials, thermal_loads, mechanical_loads,
                coupling_iterations, convergence_tolerance
            )
            
            return {
                'max_temperature': results.max_temperature,
                'max_stress': results.max_stress,
                'thermal_stress': results.thermal_stress.tolist(),
                'mechanical_stress': results.mechanical_stress.tolist(),
                'total_stress': results.total_stress.tolist(),
                'displacement': results.displacement.tolist(),
                'safety_factor': results.safety_factor.tolist(),
                'failure_locations': results.failure_locations,
                'critical_regions': results.critical_regions
            }
            
        except Exception as e:
            self.logger.error(f"Coupled analysis processing failed: {e}")
            return {'error': str(e)}
    
    def _process_safety_factor_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process safety factor analysis request."""
        try:
            # This would extract structural loads from previous analysis
            # For now, return placeholder
            material_properties = data.get('material_properties', {})
            design_factors = data.get('design_factors')
            
            # Placeholder safety factors
            safety_factors = {
                'fuselage': 2.1,
                'wings': 1.8,
                'control_surfaces': 2.3,
                'engine_mounts': 1.6
            }
            
            return {
                'safety_factors': safety_factors,
                'overall_status': 'PASS' if min(safety_factors.values()) > 1.5 else 'FAIL'
            }
            
        except Exception as e:
            self.logger.error(f"Safety factor analysis processing failed: {e}")
            return {'error': str(e)}