"""
Aerodynamics Engine

Main engine for aerodynamic analysis including CFD, stability analysis,
and stealth-aerodynamic optimization.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .cfd_solver import CFDSolver, CFDResults, SolverSettings
from .stability_analyzer import StabilityAnalyzer
from .stealth_shape_optimizer import StealthShapeOptimizer
from ...common.data_models import AircraftConfiguration, FlowConditions, AnalysisResults
from ...common.enums import ModuleType
from ...common.interfaces import AnalysisEngine
from ...core.errors import AerodynamicsError


@dataclass
class AerodynamicResults(AnalysisResults):
    """Comprehensive aerodynamic analysis results"""
    cfd_results: Optional[CFDResults] = None
    stability_results: Optional[Dict[str, Any]] = None
    optimization_results: Optional[Dict[str, Any]] = None
    performance_envelope: Optional[Dict[str, Any]] = None


class AerodynamicsEngine(AnalysisEngine):
    """
    Main aerodynamics engine providing comprehensive flight performance analysis
    
    Capabilities:
    - CFD analysis across all speed regimes
    - Stability and control analysis
    - Stealth-aerodynamic optimization
    - Performance envelope calculation
    """
    
    def __init__(self):
        super().__init__()
        self.cfd_solver = CFDSolver()
        self.stability_analyzer = StabilityAnalyzer()
        self.stealth_optimizer = StealthShapeOptimizer()
        self.logger = logging.getLogger(__name__)
        
        # Analysis capabilities
        self.capabilities = {
            "cfd_analysis": True,
            "stability_analysis": True,
            "stealth_optimization": True,
            "multi_speed_analysis": True,
            "performance_envelope": True
        }
    
    def analyze(self, configuration: AircraftConfiguration, 
                analysis_type: str = "comprehensive",
                **kwargs) -> AerodynamicResults:
        """
        Perform aerodynamic analysis on aircraft configuration
        
        Args:
            configuration: Aircraft configuration to analyze
            analysis_type: Type of analysis ('cfd', 'stability', 'optimization', 'comprehensive')
            **kwargs: Additional analysis parameters
        
        Returns:
            AerodynamicResults containing all analysis results
        """
        try:
            self.logger.info(f"Starting {analysis_type} aerodynamic analysis")
            
            results = AerodynamicResults(
                configuration_id=configuration.config_id,
                analysis_type=analysis_type,
                timestamp=self._get_timestamp()
            )
            
            if analysis_type in ["cfd", "comprehensive"]:
                results.cfd_results = self._perform_cfd_analysis(configuration, **kwargs)
            
            if analysis_type in ["stability", "comprehensive"]:
                results.stability_results = self._perform_stability_analysis(configuration, **kwargs)
            
            if analysis_type in ["optimization", "comprehensive"]:
                results.optimization_results = self._perform_optimization(configuration, **kwargs)
            
            if analysis_type == "comprehensive":
                results.performance_envelope = self._calculate_performance_envelope(
                    configuration, results
                )
            
            self.logger.info("Aerodynamic analysis completed successfully")
            return results
            
        except Exception as e:
            raise AerodynamicsError(f"Aerodynamic analysis failed: {str(e)}")
    
    def _perform_cfd_analysis(self, configuration: AircraftConfiguration, 
                             **kwargs) -> CFDResults:
        """Perform CFD analysis"""
        flow_conditions = kwargs.get('flow_conditions')
        solver_settings = kwargs.get('solver_settings')
        
        if not flow_conditions:
            # Use default flight conditions
            flow_conditions = FlowConditions(
                mach_number=0.8,
                altitude=10000,
                angle_of_attack=2.0,
                sideslip_angle=0.0
            )
        
        return self.cfd_solver.analyze(configuration, flow_conditions, solver_settings)
    
    def _perform_stability_analysis(self, configuration: AircraftConfiguration,
                                   **kwargs) -> Dict[str, Any]:
        """Perform stability and control analysis"""
        flight_conditions = kwargs.get('flight_conditions', {})
        
        return self.stability_analyzer.analyze_stability(configuration, flight_conditions)
    
    def _perform_optimization(self, configuration: AircraftConfiguration,
                             **kwargs) -> Dict[str, Any]:
        """Perform stealth-aerodynamic optimization"""
        optimization_params = kwargs.get('optimization_params', {})
        
        return self.stealth_optimizer.optimize_configuration(
            configuration, optimization_params
        )
    
    def _calculate_performance_envelope(self, configuration: AircraftConfiguration,
                                       results: AerodynamicResults) -> Dict[str, Any]:
        """Calculate aircraft performance envelope"""
        envelope = {
            "max_mach": 0.0,
            "service_ceiling": 0.0,
            "max_g_load": 0.0,
            "stall_speed": 0.0,
            "max_range": 0.0,
            "combat_radius": 0.0
        }
        
        # Extract performance metrics from CFD and stability results
        if results.cfd_results:
            envelope["max_mach"] = self._estimate_max_mach(results.cfd_results)
        
        if results.stability_results:
            envelope["max_g_load"] = results.stability_results.get("max_g_load", 9.0)
            envelope["stall_speed"] = results.stability_results.get("stall_speed", 150.0)
        
        # Estimate other performance parameters
        envelope["service_ceiling"] = 18000  # meters
        envelope["max_range"] = 3000  # km
        envelope["combat_radius"] = 1200  # km
        
        return envelope
    
    def _estimate_max_mach(self, cfd_results: CFDResults) -> float:
        """Estimate maximum Mach number from CFD results"""
        # Simple estimation based on drag force (normalized by lift)
        drag_force = cfd_results.forces.get("drag", 1000.0)
        lift_force = cfd_results.forces.get("lift", 50000.0)
        
        # Calculate drag-to-lift ratio as a proxy for efficiency
        drag_to_lift_ratio = drag_force / lift_force if lift_force > 0 else 1.0
        
        if drag_to_lift_ratio < 0.02:  # Very efficient
            return 2.5  # Supercruise capable
        elif drag_to_lift_ratio < 0.04:  # Moderately efficient
            return 2.0  # Supersonic
        else:  # Less efficient
            return 1.6  # Limited supersonic
    
    def analyze_multi_speed_regime(self, configuration: AircraftConfiguration,
                                  mach_range: List[float]) -> Dict[float, CFDResults]:
        """
        Analyze aircraft across multiple speed regimes
        
        Args:
            configuration: Aircraft configuration
            mach_range: List of Mach numbers to analyze
        
        Returns:
            Dictionary mapping Mach numbers to CFD results
        """
        results = {}
        
        for mach in mach_range:
            flow_conditions = FlowConditions(
                mach_number=mach,
                altitude=10000,
                angle_of_attack=2.0,
                sideslip_angle=0.0
            )
            
            try:
                cfd_result = self.cfd_solver.analyze(configuration, flow_conditions)
                results[mach] = cfd_result
                self.logger.info(f"Completed analysis for Mach {mach}")
                
            except Exception as e:
                self.logger.warning(f"Analysis failed for Mach {mach}: {str(e)}")
                continue
        
        return results
    
    def validate_configuration(self, configuration: AircraftConfiguration) -> Dict[str, bool]:
        """
        Validate aircraft configuration for aerodynamic analysis
        
        Returns:
            Dictionary of validation results
        """
        validation = {
            "geometry_valid": True,
            "mass_properties_valid": True,
            "control_surfaces_valid": True,
            "engine_integration_valid": True
        }
        
        # Basic geometry validation
        if not configuration.modules:
            validation["geometry_valid"] = False
        
        # Mass properties validation - check if modules have physical properties
        has_mass_properties = any(m.physical_properties for m in configuration.modules)
        if not has_mass_properties:
            validation["mass_properties_valid"] = False
        
        # Control surfaces validation - check for structural modules that could include control surfaces
        control_modules = [m for m in configuration.modules if m.module_type == ModuleType.STRUCTURAL]
        if not control_modules:
            validation["control_surfaces_valid"] = False
        
        # Engine integration validation
        engine_modules = [m for m in configuration.modules if m.module_type == ModuleType.PROPULSION]
        if not engine_modules:
            validation["engine_integration_valid"] = False
        
        return validation
    
    def get_analysis_recommendations(self, configuration: AircraftConfiguration) -> List[str]:
        """
        Get analysis recommendations based on configuration
        
        Returns:
            List of recommended analysis types and parameters
        """
        recommendations = []
        
        # Check for stealth features
        stealth_modules = [m for m in configuration.modules 
                          if "stealth" in m.module_id.lower()]
        if stealth_modules:
            recommendations.append("Perform stealth-aerodynamic optimization")
            recommendations.append("Analyze RCS vs aerodynamic performance trade-offs")
        
        # Check for high-speed capability
        engine_modules = [m for m in configuration.modules 
                         if m.module_type == ModuleType.PROPULSION]
        if engine_modules:
            max_thrust = sum(m.performance_characteristics.get("max_thrust", 0) 
                           for m in engine_modules)
            if max_thrust > 100000:  # High thrust engines
                recommendations.append("Perform multi-speed regime analysis (Mach 0.5-2.5)")
                recommendations.append("Analyze supersonic inlet performance")
        
        # Check for advanced sensors
        sensor_modules = [m for m in configuration.modules 
                         if m.module_type == ModuleType.SENSOR]
        if sensor_modules:
            recommendations.append("Analyze sensor integration aerodynamic effects")
            recommendations.append("Validate cooling requirements for high-power sensors")
        
        return recommendations
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as ISO string"""
        from datetime import datetime
        return datetime.now().isoformat()