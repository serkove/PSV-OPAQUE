"""Atmospheric loads analysis for high-altitude hypersonic flight."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

from ...core.logging import get_engine_logger


@dataclass
class AtmosphericConditions:
    """Atmospheric conditions for structural load analysis."""
    altitude: float  # m, altitude above sea level
    mach_number: float  # Mach number
    temperature: Optional[float] = None  # K, atmospheric temperature
    pressure: Optional[float] = None  # Pa, atmospheric pressure
    density: Optional[float] = None  # kg/m³, atmospheric density
    speed_of_sound: Optional[float] = None  # m/s, speed of sound
    dynamic_pressure: Optional[float] = None  # Pa, dynamic pressure
    
    def __post_init__(self):
        """Calculate missing atmospheric properties."""
        if any(prop is None for prop in [self.temperature, self.pressure, self.density, 
                                        self.speed_of_sound, self.dynamic_pressure]):
            self._calculate_atmospheric_properties()
    
    def _calculate_atmospheric_properties(self):
        """Calculate atmospheric properties using extended atmosphere model."""
        # Extended atmosphere model for 0-100 km altitude
        if self.altitude <= 11000:  # Troposphere
            self._calculate_troposphere_properties()
        elif self.altitude <= 20000:  # Lower Stratosphere
            self._calculate_lower_stratosphere_properties()
        elif self.altitude <= 32000:  # Upper Stratosphere
            self._calculate_upper_stratosphere_properties()
        elif self.altitude <= 47000:  # Stratosphere
            self._calculate_stratosphere_properties()
        elif self.altitude <= 51000:  # Mesosphere
            self._calculate_lower_mesosphere_properties()
        elif self.altitude <= 71000:  # Upper Mesosphere
            self._calculate_upper_mesosphere_properties()
        else:  # Thermosphere (simplified)
            self._calculate_thermosphere_properties()
        
        # Calculate derived properties
        if self.speed_of_sound is None:
            self.speed_of_sound = np.sqrt(1.4 * 287.0 * self.temperature)
        
        if self.dynamic_pressure is None:
            velocity = self.mach_number * self.speed_of_sound
            self.dynamic_pressure = 0.5 * self.density * velocity**2
    
    def _calculate_troposphere_properties(self):
        """Calculate properties in troposphere (0-11 km)."""
        T0, P0, rho0 = 288.15, 101325.0, 1.225
        L = -0.0065  # K/m
        
        if self.temperature is None:
            self.temperature = T0 + L * self.altitude
        
        if self.pressure is None:
            self.pressure = P0 * (self.temperature / T0) ** (-9.80665 / (287.0 * L))
        
        if self.density is None:
            self.density = self.pressure / (287.0 * self.temperature)
    
    def _calculate_lower_stratosphere_properties(self):
        """Calculate properties in lower stratosphere (11-20 km)."""
        T11, P11 = 216.65, 22632.0
        
        if self.temperature is None:
            self.temperature = T11
        
        if self.pressure is None:
            self.pressure = P11 * np.exp(-9.80665 * (self.altitude - 11000) / (287.0 * T11))
        
        if self.density is None:
            self.density = self.pressure / (287.0 * self.temperature)
    
    def _calculate_upper_stratosphere_properties(self):
        """Calculate properties in upper stratosphere (20-32 km)."""
        T20, P20 = 216.65, 5474.9
        L = 0.001  # K/m
        
        if self.temperature is None:
            self.temperature = T20 + L * (self.altitude - 20000)
        
        if self.pressure is None:
            self.pressure = P20 * (self.temperature / T20) ** (-9.80665 / (287.0 * L))
        
        if self.density is None:
            self.density = self.pressure / (287.0 * self.temperature)
    
    def _calculate_stratosphere_properties(self):
        """Calculate properties in stratosphere (32-47 km)."""
        T32, P32 = 228.65, 868.02
        L = 0.0028  # K/m
        
        if self.temperature is None:
            self.temperature = T32 + L * (self.altitude - 32000)
        
        if self.pressure is None:
            self.pressure = P32 * (self.temperature / T32) ** (-9.80665 / (287.0 * L))
        
        if self.density is None:
            self.density = self.pressure / (287.0 * self.temperature)
    
    def _calculate_lower_mesosphere_properties(self):
        """Calculate properties in lower mesosphere (47-51 km)."""
        T47, P47 = 270.65, 110.91
        
        if self.temperature is None:
            self.temperature = T47
        
        if self.pressure is None:
            self.pressure = P47 * np.exp(-9.80665 * (self.altitude - 47000) / (287.0 * T47))
        
        if self.density is None:
            self.density = self.pressure / (287.0 * self.temperature)
    
    def _calculate_upper_mesosphere_properties(self):
        """Calculate properties in upper mesosphere (51-71 km)."""
        T51, P51 = 270.65, 66.94
        L = -0.0028  # K/m
        
        if self.temperature is None:
            self.temperature = T51 + L * (self.altitude - 51000)
        
        if self.pressure is None:
            self.pressure = P51 * (self.temperature / T51) ** (-9.80665 / (287.0 * L))
        
        if self.density is None:
            self.density = self.pressure / (287.0 * self.temperature)
    
    def _calculate_thermosphere_properties(self):
        """Calculate properties in thermosphere (>71 km) - simplified."""
        # Simplified exponential model for high altitude
        h_scale = 8400  # Scale height [m]
        T71, P71, rho71 = 214.65, 3.96, 6.42e-5
        
        if self.temperature is None:
            # Temperature increases in thermosphere
            self.temperature = T71 + 0.002 * (self.altitude - 71000)
        
        if self.pressure is None:
            self.pressure = P71 * np.exp(-(self.altitude - 71000) / h_scale)
        
        if self.density is None:
            self.density = rho71 * np.exp(-(self.altitude - 71000) / h_scale)


@dataclass
class StructuralLoads:
    """Structural loads from atmospheric conditions."""
    dynamic_pressure: float  # Pa, dynamic pressure
    normal_force: np.ndarray  # N, normal forces on surfaces
    shear_force: np.ndarray  # N, shear forces on surfaces
    bending_moment: np.ndarray  # N⋅m, bending moments
    torsional_moment: np.ndarray  # N⋅m, torsional moments
    pressure_distribution: np.ndarray  # Pa, pressure distribution
    load_factor: float  # g, load factor
    safety_factor: float  # dimensionless, safety factor


@dataclass
class HypersonicLoadResults:
    """Results from hypersonic structural load analysis."""
    atmospheric_conditions: AtmosphericConditions
    structural_loads: StructuralLoads
    critical_load_locations: List[Tuple[int, str]]  # locations and load types
    max_dynamic_pressure: float  # Pa
    max_load_factor: float  # g
    structural_margins: Dict[str, float]  # safety margins by component
    recommended_modifications: List[str]  # design recommendations


class AtmosphericLoadsAnalyzer:
    """Analyzer for atmospheric loads during high-altitude hypersonic flight."""
    
    def __init__(self):
        """Initialize atmospheric loads analyzer."""
        self.logger = get_engine_logger('structural.atmospheric_loads')
        
        # Reference aircraft geometry (simplified)
        self.reference_area = 100.0  # m², reference area
        self.reference_length = 20.0  # m, reference length
        self.wing_span = 15.0  # m, wing span
        
        # Load distribution factors
        self.load_distribution_factors = {
            'fuselage': 0.4,
            'wings': 0.35,
            'control_surfaces': 0.15,
            'engine_mounts': 0.1
        }
    
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
        self.logger.info(f"Analyzing hypersonic loads for Mach {mach_number} at {altitude_range[0]/1000:.1f}-{altitude_range[1]/1000:.1f} km")
        
        try:
            # Update geometry if provided
            if aircraft_geometry:
                self._update_geometry(aircraft_geometry)
            
            # Define default load cases if not provided
            if load_cases is None:
                load_cases = ['steady_flight', 'maneuver_2g', 'maneuver_4g', 'gust_encounter']
            
            # Find critical altitude (maximum dynamic pressure)
            critical_altitude = self._find_critical_altitude(altitude_range, mach_number)
            
            # Calculate atmospheric conditions at critical altitude
            atm_conditions = AtmosphericConditions(
                altitude=critical_altitude,
                mach_number=mach_number
            )
            
            # Calculate structural loads for all load cases
            max_loads = None
            max_load_factor = 0.0
            
            for load_case in load_cases:
                loads = self._calculate_structural_loads(atm_conditions, load_case)
                
                if max_loads is None or loads.load_factor > max_load_factor:
                    max_loads = loads
                    max_load_factor = loads.load_factor
            
            # Identify critical load locations
            critical_locations = self._identify_critical_load_locations(max_loads)
            
            # Calculate structural margins
            structural_margins = self._calculate_structural_margins(max_loads, atm_conditions)
            
            # Generate design recommendations
            recommendations = self._generate_design_recommendations(
                max_loads, structural_margins, atm_conditions
            )
            
            results = HypersonicLoadResults(
                atmospheric_conditions=atm_conditions,
                structural_loads=max_loads,
                critical_load_locations=critical_locations,
                max_dynamic_pressure=atm_conditions.dynamic_pressure,
                max_load_factor=max_load_factor,
                structural_margins=structural_margins,
                recommended_modifications=recommendations
            )
            
            self.logger.info(f"Hypersonic load analysis complete. Max dynamic pressure: {results.max_dynamic_pressure:.2e} Pa")
            return results
            
        except Exception as e:
            self.logger.error(f"Hypersonic load analysis failed: {e}")
            raise
    
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
        self.logger.info("Calculating dynamic pressure envelope")
        
        try:
            altitudes = np.linspace(altitude_range[0], altitude_range[1], n_points)
            mach_numbers = np.linspace(mach_range[0], mach_range[1], n_points)
            
            alt_grid, mach_grid = np.meshgrid(altitudes, mach_numbers)
            q_grid = np.zeros_like(alt_grid)
            
            for i in range(n_points):
                for j in range(n_points):
                    conditions = AtmosphericConditions(
                        altitude=alt_grid[i, j],
                        mach_number=mach_grid[i, j]
                    )
                    q_grid[i, j] = conditions.dynamic_pressure
            
            return {
                'altitude': alt_grid,
                'mach_number': mach_grid,
                'dynamic_pressure': q_grid,
                'max_q': np.max(q_grid),
                'max_q_location': np.unravel_index(np.argmax(q_grid), q_grid.shape)
            }
            
        except Exception as e:
            self.logger.error(f"Dynamic pressure envelope calculation failed: {e}")
            raise
    
    def analyze_safety_factors(self,
                             structural_loads: StructuralLoads,
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
        self.logger.info("Calculating safety factors for extreme conditions")
        
        try:
            if design_factors is None:
                design_factors = {
                    'fuselage': 1.5,
                    'wings': 2.0,
                    'control_surfaces': 1.8,
                    'engine_mounts': 2.5
                }
            
            safety_factors = {}
            
            for component, design_factor in design_factors.items():
                # Get material properties for component
                if component in material_properties:
                    mat_props = material_properties[component]
                    ultimate_strength = mat_props.get('ultimate_strength', 500e6)  # Pa
                    yield_strength = mat_props.get('yield_strength', 350e6)  # Pa
                else:
                    # Default values for high-strength materials
                    ultimate_strength = 500e6  # Pa
                    yield_strength = 350e6  # Pa
                
                # Calculate applied stress (simplified)
                load_factor = self.load_distribution_factors.get(component, 0.25)
                max_force = np.max(np.abs(structural_loads.normal_force)) * load_factor
                
                # Assume cross-sectional area (simplified)
                cross_section_area = 0.1  # m²
                applied_stress = max_force / cross_section_area
                
                # Calculate safety factors
                sf_ultimate = ultimate_strength / applied_stress
                sf_yield = yield_strength / applied_stress
                
                # Use minimum safety factor
                sf_design = min(sf_ultimate, sf_yield) / design_factor
                
                safety_factors[component] = sf_design
            
            return safety_factors
            
        except Exception as e:
            self.logger.error(f"Safety factor calculation failed: {e}")
            raise
    
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
        self.logger.info(f"Optimizing flight profile for altitude {target_altitude/1000:.1f} km")
        
        try:
            def objective(mach):
                """Objective function to minimize (negative of efficiency)."""
                conditions = AtmosphericConditions(
                    altitude=target_altitude,
                    mach_number=mach
                )
                
                # Penalize if dynamic pressure exceeds limit
                if conditions.dynamic_pressure > max_dynamic_pressure:
                    return 1e6  # Large penalty
                
                # Efficiency metric (simplified)
                efficiency = mach / (1 + conditions.dynamic_pressure / max_dynamic_pressure)
                return -efficiency  # Minimize negative efficiency
            
            # Optimize Mach number
            result = minimize_scalar(
                objective,
                bounds=mach_range,
                method='bounded'
            )
            
            optimal_mach = result.x
            optimal_conditions = AtmosphericConditions(
                altitude=target_altitude,
                mach_number=optimal_mach
            )
            
            # Calculate trajectory parameters
            trajectory_params = {
                'optimal_mach': optimal_mach,
                'optimal_altitude': target_altitude,
                'dynamic_pressure': optimal_conditions.dynamic_pressure,
                'atmospheric_density': optimal_conditions.density,
                'flight_efficiency': -result.fun,
                'velocity': optimal_mach * optimal_conditions.speed_of_sound
            }
            
            self.logger.info(f"Optimal flight profile: Mach {optimal_mach:.1f} at {target_altitude/1000:.1f} km")
            return trajectory_params
            
        except Exception as e:
            self.logger.error(f"Flight profile optimization failed: {e}")
            raise
    
    def _update_geometry(self, aircraft_geometry: Dict[str, Any]) -> None:
        """Update aircraft geometry parameters."""
        self.reference_area = aircraft_geometry.get('reference_area', self.reference_area)
        self.reference_length = aircraft_geometry.get('reference_length', self.reference_length)
        self.wing_span = aircraft_geometry.get('wing_span', self.wing_span)
        
        # Update load distribution if provided
        if 'load_distribution' in aircraft_geometry:
            self.load_distribution_factors.update(aircraft_geometry['load_distribution'])
    
    def _find_critical_altitude(self, altitude_range: Tuple[float, float], mach_number: float) -> float:
        """Find altitude with maximum dynamic pressure."""
        altitudes = np.linspace(altitude_range[0], altitude_range[1], 100)
        max_q = 0.0
        critical_alt = altitude_range[0]
        
        for alt in altitudes:
            conditions = AtmosphericConditions(altitude=alt, mach_number=mach_number)
            if conditions.dynamic_pressure > max_q:
                max_q = conditions.dynamic_pressure
                critical_alt = alt
        
        return critical_alt
    
    def _calculate_structural_loads(self, conditions: AtmosphericConditions, load_case: str) -> StructuralLoads:
        """Calculate structural loads for given conditions and load case."""
        # Load case multipliers
        load_multipliers = {
            'steady_flight': 1.0,
            'maneuver_2g': 2.0,
            'maneuver_4g': 4.0,
            'gust_encounter': 2.5
        }
        
        load_factor = load_multipliers.get(load_case, 1.0)
        
        # Calculate forces (simplified)
        base_force = conditions.dynamic_pressure * self.reference_area
        
        # Distribute loads across components
        n_components = 10  # Simplified structural model
        normal_force = np.full(n_components, base_force * load_factor / n_components)
        shear_force = np.full(n_components, base_force * load_factor * 0.1 / n_components)
        
        # Calculate moments
        bending_moment = normal_force * self.reference_length * 0.5
        torsional_moment = shear_force * self.wing_span * 0.25
        
        # Pressure distribution (simplified)
        pressure_distribution = np.full(n_components, conditions.dynamic_pressure * load_factor)
        
        # Safety factor (preliminary)
        safety_factor = 1.5 / load_factor  # Decreases with load factor
        
        return StructuralLoads(
            dynamic_pressure=conditions.dynamic_pressure,
            normal_force=normal_force,
            shear_force=shear_force,
            bending_moment=bending_moment,
            torsional_moment=torsional_moment,
            pressure_distribution=pressure_distribution,
            load_factor=load_factor,
            safety_factor=safety_factor
        )
    
    def _identify_critical_load_locations(self, loads: StructuralLoads) -> List[Tuple[int, str]]:
        """Identify locations with critical loads."""
        critical_locations = []
        
        # Find maximum normal force location
        max_normal_idx = np.argmax(np.abs(loads.normal_force))
        critical_locations.append((max_normal_idx, 'normal_force'))
        
        # Find maximum bending moment location
        max_bending_idx = np.argmax(np.abs(loads.bending_moment))
        critical_locations.append((max_bending_idx, 'bending_moment'))
        
        # Find maximum torsional moment location
        max_torsion_idx = np.argmax(np.abs(loads.torsional_moment))
        critical_locations.append((max_torsion_idx, 'torsional_moment'))
        
        return critical_locations
    
    def _calculate_structural_margins(self, loads: StructuralLoads, conditions: AtmosphericConditions) -> Dict[str, float]:
        """Calculate structural margins for different components."""
        margins = {}
        
        # Material strength assumptions (high-strength aerospace materials)
        ultimate_strength = 500e6  # Pa
        yield_strength = 350e6  # Pa
        
        for component, load_factor in self.load_distribution_factors.items():
            # Calculate applied stress (simplified)
            max_force = np.max(np.abs(loads.normal_force)) * load_factor
            cross_section_area = 0.1  # m² (simplified)
            applied_stress = max_force / cross_section_area
            
            # Calculate margin
            margin = (yield_strength - applied_stress) / applied_stress
            margins[component] = margin
        
        return margins
    
    def _generate_design_recommendations(self, loads: StructuralLoads,
                                       margins: Dict[str, float],
                                       conditions: AtmosphericConditions) -> List[str]:
        """Generate design recommendations based on analysis."""
        recommendations = []
        
        # Check dynamic pressure (use loads.dynamic_pressure which is the actual applied pressure)
        if loads.dynamic_pressure > 50000:  # Pa
            recommendations.append("Consider active load alleviation system for high dynamic pressure")
        
        # Check structural margins
        for component, margin in margins.items():
            if margin < 0.5:  # Low margin
                recommendations.append(f"Increase {component} structural strength - low safety margin ({margin:.2f})")
            elif margin < 1.0:  # Moderate margin
                recommendations.append(f"Monitor {component} structural loads - moderate safety margin ({margin:.2f})")
        
        # Check load factors
        if loads.load_factor > 3.0:
            recommendations.append("Consider load limiting systems for high-g maneuvers")
        
        # Altitude-specific recommendations
        if conditions.altitude > 60000:  # Above 60 km
            recommendations.append("Implement thermal protection for high-altitude flight")
        
        # Mach-specific recommendations
        if conditions.mach_number > 25:
            recommendations.append("Consider plasma effects on structural loads at extreme Mach numbers")
        
        return recommendations