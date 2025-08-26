"""Thermal stress analysis for extreme temperature conditions."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from ...common.data_models import MaterialDefinition, ThermalProperties, MechanicalProperties
from ...core.logging import get_engine_logger


@dataclass
class ThermalLoadConditions:
    """Thermal loading conditions for structural analysis."""
    temperature_distribution: np.ndarray  # K, spatial temperature distribution
    temperature_gradient: np.ndarray  # K/m, temperature gradients
    heat_flux: np.ndarray  # W/m², heat flux distribution
    time_history: Optional[np.ndarray] = None  # s, time points for transient analysis
    boundary_conditions: Dict[str, float] = field(default_factory=dict)


@dataclass
class ThermalStressResults:
    """Results from thermal stress analysis."""
    thermal_stress: np.ndarray  # Pa, thermal stress distribution
    mechanical_stress: np.ndarray  # Pa, mechanical stress distribution
    total_stress: np.ndarray  # Pa, combined stress distribution
    thermal_strain: np.ndarray  # dimensionless, thermal strain
    mechanical_strain: np.ndarray  # dimensionless, mechanical strain
    displacement: np.ndarray  # m, displacement field
    safety_factor: np.ndarray  # dimensionless, safety factors
    failure_locations: List[Tuple[int, str]]  # locations and failure modes
    max_temperature: float  # K, maximum temperature
    max_stress: float  # Pa, maximum stress
    critical_regions: List[Dict[str, Any]]  # critical analysis regions


@dataclass
class StructuralGeometry:
    """Structural geometry definition for analysis."""
    nodes: np.ndarray  # Node coordinates [N x 3]
    elements: np.ndarray  # Element connectivity [E x nodes_per_element]
    element_type: str  # Element type (beam, shell, solid)
    thickness: Optional[np.ndarray] = None  # Element thickness for shells
    cross_section: Optional[Dict[str, float]] = None  # Cross-section properties for beams
    material_ids: Optional[np.ndarray] = None  # Material ID for each element


class ThermalStressAnalyzer:
    """Analyzer for thermal stress under extreme temperature conditions."""
    
    def __init__(self):
        """Initialize thermal stress analyzer."""
        self.logger = get_engine_logger('structural.thermal_stress')
        self.material_properties_cache = {}
        
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
            ThermalStressResults with stress, strain, and safety analysis
        """
        self.logger.info(f"Starting thermal stress analysis ({analysis_type})")
        
        try:
            # Validate inputs
            self._validate_thermal_inputs(geometry, materials, thermal_loads)
            
            # Calculate temperature-dependent material properties
            temp_dependent_props = self._calculate_temperature_dependent_properties(
                materials, thermal_loads.temperature_distribution
            )
            
            # Calculate thermal strains
            thermal_strain = self._calculate_thermal_strain(
                thermal_loads.temperature_distribution, temp_dependent_props
            )
            
            # Solve structural equilibrium with thermal loads
            if analysis_type == 'steady_state':
                displacement, mechanical_strain, mechanical_stress = self._solve_steady_state_thermal(
                    geometry, temp_dependent_props, thermal_strain, thermal_loads
                )
            else:
                displacement, mechanical_strain, mechanical_stress = self._solve_transient_thermal(
                    geometry, temp_dependent_props, thermal_strain, thermal_loads
                )
            
            # Calculate thermal stress
            thermal_stress = self._calculate_thermal_stress_field(
                temp_dependent_props, thermal_strain
            )
            
            # Calculate total stress
            total_stress = mechanical_stress + thermal_stress
            
            # Perform safety analysis
            safety_factor = self._calculate_safety_factors(
                total_stress, temp_dependent_props, thermal_loads.temperature_distribution
            )
            
            # Identify failure locations
            failure_locations = self._identify_failure_locations(
                total_stress, safety_factor, temp_dependent_props
            )
            
            # Identify critical regions
            critical_regions = self._identify_critical_regions(
                geometry, total_stress, thermal_loads.temperature_distribution, safety_factor
            )
            
            results = ThermalStressResults(
                thermal_stress=thermal_stress,
                mechanical_stress=mechanical_stress,
                total_stress=total_stress,
                thermal_strain=thermal_strain,
                mechanical_strain=mechanical_strain,
                displacement=displacement,
                safety_factor=safety_factor,
                failure_locations=failure_locations,
                max_temperature=np.max(thermal_loads.temperature_distribution),
                max_stress=np.max(np.abs(total_stress)),
                critical_regions=critical_regions
            )
            
            self.logger.info(f"Thermal stress analysis complete. Max stress: {results.max_stress:.2e} Pa")
            return results
            
        except Exception as e:
            self.logger.error(f"Thermal stress analysis failed: {e}")
            raise
    
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
        self.logger.info("Calculating thermal expansion effects")
        
        try:
            # Get thermal expansion coefficients
            expansion_coeffs = self._get_thermal_expansion_coefficients(
                materials, reference_temperature + temperature_change
            )
            
            # Calculate thermal strains due to expansion
            thermal_strain_expansion = np.zeros((len(geometry.nodes), 6))  # 6 strain components
            
            for i, temp_change in enumerate(temperature_change):
                if geometry.material_ids is not None:
                    mat_id = geometry.material_ids[i] if i < len(geometry.material_ids) else 0
                    alpha = expansion_coeffs[mat_id]
                else:
                    # Use first material if no material mapping
                    alpha = list(expansion_coeffs.values())[0]
                
                # Get scalar thermal expansion coefficient for this node
                if isinstance(alpha, np.ndarray):
                    alpha_scalar = alpha[i] if i < len(alpha) else alpha[0]
                else:
                    alpha_scalar = alpha
                
                # Isotropic thermal expansion (normal strains only)
                thermal_strain_expansion[i, 0:3] = alpha_scalar * temp_change  # εxx, εyy, εzz
                # Shear strains remain zero for isotropic expansion
            
            # Calculate displacement due to thermal expansion
            thermal_displacement = self._calculate_thermal_displacement(
                geometry, thermal_strain_expansion
            )
            
            # Calculate thermal stress if constrained
            constrained_thermal_stress = self._calculate_constrained_thermal_stress(
                geometry, materials, thermal_strain_expansion, temperature_change + reference_temperature
            )
            
            return {
                'thermal_strain': thermal_strain_expansion,
                'thermal_displacement': thermal_displacement,
                'constrained_stress': constrained_thermal_stress,
                'expansion_coefficients': expansion_coeffs
            }
            
        except Exception as e:
            self.logger.error(f"Thermal expansion calculation failed: {e}")
            raise
    
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
        self.logger.info("Starting coupled thermal-structural analysis")
        
        try:
            # Initialize solution
            temperature = thermal_loads.temperature_distribution.copy()
            displacement = np.zeros((len(geometry.nodes), 3))
            
            for iteration in range(coupling_iterations):
                self.logger.debug(f"Coupling iteration {iteration + 1}")
                
                # Update thermal loads based on current temperature
                updated_thermal_loads = ThermalLoadConditions(
                    temperature_distribution=temperature,
                    temperature_gradient=thermal_loads.temperature_gradient,
                    heat_flux=thermal_loads.heat_flux,
                    time_history=thermal_loads.time_history,
                    boundary_conditions=thermal_loads.boundary_conditions
                )
                
                # Solve structural problem with current temperature
                structural_results = self.analyze_thermal_stress(
                    geometry, materials, updated_thermal_loads, 'steady_state'
                )
                
                # Update temperature based on structural deformation (if significant)
                new_displacement = structural_results.displacement
                displacement_change = np.linalg.norm(new_displacement - displacement)
                
                # Check convergence
                if displacement_change < convergence_tolerance:
                    self.logger.info(f"Coupled analysis converged in {iteration + 1} iterations")
                    return structural_results
                
                displacement = new_displacement
                
                # Update temperature field based on deformation (simplified)
                # In practice, this would involve solving heat transfer with deformed geometry
                temperature = self._update_temperature_field(
                    temperature, displacement, thermal_loads
                )
            
            self.logger.warning(f"Coupled analysis did not converge in {coupling_iterations} iterations")
            return structural_results
            
        except Exception as e:
            self.logger.error(f"Coupled thermal-structural analysis failed: {e}")
            raise
    
    def _validate_thermal_inputs(self, geometry: StructuralGeometry, 
                                materials: Dict[str, MaterialDefinition],
                                thermal_loads: ThermalLoadConditions) -> None:
        """Validate inputs for thermal stress analysis."""
        if len(geometry.nodes) == 0:
            raise ValueError("Geometry must have nodes")
        
        if len(materials) == 0:
            raise ValueError("Materials dictionary cannot be empty")
        
        if len(thermal_loads.temperature_distribution) != len(geometry.nodes):
            raise ValueError("Temperature distribution must match number of nodes")
        
        # Validate material properties
        for mat_id, material in materials.items():
            if not material.thermal_properties:
                raise ValueError(f"Material {mat_id} missing thermal properties")
            if not material.mechanical_properties:
                raise ValueError(f"Material {mat_id} missing mechanical properties")
    
    def _calculate_temperature_dependent_properties(self,
                                                   materials: Dict[str, MaterialDefinition],
                                                   temperatures: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """Calculate temperature-dependent material properties."""
        temp_props = {}
        
        for mat_id, material in materials.items():
            thermal_props = material.thermal_properties
            mech_props = material.mechanical_properties
            
            # Temperature-dependent Young's modulus (simplified model)
            E_ref = mech_props.youngs_modulus
            T_ref = 293.15  # Reference temperature
            
            # Linear degradation model (can be made more sophisticated)
            E_temp = E_ref * (1.0 - 0.0005 * (temperatures - T_ref))
            E_temp = np.maximum(E_temp, 0.1 * E_ref)  # Minimum 10% of reference
            
            # Temperature-dependent thermal expansion coefficient
            alpha_base = 12e-6  # Base thermal expansion coefficient [1/K]
            alpha_temp = alpha_base * (1.0 + 0.0001 * (temperatures - T_ref))
            
            # Temperature-dependent yield strength
            yield_ref = mech_props.yield_strength
            yield_temp = yield_ref * (1.0 - 0.001 * (temperatures - T_ref))
            yield_temp = np.maximum(yield_temp, 0.1 * yield_ref)
            
            temp_props[mat_id] = {
                'youngs_modulus': E_temp,
                'poissons_ratio': np.full_like(temperatures, mech_props.poissons_ratio),
                'thermal_expansion': alpha_temp,
                'yield_strength': yield_temp,
                'density': np.full_like(temperatures, mech_props.density)
            }
        
        return temp_props
    
    def _calculate_thermal_strain(self, temperatures: np.ndarray,
                                 temp_props: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        """Calculate thermal strain from temperature distribution."""
        n_nodes = len(temperatures)
        thermal_strain = np.zeros((n_nodes, 6))  # 6 strain components
        
        # Use first material properties if no material mapping
        mat_id = list(temp_props.keys())[0]
        alpha = temp_props[mat_id]['thermal_expansion']
        
        T_ref = 293.15  # Reference temperature
        delta_T = temperatures - T_ref
        
        # Isotropic thermal expansion
        thermal_strain[:, 0] = alpha * delta_T  # εxx
        thermal_strain[:, 1] = alpha * delta_T  # εyy  
        thermal_strain[:, 2] = alpha * delta_T  # εzz
        # Shear strains remain zero
        
        return thermal_strain
    
    def _solve_steady_state_thermal(self, geometry: StructuralGeometry,
                                   temp_props: Dict[str, Dict[str, np.ndarray]],
                                   thermal_strain: np.ndarray,
                                   thermal_loads: ThermalLoadConditions) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve steady-state thermal structural problem."""
        n_nodes = len(geometry.nodes)
        n_dof = n_nodes * 3  # 3 DOF per node
        
        # Assemble stiffness matrix (simplified)
        K = self._assemble_stiffness_matrix(geometry, temp_props)
        
        # Calculate thermal force vector
        F_thermal = self._calculate_thermal_forces(geometry, temp_props, thermal_strain)
        
        # Apply boundary conditions (simplified - fixed at first node)
        K_reduced, F_reduced = self._apply_boundary_conditions(K, F_thermal)
        
        # Solve for displacements
        displacement_reduced = np.linalg.solve(K_reduced, F_reduced)
        
        # Expand to full displacement vector
        displacement = np.zeros(n_dof)
        displacement[3:] = displacement_reduced  # Skip first node (fixed)
        displacement = displacement.reshape((n_nodes, 3))
        
        # Calculate strains and stresses
        mechanical_strain = self._calculate_mechanical_strain(geometry, displacement)
        mechanical_stress = self._calculate_mechanical_stress(temp_props, mechanical_strain)
        
        return displacement, mechanical_strain, mechanical_stress
    
    def _solve_transient_thermal(self, geometry: StructuralGeometry,
                                temp_props: Dict[str, Dict[str, np.ndarray]],
                                thermal_strain: np.ndarray,
                                thermal_loads: ThermalLoadConditions) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve transient thermal structural problem."""
        # For now, use steady-state solution
        # Full transient analysis would require time integration
        return self._solve_steady_state_thermal(geometry, temp_props, thermal_strain, thermal_loads)
    
    def _calculate_thermal_stress_field(self, temp_props: Dict[str, Dict[str, np.ndarray]],
                                       thermal_strain: np.ndarray) -> np.ndarray:
        """Calculate thermal stress field."""
        n_nodes = thermal_strain.shape[0]
        thermal_stress = np.zeros((n_nodes, 6))
        
        # Use first material properties
        mat_id = list(temp_props.keys())[0]
        E = temp_props[mat_id]['youngs_modulus']
        nu = temp_props[mat_id]['poissons_ratio']
        
        # Calculate thermal stress (constrained thermal expansion)
        for i in range(n_nodes):
            D = self._get_constitutive_matrix(E[i], nu[i])
            thermal_stress[i, :] = -D @ thermal_strain[i, :]
        
        return thermal_stress
    
    def _calculate_safety_factors(self, total_stress: np.ndarray,
                                 temp_props: Dict[str, Dict[str, np.ndarray]],
                                 temperatures: np.ndarray) -> np.ndarray:
        """Calculate safety factors based on stress and temperature-dependent strength."""
        n_nodes = total_stress.shape[0]
        safety_factor = np.zeros(n_nodes)
        
        # Use first material properties
        mat_id = list(temp_props.keys())[0]
        yield_strength = temp_props[mat_id]['yield_strength']
        
        for i in range(n_nodes):
            # Von Mises stress
            von_mises = self._calculate_von_mises_stress(total_stress[i, :])
            
            # Safety factor
            if von_mises > 0:
                safety_factor[i] = yield_strength[i] / von_mises
            else:
                safety_factor[i] = np.inf
        
        return safety_factor
    
    def _identify_failure_locations(self, total_stress: np.ndarray,
                                   safety_factor: np.ndarray,
                                   temp_props: Dict[str, Dict[str, np.ndarray]]) -> List[Tuple[int, str]]:
        """Identify locations where failure criteria are exceeded."""
        failure_locations = []
        
        for i, sf in enumerate(safety_factor):
            if sf < 1.0:
                # Determine failure mode
                von_mises = self._calculate_von_mises_stress(total_stress[i, :])
                max_principal = np.max(total_stress[i, 0:3])  # Approximate
                
                if max_principal > von_mises * 0.8:
                    failure_mode = "tensile"
                else:
                    failure_mode = "yielding"
                
                failure_locations.append((i, failure_mode))
        
        return failure_locations
    
    def _identify_critical_regions(self, geometry: StructuralGeometry,
                                  total_stress: np.ndarray,
                                  temperatures: np.ndarray,
                                  safety_factor: np.ndarray) -> List[Dict[str, Any]]:
        """Identify critical regions requiring attention."""
        critical_regions = []
        
        # High stress regions
        stress_threshold = np.percentile(np.abs(total_stress).max(axis=1), 90)
        high_stress_nodes = np.where(np.abs(total_stress).max(axis=1) > stress_threshold)[0]
        
        if len(high_stress_nodes) > 0:
            critical_regions.append({
                'type': 'high_stress',
                'nodes': high_stress_nodes.tolist(),
                'max_stress': np.max(np.abs(total_stress[high_stress_nodes])),
                'description': 'Regions with high stress concentration'
            })
        
        # High temperature regions
        temp_threshold = np.percentile(temperatures, 90)
        high_temp_nodes = np.where(temperatures > temp_threshold)[0]
        
        if len(high_temp_nodes) > 0:
            critical_regions.append({
                'type': 'high_temperature',
                'nodes': high_temp_nodes.tolist(),
                'max_temperature': np.max(temperatures[high_temp_nodes]),
                'description': 'Regions with extreme temperatures'
            })
        
        # Low safety factor regions
        low_sf_nodes = np.where(safety_factor < 2.0)[0]
        
        if len(low_sf_nodes) > 0:
            critical_regions.append({
                'type': 'low_safety_factor',
                'nodes': low_sf_nodes.tolist(),
                'min_safety_factor': np.min(safety_factor[low_sf_nodes]),
                'description': 'Regions with low safety margins'
            })
        
        return critical_regions
    
    def _get_thermal_expansion_coefficients(self, materials: Dict[str, MaterialDefinition],
                                          temperatures: np.ndarray) -> Dict[str, np.ndarray]:
        """Get temperature-dependent thermal expansion coefficients."""
        expansion_coeffs = {}
        
        for mat_id, material in materials.items():
            # Base thermal expansion coefficient
            alpha_base = 12e-6  # [1/K] - typical for metals
            
            # Temperature dependence (simplified)
            T_ref = 293.15
            alpha = alpha_base * (1.0 + 0.0001 * (temperatures - T_ref))
            
            expansion_coeffs[mat_id] = alpha
        
        return expansion_coeffs
    
    def _calculate_thermal_displacement(self, geometry: StructuralGeometry,
                                      thermal_strain: np.ndarray) -> np.ndarray:
        """Calculate displacement due to thermal expansion."""
        # Simplified calculation - assumes unconstrained expansion
        n_nodes = len(geometry.nodes)
        displacement = np.zeros((n_nodes, 3))
        
        # For each node, calculate displacement based on thermal strain
        for i in range(n_nodes):
            # Assume isotropic expansion from origin
            coords = geometry.nodes[i]
            strain_avg = np.mean(thermal_strain[i, 0:3])  # Average normal strain
            displacement[i] = coords * strain_avg
        
        return displacement
    
    def _calculate_constrained_thermal_stress(self, geometry: StructuralGeometry,
                                            materials: Dict[str, MaterialDefinition],
                                            thermal_strain: np.ndarray,
                                            temperatures: np.ndarray) -> np.ndarray:
        """Calculate thermal stress when thermal expansion is constrained."""
        n_nodes = thermal_strain.shape[0]
        constrained_stress = np.zeros((n_nodes, 6))
        
        # Get material properties
        mat_id = list(materials.keys())[0]
        material = materials[mat_id]
        E = material.mechanical_properties.youngs_modulus
        nu = material.mechanical_properties.poissons_ratio
        
        # Calculate stress for constrained thermal expansion
        for i in range(n_nodes):
            D = self._get_constitutive_matrix(E, nu)
            constrained_stress[i, :] = -D @ thermal_strain[i, :]
        
        return constrained_stress
    
    def _update_temperature_field(self, temperature: np.ndarray,
                                 displacement: np.ndarray,
                                 thermal_loads: ThermalLoadConditions) -> np.ndarray:
        """Update temperature field based on structural deformation."""
        # Simplified update - in practice would solve heat transfer on deformed geometry
        return temperature
    
    def _assemble_stiffness_matrix(self, geometry: StructuralGeometry,
                                  temp_props: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        """Assemble global stiffness matrix."""
        n_nodes = len(geometry.nodes)
        n_dof = n_nodes * 3
        K = np.zeros((n_dof, n_dof))
        
        # Simplified stiffness matrix assembly
        # In practice, would loop over elements and assemble element stiffness matrices
        mat_id = list(temp_props.keys())[0]
        E = np.mean(temp_props[mat_id]['youngs_modulus'])
        
        # Simple spring model for demonstration
        for i in range(n_nodes - 1):
            for j in range(3):  # 3 DOF per node
                idx = i * 3 + j
                K[idx, idx] += E * 1e-6  # Simplified stiffness
                K[idx + 3, idx + 3] += E * 1e-6
                K[idx, idx + 3] -= E * 1e-6
                K[idx + 3, idx] -= E * 1e-6
        
        return K
    
    def _calculate_thermal_forces(self, geometry: StructuralGeometry,
                                 temp_props: Dict[str, Dict[str, np.ndarray]],
                                 thermal_strain: np.ndarray) -> np.ndarray:
        """Calculate thermal force vector."""
        n_nodes = len(geometry.nodes)
        n_dof = n_nodes * 3
        F = np.zeros(n_dof)
        
        # Simplified thermal force calculation
        # In practice, would integrate thermal strain over elements
        mat_id = list(temp_props.keys())[0]
        E = np.mean(temp_props[mat_id]['youngs_modulus'])
        
        for i in range(n_nodes):
            for j in range(3):
                idx = i * 3 + j
                F[idx] = E * thermal_strain[i, j] * 1e-6  # Simplified
        
        return F
    
    def _apply_boundary_conditions(self, K: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply boundary conditions (simplified - fix first node)."""
        # Remove first 3 DOF (fixed boundary condition)
        K_reduced = K[3:, 3:]
        F_reduced = F[3:]
        
        return K_reduced, F_reduced
    
    def _calculate_mechanical_strain(self, geometry: StructuralGeometry,
                                   displacement: np.ndarray) -> np.ndarray:
        """Calculate mechanical strain from displacement field."""
        n_nodes = len(geometry.nodes)
        mechanical_strain = np.zeros((n_nodes, 6))
        
        # Simplified strain calculation
        # In practice, would use shape functions and derivatives
        for i in range(n_nodes - 1):
            dx = geometry.nodes[i + 1] - geometry.nodes[i]
            du = displacement[i + 1] - displacement[i]
            
            if np.linalg.norm(dx) > 0:
                # Normal strains (simplified)
                mechanical_strain[i, 0] = du[0] / dx[0] if abs(dx[0]) > 1e-12 else 0
                mechanical_strain[i, 1] = du[1] / dx[1] if abs(dx[1]) > 1e-12 else 0
                mechanical_strain[i, 2] = du[2] / dx[2] if abs(dx[2]) > 1e-12 else 0
        
        return mechanical_strain
    
    def _calculate_mechanical_stress(self, temp_props: Dict[str, Dict[str, np.ndarray]],
                                   mechanical_strain: np.ndarray) -> np.ndarray:
        """Calculate mechanical stress from strain."""
        n_nodes = mechanical_strain.shape[0]
        mechanical_stress = np.zeros((n_nodes, 6))
        
        mat_id = list(temp_props.keys())[0]
        E = temp_props[mat_id]['youngs_modulus']
        nu = temp_props[mat_id]['poissons_ratio']
        
        for i in range(n_nodes):
            D = self._get_constitutive_matrix(E[i], nu[i])
            mechanical_stress[i, :] = D @ mechanical_strain[i, :]
        
        return mechanical_stress
    
    def _get_constitutive_matrix(self, E: float, nu: float) -> np.ndarray:
        """Get constitutive matrix for isotropic material."""
        D = np.zeros((6, 6))
        
        # Isotropic elasticity matrix
        factor = E / ((1 + nu) * (1 - 2 * nu))
        
        # Normal stress components
        D[0, 0] = D[1, 1] = D[2, 2] = factor * (1 - nu)
        D[0, 1] = D[0, 2] = D[1, 0] = D[1, 2] = D[2, 0] = D[2, 1] = factor * nu
        
        # Shear stress components
        D[3, 3] = D[4, 4] = D[5, 5] = factor * (1 - 2 * nu) / 2
        
        return D
    
    def _calculate_von_mises_stress(self, stress: np.ndarray) -> float:
        """Calculate von Mises equivalent stress."""
        # stress = [σxx, σyy, σzz, τxy, τxz, τyz]
        sx, sy, sz = stress[0], stress[1], stress[2]
        txy, txz, tyz = stress[3], stress[4], stress[5]
        
        von_mises = np.sqrt(0.5 * ((sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2) + 
                           3 * (txy**2 + txz**2 + tyz**2))
        
        return von_mises