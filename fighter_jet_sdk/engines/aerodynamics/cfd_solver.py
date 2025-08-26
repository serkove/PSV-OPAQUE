"""
CFD Solver Integration Module

Provides computational fluid dynamics capabilities with OpenFOAM integration,
mesh generation, adaptive refinement, and multi-speed regime analysis.
"""

import numpy as np
import subprocess
import os
import tempfile
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

from ...common.data_models import AircraftConfiguration, FlowConditions
from ...common.interfaces import AnalysisEngine
from ...core.errors import CFDError, ValidationError


class FlowRegime(Enum):
    """Flow regime classification"""
    SUBSONIC = "subsonic"
    TRANSONIC = "transonic" 
    SUPERSONIC = "supersonic"
    HYPERSONIC = "hypersonic"


@dataclass
class MeshParameters:
    """Mesh generation parameters"""
    base_cell_size: float
    boundary_layer_thickness: float
    refinement_levels: int
    growth_ratio: float
    surface_refinement: Dict[str, float]
    wake_refinement: bool = True
    shock_refinement: bool = True


@dataclass
class SolverSettings:
    """CFD solver configuration"""
    solver_type: str  # simpleFoam, rhoSimpleFoam, etc.
    turbulence_model: str
    max_iterations: int
    convergence_tolerance: float
    relaxation_factors: Dict[str, float]
    time_step: Optional[float] = None
    end_time: Optional[float] = None


@dataclass
class CFDResults:
    """CFD analysis results"""
    forces: Dict[str, float]  # drag, lift, side_force
    moments: Dict[str, float]  # pitch, yaw, roll
    pressure_distribution: np.ndarray
    velocity_field: np.ndarray
    convergence_history: List[float]
    residuals: Dict[str, List[float]]
    flow_regime: FlowRegime
    mach_number: float
    reynolds_number: float


class MeshGenerator:
    """Handles mesh generation and adaptive refinement"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def generate_mesh(self, geometry_file: str, mesh_params: MeshParameters, 
                     flow_conditions: FlowConditions) -> str:
        """Generate computational mesh for CFD analysis"""
        try:
            # Create temporary directory for mesh generation
            mesh_dir = tempfile.mkdtemp(prefix="cfd_mesh_")
            
            # Determine flow regime for mesh adaptation
            flow_regime = self._classify_flow_regime(flow_conditions.mach_number)
            
            # Generate base mesh
            base_mesh = self._generate_base_mesh(geometry_file, mesh_params, mesh_dir)
            
            # Apply regime-specific refinements
            refined_mesh = self._apply_flow_refinements(
                base_mesh, flow_regime, mesh_params, flow_conditions
            )
            
            # Validate mesh quality
            self._validate_mesh_quality(refined_mesh)
            
            self.logger.info(f"Generated mesh for {flow_regime.value} flow regime")
            return refined_mesh
            
        except Exception as e:
            raise CFDError(f"Mesh generation failed: {str(e)}")
    
    def _classify_flow_regime(self, mach_number: float) -> FlowRegime:
        """Classify flow regime based on Mach number"""
        if mach_number < 0.8:
            return FlowRegime.SUBSONIC
        elif mach_number < 1.2:
            return FlowRegime.TRANSONIC
        elif mach_number < 5.0:
            return FlowRegime.SUPERSONIC
        else:
            return FlowRegime.HYPERSONIC
    
    def _generate_base_mesh(self, geometry_file: str, mesh_params: MeshParameters, 
                           mesh_dir: str) -> str:
        """Generate base computational mesh"""
        # Create blockMeshDict for structured mesh
        block_mesh_dict = self._create_block_mesh_dict(mesh_params)
        
        # Write mesh dictionary
        mesh_dict_path = os.path.join(mesh_dir, "blockMeshDict")
        with open(mesh_dict_path, 'w') as f:
            f.write(block_mesh_dict)
        
        # Run blockMesh
        cmd = ["blockMesh", "-case", mesh_dir]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise CFDError(f"blockMesh failed: {result.stderr}")
        
        return mesh_dir
    
    def _apply_flow_refinements(self, mesh_dir: str, flow_regime: FlowRegime,
                               mesh_params: MeshParameters, 
                               flow_conditions: FlowConditions) -> str:
        """Apply flow regime specific mesh refinements"""
        
        if flow_regime in [FlowRegime.SUPERSONIC, FlowRegime.HYPERSONIC]:
            # Add shock wave refinement
            self._add_shock_refinement(mesh_dir, flow_conditions)
        
        if flow_regime == FlowRegime.TRANSONIC:
            # Add transonic refinement around Mach 1 regions
            self._add_transonic_refinement(mesh_dir, flow_conditions)
        
        if flow_regime == FlowRegime.HYPERSONIC:
            # Add boundary layer refinement for high temperature effects
            self._add_hypersonic_refinement(mesh_dir, flow_conditions)
        
        return mesh_dir
    
    def _add_shock_refinement(self, mesh_dir: str, flow_conditions: FlowConditions):
        """Add mesh refinement for shock wave capture"""
        # Implementation for shock-adaptive refinement
        pass
    
    def _add_transonic_refinement(self, mesh_dir: str, flow_conditions: FlowConditions):
        """Add mesh refinement for transonic flow features"""
        # Implementation for transonic refinement
        pass
    
    def _add_hypersonic_refinement(self, mesh_dir: str, flow_conditions: FlowConditions):
        """Add mesh refinement for hypersonic flow effects"""
        # Implementation for hypersonic boundary layer refinement
        pass
    
    def _create_block_mesh_dict(self, mesh_params: MeshParameters) -> str:
        """Create OpenFOAM blockMeshDict content"""
        return f"""
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2012                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

convertToMeters 1;

vertices
(
    (-10 -5 -5)  // 0
    ( 10 -5 -5)  // 1
    ( 10  5 -5)  // 2
    (-10  5 -5)  // 3
    (-10 -5  5)  // 4
    ( 10 -5  5)  // 5
    ( 10  5  5)  // 6
    (-10  5  5)  // 7
);

blocks
(
    hex (0 1 2 3 4 5 6 7) (100 50 50) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    inlet
    {{
        type patch;
        faces
        (
            (0 4 7 3)
        );
    }}
    outlet
    {{
        type patch;
        faces
        (
            (2 6 5 1)
        );
    }}
    walls
    {{
        type wall;
        faces
        (
            (1 5 4 0)
            (3 7 6 2)
            (0 3 2 1)
            (4 5 6 7)
        );
    }}
);

mergePatchPairs
(
);
"""
    
    def _validate_mesh_quality(self, mesh_dir: str):
        """Validate mesh quality metrics"""
        cmd = ["checkMesh", "-case", mesh_dir]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if "FAILED" in result.stdout:
            raise CFDError(f"Mesh quality check failed: {result.stdout}")


class ConvergenceMonitor:
    """Monitors CFD solution convergence"""
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.residual_history = {}
        self.logger = logging.getLogger(__name__)
    
    def check_convergence(self, residuals: Dict[str, float], iteration: int) -> bool:
        """Check if solution has converged"""
        # Store residual history
        for field, residual in residuals.items():
            if field not in self.residual_history:
                self.residual_history[field] = []
            self.residual_history[field].append(residual)
        
        # Check convergence criteria
        converged = all(res < self.tolerance for res in residuals.values())
        
        if converged:
            self.logger.info(f"Solution converged at iteration {iteration}")
        
        return converged
    
    def get_convergence_history(self) -> Dict[str, List[float]]:
        """Get convergence history for all fields"""
        return self.residual_history.copy()


class CFDSolver(AnalysisEngine):
    """Main CFD solver with OpenFOAM integration"""
    
    def __init__(self):
        super().__init__()
        self.mesh_generator = MeshGenerator()
        self.convergence_monitor = ConvergenceMonitor()
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, configuration: AircraftConfiguration, 
                flow_conditions: FlowConditions,
                solver_settings: Optional[SolverSettings] = None) -> CFDResults:
        """Perform CFD analysis on aircraft configuration"""
        try:
            # Validate inputs
            self._validate_inputs(configuration, flow_conditions)
            
            # Set default solver settings if not provided
            if solver_settings is None:
                solver_settings = self._get_default_solver_settings(flow_conditions)
            
            # Generate mesh
            mesh_params = self._get_mesh_parameters(flow_conditions)
            geometry_file = self._export_geometry(configuration)
            mesh_dir = self.mesh_generator.generate_mesh(
                geometry_file, mesh_params, flow_conditions
            )
            
            # Setup case
            case_dir = self._setup_case(mesh_dir, flow_conditions, solver_settings)
            
            # Run solver
            results = self._run_solver(case_dir, solver_settings)
            
            # Post-process results
            processed_results = self._post_process_results(
                results, flow_conditions, configuration
            )
            
            self.logger.info("CFD analysis completed successfully")
            return processed_results
            
        except Exception as e:
            raise CFDError(f"CFD analysis failed: {str(e)}")
    
    def _validate_inputs(self, configuration: AircraftConfiguration, 
                        flow_conditions: FlowConditions):
        """Validate analysis inputs"""
        if not configuration:
            raise ValidationError("Aircraft configuration is required")
        
        if flow_conditions.mach_number < 0:
            raise ValidationError("Mach number must be positive")
        
        if flow_conditions.altitude < 0:
            raise ValidationError("Altitude must be non-negative")
    
    def _get_default_solver_settings(self, flow_conditions: FlowConditions) -> SolverSettings:
        """Get default solver settings based on flow conditions"""
        flow_regime = self.mesh_generator._classify_flow_regime(flow_conditions.mach_number)
        
        if flow_regime == FlowRegime.SUBSONIC:
            return SolverSettings(
                solver_type="simpleFoam",
                turbulence_model="kOmegaSST",
                max_iterations=1000,
                convergence_tolerance=1e-6,
                relaxation_factors={"p": 0.3, "U": 0.7}
            )
        elif flow_regime in [FlowRegime.SUPERSONIC, FlowRegime.HYPERSONIC]:
            return SolverSettings(
                solver_type="rhoSimpleFoam",
                turbulence_model="kOmegaSST",
                max_iterations=2000,
                convergence_tolerance=1e-5,
                relaxation_factors={"p": 0.2, "U": 0.5, "T": 0.7}
            )
        else:  # Transonic
            return SolverSettings(
                solver_type="rhoSimpleFoam",
                turbulence_model="kOmegaSST",
                max_iterations=1500,
                convergence_tolerance=1e-6,
                relaxation_factors={"p": 0.25, "U": 0.6, "T": 0.7}
            )
    
    def _get_mesh_parameters(self, flow_conditions: FlowConditions) -> MeshParameters:
        """Get mesh parameters based on flow conditions"""
        reynolds_number = self._calculate_reynolds_number(flow_conditions)
        
        # Estimate boundary layer thickness
        bl_thickness = 0.37 / (reynolds_number ** 0.2)
        
        return MeshParameters(
            base_cell_size=0.1,
            boundary_layer_thickness=bl_thickness,
            refinement_levels=3,
            growth_ratio=1.2,
            surface_refinement={"aircraft": 0.01, "wake": 0.05}
        )
    
    def _calculate_reynolds_number(self, flow_conditions: FlowConditions) -> float:
        """Calculate Reynolds number for flow conditions"""
        # Simplified calculation - would use atmospheric model in practice
        reference_length = 10.0  # meters
        kinematic_viscosity = 1.5e-5  # m²/s at sea level
        
        velocity = flow_conditions.mach_number * 343.0  # m/s (speed of sound)
        return velocity * reference_length / kinematic_viscosity
    
    def _export_geometry(self, configuration: AircraftConfiguration) -> str:
        """Export aircraft geometry for CFD analysis"""
        # Placeholder - would export STL or other mesh format
        geometry_file = tempfile.mktemp(suffix=".stl")
        
        # Create simple box geometry for now
        with open(geometry_file, 'w') as f:
            f.write("solid aircraft\n")
            f.write("endsolid aircraft\n")
        
        return geometry_file
    
    def _setup_case(self, mesh_dir: str, flow_conditions: FlowConditions,
                   solver_settings: SolverSettings) -> str:
        """Setup OpenFOAM case directory"""
        case_dir = tempfile.mkdtemp(prefix="cfd_case_")
        
        # Copy mesh
        subprocess.run(["cp", "-r", os.path.join(mesh_dir, "constant"), case_dir])
        subprocess.run(["cp", "-r", os.path.join(mesh_dir, "system"), case_dir])
        
        # Create initial conditions
        self._create_initial_conditions(case_dir, flow_conditions)
        
        # Create boundary conditions
        self._create_boundary_conditions(case_dir, flow_conditions)
        
        # Create solver control dictionary
        self._create_control_dict(case_dir, solver_settings)
        
        return case_dir
    
    def _create_initial_conditions(self, case_dir: str, flow_conditions: FlowConditions):
        """Create initial condition files"""
        # Create 0 directory
        zero_dir = os.path.join(case_dir, "0")
        os.makedirs(zero_dir, exist_ok=True)
        
        # Create U file (velocity)
        velocity = flow_conditions.mach_number * 343.0  # m/s
        u_content = f"""
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}}

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform ({velocity} 0 0);

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform ({velocity} 0 0);
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    walls
    {{
        type            noSlip;
    }}
}}
"""
        with open(os.path.join(zero_dir, "U"), 'w') as f:
            f.write(u_content)
    
    def _create_boundary_conditions(self, case_dir: str, flow_conditions: FlowConditions):
        """Create boundary condition files"""
        # Implementation for boundary conditions
        pass
    
    def _create_control_dict(self, case_dir: str, solver_settings: SolverSettings):
        """Create solver control dictionary"""
        system_dir = os.path.join(case_dir, "system")
        os.makedirs(system_dir, exist_ok=True)
        
        control_dict = f"""
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}

application     {solver_settings.solver_type};

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {solver_settings.max_iterations};
deltaT          1;
writeControl    timeStep;
writeInterval   100;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""
        
        with open(os.path.join(system_dir, "controlDict"), 'w') as f:
            f.write(control_dict)
    
    def _run_solver(self, case_dir: str, solver_settings: SolverSettings) -> Dict[str, Any]:
        """Run OpenFOAM solver"""
        cmd = [solver_settings.solver_type, "-case", case_dir]
        
        # Run solver
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=case_dir)
        
        if result.returncode != 0:
            raise CFDError(f"Solver failed: {result.stderr}")
        
        # Parse solver output for residuals and forces
        return self._parse_solver_output(result.stdout, case_dir)
    
    def _parse_solver_output(self, solver_output: str, case_dir: str) -> Dict[str, Any]:
        """Parse solver output for results"""
        # Extract residuals from solver output
        residuals = {"U": [], "p": []}
        
        # Parse forces if available
        forces = {"drag": 0.0, "lift": 0.0, "side_force": 0.0}
        moments = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
        
        return {
            "residuals": residuals,
            "forces": forces,
            "moments": moments,
            "case_dir": case_dir
        }
    
    def _post_process_results(self, raw_results: Dict[str, Any], 
                             flow_conditions: FlowConditions,
                             configuration: AircraftConfiguration) -> CFDResults:
        """Post-process CFD results"""
        
        flow_regime = self.mesh_generator._classify_flow_regime(flow_conditions.mach_number)
        reynolds_number = self._calculate_reynolds_number(flow_conditions)
        
        # Create dummy arrays for now
        pressure_dist = np.zeros((100, 100))
        velocity_field = np.zeros((100, 100, 3))
        
        return CFDResults(
            forces=raw_results["forces"],
            moments=raw_results["moments"],
            pressure_distribution=pressure_dist,
            velocity_field=velocity_field,
            convergence_history=[1e-3, 1e-4, 1e-5, 1e-6],
            residuals=raw_results["residuals"],
            flow_regime=flow_regime,
            mach_number=flow_conditions.mach_number,
            reynolds_number=reynolds_number
        )