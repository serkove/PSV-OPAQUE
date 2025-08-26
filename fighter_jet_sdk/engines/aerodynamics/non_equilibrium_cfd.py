"""
Non-Equilibrium Chemistry CFD Module

Provides computational fluid dynamics capabilities with non-equilibrium chemistry
for extreme hypersonic conditions. Handles species transport equations, reaction
rate calculations, and ionization/dissociation mechanisms for Mach 60 flight.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import math
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

from .plasma_flow_solver import PlasmaFlowSolver, PlasmaFlowConditions, PlasmaFlowResults
from ...common.data_models import AircraftConfiguration, FlowConditions
from ...common.plasma_physics import PlasmaConditions, GasMixture, BOLTZMANN_CONSTANT, ELEMENTARY_CHARGE
from ...core.errors import CFDError, ValidationError


class ReactionType(Enum):
    """Types of chemical reactions in non-equilibrium flow."""
    DISSOCIATION = "dissociation"
    IONIZATION = "ionization"
    RECOMBINATION = "recombination"
    CHARGE_EXCHANGE = "charge_exchange"
    ELASTIC_COLLISION = "elastic_collision"


@dataclass
class ChemicalSpecies:
    """Chemical species definition for non-equilibrium chemistry."""
    name: str
    molecular_mass: float  # kg/mol
    formation_enthalpy: float  # J/mol
    charge: int  # Elementary charges
    vibrational_temperature: Optional[float] = None  # K
    electronic_levels: Optional[List[float]] = None  # J
    collision_diameter: Optional[float] = None  # m
    
    def __post_init__(self):
        """Initialize default values."""
        if self.electronic_levels is None:
            self.electronic_levels = [0.0]  # Ground state only
        if self.collision_diameter is None:
            # Estimate from molecular mass (rough approximation)
            self.collision_diameter = (self.molecular_mass / 6.022e23) ** (1/3) * 1e-9


@dataclass
class ChemicalReaction:
    """Chemical reaction definition."""
    reaction_id: str
    reaction_type: ReactionType
    reactants: Dict[str, int]  # species_name -> stoichiometric coefficient
    products: Dict[str, int]   # species_name -> stoichiometric coefficient
    activation_energy: float   # J/mol
    pre_exponential_factor: float  # Units depend on reaction order
    temperature_exponent: float = 0.0
    third_body_efficiency: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.third_body_efficiency is None:
            self.third_body_efficiency = {}


@dataclass
class NonEquilibriumState:
    """Non-equilibrium flow state with species concentrations."""
    temperature: float  # K
    pressure: float     # Pa
    density: float      # kg/m³
    velocity: np.ndarray  # m/s
    species_concentrations: Dict[str, float]  # mol/m³
    vibrational_temperatures: Dict[str, float]  # K
    electronic_temperatures: Dict[str, float]  # K
    
    @property
    def total_concentration(self) -> float:
        """Total molar concentration."""
        return sum(self.species_concentrations.values())
    
    @property
    def species_mole_fractions(self) -> Dict[str, float]:
        """Species mole fractions."""
        total = self.total_concentration
        if total > 0:
            return {species: conc / total for species, conc in self.species_concentrations.items()}
        return {species: 0.0 for species in self.species_concentrations.keys()}


class ChemicalKineticsCalculator:
    """Calculator for chemical reaction rates in non-equilibrium flow."""
    
    def __init__(self):
        """Initialize chemical kinetics calculator."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize standard atmospheric species
        self.species_database = self._initialize_species_database()
        
        # Initialize reaction mechanisms
        self.reaction_database = self._initialize_reaction_database()
        
        # Universal gas constant
        self.R_universal = 8.314462618  # J/(mol·K)
    
    def _initialize_species_database(self) -> Dict[str, ChemicalSpecies]:
        """Initialize database of chemical species."""
        species_db = {}
        
        # Neutral molecules
        species_db['N2'] = ChemicalSpecies(
            name='N2',
            molecular_mass=0.028014,  # kg/mol
            formation_enthalpy=0.0,   # J/mol (reference state)
            charge=0,
            vibrational_temperature=3395.0,  # K
            collision_diameter=3.798e-10  # m
        )
        
        species_db['O2'] = ChemicalSpecies(
            name='O2',
            molecular_mass=0.031998,
            formation_enthalpy=0.0,
            charge=0,
            vibrational_temperature=2273.0,
            collision_diameter=3.467e-10
        )
        
        species_db['NO'] = ChemicalSpecies(
            name='NO',
            molecular_mass=0.030006,
            formation_enthalpy=90250.0,  # J/mol
            charge=0,
            vibrational_temperature=2719.0,
            collision_diameter=3.492e-10
        )
        
        # Atomic species
        species_db['N'] = ChemicalSpecies(
            name='N',
            molecular_mass=0.014007,
            formation_enthalpy=472680.0,  # J/mol
            charge=0,
            collision_diameter=3.298e-10
        )
        
        species_db['O'] = ChemicalSpecies(
            name='O',
            molecular_mass=0.015999,
            formation_enthalpy=249170.0,  # J/mol
            charge=0,
            collision_diameter=2.750e-10
        )
        
        # Ionized species
        species_db['N2+'] = ChemicalSpecies(
            name='N2+',
            molecular_mass=0.028013,  # Slightly less due to electron loss
            formation_enthalpy=1503240.0,  # J/mol
            charge=1,
            vibrational_temperature=3175.0,
            collision_diameter=3.798e-10
        )
        
        species_db['O2+'] = ChemicalSpecies(
            name='O2+',
            molecular_mass=0.031997,
            formation_enthalpy=1165240.0,  # J/mol
            charge=1,
            vibrational_temperature=1905.0,
            collision_diameter=3.467e-10
        )
        
        species_db['NO+'] = ChemicalSpecies(
            name='NO+',
            molecular_mass=0.030005,
            formation_enthalpy=985080.0,  # J/mol
            charge=1,
            vibrational_temperature=2377.0,
            collision_diameter=3.492e-10
        )
        
        species_db['N+'] = ChemicalSpecies(
            name='N+',
            molecular_mass=0.014006,
            formation_enthalpy=1875240.0,  # J/mol
            charge=1,
            collision_diameter=3.298e-10
        )
        
        species_db['O+'] = ChemicalSpecies(
            name='O+',
            molecular_mass=0.015998,
            formation_enthalpy=1563300.0,  # J/mol
            charge=1,
            collision_diameter=2.750e-10
        )
        
        # Electrons
        species_db['e-'] = ChemicalSpecies(
            name='e-',
            molecular_mass=5.485799e-7,  # kg/mol (electron mass)
            formation_enthalpy=0.0,
            charge=-1,
            collision_diameter=1e-15  # Very small
        )
        
        return species_db
    
    def _initialize_reaction_database(self) -> List[ChemicalReaction]:
        """Initialize database of chemical reactions."""
        reactions = []
        
        # Dissociation reactions
        reactions.append(ChemicalReaction(
            reaction_id="N2_dissociation",
            reaction_type=ReactionType.DISSOCIATION,
            reactants={'N2': 1, 'M': 1},  # M = third body
            products={'N': 2, 'M': 1},
            activation_energy=945000.0,  # J/mol
            pre_exponential_factor=7.0e21,  # m³/(mol·s)
            temperature_exponent=-1.6,
            third_body_efficiency={'N2': 1.0, 'O2': 1.0, 'N': 4.2, 'O': 4.2, 'NO': 1.0}
        ))
        
        reactions.append(ChemicalReaction(
            reaction_id="O2_dissociation",
            reaction_type=ReactionType.DISSOCIATION,
            reactants={'O2': 1, 'M': 1},
            products={'O': 2, 'M': 1},
            activation_energy=498000.0,  # J/mol
            pre_exponential_factor=2.0e21,
            temperature_exponent=-1.5,
            third_body_efficiency={'N2': 1.0, 'O2': 1.0, 'N': 5.0, 'O': 5.0, 'NO': 1.0}
        ))
        
        reactions.append(ChemicalReaction(
            reaction_id="NO_dissociation",
            reaction_type=ReactionType.DISSOCIATION,
            reactants={'NO': 1, 'M': 1},
            products={'N': 1, 'O': 1, 'M': 1},
            activation_energy=632000.0,  # J/mol
            pre_exponential_factor=5.0e15,
            temperature_exponent=0.0,
            third_body_efficiency={'N2': 1.0, 'O2': 1.0, 'N': 22.0, 'O': 22.0, 'NO': 22.0}
        ))
        
        # Exchange reactions
        reactions.append(ChemicalReaction(
            reaction_id="NO_formation",
            reaction_type=ReactionType.CHARGE_EXCHANGE,
            reactants={'N': 1, 'O2': 1},
            products={'NO': 1, 'O': 1},
            activation_energy=31400.0,  # J/mol
            pre_exponential_factor=9.0e9,
            temperature_exponent=1.0
        ))
        
        reactions.append(ChemicalReaction(
            reaction_id="NO_formation_reverse",
            reaction_type=ReactionType.CHARGE_EXCHANGE,
            reactants={'O': 1, 'N2': 1},
            products={'NO': 1, 'N': 1},
            activation_energy=319000.0,  # J/mol
            pre_exponential_factor=6.4e17,
            temperature_exponent=-1.0
        ))
        
        # Ionization reactions
        reactions.append(ChemicalReaction(
            reaction_id="N2_ionization",
            reaction_type=ReactionType.IONIZATION,
            reactants={'N2': 1, 'e-': 1},
            products={'N2+': 1, 'e-': 2},
            activation_energy=1503240.0,  # J/mol (ionization energy)
            pre_exponential_factor=2.5e34,
            temperature_exponent=-3.82
        ))
        
        reactions.append(ChemicalReaction(
            reaction_id="O2_ionization",
            reaction_type=ReactionType.IONIZATION,
            reactants={'O2': 1, 'e-': 1},
            products={'O2+': 1, 'e-': 2},
            activation_energy=1165240.0,  # J/mol
            pre_exponential_factor=3.9e33,
            temperature_exponent=-3.78
        ))
        
        reactions.append(ChemicalReaction(
            reaction_id="N_ionization",
            reaction_type=ReactionType.IONIZATION,
            reactants={'N': 1, 'e-': 1},
            products={'N+': 1, 'e-': 2},
            activation_energy=1402560.0,  # J/mol
            pre_exponential_factor=2.5e34,
            temperature_exponent=-3.82
        ))
        
        reactions.append(ChemicalReaction(
            reaction_id="O_ionization",
            reaction_type=ReactionType.IONIZATION,
            reactants={'O': 1, 'e-': 1},
            products={'O+': 1, 'e-': 2},
            activation_energy=1314130.0,  # J/mol
            pre_exponential_factor=3.9e33,
            temperature_exponent=-3.78
        ))
        
        return reactions
    
    def calculate_reaction_rate(self, reaction: ChemicalReaction,
                              state: NonEquilibriumState) -> float:
        """Calculate reaction rate for given reaction and state.
        
        Args:
            reaction: Chemical reaction
            state: Non-equilibrium flow state
            
        Returns:
            Reaction rate in mol/(m³·s)
        """
        # Calculate rate constant using Arrhenius equation
        k = self._calculate_rate_constant(reaction, state.temperature)
        
        # Calculate concentration-dependent rate
        rate = k
        
        # Multiply by reactant concentrations
        for species, coefficient in reaction.reactants.items():
            if species == 'M':  # Third body
                # Use total concentration weighted by third body efficiencies
                third_body_conc = 0.0
                for spec, conc in state.species_concentrations.items():
                    efficiency = reaction.third_body_efficiency.get(spec, 1.0)
                    third_body_conc += efficiency * conc
                rate *= third_body_conc ** coefficient
            else:
                concentration = state.species_concentrations.get(species, 0.0)
                rate *= concentration ** coefficient
        
        return rate
    
    def _calculate_rate_constant(self, reaction: ChemicalReaction, temperature: float) -> float:
        """Calculate rate constant using Arrhenius equation."""
        # k = A * T^n * exp(-Ea / (R * T))
        k = (reaction.pre_exponential_factor * 
             temperature ** reaction.temperature_exponent *
             np.exp(-reaction.activation_energy / (self.R_universal * temperature)))
        
        return k
    
    def calculate_production_rates(self, state: NonEquilibriumState) -> Dict[str, float]:
        """Calculate species production rates from all reactions.
        
        Args:
            state: Non-equilibrium flow state
            
        Returns:
            Dictionary of species production rates in mol/(m³·s)
        """
        production_rates = {species: 0.0 for species in self.species_database.keys()}
        
        for reaction in self.reaction_database:
            rate = self.calculate_reaction_rate(reaction, state)
            
            # Subtract reactants
            for species, coefficient in reaction.reactants.items():
                if species != 'M':  # Skip third body
                    production_rates[species] -= coefficient * rate
            
            # Add products
            for species, coefficient in reaction.products.items():
                if species != 'M':  # Skip third body
                    production_rates[species] += coefficient * rate
        
        return production_rates
    
    def calculate_equilibrium_constants(self, temperature: float) -> Dict[str, float]:
        """Calculate equilibrium constants for all reactions.
        
        Args:
            temperature: Temperature in K
            
        Returns:
            Dictionary of equilibrium constants
        """
        equilibrium_constants = {}
        
        for reaction in self.reaction_database:
            # Calculate from thermodynamic data
            delta_h = 0.0  # Enthalpy change
            delta_s = 0.0  # Entropy change (simplified)
            
            # Calculate enthalpy change
            for species, coefficient in reaction.products.items():
                if species != 'M' and species in self.species_database:
                    delta_h += coefficient * self.species_database[species].formation_enthalpy
            
            for species, coefficient in reaction.reactants.items():
                if species != 'M' and species in self.species_database:
                    delta_h -= coefficient * self.species_database[species].formation_enthalpy
            
            # Simplified equilibrium constant (would need full thermodynamic data)
            k_eq = np.exp(-delta_h / (self.R_universal * temperature))
            equilibrium_constants[reaction.reaction_id] = k_eq
        
        return equilibrium_constants


class NonEquilibriumCFD(PlasmaFlowSolver):
    """CFD solver with non-equilibrium chemistry for extreme hypersonic conditions."""
    
    def __init__(self):
        """Initialize non-equilibrium CFD solver."""
        super().__init__()
        self.chemistry_calculator = ChemicalKineticsCalculator()
        self.logger = logging.getLogger(__name__)
        
        # Non-equilibrium specific parameters
        self.chemistry_enabled = True
        self.vibrational_nonequilibrium = True
        self.electronic_nonequilibrium = False  # Simplified for now
        
        # Numerical parameters
        self.chemistry_time_step_factor = 0.01  # Very small for stiff chemistry
        self.chemistry_tolerance = 1e-12
        self.max_chemistry_iterations = 1000
    
    def analyze_nonequilibrium_flow(self, configuration: AircraftConfiguration,
                                  flow_conditions: FlowConditions,
                                  magnetic_field: np.ndarray,
                                  initial_composition: Optional[Dict[str, float]] = None) -> PlasmaFlowResults:
        """Perform non-equilibrium chemistry flow analysis.
        
        Args:
            configuration: Aircraft configuration
            flow_conditions: Base flow conditions
            magnetic_field: Applied magnetic field vector
            initial_composition: Initial species composition (mole fractions)
            
        Returns:
            Plasma flow results with chemistry effects
        """
        try:
            # Validate inputs
            self._validate_chemistry_inputs(configuration, flow_conditions, initial_composition)
            
            # Set default composition if not provided
            if initial_composition is None:
                initial_composition = {'N2': 0.78, 'O2': 0.21, 'Ar': 0.01}
            
            # Calculate initial non-equilibrium state
            initial_state = self._calculate_initial_nonequilibrium_state(
                flow_conditions, initial_composition
            )
            
            # Solve chemistry evolution
            evolved_state = self._solve_chemistry_evolution(initial_state, flow_conditions)
            
            # Update flow conditions with chemistry effects
            updated_flow_conditions = self._update_flow_conditions_with_chemistry(
                flow_conditions, evolved_state
            )
            
            # Run plasma flow analysis with updated conditions
            plasma_results = self.analyze_plasma_flow(
                configuration, updated_flow_conditions, magnetic_field
            )
            
            # Add chemistry-specific results
            enhanced_results = self._add_chemistry_results(plasma_results, evolved_state)
            
            self.logger.info("Non-equilibrium chemistry flow analysis completed")
            return enhanced_results
            
        except Exception as e:
            raise CFDError(f"Non-equilibrium chemistry analysis failed: {str(e)}")
    
    def _validate_chemistry_inputs(self, configuration: AircraftConfiguration,
                                 flow_conditions: FlowConditions,
                                 initial_composition: Optional[Dict[str, float]]):
        """Validate inputs for chemistry analysis."""
        # Call base validation
        self._validate_plasma_inputs(configuration, flow_conditions, np.zeros(3))
        
        # Chemistry-specific validation
        if flow_conditions.temperature < 1000:
            self.logger.warning("Temperature may be too low for significant chemistry effects")
        
        if initial_composition is not None:
            total_fraction = sum(initial_composition.values())
            if abs(total_fraction - 1.0) > 1e-6:
                raise ValidationError("Initial composition mole fractions must sum to 1.0")
            
            # Check for unknown species
            for species in initial_composition.keys():
                if species not in self.chemistry_calculator.species_database:
                    self.logger.warning(f"Unknown species '{species}' in initial composition")
    
    def _calculate_initial_nonequilibrium_state(self, flow_conditions: FlowConditions,
                                              composition: Dict[str, float]) -> NonEquilibriumState:
        """Calculate initial non-equilibrium state from flow conditions."""
        # Calculate total concentration from ideal gas law
        total_concentration = flow_conditions.pressure / (self.chemistry_calculator.R_universal * flow_conditions.temperature)
        
        # Calculate species concentrations
        species_concentrations = {}
        for species, fraction in composition.items():
            species_concentrations[species] = fraction * total_concentration
        
        # Initialize all species (including products) with zero concentration
        for species in self.chemistry_calculator.species_database.keys():
            if species not in species_concentrations:
                species_concentrations[species] = 0.0
        
        # Calculate flow velocity
        velocity_magnitude = flow_conditions.mach_number * np.sqrt(
            1.4 * 287.0 * flow_conditions.temperature
        )
        velocity = np.array([velocity_magnitude, 0.0, 0.0])
        
        # Initialize vibrational temperatures (assume thermal equilibrium initially)
        vibrational_temperatures = {}
        electronic_temperatures = {}
        for species in self.chemistry_calculator.species_database.keys():
            vibrational_temperatures[species] = flow_conditions.temperature
            electronic_temperatures[species] = flow_conditions.temperature
        
        return NonEquilibriumState(
            temperature=flow_conditions.temperature,
            pressure=flow_conditions.pressure,
            density=flow_conditions.density,
            velocity=velocity,
            species_concentrations=species_concentrations,
            vibrational_temperatures=vibrational_temperatures,
            electronic_temperatures=electronic_temperatures
        )
    
    def _solve_chemistry_evolution(self, initial_state: NonEquilibriumState,
                                 flow_conditions: FlowConditions) -> NonEquilibriumState:
        """Solve chemistry evolution using stiff ODE solver.
        
        Args:
            initial_state: Initial non-equilibrium state
            flow_conditions: Flow conditions
            
        Returns:
            Evolved non-equilibrium state
        """
        # Estimate characteristic flow time
        characteristic_length = 1.0  # m (typical shock layer thickness)
        flow_time = characteristic_length / np.linalg.norm(initial_state.velocity)
        
        # Setup ODE system for species evolution
        species_names = list(self.chemistry_calculator.species_database.keys())
        n_species = len(species_names)
        
        def chemistry_ode(t, y):
            """ODE system for species concentrations."""
            # Create state from current concentrations
            current_concentrations = {species_names[i]: max(0.0, y[i]) for i in range(n_species)}
            
            current_state = NonEquilibriumState(
                temperature=initial_state.temperature,  # Assume constant T for now
                pressure=initial_state.pressure,
                density=initial_state.density,
                velocity=initial_state.velocity,
                species_concentrations=current_concentrations,
                vibrational_temperatures=initial_state.vibrational_temperatures,
                electronic_temperatures=initial_state.electronic_temperatures
            )
            
            # Calculate production rates
            production_rates = self.chemistry_calculator.calculate_production_rates(current_state)
            
            # Return derivatives
            return [production_rates[species_names[i]] for i in range(n_species)]
        
        # Initial conditions
        y0 = [initial_state.species_concentrations[species] for species in species_names]
        
        # Time span (from 0 to characteristic flow time)
        t_span = (0, flow_time)
        t_eval = np.linspace(0, flow_time, 100)
        
        # Solve ODE system
        try:
            solution = solve_ivp(
                chemistry_ode, t_span, y0, t_eval=t_eval,
                method='Radau',  # Good for stiff systems
                rtol=self.chemistry_tolerance,
                atol=self.chemistry_tolerance * 1e-6
            )
            
            if not solution.success:
                self.logger.warning("Chemistry ODE solver did not converge, using initial state")
                return initial_state
            
            # Extract final concentrations
            final_concentrations = {}
            for i, species in enumerate(species_names):
                final_concentrations[species] = max(0.0, solution.y[i, -1])
            
            # Create final state
            final_state = NonEquilibriumState(
                temperature=initial_state.temperature,
                pressure=initial_state.pressure,
                density=initial_state.density,
                velocity=initial_state.velocity,
                species_concentrations=final_concentrations,
                vibrational_temperatures=initial_state.vibrational_temperatures,
                electronic_temperatures=initial_state.electronic_temperatures
            )
            
            return final_state
            
        except Exception as e:
            self.logger.error(f"Chemistry evolution failed: {str(e)}")
            return initial_state
    
    def _update_flow_conditions_with_chemistry(self, original_conditions: FlowConditions,
                                             evolved_state: NonEquilibriumState) -> FlowConditions:
        """Update flow conditions based on chemistry evolution."""
        # Calculate new mixture properties
        total_mass = 0.0
        total_moles = evolved_state.total_concentration
        
        for species, concentration in evolved_state.species_concentrations.items():
            if species in self.chemistry_calculator.species_database:
                molecular_mass = self.chemistry_calculator.species_database[species].molecular_mass
                total_mass += concentration * molecular_mass
        
        # Calculate new density
        new_density = total_mass / 1000.0  # Convert from g/m³ to kg/m³
        
        # Calculate new pressure (assuming constant temperature for now)
        new_pressure = total_moles * self.chemistry_calculator.R_universal * evolved_state.temperature
        
        # Create updated flow conditions
        updated_conditions = FlowConditions(
            mach_number=original_conditions.mach_number,
            altitude=original_conditions.altitude,
            angle_of_attack=original_conditions.angle_of_attack,
            sideslip_angle=original_conditions.sideslip_angle,
            temperature=evolved_state.temperature,
            pressure=new_pressure,
            density=new_density
        )
        
        return updated_conditions
    
    def _add_chemistry_results(self, plasma_results: PlasmaFlowResults,
                             evolved_state: NonEquilibriumState) -> PlasmaFlowResults:
        """Add chemistry-specific results to plasma flow results."""
        # For now, just return the original results
        # In a full implementation, would add species concentration fields,
        # reaction rate fields, etc.
        
        self.logger.info(f"Final species composition: {evolved_state.species_mole_fractions}")
        
        return plasma_results
    
    def calculate_species_transport(self, state: NonEquilibriumState,
                                  velocity_gradient: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate species transport fluxes.
        
        Args:
            state: Non-equilibrium flow state
            velocity_gradient: Velocity gradient tensor
            
        Returns:
            Dictionary of species diffusion fluxes
        """
        species_fluxes = {}
        
        for species in state.species_concentrations.keys():
            if species in self.chemistry_calculator.species_database:
                # Simplified diffusion calculation
                # In practice, would use multicomponent diffusion
                diffusion_coefficient = self._calculate_diffusion_coefficient(species, state)
                
                # Concentration gradient (simplified as zero for now)
                concentration_gradient = np.zeros(3)
                
                # Diffusion flux: J = -D * grad(C)
                flux = -diffusion_coefficient * concentration_gradient
                species_fluxes[species] = flux
        
        return species_fluxes
    
    def _calculate_diffusion_coefficient(self, species: str, state: NonEquilibriumState) -> float:
        """Calculate binary diffusion coefficient for species.
        
        Args:
            species: Species name
            state: Non-equilibrium flow state
            
        Returns:
            Diffusion coefficient in m²/s
        """
        if species not in self.chemistry_calculator.species_database:
            return 0.0
        
        species_data = self.chemistry_calculator.species_database[species]
        
        # Simplified Chapman-Enskog theory
        # D = (3/16) * sqrt(2π * k_B * T / μ) / (n * σ²)
        # where μ is reduced mass, n is number density, σ is collision diameter
        
        # Use average molecular mass for reduced mass calculation
        avg_molecular_mass = 0.029  # kg/mol for air
        reduced_mass = (species_data.molecular_mass * avg_molecular_mass) / (species_data.molecular_mass + avg_molecular_mass)
        reduced_mass_kg = reduced_mass / 6.022e23  # Convert to kg
        
        # Number density
        number_density = state.total_concentration * 6.022e23  # molecules/m³
        
        # Collision diameter
        sigma = species_data.collision_diameter
        
        # Diffusion coefficient
        diffusion_coeff = (3.0/16.0 * np.sqrt(2 * np.pi * BOLTZMANN_CONSTANT * state.temperature / reduced_mass_kg) /
                          (number_density * sigma**2))
        
        return diffusion_coeff
    
    def get_species_database(self) -> Dict[str, ChemicalSpecies]:
        """Get the species database."""
        return self.chemistry_calculator.species_database
    
    def get_reaction_database(self) -> List[ChemicalReaction]:
        """Get the reaction database."""
        return self.chemistry_calculator.reaction_database
    
    def add_custom_species(self, species: ChemicalSpecies):
        """Add custom species to the database."""
        self.chemistry_calculator.species_database[species.name] = species
    
    def add_custom_reaction(self, reaction: ChemicalReaction):
        """Add custom reaction to the database."""
        self.chemistry_calculator.reaction_database.append(reaction)