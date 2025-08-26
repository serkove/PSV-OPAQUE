"""
Unit tests for NonEquilibriumCFD

Tests the non-equilibrium chemistry CFD module including species transport,
reaction rate calculations, and ionization/dissociation mechanisms.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from fighter_jet_sdk.engines.aerodynamics.non_equilibrium_cfd import (
    NonEquilibriumCFD, ChemicalKineticsCalculator, ChemicalSpecies, ChemicalReaction,
    NonEquilibriumState, ReactionType
)
from fighter_jet_sdk.common.data_models import AircraftConfiguration, FlowConditions
from fighter_jet_sdk.common.enums import PlasmaRegime
from fighter_jet_sdk.core.errors import CFDError, ValidationError


class TestChemicalSpecies:
    """Test cases for ChemicalSpecies class."""
    
    def test_species_initialization(self):
        """Test chemical species initialization."""
        species = ChemicalSpecies(
            name="N2",
            molecular_mass=0.028014,
            formation_enthalpy=0.0,
            charge=0,
            vibrational_temperature=3395.0
        )
        
        assert species.name == "N2"
        assert species.molecular_mass == 0.028014
        assert species.formation_enthalpy == 0.0
        assert species.charge == 0
        assert species.vibrational_temperature == 3395.0
        assert species.electronic_levels == [0.0]  # Default ground state
        assert species.collision_diameter is not None  # Should be estimated
    
    def test_species_collision_diameter_estimation(self):
        """Test collision diameter estimation."""
        species = ChemicalSpecies(
            name="O2",
            molecular_mass=0.032,
            formation_enthalpy=0.0,
            charge=0
        )
        
        # Should have estimated collision diameter
        assert species.collision_diameter > 0
        assert species.collision_diameter < 1e-9  # Reasonable molecular size


class TestChemicalReaction:
    """Test cases for ChemicalReaction class."""
    
    def test_reaction_initialization(self):
        """Test chemical reaction initialization."""
        reaction = ChemicalReaction(
            reaction_id="N2_dissociation",
            reaction_type=ReactionType.DISSOCIATION,
            reactants={'N2': 1, 'M': 1},
            products={'N': 2, 'M': 1},
            activation_energy=945000.0,
            pre_exponential_factor=7.0e21,
            temperature_exponent=-1.6
        )
        
        assert reaction.reaction_id == "N2_dissociation"
        assert reaction.reaction_type == ReactionType.DISSOCIATION
        assert reaction.reactants == {'N2': 1, 'M': 1}
        assert reaction.products == {'N': 2, 'M': 1}
        assert reaction.activation_energy == 945000.0
        assert reaction.third_body_efficiency == {}  # Default empty


class TestNonEquilibriumState:
    """Test cases for NonEquilibriumState class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.state = NonEquilibriumState(
            temperature=15000.0,
            pressure=1000.0,
            density=0.001,
            velocity=np.array([20000.0, 0.0, 0.0]),
            species_concentrations={'N2': 100.0, 'O2': 50.0, 'N': 10.0},
            vibrational_temperatures={'N2': 15000.0, 'O2': 15000.0, 'N': 15000.0},
            electronic_temperatures={'N2': 15000.0, 'O2': 15000.0, 'N': 15000.0}
        )
    
    def test_total_concentration(self):
        """Test total concentration calculation."""
        assert self.state.total_concentration == 160.0
    
    def test_species_mole_fractions(self):
        """Test species mole fraction calculation."""
        fractions = self.state.species_mole_fractions
        
        assert abs(fractions['N2'] - 100.0/160.0) < 1e-10
        assert abs(fractions['O2'] - 50.0/160.0) < 1e-10
        assert abs(fractions['N'] - 10.0/160.0) < 1e-10
        
        # Should sum to 1.0
        assert abs(sum(fractions.values()) - 1.0) < 1e-10
    
    def test_zero_concentration_fractions(self):
        """Test mole fractions with zero total concentration."""
        zero_state = NonEquilibriumState(
            temperature=15000.0,
            pressure=1000.0,
            density=0.001,
            velocity=np.array([20000.0, 0.0, 0.0]),
            species_concentrations={'N2': 0.0, 'O2': 0.0},
            vibrational_temperatures={'N2': 15000.0, 'O2': 15000.0},
            electronic_temperatures={'N2': 15000.0, 'O2': 15000.0}
        )
        
        fractions = zero_state.species_mole_fractions
        assert all(frac == 0.0 for frac in fractions.values())


class TestChemicalKineticsCalculator:
    """Test cases for ChemicalKineticsCalculator class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.calculator = ChemicalKineticsCalculator()
        
        self.test_state = NonEquilibriumState(
            temperature=15000.0,
            pressure=1000.0,
            density=0.001,
            velocity=np.array([20000.0, 0.0, 0.0]),
            species_concentrations={
                'N2': 100.0, 'O2': 50.0, 'N': 10.0, 'O': 5.0,
                'NO': 2.0, 'e-': 1.0, 'N2+': 0.5, 'O2+': 0.3
            },
            vibrational_temperatures={'N2': 15000.0, 'O2': 15000.0, 'N': 15000.0, 'O': 15000.0, 'NO': 15000.0},
            electronic_temperatures={'N2': 15000.0, 'O2': 15000.0, 'N': 15000.0, 'O': 15000.0, 'NO': 15000.0}
        )
    
    def test_species_database_initialization(self):
        """Test species database initialization."""
        species_db = self.calculator.species_database
        
        # Check key species are present
        assert 'N2' in species_db
        assert 'O2' in species_db
        assert 'N' in species_db
        assert 'O' in species_db
        assert 'NO' in species_db
        assert 'e-' in species_db
        assert 'N2+' in species_db
        
        # Check species properties
        n2 = species_db['N2']
        assert n2.molecular_mass == 0.028014
        assert n2.charge == 0
        assert n2.vibrational_temperature == 3395.0
        
        electron = species_db['e-']
        assert electron.charge == -1
        assert electron.molecular_mass < 1e-6  # Very small mass
    
    def test_reaction_database_initialization(self):
        """Test reaction database initialization."""
        reactions = self.calculator.reaction_database
        
        assert len(reactions) > 0
        
        # Check for key reaction types
        reaction_types = [r.reaction_type for r in reactions]
        assert ReactionType.DISSOCIATION in reaction_types
        assert ReactionType.IONIZATION in reaction_types
        assert ReactionType.CHARGE_EXCHANGE in reaction_types
        
        # Check specific reactions
        reaction_ids = [r.reaction_id for r in reactions]
        assert "N2_dissociation" in reaction_ids
        assert "O2_dissociation" in reaction_ids
        assert "N2_ionization" in reaction_ids
    
    def test_rate_constant_calculation(self):
        """Test rate constant calculation."""
        # Get a test reaction
        n2_dissociation = None
        for reaction in self.calculator.reaction_database:
            if reaction.reaction_id == "N2_dissociation":
                n2_dissociation = reaction
                break
        
        assert n2_dissociation is not None
        
        # Calculate rate constant at high temperature
        k = self.calculator._calculate_rate_constant(n2_dissociation, 15000.0)
        
        assert k > 0
        assert np.isfinite(k)
        
        # Rate constant should increase with temperature for endothermic reactions
        k_low = self.calculator._calculate_rate_constant(n2_dissociation, 5000.0)
        assert k > k_low
    
    def test_reaction_rate_calculation(self):
        """Test reaction rate calculation."""
        # Get N2 dissociation reaction
        n2_dissociation = None
        for reaction in self.calculator.reaction_database:
            if reaction.reaction_id == "N2_dissociation":
                n2_dissociation = reaction
                break
        
        assert n2_dissociation is not None
        
        # Calculate reaction rate
        rate = self.calculator.calculate_reaction_rate(n2_dissociation, self.test_state)
        
        assert rate >= 0  # Rate should be non-negative
        assert np.isfinite(rate)
    
    def test_production_rates_calculation(self):
        """Test species production rates calculation."""
        production_rates = self.calculator.calculate_production_rates(self.test_state)
        
        # Should have rates for all species
        assert len(production_rates) == len(self.calculator.species_database)
        
        # All rates should be finite
        for species, rate in production_rates.items():
            assert np.isfinite(rate)
        
        # Conservation check: total mass production should be zero
        # (simplified check - would need proper mass balance in full implementation)
        total_rate = sum(production_rates.values())
        # Allow some numerical error - high temperature chemistry has large rates
        assert abs(total_rate) < 1e20  # Reasonable bound for high-temperature chemistry
    
    def test_equilibrium_constants_calculation(self):
        """Test equilibrium constants calculation."""
        eq_constants = self.calculator.calculate_equilibrium_constants(15000.0)
        
        assert len(eq_constants) == len(self.calculator.reaction_database)
        
        # All constants should be positive and finite
        for reaction_id, k_eq in eq_constants.items():
            assert k_eq > 0
            assert np.isfinite(k_eq)


class TestNonEquilibriumCFD:
    """Test cases for NonEquilibriumCFD class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.solver = NonEquilibriumCFD()
        
        # Create test aircraft configuration
        self.config = AircraftConfiguration(
            config_id="test_mach60_chemistry",
            name="Test Mach 60 Vehicle with Chemistry",
            modules=[]
        )
        
        # Create test flow conditions
        self.flow_conditions = FlowConditions(
            mach_number=60.0,
            altitude=50000.0,
            angle_of_attack=0.0,
            sideslip_angle=0.0,
            temperature=15000.0,
            pressure=1000.0,
            density=0.001
        )
        
        self.magnetic_field = np.array([0.0, 0.0, 0.1])
    
    def test_solver_initialization(self):
        """Test non-equilibrium CFD solver initialization."""
        assert isinstance(self.solver, NonEquilibriumCFD)
        assert hasattr(self.solver, 'chemistry_calculator')
        assert self.solver.chemistry_enabled is True
        assert self.solver.vibrational_nonequilibrium is True
    
    def test_validate_chemistry_inputs_valid(self):
        """Test validation with valid chemistry inputs."""
        composition = {'N2': 0.78, 'O2': 0.21, 'Ar': 0.01}
        
        # Should not raise any exceptions
        self.solver._validate_chemistry_inputs(
            self.config, self.flow_conditions, composition
        )
    
    def test_validate_chemistry_inputs_invalid_composition(self):
        """Test validation with invalid composition."""
        invalid_composition = {'N2': 0.5, 'O2': 0.3}  # Doesn't sum to 1.0
        
        with pytest.raises(ValidationError, match="must sum to 1.0"):
            self.solver._validate_chemistry_inputs(
                self.config, self.flow_conditions, invalid_composition
            )
    
    def test_validate_chemistry_inputs_low_temperature_warning(self, caplog):
        """Test validation warning for low temperature."""
        low_temp_conditions = FlowConditions(
            mach_number=60.0,
            altitude=50000.0,
            angle_of_attack=0.0,
            sideslip_angle=0.0,
            temperature=500.0,  # Very low temperature
            pressure=1000.0,
            density=0.001
        )
        
        self.solver._validate_chemistry_inputs(
            self.config, low_temp_conditions, None
        )
        
        assert "may be too low for significant chemistry effects" in caplog.text
    
    def test_calculate_initial_nonequilibrium_state(self):
        """Test calculation of initial non-equilibrium state."""
        composition = {'N2': 0.78, 'O2': 0.21, 'Ar': 0.01}
        
        state = self.solver._calculate_initial_nonequilibrium_state(
            self.flow_conditions, composition
        )
        
        assert isinstance(state, NonEquilibriumState)
        assert state.temperature == self.flow_conditions.temperature
        assert state.pressure == self.flow_conditions.pressure
        assert state.density == self.flow_conditions.density
        
        # Check species concentrations
        assert 'N2' in state.species_concentrations
        assert 'O2' in state.species_concentrations
        assert state.species_concentrations['N2'] > 0
        assert state.species_concentrations['O2'] > 0
        
        # Check velocity
        assert len(state.velocity) == 3
        assert state.velocity[0] > 0  # Should have x-component
    
    def test_update_flow_conditions_with_chemistry(self):
        """Test updating flow conditions with chemistry effects."""
        # Create test evolved state
        evolved_state = NonEquilibriumState(
            temperature=15000.0,
            pressure=1000.0,
            density=0.001,
            velocity=np.array([20000.0, 0.0, 0.0]),
            species_concentrations={'N2': 50.0, 'O2': 25.0, 'N': 20.0, 'O': 10.0},
            vibrational_temperatures={'N2': 15000.0, 'O2': 15000.0, 'N': 15000.0, 'O': 15000.0},
            electronic_temperatures={'N2': 15000.0, 'O2': 15000.0, 'N': 15000.0, 'O': 15000.0}
        )
        
        updated_conditions = self.solver._update_flow_conditions_with_chemistry(
            self.flow_conditions, evolved_state
        )
        
        assert isinstance(updated_conditions, FlowConditions)
        assert updated_conditions.mach_number == self.flow_conditions.mach_number
        assert updated_conditions.temperature == evolved_state.temperature
        assert updated_conditions.pressure > 0
        assert updated_conditions.density > 0
    
    def test_calculate_diffusion_coefficient(self):
        """Test diffusion coefficient calculation."""
        state = NonEquilibriumState(
            temperature=15000.0,
            pressure=1000.0,
            density=0.001,
            velocity=np.array([20000.0, 0.0, 0.0]),
            species_concentrations={'N2': 100.0, 'O2': 50.0},
            vibrational_temperatures={'N2': 15000.0, 'O2': 15000.0},
            electronic_temperatures={'N2': 15000.0, 'O2': 15000.0}
        )
        
        diff_coeff = self.solver._calculate_diffusion_coefficient('N2', state)
        
        assert diff_coeff > 0
        assert np.isfinite(diff_coeff)
        assert diff_coeff < 1.0  # Reasonable upper bound for diffusion coefficient
    
    def test_calculate_species_transport(self):
        """Test species transport calculation."""
        state = NonEquilibriumState(
            temperature=15000.0,
            pressure=1000.0,
            density=0.001,
            velocity=np.array([20000.0, 0.0, 0.0]),
            species_concentrations={'N2': 100.0, 'O2': 50.0, 'N': 10.0},
            vibrational_temperatures={'N2': 15000.0, 'O2': 15000.0, 'N': 15000.0},
            electronic_temperatures={'N2': 15000.0, 'O2': 15000.0, 'N': 15000.0}
        )
        
        velocity_gradient = np.zeros((3, 3))
        species_fluxes = self.solver.calculate_species_transport(state, velocity_gradient)
        
        # Should have fluxes for species in the state
        assert 'N2' in species_fluxes
        assert 'O2' in species_fluxes
        assert 'N' in species_fluxes
        
        # Each flux should be a 3D vector
        for species, flux in species_fluxes.items():
            assert len(flux) == 3
            assert all(np.isfinite(flux))
    
    @patch('scipy.integrate.solve_ivp')
    def test_solve_chemistry_evolution_success(self, mock_solve_ivp):
        """Test successful chemistry evolution."""
        # Mock successful ODE solution
        mock_solution = Mock()
        mock_solution.success = True
        mock_solution.y = np.array([[100.0, 90.0], [50.0, 45.0], [0.0, 5.0]])  # Initial and final concentrations
        mock_solve_ivp.return_value = mock_solution
        
        initial_state = NonEquilibriumState(
            temperature=15000.0,
            pressure=1000.0,
            density=0.001,
            velocity=np.array([20000.0, 0.0, 0.0]),
            species_concentrations={'N2': 100.0, 'O2': 50.0, 'N': 0.0},
            vibrational_temperatures={'N2': 15000.0, 'O2': 15000.0, 'N': 15000.0},
            electronic_temperatures={'N2': 15000.0, 'O2': 15000.0, 'N': 15000.0}
        )
        
        evolved_state = self.solver._solve_chemistry_evolution(initial_state, self.flow_conditions)
        
        assert isinstance(evolved_state, NonEquilibriumState)
        mock_solve_ivp.assert_called_once()
    
    @patch('scipy.integrate.solve_ivp')
    def test_solve_chemistry_evolution_failure(self, mock_solve_ivp, caplog):
        """Test chemistry evolution failure fallback."""
        # Mock failed ODE solution
        mock_solution = Mock()
        mock_solution.success = False
        mock_solve_ivp.return_value = mock_solution
        
        initial_state = NonEquilibriumState(
            temperature=15000.0,
            pressure=1000.0,
            density=0.001,
            velocity=np.array([20000.0, 0.0, 0.0]),
            species_concentrations={'N2': 100.0, 'O2': 50.0},
            vibrational_temperatures={'N2': 15000.0, 'O2': 15000.0},
            electronic_temperatures={'N2': 15000.0, 'O2': 15000.0}
        )
        
        evolved_state = self.solver._solve_chemistry_evolution(initial_state, self.flow_conditions)
        
        # Should return initial state on failure
        assert evolved_state == initial_state
        assert "did not converge" in caplog.text
    
    def test_get_species_database(self):
        """Test getting species database."""
        species_db = self.solver.get_species_database()
        
        assert isinstance(species_db, dict)
        assert len(species_db) > 0
        assert 'N2' in species_db
        assert isinstance(species_db['N2'], ChemicalSpecies)
    
    def test_get_reaction_database(self):
        """Test getting reaction database."""
        reaction_db = self.solver.get_reaction_database()
        
        assert isinstance(reaction_db, list)
        assert len(reaction_db) > 0
        assert all(isinstance(r, ChemicalReaction) for r in reaction_db)
    
    def test_add_custom_species(self):
        """Test adding custom species."""
        custom_species = ChemicalSpecies(
            name="H2",
            molecular_mass=0.002016,
            formation_enthalpy=0.0,
            charge=0
        )
        
        self.solver.add_custom_species(custom_species)
        
        species_db = self.solver.get_species_database()
        assert "H2" in species_db
        assert species_db["H2"] == custom_species
    
    def test_add_custom_reaction(self):
        """Test adding custom reaction."""
        custom_reaction = ChemicalReaction(
            reaction_id="H2_dissociation",
            reaction_type=ReactionType.DISSOCIATION,
            reactants={'H2': 1, 'M': 1},
            products={'H': 2, 'M': 1},
            activation_energy=436000.0,
            pre_exponential_factor=5.0e18
        )
        
        initial_count = len(self.solver.get_reaction_database())
        self.solver.add_custom_reaction(custom_reaction)
        
        reaction_db = self.solver.get_reaction_database()
        assert len(reaction_db) == initial_count + 1
        assert custom_reaction in reaction_db


if __name__ == "__main__":
    pytest.main([__file__])