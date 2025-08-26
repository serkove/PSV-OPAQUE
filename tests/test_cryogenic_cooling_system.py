"""Tests for cryogenic cooling system."""

import pytest
import numpy as np
import math

from fighter_jet_sdk.engines.propulsion.cryogenic_cooling_system import (
    CryogenicCoolingSystem,
    CoolantType,
    CoolingMode,
    CoolingChannel,
    TranspirationCoolingSpec,
    FilmCoolingSpec,
    CoolingSystemPerformance
)


class TestCryogenicCoolingSystem:
    """Test cryogenic cooling system capabilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cooling_system = CryogenicCoolingSystem()
        
        # Test conditions
        self.extreme_heat_flux = 300e6  # 300 MW/m²
        self.surface_area = 1.0  # m²
        
        # Design constraints
        self.design_constraints = {
            'channel_length': 0.5,  # m
            'channel_width': 0.002,  # 2mm
            'channel_height': 0.001,  # 1mm
            'wall_thickness': 0.0005,  # 0.5mm
            'inlet_pressure': 2e6,  # 20 bar
            'max_pressure_drop': 1e6,  # 10 bar
            'min_effectiveness': 0.8,
            'max_mass_flow': 10.0  # kg/s
        }
    
    def test_system_initialization(self):
        """Test cooling system initialization and coolant database."""
        assert len(self.cooling_system.coolants) > 0
        
        # Check all coolant types are available
        expected_coolants = [
            CoolantType.LIQUID_HYDROGEN,
            CoolantType.LIQUID_NITROGEN,
            CoolantType.LIQUID_HELIUM,
            CoolantType.LIQUID_METHANE
        ]
        
        for coolant_type in expected_coolants:
            assert coolant_type in self.cooling_system.coolants
            coolant = self.cooling_system.coolants[coolant_type]
            
            # Check physical properties are reasonable
            assert coolant.boiling_point > 0
            assert coolant.critical_temperature > coolant.boiling_point
            assert coolant.density_liquid > coolant.density_vapor
            assert coolant.latent_heat_vaporization > 0
            assert coolant.thermal_conductivity_liquid > 0
    
    def test_coolant_properties(self):
        """Test coolant property calculations and consistency."""
        # Test liquid hydrogen properties
        lh2 = self.cooling_system.coolants[CoolantType.LIQUID_HYDROGEN]
        assert lh2.boiling_point < 25.0  # Very low boiling point
        assert lh2.latent_heat_vaporization > 400000.0  # High latent heat
        assert lh2.density_liquid < 100.0  # Low density
        
        # Test liquid nitrogen properties
        ln2 = self.cooling_system.coolants[CoolantType.LIQUID_NITROGEN]
        assert ln2.boiling_point > 75.0  # Higher than hydrogen
        assert ln2.density_liquid > 800.0  # Higher density than hydrogen
        
        # Test liquid helium properties
        lhe = self.cooling_system.coolants[CoolantType.LIQUID_HELIUM]
        assert lhe.boiling_point < 10.0  # Lowest boiling point
        assert lhe.critical_temperature < 10.0  # Very low critical temperature
    
    def test_regenerative_cooling_design(self):
        """Test regenerative cooling system design."""
        design_result = self.cooling_system.design_regenerative_cooling(
            self.extreme_heat_flux,
            self.surface_area,
            CoolantType.LIQUID_HYDROGEN,
            self.design_constraints
        )
        
        # Check design structure
        assert 'channel_design' in design_result
        assert 'performance' in design_result
        assert 'coolant_properties' in design_result
        assert 'total_heat_removal' in design_result
        assert 'design_margins' in design_result
        
        # Check channel design
        channels = design_result['channel_design']
        assert len(channels) > 0
        
        for channel in channels:
            assert isinstance(channel, CoolingChannel)
            assert channel.width > 0
            assert channel.height > 0
            assert channel.length > 0
            assert channel.mass_flow_rate > 0
            assert channel.inlet_pressure > 0
        
        # Check performance
        performance = design_result['performance']
        assert isinstance(performance, CoolingSystemPerformance)
        assert performance.heat_removal_rate > 0
        assert 0 <= performance.cooling_effectiveness <= 1.0
        assert performance.pressure_drop > 0
        assert performance.coolant_consumption > 0
        
        # Check total heat removal matches input
        expected_heat_load = self.extreme_heat_flux * self.surface_area
        assert abs(design_result['total_heat_removal'] - expected_heat_load) / expected_heat_load < 0.01
    
    def test_mass_flow_calculation(self):
        """Test coolant mass flow rate calculations."""
        coolant = self.cooling_system.coolants[CoolantType.LIQUID_HYDROGEN]
        heat_load = 100e6  # 100 MW
        
        mass_flow = self.cooling_system._calculate_required_mass_flow(heat_load, coolant)
        
        assert mass_flow > 0
        
        # Should scale with heat load
        double_heat_mass_flow = self.cooling_system._calculate_required_mass_flow(2 * heat_load, coolant)
        assert double_heat_mass_flow > 1.5 * mass_flow  # Should be roughly 2x
        
        # Different coolants should give different mass flows
        ln2_mass_flow = self.cooling_system._calculate_required_mass_flow(heat_load, 
                                                                         self.cooling_system.coolants[CoolantType.LIQUID_NITROGEN])
        assert ln2_mass_flow != mass_flow
    
    def test_heat_transfer_calculations(self):
        """Test heat transfer coefficient calculations."""
        # Create test channel
        channel = CoolingChannel(
            channel_id="test_channel",
            geometry_type="rectangular",
            width=0.002,  # 2mm
            height=0.001,  # 1mm
            length=0.5,   # 0.5m
            wall_thickness=0.0005,
            surface_roughness=1e-6,
            inlet_temperature=25.0,  # K
            inlet_pressure=2e6,
            mass_flow_rate=0.1  # kg/s
        )
        
        coolant = self.cooling_system.coolants[CoolantType.LIQUID_HYDROGEN]
        
        # Test Reynolds number calculation
        reynolds = self.cooling_system._calculate_reynolds_number(channel, coolant)
        assert reynolds > 0
        
        # Test Prandtl number calculation
        prandtl = self.cooling_system._calculate_prandtl_number(coolant)
        assert prandtl > 0
        
        # Test Nusselt number calculation
        nusselt = self.cooling_system._calculate_nusselt_number(reynolds, prandtl, channel)
        assert nusselt > 0
        
        # Test friction factor calculation
        friction_factor = self.cooling_system._calculate_friction_factor(reynolds, channel)
        assert friction_factor > 0
        
        # Higher Reynolds should give higher Nusselt (better heat transfer)
        high_flow_channel = CoolingChannel(
            channel_id="high_flow_channel",
            geometry_type="rectangular",
            width=channel.width,
            height=channel.height,
            length=channel.length,
            wall_thickness=channel.wall_thickness,
            surface_roughness=channel.surface_roughness,
            inlet_temperature=channel.inlet_temperature,
            inlet_pressure=channel.inlet_pressure,
            mass_flow_rate=channel.mass_flow_rate * 2  # Double flow rate
        )
        
        high_reynolds = self.cooling_system._calculate_reynolds_number(high_flow_channel, coolant)
        high_nusselt = self.cooling_system._calculate_nusselt_number(high_reynolds, prandtl, high_flow_channel)
        
        assert high_reynolds > reynolds
        assert high_nusselt > nusselt
    
    def test_transpiration_cooling(self):
        """Test transpiration cooling calculations."""
        transpiration_spec = TranspirationCoolingSpec(
            porous_material_id="porous_tungsten",
            porosity=0.3,
            permeability=1e-12,  # m²
            pore_diameter=50e-6,  # 50 microns
            thickness=0.005,  # 5mm
            coolant_injection_rate=0.1,  # kg/(m²⋅s)
            injection_temperature=25.0,  # K
            injection_pressure=1e6  # 10 bar
        )
        
        operating_conditions = {
            'surface_area': 1.0,  # m²
            'back_pressure': 1e5  # 1 bar
        }
        
        performance = self.cooling_system.calculate_transpiration_cooling(
            self.extreme_heat_flux,
            transpiration_spec,
            CoolantType.LIQUID_HYDROGEN,
            operating_conditions
        )
        
        # Check performance results
        assert isinstance(performance, CoolingSystemPerformance)
        assert performance.heat_removal_rate > 0
        assert 0 <= performance.cooling_effectiveness <= 1.0
        assert performance.pressure_drop > 0
        assert performance.coolant_consumption > 0
        assert performance.transpiration_effectiveness is not None
        assert 0 <= performance.transpiration_effectiveness <= 1.0
        
        # Higher injection rate should improve effectiveness
        high_injection_spec = TranspirationCoolingSpec(
            porous_material_id=transpiration_spec.porous_material_id,
            porosity=transpiration_spec.porosity,
            permeability=transpiration_spec.permeability,
            pore_diameter=transpiration_spec.pore_diameter,
            thickness=transpiration_spec.thickness,
            coolant_injection_rate=transpiration_spec.coolant_injection_rate * 2,
            injection_temperature=transpiration_spec.injection_temperature,
            injection_pressure=transpiration_spec.injection_pressure
        )
        
        high_performance = self.cooling_system.calculate_transpiration_cooling(
            self.extreme_heat_flux,
            high_injection_spec,
            CoolantType.LIQUID_HYDROGEN,
            operating_conditions
        )
        
        assert high_performance.cooling_effectiveness >= performance.cooling_effectiveness
        assert high_performance.coolant_consumption > performance.coolant_consumption
    
    def test_film_cooling(self):
        """Test film cooling calculations."""
        film_spec = FilmCoolingSpec(
            injection_angle=30.0,  # degrees
            hole_diameter=0.001,  # 1mm
            hole_spacing=0.005,  # 5mm
            number_of_holes=100,
            blowing_ratio=1.0,
            density_ratio=10.0,
            momentum_ratio=1.0
        )
        
        mainstream_conditions = {
            'surface_area': 1.0,  # m²
            'temperature': 3000.0,  # K
            'density': 0.1,  # kg/m³
            'velocity': 20000.0  # m/s
        }
        
        performance = self.cooling_system.calculate_film_cooling(
            self.extreme_heat_flux,
            film_spec,
            CoolantType.LIQUID_HYDROGEN,
            mainstream_conditions
        )
        
        # Check performance results
        assert isinstance(performance, CoolingSystemPerformance)
        assert performance.heat_removal_rate > 0
        assert 0 <= performance.cooling_effectiveness <= 1.0
        assert performance.pressure_drop > 0
        assert performance.coolant_consumption > 0
        assert performance.film_effectiveness is not None
        assert 0 <= performance.film_effectiveness <= 1.0
        
        # Test blowing ratio effects
        low_blowing_spec = FilmCoolingSpec(
            injection_angle=film_spec.injection_angle,
            hole_diameter=film_spec.hole_diameter,
            hole_spacing=film_spec.hole_spacing,
            number_of_holes=film_spec.number_of_holes,
            blowing_ratio=0.5,  # Lower blowing ratio
            density_ratio=film_spec.density_ratio,
            momentum_ratio=film_spec.momentum_ratio
        )
        
        low_performance = self.cooling_system.calculate_film_cooling(
            self.extreme_heat_flux,
            low_blowing_spec,
            CoolantType.LIQUID_HYDROGEN,
            mainstream_conditions
        )
        
        # Lower blowing ratio should give different performance
        assert low_performance.film_effectiveness != performance.film_effectiveness
        assert low_performance.coolant_consumption < performance.coolant_consumption
    
    def test_cooling_system_optimization(self):
        """Test cooling system optimization."""
        cooling_modes = [CoolingMode.REGENERATIVE, CoolingMode.TRANSPIRATION, CoolingMode.FILM_COOLING]
        
        # Test effectiveness optimization
        effectiveness_result = self.cooling_system.optimize_cooling_system(
            self.extreme_heat_flux,
            cooling_modes,
            CoolantType.LIQUID_HYDROGEN,
            "effectiveness"
        )
        
        assert 'best_configuration' in effectiveness_result
        assert 'best_performance' in effectiveness_result
        assert 'optimization_metric' in effectiveness_result
        assert 'cooling_mode' in effectiveness_result
        
        assert effectiveness_result['best_configuration'] is not None
        assert isinstance(effectiveness_result['best_performance'], CoolingSystemPerformance)
        assert effectiveness_result['optimization_metric'] > 0
        
        # Test mass flow optimization
        mass_flow_result = self.cooling_system.optimize_cooling_system(
            self.extreme_heat_flux,
            cooling_modes,
            CoolantType.LIQUID_HYDROGEN,
            "mass_flow"
        )
        
        assert mass_flow_result['best_configuration'] is not None
        assert isinstance(mass_flow_result['best_performance'], CoolingSystemPerformance)
        
        # Mass flow optimization should give different result than effectiveness
        assert (mass_flow_result['optimization_metric'] != 
                effectiveness_result['optimization_metric'])
    
    def test_transpiration_velocity_calculation(self):
        """Test transpiration velocity calculations."""
        spec = TranspirationCoolingSpec(
            porous_material_id="test_material",
            porosity=0.4,
            permeability=1e-11,  # m²
            pore_diameter=100e-6,
            thickness=0.01,  # 10mm
            coolant_injection_rate=0.2,
            injection_temperature=30.0,
            injection_pressure=2e6  # 20 bar
        )
        
        coolant = self.cooling_system.coolants[CoolantType.LIQUID_NITROGEN]
        conditions = {'back_pressure': 1e5}  # 1 bar
        
        velocity = self.cooling_system._calculate_transpiration_velocity(spec, coolant, conditions)
        
        assert velocity > 0
        
        # Higher pressure difference should give higher velocity
        high_pressure_conditions = {'back_pressure': 5e4}  # 0.5 bar
        high_velocity = self.cooling_system._calculate_transpiration_velocity(
            spec, coolant, high_pressure_conditions
        )
        
        assert high_velocity > velocity
        
        # Higher permeability should give higher velocity
        high_perm_spec = TranspirationCoolingSpec(
            porous_material_id=spec.porous_material_id,
            porosity=spec.porosity,
            permeability=spec.permeability * 10,  # 10x higher permeability
            pore_diameter=spec.pore_diameter,
            thickness=spec.thickness,
            coolant_injection_rate=spec.coolant_injection_rate,
            injection_temperature=spec.injection_temperature,
            injection_pressure=spec.injection_pressure
        )
        
        high_perm_velocity = self.cooling_system._calculate_transpiration_velocity(
            high_perm_spec, coolant, conditions
        )
        
        assert high_perm_velocity > velocity
    
    def test_film_effectiveness_correlation(self):
        """Test film cooling effectiveness correlations."""
        coolant = self.cooling_system.coolants[CoolantType.LIQUID_HYDROGEN]
        mainstream = {'temperature': 3000.0, 'density': 0.1, 'velocity': 20000.0}
        
        # Test different blowing ratios
        blowing_ratios = [0.3, 0.8, 1.5, 3.0]
        effectiveness_values = []
        
        for M in blowing_ratios:
            spec = FilmCoolingSpec(
                injection_angle=30.0,
                hole_diameter=0.001,
                hole_spacing=0.005,
                number_of_holes=100,
                blowing_ratio=M,
                density_ratio=10.0,
                momentum_ratio=1.0
            )
            
            effectiveness = self.cooling_system._calculate_film_effectiveness(spec, coolant, mainstream)
            effectiveness_values.append(effectiveness)
            
            assert 0 <= effectiveness <= 1.0
        
        # Check that effectiveness behavior is reasonable
        # Low blowing ratios should have lower effectiveness
        # Very high blowing ratios should also have lower effectiveness (jet lift-off)
        assert len(effectiveness_values) == len(blowing_ratios)
        
        # Find maximum effectiveness
        max_effectiveness = max(effectiveness_values)
        max_index = effectiveness_values.index(max_effectiveness)
        
        # Maximum should not be at the extremes (physical expectation)
        assert 0 < max_index < len(effectiveness_values) - 1
    
    def test_pressure_drop_calculations(self):
        """Test pressure drop calculations."""
        # Test transpiration pressure drop
        spec = TranspirationCoolingSpec(
            porous_material_id="test_material",
            porosity=0.3,
            permeability=1e-12,
            pore_diameter=50e-6,
            thickness=0.005,
            coolant_injection_rate=0.1,
            injection_temperature=25.0,
            injection_pressure=1e6
        )
        
        coolant = self.cooling_system.coolants[CoolantType.LIQUID_HYDROGEN]
        velocity = 0.1  # m/s
        
        pressure_drop = self.cooling_system._calculate_transpiration_pressure_drop(
            spec, coolant, velocity
        )
        
        assert pressure_drop > 0
        
        # Higher velocity should give higher pressure drop
        high_velocity_drop = self.cooling_system._calculate_transpiration_pressure_drop(
            spec, coolant, velocity * 2
        )
        
        assert high_velocity_drop > pressure_drop
        
        # Lower permeability should give higher pressure drop
        low_perm_spec = TranspirationCoolingSpec(
            porous_material_id=spec.porous_material_id,
            porosity=spec.porosity,
            permeability=spec.permeability / 10,  # 10x lower permeability
            pore_diameter=spec.pore_diameter,
            thickness=spec.thickness,
            coolant_injection_rate=spec.coolant_injection_rate,
            injection_temperature=spec.injection_temperature,
            injection_pressure=spec.injection_pressure
        )
        
        low_perm_drop = self.cooling_system._calculate_transpiration_pressure_drop(
            low_perm_spec, coolant, velocity
        )
        
        assert low_perm_drop > pressure_drop
    
    def test_design_validation(self):
        """Test cooling system design validation."""
        # Create test performance
        good_performance = CoolingSystemPerformance(
            heat_removal_rate=200e6,  # 200 MW
            cooling_effectiveness=0.85,
            pressure_drop=5e5,  # 5 bar
            coolant_consumption=2.0,  # kg/s
            surface_temperature_reduction=1000.0  # K
        )
        
        design_requirements = {
            'min_effectiveness': 0.8,
            'max_pressure_drop': 1e6,  # 10 bar
            'max_mass_flow': 5.0,  # kg/s
            'min_temperature_reduction': 500.0  # K
        }
        
        warnings = self.cooling_system.validate_cooling_design(good_performance, design_requirements)
        assert len(warnings) == 0  # Should pass all requirements
        
        # Test failing performance
        bad_performance = CoolingSystemPerformance(
            heat_removal_rate=100e6,
            cooling_effectiveness=0.6,  # Too low
            pressure_drop=2e6,  # Too high
            coolant_consumption=10.0,  # Too high
            surface_temperature_reduction=200.0  # Too low
        )
        
        warnings = self.cooling_system.validate_cooling_design(bad_performance, design_requirements)
        assert len(warnings) > 0  # Should have multiple warnings
        
        # Check specific warning types
        warning_text = ' '.join(warnings)
        assert 'effectiveness' in warning_text
        assert 'pressure drop' in warning_text
        assert 'consumption' in warning_text
        assert 'temperature reduction' in warning_text
    
    def test_coolant_comparison(self):
        """Test performance comparison between different coolants."""
        coolant_types = [
            CoolantType.LIQUID_HYDROGEN,
            CoolantType.LIQUID_NITROGEN,
            CoolantType.LIQUID_HELIUM,
            CoolantType.LIQUID_METHANE
        ]
        
        results = {}
        
        for coolant_type in coolant_types:
            try:
                design_result = self.cooling_system.design_regenerative_cooling(
                    self.extreme_heat_flux,
                    self.surface_area,
                    coolant_type,
                    self.design_constraints
                )
                
                results[coolant_type] = design_result['performance']
                
            except Exception as e:
                pytest.fail(f"Failed to design cooling system with {coolant_type}: {e}")
        
        # All coolants should produce valid results
        assert len(results) == len(coolant_types)
        
        # Compare performance characteristics
        for coolant_type, performance in results.items():
            assert isinstance(performance, CoolingSystemPerformance)
            assert performance.heat_removal_rate > 0
            assert performance.cooling_effectiveness > 0
            assert performance.coolant_consumption > 0
        
        # Liquid hydrogen should generally have good performance due to high specific heat
        lh2_performance = results[CoolantType.LIQUID_HYDROGEN]
        ln2_performance = results[CoolantType.LIQUID_NITROGEN]
        
        # Both should be effective, but may have different characteristics
        assert lh2_performance.cooling_effectiveness > 0.5
        assert ln2_performance.cooling_effectiveness > 0.5
    
    def test_extreme_conditions_handling(self):
        """Test system behavior under extreme conditions."""
        # Very high heat flux
        extreme_heat_flux = 1e9  # 1 GW/m²
        
        try:
            design_result = self.cooling_system.design_regenerative_cooling(
                extreme_heat_flux,
                self.surface_area,
                CoolantType.LIQUID_HYDROGEN,
                self.design_constraints
            )
            
            # Should complete without errors
            assert 'performance' in design_result
            performance = design_result['performance']
            
            # Mass flow should be very high for such extreme conditions
            assert performance.coolant_consumption > 10.0  # kg/s
            
        except Exception as e:
            # If it fails, should be due to physical limitations, not code errors
            assert "physical" in str(e).lower() or "limit" in str(e).lower()
    
    def test_channel_optimization(self):
        """Test cooling channel optimization."""
        # Create initial channels
        initial_channels = [
            CoolingChannel(
                channel_id="test_channel",
                geometry_type="rectangular",
                width=0.002,
                height=0.001,
                length=0.5,
                wall_thickness=0.0005,
                surface_roughness=1e-6,
                inlet_temperature=25.0,
                inlet_pressure=2e6,
                mass_flow_rate=0.1
            )
        ]
        
        # Create test performance
        performance = CoolingSystemPerformance(
            heat_removal_rate=100e6,
            cooling_effectiveness=0.7,  # Below target
            pressure_drop=5e5,
            coolant_consumption=1.0,
            surface_temperature_reduction=800.0
        )
        
        constraints = {
            'min_effectiveness': 0.8,
            'max_pressure_drop': 1e6
        }
        
        optimized_channels = self.cooling_system._optimize_channel_configuration(
            initial_channels, performance, constraints
        )
        
        assert len(optimized_channels) == len(initial_channels)
        
        # Optimized channels should have different dimensions
        original_channel = initial_channels[0]
        optimized_channel = optimized_channels[0]
        
        # At least one dimension should change
        dimension_changed = (
            optimized_channel.width != original_channel.width or
            optimized_channel.height != original_channel.height
        )
        assert dimension_changed


if __name__ == "__main__":
    pytest.main([__file__])