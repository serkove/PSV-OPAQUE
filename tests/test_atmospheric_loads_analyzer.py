"""Tests for atmospheric loads analyzer."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from fighter_jet_sdk.engines.structural.atmospheric_loads_analyzer import (
    AtmosphericLoadsAnalyzer, AtmosphericConditions, HypersonicLoadResults, StructuralLoads
)


class TestAtmosphericConditions:
    """Test cases for AtmosphericConditions."""
    
    def test_troposphere_conditions(self):
        """Test atmospheric conditions in troposphere (0-11 km)."""
        conditions = AtmosphericConditions(altitude=5000.0, mach_number=2.0)
        
        assert conditions.temperature is not None
        assert conditions.pressure is not None
        assert conditions.density is not None
        assert conditions.speed_of_sound is not None
        assert conditions.dynamic_pressure is not None
        
        # Check reasonable values for 5 km altitude
        assert 200 < conditions.temperature < 300  # K
        assert 50000 < conditions.pressure < 110000  # Pa
        assert 0.5 < conditions.density < 1.5  # kg/m³
        assert conditions.dynamic_pressure > 0
    
    def test_stratosphere_conditions(self):
        """Test atmospheric conditions in stratosphere (11-47 km)."""
        conditions = AtmosphericConditions(altitude=25000.0, mach_number=5.0)
        
        assert conditions.temperature is not None
        assert conditions.pressure is not None
        assert conditions.density is not None
        
        # Check reasonable values for 25 km altitude
        assert 200 < conditions.temperature < 250  # K
        assert 1000 < conditions.pressure < 10000  # Pa
        assert 0.01 < conditions.density < 0.1  # kg/m³
    
    def test_mesosphere_conditions(self):
        """Test atmospheric conditions in mesosphere (47-71 km)."""
        conditions = AtmosphericConditions(altitude=60000.0, mach_number=25.0)
        
        assert conditions.temperature is not None
        assert conditions.pressure is not None
        assert conditions.density is not None
        
        # Check reasonable values for 60 km altitude
        assert 200 < conditions.temperature < 300  # K
        assert 1 < conditions.pressure < 100  # Pa
        assert 1e-5 < conditions.density < 1e-3  # kg/m³
    
    def test_thermosphere_conditions(self):
        """Test atmospheric conditions in thermosphere (>71 km)."""
        conditions = AtmosphericConditions(altitude=80000.0, mach_number=60.0)
        
        assert conditions.temperature is not None
        assert conditions.pressure is not None
        assert conditions.density is not None
        
        # Check reasonable values for 80 km altitude
        assert 200 < conditions.temperature < 400  # K
        assert 0.1 < conditions.pressure < 10  # Pa
        assert 1e-6 < conditions.density < 1e-4  # kg/m³
    
    def test_mach_60_conditions(self):
        """Test conditions for Mach 60 flight."""
        conditions = AtmosphericConditions(altitude=50000.0, mach_number=60.0)
        
        # Calculate expected velocity
        expected_velocity = 60.0 * conditions.speed_of_sound
        actual_velocity = conditions.mach_number * conditions.speed_of_sound
        
        assert abs(expected_velocity - actual_velocity) < 1e-6
        
        # Dynamic pressure should be significant at Mach 60 (but lower at high altitude)
        assert conditions.dynamic_pressure > 1e5  # Pa (100 kPa)
    
    def test_altitude_range_30_80_km(self):
        """Test atmospheric conditions across 30-80 km altitude range."""
        altitudes = [30000, 40000, 50000, 60000, 70000, 80000]
        mach_number = 60.0
        
        temperatures = []
        pressures = []
        densities = []
        dynamic_pressures = []
        
        for alt in altitudes:
            conditions = AtmosphericConditions(altitude=alt, mach_number=mach_number)
            temperatures.append(conditions.temperature)
            pressures.append(conditions.pressure)
            densities.append(conditions.density)
            dynamic_pressures.append(conditions.dynamic_pressure)
        
        # Check that pressure and density generally decrease with altitude
        assert pressures[0] > pressures[-1]  # Pressure decreases
        assert densities[0] > densities[-1]  # Density decreases
        
        # All values should be positive and reasonable
        assert all(t > 0 for t in temperatures)
        assert all(p > 0 for p in pressures)
        assert all(rho > 0 for rho in densities)
        assert all(q > 0 for q in dynamic_pressures)


class TestAtmosphericLoadsAnalyzer:
    """Test cases for AtmosphericLoadsAnalyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = AtmosphericLoadsAnalyzer()
        
        # Test aircraft geometry
        self.aircraft_geometry = {
            'reference_area': 150.0,  # m²
            'reference_length': 25.0,  # m
            'wing_span': 18.0,  # m
            'load_distribution': {
                'fuselage': 0.45,
                'wings': 0.30,
                'control_surfaces': 0.15,
                'engine_mounts': 0.10
            }
        }
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = AtmosphericLoadsAnalyzer()
        assert analyzer.logger is not None
        assert analyzer.reference_area == 100.0  # Default value
        assert analyzer.reference_length == 20.0
        assert analyzer.wing_span == 15.0
        assert len(analyzer.load_distribution_factors) == 4
    
    def test_analyze_hypersonic_loads_mach_60(self):
        """Test hypersonic loads analysis for Mach 60."""
        altitude_range = (30000.0, 80000.0)
        mach_number = 60.0
        
        results = self.analyzer.analyze_hypersonic_loads(
            altitude_range, mach_number, self.aircraft_geometry
        )
        
        assert isinstance(results, HypersonicLoadResults)
        assert isinstance(results.atmospheric_conditions, AtmosphericConditions)
        assert isinstance(results.structural_loads, StructuralLoads)
        
        # Check that results are reasonable for Mach 60
        assert results.max_dynamic_pressure > 1e6  # Very high dynamic pressure
        assert results.max_load_factor >= 1.0
        assert len(results.critical_load_locations) >= 0
        assert len(results.structural_margins) > 0
        assert len(results.recommended_modifications) >= 0
        
        # Check atmospheric conditions
        assert 30000 <= results.atmospheric_conditions.altitude <= 80000
        assert results.atmospheric_conditions.mach_number == 60.0
    
    def test_analyze_hypersonic_loads_different_mach_numbers(self):
        """Test hypersonic loads analysis for different Mach numbers."""
        altitude_range = (40000.0, 60000.0)
        mach_numbers = [10.0, 25.0, 40.0, 60.0]
        
        max_pressures = []
        max_load_factors = []
        
        for mach in mach_numbers:
            results = self.analyzer.analyze_hypersonic_loads(
                altitude_range, mach, self.aircraft_geometry
            )
            max_pressures.append(results.max_dynamic_pressure)
            max_load_factors.append(results.max_load_factor)
        
        # Dynamic pressure should generally increase with Mach number
        assert max_pressures[1] > max_pressures[0]  # Mach 25 > Mach 10
        assert max_pressures[3] > max_pressures[2]  # Mach 60 > Mach 40
        
        # All results should be positive
        assert all(p > 0 for p in max_pressures)
        assert all(lf >= 1.0 for lf in max_load_factors)
    
    def test_calculate_dynamic_pressure_envelope(self):
        """Test dynamic pressure envelope calculation."""
        altitude_range = (30000.0, 80000.0)
        mach_range = (10.0, 60.0)
        n_points = 20
        
        envelope = self.analyzer.calculate_dynamic_pressure_envelope(
            altitude_range, mach_range, n_points
        )
        
        assert 'altitude' in envelope
        assert 'mach_number' in envelope
        assert 'dynamic_pressure' in envelope
        assert 'max_q' in envelope
        assert 'max_q_location' in envelope
        
        # Check grid dimensions
        assert envelope['altitude'].shape == (n_points, n_points)
        assert envelope['mach_number'].shape == (n_points, n_points)
        assert envelope['dynamic_pressure'].shape == (n_points, n_points)
        
        # Check that maximum dynamic pressure is positive
        assert envelope['max_q'] > 0
        
        # Check that max location is within bounds
        max_i, max_j = envelope['max_q_location']
        assert 0 <= max_i < n_points
        assert 0 <= max_j < n_points
    
    def test_analyze_safety_factors(self):
        """Test safety factor analysis."""
        # Create test structural loads
        n_components = 10
        structural_loads = StructuralLoads(
            dynamic_pressure=50000.0,
            normal_force=np.full(n_components, 1e6),  # 1 MN
            shear_force=np.full(n_components, 1e5),   # 100 kN
            bending_moment=np.full(n_components, 1e7), # 10 MN⋅m
            torsional_moment=np.full(n_components, 5e6), # 5 MN⋅m
            pressure_distribution=np.full(n_components, 50000.0),
            load_factor=2.0,
            safety_factor=1.5
        )
        
        # Material properties
        material_properties = {
            'fuselage': {'ultimate_strength': 600e6, 'yield_strength': 400e6},
            'wings': {'ultimate_strength': 500e6, 'yield_strength': 350e6},
            'control_surfaces': {'ultimate_strength': 450e6, 'yield_strength': 300e6},
            'engine_mounts': {'ultimate_strength': 700e6, 'yield_strength': 500e6}
        }
        
        safety_factors = self.analyzer.analyze_safety_factors(
            structural_loads, material_properties
        )
        
        assert len(safety_factors) == 4  # Four components
        assert all(sf > 0 for sf in safety_factors.values())
        
        # Engine mounts should have highest safety factor (strongest material)
        assert safety_factors['engine_mounts'] >= safety_factors['control_surfaces']
    
    def test_optimize_flight_profile(self):
        """Test flight profile optimization."""
        target_altitude = 50000.0  # 50 km
        max_dynamic_pressure = 100000.0  # 100 kPa
        mach_range = (20.0, 60.0)
        
        profile = self.analyzer.optimize_flight_profile(
            target_altitude, max_dynamic_pressure, mach_range
        )
        
        assert 'optimal_mach' in profile
        assert 'optimal_altitude' in profile
        assert 'dynamic_pressure' in profile
        assert 'atmospheric_density' in profile
        assert 'flight_efficiency' in profile
        assert 'velocity' in profile
        
        # Check that optimal Mach is within range
        assert mach_range[0] <= profile['optimal_mach'] <= mach_range[1]
        
        # Check that altitude matches target
        assert abs(profile['optimal_altitude'] - target_altitude) < 1e-6
        
        # Check that dynamic pressure is within limit
        assert profile['dynamic_pressure'] <= max_dynamic_pressure * 1.01  # Small tolerance
        
        # Check that velocity is consistent
        expected_velocity = profile['optimal_mach'] * np.sqrt(1.4 * 287.0 * 
                                                            AtmosphericConditions(target_altitude, profile['optimal_mach']).temperature)
        assert abs(profile['velocity'] - expected_velocity) < 100  # 100 m/s tolerance
    
    def test_load_cases(self):
        """Test different load cases."""
        altitude_range = (40000.0, 60000.0)
        mach_number = 30.0
        load_cases = ['steady_flight', 'maneuver_2g', 'maneuver_4g', 'gust_encounter']
        
        results = self.analyzer.analyze_hypersonic_loads(
            altitude_range, mach_number, self.aircraft_geometry, load_cases
        )
        
        # Should return results for the most critical load case
        assert results.max_load_factor >= 1.0
        
        # For maneuver cases, load factor should be higher
        if results.max_load_factor > 2.0:
            # Likely from maneuver_4g case
            assert results.max_load_factor <= 4.1  # Small tolerance
    
    def test_critical_altitude_finding(self):
        """Test finding critical altitude with maximum dynamic pressure."""
        altitude_range = (30000.0, 80000.0)
        mach_number = 40.0
        
        critical_alt = self.analyzer._find_critical_altitude(altitude_range, mach_number)
        
        assert altitude_range[0] <= critical_alt <= altitude_range[1]
        
        # Verify this is indeed the critical altitude
        conditions_critical = AtmosphericConditions(critical_alt, mach_number)
        
        # Check nearby altitudes have lower or equal dynamic pressure
        test_altitudes = [critical_alt - 5000, critical_alt + 5000]
        for test_alt in test_altitudes:
            if altitude_range[0] <= test_alt <= altitude_range[1]:
                test_conditions = AtmosphericConditions(test_alt, mach_number)
                assert test_conditions.dynamic_pressure <= conditions_critical.dynamic_pressure * 1.01
    
    def test_structural_load_calculation(self):
        """Test structural load calculation."""
        conditions = AtmosphericConditions(altitude=50000.0, mach_number=30.0)
        
        # Test different load cases
        load_cases = ['steady_flight', 'maneuver_2g', 'maneuver_4g', 'gust_encounter']
        
        for load_case in load_cases:
            loads = self.analyzer._calculate_structural_loads(conditions, load_case)
            
            assert isinstance(loads, StructuralLoads)
            assert loads.dynamic_pressure == conditions.dynamic_pressure
            assert len(loads.normal_force) > 0
            assert len(loads.shear_force) > 0
            assert len(loads.bending_moment) > 0
            assert len(loads.torsional_moment) > 0
            assert len(loads.pressure_distribution) > 0
            assert loads.load_factor >= 1.0
            assert loads.safety_factor > 0
            
            # Check load factor matches expected values
            if load_case == 'steady_flight':
                assert abs(loads.load_factor - 1.0) < 1e-6
            elif load_case == 'maneuver_2g':
                assert abs(loads.load_factor - 2.0) < 1e-6
            elif load_case == 'maneuver_4g':
                assert abs(loads.load_factor - 4.0) < 1e-6
            elif load_case == 'gust_encounter':
                assert abs(loads.load_factor - 2.5) < 1e-6
    
    def test_critical_load_location_identification(self):
        """Test identification of critical load locations."""
        # Create test loads with known maximum locations
        n_components = 10
        normal_force = np.random.rand(n_components) * 1e6
        normal_force[3] = 5e6  # Maximum at index 3
        
        bending_moment = np.random.rand(n_components) * 1e7
        bending_moment[7] = 8e7  # Maximum at index 7
        
        torsional_moment = np.random.rand(n_components) * 1e6
        torsional_moment[2] = 6e6  # Maximum at index 2
        
        loads = StructuralLoads(
            dynamic_pressure=50000.0,
            normal_force=normal_force,
            shear_force=np.random.rand(n_components) * 1e5,
            bending_moment=bending_moment,
            torsional_moment=torsional_moment,
            pressure_distribution=np.full(n_components, 50000.0),
            load_factor=2.0,
            safety_factor=1.5
        )
        
        critical_locations = self.analyzer._identify_critical_load_locations(loads)
        
        assert len(critical_locations) == 3  # Three types of critical loads
        
        # Check that correct locations are identified
        location_dict = {load_type: location for location, load_type in critical_locations}
        assert location_dict['normal_force'] == 3
        assert location_dict['bending_moment'] == 7
        assert location_dict['torsional_moment'] == 2
    
    def test_design_recommendations(self):
        """Test generation of design recommendations."""
        # Create loads with high dynamic pressure
        loads = StructuralLoads(
            dynamic_pressure=100000.0,  # High dynamic pressure
            normal_force=np.full(10, 1e6),
            shear_force=np.full(10, 1e5),
            bending_moment=np.full(10, 1e7),
            torsional_moment=np.full(10, 5e6),
            pressure_distribution=np.full(10, 100000.0),
            load_factor=4.0,  # High load factor
            safety_factor=1.2
        )
        
        # Create margins with some low values
        margins = {
            'fuselage': 0.3,  # Low margin
            'wings': 1.2,     # Good margin
            'control_surfaces': 0.8,  # Moderate margin
            'engine_mounts': 0.4      # Low margin
        }
        
        conditions = AtmosphericConditions(altitude=65000.0, mach_number=50.0)
        
        recommendations = self.analyzer._generate_design_recommendations(
            loads, margins, conditions
        )
        
        assert len(recommendations) > 0
        
        # Should recommend load alleviation for high dynamic pressure (100,000 Pa > 50,000 Pa threshold)
        load_alleviation_rec = any('load alleviation' in rec.lower() for rec in recommendations)
        # Print recommendations for debugging
        if not load_alleviation_rec:
            print(f"Recommendations: {recommendations}")
            print(f"Dynamic pressure: {loads.dynamic_pressure}")
        assert load_alleviation_rec
        
        # Should recommend structural improvements for low margins
        fuselage_rec = any('fuselage' in rec.lower() and 'strength' in rec.lower() for rec in recommendations)
        engine_mount_rec = any('engine_mounts' in rec.lower() and 'strength' in rec.lower() for rec in recommendations)
        assert fuselage_rec or engine_mount_rec
        
        # Should recommend load limiting for high load factor
        load_limiting_rec = any('load limiting' in rec.lower() for rec in recommendations)
        assert load_limiting_rec
        
        # Should recommend thermal protection for high altitude
        thermal_rec = any('thermal protection' in rec.lower() for rec in recommendations)
        assert thermal_rec
        
        # Should recommend plasma considerations for high Mach
        plasma_rec = any('plasma' in rec.lower() for rec in recommendations)
        assert plasma_rec
    
    def test_geometry_update(self):
        """Test aircraft geometry update."""
        new_geometry = {
            'reference_area': 200.0,
            'reference_length': 30.0,
            'wing_span': 25.0,
            'load_distribution': {
                'fuselage': 0.5,
                'wings': 0.3,
                'control_surfaces': 0.1,
                'engine_mounts': 0.1
            }
        }
        
        self.analyzer._update_geometry(new_geometry)
        
        assert self.analyzer.reference_area == 200.0
        assert self.analyzer.reference_length == 30.0
        assert self.analyzer.wing_span == 25.0
        assert self.analyzer.load_distribution_factors['fuselage'] == 0.5
    
    def test_extreme_conditions_mach_60_80km(self):
        """Test analysis under extreme conditions: Mach 60 at 80 km."""
        altitude_range = (75000.0, 80000.0)
        mach_number = 60.0
        
        results = self.analyzer.analyze_hypersonic_loads(
            altitude_range, mach_number, self.aircraft_geometry
        )
        
        assert isinstance(results, HypersonicLoadResults)
        
        # At 80 km, atmospheric density is very low
        assert results.atmospheric_conditions.density < 1e-4  # kg/m³
        
        # But dynamic pressure can still be significant due to very high velocity
        velocity = mach_number * results.atmospheric_conditions.speed_of_sound
        assert velocity > 15000  # m/s (over 15 km/s)
        
        # Should have recommendations for extreme conditions
        assert len(results.recommended_modifications) > 0
        
        # Should recommend plasma effects consideration
        plasma_rec = any('plasma' in rec.lower() for rec in results.recommended_modifications)
        assert plasma_rec
    
    @patch('fighter_jet_sdk.engines.structural.atmospheric_loads_analyzer.get_engine_logger')
    def test_logging(self, mock_logger):
        """Test logging functionality."""
        mock_log = Mock()
        mock_logger.return_value = mock_log
        
        analyzer = AtmosphericLoadsAnalyzer()
        
        # Test successful analysis logging
        results = analyzer.analyze_hypersonic_loads(
            (40000.0, 60000.0), 30.0, self.aircraft_geometry
        )
        
        # Check that info logs were called
        mock_log.info.assert_called()
        
        # Check that the completion log includes max dynamic pressure
        completion_calls = [call for call in mock_log.info.call_args_list 
                          if 'complete' in str(call)]
        assert len(completion_calls) > 0