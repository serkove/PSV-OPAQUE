"""Quality control and inspection systems module."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
import statistics

from ...common.data_models import MaterialDefinition, Module
from ...core.logging import get_engine_logger


class InspectionType(Enum):
    """Types of inspection methods."""
    VISUAL = "visual"
    DIMENSIONAL = "dimensional"
    NON_DESTRUCTIVE_TESTING = "non_destructive_testing"
    ELECTRICAL = "electrical"
    MECHANICAL = "mechanical"
    STEALTH_COATING = "stealth_coating"
    MATERIAL_PROPERTIES = "material_properties"
    FUNCTIONAL = "functional"


class NDTMethod(Enum):
    """Non-destructive testing methods."""
    ULTRASONIC = "ultrasonic"
    RADIOGRAPHIC = "radiographic"
    MAGNETIC_PARTICLE = "magnetic_particle"
    DYE_PENETRANT = "dye_penetrant"
    EDDY_CURRENT = "eddy_current"
    THERMOGRAPHIC = "thermographic"
    LASER_SHEAROGRAPHY = "laser_shearography"


class StealthCoatingTest(Enum):
    """Stealth coating inspection methods."""
    RADAR_CROSS_SECTION = "radar_cross_section"
    SURFACE_ROUGHNESS = "surface_roughness"
    COATING_THICKNESS = "coating_thickness"
    ADHESION_TEST = "adhesion_test"
    ELECTROMAGNETIC_PROPERTIES = "electromagnetic_properties"
    WEATHERING_RESISTANCE = "weathering_resistance"


class InspectionResult(Enum):
    """Inspection result status."""
    PASS = "pass"
    FAIL = "fail"
    CONDITIONAL_PASS = "conditional_pass"
    RETEST_REQUIRED = "retest_required"


@dataclass
class InspectionCriteria:
    """Inspection acceptance criteria."""
    criteria_id: str
    parameter_name: str
    nominal_value: float
    tolerance_upper: float
    tolerance_lower: float
    units: str
    critical: bool = False  # Critical dimension/parameter
    measurement_method: str = ""
    sampling_plan: str = "100%"  # Percentage or specific plan


@dataclass
class InspectionProcedure:
    """Detailed inspection procedure."""
    procedure_id: str
    procedure_name: str
    inspection_type: InspectionType
    ndt_method: Optional[NDTMethod] = None
    stealth_test: Optional[StealthCoatingTest] = None
    equipment_required: List[str] = field(default_factory=list)
    setup_time: float = 0.0  # minutes
    inspection_time: float = 0.0  # minutes per unit
    skill_level_required: int = 1  # 1=basic, 5=expert
    environmental_requirements: Dict[str, Any] = field(default_factory=dict)
    safety_requirements: List[str] = field(default_factory=list)
    acceptance_criteria: List[InspectionCriteria] = field(default_factory=list)
    documentation_requirements: List[str] = field(default_factory=list)


@dataclass
class InspectionRecord:
    """Individual inspection record."""
    record_id: str
    procedure_id: str
    part_id: str
    inspector_id: str
    inspection_date: datetime
    measurements: Dict[str, float] = field(default_factory=dict)
    result: InspectionResult = InspectionResult.PASS
    notes: str = ""
    defects_found: List[str] = field(default_factory=list)
    corrective_actions: List[str] = field(default_factory=list)
    reinspection_required: bool = False


@dataclass
class StatisticalProcessControl:
    """Statistical process control data."""
    parameter_name: str
    measurements: List[float] = field(default_factory=list)
    control_limits_upper: float = 0.0
    control_limits_lower: float = 0.0
    mean: float = 0.0
    std_dev: float = 0.0
    cpk: float = 0.0  # Process capability index
    out_of_control_points: List[int] = field(default_factory=list)
    trend_analysis: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityTrend:
    """Quality trend analysis result."""
    parameter_name: str
    time_period: Tuple[datetime, datetime]
    trend_direction: str  # "improving", "degrading", "stable"
    trend_magnitude: float
    statistical_significance: float
    recommended_actions: List[str] = field(default_factory=list)


@dataclass
class DefectAnalysis:
    """Defect analysis and categorization."""
    defect_type: str
    frequency: int
    severity: str  # "critical", "major", "minor"
    root_causes: List[str] = field(default_factory=list)
    cost_impact: float = 0.0  # USD
    prevention_measures: List[str] = field(default_factory=list)


class QualityController:
    """Quality control and inspection system for advanced materials and stealth coatings."""
    
    def __init__(self):
        """Initialize quality control system."""
        self.logger = get_engine_logger('quality_controller')
        self.inspection_procedures = {}
        self.inspection_records = []
        self.spc_data = {}
        self.equipment_database = {}
        self._initialize_standard_procedures()
        self._initialize_equipment_database()
    
    def _initialize_standard_procedures(self):
        """Initialize standard inspection procedures."""
        # Stealth coating inspection procedures
        self.inspection_procedures['stealth_rcs_measurement'] = InspectionProcedure(
            procedure_id='stealth_rcs_measurement',
            procedure_name='Radar Cross Section Measurement',
            inspection_type=InspectionType.STEALTH_COATING,
            stealth_test=StealthCoatingTest.RADAR_CROSS_SECTION,
            equipment_required=['anechoic_chamber', 'network_analyzer', 'horn_antennas'],
            setup_time=120.0,  # 2 hours
            inspection_time=60.0,  # 1 hour per measurement
            skill_level_required=4,
            environmental_requirements={
                'temperature': (20.0, 25.0),  # °C
                'humidity': (40.0, 60.0),  # %RH
                'electromagnetic_isolation': True
            },
            safety_requirements=['RF_safety_training', 'PPE_required'],
            acceptance_criteria=[
                InspectionCriteria(
                    criteria_id='rcs_x_band',
                    parameter_name='RCS_X_band',
                    nominal_value=-30.0,  # dBsm
                    tolerance_upper=-25.0,
                    tolerance_lower=-40.0,
                    units='dBsm',
                    critical=True,
                    measurement_method='Monostatic RCS at 10 GHz'
                )
            ],
            documentation_requirements=['measurement_report', 'calibration_certificate']
        )
        
        # Coating thickness inspection
        self.inspection_procedures['coating_thickness'] = InspectionProcedure(
            procedure_id='coating_thickness',
            procedure_name='Stealth Coating Thickness Measurement',
            inspection_type=InspectionType.STEALTH_COATING,
            stealth_test=StealthCoatingTest.COATING_THICKNESS,
            equipment_required=['ultrasonic_thickness_gauge', 'eddy_current_probe'],
            setup_time=15.0,
            inspection_time=5.0,  # 5 minutes per measurement point
            skill_level_required=2,
            environmental_requirements={
                'temperature': (15.0, 30.0),
                'surface_cleanliness': 'Class_A'
            },
            safety_requirements=['PPE_required'],
            acceptance_criteria=[
                InspectionCriteria(
                    criteria_id='coating_thickness_nominal',
                    parameter_name='coating_thickness',
                    nominal_value=0.5,  # mm
                    tolerance_upper=0.6,
                    tolerance_lower=0.4,
                    units='mm',
                    critical=True,
                    measurement_method='Ultrasonic thickness measurement',
                    sampling_plan='Grid pattern 100mm spacing'
                )
            ]
        )
        
        # Composite material NDT
        self.inspection_procedures['composite_ultrasonic'] = InspectionProcedure(
            procedure_id='composite_ultrasonic',
            procedure_name='Composite Ultrasonic Inspection',
            inspection_type=InspectionType.NON_DESTRUCTIVE_TESTING,
            ndt_method=NDTMethod.ULTRASONIC,
            equipment_required=['ultrasonic_flaw_detector', 'immersion_tank', 'transducers'],
            setup_time=45.0,
            inspection_time=30.0,  # 30 minutes per square meter
            skill_level_required=3,
            environmental_requirements={
                'temperature': (18.0, 25.0),
                'water_quality': 'deionized'
            },
            safety_requirements=['hearing_protection', 'electrical_safety'],
            acceptance_criteria=[
                InspectionCriteria(
                    criteria_id='void_content',
                    parameter_name='void_content',
                    nominal_value=0.0,
                    tolerance_upper=2.0,  # Maximum 2% void content
                    tolerance_lower=0.0,
                    units='%',
                    critical=True,
                    measurement_method='C-scan ultrasonic'
                ),
                InspectionCriteria(
                    criteria_id='delamination_size',
                    parameter_name='delamination_diameter',
                    nominal_value=0.0,
                    tolerance_upper=6.0,  # Maximum 6mm diameter
                    tolerance_lower=0.0,
                    units='mm',
                    critical=True,
                    measurement_method='Pulse-echo ultrasonic'
                )
            ]
        )
        
        # Dimensional inspection
        self.inspection_procedures['precision_dimensional'] = InspectionProcedure(
            procedure_id='precision_dimensional',
            procedure_name='Precision Dimensional Inspection',
            inspection_type=InspectionType.DIMENSIONAL,
            equipment_required=['cmm_machine', 'laser_tracker', 'surface_plates'],
            setup_time=30.0,
            inspection_time=45.0,  # 45 minutes per part
            skill_level_required=3,
            environmental_requirements={
                'temperature': (20.0, 20.0),  # ±0.1°C
                'vibration_isolation': True
            },
            safety_requirements=['safety_glasses'],
            acceptance_criteria=[
                InspectionCriteria(
                    criteria_id='geometric_tolerance',
                    parameter_name='geometric_tolerance',
                    nominal_value=0.0,
                    tolerance_upper=0.1,  # ±0.1mm
                    tolerance_lower=-0.1,
                    units='mm',
                    critical=True,
                    measurement_method='CMM measurement per GD&T'
                )
            ]
        )
    
    def _initialize_equipment_database(self):
        """Initialize inspection equipment database."""
        self.equipment_database = {
            'anechoic_chamber': {
                'type': 'RF_measurement',
                'frequency_range': (1e9, 40e9),  # 1-40 GHz
                'accuracy': '±0.5 dB',
                'calibration_interval': 365,  # days
                'cost_per_hour': 500.0
            },
            'network_analyzer': {
                'type': 'RF_measurement',
                'frequency_range': (10e6, 67e9),  # 10 MHz - 67 GHz
                'accuracy': '±0.1 dB',
                'calibration_interval': 365,
                'cost_per_hour': 150.0
            },
            'ultrasonic_thickness_gauge': {
                'type': 'thickness_measurement',
                'range': (0.1, 500.0),  # mm
                'accuracy': '±0.01 mm',
                'calibration_interval': 180,
                'cost_per_hour': 25.0
            },
            'cmm_machine': {
                'type': 'dimensional_measurement',
                'accuracy': '±0.002 mm',
                'measurement_volume': (1000, 1000, 600),  # mm
                'calibration_interval': 365,
                'cost_per_hour': 75.0
            },
            'ultrasonic_flaw_detector': {
                'type': 'ndt_equipment',
                'frequency_range': (0.5e6, 20e6),  # 0.5-20 MHz
                'sensitivity': '-80 dB',
                'calibration_interval': 180,
                'cost_per_hour': 50.0
            }
        }
    
    def generate_inspection_protocols(
        self,
        material: MaterialDefinition,
        part_geometry: Dict[str, Any],
        manufacturing_process: str,
        criticality_level: int = 3
    ) -> List[InspectionProcedure]:
        """Generate inspection protocols for stealth coatings and metamaterials."""
        self.logger.info(f"Generating inspection protocols for material {material.name}")
        
        protocols = []
        
        # Determine required inspections based on material type
        if material.base_material_type.name in ['METAMATERIAL', 'STEALTH_COATING']:
            protocols.extend(self._generate_stealth_protocols(material, part_geometry, criticality_level))
        
        if material.base_material_type.name == 'COMPOSITE':
            protocols.extend(self._generate_composite_protocols(material, part_geometry, criticality_level))
        
        # Always include dimensional inspection for critical parts
        if criticality_level >= 3:
            protocols.append(self.inspection_procedures['precision_dimensional'])
        
        # Add process-specific inspections
        if manufacturing_process.lower() in ['autoclave', 'out_of_autoclave']:
            protocols.extend(self._generate_cure_quality_protocols(material, manufacturing_process))
        
        return protocols
    
    def _generate_stealth_protocols(
        self,
        material: MaterialDefinition,
        part_geometry: Dict[str, Any],
        criticality_level: int
    ) -> List[InspectionProcedure]:
        """Generate stealth coating specific inspection protocols."""
        protocols = []
        
        # RCS measurement for all stealth materials
        protocols.append(self.inspection_procedures['stealth_rcs_measurement'])
        
        # Coating thickness measurement
        protocols.append(self.inspection_procedures['coating_thickness'])
        
        # Surface roughness for stealth coatings
        if criticality_level >= 4:
            surface_roughness_procedure = InspectionProcedure(
                procedure_id='surface_roughness_stealth',
                procedure_name='Stealth Surface Roughness Measurement',
                inspection_type=InspectionType.STEALTH_COATING,
                stealth_test=StealthCoatingTest.SURFACE_ROUGHNESS,
                equipment_required=['profilometer', 'optical_scanner'],
                setup_time=20.0,
                inspection_time=15.0,
                skill_level_required=2,
                acceptance_criteria=[
                    InspectionCriteria(
                        criteria_id='surface_roughness_ra',
                        parameter_name='surface_roughness_Ra',
                        nominal_value=0.8,  # μm
                        tolerance_upper=1.6,
                        tolerance_lower=0.0,
                        units='μm',
                        critical=True,
                        measurement_method='Stylus profilometry'
                    )
                ]
            )
            protocols.append(surface_roughness_procedure)
        
        # Electromagnetic properties verification
        if material.electromagnetic_properties:
            em_properties_procedure = InspectionProcedure(
                procedure_id='em_properties_verification',
                procedure_name='Electromagnetic Properties Verification',
                inspection_type=InspectionType.STEALTH_COATING,
                stealth_test=StealthCoatingTest.ELECTROMAGNETIC_PROPERTIES,
                equipment_required=['vector_network_analyzer', 'waveguide_fixtures'],
                setup_time=60.0,
                inspection_time=90.0,
                skill_level_required=4,
                acceptance_criteria=[
                    InspectionCriteria(
                        criteria_id='permittivity_real',
                        parameter_name='permittivity_real',
                        nominal_value=material.electromagnetic_properties.permittivity.real,
                        tolerance_upper=material.electromagnetic_properties.permittivity.real * 1.1,
                        tolerance_lower=material.electromagnetic_properties.permittivity.real * 0.9,
                        units='',
                        critical=True,
                        measurement_method='Waveguide transmission method'
                    )
                ]
            )
            protocols.append(em_properties_procedure)
        
        return protocols
    
    def _generate_composite_protocols(
        self,
        material: MaterialDefinition,
        part_geometry: Dict[str, Any],
        criticality_level: int
    ) -> List[InspectionProcedure]:
        """Generate composite material inspection protocols."""
        protocols = []
        
        # Ultrasonic inspection for all composite parts
        protocols.append(self.inspection_procedures['composite_ultrasonic'])
        
        # Thermographic inspection for large parts
        part_area = part_geometry.get('length', 1.0) * part_geometry.get('width', 1.0)
        if part_area > 1.0:  # > 1 m²
            thermographic_procedure = InspectionProcedure(
                procedure_id='thermographic_inspection',
                procedure_name='Thermographic Inspection',
                inspection_type=InspectionType.NON_DESTRUCTIVE_TESTING,
                ndt_method=NDTMethod.THERMOGRAPHIC,
                equipment_required=['thermal_camera', 'heat_source'],
                setup_time=30.0,
                inspection_time=20.0,
                skill_level_required=3,
                acceptance_criteria=[
                    InspectionCriteria(
                        criteria_id='thermal_anomaly',
                        parameter_name='temperature_difference',
                        nominal_value=0.0,
                        tolerance_upper=2.0,  # Maximum 2°C difference
                        tolerance_lower=-2.0,
                        units='°C',
                        critical=False,
                        measurement_method='Active thermography'
                    )
                ]
            )
            protocols.append(thermographic_procedure)
        
        return protocols
    
    def _generate_cure_quality_protocols(
        self,
        material: MaterialDefinition,
        manufacturing_process: str
    ) -> List[InspectionProcedure]:
        """Generate cure quality inspection protocols."""
        protocols = []
        
        # Degree of cure measurement
        cure_analysis_procedure = InspectionProcedure(
            procedure_id='cure_analysis',
            procedure_name='Degree of Cure Analysis',
            inspection_type=InspectionType.MATERIAL_PROPERTIES,
            equipment_required=['dsc_analyzer', 'sample_preparation_kit'],
            setup_time=45.0,
            inspection_time=120.0,  # 2 hours per sample
            skill_level_required=4,
            acceptance_criteria=[
                InspectionCriteria(
                    criteria_id='degree_of_cure',
                    parameter_name='degree_of_cure',
                    nominal_value=95.0,  # 95% cure
                    tolerance_upper=100.0,
                    tolerance_lower=90.0,
                    units='%',
                    critical=True,
                    measurement_method='Differential Scanning Calorimetry'
                )
            ]
        )
        protocols.append(cure_analysis_procedure)
        
        return protocols
    
    def perform_inspection(
        self,
        procedure: InspectionProcedure,
        part_id: str,
        inspector_id: str,
        measurements: Dict[str, float]
    ) -> InspectionRecord:
        """Perform inspection and record results."""
        self.logger.info(f"Performing inspection {procedure.procedure_name} on part {part_id}")
        
        # Create inspection record
        record = InspectionRecord(
            record_id=f"INS_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{part_id}",
            procedure_id=procedure.procedure_id,
            part_id=part_id,
            inspector_id=inspector_id,
            inspection_date=datetime.now(),
            measurements=measurements
        )
        
        # Evaluate against acceptance criteria
        defects = []
        overall_result = InspectionResult.PASS
        
        for criteria in procedure.acceptance_criteria:
            if criteria.parameter_name in measurements:
                measured_value = measurements[criteria.parameter_name]
                
                # Check if measurement is within tolerance
                if not (criteria.tolerance_lower <= measured_value <= criteria.tolerance_upper):
                    defect_description = (
                        f"{criteria.parameter_name}: {measured_value} {criteria.units} "
                        f"(spec: {criteria.tolerance_lower} to {criteria.tolerance_upper} {criteria.units})"
                    )
                    defects.append(defect_description)
                    
                    if criteria.critical:
                        overall_result = InspectionResult.FAIL
                    elif overall_result == InspectionResult.PASS:
                        overall_result = InspectionResult.CONDITIONAL_PASS
        
        record.defects_found = defects
        record.result = overall_result
        
        # Add to inspection records
        self.inspection_records.append(record)
        
        # Update SPC data
        self._update_spc_data(procedure.procedure_id, measurements)
        
        return record
    
    def _update_spc_data(self, procedure_id: str, measurements: Dict[str, float]):
        """Update statistical process control data."""
        if procedure_id not in self.spc_data:
            self.spc_data[procedure_id] = {}
        
        for parameter, value in measurements.items():
            if parameter not in self.spc_data[procedure_id]:
                self.spc_data[procedure_id][parameter] = StatisticalProcessControl(
                    parameter_name=parameter
                )
            
            spc = self.spc_data[procedure_id][parameter]
            spc.measurements.append(value)
            
            # Update statistics if we have enough data points
            if len(spc.measurements) >= 10:
                spc.mean = statistics.mean(spc.measurements)
                spc.std_dev = statistics.stdev(spc.measurements)
                
                # Calculate control limits (3-sigma)
                spc.control_limits_upper = spc.mean + 3 * spc.std_dev
                spc.control_limits_lower = spc.mean - 3 * spc.std_dev
                
                # Check for out-of-control points
                spc.out_of_control_points = []
                for i, measurement in enumerate(spc.measurements):
                    if measurement > spc.control_limits_upper or measurement < spc.control_limits_lower:
                        spc.out_of_control_points.append(i)
    
    def analyze_quality_trends(
        self,
        parameter_name: str,
        time_period_days: int = 30
    ) -> QualityTrend:
        """Analyze quality trends for a specific parameter."""
        self.logger.info(f"Analyzing quality trends for {parameter_name}")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_period_days)
        
        # Collect measurements within time period
        measurements = []
        dates = []
        
        for record in self.inspection_records:
            if (start_date <= record.inspection_date <= end_date and 
                parameter_name in record.measurements):
                measurements.append(record.measurements[parameter_name])
                dates.append(record.inspection_date)
        
        if len(measurements) < 5:
            return QualityTrend(
                parameter_name=parameter_name,
                time_period=(start_date, end_date),
                trend_direction="insufficient_data",
                trend_magnitude=0.0,
                statistical_significance=0.0
            )
        
        # Calculate trend using linear regression
        x_values = [(date - start_date).total_seconds() for date in dates]
        slope, intercept = np.polyfit(x_values, measurements, 1)
        
        # Determine trend direction and magnitude
        if abs(slope) < 0.001:  # Threshold for "stable"
            trend_direction = "stable"
        elif slope > 0:
            trend_direction = "degrading" if parameter_name.lower() in ['defect', 'error', 'deviation'] else "improving"
        else:
            trend_direction = "improving" if parameter_name.lower() in ['defect', 'error', 'deviation'] else "degrading"
        
        # Calculate statistical significance (simplified)
        correlation = np.corrcoef(x_values, measurements)[0, 1]
        statistical_significance = abs(correlation)
        
        # Generate recommendations
        recommendations = []
        if trend_direction == "degrading" and statistical_significance > 0.7:
            recommendations.append("Investigate root cause of degrading trend")
            recommendations.append("Review process parameters and controls")
            recommendations.append("Increase inspection frequency")
        elif trend_direction == "stable" and statistical_significance < 0.3:
            recommendations.append("Process appears stable - maintain current controls")
        
        return QualityTrend(
            parameter_name=parameter_name,
            time_period=(start_date, end_date),
            trend_direction=trend_direction,
            trend_magnitude=abs(slope),
            statistical_significance=statistical_significance,
            recommended_actions=recommendations
        )
    
    def analyze_defects(self, time_period_days: int = 30) -> List[DefectAnalysis]:
        """Analyze defect patterns and root causes."""
        self.logger.info("Analyzing defect patterns")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_period_days)
        
        # Collect defects within time period
        defect_counts = {}
        
        for record in self.inspection_records:
            if start_date <= record.inspection_date <= end_date:
                for defect in record.defects_found:
                    # Extract defect type (parameter name)
                    defect_type = defect.split(':')[0] if ':' in defect else defect
                    defect_counts[defect_type] = defect_counts.get(defect_type, 0) + 1
        
        # Create defect analysis
        defect_analyses = []
        
        for defect_type, frequency in defect_counts.items():
            # Determine severity based on frequency and type
            if frequency > 10:
                severity = "critical"
            elif frequency > 5:
                severity = "major"
            else:
                severity = "minor"
            
            # Generate root causes and prevention measures (simplified)
            root_causes = self._identify_root_causes(defect_type)
            prevention_measures = self._generate_prevention_measures(defect_type)
            
            # Estimate cost impact
            cost_impact = self._estimate_defect_cost(defect_type, frequency, severity)
            
            defect_analyses.append(DefectAnalysis(
                defect_type=defect_type,
                frequency=frequency,
                severity=severity,
                root_causes=root_causes,
                cost_impact=cost_impact,
                prevention_measures=prevention_measures
            ))
        
        # Sort by frequency (most common first)
        defect_analyses.sort(key=lambda x: x.frequency, reverse=True)
        
        return defect_analyses
    
    def _identify_root_causes(self, defect_type: str) -> List[str]:
        """Identify potential root causes for defect types."""
        root_cause_map = {
            'coating_thickness': [
                'Spray gun distance variation',
                'Material viscosity variation',
                'Environmental conditions',
                'Operator technique variation'
            ],
            'surface_roughness_Ra': [
                'Substrate preparation inadequate',
                'Contamination during application',
                'Curing temperature variation',
                'Material aging'
            ],
            'void_content': [
                'Inadequate vacuum during layup',
                'Resin flow issues',
                'Contamination in prepreg',
                'Cure cycle deviation'
            ],
            'geometric_tolerance': [
                'Tooling dimensional drift',
                'Thermal expansion effects',
                'Fixture wear',
                'Measurement system variation'
            ]
        }
        
        return root_cause_map.get(defect_type, ['Process variation', 'Material variation', 'Environmental factors'])
    
    def _generate_prevention_measures(self, defect_type: str) -> List[str]:
        """Generate prevention measures for defect types."""
        prevention_map = {
            'coating_thickness': [
                'Implement spray gun distance controls',
                'Monitor material viscosity',
                'Control environmental conditions',
                'Provide operator training'
            ],
            'surface_roughness_Ra': [
                'Improve substrate preparation procedures',
                'Implement contamination controls',
                'Tighten curing temperature control',
                'Implement material shelf life controls'
            ],
            'void_content': [
                'Improve vacuum bagging procedures',
                'Optimize resin flow parameters',
                'Implement prepreg quality controls',
                'Validate cure cycle parameters'
            ],
            'geometric_tolerance': [
                'Implement tooling maintenance program',
                'Control environmental temperature',
                'Implement fixture inspection program',
                'Calibrate measurement systems'
            ]
        }
        
        return prevention_map.get(defect_type, ['Implement process controls', 'Increase inspection frequency', 'Provide training'])
    
    def _estimate_defect_cost(self, defect_type: str, frequency: int, severity: str) -> float:
        """Estimate cost impact of defects."""
        # Base cost per defect by severity
        base_costs = {
            'critical': 5000.0,  # $5000 per critical defect
            'major': 1000.0,    # $1000 per major defect
            'minor': 200.0      # $200 per minor defect
        }
        
        # Defect-specific multipliers
        defect_multipliers = {
            'coating_thickness': 1.5,  # Expensive to rework
            'surface_roughness_Ra': 1.2,
            'void_content': 2.0,  # May require part replacement
            'geometric_tolerance': 1.8
        }
        
        base_cost = base_costs.get(severity, 500.0)
        multiplier = defect_multipliers.get(defect_type, 1.0)
        
        return base_cost * multiplier * frequency
    
    def generate_inspection_report(
        self,
        part_id: str,
        time_period_days: int = 7
    ) -> Dict[str, Any]:
        """Generate comprehensive inspection report for a part."""
        self.logger.info(f"Generating inspection report for part {part_id}")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_period_days)
        
        # Collect inspection records for the part
        part_records = [
            record for record in self.inspection_records
            if record.part_id == part_id and start_date <= record.inspection_date <= end_date
        ]
        
        if not part_records:
            return {
                'part_id': part_id,
                'status': 'no_inspections_found',
                'time_period': (start_date, end_date)
            }
        
        # Calculate summary statistics
        total_inspections = len(part_records)
        passed_inspections = len([r for r in part_records if r.result == InspectionResult.PASS])
        failed_inspections = len([r for r in part_records if r.result == InspectionResult.FAIL])
        conditional_passes = len([r for r in part_records if r.result == InspectionResult.CONDITIONAL_PASS])
        
        pass_rate = (passed_inspections / total_inspections) * 100.0 if total_inspections > 0 else 0.0
        
        # Collect all defects
        all_defects = []
        for record in part_records:
            all_defects.extend(record.defects_found)
        
        # Generate report
        report = {
            'part_id': part_id,
            'report_date': datetime.now(),
            'time_period': (start_date, end_date),
            'summary': {
                'total_inspections': total_inspections,
                'passed_inspections': passed_inspections,
                'failed_inspections': failed_inspections,
                'conditional_passes': conditional_passes,
                'pass_rate_percent': pass_rate
            },
            'defects': {
                'total_defects': len(all_defects),
                'defect_list': all_defects,
                'unique_defect_types': list(set([d.split(':')[0] for d in all_defects if ':' in d]))
            },
            'inspection_records': [
                {
                    'record_id': record.record_id,
                    'procedure': record.procedure_id,
                    'date': record.inspection_date,
                    'result': record.result.value,
                    'defects': record.defects_found
                }
                for record in part_records
            ],
            'recommendations': self._generate_part_recommendations(part_records)
        }
        
        return report
    
    def _generate_part_recommendations(self, records: List[InspectionRecord]) -> List[str]:
        """Generate recommendations based on inspection history."""
        recommendations = []
        
        # Check for recurring defects
        all_defects = []
        for record in records:
            all_defects.extend(record.defects_found)
        
        defect_counts = {}
        for defect in all_defects:
            defect_type = defect.split(':')[0] if ':' in defect else defect
            defect_counts[defect_type] = defect_counts.get(defect_type, 0) + 1
        
        # Generate recommendations based on patterns
        for defect_type, count in defect_counts.items():
            if count > 2:
                recommendations.append(f"Investigate recurring {defect_type} issues - occurred {count} times")
        
        # Check pass rate
        pass_rate = len([r for r in records if r.result == InspectionResult.PASS]) / len(records) * 100.0
        if pass_rate < 80.0:
            recommendations.append(f"Low pass rate ({pass_rate:.1f}%) - review manufacturing process")
        
        if not recommendations:
            recommendations.append("Part quality appears acceptable - continue current inspection schedule")
        
        return recommendations