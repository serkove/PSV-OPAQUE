"""Enumeration types for the Fighter Jet SDK."""

from enum import Enum, auto


class ModuleType(Enum):
    """Types of aircraft modules."""
    COCKPIT = auto()
    SENSOR = auto()
    PAYLOAD = auto()
    PROPULSION = auto()
    AVIONICS = auto()
    DEFENSIVE = auto()
    STRUCTURAL = auto()


class MaterialType(Enum):
    """Types of materials supported by the system."""
    METAMATERIAL = auto()
    CONDUCTIVE_POLYMER = auto()
    ULTRA_HIGH_TEMP_CERAMIC = auto()
    COMPOSITE = auto()
    STEALTH_COATING = auto()
    CONVENTIONAL_METAL = auto()


class SensorType(Enum):
    """Types of sensor systems."""
    AESA_RADAR = auto()
    EO_IR_TRACKING = auto()
    LASER_BASED = auto()
    PLASMA_BASED = auto()
    RADIATION_DETECTION = auto()
    PASSIVE_RF = auto()


class EngineType(Enum):
    """Types of propulsion systems."""
    TURBOFAN = auto()
    TURBOJET = auto()
    RAMJET = auto()
    SCRAMJET = auto()
    HYBRID = auto()


class ExtremePropulsionType(Enum):
    """Extreme hypersonic propulsion types for Mach 60+ flight."""
    DUAL_MODE_SCRAMJET = auto()
    COMBINED_CYCLE_AIRBREATHING = auto()
    ROCKET_ASSISTED_SCRAMJET = auto()
    MAGNETOPLASMADYNAMIC = auto()
    NUCLEAR_THERMAL = auto()


class FlightRegime(Enum):
    """Flight speed regimes."""
    SUBSONIC = auto()
    TRANSONIC = auto()
    SUPERSONIC = auto()
    HYPERSONIC = auto()
    EXTREME_HYPERSONIC = auto()  # Mach 25+


class ManufacturingProcess(Enum):
    """Manufacturing process types."""
    AUTOCLAVE = auto()
    OUT_OF_AUTOCLAVE = auto()
    FIBER_PLACEMENT = auto()
    RESIN_TRANSFER_MOLDING = auto()
    ADDITIVE_MANUFACTURING = auto()
    CONVENTIONAL_MACHINING = auto()


class PlasmaRegime(Enum):
    """Plasma ionization regimes for extreme hypersonic flight."""
    WEAKLY_IONIZED = auto()
    PARTIALLY_IONIZED = auto()
    FULLY_IONIZED = auto()
    MAGNETIZED_PLASMA = auto()


class ThermalProtectionType(Enum):
    """Thermal protection system types for extreme conditions."""
    PASSIVE_ABLATIVE = auto()
    ACTIVE_TRANSPIRATION = auto()
    REGENERATIVE_COOLING = auto()
    RADIATIVE_COOLING = auto()
    HYBRID_SYSTEM = auto()