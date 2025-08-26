"""Error handling utilities for the Fighter Jet SDK."""

from typing import Any, Dict, List, Optional, Union
import traceback
from datetime import datetime


class SDKError(Exception):
    """Base exception class for Fighter Jet SDK errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None, 
                 cause: Optional[Exception] = None):
        """Initialize SDK error.
        
        Args:
            message: Error message.
            error_code: Unique error code for categorization.
            context: Additional context information.
            cause: Original exception that caused this error.
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.cause = cause
        self.timestamp = datetime.now()
        
        # Add traceback information
        self.traceback_info = traceback.format_exc() if cause else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation."""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'context': self.context,
            'timestamp': self.timestamp.isoformat(),
            'traceback': self.traceback_info,
            'cause': str(self.cause) if self.cause else None
        }
    
    def __str__(self) -> str:
        """String representation of error."""
        base_msg = f"[{self.error_code}] {self.message}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            base_msg += f" (Context: {context_str})"
        return base_msg


class ValidationError(SDKError):
    """Error raised when validation fails."""
    
    def __init__(self, message: str, validation_errors: Optional[List[str]] = None,
                 field_name: Optional[str] = None, **kwargs):
        """Initialize validation error.
        
        Args:
            message: Error message.
            validation_errors: List of specific validation errors.
            field_name: Name of the field that failed validation.
            **kwargs: Additional arguments for SDKError.
        """
        super().__init__(message, **kwargs)
        self.validation_errors = validation_errors or []
        self.field_name = field_name
        
        # Add validation context
        self.context.update({
            'validation_errors': self.validation_errors,
            'field_name': self.field_name
        })


class ConfigurationError(SDKError):
    """Error raised when configuration is invalid."""
    
    def __init__(self, message: str, config_section: Optional[str] = None,
                 config_key: Optional[str] = None, **kwargs):
        """Initialize configuration error.
        
        Args:
            message: Error message.
            config_section: Configuration section with error.
            config_key: Specific configuration key with error.
            **kwargs: Additional arguments for SDKError.
        """
        super().__init__(message, **kwargs)
        self.config_section = config_section
        self.config_key = config_key
        
        # Add configuration context
        self.context.update({
            'config_section': self.config_section,
            'config_key': self.config_key
        })


class SimulationError(SDKError):
    """Error raised during simulation operations."""
    
    def __init__(self, message: str, simulation_type: Optional[str] = None,
                 convergence_data: Optional[Dict[str, float]] = None,
                 iteration_count: Optional[int] = None, **kwargs):
        """Initialize simulation error.
        
        Args:
            message: Error message.
            simulation_type: Type of simulation that failed.
            convergence_data: Convergence data at time of failure.
            iteration_count: Number of iterations completed.
            **kwargs: Additional arguments for SDKError.
        """
        super().__init__(message, **kwargs)
        self.simulation_type = simulation_type
        self.convergence_data = convergence_data or {}
        self.iteration_count = iteration_count
        
        # Add simulation context
        self.context.update({
            'simulation_type': self.simulation_type,
            'convergence_data': self.convergence_data,
            'iteration_count': self.iteration_count
        })


class EngineError(SDKError):
    """Error raised by SDK engines."""
    
    def __init__(self, message: str, engine_name: Optional[str] = None,
                 operation: Optional[str] = None, **kwargs):
        """Initialize engine error.
        
        Args:
            message: Error message.
            engine_name: Name of the engine that raised the error.
            operation: Operation being performed when error occurred.
            **kwargs: Additional arguments for SDKError.
        """
        super().__init__(message, **kwargs)
        self.engine_name = engine_name
        self.operation = operation
        
        # Add engine context
        self.context.update({
            'engine_name': self.engine_name,
            'operation': self.operation
        })


class DataError(SDKError):
    """Error raised when data operations fail."""
    
    def __init__(self, message: str, data_type: Optional[str] = None,
                 data_id: Optional[str] = None, **kwargs):
        """Initialize data error.
        
        Args:
            message: Error message.
            data_type: Type of data involved in error.
            data_id: ID of specific data object.
            **kwargs: Additional arguments for SDKError.
        """
        super().__init__(message, **kwargs)
        self.data_type = data_type
        self.data_id = data_id
        
        # Add data context
        self.context.update({
            'data_type': self.data_type,
            'data_id': self.data_id
        })


class OptimizationError(SDKError):
    """Error raised during optimization operations."""
    
    def __init__(self, message: str, optimizer_type: Optional[str] = None,
                 objective_value: Optional[float] = None,
                 constraint_violations: Optional[List[str]] = None, **kwargs):
        """Initialize optimization error.
        
        Args:
            message: Error message.
            optimizer_type: Type of optimizer that failed.
            objective_value: Last objective function value.
            constraint_violations: List of constraint violations.
            **kwargs: Additional arguments for SDKError.
        """
        super().__init__(message, **kwargs)
        self.optimizer_type = optimizer_type
        self.objective_value = objective_value
        self.constraint_violations = constraint_violations or []
        
        # Add optimization context
        self.context.update({
            'optimizer_type': self.optimizer_type,
            'objective_value': self.objective_value,
            'constraint_violations': self.constraint_violations
        })


class MaterialsError(EngineError):
    """Error specific to materials engine operations."""
    
    def __init__(self, message: str, material_id: Optional[str] = None,
                 frequency: Optional[float] = None, temperature: Optional[float] = None,
                 **kwargs):
        """Initialize materials error.
        
        Args:
            message: Error message.
            material_id: ID of material involved in error.
            frequency: Frequency at which error occurred (Hz).
            temperature: Temperature at which error occurred (K).
            **kwargs: Additional arguments for EngineError.
        """
        kwargs['engine_name'] = 'materials'
        super().__init__(message, **kwargs)
        self.material_id = material_id
        self.frequency = frequency
        self.temperature = temperature
        
        # Add materials-specific context
        self.context.update({
            'material_id': self.material_id,
            'frequency': self.frequency,
            'temperature': self.temperature
        })


class AerodynamicsError(EngineError):
    """Error specific to aerodynamics engine operations."""
    
    def __init__(self, message: str, mach_number: Optional[float] = None,
                 altitude: Optional[float] = None, angle_of_attack: Optional[float] = None,
                 **kwargs):
        """Initialize aerodynamics error.
        
        Args:
            message: Error message.
            mach_number: Mach number at which error occurred.
            altitude: Altitude at which error occurred (m).
            angle_of_attack: Angle of attack at which error occurred (degrees).
            **kwargs: Additional arguments for EngineError.
        """
        kwargs['engine_name'] = 'aerodynamics'
        super().__init__(message, **kwargs)
        self.mach_number = mach_number
        self.altitude = altitude
        self.angle_of_attack = angle_of_attack
        
        # Add aerodynamics-specific context
        self.context.update({
            'mach_number': self.mach_number,
            'altitude': self.altitude,
            'angle_of_attack': self.angle_of_attack
        })


class CFDError(AerodynamicsError):
    """Error specific to CFD operations."""
    
    def __init__(self, message: str, solver_type: Optional[str] = None,
                 mesh_quality: Optional[float] = None, convergence_residual: Optional[float] = None,
                 **kwargs):
        """Initialize CFD error.
        
        Args:
            message: Error message.
            solver_type: Type of CFD solver being used.
            mesh_quality: Mesh quality metric at time of error.
            convergence_residual: Convergence residual at time of error.
            **kwargs: Additional arguments for AerodynamicsError.
        """
        kwargs['operation'] = 'cfd_analysis'
        super().__init__(message, **kwargs)
        self.solver_type = solver_type
        self.mesh_quality = mesh_quality
        self.convergence_residual = convergence_residual
        
        # Add CFD-specific context
        self.context.update({
            'solver_type': self.solver_type,
            'mesh_quality': self.mesh_quality,
            'convergence_residual': self.convergence_residual
        })


class ReliabilityError(EngineError):
    """Error specific to reliability analysis operations."""
    
    def __init__(self, message: str, component_id: Optional[str] = None,
                 failure_mode: Optional[str] = None, analysis_type: Optional[str] = None,
                 **kwargs):
        """Initialize reliability error.
        
        Args:
            message: Error message.
            component_id: ID of component involved in error.
            failure_mode: Failure mode being analyzed.
            analysis_type: Type of reliability analysis.
            **kwargs: Additional arguments for EngineError.
        """
        kwargs['engine_name'] = 'reliability'
        super().__init__(message, **kwargs)
        self.component_id = component_id
        self.failure_mode = failure_mode
        self.analysis_type = analysis_type
        
        # Add reliability-specific context
        self.context.update({
            'component_id': self.component_id,
            'failure_mode': self.failure_mode,
            'analysis_type': self.analysis_type
        })


class ErrorHandler:
    """Centralized error handling for the Fighter Jet SDK."""
    
    def __init__(self):
        """Initialize error handler."""
        self.error_history: List[Dict[str, Any]] = []
        self.error_callbacks: Dict[str, List[callable]] = {}
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """Handle an error with appropriate logging and recovery.
        
        Args:
            error: Exception to handle.
            context: Additional context information.
        """
        from .logging import get_logger
        
        logger = get_logger('error_handler')
        
        # Convert to SDK error if needed
        if not isinstance(error, SDKError):
            sdk_error = SDKError(
                message=str(error),
                context=context,
                cause=error
            )
        else:
            sdk_error = error
            if context:
                sdk_error.context.update(context)
        
        # Log the error
        logger.error(
            f"Error occurred: {sdk_error}",
            extra={
                'error_code': sdk_error.error_code,
                'error_context': sdk_error.context
            }
        )
        
        # Add to error history
        self.error_history.append(sdk_error.to_dict())
        
        # Execute callbacks
        error_type = type(sdk_error).__name__
        if error_type in self.error_callbacks:
            for callback in self.error_callbacks[error_type]:
                try:
                    callback(sdk_error)
                except Exception as callback_error:
                    logger.error(f"Error in error callback: {callback_error}")
    
    def register_error_callback(self, error_type: str, callback: callable) -> None:
        """Register callback for specific error type.
        
        Args:
            error_type: Type of error to handle.
            callback: Callback function to execute.
        """
        if error_type not in self.error_callbacks:
            self.error_callbacks[error_type] = []
        self.error_callbacks[error_type].append(callback)
    
    def get_error_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get error history.
        
        Args:
            limit: Maximum number of errors to return.
            
        Returns:
            List of error dictionaries.
        """
        if limit:
            return self.error_history[-limit:]
        return self.error_history.copy()
    
    def clear_error_history(self) -> None:
        """Clear error history."""
        self.error_history.clear()
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics.
        
        Returns:
            Dictionary with error statistics.
        """
        if not self.error_history:
            return {'total_errors': 0}
        
        error_types = {}
        error_codes = {}
        
        for error in self.error_history:
            error_type = error.get('error_type', 'Unknown')
            error_code = error.get('error_code', 'Unknown')
            
            error_types[error_type] = error_types.get(error_type, 0) + 1
            error_codes[error_code] = error_codes.get(error_code, 0) + 1
        
        return {
            'total_errors': len(self.error_history),
            'error_types': error_types,
            'error_codes': error_codes,
            'most_recent': self.error_history[-1] if self.error_history else None
        }


# Global error handler instance
_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get global error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


def handle_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """Handle an error using the global error handler."""
    get_error_handler().handle_error(error, context)


def register_error_callback(error_type: str, callback: callable) -> None:
    """Register error callback using the global error handler."""
    get_error_handler().register_error_callback(error_type, callback)