"""Logging framework for the Fighter Jet SDK."""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json

from .config import get_config


class SDKFormatter(logging.Formatter):
    """Custom formatter for SDK logging."""
    
    def __init__(self, include_context: bool = True):
        """Initialize formatter.
        
        Args:
            include_context: Whether to include additional context in log messages.
        """
        self.include_context = include_context
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record."""
        # Add SDK-specific context
        if self.include_context:
            if hasattr(record, 'engine'):
                record.engine_name = f"[{record.engine}] "
            else:
                record.engine_name = ""
            
            if hasattr(record, 'operation'):
                record.operation_name = f"({record.operation}) "
            else:
                record.operation_name = ""
        else:
            record.engine_name = ""
            record.operation_name = ""
        
        # Use config format or default
        config = get_config()
        format_string = config.log_format
        
        # Enhance format with SDK context
        if self.include_context:
            format_string = format_string.replace(
                "%(name)s", 
                "%(name)s %(engine_name)s%(operation_name)s"
            )
        
        formatter = logging.Formatter(format_string)
        return formatter.format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add SDK-specific fields
        if hasattr(record, 'engine'):
            log_entry['engine'] = record.engine
        
        if hasattr(record, 'operation'):
            log_entry['operation'] = record.operation
        
        if hasattr(record, 'config_id'):
            log_entry['config_id'] = record.config_id
        
        if hasattr(record, 'performance_data'):
            log_entry['performance'] = record.performance_data
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


class PerformanceFilter(logging.Filter):
    """Filter for performance-related log messages."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter performance logs."""
        return hasattr(record, 'performance_data')


class EngineFilter(logging.Filter):
    """Filter for specific engine log messages."""
    
    def __init__(self, engine_name: str):
        """Initialize filter for specific engine.
        
        Args:
            engine_name: Name of the engine to filter for.
        """
        super().__init__()
        self.engine_name = engine_name
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter engine logs."""
        return getattr(record, 'engine', '') == self.engine_name


class LogManager:
    """Centralized logging manager for the Fighter Jet SDK."""
    
    def __init__(self):
        """Initialize log manager."""
        self.loggers: Dict[str, logging.Logger] = {}
        self.handlers: Dict[str, logging.Handler] = {}
        self.initialized = False
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        config = get_config()
        
        # Configure root logger
        root_logger = logging.getLogger('fighter_jet_sdk')
        root_logger.setLevel(getattr(logging, config.log_level.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, config.log_level.upper()))
        console_handler.setFormatter(SDKFormatter(include_context=True))
        root_logger.addHandler(console_handler)
        self.handlers['console'] = console_handler
        
        # File handler (if configured)
        if config.log_file:
            log_path = Path(config.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(SDKFormatter(include_context=True))
            root_logger.addHandler(file_handler)
            self.handlers['file'] = file_handler
        
        # Performance log handler
        perf_handler = logging.StreamHandler(sys.stdout)
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(JSONFormatter())
        perf_handler.addFilter(PerformanceFilter())
        
        perf_logger = logging.getLogger('fighter_jet_sdk.performance')
        perf_logger.addHandler(perf_handler)
        perf_logger.propagate = False
        self.handlers['performance'] = perf_handler
        
        self.initialized = True
    
    def get_logger(self, name: str, engine: Optional[str] = None) -> logging.Logger:
        """Get logger instance.
        
        Args:
            name: Logger name.
            engine: Engine name for context.
            
        Returns:
            Logger instance.
        """
        full_name = f"fighter_jet_sdk.{name}"
        
        if full_name not in self.loggers:
            logger = logging.getLogger(full_name)
            
            # Add engine context if provided
            if engine:
                logger = EngineLoggerAdapter(logger, {'engine': engine})
            
            self.loggers[full_name] = logger
        
        return self.loggers[full_name]
    
    def get_engine_logger(self, engine_name: str) -> logging.Logger:
        """Get logger for specific engine.
        
        Args:
            engine_name: Name of the engine.
            
        Returns:
            Logger instance with engine context.
        """
        return self.get_logger(f"engines.{engine_name}", engine=engine_name)
    
    def log_performance(self, operation: str, duration: float, 
                       engine: Optional[str] = None, **kwargs) -> None:
        """Log performance data.
        
        Args:
            operation: Operation name.
            duration: Operation duration in seconds.
            engine: Engine name.
            **kwargs: Additional performance data.
        """
        perf_logger = logging.getLogger('fighter_jet_sdk.performance')
        
        performance_data = {
            'operation': operation,
            'duration_seconds': duration,
            **kwargs
        }
        
        perf_logger.info(
            f"Performance: {operation} completed in {duration:.3f}s",
            extra={
                'performance_data': performance_data,
                'engine': engine,
                'operation': operation
            }
        )
    
    def add_engine_handler(self, engine_name: str, handler: logging.Handler) -> None:
        """Add handler for specific engine.
        
        Args:
            engine_name: Name of the engine.
            handler: Log handler to add.
        """
        engine_logger = logging.getLogger(f'fighter_jet_sdk.engines.{engine_name}')
        handler.addFilter(EngineFilter(engine_name))
        engine_logger.addHandler(handler)
        
        handler_key = f"{engine_name}_custom"
        self.handlers[handler_key] = handler
    
    def set_log_level(self, level: str, logger_name: Optional[str] = None) -> None:
        """Set log level.
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            logger_name: Specific logger name, or None for root logger.
        """
        log_level = getattr(logging, level.upper())
        
        if logger_name:
            logger = logging.getLogger(f"fighter_jet_sdk.{logger_name}")
            logger.setLevel(log_level)
        else:
            root_logger = logging.getLogger('fighter_jet_sdk')
            root_logger.setLevel(log_level)
            
            # Update all handlers
            for handler in self.handlers.values():
                handler.setLevel(log_level)
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics.
        
        Returns:
            Dictionary with logging statistics.
        """
        stats = {
            'initialized': self.initialized,
            'loggers_count': len(self.loggers),
            'handlers_count': len(self.handlers),
            'handlers': list(self.handlers.keys())
        }
        
        # Add handler-specific stats
        for name, handler in self.handlers.items():
            if hasattr(handler, 'stream') and hasattr(handler.stream, 'name'):
                stats[f'{name}_output'] = handler.stream.name
        
        return stats


class EngineLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds engine context to log messages."""
    
    def process(self, msg, kwargs):
        """Process log message with engine context."""
        # Add engine context to extra data
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        kwargs['extra'].update(self.extra)
        
        return msg, kwargs


# Global log manager instance
_log_manager: Optional[LogManager] = None


def get_log_manager() -> LogManager:
    """Get global log manager instance."""
    global _log_manager
    if _log_manager is None:
        _log_manager = LogManager()
    return _log_manager


def get_logger(name: str, engine: Optional[str] = None) -> logging.Logger:
    """Get logger instance."""
    return get_log_manager().get_logger(name, engine)


def get_engine_logger(engine_name: str) -> logging.Logger:
    """Get logger for specific engine."""
    return get_log_manager().get_engine_logger(engine_name)


def log_performance(operation: str, duration: float, 
                   engine: Optional[str] = None, **kwargs) -> None:
    """Log performance data."""
    get_log_manager().log_performance(operation, duration, engine, **kwargs)