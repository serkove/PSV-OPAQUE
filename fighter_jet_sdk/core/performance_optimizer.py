"""Performance optimization utilities for the Fighter Jet SDK."""

import time
import psutil
import threading
import multiprocessing
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
import gc
import sys
from pathlib import Path

from .config import get_config
from .errors import PerformanceError


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    peak_memory_mb: float
    cache_hits: int = 0
    cache_misses: int = 0


@dataclass
class OptimizationSettings:
    """Settings for performance optimization."""
    enable_parallel_processing: bool = True
    max_threads: Optional[int] = None
    max_processes: Optional[int] = None
    enable_caching: bool = True
    cache_size_mb: int = 1024
    enable_memory_optimization: bool = True
    gc_threshold: int = 1000  # Number of operations before garbage collection


class PerformanceOptimizer:
    """Performance optimization manager for the Fighter Jet SDK."""
    
    def __init__(self):
        """Initialize performance optimizer."""
        self.logger = logging.getLogger(__name__)
        self.config = get_config()
        self.settings = self._load_optimization_settings()
        self.metrics_history: List[PerformanceMetrics] = []
        self.operation_count = 0
        self._cache: Dict[str, Any] = {}
        self._cache_stats = {"hits": 0, "misses": 0}
        
        # Initialize thread and process pools
        self.thread_pool = None
        self.process_pool = None
        self._initialize_pools()
    
    def _load_optimization_settings(self) -> OptimizationSettings:
        """Load optimization settings from configuration."""
        return OptimizationSettings(
            enable_parallel_processing=self.config.parallel_processing,
            max_threads=self.config.max_threads,
            enable_caching=self.config.cache_enabled,
            cache_size_mb=self.config.cache_size_mb,
            enable_memory_optimization=True
        )
    
    def _initialize_pools(self):
        """Initialize thread and process pools."""
        if self.settings.enable_parallel_processing:
            max_threads = self.settings.max_threads or min(32, (multiprocessing.cpu_count() or 1) + 4)
            max_processes = self.settings.max_processes or multiprocessing.cpu_count()
            
            self.thread_pool = ThreadPoolExecutor(max_workers=max_threads)
            self.process_pool = ProcessPoolExecutor(max_workers=max_processes)
            
            self.logger.info(f"Initialized thread pool with {max_threads} workers")
            self.logger.info(f"Initialized process pool with {max_processes} workers")
    
    def optimize_function(self, func: Callable, *args, use_cache: bool = True, **kwargs) -> Any:
        """Optimize function execution with caching and performance monitoring.
        
        Args:
            func: Function to optimize
            *args: Function arguments
            use_cache: Whether to use caching
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        # Generate cache key
        cache_key = None
        if use_cache and self.settings.enable_caching:
            cache_key = self._generate_cache_key(func, args, kwargs)
            
            # Check cache
            if cache_key in self._cache:
                self._cache_stats["hits"] += 1
                return self._cache[cache_key]
            else:
                self._cache_stats["misses"] += 1
        
        # Monitor performance
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = psutil.cpu_percent()
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result if enabled
            if cache_key and self.settings.enable_caching:
                self._add_to_cache(cache_key, result)
            
            # Record metrics
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_cpu = psutil.cpu_percent()
            
            metrics = PerformanceMetrics(
                execution_time=end_time - start_time,
                memory_usage_mb=end_memory,
                cpu_usage_percent=end_cpu,
                peak_memory_mb=max(start_memory, end_memory),
                cache_hits=self._cache_stats["hits"],
                cache_misses=self._cache_stats["misses"]
            )
            
            self.metrics_history.append(metrics)
            self.operation_count += 1
            
            # Perform garbage collection if needed
            if (self.settings.enable_memory_optimization and 
                self.operation_count % self.settings.gc_threshold == 0):
                gc.collect()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in optimized function execution: {e}")
            raise PerformanceError(f"Function optimization failed: {e}")
    
    def parallel_map(self, func: Callable, items: List[Any], use_processes: bool = False) -> List[Any]:
        """Execute function in parallel over a list of items.
        
        Args:
            func: Function to execute
            items: List of items to process
            use_processes: Whether to use processes instead of threads
            
        Returns:
            List of results
        """
        if not self.settings.enable_parallel_processing or len(items) < 2:
            return [func(item) for item in items]
        
        try:
            executor = self.process_pool if use_processes else self.thread_pool
            if executor is None:
                return [func(item) for item in items]
            
            futures = [executor.submit(func, item) for item in items]
            results = [future.result() for future in futures]
            
            self.logger.debug(f"Parallel execution completed for {len(items)} items")
            return results
            
        except Exception as e:
            self.logger.warning(f"Parallel execution failed, falling back to sequential: {e}")
            return [func(item) for item in items]
    
    def batch_optimize(self, operations: List[Callable], batch_size: int = 10) -> List[Any]:
        """Optimize batch execution of operations.
        
        Args:
            operations: List of operations to execute
            batch_size: Size of each batch
            
        Returns:
            List of results
        """
        results = []
        
        for i in range(0, len(operations), batch_size):
            batch = operations[i:i + batch_size]
            
            # Execute batch in parallel if possible
            if self.settings.enable_parallel_processing and len(batch) > 1:
                batch_results = self.parallel_map(lambda op: op(), batch)
            else:
                batch_results = [op() for op in batch]
            
            results.extend(batch_results)
            
            # Memory cleanup between batches
            if self.settings.enable_memory_optimization:
                gc.collect()
        
        return results
    
    def _generate_cache_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Generate cache key for function call."""
        try:
            func_name = f"{func.__module__}.{func.__name__}"
            args_str = str(hash(args))
            kwargs_str = str(hash(tuple(sorted(kwargs.items()))))
            return f"{func_name}_{args_str}_{kwargs_str}"
        except Exception:
            # Fallback to simple string representation
            return f"{func.__name__}_{hash(str(args))}_{hash(str(kwargs))}"
    
    def _add_to_cache(self, key: str, value: Any):
        """Add item to cache with size management."""
        # Simple cache size management
        if len(self._cache) > 1000:  # Arbitrary limit
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self._cache.keys())[:100]
            for k in keys_to_remove:
                del self._cache[k]
        
        self._cache[key] = value
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics.
        
        Returns:
            Dictionary with performance statistics
        """
        if not self.metrics_history:
            return {"message": "No performance data available"}
        
        execution_times = [m.execution_time for m in self.metrics_history]
        memory_usage = [m.memory_usage_mb for m in self.metrics_history]
        cpu_usage = [m.cpu_usage_percent for m in self.metrics_history]
        
        return {
            "total_operations": len(self.metrics_history),
            "average_execution_time": sum(execution_times) / len(execution_times),
            "max_execution_time": max(execution_times),
            "min_execution_time": min(execution_times),
            "average_memory_usage_mb": sum(memory_usage) / len(memory_usage),
            "peak_memory_usage_mb": max(memory_usage),
            "average_cpu_usage": sum(cpu_usage) / len(cpu_usage),
            "cache_hit_rate": (
                self._cache_stats["hits"] / 
                (self._cache_stats["hits"] + self._cache_stats["misses"])
                if (self._cache_stats["hits"] + self._cache_stats["misses"]) > 0 else 0
            ),
            "cache_size": len(self._cache)
        }
    
    def optimize_memory_usage(self):
        """Perform memory optimization operations."""
        if not self.settings.enable_memory_optimization:
            return
        
        # Clear cache if it's too large
        cache_size_mb = sys.getsizeof(self._cache) / 1024 / 1024
        if cache_size_mb > self.settings.cache_size_mb:
            self._cache.clear()
            self._cache_stats = {"hits": 0, "misses": 0}
            self.logger.info("Cache cleared due to size limit")
        
        # Force garbage collection
        collected = gc.collect()
        self.logger.debug(f"Garbage collection freed {collected} objects")
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self.metrics_history.clear()
        self.operation_count = 0
        self._cache_stats = {"hits": 0, "misses": 0}
        self.logger.info("Performance metrics reset")
    
    def shutdown(self):
        """Shutdown performance optimizer and cleanup resources."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        
        self._cache.clear()
        self.logger.info("Performance optimizer shutdown complete")


# Global performance optimizer instance
_performance_optimizer: Optional[PerformanceOptimizer] = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer


def optimize_function(func: Callable, *args, **kwargs) -> Any:
    """Convenience function for optimizing function execution."""
    return get_performance_optimizer().optimize_function(func, *args, **kwargs)


def parallel_map(func: Callable, items: List[Any], use_processes: bool = False) -> List[Any]:
    """Convenience function for parallel execution."""
    return get_performance_optimizer().parallel_map(func, items, use_processes)