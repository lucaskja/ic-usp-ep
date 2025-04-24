"""
Utility module for marking functions as deprecated.

This module provides decorators and utilities to mark functions, methods, or classes
as deprecated, providing warnings to users when they are used.
"""
import functools
import warnings
import inspect
from typing import Callable, Optional, Any, Union, TypeVar

F = TypeVar('F', bound=Callable[..., Any])

def deprecated(func: Optional[F] = None, *, 
               reason: str = "This function is deprecated and will be removed in a future version.",
               alternative: Optional[str] = None) -> Union[F, Callable[[F], F]]:
    """
    Decorator to mark functions, methods, or classes as deprecated.
    
    Args:
        func: The function, method, or class to mark as deprecated
        reason: The reason for deprecation
        alternative: The recommended alternative to use instead
        
    Returns:
        The wrapped function/method/class with deprecation warning
        
    Example:
        @deprecated(reason="Use new_function() instead", alternative="new_function")
        def old_function():
            pass
            
        @deprecated
        def simple_old_function():
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = reason
            if alternative:
                message += f" Use {alternative} instead."
                
            # Get caller information
            frame = inspect.currentframe().f_back
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno
            
            # Format warning message with caller information
            warning_msg = f"{func.__name__} is deprecated. {message} (called from {filename}:{lineno})"
            
            # Issue deprecation warning
            warnings.warn(warning_msg, category=DeprecationWarning, stacklevel=2)
            
            return func(*args, **kwargs)
        
        # Add deprecation notice to docstring
        if func.__doc__:
            wrapper.__doc__ = f"{func.__doc__}\n\nDeprecated: {reason}"
        else:
            wrapper.__doc__ = f"Deprecated: {reason}"
            
        if alternative:
            wrapper.__doc__ += f" Use {alternative} instead."
            
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func)


class DeprecatedClass:
    """
    Base class for marking entire classes as deprecated.
    
    Example:
        class OldClass(DeprecatedClass):
            def __init__(self):
                super().__init__(reason="Use NewClass instead", alternative="NewClass")
                # Rest of initialization
    """
    def __init__(self, reason: str = "This class is deprecated and will be removed in a future version.", 
                 alternative: Optional[str] = None):
        """
        Initialize the deprecated class.
        
        Args:
            reason: The reason for deprecation
            alternative: The recommended alternative to use instead
        """
        message = reason
        if alternative:
            message += f" Use {alternative} instead."
            
        # Get caller information
        frame = inspect.currentframe().f_back
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        
        # Format warning message with caller information
        class_name = self.__class__.__name__
        warning_msg = f"{class_name} is deprecated. {message} (instantiated at {filename}:{lineno})"
        
        # Issue deprecation warning
        warnings.warn(warning_msg, category=DeprecationWarning, stacklevel=2)
