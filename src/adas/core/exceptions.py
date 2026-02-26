"""Custom exceptions for ADAS system.

This module defines domain-specific exceptions for error handling.
"""

from __future__ import annotations


class ADASException(Exception):
    """Base exception for all ADAS errors."""
    pass


class ValidationError(ADASException):
    """Raised when input validation fails."""
    pass


class SensorError(ADASException):
    """Raised when sensor data is invalid or unavailable."""
    pass


class SafetyViolation(ADASException):
    """Raised when a safety constraint is violated."""
    pass


class ConfigurationError(ADASException):
    """Raised when configuration is invalid."""
    pass


class PerceptionError(ADASException):
    """Raised when perception pipeline fails."""
    pass


class TrackingError(ADASException):
    """Raised when object tracking fails."""
    pass


class PlanningError(ADASException):
    """Raised when motion planning fails."""
    pass


class ControlError(ADASException):
    """Raised when control command generation fails."""
    pass
