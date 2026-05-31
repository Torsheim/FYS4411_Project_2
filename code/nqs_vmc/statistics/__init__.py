"""Statistical error analysis."""

from .blocking import blocking_analysis, blocking_error
from .error_estimates import standard_error, confidence_interval

__all__ = ["blocking_analysis", "blocking_error", "standard_error", "confidence_interval"]
