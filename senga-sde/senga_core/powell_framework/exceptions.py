# File: senga_core/powell_framework/exceptions.py
class OptimizationFailedException(Exception):
    """Raised when tactical optimization fails to converge"""
    pass

class ConvergenceException(Exception):
    """Raised when VFA fails to converge within acceptable bounds"""
    pass

class FeatureExtractionError(Exception):
    """Raised when feature extraction fails"""
    pass

class CostCalculationError(Exception):
    """Raised when cost calculation fails"""
    pass