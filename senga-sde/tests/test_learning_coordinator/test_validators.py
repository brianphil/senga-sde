# File: tests/test_learning_coordinator/test_validators.py
import pytest
from senga_core.learning_coordinator.validators import LearningVelocityValidator
from senga_core.learning_coordinator.learning_engine import RealTimeLearningEngine
from datetime import datetime, timedelta
def test_learning_velocity_validator():
    engine = RealTimeLearningEngine()
    validator = LearningVelocityValidator(engine)
    
    # Add some learning history with improving errors
    for i in range(20):
        engine.learning_history.append({
            'timestamp': datetime.now() - timedelta(days=i),
            'prediction_errors': {'efficiency_error': max(0.01, 0.2 - (i * 0.01))},
            'extracted_patterns': []
        })
    
    result = validator.validate_learning_velocity(window_days=10)
    
    assert 'valid' in result
    assert 'velocity_metrics' in result
    assert 'meets_requirement' in result
    assert isinstance(result['valid'], bool)

def test_validate_convergence_properties():
    validator = LearningVelocityValidator(None)
    
    # Test with None models
    result = validator.validate_convergence_properties()
    
    assert result['valid'] == True  # Should pass when no models provided
    assert len(result['issues']) == 0