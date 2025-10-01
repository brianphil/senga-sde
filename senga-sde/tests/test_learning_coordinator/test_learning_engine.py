# File: tests/test_learning_coordinator/test_learning_engine.py
import pytest
import numpy as np
from datetime import datetime, timedelta
from senga_core.learning_coordinator.learning_engine import RealTimeLearningEngine, DecisionOutcome

def test_learning_engine_initialization():
    engine = RealTimeLearningEngine()
    assert engine is not None
    assert len(engine.pattern_learners) == 3

def test_process_outcome_feedback():
    engine = RealTimeLearningEngine()
    
    # Create test decision outcome
    outcome = DecisionOutcome(
        decision_id="DEC001",
        original_decision={
            'expected_outcomes': {
                'predicted_efficiency': 0.8,
                'predicted_time': 45.0,
                'predicted_cost': 100.0
            },
            'delivery_sequence': [{'address_confidence': 0.8}]
        },
        actual_result={
            'measured_efficiency': 0.75,
            'actual_time': 50.0,
            'actual_cost': 110.0,
            'delivery_success': True,
            'traffic_index': 0.6,
            'time_of_day': 14,
            'day_of_week': 1,
            'weather_index': 0.3
        },
        timestamp=datetime.now(),
        scale='tactical'
    )
    
    patterns = engine.process_outcome_feedback(outcome)
    
    assert isinstance(patterns, list)
    assert len(patterns) > 0  # Should learn at least traffic pattern

def test_calculate_prediction_errors():
    engine = RealTimeLearningEngine()
    
    decision = {
        'expected_outcomes': {
            'predicted_efficiency': 0.8,
            'predicted_time': 45.0,
            'predicted_cost': 100.0
        }
    }
    
    actual_result = {
        'measured_efficiency': 0.75,
        'actual_time': 50.0,
        'actual_cost': 110.0
    }
    
    errors = engine.calculate_prediction_errors(decision, actual_result)
    
    assert 'efficiency_error' in errors
    assert 'time_error' in errors
    assert 'cost_error' in errors
    assert errors['efficiency_error'] == 0.05
    assert abs(errors['time_error'] - 0.111) < 0.001
    assert errors['cost_error'] == 0.1

def test_assess_learning_velocity():
    engine = RealTimeLearningEngine()
    
    # Add some learning history
    for i in range(10):
        engine.learning_history.append({
            'timestamp': datetime.now() - timedelta(days=i),
            'prediction_errors': {'efficiency_error': 0.1 - (i * 0.01)},
            'extracted_patterns': []
        })
    
    velocity = engine.assess_learning_velocity(window_days=5)
    
    assert 'accuracy_improvement_rate' in velocity
    assert 'learning_velocity_score' in velocity
    assert 'meets_target' in velocity
    assert isinstance(velocity['accuracy_improvement_rate'], float)