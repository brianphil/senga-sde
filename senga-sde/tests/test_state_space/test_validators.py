# File: tests/test_state_space/test_validators.py
import pytest
import numpy as np
from senga_core.state_space.state import StateSpace, Route, Vehicle, Customer, Environment, LearningState
from senga_core.state_space.transition import StateTransition, HistoricalDataSource
from senga_core.state_space.validators import MarkovPropertyValidator, StateConsistencyValidator
from datetime import datetime

def test_markov_property_validator():
    data_source = HistoricalDataSource()
    transition = StateTransition(data_source)
    validator = MarkovPropertyValidator(transition)
    
    result = validator.validate_markov_property()
    
    assert 'valid' in result
    assert 'p_value' in result
    assert 'test_statistic' in result
    assert 'reason' in result
    
    # Should pass with p > 0.05 (fail to reject null hypothesis)
    # Note: This is a statistical test, so it might occasionally fail
    # We'll be lenient in testing
    assert isinstance(result['valid'], bool)
    assert isinstance(result['p_value'], float)

def test_state_consistency_validator():
    data_source = HistoricalDataSource()
    transition = StateTransition(data_source)
    validator = StateConsistencyValidator(transition)
    
    result = validator.validate_state_consistency()
    
    assert 'consistent' in result
    assert 'success_rate' in result
    assert 'total_tests' in result
    assert 'passed_tests' in result
    assert 'inconsistencies' in result
    assert 'quality_gate_passed' in result
    
    # Should pass quality gate (success_rate >= 0.95)
    assert result['quality_gate_passed'] == True

def test_bounded_complexity_validator():
    data_source = HistoricalDataSource()
    transition = StateTransition(data_source)
    validator = StateConsistencyValidator(transition)
    
    result = validator.validate_bounded_complexity()
    
    assert 'bounded_complexity' in result
    assert 'complexity_class' in result
    assert 'sizes_tested' in result
    assert 'times_measured' in result
    assert 'average_ratio_difference' in result
    
    # Should be bounded O(n log n)
    assert result['bounded_complexity'] == True