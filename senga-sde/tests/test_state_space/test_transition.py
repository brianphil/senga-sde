# File: tests/test_state_space/test_transition.py
import pytest
import numpy as np
from senga_core.state_space.state import StateSpace, Route, Vehicle, Customer, Environment, LearningState
from senga_core.state_space.transition import StateTransition, HistoricalDataSource
from datetime import datetime

def test_historical_data_source_initialization():
    data_source = HistoricalDataSource()
    assert len(data_source.transitions) == 1000  # As generated

def test_state_transition_probability():
    data_source = HistoricalDataSource()
    transition = StateTransition(data_source)
    
    # Get a sample state and action from historical data
    sample_trans = data_source.transitions[0]
    state = sample_trans['state_before']
    action = sample_trans['action']
    next_state = sample_trans['state_after']
    
    # Calculate probability
    prob = transition.transition_probability(state, action, next_state)
    
    assert isinstance(prob, float)
    assert prob >= 0.0
    assert prob <= 1.0

def test_state_transition_sum_to_one():
    data_source = HistoricalDataSource()
    transition = StateTransition(data_source)
    
    # Test with first 10 transitions
    for i in range(10):
        trans = data_source.transitions[i]
        state = trans['state_before']
        action = trans['action']
        
        # Get all possible next states
        transitions = transition.get_possible_transitions(state, action)
        total_prob = sum(prob for _, prob in transitions)
        
        assert abs(total_prob - 1.0) < 0.01, f"Probabilities sum to {total_prob}, not 1.0"

def test_state_transition_caching():
    data_source = HistoricalDataSource()
    transition = StateTransition(data_source)
    
    # Get a sample transition
    trans = data_source.transitions[0]
    state = trans['state_before']
    action = trans['action']
    next_state = trans['state_after']
    
    # First call
    prob1 = transition.transition_probability(state, action, next_state)
    
    # Second call should be cached
    prob2 = transition.transition_probability(state, action, next_state)
    
    assert prob1 == prob2