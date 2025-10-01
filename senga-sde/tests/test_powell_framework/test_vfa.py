# File: tests/test_powell_framework/test_vfa.py
import pytest
import numpy as np
from datetime import datetime
from senga_core.state_space.state import StateSpace, Route, Vehicle, Customer, Environment, LearningState
from senga_core.powell_framework.vfa import StrategicVFA
from senga_core.powell_framework.exceptions import ConvergenceException, FeatureExtractionError

def create_test_state():
    """Create a test state for VFA testing"""
    routes = [Route("R001", [(-1.2864, 36.8172)], 10.0, 30.0, 0.8)]
    fleet = [Vehicle("V01", 2000.0, 0.5, "good", (-1.2864, 36.8172), 0.7)]
    customers = [Customer("C001", (-1.2864, 36.8172), 100.0, (9, 17), 0.8, "mpesa")]
    env = Environment(0.5, "clear", "online", 12, 1)
    learning = LearningState(0.85, "v1.0", datetime.now(), 0.1)
    
    return StateSpace(routes, fleet, customers, env, learning)

def test_vfa_initialization():
    vfa = StrategicVFA(learning_rate=0.001, gamma=0.95)
    assert vfa.learning_rate == 0.001
    assert vfa.gamma == 0.95
    assert len(vfa.theta) == 14  # Feature dimensionality
    assert len(vfa.convergence_metrics) == 0

def test_vfa_td_update():
    vfa = StrategicVFA(learning_rate=0.001, gamma=0.95)
    state = create_test_state()
    next_state = create_test_state()  # Same state for simplicity
    
    # Perform TD update
    metric = vfa.td_update(state, "test_action", 10.0, next_state)
    
    assert isinstance(metric, dict)
    assert 'td_error' in metric
    assert 'parameter_change_norm' in metric
    assert 'timestamp' in metric
    assert len(vfa.convergence_metrics) == 1

def test_vfa_predict_value():
    vfa = StrategicVFA(learning_rate=0.001, gamma=0.95)
    state = create_test_state()
    
    value = vfa.predict_value(state)
    assert isinstance(value, float)

def test_vfa_convergence_check():
    vfa = StrategicVFA(learning_rate=0.001, gamma=0.95)
    state = create_test_state()
    next_state = create_test_state()
    
    # Perform 50 updates
    for i in range(50):
        vfa.td_update(state, f"action_{i}", 10.0 + i*0.1, next_state)
    
    # Check convergence (should fail with small window)
    result = vfa.check_convergence(window_size=100, threshold=1e-4)
    assert result['converged'] == False
    assert result['reason'] == 'Insufficient updates (50 < 100)'