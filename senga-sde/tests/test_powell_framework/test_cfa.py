# File: tests/test_powell_framework/test_cfa.py
import pytest
import numpy as np
from senga_core.state_space.state import StateSpace, Route, Vehicle, Customer, Environment, LearningState
from senga_core.powell_framework.cfa import TacticalCFA
from senga_core.powell_framework.exceptions import OptimizationFailedException

def create_test_state():
    """Create a test state for CFA testing"""
    routes = [Route("R001", [(-1.2864, 36.8172)], 10.0, 30.0, 0.8)]
    fleet = [Vehicle("V01", 2000.0, 0.5, "good", (-1.2864, 36.8172), 0.7)]
    customers = [Customer("C001", (-1.2864, 36.8172), 100.0, (9, 17), 0.8, "mpesa")]
    env = Environment(0.5, "clear", "online", 12, 1)
    learning = LearningState(0.85, "v1.0", None, 0.1)
    
    return StateSpace(routes, fleet, customers, env, learning)

def test_cfa_initialization():
    cfa = TacticalCFA()
    assert cfa.cost_function is not None
    assert len(cfa.optimization_history) == 0

# In test_cfa_optimization test

def test_cfa_optimization():
    cfa = TacticalCFA()
    state = create_test_state()
    
    # Simple, well-behaved constraint: cost must be positive (always true)
    def positive_cost_constraint(state, action):
        # Return large positive value to satisfy inequality constraint
        return 1000.0  # g(x) >= 0 is easily satisfied
    
    # Perform optimization
    action = cfa.optimize_tactical_decision(state, [positive_cost_constraint])
    
    assert isinstance(action, dict)
    assert 'route_plan' in action
    assert 'schedule' in action
    assert 'driver_assignments' in action
    assert 'time_windows' in action
    assert 'optimization_metadata' in action
    assert action['optimization_metadata']['success'] == True
    assert np.isfinite(action['optimization_metadata']['cost'])

def test_cfa_different_inputs_different_outputs():
    cfa = TacticalCFA()
    
    # Create two different states
    state1 = create_test_state()
    
    routes2 = [Route("R002", [(-1.2864, 36.8172)], 20.0, 60.0, 0.6)]
    fleet2 = [Vehicle("V02", 3000.0, 0.3, "warning", (-1.2864, 36.8172), 0.9)]
    customers2 = [Customer("C002", (-1.2864, 36.8172), 200.0, (10, 18), 0.6, "cash")]
    env2 = Environment(0.8, "rain", "degraded", 14, 2)
    learning2 = LearningState(0.75, "v1.0", None, 0.2)
    state2 = StateSpace(routes2, fleet2, customers2, env2, learning2)
    
    # Optimize for both states
    action1 = cfa.optimize_tactical_decision(state1)
    action2 = cfa.optimize_tactical_decision(state2)
    
    # Actions should be different
    assert action1 != action2
    assert action1['route_plan'][0]['distance_km'] != action2['route_plan'][0]['distance_km']
    assert action1['schedule']['total_hours'] != action2['schedule']['total_hours']

def test_cfa_optimization_history():
    cfa = TacticalCFA()
    state = create_test_state()
    
    # Perform optimization
    action = cfa.optimize_tactical_decision(state)
    
    # Check history
    history = cfa.get_optimization_history()
    assert len(history) == 1
    assert history[0]['success'] == True
    assert 'cost' in history[0]
    assert 'iterations' in history[0]