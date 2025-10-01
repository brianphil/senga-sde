# File: tests/test_powell_framework/test_costs.py
import pytest
import numpy as np
from senga_core.state_space.state import StateSpace, Route, Vehicle, Customer, Environment, LearningState
from senga_core.powell_framework.costs import SengaCostFunction, CostCalculationError

def create_test_state():
    """Create a test state for cost calculation testing"""
    routes = [Route("R001", [(-1.2864, 36.8172)], 10.0, 30.0, 0.8)]
    fleet = [Vehicle("V01", 2000.0, 0.5, "good", (-1.2864, 36.8172), 0.7)]
    customers = [Customer("C001", (-1.2864, 36.8172), 100.0, (9, 17), 0.8, "mpesa")]
    env = Environment(0.5, "clear", "online", 12, 1)
    learning = LearningState(0.85, "v1.0", None, 0.1)
    
    return StateSpace(routes, fleet, customers, env, learning)

def test_cost_function_initialization():
    cost_func = SengaCostFunction()
    assert cost_func.cost_weights is not None
    assert 'address_uncertainty' in cost_func.cost_weights
    assert 'cultural_misalignment' in cost_func.cost_weights
    assert 'connectivity_risk' in cost_func.cost_weights

def test_multi_objective_cost():
    cost_func = SengaCostFunction()
    state = create_test_state()
    
    # Create a simple action
    action = {
        'route_plan': [{'distance_km': 50.0}],
        'schedule': {'total_hours': 8.0},
        'driver_assignments': [{'hours': 8.0}],
        'time_windows': {'C001': {'start_hour': 10, 'end_hour': 12}},
        'vehicle_usage': {'total_distance_km': 50.0},
        'delivery_sequence': [{'customer_id': 'C001', 'address_confidence': 0.8}]
    }
    
    cost = cost_func.multi_objective_cost(state, action)
    assert isinstance(cost, float)
    assert cost >= 0.0

def test_address_resolution_risk():
    cost_func = SengaCostFunction()
    
    delivery_sequence = [
        {'customer_id': 'C001', 'address_confidence': 0.9},
        {'customer_id': 'C002', 'address_confidence': 0.5},
        {'customer_id': 'C003', 'address_confidence': 0.1}
    ]
    
    risk = cost_func.calculate_address_resolution_risk(delivery_sequence)
    assert isinstance(risk, float)
    assert risk > 0.0  # Lower confidence = higher risk

def test_cultural_timing_penalty():
    cost_func = SengaCostFunction()
    customers = [Customer("C001", (-1.2864, 36.8172), 100.0, (9, 17), 0.8, "mpesa")]
    
    time_windows = {
        'C001': {'start_hour': 10, 'end_hour': 12}  # Within preferred hours
    }
    
    penalty = cost_func.calculate_cultural_timing_penalty(time_windows, customers)
    assert isinstance(penalty, float)
    assert penalty >= 0.0