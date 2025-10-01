# File: tests/test_state_space/test_state.py
import pytest
import numpy as np
from senga_core.state_space.state import StateSpace, Route, Vehicle, Customer, Environment, LearningState
from datetime import datetime

def test_state_space_initialization():
    # Create minimal state
    routes = [Route("R001", [(-1.2864, 36.8172)], 10.0, 30.0, 0.8)]
    fleet = [Vehicle("V01", 2000.0, 0.5, "good", (-1.2864, 36.8172), 0.7)]
    customers = [Customer("C001", (-1.2864, 36.8172), 100.0, (9, 17), 0.8, "mpesa")]
    env = Environment(0.5, "clear", "online", 12, 1)
    learning = LearningState(0.85, "v1.0", datetime.now(), 0.1)
    
    state = StateSpace(routes, fleet, customers, env, learning)
    
    assert state.routes == routes
    assert state.fleet == fleet
    assert state.customers == customers
    assert state.environment == env
    assert state.learning == learning

def test_state_to_vector():
    routes = [Route("R001", [(-1.2864, 36.8172)], 10.0, 30.0, 0.8)]
    fleet = [Vehicle("V01", 2000.0, 0.5, "good", (-1.2864, 36.8172), 0.7)]
    customers = [Customer("C001", (-1.2864, 36.8172), 100.0, (9, 17), 0.8, "mpesa")]
    env = Environment(0.5, "clear", "online", 12, 1)
    learning = LearningState(0.85, "v1.0", datetime.now(), 0.1)
    
    state = StateSpace(routes, fleet, customers, env, learning)
    vector = state.to_vector()
    
    assert isinstance(vector, np.ndarray)
    assert len(vector) == 14  # Expected feature count
    assert vector[0] == 0.8  # routes mean efficiency
    assert vector[1] == 1.0  # routes count
    assert vector[2] == 0.5  # fleet mean fuel
    assert vector[3] == 0.7  # fleet mean utilization
    assert vector[4] == 1.0  # fleet count
    assert vector[5] == 0.8  # customers mean availability
    assert vector[6] == 100.0  # customers total demand
    assert vector[7] == 0.5  # traffic index
    assert abs(vector[8] - 0.5) < 0.01  # time_of_day normalized
    assert abs(vector[9] - 0.142) < 0.01  # day_of_week normalized
    assert vector[10] == 0.85  # prediction accuracy
    assert vector[11] == 0.1  # uncertainty estimate

def test_state_signature():
    routes = [Route("R001", [(-1.2864, 36.8172)], 10.0, 30.0, 0.8)]
    fleet = [Vehicle("V01", 2000.0, 0.5, "good", (-1.2864, 36.8172), 0.7)]
    customers = [Customer("C001", (-1.2864, 36.8172), 100.0, (9, 17), 0.8, "mpesa")]
    env = Environment(0.5, "clear", "online", 12, 1)
    learning = LearningState(0.85, "v1.0", datetime.now(), 0.1)
    
    state1 = StateSpace(routes, fleet, customers, env, learning)
    state2 = StateSpace(routes, fleet, customers, env, learning)  # Identical
    
    assert state1.get_state_signature() == state2.get_state_signature()
    assert hash(state1) == hash(state2)
    assert state1 == state2