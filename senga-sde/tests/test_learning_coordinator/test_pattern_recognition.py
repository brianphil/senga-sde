# File: tests/test_learning_coordinator/test_pattern_recognition.py
import pytest
import numpy as np
from datetime import datetime
from senga_core.learning_coordinator.pattern_recognition import (
    TrafficPatternLearner, CustomerBehaviorLearner, RouteEfficiencyLearner
)

def test_traffic_pattern_learner():
    learner = TrafficPatternLearner()
    
    traffic_data = [
        {'hour': 8, 'day_of_week': 1, 'weather_index': 0.3, 'traffic_index': 0.8},
        {'hour': 17, 'day_of_week': 1, 'weather_index': 0.4, 'traffic_index': 0.9},
        {'hour': 12, 'day_of_week': 1, 'weather_index': 0.2, 'traffic_index': 0.6}
    ]
    
    pattern = learner.learn_traffic_patterns(traffic_data)
    
    assert pattern.pattern_type == "traffic"
    assert pattern.confidence > 0.0
    assert 'training_samples' in pattern.data

def test_customer_behavior_learner():
    learner = CustomerBehaviorLearner()
    
    customer_data = [
        {
            'customer_id': 'C001',
            'customer_type': 'retail_shop',
            'interaction_history': [
                {'timestamp': datetime(2023, 1, 1, 10), 'success': True},
                {'timestamp': datetime(2023, 1, 1, 14), 'success': True},
                {'timestamp': datetime(2023, 1, 2, 11), 'success': True}
            ]
        }
    ]
    
    pattern = learner.learn_customer_patterns(customer_data)
    
    assert pattern.pattern_type == "customer"
    assert pattern.confidence > 0.0
    assert 'customer_patterns' in pattern.data

def test_route_efficiency_learner():
    learner = RouteEfficiencyLearner()
    
    route_data = [
        {
            'route_id': 'R001',
            'traffic_index': 0.6,
            'weather_index': 0.3,
            'vehicle_utilization': 0.8,
            'time_of_day': 14,
            'efficiency_score': 0.75,
            'distance_km': 25.0,
            'actual_time_min': 45.0
        }
    ]
    
    pattern = learner.learn_route_efficiency(route_data)
    
    assert pattern.pattern_type == "route"
    assert pattern.confidence > 0.0
    assert 'best_route' in pattern.data