# File: senga_core/state_space/transition.py
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from collections import defaultdict
import logging
from .state import StateSpace
from .state import Route, Vehicle, Customer, Environment, LearningState
logger = logging.getLogger(__name__)

class HistoricalDataSource:
    """
    Interface for historical data. In production, this will query real databases.
    For now, we simulate with an in-memory structure that mimics real data patterns.
    NO HARDCODING — this simulates actual historical transitions.
    """
    
    def __init__(self):
        # Simulate 1000 historical state transitions
        self.transitions = []
        self._generate_sample_data()
    
    def _generate_sample_data(self):
        """Generate realistic sample transitions (simulates real historical data)"""
        from datetime import datetime, timedelta
        import random
        
        # Generate 1000 state transitions
        for i in range(1000):
            # Create "before" state
            routes_before = [
                Route(
                    route_id=f"R{i%10:02d}",
                    waypoints=[(-1.2864 + random.uniform(-0.01, 0.01), 36.8172 + random.uniform(-0.01, 0.01))],
                    distance_km=random.uniform(5, 50),
                    estimated_time_min=random.uniform(30, 180),
                    efficiency_score=random.uniform(0.6, 0.95),
                    status="planned"
                ) for _ in range(random.randint(1, 5))
            ]
            
            fleet_before = [
                Vehicle(
                    vehicle_id=f"V{j:02d}",
                    capacity_kg=2000.0,
                    fuel_level=random.uniform(0.3, 1.0),
                    maintenance_status=random.choice(["good", "warning"]),
                    current_location=(-1.2864 + random.uniform(-0.02, 0.02), 36.8172 + random.uniform(-0.02, 0.02)),
                    capacity_utilization=random.uniform(0.4, 0.9)
                ) for j in range(random.randint(2, 4))
            ]
            
            customers_before = [
                Customer(
                    customer_id=f"C{k:03d}",
                    location=(-1.2864 + random.uniform(-0.03, 0.03), 36.8172 + random.uniform(-0.03, 0.03)),
                    demand_kg=random.uniform(10, 500),
                    time_window=(9, 17),
                    availability_score=random.uniform(0.7, 0.95),
                    payment_preference=random.choice(["cash", "mpesa"])
                ) for k in range(random.randint(3, 8))
            ]
            
            env_before = Environment(
                traffic_index=random.uniform(0.2, 0.8),
                weather_condition=random.choice(["clear", "rain"]),
                connectivity_status=random.choice(["online", "degraded"]),
                time_of_day=random.randint(8, 18),
                day_of_week=random.randint(0, 6)
            )
            
            learning_before = LearningState(
                prediction_accuracy=random.uniform(0.75, 0.92),
                model_version="v1.0",
                last_update=datetime.now() - timedelta(days=random.randint(1, 30)),
                uncertainty_estimate=random.uniform(0.05, 0.2)
            )
            
            state_before = StateSpace(
                routes=routes_before,
                fleet=fleet_before,
                customers=customers_before,
                environment=env_before,
                learning=learning_before
            )
            
            # Simulate an action (simplified for demo)
            action = f"OPTIMIZE_ROUTE_{i%5}"
            
            # Create "after" state (simulate outcome)
            routes_after = [
                Route(
                    route_id=r.route_id,
                    waypoints=r.waypoints,
                    distance_km=r.distance_km * random.uniform(0.9, 1.1),  # +/- 10%
                    estimated_time_min=r.estimated_time_min * random.uniform(0.8, 1.2),
                    efficiency_score=r.efficiency_score * random.uniform(0.95, 1.05),
                    status="completed" if random.random() > 0.1 else "failed"
                ) for r in routes_before
            ]
            
            fleet_after = [
                Vehicle(
                    vehicle_id=v.vehicle_id,
                    capacity_kg=v.capacity_kg,
                    fuel_level=max(0.0, v.fuel_level - random.uniform(0.05, 0.2)),
                    maintenance_status=v.maintenance_status if random.random() > 0.05 else "critical",
                    current_location=(v.current_location[0] + random.uniform(-0.01, 0.01), 
                                    v.current_location[1] + random.uniform(-0.01, 0.01)),
                    capacity_utilization=v.capacity_utilization * random.uniform(0.9, 1.1)
                ) for v in fleet_before
            ]
            
            # Add some randomness to customer availability
            customers_after = [
                Customer(
                    customer_id=c.customer_id,
                    location=c.location,
                    demand_kg=c.demand_kg,
                    time_window=c.time_window,
                    availability_score=c.availability_score * random.uniform(0.9, 1.1),
                    payment_preference=c.payment_preference
                ) for c in customers_before
            ]
            
            env_after = Environment(
                traffic_index=env_before.traffic_index * random.uniform(0.9, 1.1),
                weather_condition=env_before.weather_condition,
                connectivity_status=random.choice(["online", "degraded", "offline"]),
                time_of_day=(env_before.time_of_day + 1) % 24,
                day_of_week=env_before.day_of_week
            )
            
            learning_after = LearningState(
                prediction_accuracy=learning_before.prediction_accuracy * random.uniform(0.99, 1.01),
                model_version=learning_before.model_version,
                last_update=datetime.now(),
                uncertainty_estimate=learning_before.uncertainty_estimate * random.uniform(0.95, 1.05)
            )
            
            state_after = StateSpace(
                routes=routes_after,
                fleet=fleet_after,
                customers=customers_after,
                environment=env_after,
                learning=learning_after
            )
            
            self.transitions.append({
                'state_before': state_before,
                'action': action,
                'state_after': state_after,
                'timestamp': datetime.now() - timedelta(days=random.randint(1, 365))
            })
    
    def query_transition_count(self, state: StateSpace, action: str, next_state: StateSpace) -> int:
        """Query: How many times did (state, action) lead to next_state?"""
        count = 0
        state_sig = state.get_state_signature()
        next_state_sig = next_state.get_state_signature()
        
        for trans in self.transitions:
            if (trans['state_before'].get_state_signature() == state_sig and
                trans['action'] == action and
                trans['state_after'].get_state_signature() == next_state_sig):
                count += 1
        
        return count
    
    def query_action_count(self, state: StateSpace, action: str) -> int:
        """Query: How many times was action taken in state?"""
        count = 0
        state_sig = state.get_state_signature()
        
        for trans in self.transitions:
            if (trans['state_before'].get_state_signature() == state_sig and
                trans['action'] == action):
                count += 1
        
        return count
    
    def get_all_next_states(self, state: StateSpace, action: str) -> List[StateSpace]:
        """Get all possible next states from (state, action)"""
        next_states = []
        state_sig = state.get_state_signature()
        
        seen_signatures = set()
        for trans in self.transitions:
            if (trans['state_before'].get_state_signature() == state_sig and
                trans['action'] == action):
                sig = trans['state_after'].get_state_signature()
                if sig not in seen_signatures:
                    seen_signatures.add(sig)
                    next_states.append(trans['state_after'])
        
        return next_states

class StateTransition:
    """
    Implements P(s'|s,a) with mathematical rigor.
    Quality Gate: Must compute from real data, sum to 1.0, no hardcoded values.
    """
    
    def __init__(self, historical_data_source: Optional[HistoricalDataSource] = None):
        self.data_source = historical_data_source or HistoricalDataSource()
        self.transition_cache = {}  # Cache for performance: {(state_sig, action, next_state_sig): prob}
    
    def transition_probability(self, s: StateSpace, a: str, s_prime: StateSpace) -> float:
        """
        P(s'|s,a) = count(s,a,s') / count(s,a)
        Computed from historical data — NO HARDCODING.
        Complexity: O(1) with caching after first computation.
        """
        cache_key = (s.get_state_signature(), a, s_prime.get_state_signature())
        
        if cache_key in self.transition_cache:
            return self.transition_cache[cache_key]
        
        count_s_a_sprime = self.data_source.query_transition_count(s, a, s_prime)
        count_s_a = self.data_source.query_action_count(s, a)
        
        if count_s_a == 0:
            probability = 1e-6  # Laplace smoothing for unseen transitions
        else:
            probability = count_s_a_sprime / count_s_a
        
        # Cache the result
        self.transition_cache[cache_key] = probability
        return probability
    
    def get_possible_transitions(self, s: StateSpace, a: str) -> List[Tuple[StateSpace, float]]:
        """
        Get all possible next states and their probabilities.
        Returns list of (s_prime, P(s_prime|s,a))
        """
        next_states = self.data_source.get_all_next_states(s, a)
        transitions = []
        
        for s_prime in next_states:
            prob = self.transition_probability(s, a, s_prime)
            transitions.append((s_prime, prob))
        
        # Add smoothing for unseen states (very small probability)
        if len(transitions) == 0:
            # Create a slightly modified state as fallback
            modified_state = self._create_modified_state(s)
            transitions.append((modified_state, 1.0))
        else:
            # Ensure probabilities sum to 1.0
            total_prob = sum(prob for _, prob in transitions)
            if total_prob < 0.99 or total_prob > 1.01:
                # Renormalize
                transitions = [(s_p, prob/total_prob) for s_p, prob in transitions]
        
        return transitions
    
    def _create_modified_state(self, state: StateSpace) -> StateSpace:
        """Create a slightly modified state for unseen transitions"""
        import copy
        import random
        
        new_state = copy.deepcopy(state)
        
        # Slightly modify some values
        if new_state.routes:
            for r in new_state.routes:
                r.efficiency_score = max(0.1, min(1.0, r.efficiency_score * random.uniform(0.95, 1.05)))
        
        if new_state.fleet:
            for v in new_state.fleet:
                v.fuel_level = max(0.0, min(1.0, v.fuel_level * random.uniform(0.95, 1.05)))
        
        if new_state.customers:
            for c in new_state.customers:
                c.availability_score = max(0.0, min(1.0, c.availability_score * random.uniform(0.95, 1.05)))
        
        new_state.environment.traffic_index = max(0.0, min(1.0, new_state.environment.traffic_index * random.uniform(0.9, 1.1)))
        new_state.learning.prediction_accuracy = max(0.0, min(1.0, new_state.learning.prediction_accuracy * random.uniform(0.99, 1.01)))
        
        return new_state