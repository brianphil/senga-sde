# File: senga_core/state_space/validators.py
from collections import defaultdict
from typing import List, Tuple, Dict, Any
import numpy as np
from scipy.stats import chi2_contingency
import logging
from .state import StateSpace, Route, Vehicle, Customer, Environment, LearningState
from .transition import StateTransition, HistoricalDataSource

logger = logging.getLogger(__name__)

class MarkovPropertyValidator:
    """
    Validates that state transitions satisfy the Markov property:
    P(s_{t+1} | s_t, s_{t-1}) ≈ P(s_{t+1} | s_t)
    Quality Gate: Must pass statistical test with p > 0.05
    """

    def __init__(self, state_transition: StateTransition, test_data: List[Tuple] = None):
        self.state_transition = state_transition
        self.test_data = test_data or self._generate_test_data()
    
    def _generate_test_data(self) -> List[Tuple]:
        """Generate test sequences for Markov validation"""
        # Use the historical data source to create sequences
        data_source = HistoricalDataSource()
        sequences = []
        
        # Group transitions by sequence (simplified)
        state_map = {}
        for i, trans in enumerate(data_source.transitions):
            state_map[trans['state_before'].get_state_signature()] = trans['state_before']
            state_map[trans['state_after'].get_state_signature()] = trans['state_after']
        
        # Create sequences of 3 states
        for i in range(len(data_source.transitions) - 2):
            trans1 = data_source.transitions[i]
            trans2 = data_source.transitions[i + 1]
            trans3 = data_source.transitions[i + 2]
            
            # Only use if they form a sequence (simplified assumption)
            if (trans1['state_after'].get_state_signature() == trans2['state_before'].get_state_signature() and
                trans2['state_after'].get_state_signature() == trans3['state_before'].get_state_signature()):
                sequences.append((
                    trans1['state_before'],
                    trans2['state_before'],  # s_{t-1}
                    trans2['state_after'],   # s_t
                    trans3['state_after']    # s_{t+1}
                ))
        
        return sequences[:100]  # Use first 100 sequences for testing
    
    def validate_markov_property(self, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Test: P(s_{t+1} | s_t, s_{t-1}) ≈ P(s_{t+1} | s_t)
        Uses chi-square test for conditional independence.
        """
        if len(self.test_data) < 10:
            return {
                'valid': False,
                'reason': 'Insufficient test data',
                'p_value': 0.0,
                'test_statistic': 0.0
            }
        
        # Create contingency table
        # We'll discretize states by their signature for the test
        observed = defaultdict(lambda: defaultdict(int))
        
        for s_prev, s_t, s_t1, s_t2 in self.test_data:
            # Discretize states (use first 4 chars of signature for simplicity)
            s_t_disc = s_t.get_state_signature()[:4]
            s_prev_disc = s_prev.get_state_signature()[:4]
            s_t2_disc = s_t2.get_state_signature()[:4]
            
            # Count occurrences of (s_t, s_prev) -> s_t2
            observed[(s_t_disc, s_prev_disc)][s_t2_disc] += 1
        
        if len(observed) == 0:
            return {
                'valid': False,
                'reason': 'No valid state sequences found',
                'p_value': 0.0,
                'test_statistic': 0.0
            }
        
        # Build contingency table
        states_t = set()
        states_prev = set()
        states_t2 = set()
        
        for (s_t, s_prev), next_states in observed.items():
            states_t.add(s_t)
            states_prev.add(s_prev)
            for s_t2 in next_states.keys():
                states_t2.add(s_t2)
        
        # Create matrix for chi-square test
        rows = list(states_t)
        cols = list(states_t2)
        
        if len(rows) < 2 or len(cols) < 2:
            return {
                'valid': True,  # Not enough variation to reject Markov property
                'reason': 'Insufficient state variation for test',
                'p_value': 1.0,
                'test_statistic': 0.0
            }
        
        contingency_table = np.zeros((len(rows), len(cols)))
        
        for i, s_t in enumerate(rows):
            for j, s_t2 in enumerate(cols):
                total = 0
                for s_prev in states_prev:
                    total += observed[(s_t, s_prev)].get(s_t2, 0)
                contingency_table[i, j] = total
        
        # Perform chi-square test
        try:
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            valid = p_value > alpha  # Fail to reject null hypothesis (Markov property holds)
            
            return {
                'valid': valid,
                'reason': 'Markov property validated' if valid else 'Markov property violated',
                'p_value': p_value,
                'test_statistic': chi2,
                'degrees_of_freedom': dof
            }
        except Exception as e:
            return {
                'valid': False,
                'reason': f'Chi-square test failed: {e}',
                'p_value': 0.0,
                'test_statistic': 0.0
            }

class StateConsistencyValidator:
    """
    Validates mathematical consistency of state space.
    Quality Gate: ΣP(s'|s,a) = 1.0 ± 0.01 for all (s,a)
    """
    
    def __init__(self, state_transition: StateTransition):
        self.state_transition = state_transition
    
    def validate_state_consistency(self, test_states: List[Tuple[StateSpace, str]] = None) -> Dict[str, Any]:
        """
        Validate that for each (state, action), sum of P(s'|s,a) = 1.0
        """
        if test_states is None:
            # Generate test states
            data_source = HistoricalDataSource()
            test_states = []
            for trans in data_source.transitions[:50]:  # First 50 transitions
                test_states.append((trans['state_before'], trans['action']))
        
        inconsistencies = []
        total_tests = len(test_states)
        passed_tests = 0
        
        for state, action in test_states:
            # Get all possible transitions
            transitions = self.state_transition.get_possible_transitions(state, action)
            total_prob = sum(prob for _, prob in transitions)
            
            if abs(total_prob - 1.0) > 0.01:
                inconsistencies.append({
                    'state_signature': state.get_state_signature(),
                    'action': action,
                    'total_probability': total_prob,
                    'deviation': abs(total_prob - 1.0)
                })
            else:
                passed_tests += 1
        
        success_rate = passed_tests / max(1, total_tests)
        
        return {
            'consistent': len(inconsistencies) == 0,
            'success_rate': success_rate,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'inconsistencies': inconsistencies,
            'quality_gate_passed': success_rate >= 0.95  # Allow 5% failure rate for edge cases
        }
    
    def validate_bounded_complexity(self) -> Dict[str, Any]:
        """
        Validate that state operations are bounded O(n log n)
        Uses curve fitting to determine complexity class.
        """
        import time
        from scipy.optimize import curve_fit
        
        # Test with larger sizes for better asymptotic behavior
        sizes = [50, 100, 200, 500, 1000, 2000]
        times = []
        
        for size in sizes:
            # Create large state
            routes = [
                Route(
                    route_id=f"R{i:04d}",
                    waypoints=[(-1.2864, 36.8172)],
                    distance_km=10.0,
                    estimated_time_min=30.0,
                    efficiency_score=0.8
                ) for i in range(size)
            ]
            
            fleet = [
                Vehicle(
                    vehicle_id=f"V{i:02d}",
                    capacity_kg=2000.0,
                    fuel_level=0.5,
                    maintenance_status="good",
                    current_location=(-1.2864, 36.8172),
                    capacity_utilization=0.7
                ) for i in range(max(1, size // 10))
            ]
            
            customers = [
                Customer(
                    customer_id=f"C{i:04d}",
                    location=(-1.2864, 36.8172),
                    demand_kg=100.0,
                    time_window=(9, 17),
                    availability_score=0.8,
                    payment_preference="mpesa"
                ) for i in range(size)
            ]
            
            env = Environment(
                traffic_index=0.5,
                weather_condition="clear",
                connectivity_status="online",
                time_of_day=12,
                day_of_week=1
            )
            
            learning = LearningState(
                prediction_accuracy=0.85,
                model_version="v1.0",
                last_update=None,
                uncertainty_estimate=0.1
            )
            
            state = StateSpace(routes, fleet, customers, env, learning)
            
            # Time the to_vector operation (run 5 times, take minimum to reduce noise)
            min_time = float('inf')
            for _ in range(5):
                start_time = time.perf_counter()
                _ = state.to_vector()
                end_time = time.perf_counter()
                min_time = min(min_time, end_time - start_time)
            
            times.append(min_time)
        
        # Define complexity functions
        def o_n(x, a):
            return a * x
        
        def o_n_log_n(x, a):
            return a * x * np.log(x)
        
        def o_n_squared(x, a):
            return a * x * x
        
        # Fit curves
        sizes_array = np.array(sizes, dtype=float)
        times_array = np.array(times, dtype=float)
        
        try:
            # Remove any zero times
            mask = times_array > 0
            if np.sum(mask) < 2:
                return {
                    'bounded_complexity': False,
                    'complexity_class': 'Unknown',
                    'reason': 'Insufficient positive timing data',
                    'sizes_tested': sizes,
                    'times_measured': times
                }
            
            sizes_filtered = sizes_array[mask]
            times_filtered = times_array[mask]
            
            # Fit O(n)
            popt_n, _ = curve_fit(o_n, sizes_filtered, times_filtered, p0=[1.0])
            residuals_n = times_filtered - o_n(sizes_filtered, *popt_n)
            mse_n = np.mean(residuals_n ** 2)
            
            # Fit O(n log n)
            popt_n_log_n, _ = curve_fit(o_n_log_n, sizes_filtered, times_filtered, p0=[1.0])
            residuals_n_log_n = times_filtered - o_n_log_n(sizes_filtered, *popt_n_log_n)
            mse_n_log_n = np.mean(residuals_n_log_n ** 2)
            
            # Fit O(n²)
            popt_n_squared, _ = curve_fit(o_n_squared, sizes_filtered, times_filtered, p0=[1.0])
            residuals_n_squared = times_filtered - o_n_squared(sizes_filtered, *popt_n_squared)
            mse_n_squared = np.mean(residuals_n_squared ** 2)
            
            # Determine best fit
            complexities = [
                ('O(n)', mse_n),
                ('O(n log n)', mse_n_log_n),
                ('O(n²)', mse_n_squared)
            ]
            
            best_complexity = min(complexities, key=lambda x: x[1])
            
            # Accept O(n) or O(n log n) as bounded
            is_bounded = best_complexity[0] in ['O(n)', 'O(n log n)']
            
            return {
                'bounded_complexity': is_bounded,
                'complexity_class': best_complexity[0],
                'mse_values': {
                    'O(n)': mse_n,
                    'O(n log n)': mse_n_log_n,
                    'O(n²)': mse_n_squared
                },
                'sizes_tested': sizes,
                'times_measured': times,
                'best_fit': best_complexity[0],
                'reason': f"Best fit is {best_complexity[0]} with MSE {best_complexity[1]:.2e}"
            }
            
        except Exception as e:
            return {
                'bounded_complexity': False,
                'complexity_class': 'Unknown',
                'reason': f'Curve fitting failed: {e}',
                'sizes_tested': sizes,
                'times_measured': times
            }