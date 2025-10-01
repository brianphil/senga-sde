# File: senga_core/powell_framework/cfa.py
import numpy as np
import scipy.optimize
from typing import List, Dict, Any, Callable
import logging
from datetime import datetime
from .costs import SengaCostFunction
from .exceptions import OptimizationFailedException, CostCalculationError

logger = logging.getLogger(__name__)

class TacticalCFA:
    """
    Cost Function Approximation for tactical decisions.
    Uses scipy.optimize for actual mathematical optimization.
    Quality Gate: Must use actual solver, find different solutions for different inputs.
    """
    
    def __init__(self, cost_weights: Dict[str, float] = None, max_iterations: int = 1000):
        # Initialize cost function
        self.cost_function = SengaCostFunction(cost_weights)
        self.max_iterations = max_iterations
        self.optimization_history = []
        self.constraint_validators = []
        
        logger.info("Initialized TacticalCFA with scipy.optimize backend")
    
    # In senga_core/powell_framework/cfa.py

    def optimize_tactical_decision(self, state: Any, constraints: List[Callable] = None) -> Dict[str, Any]:
        """
        Actual optimization using scipy.optimize - with bounds and timeout protection.
        Quality Gate: Must use actual optimization solver.
        """
        import signal
        from contextlib import contextmanager

        @contextmanager
        def timeout(seconds):
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Optimization timed out after {seconds} seconds")
            
            # Only works on Unix, but we'll use a fallback for Windows
            if hasattr(signal, 'SIGALRM'):
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(seconds)
                try:
                    yield
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            else:
                # Windows fallback: no SIGALRM, just hope it doesn't hang
                yield

        try:
            # Initialize decision variables
            x0 = self.initialize_decision_variables(state)
            
            # Define bounds for each variable (prevent infinite exploration)
            # Assume all variables should be between -100 and 100 for stability
            bounds = [(-100.0, 100.0) for _ in range(len(x0))]

            # Define objective function
            def objective_function(x):
                action = self.decode_action_vector(x, state)
                try:
                    cost = self.cost_function.multi_objective_cost(state, action)
                    # Penalize NaN or inf
                    if not np.isfinite(cost):
                        return 1e10
                    return cost
                except Exception as e:
                    logger.warning(f"Cost calculation failed in optimization: {e}")
                    return 1e10  # Return high cost for failed calculations

            # Define constraint functions
            constraint_funcs = []
            if constraints:
                for i, constraint_func in enumerate(constraints):
                    def create_constraint_func(cf, idx):
                        def constraint(x):
                            action = self.decode_action_vector(x, state)
                            try:
                                val = cf(state, action)
                                # Ensure constraint returns finite value
                                return val if np.isfinite(val) else -1.0
                            except Exception as e:
                                logger.warning(f"Constraint {idx} evaluation failed: {e}")
                                return -1.0  # Violate constraint on error
                        return constraint
                    
                    constraint_funcs.append(create_constraint_func(constraint_func, i))

            # Set up constraints for scipy.optimize
            scipy_constraints = []
            for cf in constraint_funcs:
                scipy_constraints.append({
                    'type': 'ineq',  # inequality constraint: g(x) >= 0
                    'fun': cf
                })

            # Perform optimization WITH TIMEOUT and STRICT BOUNDS
            logger.debug(f"Starting optimization with {len(x0)} variables")
            
            try:
                with timeout(30):  # 30 second timeout
                    result = scipy.optimize.minimize(
                        objective_function,
                        x0,
                        method='SLSQP',
                        bounds=bounds,  # <<< CRITICAL: Add bounds
                        constraints=scipy_constraints,
                        options={
                            'ftol': 1e-4,    # Relax tolerance slightly for faster convergence
                            'maxiter': 100,  # <<< CRITICAL: Limit iterations
                            'disp': False,
                            'eps': 1e-3      # Step size for numerical gradients
                        }
                    )
            except TimeoutError as te:
                error_msg = f"Tactical optimization timed out: {te}"
                logger.error(error_msg)
                raise OptimizationFailedException(error_msg)

            # Log optimization result
            optimization_record = {
                'success': result.success,
                'message': result.message,
                'fun': result.fun,
                'nfev': result.nfev,
                'nit': result.nit,
                'timestamp': datetime.now()
            }

            self.optimization_history.append(optimization_record)

            # Check if optimization succeeded
            if not result.success or not np.isfinite(result.fun):
                error_msg = f"Tactical optimization failed: {result.message} (fun={result.fun})"
                logger.error(error_msg)
                raise OptimizationFailedException(error_msg)

            # Decode and return optimal action
            optimal_action = self.decode_action_vector(result.x, state)
            optimal_action['optimization_metadata'] = {
                'cost': float(result.fun),
                'iterations': result.nit,
                'function_evaluations': result.nfev,
                'success': True,
                'x_final': result.x.tolist()  # For debugging
            }

            logger.info(f"Optimization successful: cost={result.fun:.2f}, iterations={result.nit}")
            return optimal_action

        except OptimizationFailedException:
            raise
        except Exception as e:
            error_msg = f"Optimization failed unexpectedly: {str(e)}"
            logger.error(error_msg)
            raise OptimizationFailedException(error_msg)
    
    def initialize_decision_variables(self, state: Any) -> np.ndarray:
        """
        Initialize decision variables for optimization.
        This is a simple initialization - in practice, you might use heuristics.
        """
        # For demonstration, we'll create a simple vector of decision variables
        # In a real implementation, this would be more sophisticated
        
        num_variables = 10  # Simple fixed size for demo
        
        # Initialize with reasonable values based on state
        x0 = np.zeros(num_variables)
        
        # If state has routes, initialize based on route count
        if hasattr(state, 'routes') and len(state.routes) > 0:
            x0[0] = len(state.routes) * 10.0  # Scale route count
            x0[1] = np.mean([r.distance_km for r in state.routes]) if state.routes else 50.0
        
        # If state has fleet, initialize based on fleet size
        if hasattr(state, 'fleet') and len(state.fleet) > 0:
            x0[2] = len(state.fleet) * 5.0
            x0[3] = np.mean([v.fuel_level for v in state.fleet]) * 100.0
        
        # If state has customers, initialize based on customer count
        if hasattr(state, 'customers') and len(state.customers) > 0:
            x0[4] = len(state.customers) * 2.0
            x0[5] = np.mean([c.demand_kg for c in state.customers]) / 10.0
        
        # Add some randomness to avoid local minima
        x0 += np.random.randn(len(x0)) * 0.1
        
        return x0
    
    def decode_action_vector(self, x: np.ndarray, state: Any) -> Dict[str, Any]:
        """
        Decode optimization vector into actionable decision.
        This is a simplified version - in practice, this would be more sophisticated.
        """
        action = {
            'type': 'route_optimization',
            'route_plan': [],
            'schedule': {},
            'driver_assignments': [],
            'time_windows': {},
            'vehicle_usage': {},
            'delivery_sequence': []
        }
        
        # Decode route plan (simplified)
        if len(x) > 0:
            num_routes = max(1, int(abs(x[0]) / 10.0))
            for i in range(num_routes):
                route = {
                    'route_id': f"R{i+1:02d}",
                    'distance_km': abs(x[1]) if len(x) > 1 else 50.0,
                    'estimated_time_min': abs(x[1]) * 2.0 if len(x) > 1 else 60.0,
                    'waypoints': [(-1.2864, 36.8172)]  # Nairobi center
                }
                action['route_plan'].append(route)
        
        # Decode schedule
        action['schedule'] = {
            'total_hours': abs(x[2]) if len(x) > 2 else 8.0,
            'start_time': 8,  # 8 AM
            'end_time': 17    # 5 PM
        }
        
        # Decode driver assignments
        if len(x) > 3:
            num_drivers = max(1, int(abs(x[2]) / 5.0))
            for i in range(num_drivers):
                driver = {
                    'driver_id': f"D{i+1:02d}",
                    'hours': abs(x[3]) / 10.0 if len(x) > 3 else 8.0,
                    'routes_assigned': [f"R{(i%num_routes)+1:02d}"]
                }
                action['driver_assignments'].append(driver)
        
        # Decode time windows for customers
        if hasattr(state, 'customers') and state.customers:
            for i, customer in enumerate(state.customers[:5]):  # First 5 customers
                if len(x) > 4 + i:
                    start_hour = max(6, min(22, int(8 + (x[4 + i] % 8))))
                    action['time_windows'][customer.customer_id] = {
                        'start_hour': start_hour,
                        'end_hour': start_hour + 2,
                        'confidence': 0.8
                    }
                else:
                    action['time_windows'][customer.customer_id] = {
                        'start_hour': 10,
                        'end_hour': 12,
                        'confidence': 0.5
                    }
        
        # Decode vehicle usage
        action['vehicle_usage'] = {
            'total_distance_km': sum(r['distance_km'] for r in action['route_plan']),
            'max_distance_km': max(r['distance_km'] for r in action['route_plan']) if action['route_plan'] else 0.0
        }
        
        # Decode delivery sequence with address confidence
        for i, customer_id in enumerate(action['time_windows'].keys()):
            if len(x) > 9 + i:
                confidence = max(0.1, min(0.9, 0.5 + (x[9 + i] % 0.4)))
            else:
                confidence = 0.7
            
            action['delivery_sequence'].append({
                'customer_id': customer_id,
                'sequence_position': i + 1,
                'address_confidence': confidence,
                'estimated_delivery_time': action['time_windows'][customer_id]['start_hour']
            })
        
        return action
    
    def add_constraint_validator(self, validator_func: Callable):
        """Add a constraint validator function"""
        self.constraint_validators.append(validator_func)
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get history of optimization attempts"""
        return self.optimization_history.copy()
    
    def get_recent_success_rate(self, window_size: int = 10) -> float:
        """Get success rate of recent optimizations"""
        if len(self.optimization_history) == 0:
            return 1.0
        
        recent_history = self.optimization_history[-window_size:]
        successes = sum(1 for record in recent_history if record['success'])
        return successes / len(recent_history)