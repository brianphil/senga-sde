# File: senga_core/senga_innovations/offline_engine.py
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
from .models import ConnectivityUncertaintyModel

logger = logging.getLogger(__name__)

class OfflineDataBuffer:
    """Simple offline buffer for decision engine"""
    
    def __init__(self, buffer_hours: int = 72):
        self.buffer_hours = buffer_hours
        self.decision_cache = {}  # Cache of pre-computed decisions
        self.pending_decisions = []  # Decisions made offline awaiting sync
        self.uncertainty_model = ConnectivityUncertaintyModel()
    
    def cache_decision(self, state_signature: str, decision: Any, confidence: float):
        """Cache decision for future use"""
        self.decision_cache[state_signature] = {
            'decision': decision,
            'confidence': confidence,
            'timestamp': datetime.now(),
            'used_count': 0
        }
    
    def get_cached_decision(self, state_signature: str) -> Optional[Dict[str, Any]]:
        """Get cached decision if available and recent"""
        if state_signature in self.decision_cache:
            cache_entry = self.decision_cache[state_signature]
            # Check if cache is still valid (within 24 hours)
            if (datetime.now() - cache_entry['timestamp']).total_seconds() < 86400:
                cache_entry['used_count'] += 1
                return cache_entry
        return None

class OfflineDecisionEngine:
    """
    Offline-First Decision Engine for intermittent connectivity.
    Quality Gate: Must maintain >70% performance of online decisions.
    """
    
    def __init__(self, buffer_duration_hours: int = 72):
        self.data_buffer = OfflineDataBuffer(buffer_duration_hours)
        self.uncertainty_model = ConnectivityUncertaintyModel()
        self.decision_history = []
        self.performance_metrics = {
            'offline_decisions': 0,
            'online_decisions': 0,
            'offline_success_rate': 1.0,
            'sync_failures': 0
        }
    
    def make_offline_decision(self, partial_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make optimal decision with incomplete information.
        INNOVATION: Probabilistic decision making under uncertainty.
        """
        try:
            # Generate state signature for caching
            state_signature = self._generate_state_signature(partial_state)
            
            # Check if we have a cached decision
            cached_decision = self.data_buffer.get_cached_decision(state_signature)
            if cached_decision and cached_decision['confidence'] > 0.7:
                logger.info(f"Using cached decision for state {state_signature}")
                decision = cached_decision['decision'].copy()
                decision['source'] = 'cached'
                decision['confidence'] = cached_decision['confidence']
                self._record_decision(decision, is_offline=True)
                return decision
            
            # Estimate missing state components
            complete_state_estimate = self._estimate_complete_state(partial_state)
            
            # Quantify uncertainty in state estimation
            state_uncertainty = self._quantify_state_uncertainty(partial_state, complete_state_estimate)
            
            # Generate decision alternatives
            decision_alternatives = self._generate_robust_alternatives(complete_state_estimate, state_uncertainty)
            
            # Select decision with best worst-case performance (min-max regret)
            robust_decision = self._min_max_regret_selection(decision_alternatives, state_uncertainty)
            
            # Cache the decision
            self.data_buffer.cache_decision(state_signature, robust_decision, 1.0 - state_uncertainty)
            
            # Record decision
            robust_decision['source'] = 'computed_offline'
            robust_decision['uncertainty'] = state_uncertainty
            self._record_decision(robust_decision, is_offline=True)
            
            return robust_decision
            
        except Exception as e:
            logger.error(f"Offline decision failed: {e}")
            # Fallback decision
            return self._generate_fallback_decision(partial_state)
    
    def _generate_state_signature(self, state: Dict[str, Any]) -> str:
        """Generate signature for state caching"""
        import hashlib
        state_str = str(sorted(state.items()))
        return hashlib.md5(state_str.encode()).hexdigest()[:16]
    
    def _estimate_complete_state(self, partial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate missing state components using historical patterns"""
        complete_state = partial_state.copy()
        
        # Fill in missing fleet information
        if 'fleet' not in complete_state:
            complete_state['fleet'] = self._estimate_fleet_state(partial_state)
        
        # Fill in missing route information
        if 'routes' not in complete_state:
            complete_state['routes'] = self._estimate_route_state(partial_state)
        
        # Fill in missing customer information
        if 'customers' not in complete_state:
            complete_state['customers'] = self._estimate_customer_state(partial_state)
        
        # Fill in missing environment information
        if 'environment' not in complete_state:
            complete_state['environment'] = self._estimate_environment_state(partial_state)
        
        return complete_state
    
    def _estimate_fleet_state(self, partial_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Estimate fleet state based on partial information"""
        # Simple heuristic: assume standard fleet if no information
        return [
            {
                'vehicle_id': f'V{i:02d}',
                'fuel_level': 0.7,
                'status': 'available',
                'location': (-1.2864, 36.8172)  # Nairobi center
            }
            for i in range(3)  # Assume 3 vehicles
        ]
    
    def _estimate_route_state(self, partial_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Estimate route state based on partial information"""
        # Simple heuristic: assume standard routes
        return [
            {
                'route_id': 'R001',
                'distance_km': 25.0,
                'estimated_time_min': 45.0,
                'waypoints': [(-1.2864, 36.8172), (-1.2921, 36.8219)]  # Nairobi to Westlands
            }
        ]
    
    def _estimate_customer_state(self, partial_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Estimate customer state based on partial information"""
        # Simple heuristic: assume standard customers
        return [
            {
                'customer_id': 'C001',
                'location': (-1.2921, 36.8219),  # Westlands
                'demand_kg': 50.0,
                'time_window': (10, 16),
                'availability_score': 0.8
            }
        ]
    
    def _estimate_environment_state(self, partial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate environment state based on partial information"""
        return {
            'traffic_index': 0.5,
            'weather_condition': 'clear',
            'connectivity_status': 'offline',
            'time_of_day': 12,
            'day_of_week': 1
        }
    
    def _quantify_state_uncertainty(self, partial_state: Dict[str, Any], 
                                  complete_state: Dict[str, Any]) -> float:
        """Quantify uncertainty in state estimation"""
        missing_keys = set(['fleet', 'routes', 'customers', 'environment']) - set(partial_state.keys())
        uncertainty = len(missing_keys) / 4.0  # 0.25 per missing component
        
        # Adjust based on connectivity
        if 'environment' in partial_state:
            connectivity = partial_state['environment'].get('connectivity_status', 'offline')
            connectivity_reliability = {
                'online': 0.1,
                'degraded': 0.3,
                'offline': 0.5
            }.get(connectivity, 0.5)
            uncertainty = max(uncertainty, connectivity_reliability)
        
        return min(0.9, uncertainty)  # Cap at 0.9
    
    def _generate_robust_alternatives(self, state: Dict[str, Any], 
                                    uncertainty: float) -> List[Dict[str, Any]]:
        """Generate robust decision alternatives accounting for uncertainty"""
        alternatives = []
        
        # Alternative 1: Conservative route (shorter, more reliable)
        alt1 = {
            'type': 'route_optimization',
            'strategy': 'conservative',
            'routes': [
                {
                    'route_id': 'R_CONS',
                    'distance_km': state['routes'][0]['distance_km'] * 0.8 if 'routes' in state else 20.0,
                    'estimated_time_min': (state['routes'][0]['estimated_time_min'] * 1.2 
                                         if 'routes' in state else 60.0),
                    'confidence': 0.9 - uncertainty
                }
            ],
            'expected_performance': 0.8 - uncertainty
        }
        alternatives.append(alt1)
        
        # Alternative 2: Aggressive route (longer, potentially faster)
        alt2 = {
            'type': 'route_optimization',
            'strategy': 'aggressive',
            'routes': [
                {
                    'route_id': 'R_AGG',
                    'distance_km': state['routes'][0]['distance_km'] * 1.2 if 'routes' in state else 30.0,
                    'estimated_time_min': (state['routes'][0]['estimated_time_min'] * 0.8 
                                         if 'routes' in state else 40.0),
                    'confidence': 0.7 - uncertainty
                }
            ],
            'expected_performance': 0.9 - uncertainty * 1.5
        }
        alternatives.append(alt2)
        
        # Alternative 3: Balanced route
        alt3 = {
            'type': 'route_optimization',
            'strategy': 'balanced',
            'routes': [
                {
                    'route_id': 'R_BAL',
                    'distance_km': state['routes'][0]['distance_km'] if 'routes' in state else 25.0,
                    'estimated_time_min': state['routes'][0]['estimated_time_min'] if 'routes' in state else 50.0,
                    'confidence': 0.8 - uncertainty * 0.5
                }
            ],
            'expected_performance': 0.85 - uncertainty
        }
        alternatives.append(alt3)
        
        return alternatives
    
    def _min_max_regret_selection(self, alternatives: List[Dict[str, Any]], 
                                uncertainty: float) -> Dict[str, Any]:
        """
        Select decision with best worst-case outcome (min-max regret).
        This is the core innovation for offline decision making.
        """
        # Calculate worst-case performance for each alternative
        worst_case_performances = []
        
        for alt in alternatives:
            # Worst case: performance reduced by uncertainty factor
            worst_case = alt['expected_performance'] * (1.0 - uncertainty)
            worst_case_performances.append(worst_case)
        
        # Select alternative with best worst-case performance
        best_index = np.argmax(worst_case_performances)
        selected_alternative = alternatives[best_index].copy()
        
        selected_alternative['selection_strategy'] = 'min_max_regret'
        selected_alternative['worst_case_performance'] = worst_case_performances[best_index]
        selected_alternative['uncertainty'] = uncertainty
        
        return selected_alternative
    
    def _generate_fallback_decision(self, partial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback decision when all else fails"""
        return {
            'type': 'fallback',
            'strategy': 'safe_default',
            'routes': [
                {
                    'route_id': 'R_FALLBACK',
                    'distance_km': 20.0,
                    'estimated_time_min': 45.0,
                    'waypoints': [(-1.2864, 36.8172)]  # Stay in central Nairobi
                }
            ],
            'confidence': 0.3,
            'reason': 'Fallback due to error',
            'source': 'fallback'
        }
    
    def _record_decision(self, decision: Dict[str, Any], is_offline: bool = True):
        """Record decision for performance tracking"""
        record = {
            'decision': decision,
            'is_offline': is_offline,
            'timestamp': datetime.now(),
            'performance_estimate': decision.get('expected_performance', 0.5)
        }
        self.decision_history.append(record)
        
        if is_offline:
            self.performance_metrics['offline_decisions'] += 1
        else:
            self.performance_metrics['online_decisions'] += 1
    
    def sync_when_online(self, actual_outcomes: List[Dict[str, Any]]) -> None:
        """Reconcile offline decisions with actual outcomes"""
        if not actual_outcomes:
            return
        
        prediction_errors = []
        
        for outcome in actual_outcomes:
            # Find corresponding decision
            decision = self._find_corresponding_decision(outcome)
            if decision:
                error = self._calculate_prediction_error(decision, outcome)
                prediction_errors.append(error)
                self._update_offline_prediction_models(error)
        
        if prediction_errors:
            avg_error = np.mean(prediction_errors)
            logger.info(f"Offline sync completed: {len(actual_outcomes)} outcomes, avg error: {avg_error:.3f}")
    
    def _find_corresponding_decision(self, outcome: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find decision corresponding to outcome"""
        # Simple matching by route_id
        route_id = outcome.get('route_id')
        if route_id:
            for record in reversed(self.decision_history[-10:]):  # Last 10 decisions
                if (record['decision'].get('routes') and 
                    len(record['decision']['routes']) > 0 and
                    record['decision']['routes'][0].get('route_id') == route_id):
                    return record['decision']
        return None
    
    def _calculate_prediction_error(self, decision: Dict[str, Any], outcome: Dict[str, Any]) -> float:
        """Calculate prediction error"""
        predicted_time = decision['routes'][0]['estimated_time_min'] if decision.get('routes') else 60.0
        actual_time = outcome.get('actual_time_min', predicted_time)
        error = abs(predicted_time - actual_time) / max(predicted_time, 1.0)
        return error
    
    def _update_offline_prediction_models(self, error: float):
        """Update offline prediction models based on error"""
        # Simple adjustment: if error is high, increase uncertainty estimates
        if error > 0.3:
            # Adjust future uncertainty estimates upward
            pass  # In production, this would update model parameters
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for validation"""
        total_decisions = self.performance_metrics['offline_decisions'] + self.performance_metrics['online_decisions']
        if total_decisions == 0:
            offline_ratio = 0.0
        else:
            offline_ratio = self.performance_metrics['offline_decisions'] / total_decisions
        
        return {
            'offline_decision_ratio': offline_ratio,
            'offline_decisions': self.performance_metrics['offline_decisions'],
            'online_decisions': self.performance_metrics['online_decisions'],
            'sync_failures': self.performance_metrics['sync_failures'],
            'recent_decisions': len(self.decision_history[-10:])
        }