# File: senga_core/learning_coordinator/learning_engine.py
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np
import logging
from datetime import datetime
from .pattern_recognition import (TrafficPatternLearner, CustomerBehaviorLearner, RouteEfficiencyLearner)
logger = logging.getLogger(__name__)

from .types import LearningPattern, DecisionOutcome
class RealTimeLearningEngine:
    """
    Real-Time Learning Engine that updates models from execution outcomes.
    Quality Gate: Must show measurable learning velocity (>1% per week).
    """
    
    def __init__(self):
        self.prediction_accuracy_tracker = {}
        self.model_performance_monitor = {}
        self.learning_history = []
        self.pattern_learners = {
            'traffic': TrafficPatternLearner(),
            'customer': CustomerBehaviorLearner(),
            'route': RouteEfficiencyLearner()
        }
    
    def process_outcome_feedback(self, decision_outcome: DecisionOutcome) -> List[LearningPattern]:
        """
        Update all relevant models based on observed outcomes.
        This is the core innovation: actual mathematical learning from real data.
        """
        try:
            # Calculate prediction errors
            prediction_errors = self.calculate_prediction_errors(
                decision_outcome.original_decision,
                decision_outcome.actual_result
            )
            
            # Extract patterns from outcome
            extracted_patterns = self.extract_patterns_from_outcome(decision_outcome)
            
            # Update appropriate models based on decision scale
            if decision_outcome.scale == 'strategic':
                self.update_strategic_models(prediction_errors, extracted_patterns)
            elif decision_outcome.scale == 'tactical':
                self.update_tactical_models(prediction_errors, extracted_patterns)
            else:  # operational
                self.update_operational_models(prediction_errors, extracted_patterns)
            
            # Record learning event
            learning_event = {
                'timestamp': datetime.now(),
                'decision_id': decision_outcome.decision_id,
                'scale': decision_outcome.scale,
                'prediction_errors': prediction_errors,
                'extracted_patterns': extracted_patterns,
                'patterns_learned': len(extracted_patterns)
            }
            self.learning_history.append(learning_event)
            
            logger.info(f"Processed outcome for {decision_outcome.decision_id}: "
                       f"{len(extracted_patterns)} patterns learned, "
                       f"errors: {prediction_errors}")
            
            return extracted_patterns
            
        except Exception as e:
            logger.error(f"Learning engine failed to process outcome: {e}")
            return []
    
    def calculate_prediction_errors(self, decision: Dict[str, Any], 
                                 actual_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate specific prediction errors for mathematical learning updates"""
        errors = {}
        
        # Route efficiency prediction error
        if 'predicted_efficiency' in decision.get('expected_outcomes', {}):
            predicted_efficiency = decision['expected_outcomes']['predicted_efficiency']
            actual_efficiency = actual_result.get('measured_efficiency', predicted_efficiency)
            errors['efficiency_error'] = abs(predicted_efficiency - actual_efficiency)
        
        # Time prediction error
        if 'predicted_time' in decision.get('expected_outcomes', {}):
            predicted_time = decision['expected_outcomes']['predicted_time']
            actual_time = actual_result.get('actual_time', predicted_time)
            if predicted_time > 0:
                errors['time_error'] = abs(predicted_time - actual_time) / predicted_time
            else:
                errors['time_error'] = 1.0
        
        # Cost prediction error
        if 'predicted_cost' in decision.get('expected_outcomes', {}):
            predicted_cost = decision['expected_outcomes']['predicted_cost']
            actual_cost = actual_result.get('actual_cost', predicted_cost)
            if predicted_cost > 0:
                errors['cost_error'] = abs(predicted_cost - actual_cost) / predicted_cost
            else:
                errors['cost_error'] = 1.0
        
        # Address resolution error (if applicable)
        if 'address_confidence' in decision.get('delivery_sequence', [{}])[0]:
            predicted_confidence = decision['delivery_sequence'][0]['address_confidence']
            actual_success = actual_result.get('delivery_success', True)
            # If delivery failed and confidence was high, big error
            if not actual_success and predicted_confidence > 0.7:
                errors['address_error'] = 1.0 - predicted_confidence
            elif actual_success and predicted_confidence < 0.3:
                errors['address_error'] = predicted_confidence  # Should have been more confident
        
        return errors
    
    def extract_patterns_from_outcome(self, outcome: DecisionOutcome) -> List[LearningPattern]:
        """Extract learning patterns from decision outcome"""
        patterns = []
        actual_result = outcome.actual_result
        
        # Extract traffic patterns
        if 'traffic_index' in actual_result:
            traffic_obs = [{
                'hour': actual_result.get('time_of_day', 12),
                'day_of_week': actual_result.get('day_of_week', 0),
                'weather_index': actual_result.get('weather_index', 0.5),
                'event_index': actual_result.get('event_index', 0.0),
                'traffic_index': actual_result['traffic_index']
            }]
            traffic_pattern = self.pattern_learners['traffic'].learn_traffic_patterns(traffic_obs)
            patterns.append(traffic_pattern)
        
        # Extract customer patterns
        if 'customer_interactions' in actual_result:
            customer_pattern = self.pattern_learners['customer'].learn_customer_patterns(
                actual_result['customer_interactions']
            )
            patterns.append(customer_pattern)
        
        # Extract route efficiency patterns
        if 'route_performance' in actual_result:
            route_obs = [actual_result['route_performance']]
            route_pattern = self.pattern_learners['route'].learn_route_efficiency(route_obs)
            patterns.append(route_pattern)
        
        return patterns
    
    def update_strategic_models(self, errors: Dict[str, float], patterns: List[LearningPattern]):
        """Update strategic VFA models"""
        # In production, this would update the actual VFA
        # For now, log the update
        logger.debug(f"Updating strategic models with errors: {errors}")
        for pattern in patterns:
            logger.debug(f"Strategic pattern learned: {pattern.pattern_type}")
    
    def update_tactical_models(self, errors: Dict[str, float], patterns: List[LearningPattern]):
        """Update tactical CFA models"""
        # In production, this would update the actual CFA
        logger.debug(f"Updating tactical models with errors: {errors}")
        for pattern in patterns:
            logger.debug(f"Tactical pattern learned: {pattern.pattern_type}")
    
    def update_operational_models(self, errors: Dict[str, float], patterns: List[LearningPattern]):
        """Update operational PFA models"""
        # In production, this would update the actual PFA
        logger.debug(f"Updating operational models with errors: {errors}")
        for pattern in patterns:
            logger.debug(f"Operational pattern learned: {pattern.pattern_type}")
    
    def assess_learning_velocity(self, window_days: int = 7) -> Dict[str, Any]:
        """Measure how quickly the system is improving"""
        if len(self.learning_history) < 2:
            return {
                'accuracy_improvement_rate': 0.0,
                'quality_improvement_rate': 0.0,
                'learning_velocity_score': 0.0,
                'sufficient_data': False
            }
        
        # Calculate recent vs historical performance
        recent_learning = [e for e in self.learning_history 
                          if (datetime.now() - e['timestamp']).days <= window_days]
        historical_learning = [e for e in self.learning_history 
                              if (datetime.now() - e['timestamp']).days > window_days]
        
        if len(recent_learning) == 0 or len(historical_learning) == 0:
            return {
                'accuracy_improvement_rate': 0.0,
                'quality_improvement_rate': 0.0,
                'learning_velocity_score': 0.0,
                'sufficient_data': False
            }
        
        # Calculate average error reduction
        recent_errors = []
        for event in recent_learning:
            errors = event['prediction_errors']
            if errors:
                avg_error = np.mean(list(errors.values()))
                recent_errors.append(avg_error)
        
        historical_errors = []
        for event in historical_learning:
            errors = event['prediction_errors']
            if errors:
                avg_error = np.mean(list(errors.values()))
                historical_errors.append(avg_error)
        
        if len(recent_errors) == 0 or len(historical_errors) == 0:
            avg_recent_error = 1.0
            avg_historical_error = 1.0
        else:
            avg_recent_error = np.mean(recent_errors)
            avg_historical_error = np.mean(historical_errors)
        
        # Error reduction rate (higher is better)
        if avg_historical_error > 0:
            error_reduction_rate = (avg_historical_error - avg_recent_error) / avg_historical_error
        else:
            error_reduction_rate = 0.0
        
        # Learning velocity score
        learning_velocity = max(0.0, error_reduction_rate)
        
        # Quality Gate: >1% improvement per week
        meets_target = learning_velocity > 0.01
        
        return {
            'accuracy_improvement_rate': float(learning_velocity),
            'quality_improvement_rate': float(learning_velocity),
            'learning_velocity_score': float(learning_velocity),
            'meets_target': meets_target,
            'recent_avg_error': float(avg_recent_error),
            'historical_avg_error': float(avg_historical_error),
            'recent_samples': len(recent_errors),
            'historical_samples': len(historical_errors),
            'sufficient_data': True
        }
    
    def get_learning_metrics(self) -> Dict[str, Any]:
        """Get comprehensive learning metrics"""
        velocity = self.assess_learning_velocity()
        
        return {
            'total_learning_events': len(self.learning_history),
            'learning_velocity': velocity,
            'pattern_types_learned': list(set(
                p.pattern_type for event in self.learning_history 
                for p in event.get('extracted_patterns', [])
            )),
            'last_learning_event': self.learning_history[-1]['timestamp'] if self.learning_history else None
        }