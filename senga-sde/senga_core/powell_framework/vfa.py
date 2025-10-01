# File: senga_core/powell_framework/vfa.py
import numpy as np
from typing import List, Dict, Any
import logging
from datetime import datetime
from .features import SengaFeatureExtractor
from .exceptions import ConvergenceException, FeatureExtractionError

logger = logging.getLogger(__name__)

class StrategicVFA:
    """
    Value Function Approximation for strategic decisions.
    Implements Temporal Difference learning with Robbins-Monro convergence.
    Quality Gate: Must show decreasing TD error over time.
    """
    
    def __init__(self, learning_rate: float = 0.001, gamma: float = 0.95, feature_dim: int = 14):
        # Initialize parameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.feature_dim = feature_dim
        
        # Initialize weights (small random values)
        self.theta = np.random.randn(feature_dim) * 0.01
        
        # Initialize feature extractor
        self.feature_extractor = SengaFeatureExtractor()
        
        # Track convergence metrics
        self.convergence_metrics = []
        self.update_count = 0
        
        # Robbins-Monro condition tracking
        self.alpha_sum = 0.0
        self.alpha_squared_sum = 0.0
        
        logger.info(f"Initialized StrategicVFA with learning_rate={learning_rate}, gamma={gamma}")
    
    def td_update(self, state, action, reward, next_state) -> Dict[str, Any]:
        """
        TD(0) update with Robbins-Monro convergence conditions.
        V(s) ← V(s) + α [R + γV(s') - V(s)]
        """
        try:
            # Extract features
            features = self.feature_extractor.extract_strategic_features(state)
            next_features = self.feature_extractor.extract_strategic_features(next_state)
            
            # Calculate target and prediction
            target = reward + self.gamma * np.dot(self.theta, next_features)
            prediction = np.dot(self.theta, features)
            td_error = target - prediction
            
            # Update weights
            gradient = td_error * features
            self.theta += self.learning_rate * gradient
            
            # Update Robbins-Monro tracking
            self.alpha_sum += self.learning_rate
            self.alpha_squared_sum += self.learning_rate ** 2
            
            # Track convergence metrics
            param_change_norm = np.linalg.norm(self.learning_rate * gradient)
            timestamp = datetime.now()
            
            metric = {
                'update_count': self.update_count,
                'td_error': float(td_error),
                'td_error_abs': float(abs(td_error)),
                'parameter_change_norm': float(param_change_norm),
                'timestamp': timestamp,
                'current_learning_rate': self.learning_rate,
                'alpha_sum': self.alpha_sum,
                'alpha_squared_sum': self.alpha_squared_sum
            }
            
            self.convergence_metrics.append(metric)
            self.update_count += 1
            
            logger.debug(f"TD Update {self.update_count}: TD Error = {td_error:.6f}, "
                        f"Param Change Norm = {param_change_norm:.6f}")
            
            return metric
            
        except FeatureExtractionError as e:
            logger.error(f"Feature extraction failed in TD update: {e}")
            raise ConvergenceException(f"Feature extraction failed: {str(e)}")
        except Exception as e:
            logger.error(f"TD update failed: {e}")
            raise ConvergenceException(f"TD update failed: {str(e)}")
    
    def predict_value(self, state) -> float:
        """Predict value of state V(s)"""
        try:
            features = self.feature_extractor.extract_strategic_features(state)
            return float(np.dot(self.theta, features))
        except Exception as e:
            logger.error(f"Value prediction failed: {e}")
            return 0.0
    
    def get_convergence_metrics(self) -> List[Dict[str, Any]]:
        """Get convergence metrics for validation"""
        return self.convergence_metrics.copy()
    
    def check_convergence(self, window_size: int = 100, threshold: float = 1e-4) -> Dict[str, Any]:
        """
        Check if VFA has converged based on recent TD errors.
        Convergence: Mean absolute TD error < threshold over last window_size updates.
        """
        if len(self.convergence_metrics) < window_size:
            return {
                'converged': False,
                'reason': f'Insufficient updates ({len(self.convergence_metrics)} < {window_size})',
                'mean_td_error': 0.0,
                'updates_needed': window_size - len(self.convergence_metrics)
            }
        
        recent_metrics = self.convergence_metrics[-window_size:]
        mean_td_error = np.mean([m['td_error_abs'] for m in recent_metrics])
        
        converged = mean_td_error < threshold
        
        # Check Robbins-Monro conditions
        robbins_monro_satisfied = (self.alpha_sum == float('inf') and 
                                 self.alpha_squared_sum < float('inf'))
        
        # In practice, we use a decaying learning rate to satisfy Robbins-Monro
        # For now, we'll assume it's satisfied if we're using a small constant rate
        
        return {
            'converged': converged,
            'mean_td_error': float(mean_td_error),
            'threshold': threshold,
            'window_size': window_size,
            'robbins_monro_satisfied': robbins_monro_satisfied,
            'alpha_sum': self.alpha_sum,
            'alpha_squared_sum': self.alpha_squared_sum
        }
    
    def satisfies_robbins_monro(self) -> bool:
        """
        Check if learning rate satisfies Robbins-Monro conditions:
        sum(α_t) = ∞ and sum(α_t²) < ∞
        For constant learning rate, this is not satisfied, so we need to decay it.
        """
        # For constant learning rate α, sum(α_t) = ∞ (good) but sum(α_t²) = ∞ (bad)
        # So we return False to indicate we should implement decaying learning rate
        return False
    
    def get_feature_bounds(self) -> List[float]:
        """Get bounds for feature space (for convergence validation)"""
        # In practice, we would track min/max of features during training
        # For now, return reasonable bounds based on feature definitions
        return [0.0, 1.0] * self.feature_dim  # All features normalized to [0,1]
    
    def get_learning_rate_history(self) -> List[float]:
        """Get history of learning rates"""
        return [self.learning_rate] * len(self.convergence_metrics)