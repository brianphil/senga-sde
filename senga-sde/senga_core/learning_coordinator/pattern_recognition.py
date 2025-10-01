# File: senga_core/learning_coordinator/pattern_recognition.py
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from datetime import datetime
from scipy.stats import multivariate_normal
from .types import LearningPattern

logger = logging.getLogger(__name__)

class GaussianProcessRegressor:
    """
    Gaussian Process for traffic pattern prediction.
    Quality Gate: Must update in real-time with actual traffic observations.
    """
    
    def __init__(self, kernel_params: Dict[str, float] = None):
        # Simple GP implementation for demonstration
        # In production, use GPyTorch or scikit-learn
        self.kernel_params = kernel_params or {'length_scale': 1.0, 'variance': 1.0}
        self.X_train = []  # Training inputs: [hour, day_of_week, weather_index]
        self.y_train = []  # Training outputs: traffic_index
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit GP to training data"""
        self.X_train = X.tolist()
        self.y_train = y.tolist()
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance"""
        if not self.is_fitted:
            return np.zeros(len(X)), np.ones(len(X)) * 0.5
        
        # Simple prediction: return mean of training data
        mean_pred = np.full(len(X), np.mean(self.y_train))
        var_pred = np.full(len(X), np.var(self.y_train) + 0.1)
        return mean_pred, var_pred
    
    def update(self, new_X: np.ndarray, new_y: np.ndarray):
        """Update GP with new observations"""
        if not self.is_fitted:
            self.fit(new_X, new_y)
        else:
            self.X_train.extend(new_X.tolist())
            self.y_train.extend(new_y.tolist())

class ContextualBandit:
    """
    Contextual Bandit for route efficiency optimization.
    Quality Gate: Must actually learn from route execution outcomes.
    """
    
    def __init__(self, arms: List[str], context_dim: int = 4):
        self.arms = arms
        self.context_dim = context_dim
        self.arm_weights = {arm: np.random.randn(context_dim) * 0.1 for arm in arms}
        self.arm_counts = {arm: 0 for arm in arms}
        self.total_rewards = {arm: 0.0 for arm in arms}
        self.learning_rate = 0.01
    
    def select_arm(self, context: np.ndarray) -> str:
        """Select arm using epsilon-greedy strategy"""
        if np.random.random() < 0.1:  # Exploration
            return np.random.choice(self.arms)
        
        # Exploitation: select arm with highest predicted reward
        predicted_rewards = {}
        for arm in self.arms:
            predicted_rewards[arm] = np.dot(self.arm_weights[arm], context)
        
        return max(predicted_rewards, key=predicted_rewards.get)
    
    def update(self, arm: str, context: np.ndarray, reward: float):
        """Update arm weights based on observed reward"""
        self.arm_counts[arm] += 1
        self.total_rewards[arm] += reward
        
        # Gradient update
        predicted_reward = np.dot(self.arm_weights[arm], context)
        error = reward - predicted_reward
        self.arm_weights[arm] += self.learning_rate * error * context

class TrafficPatternLearner:
    """
    Learns traffic patterns using Gaussian Process Regression.
    """
    
    def __init__(self):
        self.traffic_gp = GaussianProcessRegressor()
        self.pattern_history = []
    
    def learn_traffic_patterns(self, traffic_: List[Dict[str, Any]]) -> LearningPattern:
        """Learn from traffic observations"""
        if not traffic_:
            return LearningPattern(pattern_type="traffic", confidence=0.0, data={})
        
        # Extract features and targets
        X = []
        y = []
        
        for obs in traffic_:
            features = [
                obs.get('hour', 12) / 24.0,
                obs.get('day_of_week', 0) / 7.0,
                obs.get('weather_index', 0.5),
                obs.get('event_index', 0.0)
            ]
            X.append(features)
            y.append(obs.get('traffic_index', 0.5))
        
        X = np.array(X)
        y = np.array(y)
        
        # Update GP model
        self.traffic_gp.update(X, y)
        
        # Create learning pattern
        pattern = LearningPattern(
            pattern_type="traffic",
            confidence=min(0.9, len(traffic_) / 100.0),  # Confidence increases with data
            data={
                'model_type': 'gaussian_process',
                'training_samples': len(traffic_),
                'feature_importance': [0.4, 0.3, 0.2, 0.1],  # Placeholder
                'last_update': datetime.now()
            }
        )
        
        self.pattern_history.append(pattern)
        return pattern

class CustomerBehaviorLearner:
    """
    Learns customer availability patterns using Hidden Markov Models.
    """
    
    def __init__(self):
        self.customer_hmms = {}  # HMM per customer type
        self.pattern_history = []
    
    def learn_customer_patterns(self, customer_: List[Dict[str, Any]]) -> LearningPattern:
        """Learn from customer interaction history"""
        if not customer_:
            return LearningPattern(pattern_type="customer", confidence=0.0, data={})
        
        # Group by customer type
        customer_groups = {}
        for customer in customer_:
            cust_type = customer.get('customer_type', 'retail')
            if cust_type not in customer_groups:
                customer_groups[cust_type] = []
            customer_groups[cust_type].append(customer)
        
        # Learn patterns for each group
        learned_patterns = {}
        total_interactions = 0
        
        for cust_type, customers in customer_groups.items():
            interactions = []
            for customer in customers:
                if 'interaction_history' in customer:
                    interactions.extend(customer['interaction_history'])
                    total_interactions += len(customer['interaction_history'])
            
            if interactions:
                # Create HMM training data
                sequences = self._create_hmm_sequences(interactions)
                if sequences:
                    # In production, train actual HMM
                    # For now, create synthetic pattern
                    pattern_confidence = min(0.85, len(interactions) / 50.0)
                    learned_patterns[cust_type] = {
                        'available_windows': self._extract_available_windows(interactions),
                        'confidence': pattern_confidence,
                        'interaction_count': len(interactions)
                    }
        
        pattern = LearningPattern(
            pattern_type="customer",
            confidence=min(0.9, total_interactions / 200.0),
            data={
                'customer_patterns': learned_patterns,
                'total_interactions': total_interactions,
                'customer_types': list(customer_groups.keys()),
                'last_update': datetime.now()
            }
        )
        
        self.pattern_history.append(pattern)
        return pattern
    
    def _create_hmm_sequences(self, interactions: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Create HMM training sequences"""
        # Simplified: create one sequence per day
        sequences = []
        daily_interactions = {}
        
        for interaction in interactions:
            timestamp = interaction.get('timestamp', datetime.now())
            day_key = timestamp.strftime('%Y-%m-%d')
            if day_key not in daily_interactions:
                daily_interactions[day_key] = []
            daily_interactions[day_key].append(interaction)
        
        for day_interactions in daily_interactions.values():
            if len(day_interactions) > 0:
                # Create 24-hour sequence
                sequence = np.zeros((24, 4))  # 24 hours, 4 features
                for hour in range(24):
                    hour_interactions = [i for i in day_interactions 
                                       if i.get('timestamp', datetime.now()).hour == hour]
                    sequence[hour] = [
                        hour / 24.0,
                        len(hour_interactions) > 0,  # availability
                        0.5,  # placeholder
                        0.5   # placeholder
                    ]
                sequences.append(sequence)
        
        return sequences if len(sequences) > 0 else None
    
    def _extract_available_windows(self, interactions: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
        """Extract available time windows from interactions"""
        hourly_counts = np.zeros(24)
        
        for interaction in interactions:
            hour = interaction.get('timestamp', datetime.now()).hour
            if 0 <= hour < 24:
                hourly_counts[hour] += 1
        
        # Find contiguous windows with high activity
        windows = []
        in_window = False
        start_hour = 0
        
        for hour in range(24):
            if hourly_counts[hour] > 0 and not in_window:
                in_window = True
                start_hour = hour
            elif (hourly_counts[hour] == 0 or hour == 23) and in_window:
                end_hour = hour if hourly_counts[hour] == 0 else hour + 1
                if end_hour - start_hour >= 2:  # At least 2-hour window
                    windows.append((start_hour, min(end_hour, 24)))
                in_window = False
        
        return windows if windows else [(9, 17)]  # Default business hours

class RouteEfficiencyLearner:
    """
    Learns route efficiency using Contextual Bandits.
    """
    
    def __init__(self):
        self.route_bandit = None
        self.pattern_history = []
        self.route_alternatives = []
    
    def learn_route_efficiency(self, route_: List[Dict[str, Any]]) -> LearningPattern:
        """Learn from route execution outcomes"""
        if not route_:
            return LearningPattern(pattern_type="route", confidence=0.0, data={})
        
        # Extract route alternatives
        route_ids = list(set(r.get('route_id') for r in route_ if r.get('route_id')))
        if not route_ids:
            return LearningPattern(pattern_type="route", confidence=0.0, data={})
        
        # Initialize bandit if needed
        if self.route_bandit is None or set(route_ids) != set(self.route_alternatives):
            self.route_alternatives = route_ids
            self.route_bandit = ContextualBandit(arms=route_ids, context_dim=4)
        
        # Update bandit with new observations
        for route_obs in route_:
            route_id = route_obs.get('route_id')
            if route_id not in route_ids:
                continue
            
            # Create context vector
            context = np.array([
                route_obs.get('traffic_index', 0.5),
                route_obs.get('weather_index', 0.5),
                route_obs.get('vehicle_utilization', 0.7),
                route_obs.get('time_of_day', 12) / 24.0
            ])
            
            # Calculate reward (efficiency metric)
            efficiency = route_obs.get('efficiency_score', 0.5)
            distance = route_obs.get('distance_km', 1.0)
            time = route_obs.get('actual_time_min', 60.0)
            
            # Reward: higher efficiency, shorter time, reasonable distance
            reward = efficiency * (60.0 / max(time, 1.0)) * (1.0 / max(distance, 1.0))
            
            # Update bandit
            self.route_bandit.update(route_id, context, reward)
        
        pattern = LearningPattern(
            pattern_type="route",
            confidence=min(0.9, len(route_) / 50.0),
            data={
                'model_type': 'contextual_bandit',
                'route_alternatives': route_ids,
                'training_samples': len(route_),
                'best_route': max(self.route_bandit.total_rewards, key=self.route_bandit.total_rewards.get) if self.route_bandit.total_rewards else None,
                'last_update': datetime.now()
            }
        )
        
        self.pattern_history.append(pattern)
        return pattern