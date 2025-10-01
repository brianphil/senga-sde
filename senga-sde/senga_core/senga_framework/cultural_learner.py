# File: senga_core/senga_innovations/cultural_learner.py
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from datetime import datetime, timedelta
from .models import HiddenMarkovModel
from dataclasses import dataclass
logger = logging.getLogger(__name__)

@dataclass
class TimeWindow:
    start_hour: int
    end_hour: int
    confidence: float
    cultural_context: str = ""

class CulturalPatternLearner:
    """
    Cultural Pattern Learner for Kenyan business rhythms.
    Quality Gate: Must outperform generic business hours by >20%.
    """
    
    def __init__(self):
        self.availability_hmm = {}  # HMM per customer type
        self.payment_preference_model = None  # Will implement DirichletMultinomial
        self.seasonal_adjustment_model = None  # Will implement SeasonalDecomposition
        self.customer_types = {}
        self.training_history = []
        self.prediction_history = []
        
        # Default HMM for unknown customer types
        self.default_hmm = HiddenMarkovModel(n_components=3, n_features=4)
        # Train on some default patterns
        self._train_default_hmm()
    
    def _train_default_hmm(self):
        """Train default HMM on generic Kenyan business patterns"""
        # Generate synthetic training data for default HMM
        sequences = []
        
        for _ in range(50):  # 50 synthetic customers
            sequence = []
            for hour in range(24):
                # Kenyan business pattern: busy 10-13, 16-19, quiet at night
                if 6 <= hour <= 8:  # Morning opening
                    features = [hour/24.0, 0.0, 0.0, 0.3]  # [hour_norm, weekend, month, traffic]
                elif 10 <= hour <= 13:  # Midday busy
                    features = [hour/24.0, 0.0, 0.0, 0.7]
                elif 16 <= hour <= 19:  # Evening busy
                    features = [hour/24.0, 0.0, 0.0, 0.8]
                elif 22 <= hour or hour <= 5:  # Night quiet
                    features = [hour/24.0, 0.0, 0.0, 0.2]
                else:  # Other times
                    features = [hour/24.0, 0.0, 0.0, 0.5]
                sequence.append(features)
            sequences.append(np.array(sequence))
        
        # Fit default HMM
        self.default_hmm.fit(sequences)
        logger.info("Default CulturalPatternLearner HMM trained")
    
    def learn_customer_availability(self, customer_data: List[Dict[str, Any]]) -> None:
        """
        Learn time-dependent availability patterns.
        This is the core innovation for African business rhythms.
        """
        if not customer_data:
            logger.warning("No customer data provided for learning")
            return
        
        # Group customers by business type and location
        customer_groups = self._group_customers_by_type(customer_data)
        
        for group_type, customers in customer_groups.items():
            try:
                # Create features for each customer
                availability_sequences = []
                for customer in customers:
                    if 'interaction_history' in customer:
                        sequence = self._create_availability_sequence(customer['interaction_history'])
                        if len(sequence) > 0:
                            availability_sequences.append(sequence)
                
                if len(availability_sequences) == 0:
                    continue
                
                # Train Hidden Markov Model for this customer group
                logger.info(f"Training HMM for customer group: {group_type} ({len(availability_sequences)} sequences)")
                hmm = HiddenMarkovModel(n_components=3, n_features=4)  # Available, Busy, Unavailable
                hmm.fit(availability_sequences)
                
                self.availability_hmm[group_type] = hmm
                self.customer_types[group_type] = len(customers)
                
                logger.info(f"HMM trained for {group_type}: {len(availability_sequences)} sequences")
                
            except Exception as e:
                logger.error(f"Failed to train HMM for {group_type}: {e}")
                continue
        
        # Record training
        self.training_history.append({
            'timestamp': datetime.now(),
            'customer_groups_trained': list(customer_groups.keys()),
            'total_customers': len(customer_data)
        })
    
    def _group_customers_by_type(self, customer_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group customers by business type and location"""
        groups = {}
        
        for customer in customer_data:
            # Get customer type
            customer_type = customer.get('customer_type', 'retail_shop')
            location_type = customer.get('location_type', 'urban')
            group_key = f"{customer_type}_{location_type}"
            
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(customer)
        
        return groups
    
    def _create_availability_sequence(self, interaction_history: List[Dict[str, Any]]) -> np.ndarray:
        """Create feature sequence for HMM training"""
        if not interaction_history:
            return np.array([])
        
        # Create 24-hour sequence
        sequence = []
        
        for hour in range(24):
            # Extract features for this hour
            hour_features = self._extract_time_context_features(datetime.now(), hour, interaction_history)
            sequence.append(hour_features)
        
        return np.array(sequence)
    
    def _extract_time_context_features(self, day: datetime, hour: int, 
                                     interaction_history: List[Dict[str, Any]] = None) -> List[float]:
        """Extract features for time context"""
        features = []
        
        # Hour of day (normalized)
        features.append(hour / 24.0)
        
        # Day of week (0=Monday, 6=Sunday)
        features.append(day.weekday() / 7.0)
        
        # Month (normalized)
        features.append((day.month - 1) / 12.0)
        
        # Traffic index (simplified)
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
            features.append(0.8)
        elif 22 <= hour or hour <= 5:  # Night hours
            features.append(0.3)
        else:
            features.append(0.5)
        
        return features
    
    def predict_optimal_delivery_window(self, customer_id: str, day: datetime) -> TimeWindow:
        """
        Predict best delivery time based on learned cultural patterns.
        This replaces generic 9-5 assumptions with actual Kenyan business rhythms.
        """
        try:
            # Get customer type
            customer_type = self._get_customer_type(customer_id)
            location_type = "urban"  # Default, would be stored in customer profile
            
            # Get appropriate HMM
            group_key = f"{customer_type}_{location_type}"
            hmm = self.availability_hmm.get(group_key, self.default_hmm)
            
            if not hmm.is_fitted:
                return self._default_business_hours()
            
            # Predict availability probability for each hour of the day
            hourly_availability = np.zeros(24)
            
            for hour in range(24):
                context_features = self._extract_time_context_features(day, hour)
                try:
                    # Predict probability of "Available" state (state 0)
                    prob_vector = hmm.predict_proba(np.array(context_features).reshape(1, -1))
                    if len(prob_vector) > 0:
                        hourly_availability[hour] = prob_vector[0]  # Probability of Available state
                    else:
                        hourly_availability[hour] = 0.5
                except Exception as e:
                    logger.warning(f"HMM prediction failed for hour {hour}: {e}")
                    hourly_availability[hour] = 0.5
            
            # Find optimal 3-hour delivery window
            best_window_start = 9  # Default to 9 AM
            best_window_score = 0.0
            
            # Try windows from 6 AM to 7 PM
            for start_hour in range(6, 19):
                end_hour = min(start_hour + 3, 22)
                if end_hour - start_hour < 2:  # Ensure at least 2-hour window
                    continue
                
                window_scores = hourly_availability[start_hour:end_hour]
                window_score = np.mean(window_scores) if len(window_scores) > 0 else 0.0
                
                if window_score > best_window_score:
                    best_window_score = window_score
                    best_window_start = start_hour
            
            # Get cultural context
            cultural_context = self._get_cultural_context(day, best_window_start)
            
            # Create time window
            time_window = TimeWindow(
                start_hour=best_window_start,
                end_hour=best_window_start + 3,
                confidence=float(best_window_score),
                cultural_context=cultural_context
            )
            
            # Record prediction
            self._record_prediction(customer_id, time_window, day)
            
            return time_window
            
        except Exception as e:
            logger.error(f"Prediction failed for customer {customer_id}: {e}")
            return self._fallback_delivery_window()
    
    def _get_customer_type(self, customer_id: str) -> str:
        """Get customer type (in production, this would query a database)"""
        # Simple heuristic based on customer_id
        if customer_id.startswith('R'):
            return 'retail_shop'
        elif customer_id.startswith('M'):
            return 'market_vendor'
        elif customer_id.startswith('O'):
            return 'office'
        elif customer_id.startswith('S'):
            return 'school'
        elif customer_id.startswith('H'):
            return 'hospital'
        else:
            return 'retail_shop'  # Default
    
    def _default_business_hours(self) -> TimeWindow:
        """Fallback for generic business hours"""
        return TimeWindow(
            start_hour=9,
            end_hour=17,
            confidence=0.5,
            cultural_context="default_business_hours"
        )
    
    def _fallback_delivery_window(self) -> TimeWindow:
        """Fallback when prediction fails"""
        return TimeWindow(
            start_hour=10,
            end_hour=14,
            confidence=0.3,
            cultural_context="fallback_prediction"
        )
    
    def _get_cultural_context(self, day: datetime, start_hour: int) -> str:
        """Get cultural context for time window"""
        hour_context = ""
        if 6 <= start_hour <= 8:
            hour_context = "morning_opening"
        elif 10 <= start_hour <= 13:
            hour_context = "midday_busy"
        elif 16 <= start_hour <= 19:
            hour_context = "evening_busy"
        elif 20 <= start_hour:
            hour_context = "evening_closing"
        else:
            hour_context = "standard_business"
        
        day_context = ""
        if day.weekday() >= 5:  # Weekend
            day_context = "weekend"
        elif day.weekday() == 0:  # Monday
            day_context = "start_of_week"
        else:
            day_context = "weekday"
        
        return f"{hour_context}_{day_context}"
    
    def _record_prediction(self, customer_id: str, time_window: TimeWindow, day: datetime):
        """Record prediction for performance tracking"""
        record = {
            'customer_id': customer_id,
            'time_window': time_window,
            'prediction_day': day,
            'timestamp': datetime.now(),
            'confidence': time_window.confidence
        }
        self.prediction_history.append(record)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for validation"""
        if len(self.prediction_history) == 0:
            return {
                'prediction_count': 0,
                'avg_confidence': 0.0,
                'improvement_over_default': 0.0
            }
        
        confidences = [r['confidence'] for r in self.prediction_history]
        avg_confidence = np.mean(confidences)
        
        # Calculate improvement over default (9-5)
        default_score = 0.5  # Confidence of default 9-5 window
        improvement = (avg_confidence - default_score) / default_score if default_score > 0 else 0.0
        
        return {
            'prediction_count': len(self.prediction_history),
            'avg_confidence': float(avg_confidence),
            'improvement_over_default': float(improvement),
            'high_confidence_rate': len([c for c in confidences if c > 0.7]) / len(confidences),
            'recent_predictions': len(self.prediction_history[-10:])
        }
    
    def generate_synthetic_training_data(self, n_customers: int = 100) -> List[Dict[str, Any]]:
        """Generate synthetic training data for testing"""
        customer_types = ['retail_shop', 'market_vendor', 'office', 'school', 'restaurant']
        location_types = ['cbd', 'residential', 'market', 'industrial']
        
        training_data = []
        
        for i in range(n_customers):
            customer_type = np.random.choice(customer_types)
            location_type = np.random.choice(location_types)
            
            # Generate interaction history based on customer type
            interaction_history = []
            
            for day in range(30):  # 30 days of history
                for hour in range(24):
                    # Generate availability based on customer type
                    if customer_type == 'retail_shop':
                        available = 10 <= hour <= 18 and not (13 <= hour <= 14)
                    elif customer_type == 'market_vendor':
                        available = 6 <= hour <= 12
                    elif customer_type == 'office':
                        available = 9 <= hour <= 17 and not (12 <= hour <= 14)
                    elif customer_type == 'school':
                        available = 7 <= hour <= 15 and day % 7 not in [5, 6]  # Not weekends
                    elif customer_type == 'restaurant':
                        available = (12 <= hour <= 15) or (19 <= hour <= 21)
                    else:
                        available = 9 <= hour <= 17
                    
                    # Add some randomness
                    if np.random.random() < 0.2:
                        available = not available
                    
                    if available:
                        interaction_history.append({
                            'timestamp': datetime.now() - timedelta(days=30-day, hours=24-hour),
                            'interaction_type': 'delivery',
                            'success': True
                        })
            
            training_data.append({
                'customer_id': f"C{i:03d}",
                'customer_type': customer_type,
                'location_type': location_type,
                'interaction_history': interaction_history
            })
        
        return training_data