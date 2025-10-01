# File: senga_core/powell_framework/features.py
import numpy as np
from typing import List, Dict, Any
import logging
from senga_core.state_space.state import StateSpace, Route, Vehicle, Customer, Environment
from .exceptions import FeatureExtractionError, CostCalculationError
logger = logging.getLogger(__name__)

class SengaFeatureExtractor:
    """
    Extracts Senga-specific features for Value Function Approximation.
    Quality Gate: Must include infrastructure reliability, cultural alignment, cascade risk.
    """
    
    def __init__(self):
        # Initialize any models or databases needed
        self.cultural_patterns = self._load_cultural_patterns()
        self.infrastructure_map = self._load_infrastructure_map()
    
    def _load_cultural_patterns(self) -> Dict[str, Any]:
        """Load cultural patterns for Kenyan business rhythms"""
        # In production, this would load from a database
        # For now, use simple heuristics
        return {
            'business_types': {
                'retail_shop': {'peak_hours': [10, 18], 'lunch_break': [13, 14]},
                'restaurant': {'peak_hours': [12, 15, 19, 21], 'closed_days': ['monday']},
                'market_vendor': {'peak_hours': [6, 12], 'closed_days': []},
                'office': {'peak_hours': [9, 17], 'lunch_break': [12, 14]},
                'school': {'peak_hours': [7, 15], 'closed_days': ['saturday', 'sunday']}
            },
            'locations': {
                'cbd': {'traffic_patterns': {'morning_rush': [7, 9], 'evening_rush': [17, 19]}},
                'residential': {'delivery_windows': [9, 18]},
                'industrial': {'delivery_windows': [8, 16]},
                'peri_urban': {'delivery_windows': [8, 17]}
            }
        }
    
    def _load_infrastructure_map(self) -> Dict[str, float]:
        """Load infrastructure reliability scores"""
        # In production, this would be learned from data
        # For now, use simple heuristics based on Implementation Guide
        return {
            'highway': 0.95,
            'urban_road': 0.85,
            'rural_road': 0.70,
            'informal_road': 0.60,
            'construction_zone': 0.40
        }
    
    def extract_strategic_features(self, state: StateSpace) -> np.ndarray:
        """
        Extract features specific to African logistics reality.
        Quality Gate: Must include infrastructure reliability, cultural alignment, cascade risk.
        """
        try:
            features = []
            
            # 1. Infrastructure reliability feature (unique to African context)
            connectivity_reliability = self.calculate_infrastructure_reliability(state.environment)
            features.append(connectivity_reliability)
            
            # 2. Cultural timing alignment (business rhythm learning)
            cultural_alignment = self.assess_cultural_timing_efficiency(state)
            features.append(cultural_alignment)
            
            # 3. Cascade effect potential (network fragility unique to direct delivery)
            cascade_risk = self.calculate_cascade_vulnerability(state.routes)
            features.append(cascade_risk)
            
            # 4. Fleet utilization efficiency
            if state.fleet:
                avg_utilization = np.mean([v.capacity_utilization for v in state.fleet])
                features.append(avg_utilization)
            else:
                features.append(0.0)
            
            # 5. Route efficiency score
            if state.routes:
                avg_efficiency = np.mean([r.efficiency_score for r in state.routes])
                features.append(avg_efficiency)
            else:
                features.append(0.0)
            
            # 6. Customer availability score
            if state.customers:
                avg_availability = np.mean([c.availability_score for c in state.customers])
                features.append(avg_availability)
            else:
                features.append(0.0)
            
            # 7. Traffic index (normalized)
            features.append(state.environment.traffic_index)
            
            # 8. Time of day (normalized)
            features.append(state.environment.time_of_day / 24.0)
            
            # 9. Day of week (normalized)
            features.append(state.environment.day_of_week / 7.0)
            
            # 10. Learning state accuracy
            features.append(state.learning.prediction_accuracy)
            
            # 11. Learning state uncertainty
            features.append(state.learning.uncertainty_estimate)
            
            # 12. Number of routes
            features.append(float(len(state.routes)))
            
            # 13. Number of vehicles
            features.append(float(len(state.fleet)))
            
            # 14. Number of customers
            features.append(float(len(state.customers)))
            
            # Ensure we have exactly 14 features (must match state vector dimensionality)
            if len(features) != 14:
                raise FeatureExtractionError(f"Expected 14 features, got {len(features)}")
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise FeatureExtractionError(f"Feature extraction failed: {str(e)}")
    
    def calculate_infrastructure_reliability(self, environment: Environment) -> float:
        """Calculate infrastructure reliability based on location type and connectivity"""
        # Simple heuristic based on location type
        location_reliability = {
            'online': 0.95,
            'degraded': 0.80,
            'offline': 0.65
        }
        
        reliability = location_reliability.get(environment.connectivity_status, 0.75)
        
        # Adjust based on traffic (higher traffic might indicate better infrastructure)
        traffic_adjustment = environment.traffic_index * 0.1  # Add up to 0.1 based on traffic
        reliability = min(1.0, reliability + traffic_adjustment)
        
        return reliability
    
    def assess_cultural_timing_efficiency(self, state: StateSpace) -> float:
        """Assess how well current operations align with cultural business patterns"""
        if not state.customers or not state.routes:
            return 0.5  # Default score
        
        alignment_scores = []
        
        for customer in state.customers:
            # Get customer type and location
            customer_type = getattr(customer, 'customer_type', 'retail_shop')
            location_type = getattr(customer, 'location_type', 'urban')
            
            # Get preferred hours for this customer type
            preferred_hours = self.cultural_patterns['business_types'].get(
                customer_type, {}).get('peak_hours', [9, 17])
            
            # Check if current time is within preferred hours
            current_hour = state.environment.time_of_day
            if len(preferred_hours) >= 2:
                if preferred_hours[0] <= current_hour <= preferred_hours[-1]:
                    alignment_scores.append(1.0)
                else:
                    alignment_scores.append(0.5)
            else:
                alignment_scores.append(0.7)  # Default if no clear pattern
        
        return np.mean(alignment_scores) if alignment_scores else 0.5
    
    def calculate_cascade_vulnerability(self, routes: List[Route]) -> float:
        """Calculate cascade effect potential based on route interdependencies"""
        if len(routes) <= 1:
            return 0.1  # Low risk with single route
        
        # Simple heuristic: more routes = higher cascade risk
        # Also consider route efficiency (lower efficiency = higher risk)
        avg_efficiency = np.mean([r.efficiency_score for r in routes]) if routes else 0.8
        route_count_factor = min(1.0, len(routes) / 10.0)  # Cap at 10 routes
        
        # Cascade risk increases with route count but decreases with efficiency
        cascade_risk = route_count_factor * (1.0 - avg_efficiency)
        
        return cascade_risk