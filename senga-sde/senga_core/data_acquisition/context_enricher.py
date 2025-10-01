# File: senga_core/data_acquisition/context_enricher.py
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class CulturalPatternDB:
    """Simple in-memory DB for cultural patterns - will be replaced with actual HMM later"""
    
    def __init__(self):
        # These will be learned from data in Module 4
        # For now, use simple heuristics based on Implementation Guide
        self.cultural_patterns = {
            'business_type': {
                'retail_shop': {'peak_hours': [10, 18], 'lunch_break': [13, 14]},
                'restaurant': {'peak_hours': [12, 15, 19, 21], 'closed_days': ['monday']},
                'market_vendor': {'peak_hours': [6, 12], 'closed_days': []},
                'office': {'peak_hours': [9, 17], 'lunch_break': [12, 14]},
                'school': {'peak_hours': [7, 15], 'closed_days': ['saturday', 'sunday']}
            },
            'location': {
                'cbd': {'traffic_patterns': {'morning_rush': [7, 9], 'evening_rush': [17, 19]}},
                'residential': {'delivery_windows': [9, 18]},
                'industrial': {'delivery_windows': [8, 16]},
                'peri_urban': {'delivery_windows': [8, 17]}
            }
        }
    
    def get_customer_context(self, customer_id: str, customer_type: str = None, location_type: str = None) -> Dict[str, Any]:
        """Get cultural context for customer"""
        context = {
            'customer_id': customer_id,
            'cultural_alignment_score': 0.5,  # Default score
            'preferred_delivery_windows': [],
            'avoidance_periods': [],
            'payment_preferences': ['cash', 'mpesa'],  # Default Kenyan payment methods
            'last_updated': datetime.utcnow().isoformat()
        }
        
        if customer_type and customer_type in self.cultural_patterns['business_type']:
            pattern = self.cultural_patterns['business_type'][customer_type]
            context['preferred_delivery_windows'] = pattern.get('peak_hours', [])
            context['avoidance_periods'] = pattern.get('lunch_break', []) + pattern.get('closed_days', [])
            context['cultural_alignment_score'] = 0.8  # Higher confidence for known types
        
        if location_type and location_type in self.cultural_patterns['location']:
            pattern = self.cultural_patterns['location'][location_type]
            if 'delivery_windows' in pattern:
                context['preferred_delivery_windows'] = pattern['delivery_windows']
            if 'traffic_patterns' in pattern:
                context['traffic_patterns'] = pattern['traffic_patterns']
            context['cultural_alignment_score'] = max(context['cultural_alignment_score'], 0.7)
        
        return context

class InfrastructureReliabilityDB:
    """Simple in-memory DB for infrastructure reliability"""
    
    def __init__(self):
        # These will be learned from data in Module 4
        # For now, use simple heuristics based on Implementation Guide
        self.route_reliability = {
            'highway': 0.95,
            'urban_road': 0.85,
            'rural_road': 0.70,
            'informal_road': 0.60,
            'construction_zone': 0.40
        }
        
        self.connectivity_reliability = {
            'cbd': 0.90,
            'urban': 0.85,
            'peri_urban': 0.75,
            'rural': 0.65,
            'remote': 0.50
        }
    
    def get_route_reliability(self, route_id: str, road_type: str = None) -> float:
        """Get infrastructure reliability score for route"""
        if road_type and road_type in self.route_reliability:
            return self.route_reliability[road_type]
        
        # Default based on route_id patterns
        if 'HW' in route_id.upper():
            return 0.95
        elif 'URB' in route_id.upper():
            return 0.85
        elif 'RUR' in route_id.upper():
            return 0.70
        else:
            return 0.75  # Default for unknown routes
    
    def get_connectivity_reliability(self, location_type: str = None) -> float:
        """Get connectivity reliability score for location"""
        if location_type and location_type in self.connectivity_reliability:
            return self.connectivity_reliability[location_type]
        
        return 0.75  # Default

class ContextEnricher:
    """
    Enriches data with Senga-specific context.
    Quality Gate: Must add at least 2 Senga-specific context fields.
    """
    
    def __init__(self):
        self.cultural_db = CulturalPatternDB()
        self.infrastructure_db = InfrastructureReliabilityDB()
    
    def enrich(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich raw data with Senga-specific context"""
        enriched_data = raw_data.copy()
        
        try:
            # Add cultural context if customer info available
            if 'customer_id' in raw_data:
                customer_type = raw_data.get('customer_type', 'unknown')
                location_type = raw_data.get('location_type', 'urban')
                cultural_context = self.cultural_db.get_customer_context(
                    raw_data['customer_id'], 
                    customer_type, 
                    location_type
                )
                enriched_data['cultural_context'] = cultural_context
                enriched_data['cultural_alignment_score'] = cultural_context['cultural_alignment_score']
            
            # Add infrastructure reliability if route info available
            if 'route_id' in raw_data:
                road_type = raw_data.get('road_type')
                route_reliability = self.infrastructure_db.get_route_reliability(
                    raw_data['route_id'], 
                    road_type
                )
                enriched_data['infrastructure_reliability'] = route_reliability
            
            # Add connectivity reliability if location info available
            if 'location_type' in raw_data:
                connectivity_reliability = self.infrastructure_db.get_connectivity_reliability(
                    raw_data['location_type']
                )
                enriched_data['connectivity_reliability'] = connectivity_reliability
            
            # Add Senga-specific innovation flag
            enriched_data['senga_enriched'] = True
            enriched_data['enrichment_timestamp'] = datetime.utcnow().isoformat()
            
            logger.debug(f"Enriched data for {raw_data.get('customer_id', 'unknown') or raw_data.get('route_id', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Failed to enrich data: {e}")
            enriched_data['enrichment_failed'] = True
            enriched_data['enrichment_error'] = str(e)
        
        return enriched_data