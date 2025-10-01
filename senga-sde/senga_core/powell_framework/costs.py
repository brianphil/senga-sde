# File: senga_core/powell_framework/costs.py
import numpy as np
from typing import Dict, Any, List
import logging
from senga_core.state_space.state import StateSpace, Route, Vehicle, Customer
from .exceptions import CostCalculationError
logger = logging.getLogger(__name__)

class SengaCostFunction:
    """
    Multi-objective cost function incorporating African logistics realities.
    Quality Gate: Must include address uncertainty, cultural misalignment, connectivity risk.
    """
    
    def __init__(self, cost_weights: Dict[str, float] = None):
        # Default cost weights
        self.cost_weights = cost_weights or {
            'fuel': 1.0,
            'time': 1.0,
            'address_uncertainty': 2.0,  # Higher weight for Senga-specific challenge
            'cultural_misalignment': 1.5,  # Higher weight for Senga-specific challenge
            'connectivity_risk': 1.5,     # Higher weight for Senga-specific challenge
            'labor': 0.8,
            'maintenance': 0.5
        }
        
        # Initialize cultural patterns
        self.cultural_patterns = self._load_cultural_patterns()
    
    def _load_cultural_patterns(self) -> Dict[str, Any]:
        """Load cultural patterns for cost calculation"""
        return {
            'business_types': {
                'retail_shop': {'peak_hours': [10, 18], 'lunch_break': [13, 14]},
                'restaurant': {'peak_hours': [12, 15, 19, 21], 'closed_days': ['monday']},
                'market_vendor': {'peak_hours': [6, 12], 'closed_days': []},
                'office': {'peak_hours': [9, 17], 'lunch_break': [12, 14]},
                'school': {'peak_hours': [7, 15], 'closed_days': ['saturday', 'sunday']}
            }
        }
    
    def multi_objective_cost(self, state: StateSpace, action: Dict[str, Any]) -> float:
        """
        Cost function incorporating African logistics realities.
        Quality Gate: Must include address uncertainty, cultural misalignment, connectivity risk.
        """
        try:
            costs = {}
            
            # 1. Standard logistics costs
            costs['fuel'] = self.calculate_fuel_cost(action.get('route_plan', []))
            costs['time'] = self.calculate_time_cost(action.get('schedule', {}))
            costs['labor'] = self.calculate_labor_cost(action.get('driver_assignments', []))
            costs['maintenance'] = self.calculate_maintenance_cost(action.get('vehicle_usage', {}))
            
            # 2. SENGA-SPECIFIC: Informal address resolution uncertainty cost
            costs['address_uncertainty'] = self.calculate_address_resolution_risk(
                action.get('delivery_sequence', [])
            )
            
            # 3. SENGA-SPECIFIC: Cultural timing misalignment penalty
            costs['cultural_misalignment'] = self.calculate_cultural_timing_penalty(
                action.get('time_windows', {}),
                state.customers if hasattr(state, 'customers') else []
            )
            
            # 4. SENGA-SPECIFIC: Connectivity loss risk (offline operation capability)
            costs['connectivity_risk'] = self.calculate_offline_operation_difficulty(
                action.get('route_complexity', {}),
                state.environment if hasattr(state, 'environment') else None
            )
            
            # Calculate total cost
            total_cost = 0.0
            for cost_name, cost_value in costs.items():
                weight = self.cost_weights.get(cost_name, 1.0)
                total_cost += weight * cost_value
            
            logger.debug(f"Calculated total cost: {total_cost:.2f} (components: {costs})")
            return total_cost
            
        except Exception as e:
            logger.error(f"Cost calculation failed: {e}")
            raise CostCalculationError(f"Cost calculation failed: {str(e)}")
    
    def calculate_fuel_cost(self, route_plan: List[Dict[str, Any]]) -> float:
        """Calculate fuel cost based on route distance and vehicle efficiency"""
        if not route_plan:
            return 0.0
        
        total_distance = sum(route.get('distance_km', 0.0) for route in route_plan)
        # Assume 10 km/liter and $1.20 per liter (Kenyan fuel price)
        fuel_consumption = total_distance / 10.0
        fuel_cost = fuel_consumption * 1.20
        
        return fuel_cost
    
    def calculate_time_cost(self, schedule: Dict[str, Any]) -> float:
        """Calculate time cost based on driver hours and opportunity cost"""
        total_hours = schedule.get('total_hours', 0.0)
        # Assume $5/hour driver cost
        time_cost = total_hours * 5.0
        
        return time_cost
    
    def calculate_labor_cost(self, driver_assignments: List[Dict[str, Any]]) -> float:
        """Calculate labor cost based on driver assignments"""
        if not driver_assignments:
            return 0.0
        
        total_hours = sum(driver.get('hours', 0.0) for driver in driver_assignments)
        # Assume $5/hour
        labor_cost = total_hours * 5.0
        
        return labor_cost
    
    def calculate_maintenance_cost(self, vehicle_usage: Dict[str, Any]) -> float:
        """Calculate maintenance cost based on vehicle usage"""
        total_distance = vehicle_usage.get('total_distance_km', 0.0)
        # Assume $0.10 per km maintenance cost
        maintenance_cost = total_distance * 0.10
        
        return maintenance_cost
    
    def calculate_address_resolution_risk(self, delivery_sequence: List[Dict[str, Any]]) -> float:
        """
        SENGA-SPECIFIC: Calculate risk cost from informal address resolution uncertainty.
        Higher uncertainty = higher cost.
        """
        if not delivery_sequence:
            return 0.0
        
        total_risk = 0.0
        for delivery in delivery_sequence:
            # Get address confidence (lower confidence = higher risk)
            address_confidence = delivery.get('address_confidence', 0.5)
            # Risk is inverse of confidence
            risk = 1.0 - address_confidence
            total_risk += risk
        
        # Normalize by number of deliveries
        avg_risk = total_risk / len(delivery_sequence)
        
        # Scale up for cost function (address uncertainty is critical in African context)
        return avg_risk * 100.0  # Scale to make it significant in cost function
    
    def calculate_cultural_timing_penalty(self, time_windows: Dict[str, Any], 
                                        customers: List[Customer]) -> float:
        """
        SENGA-SPECIFIC: Calculate penalty for cultural timing misalignment.
        Delivering outside preferred hours = higher penalty.
        """
        if not time_windows or not customers:
            return 0.0
        
        total_penalty = 0.0
        customer_dict = {c.customer_id: c for c in customers}
        
        for customer_id, window in time_windows.items():
            if customer_id not in customer_dict:
                continue
            
            customer = customer_dict[customer_id]
            customer_type = getattr(customer, 'customer_type', 'retail_shop')
            
            # Get preferred hours for this customer type
            preferred_hours = self.cultural_patterns['business_types'].get(
                customer_type, {}).get('peak_hours', [9, 17])
            
            # Get actual delivery window
            start_hour = window.get('start_hour', 12)
            end_hour = window.get('end_hour', start_hour + 1)
            
            # Calculate overlap with preferred hours
            if len(preferred_hours) >= 2:
                pref_start = preferred_hours[0]
                pref_end = preferred_hours[-1]
                
                # Calculate overlap
                overlap_start = max(start_hour, pref_start)
                overlap_end = min(end_hour, pref_end)
                overlap = max(0, overlap_end - overlap_start)
                total_window = end_hour - start_hour
                
                if total_window > 0:
                    alignment = overlap / total_window
                else:
                    alignment = 0.0
                
                # Penalty is inverse of alignment
                penalty = 1.0 - alignment
            else:
                penalty = 0.3  # Default penalty if no clear pattern
            
            total_penalty += penalty
        
        # Normalize by number of customers
        avg_penalty = total_penalty / max(1, len(time_windows))
        
        # Scale up for cost function (cultural alignment is critical in African context)
        return avg_penalty * 50.0  # Scale to make it significant in cost function
    
    def calculate_offline_operation_difficulty(self, route_complexity: Dict[str, Any], 
                                             environment) -> float:
        """
        SENGA-SPECIFIC: Calculate risk cost from connectivity loss (offline operation).
        More complex routes in offline mode = higher risk.
        """
        # Get connectivity status
        connectivity_status = environment.connectivity_status if environment else 'online'
        
        # Base risk based on connectivity
        base_risk = {
            'online': 0.1,
            'degraded': 0.5,
            'offline': 1.0
        }.get(connectivity_status, 0.5)
        
        # Adjust based on route complexity
        complexity_factor = route_complexity.get('complexity_score', 1.0)
        
        # Total risk
        total_risk = base_risk * complexity_factor
        
        # Scale up for cost function (offline operation is critical in African context)
        return total_risk * 30.0  # Scale to make it significant in cost function