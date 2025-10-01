# File: senga_core/state_space/state.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class Route:
    route_id: str
    waypoints: List[Tuple[float, float]]  # (lat, lng)
    distance_km: float
    estimated_time_min: float
    efficiency_score: float  # 0.0 to 1.0
    vehicle_id: Optional[str] = None
    status: str = "planned"  # planned, active, completed, failed

@dataclass
class Vehicle:
    vehicle_id: str
    capacity_kg: float
    fuel_level: float  # 0.0 to 1.0
    maintenance_status: str  # good, warning, critical
    current_location: Tuple[float, float]
    capacity_utilization: float  # 0.0 to 1.0

@dataclass
class Customer:
    customer_id: str
    location: Tuple[float, float]
    demand_kg: float
    time_window: Tuple[int, int]  # (start_hour, end_hour)
    availability_score: float  # 0.0 to 1.0 (from CulturalPatternLearner)
    payment_preference: str  # cash, mpesa, credit

@dataclass
class Environment:
    traffic_index: float  # 0.0 (free) to 1.0 (gridlock)
    weather_condition: str  # clear, rain, storm
    connectivity_status: str  # online, degraded, offline
    time_of_day: int  # 0-23
    day_of_week: int  # 0=Monday, 6=Sunday

@dataclass
class LearningState:
    prediction_accuracy: float  # 0.0 to 1.0
    model_version: str
    last_update: datetime
    uncertainty_estimate: float  # 0.0 to 1.0

@dataclass
class StateSpace:
    """
    Formal State Space S_t = {Routes, Fleet, Customers, Environment, Learning}
    Quality Gate: Must have bounded operations O(n log n) and formal structure.
    """
    routes: List[Route]
    fleet: List[Vehicle]
    customers: List[Customer]
    environment: Environment
    learning: LearningState

    def to_vector(self) -> np.ndarray:
        """Convert state to feature vector for ML models. O(n) complexity."""
        features = []
        
        # Routes: mean efficiency, count
        if len(self.routes) > 0:
            features.append(np.mean([r.efficiency_score for r in self.routes]))
            features.append(float(len(self.routes)))
        else:
            features.extend([0.0, 0.0])
        
        # Fleet: mean fuel, mean utilization, count
        if len(self.fleet) > 0:
            features.append(np.mean([v.fuel_level for v in self.fleet]))
            features.append(np.mean([v.capacity_utilization for v in self.fleet]))
            features.append(float(len(self.fleet)))
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Customers: mean availability, total demand
        if len(self.customers) > 0:
            features.append(np.mean([c.availability_score for c in self.customers]))
            features.append(float(sum(c.demand_kg for c in self.customers)))
        else:
            features.extend([0.0, 0.0])
        
        # Environment
        features.append(self.environment.traffic_index)
        features.append(self.environment.time_of_day / 24.0)  # Normalize to [0,1]
        features.append(self.environment.day_of_week / 7.0)   # Normalize to [0,1]
        
        # Learning
        features.append(self.learning.prediction_accuracy)
        features.append(self.learning.uncertainty_estimate)
        
        # Ensure we have exactly 14 features
        assert len(features) == 14, f"Expected 14 features, got {len(features)}"
        
        return np.array(features, dtype=np.float32)

    def get_state_signature(self) -> str:
        """Generate a hashable signature for this state (for transition counting)"""
        import hashlib
        # Use only key attributes for signature
        sig_data = (
            f"R{len(self.routes)}F{len(self.fleet)}C{len(self.customers)}"
            f"T{self.environment.traffic_index:.2f}"
            f"A{self.learning.prediction_accuracy:.2f}"
        )
        return hashlib.md5(sig_data.encode()).hexdigest()[:16]

    def __hash__(self):
        return hash(self.get_state_signature())

    def __eq__(self, other):
        if not isinstance(other, StateSpace):
            return False
        return self.get_state_signature() == other.get_state_signature()