# File: senga_core/learning_coordinator/types.py
from dataclasses import dataclass
from typing import Dict, Any
from datetime import datetime

@dataclass
class LearningPattern:
    pattern_type: str  # 'traffic', 'customer', 'route'
    confidence: float
    data: Dict[str, Any]

@dataclass
class DecisionOutcome:
    decision_id: str
    original_decision: Dict[str, Any]
    actual_result: Dict[str, Any]
    timestamp: datetime
    scale: str  # 'strategic', 'tactical', 'operational'