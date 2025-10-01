# File: tests/test_learning_coordinator/test_coordinator.py
import pytest
from datetime import datetime
from senga_core.learning_coordinator.coordinator import MultiScaleLearningCoordinator
from senga_core.learning_coordinator.learning_engine import DecisionOutcome

def test_coordinator_initialization():
    coordinator = MultiScaleLearningCoordinator()
    assert coordinator is not None
    assert coordinator.consistency_validator is not None

def test_propagate_learning():
    coordinator = MultiScaleLearningCoordinator()
    
    outcome = DecisionOutcome(
        decision_id="DEC001",
        original_decision={'test': 'data'},
        actual_result={'test': 'result'},
        timestamp=datetime.now(),
        scale='operational'
    )
    
    result = coordinator.propagate_learning(outcome)
    
    assert 'decision_id' in result
    assert 'propagations' in result
    assert isinstance(result['propagations'], list)

def test_validate_consistency():
    coordinator = MultiScaleLearningCoordinator()
    
    report = coordinator.validate_consistency()
    
    assert hasattr(report, 'consistent')
    assert hasattr(report, 'inconsistencies')
    assert hasattr(report, 'timestamp')
    assert isinstance(report.inconsistencies, list)

def test_get_coordination_metrics():
    coordinator = MultiScaleLearningCoordinator()
    
    metrics = coordinator.get_coordination_metrics()
    
    assert 'total_propagations' in metrics
    assert 'propagation_success_rate' in metrics
    assert 'active_scales' in metrics
    assert isinstance(metrics['propagation_success_rate'], float)
    assert isinstance(metrics['active_scales'], list)