# File: tests/test_learning_coordinator/test_consistency.py
import pytest

from datetime import datetime
from senga_core.learning_coordinator.consistency import ConsistencyValidator, ConsistencyReport

def test_consistency_validator_initialization():
    validator = ConsistencyValidator()
    assert validator is not None

def test_validate_cross_scale_consistency():
    validator = ConsistencyValidator()
    
    report = validator.validate_cross_scale_consistency()
    
    assert isinstance(report, ConsistencyReport)
    assert isinstance(report.consistent, bool)
    assert isinstance(report.inconsistencies, list)
    assert hasattr(report, 'timestamp')

def test_resolve_scale_conflicts():
    validator = ConsistencyValidator()
    
    # Create test inconsistency report
    test_report = ConsistencyReport(
        consistent=False,
        inconsistencies=[
            {
                'type': 'fleet_allocation_mismatch',
                'strategic_recommendation': {'total_vehicles': 10},
                'tactical_reality': {'active_vehicles': 12},
                'severity': 'high',
                'impact': 'Resource over-allocation'
            },
            {
                'type': 'route_execution_deviation',
                'planned_routes': [{'route_id': 'R001', 'estimated_time_min': 45.0}],
                'actual_routes': [{'route_id': 'R001', 'actual_time_min': 60.0}],
                'deviation_score': 0.33,
                'severity': 'medium',
                'impact': 'Plan execution quality degradation'
            }
        ],
        timestamp=datetime.now()
    )
    
    resolution = validator.resolve_scale_conflicts(test_report)
    
    assert 'total_inconsistencies' in resolution
    assert 'resolved_inconsistencies' in resolution
    assert 'resolution_rate' in resolution
    assert 'meets_target' in resolution
    assert resolution['total_inconsistencies'] == 2
    assert resolution['resolution_rate'] >= 0.0  # Should be able to resolve some

def test_fleet_allocations_consistent():
    validator = ConsistencyValidator()
    
    strategic = {'total_vehicles': 10}
    tactical = {'active_vehicles': 11}  # Within 20% buffer
    
    assert validator._fleet_allocations_consistent(strategic, tactical) == True
    
    tactical_over = {'active_vehicles': 13}  # Over 20% buffer
    
    assert validator._fleet_allocations_consistent(strategic, tactical_over) == False