# File: senga_core/learning_coordinator/validators.py
from typing import Dict, Any, List
import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class LearningVelocityValidator:
    """
    Validates that the system shows measurable learning velocity.
    Quality Gate: Must show >1% improvement per week in prediction accuracy.
    """
    
    def __init__(self, learning_engine):
        self.learning_engine = learning_engine
    
    def validate_learning_velocity(self, window_days: int = 7) -> Dict[str, Any]:
        """Validate learning velocity meets Quality Gate requirements"""
        velocity_metrics = self.learning_engine.assess_learning_velocity(window_days)
        
        # Quality Gate: >1% improvement per week
        meets_requirement = velocity_metrics.get('meets_target', False)
        
        validation_result = {
            'valid': meets_requirement,
            'velocity_metrics': velocity_metrics,
            'requirement': 'learning_velocity > 0.01 (1% per week)',
            'actual_velocity': velocity_metrics.get('learning_velocity_score', 0.0),
            'meets_requirement': meets_requirement,
            'recommendation': 'Proceed' if meets_requirement else 'Increase learning rate or data quality'
        }
        
        if not meets_requirement:
            validation_result['blocking_issue'] = 'Learning velocity below 1% per week threshold'
        
        return validation_result
    
    def validate_convergence_properties(self, vfa=None, cfa=None) -> Dict[str, Any]:
        """Validate that learning algorithms have proper convergence properties"""
        issues = []
        
        # Validate VFA convergence
        if vfa:
            if not vfa.satisfies_robbins_monro():
                issues.append("VFA learning rate does not satisfy Robbins-Monro convergence conditions")
            
            convergence_status = vfa.check_convergence()
            if not convergence_status.get('converged', False):
                issues.append(f"VFA has not achieved empirical convergence (mean TD error: {convergence_status.get('mean_td_error', 0.0):.6f})")
        
        # Validate CFA optimization
        if cfa:
            success_rate = cfa.get_recent_success_rate()
            if success_rate < 0.95:
                issues.append(f"CFA optimization success rate below 95% ({success_rate:.1%})")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'convergence_valid': len(issues) == 0
        }
    
    def validate_multi_scale_coordination(self, coordinator) -> Dict[str, Any]:
        """Validate that multi-scale coordination is functioning properly"""
        coordination_metrics = coordinator.get_coordination_metrics()
        
        propagation_success_rate = coordination_metrics.get('propagation_success_rate', 0.0)
        active_scales = len(coordination_metrics.get('active_scales', []))
        
        issues = []
        if propagation_success_rate < 0.9:
            issues.append(f"Learning propagation success rate below 90% ({propagation_success_rate:.1%})")
        
        if active_scales < 2:
            issues.append(f"Insufficient active decision scales ({active_scales})")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'coordination_valid': len(issues) == 0,
            'metrics': coordination_metrics
        }
    
    def validate_consistency_resolution(self, consistency_validator) -> Dict[str, Any]:
        """Validate that consistency issues are being resolved effectively"""
        if not consistency_validator.resolution_history:
            return {
                'valid': False,
                'issues': ['No consistency resolution history available'],
                'resolution_valid': False
            }
        
        latest_resolution = consistency_validator.resolution_history[-1]
        resolution_rate = latest_resolution.get('resolution_rate', 0.0)
        
        issues = []
        if resolution_rate < 0.95:
            issues.append(f"Consistency resolution rate below 95% ({resolution_rate:.1%})")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'resolution_valid': len(issues) == 0,
            'latest_resolution': latest_resolution
        }