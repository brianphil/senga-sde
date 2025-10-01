# File: senga_core/learning_coordinator/coordinator.py
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from .learning_engine import LearningPattern, DecisionOutcome
from .consistency import ConsistencyValidator, ConsistencyReport

logger = logging.getLogger(__name__)

class MultiScaleLearningCoordinator:
    """
    Coordinates learning across VFA, CFA, PFA, and DLA layers.
    Quality Gate: Must propagate learning across all scales simultaneously.
    """
    
    def __init__(self, strategic_vfa=None, tactical_cfa=None, operational_pfa=None):
        self.strategic_vfa = strategic_vfa
        self.tactical_cfa = tactical_cfa
        self.operational_pfa = operational_pfa
        self.consistency_validator = ConsistencyValidator()
        self.learning_propagation_log = []
        self.scale_dependencies = self._build_dependency_graph()
    
    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """Build dependency graph between decision scales"""
        return {
            'operational': ['tactical', 'strategic'],
            'tactical': ['strategic'],
            'strategic': []
        }
    
    def propagate_learning(self, decision_outcome: DecisionOutcome) -> Dict[str, Any]:
        """
        Propagate learning from operational outcomes to tactical and strategic levels.
        This is the core innovation: true multi-scale coordination.
        """
        propagation_results = {
            'decision_id': decision_outcome.decision_id,
            'original_scale': decision_outcome.scale,
            'propagations': [],
            'timestamp': datetime.now()
        }
        
        try:
            # Extract operational patterns
            pattern = self._extract_pattern_from_outcome(decision_outcome)
            
            # Determine propagation targets based on scale
            if decision_outcome.scale == 'operational':
                # Operational → Tactical → Strategic
                if self._is_tactically_significant(pattern):
                    tactical_update = self._convert_to_tactical(pattern)
                    self._update_tactical_models(tactical_update)
                    propagation_results['propagations'].append({
                        'from': 'operational',
                        'to': 'tactical',
                        'pattern': pattern.pattern_type,
                        'success': True
                    })
                
                if self._is_strategically_significant(pattern):
                    strategic_update = self._convert_to_strategic(pattern)
                    self._update_strategic_models(strategic_update)
                    propagation_results['propagations'].append({
                        'from': 'operational',
                        'to': 'strategic',
                        'pattern': pattern.pattern_type,
                        'success': True
                    })
            
            elif decision_outcome.scale == 'tactical':
                # Tactical → Strategic
                if self._is_strategically_significant(pattern):
                    strategic_update = self._convert_to_strategic(pattern)
                    self._update_strategic_models(strategic_update)
                    propagation_results['propagations'].append({
                        'from': 'tactical',
                        'to': 'strategic',
                        'pattern': pattern.pattern_type,
                        'success': True
                    })
            
            # Log propagation
            self.learning_propagation_log.append(propagation_results)
            logger.info(f"Learning propagated: {len(propagation_results['propagations'])} updates")
            
            return propagation_results
            
        except Exception as e:
            logger.error(f"Learning propagation failed: {e}")
            propagation_results['error'] = str(e)
            return propagation_results
    
    def _extract_pattern_from_outcome(self, outcome: DecisionOutcome) -> LearningPattern:
        """Extract learning pattern from decision outcome"""
        # Simplified pattern extraction
        if outcome.scale == 'operational':
            pattern_type = 'route_execution'
        elif outcome.scale == 'tactical':
            pattern_type = 'route_planning'
        else:
            pattern_type = 'fleet_allocation'
        
        return LearningPattern(
            pattern_type=pattern_type,
            confidence=0.8,
            data={
                'decision_id': outcome.decision_id,
                'scale': outcome.scale,
                'timestamp': outcome.timestamp,
                'actual_result': outcome.actual_result
            }
        )
    
    def _is_tactically_significant(self, pattern: LearningPattern) -> bool:
        """Determine if pattern is significant enough for tactical update"""
        # Simple heuristic: always significant for now
        return True
    
    def _is_strategically_significant(self, pattern: LearningPattern) -> bool:
        """Determine if pattern is significant enough for strategic update"""
        # Simple heuristic: significant if confidence > 0.7
        return pattern.confidence > 0.7
    
    def _convert_to_tactical(self, pattern: LearningPattern) -> Dict[str, Any]:
        """Convert operational pattern to tactical learning update"""
        return {
            'pattern_type': pattern.pattern_type,
            'confidence': pattern.confidence,
            'tactical_impact': 'route_optimization_adjustment',
            'data': pattern.data
        }
    
    def _convert_to_strategic(self, pattern: LearningPattern) -> Dict[str, Any]:
        """Convert pattern to strategic learning update"""
        return {
            'pattern_type': pattern.pattern_type,
            'confidence': pattern.confidence,
            'strategic_impact': 'fleet_allocation_adjustment',
            'data': pattern.data
        }
    
    def _update_tactical_models(self, update: Dict[str, Any]):
        """Update tactical CFA models"""
        if self.tactical_cfa:
            # In production, this would call actual CFA update methods
            logger.debug(f"Updating tactical models with: {update}")
        else:
            logger.debug(f"Tactical update (no CFA): {update}")
    
    def _update_strategic_models(self, update: Dict[str, Any]):
        """Update strategic VFA models"""
        if self.strategic_vfa:
            # In production, this would call actual VFA update methods
            logger.debug(f"Updating strategic models with: {update}")
        else:
            logger.debug(f"Strategic update (no VFA): {update}")
    
    def validate_consistency(self) -> ConsistencyReport:
        """Validate consistency across all decision scales"""
        return self.consistency_validator.validate_cross_scale_consistency(
            strategic_vfa=self.strategic_vfa,
            tactical_cfa=self.tactical_cfa,
            operational_pfa=self.operational_pfa
        )
    
    def resolve_consistency_conflicts(self, report: ConsistencyReport) -> Dict[str, Any]:
        """Automatically resolve conflicts between decision scales"""
        return self.consistency_validator.resolve_scale_conflicts(report)
    
    def get_coordination_metrics(self) -> Dict[str, Any]:
        """Get metrics about multi-scale coordination"""
        return {
            'total_propagations': len(self.learning_propagation_log),
            'propagation_success_rate': self._calculate_propagation_success_rate(),
            'active_scales': self._get_active_scales(),
            'last_propagation': self.learning_propagation_log[-1]['timestamp'] if self.learning_propagation_log else None
        }
    
    def _calculate_propagation_success_rate(self) -> float:
        """Calculate success rate of learning propagations"""
        if len(self.learning_propagation_log) == 0:
            return 1.0
        
        successful = sum(1 for log in self.learning_propagation_log 
                        if 'error' not in log)
        return successful / len(self.learning_propagation_log)
    
    def _get_active_scales(self) -> List[str]:
        """Get list of active decision scales"""
        active = []
        if self.operational_pfa:
            active.append('operational')
        if self.tactical_cfa:
            active.append('tactical')
        if self.strategic_vfa:
            active.append('strategic')
        return active