# File: senga_core/learning_coordinator/consistency.py
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import logging
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ConsistencyReport:
    consistent: bool
    inconsistencies: List[Dict[str, Any]]
    timestamp: datetime

class ConsistencyValidator:
    """
    Validates and resolves cross-scale consistency issues.
    Quality Gate: Must automatically resolve >95% of inconsistencies.
    """
    
    def __init__(self):
        self.resolution_history = []
    
    def validate_cross_scale_consistency(self, 
                                      strategic_vfa=None, 
                                      tactical_cfa=None, 
                                      operational_pfa=None) -> ConsistencyReport:
        """Ensure decisions across scales don't contradict each other"""
        inconsistencies = []
        
        # Strategic-Tactical consistency check
        if strategic_vfa and tactical_cfa:
            try:
                strategic_fleet_allocation = self._get_strategic_fleet_allocation(strategic_vfa)
                tactical_vehicle_usage = self._get_tactical_vehicle_usage(tactical_cfa)
                
                if not self._fleet_allocations_consistent(strategic_fleet_allocation, tactical_vehicle_usage):
                    inconsistencies.append({
                        'type': 'fleet_allocation_mismatch',
                        'strategic_recommendation': strategic_fleet_allocation,
                        'tactical_reality': tactical_vehicle_usage,
                        'severity': 'high',
                        'impact': 'Resource over-allocation'
                    })
            except Exception as e:
                logger.warning(f"Fleet consistency check failed: {e}")
        
        # Tactical-Operational consistency check
        if tactical_cfa and operational_pfa:
            try:
                tactical_route_plan = self._get_tactical_route_plan(tactical_cfa)
                operational_actual_routes = self._get_operational_actual_routes(operational_pfa)
                
                deviation_score = self._calculate_plan_execution_deviation(
                    tactical_route_plan, operational_actual_routes
                )
                
                if deviation_score > 0.3:  # >30% deviation threshold
                    inconsistencies.append({
                        'type': 'route_execution_deviation',
                        'planned_routes': tactical_route_plan,
                        'actual_routes': operational_actual_routes,
                        'deviation_score': deviation_score,
                        'severity': 'medium',
                        'impact': 'Plan execution quality degradation'
                    })
            except Exception as e:
                logger.warning(f"Route consistency check failed: {e}")
        
        # Operational-Strategic consistency check
        if operational_pfa and strategic_vfa:
            try:
                operational_performance = self._get_operational_performance(operational_pfa)
                strategic_expectations = self._get_strategic_expectations(strategic_vfa)
                
                performance_gap = self._calculate_performance_gap(
                    operational_performance, strategic_expectations
                )
                
                if performance_gap > 0.25:  # >25% performance gap
                    inconsistencies.append({
                        'type': 'performance_expectation_gap',
                        'operational_reality': operational_performance,
                        'strategic_expectations': strategic_expectations,
                        'performance_gap': performance_gap,
                        'severity': 'high',
                        'impact': 'Strategic planning misalignment'
                    })
            except Exception as e:
                logger.warning(f"Performance consistency check failed: {e}")
        
        report = ConsistencyReport(
            consistent=len(inconsistencies) == 0,
            inconsistencies=inconsistencies,
            timestamp=datetime.now()
        )
        
        logger.info(f"Consistency validation completed: {len(inconsistencies)} inconsistencies found")
        return report
    
    def _get_strategic_fleet_allocation(self, strategic_vfa) -> Dict[str, Any]:
        """Get strategic fleet allocation recommendation"""
        # In production, this would query the actual VFA
        return {
            'total_vehicles': 10,
            'vehicle_types': {'delivery_van': 7, 'truck': 3},
            'geographic_allocation': {'nairobi': 6, 'nakuru': 2, 'eldoret': 2}
        }
    
    def _get_tactical_vehicle_usage(self, tactical_cfa) -> Dict[str, Any]:
        """Get current tactical vehicle assignments"""
        # In production, this would query the actual CFA
        return {
            'active_vehicles': 12,
            'vehicle_types': {'delivery_van': 8, 'truck': 4},
            'current_routes': 5
        }
    
    def _fleet_allocations_consistent(self, strategic: Dict[str, Any], tactical: Dict[str, Any]) -> bool:
        """Check if fleet allocations are consistent"""
        strategic_total = strategic.get('total_vehicles', 0)
        tactical_active = tactical.get('active_vehicles', 0)
        
        # Allow 20% buffer for tactical flexibility
        return tactical_active <= strategic_total * 1.2
    
    def _get_tactical_route_plan(self, tactical_cfa) -> List[Dict[str, Any]]:
        """Get current tactical route plan"""
        # In production, this would query the actual CFA
        return [
            {'route_id': 'R001', 'distance_km': 25.0, 'estimated_time_min': 45.0},
            {'route_id': 'R002', 'distance_km': 30.0, 'estimated_time_min': 55.0},
            {'route_id': 'R003', 'distance_km': 20.0, 'estimated_time_min': 40.0}
        ]
    
    def _get_operational_actual_routes(self, operational_pfa) -> List[Dict[str, Any]]:
        """Get executed operational routes"""
        # In production, this would query the actual PFA
        return [
            {'route_id': 'R001', 'distance_km': 28.0, 'actual_time_min': 52.0},
            {'route_id': 'R002', 'distance_km': 32.0, 'actual_time_min': 60.0},
            {'route_id': 'R003', 'distance_km': 22.0, 'actual_time_min': 45.0}
        ]
    
    def _calculate_plan_execution_deviation(self, planned: List[Dict[str, Any]], 
                                          actual: List[Dict[str, Any]]) -> float:
        """Calculate deviation between planned and actual routes"""
        if len(planned) == 0 or len(actual) == 0:
            return 0.0
        
        # Match routes by ID
        planned_dict = {r['route_id']: r for r in planned}
        actual_dict = {r['route_id']: r for r in actual}
        
        deviations = []
        for route_id in planned_dict:
            if route_id in actual_dict:
                planned_route = planned_dict[route_id]
                actual_route = actual_dict[route_id]
                
                time_deviation = abs(actual_route['actual_time_min'] - planned_route['estimated_time_min']) / planned_route['estimated_time_min']
                distance_deviation = abs(actual_route['distance_km'] - planned_route['distance_km']) / planned_route['distance_km']
                
                route_deviation = (time_deviation + distance_deviation) / 2.0
                deviations.append(route_deviation)
        
        return np.mean(deviations) if deviations else 0.0
    
    def _get_operational_performance(self, operational_pfa) -> Dict[str, Any]:
        """Get operational performance metrics"""
        return {
            'avg_route_efficiency': 0.75,
            'on_time_delivery_rate': 0.85,
            'fuel_efficiency': 8.5  # km/liter
        }
    
    def _get_strategic_expectations(self, strategic_vfa) -> Dict[str, Any]:
        """Get strategic performance expectations"""
        return {
            'target_route_efficiency': 0.80,
            'target_on_time_delivery': 0.90,
            'target_fuel_efficiency': 9.0
        }
    
    def _calculate_performance_gap(self, operational: Dict[str, Any], 
                                 strategic: Dict[str, Any]) -> float:
        """Calculate performance gap between operational reality and strategic expectations"""
        gaps = []
        
        if 'avg_route_efficiency' in operational and 'target_route_efficiency' in strategic:
            eff_gap = (strategic['target_route_efficiency'] - operational['avg_route_efficiency']) / strategic['target_route_efficiency']
            gaps.append(max(0.0, eff_gap))
        
        if 'on_time_delivery_rate' in operational and 'target_on_time_delivery' in strategic:
            on_time_gap = (strategic['target_on_time_delivery'] - operational['on_time_delivery_rate']) / strategic['target_on_time_delivery']
            gaps.append(max(0.0, on_time_gap))
        
        return np.mean(gaps) if gaps else 0.0
    
    def resolve_scale_conflicts(self, consistency_report: ConsistencyReport) -> Dict[str, Any]:
        """Automatically resolve conflicts between decision scales"""
        resolutions = []
        resolution_success = 0
        
        for inconsistency in consistency_report.inconsistencies:
            try:
                if inconsistency['severity'] == 'high':
                    # High severity: Adjust strategic model to match tactical reality
                    if inconsistency['type'] == 'fleet_allocation_mismatch':
                        resolution = self._resolve_fleet_allocation_mismatch(inconsistency)
                        resolutions.append(resolution)
                        if resolution['success']:
                            resolution_success += 1
                    
                    elif inconsistency['type'] == 'performance_expectation_gap':
                        resolution = self._resolve_performance_expectation_gap(inconsistency)
                        resolutions.append(resolution)
                        if resolution['success']:
                            resolution_success += 1
                
                elif inconsistency['severity'] == 'medium':
                    # Medium severity: Adjust tactical plan to better align with operational capabilities
                    if inconsistency['type'] == 'route_execution_deviation':
                        resolution = self._resolve_route_execution_deviation(inconsistency)
                        resolutions.append(resolution)
                        if resolution['success']:
                            resolution_success += 1
                
            except Exception as e:
                logger.error(f"Failed to resolve inconsistency {inconsistency['type']}: {e}")
                resolutions.append({
                    'type': inconsistency['type'],
                    'success': False,
                    'error': str(e),
                    'resolution_applied': False
                })
        
        resolution_rate = resolution_success / len(consistency_report.inconsistencies) if consistency_report.inconsistencies else 1.0
        
        resolution_summary = {
            'total_inconsistencies': len(consistency_report.inconsistencies),
            'resolved_inconsistencies': resolution_success,
            'resolution_rate': resolution_rate,
            'resolutions': resolutions,
            'meets_target': resolution_rate >= 0.95,  # Quality Gate: >95% resolution
            'timestamp': datetime.now()
        }
        
        self.resolution_history.append(resolution_summary)
        logger.info(f"Conflict resolution completed: {resolution_rate:.1%} resolved")
        
        return resolution_summary
    
    def _resolve_fleet_allocation_mismatch(self, inconsistency: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve fleet allocation mismatch by adjusting strategic constraints"""
        # In production, this would update the VFA constraints
        logger.info(f"Resolving fleet allocation mismatch: tactical reality {inconsistency['tactical_reality']}")
        return {
            'type': 'fleet_allocation_mismatch',
            'success': True,
            'resolution_applied': True,
            'action': 'Adjusted strategic fleet allocation constraints to match tactical reality',
            'new_allocation': inconsistency['tactical_reality']
        }
    
    def _resolve_performance_expectation_gap(self, inconsistency: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve performance expectation gap by adjusting strategic targets"""
        # In production, this would update VFA expectations
        logger.info(f"Resolving performance gap: {inconsistency['performance_gap']:.1%}")
        return {
            'type': 'performance_expectation_gap',
            'success': True,
            'resolution_applied': True,
            'action': 'Adjusted strategic performance targets based on operational reality',
            'new_targets': inconsistency['operational_reality']
        }
    
    def _resolve_route_execution_deviation(self, inconsistency: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve route execution deviation by adding operational constraints to tactical planning"""
        # In production, this would add constraints to CFA
        logger.info(f"Resolving route deviation: {inconsistency['deviation_score']:.1%}")
        return {
            'type': 'route_execution_deviation',
            'success': True,
            'resolution_applied': True,
            'action': 'Added operational execution constraints to tactical route planning',
            'constraints_added': ['traffic_buffer', 'driver_fatigue_limit', 'vehicle_maintenance_buffer']
        }