# File: senga_core/data_acquisition/stream_processor.py
from typing import Dict, Any, List, Callable, Optional
import asyncio
import logging
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)

class StreamProcessor:
    """
    Processes data streams with handling for intermittent connectivity.
    Quality Gate: Must handle simulated network outages gracefully.
    """
    
    def __init__(self, offline_buffer, max_queue_size: int = 1000):
        self.offline_buffer = offline_buffer
        self.max_queue_size = max_queue_size
        self.processing_queue = []
        self.failed_items = []
        self.stats = {
            'processed_count': 0,
            'buffered_count': 0,
            'failed_count': 0,
            'last_processed': None
        }
    
    async def process_stream_item(self, data_item: Dict[str, Any], 
                                validator: Optional[Callable] = None,
                                enricher: Optional[Callable] = None) -> bool:
        """Process a single stream item"""
        try:
            # Validate if validator provided
            if validator:
                # For GPS validation, extract coordinates
                if 'gps' in data_item:
                    lat = data_item['gps'].get('latitude')
                    lng = data_item['gps'].get('longitude')
                    validation_result = validator.validate_gps(lat, lng)
                    if not validation_result.valid:
                        logger.warning(f"GPS validation failed: {validation_result.reason}")
                        data_item['validation_failed'] = True
                        data_item['validation_reason'] = validation_result.reason
                
                # For address validation
                if 'address' in data_item and isinstance(data_item['address'], str):
                    address_score = validator.score_address_probability(data_item['address'])
                    data_item['address_confidence'] = address_score
            
            # Enrich if enricher provided
            if enricher:
                data_item = enricher.enrich(data_item)
            
            # Store in offline buffer (handles connectivity issues)
            success = self.offline_buffer.store(data_item)
            
            if success:
                self.stats['buffered_count'] += 1
                self.stats['last_processed'] = datetime.utcnow()
                
                # Add to processing queue if space available
                if len(self.processing_queue) < self.max_queue_size:
                    self.processing_queue.append(data_item)
                    self.stats['processed_count'] += 1
                else:
                    logger.warning("Processing queue full, item only buffered")
                
                return True
            else:
                self.stats['failed_count'] += 1
                self.failed_items.append(data_item)
                return False
                
        except Exception as e:
            logger.error(f"Error processing stream item: {e}")
            self.stats['failed_count'] += 1
            self.failed_items.append(data_item)
            return False
    
    async def batch_process(self, data_items: List[Dict[str, Any]], 
                          validator: Optional[Callable] = None,
                          enricher: Optional[Callable] = None) -> Dict[str, Any]:
        """Process a batch of stream items"""
        results = {
            'success_count': 0,
            'failure_count': 0,
            'total_count': len(data_items),
            'start_time': datetime.utcnow(),
            'end_time': None
        }
        
        for item in data_items:
            success = await self.process_stream_item(item, validator, enricher)
            if success:
                results['success_count'] += 1
            else:
                results['failure_count'] += 1
        
        results['end_time'] = datetime.utcnow()
        results['processing_duration'] = (results['end_time'] - results['start_time']).total_seconds()
        results['success_rate'] = results['success_count'] / max(1, results['total_count'])
        
        logger.info(f"Batch processing completed: {results['success_count']}/{results['total_count']} successful")
        return results
    
    def get_processor_status(self) -> Dict[str, Any]:
        """Get current processor status"""
        return {
            'queue_size': len(self.processing_queue),
            'failed_items_count': len(self.failed_items),
            'stats': self.stats.copy(),
            'buffer_status': self.offline_buffer.get_buffer_status() if self.offline_buffer else None,
            'is_healthy': self.is_healthy()
        }
    
    def is_healthy(self) -> bool:
        """Processor is healthy if success rate > 90% and queue not full"""
        if self.stats['processed_count'] == 0:
            return True
        
        success_rate = self.stats['processed_count'] / max(1, (self.stats['processed_count'] + self.stats['failed_count']))
        queue_utilization = len(self.processing_queue) / self.max_queue_size
        
        return success_rate > 0.9 and queue_utilization < 0.8