# File: senga_core/data_acquisition/offline_buffer.py
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class OfflineDataBuffer:
    """
    Implements 72-hour offline buffer with sync capability.
    Quality Gate: Must survive simulated 72-hour outage without data loss.
    """
    
    def __init__(self, buffer_hours: int = 72):
        self.buffer_hours = buffer_hours
        self.buffer = deque(maxlen=buffer_hours * 3600)  # 1-second resolution
        self.last_sync = None
        self.sync_failures = 0
        self.max_sync_retries = 3

    def store(self, data_point: Dict[str, Any]) -> bool:
        """Store data point with timestamp and sync status"""
        try:
            buffered_item = {
                'timestamp': datetime.utcnow(),
                'data': data_point,
                'synced': False,
                'retry_count': 0
            }
            self.buffer.append(buffered_item)
            logger.debug(f"Stored data point. Buffer size: {len(self.buffer)}")
            return True
        except Exception as e:
            logger.error(f"Failed to store data: {e}")
            return False

    def sync_to_cloud(self, cloud_client) -> int:
        """
        Sync all unsynced data points to cloud.
        Returns number of successfully synced items.
        Implements exponential backoff for retries.
        """
        synced_count = 0
        current_time = datetime.utcnow()
        
        # Create a list of indices to sync (avoid modifying deque while iterating)
        items_to_sync = []
        for i, item in enumerate(self.buffer):
            if not item['synced'] and item['retry_count'] < self.max_sync_retries:
                items_to_sync.append(i)
        
        for i in items_to_sync:
            item = self.buffer[i]
            success = False
            
            for attempt in range(self.max_sync_retries - item['retry_count']):
                try:
                    if cloud_client.push(item['data']):
                        item['synced'] = True
                        item['synced_at'] = current_time
                        synced_count += 1
                        success = True
                        break
                    else:
                        # Exponential backoff
                        import time
                        time.sleep(2 ** attempt)
                except Exception as e:
                    logger.warning(f"Sync attempt {attempt + 1} failed: {e}")
                    continue
            
            if not success:
                item['retry_count'] += self.max_sync_retries - item['retry_count']
                self.sync_failures += 1
        
        self.last_sync = current_time
        logger.info(f"Sync completed: {synced_count} items synced, {self.sync_failures} failures")
        return synced_count

    def get_buffer_status(self) -> Dict[str, Any]:
        """Return buffer health metrics"""
        total_items = len(self.buffer)
        unsynced_items = sum(1 for item in self.buffer if not item['synced'])
        avg_retry_count = sum(item['retry_count'] for item in self.buffer) / max(1, total_items)
        
        return {
            'total_items': total_items,
            'unsynced_items': unsynced_items,
            'sync_success_rate': (total_items - unsynced_items) / max(1, total_items),
            'avg_retry_count': avg_retry_count,
            'buffer_utilization': total_items / (self.buffer_hours * 3600),
            'last_sync': self.last_sync
        }

    def is_healthy(self) -> bool:
        """Buffer is healthy if utilization < 95% and sync success rate > 90%"""
        status = self.get_buffer_status()
        return (status['buffer_utilization'] < 0.95 and 
                status['sync_success_rate'] > 0.9)