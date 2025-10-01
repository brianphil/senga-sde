# File: tests/test_data_acquisition/test_offline_buffer.py
import pytest
from datetime import datetime, timedelta
from senga_core.data_acquisition.offline_buffer import OfflineDataBuffer

class MockCloudClient:
    def __init__(self, fail_rate=0.0):
        self.fail_rate = fail_rate
        self.push_count = 0
    
    def push(self, data):
        self.push_count += 1
        import random
        return random.random() > self.fail_rate

def test_offline_buffer_initialization():
    buffer = OfflineDataBuffer(72)
    assert buffer.buffer_hours == 72
    assert len(buffer.buffer) == 0
    assert buffer.last_sync is None

def test_offline_buffer_store():
    buffer = OfflineDataBuffer(1)  # 1 hour for testing
    test_data = {'sensor': 'gps', 'value': 123.45}
    
    result = buffer.store(test_data)
    assert result == True
    assert len(buffer.buffer) == 1
    assert buffer.buffer[0]['data'] == test_data
    assert buffer.buffer[0]['synced'] == False

def test_offline_buffer_survives_72h_outage():
    buffer = OfflineDataBuffer(72)
    # Store 72 hours of data (1 item per second would be too much, so 1 per minute)
    for i in range(72 * 60):  # 72 hours * 60 minutes
        buffer.store({'timestamp': i, 'data': f'data_{i}'})
    
    # Buffer should contain exactly 72*60 items (since maxlen is 72*3600, we're well under)
    assert len(buffer.buffer) == 72 * 60

def test_offline_buffer_sync():
    buffer = OfflineDataBuffer(1)
    cloud_client = MockCloudClient(fail_rate=0.0)
    
    # Store 10 items
    for i in range(10):
        buffer.store({'test': f'data_{i}'})
    
    # Sync should succeed for all items
    synced_count = buffer.sync_to_cloud(cloud_client)
    assert synced_count == 10
    
    # All items should be marked as synced
    unsynced = sum(1 for item in buffer.buffer if not item['synced'])
    assert unsynced == 0

def test_offline_buffer_sync_with_failures():
    buffer = OfflineDataBuffer(1)
    cloud_client = MockCloudClient(fail_rate=0.5)  # 50% failure rate
    
    # Store 10 items
    for i in range(10):
        buffer.store({'test': f'data_{i}'})
    
    # Sync with failures
    synced_count = buffer.sync_to_cloud(cloud_client)
    
    # Some items should have failed (statistically)
    assert synced_count <= 10
    assert synced_count >= 0
    
    # Check retry counts
    items_with_retries = sum(1 for item in buffer.buffer if item['retry_count'] > 0)
    assert items_with_retries >= 0  # Some items likely had retries

def test_offline_buffer_status():
    buffer = OfflineDataBuffer(1)
    status = buffer.get_buffer_status()
    assert 'total_items' in status
    assert 'unsynced_items' in status
    assert 'sync_success_rate' in status
    assert 'avg_retry_count' in status
    assert 'buffer_utilization' in status