import pytest
from senga_core.data_acquisition.stream_processor import process_stream

def test_process_stream_yields_records():
    source = [{'id': 1}, {'id': 2}]
    results = list(process_stream(source))
    assert results == [{'id': 1}, {'id': 2}]
# File: tests/test_data_acquisition/test_stream_processor.py
import pytest
import asyncio
from senga_core.data_acquisition.offline_buffer import OfflineDataBuffer
from senga_core.data_acquisition.validators import DataQualityValidator
from senga_core.data_acquisition.context_enricher import ContextEnricher
from senga_core.data_acquisition.stream_processor import StreamProcessor

@pytest.mark.asyncio
async def test_stream_processor_initialization():
    buffer = OfflineDataBuffer(1)
    processor = StreamProcessor(buffer)
    
    assert processor.offline_buffer == buffer
    assert len(processor.processing_queue) == 0
    assert processor.stats['processed_count'] == 0

@pytest.mark.asyncio
async def test_stream_processor_single_item():
    buffer = OfflineDataBuffer(1)
    validator = DataQualityValidator()
    enricher = ContextEnricher()
    processor = StreamProcessor(buffer)
    
    test_item = {
        'customer_id': 'CUST001',
        'gps': {'latitude': -1.2864, 'longitude': 36.8172},
        'address': 'near the big tree'
    }
    
    result = await processor.process_stream_item(test_item, validator, enricher)
    assert result == True
    
    # Check stats
    assert processor.stats['buffered_count'] == 1
    assert processor.stats['processed_count'] == 1
    
    # Check that item was enriched
    assert len(processor.processing_queue) == 1
    processed_item = processor.processing_queue[0]
    assert 'cultural_context' in processed_item
    assert 'address_confidence' in processed_item
    assert processed_item['address_confidence'] > 0.5

@pytest.mark.asyncio
async def test_stream_processor_batch():
    buffer = OfflineDataBuffer(1)
    validator = DataQualityValidator()
    enricher = ContextEnricher()
    processor = StreamProcessor(buffer)
    
    test_items = [
        {'customer_id': f'CUST{i:03d}', 'gps': {'latitude': -1.2864, 'longitude': 36.8172}} 
        for i in range(10)
    ]
    
    results = await processor.batch_process(test_items, validator, enricher)
    
    assert results['success_count'] == 10
    assert results['failure_count'] == 0
    assert results['total_count'] == 10
    assert results['success_rate'] == 1.0
    
    # Check processor stats
    assert processor.stats['buffered_count'] == 10
    assert processor.stats['processed_count'] == 10

@pytest.mark.asyncio
async def test_stream_processor_with_invalid_gps():
    buffer = OfflineDataBuffer(1)
    validator = DataQualityValidator()
    enricher = ContextEnricher()
    processor = StreamProcessor(buffer)
    
    # Invalid GPS (out of bounds)
    test_item = {
        'customer_id': 'CUST001',
        'gps': {'latitude': 0.0, 'longitude': 0.0}  # Invalid for Nairobi
    }
    
    result = await processor.process_stream_item(test_item, validator, enricher)
    assert result == True  # Should still store in buffer even with validation failures
    
    # Check that validation failed flag was added
    assert len(processor.processing_queue) == 1
    processed_item = processor.processing_queue[0]
    assert 'validation_failed' in processed_item
    assert processed_item['validation_failed'] == True