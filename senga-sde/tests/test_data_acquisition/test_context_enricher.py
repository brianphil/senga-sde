import pytest
from senga_core.data_acquisition.context_enricher import enrich

def test_enrich_adds_context():
    record = {'id': 1}
    context = {'source': 'sensorA'}
    enriched = enrich(record, context)
    assert enriched['id'] == 1
    assert enriched['source'] == 'sensorA'
    assert record == {'id': 1}  # original unchanged
# File: tests/test_data_acquisition/test_context_enricher.py
import pytest
from senga_core.data_acquisition.context_enricher import ContextEnricher

def test_context_enricher_initialization():
    enricher = ContextEnricher()
    assert enricher.cultural_db is not None
    assert enricher.infrastructure_db is not None

def test_context_enricher_with_customer_data():
    enricher = ContextEnricher()
    raw_data = {
        'customer_id': 'CUST001',
        'customer_type': 'retail_shop',
        'location_type': 'cbd'
    }
    
    enriched = enricher.enrich(raw_data)
    
    # Should have added cultural context
    assert 'cultural_context' in enriched
    assert 'cultural_alignment_score' in enriched
    assert enriched['cultural_alignment_score'] >= 0.5
    
    # Should have added Senga enrichment flag
    assert enriched['senga_enriched'] == True

def test_context_enricher_with_route_data():
    enricher = ContextEnricher()
    raw_data = {
        'route_id': 'HW001',
        'road_type': 'highway'
    }
    
    enriched = enricher.enrich(raw_data)
    
    # Should have added infrastructure reliability
    assert 'infrastructure_reliability' in enriched
    assert enriched['infrastructure_reliability'] >= 0.7  # Highway should be high reliability

def test_context_enricher_with_location_data():
    enricher = ContextEnricher()
    raw_data = {
        'location_type': 'rural'
    }
    
    enriched = enricher.enrich(raw_data)
    
    # Should have added connectivity reliability
    assert 'connectivity_reliability' in enriched
    assert enriched['connectivity_reliability'] <= 0.75  # Rural should have lower connectivity

def test_context_enricher_adds_senga_specific_fields():
    enricher = ContextEnricher()
    raw_data = {
        'customer_id': 'CUST001',
        'route_id': 'RTE001',
        'location_type': 'urban'
    }
    
    enriched = enricher.enrich(raw_data)
    
    # Quality Gate: Must add at least 2 Senga-specific context fields
    senga_fields = 0
    expected_fields = [
        'cultural_context', 
        'cultural_alignment_score', 
        'infrastructure_reliability', 
        'connectivity_reliability',
        'senga_enriched'
    ]
    
    for field in expected_fields:
        if field in enriched:
            senga_fields += 1
    
    assert senga_fields >= 2, f"Only added {senga_fields} Senga-specific fields, need at least 2"