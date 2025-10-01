import pytest
from senga_core.data_acquisition.validators import validate_record

def test_valid_record():
    record = {'id': 123, 'value': 'ok'}
    assert validate_record(record)

def test_invalid_record():
    assert not validate_record('not a dict')
    assert not validate_record({})
# File: tests/test_data_acquisition/test_validators.py
import pytest
from senga_core.data_acquisition.validators import DataQualityValidator, ValidationResult

def test_gps_validator_within_bounds():
    validator = DataQualityValidator()
    
    # Test valid Nairobi coordinates
    result = validator.validate_gps(-1.2864, 36.8172)  # Central Nairobi
    assert result.valid == True
    assert result.uncertainty_flag == False

def test_gps_validator_out_of_bounds():
    validator = DataQualityValidator()
    
    # Test invalid latitude (too far north)
    result = validator.validate_gps(-1.0, 36.8172)
    assert result.valid == False
    assert result.uncertainty_flag == True
    assert "Latitude" in result.reason
    
    # Test invalid longitude (too far west)
    result = validator.validate_gps(-1.2864, 36.5)
    assert result.valid == False
    assert result.uncertainty_flag == True
    assert "Longitude" in result.reason

def test_address_probability_scoring():
    validator = DataQualityValidator()
    
    # Test "near the big tree" - should score >0.5
    score = validator.score_address_probability("near the big tree")
    assert score > 0.5, f"Score {score} not > 0.5 for 'near the big tree'"
    
    # Test very specific address
    score = validator.score_address_probability("1234 Main Street, Nairobi")
    assert score > 0.1 and score < 0.9
    
    # Test empty address
    score = validator.score_address_probability("")
    assert score == 0.1
    
    # Test address with many keywords
    score = validator.score_address_probability("near the big red tree opposite the blue shop next to the market")
    assert score > 0.5

def test_temporal_consistency_validator():
    validator = DataQualityValidator()
    from datetime import datetime, timedelta
    
    now = datetime.utcnow()
    
    # Test valid timestamp (now)
    result = validator.validate_temporal_consistency(now)
    assert result.valid == True
    
    # Test future timestamp (should fail)
    future = now + timedelta(hours=1)
    result = validator.validate_temporal_consistency(future)
    assert result.valid == False
    assert "future" in result.reason
    
    # Test past timestamp (7 days ago - should fail)
    past = now - timedelta(days=8)
    result = validator.validate_temporal_consistency(past)
    assert result.valid == False
    assert "past" in result.reason
    
    # Test string timestamp
    result = validator.validate_temporal_consistency(now.isoformat())
    assert result.valid == True