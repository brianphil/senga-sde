# File: tests/test_powell_framework/test_features.py
import pytest
import numpy as np
from senga_core.state_space.state import StateSpace, Route, Vehicle, Customer, Environment, LearningState
from senga_core.powell_framework.features import SengaFeatureExtractor, FeatureExtractionError

def create_test_state():
    """Create a test state for feature extraction testing"""
    routes = [Route("R001", [(-1.2864, 36.8172)], 10.0, 30.0, 0.8)]
    fleet = [Vehicle("V01", 2000.0, 0.5, "good", (-1.2864, 36.8172), 0.7)]
    customers = [Customer("C001", (-1.2864, 36.8172), 100.0, (9, 17), 0.8, "mpesa")]
    env = Environment(0.5, "clear", "online", 12, 1)
    learning = LearningState(0.85, "v1.0", None, 0.1)
    
    return StateSpace(routes, fleet, customers, env, learning)

def test_feature_extractor_initialization():
    extractor = SengaFeatureExtractor()
    assert extractor is not None

def test_feature_extraction():
    extractor = SengaFeatureExtractor()
    state = create_test_state()
    
    features = extractor.extract_strategic_features(state)
    
    assert isinstance(features, np.ndarray)
    assert len(features) == 14  # Must match expected dimensionality
    assert all(isinstance(f, float) for f in features)
    assert all(0.0 <= f <= 1.0 for f in features[:11])  # First 11 features should be normalized

def test_infrastructure_reliability():
    extractor = SengaFeatureExtractor()
    env = Environment(0.5, "clear", "online", 12, 1)
    
    reliability = extractor.calculate_infrastructure_reliability(env)
    assert isinstance(reliability, float)
    assert 0.0 <= reliability <= 1.0

def test_cultural_timing_efficiency():
    extractor = SengaFeatureExtractor()
    state = create_test_state()
    
    efficiency = extractor.assess_cultural_timing_efficiency(state)
    assert isinstance(efficiency, float)
    assert 0.0 <= efficiency <= 1.0