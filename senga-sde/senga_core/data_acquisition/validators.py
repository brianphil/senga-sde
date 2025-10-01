# File: senga_core/data_acquisition/validators.py
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
import re
import logging
import datetime
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    valid: bool
    reason: str
    uncertainty_flag: bool = False
    confidence: float = 1.0

class DataQualityValidator:
    """
    Validates data quality with Senga-specific context.
    Quality Gate: 
    - GPS validator must reject 100% of out-of-bounds coordinates
    - Address scorer must return >0.5 for "near the big tree"
    """
    
    # Nairobi geographical bounds (from Implementation Guide)
    NAIROBI_BOUNDS = {
        'lat_min': -1.4441,
        'lat_max': -1.1634,
        'lng_min': 36.6573,
        'lng_max': 37.1058
    }
    
    # Informal address keywords (Senga-specific innovation)
    INFORMAL_ADDRESS_KEYWORDS = [
        "near", "opposite", "next to", "behind", "in front of",
        "tree", "shop", "market", "junction", "roundabout",
        "blue", "red", "big", "small", "old", "new",
        "kiosk", "mama", "dukas", "stage", "parking"
    ]
    
    def validate_gps(self, lat: float, lng: float) -> ValidationResult:
        """Validate GPS coordinates against Nairobi bounds"""
        if not isinstance(lat, (int, float)) or not isinstance(lng, (int, float)):
            return ValidationResult(
                valid=False,
                reason="GPS coordinates must be numeric",
                uncertainty_flag=True
            )
        
        if not (self.NAIROBI_BOUNDS['lat_min'] <= lat <= self.NAIROBI_BOUNDS['lat_max']):
            return ValidationResult(
                valid=False,
                reason=f"Latitude {lat} out of Nairobi bounds [{self.NAIROBI_BOUNDS['lat_min']}, {self.NAIROBI_BOUNDS['lat_max']}]",
                uncertainty_flag=True
            )
        
        if not (self.NAIROBI_BOUNDS['lng_min'] <= lng <= self.NAIROBI_BOUNDS['lng_max']):
            return ValidationResult(
                valid=False,
                reason=f"Longitude {lng} out of Nairobi bounds [{self.NAIROBI_BOUNDS['lng_min']}, {self.NAIROBI_BOUNDS['lng_max']}]",
                uncertainty_flag=True
            )
        
        return ValidationResult(
            valid=True,
            reason="GPS coordinates within Nairobi bounds",
            uncertainty_flag=False,
            confidence=1.0
        )
    
    def score_address_probability(self, address_text: str) -> float:
        """
        Score probability that address can be resolved.
        Uses heuristic based on presence of informal address keywords.
        This is a placeholder until probabilistic resolver is implemented.
        Quality Gate: Must return >0.5 for "near the big tree"
        """
        if not isinstance(address_text, str) or len(address_text.strip()) == 0:
            return 0.1
        
        text_lower = address_text.lower()
        keyword_matches = sum(1 for kw in self.INFORMAL_ADDRESS_KEYWORDS if kw in text_lower)
        max_possible_matches = len(self.INFORMAL_ADDRESS_KEYWORDS)
        
        # Calculate base score
        base_score = keyword_matches / max(1, max_possible_matches)
        
        # Boost score for very short addresses (likely landmarks)
        if len(address_text.split()) <= 3:
            base_score = min(1.0, base_score * 1.5)
        
        # Clamp between 0.1 and 0.9 (never 100% certain for informal addresses)
        final_score = max(0.1, min(0.9, base_score))
        
        logger.debug(f"Address '{address_text}' scored {final_score} (matched {keyword_matches}/{max_possible_matches} keywords)")
        return final_score
    
    def validate_temporal_consistency(self, timestamp: Any, reference_time: datetime = None) -> ValidationResult:
        """Validate that timestamp is reasonable (not in future, not too far in past)"""
        if reference_time is None:
            reference_time = datetime.utcnow()
        
        try:
            if isinstance(timestamp, str):
                from dateutil import parser
                timestamp = parser.parse(timestamp)
            elif isinstance(timestamp, (int, float)):
                from datetime import datetime
                timestamp = datetime.fromtimestamp(timestamp)
            
            if not isinstance(timestamp, datetime):
                return ValidationResult(
                    valid=False,
                    reason="Could not parse timestamp",
                    uncertainty_flag=True
                )
            
            # Check if timestamp is in future
            if timestamp > reference_time + datetime.timedelta(minutes=5):
                return ValidationResult(
                    valid=False,
                    reason="Timestamp is in the future",
                    uncertainty_flag=True
                )
            
            # Check if timestamp is too far in past (> 7 days)
            if timestamp < reference_time - timedelta(days=7):
                return ValidationResult(
                    valid=False,
                    reason="Timestamp is too far in the past",
                    uncertainty_flag=True
                )
            
            return ValidationResult(
                valid=True,
                reason="Timestamp is temporally consistent",
                uncertainty_flag=False,
                confidence=1.0
            )
            
        except Exception as e:
            return ValidationResult(
                valid=False,
                reason=f"Timestamp validation error: {e}",
                uncertainty_flag=True
            )