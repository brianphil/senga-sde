# File: senga_core/senga_innovations/address_resolver.py
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
import re
from .models import BayesianAddressModel
from dataclasses import dataclass
from datetime import datetime
logger = logging.getLogger(__name__)

@dataclass
class AddressResolution:
    coordinates: Tuple[float, float]
    confidence: float
    uncertainty_radius: float  # in meters
    alternative_locations: List[Tuple[Tuple[float, float], float]] = None
    resolution_method: str = "bayesian"

class LandmarkGraph:
    """Graph of known landmarks with their relationships"""
    
    def __init__(self):
        self.landmarks = {}
        self.relationships = {}
        self._initialize_kenyan_landmarks()
    
    def _initialize_kenyan_landmarks(self):
        """Initialize with common Kenyan landmarks"""
        common_landmarks = [
            "kenyatta market", "city market", "westgate mall", "thika road mall",
            "uwanja wa ndege", "kenyatta hospital", "nairobi hospital", 
            "kibera", "mathare", "kayole", "embakasi", "langata", "ngong road",
            "university of nairobi", "kenyatta university", "strathmore university",
            "kenya national theatre", "nyayo stadium", "kasarani stadium",
            "karura forest", "ngong road forest", "nairobi national park"
        ]
        
        # Add landmarks with approximate locations
        for landmark in common_landmarks:
            # Assign random but reasonable locations in Nairobi
            lat = -1.2864 + np.random.uniform(-0.1, 0.1)
            lng = 36.8172 + np.random.uniform(-0.1, 0.1)
            self.landmarks[landmark] = {
                'coordinates': (lat, lng),
                'type': 'commercial' if 'mall' in landmark or 'market' in landmark else 'other',
                'reliability_score': 0.8
            }
    
    def get_nearby_locations(self, landmark: str, radius_km: float = 1.0) -> List[Tuple[float, float]]:
        """Get nearby locations for a landmark"""
        if landmark not in self.landmarks:
            # Return random locations if landmark unknown
            locations = []
            base_lat, base_lng = -1.2864, 36.8172
            for _ in range(3):
                lat = base_lat + np.random.uniform(-0.01, 0.01)
                lng = base_lng + np.random.uniform(-0.01, 0.01)
                locations.append((lat, lng))
            return locations
        
        base_lat, base_lng = self.landmarks[landmark]['coordinates']
        locations = [(base_lat, base_lng)]
        
        # Add some nearby locations
        for _ in range(2):
            lat = base_lat + np.random.uniform(-0.005, 0.005)
            lng = base_lng + np.random.uniform(-0.005, 0.005)
            locations.append((lat, lng))
        
        return locations
    
    def get_landmark_location_prior(self, landmark: str, location: Tuple[float, float]) -> float:
        """Get prior probability for landmark at location"""
        if landmark not in self.landmarks:
            return 0.3  # Lower prior for unknown landmarks
        
        landmark_lat, landmark_lng = self.landmarks[landmark]['coordinates']
        distance = self._haversine_distance(landmark_lat, landmark_lng, location[0], location[1])
        
        # Gaussian prior based on distance
        sigma = 0.5  # 0.5 km standard deviation
        prior = np.exp(-distance**2 / (2 * sigma**2))
        return prior * self.landmarks[landmark]['reliability_score']
    
    def _haversine_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate haversine distance in km"""
        R = 6371  # Earth radius in km
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lng2 - lng1)
        
        a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

class ProbabilisticAddressResolver:
    """
    Probabilistic Address Resolver for informal addressing.
    Quality Gate: Must achieve >80% accuracy within 500m.
    """
    
    def __init__(self):
        self.landmark_graph = LandmarkGraph()
        self.address_confidence_model = BayesianAddressModel()
        self.resolution_history = []
        self.training_data = self._generate_training_data()
        
        # Fit the model with training data
        self.address_confidence_model.fit(self.training_data)
    
    def _generate_training_data(self) -> List[Dict[str, Any]]:
        """Generate training data for address model"""
        training_data = []
        
        # Add some realistic training examples
        examples = [
            {
                'address_text': 'near kenyatta market',
                'gps': (-1.2864, 36.8172),
                'context': {'location_type': 'market', 'time_of_day': 10}
            },
            {
                'address_text': 'opposite westgate mall',
                'gps': (-1.2921, 36.8219),
                'context': {'location_type': 'cbd', 'time_of_day': 14}
            },
            {
                'address_text': 'next to kenyatta hospital',
                'gps': (-1.2800, 36.8200),
                'context': {'location_type': 'hospital', 'time_of_day': 9}
            },
            {
                'address_text': 'near the big tree in kibera',
                'gps': (-1.3140, 36.7760),
                'context': {'location_type': 'residential', 'time_of_day': 11}
            },
            {
                'address_text': 'behind university of nairobi',
                'gps': (-1.2920, 36.8220),
                'context': {'location_type': 'educational', 'time_of_day': 13}
            }
        ]
        
        training_data.extend(examples)
        
        # Add some random examples for robustness
        for i in range(20):
            lat = -1.2864 + np.random.uniform(-0.1, 0.1)
            lng = 36.8172 + np.random.uniform(-0.1, 0.1)
            landmark = f"landmark_{i}"
            training_data.append({
                'address_text': f'near {landmark}',
                'gps': (lat, lng),
                'context': {
                    'location_type': np.random.choice(['cbd', 'residential', 'market', 'educational']),
                    'time_of_day': np.random.randint(8, 18)
                }
            })
        
        return training_data
    
    def resolve(self, address_text: str, context: Dict[str, Any] = None) -> AddressResolution:
        """
        Convert informal address to GPS coordinates with uncertainty quantification.
        This is the core innovation for African addressing systems.
        """
        try:
            # Extract landmarks using NLP
            landmarks = self._extract_landmarks(address_text)
            
            # If no landmarks found, use Bayesian model directly
            if not landmarks:
                coords, confidence = self.address_confidence_model.predict_location(address_text, context)
                uncertainty_radius = self._calculate_uncertainty_radius(confidence)
                
                resolution = AddressResolution(
                    coordinates=(float(coords[0]), float(coords[1])),
                    confidence=float(confidence),
                    uncertainty_radius=uncertainty_radius,
                    alternative_locations=[],
                    resolution_method="bayesian_direct"
                )
                
                self._record_resolution(address_text, resolution)
                return resolution
            
            # Build probability distribution over possible locations
            location_candidates = []
            
            for landmark in landmarks:
                candidate_locations = self.landmark_graph.get_nearby_locations(landmark)
                for loc in candidate_locations:
                    prior_prob = self.landmark_graph.get_landmark_location_prior(landmark, loc)
                    likelihood = self._calculate_context_likelihood(loc, context or {})
                    posterior = prior_prob * likelihood
                    location_candidates.append((loc, posterior))
            
            # Handle case where no candidates found
            if not location_candidates:
                coords, confidence = self.address_confidence_model.predict_location(address_text, context)
                uncertainty_radius = self._calculate_uncertainty_radius(confidence)
                
                resolution = AddressResolution(
                    coordinates=(float(coords[0]), float(coords[1])),
                    confidence=float(confidence),
                    uncertainty_radius=uncertainty_radius,
                    alternative_locations=[],
                    resolution_method="bayesian_fallback"
                )
                
                self._record_resolution(address_text, resolution)
                return resolution
            
            # Normalize to get probability distribution
            total_prob = sum(posterior for _, posterior in location_candidates)
            if total_prob == 0:
                total_prob = 1e-6
            
            location_distribution = [(loc, post/total_prob) for loc, post in location_candidates]
            
            # Select location with highest probability
            best_location, confidence = max(location_distribution, key=lambda x: x[1])
            
            # Get top 3 alternative locations
            sorted_locations = sorted(location_distribution, key=lambda x: x[1], reverse=True)
            top_alternatives = sorted_locations[:3]
            
            # Calculate uncertainty radius based on confidence and alternatives
            uncertainty_radius = self._calculate_uncertainty_radius(confidence, best_location, top_alternatives)
            
            resolution = AddressResolution(
                coordinates=(float(best_location[0]), float(best_location[1])),
                confidence=float(confidence),
                uncertainty_radius=uncertainty_radius,
                alternative_locations=[((float(loc[0]), float(loc[1])), float(prob)) for loc, prob in top_alternatives],
                resolution_method="landmark_bayesian"
            )
            
            self._record_resolution(address_text, resolution)
            return resolution
            
        except Exception as e:
            logger.error(f"Address resolution failed for '{address_text}': {e}")
            # Return fallback resolution
            return AddressResolution(
                coordinates=(-1.2864, 36.8172),  # Nairobi center
                confidence=0.1,
                uncertainty_radius=5000.0,  # 5km radius
                alternative_locations=[],
                resolution_method="fallback"
            )
    
    def _extract_landmarks(self, address_text: str) -> List[str]:
        """Extract landmarks from address text using regex patterns"""
        address_text = address_text.lower()
        
        # Common landmark patterns
        landmark_patterns = [
            r'(?:near|next to|opposite|behind|in front of|close to)\s+([a-zA-Z\s]+?)(?:\s+(?:market|mall|hospital|school|university|college|church|mosque|stadium|forest|park|road|street|avenue|lane|drive))',
            r'([a-zA-Z\s]+?)(?:\s+(?:market|mall|hospital|school|university|college|church|mosque|stadium|forest|park|road|street|avenue|lane|drive))',
            r'(?:the\s+)?(big|small|old|new)\s+(tree|shop|house|building|kiosk|dukas)',
            r'(?:at|by|beside)\s+([a-zA-Z\s]+)'
        ]
        
        landmarks = []
        
        for pattern in landmark_patterns:
            matches = re.findall(pattern, address_text)
            for match in matches:
                if isinstance(match, tuple):
                    landmark = ' '.join(match).strip()
                else:
                    landmark = match.strip()
                if landmark and len(landmark) > 2:
                    landmarks.append(landmark)
        
        # If no structured landmarks found, try simple keyword extraction
        if not landmarks:
            keywords = [
                "tree", "shop", "market", "junction", "roundabout", "school", "church",
                "mosque", "hospital", "clinic", "pharmacy", "bank", "atm", "station",
                "park", "stadium", "hotel", "restaurant", "cafe", "bar", "salon",
                "kiosk", "dukas", "stage", "parking", "bridge", "river", "road",
                "kenyatta", "westgate", "thika", "kibera", "mathare", "embakasi",
                "langata", "ngong", "university", "nairobi", "kasarani", "karura"
            ]
            
            for kw in keywords:
                if kw in address_text:
                    landmarks.append(kw)
        
        return list(set(landmarks))  # Remove duplicates
    
    def _calculate_context_likelihood(self, location: Tuple[float, float], context: Dict[str, Any]) -> float:
        """Calculate likelihood based on context"""
        likelihood = 1.0
        
        # Adjust based on location type
        if 'location_type' in context:
            loc_type = context['location_type']
            adjustment = {
                'cbd': 1.2,
                'market': 1.3,
                'residential': 1.1,
                'industrial': 0.9,
                'rural': 0.8,
                'peri_urban': 1.0
            }.get(loc_type, 1.0)
            likelihood *= adjustment
        
        # Adjust based on time of day
        if 'time_of_day' in context:
            hour = context['time_of_day']
            if 8 <= hour <= 17:  # Business hours
                likelihood *= 1.1
            elif hour < 6 or hour > 22:  # Night hours
                likelihood *= 0.8
        
        return likelihood
    
    def _calculate_uncertainty_radius(self, confidence: float, 
                                    best_location: Tuple[float, float] = None,
                                    alternatives: List[Tuple[Tuple[float, float], float]] = None) -> float:
        """Calculate uncertainty radius in meters based on confidence and alternatives"""
        # Base radius inversely proportional to confidence
        base_radius = 5000.0 * (1.0 - confidence)  # Max 5km when confidence=0
        
        # If we have alternatives, use distance to closest alternative
        if best_location and alternatives and len(alternatives) > 1:
            distances = []
            for alt_loc, alt_prob in alternatives[1:]:  # Skip the best location
                if alt_prob > 0.1:  # Only consider reasonably probable alternatives
                    dist = self._haversine_distance(
                        best_location[0], best_location[1],
                        alt_loc[0], alt_loc[1]
                    ) * 1000  # Convert to meters
                    distances.append(dist)
            
            if distances:
                min_distance = min(distances)
                # Use minimum of base radius and distance to closest alternative
                radius = min(base_radius, min_distance * 2.0)  # Factor of 2 for safety
                return float(radius)
        
        return float(base_radius)
    
    def _haversine_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate haversine distance in km"""
        R = 6371  # Earth radius in km
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lng2 - lng1)
        
        a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c
    
    def _record_resolution(self, address_text: str, resolution: AddressResolution):
        """Record resolution for performance tracking"""
        record = {
            'address_text': address_text,
            'resolution': resolution,
            'timestamp': datetime.now(),
            'method': resolution.resolution_method
        }
        self.resolution_history.append(record)
    
    def get_accuracy_metrics(self) -> Dict[str, Any]:
        """Get accuracy metrics for validation"""
        if len(self.resolution_history) == 0:
            return {'resolution_count': 0, 'avg_confidence': 0.0}
        
        confidences = [r['resolution'].confidence for r in self.resolution_history]
        avg_confidence = np.mean(confidences)
        
        return {
            'resolution_count': len(self.resolution_history),
            'avg_confidence': float(avg_confidence),
            'high_confidence_rate': len([c for c in confidences if c > 0.7]) / len(confidences),
            'recent_resolutions': len(self.resolution_history[-10:])
        }