# File: senga_core/senga_innovations/models.py
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from scipy.stats import dirichlet, multivariate_normal
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

class HiddenMarkovModel:
    """
    Continuous Hidden Markov Model for customer availability patterns.
    Quality Gate: Must actually learn from data, not use static templates.
    """
    
    def __init__(self, n_components: int = 3, n_features: int = 4):
        self.n_components = n_components  # e.g., Available, Busy, Unavailable
        self.n_features = n_features      # e.g., hour, day, month, traffic
        
        # Initialize parameters
        self.initial_state_probs = np.ones(n_components) / n_components
        self.transition_matrix = np.ones((n_components, n_components)) / n_components
        self.emission_means = np.random.randn(n_components, n_features) * 0.1
        self.emission_covs = np.array([np.eye(n_features) * 0.5 for _ in range(n_components)])
        
        self.is_fitted = False
        self.training_history = []
    
    def fit(self, sequences: List[np.ndarray], max_iter: int = 100, tol: float = 1e-4):
        """
        Fit HMM using Baum-Welch algorithm (EM for HMM).
        Sequences: List of observation sequences [seq1, seq2, ...]
        Each sequence: np.array of shape (T, n_features)
        """
        if len(sequences) == 0:
            raise ValueError("No training sequences provided")
        
        prev_log_likelihood = -np.inf
        
        for iteration in range(max_iter):
            total_log_likelihood = 0.0
            accumulated_stats = self._initialize_accumulators()
            
            # E-step: Forward-backward algorithm for each sequence
            for seq in sequences:
                if len(seq) == 0:
                    continue
                
                # Forward probabilities
                alpha, log_likelihood = self._forward(seq)
                total_log_likelihood += log_likelihood
                
                # Backward probabilities
                beta = self._backward(seq)
                
                # Update accumulators
                self._accumulate_statistics(seq, alpha, beta, accumulated_stats)
            
            # M-step: Update parameters
            self._update_parameters(accumulated_stats, len(sequences))
            
            # Check convergence
            avg_log_likelihood = total_log_likelihood / len(sequences)
            self.training_history.append({
                'iteration': iteration,
                'avg_log_likelihood': avg_log_likelihood,
                'param_change': np.linalg.norm(self.transition_matrix - accumulated_stats['transition_counts'])
            })
            
            if abs(avg_log_likelihood - prev_log_likelihood) < tol:
                logger.info(f"HMM converged at iteration {iteration}")
                break
            
            prev_log_likelihood = avg_log_likelihood
        
        self.is_fitted = True
        logger.info(f"HMM training completed: {len(sequences)} sequences, final log-likelihood: {avg_log_likelihood:.4f}")
    
    def _initialize_accumulators(self) -> Dict[str, Any]:
        """Initialize accumulators for Baum-Welch algorithm"""
        return {
            'initial_counts': np.zeros(self.n_components),
            'transition_counts': np.zeros((self.n_components, self.n_components)),
            'emission_sums': np.zeros((self.n_components, self.n_features)),
            'emission_counts': np.zeros(self.n_components),
            'emission_outer_products': np.zeros((self.n_components, self.n_features, self.n_features))
        }
    
    def _forward(self, sequence: np.ndarray) -> Tuple[np.ndarray, float]:
        """Forward algorithm to compute alpha probabilities"""
        T = len(sequence)
        alpha = np.zeros((T, self.n_components))
        
        # Initialize
        for i in range(self.n_components):
            emission_prob = self._emission_probability(sequence[0], i)
            alpha[0, i] = self.initial_state_probs[i] * emission_prob
        
        # Normalize
        alpha[0] = alpha[0] / np.sum(alpha[0])
        
        # Recursion
        for t in range(1, T):
            for j in range(self.n_components):
                alpha[t, j] = sum(
                    alpha[t-1, i] * self.transition_matrix[i, j] * self._emission_probability(sequence[t], j)
                    for i in range(self.n_components)
                )
            # Normalize
            if np.sum(alpha[t]) > 0:
                alpha[t] = alpha[t] / np.sum(alpha[t])
            else:
                alpha[t] = np.ones(self.n_components) / self.n_components
        
        log_likelihood = np.log(np.sum(alpha[-1])) if np.sum(alpha[-1]) > 0 else -np.inf
        return alpha, log_likelihood
    
    def _backward(self, sequence: np.ndarray) -> np.ndarray:
        """Backward algorithm to compute beta probabilities"""
        T = len(sequence)
        beta = np.zeros((T, self.n_components))
        
        # Initialize
        beta[-1] = np.ones(self.n_components)
        
        # Recursion
        for t in range(T-2, -1, -1):
            for i in range(self.n_components):
                beta[t, i] = sum(
                    self.transition_matrix[i, j] * self._emission_probability(sequence[t+1], j) * beta[t+1, j]
                    for j in range(self.n_components)
                )
            # Normalize
            if np.sum(beta[t]) > 0:
                beta[t] = beta[t] / np.sum(beta[t])
            else:
                beta[t] = np.ones(self.n_components) / self.n_components
        
        return beta
    
    def _emission_probability(self, observation: np.ndarray, state: int) -> float:
        """Compute emission probability for observation given state"""
        try:
            mvn = multivariate_normal(mean=self.emission_means[state], cov=self.emission_covs[state])
            prob = mvn.pdf(observation)
            return max(prob, 1e-10)  # Avoid zero probabilities
        except:
            return 1e-10
    
    def _accumulate_statistics(self, sequence: np.ndarray, alpha: np.ndarray, beta: np.ndarray, stats: Dict[str, Any]):
        """Accumulate statistics for M-step"""
        T = len(sequence)
        
        # Gamma: P(state_t = i | observations)
        gamma = alpha * beta
        gamma = gamma / np.sum(gamma, axis=1, keepdims=True)
        
        # Xi: P(state_t = i, state_{t+1} = j | observations)
        xi = np.zeros((T-1, self.n_components, self.n_components))
        for t in range(T-1):
            for i in range(self.n_components):
                for j in range(self.n_components):
                    xi[t, i, j] = alpha[t, i] * self.transition_matrix[i, j] * self._emission_probability(sequence[t+1], j) * beta[t+1, j]
            # Normalize
            if np.sum(xi[t]) > 0:
                xi[t] = xi[t] / np.sum(xi[t])
        
        # Accumulate
        stats['initial_counts'] += gamma[0]
        for t in range(T-1):
            stats['transition_counts'] += xi[t]
        for t in range(T):
            for i in range(self.n_components):
                stats['emission_sums'][i] += gamma[t, i] * sequence[t]
                stats['emission_counts'][i] += gamma[t, i]
                stats['emission_outer_products'][i] += gamma[t, i] * np.outer(sequence[t], sequence[t])
    
    def _update_parameters(self, stats: Dict[str, Any], n_sequences: int):
        """Update HMM parameters based on accumulated statistics"""
        # Update initial state probabilities
        if np.sum(stats['initial_counts']) > 0:
            self.initial_state_probs = stats['initial_counts'] / np.sum(stats['initial_counts'])
        
        # Update transition matrix
        for i in range(self.n_components):
            if np.sum(stats['transition_counts'][i]) > 0:
                self.transition_matrix[i] = stats['transition_counts'][i] / np.sum(stats['transition_counts'][i])
        
        # Update emission parameters
        for i in range(self.n_components):
            if stats['emission_counts'][i] > 0:
                self.emission_means[i] = stats['emission_sums'][i] / stats['emission_counts'][i]
                # Update covariance
                mean_outer = np.outer(self.emission_means[i], self.emission_means[i])
                cov = (stats['emission_outer_products'][i] / stats['emission_counts'][i]) - mean_outer
                # Ensure positive definite
                cov = cov + np.eye(self.n_features) * 1e-6
                self.emission_covs[i] = cov
    
    def predict_proba(self, sequence: np.ndarray) -> np.ndarray:
        """Predict state probabilities for sequence"""
        if not self.is_fitted:
            raise ValueError("HMM must be fitted before prediction")
        
        if len(sequence.shape) == 1:
            sequence = sequence.reshape(1, -1)
        
        alpha, _ = self._forward(sequence)
        return alpha[-1]  # Return probabilities for last state

class BayesianAddressModel:
    """
    Bayesian model for probabilistic address resolution.
    Quality Gate: Must achieve >80% accuracy within 500m.
    """
    
    def __init__(self):
        # Prior distributions for landmark locations
        self.landmark_priors = {}  # landmark -> (mean_lat, mean_lng, cov_matrix)
        self.context_likelihood_models = {}  # context_type -> parameters
        
        # Default prior for unknown landmarks
        self.default_prior = {
            'mean': np.array([-1.2864, 36.8172]),  # Nairobi center
            'cov': np.array([[0.01, 0], [0, 0.01]])  # ~1.1km radius
        }
    
    def fit(self, training_data: List[Dict[str, Any]]):
        """Fit model using training data: [{'address_text': str, 'gps': (lat, lng), 'context': dict}, ...]"""
        from collections import defaultdict
        
        # Group data by landmarks
        landmark_data = defaultdict(list)
        
        for item in training_data:
            address_text = item['address_text'].lower()
            lat, lng = item['gps']
            context = item.get('context', {})
            
            # Extract landmarks (simple keyword extraction)
            landmarks = self._extract_landmarks(address_text)
            for landmark in landmarks:
                landmark_data[landmark].append({
                    'location': np.array([lat, lng]),
                    'context': context
                })
        
        # Fit priors for each landmark
        for landmark, data in landmark_data.items():
            locations = np.array([d['location'] for d in data])
            if len(locations) >= 2:
                mean = np.mean(locations, axis=0)
                cov = np.cov(locations, rowvar=False) + np.eye(2) * 1e-6  # Regularization
                self.landmark_priors[landmark] = {
                    'mean': mean,
                    'cov': cov,
                    'count': len(data)
                }
            else:
                # Single point - use default covariance
                self.landmark_priors[landmark] = {
                    'mean': locations[0],
                    'cov': self.default_prior['cov'],
                    'count': len(data)
                }
        
        logger.info(f"BayesianAddressModel fitted with {len(self.landmark_priors)} landmarks")
    
    def _extract_landmarks(self, address_text: str) -> List[str]:
        """Extract landmarks from address text"""
        # Simple keyword extraction - in production, use NLP
        keywords = [
            "tree", "shop", "market", "junction", "roundabout", "school", "church",
            "mosque", "hospital", "clinic", "pharmacy", "bank", "atm", "station",
            "park", "stadium", "hotel", "restaurant", "cafe", "bar", "salon",
            "kiosk", "dukas", "stage", "parking", "bridge", "river", "road"
        ]
        
        landmarks = []
        for kw in keywords:
            if kw in address_text.lower():
                landmarks.append(kw)
        
        return landmarks if landmarks else ["unknown"]
    
    def predict_location(self, address_text: str, context: Dict[str, Any] = None) -> Tuple[np.ndarray, float]:
        """Predict location and confidence"""
        landmarks = self._extract_landmarks(address_text)
        candidates = []
        
        for landmark in landmarks:
            if landmark in self.landmark_priors:
                prior = self.landmark_priors[landmark]
                mean, cov = prior['mean'], prior['cov']
            else:
                mean, cov = self.default_prior['mean'], self.default_prior['cov']
            
            # Calculate likelihood based on context (simplified)
            likelihood = self._calculate_context_likelihood(mean, context or {})
            
            # Sample from posterior (mean of posterior)
            posterior_mean = mean  # In full Bayesian, this would be updated
            posterior_cov = cov
            
            # Confidence based on covariance determinant
            confidence = 1.0 / (np.sqrt(np.linalg.det(cov)) + 1e-6)
            candidates.append((posterior_mean, confidence * likelihood, posterior_cov))
        
        if not candidates:
            return self.default_prior['mean'], 0.1
        
        # Select best candidate
        best_mean, best_confidence, best_cov = max(candidates, key=lambda x: x[1])
        
        # Normalize confidence
        total_confidence = sum(c[1] for c in candidates)
        normalized_confidence = best_confidence / max(total_confidence, 1e-6)
        
        return best_mean, min(0.9, max(0.1, normalized_confidence))  # Clamp confidence
    
    def _calculate_context_likelihood(self, location: np.ndarray, context: Dict[str, Any]) -> float:
        """Calculate likelihood based on context"""
        likelihood = 1.0
        
        # Adjust based on location type
        if 'location_type' in context:
            loc_type = context['location_type']
            if loc_type == 'cbd':
                # CBD areas have higher density, lower uncertainty
                likelihood *= 1.2
            elif loc_type == 'rural':
                # Rural areas have higher uncertainty
                likelihood *= 0.8
        
        return likelihood

class ConnectivityUncertaintyModel:
    """
    Model for connectivity uncertainty in offline scenarios.
    Quality Gate: Must quantify uncertainty for offline decision making.
    """
    
    def __init__(self):
        self.connectivity_patterns = {
            'online': {'reliability': 0.95, 'latency': 0.1},
            'degraded': {'reliability': 0.75, 'latency': 1.0},
            'offline': {'reliability': 0.30, 'latency': 10.0}
        }
        self.time_patterns = {}  # hour -> reliability adjustment
    
    def predict_connectivity_reliability(self, location_type: str, time_of_day: int) -> float:
        """Predict connectivity reliability based on location and time"""
        base_reliability = self.connectivity_patterns.get(
            'degraded',  # Assume degraded as default
            self.connectivity_patterns['offline']
        )['reliability']
        
        # Adjust based on location
        location_adjustment = {
            'cbd': 1.2,
            'urban': 1.0,
            'peri_urban': 0.8,
            'rural': 0.6,
            'remote': 0.4
        }.get(location_type, 1.0)
        
        # Adjust based on time (simplified)
        time_adjustment = 1.0
        if 7 <= time_of_day <= 9 or 17 <= time_of_day <= 19:  # Rush hours
            time_adjustment = 0.8
        elif 22 <= time_of_day or time_of_day <= 5:  # Night hours
            time_adjustment = 0.9
        
        reliability = base_reliability * location_adjustment * time_adjustment
        return max(0.1, min(0.95, reliability))  # Clamp between 0.1 and 0.95