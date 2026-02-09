"""
Intent Recognition Classifier
Perform user intent recognition based on embedding vector similarity
"""
import os
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
from .base_agent import BaseAgent

class IntentClassifier(BaseAgent):
    """Embedding-based intent recognition classifier"""
    
    def __init__(self):
        super().__init__("IntentClassifier")

        # Set OpenAI API key and configuration
        openai.api_base = ""
        openai.api_key = ""
        
        # Predefined intent vector library
        self.intent_examples = {
            'hydraulic_simulation': [
                "run hydraulic simulation",
                "calculate network pressure",
                "analyze flow distribution",
                "simulate hydraulic performance",
                "hydraulic analysis",
                "pressure calculation",
                "flow simulation",
                "perform hydraulic analysis",
                "calculate flow velocity"
            ],
            'network_analysis': [
                "analyze network structure",
                "view network info",
                "network topology analysis",
                "network connectivity",
                "pipe statistics",
                "node distribution",
                "network overview",
                "Network structure",
                "topology structure",
                "network composition"
            ],
            'partition_analysis': [
                "Network partition",
                "clustering analysis",
                "FCM clustering",
                "fuzzy clustering",
                "partition analysis",
                "region division",
                "network division",
                "node clustering",
                "partition optimization",
                "clustering",
                "partition",
                "divide into how many zones",
                "how many partitions",
                "cluster count",
                "outlier detection",
                "anomaly point detection",
                "partition visualization",
                "clustering visualization"
            ],
            'sensor_placement': [
                "Sensor placement",
                "sensor optimization",
                "pressure monitoring point placement",
                "monitoring point optimization",
                "sensor selection",
                "pressure sensor",
                "monitoring point selection",
                "sensor configuration",
                "monitoring network optimization",
                "sensor network",
                "pressure monitoring",
                "monitoring points",
                "resilience analysis",
                "sensor resilience",
                "fault analysis",
                "detection coverage",
                "sensitivity analysis",
                "detection point placement",
                "detection point optimization",
                "pressure detection",
                "monitoring system",
                "sensor system"
            ],
            'leak_detection': [
                "Leak detection",
                "leak analysis",
                "leakage detection",
                "leakage analysis",
                "anomaly detection",
                "fault detection",
                "leak identification",
                "leakage identification",
                "leak localization",
                "leakage localization",
                "leak monitoring",
                "leakage monitoring",
                "train leak model",
                "train detection model",
                "leak model training",
                "detection model training",
                "leak machine learning",
                "leak AI",
                "leak prediction",
                "leakage prediction",
                "anomaly prediction",
                "fault prediction",
                "leak diagnosis",
                "leakage diagnosis"
            ],
            'general_inquiry': [
                "what file is this",
                "file content introduction",
                "basic info query",
                "help info",
                "usage instructions",
                "feature introduction",
                "how to use",
                "what can you do"
            ]
        }
        
        # Calculate intent vectors
        self.intent_embeddings = None
        self._compute_intent_embeddings()
    
    def _get_embedding(self, text: str):
        """Get embedding vector for text"""
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=text
            )
            return np.array(response['data'][0]['embedding'])
        except Exception as e:
            self.log_error(f"Failed to get embedding: {e}")
            return None
    
    def _compute_intent_embeddings(self):
        """Calculate embedding vectors for each intent"""
        self.log_info("Starting to calculate intent vectors...")
        
        try:
            self.intent_embeddings = {}
            
            for intent, examples in self.intent_examples.items():
                embeddings = []
                
                for example in examples:
                    embedding = self._get_embedding(example)
                    if embedding is not None:
                        embeddings.append(embedding)
                
                if embeddings:
                    # Calculate average vector as intent vector
                    self.intent_embeddings[intent] = np.mean(embeddings, axis=0)
                    self.log_info(f"Intent '{intent}' vector calculation complete, sample count: {len(embeddings)}")
                else:
                    self.log_error(f"Intent '{intent}' has no valid embedding vectors")
            
            self.log_info("All intent vector calculation complete")
            
        except Exception as e:
            self.log_error(f"Failed to calculate intent vectors: {e}")
            self.intent_embeddings = {}
    
    def classify_intent(self, user_message: str):
        """Classify user intent"""
        if not self.intent_embeddings:
            self.log_error("Intent vectors not initialized")
            return {
                'intent': 'general_inquiry',
                'confidence': 0.0,
                'all_similarities': {},
                'error': 'Intent vectors not initialized'
            }
        
        try:
            # Get user message embedding
            user_embedding = self._get_embedding(user_message)
            if user_embedding is None:
                return {
                    'intent': 'general_inquiry',
                    'confidence': 0.0,
                    'all_similarities': {},
                    'error': 'Cannot get user message embedding'
                }
            
            # Calculate similarity with each intent
            similarities = {}
            user_embedding = user_embedding.reshape(1, -1)
            
            for intent, intent_vector in self.intent_embeddings.items():
                intent_vector = intent_vector.reshape(1, -1)
                similarity = cosine_similarity(user_embedding, intent_vector)[0][0]
                similarities[intent] = float(similarity)
            
            # Return intent with highest similarity and confidence
            best_intent = max(similarities, key=similarities.get)
            confidence = similarities[best_intent]
            
            self.log_info(f"Intent recognition result: {best_intent} (Confidence: {confidence:.3f})")
            
            return {
                'intent': best_intent,
                'confidence': confidence,
                'all_similarities': similarities
            }
            
        except Exception as e:
            self.log_error(f"Intent recognition failed: {e}")
            return {
                'intent': 'general_inquiry',
                'confidence': 0.0,
                'all_similarities': {},
                'error': str(e)
            }
    
    def process(self, user_message: str):
        """Process user message, return intent recognition result"""
        return self.classify_intent(user_message)
