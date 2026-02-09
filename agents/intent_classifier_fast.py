"""
Fast Intent Recognition Classifier
Use caching mechanism to avoid repeated embedding vector calculation
"""
import os
import json
import pickle
import hashlib
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
from .base_agent import BaseAgent

class FastIntentClassifier(BaseAgent):
    """Cache-based fast intent recognition classifier"""
    
    def __init__(self):
        super().__init__("FastIntentClassifier")

        # Set OpenAI API key and configuration
        openai.api_base = ""
        openai.api_key = ""
        
        # Cache file path
        self.cache_dir = "cache"
        self.embeddings_cache_file = os.path.join(self.cache_dir, "intent_embeddings.pkl")
        self.examples_hash_file = os.path.join(self.cache_dir, "examples_hash.txt")
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Predefined intent vector library (optimized version - reduced examples while maintaining accuracy)
        self.intent_examples = {
            'hydraulic_simulation': [
                "Perform hydraulic calculation", "Run hydraulic simulation", "Calculate network pressure", "Analyze flow distribution", "Hydraulic analysis",
                "Pressure calculation", "Flow simulation", "Hydraulic simulation", "Network simulation", "Pressure analysis"
            ],
            'network_analysis': [
                "Analyze network structure", "View network info", "Network topology analysis", "Network overview", "Network structure",
                "Topology structure", "Network composition", "Node distribution", "Pipe statistics"
            ],
            'partition_analysis': [
                "Network partition", "Network Partition", "Clustering analysis", "Partition analysis", "Region division",
                "Divide into how many areas", "Divide into how many partitions", "Cluster into how many", "Network division", "Node clustering",
                "Partition optimization", "Outlier detection", "Partition visualization", "FCM clustering", "Fuzzy clustering"
            ],
            'sensor_placement': [
                "Sensor placement", "Sensor optimization", "Pressure monitoring point layout", "Monitoring point optimization", "Sensor selection",
                "Pressure sensor", "Monitoring point selection", "Sensor configuration", "Monitoring network optimization", "Resilience analysis",
                "Detection coverage", "Sensitivity analysis", "Detection point layout", "Pressure detection"
            ],
            'leak_detection': [
                "Leak detection", "Leakage analysis", "Leak testing", "Anomaly detection", "Train leak model", "Leak model training",
                "Fault detection", "Leak identification", "Leak recognition", "Leak localization", "Leak monitoring", "Detection model training",
                "Leak machine learning", "Leak prediction", "Anomaly prediction", "Fault prediction"
            ],
            'general_inquiry': [
                "What is this file", "File content introduction", "Basic info query", "Help info", "Usage instructions", "Function introduction"
            ]
        }
        
        # Calculate intent vectors (using cache)
        self.intent_embeddings = None
        self._load_or_compute_embeddings()
    
    def _compute_examples_hash(self):
        """Calculate hash value of example texts to detect changes"""
        examples_str = json.dumps(self.intent_examples, sort_keys=True)
        return hashlib.md5(examples_str.encode()).hexdigest()
    
    def _should_recompute_embeddings(self):
        """Check if embeddings need to be recomputed"""
        if not os.path.exists(self.embeddings_cache_file):
            return True
        
        if not os.path.exists(self.examples_hash_file):
            return True
        
        # Check if examples have changed
        current_hash = self._compute_examples_hash()
        try:
            with open(self.examples_hash_file, 'r') as f:
                cached_hash = f.read().strip()
            return current_hash != cached_hash
        except:
            return True
    
    def _save_embeddings_cache(self):
        """Save embeddings to cache file"""
        try:
            with open(self.embeddings_cache_file, 'wb') as f:
                pickle.dump(self.intent_embeddings, f)
            
            # Save hash value
            current_hash = self._compute_examples_hash()
            with open(self.examples_hash_file, 'w') as f:
                f.write(current_hash)
            
            self.log_info("Embeddings cache saved")
        except Exception as e:
            self.log_error(f"Failed to save embeddings cache: {e}")
    
    def _load_embeddings_cache(self):
        """Load embeddings from cache file"""
        try:
            with open(self.embeddings_cache_file, 'rb') as f:
                self.intent_embeddings = pickle.load(f)
            self.log_info("Successfully loaded embeddings from cache")
            return True
        except Exception as e:
            self.log_error(f"Failed to load embeddings cache: {e}")
            return False
    
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
            total_examples = sum(len(examples) for examples in self.intent_examples.values())
            processed = 0
            
            for intent, examples in self.intent_examples.items():
                embeddings = []
                
                for example in examples:
                    embedding = self._get_embedding(example)
                    if embedding is not None:
                        embeddings.append(embedding)
                    
                    processed += 1
                    if processed % 5 == 0:
                        self.log_info(f"Progress: {processed}/{total_examples}")
                
                if embeddings:
                    # Calculate average vector as intent vector
                    self.intent_embeddings[intent] = np.mean(embeddings, axis=0)
                    self.log_info(f"Intent '{intent}' vector calculation complete, sample count: {len(embeddings)}")
                else:
                    self.log_error(f"Intent '{intent}' has no valid embedding vectors")
            
            self.log_info("All intent vector calculation complete")
            
            # Save to cache
            self._save_embeddings_cache()
            
        except Exception as e:
            self.log_error(f"Failed to calculate intent vectors: {e}")
            self.intent_embeddings = {}
    
    def _load_or_compute_embeddings(self):
        """Load or compute embeddings"""
        if self._should_recompute_embeddings():
            self.log_info("Need to recompute embeddings")
            self._compute_intent_embeddings()
        else:
            self.log_info("Loading embeddings from cache")
            if not self._load_embeddings_cache():
                self.log_info("Cache loading failed, recomputing")
                self._compute_intent_embeddings()
    
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
