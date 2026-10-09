"""
Educational Embeddings Module
=============================
Explains how AI maps concepts to numbers:
1. Vectors: A list of numbers [v1, v2, ... vn] where each dimension represents an abstract concept.
2. Cosine Similarity: Measuring the angle between two vectors to determine semantic closeness.
3. Vector Arithmetic: How math works on concepts: King - Man + Woman = Queen.
4. Dimensionality Reduction (PCA): Compressing 16D vectors to 2D for human visualization.
"""

import math
import numpy as np
from typing import List, Dict, Any, Tuple


# Semantic axes for our 16-dimensional educational space:
# [0: Living, 1: Human, 2: Royalty/Power, 3: Female, 4: Tech/Digital, 
#  5: Animal, 6: Canine, 7: Feline, 8: Food, 9: Emotion_Positive,
#  10: Abstract, 11: Action/Verb, 12: Vehicle/Transport, 13: Size_Large,
#  14: Science/Math, 15: Natural]
AXIS_NAMES = [
    "Living", "Human", "Royalty/Power", "Female", "Tech/Digital",
    "Animal", "Canine", "Feline", "Food", "Emotion_Positive",
    "Abstract", "Action/Verb", "Vehicle", "Size_Large",
    "Science/Math", "Natural"
]

# Base seed word vectors
SEED_VECTORS: Dict[str, List[float]] = {
    # Royalty & Gender
    "king":     [0.9, 0.9, 0.95, -0.8, -0.9, -0.9, -0.9, -0.9, -0.9, 0.2, 0.3, -0.5, -0.8, 0.4, -0.7, -0.5],
    "queen":    [0.9, 0.9, 0.95,  0.8, -0.9, -0.9, -0.9, -0.9, -0.9, 0.3, 0.3, -0.5, -0.8, 0.2, -0.7, -0.5],
    "man":      [0.9, 0.9, 0.1,  -0.8, -0.8, -0.9, -0.9, -0.9, -0.9, 0.1, 0.2, -0.3, -0.7, 0.3, -0.5, -0.3],
    "woman":    [0.9, 0.9, 0.1,   0.8, -0.8, -0.9, -0.9, -0.9, -0.9, 0.2, 0.2, -0.3, -0.7, 0.1, -0.5, -0.3],
    "prince":   [0.9, 0.9, 0.7,  -0.8, -0.9, -0.9, -0.9, -0.9, -0.9, 0.4, 0.3, -0.4, -0.8, 0.2, -0.7, -0.5],
    "princess": [0.9, 0.9, 0.7,   0.8, -0.9, -0.9, -0.9, -0.9, -0.9, 0.5, 0.3, -0.4, -0.8, 0.1, -0.7, -0.5],

    # Animals
    "dog":      [0.9, -0.9, -0.8, 0.0, -0.9,  0.9,  0.9, -0.9, -0.9, 0.6, -0.8, 0.1, -0.8, 0.2, -0.8, 0.8],
    "puppy":    [0.9, -0.9, -0.8, 0.0, -0.9,  0.9,  0.95,-0.9, -0.9, 0.8, -0.8, 0.3, -0.9, -0.7, -0.8, 0.8],
    "wolf":     [0.9, -0.9, -0.4, 0.0, -0.9,  0.9,  0.8, -0.9, -0.9, -0.2, -0.7, 0.4, -0.8, 0.5, -0.8, 0.9],
    "cat":      [0.9, -0.9, -0.8, 0.0, -0.9,  0.9, -0.9,  0.9, -0.9, 0.5, -0.8, 0.0, -0.8, -0.3, -0.8, 0.8],
    "kitten":   [0.9, -0.9, -0.8, 0.0, -0.9,  0.9, -0.9,  0.95,-0.9, 0.8, -0.8, 0.2, -0.9, -0.8, -0.8, 0.8],
    "lion":     [0.9, -0.9,  0.6, 0.0, -0.9,  0.9, -0.7,  0.8, -0.9, 0.1, -0.7, 0.5, -0.8, 0.8, -0.8, 0.9],

    # Technology & AI
    "computer": [-0.9, -0.8, 0.0, 0.0, 0.95, -0.9, -0.9, -0.9, -0.9, 0.0, 0.4, -0.6, -0.7, 0.3, 0.9, -0.9],
    "robot":    [-0.5,  0.2, 0.1, 0.0, 0.9,  -0.8, -0.8, -0.8, -0.9, 0.1, 0.5,  0.2,  0.2, 0.4, 0.9, -0.9],
    "ai":       [-0.7,  0.3, 0.3, 0.0, 0.98, -0.9, -0.9, -0.9, -0.9, 0.3, 0.9,  0.1, -0.8, 0.5, 0.95, -0.9],
    "chatbot":  [-0.6,  0.4, 0.0, 0.0, 0.92, -0.9, -0.9, -0.9, -0.9, 0.4, 0.8,  0.3, -0.8, 0.2, 0.85, -0.9],
    "neural":   [-0.2,  0.1, 0.0, 0.0, 0.85, -0.7, -0.7, -0.7, -0.9, 0.1, 0.8,  0.0, -0.9, 0.1, 0.92, -0.4],
    "code":     [-0.9, -0.7, 0.0, 0.0, 0.9,  -0.9, -0.9, -0.9, -0.9, 0.2, 0.7,  0.2, -0.9, 0.0, 0.85, -0.9],

    # Food
    "apple":    [0.7, -0.9, -0.9, 0.0, -0.7, -0.9, -0.9, -0.9,  0.95, 0.6, -0.8, -0.8, -0.9, -0.6, -0.6, 0.9],
    "pizza":    [-0.8, -0.9, -0.9, 0.0, -0.9, -0.9, -0.9, -0.9,  0.98, 0.8, -0.8, -0.8, -0.9, 0.1, -0.7, -0.3],
    "banana":   [0.7, -0.9, -0.9, 0.0, -0.8, -0.9, -0.9, -0.9,  0.95, 0.5, -0.8, -0.8, -0.9, -0.5, -0.6, 0.9],

    # Vehicles
    "car":      [-0.9, -0.8, 0.0, 0.0, 0.5, -0.9, -0.9, -0.9, -0.9, 0.3, -0.6, 0.6, 0.95, 0.6, 0.6, -0.8],
    "airplane": [-0.9, -0.8, 0.2, 0.0, 0.7, -0.9, -0.9, -0.9, -0.9, 0.4, -0.6, 0.8, 0.98, 0.95, 0.8, -0.8],
    "bicycle":  [-0.9, -0.8, -0.5, 0.0, -0.3, -0.9, -0.9, -0.9, -0.9, 0.5, -0.7, 0.7, 0.8, -0.3, 0.3, 0.2],

    # Emotions
    "happy":    [0.5, 0.7, 0.0, 0.0, -0.6, -0.6, -0.6, -0.6, -0.5,  0.95, 0.8, -0.2, -0.7, 0.3, -0.5, 0.4],
    "sad":      [0.5, 0.7, 0.0, 0.0, -0.6, -0.6, -0.6, -0.6, -0.5, -0.95, 0.8, -0.6, -0.7, -0.3, -0.5, 0.2],
    "love":     [0.6, 0.8, 0.1, 0.2, -0.7, -0.5, -0.5, -0.5, -0.6,  0.98, 0.9,  0.1, -0.8, 0.6, -0.6, 0.5]
}


class EmbeddingEngine:
    def __init__(self):
        # Normalize all seed vectors to unit sphere (length = 1.0)
        self.vectors: Dict[str, np.ndarray] = {}
        for word, vec in SEED_VECTORS.items():
            arr = np.array(vec, dtype=np.float32)
            norm = np.linalg.norm(arr)
            self.vectors[word] = arr / (norm + 1e-9)
        self.dim = 16

    def _hash_word_vector(self, word: str) -> np.ndarray:
        """
        Deterministic pseudo-embedding for words not in the seed dictionary,
        preserving subword and character n-gram similarities.
        """
        v = np.zeros(self.dim, dtype=np.float32)
        w = word.lower().strip()
        if not w:
            return v
        
        # Check if known seeds exist as substrings
        found_seeds = []
        for seed_word, s_vec in self.vectors.items():
            if seed_word in w:
                found_seeds.append(s_vec)
        
        if found_seeds:
            v = np.mean(found_seeds, axis=0)

        # Add character hash influence
        rng = np.random.RandomState(abs(hash(w)) % (2**31 - 1))
        noise = rng.normal(0, 0.35, size=self.dim).astype(np.float32)
        v = v + noise

        norm = np.linalg.norm(v)
        return v / (norm + 1e-9)

    def get_word_embedding(self, word: str) -> np.ndarray:
        w = word.lower().strip()
        if w in self.vectors:
            return self.vectors[w]
        return self._hash_word_vector(w)

    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Embeds an entire sentence/prompt by tokenizing and computing
        the normalized mean of its word vectors (Bag-of-Embeddings).
        """
        words = [w.strip() for w in text.lower().split() if w.strip()]
        if not words:
            return np.zeros(self.dim, dtype=np.float32)
        
        vectors = [self.get_word_embedding(w) for w in words]
        combined = np.mean(vectors, axis=0)
        norm = np.linalg.norm(combined)
        return combined / (norm + 1e-9)

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> Dict[str, Any]:
        """
        Computes cosine similarity with mathematical walkthrough:
        cos(theta) = (A . B) / (||A|| * ||B||)
        """
        dot_product = float(np.dot(vec1, vec2))
        norm1 = float(np.linalg.norm(vec1))
        norm2 = float(np.linalg.norm(vec2))
        
        if norm1 == 0 or norm2 == 0:
            score = 0.0
        else:
            score = dot_product / (norm1 * norm2)
            
        score = max(-1.0, min(1.0, score))
        angle_rad = math.acos(score)
        angle_deg = math.degrees(angle_rad)

        return {
            "similarity": round(score, 4),
            "percentage": round(score * 100, 1),
            "angle_degrees": round(angle_deg, 1),
            "dot_product": round(dot_product, 4),
            "norm_a": round(norm1, 4),
            "norm_b": round(norm2, 4),
            "interpretation": self._interpret_similarity(score)
        }

    def _interpret_similarity(self, score: float) -> str:
        if score > 0.85:
            return "Extremely Similar (Nearly identical meaning or strong semantic synonym)"
        elif score > 0.65:
            return "Highly Related (Belong to the same semantic category or topic)"
        elif score > 0.35:
            return "Moderately Related (Share some contextual attributes or overlap)"
        elif score > 0.0:
            return "Weakly Related (Slight positive correlation)"
        elif score > -0.35:
            return "Unrelated / Orthogonal (Different topics or unrelated domains)"
        else:
            return "Opposite / Inverted (Contrasting meanings or polar attributes)"

    def vector_arithmetic(self, positive_words: List[str], negative_words: List[str]) -> Dict[str, Any]:
        """
        Calculates King - Man + Woman = Queen vector analogy.
        Result Vector = sum(positive) - sum(negative).
        Then finds the nearest neighbors in the vocabulary!
        """
        result_vec = np.zeros(self.dim, dtype=np.float32)

        for w in positive_words:
            result_vec += self.get_word_embedding(w)
        for w in negative_words:
            result_vec -= self.get_word_embedding(w)

        norm = np.linalg.norm(result_vec)
        if norm > 0:
            result_vec = result_vec / norm

        # Exclude input words from top matches
        exclude = set([w.lower().strip() for w in positive_words + negative_words])
        
        matches = []
        for word, vec in self.vectors.items():
            sim = float(np.dot(result_vec, vec))
            matches.append((word, sim))

        matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = [
            {"word": w, "similarity": round(s, 4), "is_input": w in exclude}
            for w, s in matches[:8]
        ]

        formula = " + ".join(positive_words)
        if negative_words:
            formula += " - " + " - ".join(negative_words)

        return {
            "formula": formula,
            "result_vector": [round(float(x), 3) for x in result_vec],
            "top_matches": top_matches,
            "winner": [m for m in top_matches if not m["is_input"]][0]["word"] if top_matches else "N/A"
        }

    def get_2d_projection(self, words: List[str] = None) -> List[Dict[str, Any]]:
        """
        Projects high-dimensional word vectors into 2D using Principal Component Analysis (PCA)
        for visualization on an interactive 2D scatter plot.
        """
        if not words:
            words = list(self.vectors.keys())

        # Collect vectors
        matrix = []
        valid_words = []
        for w in words:
            vec = self.get_word_embedding(w)
            matrix.append(vec)
            valid_words.append(w)

        X = np.array(matrix, dtype=np.float32)
        # Center the data
        X_centered = X - np.mean(X, axis=0)

        # SVD for PCA: X = U * S * V^T
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        # Project onto first 2 principal components
        coords_2d = np.dot(X_centered, Vt[:2].T)

        # Normalize 2D coordinates between -100 and +100 for display
        max_val = np.max(np.abs(coords_2d)) if np.max(np.abs(coords_2d)) > 0 else 1.0
        coords_scaled = (coords_2d / max_val) * 80.0

        points = []
        for i, w in enumerate(valid_words):
            # Category detection for UI color-coding
            cat = "other"
            if w in ["king", "queen", "man", "woman", "prince", "princess"]:
                cat = "royalty_people"
            elif w in ["dog", "puppy", "wolf", "cat", "kitten", "lion"]:
                cat = "animals"
            elif w in ["computer", "robot", "ai", "chatbot", "neural", "code"]:
                cat = "tech_ai"
            elif w in ["apple", "pizza", "banana"]:
                cat = "food"
            elif w in ["car", "airplane", "bicycle"]:
                cat = "vehicles"
            elif w in ["happy", "sad", "love"]:
                cat = "emotions"

            points.append({
                "word": w,
                "x": round(float(coords_scaled[i, 0]), 2),
                "y": round(float(coords_scaled[i, 1]), 2),
                "category": cat,
                "vector": [round(float(v), 3) for v in X[i]]
            })

        return points


# Singleton instance
EMBEDDING_ENGINE = EmbeddingEngine()
