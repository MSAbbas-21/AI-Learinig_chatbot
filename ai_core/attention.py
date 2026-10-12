"""
Educational Self-Attention Module
=================================
Visualizes the core mechanism of Transformers:
Attention(Q, K, V) = Softmax(Q K^T / sqrt(d_k)) V

Features:
1. Multi-Head Attention Simulation (4 educational heads):
   - Head 1: Coreference & Semantic Link (e.g. 'it' -> 'animal' or 'street')
   - Head 2: Syntactic Dependency (verbs to nouns, adjectives to nouns)
   - Head 3: Positional Locality (neighboring tokens)
   - Head 4: Global Salience / Topic Focus
2. Real-time Attention Matrix computation for any sentence.
3. Interactive token attention extraction (which tokens does token X attend to).
"""

import math
import numpy as np
from typing import List, Dict, Any, Tuple
from ai_core.tokenizer import BPETokenizer
from ai_core.embeddings import EMBEDDING_ENGINE


class AttentionEngine:
    def __init__(self):
        self.tokenizer = BPETokenizer()
        self.d_k = 16  # dimension of keys/queries

    def compute_attention(self, text: str, head_type: str = "coreference") -> Dict[str, Any]:
        """
        Computes attention matrix between tokens of the input text.
        Returns tokens, raw dot-product scores, softmax attention weights, and head explanations.
        """
        tokens = self.tokenizer.tokenize(text)
        if not tokens:
            tokens = ["Hello", "AI"]
        
        n = len(tokens)
        token_lower = [t.lower().strip() for t in tokens]

        # Base vectors from embedding engine
        vectors = [EMBEDDING_ENGINE.get_text_embedding(t) for t in token_lower]

        # Calculate attention weights matrix A of size (n, n)
        # A[i][j] represents how much token i attends to token j
        weights = np.zeros((n, n), dtype=np.float64)

        # Detect specific linguistic patterns for educational clarity
        # Winograd Schema / Coreference resolution
        has_animal = any("animal" in t or "dog" in t or "cat" in t for t in token_lower)
        has_street = any("street" in t or "road" in t for t in token_lower)
        has_tired = any("tired" in t or "hungry" in t or "sleepy" in t for t in token_lower)
        has_wide = any("wide" in t or "crowded" in t or "narrow" in t for t in token_lower)
        
        has_river = any("river" in t or "stream" in t or "water" in t for t in token_lower)
        has_money = any("money" in t or "loan" in t or "deposit" in t for t in token_lower)

        for i in range(n):
            for j in range(n):
                ti = token_lower[i]
                tj = token_lower[j]

                # Dot product baseline
                vec_i = vectors[i]
                vec_j = vectors[j]
                norm_i = np.linalg.norm(vec_i) or 1.0
                norm_j = np.linalg.norm(vec_j) or 1.0
                base_sim = float(np.dot(vec_i, vec_j) / (norm_i * norm_j))

                # Distance penalty / bias
                dist = abs(i - j)

                if head_type == "locality":
                    # Head favors nearby tokens
                    score = 2.5 / (1.0 + dist) + 0.3 * base_sim
                elif head_type == "syntactic":
                    # Head connects modifiers and verbs
                    score = base_sim * 1.5 + (1.2 if dist in [1, 2] else 0.2)
                elif head_type == "global":
                    # Head connects to first token / key nouns
                    score = (1.5 if j == 0 else 0.5) + base_sim * 1.2
                else:
                    # Default: Coreference / Semantic Head
                    score = base_sim * 1.8

                    # Coreference: 'it' resolving to 'animal' vs 'street'
                    if ti in ["it", "they", "its"] or "it" in ti:
                        if has_tired and "animal" in tj:
                            score += 3.8
                        elif has_wide and "street" in tj:
                            score += 3.8
                        elif has_tired and "tired" in tj:
                            score += 1.8
                        elif has_wide and "wide" in tj:
                            score += 1.8

                    # Polysemy: 'bank' resolving to river vs money
                    if "bank" in ti:
                        if has_river and ("river" in tj or "water" in tj):
                            score += 3.5
                        elif has_money and ("money" in tj or "loan" in tj):
                            score += 3.5

                    # Self-attention diagonal boost (tokens always attend somewhat to themselves)
                    if i == j:
                        score += 1.0

                weights[i][j] = score

        # Apply Softmax across rows: Softmax(scores / sqrt(d_k))
        scale = math.sqrt(self.d_k)
        scaled_weights = weights / scale
        
        attention_matrix = []
        for i in range(n):
            row = scaled_weights[i]
            # stable softmax
            exp_row = np.exp(row - np.max(row))
            probs = exp_row / np.sum(exp_row)
            attention_matrix.append([round(float(p), 4) for p in probs])

        # Find top attended token for each token
        top_attended = []
        for i in range(n):
            row_probs = attention_matrix[i]
            # Sort indices excluding self if possible, or include highest
            indexed = sorted([(j, p) for j, p in enumerate(row_probs)], key=lambda x: x[1], reverse=True)
            best_j, best_prob = indexed[0]
            top_attended.append({
                "from_index": i,
                "from_token": tokens[i],
                "to_index": best_j,
                "to_token": tokens[best_j],
                "weight": best_prob,
                "weight_percentage": round(best_prob * 100, 1)
            })

        head_descriptions = {
            "coreference": "Resolves pronouns ('it', 'they') to their antecedent nouns based on context.",
            "syntactic": "Tracks grammatical relationships (subjects to verbs, adjectives to nouns).",
            "locality": "Focuses on adjacent neighboring words for immediate phrase cohesion.",
            "global": "Attends to the global sentence theme and high-salience concepts."
        }

        return {
            "tokens": tokens,
            "token_count": n,
            "head_type": head_type,
            "head_description": head_descriptions.get(head_type, ""),
            "attention_matrix": attention_matrix,
            "top_attended": top_attended,
            "formula": "Attention(Q, K, V) = Softmax(Q·Kᵀ / √d_k) · V"
        }


ATTENTION_ENGINE = AttentionEngine()
