"""
Educational Chatbot Generator
=============================
Integrates the complete Glass Box AI Pipeline:
1. Tokenizer (BPE subwords & IDs)
2. Semantic Embeddings & Vector Search (RAG)
3. Prompt Context Assembly
4. Autoregressive Next-Token Probability Prediction (Logits, Temperature, Softmax)
5. Generative Output with Step-by-Step Telemetry
"""

import time
import re
from typing import List, Dict, Any, Generator
import numpy as np

from ai_core.tokenizer import analyze_text_tokens, BPETokenizer
from ai_core.embeddings import EMBEDDING_ENGINE
from ai_core.rag_engine import RAG_ENGINE
from ai_core.sampling import simulate_sampling_pipeline, apply_softmax


# Educational knowledge topics and responses for local Glass-Box model
TOPIC_ANSWERS = {
    "transformer": (
        "Transformers are the foundational architecture powering modern AI models like Gemini and GPT. "
        "Unlike older recurrent neural networks that processed text sequentially one word at a time, "
        "Transformers process an entire sequence in parallel using **Self-Attention**. "
        "This allows the model to compute direct mathematical connections between any two words regardless of how far apart they are."
    ),
    "token": (
        "Tokenization is the translation layer between human language and neural networks. "
        "Because computers only calculate numbers (matrix multiplications), text is split into subwords called **tokens**. "
        "Each unique token corresponds to an integer ID in the model's vocabulary (typically 32,000 to 256,000 tokens). "
        "For example, common words are a single token, while rare or novel words are split into pieces."
    ),
    "embedding": (
        "An embedding is a representation of words or sentences as dense vectors of numbers in a high-dimensional space. "
        "Words with similar meanings or contexts end up close together geometrically. "
        "For instance, the vector for 'cat' is closer to 'dog' than to 'computer'. "
        "This allows computers to calculate semantic similarity using Cosine Similarity: $\\cos(\\theta) = \\frac{u \\cdot v}{\\|u\\| \\|v\\|}$."
    ),
    "temperature": (
        "Temperature ($T$) is a hyperparameter that controls randomness during token generation. "
        "The model calculates raw scores (logits) for every word in its vocabulary. "
        "Dividing logits by temperature before applying Softmax controls the shape of the probability distribution:\n"
        "- **Low Temperature ($T < 0.5$)**: Sharpens probabilities, picking the most likely word (predictable, deterministic, factual).\n"
        "- **High Temperature ($T > 1.0$)**: Flattens probabilities, giving rarer tokens a chance (creative, diverse, but can hallucinate)."
    ),
    "rag": (
        "RAG stands for **Retrieval-Augmented Generation**. "
        "Instead of retraining a massive model every time new information is created, RAG connects the model to a vector database. "
        "When you ask a question, the system embeds your query, searches for relevant documents using vector similarity, "
        "and injects those facts directly into the prompt context so the AI can answer accurately with citations."
    ),
    "backprop": (
        "Backpropagation (backward propagation of errors) is how neural networks learn during training. "
        "1. **Forward Pass**: Input data passes through network layers to make a prediction.\n"
        "2. **Loss Calculation**: A loss function measures the gap between the prediction and the correct answer.\n"
        "3. **Backward Pass**: Using the chain rule from calculus, gradients are calculated to determine how much each weight contributed to the error.\n"
        "4. **Weight Update**: An optimizer (like AdamW or SGD) nudges the weights to reduce the error."
    ),
    "attention": (
        "Self-Attention is the core engine of the Transformer. "
        "For every token, the model calculates three vectors: **Query ($Q$)**, **Key ($K$)**, and **Value ($V$)**.\n"
        "The attention score is computed as: $\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}}\\right) V$.\n"
        "This computes a weighted sum of all values based on how much the query matches each key, allowing words to adapt their meaning to surrounding context."
    )
}


class GlassBoxChatbot:
    def __init__(self):
        self.tokenizer = BPETokenizer()

    def generate_response(
        self,
        user_message: str,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        enable_rag: bool = True
    ) -> Dict[str, Any]:
        """
        Executes the complete transparent 5-stage AI pipeline.
        """
        start_time = time.time()

        # STAGE 1: TOKENIZE USER INPUT
        token_analysis = analyze_text_tokens(user_message)

        # STAGE 2 & 3: VECTOR RETRIEVAL (RAG) & PROMPT ASSEMBLY
        rag_data = None
        if enable_rag:
            rag_data = RAG_ENGINE.construct_augmented_prompt(user_message, top_k=2)
            retrieved_context = rag_data["context_str"]
            full_prompt = rag_data["full_prompt"]
        else:
            retrieved_context = ""
            full_prompt = (
                f"=== SYSTEM INSTRUCTION ===\nYou are an educational AI assistant.\n\n"
                f"=== USER QUESTION ===\n{user_message}\n\n=== ASSISTANT ANSWER ==="
            )

        # STAGE 4: AUTOREGRESSIVE GENERATION & PROBABILITY TELEMETRY
        # Determine the core topic to answer meaningfully
        cleaned_msg = user_message.lower()
        response_text = ""
        matched_topic = None

        for key, ans in TOPIC_ANSWERS.items():
            if key in cleaned_msg:
                response_text = ans
                matched_topic = key
                break

        # If no specific key, check if RAG found a high-confidence match
        if not response_text and rag_data and rag_data["retrieved_docs"]:
            top_doc = rag_data["retrieved_docs"][0]
            if top_doc["similarity_score"] > 0.40:
                response_text = (
                    f"Based on retrieved knowledge regarding **{top_doc['title']}**:\n\n"
                    f"{top_doc['content']}\n\n"
                    f"*(Retrieved via Vector Cosine Similarity score: {top_doc['similarity_percentage']}%)*"
                )
                matched_topic = top_doc['topic']

        # Fallback general conversational response
        if not response_text:
            response_text = (
                f"I processed your query: '{user_message}'. "
                "In a generative AI model, your text is transformed into tokens, projected through multi-head self-attention layers, "
                "and decoded autoregressively by predicting the most probable next token one-by-one. "
                "Try asking me about **Transformers**, **Tokens**, **Embeddings**, **Temperature**, **Backpropagation**, or **RAG** "
                "to see how the underlying mathematics and vector similarity mechanics operate!"
            )

        # Tokenize the generated response
        generated_tokens = self.tokenizer.tokenize(response_text)
        total_tokens_generated = len(generated_tokens)

        # Simulate step-by-step next token prediction for the first few generation steps
        generation_steps = []
        sample_words = ["The", "model", "calculates", "probabilities", "using", "attention"]
        
        # Pick 3 representative tokens to illustrate logit/probability telemetry
        step_indices = [0, min(2, len(generated_tokens)-1), min(5, len(generated_tokens)-1)]
        for idx, step_i in enumerate(step_indices):
            actual_token = generated_tokens[step_i]
            
            # Synthesize realistic competing candidate logits for educational display
            competing = [
                (actual_token, 4.2),
                ("a", 2.1),
                ("is", 1.8),
                ("in", 1.4),
                ("to", 0.9),
                ("data", 0.3),
                ("neural", -0.2),
                ("banana", -3.5)
            ]
            cand_tokens = [c[0] for c in competing]
            raw_logits = [c[1] for c in competing]

            step_telemetry = simulate_sampling_pipeline(
                tokens=cand_tokens,
                raw_logits=raw_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=42 + idx
            )
            step_telemetry["step_number"] = step_i + 1
            step_telemetry["context_prefix"] = "".join(generated_tokens[:step_i]) if step_i > 0 else "<START_OF_RESPONSE>"
            step_telemetry["selected_token"] = actual_token
            generation_steps.append(step_telemetry)

        elapsed_ms = round((time.time() - start_time) * 1000, 1)

        return {
            "user_message": user_message,
            "response": response_text,
            "matched_topic": matched_topic,
            "telemetry": {
                "elapsed_ms": elapsed_ms,
                "input_tokens_count": token_analysis["bpe"]["token_count"],
                "output_tokens_count": total_tokens_generated,
                "total_tokens": token_analysis["bpe"]["token_count"] + total_tokens_generated,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p
            },
            "stages": {
                "stage_1_tokenization": token_analysis,
                "stage_2_vector_search": rag_data["retrieved_docs"] if rag_data else [],
                "stage_3_prompt_assembly": {
                    "full_prompt": full_prompt,
                    "system_prompt_length": 140,
                    "context_length": len(retrieved_context),
                    "user_length": len(user_message)
                },
                "stage_4_sampling_telemetry": generation_steps
            }
        }


# Singleton chatbot instance
CHATBOT_ENGINE = GlassBoxChatbot()
