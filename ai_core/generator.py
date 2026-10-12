"""
Educational Chatbot Generator
=============================
Integrates the complete Glass Box AI Pipeline:
1. Tokenizer (BPE subwords & IDs)
2. Semantic Embeddings & Vector Search (RAG)
3. Prompt Context Assembly
4. Autoregressive Next-Token Probability Prediction (Logits, Temperature, Softmax)
5. Generative Output with Real-Time Streaming & Step-by-Step Telemetry
"""

import time
import re
from typing import List, Dict, Any, Generator, Tuple
import numpy as np

from ai_core.tokenizer import analyze_text_tokens, BPETokenizer
from ai_core.embeddings import EMBEDDING_ENGINE
from ai_core.rag_engine import RAG_ENGINE
from ai_core.sampling import simulate_sampling_pipeline, apply_softmax


TOPIC_ANSWERS = {
    "real time": (
        "Operating an AI chatbot in **real time** requires an intricate, synchronized pipeline between GPUs, neural networks, and the web browser:\n\n"
        "1. **Autoregressive Decoding:** LLMs cannot generate an entire paragraph at once. The model predicts exactly **one token at a time**. "
        "Each newly predicted token is appended to the prompt history and fed back into the model to predict the next.\n\n"
        "2. **Real-Time Streaming (SSE):** Instead of making you wait 10 seconds for the full response, servers use **Server-Sent Events (SSE)** or WebSockets to stream tokens to your screen the millisecond they are sampled (at 30–100+ tokens/second).\n\n"
        "3. **KV Caching (Key-Value Cache):** In naive generation, re-calculating attention across 2,000 previous tokens for every new word would cause quadratic slowdown ($O(N^2)$). "
        "In real-time inference, the model saves past Key and Value attention matrices in high-speed GPU VRAM, so only the newest token is computed.\n\n"
        "4. **Latency Metrics:** AI speed is measured by **TTFT** (Time To First Token) and **ITL** (Inter-Token Latency, typically 10–25ms per token on modern hardware like NVIDIA H100s or Google TPUs)."
    ),
    "transformer": (
        "Transformers are the foundational neural network architecture powering modern AI models like Gemini and GPT.\n\n"
        "Unlike older recurrent neural networks (RNNs) that processed text sequentially word-by-word, "
        "Transformers process an entire sequence in parallel using **Self-Attention**.\n\n"
        "This allows the model to compute direct mathematical connections between any two words regardless of how far apart they are in the context window. "
        "A standard Transformer consists of alternating layers of **Multi-Head Self-Attention**, **Layer Normalization**, and **Feed-Forward Networks** (FFNs)."
    ),
    "token": (
        "Tokenization is the translation layer between human language and neural networks.\n\n"
        "Because computers only perform matrix multiplications on numbers, text is split into subwords called **tokens**. "
        "Each unique token corresponds to an integer ID in the model's vocabulary (typically 32,000 to 256,000 tokens).\n\n"
        "Modern LLMs use **Byte-Pair Encoding (BPE)**: common words are a single token (e.g. `robot` $\\to$ `#1284`), "
        "while rare or compound words are broken into subwords (e.g. `strawberry` $\\to$ `['straw', 'berry']`)."
    ),
    "embedding": (
        "An embedding is a representation of words or concepts as dense vectors of numbers in a high-dimensional space.\n\n"
        "Words with similar meanings or contexts end up close together geometrically. "
        "For instance, the vector for `cat` is close to `dog`, while `computer` is far away.\n\n"
        "This enables computers to calculate semantic similarity using **Cosine Similarity**: "
        "$$\\cos(\\theta) = \\frac{\\mathbf{u} \\cdot \\mathbf{v}}{\\|\\mathbf{u}\\| \\|\\mathbf{v}\\|}$$\n\n"
        "Remarkably, concept algebra emerges naturally: $\\vec{\\text{King}} - \\vec{\\text{Man}} + \\vec{\\text{Woman}} \\approx \\vec{\\text{Queen}}$."
    ),
    "temperature": (
        "Temperature ($T$) is a hyperparameter that controls the randomness of next-token selection during sampling.\n\n"
        "The model's final layer outputs raw unconstrained scores called **Logits**. "
        "Dividing logits by temperature before applying the Softmax function reshapes the probability distribution:\n\n"
        "- **Low Temperature ($T < 0.4$, Cold/Greedy):** Exaggerates differences between logits. The top word gets ~99% probability. Predictable, deterministic, factual.\n"
        "- **High Temperature ($T > 1.2$, Hot):** Flattens the distribution. Lesser-known words get picked. Highly creative and poetic, but risks hallucinations and nonsense.\n\n"
        "Coupled with **Top-K** and **Top-P (Nucleus)** filtering, it prevents the model from choosing nonsensical long-tail words."
    ),
    "rag": (
        "RAG stands for **Retrieval-Augmented Generation**.\n\n"
        "Large Language Models have static frozen weights; they cannot know real-time news or private corporate documents. If asked, they often **hallucinate** plausible falsehoods.\n\n"
        "**How RAG works in real time:**\n"
        "1. External documents are chunked and converted into vector embeddings.\n"
        "2. When you ask a question, your query is embedded into a vector.\n"
        "3. A Vector Database performs Cosine Similarity search to retrieve the top matching facts.\n"
        "4. Those retrieved facts are prepended directly into the prompt context for the LLM to read before generating its response."
    ),
    "backprop": (
        "Backpropagation (backward propagation of errors) is the mathematical engine of neural network training:\n\n"
        "1. **Forward Pass:** Input data flows through network layers, producing an output prediction.\n"
        "2. **Loss Calculation:** A loss function (like Cross-Entropy) calculates the numerical error between the prediction and the ground truth.\n"
        "3. **Backward Pass:** Applying the calculus Chain Rule, gradients are computed backwards layer-by-layer to find how much each weight contributed to the error.\n"
        "4. **Optimizer (AdamW / SGD):** Gradient Descent updates billions of weights in the direction that minimizes loss."
    ),
    "attention": (
        "Self-Attention is the core mathematical innovation of the Transformer:\n\n"
        "For every token, the model projects three vectors:\n"
        "- **Query ($Q$):** What this token is searching for.\n"
        "- **Key ($K$):** What this token offers to others.\n"
        "- **Value ($V$):** The actual informational content.\n\n"
        "The attention weights are calculated as:\n"
        "$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}}\\right) V$$\n\n"
        "This allows words like *'it'* in *'The animal didn't cross the street because it was too tired'* to mathematically route 85%+ of their attention to *'animal'*!"
    ),
    "strawberry": (
        "The famous 'strawberry' problem illustrates why LLMs are not character-based programs:\n\n"
        "When you ask an AI model *'How many r's are in strawberry?'*, humans expect the model to inspect letters: `s-t-r-a-w-b-e-r-r-y`.\n"
        "However, LLMs **never see individual characters**. The tokenizer compresses the word into subwords: `['straw', 'berry']`.\n\n"
        "Because the letters are packaged inside token IDs, the model has to infer spelling from indirect training memory or use multi-step reasoning ('Chain-of-Thought') to spell the word out letter-by-letter!"
    ),
    "hardware": (
        "Running AI models in real time demands immense specialized hardware:\n\n"
        "- **GPUs / TPUs:** Modern LLMs require thousands of Tensor Cores capable of doing half-precision matrix multiplications ($FP16$ or $BF16$) at trillions of FLOPs per second.\n"
        "- **Memory Bandwidth Bottleneck:** During real-time inference, the model must read all weights from memory for every single token. High Bandwidth Memory (HBM3) is critical.\n"
        "- **Quantization (INT8 / INT4):** Compressing 16-bit weights into 4-bit integers reduces VRAM usage by 75%, allowing massive models to run smoothly on laptops and local machines."
    ),
    "hallucination": (
        "Hallucination is when an AI generates statements that sound confident and authoritative but are completely fabricated.\n\n"
        "Why does it happen? Because LLMs are **statistical token predictors**, not truth databases. "
        "The model selects words that are statistically coherent in that sentence structure. "
        "If the training data has gaps, or if high temperature sampling pushes the model down a rare probability branch, it will seamlessly generate plausible-sounding fiction."
    )
}


class GlassBoxChatbot:
    def __init__(self):
        self.tokenizer = BPETokenizer()

    def get_response_text(self, user_message: str, enable_rag: bool = True) -> Tuple[str, str, Dict[str, Any]]:
        """
        Determines the educational response text and retrieves RAG context if applicable.
        """
        cleaned_msg = user_message.lower()
        response_text = ""
        matched_topic = None
        rag_data = None

        if enable_rag:
            rag_data = RAG_ENGINE.construct_augmented_prompt(user_message, top_k=2)

        # Match specific knowledge keys
        for key, ans in TOPIC_ANSWERS.items():
            if key in cleaned_msg:
                response_text = ans
                matched_topic = key
                break

        # Check RAG hits if no direct key matched
        if not response_text and rag_data and rag_data["retrieved_docs"]:
            top_doc = rag_data["retrieved_docs"][0]
            if top_doc["similarity_score"] > 0.38:
                response_text = (
                    f"Based on retrieved knowledge regarding **{top_doc['title']}**:\n\n"
                    f"{top_doc['content']}\n\n"
                    f"*(Retrieved via Vector Cosine Similarity score: {top_doc['similarity_percentage']}%)*"
                )
                matched_topic = top_doc.get("topic", "RAG Knowledge")

        # Fallback general conversational response
        if not response_text:
            response_text = (
                f"I processed your query: '{user_message}'.\n\n"
                "In a generative AI model, your text is transformed into numerical tokens, mapped into dense embedding vectors, "
                "projected across dozens of self-attention Transformer layers, and decoded autoregressively by predicting the most probable next token one-by-one.\n\n"
                "Try asking me about:\n"
                "- **'How AI works in real time'** (Autoregressive decoding, streaming, KV caching)\n"
                "- **'What is a Transformer and Self-Attention?'**\n"
                "- **'Why do LLMs fail to count letters in strawberry?'**\n"
                "- **'How do Embeddings and Cosine Similarity work?'**\n"
                "- **'How does Temperature affect AI creativity?'**\n"
                "- **'How does RAG eliminate hallucinations?'**"
            )

        return response_text, matched_topic, rag_data

    def generate_response(
        self,
        user_message: str,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        enable_rag: bool = True
    ) -> Dict[str, Any]:
        """
        Executes the synchronous 5-stage glass-box pipeline.
        """
        start_time = time.time()

        # Stage 1: Tokenize input
        token_analysis = analyze_text_tokens(user_message)

        # Stage 2 & 3: RAG & Prompt
        response_text, matched_topic, rag_data = self.get_response_text(user_message, enable_rag=enable_rag)
        retrieved_context = rag_data["context_str"] if rag_data else ""
        full_prompt = rag_data["full_prompt"] if rag_data else (
            f"=== SYSTEM INSTRUCTION ===\nYou are an educational AI assistant.\n\n"
            f"=== USER QUESTION ===\n{user_message}\n\n=== ASSISTANT ANSWER ==="
        )

        # Stage 4: Autoregressive next-token sampling simulation
        generated_tokens = self.tokenizer.tokenize(response_text)
        total_tokens_generated = len(generated_tokens)

        # Sample telemetry steps
        generation_steps = []
        step_indices = [0, min(2, len(generated_tokens)-1), min(5, len(generated_tokens)-1)]
        for idx, step_i in enumerate(step_indices):
            actual_token = generated_tokens[step_i]
            competing = [
                (actual_token, 4.2),
                ("the", 2.1),
                ("is", 1.8),
                ("in", 1.4),
                ("to", 0.9),
                ("model", 0.4),
                ("data", 0.1),
                ("banana", -3.5)
            ]
            step_telemetry = simulate_sampling_pipeline(
                tokens=[c[0] for c in competing],
                raw_logits=[c[1] for c in competing],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=42 + idx
            )
            step_telemetry["step_number"] = step_i + 1
            step_telemetry["context_prefix"] = "".join(generated_tokens[:step_i]) if step_i > 0 else "<START>"
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
                "tokens_per_second": round(total_tokens_generated / (max(0.001, elapsed_ms) / 1000.0), 1),
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

    def generate_response_stream(
        self,
        user_message: str,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        enable_rag: bool = True,
        token_delay_ms: float = 25.0
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Yields real-time events for Server-Sent Events (SSE) streaming:
        1. Initialization & Tokenization event
        2. RAG & Prompt Assembly event
        3. Real-time token-by-token events with logits & attention telemetry
        4. Completion event
        """
        start_time = time.time()

        # Event 1: Tokenization
        token_analysis = analyze_text_tokens(user_message)
        yield {
            "stage": "stage_1_tokenization",
            "data": token_analysis,
            "message": "Tokenized user input into BPE subword tokens"
        }

        # Event 2: RAG Vector Search & Prompt Assembly
        response_text, matched_topic, rag_data = self.get_response_text(user_message, enable_rag=enable_rag)
        full_prompt = rag_data["full_prompt"] if rag_data else (
            f"=== SYSTEM INSTRUCTION ===\nYou are an educational AI assistant.\n\n"
            f"=== USER QUESTION ===\n{user_message}\n\n=== ASSISTANT ANSWER ==="
        )

        yield {
            "stage": "stage_2_rag",
            "retrieved_docs": rag_data["retrieved_docs"] if rag_data else [],
            "full_prompt": full_prompt,
            "matched_topic": matched_topic,
            "message": "Retrieved relevant vectors and assembled context window"
        }

        # Event 3: Token Streaming
        generated_tokens = self.tokenizer.tokenize(response_text)
        total_tokens = len(generated_tokens)
        input_tokens = token_analysis["bpe"]["tokens"]

        distractors_pool = ["the", "is", "a", "model", "neural", "attention", "data", "weights", "vector", "state", "output"]

        for i, token in enumerate(generated_tokens):
            # Synthesize realistic competing logits for current token
            base_logit = 3.8 + np.random.uniform(-0.4, 0.6)
            alt1 = distractors_pool[(i * 3) % len(distractors_pool)]
            alt2 = distractors_pool[(i * 7 + 1) % len(distractors_pool)]
            alt3 = distractors_pool[(i * 5 + 2) % len(distractors_pool)]
            
            cands = [token, alt1, alt2, alt3, "banana"]
            raw_logits = [base_logit, base_logit - 1.4, base_logit - 2.2, base_logit - 3.1, -4.5]

            step_sim = simulate_sampling_pipeline(
                tokens=cands,
                raw_logits=raw_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=i + 100
            )

            # Attention link: which input token or recent token this step attends to
            attended_idx = max(0, min(len(input_tokens) - 1, i % len(input_tokens)))
            attended_token = input_tokens[attended_idx] if input_tokens else ""

            elapsed = round((time.time() - start_time) * 1000, 1)
            tok_per_sec = round((i + 1) / max(0.001, (time.time() - start_time)), 1)

            yield {
                "stage": "stage_stream_token",
                "token": token,
                "token_index": i + 1,
                "total_tokens": total_tokens,
                "attended_token": attended_token,
                "attended_index": attended_idx,
                "telemetry": step_sim,
                "stats": {
                    "elapsed_ms": elapsed,
                    "tok_per_sec": tok_per_sec,
                    "progress_pct": round(((i + 1) / total_tokens) * 100, 1)
                }
            }

            if token_delay_ms > 0:
                time.sleep(token_delay_ms / 1000.0)

        # Event 4: Completion
        total_elapsed_ms = round((time.time() - start_time) * 1000, 1)
        yield {
            "stage": "stage_complete",
            "full_response": response_text,
            "matched_topic": matched_topic,
            "telemetry": {
                "elapsed_ms": total_elapsed_ms,
                "input_tokens_count": token_analysis["bpe"]["token_count"],
                "output_tokens_count": total_tokens,
                "total_tokens": token_analysis["bpe"]["token_count"] + total_tokens,
                "tokens_per_second": round(total_tokens / max(0.001, total_elapsed_ms / 1000.0), 1)
            }
        }


# Singleton chatbot instance
CHATBOT_ENGINE = GlassBoxChatbot()
