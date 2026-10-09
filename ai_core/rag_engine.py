"""
Educational RAG (Retrieval-Augmented Generation) Engine
========================================================
Demystifies how modern AI bots search through company documents, books, and
private knowledge bases without retraining the model.

Pipeline:
1. Knowledge Ingestion: Documents split into text chunks.
2. Vector Indexing: Each chunk is converted into an embedding vector.
3. Vector Retrieval: User question is converted into a vector, and Cosine
   Similarity finds the top-K closest chunks.
4. Context Augmentation: The retrieved chunks are injected into the Prompt.
5. Grounded Generation: Model answers using the provided facts!
"""

import json
from typing import List, Dict, Any, Tuple
import numpy as np
from ai_core.embeddings import EMBEDDING_ENGINE


DEFAULT_KNOWLEDGE_BASE = [
    {
        "id": "kb_1",
        "title": "What is a Transformer Architecture?",
        "topic": "Neural Networks",
        "content": "Introduced in the 2017 paper 'Attention Is All You Need' by Vaswani et al. at Google, Transformers replaced recurrent neural networks (RNNs) and LSTMs. Transformers process all words in parallel using Self-Attention, allowing models like GPT and Gemini to capture long-range dependencies efficiently across thousands of tokens."
    },
    {
        "id": "kb_2",
        "title": "Why do LLMs struggle to count letters in words (e.g. 'strawberry')?",
        "topic": "Tokenization",
        "content": "LLMs never see raw characters like s-t-r-a-w-b-e-r-r-y. Instead, the tokenizer groups letters into subword tokens such as ['straw', 'berry'] with unique numerical IDs. Because the model processes token IDs rather than individual letters, it must indirectly guess or deduce character counts unless it has been explicitly trained or prompted to break words down letter-by-letter."
    },
    {
        "id": "kb_3",
        "title": "How does Backpropagation and Gradient Descent work?",
        "topic": "Model Training",
        "content": "During pre-training, a neural network makes a prediction. A Loss Function measures the error between the prediction and the ground truth. Backpropagation uses the chain rule of calculus to calculate gradients—telling how much each weight contributed to the error. An optimizer (like AdamW) then updates the weights via Gradient Descent in the direction that minimizes loss."
    },
    {
        "id": "kb_4",
        "title": "What is Self-Attention in simple terms?",
        "topic": "Attention Mechanism",
        "content": "Self-Attention allows each word in a sentence to look at ('attend to') all other words to understand context. For example, in 'The bank of the river', the word 'bank' attends strongly to 'river', so the model knows it means a shoreline rather than a financial institution. It computes Query (Q), Key (K), and Value (V) matrices to calculate contextual relevance scores."
    },
    {
        "id": "kb_5",
        "title": "Pre-training vs Fine-Tuning vs RLHF",
        "topic": "AI Alignment",
        "content": "Pre-training teaches the model language patterns by predicting the next token on terabytes of internet text (resulting in a 'base model'). Fine-tuning (Instruction Tuning) trains the base model to answer user questions politely and clearly. Finally, RLHF (Reinforcement Learning from Human Feedback) or DPO aligns the model with human preferences, safety guidelines, and helpfulness."
    },
    {
        "id": "kb_6",
        "title": "What is Hallucination and why does it happen?",
        "topic": "Model Limitations",
        "content": "Hallucination occurs when an AI generates text that sounds confident and plausible but is factually incorrect. Because LLMs are probabilistic word predictors rather than truth databases, if their training data lacked the fact, or if temperature sampling picked an unlikely token path, the model will smoothly invent plausible-sounding details."
    },
    {
        "id": "kb_7",
        "title": "What is RAG (Retrieval-Augmented Generation)?",
        "topic": "Information Retrieval",
        "content": "RAG connects an AI to external live databases. Instead of relying solely on parametric memory (weights), the system converts the user's question into a vector embedding, queries a vector database for relevant documents using cosine similarity, and prepends those facts into the prompt context for the LLM to read before answering."
    }
]


class RAGEngine:
    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self.vectors: List[np.ndarray] = []
        self._load_initial_knowledge()

    def _load_initial_knowledge(self):
        for doc in DEFAULT_KNOWLEDGE_BASE:
            self.add_document(doc["title"], doc["content"], doc.get("topic", "General"), doc.get("id"))

    def add_document(self, title: str, content: str, topic: str = "Custom", doc_id: str = None) -> Dict[str, Any]:
        """
        Adds a new document to the RAG knowledge base and calculates its vector embedding.
        """
        if not doc_id:
            doc_id = f"custom_{len(self.documents) + 1}"

        text_to_embed = f"{title} {topic} {content}"
        vector = EMBEDDING_ENGINE.get_text_embedding(text_to_embed)

        doc_entry = {
            "id": doc_id,
            "title": title,
            "topic": topic,
            "content": content,
            "snippet": content[:120] + "..." if len(content) > 120 else content
        }

        self.documents.append(doc_entry)
        self.vectors.append(vector)
        return doc_entry

    def search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Performs semantic vector search on the knowledge base:
        1. Embeds query into a vector.
        2. Calculates cosine similarity with all document vectors.
        3. Returns top-K ranked documents with similarity scores.
        """
        if not self.documents:
            return []

        query_vec = EMBEDDING_ENGINE.get_text_embedding(query)
        scored_docs = []

        for i, (doc, doc_vec) in enumerate(zip(self.documents, self.vectors)):
            sim_data = EMBEDDING_ENGINE.cosine_similarity(query_vec, doc_vec)
            scored_docs.append({
                **doc,
                "similarity_score": sim_data["similarity"],
                "similarity_percentage": sim_data["percentage"],
                "interpretation": sim_data["interpretation"]
            })

        # Sort descending by similarity
        scored_docs.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored_docs[:top_k]

    def construct_augmented_prompt(self, user_query: str, top_k: int = 2) -> Dict[str, Any]:
        """
        Demonstrates the exact prompt injection process in modern RAG.
        """
        retrieved_docs = self.search(user_query, top_k=top_k)
        
        context_blocks = []
        for i, doc in enumerate(retrieved_docs):
            context_blocks.append(f"[Document {i+1}: {doc['title']}]\n{doc['content']}")

        context_str = "\n\n".join(context_blocks)

        system_instruction = (
            "You are an educational AI assistant. Answer the user's question accurately. "
            "Use the provided Knowledge Context whenever relevant to ground your answer."
        )

        full_prompt = (
            f"=== SYSTEM INSTRUCTION ===\n{system_instruction}\n\n"
            f"=== RETRIEVED KNOWLEDGE BASE CONTEXT (RAG) ===\n{context_str}\n\n"
            f"=== USER QUESTION ===\n{user_query}\n\n"
            f"=== ASSISTANT ANSWER ==="
        )

        return {
            "query": user_query,
            "retrieved_docs": retrieved_docs,
            "context_str": context_str,
            "full_prompt": full_prompt,
            "top_similarity": retrieved_docs[0]["similarity_score"] if retrieved_docs else 0.0
        }

    def get_all_documents(self) -> List[Dict[str, Any]]:
        return self.documents


# Singleton instance
RAG_ENGINE = RAGEngine()
