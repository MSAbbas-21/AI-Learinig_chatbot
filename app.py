"""
AI Learning Chatbot & Interactive Architecture Studio
=====================================================
A complete web application to understand the internals of Generative AI,
Tokenization, Embeddings, RAG, Attention, and Softmax Sampling.
"""

import os
import sys
from flask import Flask, render_template, request, jsonify

# Add directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_core.tokenizer import analyze_text_tokens
from ai_core.embeddings import EMBEDDING_ENGINE
from ai_core.sampling import simulate_sampling_pipeline
from ai_core.rag_engine import RAG_ENGINE
from ai_core.generator import CHATBOT_ENGINE

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Processes chat prompt and returns the generated answer + full glass-box telemetry.
    """
    try:
        data = request.get_json() or {}
        message = data.get("message", "").strip()
        if not message:
            return jsonify({"error": "Message cannot be empty"}), 400

        temperature = float(data.get("temperature", 0.7))
        top_k = int(data.get("top_k", 40))
        top_p = float(data.get("top_p", 0.9))
        enable_rag = bool(data.get("enable_rag", True))

        result = CHATBOT_ENGINE.generate_response(
            user_message=message,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            enable_rag=enable_rag
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/tokenize", methods=["POST"])
def tokenize():
    """
    Tokenizes input text across Char, Word, and BPE tokenizers.
    """
    try:
        data = request.get_json() or {}
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "Text cannot be empty"}), 400

        analysis = analyze_text_tokens(text)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/embeddings/similarity", methods=["POST"])
def similarity():
    """
    Calculates Cosine Similarity between two words or sentences.
    """
    try:
        data = request.get_json() or {}
        text1 = data.get("text1", "").strip()
        text2 = data.get("text2", "").strip()
        if not text1 or not text2:
            return jsonify({"error": "Both inputs are required"}), 400

        vec1 = EMBEDDING_ENGINE.get_text_embedding(text1)
        vec2 = EMBEDDING_ENGINE.get_text_embedding(text2)
        sim_data = EMBEDDING_ENGINE.cosine_similarity(vec1, vec2)
        
        sim_data["text1"] = text1
        sim_data["text2"] = text2
        sim_data["vector1"] = [round(float(x), 3) for x in vec1]
        sim_data["vector2"] = [round(float(x), 3) for x in vec2]
        return jsonify(sim_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/embeddings/arithmetic", methods=["POST"])
def arithmetic():
    """
    Calculates vector math like King - Man + Woman = Queen.
    """
    try:
        data = request.get_json() or {}
        positive = data.get("positive", [])
        negative = data.get("negative", [])
        
        if not positive:
            return jsonify({"error": "At least one positive word is required"}), 400

        result = EMBEDDING_ENGINE.vector_arithmetic(positive, negative)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/embeddings/projection", methods=["GET"])
def projection():
    """
    Returns 2D PCA projected coordinates for vocabulary words.
    """
    try:
        points = EMBEDDING_ENGINE.get_2d_projection()
        return jsonify({"points": points})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sampling/simulate", methods=["POST"])
def sampling_simulate():
    """
    Simulates temperature, Top-K, and Top-P filtering on candidate tokens.
    """
    try:
        data = request.get_json() or {}
        tokens = data.get("tokens", ["coffee", "tea", "water", "juice", "soda", "lava"])
        logits = data.get("logits", [4.5, 3.8, 2.5, 1.8, 0.5, -4.0])
        temperature = float(data.get("temperature", 1.0))
        top_k = int(data.get("top_k", 50))
        top_p = float(data.get("top_p", 1.0))

        result = simulate_sampling_pipeline(
            tokens=tokens,
            raw_logits=logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rag/documents", methods=["GET", "POST"])
def rag_documents():
    """
    Get all documents or add a new custom document to the RAG knowledge base.
    """
    try:
        if request.method == "POST":
            data = request.get_json() or {}
            title = data.get("title", "").strip()
            content = data.get("content", "").strip()
            topic = data.get("topic", "Custom")
            if not title or not content:
                return jsonify({"error": "Title and content are required"}), 400
            
            new_doc = RAG_ENGINE.add_document(title, content, topic)
            return jsonify({"status": "success", "document": new_doc})
        else:
            return jsonify({"documents": RAG_ENGINE.get_all_documents()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rag/search", methods=["POST"])
def rag_search():
    """
    Runs vector search against the RAG knowledge base.
    """
    try:
        data = request.get_json() or {}
        query = data.get("query", "").strip()
        top_k = int(data.get("top_k", 2))
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400

        res = RAG_ENGINE.construct_augmented_prompt(query, top_k=top_k)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n==========================================================")
    print(f"  AI Learning Studio & Glass-Box Chatbot is running!")
    print(f"  Open in your browser: http://localhost:{port}")
    print(f"==========================================================\n")
    app.run(host="127.0.0.1", port=port, debug=False)
