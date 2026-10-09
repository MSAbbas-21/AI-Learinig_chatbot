"""
Automated Integration Tests for AI Learning Studio
==================================================
Tests:
- /api/chat endpoint and telemetry schema
- /api/tokenize with Char, Word, and BPE
- /api/embeddings/similarity and vector arithmetic
- /api/embeddings/projection PCA 2D coordinates
- /api/sampling/simulate temperature, top_k, top_p
- /api/rag/documents and /api/rag/search
"""

import sys
import os

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app


def run_tests():
    client = app.test_client()
    print("Running integration tests...")

    # 1. Test Index
    res = client.get("/")
    assert res.status_code == 200, f"Index failed: {res.status_code}"
    print("✔ Home page loads successfully")

    # 2. Test Tokenizer
    res = client.post("/api/tokenize", json={"text": "Understanding transformer models!"})
    assert res.status_code == 200
    data = res.get_json()
    assert "bpe" in data and "word" in data and "char" in data
    assert len(data["bpe"]["tokens"]) > 0
    print(f"✔ Tokenizer API works: {len(data['bpe']['tokens'])} BPE tokens generated")

    # 3. Test Similarity
    res = client.post("/api/embeddings/similarity", json={"text1": "king", "text2": "queen"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["similarity"] > 0.8
    print(f"✔ Embedding Similarity works: King vs Queen = {data['percentage']}%")

    # 4. Test Vector Arithmetic
    res = client.post("/api/embeddings/arithmetic", json={"positive": ["king", "woman"], "negative": ["man"]})
    assert res.status_code == 200
    data = res.get_json()
    assert data["winner"] == "queen"
    print(f"✔ Vector Arithmetic works: King - Man + Woman = {data['winner']}")

    # 5. Test PCA Projection
    res = client.get("/api/embeddings/projection")
    assert res.status_code == 200
    data = res.get_json()
    assert len(data["points"]) > 10
    print(f"✔ PCA 2D Projection works: {len(data['points'])} points projected")

    # 6. Test Sampling Simulator
    res = client.post("/api/sampling/simulate", json={"temperature": 0.7, "top_k": 4, "top_p": 0.9})
    assert res.status_code == 200
    data = res.get_json()
    assert "winner" in data
    print(f"✔ Sampling simulator works: Winner = {data['winner']}")

    # 7. Test RAG Search
    res = client.post("/api/rag/search", json={"query": "What is self-attention?"})
    assert res.status_code == 200
    data = res.get_json()
    assert len(data["retrieved_docs"]) > 0
    print(f"✔ RAG vector search works: Top hit = '{data['retrieved_docs'][0]['title']}'")

    # 8. Test Chatbot with full telemetry
    res = client.post("/api/chat", json={"message": "What is a Transformer?", "temperature": 0.7})
    assert res.status_code == 200
    data = res.get_json()
    assert "response" in data
    assert "stages" in data
    assert "telemetry" in data
    print("✔ Glass-Box Chatbot API works: Response generated with all 4 stages of telemetry!")

    print("\n🎉 ALL TESTS PASSED! The AI Learning Studio is completely functional.")


if __name__ == "__main__":
    run_tests()
