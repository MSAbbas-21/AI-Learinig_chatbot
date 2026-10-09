"""
Interactive AI Educational CLI
==============================
Run this script in your terminal to explore how AI really works:
`python learn_ai_cli.py`

Covers:
1. Interactive Tokenizer (Char vs Word vs Subword/BPE)
2. Vector Arithmetic & Cosine Similarity (King - Man + Woman = Queen)
3. Logits, Softmax & Temperature Simulator
4. Live RAG Vector Search
5. Chat with the Glass-Box AI Model
"""

import sys
import os

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Ensure current dir is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_core.tokenizer import analyze_text_tokens
from ai_core.embeddings import EMBEDDING_ENGINE
from ai_core.sampling import simulate_sampling_pipeline
from ai_core.rag_engine import RAG_ENGINE
from ai_core.generator import CHATBOT_ENGINE


def print_header(title: str):
    print("\n" + "=" * 65)
    print(f"  {title.upper()}")
    print("=" * 65)


def run_tokenizer_demo():
    print_header("Lesson 1: Tokenization (How AI Reads Text)")
    print("Neural networks cannot read raw English letters. They only calculate numbers.")
    print("Text must be split into 'tokens' and mapped to vocabulary integer IDs.\n")
    
    text = input("Enter any sentence to tokenize [or press Enter for default]: ").strip()
    if not text:
        text = "Understanding artificial intelligence and chatbots with transformers!"

    analysis = analyze_text_tokens(text)
    
    print(f"\nOriginal Text: \"{text}\"")
    print(f"Total Characters: {analysis['char_count']}")
    
    print("\n1. Byte-Pair Encoding (Subwords - like modern GPT/Gemini):")
    tokens = analysis['bpe']['tokens']
    ids = analysis['bpe']['token_ids']
    print("  Tokens : " + " | ".join(f"'{t}'" for t in tokens))
    print("  IDs    : " + " | ".join(str(i) for i in ids))
    print(f"  Token Count: {analysis['bpe']['token_count']} (Compression ratio: {analysis['bpe']['compression_ratio']} chars/token)")

    print("\n2. Word Tokenizer:")
    print("  Tokens : " + " | ".join(analysis['word']['tokens']))
    print(f"  Count  : {analysis['word']['token_count']} tokens")

    print("\nKey Takeaway: Modern LLMs use Subwords (BPE). This allows them to represent")
    print("rare words, prefixes, and code without having an infinite vocabulary!")


def run_embeddings_demo():
    print_header("Lesson 2: Embeddings & Vector Math (Meaning in Geometry)")
    print("An embedding turns words into vectors (coordinates in high-dimensional space).")
    print("Concepts with similar meaning point in nearly identical directions!\n")

    print("Try comparing two words to measure their Cosine Similarity:")
    word_a = input("Enter Word 1 [default: cat]: ").strip() or "cat"
    word_b = input("Enter Word 2 [default: dog]: ").strip() or "dog"

    vec_a = EMBEDDING_ENGINE.get_word_embedding(word_a)
    vec_b = EMBEDDING_ENGINE.get_word_embedding(word_b)
    sim_data = EMBEDDING_ENGINE.cosine_similarity(vec_a, vec_b)

    print(f"\nCosine Similarity between '{word_a}' and '{word_b}':")
    print(f"  Score: {sim_data['similarity']} ({sim_data['percentage']}%)")
    print(f"  Angle between vectors: {sim_data['angle_degrees']} degrees")
    print(f"  Interpretation: {sim_data['interpretation']}")

    print("\nNow let's test Vector Arithmetic (Concept Algebra):")
    print("Formula: King - Man + Woman = ?")
    arith = EMBEDDING_ENGINE.vector_arithmetic(["king", "woman"], ["man"])
    print(f"  Top result: {arith['winner']}")
    print("  Top 5 nearest neighbors:")
    for match in arith['top_matches'][:5]:
        print(f"    - {match['word']:<10} (similarity: {match['similarity']})")


def run_sampling_demo():
    print_header("Lesson 3: Logits, Softmax & Temperature (How AI Picks Words)")
    print("The neural network outputs raw confidence scores (Logits) for every word.")
    print("Softmax turns logits into probabilities that sum to 100%.")
    print("Temperature controls whether the model plays it safe or takes creative risks.\n")

    tokens = ["coffee", "tea", "water", "juice", "soda", "lava"]
    logits = [4.5, 3.8, 2.5, 1.8, 0.5, -4.0]

    for temp in [0.2, 1.0, 1.8]:
        res = simulate_sampling_pipeline(tokens, logits, temperature=temp, seed=42)
        print(f"\n--- Temperature = {temp} ---")
        for c in res["candidates"]:
            bar = "#" * int(c["temp_prob"] * 30)
            print(f"  {c['token']:<8} (Logit: {c['logit']:>4.1f}) -> Prob: {c['temp_prob']:>6.2%} | {bar}")
        print(f"  Selected Word: '{res['winner']}'")

    print("\nKey Takeaway: Low temperature focuses all probability onto the top choice.")
    print("High temperature flattens the chart, giving unexpected words a chance!")


def run_rag_demo():
    print_header("Lesson 4: RAG (Retrieval-Augmented Generation)")
    print("RAG prevents AI hallucinations by searching an external vector database")
    print("and pasting relevant knowledge into the prompt context.\n")

    query = input("Ask a question to search our knowledge base [e.g., 'What is attention?']: ").strip()
    if not query:
        query = "How does attention work?"

    rag_res = RAG_ENGINE.construct_augmented_prompt(query, top_k=2)
    print(f"\nSearch Query: '{query}'")
    print(f"Found {len(rag_res['retrieved_docs'])} relevant documents:")
    for i, doc in enumerate(rag_res['retrieved_docs']):
        print(f"  [{i+1}] {doc['title']} (Cosine Match: {doc['similarity_percentage']}%)")
        print(f"      Topic: {doc['topic']}")
        print(f"      Snippet: {doc['snippet']}")

    print("\nThis retrieved context is injected into the LLM's prompt automatically!")


def run_chatbot_demo():
    print_header("Lesson 5: Glass-Box Chatbot (Full Interactive Pipeline)")
    print("Ask any question about AI concepts. Every step will be displayed!")
    print("Type 'exit' to return to menu.\n")

    while True:
        prompt = input("You: ").strip()
        if not prompt or prompt.lower() in ["exit", "quit", "q"]:
            break

        res = CHATBOT_ENGINE.generate_response(prompt, temperature=0.7)
        
        print("\n--- Pipeline Telemetry ---")
        print(f"Tokens in prompt: {res['telemetry']['input_tokens_count']} | Generated: {res['telemetry']['output_tokens_count']}")
        print(f"Execution time  : {res['telemetry']['elapsed_ms']} ms")
        if res['stages']['stage_2_vector_search']:
            top_match = res['stages']['stage_2_vector_search'][0]
            print(f"RAG Retrieved   : '{top_match['title']}' (Score: {top_match['similarity_percentage']}%)")
        
        print("\nAI Response:")
        print(res["response"])
        print("-" * 50 + "\n")


def main():
    while True:
        print_header("AI Under-The-Hood: Educational Studio")
        print("1. Lesson 1: Tokenization (Char vs Word vs BPE)")
        print("2. Lesson 2: Embeddings & Vector Math (King - Man + Woman = Queen)")
        print("3. Lesson 3: Logits, Softmax & Temperature")
        print("4. Lesson 4: RAG (Retrieval-Augmented Generation)")
        print("5. Lesson 5: Glass-Box Chatbot with Live Telemetry")
        print("6. Exit")
        
        choice = input("\nSelect a lesson [1-6]: ").strip()
        if choice == "1":
            run_tokenizer_demo()
        elif choice == "2":
            run_embeddings_demo()
        elif choice == "3":
            run_sampling_demo()
        elif choice == "4":
            run_rag_demo()
        elif choice == "5":
            run_chatbot_demo()
        elif choice == "6":
            print("\nHappy AI Learning! Goodbye!\n")
            break
        else:
            print("Invalid choice. Please enter 1-6.")


if __name__ == "__main__":
    main()
