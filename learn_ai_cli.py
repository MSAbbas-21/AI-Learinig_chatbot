"""
Interactive AI Educational CLI
==============================
Run this script in your terminal to explore how AI really works in real time:
`python learn_ai_cli.py`

Covers:
1. Interactive Tokenizer (Char vs Word vs Subword/BPE)
2. Vector Arithmetic & Cosine Similarity (King - Man + Woman = Queen)
3. Logits, Softmax & Temperature Simulator
4. Self-Attention Engine & Heatmap Matrix
5. Live RAG Vector Search
6. Real-Time Streaming Glass-Box Chatbot
"""

import sys
import os
import time

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
from ai_core.attention import ATTENTION_ENGINE


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
    print("  Tokens : " + " | ".join(f"'{t}'" for t in analysis['word']['tokens']))
    print(f"  Token Count: {analysis['word']['token_count']}")

    print("\n3. Character Tokenizer:")
    print(f"  Token Count: {analysis['char']['token_count']} (Very long sequence, memory hungry!)")

    print("\n💡 Why LLMs struggle with 'strawberry':")
    print("Because 'strawberry' is compressed into tokens ['straw', 'berry'], the neural network")
    print("never sees individual letters directly. It must guess character counts from context!")


def run_embeddings_demo():
    print_header("Lesson 2: Embeddings & Vector Arithmetic")
    print("Words become dense vectors in high-dimensional space.")
    print("Semantics are measured with Cosine Similarity: cos(theta) = (u . v) / (||u|| ||v||).\n")

    w1 = input("Word 1 [default: 'king']: ").strip() or "king"
    w2 = input("Word 2 [default: 'queen']: ").strip() or "queen"

    vec1 = EMBEDDING_ENGINE.get_text_embedding(w1)
    vec2 = EMBEDDING_ENGINE.get_text_embedding(w2)
    sim = EMBEDDING_ENGINE.cosine_similarity(vec1, vec2)

    print(f"\nComparing '{w1}' vs '{w2}':")
    print(f"  Cosine Similarity Score : {sim['percentage']}% ({sim['similarity']:.4f})")
    print(f"  Angle between vectors   : {sim['angle_degrees']} degrees")
    print(f"  Interpretation          : {sim['interpretation']}")

    print("\nNow testing Vector Concept Algebra: King - Man + Woman = ?")
    arith = EMBEDDING_ENGINE.vector_arithmetic(["king", "woman"], ["man"])
    print(f"  Calculated Winner: '{arith['winner'].upper()}'!")
    print("  Top Matches:")
    for m in arith['top_matches'][:4]:
        print(f"    - {m['word']}: {m['similarity']*100:.1f}%")


def run_sampling_demo():
    print_header("Lesson 3: Logits, Softmax & Temperature")
    print("Neural networks output unbounded raw scores (Logits).")
    print("Softmax converts logits into probabilities summing to 100%:")
    print("  P_i = exp(z_i / T) / sum(exp(z_j / T))\n")

    tokens = ["coffee", "tea", "water", "juice", "soda", "lava"]
    logits = [4.5, 3.8, 2.5, 1.8, 0.5, -4.0]

    for temp in [0.2, 1.0, 1.8]:
        res = simulate_sampling_pipeline(tokens, logits, temperature=temp, top_k=6, top_p=1.0)
        print(f"--- Temperature T = {temp} ---")
        for c in res["candidates"][:4]:
            bar = "#" * int(c["percentage"] / 4)
            print(f"  {c['token']:<8} | Logit: {c['logit']:>4} | Prob: {c['percentage']:>5.1f}% | {bar}")
        print(f"  >> Winner Sampled: '{res['winner']}'\n")


def run_attention_demo():
    print_header("Lesson 4: Self-Attention & Transformer Mechanics")
    print("Self-Attention computes how much each word attends to every other word:")
    print("  Attention(Q, K, V) = Softmax(Q K^T / sqrt(d_k)) V\n")

    sentence = input("Enter sentence [default: 'The animal didn\\'t cross the street because it was too tired']: ").strip()
    if not sentence:
        sentence = "The animal didn't cross the street because it was too tired"

    att = ATTENTION_ENGINE.compute_attention(sentence, head_type="coreference")
    tokens = att["tokens"]
    matrix = att["attention_matrix"]

    print(f"\nSentence tokens ({len(tokens)}):")
    print("  " + " | ".join(f"[{i}] {t}" for i, t in enumerate(tokens)))

    print("\nKey Attention Connections:")
    for top in att["top_attended"][:5]:
        print(f"  Token '{top['from_token']}' attends most strongly to '{top['to_token']}' ({top['weight_percentage']}%)")


def run_rag_demo():
    print_header("Lesson 5: RAG (Retrieval-Augmented Generation)")
    print("Instead of retraining the model, RAG searches external documents using vectors\n")

    query = input("Ask a question [default: 'What is attention?']: ").strip()
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
    print_header("Lesson 6: Real-Time Glass-Box Chatbot with Live Streaming")
    print("Ask any question about AI concepts. Watch the real-time token streaming & pipeline!")
    print("Type 'exit' to return to menu.\n")

    while True:
        prompt = input("\nYou: ").strip()
        if not prompt or prompt.lower() in ["exit", "quit", "q"]:
            break

        print("\n🤖 AI [Streaming tokens in real time]:\n")
        start_t = time.time()
        token_count = 0

        for event in CHATBOT_ENGINE.generate_response_stream(prompt, temperature=0.7, token_delay_ms=20):
            if event["stage"] == "stage_stream_token":
                sys.stdout.write(event["token"])
                sys.stdout.flush()
                token_count += 1
            elif event["stage"] == "stage_complete":
                elapsed = event["telemetry"]["elapsed_ms"]
                tok_sec = event["telemetry"]["tokens_per_second"]
                print(f"\n\n[⚡ Generated {token_count} tokens in {elapsed}ms | Speed: {tok_sec} tok/s]")
                if event.get("matched_topic"):
                    print(f"[Topic Grounded: '{event['matched_topic']}']")


def main():
    while True:
        print_header("Real-Time AI Under-The-Hood: Interactive CLI")
        print("1. Lesson 1: Tokenization (Char vs Word vs BPE)")
        print("2. Lesson 2: Embeddings & Vector Math (King - Man + Woman = Queen)")
        print("3. Lesson 3: Logits, Softmax & Temperature")
        print("4. Lesson 4: Self-Attention & Transformer Heatmap")
        print("5. Lesson 5: RAG (Retrieval-Augmented Generation)")
        print("6. Lesson 6: Real-Time Streaming Glass-Box Chatbot")
        print("7. Exit")
        
        choice = input("\nSelect a lesson [1-7]: ").strip()
        if choice == "1":
            run_tokenizer_demo()
        elif choice == "2":
            run_embeddings_demo()
        elif choice == "3":
            run_sampling_demo()
        elif choice == "4":
            run_attention_demo()
        elif choice == "5":
            run_rag_demo()
        elif choice == "6":
            run_chatbot_demo()
        elif choice == "7":
            print("\nHappy AI Learning! Goodbye!\n")
            break
        else:
            print("Invalid choice. Please enter 1-7.")


if __name__ == "__main__":
    main()
