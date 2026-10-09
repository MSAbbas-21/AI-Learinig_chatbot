# 🧠 AI Under The Hood: Interactive Learning Studio & Glass-Box Chatbot

A hands-on educational application designed to demystify **how Artificial Intelligence and Large Language Models (LLMs) actually work under the hood**.

No black boxes, no mysterious third-party APIs required. Every calculation—from **Tokenization** and **Vector Embeddings** to **Cosine Similarity**, **RAG Retrieval**, **Logits**, and **Softmax Temperature Sampling**—is implemented transparently with pure Python and NumPy.

---

## 🚀 Quick Start

### 1. Launch the Interactive Web Studio
```bash
cd C:\Users\Unique\.gemini\antigravity\scratch\ai_learning_chatbot
python app.py
```
Then open your browser at **[http://localhost:5000](http://localhost:5000)**.

### 2. Run the Terminal-Based Learning CLI
If you prefer an interactive terminal walkthrough:
```bash
python learn_ai_cli.py
```

### 3. Run the Automated Tests
```bash
python test_app.py
```

---

## 🔍 Core Concepts You Will Learn

### 1. Tokenization: How Computers Read Text
* **The Problem:** Computers cannot read words or letters. Neural networks only perform matrix multiplications on numbers.
* **Why not characters?** Character tokenization (`c-a-t`) creates massive sequence lengths, causing the quadratic attention cost $O(N^2)$ to explode.
* **Why not full words?** Word tokenization fails on typos, code syntax, and requires an infinite vocabulary dictionary.
* **The Solution (BPE - Byte-Pair Encoding):** Modern LLMs (GPT, Gemini, Claude) break words into frequent subword chunks (e.g. `strawberry` $\to$ `['straw', 'berry']`).
* *Interactive Feature:* Explore color-coded subwords, token IDs, and compression ratios in the **Tokenizer Lab**.

---

### 2. Embeddings & Geometry of Meaning
* **The Problem:** A token ID like `42` has no inherent meaning.
* **The Solution:** The model maps each token ID to an **Embedding Vector** (a list of numbers representing coordinates in a high-dimensional space).
* **Cosine Similarity:** Measures the angle $\theta$ between two vectors:
  $$\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$
* **Concept Algebra:** Notice how concepts add and subtract geometrically:
  $$\vec{\text{King}} - \vec{\text{Man}} + \vec{\text{Woman}} \approx \vec{\text{Queen}}$$
* *Interactive Feature:* Calculate semantic distances and explore a live 2D PCA projection map in the **Embeddings & Vectors** tab.

---

### 3. RAG: Retrieval-Augmented Generation
* **Why RAG?** LLMs have frozen weights and cannot know real-time data or private company documents. If forced to answer unfamiliar topics, they **hallucinate** plausible-sounding fictions.
* **How RAG works:**
  1. Chunk external knowledge into documents.
  2. Embed chunks into vectors using an embedding model.
  3. When a user asks a question, embed the user query.
  4. Perform **Cosine Similarity vector search** to find top-$K$ matching documents.
  5. Stuff the retrieved facts into the prompt context for the model.
* *Interactive Feature:* Add custom documents to the vector database and watch live semantic search ranking in the **RAG Vector DB** tab.

---

### 4. Attention & The Transformer Architecture
* **Self-Attention:** Allows each word to look at all other words in the sentence to adapt its meaning to context:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
  * **Query ($Q$):** What this token is seeking.
  * **Key ($K$):** What this token provides.
  * **Value ($V$):** The actual semantic information transferred.

---

### 5. Next-Token Prediction & Softmax Sampling
* **Logits:** Raw unbounded scores output by the final projection layer.
* **Softmax Formula:** Converts logits into a probability distribution summing to 100%:
  $$P_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
* **Temperature ($T$):**
  * $T \to 0$ (Cold / Greedy): The highest logit dominates ($P \to 1.0$). Robotic, factual, deterministic.
  * $T > 1.0$ (Hot): Probabilities flatten. Unlikely words get chosen. Creative, wild, risk of hallucinations.
* **Top-K:** Truncates candidates to the top $K$ choices.
* **Top-P (Nucleus):** Dynamically retains only the smallest subset of candidates whose cumulative probability reaches $P$.
* *Interactive Feature:* Move real-time sliders and watch probability bars reshape in the **Temperature & Sampling** tab.

---

## 📁 Project Architecture

```
ai_learning_chatbot/
├── app.py                  # Flask Web server & REST API
├── learn_ai_cli.py         # Terminal-based interactive tutorial
├── test_app.py             # Integration test suite
├── ai_core/
│   ├── tokenizer.py        # Char, Word, and Subword (BPE) tokenizers
│   ├── embeddings.py       # 16D semantic space, cosine similarity & PCA
│   ├── sampling.py         # Logits, Softmax, Temperature, Top-K & Top-P
│   ├── rag_engine.py       # Vector DB, cosine search & prompt augmentation
│   └── generator.py        # 5-stage glass-box generation pipeline
├── templates/
│   └── index.html          # Web UI layout with 6 interactive tabs
├── static/
│   ├── css/style.css       # Modern dark-theme styling
│   └── js/app.js           # Interactive client scripts, charts & canvas
└── data/                   # Data storage directory
```

---

## 🎓 Recommended Learning Order

1. **Start with the Tokenizer Lab**: Type different words, code snippets, and sentences to see how words turn into token IDs.
2. **Move to the Embeddings Tab**: Experiment with comparing words (`dog` vs `cat`, `cat` vs `computer`) and test vector arithmetic.
3. **Play with Temperature & Sampling**: Slide the temperature from `0.1` to `2.0` and observe how probability distributions change.
4. **Test the RAG Vector DB**: Query the knowledge base, add a custom document, and see how cosine similarity matches your query.
5. **Chat with the Glass-Box Bot**: Send messages and observe all 4 telemetry stages operating simultaneously in real time!
