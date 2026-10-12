# ⚡ Real-Time AI Chatbot & Interactive Architecture Studio

A hands-on, glass-box educational platform designed to demonstrate **how Artificial Intelligence and Large Language Models (LLMs) operate in real time**.

No black boxes, no mysterious third-party API dependencies required. Every calculation—from **Tokenization** and **Vector Embeddings** to **Multi-Head Self-Attention**, **Cosine Similarity**, **RAG Retrieval**, **Logits**, **Softmax Temperature Sampling**, and **Server-Sent Events (SSE) Streaming**—is implemented transparently with pure Python and NumPy.

---

## 🚀 Quick Start

### 1. Launch the Interactive Web Studio
```bash
cd C:\Users\Unique\.gemini\antigravity\scratch\ai_learning_chatbot
python app.py
```
Open your browser at **[http://localhost:5000](http://localhost:5000)**.

### 2. Run the Terminal-Based Learning CLI (with Live Streaming)
```bash
python learn_ai_cli.py
```

### 3. Run the Automated Test Suite
```bash
python test_app.py
```

---

## ⚡ What Makes It "Real Time"?

1. **Autoregressive Streaming Generation (Server-Sent Events - SSE):**
   - The chatbot does not make you wait for the full paragraph. It predicts and streams **one token at a time** in real time, mimicking modern models like Gemini and ChatGPT.
   - Includes adjustable stream delay controls (from 5ms high-speed up to slow-motion 150ms).

2. **Live Next-Token Candidate Probability Race:**
   - As each token is emitted, an animated live horizontal bar chart shows which candidate tokens competed across the vocabulary and why the winner was chosen based on logits and Softmax temperature.

3. **Active Attention Link Radar:**
   - Visualizes which words in the context window the model paid attention to while generating the current word.

4. **Live As-You-Type Tokenizer HUD:**
   - Watch text transform into Byte-Pair Encoding (BPE) subword chips with integer IDs as you type before even pressing Send.

5. **Live Telemetry HUD:**
   - Real-time speedometer tracking tokens per second (tok/s), completion percentage, elapsed milliseconds, and context window consumption.

---

## 🔬 Interactive Labs & Modules

| Lab | Features & Concepts |
| :--- | :--- |
| **💬 Real-Time Glass-Box Chat** | Live token streaming, animated candidate probability race, active attention connections, live pipeline flowchart. |
| **⚡ Transformer Attention Engine** | Interactive $N \times N$ attention heatmap matrix ($\text{Softmax}(Q K^T / \sqrt{d})$), hover token arc webs, 4 multi-head filters (Coreference, Syntactic, Locality, Global). |
| **🔤 Tokenizer Laboratory** | Subword BPE vs Word vs Character tokenizers, compression ratios, and the famous "strawberry" letter-counting breakdown. |
| **📐 Embeddings & Vectors** | Cosine similarity gauge, 16D semantic space, vector math (`King - Man + Woman = Queen`), interactive 2D PCA cluster map. |
| **🎲 Temperature & Softmax** | Dynamic sliders for Temperature ($T$), Top-K, and Top-P (Nucleus), live probability distribution reshape and entropy analysis. |
| **📚 RAG Knowledge Base** | Real-time semantic search, cosine similarity ranking, custom document ingestion and prompt context injection. |
| **📖 How AI Works Guide** | In-depth walkthrough covering Autoregressive decoding, KV Caching, GPUs & Tensor Cores, Loss & Backpropagation, and Alignment. |

---

## 📁 Project Architecture

```
ai_learning_chatbot/
├── app.py                  # Flask Web server with SSE streaming & REST APIs
├── learn_ai_cli.py         # Terminal-based interactive tutorial with live streaming
├── test_app.py             # Automated integration test suite
├── ai_core/
│   ├── attention.py        # Multi-head self-attention matrix & coreference engine
│   ├── tokenizer.py        # Char, Word, and Subword (BPE) tokenizers
│   ├── embeddings.py       # 16D semantic space, cosine similarity & PCA
│   ├── sampling.py         # Logits, Softmax, Temperature, Top-K & Top-P
│   ├── rag_engine.py       # Vector DB, cosine search & prompt augmentation
│   └── generator.py        # Glass-box real-time streaming response engine
├── templates/
│   └── index.html          # Responsive glass-box UI & HUD layout
└── static/
    ├── css/style.css       # Dark-theme HUD styles & animations
    └── js/app.js           # Real-time SSE streaming client & attention visualizers
```
