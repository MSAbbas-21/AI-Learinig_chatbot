/**
 * AI Under The Hood — Interactive Studio Client
 */

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initChat();
  initTokenizerLab();
  initEmbeddingsLab();
  initSamplingLab();
  initRagLab();
});

/* ==================== 1. TAB NAVIGATION ==================== */
function initTabs() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));

      tab.classList.add("active");
      const targetPane = document.getElementById(tab.dataset.tab);
      if (targetPane) {
        targetPane.classList.add("active");
        if (tab.dataset.tab === "tab-embeddings") {
          loadPcaMap();
        }
      }
    });
  });
}

/* ==================== 2. TAB 1: GLASS-BOX CHAT ==================== */
function initChat() {
  const form = document.getElementById("chatForm");
  const input = document.getElementById("chatInput");
  const messagesBox = document.getElementById("chatMessages");
  const tempSlider = document.getElementById("chatTemp");
  const topPSlider = document.getElementById("chatTopP");
  const topKSlider = document.getElementById("chatTopK");

  tempSlider.addEventListener("input", (e) => document.getElementById("tempVal").innerText = e.target.value);
  topPSlider.addEventListener("input", (e) => document.getElementById("topPVal").innerText = e.target.value);
  topKSlider.addEventListener("input", (e) => document.getElementById("topKVal").innerText = e.target.value);

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = input.value.trim();
    if (!query) return;

    // Append User Message
    appendChatMessage("user", query);
    input.value = "";
    input.disabled = true;

    // Set badge to thinking
    const badge = document.getElementById("telemetryBadge");
    badge.innerText = "Processing...";
    badge.style.color = "var(--warning)";

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: query,
          temperature: parseFloat(tempSlider.value),
          top_p: parseFloat(topPSlider.value),
          top_k: parseInt(topKSlider.value),
          enable_rag: document.getElementById("chatRagToggle").checked
        })
      });

      const data = await res.json();
      if (data.error) {
        appendChatMessage("bot", `⚠️ Error: ${data.error}`);
      } else {
        appendChatMessage("bot", data.response);
        renderChatTelemetry(data);
      }
    } catch (err) {
      appendChatMessage("bot", `⚠️ Connection error: ${err.message}`);
    } finally {
      input.disabled = false;
      input.focus();
      badge.innerText = "Completed";
      badge.style.color = "var(--success)";
    }
  });
}

window.sendQuickPrompt = function(promptText) {
  const input = document.getElementById("chatInput");
  input.value = promptText;
  document.getElementById("chatForm").dispatchEvent(new Event("submit"));
};

function appendChatMessage(role, content) {
  const box = document.getElementById("chatMessages");
  const bubble = document.createElement("div");
  bubble.className = `message-bubble ${role === "user" ? "user-bubble" : "bot-bubble"}`;

  const header = document.createElement("div");
  header.className = "msg-header";
  header.innerHTML = `
    <span class="sender-name">${role === "user" ? "👤 You" : "🤖 Glass-Box AI"}</span>
    <span class="msg-time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
  `;

  const body = document.createElement("div");
  body.className = "msg-content";
  body.innerHTML = formatMarkdown(content);

  bubble.appendChild(header);
  bubble.appendChild(body);
  box.appendChild(bubble);
  box.scrollTop = box.scrollHeight;
}

function formatMarkdown(text) {
  if (!text) return "";
  let html = text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n- (.*?)/g, '<li>$1</li>');

  if (html.includes('<li>')) {
    html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
  }
  return `<p>${html}</p>`;
}

function renderChatTelemetry(data) {
  // Stage 1: Tokenization
  const tokensBox = document.getElementById("telemetryTokens");
  tokensBox.innerHTML = "";
  const bpe = data.stages.stage_1_tokenization.bpe;
  bpe.chips.forEach(chip => {
    const el = document.createElement("span");
    el.className = "token-chip";
    el.style.backgroundColor = chip.color;
    el.innerHTML = `${chip.is_space ? '␣' : escapeHtml(chip.token)} <span class="token-id-tag">#${chip.id}</span>`;
    tokensBox.appendChild(el);
  });

  document.getElementById("statChars").innerText = data.stages.stage_1_tokenization.char_count;
  document.getElementById("statTokens").innerText = bpe.token_count;
  document.getElementById("statRatio").innerText = bpe.compression_ratio;

  // Stage 2: RAG Search
  const ragBox = document.getElementById("telemetryRagBox");
  ragBox.innerHTML = "";
  const hits = data.stages.stage_2_vector_search;
  if (!hits || hits.length === 0) {
    ragBox.innerHTML = '<span class="placeholder-text">RAG disabled or no matching documents retrieved.</span>';
  } else {
    hits.forEach(hit => {
      const card = document.createElement("div");
      card.className = "rag-hit-card";
      card.innerHTML = `
        <div class="rag-hit-title">
          <span>${escapeHtml(hit.title)}</span>
          <span class="rag-hit-score">${hit.similarity_percentage}% match</span>
        </div>
        <div class="rag-hit-body">${escapeHtml(hit.snippet)}</div>
      `;
      ragBox.appendChild(card);
    });
  }

  // Stage 3: Prompt Assembly
  const promptBox = document.getElementById("telemetryPromptCode");
  promptBox.innerText = data.stages.stage_3_prompt_assembly.full_prompt;

  // Stage 4: Next-Token Sampling Telemetry
  const samplingBox = document.getElementById("telemetrySamplingBox");
  samplingBox.innerHTML = "";
  const steps = data.stages.stage_4_sampling_telemetry;
  if (steps && steps.length > 0) {
    steps.forEach((step, idx) => {
      const stepEl = document.createElement("div");
      stepEl.className = "telemetry-step-block";
      stepEl.style.marginBottom = "0.75rem";
      
      let candidatesHtml = step.candidates.slice(0, 4).map(c => `
        <div style="display:flex; justify-content:space-between; font-size:0.75rem; margin-top:2px;">
          <span>${c.is_winner ? '🏆 ' : ''}<code>${escapeHtml(c.token)}</code> (logit: ${c.logit})</span>
          <span style="font-family:var(--font-mono); color:${c.is_winner ? 'var(--success)' : 'var(--text-muted)'}">${c.percentage}%</span>
        </div>
      `).join("");

      stepEl.innerHTML = `
        <div style="font-size:0.78rem; font-weight:600; color:var(--primary); margin-bottom:0.2rem;">
          Token Step #${step.step_number}: Chosen: <code>${escapeHtml(step.winner)}</code>
        </div>
        ${candidatesHtml}
      `;
      samplingBox.appendChild(stepEl);
    });
  }
}

/* ==================== 3. TAB 2: TOKENIZER LAB ==================== */
function initTokenizerLab() {
  const analyzeBtn = document.getElementById("tokAnalyzeBtn");
  const textarea = document.getElementById("tokInputText");

  async function runTokenize() {
    const text = textarea.value.trim();
    if (!text) return;

    try {
      const res = await fetch("/api/tokenize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });
      const data = await res.json();
      renderTokenizerResults(data);
    } catch (err) {
      console.error("Tokenize failed:", err);
    }
  }

  analyzeBtn.addEventListener("click", runTokenize);
  runTokenize(); // Initial run
}

window.setTokPreset = function(presetText) {
  const textarea = document.getElementById("tokInputText");
  textarea.value = presetText;
  document.getElementById("tokAnalyzeBtn").click();
};

function renderTokenizerResults(data) {
  // BPE
  const bpeBox = document.getElementById("bpeDisplayChips");
  bpeBox.innerHTML = "";
  data.bpe.chips.forEach(chip => {
    const el = document.createElement("span");
    el.className = "token-chip";
    el.style.backgroundColor = chip.color;
    el.title = `Token ID: ${chip.id}`;
    el.innerHTML = `${chip.is_space ? '␣' : escapeHtml(chip.token)} <span class="token-id-tag">#${chip.id}</span>`;
    bpeBox.appendChild(el);
  });
  document.getElementById("bpeCount").innerText = data.bpe.token_count;
  document.getElementById("bpeRatio").innerText = data.bpe.compression_ratio;

  // Word
  const wordBox = document.getElementById("wordDisplayChips");
  wordBox.innerHTML = "";
  data.word.tokens.forEach((t, i) => {
    const el = document.createElement("span");
    el.className = "token-chip";
    el.style.backgroundColor = "#475569";
    el.innerHTML = `${escapeHtml(t)} <span class="token-id-tag">#${data.word.token_ids[i]}</span>`;
    wordBox.appendChild(el);
  });
  document.getElementById("wordCount").innerText = data.word.token_count;

  // Char
  const charBox = document.getElementById("charDisplayChips");
  charBox.innerHTML = "";
  data.char.tokens.slice(0, 45).forEach((ch, i) => {
    const el = document.createElement("span");
    el.className = "token-chip";
    el.style.backgroundColor = "#334155";
    el.innerHTML = `${ch === ' ' ? '␣' : escapeHtml(ch)}`;
    charBox.appendChild(el);
  });
  if (data.char.tokens.length > 45) {
    charBox.innerHTML += `<span style="font-size:0.75rem; color:var(--text-dim); align-self:center;">...+${data.char.tokens.length - 45} chars</span>`;
  }
  document.getElementById("charCount").innerText = data.char.token_count;
}

/* ==================== 4. TAB 3: EMBEDDINGS LAB ==================== */
function initEmbeddingsLab() {
  const calcBtn = document.getElementById("calcSimBtn");
  const arithBtn = document.getElementById("runArithBtn");
  const reloadPcaBtn = document.getElementById("reloadPcaBtn");

  calcBtn.addEventListener("click", async () => {
    const a = document.getElementById("simWordA").value.trim();
    const b = document.getElementById("simWordB").value.trim();
    if (!a || !b) return;

    try {
      const res = await fetch("/api/embeddings/similarity", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text1: a, text2: b })
      });
      const data = await res.json();
      document.getElementById("simScoreVal").innerText = `${data.percentage}%`;
      document.getElementById("simAngleVal").innerText = `${data.angle_degrees}°`;
      document.getElementById("simDotVal").innerText = data.dot_product;
      document.getElementById("simInterpVal").innerText = data.interpretation;
    } catch (err) {
      console.error("Similarity calc failed:", err);
    }
  });

  arithBtn.addEventListener("click", async () => {
    const pos1 = document.getElementById("arithPos1").value.trim();
    const neg1 = document.getElementById("arithNeg1").value.trim();
    const pos2 = document.getElementById("arithPos2").value.trim();
    if (!pos1 || !pos2) return;

    try {
      const res = await fetch("/api/embeddings/arithmetic", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ positive: [pos1, pos2], negative: neg1 ? [neg1] : [] })
      });
      const data = await res.json();
      document.getElementById("arithWinnerWord").innerText = data.winner;

      const list = document.getElementById("arithMatchesList");
      list.innerHTML = "";
      data.top_matches.forEach(m => {
        const item = document.createElement("div");
        item.className = "match-item";
        item.innerHTML = `
          <span>${m.is_input ? '(Input) ' : ''}<strong>${escapeHtml(m.word)}</strong></span>
          <span class="match-score">${(m.similarity * 100).toFixed(1)}%</span>
        `;
        list.appendChild(item);
      });
    } catch (err) {
      console.error("Vector arith failed:", err);
    }
  });

  reloadPcaBtn.addEventListener("click", loadPcaMap);

  // Initial runs
  calcBtn.click();
  arithBtn.click();
}

window.setSimPair = function(w1, w2) {
  document.getElementById("simWordA").value = w1;
  document.getElementById("simWordB").value = w2;
  document.getElementById("calcSimBtn").click();
};

async function loadPcaMap() {
  const canvas = document.getElementById("pcaCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  try {
    const res = await fetch("/api/embeddings/projection");
    const data = await res.json();
    const points = data.points;

    // Draw background & grid
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    ctx.strokeStyle = "#1e293b";
    ctx.lineWidth = 1;

    // Center axes
    ctx.beginPath();
    ctx.moveTo(w / 2, 0);
    ctx.lineTo(w / 2, h);
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    const catColors = {
      royalty_people: "#a855f7",
      animals: "#10b981",
      tech_ai: "#38bdf8",
      food: "#f59e0b",
      vehicles: "#ef4444",
      emotions: "#ec4899",
      other: "#94a3b8"
    };

    // Draw points
    points.forEach(pt => {
      // Map [-80, 80] to canvas coords
      const px = (w / 2) + (pt.x / 80) * (w / 2 - 50);
      const py = (h / 2) - (pt.y / 80) * (h / 2 - 40);

      const color = catColors[pt.category] || "#94a3b8";

      // Outer glow circle
      ctx.beginPath();
      ctx.arc(px, py, 6, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();

      // Label text
      ctx.fillStyle = "#f8fafc";
      ctx.font = "12px 'Inter', sans-serif";
      ctx.fillText(pt.word, px + 9, py + 4);
    });
  } catch (err) {
    console.error("Failed loading PCA:", err);
  }
}

/* ==================== 5. TAB 4: SAMPLING LAB ==================== */
function initSamplingLab() {
  const tempSlider = document.getElementById("labTempSlider");
  const topKSlider = document.getElementById("labTopKSlider");
  const topPSlider = document.getElementById("labTopPSlider");
  const sampleBtn = document.getElementById("sampleTokenBtn");

  const updateSim = async () => {
    document.getElementById("labTempBadge").innerText = tempSlider.value;
    document.getElementById("labTopKBadge").innerText = topKSlider.value;
    document.getElementById("labTopPBadge").innerText = topPSlider.value;

    try {
      const res = await fetch("/api/sampling/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          temperature: parseFloat(tempSlider.value),
          top_k: parseInt(topKSlider.value),
          top_p: parseFloat(topPSlider.value)
        })
      });
      const data = await res.json();
      renderSamplingBars(data);
    } catch (err) {
      console.error("Sampling sim failed:", err);
    }
  };

  tempSlider.addEventListener("input", updateSim);
  topKSlider.addEventListener("input", updateSim);
  topPSlider.addEventListener("input", updateSim);
  sampleBtn.addEventListener("click", updateSim);

  updateSim(); // Initial render
}

function renderSamplingBars(data) {
  const container = document.getElementById("probabilityBars");
  container.innerHTML = "";

  document.getElementById("samplingWinnerWord").innerText = data.winner;

  data.candidates.forEach(c => {
    const row = document.createElement("div");
    row.className = "bar-row";

    const isCut = !c.is_active;
    const isWin = c.is_winner;

    row.innerHTML = `
      <div class="bar-label-group">
        <span class="bar-word" style="color:${isWin ? 'var(--success)' : isCut ? 'var(--text-dim)' : 'var(--text-main)'}">
          ${isWin ? '🏆 ' : ''}${escapeHtml(c.token)} ${isCut ? '(Filtered Out)' : ''}
        </span>
        <span class="bar-stats">
          Logit: ${c.logit} | Prob: <strong>${c.percentage}%</strong>
        </span>
      </div>
      <div class="bar-track">
        <div class="bar-fill ${isWin ? 'bar-winner' : isCut ? 'bar-cut' : ''}" style="width:${Math.max(c.percentage, 0.5)}%"></div>
      </div>
    `;
    container.appendChild(row);
  });

  // Notes
  const notesBox = document.getElementById("samplingNotesBox");
  notesBox.innerHTML = "";
  data.notes.forEach(note => {
    const p = document.createElement("p");
    p.innerText = note;
    notesBox.appendChild(p);
  });
}

/* ==================== 6. TAB 5: RAG LAB ==================== */
function initRagLab() {
  const searchInput = document.getElementById("ragSearchInput");
  const searchBtn = document.getElementById("ragSearchBtn");
  const addForm = document.getElementById("addDocForm");

  searchBtn.addEventListener("click", async () => {
    const q = searchInput.value.trim();
    if (!q) return;

    try {
      const res = await fetch("/api/rag/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, top_k: 3 })
      });
      const data = await res.json();
      renderRagSearchResults(data.retrieved_docs);
    } catch (err) {
      console.error("RAG search failed:", err);
    }
  });

  addForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const title = document.getElementById("docTitle").value.trim();
    const topic = document.getElementById("docTopic").value.trim();
    const content = document.getElementById("docContent").value.trim();

    try {
      const res = await fetch("/api/rag/documents", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, topic, content })
      });
      const data = await res.json();
      if (data.status === "success") {
        alert("Document successfully indexed & embedded into the vector database!");
        addForm.reset();
        loadRagDocuments();
      }
    } catch (err) {
      console.error("Add doc failed:", err);
    }
  });

  loadRagDocuments();
}

async function loadRagDocuments() {
  try {
    const res = await fetch("/api/rag/documents");
    const data = await res.json();
    const docs = data.documents;

    document.getElementById("totalDocsCount").innerText = docs.length;
    const accordion = document.getElementById("docsAccordion");
    accordion.innerHTML = "";

    docs.forEach(doc => {
      const card = document.createElement("div");
      card.className = "doc-item-card";
      card.innerHTML = `
        <h5>
          <span>${escapeHtml(doc.title)}</span>
          <span class="tag-pill tag-primary">${escapeHtml(doc.topic)}</span>
        </h5>
        <p>${escapeHtml(doc.content)}</p>
      `;
      accordion.appendChild(card);
    });
  } catch (err) {
    console.error("Load docs failed:", err);
  }
}

function renderRagSearchResults(results) {
  const box = document.getElementById("ragSearchResults");
  box.innerHTML = "";

  if (!results || results.length === 0) {
    box.innerHTML = '<span class="placeholder-text">No matches found.</span>';
    return;
  }

  results.forEach((r, idx) => {
    const item = document.createElement("div");
    item.className = "rag-hit-card";
    item.innerHTML = `
      <div class="rag-hit-title">
        <span>#${idx+1} ${escapeHtml(r.title)} (${escapeHtml(r.topic)})</span>
        <span class="rag-hit-score">${r.similarity_percentage}% match</span>
      </div>
      <div class="rag-hit-body">${escapeHtml(r.content)}</div>
      <div style="font-size:0.7rem; color:var(--text-dim); margin-top:3px;">
        Assessment: ${escapeHtml(r.interpretation)}
      </div>
    `;
    box.appendChild(item);
  });
}

/* ==================== UTILITIES ==================== */
function escapeHtml(text) {
  if (!text) return "";
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
