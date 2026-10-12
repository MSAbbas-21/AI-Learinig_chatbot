/**
 * Real-Time AI Chatbot & Architecture Studio Client
 */

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initChat();
  initAttentionLab();
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
        } else if (tab.dataset.tab === "tab-attention") {
          if (!window._attentionLoaded) {
            document.getElementById("attComputeBtn").click();
            window._attentionLoaded = true;
          }
        }
      }
    });
  });
}

/* ==================== 2. REAL-TIME GLASS-BOX CHAT ==================== */
function initChat() {
  const form = document.getElementById("chatForm");
  const input = document.getElementById("chatInput");
  const tempSlider = document.getElementById("chatTemp");
  const topPSlider = document.getElementById("chatTopP");
  const topKSlider = document.getElementById("chatTopK");
  const speedSlider = document.getElementById("chatSpeed");
  const streamToggle = document.getElementById("chatStreamToggle");
  const ragToggle = document.getElementById("chatRagToggle");

  tempSlider.addEventListener("input", (e) => document.getElementById("tempVal").innerText = e.target.value);
  topPSlider.addEventListener("input", (e) => document.getElementById("topPVal").innerText = e.target.value);
  topKSlider.addEventListener("input", (e) => document.getElementById("topKVal").innerText = e.target.value);
  speedSlider.addEventListener("input", (e) => document.getElementById("speedVal").innerText = `${e.target.value}ms`);

  // Live As-You-Type Tokenizer HUD
  let typeTimer = null;
  input.addEventListener("input", () => {
    clearTimeout(typeTimer);
    const text = input.value;
    if (!text) {
      document.getElementById("typeHudStats").innerText = "0 characters · 0 tokens";
      document.getElementById("typeHudChips").innerHTML = '<span class="placeholder-text-dim">Start typing below to see instant subword token segmentation...</span>';
      return;
    }

    typeTimer = setTimeout(async () => {
      try {
        const res = await fetch("/api/tokenize", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text })
        });
        const data = await res.json();
        const chipsBox = document.getElementById("typeHudChips");
        chipsBox.innerHTML = "";
        data.bpe.chips.forEach(chip => {
          const el = document.createElement("span");
          el.className = "token-chip";
          el.style.backgroundColor = chip.color;
          el.style.fontSize = "0.75rem";
          el.style.padding = "2px 6px";
          el.innerHTML = `${chip.is_space ? '␣' : escapeHtml(chip.token)} <span class="token-id-tag">#${chip.id}</span>`;
          chipsBox.appendChild(el);
        });
        document.getElementById("typeHudStats").innerText = `${data.char_count} chars · ${data.bpe.token_count} BPE tokens (Ratio: ${data.bpe.compression_ratio})`;
      } catch (e) {
        // silent fail for quick typing
      }
    }, 150);
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = input.value.trim();
    if (!query) return;

    // Reset input
    input.value = "";
    document.getElementById("typeHudChips").innerHTML = '<span class="placeholder-text-dim">Start typing below to see instant subword token segmentation...</span>';
    document.getElementById("typeHudStats").innerText = "0 characters · 0 tokens";
    input.disabled = true;

    // Append User Message bubble
    appendChatMessage("user", query);

    const isStreaming = streamToggle.checked;
    if (isStreaming) {
      await handleStreamingChat(query);
    } else {
      await handleSyncChat(query);
    }

    input.disabled = false;
    input.focus();
  });
}

function updatePipelineStep(stepNum) {
  for (let i = 1; i <= 6; i++) {
    const el = document.getElementById(`flowStep${i}`);
    if (!el) continue;
    if (i < stepNum) {
      el.className = "flow-step completed";
    } else if (i === stepNum) {
      el.className = "flow-step active";
    } else {
      el.className = "flow-step";
    }
  }
}

async function handleStreamingChat(query) {
  const badge = document.getElementById("telemetryBadge");
  const statusInd = document.getElementById("streamStatusIndicator");
  badge.innerText = "Streaming...";
  badge.style.color = "var(--warning)";
  if (statusInd) statusInd.classList.add("busy");

  // Create bot bubble with active cursor
  const { bubble, contentEl, cursor } = createStreamingBotBubble();

  // Highlight step 1: Prompt Input
  updatePipelineStep(1);

  const speedMs = parseInt(document.getElementById("chatSpeed").value) || 30;
  const temp = parseFloat(document.getElementById("chatTemp").value) || 0.7;
  const topK = parseInt(document.getElementById("chatTopK").value) || 40;
  const topP = parseFloat(document.getElementById("chatTopP").value) || 0.9;
  const enableRag = document.getElementById("chatRagToggle").checked;

  let accumulatedText = "";
  const url = `/api/chat/stream?message=${encodeURIComponent(query)}&temperature=${temp}&top_k=${topK}&top_p=${topP}&enable_rag=${enableRag}&delay_ms=${speedMs}`;

  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop(); // Keep unfinished tail

      for (const block of lines) {
        const line = block.trim();
        if (line.startsWith("data: ")) {
          try {
            const event = JSON.parse(line.substring(6));
            processStreamEvent(event, contentEl, cursor, (chunk) => {
              accumulatedText += chunk;
            });
          } catch (err) {
            console.error("Failed to parse SSE event:", err, line);
          }
        }
      }
    }

    // Finished
    updatePipelineStep(7); // all done
    cursor.remove();
    contentEl.innerHTML = formatMarkdown(accumulatedText);
    badge.innerText = "Complete";
    badge.style.color = "var(--success)";
  } catch (err) {
    cursor.remove();
    contentEl.innerHTML = `<span style="color:var(--danger)">⚠️ Streaming error: ${escapeHtml(err.message)}</span>`;
    badge.innerText = "Error";
    badge.style.color = "var(--danger)";
  } finally {
    if (statusInd) statusInd.classList.remove("busy");
  }
}

function processStreamEvent(event, contentEl, cursor, appendText) {
  if (event.stage === "stage_1_tokenization") {
    updatePipelineStep(2);
    renderStage1Telemetry(event.data);
  } else if (event.stage === "stage_2_rag") {
    updatePipelineStep(3);
    renderStage2Telemetry(event.retrieved_docs, event.full_prompt);
  } else if (event.stage === "stage_stream_token") {
    updatePipelineStep(5); // In logits & softmax sampling

    // Append token to text
    appendText(event.token);
    cursor.insertAdjacentText("beforebegin", event.token);
    const box = document.getElementById("chatMessages");
    box.scrollTop = box.scrollHeight;

    // Update real-time HUD metrics
    if (event.stats) {
      document.getElementById("mTokSec").innerText = event.stats.tok_per_sec;
      document.getElementById("mProgress").innerText = `${event.stats.progress_pct}%`;
      document.getElementById("mElapsed").innerText = `${event.stats.elapsed_ms}ms`;
      document.getElementById("mTokenTotal").innerText = `${event.token_index} / ${event.total_tokens}`;
    }

    // Update current token & attended token pills
    document.getElementById("liveCurrentToken").innerText = event.token || "<space>";
    document.getElementById("liveAttendedToken").innerText = event.attended_token ? `"${event.attended_token}"` : "None";

    // Update Live Candidate Race Bars
    renderLiveCandidateRace(event.telemetry);
  } else if (event.stage === "stage_complete") {
    updatePipelineStep(6);
    if (event.telemetry) {
      document.getElementById("telemetryBadge").innerText = `Done (${event.telemetry.tokens_per_second} tok/s)`;
    }
  }
}

function renderLiveCandidateRace(telemetry) {
  const container = document.getElementById("liveCandidatesList");
  if (!container || !telemetry || !telemetry.candidates) return;

  container.innerHTML = "";
  const topCands = telemetry.candidates.slice(0, 5);

  topCands.forEach(c => {
    const isWinner = c.is_winner;
    const row = document.createElement("div");
    row.className = "candidate-row";
    row.innerHTML = `
      <span class="candidate-name ${isWinner ? 'winner' : ''}" title="${escapeHtml(c.token)}">
        ${isWinner ? '👑 ' : ''}<code>${escapeHtml(c.token)}</code>
      </span>
      <div class="cand-bar-wrap">
        <div class="cand-bar-fill ${isWinner ? 'winner-bar' : ''}" style="width: ${Math.max(c.percentage, 1)}%"></div>
      </div>
      <span class="cand-pct-badge">${c.percentage}%</span>
    `;
    container.appendChild(row);
  });
}

function renderStage1Telemetry(tokenData) {
  const tokensBox = document.getElementById("telemetryTokens");
  tokensBox.innerHTML = "";
  const bpe = tokenData.bpe;
  bpe.chips.forEach(chip => {
    const el = document.createElement("span");
    el.className = "token-chip";
    el.style.backgroundColor = chip.color;
    el.innerHTML = `${chip.is_space ? '␣' : escapeHtml(chip.token)} <span class="token-id-tag">#${chip.id}</span>`;
    tokensBox.appendChild(el);
  });

  document.getElementById("statChars").innerText = tokenData.char_count;
  document.getElementById("statTokens").innerText = bpe.token_count;
  document.getElementById("statRatio").innerText = bpe.compression_ratio;
}

function renderStage2Telemetry(hits, fullPrompt) {
  const ragBox = document.getElementById("telemetryRagBox");
  ragBox.innerHTML = "";
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

  const promptBox = document.getElementById("telemetryPromptCode");
  if (promptBox) promptBox.innerText = fullPrompt || "// Prompt assembly preview";
}

async function handleSyncChat(query) {
  const badge = document.getElementById("telemetryBadge");
  badge.innerText = "Processing...";
  badge.style.color = "var(--warning)";

  updatePipelineStep(1);

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: query,
        temperature: parseFloat(document.getElementById("chatTemp").value),
        top_p: parseFloat(document.getElementById("chatTopP").value),
        top_k: parseInt(document.getElementById("chatTopK").value),
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
    badge.innerText = "Completed";
    badge.style.color = "var(--success)";
    updatePipelineStep(6);
  }
}

function renderChatTelemetry(data) {
  renderStage1Telemetry(data.stages.stage_1_tokenization);
  renderStage2Telemetry(data.stages.stage_2_vector_search, data.stages.stage_3_prompt_assembly.full_prompt);

  // Sampling steps
  const steps = data.stages.stage_4_sampling_telemetry;
  if (steps && steps.length > 0) {
    renderLiveCandidateRace(steps[0]);
    document.getElementById("liveCurrentToken").innerText = steps[0].winner || "-";
    document.getElementById("liveAttendedToken").innerText = "Context words";
  }

  if (data.telemetry) {
    document.getElementById("mTokSec").innerText = data.telemetry.tokens_per_second || "35.0";
    document.getElementById("mProgress").innerText = "100%";
    document.getElementById("mElapsed").innerText = `${data.telemetry.elapsed_ms}ms`;
    document.getElementById("mTokenTotal").innerText = `${data.telemetry.output_tokens_count}`;
  }
}

function createStreamingBotBubble() {
  const box = document.getElementById("chatMessages");
  const bubble = document.createElement("div");
  bubble.className = "message-bubble bot-bubble";

  const header = document.createElement("div");
  header.className = "msg-header";
  header.innerHTML = `
    <span class="sender-name">🤖 Real-Time Glass-Box AI</span>
    <span class="msg-time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
  `;

  const contentEl = document.createElement("div");
  contentEl.className = "msg-content";

  const cursor = document.createElement("span");
  cursor.className = "blinking-cursor";
  contentEl.appendChild(cursor);

  bubble.appendChild(header);
  bubble.appendChild(contentEl);
  box.appendChild(bubble);
  box.scrollTop = box.scrollHeight;

  return { bubble, contentEl, cursor };
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
    <span class="sender-name">${role === "user" ? "👤 You" : "🤖 Real-Time Glass-Box AI"}</span>
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

/* ==================== 3. ATTENTION ENGINE LAB ==================== */
function initAttentionLab() {
  const computeBtn = document.getElementById("attComputeBtn");
  const input = document.getElementById("attSentenceInput");
  const headBtns = document.querySelectorAll(".head-btn");

  let currentHead = "coreference";

  headBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      headBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentHead = btn.dataset.head;
      computeAttention();
    });
  });

  async function computeAttention() {
    const text = input.value.trim();
    if (!text) return;

    try {
      const res = await fetch("/api/attention", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, head_type: currentHead })
      });
      const data = await res.json();
      renderAttentionResults(data);
    } catch (err) {
      console.error("Attention calc failed:", err);
    }
  }

  computeBtn.addEventListener("click", computeAttention);
}

window.setAttPreset = function(sentence, head) {
  document.getElementById("attSentenceInput").value = sentence;
  const btn = document.querySelector(`.head-btn[data-head="${head}"]`);
  if (btn) btn.click();
  else document.getElementById("attComputeBtn").click();
};

function renderAttentionResults(data) {
  const tokens = data.tokens;
  const matrix = data.attention_matrix;
  const n = tokens.length;

  document.getElementById("headExplainerText").innerText = data.head_description;

  // 1. Render Interactive Token Arc Pills
  const arcRow = document.getElementById("tokenArcsRow");
  arcRow.innerHTML = "";
  const focusCard = document.getElementById("arcFocusCard");

  tokens.forEach((tok, idx) => {
    const pill = document.createElement("button");
    pill.className = "arc-token-pill";
    pill.innerHTML = `<code>${escapeHtml(tok)}</code> <span style="font-size:0.68rem; opacity:0.7">#${idx}</span>`;
    
    pill.addEventListener("mouseenter", () => {
      document.querySelectorAll(".arc-token-pill").forEach(p => p.classList.remove("active"));
      pill.classList.add("active");

      // Show top attended connections
      const row = matrix[idx];
      const sorted = row.map((w, j) => ({ j, tok: tokens[j], w })).sort((a, b) => b.w - a.w);
      const top3 = sorted.slice(0, 3);

      focusCard.innerHTML = `
        <div style="font-weight:600; color:var(--primary); margin-bottom:4px;">
          Token #${idx} <code>${escapeHtml(tok)}</code> is attending to:
        </div>
        <div style="display:flex; flex-direction:column; gap:4px;">
          ${top3.map(c => `
            <div style="display:flex; justify-content:space-between; font-size:0.8rem;">
              <span>➔ <code>${escapeHtml(c.tok)}</code> (position #${c.j})</span>
              <strong style="color:${c.w > 0.3 ? 'var(--success)' : 'var(--text-muted)'}">${(c.w * 100).toFixed(1)}% weight</strong>
            </div>
          `).join("")}
        </div>
      `;

      // Highlight heatmap row
      highlightHeatmapRow(idx);
    });

    arcRow.appendChild(pill);
  });

  // Default focus on token 'it' if present, or first token
  const itIdx = tokens.findIndex(t => t.toLowerCase() === "it" || t.toLowerCase() === "bank");
  const targetPill = itIdx >= 0 ? arcRow.children[itIdx] : arcRow.children[0];
  if (targetPill) targetPill.dispatchEvent(new Event("mouseenter"));

  // 2. Render Heatmap Grid
  const grid = document.getElementById("attentionHeatmapGrid");
  grid.style.gridTemplateColumns = `repeat(${n + 1}, 38px)`;
  grid.innerHTML = "";

  // Corner empty
  grid.appendChild(createGridCell("", "heatmap-header-token"));

  // Top header tokens (Keys)
  tokens.forEach(t => {
    grid.appendChild(createGridCell(t, "heatmap-header-token", `Key: "${t}"`));
  });

  // Rows (Queries)
  for (let i = 0; i < n; i++) {
    // Row header (Query)
    grid.appendChild(createGridCell(tokens[i], "heatmap-header-token", `Query: "${tokens[i]}"`));

    for (let j = 0; j < n; j++) {
      const w = matrix[i][j];
      const cell = document.createElement("div");
      cell.className = `heatmap-cell cell-row-${i}`;
      cell.title = `Query "${tokens[i]}" ➔ Key "${tokens[j]}": ${(w * 100).toFixed(1)}%`;
      
      // Calculate color intensity
      // low = #111a30, mid = #3b82f6, high = #ec4899
      if (w < 0.2) {
        cell.style.backgroundColor = `rgba(59, 130, 246, ${Math.max(0.15, w * 2)})`;
      } else if (w < 0.5) {
        cell.style.backgroundColor = `rgba(168, 85, 247, ${w * 1.5})`;
      } else {
        cell.style.backgroundColor = `rgba(236, 72, 153, ${Math.min(1.0, w * 1.2)})`;
      }

      cell.innerText = Math.round(w * 100);
      grid.appendChild(cell);
    }
  }
}

function createGridCell(text, className, title = "") {
  const el = document.createElement("div");
  el.className = className;
  el.innerText = text;
  if (title) el.title = title;
  return el;
}

function highlightHeatmapRow(rowIdx) {
  document.querySelectorAll(".heatmap-cell").forEach(cell => {
    if (cell.classList.contains(`cell-row-${rowIdx}`)) {
      cell.style.outline = "2px solid #38bdf8";
      cell.style.zIndex = "5";
    } else {
      cell.style.outline = "none";
      cell.style.zIndex = "1";
    }
  });
}

/* ==================== 4. TOKENIZER LAB ==================== */
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
  runTokenize();
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
  data.char.tokens.slice(0, 45).forEach((ch) => {
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

/* ==================== 5. EMBEDDINGS LAB ==================== */
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

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    ctx.strokeStyle = "#1e293b";
    ctx.lineWidth = 1;

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

    points.forEach(pt => {
      const px = (w / 2) + (pt.x / 80) * (w / 2 - 50);
      const py = (h / 2) - (pt.y / 80) * (h / 2 - 40);

      const color = catColors[pt.category] || "#94a3b8";

      ctx.beginPath();
      ctx.arc(px, py, 6, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();

      ctx.fillStyle = "#f8fafc";
      ctx.font = "12px 'Inter', sans-serif";
      ctx.fillText(pt.word, px + 9, py + 4);
    });
  } catch (err) {
    console.error("Failed loading PCA:", err);
  }
}

/* ==================== 6. SAMPLING LAB ==================== */
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

  updateSim();
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

  const notesBox = document.getElementById("samplingNotesBox");
  notesBox.innerHTML = "";
  data.notes.forEach(note => {
    const p = document.createElement("p");
    p.innerText = note;
    notesBox.appendChild(p);
  });
}

/* ==================== 7. RAG LAB ==================== */
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
