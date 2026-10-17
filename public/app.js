const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const APPROACHES = [
  { key: "rules", title: "Rules-based search", how: "Matches the question's keywords against policy titles and text. No AI." },
  { key: "llm", title: "LLM without a vector index", how: "Pastes all policies into the prompt and lets the LLM find the answer." },
  { key: "vector", title: "LLM with a vector index", how: "Finds the closest policies by embedding similarity, then the LLM answers from those." },
];
const EXAMPLES = [
  "How many vacation days do I get per year?",
  "Can I work from home?",
  "My grandmother passed away. How much time off can I take?",
  "A reporter called and asked for a comment. Can I answer?",
  "Can I bring my dog to the office?",
  "Is there a company pension plan?",
];
const GROUND_LABEL = { supported: "Supported by the policy", unsupported: "Not supported by the policy", abstained: "Says it has no answer" };

let status = null;

async function api(path, opts) {
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

// ---------- header / status ----------
async function loadStatus() {
  status = await api("/api/status");
  const ready = status.index.models.filter((m) => m.ready).map((m) => m.model);
  $("#models").textContent = `Answer models, in the order tried: ${status.llm_models.join(", ")}. Embedding models, in the order tried: ${status.embedding_models.join(", ")} (indexes ready: ${ready.join(", ") || "none yet"}; ${status.embedding_dims} dimensions). Policies retrieved per question: ${status.top_k}.`;
  const banner = $("#banner");
  if (!status.has_api_key) {
    banner.hidden = false;
    banner.textContent = "No Gemini API key is configured, so only the rules-based approach will work. Add GEMINI_API_KEY to the .env file and restart.";
  } else if (!status.index.ready && status.index.error) {
    banner.hidden = false;
    banner.textContent = `The vector index could not be built: ${status.index.error}`;
    setTimeout(loadStatus, 15000);
  } else if (!status.index.ready) {
    banner.hidden = false;
    banner.textContent = "The vector index is still being built. The third answer will work in a few seconds.";
    setTimeout(loadStatus, 3000);
  } else if (status.index.models.some((m) => !m.ready)) {
    banner.hidden = false;
    banner.textContent = `Backup embedding index not ready yet (${status.index.models.filter((m) => !m.ready).map((m) => m.model).join(", ")}). The site works, but has no fallback for embeddings until it finishes.`;
    setTimeout(loadStatus, 10000);
  } else banner.hidden = true;
}

// ---------- ask ----------
const fmtMs = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`);
const fmtTok = (n) => Number(n).toLocaleString("en-US");

function card(a, r) {
  const head = `<h3>${a.title}</h3><p class="how">${a.how}</p>`;
  if (!r) return `<article class="card ${a.key}">${head}<div class="skeleton"></div><div class="skeleton" style="width:80%"></div><div class="skeleton" style="width:60%"></div></article>`;
  if (r.error) return `<article class="card ${a.key}">${head}<p class="err">${esc(r.error)}</p><dl class="metrics"><div><dt>Response time</dt><dd>${fmtMs(r.metrics.ms)}</dd></div></dl></article>`;
  const m = r.metrics, t = m.tokens;
  const policy = r.policy
    ? `<div><p class="label">Relevant policy</p><div class="policy"><strong>${esc(r.policy.title)}</strong>${esc(r.policy.text)}<br><span>${esc(r.policy.department)} department, ${esc(r.policy.category)}</span></div></div>`
    : `<div><p class="label">Relevant policy</p><div class="policy"><strong>None identified</strong>${r.cited_title ? `<span>The answer cited "${esc(r.cited_title)}", which is not in the database.</span>` : ""}</div></div>`;
  const g = r.grounding;
  const reasons = g.reasons?.length && g.status === "unsupported" ? `<ul class="reasons">${g.reasons.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : "";
  const tokenCell = a.key === "rules"
    ? `<dd>0<small>no model used</small></dd>`
    : `<dd>${fmtTok(t.total)}<small>${fmtTok(t.input)} in, ${fmtTok(t.output)} out${t.thinking ? `, ${fmtTok(t.thinking)} thinking` : ""}${m.embedding_tokens ? `<br>+ ${fmtTok(m.embedding_tokens)} embedding${m.embedding_tokens_estimated ? " (estimated)" : ""}` : ""}</small></dd>`;
  const timeExtra = m.breakdown ? `<small>embed ${fmtMs(m.breakdown.embed_ms)}, search ${fmtMs(m.breakdown.search_ms)}, LLM ${fmtMs(m.breakdown.llm_ms)}</small>` : "";
  const usedModel = a.key === "rules" ? "" : `<p class="tag">Answered by ${esc(m.model)}${m.embedding_model ? `, retrieved with ${esc(m.embedding_model)}` : ""}</p>`;
  const switched = m.fallbacks?.length ? `<p class="switched">Switched model: ${m.fallbacks.map(esc).join("; ")}</p>` : "";
  let more = "";
  if (a.key === "vector") more = `<details><summary>Policies the index retrieved</summary><ol>${r.details.retrieved.map((h) => `<li>${esc(h.title)} <span class="tag">similarity ${h.score}</span></li>`).join("")}</ol></details>`;
  if (a.key === "rules") more = `<details><summary>How the keywords matched</summary><p>Keywords used: ${esc(r.details.query_terms.join(", ") || "none")}</p>${r.details.candidates.length ? `<ol>${r.details.candidates.map((c) => `<li>${esc(c.title)} <span class="tag">${c.score}, matched ${esc(c.matched.join(", "))}</span></li>`).join("")}</ol>` : ""}</details>`;
  if (a.key === "llm") more = `<details><summary>What the LLM was given</summary><p>All ${r.details.policies_in_prompt} policies, every time.</p></details>`;
  return `<article class="card ${a.key}">${head}
    <div><p class="label">Answer</p><p class="answer">${esc(r.answer)}</p></div>
    ${policy}
    <div><span class="badge ${g.status}">${GROUND_LABEL[g.status]}</span>${reasons}</div>
    <dl class="metrics"><div><dt>Response time</dt><dd>${fmtMs(m.ms)}${timeExtra}</dd></div><div><dt>Tokens used</dt>${tokenCell}</div></dl>
    ${usedModel}${switched}
    ${more}</article>`;
}

function renderCards(results) {
  $("#results").hidden = false;
  $("#results").innerHTML = APPROACHES.map((a) => card(a, results?.[a.key])).join("");
}

$("#ask").addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = $("#q").value.trim();
  if (!question) return;
  const btn = $("#go");
  btn.disabled = true; btn.textContent = "Asking...";
  renderCards(null);
  try {
    const data = await api("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question }) });
    renderCards(data.results);
  } catch (err) {
    $("#results").innerHTML = `<p class="err">${esc(err.message)}</p>`;
  } finally {
    btn.disabled = false; btn.textContent = "Ask all three";
    loadStatus();
  }
});

$("#chips").innerHTML = EXAMPLES.map((q) => `<button type="button" class="chip">${esc(q)}</button>`).join("");
$("#chips").addEventListener("click", (e) => {
  if (!e.target.classList.contains("chip")) return;
  $("#q").value = e.target.textContent;
  $("#ask").requestSubmit();
});

// ---------- benchmark ----------
const pct = (x) => `${x.n}/${x.of}`;
let benchTimer = null;

function renderBench(b) {
  const btn = $("#runBench");
  const running = b.status === "running";
  btn.disabled = running || b.cooldown_seconds > 0;
  btn.textContent = running ? `Running ${b.done}/${b.total}...` : "Run benchmark";
  $("#benchStatus").textContent = running
    ? "Running. This takes a minute or two."
    : b.summary
      ? `Last run finished ${b.finished_at.replace("T", " ")}${b.models ? ` using answer models ${b.models.llm} and embedding models ${b.models.embedding}` : ""}.${b.cooldown_seconds > 0 ? ` You can run it again in ${b.cooldown_seconds} s.` : ""}`
      : b.status === "error" ? `The run failed: ${b.error}` : "No benchmark has been run yet. Click Run benchmark.";
  if (!b.summary) { $("#benchTable").innerHTML = ""; $("#benchDetail").hidden = true; return; }

  const s = b.summary;
  const row = (key, name) => {
    const x = s[key];
    return `<tr><th scope="row">${name}</th>
      <td class="num">${pct(x.correct_policy)}</td>
      <td class="num">${x.by_type.direct.correct}/${x.by_type.direct.total}</td>
      <td class="num">${x.by_type.paraphrase.correct}/${x.by_type.paraphrase.total}</td>
      <td class="num">${pct(x.declined_out_of_scope)}</td>
      <td class="num ${x.answered_out_of_scope.n ? "bad-text" : ""}">${pct(x.answered_out_of_scope)}</td>
      <td class="num ${x.unsupported_claims.n ? "bad-text" : ""}">${pct(x.unsupported_claims)}</td>
      <td class="num">${fmtMs(x.avg_ms)}</td>
      <td class="num">${fmtTok(x.avg_tokens)}</td></tr>`;
  };
  $("#benchTable").innerHTML = `<div class="table-wrap"><table>
    <thead><tr><th>Approach</th><th class="num">Right policy (20)</th><th class="num">Policy-worded (7)</th><th class="num">Everyday wording (13)</th><th class="num">Correctly declined (7)</th><th class="num">Answered anyway (7)</th><th class="num">Unsupported claims (27)</th><th class="num">Avg time</th><th class="num">Avg tokens</th></tr></thead>
    <tbody>${row("rules", "Rules-based")}${row("llm", "LLM, no index")}${row("vector", "LLM + vector index")}</tbody></table></div>
    <p class="sub" style="margin-top:12px">"Answered anyway" counts out-of-scope questions where the approach gave an answer instead of saying the database has no policy. "Unsupported claims" counts answers that cite a policy that does not exist, quote text that is not in the policy, or add numbers the policy does not contain.</p>`;

  $("#benchDetail").hidden = false;
  $("#benchRows").innerHTML = b.results.map(({ q, results }) => {
    const cell = (k, name) => {
      const r = results[k];
      if (r.error) return `<div><span class="tag">${name}</span><br><span class="err">${esc(r.error)}</span></div>`;
      const right = q.expected ? (r.answerable && r.policy?.title === q.expected) : !r.answerable;
      return `<div><span class="tag">${name}</span><br><strong class="${right ? "ok-text" : "bad-text"}">${right ? "Correct" : "Wrong"}</strong>: ${esc(r.policy?.title || (r.answerable ? "(unknown policy)" : "no policy"))}<br><span class="tag">${fmtMs(r.metrics.ms)}, ${fmtTok(r.metrics.tokens.total)} tokens, ${GROUND_LABEL[r.grounding.status].toLowerCase()}${r.metrics.model ? `, ${esc(r.metrics.model)}` : ""}</span></div>`;
    };
    return `<div class="qrow"><h4>${esc(q.question)} <span class="tag">expected: ${esc(q.expected || "no policy")}</span></h4><div class="qgrid">${cell("rules", "Rules")}${cell("llm", "LLM, no index")}${cell("vector", "LLM + vector")}</div></div>`;
  }).join("");
}

async function pollBench() {
  try {
    const b = await api("/api/benchmark");
    renderBench(b);
    clearTimeout(benchTimer);
    if (b.status === "running") benchTimer = setTimeout(pollBench, 2000);
    else if (b.cooldown_seconds > 0) benchTimer = setTimeout(pollBench, 10000);
  } catch (err) { $("#benchStatus").textContent = err.message; }
}

$("#runBench").addEventListener("click", async () => {
  try { await api("/api/benchmark", { method: "POST" }); }
  catch (err) { $("#benchStatus").textContent = err.message; return; }
  pollBench();
});

// ---------- policy browser ----------
async function loadPolicies() {
  const all = await api("/api/policies");
  $("#count").textContent = all.length;
  const body = $("#policyTable tbody");
  const draw = (list) => { body.innerHTML = list.map((p) => `<tr><td>${esc(p.title)}</td><td>${esc(p.text)}</td><td>${esc(p.department)}</td><td>${esc(p.category)}</td></tr>`).join(""); };
  draw(all);
  $("#filter").addEventListener("input", (e) => {
    const f = e.target.value.toLowerCase();
    draw(all.filter((p) => Object.values(p).join(" ").toLowerCase().includes(f)));
  });
}

loadStatus(); pollBench(); loadPolicies();
