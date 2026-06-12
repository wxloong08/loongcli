/* loongcli archive SPA — hash 路由，无框架 */
"use strict";

const stage = document.getElementById("stage");
const MEMORY_TYPES = ["user", "feedback", "project", "reference"];
const PAGE_CHUNK = 200;

/* ── 工具函数 ───────────────────────────────── */

function esc(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;").replaceAll("'", "&#39;");
}

/* 拦截危险协议的链接/图片：marked 默认不过滤 javascript:/data: URL */
function unsafeUrl(href) {
  const t = String(href ?? "").trim().toLowerCase();
  return t.startsWith("javascript:") || t.startsWith("data:") || t.startsWith("vbscript:");
}

marked.use({
  renderer: {
    link(href, title, text) { return unsafeUrl(href) ? text : false; },
    image(href, title, text) { return unsafeUrl(href) ? esc(text) : false; },
  },
});

/* markdown 渲染：先转义源文本中的内嵌 HTML，再交给 marked，
   防止会话/记忆里携带的原始 HTML（如抓取的网页内容）注入执行 */
function md(text) {
  const safe = String(text ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;");
  return `<div class="md">${marked.parse(safe)}</div>`;
}

function relTime(iso) {
  if (!iso) return "?";
  const sec = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
  if (isNaN(sec)) return "?";
  if (sec < 60) return "刚刚";
  if (sec < 3600) return `${Math.floor(sec / 60)} 分钟前`;
  if (sec < 86400) return `${Math.floor(sec / 3600)} 小时前`;
  const days = Math.floor(sec / 86400);
  if (days < 30) return `${days} 天前`;
  if (days < 365) return `${Math.floor(days / 30)} 个月前`;
  return `${Math.floor(days / 365)} 年前`;
}

function fmtSize(n) {
  if (n == null) return "?";
  if (n < 1024) return `${n}B`;
  if (n < 1048576) return `${(n / 1024).toFixed(1)}KB`;
  return `${(n / 1048576).toFixed(1)}MB`;
}

function shortSlug(slug) {
  // "D-HKUDS-loongcli" → "loongcli"，太短就显示原样
  const parts = slug.split("-").filter(Boolean);
  return parts.length > 1 ? parts[parts.length - 1] : slug;
}

function toast(msg, isError = false) {
  const el = document.createElement("div");
  el.className = `toast${isError ? " toast-error" : ""}`;
  el.textContent = msg;
  document.getElementById("toast-zone").appendChild(el);
  setTimeout(() => el.remove(), 3800);
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await res.json().catch(() => ({ error: "响应解析失败" }));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

/* ── 路由 ───────────────────────────────────── */

function route() {
  const hash = location.hash.replace(/^#\/?/, "");
  const parts = hash.split("/").filter(Boolean).map(decodeURIComponent);

  document.querySelectorAll(".nav a").forEach(a => {
    a.classList.toggle("active", a.dataset.nav === (parts[0] || "sessions"));
  });

  if (parts[0] === "memories") {
    if (parts[1] === "new") return renderMemoryForm(null);
    if (parts[2] === "edit") return renderMemoryForm(parts[1]);
    if (parts[1]) return renderMemoryDetail(parts[1]);
    return renderMemoryList();
  }
  if (parts[0] === "sessions" && parts[1] && parts[2]) return renderSessionDetail(parts[1], parts[2]);
  return renderSessionList();
}

window.addEventListener("hashchange", route);

/* ── 会话列表 ───────────────────────────────── */

async function renderSessionList() {
  stage.innerHTML = `<div class="empty">加载中…</div>`;
  let data;
  try { data = await api("/api/projects"); }
  catch (e) { stage.innerHTML = `<div class="empty">${esc(e.message)}</div>`; return; }

  const projects = data.projects;
  const saved = localStorage.getItem("loong.project");
  const slugs = projects.map(p => p.slug);
  const selected = slugs.includes(saved) ? saved
    : slugs.includes(data.current) ? data.current
    : slugs[0];

  const options = projects.map(p =>
    `<option value="${esc(p.slug)}"${p.slug === selected ? " selected" : ""}>` +
    `${esc(shortSlug(p.slug))} (${p.session_count})</option>`).join("");

  stage.innerHTML = `
    <div class="page-head">
      <div class="page-title">会话<span class="en">SESSIONS</span></div>
      <div class="page-sub">历史会话只读浏览 · ${projects.length} 个项目</div>
    </div>
    <div class="toolbar">
      <select id="project-select" title="选择项目">${options}</select>
      <input type="text" id="session-search" placeholder="⌕ 搜索标题…">
    </div>
    <div class="card-list" id="session-list"><div class="empty">加载中…</div></div>`;

  let sessions = [];
  let currentSlug = selected;

  const renderList = () => {
    const list = document.getElementById("session-list");
    const q = document.getElementById("session-search").value.trim().toLowerCase();
    const filtered = q ? sessions.filter(s => (s.title || "").toLowerCase().includes(q)) : sessions;
    if (!filtered.length) {
      list.innerHTML = `<div class="empty"><span class="glyph">空</span>${q ? "没有匹配的会话" : "该项目没有会话"}</div>`;
      return;
    }
    list.innerHTML = filtered.map(s => `
      <a class="card" href="#/sessions/${encodeURIComponent(currentSlug)}/${encodeURIComponent(s.session_id)}">
        <div class="card-title">${esc(s.title || "(无标题)")}</div>
        <div class="card-meta">
          <span>${relTime(s.updated_at)}</span>
          <span>${s.turn_count ?? "?"} 轮</span>
          <span>${fmtSize(s.file_size)}</span>
          <span>${esc(s.session_id)}</span>
        </div>
      </a>`).join("");
  };

  const loadSessions = async (slug) => {
    currentSlug = slug;
    try {
      const data = await api(`/api/projects/${encodeURIComponent(slug)}/sessions?limit=100`);
      sessions = data.sessions;
    } catch (e) {
      sessions = [];
      document.getElementById("session-list").innerHTML = `<div class="empty">${esc(e.message)}</div>`;
      return;
    }
    renderList();
  };

  const sel = document.getElementById("project-select");
  sel.addEventListener("change", () => {
    localStorage.setItem("loong.project", sel.value);
    loadSessions(sel.value);
  });
  document.getElementById("session-search").addEventListener("input", renderList);

  if (selected) loadSessions(selected);
  else document.getElementById("session-list").innerHTML =
    `<div class="empty"><span class="glyph">空</span>还没有任何会话</div>`;
}

/* ── 会话详情 ───────────────────────────────── */

function pairToolResults(messages) {
  // tool 消息按 tool_call_id 归入前面 assistant 的 tool_calls
  const results = new Map();
  for (const m of messages) {
    if (m.role === "tool" && m.tool_call_id) results.set(m.tool_call_id, m);
  }
  return results;
}

function renderToolBlock(tc, result) {
  let args = tc.function?.arguments ?? "";
  let argsPretty = args;
  try { argsPretty = JSON.stringify(JSON.parse(args), null, 2); } catch { /* 保持原样 */ }
  const preview = args.replace(/\s+/g, " ").slice(0, 80);
  const resultText = typeof result?.content === "string"
    ? result.content : result ? JSON.stringify(result.content) : "(无结果)";
  return `
    <details class="tool-block">
      <summary>
        <span class="tool-name">${esc(tc.function?.name || "?")}</span>
        <span class="tool-args-preview">${esc(preview)}</span>
      </summary>
      <div class="tool-body">
        <h5>ARGUMENTS</h5>
        <pre>${esc(argsPretty)}</pre>
        <h5>RESULT</h5>
        <pre>${esc(resultText)}</pre>
      </div>
    </details>`;
}

function renderMessage(m, toolResults) {
  if (m.role === "user") {
    const text = typeof m.content === "string" ? m.content : JSON.stringify(m.content);
    return `<div class="msg msg-user">${esc(text)}</div>`;
  }
  if (m.role === "system") {
    return `<div class="msg msg-system" data-sys hidden>` +
      `<span class="role-label">SYSTEM</span>${esc(m.content || "")}</div>`;
  }
  if (m.role === "assistant") {
    let html = "";
    if (m.content) html += md(m.content);
    for (const tc of m.tool_calls || []) {
      html += renderToolBlock(tc, toolResults.get(tc.id));
    }
    if (!html) return "";
    return `<div class="msg msg-assistant">${html}</div>`;
  }
  if (m.role === "tool") {
    if (m.tool_call_id) return "";  // 已归入 assistant 折叠块
    return `<div class="msg msg-system">` +
      `<span class="role-label">TOOL</span>${esc(String(m.content ?? ""))}</div>`;
  }
  return "";
}

async function renderSessionDetail(slug, id) {
  stage.innerHTML = `<div class="empty">加载中…</div>`;
  let data;
  try { data = await api(`/api/projects/${encodeURIComponent(slug)}/sessions/${encodeURIComponent(id)}`); }
  catch (e) { stage.innerHTML = `<div class="empty">${esc(e.message)}</div>`; return; }

  const meta = data.meta || {};
  const hasCompact = Array.isArray(data.compact_messages) && data.compact_messages.length > 0;

  stage.innerHTML = `
    <div class="crumb"><a href="#/sessions">会话</a> / ${esc(shortSlug(slug))} / ${esc(id)}</div>
    <div class="page-head">
      <div class="page-title">${esc(meta.title || "(无标题)")}</div>
    </div>
    <div class="dialog-meta">
      <span>创建 ${relTime(meta.created_at)}</span>
      <span>更新 ${relTime(meta.updated_at)}</span>
      <span>${meta.turn_count ?? "?"} 轮</span>
      <label class="toggle-sys"><input type="checkbox" id="toggle-sys"> 显示 system 消息</label>
      ${hasCompact ? `<label class="toggle-sys"><input type="checkbox" id="toggle-compact"> 查看压缩后视图</label>` : ""}
    </div>
    <div id="dialog-zone"></div>`;

  const renderView = (useCompact) => {
    const messages = useCompact ? data.compact_messages : (data.messages || []);
    const toolResults = pairToolResults(messages);
    const zone = document.getElementById("dialog-zone");
    let start = Math.max(0, messages.length - PAGE_CHUNK);

    const renderRange = () => {
      const visible = messages.slice(start);
      const moreBtn = start > 0
        ? `<button class="btn show-more-btn" id="show-more">↑ 加载更早的 ${Math.min(PAGE_CHUNK, start)} 条（共 ${messages.length} 条）</button>` : "";
      zone.innerHTML = `${moreBtn}<div class="dialog">` +
        visible.map(m => renderMessage(m, toolResults)).join("") + `</div>`;
      syncSysVisibility();
      document.getElementById("show-more")?.addEventListener("click", () => {
        start = Math.max(0, start - PAGE_CHUNK);
        renderRange();
      });
    };
    renderRange();
  };

  const syncSysVisibility = () => {
    const show = document.getElementById("toggle-sys").checked;
    document.querySelectorAll("[data-sys]").forEach(el => { el.hidden = !show; });
  };

  document.getElementById("toggle-sys").addEventListener("change", syncSysVisibility);
  document.getElementById("toggle-compact")?.addEventListener("change", e => renderView(e.target.checked));
  renderView(false);
}

/* ── 记忆列表 ───────────────────────────────── */

let memoryFilter = "";

async function renderMemoryList() {
  stage.innerHTML = `<div class="empty">加载中…</div>`;
  let data;
  try { data = await api(`/api/memories${memoryFilter ? `?type=${memoryFilter}` : ""}`); }
  catch (e) { stage.innerHTML = `<div class="empty">${esc(e.message)}</div>`; return; }

  const tabs = ["", ...MEMORY_TYPES].map(t =>
    `<button data-type="${t}" class="${t === memoryFilter ? "active" : ""}">${t || "全部"}</button>`).join("");

  stage.innerHTML = `
    <div class="page-head">
      <div class="page-title">记忆<span class="en">MEMORY</span></div>
      <div class="page-sub">~/.loongcli/memory · ${data.memories.length} 条</div>
    </div>
    <div class="toolbar">
      <div class="filter-tabs">${tabs}</div>
      <input type="text" id="memory-search" placeholder="⌕ 搜索名称/描述…">
      <button class="btn btn-gold" id="new-memory">+ 新建记忆</button>
    </div>
    <div class="card-list" id="memory-list"></div>`;

  const renderCards = () => {
    const q = document.getElementById("memory-search").value.trim().toLowerCase();
    const filtered = q
      ? data.memories.filter(m =>
          m.name.toLowerCase().includes(q) || (m.description || "").toLowerCase().includes(q))
      : data.memories;
    document.getElementById("memory-list").innerHTML = filtered.map(m => `
      <a class="card" href="#/memories/${encodeURIComponent(m.name)}">
        <div class="card-title"><span class="tag tag-${esc(m.type)}">${esc(m.type)}</span> &nbsp;${esc(m.name)}</div>
        <div class="card-meta">
          <span>${esc(m.description)}</span>
          <span>${relTime(m.updated_at)}</span>
        </div>
      </a>`).join("") || `<div class="empty"><span class="glyph">忘</span>${q ? "没有匹配的记忆" : "没有记忆"}</div>`;
  };
  renderCards();

  document.querySelectorAll(".filter-tabs button").forEach(b =>
    b.addEventListener("click", () => { memoryFilter = b.dataset.type; renderMemoryList(); }));
  document.getElementById("memory-search").addEventListener("input", renderCards);
  document.getElementById("new-memory").addEventListener("click", () => { location.hash = "#/memories/new"; });
}

/* ── 记忆详情 ───────────────────────────────── */

async function renderMemoryDetail(name) {
  stage.innerHTML = `<div class="empty">加载中…</div>`;
  let m;
  try { m = await api(`/api/memories/${encodeURIComponent(name)}`); }
  catch (e) { stage.innerHTML = `<div class="empty">${esc(e.message)}</div>`; return; }

  stage.innerHTML = `
    <div class="crumb"><a href="#/memories">记忆</a> / ${esc(m.name)}</div>
    <div class="memo-head">
      <div class="page-head" style="margin-bottom:0">
        <div class="page-title">${esc(m.name)}</div>
        <div class="page-sub"><span class="tag tag-${esc(m.type)}">${esc(m.type)}</span> &nbsp;${esc(m.description)}</div>
      </div>
      <div class="form-actions">
        <button class="btn" id="edit-btn">编辑</button>
        <button class="btn btn-danger" id="delete-btn">删除</button>
      </div>
    </div>
    <dl class="memo-frontmatter">
      <dt>CREATED</dt><dd>${esc(m.created_at || "?")}</dd>
      <dt>UPDATED</dt><dd>${esc(m.updated_at || "?")}</dd>
    </dl>
    ${md(m.content)}`;

  document.getElementById("edit-btn").addEventListener("click", () => {
    location.hash = `#/memories/${encodeURIComponent(m.name)}/edit`;
  });
  document.getElementById("delete-btn").addEventListener("click", async () => {
    if (!confirm(`确认删除记忆「${m.name}」？\n\n${m.description}\n\n此操作不可恢复。`)) return;
    try {
      await api(`/api/memories/${encodeURIComponent(m.name)}`, { method: "DELETE" });
      toast(`已删除 ${m.name}`);
      location.hash = "#/memories";
    } catch (e) { toast(e.message, true); }
  });
}

/* ── 记忆表单（新建/编辑） ───────────────────── */

async function renderMemoryForm(name) {
  let m = { name: "", description: "", type: "project", content: "" };
  if (name) {
    try { m = await api(`/api/memories/${encodeURIComponent(name)}`); }
    catch (e) { stage.innerHTML = `<div class="empty">${esc(e.message)}</div>`; return; }
  }

  const typeOptions = MEMORY_TYPES.map(t =>
    `<option value="${t}"${t === m.type ? " selected" : ""}>${t}</option>`).join("");

  stage.innerHTML = `
    <div class="crumb"><a href="#/memories">记忆</a> / ${name ? esc(name) : "新建"}</div>
    <div class="page-head">
      <div class="page-title">${name ? "编辑记忆" : "新建记忆"}<span class="en">${name ? "EDIT" : "NEW"}</span></div>
    </div>
    <form class="form" id="memory-form">
      <label>NAME（字母/数字/横线/下划线）
        <input type="text" id="f-name" value="${esc(m.name)}" ${name ? "readonly" : ""} required
               pattern="[A-Za-z0-9_-]+" placeholder="project-some-topic">
      </label>
      <label>TYPE
        <select id="f-type">${typeOptions}</select>
      </label>
      <label>DESCRIPTION（一句话摘要，用于索引与召回）
        <input type="text" id="f-desc" value="${esc(m.description)}" required maxlength="500">
      </label>
      <label>CONTENT（Markdown）
        <textarea id="f-content" required>${esc(m.content)}</textarea>
      </label>
      <div class="form-actions">
        <button type="submit" class="btn btn-gold">保存</button>
        <button type="button" class="btn" id="cancel-btn">取消</button>
      </div>
    </form>`;

  document.getElementById("cancel-btn").addEventListener("click", () => {
    location.hash = name ? `#/memories/${encodeURIComponent(name)}` : "#/memories";
  });

  document.getElementById("memory-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const body = {
      name: document.getElementById("f-name").value.trim(),
      type: document.getElementById("f-type").value,
      description: document.getElementById("f-desc").value.trim(),
      content: document.getElementById("f-content").value,
    };
    try {
      const result = name
        ? await api(`/api/memories/${encodeURIComponent(name)}`, { method: "PUT", body: JSON.stringify(body) })
        : await api("/api/memories", { method: "POST", body: JSON.stringify(body) });
      if (result.merged) toast(`描述与已有记忆相似，已合并到「${result.name}」`);
      else toast(`已保存 ${result.name}`);
      location.hash = `#/memories/${encodeURIComponent(result.name)}`;
    } catch (err) { toast(err.message, true); }
  });
}

/* ── 启动 ───────────────────────────────────── */

if (!location.hash) location.hash = "#/sessions";
route();
