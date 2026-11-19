/*
  UI prototype implementing the 5 interface conditions described in rp.docx.
  - C0: No-AI baseline
  - C1: Prediction-only
  - C2: Static SHAP heatmap
  - C3: Sequential reveal
  - C4: Sequential reveal + boundary cue
*/

const CONDITIONS = {
  C0: { id: "C0", name: "No-AI baseline", hasAI: false, hasXAI: false, sequential: false, boundaryCue: false },
  C1: { id: "C1", name: "Prediction-only", hasAI: true, hasXAI: false, sequential: false, boundaryCue: false },
  C2: { id: "C2", name: "Static SHAP heatmap", hasAI: true, hasXAI: true, sequential: false, boundaryCue: false },
  C3: { id: "C3", name: "Sequential reveal", hasAI: true, hasXAI: true, sequential: true, boundaryCue: false },
  C4: { id: "C4", name: "Sequential + boundary cue", hasAI: true, hasXAI: true, sequential: true, boundaryCue: true },
};

const CONDITION_ORDER = ["C0", "C1", "C2", "C3", "C4"];

const DEFAULT_TRIALS_PER_CONDITION = 5; // 25 trials total (main study default)
const DEFAULT_STIMULI_URLS = ["./data/final/stimuli_pool_v11_twoyear_neutral.json"];
const FORMSPREE_ENDPOINT = ""; // Public demonstration: remote collection is disabled.
const SUMMARY_PAYLOAD_VERSION = "v2";
const AUTO_LIST_IDS = [1, 2, 3, 4, 5];

// Sequential conditions (C3/C4): reveal top-|SHAP| highlights until a target is reached.
// This keeps trials workable even when the text has many tokens.
const SEQUENTIAL_REVEAL_TARGET = 24;
const SEQUENTIAL_REVEAL_STEP = 3;

function nowISO() {
  return new Date().toISOString();
}

function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}

function fmtPct(p) {
  if (!Number.isFinite(p)) return "--";
  const pct = p * 100;
  if (pct < 0.01) return "<0.01%";
  if (pct > 99.99) return ">99.99%";
  return pct.toFixed(2) + "%";
}

function pickPred(ai) {
  const probs = ai?.probs || {};
  const label = typeof ai?.label === "string" ? ai.label : "";
  const entries = Object.entries(probs).filter(([, v]) => Number.isFinite(v));
  if (!entries.length) return { label: ai?.label || "--", prob: NaN };
  entries.sort((a, b) => (b[1] ?? 0) - (a[1] ?? 0));
  return { label: entries[0][0], prob: entries[0][1] };
}

function cleanSegmentText(s) {
  return decodeBasicEntities(String(s || "")).replace(/</g, "").replace(/>/g, "");
}

function decodeBasicEntities(s) {
  return String(s || "")
    .replace(/&#038;|&amp;/gi, "&")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;|&apos;/gi, "'")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">");
}

function stripNoisyFragments(s) {
  let t = String(s || "");
  t = t.replace(/<\s*cdata\s*/gi, " ");
  t = t.replace(/function\(d\s*,?\s*s[\s\S]*?\}\s*\([^)]*\)\s*;?/gi, " ");
  t = t.replace(/\/\/connect\.facebook\.net\S*/gi, " ");
  t = t.replace(/\b(getelementsbytagname|getelementbyid|jssdk|xfbml=1|document\s+script)\b/gi, " ");
  return t;
}

function sentenceCase(s) {
  const t = String(s || "");
  if (!t) return "";
  let out = "";
  let capNext = true;
  for (let i = 0; i < t.length; i++) {
    const ch = t[i];
    if (capNext && /[a-z]/.test(ch)) {
      out += ch.toUpperCase();
      capNext = false;
      continue;
    }
    out += ch;
    if (/[.!?]\s*$/.test(out) || ch === "\n") capNext = true;
    if (/[A-Za-z0-9]/.test(ch)) capNext = false;
  }
  return out;
}

function cleanDisplayText(s, isTitle = false) {
  let t = decodeBasicEntities(String(s || ""));
  t = stripNoisyFragments(t);
  t = t.replace(/\.\./g, ".");
  t = t.replace(/\s+/g, " ").trim();
  if (!t) return "";
  // Collapse immediate duplicated first sentence like: "X. X. Y"
  t = t.replace(/^(.+?[.!?])\s+\1\s+/i, "$1 ");
  t = sentenceCase(t);
  if (isTitle && t.length > 1) {
    t = t[0].toUpperCase() + t.slice(1);
  } else if (!isTitle && !/[.!?]["']?\s*$/.test(t)) {
    t = t.replace(/[,:;\-\s]+$/, "").trim();
    if (t) t += ".";
  }
  return t;
}

function hasNoisyMarkers(text) {
  const t = String(text || "");
  return /(\bcdata\b|getelementsbytagname|getelementbyid|connect\.facebook\.net|\bjssdk\b|document\s+script|function\(d\s+s|xfbml=1)/i.test(t);
}

function isUsableStimulusItem(item) {
  if (!item) return false;
  const title = String(item.title || "").trim();
  const content = String(item.content || "").trim();
  if (!title || !content) return false;
  if (hasNoisyMarkers(title) || hasNoisyMarkers(content)) return false;
  return true;
}

function isDelimiterToken(text) {
  const t = String(text || "").trim().toLowerCase();
  if (!t) return true;
  if (t === "<" || t === ">") return true;
  return false;
}

function cleanSegments(segments) {
  const out = [];
  for (const seg of segments || []) {
    const textRaw = seg?.text ?? "";
    const text = cleanSegmentText(textRaw);
    if (isDelimiterToken(textRaw)) continue;
    if (!String(text).trim()) continue;
    out.push({ text, shap: Number(seg?.shap ?? 0) });
  }
  return out;
}

function tokenizeZeroShap(text) {
  const src = String(text || "");
  if (!src.trim()) return [];
  const parts = src.match(/\S+\s*/g) || [src];
  return parts.map((part) => ({ text: part, shap: 0 }));
}

function repairSegmentsForDisplay(rawText, rawSegments) {
  const source = cleanSegmentText(rawText);
  const normalized = cleanSegments(rawSegments);
  if (!String(source).trim()) return normalized;
  if (!normalized.length) return tokenizeZeroShap(source);

  const joined = normalized.map((s) => s.text).join("");
  if (!joined.trim()) return tokenizeZeroShap(source);

  const joinedTrim = joined.replace(/\s+$/g, "");
  const sourceTrim = source.replace(/\s+$/g, "");
  if (joined === source || joinedTrim === sourceTrim) {
    return normalized;
  }

  function appendTail(tail) {
    if (!tail) return;
    const last = normalized[normalized.length - 1];
    const wordJoin = last && /[A-Za-z]$/.test(last.text || "") && /^[A-Za-z]/.test(tail);
    if (wordJoin) {
      const m = tail.match(/^[A-Za-z]+/);
      const prefix = m ? m[0] : "";
      if (prefix) {
        last.text += prefix;
        tail = tail.slice(prefix.length);
      }
    }
    if (tail) normalized.push({ text: tail, shap: 0 });
  }

  if (source.startsWith(joined)) {
    const tail = source.slice(joined.length);
    appendTail(tail);
    return normalized;
  }
  if (source.startsWith(joinedTrim)) {
    const tail = source.slice(joinedTrim.length);
    appendTail(tail);
    return normalized;
  }

  return normalized;
}

function maxAbsShap(segments) {
  let m = 0;
  for (const s of segments) {
    const v = Math.abs(Number(s?.shap ?? 0));
    if (v > m) m = v;
  }
  return m || 1;
}

function tokenBg(shapVal, maxAbs, revealed) {
  if (!revealed) return "transparent";
  const v = Number(shapVal);
  if (!Number.isFinite(v) || v === 0) return "transparent";
  const a = clamp01(Math.abs(v) / maxAbs);
  const alpha = 0.06 + a * 0.34;
  if (v >= 0) {
    return `rgba(224, 54, 86, ${alpha.toFixed(3)})`;
  }
  return `rgba(30, 136, 229, ${alpha.toFixed(3)})`;
}

function logit(p) {
  const x = Number(p);
  if (!Number.isFinite(x)) return NaN;
  const eps = 1e-6;
  const q = Math.max(eps, Math.min(1 - eps, x));
  return Math.log(q / (1 - q));
}

function fmtSigned(v) {
  if (!Number.isFinite(v)) return "--";
  return (v >= 0 ? "+" : "") + v.toFixed(3);
}

function directionFromLogit(v) {
  if (!Number.isFinite(v)) return "--";
  if (v > 0) return "toward FAKE";
  if (v < 0) return "toward TRUE";
  return "neutral";
}

function buildRevealOrder(segments) {
  const idx = [];
  for (let i = 0; i < segments.length; i++) {
    const v = Math.abs(Number(segments[i]?.shap ?? 0));
    if (v > 1e-9) idx.push(i);
  }
  idx.sort((a, b) => Math.abs(segments[b].shap) - Math.abs(segments[a].shap));
  return idx;
}

// Embedded fallback items: segments extracted from explain_0000.html / explain_0001.html.
// Note: explain_0001 contains <title>/<content>/<end> artifacts; we clean them.
const FALLBACK_ITEMS = [
  {
    id: "item-0000",
    source: "demo",
    title: "Scientists confirm the moon is made of cheese",
    content: "A leaked report claims astronauts tasted the surface and found cheddar.",
    ai: {
      label: "TRUE",
      probs: {
        FAKE: 0.00014231020759325475,
        TRUE: 0.9998576641082764,
      },
    },
    xai: {
      target: "logit(P(FAKE))",
      title_segments: cleanSegments([
        { text: "", shap: -0.081 },
        { text: "Scientists ", shap: -2.47 },
        { text: "confirm ", shap: -0.59 },
        { text: "the ", shap: 0.404 },
        { text: "moon ", shap: -0.688 },
        { text: "is ", shap: -0.455 },
        { text: "made ", shap: -1.014 },
        { text: "of ", shap: -1.34 },
        { text: "cheese", shap: -0.412 },
      ]),
      content_segments: cleanSegments([
        { text: ": ", shap: -1.344 },
        { text: "A ", shap: -0.323 },
        { text: "leaked ", shap: -1.981 },
        { text: "report ", shap: -3.278 },
        { text: "claims ", shap: 0.177 },
        { text: "astronauts ", shap: -2.086 },
        { text: "tasted ", shap: 0.098 },
        { text: "the ", shap: -0.71 },
        { text: "surface ", shap: -1.046 },
        { text: "and ", shap: -0.958 },
        { text: "found ", shap: 0.608 },
        { text: "c", shap: 0.187 },
        { text: "heddar", shap: 0.401 },
        { text: ".", shap: 1.211 },
        { text: "", shap: 0.0 },
      ]),
    },
  },
  {
    id: "item-0001",
    source: "demo",
    title: "City council approves 2026 budget after late-night vote",
    content: "The budget includes funding for road repairs and a new public library branch.",
    ai: {
      label: "TRUE",
      probs: {
        FAKE: 0.00003319943425594829,
        TRUE: 0.9999667406082153,
      },
    },
    xai: {
      target: "logit(P(FAKE))",
      title_segments: cleanSegments([
        { text: "", shap: -0.492 },
        { text: "<", shap: -0.466 },
        { text: "title", shap: -0.459 },
        { text: ">", shap: -0.504 },
        { text: "City ", shap: -0.042 },
        { text: "council ", shap: -1.347 },
        { text: "approves ", shap: -0.633 },
        { text: "20", shap: -0.177 },
        { text: "26 ", shap: -1.102 },
        { text: "budget ", shap: -2.272 },
        { text: "after ", shap: -2.058 },
        { text: "late", shap: -1.219 },
        { text: "-", shap: -1.183 },
        { text: "night ", shap: -1.949 },
        { text: "vote", shap: -1.867 },
      ]),
      content_segments: cleanSegments([
        { text: "<", shap: -1.077 },
        { text: "content", shap: -0.051 },
        { text: ">", shap: 0.046 },
        { text: "The ", shap: 0.074 },
        { text: "budget ", shap: -0.21 },
        { text: "includes ", shap: 0.007 },
        { text: "funding ", shap: -0.156 },
        { text: "for ", shap: -0.062 },
        { text: "road ", shap: -0.072 },
        { text: "repairs ", shap: -0.025 },
        { text: "and ", shap: 0.041 },
        { text: "a ", shap: -0.079 },
        { text: "new ", shap: -0.038 },
        { text: "public ", shap: -0.053 },
        { text: "library ", shap: 0.145 },
        { text: "branch", shap: 0.027 },
        { text: ".<", shap: 0.092 },
        { text: "end", shap: 0.107 },
        { text: ">", shap: 0.03 },
        { text: "", shap: 0.034 },
      ]),
    },
  },
];

const els = {
  introPage: document.getElementById("introPage"),
  experimentRoot: document.getElementById("experimentRoot"),
  introAssignedList: document.getElementById("introAssignedList"),
  introStartBtn: document.getElementById("introStartBtn"),
  listSelect: document.getElementById("listSelect"),
  conditionPill: document.getElementById("conditionPill"),
  trialPill: document.getElementById("trialPill"),
  seedPill: document.getElementById("seedPill"),
  restartBtn: document.getElementById("restartBtn"),
  sourceBadge: document.getElementById("sourceBadge"),
  itemId: document.getElementById("itemId"),
  headline: document.getElementById("headline"),
  excerpt: document.getElementById("excerpt"),
  xaiCard: document.getElementById("xaiCard"),
  xaiTarget: document.getElementById("xaiTarget"),
  xaiText: document.getElementById("xaiText"),
  xaiStats: document.getElementById("xaiStats"),
  xaiControls: document.getElementById("xaiControls"),
  revealNextBtn: document.getElementById("revealNextBtn"),
  revealAllBtn: document.getElementById("revealAllBtn"),
  revealCounter: document.getElementById("revealCounter"),
  boundaryCue: document.getElementById("boundaryCue"),
  stageKicker: document.getElementById("stageKicker"),
  stageTitle: document.getElementById("stageTitle"),
  timer: document.getElementById("timer"),
  chooseTrue: document.getElementById("chooseTrue"),
  chooseFake: document.getElementById("chooseFake"),
  confSlider: document.getElementById("confSlider"),
  confValue: document.getElementById("confValue"),
  aiPanel: document.getElementById("aiPanel"),
  aiLabel: document.getElementById("aiLabel"),
  aiProb: document.getElementById("aiProb"),
  aiMeter: document.getElementById("aiMeter"),
  aiProbFoot: document.getElementById("aiProbFoot"),
  noteBox: document.getElementById("noteBox"),
  uploadStatus: document.getElementById("uploadStatus"),
  summaryBox: document.getElementById("summaryBox"),
  initialSummary: document.getElementById("initialSummary"),
  initialConfSummary: document.getElementById("initialConfSummary"),
  controlsBox: document.getElementById("controlsBox"),
  whyBox: document.getElementById("whyBox"),
  whyText: document.getElementById("whyText"),
  primaryBtn: document.getElementById("primaryBtn"),
  retryUploadBtn: document.getElementById("retryUploadBtn"),
  downloadBtn: document.getElementById("downloadBtn"),
  debriefPage: document.getElementById("debriefPage"),
  debriefFinishBtn: document.getElementById("debriefFinishBtn"),
  debriefSaveBtn: document.getElementById("debriefSaveBtn"),
  debriefCloseHint: document.getElementById("debriefCloseHint"),
  debriefClarificationDisclosure: document.getElementById("debriefClarificationDisclosure"),
  debriefClarificationList: document.getElementById("debriefClarificationList"),
};

let state = null;
let timerHandle = null;

const AUTO_LIST_SESSION_KEY = "xai_auto_list_assignment_v1";
const AUTO_LIST_COUNT_KEY = "xai_auto_list_counts_v1";
const PARTICIPANT_ID_STORAGE_KEY = "xai_participant_id_v1";

function randomShortId() {
  return Math.random().toString(36).slice(2, 10);
}

function buildSessionId() {
  return `s_${Date.now().toString(36)}_${randomShortId()}`;
}

function resolveParticipantId(q) {
  if (q?.pid) return q.pid;
  try {
    const existing = String(localStorage.getItem(PARTICIPANT_ID_STORAGE_KEY) || "").trim();
    if (existing) return existing;
    const generated = `anon_${randomShortId()}`;
    localStorage.setItem(PARTICIPANT_ID_STORAGE_KEY, generated);
    return generated;
  } catch (_err) {
    return `anon_${randomShortId()}`;
  }
}

function normalizeListId(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return null;
  const list = Math.floor(n);
  return AUTO_LIST_IDS.includes(list) ? list : null;
}

function hashStringToUint(str) {
  const s = String(str || "");
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function chooseAutoListRoundRobin() {
  let counts = { 2: 0, 3: 0, 4: 0 };
  try {
    const raw = localStorage.getItem(AUTO_LIST_COUNT_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      for (const listId of AUTO_LIST_IDS) {
        const n = Number(parsed?.[listId]);
        counts[listId] = Number.isFinite(n) && n >= 0 ? Math.floor(n) : 0;
      }
    }
  } catch (_err) {}

  let min = Infinity;
  for (const listId of AUTO_LIST_IDS) min = Math.min(min, counts[listId]);
  const candidates = [];
  for (const listId of AUTO_LIST_IDS) if (counts[listId] === min) candidates.push(listId);
  const chosen = candidates[Math.floor(Math.random() * candidates.length)] || AUTO_LIST_IDS[0];
  counts[chosen] += 1;

  try {
    localStorage.setItem(AUTO_LIST_COUNT_KEY, JSON.stringify(counts));
  } catch (_err) {}
  return chosen;
}

function resolveAssignedList(q) {
  const direct = normalizeListId(q?.list);
  if (direct) return direct;

  if (q?.pid) {
    return AUTO_LIST_IDS[hashStringToUint(q.pid) % AUTO_LIST_IDS.length];
  }

  try {
    const fromSession = normalizeListId(sessionStorage.getItem(AUTO_LIST_SESSION_KEY));
    if (fromSession) return fromSession;
  } catch (_err) {}

  const chosen = chooseAutoListRoundRobin();
  try {
    sessionStorage.setItem(AUTO_LIST_SESSION_KEY, String(chosen));
  } catch (_err) {}
  return chosen;
}

function ensureQueryHasAssignedList(assignedList) {
  const current = parseQuery();
  const safeList = normalizeListId(assignedList);
  if (!safeList) return;
  if (normalizeListId(current.list) === safeList && current.mode === "AUTO") return;

  const params = new URLSearchParams(window.location.search || "");
  params.set("list", String(safeList));
  if (!params.get("mode")) params.set("mode", "AUTO");
  const next = `${window.location.pathname}?${params.toString()}`;
  window.history.replaceState({}, "", next);
}

async function requestServerAssignedList(q) {
  try {
    const params = new URLSearchParams();
    if (q?.pid) params.set("pid", q.pid);
    const url = params.toString() ? `./api/assign-list?${params.toString()}` : "./api/assign-list";
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) return null;
    const data = await res.json();
    return normalizeListId(data?.list);
  } catch (_err) {
    return null;
  }
}

function showIntroScreen(show) {
  if (els.introPage) els.introPage.hidden = !show;
  if (els.experimentRoot) els.experimentRoot.hidden = show;
}

function renderIntroScreen() {
  const assignedList = normalizeListId(state?.list_id);

  if (els.introAssignedList) {
    els.introAssignedList.textContent = assignedList ? "You are ready to begin." : "Preparing study...";
  }

  if (els.introStartBtn) {
    els.introStartBtn.disabled = !assignedList;
    els.introStartBtn.textContent = assignedList ? "Start study" : "Preparing study...";
  }

}

function beginSessionFromIntro() {
  if (!state) return;
  if (!state.started_at) state.started_at = nowISO();
  state.trialStartAt = Date.now();
  state.stageStartAt = Date.now();
  logEvent("session_start", {});
  renderTop();
  renderItem();
  renderAI();
  renderXAI();
  renderStage();
  showIntroScreen(false);
  startTimer();
  tickTimer();
}

function parseQuery() {
  const q = new URLSearchParams(window.location.search || "");
  const pid = (q.get("pid") || "").trim();
  const seedRaw = (q.get("seed") || "").trim();
  const nRaw = (q.get("n") || "").trim();
  const listRaw = (q.get("list") || "").trim();
  const seed = seedRaw ? Number(seedRaw) : null;
  const n = nRaw ? Number(nRaw) : null;
  const list = listRaw ? Number(listRaw) : null;
  const mode = (q.get("mode") || "").trim().toUpperCase();
  const autostartRaw = (q.get("autostart") || "").trim();
  const sidRaw = (q.get("sid") || "").trim();
  const previewRaw = (q.get("preview") || "").trim().toLowerCase();
  return {
    pid: pid || null,
    sid: sidRaw || null,
    seed: Number.isFinite(seed) ? seed : null,
    n: Number.isFinite(n) ? n : null,
    list: Number.isFinite(list) ? Math.floor(list) : null,
    mode: mode || null,
    autostart: autostartRaw === "1",
    preview: previewRaw || null,
  };
}

function isDebriefPreviewRequested(q) {
  return q?.preview === "debrief" || q?.preview === "done";
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffleInPlace(arr, rand) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
  return arr;
}

function chooseSample(items, k, rand) {
  const idx = items.map((_, i) => i);
  shuffleInPlace(idx, rand);
  return idx.slice(0, Math.min(k, idx.length)).map((i) => items[i]);
}

function normalizeStimuliPayload(payload) {
  if (Array.isArray(payload)) {
    return { items: payload, meta: {} };
  }
  if (payload && Array.isArray(payload.items)) {
    return { items: payload.items, meta: payload.meta || {} };
  }
  return { items: null, meta: {} };
}

async function fetchFirstJson(urls) {
  for (const url of urls) {
    try {
      const r = await fetch(url, { cache: "no-store" });
      if (!r.ok) continue;
      const j = await r.json();
      const normalized = normalizeStimuliPayload(j);
      if (normalized.items && normalized.items.length) {
        return { url, items: normalized.items, meta: normalized.meta };
      }
    } catch (_) {
      // ignore and try next
    }
  }
  return { url: null, items: FALLBACK_ITEMS, meta: {} };
}

function buildTrialsFromListDesign(itemsAll, listId, design, rand, nWanted) {
  const listKey = String(listId || "");
  const mapping = design?.condition_lists?.[listKey] || null;
  if (!mapping) return null;

  const byGroup = new Map();
  for (const it of itemsAll) {
    const g = String(it?.group || "").trim();
    if (!g) continue;
    if (!byGroup.has(g)) byGroup.set(g, []);
    byGroup.get(g).push(it);
  }

  const perConditionPools = new Map();
  for (const cond of CONDITION_ORDER) {
    const g = String(mapping[cond] || "").trim();
    const pool = (byGroup.get(g) || []).slice();
    shuffleInPlace(pool, rand);
    perConditionPools.set(cond, pool);
  }

  let perCondLimit = null;
  if (Number.isFinite(nWanted) && nWanted > 0) {
    const per = Math.floor(nWanted / CONDITION_ORDER.length);
    if (per > 0) perCondLimit = per;
  }

  const trials = [];
  for (const cond of CONDITION_ORDER) {
    const pool = perConditionPools.get(cond) || [];
    const take = perCondLimit == null ? pool.length : Math.min(perCondLimit, pool.length);
    for (let i = 0; i < take; i++) {
      trials.push({ item: pool[i], condition: cond, list_id: Number(listId), list_group: mapping[cond] || null });
    }
  }

  if (!trials.length) return null;
  shuffleInPlace(trials, rand);
  return trials;
}

function buildConditionSchedule(totalTrials, rand, modeValue) {
  if (modeValue && modeValue !== "AUTO" && CONDITIONS[modeValue]) {
    return Array.from({ length: totalTrials }, () => modeValue);
  }

  // AUTO: balanced across C0-C4. Ensure multiple of 5.
  const per = Math.floor(totalTrials / CONDITION_ORDER.length);
  const n = per * CONDITION_ORDER.length;
  const out = [];
  for (const c of CONDITION_ORDER) {
    for (let i = 0; i < per; i++) out.push(c);
  }
  shuffleInPlace(out, rand);
  return out;
}

function resetState(itemsAll, payloadUrl, payloadMeta) {
  const q = parseQuery();
  const modeValue = (q.mode && (q.mode === "AUTO" || CONDITIONS[q.mode])) ? q.mode : "AUTO";
  const selectedList = Number.isFinite(q.list) && q.list > 0 ? Math.floor(q.list) : Number(els.listSelect.value || 1);
  const totalTrialsWanted = Number.isFinite(q.n) && q.n > 0 ? Math.floor(q.n) : DEFAULT_TRIALS_PER_CONDITION * CONDITION_ORDER.length;

  const seed = Number.isFinite(q.seed) ? q.seed : Math.floor(Math.random() * 1e9);
  const rand = mulberry32(seed);
  const design = payloadMeta?.design || null;
  const itemsClean = (itemsAll || []).filter(isUsableStimulusItem);
  const itemsForSampling = itemsClean.length ? itemsClean : itemsAll;

  let trials = null;
  if (modeValue === "AUTO" && Number.isFinite(selectedList) && selectedList > 0 && design) {
    trials = buildTrialsFromListDesign(itemsForSampling, selectedList, design, rand, totalTrialsWanted);
  }

  if (!trials) {
    const schedule = buildConditionSchedule(totalTrialsWanted, rand, modeValue);
    const totalTrials = schedule.length;
    const pickedItems = chooseSample(itemsForSampling, totalTrials, rand);
    trials = pickedItems.map((it, i) => ({ item: it, condition: schedule[i], list_id: null, list_group: null }));
  }

  state = {
    mode: modeValue,
    participant_id: resolveParticipantId(q),
    session_id: q.sid || buildSessionId(),
    list_id: Number.isFinite(selectedList) && selectedList > 0 ? selectedList : null,
    seed,
    stimuli_url: payloadUrl,
    design,
    trials,
    idx: 0,
    stage: "initial", // initial | ai | final | done
    selection: null,
    conf: 50,
    finalSelection: null,
    finalConf: 50,
    reveals: 0,
    revealOrder: [],
    revealedSet: new Set(),
    log: [],
    upload: {
      status: "idle", // idle | uploading | success | failed
      attempts: 0,
      error: null,
      submitted_at: null,
      auto_triggered: false,
    },
    trialStartAt: null,
    stageStartAt: null,
    started_at: null,
    completed_at: null,
  };
}

function buildAnswerSummary() {
  const finals = state.log.filter((e) => e.type === "submit_final");
  return finals.map((e, i) => ({
    trial_index: i + 1,
    item_id: e.item_id || null,
    condition: e.condition || null,
    initial_label: e.initial_label || null,
    initial_confidence: Number.isFinite(e.initial_confidence) ? e.initial_confidence : null,
    final_label: e.label || null,
    final_confidence: Number.isFinite(e.confidence) ? e.confidence : null,
    changed_mind: Boolean(e.initial_label && e.label && e.initial_label !== e.label),
    ai_label: e.ai_label || null,
    why: String(e.why || "").trim(),
  }));
}

function truthLabelForItem(item) {
  const raw = String(item?.truth?.label || item?.label || "").trim().toUpperCase();
  if (raw === "TRUE" || raw === "FAKE") return raw;
  return "UNSPECIFIED";
}

function renderDebriefClarifications() {
  if (!els.debriefClarificationList) return;
  els.debriefClarificationList.innerHTML = "";
  if (els.debriefClarificationDisclosure) {
    els.debriefClarificationDisclosure.open = false;
  }

  const trials = Array.isArray(state?.trials) ? state.trials : [];
  if (!trials.length) {
    const li = document.createElement("li");
    li.textContent = "No study items were available to summarize.";
    els.debriefClarificationList.appendChild(li);
    return;
  }

  for (let i = 0; i < trials.length; i++) {
    const item = trials[i]?.item || {};
    const li = document.createElement("li");
    const title = cleanDisplayText(item.title, true) || String(item.id || `Item ${i + 1}`);
    const truth = truthLabelForItem(item);
    li.textContent = `Item ${i + 1}: ${title} - veracity status: ${truth}.`;
    els.debriefClarificationList.appendChild(li);
  }
}

function buildSessionSummaryPayload() {
  const answers = buildAnswerSummary();
  const completedAt = state.completed_at || nowISO();
  const startedAt = state.started_at || completedAt;
  const startedMs = Date.parse(startedAt);
  const completedMs = Date.parse(completedAt);
  const durationSec = Number.isFinite(startedMs) && Number.isFinite(completedMs)
    ? Math.max(0, Math.round((completedMs - startedMs) / 1000))
    : null;
  const payload = {
    participant_id: state.participant_id || "anon",
    session_id: state.session_id || null,
    started_at: startedAt,
    completed_at: completedAt,
    duration_sec: durationSec,
    list_id: Number.isFinite(state.list_id) ? state.list_id : null,
    seed: state.seed,
    mode: state.mode,
    trial_count: answers.length,
    submitted_at: nowISO(),
    payload_version: SUMMARY_PAYLOAD_VERSION,
    answers_json: JSON.stringify(answers),
    events_json: JSON.stringify(state.log || []),
  };
  return payload;
}

function renderUploadStatus() {
  if (!els.uploadStatus || !els.retryUploadBtn) return;
  const upload = state?.upload;
  if (!upload || state.stage !== "done") {
    els.uploadStatus.hidden = true;
    els.retryUploadBtn.hidden = true;
    return;
  }

  els.uploadStatus.hidden = false;
  if (upload.status === "uploading") {
    els.uploadStatus.textContent = "Uploading summary to Formspree...";
    els.retryUploadBtn.hidden = true;
    return;
  }
  if (upload.status === "success") {
    els.uploadStatus.textContent = "Upload successful. Your session summary was saved.";
    els.retryUploadBtn.hidden = true;
    return;
  }
  if (upload.status === "failed") {
    const msg = upload.error ? `Upload failed: ${upload.error}` : "Upload failed. Please retry.";
    els.uploadStatus.textContent = msg;
    els.retryUploadBtn.hidden = false;
    return;
  }

  els.uploadStatus.textContent = "Submission complete. Preparing upload...";
  els.retryUploadBtn.hidden = true;
}

async function submitSessionSummary(options = {}) {
  const manual = Boolean(options.manual);
  if (!state?.upload) return;
  if (!manual && state.upload.status !== "idle") return;
  if (state.upload.status === "uploading") return;

  state.upload.status = "uploading";
  state.upload.error = null;
  state.upload.attempts += 1;
  renderUploadStatus();

  const payload = buildSessionSummaryPayload();
  try {
    if (!FORMSPREE_ENDPOINT) throw new Error("Demo mode: remote collection is disabled. Download the session locally.");
    const r = await fetch(FORMSPREE_ENDPOINT, {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    if (!r.ok) {
      let detail = `HTTP ${r.status}`;
      try {
        const j = await r.json();
        if (j?.errors?.[0]?.message) detail = j.errors[0].message;
      } catch (_) {
        // ignore parsing errors and keep status code
      }
      throw new Error(detail);
    }

    state.upload.status = "success";
    state.upload.submitted_at = nowISO();
    logEvent("remote_submit_success", { attempts: state.upload.attempts, summary_trials: payload.trial_count });
  } catch (err) {
    state.upload.status = "failed";
    state.upload.error = err?.message || "Unknown error";
    logEvent("remote_submit_failed", { attempts: state.upload.attempts, error: state.upload.error });
  }
  renderUploadStatus();
}

function maybeAutoSubmitSummary() {
  if (!state?.upload || state.upload.auto_triggered) return;
  state.upload.auto_triggered = true;
  submitSessionSummary();
}

function logEvent(type, data) {
  const t = state.trials[state.idx] || {};
  const now = Date.now();
  const elapsedStageMs = Number.isFinite(state.stageStartAt) ? Math.max(0, now - state.stageStartAt) : null;
  const elapsedTrialMs = Number.isFinite(state.trialStartAt) ? Math.max(0, now - state.trialStartAt) : null;
  state.log.push({
    t: nowISO(),
    type,
    mode: state.mode,
    participant_id: state.participant_id,
    list_id: state.list_id,
    seed: state.seed,
    condition: t.condition,
    list_group: t.list_group || null,
    item_id: t.item?.id,
    stage: state.stage,
    elapsed_stage_ms: elapsedStageMs,
    elapsed_trial_ms: elapsedTrialMs,
    ...data,
  });
}

function setChoice(which, isFinal) {
  if (!isFinal) {
    state.selection = which;
    els.chooseTrue.setAttribute("aria-pressed", which === "TRUE" ? "true" : "false");
    els.chooseFake.setAttribute("aria-pressed", which === "FAKE" ? "true" : "false");
  } else {
    state.finalSelection = which;
    els.chooseTrue.setAttribute("aria-pressed", which === "TRUE" ? "true" : "false");
    els.chooseFake.setAttribute("aria-pressed", which === "FAKE" ? "true" : "false");
  }
}

function renderTop() {
  const cond = CONDITIONS[state.trials[state.idx]?.condition] || CONDITIONS.C1;
  const t = state.trials[state.idx] || {};
  els.conditionPill.textContent = `${cond.id} - ${cond.name}`;
  els.trialPill.textContent = `Trial ${state.idx + 1} / ${state.trials.length}`;
  const listPart = state.list_id ? ` list=${state.list_id}` : "";
  const groupPart = t.list_group ? ` group=${t.list_group}` : "";
  els.seedPill.textContent = `pid=${state.participant_id || "--"}${listPart}${groupPart} seed=${state.seed}`;
  document.body.dataset.condition = cond.id;
  document.body.dataset.stage = state.stage;
}

function renderItem() {
  const item = state.trials[state.idx]?.item;
  els.sourceBadge.textContent = item.source || "Stimulus";
  els.itemId.textContent = item.id;
  const headline = cleanDisplayText(item.title, true);
  const excerpt = cleanDisplayText(item.content, false);
  els.headline.textContent = headline || item.title;
  els.excerpt.textContent = excerpt || item.content;
}

function renderAI() {
  const cond = CONDITIONS[state.trials[state.idx]?.condition] || CONDITIONS.C1;
  const item = state.trials[state.idx]?.item;
  if (state.stage === "initial" || state.stage === "done") {
    // Critical for the study: no AI shown before initial judgment.
    els.aiPanel.hidden = true;
    return;
  }
  if (!cond.hasAI) {
    els.aiPanel.hidden = true;
    return;
  }

  els.aiPanel.hidden = false;
  const pred = pickPred(item.ai);
  els.aiLabel.textContent = pred.label;
  els.aiProb.textContent = fmtPct(pred.prob);
  els.aiMeter.style.width = `${clamp01(pred.prob) * 100}%`;

  const pTrue = item.ai?.probs?.TRUE;
  const pFake = item.ai?.probs?.FAKE;
  els.aiProbFoot.textContent = `P(TRUE)=${fmtPct(pTrue)}, P(FAKE)=${fmtPct(pFake)}`;
}

function combinedSegments(item) {
  const titleSegs = repairSegmentsForDisplay(item?.title || "", item?.xai?.title_segments || []);
  const contentSegs = repairSegmentsForDisplay(item?.content || "", item?.xai?.content_segments || []);
  const out = [];
  for (const s of titleSegs) out.push({ ...s, where: "title" });
  out.push({ type: "break" });
  for (const s of contentSegs) out.push({ ...s, where: "content" });
  return out;
}

function renderXAI() {
  const cond = CONDITIONS[state.trials[state.idx]?.condition] || CONDITIONS.C1;
  const item = state.trials[state.idx]?.item;

  // Default: hide sequential controls (avoid carryover across conditions/stages).
  // Note: we also hide the individual buttons so they stay hidden even if CSS overrides
  // the container's `hidden` state.
  els.xaiControls.hidden = true;
  els.xaiControls.style.display = "none";
  els.revealNextBtn.hidden = true;
  els.revealNextBtn.style.display = "none";
  els.revealAllBtn.hidden = true;
  els.revealAllBtn.style.display = "none";
  els.revealCounter.hidden = true;
  els.revealCounter.style.display = "none";

  if (state.stage === "initial" || state.stage === "done") {
    // No explanation before initial judgment.
    els.xaiCard.hidden = true;
    els.xaiStats.textContent = "";
    return;
  }

  if (!cond.hasXAI || !item?.xai) {
    els.xaiCard.hidden = true;
    els.xaiStats.textContent = "";
    return;
  }

  els.xaiCard.hidden = false;
  els.xaiTarget.textContent = "";
  els.xaiTarget.removeAttribute("title");

  // Flatten only real segments for reveal ordering.
  const segsAll = combinedSegments(item);
  const segsFlat = segsAll.filter((s) => !s.type);
  const maxAbs = maxAbsShap(segsFlat);

  // Reveal model
  state.revealOrder = buildRevealOrder(segsFlat);
  const totalRevealable = state.revealOrder.length;
  const revealTarget = Math.min(SEQUENTIAL_REVEAL_TARGET, totalRevealable);
  state.revealTarget = revealTarget;
  state.totalRevealable = totalRevealable;

  // Determine which indices are revealed
  state.revealedSet = new Set();
  if (!cond.sequential) {
    for (let i = 0; i < segsFlat.length; i++) state.revealedSet.add(i);
  } else {
    for (let i = 0; i < state.reveals; i++) {
      const idx = state.revealOrder[i];
      if (idx === undefined) break;
      state.revealedSet.add(idx);
    }
  }

  // Controls (sequential reveal only)
  const showSeqControls = Boolean(cond.sequential && state.stage === "ai" && totalRevealable > 0);
  els.xaiControls.hidden = !showSeqControls;
  els.xaiControls.style.display = showSeqControls ? "flex" : "none";
  els.revealNextBtn.hidden = !showSeqControls;
  els.revealNextBtn.style.display = showSeqControls ? "" : "none";
  els.revealAllBtn.hidden = !showSeqControls;
  els.revealAllBtn.style.display = showSeqControls ? "" : "none";
  els.revealCounter.hidden = !showSeqControls;
  els.revealCounter.style.display = showSeqControls ? "" : "none";
  if (cond.sequential) {
    const shownCount = Math.min(state.revealedSet.size, revealTarget);
    const extra = revealTarget < totalRevealable ? ` (${totalRevealable} tokens available)` : "";
    els.revealCounter.textContent = `${shownCount} / ${revealTarget} highlights revealed${extra}`;
    els.revealNextBtn.disabled = shownCount >= revealTarget;
    els.revealNextBtn.textContent = SEQUENTIAL_REVEAL_STEP === 1 ? "Reveal next highlight" : `Reveal next ${SEQUENTIAL_REVEAL_STEP} highlights`;
  } else {
    els.revealCounter.textContent = "";
    els.revealNextBtn.disabled = true;
  }

  const shownSum = segsFlat.reduce((acc, s, i) => (state.revealedSet.has(i) ? acc + Number(s.shap || 0) : acc), 0);
  const allSum = segsFlat.reduce((acc, s) => acc + Number(s.shap || 0), 0);
  const pFake = item?.ai?.probs?.FAKE;
  const finalLogit = logit(pFake);
  const baseFromData = Number(item?.xai?.base_value);
  const baseValue = Number.isFinite(baseFromData) ? baseFromData : (Number.isFinite(finalLogit) ? finalLogit - allSum : NaN);
  const shownTotal = Number.isFinite(baseValue) ? baseValue + shownSum : NaN;
  const fullTotal = Number.isFinite(baseValue) ? baseValue + allSum : NaN;
  const shownTxt = Number.isFinite(shownSum) ? shownSum.toFixed(3) : "--";
  const allTxt = Number.isFinite(allSum) ? allSum.toFixed(3) : "--";
  const baseTxt = Number.isFinite(baseValue) ? baseValue.toFixed(3) : "--";
  const shownNetTxt = fmtSigned(shownTotal);
  const fullNetTxt = fmtSigned(fullTotal);
  const shownDir = directionFromLogit(shownTotal);
  const fullDir = directionFromLogit(fullTotal);
  const partialTag = cond.sequential ? "(shown)" : "";
  els.xaiStats.textContent = `SHAP sum shown=${shownTxt} ${partialTag} | SHAP sum all=${allTxt} | base=${baseTxt} | net shown=${shownNetTxt} (${shownDir}) | net all=${fullNetTxt} (${fullDir})`;

  els.boundaryCue.hidden = !cond.boundaryCue;

  // Render text
  els.xaiText.innerHTML = "";
  let flatIdx = 0;
  for (const seg of segsAll) {
    if (seg.type === "break") {
      const br = document.createElement("div");
      br.style.height = "10px";
      br.style.borderTop = "1px dashed rgba(16,33,46,0.18)";
      br.style.margin = "10px 0 8px";
      els.xaiText.appendChild(br);
      continue;
    }

    const revealed = state.revealedSet.has(flatIdx);
    const span = document.createElement("span");
    span.className = "token";
    span.style.background = tokenBg(seg.shap, maxAbs, revealed);
    span.dataset.show = "false";
    span.title = revealed ? `SHAP: ${seg.shap.toFixed(3)}` : "(hidden until revealed)";

    const txt = document.createElement("span");
    txt.textContent = seg.text;
    span.appendChild(txt);

    const val = document.createElement("div");
    val.className = "token__val";
    val.textContent = revealed ? seg.shap.toFixed(3) : "";
    span.appendChild(val);

    span.addEventListener("click", () => {
      if (!revealed) return;
      const on = span.dataset.show === "true";
      span.dataset.show = on ? "false" : "true";
      logEvent("token_toggle", { token: seg.text.trim().slice(0, 40), shap: seg.shap, show: !on });
    });

    els.xaiText.appendChild(span);
    flatIdx++;
  }
}

function renderStage() {
  const cond = CONDITIONS[state.trials[state.idx]?.condition] || CONDITIONS.C1;

  els.primaryBtn.disabled = false;

  // Reset UI pieces
  els.whyBox.hidden = true;
  els.primaryBtn.hidden = false;
  els.downloadBtn.hidden = true;
  if (els.uploadStatus) els.uploadStatus.hidden = true;
  if (els.retryUploadBtn) els.retryUploadBtn.hidden = true;
  if (els.debriefPage) els.debriefPage.hidden = true;
  if (els.debriefCloseHint) els.debriefCloseHint.textContent = "";

  // Controls/summary visibility
  els.summaryBox.hidden = true;
  els.controlsBox.hidden = false;

  if (state.stage === "initial") {
    els.stageKicker.textContent = "Stage 1";
    els.stageTitle.textContent = "Initial judgment";
    if (cond.hasAI) {
      els.noteBox.textContent = "Choose a label and set your confidence. Then reveal the next screen.";
      els.primaryBtn.textContent = "Reveal AI";
    } else {
      els.noteBox.textContent =
        "Choose a label and set your confidence. Next is a brief review screen (no AI) before your final judgment.";
      els.primaryBtn.textContent = "Review (no AI)";
    }
    els.controlsBox.hidden = false;
    els.summaryBox.hidden = true;
    renderAI();
    renderXAI();
    return;
  }

  if (state.stage === "ai") {
    els.stageKicker.textContent = "Stage 2";
    els.stageTitle.textContent = cond.hasAI ? "AI output" : "Review (no AI)";

    // Lock initial judgment and show summary
    els.controlsBox.hidden = true;
    els.summaryBox.hidden = false;
    els.initialSummary.textContent = state.selection ? state.selection : "--";
    els.initialConfSummary.textContent = Number.isFinite(state.conf) ? String(state.conf) : "--";

    renderAI();
    renderXAI();

    if (!cond.hasAI) {
      els.noteBox.textContent =
        "Review screen: re-read the headline/excerpt. No AI is available for this trial. Proceed when ready.";
      els.primaryBtn.textContent = "Final judgment";
      return;
    }

    if (cond.hasXAI && cond.sequential) {
      const target = Number.isFinite(state.revealTarget) ? state.revealTarget : 0;
      if (target === 0) {
        els.noteBox.textContent = "No highlighted tokens are available for this item. Proceed when ready.";
      } else {
        els.noteBox.textContent = `Reveal highlights step by step. The colors show which words influenced the AI, but the AI can still be wrong. Proceed after ${target} highlighted words.`;
      }
      const done = els.revealNextBtn.disabled;
      els.primaryBtn.textContent = done ? "Final judgment" : "Keep revealing";
      els.primaryBtn.disabled = !done;
      return;
    }

    if (cond.hasXAI) {
      els.noteBox.textContent = "Explanation highlights are shown with the AI prediction. Use them as guidance, not as proof. Proceed when ready.";
      els.primaryBtn.textContent = "Final judgment";
      return;
    }

    els.noteBox.textContent = "AI label + probability are shown. Proceed when ready.";
    els.primaryBtn.textContent = "Final judgment";
    return;
  }

  if (state.stage === "final") {
    els.stageKicker.textContent = "Stage 3";
    els.stageTitle.textContent = "Final judgment";
    els.noteBox.textContent = "Record your final decision and confidence. Optionally write what influenced you.";
    els.primaryBtn.textContent = state.idx + 1 < state.trials.length ? "Submit & next" : "Finish";
    els.primaryBtn.disabled = false;
    els.controlsBox.hidden = false;
    els.summaryBox.hidden = false;
    els.initialSummary.textContent = state.selection ? state.selection : "--";
    els.initialConfSummary.textContent = Number.isFinite(state.conf) ? String(state.conf) : "--";
    els.whyBox.hidden = false;
    renderAI();
    renderXAI();
    return;
  }

  if (state.stage === "done") {
    els.stageKicker.textContent = "Complete";
    els.stageTitle.textContent = "Submission complete";
    els.noteBox.textContent = "Thanks for your participation. Your submission is saved automatically.";
    els.primaryBtn.hidden = true;
    els.downloadBtn.hidden = true;
    els.controlsBox.hidden = true;
    els.summaryBox.hidden = true;
    if (els.debriefPage) els.debriefPage.hidden = false;
    renderDebriefClarifications();
    renderUploadStatus();
    maybeAutoSubmitSummary();
    renderAI();
    renderXAI();
    return;
  }
}

function setStage(next) {
  state.stage = next;
  state.stageStartAt = Date.now();
  if (next === "done" && !state.completed_at) {
    state.completed_at = nowISO();
  }
  logEvent("stage", { to: next });
  document.body.dataset.stage = next;
  // Reset choice UI state per stage
  els.chooseTrue.setAttribute("aria-pressed", "false");
  els.chooseFake.setAttribute("aria-pressed", "false");
  if (next === "final") {
    state.finalSelection = null;
    state.finalConf = 50;
    els.confSlider.value = "50";
    els.confValue.textContent = "50";
    els.whyText.value = "";
  }
  if (next === "ai") {
    state.reveals = 0;
  }
  renderStage();
}

function downloadLog() {
  const lines = state.log.map((e) => JSON.stringify(e));
  const blob = new Blob([lines.join("\n") + "\n"], { type: "application/x-ndjson" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  const pid = state.participant_id || "anon";
  const listPart = state.list_id ? `_list${state.list_id}` : "";
  a.download = `session_${pid}_${state.mode}${listPart}_seed${state.seed}_${new Date().toISOString().replace(/[:.]/g, "-")}.jsonl`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function finishDebrief() {
  if (window.opener) {
    window.close();
    return;
  }
  if (els.debriefCloseHint) {
    els.debriefCloseHint.textContent = "You can now close this page.";
  }
}

function tickTimer() {
  if (!state?.stageStartAt) return;
  const elapsed = Math.max(0, Date.now() - state.stageStartAt);
  const s = Math.floor(elapsed / 1000);
  const mm = String(Math.floor(s / 60)).padStart(2, "0");
  const ss = String(s % 60).padStart(2, "0");
  els.timer.textContent = `${mm}:${ss}`;
}

function startTimer() {
  if (timerHandle) clearInterval(timerHandle);
  timerHandle = setInterval(tickTimer, 250);
}

async function bootAsync() {
  const q = parseQuery();
  let assignedList = normalizeListId(q.list);
  if (!assignedList) {
    assignedList = await requestServerAssignedList(q);
  }
  if (!assignedList) {
    assignedList = resolveAssignedList(q);
  }
  ensureQueryHasAssignedList(assignedList);
  if (assignedList) {
    els.listSelect.value = String(assignedList);
  }
  if (els.listSelect) {
    els.listSelect.disabled = true;
    els.listSelect.title = "List is assigned automatically for this study.";
  }

  const loaded = await fetchFirstJson(DEFAULT_STIMULI_URLS);
  resetState(loaded.items, loaded.url, loaded.meta || {});
  logEvent("boot", { trials: state.trials.length, stimuli_url: state.stimuli_url });
  renderIntroScreen();

  if (isDebriefPreviewRequested(q)) {
    if (!state.started_at) state.started_at = nowISO();
    state.completed_at = nowISO();
    setStage("done");
    showIntroScreen(false);
    if (timerHandle) clearInterval(timerHandle);
    return;
  }

  if (q.autostart) {
    beginSessionFromIntro();
    return;
  }

  if (timerHandle) clearInterval(timerHandle);
  showIntroScreen(true);
}

// Events
els.listSelect.addEventListener("change", () => {
  // Reboot so schedule + sampling stay coherent.
  bootAsync();
});

els.restartBtn.addEventListener("click", () => {
  bootAsync();
});

if (els.introStartBtn) {
  els.introStartBtn.addEventListener("click", () => {
    beginSessionFromIntro();
  });
}

els.confSlider.addEventListener("input", () => {
  const v = Number(els.confSlider.value);
  els.confValue.textContent = String(v);
  if (state.stage === "final") state.finalConf = v;
  else state.conf = v;
});

els.chooseTrue.addEventListener("click", () => {
  const isFinal = state.stage === "final";
  setChoice("TRUE", isFinal);
  logEvent("choose", { which: "TRUE", final: isFinal });
});
els.chooseFake.addEventListener("click", () => {
  const isFinal = state.stage === "final";
  setChoice("FAKE", isFinal);
  logEvent("choose", { which: "FAKE", final: isFinal });
});

els.revealNextBtn.addEventListener("click", () => {
  const target = Number.isFinite(state.revealTarget) ? state.revealTarget : state.reveals + SEQUENTIAL_REVEAL_STEP;
  state.reveals = Math.min(state.reveals + SEQUENTIAL_REVEAL_STEP, target);
  logEvent("reveal_next", { reveals: state.reveals });
  renderXAI();
  renderStage();
});

els.revealAllBtn.addEventListener("click", () => {
  state.reveals = Number.isFinite(state.revealTarget) ? state.revealTarget : state.revealOrder.length;
  logEvent("reveal_all", { reveals: state.reveals });
  renderXAI();
  renderStage();
});

els.primaryBtn.addEventListener("click", () => {
  const cond = CONDITIONS[state.trials[state.idx]?.condition] || CONDITIONS.C1;
  const item = state.trials[state.idx]?.item;

  if (state.stage === "initial") {
    if (!state.selection) {
      els.noteBox.textContent = "Please choose TRUE or FAKE before continuing.";
      return;
    }
    logEvent("submit_initial", { label: state.selection, confidence: state.conf });
    els.primaryBtn.disabled = false;
    setStage("ai");
    // In sequential conditions, we start with 0 reveals
    state.reveals = 0;
    renderAI();
    renderXAI();
    renderStage();
    return;
  }

  if (state.stage === "ai") {
    if (cond.hasXAI && cond.sequential) {
      // Must reveal all before proceeding
      const target = Number.isFinite(state.revealTarget) ? state.revealTarget : 0;
      if (state.revealedSet.size < target) return;
    }
    setStage("final");
    return;
  }

  if (state.stage === "final") {
    if (!state.finalSelection) {
      els.noteBox.textContent = "Please choose TRUE or FAKE for your final judgment.";
      return;
    }
    logEvent("submit_final", {
      label: state.finalSelection,
      confidence: state.finalConf,
      why: els.whyText.value || "",
      initial_label: state.selection,
      initial_confidence: state.conf,
      ai_label: cond.hasAI ? pickPred(item.ai).label : null,
    });

    // Next item or done
    if (state.idx + 1 < state.trials.length) {
      state.idx += 1;
      state.trialStartAt = Date.now();
      state.selection = null;
      state.conf = 50;
      els.confSlider.value = "50";
      els.confValue.textContent = "50";
      els.whyText.value = "";
      renderTop();
      renderItem();
      setStage("initial");
      return;
    }

    setStage("done");
    els.primaryBtn.disabled = false;
    return;
  }

  if (state.stage === "done") return;
});

els.downloadBtn.addEventListener("click", downloadLog);
if (els.debriefFinishBtn) {
  els.debriefFinishBtn.addEventListener("click", finishDebrief);
}
if (els.debriefSaveBtn) {
  els.debriefSaveBtn.addEventListener("click", () => {
    window.print();
  });
}
if (els.retryUploadBtn) {
  els.retryUploadBtn.addEventListener("click", () => {
    submitSessionSummary({ manual: true });
  });
}

// Init
bootAsync();
