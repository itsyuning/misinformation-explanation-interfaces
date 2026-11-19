const AUTO_LIST_IDS = [1, 2, 3, 4, 5];

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

async function kvCall(path) {
  const base = process.env.KV_REST_API_URL;
  const token = process.env.KV_REST_API_TOKEN;
  if (!base || !token) return null;
  const res = await fetch(`${base}${path}`, {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
  if (!res.ok) {
    throw new Error(`KV request failed: ${res.status}`);
  }
  const data = await res.json();
  return data?.result;
}

async function assignFromKV(pid) {
  const mapKey = pid ? `assign:pid:${pid}` : null;
  if (mapKey) {
    const existing = normalizeListId(await kvCall(`/get/${encodeURIComponent(mapKey)}`));
    if (existing) return existing;
  }

  const counter = Number(await kvCall("/incr/assign:counter"));
  if (!Number.isFinite(counter) || counter <= 0) {
    throw new Error("Invalid KV counter result");
  }
  const list = AUTO_LIST_IDS[(Math.floor(counter) - 1) % AUTO_LIST_IDS.length];

  if (mapKey) {
    await kvCall(`/set/${encodeURIComponent(mapKey)}/${list}`);
  }
  return list;
}

export default async function handler(req, res) {
  if (req.method !== "GET") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  const pid = String(req.query?.pid || "").trim();

  try {
    const list = await assignFromKV(pid || null);
    res.status(200).json({ list, source: "kv_round_robin" });
    return;
  } catch (_err) {
    // Fallback mode when KV is not configured or unavailable.
    const list = pid
      ? AUTO_LIST_IDS[hashStringToUint(pid) % AUTO_LIST_IDS.length]
      : AUTO_LIST_IDS[Math.floor(Math.random() * AUTO_LIST_IDS.length)];
    res.status(200).json({ list, source: pid ? "pid_hash_fallback" : "random_fallback" });
  }
}
