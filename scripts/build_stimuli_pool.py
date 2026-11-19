#!/usr/bin/env python3
"""Build a stimuli pool JSON for the XAI UI.

Outputs a JSON file consumed by `xai_ui/app.js`.

Default dataset is `Ambrosio1994/real-and-fake-news` because it provides title + text + label.
Default model is `hamzab/roberta-fake-news-classification`.

Notes:
- The model expects a formatted prompt: <title>...<content>...<end>
- Explanations use SHAP on logit(P(FAKE)) so tokens that increase FAKE are red in the UI.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import string
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING


DEFAULT_DATASET = "Ambrosio1994/real-and-fake-news"
DEFAULT_SPLIT = "train"
DEFAULT_MODEL = "hamzab/roberta-fake-news-classification"
CONDITION_ORDER = ["C0", "C1", "C2", "C3", "C4"]
TOPIC_ORDER = ["POLITICS", "HEALTH", "SCI_TECH", "BUSINESS", "SOCIETY"]

NOISY_RE = re.compile(
    r"(\bcdata\b|getelementsbytagname|getelementbyid|connect\.facebook\.net|\bjssdk\b|document\s+script|function\(d\s+s|xfbml=1)",
    flags=re.IGNORECASE,
)

BOILERPLATE_RE = re.compile(
    r"(this\s+is\s+a\s+satirical\s+website|don't\s+take\s+it\s+seriously|it'?s\s+a\s+joke|report\s+abuse|\bloading\b|\b\d{2,}\s+\d{2,}\s+shares\b)",
    flags=re.IGNORECASE,
)

TOPIC_PATTERNS: Dict[str, re.Pattern[str]] = {
    "POLITICS": re.compile(
        r"\b(president|prime minister|senate|congress|parliament|election|electoral|campaign|voter|white house|government|minister|policy|supreme court|lawmakers?)\b",
        flags=re.IGNORECASE,
    ),
    "HEALTH": re.compile(
        r"\b(health|medical|medicine|hospital|doctor|disease|virus|vaccine|covid|flu|clinical|patient|treatment|drug|cdc|who)\b",
        flags=re.IGNORECASE,
    ),
    "SCI_TECH": re.compile(
        r"\b(science|scientist|research|study|technology|tech|software|internet|ai\b|artificial intelligence|cyber|nasa|space|satellite|robot|chip|semiconductor)\b",
        flags=re.IGNORECASE,
    ),
    "BUSINESS": re.compile(
        r"\b(business|economy|economic|market|stock|trade|bank|inflation|gdp|company|industry|investor|revenue|profit|oil|currency|federal reserve)\b",
        flags=re.IGNORECASE,
    ),
    "SOCIETY": re.compile(
        r"\b(school|education|housing|community|police|crime|court case|family|culture|sports?|weather|climate|environment|transport|wildfire|flood|immigration)\b",
        flags=re.IGNORECASE,
    ),
}

ENTERTAINMENT_RE = re.compile(
    r"\b(celebrity|celeb|actor|actress|singer|song|album|movie|film|tv|episode|season|showbiz|hollywood|grammy|oscar|red carpet|kardashian|jenner|bieber|swift|stefani|mariah|lopez|thrones|bachelor|instagram|romance|boyfriend|girlfriend|wedding|baby\s+number)\b",
    flags=re.IGNORECASE,
)

RELATIONSHIP_RE = re.compile(
    r"\b(dating|romance|boyfriend|girlfriend|husband|wife|wedding|pregnant|baby\s+girl|baby\s+boy|split|breakup|cheating)\b",
    flags=re.IGNORECASE,
)

STRICT_NON_NEWS_TITLE_RE = re.compile(
    r"\b(\d+\s+(things|ways|ideas|tips)|how\s+to|what\s+you\s+need\s+to\s+know|tricks\s+that|best\s+\d+|unique\s+\w+\s+ideas)\b",
    flags=re.IGNORECASE,
)

CLICKBAIT_RE = re.compile(
    r"(\[video\]|\[image\]|\bbreaking\b|\bnew low\b|\bstunning\b|\bfreaking out\b|\bshocking\b|\bmic drop\b|\bf\*ck\b|\bfull of sh\*t\b)",
    flags=re.IGNORECASE,
)

EN_STOP = {
    "the",
    "and",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "that",
    "is",
    "was",
    "are",
    "be",
    "as",
    "by",
    "from",
    "at",
    "it",
}


# Only strip literal bracket tokens used in marker formatting.
# Do not strip plain words like "end"/"content" here, otherwise normal words
# (e.g., "Defend") can be broken when subword tokens are merged for display.
DELIM_TOKENS = {"<", ">"}
SPECIAL_TOKENS = {"<s>", "</s>", "[CLS]", "[SEP]", "[PAD]"}


if TYPE_CHECKING:

    def _make_shap_text_masker(tokenizer: Any) -> Any:  # pragma: no cover
        ...

else:

    def _make_shap_text_masker(tokenizer: Any) -> Any:
        import shap  # type: ignore

        return shap.maskers.Text(tokenizer)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def clean_ws(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    return s


def strip_social_fragments(text: str) -> str:
    t = str(text or "")
    # Remove embedded tweet/link artifacts that reduce readability.
    t = re.sub(r"^\s*\d+\s*/\s*\d+\s+", " ", t)
    t = re.sub(r"File\s+Photo\s+by\s+[^|]{0,120}\|\s*License\s+Photo", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\b[A-Z][a-z]+\s+\d{1,2}\s*\(UPI\)\s*--", " ", t)
    t = re.sub(r"\bAdvertisement\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"https?://\S+", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\bpic\.twitter\.com/\S+", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"[@#][A-Za-z0-9_]+", " ", t)
    t = re.sub(r"\bFollow\s+live\s+at\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+[\-\u2014]\s*[A-Za-z0-9_\s]+\([^)]*\)\s+[A-Z][a-z]+\s+\d{1,2},\s+\d{4}", " ", t)
    t = re.sub(r"([.!?])([A-Z])", r"\1 \2", t)
    return clean_ws(t)


def is_noisy_text(s: str) -> bool:
    return bool(NOISY_RE.search(str(s or "")))


def has_boilerplate_noise(s: str) -> bool:
    return bool(BOILERPLATE_RE.search(str(s or "")))


def is_englishish(s: str, min_stop_hits: int = 4) -> bool:
    txt = str(s or "")
    if not txt:
        return False
    letters = sum(ch.isalpha() for ch in txt)
    ascii_letters = sum(("a" <= ch.lower() <= "z") for ch in txt)
    if letters <= 0:
        return False
    if (ascii_letters / letters) < 0.92:
        return False
    toks = re.findall(r"[a-z]+", txt.lower())
    stop_hits = sum(1 for t in toks if t in EN_STOP)
    return stop_hits >= min_stop_hits


def is_entertainment_like(title: str, content: str) -> bool:
    return bool(ENTERTAINMENT_RE.search(f"{title} {content}"))


def looks_person_name_heavy(title: str) -> bool:
    toks = re.findall(r"\b[A-Z][a-z]{2,}\b", str(title or ""))
    if len(toks) < 2:
        return False
    # Discount common sentence-leading words.
    stop = {
        "The",
        "This",
        "That",
        "These",
        "Those",
        "How",
        "Why",
        "What",
        "When",
        "Where",
        "Who",
    }
    toks2 = [t for t in toks if t not in stop]
    return len(toks2) >= 2


def has_topic_signal(topic: str, title: str, content: str) -> bool:
    pat = TOPIC_PATTERNS.get(str(topic or ""))
    if pat is None:
        return False
    text = f"{title} {content}"
    return len(pat.findall(text)) >= 1


def has_excessive_uppercase(text: str) -> bool:
    t = str(text or "")
    letters = [ch for ch in t if ch.isalpha()]
    if len(letters) < 20:
        return False
    upper = sum(1 for ch in letters if ch.isupper())
    return (upper / len(letters)) >= 0.35


def detect_topic(title: str, content: str) -> Optional[str]:
    text = f"{title} {content}"
    best_topic: Optional[str] = None
    best_count = 0
    for topic in TOPIC_ORDER:
        c = len(TOPIC_PATTERNS[topic].findall(text))
        if c > best_count:
            best_count = c
            best_topic = topic
    if best_count <= 0:
        return None
    return best_topic


def normalize_split_truth(split_name: str) -> Optional[str]:
    s = str(split_name or "").strip().upper()
    if not s:
        return None
    if s.endswith("F"):
        return "FAKE"
    if s.endswith("R"):
        return "TRUE"
    return None


def get_first_nonempty(row: Dict[str, Any], keys: Sequence[str]) -> str:
    for k in keys:
        v = row.get(k)
        if v is None:
            continue
        txt = clean_ws(str(v))
        if txt:
            return txt
    return ""


def load_whitelist_rows(path: str) -> List[Dict[str, Any]]:
    p = str(path or "").strip()
    if not p:
        return []
    ext = os.path.splitext(p)[1].lower()
    if ext == ".jsonl":
        out: List[Dict[str, Any]] = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if not t:
                    continue
                v = json.loads(t)
                if isinstance(v, dict):
                    out.append(v)
        return out
    if ext == ".json":
        v = json.load(open(p, "r", encoding="utf-8"))
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
        if isinstance(v, dict) and isinstance(v.get("items"), list):
            return [x for x in v["items"] if isinstance(x, dict)]
        raise RuntimeError(f"Unsupported JSON whitelist shape in {p}")
    if ext == ".csv":
        with open(p, "r", encoding="utf-8", newline="") as f:
            return [dict(r) for r in csv.DictReader(f)]
    raise RuntimeError(f"Unsupported whitelist file type for {p}; expected .csv/.json/.jsonl")


def truncate_words(text: str, max_words: int) -> str:
    txt = clean_ws(text)
    words = txt.split(" ")
    if len(words) <= max_words:
        return txt

    # Strict complete-sentence truncation.
    # Keep as many whole sentences as possible under max_words.
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", txt) if s.strip()]
    if not sents:
        return ""

    out: List[str] = []
    used = 0
    for s in sents:
        n = len(s.split(" "))
        if used + n > max_words:
            break
        out.append(s)
        used += n

    # If no sentence fits under max_words, keep the first full sentence when it is
    # only moderately longer than the cap. This preserves sentence completeness.
    if not out:
        first_n = len(sents[0].split(" "))
        if first_n <= (max_words + 60):
            return sents[0]
        return ""

    return " ".join(out).strip()


def is_low_quality_text(text: str) -> bool:
    t = clean_ws(text)
    if not t:
        return True
    words = [w for w in re.split(r"\s+", t) if w]
    if len(words) < 6:
        return True
    alpha_chars = sum(ch.isalpha() for ch in t)
    if alpha_chars <= 0:
        return True
    non_alpha_ratio = 1.0 - (alpha_chars / max(1, len(t)))
    if non_alpha_ratio > 0.42:
        return True
    return False


def make_prompt(title: str, content: str) -> str:
    # Keep this consistent with the notebook + prior work.
    return f"<title>{title}<content>{content}<end>"


def find_label_names(id2label: Dict[int, str]) -> List[str]:
    n = len(id2label)
    return [str(id2label[i]) for i in range(n)]


def pick_fake_index(labels: Sequence[str]) -> int:
    # Prefer exact FAKE / REAL-ish names.
    upper = [str(x).upper() for x in labels]
    if "FAKE" in upper:
        return upper.index("FAKE")
    if "FALSE" in upper:
        return upper.index("FALSE")
    # fall back: assume binary and class-1 is fake
    if len(labels) == 2:
        return 1
    raise RuntimeError(f"Cannot infer fake label from labels={labels}")


def normalize_truth(label_val: Any) -> str:
    # Common conventions: 0=REAL/TRUE, 1=FAKE.
    if isinstance(label_val, str):
        u = label_val.strip().upper()
        if u in {"REAL", "TRUE", "0", "FACT", "FACTUAL", "ACCURATE", "CORRECT"}:
            return "TRUE"
        if u in {
            "FAKE",
            "FALSE",
            "1",
            "MISINFORMATION",
            "MISINFO",
            "DISINFORMATION",
            "HOAX",
            "RUMOR",
            "RUMOUR",
            "INACCURATE",
            "INCORRECT",
        }:
            return "FAKE"
    try:
        iv = int(label_val)
        return "FAKE" if iv == 1 else "TRUE"
    except Exception:
        return "TRUE"


def token_display(tok: str) -> str:
    t = str(tok)
    if t in SPECIAL_TOKENS:
        return ""
    if t.startswith("Ġ"):
        return " " + t[1:]
    if t.startswith("▁"):
        return " " + t[1:]
    return t


def is_delim_token(tok: str) -> bool:
    t = str(tok).strip().lower()
    return t in DELIM_TOKENS


def safe_float(x: Any) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return 0.0


def confidence_of_probs(probs: Dict[str, float]) -> float:
    if not probs:
        return float("nan")
    try:
        return float(max(float(v) for v in probs.values()))
    except Exception:
        return float("nan")


def infer_pred_label(probs: Dict[str, float], fake_threshold: float = 0.5) -> str:
    p_fake = safe_float(probs.get("FAKE", float("nan")))
    p_true = safe_float(probs.get("TRUE", float("nan")))
    if math.isfinite(p_fake) and math.isfinite(p_true):
        return "FAKE" if p_fake >= fake_threshold else "TRUE"
    if probs:
        return max(probs.items(), key=lambda kv: kv[1])[0]
    return "--"


def confidence_bin(conf: float, low_max: float, mid_max: float) -> str:
    c = safe_float(conf)
    if c <= low_max:
        return "low"
    if c <= mid_max:
        return "mid"
    return "high"


def pick_balanced_conf(
    lst: List[Dict[str, Any]],
    k: int,
    rng: random.Random,
    low_max: float,
    mid_max: float,
) -> List[Dict[str, Any]]:
    if k <= 0 or not lst:
        return []

    bins: Dict[str, List[Dict[str, Any]]] = {"low": [], "mid": [], "high": []}
    for e in lst:
        conf = safe_float(e.get("ai_conf"))
        b = confidence_bin(conf, low_max=low_max, mid_max=mid_max)
        bins[b].append(e)

    for b in bins:
        rng.shuffle(bins[b])

    # Try to spread selection across confidence bins, then fill remainder from any available bin.
    base = k // 3
    targets: Dict[str, int] = {"low": base, "mid": base, "high": k - 2 * base}

    selected: List[Dict[str, Any]] = []
    used_ids = set()
    taken: Dict[str, int] = {"low": 0, "mid": 0, "high": 0}

    for b in ["low", "mid", "high"]:
        need = targets[b]
        while need > 0 and bins[b]:
            e = bins[b].pop()
            key = (e.get("row_idx"), e.get("title"))
            if key in used_ids:
                continue
            used_ids.add(key)
            selected.append(e)
            taken[b] += 1
            need -= 1

    while len(selected) < k:
        # Prefer bins with most leftovers to avoid overfitting to one confidence stratum.
        choices = sorted(["low", "mid", "high"], key=lambda b: len(bins[b]), reverse=True)
        if not bins[choices[0]]:
            break
        for b in choices:
            if not bins[b]:
                continue
            e = bins[b].pop()
            key = (e.get("row_idx"), e.get("title"))
            if key in used_ids:
                continue
            used_ids.add(key)
            selected.append(e)
            taken[b] += 1
            break

    return selected[:k]


def assign_item_groups(items: List[Dict[str, Any]], n_groups: int, seed: int, low_max: float, mid_max: float) -> List[str]:
    if n_groups <= 0:
        raise RuntimeError("n_groups must be > 0")

    labels = list(string.ascii_uppercase[:n_groups])
    rng = random.Random(seed + 911)

    strata: Dict[Tuple[str, bool, str], List[Dict[str, Any]]] = {}
    for it in items:
        truth = str(it.get("truth", {}).get("label", "TRUE"))
        ai_correct = bool(it.get("ai", {}).get("correct", False))
        conf = confidence_of_probs(it.get("ai", {}).get("probs", {}))
        cbin = confidence_bin(conf, low_max=low_max, mid_max=mid_max)
        key = (truth, ai_correct, cbin)
        strata.setdefault(key, []).append(it)

    for key in strata:
        rng.shuffle(strata[key])

    group_buckets: Dict[str, List[Dict[str, Any]]] = {g: [] for g in labels}
    gi = 0
    for key in sorted(strata.keys()):
        for it in strata[key]:
            g = labels[gi % n_groups]
            group_buckets[g].append(it)
            gi += 1

    for g in labels:
        for it in group_buckets[g]:
            it["group"] = g

    return labels


def build_latin_square_lists(group_labels: Sequence[str], conditions: Sequence[str]) -> Dict[str, Dict[str, str]]:
    if len(group_labels) != len(conditions):
        return {}

    out: Dict[str, Dict[str, str]] = {}
    n = len(group_labels)
    for r in range(n):
        mapping = {}
        for c_idx, cond in enumerate(conditions):
            mapping[cond] = group_labels[(c_idx + r) % n]
        out[str(r + 1)] = mapping
    return out


def select_balanced_study_pool(
    scored: List[Dict[str, Any]],
    total_n: int,
    n_groups: int,
    n_wrong_target: int,
    rng: random.Random,
    low_max: float,
    mid_max: float,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    if n_groups != len(TOPIC_ORDER):
        raise RuntimeError(f"For study design, --n-groups must be {len(TOPIC_ORDER)}")
    if total_n != n_groups * len(TOPIC_ORDER):
        raise RuntimeError(f"For study design, --n must equal {n_groups * len(TOPIC_ORDER)}")
    if n_wrong_target % len(TOPIC_ORDER) != 0:
        raise RuntimeError("For study design, --n-wrong must be divisible by number of topics")

    per_topic = total_n // len(TOPIC_ORDER)
    wrong_per_topic = n_wrong_target // len(TOPIC_ORDER)
    correct_per_topic = per_topic - wrong_per_topic
    if wrong_per_topic <= 0 or correct_per_topic <= 0:
        raise RuntimeError("Invalid per-topic correct/wrong allocation")

    group_labels = list(string.ascii_uppercase[:n_groups])
    selected_by_topic: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for topic in TOPIC_ORDER:
        wrong_pool = [x for x in scored if x.get("topic") == topic and (not bool(x.get("ai_correct")))]
        correct_pool = [x for x in scored if x.get("topic") == topic and bool(x.get("ai_correct"))]

        if len(wrong_pool) < wrong_per_topic or len(correct_pool) < correct_per_topic:
            raise RuntimeError(
                f"Insufficient items for topic={topic}: "
                f"need wrong/correct={wrong_per_topic}/{correct_per_topic}, "
                f"have {len(wrong_pool)}/{len(correct_pool)}"
            )

        selected_wrong = pick_balanced_conf(
            wrong_pool,
            wrong_per_topic,
            rng=rng,
            low_max=low_max,
            mid_max=mid_max,
        )
        selected_correct = pick_balanced_conf(
            correct_pool,
            correct_per_topic,
            rng=rng,
            low_max=low_max,
            mid_max=mid_max,
        )
        if len(selected_wrong) != wrong_per_topic or len(selected_correct) != correct_per_topic:
            raise RuntimeError(f"Failed selecting balanced confidence subset for topic={topic}")
        selected_by_topic[topic] = {"wrong": selected_wrong, "correct": selected_correct}

    # Assign per topic to exactly one item per group.
    # Wrong pattern: for topic t, groups t and (t+1) are wrong -> each group gets 2 wrong total.
    assigned: List[Dict[str, Any]] = []
    for t_idx, topic in enumerate(TOPIC_ORDER):
        wrong = list(selected_by_topic[topic]["wrong"])
        correct = list(selected_by_topic[topic]["correct"])
        rng.shuffle(wrong)
        rng.shuffle(correct)

        wrong_group_idx = [t_idx % n_groups, (t_idx + 1) % n_groups]
        correct_group_idx = [i for i in range(n_groups) if i not in wrong_group_idx]

        if len(wrong) != len(wrong_group_idx) or len(correct) != len(correct_group_idx):
            raise RuntimeError("Topic allocation shape mismatch")

        for gi, e in zip(wrong_group_idx, wrong):
            x = dict(e)
            x["group"] = group_labels[gi]
            assigned.append(x)
        for gi, e in zip(correct_group_idx, correct):
            x = dict(e)
            x["group"] = group_labels[gi]
            assigned.append(x)

    # Final sanity checks.
    by_group_topic = {(g, t): 0 for g in group_labels for t in TOPIC_ORDER}
    wrong_by_group = {g: 0 for g in group_labels}
    for e in assigned:
        g = str(e.get("group"))
        t = str(e.get("topic"))
        by_group_topic[(g, t)] = by_group_topic.get((g, t), 0) + 1
        if not bool(e.get("ai_correct")):
            wrong_by_group[g] = wrong_by_group.get(g, 0) + 1

    for g in group_labels:
        for t in TOPIC_ORDER:
            if by_group_topic.get((g, t), 0) != 1:
                raise RuntimeError(f"Group/topic assignment invalid for {g}/{t}")
        if wrong_by_group.get(g, 0) != wrong_per_topic:
            raise RuntimeError(f"Group wrong-count invalid for {g}: {wrong_by_group.get(g, 0)} != {wrong_per_topic}")

    rng.shuffle(assigned)
    return assigned, group_labels


def select_topic_balanced_pool(
    scored: List[Dict[str, Any]],
    total_n: int,
    n_groups: int,
    n_wrong_target: int,
    rng: random.Random,
    low_max: float,
    mid_max: float,
    min_per_topic: int,
) -> Tuple[List[Dict[str, Any]], List[str], int]:
    if n_groups != len(TOPIC_ORDER):
        raise RuntimeError(f"Topic-balanced fallback expects --n-groups={len(TOPIC_ORDER)}")
    if min_per_topic <= 0:
        raise RuntimeError("min_per_topic must be > 0")
    min_total = min_per_topic * len(TOPIC_ORDER)
    if total_n < min_total:
        raise RuntimeError(f"Need --n >= {min_total} for min_per_topic={min_per_topic}")
    if total_n % n_groups != 0:
        raise RuntimeError("Topic-balanced fallback expects --n divisible by --n-groups")

    per_group = total_n // n_groups
    group_labels = list(string.ascii_uppercase[:n_groups])

    by_topic_wrong: Dict[str, List[Dict[str, Any]]] = {}
    by_topic_correct: Dict[str, List[Dict[str, Any]]] = {}
    topic_quota: Dict[str, int] = {t: min_per_topic for t in TOPIC_ORDER}

    for topic in TOPIC_ORDER:
        wrong_pool = [x for x in scored if x.get("topic") == topic and (not bool(x.get("ai_correct")))]
        correct_pool = [x for x in scored if x.get("topic") == topic and bool(x.get("ai_correct"))]
        if len(wrong_pool) + len(correct_pool) < min_per_topic:
            raise RuntimeError(
                f"Not enough items for topic={topic}: have {len(wrong_pool)+len(correct_pool)}, need >= {min_per_topic}"
            )
        by_topic_wrong[topic] = wrong_pool
        by_topic_correct[topic] = correct_pool

    # Distribute extra slots to topics with larger usable pools and wrong capacity.
    extra = total_n - min_total
    topic_caps = {
        t: len(by_topic_wrong[t]) + len(by_topic_correct[t]) for t in TOPIC_ORDER
    }
    while extra > 0:
        choices = [t for t in TOPIC_ORDER if topic_quota[t] < topic_caps[t]]
        if not choices:
            break
        t = max(choices, key=lambda x: (len(by_topic_wrong[x]), topic_caps[x] - topic_quota[x], -topic_quota[x]))
        topic_quota[t] += 1
        extra -= 1

    max_wrong_possible = sum(min(len(by_topic_wrong[t]), topic_quota[t]) for t in TOPIC_ORDER)
    min_wrong_required = {
        t: max(0, topic_quota[t] - len(by_topic_correct[t])) for t in TOPIC_ORDER
    }
    min_wrong_total = sum(min_wrong_required.values())

    desired_wrong = min(n_wrong_target, max_wrong_possible)
    desired_wrong = max(desired_wrong, min_wrong_total)

    wrong_take: Dict[str, int] = {t: min(len(by_topic_wrong[t]), topic_quota[t]) for t in TOPIC_ORDER}
    cur_wrong = sum(wrong_take.values())
    while cur_wrong > desired_wrong:
        candidates = [t for t in TOPIC_ORDER if wrong_take[t] > min_wrong_required[t]]
        if not candidates:
            break
        t = max(candidates, key=lambda x: wrong_take[x])
        wrong_take[t] -= 1
        cur_wrong -= 1

    selected_by_topic: Dict[str, List[Dict[str, Any]]] = {}
    for topic in TOPIC_ORDER:
        w = wrong_take[topic]
        c = topic_quota[topic] - w
        wrong_sel = pick_balanced_conf(by_topic_wrong[topic], w, rng=rng, low_max=low_max, mid_max=mid_max)
        correct_sel = pick_balanced_conf(by_topic_correct[topic], c, rng=rng, low_max=low_max, mid_max=mid_max)
        if len(wrong_sel) != w or len(correct_sel) != c:
            raise RuntimeError(f"Selection failed for topic={topic}")
        topic_items = list(wrong_sel) + list(correct_sel)
        rng.shuffle(topic_items)
        selected_by_topic[topic] = topic_items

    selected = []
    for topic in TOPIC_ORDER:
        selected.extend(selected_by_topic[topic])

    assigned: List[Dict[str, Any]] = []
    wrong_by_group = {g: 0 for g in group_labels}
    group_sizes = {g: 0 for g in group_labels}
    topic_by_group: Dict[str, Dict[str, int]] = {g: {t: 0 for t in TOPIC_ORDER} for g in group_labels}

    rng.shuffle(selected)
    selected.sort(key=lambda e: (not bool(e.get("ai_correct")),), reverse=True)
    for e in selected:
        topic = str(e.get("topic") or "")
        is_wrong = not bool(e.get("ai_correct"))
        candidates = [g for g in group_labels if group_sizes[g] < per_group]
        if not candidates:
            raise RuntimeError("Group assignment overflow")

        def key(g: str):
            if is_wrong:
                return (group_sizes[g], wrong_by_group[g], topic_by_group[g].get(topic, 0), rng.random())
            return (group_sizes[g], topic_by_group[g].get(topic, 0), wrong_by_group[g], rng.random())

        best = min(candidates, key=key)
        x = dict(e)
        x["group"] = best
        assigned.append(x)
        group_sizes[best] += 1
        topic_by_group[best][topic] = topic_by_group[best].get(topic, 0) + 1
        if is_wrong:
            wrong_by_group[best] += 1

    for g in group_labels:
        if group_sizes[g] != per_group:
            raise RuntimeError(f"Group size mismatch for {g}: {group_sizes[g]} != {per_group}")

    rng.shuffle(assigned)
    return assigned, group_labels, desired_wrong


def find_marker_span(tokens: Sequence[str]) -> Tuple[Optional[Tuple[int, int]], Tuple[int, int]]:
    """Split token indices into title/content spans.

    Returns (title_span, content_span) as (start_idx, end_idx) inclusive-exclusive.
    If markers are not found reliably, returns (None, (0, len(tokens))).
    """

    tl = [str(t).strip().lower() for t in tokens]

    def is_marker_name(i: int, name: str) -> bool:
        if i < 0 or i >= len(tl) or tl[i] != name:
            return False
        return (i > 0 and tl[i - 1] == "<") and (i + 1 < len(tl) and tl[i + 1] == ">")

    def first_idx(name: str, start: int = 0) -> Optional[int]:
        for i in range(start, len(tl)):
            if is_marker_name(i, name):
                return i
        return None

    def next_gt(start: int) -> Optional[int]:
        for i in range(start, len(tl)):
            if tl[i] == ">":
                return i
        return None

    def prev_lt(before: int) -> Optional[int]:
        for i in range(before - 1, -1, -1):
            if tl[i] == "<":
                return i
        return None

    i_title = first_idx("title")
    i_content = first_idx("content", (i_title + 1) if i_title is not None else 0)
    if i_content is None:
        i_content = first_idx("context", (i_title + 1) if i_title is not None else 0)
    i_end = first_idx("end", (i_content + 1) if i_content is not None else 0)

    if i_title is None or i_content is None:
        return None, (0, len(tokens))

    # Title span: after '>' following title marker, until '<' before content marker.
    gt_title = next_gt(i_title)
    start_title = (gt_title + 1) if gt_title is not None else (i_title + 1)
    lt_content = prev_lt(i_content)
    end_title = lt_content if (lt_content is not None and lt_content > start_title) else i_content
    end_title = max(start_title, end_title)

    # Content span: after '>' following content marker, until '<' before end marker (or end-of-seq).
    gt_content = next_gt(i_content)
    start_content = (gt_content + 1) if gt_content is not None else (i_content + 1)
    if i_end is None:
        end_content = len(tokens)
    else:
        lt_end = prev_lt(i_end)
        end_content = lt_end if (lt_end is not None and lt_end > start_content) else i_end
    end_content = max(start_content, end_content)

    return (start_title, end_title), (start_content, end_content)


def build_segments(tokens: Sequence[str], values: Sequence[float], span: Tuple[int, int]) -> List[Dict[str, Any]]:
    """Build UI-friendly word segments by merging subword tokens.

    RoBERTa BPE uses a leading marker (U+0120) to indicate a new word.
    `token_display()` converts that to a leading space, so we can merge by detecting
    whether the displayed token starts with a space.
    """

    start, end = span
    out: List[Dict[str, Any]] = []
    cur_text = ""
    cur_shap = 0.0

    for t, v in zip(tokens[start:end], values[start:end]):
        if is_delim_token(t):
            continue
        disp = token_display(t)
        if not disp.strip():
            continue

        # Heuristic word boundary handling for two common token styles:
        # - RoBERTa BPE: new words often start with a leading space (via U+0120 marker).
        # - SHAP Text masker: tokens may include trailing whitespace for word boundaries.
        if cur_text and disp.startswith(" "):
            out.append({"text": cur_text, "shap": cur_shap})
            cur_text = ""
            cur_shap = 0.0

        cur_text += disp
        cur_shap += safe_float(v)

        if disp[-1].isspace():
            out.append({"text": cur_text, "shap": cur_shap})
            cur_text = ""
            cur_shap = 0.0

    if cur_text:
        out.append({"text": cur_text, "shap": cur_shap})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--split", default=DEFAULT_SPLIT)
    ap.add_argument(
        "--source-plan",
        default="",
        help="Optional curated source plan. Use 'relaxed_fnn' for GossipCop+PolitiFact+HEALTH mix.",
    )
    ap.add_argument("--gossipcop-dataset", default="Jinyan1/GossipCop")
    ap.add_argument("--politifact-dataset", default="Jinyan1/PolitiFact")
    ap.add_argument("--coaid-dataset", default="raquiba/CoAid_title")
    ap.add_argument("--health-fallback-dataset", default="ClassyB/Health_Misinformation")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="xai_ui/stimuli_pool.json")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument(
        "--n-wrong",
        type=int,
        default=0,
        help="Target number of AI-incorrect items in the final pool (0=auto, default ~20%).",
    )
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--scan-limit",
        type=int,
        default=5000,
        help="Max candidate rows to score for selection (model-only; SHAP computed only for selected items).",
    )
    ap.add_argument("--scan-batch-size", type=int, default=16)
    ap.add_argument("--no-balance-truth", action="store_true", help="Do not force 50/50 TRUE/FAKE")
    ap.add_argument(
        "--no-prefer-high-conf-wrong",
        action="store_true",
        help="If set, do not preferentially select high-confidence AI mistakes.",
    )
    ap.add_argument(
        "--real-requires-reuters",
        action="store_true",
        help="If set, keep TRUE items only if their text contains 'reuters'.",
    )
    ap.add_argument("--max-title-words", type=int, default=25)
    ap.add_argument("--max-content-words", type=int, default=180)
    ap.add_argument("--min-content-words", type=int, default=60)
    ap.add_argument("--model-max-length", type=int, default=256)
    ap.add_argument("--max-evals", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument(
        "--fake-threshold",
        type=float,
        default=0.5,
        help="Decision threshold for FAKE class: predict FAKE iff P(FAKE) >= threshold.",
    )
    ap.add_argument("--no-shap", action="store_true")
    ap.add_argument("--n-groups", type=int, default=5, help="Assign selected items to this many rotation groups (A..).")
    ap.add_argument(
        "--conf-low-max",
        type=float,
        default=0.70,
        help="Upper bound for low-confidence bin using max class probability.",
    )
    ap.add_argument(
        "--conf-mid-max",
        type=float,
        default=0.90,
        help="Upper bound for mid-confidence bin; above this is high confidence.",
    )
    ap.add_argument(
        "--study-design",
        action="store_true",
        help="Enable strict 5-topic x 5-group design (25 items; each group: 3 correct + 2 incorrect).",
    )
    ap.add_argument(
        "--allow-topic-fallback",
        action="store_true",
        help="Allow non-topic-balanced fallback selection if strict study-design constraints cannot be met.",
    )
    ap.add_argument(
        "--strict-news-filter",
        action="store_true",
        help="Apply extra title filtering to drop listicle/soft-entertainment phrasing.",
    )
    ap.add_argument(
        "--min-per-topic",
        type=int,
        default=5,
        help="When using topic fallback, keep at least this many items per topic.",
    )
    ap.add_argument(
        "--whitelist-file",
        action="append",
        default=[],
        help="Path to curated candidates file (.csv/.json/.jsonl). Can be passed multiple times.",
    )
    ap.add_argument(
        "--whitelist-source-tag",
        default="WHITELIST_NEWS",
        help="Default source tag used for whitelist rows missing a source value.",
    )
    ap.add_argument(
        "--whitelist-only",
        action="store_true",
        help="Skip dataset crawling and build candidate pool only from --whitelist-file rows.",
    )
    args = ap.parse_args()

    random.seed(args.seed)

    if not (0.0 < float(args.fake_threshold) < 1.0):
        raise RuntimeError("Need 0 < --fake-threshold < 1")

    from datasets import load_dataset  # type: ignore
    import torch  # type: ignore
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

    candidates: List[Dict[str, Any]] = []

    def add_candidate(
        *,
        row_idx: str,
        title: str,
        text: str,
        truth: str,
        topic: Optional[str],
        source_tag: str,
    ) -> None:
        title2 = clean_ws(title)
        text_raw = strip_social_fragments(text)
        if not title2 or not text_raw:
            return
        if is_noisy_text(title2) or is_noisy_text(text_raw):
            return
        if has_boilerplate_noise(title2) or has_boilerplate_noise(text_raw):
            return
        if not is_englishish(title2 + " " + text_raw):
            return
        if len(title2.split(" ")) > args.max_title_words:
            return
        text2 = truncate_words(text_raw, args.max_content_words)
        if not text2:
            return
        if len(text2.split(" ")) < args.min_content_words:
            return
        if is_low_quality_text(text2):
            return
        if CLICKBAIT_RE.search(title2) or CLICKBAIT_RE.search(text2):
            return
        if args.strict_news_filter and STRICT_NON_NEWS_TITLE_RE.search(title2):
            return
        if has_excessive_uppercase(title2):
            return

        truth2 = normalize_truth(truth)
        topic2 = topic or detect_topic(title2, text2)
        if topic2 is None:
            return

        # Reduce entertainment/gossip material in the research pool.
        if is_entertainment_like(title2, text2):
            return
        if RELATIONSHIP_RE.search(f"{title2} {text2}"):
            return
        if (
            source_tag == "FakeNewsNet/GossipCop"
            and looks_person_name_heavy(title2)
            and (not has_topic_signal(topic2, title2, text2))
        ):
            return

        if args.real_requires_reuters and truth2 == "TRUE" and ("reuters" not in text_raw.lower()):
            return

        candidates.append(
            {
                "row_idx": row_idx,
                "title": title2,
                "content": text2,
                "truth_label": truth2,
                "topic": topic2,
                "source": source_tag,
            }
        )

    source_plan = str(args.source_plan).strip().lower()

    if args.whitelist_only:
        if not args.whitelist_file:
            raise RuntimeError("--whitelist-only requires at least one --whitelist-file")
    elif source_plan == "relaxed_fnn":
        # 1) GossipCop as main pool (map mostly to SOCIETY, allow SCI_TECH/BUSINESS by keyword).
        ds_gc = load_dataset(args.gossipcop_dataset)
        for split_name in ds_gc.keys():
            split_truth = normalize_split_truth(str(split_name))
            if split_truth is None:
                continue
            dsi = ds_gc[split_name]
            for i in range(len(dsi)):
                row = dsi[i]
                title = get_first_nonempty(row, ["title", "headline"]) 
                text = get_first_nonempty(row, ["text", "content", "description"])
                if not title or not text:
                    continue
                raw_topic = detect_topic(title, text) or "SOCIETY"
                topic = raw_topic if raw_topic in {"SCI_TECH", "BUSINESS", "SOCIETY"} else "SOCIETY"
                add_candidate(
                    row_idx=f"gc_{split_name}_{i}",
                    title=title,
                    text=text,
                    truth=split_truth,
                    topic=topic,
                    source_tag="FakeNewsNet/GossipCop",
                )

        # 2) PolitiFact as POLITICS supplement.
        ds_pf = load_dataset(args.politifact_dataset)
        for split_name in ds_pf.keys():
            split_truth = normalize_split_truth(str(split_name))
            if split_truth is None:
                continue
            dsi = ds_pf[split_name]
            for i in range(len(dsi)):
                row = dsi[i]
                title = get_first_nonempty(row, ["title", "headline"])
                text = get_first_nonempty(row, ["text", "content", "description"])
                if not title or not text:
                    continue
                add_candidate(
                    row_idx=f"pf_{split_name}_{i}",
                    title=title,
                    text=text,
                    truth=split_truth,
                    topic="POLITICS",
                    source_tag="FakeNewsNet/PolitiFact",
                )

        # 3) HEALTH supplement: try CoAID title dataset first, fallback to accessible health set.
        health_loaded = False
        try:
            ds_h = load_dataset(args.coaid_dataset)
            for split_name in ds_h.keys():
                dsi = ds_h[split_name]
                cols_h = set(dsi.column_names)
                label_keys = ["label", "real", "class", "True or Misinformation"]
                title_keys = ["title", "headline", "Statement"]
                text_keys = ["text", "content", "description", "Reason", "summary", "body"]
                for i in range(len(dsi)):
                    row = dsi[i]
                    title = get_first_nonempty(row, title_keys)
                    text = get_first_nonempty(row, text_keys)
                    if not text:
                        text = title
                    truth_raw = None
                    for lk in label_keys:
                        if lk in cols_h:
                            truth_raw = row.get(lk)
                            break
                    if truth_raw is None:
                        continue
                    add_candidate(
                        row_idx=f"coaid_{split_name}_{i}",
                        title=title,
                        text=text,
                        truth=str(truth_raw),
                        topic="HEALTH",
                        source_tag="CoAID",
                    )
            health_loaded = True
        except Exception as e:
            print(f"CoAID load failed ({args.coaid_dataset}): {e}")

        if not health_loaded:
            ds_hf = load_dataset(args.health_fallback_dataset)
            for split_name in ds_hf.keys():
                dsi = ds_hf[split_name]
                for i in range(len(dsi)):
                    row = dsi[i]
                    title = get_first_nonempty(row, ["Statement", "title", "headline"])
                    reason = get_first_nonempty(row, ["Reason", "text", "content", "description"])
                    text = f"{title}. {reason}".strip()
                    truth_raw = row.get("True or Misinformation", row.get("label", row.get("real", "")))
                    add_candidate(
                        row_idx=f"hfb_{split_name}_{i}",
                        title=title,
                        text=text,
                        truth=str(truth_raw),
                        topic="HEALTH",
                        source_tag="HEALTH_FALLBACK",
                    )
            print(f"Using fallback health dataset: {args.health_fallback_dataset}")

    elif source_plan == "quality_mix":
        # 1) Base general-news pool from default dataset, restricted to non-politics/non-health topics.
        ds_base = load_dataset(args.dataset, split=args.split)
        cols = set(ds_base.column_names)
        title_col = "title" if "title" in cols else None
        text_col = "text" if "text" in cols else ("content" if "content" in cols else None)
        label_col = "label" if "label" in cols else ("real" if "real" in cols else None)
        if not title_col or not text_col or not label_col:
            raise RuntimeError(f"Unsupported base dataset schema columns={sorted(cols)}")

        for i in range(len(ds_base)):
            row = ds_base[i]
            title = str(row.get(title_col, ""))
            text = str(row.get(text_col, ""))
            topic = detect_topic(title, text)
            if topic not in {"SCI_TECH", "BUSINESS", "SOCIETY"}:
                continue
            add_candidate(
                row_idx=f"base_{i}",
                title=title,
                text=text,
                truth=str(row.get(label_col, "")),
                topic=topic,
                source_tag="BASE_NEWS",
            )

        # 2) PolitiFact as POLITICS source.
        ds_pf = load_dataset(args.politifact_dataset)
        for split_name in ds_pf.keys():
            split_truth = normalize_split_truth(str(split_name))
            if split_truth is None:
                continue
            dsi = ds_pf[split_name]
            for i in range(len(dsi)):
                row = dsi[i]
                title = get_first_nonempty(row, ["title", "headline"])
                text = get_first_nonempty(row, ["text", "content", "description"])
                if not title or not text:
                    continue
                add_candidate(
                    row_idx=f"pf_{split_name}_{i}",
                    title=title,
                    text=text,
                    truth=split_truth,
                    topic="POLITICS",
                    source_tag="FakeNewsNet/PolitiFact",
                )

        # 3) HEALTH source (CoAID -> fallback).
        health_loaded = False
        try:
            ds_h = load_dataset(args.coaid_dataset)
            for split_name in ds_h.keys():
                dsi = ds_h[split_name]
                cols_h = set(dsi.column_names)
                label_keys = ["label", "real", "class", "True or Misinformation"]
                title_keys = ["title", "headline", "Statement"]
                text_keys = ["text", "content", "description", "Reason", "summary", "body"]
                for i in range(len(dsi)):
                    row = dsi[i]
                    title = get_first_nonempty(row, title_keys)
                    text = get_first_nonempty(row, text_keys)
                    if not text:
                        text = title
                    truth_raw = None
                    for lk in label_keys:
                        if lk in cols_h:
                            truth_raw = row.get(lk)
                            break
                    if truth_raw is None:
                        continue
                    add_candidate(
                        row_idx=f"coaid_{split_name}_{i}",
                        title=title,
                        text=text,
                        truth=str(truth_raw),
                        topic="HEALTH",
                        source_tag="CoAID",
                    )
            health_loaded = True
        except Exception as e:
            print(f"CoAID load failed ({args.coaid_dataset}): {e}")

        if not health_loaded:
            ds_hf = load_dataset(args.health_fallback_dataset)
            for split_name in ds_hf.keys():
                dsi = ds_hf[split_name]
                for i in range(len(dsi)):
                    row = dsi[i]
                    title = get_first_nonempty(row, ["Statement", "title", "headline"])
                    reason = get_first_nonempty(row, ["Reason", "text", "content", "description"])
                    text = f"{title}. {reason}".strip()
                    truth_raw = row.get("True or Misinformation", row.get("label", row.get("real", "")))
                    add_candidate(
                        row_idx=f"hfb_{split_name}_{i}",
                        title=title,
                        text=text,
                        truth=str(truth_raw),
                        topic="HEALTH",
                        source_tag="HEALTH_FALLBACK",
                    )
            print(f"Using fallback health dataset: {args.health_fallback_dataset}")

    else:
        ds = load_dataset(args.dataset, split=args.split)
        cols = set(ds.column_names)

        # Heuristics for common column names.
        title_col = "title" if "title" in cols else None
        text_col = "text" if "text" in cols else ("content" if "content" in cols else None)
        label_col = "label" if "label" in cols else ("real" if "real" in cols else None)
        if not title_col or not text_col or not label_col:
            raise RuntimeError(f"Unsupported dataset schema columns={sorted(cols)}")

        # Collect candidates with manageable reading length.
        for i in range(len(ds)):
            row = ds[i]
            add_candidate(
                row_idx=str(i),
                title=str(row.get(title_col, "")),
                text=str(row.get(text_col, "")),
                truth=str(row.get(label_col, "")),
                topic=None,
                source_tag=args.dataset,
            )

    if args.whitelist_file:
        wl_added = 0
        for wp in args.whitelist_file:
            rows = load_whitelist_rows(wp)
            for i, row in enumerate(rows):
                title = get_first_nonempty(row, ["title", "headline"])
                text = get_first_nonempty(row, ["content", "text", "description", "body"])
                truth = get_first_nonempty(row, ["truth_label", "truth", "label", "real"])
                topic_raw = get_first_nonempty(row, ["topic"])
                topic = topic_raw.upper() if topic_raw else None
                if topic and topic not in TOPIC_ORDER:
                    topic = None
                source = get_first_nonempty(row, ["source", "source_tag", "publisher"]) or str(args.whitelist_source_tag)
                if not title or not text or not truth:
                    continue
                add_candidate(
                    row_idx=f"wl_{os.path.basename(str(wp))}_{i}",
                    title=title,
                    text=text,
                    truth=truth,
                    topic=topic,
                    source_tag=source,
                )
                wl_added += 1
        print(f"Whitelist candidates added: {wl_added}")

    if not candidates:
        raise RuntimeError("No candidates after filtering; adjust thresholds.")

    src_counts: Dict[str, int] = {}
    topic_counts_pre: Dict[str, int] = {t: 0 for t in TOPIC_ORDER}
    for c in candidates:
        src = str(c.get("source") or "unknown")
        src_counts[src] = src_counts.get(src, 0) + 1
        t = str(c.get("topic") or "")
        if t in topic_counts_pre:
            topic_counts_pre[t] += 1
    print("Candidate sources:", ", ".join(f"{k}={v}" for k, v in sorted(src_counts.items())))
    print("Candidate topics:", ", ".join(f"{t}={topic_counts_pre[t]}" for t in TOPIC_ORDER))

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.eval()

    labels = find_label_names(model.config.id2label)
    fake_index = pick_fake_index(labels)

    balance_truth = not args.no_balance_truth
    total_n = int(args.n)
    if total_n <= 0:
        raise RuntimeError("--n must be > 0")

    n_wrong_target = int(args.n_wrong)
    if n_wrong_target <= 0:
        n_wrong_target = max(5, total_n // 5)
    n_wrong_target = min(n_wrong_target, total_n)

    def batched(seq: List[Dict[str, Any]], size: int):
        for j in range(0, len(seq), size):
            yield seq[j : j + size]

    def infer_batch(prompts: List[str]) -> List[Dict[str, float]]:
        with torch.no_grad():
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max(64, int(args.model_max_length)),
            )
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        out: List[Dict[str, float]] = []
        for row in probs:
            out.append({labels[i]: float(row[i]) for i in range(len(labels))})
        return out

    # Score candidates with the model to find enough AI-incorrect items.
    scan = list(candidates)
    random.shuffle(scan)
    if args.scan_limit and args.scan_limit > 0:
        scan = scan[: min(int(args.scan_limit), len(scan))]

    buckets: Dict[Tuple[str, bool], List[Dict[str, Any]]] = {
        ("TRUE", True): [],
        ("TRUE", False): [],
        ("FAKE", True): [],
        ("FAKE", False): [],
    }
    scored_entries: List[Dict[str, Any]] = []

    scan_bs = max(1, int(args.scan_batch_size))
    for chunk in batched(scan, scan_bs):
        prompts = [make_prompt(c["title"], c["content"]) for c in chunk]
        probs_list = infer_batch(prompts)
        for c, prompt, probs in zip(chunk, prompts, probs_list):
            pred_label = infer_pred_label(probs, fake_threshold=float(args.fake_threshold))
            correct = bool(pred_label == c["truth_label"])
            conf = max(probs.values()) if probs else float("nan")
            entry = dict(c)
            entry["prompt"] = prompt
            entry["probs"] = probs
            entry["pred_label"] = pred_label
            entry["ai_correct"] = correct
            entry["ai_conf"] = conf
            buckets[(c["truth_label"], correct)].append(entry)
            scored_entries.append(entry)

    wrong_true = buckets[("TRUE", False)]
    wrong_fake = buckets[("FAKE", False)]
    correct_true = buckets[("TRUE", True)]
    correct_fake = buckets[("FAKE", True)]

    print(
        "Scanned",
        len(scan),
        "candidates ->",
        f"wrong TRUE={len(wrong_true)}, wrong FAKE={len(wrong_fake)}, correct TRUE={len(correct_true)}, correct FAKE={len(correct_fake)}",
    )

    topic_counts: Dict[str, int] = {t: 0 for t in TOPIC_ORDER}
    topic_wrong: Dict[str, int] = {t: 0 for t in TOPIC_ORDER}
    for e in scored_entries:
        t = str(e.get("topic") or "")
        if t in topic_counts:
            topic_counts[t] += 1
            if not bool(e.get("ai_correct")):
                topic_wrong[t] += 1
    print("Topic availability:", ", ".join(f"{t}={topic_counts[t]} (wrong={topic_wrong[t]})" for t in TOPIC_ORDER))

    if not (0.0 < float(args.conf_low_max) < float(args.conf_mid_max) < 1.0):
        raise RuntimeError("Need 0 < --conf-low-max < --conf-mid-max < 1")

    pick_rng = random.Random(args.seed + 17)
    picked: List[Dict[str, Any]] = []
    strict_group_labels: Optional[List[str]] = None
    wrong_true_target = 0
    wrong_fake_target = 0
    correct_true_target = 0
    correct_fake_target = 0

    if args.study_design:
        try:
            picked, strict_group_labels = select_balanced_study_pool(
                scored_entries,
                total_n=total_n,
                n_groups=int(args.n_groups),
                n_wrong_target=n_wrong_target,
                rng=pick_rng,
                low_max=float(args.conf_low_max),
                mid_max=float(args.conf_mid_max),
            )
            print("Using strict study design selection (topic-balanced + group-balanced).")
        except RuntimeError as e:
            if not args.allow_topic_fallback:
                raise
            print(f"Strict study design failed; falling back to generic selection: {e}")
            try:
                picked, strict_group_labels, capped_wrong = select_topic_balanced_pool(
                    scored_entries,
                    total_n=total_n,
                    n_groups=int(args.n_groups),
                    n_wrong_target=n_wrong_target,
                    rng=pick_rng,
                    low_max=float(args.conf_low_max),
                    mid_max=float(args.conf_mid_max),
                    min_per_topic=max(1, int(args.min_per_topic)),
                )
                print(
                    "Using topic-balanced fallback selection "
                    f"(target wrong={n_wrong_target}, achievable wrong={capped_wrong})."
                )
            except RuntimeError as e2:
                print(f"Topic-balanced fallback unavailable; using generic selection: {e2}")

    if not picked:
        if balance_truth:
            n_true_target = total_n // 2
            n_fake_target = total_n - n_true_target
        else:
            n_true_target = min(total_n, len(wrong_true) + len(correct_true))
            n_fake_target = total_n - n_true_target

        # Allocate AI-incorrect items across truth labels (best-effort).
        wrong_true_target = min(len(wrong_true), n_true_target, n_wrong_target // 2)
        wrong_fake_target = min(len(wrong_fake), n_fake_target, n_wrong_target - wrong_true_target)
        remaining_wrong = n_wrong_target - (wrong_true_target + wrong_fake_target)
        while remaining_wrong > 0:
            # Fill whichever has remaining capacity.
            if wrong_true_target < n_true_target and wrong_true_target < len(wrong_true):
                wrong_true_target += 1
                remaining_wrong -= 1
                continue
            if wrong_fake_target < n_fake_target and wrong_fake_target < len(wrong_fake):
                wrong_fake_target += 1
                remaining_wrong -= 1
                continue
            break

        correct_true_target = n_true_target - wrong_true_target
        correct_fake_target = n_fake_target - wrong_fake_target

        if correct_true_target > len(correct_true) or correct_fake_target > len(correct_fake):
            raise RuntimeError(
                "Not enough candidates to fill pool with requested constraints. "
                f"Need correct TRUE={correct_true_target}/{len(correct_true)}, correct FAKE={correct_fake_target}/{len(correct_fake)}. "
                "Try increasing --scan-limit or loosening filters."
            )

    def pick_any(lst: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        return pick_balanced_conf(
            lst,
            k,
            rng=pick_rng,
            low_max=float(args.conf_low_max),
            mid_max=float(args.conf_mid_max),
        )

    if not picked:
        picked = (
            pick_any(wrong_true, wrong_true_target)
            + pick_any(wrong_fake, wrong_fake_target)
            + pick_any(correct_true, correct_true_target)
            + pick_any(correct_fake, correct_fake_target)
        )

        if len(picked) != (wrong_true_target + wrong_fake_target + correct_true_target + correct_fake_target):
            raise RuntimeError("Internal selection error: picked wrong count")

        random.shuffle(picked)

    explainer = None
    if not args.no_shap:
        import shap  # type: ignore

        def model_logit(texts: Sequence[str]):
            with torch.no_grad():
                enc = tokenizer(
                    list(texts),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max(64, int(args.model_max_length)),
                )
                logits = model(**enc).logits
                out = logits[:, fake_index].detach().cpu().numpy()
            return out

        masker = _make_shap_text_masker(tokenizer)
        explainer = shap.Explainer(model_logit, masker)

    items_out = []
    for k, c in enumerate(picked):
        title = c["title"]
        content = c["content"]
        truth = c["truth_label"]
        prompt = str(c.get("prompt") or make_prompt(title, content))
        probs = c.get("probs") or {}
        pred_label = str(c.get("pred_label") or infer_pred_label(probs, fake_threshold=float(args.fake_threshold)))
        ai_correct = bool(c.get("ai_correct") if ("ai_correct" in c) else (pred_label == truth))
        ai_conf = confidence_of_probs(probs)
        ai_conf_bin = confidence_bin(ai_conf, low_max=float(args.conf_low_max), mid_max=float(args.conf_mid_max))

        xai = None
        if explainer is not None:
            exp = explainer([prompt], max_evals=args.max_evals, batch_size=args.batch_size)
            toks = [str(t) for t in exp.data[0]]
            vals = [safe_float(v) for v in exp.values[0]]
            base_value = safe_float(exp.base_values[0])

            title_span, content_span = find_marker_span(toks)
            if title_span is None:
                title_segments = []
                content_segments = build_segments(toks, vals, content_span)
            else:
                title_segments = build_segments(toks, vals, title_span)
                content_segments = build_segments(toks, vals, content_span)

            xai = {
                "target": "logit(P(FAKE))",
                "base_value": base_value,
                "title_segments": title_segments,
                "content_segments": content_segments,
            }

        items_out.append({
            "id": f"{args.dataset.replace('/', '_')}_{args.split}_{c['row_idx']}",
            "source": c.get("source") or args.dataset,
            "topic": c.get("topic"),
            "group": c.get("group"),
            "title": title,
            "content": content,
            "truth": {"label": truth},
            "ai": {
                "label": pred_label,
                "probs": probs,
                "correct": ai_correct,
                "confidence": ai_conf,
                "conf_bin": ai_conf_bin,
            },
            "xai": xai,
        })

        if (k + 1) % 5 == 0:
            print(f"[{k+1}/{len(picked)}] processed")

    if strict_group_labels is None:
        group_labels = assign_item_groups(
            items_out,
            n_groups=int(args.n_groups),
            seed=int(args.seed),
            low_max=float(args.conf_low_max),
            mid_max=float(args.conf_mid_max),
        )
    else:
        group_labels = strict_group_labels
    condition_lists = build_latin_square_lists(group_labels, CONDITION_ORDER)

    conf_counts = {"low": 0, "mid": 0, "high": 0}
    corr_counts = {True: 0, False: 0}
    topic_final = {t: 0 for t in TOPIC_ORDER}
    group_final: Dict[str, int] = {g: 0 for g in group_labels}
    group_wrong_final: Dict[str, int] = {g: 0 for g in group_labels}
    for it in items_out:
        b = str(it.get("ai", {}).get("conf_bin", "mid"))
        if b in conf_counts:
            conf_counts[b] += 1
        is_corr = bool(it.get("ai", {}).get("correct", False))
        corr_counts[is_corr] += 1
        t = str(it.get("topic") or "")
        if t in topic_final:
            topic_final[t] += 1
        g = str(it.get("group") or "")
        if g in group_final:
            group_final[g] += 1
            if not is_corr:
                group_wrong_final[g] += 1
    print(
        "Selected pool summary:",
        f"AI-correct={corr_counts[True]}, AI-incorrect={corr_counts[False]},",
        f"conf bins low/mid/high={conf_counts['low']}/{conf_counts['mid']}/{conf_counts['high']}",
    )
    print("Topic final:", ", ".join(f"{t}={topic_final[t]}" for t in TOPIC_ORDER))
    print(
        "Group final:",
        ", ".join(f"{g}=n{group_final[g]}/wrong{group_wrong_final[g]}" for g in group_labels),
    )

    out_payload = {
        "schema_version": 1,
        "meta": {
            "built_at": now_iso(),
            "dataset": (f"source_plan:{args.source_plan}" if str(args.source_plan).strip() else args.dataset),
            "split": args.split,
            "model": args.model,
            "seed": args.seed,
            "n": len(items_out),
            "n_groups": int(args.n_groups),
            "study_design": bool(args.study_design),
            "min_per_topic": int(args.min_per_topic),
            "confidence_bins": {
                "low_max": float(args.conf_low_max),
                "mid_max": float(args.conf_mid_max),
            },
            "decision": {
                "fake_threshold": float(args.fake_threshold),
            },
            "shap": {
                "enabled": (not args.no_shap),
                "max_evals": args.max_evals,
                "batch_size": args.batch_size,
                "target": "logit(P(FAKE))",
            },
            "design": {
                "group_labels": group_labels,
                "topic_order": TOPIC_ORDER,
                "condition_order": CONDITION_ORDER,
                "condition_lists": condition_lists,
            },
        },
        "items": items_out,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as outf:
        json.dump(out_payload, outf, ensure_ascii=True, indent=2)
        outf.write("\n")

    print(f"Wrote {args.out} with {len(items_out)} items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
