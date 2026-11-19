#!/usr/bin/env python3
"""Quick model bake-off for fake-news classification.

Evaluates one or more Hugging Face sequence-classification checkpoints on an
items JSON (same schema as xai_ui stimuli files), then reports:
- prediction balance (TRUE/FAKE counts)
- AI-wrong count
- topic-level coupling stats
- best threshold for P(FAKE) if calibration is enabled
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def make_prompt(title: str, content: str) -> str:
    return f"<title>{title}<content>{content}<end>"


def find_label_names(id2label: Dict[int, str]) -> List[str]:
    n = len(id2label)
    return [str(id2label[i]) for i in range(n)]


def pick_fake_index(labels: Sequence[str]) -> int:
    upper = [str(x).upper() for x in labels]
    if "FAKE" in upper:
        return upper.index("FAKE")
    if "FALSE" in upper:
        return upper.index("FALSE")
    if len(labels) == 2:
        return 1
    raise RuntimeError(f"Cannot infer FAKE class from labels={labels}")


def evaluate_with_threshold(items: List[Dict[str, Any]], probs_fake: List[float], threshold: float) -> Dict[str, Any]:
    pred = ["FAKE" if p >= threshold else "TRUE" for p in probs_fake]
    truth = [str((it.get("truth") or {}).get("label", "TRUE")) for it in items]
    topic = [str(it.get("topic") or "UNKNOWN") for it in items]

    pred_counts = Counter(pred)
    truth_counts = Counter(truth)
    wrong = sum(1 for p, t in zip(pred, truth) if p != t)
    correct = len(items) - wrong

    by_topic: Dict[str, Counter] = defaultdict(Counter)
    for p, t, top in zip(pred, truth, topic):
        c = by_topic[top]
        c["n"] += 1
        c[f"truth_{t}"] += 1
        c[f"pred_{p}"] += 1
        c[f"corr_{p == t}"] += 1

    return {
        "threshold": threshold,
        "n": len(items),
        "wrong": wrong,
        "correct": correct,
        "pred_counts": dict(pred_counts),
        "truth_counts": dict(truth_counts),
        "by_topic": {k: dict(v) for k, v in sorted(by_topic.items())},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, help="Path to items JSON (with .items list)")
    ap.add_argument(
        "--models",
        nargs="+",
        default=[
            "hamzab/roberta-fake-news-classification",
            "mrm8488/bert-tiny-finetuned-fake-news-detection",
        ],
    )
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--target-wrong", type=int, default=8)
    ap.add_argument("--target-pred-true", type=int, default=8)
    ap.add_argument("--out", default="xai_ui/model_bakeoff_report.json")
    args = ap.parse_args()

    payload = json.loads(Path(args.items).read_text())
    items = payload.get("items", payload)
    if not isinstance(items, list) or not items:
        raise RuntimeError("Invalid --items payload (no list found)")

    prompts = [make_prompt(str(it.get("title", "")), str(it.get("content", ""))) for it in items]

    import torch  # type: ignore
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

    report: Dict[str, Any] = {"items": args.items, "n": len(items), "results": []}

    for model_id in args.models:
        print(f"\n=== Evaluating {model_id} ===")
        try:
            tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
            mdl.eval()
            labels = find_label_names(mdl.config.id2label)
            fake_idx = pick_fake_index(labels)

            probs_fake: List[float] = []
            bs = max(1, int(args.batch_size))
            with torch.no_grad():
                for i in range(0, len(prompts), bs):
                    chunk = prompts[i : i + bs]
                    enc = tok(
                        chunk,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max(64, int(args.max_length)),
                    )
                    logits = mdl(**enc).logits
                    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                    for row in probs:
                        probs_fake.append(float(row[fake_idx]))

            # Baseline argmax threshold for binary with fake_idx convention.
            baseline = evaluate_with_threshold(items, probs_fake, threshold=0.5)

            # Threshold sweep to match study constraints better.
            best = None
            for t_i in range(5, 96):
                thr = t_i / 100.0
                cur = evaluate_with_threshold(items, probs_fake, threshold=thr)
                pred_true = int(cur["pred_counts"].get("TRUE", 0))
                score = (
                    abs(int(cur["wrong"]) - int(args.target_wrong)),
                    abs(pred_true - int(args.target_pred_true)),
                )
                if best is None or score < best[0]:
                    best = (score, cur)

            result = {
                "model": model_id,
                "labels": labels,
                "fake_index": fake_idx,
                "baseline": baseline,
                "best_calibrated": best[1] if best is not None else None,
            }
            report["results"].append(result)
            b = baseline
            c = result["best_calibrated"]
            print(
                f"baseline wrong={b['wrong']} pred_TRUE={b['pred_counts'].get('TRUE', 0)} pred_FAKE={b['pred_counts'].get('FAKE', 0)}"
            )
            print(
                f"calibrated thr={c['threshold']:.2f} wrong={c['wrong']} pred_TRUE={c['pred_counts'].get('TRUE', 0)} pred_FAKE={c['pred_counts'].get('FAKE', 0)}"
            )

        except Exception as e:
            report["results"].append({"model": model_id, "error": str(e)})
            print(f"failed: {e}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
