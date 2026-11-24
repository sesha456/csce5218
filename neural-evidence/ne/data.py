"""Loading processed JSONL, evidence-mode selection and the eval-set registry."""
import json
import os
import random

EVIDENCE_MODES = ("gold", "retrieved", "none")


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def write_jsonl(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def evidence_of(row, mode: str):
    """Evidence text for an E4 condition: 'gold', 'retrieved' or 'none' (claim only)."""
    if mode == "gold":
        return row["evidence_gold"]
    if mode == "retrieved":
        return row["evidence_retrieved"]
    if mode == "none":
        return None
    raise ValueError(f"unknown evidence mode {mode!r}")


def fixed_subset(rows, n: int, seed: int = 0):
    """Same n examples for every model/seed (subset seed is fixed on purpose)."""
    if not n or n >= len(rows):
        return rows
    idx = sorted(random.Random(seed).sample(range(len(rows)), n))
    return [rows[i] for i in idx]


def eval_sets(proc_dir: str) -> dict:
    """Name -> path for every evaluation file that exists under proc_dir."""
    cands = {
        "fever_val": "fever/val.jsonl",
        "fever_test": "fever/test.jsonl",
        "scifact_test": "scifact/test.jsonl",
        "fever_test_distract": "robustness/fever_test_distract.jsonl",
        "fever_test_distract_sent": "robustness/fever_test_distract_sent.jsonl",
        "fever_test_para": "robustness/fever_test_para.jsonl",
    }
    return {k: os.path.join(proc_dir, v) for k, v in cands.items() if os.path.exists(os.path.join(proc_dir, v))}
