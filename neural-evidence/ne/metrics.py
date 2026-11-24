"""Shared evaluation: every experiment reports through these functions."""
import json

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from . import LABELS

SHORT = {"SUPPORTS": "S", "REFUTES": "R", "NOT ENOUGH INFO": "NEI"}


def evaluate(y_true, y_pred) -> dict:
    """Macro-F1 (primary), accuracy, per-class P/R/F1 and the confusion matrix."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(len(LABELS)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "macro_f1": float(f.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": {
            name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, name in enumerate(LABELS)
        },
        "refutes_recall": float(r[1]),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n": int(len(y_true)),
    }


def hf_compute_metrics(eval_pred):
    """Drop-in `compute_metrics` for transformers.Trainer (flat scalars only)."""
    logits, labels = eval_pred
    m = evaluate(labels, np.argmax(logits, axis=-1))
    flat = {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"]}
    for name, v in m["per_class"].items():
        for key in ("precision", "recall", "f1"):
            flat[f"{SHORT[name]}_{key}"] = v[key]
    return flat


def aggregate_seeds(results: list) -> dict:
    """Mean and std of the scalar metrics over runs with different seeds."""
    keys = ["macro_f1", "accuracy", "refutes_recall"]
    out = {"n_seeds": len(results)}
    for k in keys:
        vals = np.array([r[k] for r in results], dtype=float)
        out[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
    return out


def format_report(m: dict, title: str = "") -> str:
    """Plain-text table for logs and notebooks."""
    lines = [title] if title else []
    lines.append(f"macro-F1 {m['macro_f1']:.4f} | accuracy {m['accuracy']:.4f} | n={m['n']}")
    lines.append(f"{'class':<16}{'P':>8}{'R':>8}{'F1':>8}{'support':>9}")
    for name, v in m["per_class"].items():
        lines.append(f"{name:<16}{v['precision']:>8.3f}{v['recall']:>8.3f}{v['f1']:>8.3f}{v['support']:>9}")
    lines.append("confusion (rows=true, cols=pred; S, R, NEI):")
    lines.extend("  " + " ".join(f"{x:>7}" for x in row) for row in m["confusion_matrix"])
    return "\n".join(lines)


def save(m: dict, path: str):
    with open(path, "w") as fh:
        json.dump(m, fh, indent=2)
