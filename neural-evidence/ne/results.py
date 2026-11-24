"""Saving non-transformer runs in the shared layout, and summarising all runs.

Layout: results/<exp>/<name>_seed<seed>/{config.json, metrics_<evalset>.json, done.json}
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd

from . import metrics


def run_dir(results_dir, exp, name, seed):
    return os.path.join(results_dir, exp, f"{name}_seed{seed}")


def is_done(results_dir, exp, name, seed):
    return os.path.exists(os.path.join(run_dir(results_dir, exp, name, seed), "done.json"))


def save_run(results_dir, exp, name, seed, config: dict, preds_by_set: dict, labels_by_set: dict,
             extra: dict = None, rows_by_set: dict = None):
    """preds/labels: evalset -> int arrays. rows_by_set (optional) enables the *_clean FEVER subset."""
    d = run_dir(results_dir, exp, name, seed)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.json"), "w") as fh:
        json.dump({"exp": exp, "name": name, "seed": seed, **config}, fh, indent=2)
    headline = {}
    for k, pred in preds_by_set.items():
        y = np.asarray(labels_by_set[k])
        pred = np.asarray(pred)
        m = metrics.evaluate(y, pred)
        metrics.save(m, os.path.join(d, f"metrics_{k}.json"))
        print(metrics.format_report(m, f"[{exp}/{name} seed {seed}] {k}"))
        headline[k] = {"macro_f1": m["macro_f1"], "refutes_recall": m["refutes_recall"]}
        rows = (rows_by_set or {}).get(k)
        if rows and "text_in_train" in rows[0]:
            keep = np.array([not r["text_in_train"] for r in rows])
            metrics.save(metrics.evaluate(y[keep], pred[keep]), os.path.join(d, f"metrics_{k}_clean.json"))
    with open(os.path.join(d, "done.json"), "w") as fh:
        json.dump({"headline": headline, **(extra or {})}, fh, indent=2)


def collect(results_dir: str) -> pd.DataFrame:
    """One row per (run, eval set) with the main metrics."""
    rows = []
    for mpath in glob.glob(os.path.join(results_dir, "*", "*", "metrics_*.json")):
        d = os.path.dirname(mpath)
        exp = os.path.basename(os.path.dirname(d))
        m_run = re.match(r"(.+)_seed(\d+)$", os.path.basename(d))
        if not m_run:
            continue
        done = json.load(open(os.path.join(d, "done.json"))) if os.path.exists(os.path.join(d, "done.json")) else {}
        m = json.load(open(mpath))
        rows.append({
            "exp": exp, "run": m_run.group(1), "seed": int(m_run.group(2)),
            "eval_set": os.path.basename(mpath)[len("metrics_"):-len(".json")],
            "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
            "S_f1": m["per_class"]["SUPPORTS"]["f1"], "R_f1": m["per_class"]["REFUTES"]["f1"],
            "NEI_f1": m["per_class"]["NOT ENOUGH INFO"]["f1"], "R_recall": m["refutes_recall"],
            "n": m["n"], "train_min": round(done.get("train_seconds", 0) / 60, 1) or None,
            "peak_vram_gb": done.get("peak_vram_gb"),
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std over seeds for each (exp, run, eval set)."""
    if df.empty:
        return df
    g = df.groupby(["exp", "run", "eval_set"])
    out = g.agg(seeds=("seed", "nunique"),
                macro_f1=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
                accuracy=("accuracy", "mean"), R_recall=("R_recall", "mean"), R_recall_std=("R_recall", "std"),
                S_f1=("S_f1", "mean"), R_f1=("R_f1", "mean"), NEI_f1=("NEI_f1", "mean"),
                train_min=("train_min", "mean"), peak_vram_gb=("peak_vram_gb", "max")).reset_index()
    return out.round(4).fillna({"macro_f1_std": 0, "R_recall_std": 0})
