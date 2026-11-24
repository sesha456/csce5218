"""Check whether the E6 distractor collapse is a construction artefact, and build a fairer set.

python scripts/diagnose_distractors.py --proc <data/processed> --results <results>

1. Evidence shape per label in FEVER train: NEI evidence is always BM25-retrieved, while
   SUPPORTS/REFUTES evidence is the annotated gold set. Sentence and page counts show how
   easy it is to tell them apart without reading the claim.
2. For every finished run scored on fever_test_distract: prediction distribution and
   per-class recall on the clean vs distractor test set.
3. Writes robustness/fever_test_distract_sent.jsonl (one single gold sentence from another
   claim appended), if it does not exist yet.
"""
import argparse
import glob
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ne import LABELS  # noqa: E402
from ne.data import read_jsonl, write_jsonl  # noqa: E402
from ne.perturb import add_sentence_distractors  # noqa: E402


def evidence_shape(rows):
    print("\n== 1. Evidence shape by label (FEVER train) ==")
    by = {}
    for r in rows:
        ids = r["gold_ids"] if r.get("evidence_gold_source") == "gold" else r["retrieved_ids"]
        by.setdefault(r["label_name"], []).append((len(ids), len({p for p, _ in ids}), len(r["evidence_gold"].split())))
    print(f"{'label':<18}{'n':>8}{'sents':>8}{'pages':>8}{'words':>8}{'%2+ sents':>11}")
    for lab in LABELS:
        a = np.array(by.get(lab, [(0, 0, 0)]))
        print(f"{lab:<18}{len(a):>8}{a[:, 0].mean():>8.2f}{a[:, 1].mean():>8.2f}{a[:, 2].mean():>8.1f}"
              f"{100 * (a[:, 0] >= 2).mean():>10.1f}%")


def per_class(y, pred):
    rec = [float((pred[y == k] == k).mean()) if (y == k).any() else float("nan") for k in range(3)]
    dist = np.bincount(pred, minlength=3) / len(pred)
    return rec, dist


def runs_report(results, y_clean, y_dis):
    print("\n== 2. Clean vs distractor test, per run ==")
    print(f"{'run':<45}{'set':<10}{'recall S/R/NEI':<22}{'pred share S/R/NEI'}")
    for f in sorted(glob.glob(f"{results}/*/*/preds_fever_test_distract.npy")):
        d = os.path.dirname(f)
        run = os.path.relpath(d, results)
        for name, y in (("clean", y_clean), ("distract", y_dis)):
            p = f"{d}/preds_fever_test.npy" if name == "clean" else f
            if not os.path.exists(p):
                continue
            pred = np.load(p).argmax(-1)
            if len(pred) != len(y):
                print(f"{run:<45}{name:<10}length mismatch ({len(pred)} vs {len(y)})")
                continue
            rec, dist = per_class(y, pred)
            print(f"{run:<45}{name:<10}{' / '.join(f'{x:.2f}' for x in rec):<22}{' / '.join(f'{x:.2f}' for x in dist)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", required=True)
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    evidence_shape(read_jsonl(f"{a.proc}/fever/train.jsonl"))
    test = read_jsonl(f"{a.proc}/fever/test.jsonl")
    dis = read_jsonl(f"{a.proc}/robustness/fever_test_distract.jsonl")
    added = Counter(len(d["evidence_gold"].split()) - len(t["evidence_gold"].split()) > 0 for d, t in zip(dis, test))
    extra_words = np.mean([len(d["evidence_gold"].split()) - len(t["evidence_gold"].split()) for d, t in zip(dis, test)])
    print(f"\ncurrent distractor set: {len(dis)} rows, {extra_words:.0f} extra words per example on average ({added[True]} changed)")
    runs_report(a.results, np.array([r["label"] for r in test]), np.array([r["label"] for r in dis]))
    out = f"{a.proc}/robustness/fever_test_distract_sent.jsonl"
    if not os.path.exists(out):
        new = add_sentence_distractors(test, n_sents=1, seed=0)
        write_jsonl(new, out)
        extra = np.mean([len(n["evidence_gold"].split()) - len(t["evidence_gold"].split()) for n, t in zip(new, test)])
        print(f"\nwrote {out}: {len(new)} rows, {extra:.0f} extra words per example on average")
    else:
        print(f"\n{out} already exists")


if __name__ == "__main__":
    main()
