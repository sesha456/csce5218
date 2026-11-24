"""Build the SciFact out-of-domain test set (train + dev claims, since test labels are hidden).

python scripts/prepare_scifact.py --raw data/raw/scifact --out data/processed/scifact
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ne.fever import read_jsonl, write_jsonl  # noqa: E402
from ne.scifact import build_examples, load_corpus  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--k-sents", type=int, default=2)
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
corpus = load_corpus(os.path.join(args.raw, "corpus.jsonl"))
rows = []
for split in ("train", "dev"):
    rows += build_examples(read_jsonl(os.path.join(args.raw, f"claims_{split}.jsonl")),
                           corpus, args.k_sents, split_name=f"scifact_{split}")
write_jsonl(rows, os.path.join(args.out, "test.jsonl"))
stats = {"examples": len(rows), "labels": Counter(r["label_name"] for r in rows)}
with open(os.path.join(args.out, "stats.json"), "w") as fh:
    json.dump(stats, fh, indent=2)
print(json.dumps(stats))
