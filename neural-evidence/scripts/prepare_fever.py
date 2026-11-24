"""Build FEVER train/val/test JSONL files.

train/val: FEVER train.jsonl split ~90/10 by claim-text group (no claim text in both).
test:      FEVER shared_task_dev.jsonl (labelled dev), flagged where its claim text
           also appears in train (`text_in_train`), so leakage can be filtered/reported.

python scripts/prepare_fever.py --raw data/raw/fever --db data/wiki/wiki.db \
    --index data/wiki/bm25_index --out data/processed/fever
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ne.fever import build_examples, grouped_split, read_jsonl, write_jsonl  # noqa: E402
from ne.retrieval import WikiRetriever  # noqa: E402
from ne.text import claim_group  # noqa: E402
from ne.wiki import WikiDB  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True, help="dir with train.jsonl and shared_task_dev.jsonl")
ap.add_argument("--db", required=True)
ap.add_argument("--index", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--val-frac", type=float, default=0.10)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--k-pages", type=int, default=5)
ap.add_argument("--k-sents", type=int, default=2)
ap.add_argument("--limit", type=int, default=0, help="debug: use only the first N claims per file")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
db = WikiDB(args.db)
retriever = WikiRetriever(db, args.index)

train_raw = read_jsonl(os.path.join(args.raw, "train.jsonl"))
dev_raw = read_jsonl(os.path.join(args.raw, "shared_task_dev.jsonl"))
if args.limit:
    train_raw, dev_raw = train_raw[:args.limit], dev_raw[:args.limit]

groups = [claim_group(r["claim"]) for r in train_raw]
assign = grouped_split(groups, args.val_frac, args.seed)
train_groups = {g for g, s in assign.items() if s == "train"}
dup_ids = sum(c for c in Counter(groups).values() if c > 1)

stats = {"train_claim_ids": len(train_raw), "train_unique_groups": len(assign),
         "train_ids_sharing_text": dup_ids}
parts = {
    "train": [r for r, g in zip(train_raw, groups) if assign[g] == "train"],
    "val": [r for r, g in zip(train_raw, groups) if assign[g] == "val"],
    "test": dev_raw,
}
for name, recs in parts.items():
    rows, st = build_examples(recs, db, retriever, args.k_pages, args.k_sents, split_name=name)
    if name == "test":
        for r in rows:
            r["text_in_train"] = r["group"] in train_groups
        st["text_in_train"] = sum(r["text_in_train"] for r in rows)
    write_jsonl(rows, os.path.join(args.out, f"{name}.jsonl"))
    stats[name] = {"examples": len(rows), **st}
    print(name, json.dumps(stats[name]))

assert not ({r["group"] for r in read_jsonl(os.path.join(args.out, "train.jsonl"))}
            & {r["group"] for r in read_jsonl(os.path.join(args.out, "val.jsonl"))}), "train/val leak"
with open(os.path.join(args.out, "stats.json"), "w") as fh:
    json.dump(stats, fh, indent=2)
print("wrote", args.out)
