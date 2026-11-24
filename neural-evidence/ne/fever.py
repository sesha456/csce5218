"""Turn FEVER claims into (claim, evidence, label) examples with leak-free splits."""
import json
import random
from collections import Counter

from tqdm.auto import tqdm

from . import LABEL2ID
from .retrieval import format_evidence
from .text import claim_group


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def gold_evidence(record, wiki_db):
    """Pick the first annotated evidence set whose sentences all exist in the dump.

    FEVER records hold several evidence sets (one per annotation); each is a list
    of [annotation_id, evidence_id, page_id, sentence_id]. Using one complete set
    keeps the evidence minimal and faithful to a single annotator's judgement.
    Returns [(page_id, sent_id, text)] or None if no set can be resolved.
    """
    for ev_set in record.get("evidence", []):
        items, seen = [], set()
        for _, _, page, sid in ev_set:
            if page is None or sid is None:
                items = None
                break
            if (page, sid) in seen:
                continue
            seen.add((page, sid))
            text = wiki_db.sentence(page, sid)
            if text is None:
                items = None
                break
            items.append((page, sid, text))
        if items:
            return items
    return None


def grouped_split(groups, val_frac: float, seed: int):
    """Assign each group key to 'train' or 'val' (no group spans both)."""
    uniq = sorted(set(groups))
    random.Random(seed).shuffle(uniq)
    n_val = int(round(len(uniq) * val_frac))
    val = set(uniq[:n_val])
    return {g: ("val" if g in val else "train") for g in uniq}


def build_examples(records, wiki_db, retriever, k_pages=5, k_sents=2, split_name=None):
    """Return (examples, stats). Every example has gold-style and retrieved evidence."""
    stats = Counter()
    claims = [r["claim"] for r in records]
    pages = retriever.top_pages(claims, k=k_pages)
    out = []
    for rec, top in tqdm(zip(records, pages), total=len(records), desc="build examples"):
        label = rec["label"]
        retrieved = retriever.top_sentences(rec["claim"], top, k=k_sents)
        if label == "NOT ENOUGH INFO":
            gold, gold_source = retrieved, "retrieved"
        else:
            gold, gold_source = gold_evidence(rec, wiki_db), "gold"
            if gold is None:
                stats["dropped_unresolved_evidence"] += 1
                continue
        if not gold:
            stats["dropped_empty_evidence"] += 1
            continue
        stats[label] += 1
        out.append({
            "id": rec["id"],
            "claim": rec["claim"],
            "group": claim_group(rec["claim"]),
            "label": LABEL2ID[label],
            "label_name": label,
            "evidence_gold": format_evidence((p, s) for p, _, s in gold),
            "evidence_gold_source": gold_source,
            "evidence_retrieved": format_evidence((p, s) for p, _, s in retrieved),
            "gold_ids": [[p, i] for p, i, _ in gold] if gold_source == "gold" else [],
            "retrieved_ids": [[p, i] for p, i, _ in retrieved],
            "split": split_name,
        })
    return out, stats
