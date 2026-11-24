"""Turn SciFact claims into one (claim, evidence, label) example per claim."""
import json

from . import LABEL2ID, bm25
from .text import claim_group

SCIFACT_TO_FEVER = {"SUPPORT": "SUPPORTS", "CONTRADICT": "REFUTES"}


def load_corpus(path):
    with open(path, encoding="utf-8") as fh:
        return {int(d["doc_id"]): d for d in map(json.loads, fh) if d}


def build_examples(claims, corpus, k_sents=2, split_name=None):
    """SUPPORT/CONTRADICT -> rationale sentences of the first evidence doc.
    NEI (no evidence) -> top-k BM25 sentences of the first cited abstract.
    `evidence_retrieved` is always the top-k BM25 sentences of that same doc.
    """
    out = []
    for c in claims:
        if c["evidence"]:
            doc_id = sorted(int(d) for d in c["evidence"])[0]
            ann = c["evidence"][str(doc_id)]
            label = SCIFACT_TO_FEVER[ann[0]["label"]]
            sent_ids = sorted({s for a in ann for s in a["sentences"]})
            source = "gold"
        else:
            doc_id = int(c["cited_doc_ids"][0])
            label, sent_ids, source = "NOT ENOUGH INFO", None, "retrieved"
        abstract = [" ".join(x.split()) for x in corpus[doc_id]["abstract"]]
        retrieved_ids = sorted(bm25.rank(c["claim"], abstract)[:k_sents])
        if sent_ids is None:
            sent_ids = retrieved_ids
        out.append({
            "id": c["id"],
            "claim": c["claim"],
            "group": claim_group(c["claim"]),
            "label": LABEL2ID[label],
            "label_name": label,
            "evidence_gold": " ".join(abstract[i] for i in sent_ids),
            "evidence_gold_source": source,
            "evidence_retrieved": " ".join(abstract[i] for i in retrieved_ids),
            "doc_id": doc_id,
            "gold_ids": sent_ids if source == "gold" else [],
            "retrieved_ids": retrieved_ids,
            "split": split_name,
        })
    return out
