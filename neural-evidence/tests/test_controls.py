"""Distractor and paraphrase controls on tiny hand-made rows (no GPU, no downloads)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ne import controls  # noqa: E402

PAGES = {
    "Barack_Obama": ["Barack Obama is an American politician and lawyer .",
                     "He served as the 44th President of the United States .",
                     "Obama was born in Honolulu , Hawaii .",
                     "Obama graduated from Columbia University in 1983 ."],
    "Paris": ["Paris is the capital of France and its largest city .",
              "The Eiffel Tower is located in Paris , France .",
              "Paris hosted the Summer Olympics in 1900 and 1924 ."],
    "Mount_Everest": ["Mount Everest is Earth 's highest mountain above sea level .",
                      "It lies in the Mahalangur Himal range of the Himalayas ."],
}


class FakeDB:
    def sentences(self, page):
        return tuple(PAGES.get(page, ()))


class FakeScorer:
    """Calls everything neutral except sentences mentioning Honolulu."""
    def probs(self, premises, hypotheses, bs=64):
        out = np.tile([0.1, 0.1, 0.8], (len(premises), 1))
        for i, p in enumerate(premises):
            if "Honolulu" in p:
                out[i] = [0.9, 0.05, 0.05]
        return out


def _row(i, claim, label, gold, retrieved, source="gold"):
    from ne.retrieval import format_evidence
    ids = gold if source == "gold" else retrieved
    ev = format_evidence((p, PAGES[p][s]) for p, s in ids)
    return {"id": i, "claim": claim, "label": label, "group": f"g{i}", "evidence_gold": ev,
            "evidence_gold_source": source, "evidence_retrieved": ev,
            "gold_ids": [list(x) for x in gold] if source == "gold" else [],
            "retrieved_ids": [list(x) for x in retrieved]}


ROWS = [
    _row(1, "Barack Obama was born in Kenya.", 1, [("Barack_Obama", 2)], [("Barack_Obama", 0), ("Barack_Obama", 1)]),
    _row(2, "Paris is the capital of France.", 0, [("Paris", 0)], [("Paris", 0), ("Paris", 1)]),
    _row(3, "Mount Everest is the highest mountain.", 0, [("Mount_Everest", 0)], [("Mount_Everest", 0), ("Mount_Everest", 1)]),
    _row(4, "Obama likes surfing.", 2, [], [("Barack_Obama", 0), ("Barack_Obama", 3)], source="retrieved"),
]
RAW = [{"id": 1, "evidence": [[[0, 0, "Barack_Obama", 2]], [[0, 1, "Barack_Obama", 3]]]},
       {"id": 2, "evidence": [[[0, 0, "Paris", 0]]]}, {"id": 3, "evidence": [[[0, 0, "Mount_Everest", 0]]]},
       {"id": 4, "evidence": [[[0, None, None, None]]]}]


def test_candidates_respect_exclusions():
    ann = controls.annotated_sentences(RAW)
    assert ann["1"] == {("Barack_Obama", 2), ("Barack_Obama", 3)}
    cands = controls.candidates(ROWS, FakeDB(), ann, n_cand=3)
    for r, c in zip(ROWS, cands):
        excl = controls._excluded(r, ann)
        own_pages = {p for p, _ in controls.own_ids(r)} | {p for p, _ in r["retrieved_ids"]}
        for cond in controls.CONDITIONS:
            assert all((p, s) not in excl for p, s, _ in c[cond]), (r["id"], cond)
        assert all(p not in own_pages for p, _, _ in c["random"])
        assert all(p == controls.own_ids(r)[0][0] for p, _, _ in c["entity"])
    # claim 1: sentences 0-1 retrieved, 2-3 annotated, so its own page has nothing left
    assert cands[0]["entity"] == [] and cands[0]["topical"] == []
    # NEI rows use the same procedure: their entity page is the first retrieved page
    assert {s for _, s, _ in cands[3]["entity"]} == {1, 2}


def test_neutral_filter_and_rows():
    ann = controls.annotated_sentences(RAW)
    cands = controls.candidates(ROWS, FakeDB(), ann, n_cand=3)
    kept, share = controls.neutral_filter(cands, [r["claim"] for r in ROWS], FakeScorer())
    assert all("Honolulu" not in t for c in kept for cond in controls.CONDITIONS for _, _, t in c[cond])
    assert 0 < share["entity"] < 1
    rows = controls.distracted_rows(ROWS, kept, "entity", 0)
    assert len(rows) == len(ROWS) and [r["label"] for r in rows] == [r["label"] for r in ROWS]
    for r, o, k in zip(rows, ROWS, kept):
        if k["entity"]:
            assert r["has_distractor"] and r["evidence_gold"] == f"{o['evidence_gold']} {k['entity'][0][2]}"
        else:
            assert not r["has_distractor"] and r["evidence_gold"] == o["evidence_gold"]
    mask = controls.coverage_mask(kept, 1)
    assert mask.dtype == bool and not mask[0]


def test_condition_stats_and_overlap():
    y = np.array([0, 1, 2, 0])
    clean, dis = np.array([0, 1, 2, 0]), np.array([2, 1, 2, 0])
    s = controls.condition_stats(y, clean, dis, np.ones(4, bool))
    assert s["flip_rate"] == 0.25 and abs(s["SR_to_NEI"] - 1 / 3) < 1e-9 and s["S_recall"] == 0.5
    assert controls.claim_overlap("Paris capital France", "Paris: Paris is in France.") == 2 / 3


def test_matched_paraphrase():
    test = [{"id": i, "claim": f"c{i}", "label": i % 3} for i in range(6)]
    para = [{"id": f"{i}_para", "claim": f"p{i}", "claim_original": f"c{i}", "label": i % 3} for i in (1, 2, 4, 5)]
    lt = np.eye(3)[[0, 1, 2, 0, 1, 2]]
    lp = np.eye(3)[[1, 2, 0, 2]]   # wrong only on claim 4's paraphrase
    m = controls.matched_paraphrase(test, para, lt, lp, n_boot=50)
    assert m["n"] == 4 and m["orig_acc"] == 1.0 and m["para_acc"] == 0.75 and m["flip_rate"] == 0.25
    assert list(m["index"]) == [1, 2, 4, 5] and m["delta"] < 0
