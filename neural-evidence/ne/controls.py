"""Control experiments for the E6 distractor and E7 paraphrase results (evaluation only).

Distractor controls. One sentence is appended to the evidence of every FEVER test claim
(all three labels, same procedure), under three conditions:

* random   - an annotated single-sentence gold evidence sentence taken from another claim,
             from a page that is not among the claim's own evidence or retrieved pages
* entity   - a random sentence from the same Wikipedia page as the first sentence of the
             claim's own evidence (same entity, so it shares the page title)
* topical  - the sentences that BM25 ranks closest to the claim on the claim's evidence and
             retrieved pages (shares the topic and the claim's words)

For every condition, sentences that belong to *any* annotated FEVER evidence set for the
claim, the claim's own evidence and its two BM25-retrieved sentences are excluded, and
candidates are kept only if an MNLI-only model (roberta-large-mnli) labels them neutral
towards the claim, so the added sentence should not change the gold label. Each condition
has several samples (sample j uses the j-th surviving candidate), and rows are compared
on the subset that has a distractor in every condition and sample.

Paraphrase control: the base and E7 models are compared on exactly the same claims,
original vs back-translated, using the saved predictions.
"""
import random
from collections import Counter

import numpy as np

from . import bm25, metrics
from .retrieval import format_evidence
from .text import clean_wiki_text, page_title_to_text

CONDITIONS = ("random", "entity", "topical")
MIN_WORDS = 4   # skip list fragments and headings


def annotated_sentences(raw_records) -> dict:
    """FEVER claim id -> every (page, sentence id) in any of its annotated evidence sets."""
    out = {}
    for rec in raw_records:
        s = set()
        for ev_set in rec.get("evidence", []):
            for _, _, page, sid in ev_set:
                if page is not None and sid is not None:
                    s.add((page, int(sid)))
        out[str(rec["id"])] = s
    return out


def own_ids(row):
    ids = row["gold_ids"] if row.get("evidence_gold_source") == "gold" else row["retrieved_ids"]
    return [(p, int(i)) for p, i in ids]


def _excluded(row, annotated):
    return (set(own_ids(row)) | {(p, int(i)) for p, i in row["retrieved_ids"]}
            | annotated.get(str(row["id"]), set()))


def _ok(sent):
    return bool(sent) and len(sent.split()) >= MIN_WORDS


def candidates(rows, wiki_db, annotated, n_cand=6, seed=0):
    """Per row: {condition: [(page, sent_id, formatted text), ...]} in preference order.

    random/entity candidates are in a seeded random order, topical ones in BM25 order.
    """
    pool = [(r["gold_ids"][0][0], int(r["gold_ids"][0][1]), r["evidence_gold"], r["group"]) for r in rows
            if r.get("evidence_gold_source") == "gold" and len(r.get("gold_ids") or []) == 1]
    if not pool:
        raise ValueError("no single-sentence gold evidence to sample random distractors from")
    out = []
    for r in rows:
        rng = random.Random(f"{seed}-{r['id']}")
        excl = _excluded(r, annotated)
        own = own_ids(r)
        pages = list(dict.fromkeys([p for p, _ in own] + [p for p, _ in r["retrieved_ids"]]))
        # random: another claim's gold sentence from an unrelated page
        rand, tries = [], 0
        while len(rand) < n_cand and tries < 50 * n_cand:
            tries += 1
            p, sid, text, g = pool[rng.randrange(len(pool))]
            if g != r["group"] and p not in pages and (p, sid) not in excl and all(text != t for *_, t in rand):
                rand.append((p, sid, text))
        # entity: other sentences of the page the evidence starts with
        ent_page = own[0][0] if own else None
        ent = [(ent_page, sid, s) for sid, s in enumerate(wiki_db.sentences(ent_page))
               if _ok(s) and (ent_page, sid) not in excl] if ent_page else []
        rng.shuffle(ent)
        ent = ent[:n_cand]
        # topical: best BM25 matches to the claim on the evidence/retrieved pages
        cands = [(p, sid, s) for p in pages for sid, s in enumerate(wiki_db.sentences(p))
                 if _ok(s) and (p, sid) not in excl]
        docs = [f"{page_title_to_text(p)} {clean_wiki_text(s)}" for p, _, s in cands]
        top = [cands[i] for i in bm25.rank(r["claim"], docs)[:n_cand]] if cands else []
        fmt = lambda items: [(p, sid, format_evidence([(p, s)])) for p, sid, s in items]
        out.append({"random": rand, "entity": fmt(ent), "topical": fmt(top)})
    return out


def neutral_filter(cands, claims, scorer, bs=256):
    """Keep candidates that the MNLI scorer labels neutral (argmax) towards the claim.

    scorer.probs(premises, hypotheses) -> (n, 3) probabilities in FEVER order (S, R, NEI).
    Returns the filtered structure plus the share of candidates kept per condition.
    """
    flat = [(i, c, k) for i, row in enumerate(cands) for c in CONDITIONS for k in range(len(row[c]))]
    if not flat:
        return cands, {}
    probs = scorer.probs([cands[i][c][k][2] for i, c, k in flat], [claims[i] for i, _, _ in flat], bs=bs)
    keep = probs.argmax(-1) == 2
    out = [{c: [] for c in CONDITIONS} for _ in cands]
    seen, kept = Counter(), Counter()
    for (i, c, k), ok in zip(flat, keep):
        seen[c] += 1
        if ok:
            kept[c] += 1
            out[i][c].append(cands[i][c][k])
    return out, {c: kept[c] / seen[c] for c in CONDITIONS if seen[c]}


def distracted_rows(rows, picks, condition, sample):
    """Copy of rows with picks[i][condition][sample] appended to both evidence fields.

    Rows without such a candidate keep their evidence and get has_distractor=False,
    so the file stays aligned with the clean test set.
    """
    out = []
    for r, p in zip(rows, picks):
        new = dict(r)
        cand = p[condition][sample] if len(p[condition]) > sample else None
        new["has_distractor"] = cand is not None
        new["perturbation"] = f"control_{condition}_{sample}"
        if cand is not None:
            new["distractor"] = cand[2]
            new["distractor_ids"] = [cand[0], cand[1]]
            for f in ("evidence_gold", "evidence_retrieved"):
                new[f] = f"{r[f]} {cand[2]}"
        out.append(new)
    return out


def coverage_mask(picks, n_samples):
    return np.array([all(len(p[c]) >= n_samples for c in CONDITIONS) for p in picks])


def claim_overlap(claim, text):
    """Share of the claim's content words that also appear in `text`."""
    q = set(bm25.tokenize(claim))
    return len(q & set(bm25.tokenize(text))) / len(q) if q else 0.0


def token_lengths(tok, rows):
    """Untruncated length (claim + evidence, with special tokens) of every row."""
    enc = tok([r["claim"] for r in rows], [r["evidence_gold"] for r in rows], truncation=False)
    return np.array([len(x) for x in enc["input_ids"]])


def condition_stats(y, pred_clean, pred, mask):
    """Macro-F1 and how the predictions moved on the rows in `mask`."""
    y, pc, pd_ = y[mask], pred_clean[mask], pred[mask]
    m = metrics.evaluate(y, pd_)
    rec = {"n": int(mask.sum()), "macro_f1": m["macro_f1"],
           "flip_rate": float((pc != pd_).mean()),
           "pred_nei_share": float((pd_ == 2).mean())}
    for k, name in enumerate(("S", "R", "NEI")):
        rec[f"{name}_recall"] = m["per_class"][metrics.LABELS[k]]["recall"]
    sr = y != 2
    rec["SR_to_NEI"] = float(((pc == y) & (pd_ == 2))[sr].sum() / max(1, (pc == y)[sr].sum()))
    return rec


def matched_paraphrase(test_rows, para_rows, logits_test, logits_para, n_boot=1000, seed=0):
    """Score one model on exactly the claims that have a paraphrase: original vs paraphrase.

    Paraphrase rows carry id '<orig id>_para' and the original claim in claim_original.
    Returns macro-F1 on both, the paired difference with a bootstrap 95% interval, and the
    share of claims whose prediction changed.
    """
    pos = {str(r["id"]): i for i, r in enumerate(test_rows)}
    idx = []
    for p in para_rows:
        i = pos[str(p["id"])[:-len("_para")]]
        assert test_rows[i]["claim"] == p["claim_original"] and test_rows[i]["label"] == p["label"]
        idx.append(i)
    idx = np.array(idx)
    y = np.array([p["label"] for p in para_rows])
    po, pp = logits_test[idx].argmax(-1), logits_para.argmax(-1)
    f_o, f_p = metrics.evaluate(y, po)["macro_f1"], metrics.evaluate(y, pp)["macro_f1"]
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        b = rng.integers(0, len(y), len(y))
        deltas.append(metrics.evaluate(y[b], pp[b])["macro_f1"] - metrics.evaluate(y[b], po[b])["macro_f1"])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"n": int(len(y)), "orig_macro_f1": f_o, "para_macro_f1": f_p, "delta": f_p - f_o,
            "delta_ci95": [float(lo), float(hi)], "flip_rate": float((po != pp).mean()),
            "orig_acc": float((po == y).mean()), "para_acc": float((pp == y).mean()), "index": idx}
