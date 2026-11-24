"""E6/E7 data: distractor evidence, back-translated claims, and paraphrase filtering."""
import random
import time

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .text import normalize_claim


def add_distractors(rows, n_sents=2, seed=0, position="end"):
    """Append (or prepend) `n_sents` whole retrieved-evidence blocks from *other* claims
    (each block is k_sents=2 BM25 sentences, so n_sents=2 adds up to 4 sentences).
    Labels are unchanged. See `add_sentence_distractors` for the style-matched variant."""
    rng = random.Random(seed)
    pool = [(r["group"], r["evidence_retrieved"]) for r in rows if r.get("evidence_retrieved")]
    out = []
    for r in rows:
        picks = []
        while len(picks) < n_sents:
            g, ev = pool[rng.randrange(len(pool))]
            if g != r["group"]:
                picks.append(ev)
        extra = " ".join(picks)
        new = dict(r)
        for f in ("evidence_gold", "evidence_retrieved"):
            new[f] = f"{r[f]} {extra}" if position == "end" else f"{extra} {r[f]}"
        new["perturbation"] = f"distractor_{n_sents}_{position}"
        out.append(new)
    return out


def add_sentence_distractors(rows, n_sents=1, seed=0, position="end"):
    """Fairer E6 set: append `n_sents` *single* annotated gold sentences taken from other
    claims (SUPPORTS/REFUTES rows whose gold evidence is exactly one sentence).

    `add_distractors` appends whole retrieved-evidence blocks (2 BM25 sentences each), which
    look exactly like the evidence attached to NOT ENOUGH INFO claims in training, so models
    can read "extra off-topic retrieved text" as a label cue. Single gold sentences match the
    style of real SUPPORTS/REFUTES evidence instead. Labels are unchanged."""
    rng = random.Random(seed)
    pool = [(r["group"], r["evidence_gold"]) for r in rows
            if r.get("evidence_gold_source") == "gold" and len(r.get("gold_ids") or []) == 1]
    if not pool:
        raise ValueError("no single-sentence gold evidence to sample distractors from")
    out = []
    for r in rows:
        picks = []
        while len(picks) < n_sents:
            g, ev = pool[rng.randrange(len(pool))]
            if g != r["group"]:
                picks.append(ev)
        extra = " ".join(picks)
        new = dict(r)
        for f in ("evidence_gold", "evidence_retrieved"):
            new[f] = f"{r[f]} {extra}" if position == "end" else f"{extra} {r[f]}"
        new["perturbation"] = f"distractor_sent{n_sents}_{position}"
        out.append(new)
    return out


class BackTranslator:
    """English -> pivot -> English with MarianMT (Helsinki-NLP opus-mt)."""

    def __init__(self, pivot="de"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.fwd = self._load(f"Helsinki-NLP/opus-mt-en-{pivot}", dtype)
        self.bwd = self._load(f"Helsinki-NLP/opus-mt-{pivot}-en", dtype)

    def _load(self, name, dtype):
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSeq2SeqLM.from_pretrained(name, torch_dtype=dtype).to(self.device).eval()
        return tok, model

    @torch.no_grad()
    def _translate(self, pair, texts, bs, sample=False):
        tok, model = pair
        out = [None] * len(texts)
        order = np.argsort([len(t) for t in texts])
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True, truncation=True,
                      max_length=128).to(self.device)
            gen = model.generate(**enc, max_new_tokens=128,
                                 **({"do_sample": True, "top_p": 0.9} if sample else {"num_beams": 4}))
            for j, t in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
                out[j] = t
        return out

    def paraphrase(self, texts, bs=64):
        """Beam back-translation; claims that come back unchanged are retried with sampling."""
        t0 = time.time()
        para = self._translate(self.bwd, self._translate(self.fwd, texts, bs), bs)
        same = [i for i, (a, b) in enumerate(zip(texts, para)) if normalize_claim(a) == normalize_claim(b)]
        if same:
            sub = [texts[i] for i in same]
            retry = self._translate(self.bwd, self._translate(self.fwd, sub, bs, sample=True), bs, sample=True)
            for i, t in zip(same, retry):
                para[i] = t
        print(f"back-translated {len(texts)} claims in {time.time()-t0:.0f}s ({len(same)} retried)")
        return para


def paraphrase_rows(rows, paraphrases, nli_scores=None, threshold=0.5):
    """Copy rows with the claim replaced; keep evidence, label and group (same split)."""
    out = []
    for i, (r, p) in enumerate(zip(rows, paraphrases)):
        new = dict(r)
        new["claim_original"], new["claim"] = r["claim"], p
        new["id"] = f"{r['id']}_para"
        new["para_changed"] = normalize_claim(p) != normalize_claim(r["claim"])
        if nli_scores is not None:
            new["para_nli_entail"] = float(nli_scores[i])
            new["para_nli_ok"] = bool(nli_scores[i] >= threshold)
        new["perturbation"] = "backtranslation"
        out.append(new)
    return out


def mutual_entailment(scorer, a, b):
    """min(P(a entails b), P(b entails a)) from an MNLI model: a meaning-preservation score."""
    p_ab = scorer.probs(a, b)[:, 0]
    p_ba = scorer.probs(b, a)[:, 0]
    return np.minimum(p_ab, p_ba)
