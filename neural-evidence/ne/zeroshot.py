"""E1 zero-shot baseline: an MNLI-only model applied without any FEVER training.

premise = evidence, hypothesis = claim
entailment -> SUPPORTS, contradiction -> REFUTES, neutral -> NOT ENOUGH INFO
Use an MNLI-only checkpoint (default roberta-large-mnli). Checkpoints such as
MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli were trained on FEVER and would leak.
"""
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

NLI_TO_FEVER = {"entailment": 0, "contradiction": 1, "neutral": 2}


class NLIScorer:
    def __init__(self, model_name="roberta-large-mnli", max_length=256):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=dtype).to(self.device).eval()
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.col = [next(i for i, n in id2label.items() if n.startswith(lbl))
                    for lbl in ("entailment", "contradiction", "neutral")]  # -> FEVER order S, R, NEI
        self.max_length = max_length

    @torch.no_grad()
    def probs(self, premises, hypotheses, bs=64):
        """Return an (n, 3) array of probabilities in FEVER label order (S, R, NEI)."""
        order = np.argsort([len(p) + len(h) for p, h in zip(premises, hypotheses)])
        out = np.zeros((len(premises), 3), dtype=np.float32)
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            enc = self.tok([premises[j] for j in idx], [hypotheses[j] for j in idx], truncation="only_first",
                           max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
            p = self.model(**enc).logits.float().softmax(-1)[:, self.col]
            out[idx] = p.cpu().numpy()
        return out


def predict_rows(scorer, rows, evidence_field="evidence_gold"):
    t0 = time.time()
    p = scorer.probs([r[evidence_field] for r in rows], [r["claim"] for r in rows])
    return p.argmax(-1), round(time.time() - t0, 1)
