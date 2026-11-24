"""E1 neural baseline: siamese BiLSTM sentence-pair classifier with GloVe embeddings.

claim -> BiLSTM -> max-pool = u ; evidence -> same BiLSTM -> max-pool = v
features [u, v, |u - v|, u * v] -> MLP -> 3 classes  (InferSent-style, Conneau et al. 2017)
Trained from scratch; the GloVe embedding table is frozen by default.
"""
import os
import re
import time
import zipfile
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from . import metrics
from .data import evidence_of

_TOK = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")
PAD, UNK = 0, 1
GLOVE_URLS = ["https://nlp.stanford.edu/data/glove.6B.zip",
              "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"]


def tokenize(text):
    return _TOK.findall((text or "").lower())


def build_vocab(rows, evidence, min_freq=2, max_size=100_000):
    c = Counter()
    for r in rows:
        c.update(tokenize(r["claim"]))
        if evidence != "none":
            c.update(tokenize(evidence_of(r, evidence)))
    words = [w for w, n in c.most_common(max_size) if n >= min_freq]
    return {w: i + 2 for i, w in enumerate(words)}


def download_glove(dest_dir):
    """Fetch glove.6B.300d.txt into dest_dir (cached)."""
    txt = os.path.join(dest_dir, "glove.6B.300d.txt")
    if os.path.exists(txt):
        return txt
    os.makedirs(dest_dir, exist_ok=True)
    zpath = os.path.join(dest_dir, "glove.6B.zip")
    import urllib.request
    for url in GLOVE_URLS:
        try:
            print("downloading", url)
            urllib.request.urlretrieve(url, zpath)
            break
        except Exception as e:  # try the mirror
            print("  failed:", e)
    with zipfile.ZipFile(zpath) as zf:
        zf.extract("glove.6B.300d.txt", dest_dir)
    os.remove(zpath)
    return txt


def load_embeddings(vocab, glove_path, dim=300, seed=0):
    rng = np.random.default_rng(seed)
    emb = rng.normal(0, 0.1, (len(vocab) + 2, dim)).astype(np.float32)
    emb[PAD] = 0
    hits = 0
    if glove_path:
        with open(glove_path, encoding="utf-8") as fh:
            for line in fh:
                w, rest = line.split(" ", 1)
                if w in vocab:
                    emb[vocab[w]] = np.asarray(rest.split(), dtype=np.float32)
                    hits += 1
    print(f"GloVe coverage: {hits}/{len(vocab)} words")
    return torch.from_numpy(emb)


class BiLSTMPair(nn.Module):
    def __init__(self, emb, hidden=256, layers=2, dropout=0.3, freeze=True, claim_only=False):
        super().__init__()
        self.claim_only = claim_only
        self.emb = nn.Embedding.from_pretrained(emb, freeze=freeze, padding_idx=PAD)
        self.lstm = nn.LSTM(emb.size(1), hidden, num_layers=layers, bidirectional=True,
                            batch_first=True, dropout=dropout)
        d = 2 * hidden
        self.mlp = nn.Sequential(nn.Dropout(dropout), nn.Linear(d if claim_only else 4 * d, 512), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(512, 3))

    def encode(self, ids, lens):
        x = self.emb(ids)
        packed = pack_padded_sequence(x, lens.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, padding_value=-1e4)
        return out.max(dim=1).values

    def forward(self, c, cl, e=None, el=None):
        u = self.encode(c, cl)
        if self.claim_only:
            return self.mlp(u)
        v = self.encode(e, el)
        return self.mlp(torch.cat([u, v, (u - v).abs(), u * v], dim=-1))


def _ids(text, vocab, max_len):
    ids = [vocab.get(t, UNK) for t in tokenize(text)][:max_len]
    return ids or [UNK]


def _batches(rows, vocab, evidence, bs, shuffle, rng, max_claim=48, max_ev=128):
    order = rng.permutation(len(rows)) if shuffle else np.arange(len(rows))
    for i in range(0, len(order), bs):
        chunk = [rows[j] for j in order[i:i + bs]]

        def pad(seqs):
            L = max(len(s) for s in seqs)
            return (torch.tensor([s + [PAD] * (L - len(s)) for s in seqs]), torch.tensor([len(s) for s in seqs]))
        c, cl = pad([_ids(r["claim"], vocab, max_claim) for r in chunk])
        if evidence == "none":
            e = el = None
        else:
            e, el = pad([_ids(evidence_of(r, evidence), vocab, max_ev) for r in chunk])
        yield c, cl, e, el, torch.tensor([r["label"] for r in chunk]), order[i:i + bs]


def predict(model, rows, vocab, evidence, device, bs=512):
    model.eval()
    out = np.zeros(len(rows), dtype=int)
    with torch.no_grad():
        for c, cl, e, el, _, idx in _batches(rows, vocab, evidence, bs, False, None):
            args = (c.to(device), cl) if e is None else (c.to(device), cl, e.to(device), el)
            out[idx] = model(*args).argmax(-1).cpu().numpy()
    return out


def train(train_rows, val_rows, glove_path, evidence="gold", epochs=6, bs=128, lr=1e-3, seed=42,
          patience=2, log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = build_vocab(train_rows, evidence)
    model = BiLSTMPair(load_embeddings(vocab, glove_path), claim_only=evidence == "none").to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    lossf = nn.CrossEntropyLoss()
    best, best_state, bad = -1.0, None, 0
    yv = [r["label"] for r in val_rows]
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for c, cl, e, el, y, _ in _batches(train_rows, vocab, evidence, bs, True, rng):
            args = (c.to(device), cl) if e is None else (c.to(device), cl, e.to(device), el)
            loss = lossf(model(*args), y.to(device))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(y)
        f1 = metrics.evaluate(yv, predict(model, val_rows, vocab, evidence, device))["macro_f1"]
        log(f"epoch {ep+1}: train loss {tot/len(train_rows):.4f}, val macro-F1 {f1:.4f}")
        if f1 > best:
            best, bad = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, vocab, {"train_seconds": round(time.time() - t0, 1), "best_val_macro_f1": best,
                          "trainable_params": n_params}
