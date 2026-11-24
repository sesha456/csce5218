"""Two-stage BM25 evidence retrieval over the FEVER Wikipedia dump.

Stage 1 (documents): bm25s over "page title + first sentence" for every page.
Stage 2 (sentences): plain BM25 re-ranking of all sentences in the top pages.
This mirrors the TF-IDF/BM25 baselines of the FEVER shared task and is used both
to attach evidence to NOT ENOUGH INFO claims and for the "retrieved evidence"
condition of experiment E4.
"""
import json
import os

import bm25s
import numpy as np
from tqdm.auto import tqdm

from . import bm25
from .text import clean_wiki_text, page_title_to_text

try:
    import Stemmer  # PyStemmer
    _STEMMER = Stemmer.Stemmer("english")
except ImportError:  # pragma: no cover
    _STEMMER = None


def _tok(texts, show_progress=False):
    return bm25s.tokenize(texts, stopwords="en", stemmer=_STEMMER, show_progress=show_progress)


def build_doc_index(wiki_db, index_dir: str) -> int:
    """Index title + intro of every page in wiki_db. Returns #pages indexed."""
    os.makedirs(index_dir, exist_ok=True)
    ids, texts = [], []
    for page_id, intro in tqdm(wiki_db.iter_intros(), desc="collect intros", unit="page"):
        ids.append(page_id)
        texts.append(f"{page_title_to_text(page_id)} {clean_wiki_text(intro)}")
    tokens = _tok(texts, show_progress=True)
    del texts  # free memory before indexing (~5.4M pages)
    retriever = bm25s.BM25()
    retriever.index(tokens)
    retriever.save(index_dir)
    with open(os.path.join(index_dir, "page_ids.json"), "w") as fh:
        json.dump(ids, fh)
    return len(ids)


def format_evidence(items) -> str:
    """items: iterable of (page_id, sentence_text) -> one evidence string."""
    return " ".join(f"{page_title_to_text(p)}: {clean_wiki_text(s)}" for p, s in items)


class WikiRetriever:
    def __init__(self, wiki_db, index_dir: str):
        self.db = wiki_db
        self.bm25 = bm25s.BM25.load(index_dir, mmap=True)
        with open(os.path.join(index_dir, "page_ids.json")) as fh:
            self.page_ids = json.load(fh)

    def top_pages(self, claims, k: int = 5, batch_size: int = 2048, n_threads: int = 0):
        """Return a list (one per claim) of the top-k page ids."""
        out = []
        for start in tqdm(range(0, len(claims), batch_size), desc="doc retrieval"):
            chunk = claims[start:start + batch_size]
            res, _ = self.bm25.retrieve(_tok(chunk), k=k, n_threads=n_threads, show_progress=False)
            out.extend([[self.page_ids[int(i)] for i in row] for row in np.asarray(res)])
        return out

    def top_sentences(self, claim: str, pages, k: int = 2):
        """Re-rank sentences from `pages`; return [(page_id, sent_id, text)]."""
        cands = []
        for p in pages:
            for sid, sent in enumerate(self.db.sentences(p)):
                if sent:
                    cands.append((p, sid, sent))
        if not cands:
            return []
        docs = [f"{page_title_to_text(p)} {clean_wiki_text(s)}" for p, _, s in cands]
        return [cands[i] for i in bm25.rank(claim, docs)[:k]]
