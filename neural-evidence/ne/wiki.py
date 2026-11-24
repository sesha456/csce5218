"""Store the FEVER Wikipedia dump in SQLite for random access to sentences.

The dump (wiki-pages.zip, ~5.4M pages) is far too big to keep in memory, so we
stream it once into a SQLite file: page id -> raw `lines` field. Building takes
roughly 15-30 minutes on Colab and the resulting file (~7 GB) can be cached in
Google Drive.
"""
import io
import json
import os
import sqlite3
import zipfile
from functools import lru_cache

from tqdm.auto import tqdm

from .text import parse_wiki_lines


def iter_wiki_pages(source: str):
    """Yield page dicts from wiki-pages.zip or an extracted wiki-pages/ dir."""
    if source.endswith(".zip"):
        with zipfile.ZipFile(source) as zf:
            # skip macOS resource-fork entries (__MACOSX/._*.jsonl): they are binary, not JSON
            names = sorted(
                n for n in zf.namelist()
                if n.endswith(".jsonl") and "__MACOSX" not in n and not os.path.basename(n).startswith("._")
            )
            for name in names:
                with zf.open(name) as fh:
                    for line in io.TextIOWrapper(fh, encoding="utf-8"):
                        if line.strip():
                            yield json.loads(line)
    else:
        for name in sorted(os.listdir(source)):
            if name.endswith(".jsonl") and not name.startswith("._"):
                with open(os.path.join(source, name), encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            yield json.loads(line)


def build_wiki_db(source: str, db_path: str, batch: int = 50_000) -> int:
    """Create db_path from the dump. Returns the number of pages stored."""
    if os.path.exists(db_path):
        os.remove(db_path)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("CREATE TABLE pages (id TEXT PRIMARY KEY, lines TEXT)")
    rows, n = [], 0
    for page in tqdm(iter_wiki_pages(source), desc="wiki pages", unit="page"):
        if not page.get("id"):
            continue
        rows.append((page["id"], page.get("lines", "")))
        if len(rows) >= batch:
            con.executemany("INSERT OR REPLACE INTO pages VALUES (?, ?)", rows)
            n += len(rows)
            rows = []
    con.executemany("INSERT OR REPLACE INTO pages VALUES (?, ?)", rows)
    n += len(rows)
    con.commit()
    con.close()
    return n


class WikiDB:
    """Read-only access to page sentences."""

    def __init__(self, db_path: str):
        self.con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self._sentences = lru_cache(maxsize=200_000)(self._load)

    def _load(self, page_id: str):
        row = self.con.execute("SELECT lines FROM pages WHERE id = ?", (page_id,)).fetchone()
        return tuple(parse_wiki_lines(row[0])) if row else ()

    def sentences(self, page_id: str) -> tuple:
        return self._sentences(page_id)

    def sentence(self, page_id: str, sent_id: int):
        sents = self.sentences(page_id)
        return sents[sent_id] if 0 <= sent_id < len(sents) and sents[sent_id] else None

    def iter_intros(self):
        """Yield (page_id, first non-empty sentence) for every page."""
        for page_id, lines in self.con.execute("SELECT id, lines FROM pages"):
            first = next((s for s in parse_wiki_lines(lines) if s), "")
            yield page_id, first
