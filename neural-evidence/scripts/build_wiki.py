"""Build the SQLite page store and the BM25 page index (run once, cache in Drive).

python scripts/build_wiki.py --wiki data/raw/fever/wiki-pages.zip \
    --db data/wiki/wiki.db --index data/wiki/bm25_index
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ne.retrieval import build_doc_index  # noqa: E402
from ne.wiki import WikiDB, build_wiki_db  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--wiki", required=True, help="wiki-pages.zip or extracted dir")
ap.add_argument("--db", required=True)
ap.add_argument("--index", required=True)
ap.add_argument("--skip-db", action="store_true", help="reuse an existing --db")
args = ap.parse_args()

os.makedirs(os.path.dirname(os.path.abspath(args.db)), exist_ok=True)
t = time.time()
if not args.skip_db:
    print("pages stored:", build_wiki_db(args.wiki, args.db), f"({time.time()-t:.0f}s)")
t = time.time()
print("pages indexed:", build_doc_index(WikiDB(args.db), args.index), f"({time.time()-t:.0f}s)")
