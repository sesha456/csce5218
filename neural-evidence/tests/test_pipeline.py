"""End-to-end test on a tiny synthetic FEVER dump. Run: python -m pytest tests -q
(or just `python tests/test_pipeline.py`). Needs no GPU and no downloads."""
import json
import os
import subprocess
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ne import metrics  # noqa: E402
from ne.text import clean_wiki_text, parse_wiki_lines  # noqa: E402

PAGES = {
    "Barack_Obama": ["Barack Obama is an American politician .",
                     "He served as the 44th President of the United States from 2009 to 2017 .",
                     "Obama was born in Honolulu , Hawaii ."],
    "Honolulu": ["Honolulu is the capital of Hawaii .", "It lies on the island of Oahu ."],
    "Paris": ["Paris is the capital of France -LRB- the French Republic -RRB- .",
              "The Eiffel Tower is located in Paris ."],
    "Mount_Everest": ["Mount Everest is Earth 's highest mountain above sea level ."],
}
CLAIMS = [
    (1, "Barack Obama was born in Honolulu.", "SUPPORTS", [[[0, 0, "Barack_Obama", 2]]]),
    (2, "Barack Obama was born in Kenya.", "REFUTES", [[[0, 0, "Barack_Obama", 2]]]),
    (3, "Barack Obama likes surfing in Oahu.", "NOT ENOUGH INFO", [[[0, None, None, None]]]),
    (4, "Paris is the capital of France.", "SUPPORTS", [[[0, 0, "Paris", 0]]]),
    (5, "Paris is the capital of France.", "SUPPORTS", [[[0, 0, "Paris", 0]]]),  # duplicate text
    (6, "The Eiffel Tower is in Berlin.", "REFUTES", [[[0, 0, "Paris", 1]]]),
    (7, "Mount Everest is the highest mountain.", "SUPPORTS", [[[0, 0, "Mount_Everest", 0]]]),
    (8, "Mount Everest is in Hawaii.", "REFUTES", [[[0, 0, "Missing_Page", 0]]]),  # unresolvable
]


def _write(path, rows):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_text_helpers():
    assert clean_wiki_text("France -LRB- the Republic -RRB- , Earth 's") == "France (the Republic), Earth's"
    assert parse_wiki_lines("0\tA .\tx\n1\t\n2\tC .") == ["A .", "", "C ."]


def test_metrics():
    m = metrics.evaluate([0, 1, 2, 1], [0, 1, 2, 2])
    assert m["accuracy"] == 0.75 and m["refutes_recall"] == 0.5
    assert m["confusion_matrix"][1] == [0, 1, 1]
    agg = metrics.aggregate_seeds([m, m])
    assert agg["macro_f1"]["std"] == 0.0


def test_fever_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        wiki = os.path.join(d, "wiki-pages.zip")
        with zipfile.ZipFile(wiki, "w") as zf:
            body = "\n".join(json.dumps({"id": pid, "text": " ".join(s), "lines": "\n".join(
                f"{i}\t{t}\tlink" for i, t in enumerate(s))}) for pid, s in PAGES.items())
            zf.writestr("wiki-pages/wiki-001.jsonl", body + "\n" + json.dumps({"id": "", "text": "", "lines": ""}))
        raw = os.path.join(d, "raw")
        os.makedirs(raw)
        rows = [{"id": i, "claim": c, "label": l, "evidence": e, "verifiable": "x"} for i, c, l, e in CLAIMS]
        _write(os.path.join(raw, "train.jsonl"), rows)
        _write(os.path.join(raw, "shared_task_dev.jsonl"), rows[:4])
        db, idx, out = (os.path.join(d, x) for x in ("wiki.db", "idx", "out"))
        run = lambda *a: subprocess.run([sys.executable, *a], check=True, cwd=ROOT, capture_output=True, text=True)
        run("scripts/build_wiki.py", "--wiki", wiki, "--db", db, "--index", idx)
        run("scripts/prepare_fever.py", "--raw", raw, "--db", db, "--index", idx, "--out", out,
            "--val-frac", "0.4", "--k-pages", "2")
        load = lambda n: [json.loads(l) for l in open(os.path.join(out, f"{n}.jsonl"))]
        train, val, test = load("train"), load("val"), load("test")
        allrows = train + val
        assert 8 not in {r["id"] for r in allrows}, "unresolvable evidence must be dropped"
        assert not ({r["group"] for r in train} & {r["group"] for r in val}), "leak"
        g = {r["id"]: r["split"] for r in allrows}
        assert g[4] == g[5], "duplicate claim texts must share a split"
        by = {r["id"]: r for r in allrows}
        assert by[1]["evidence_gold"] == "Barack Obama: Obama was born in Honolulu, Hawaii."
        assert by[3]["evidence_gold_source"] == "retrieved" and by[3]["evidence_gold"]
        assert "(the French Republic)" in by[4]["evidence_gold"]
        assert all("text_in_train" in r for r in test)
        stats = json.load(open(os.path.join(out, "stats.json")))
        assert stats["train_ids_sharing_text"] == 2


if __name__ == "__main__":
    test_text_helpers(); test_metrics(); test_fever_end_to_end(); print("all tests passed")
