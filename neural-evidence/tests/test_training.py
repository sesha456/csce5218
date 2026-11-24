"""Smoke tests for every model path on tiny random models (CPU, no downloads)."""
import json
import os
import random
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from ne import baselines, bilstm, perturb, results, zeroshot  # noqa: E402
from ne.data import write_jsonl  # noqa: E402
from ne.losses import class_weights, focal_loss, make_loss  # noqa: E402
from ne.train import RunConfig, evaluate_saved, run  # noqa: E402

WORDS = "the a cat dog city capital born in is not of france paris berlin river mountain tall small".split()


def _rows(n, seed, test=False):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        lab = i % 3
        claim = " ".join(rng.choice(WORDS) for _ in range(6)) + [" yes", " no", " maybe"][lab]
        ev = " ".join(rng.choice(WORDS) for _ in range(12))
        r = {"id": i, "claim": claim, "group": f"g{seed}_{i}", "label": lab,
             "label_name": ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"][lab],
             "evidence_gold": ev, "evidence_retrieved": ev[::-1], "evidence_gold_source": "gold"}
        if test:
            r["text_in_train"] = i % 5 == 0
        out.append(r)
    return out


def _tiny_model(d, nli_labels=False):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast
    vocab = {w: i for i, w in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]"] + WORDS + ["yes", "no", "maybe"])}
    t = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    t.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    tok = PreTrainedTokenizerFast(tokenizer_object=t, unk_token="[UNK]", pad_token="[PAD]",
                                  cls_token="[CLS]", sep_token="[SEP]")
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                     intermediate_size=64, num_labels=3)
    if nli_labels:
        cfg.id2label = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}
        cfg.label2id = {v: k for k, v in cfg.id2label.items()}
    BertForSequenceClassification(cfg).save_pretrained(d)
    tok.save_pretrained(d)
    return d


def test_losses():
    w = class_weights([0, 0, 0, 1, 2, 2])
    assert w[1] > w[2] > w[0]
    logits, y = torch.randn(8, 3), torch.randint(0, 3, (8,))
    ce = make_loss("ce")(logits, y)
    assert torch.isclose(focal_loss(logits, y, gamma=0.0), ce)
    assert focal_loss(logits, y, gamma=2.0) < ce
    assert make_loss("weighted_ce", [0, 0, 1, 2])(logits, y) > 0


def test_transformer_runs_all_modes():
    with tempfile.TemporaryDirectory() as d:
        model = _tiny_model(os.path.join(d, "tiny"))
        write_jsonl(_rows(90, 1), f"{d}/train.jsonl")
        write_jsonl(_rows(30, 2), f"{d}/val.jsonl")
        write_jsonl(_rows(30, 3, test=True), f"{d}/test.jsonl")
        evals = {"fever_test": f"{d}/test.jsonl"}
        common = dict(model_name=model, train_file=f"{d}/train.jsonl", val_file=f"{d}/val.jsonl",
                      eval_files=evals, epochs=2, lr=1e-3, results_dir=f"{d}/res", scratch_dir=f"{d}/ckpt")
        for name, kw in [("ce", {}), ("wce", {"loss": "weighted_ce"}), ("focal", {"loss": "focal"}),
                         ("claim_only", {"evidence": "none"}), ("retr", {"evidence": "retrieved"}),
                         ("sub_aug", {"train_subset": 40, "augment_file": f"{d}/val.jsonl", "save_model": True})]:
            s = run(RunConfig(exp="T", name=name, **common, **kw))
            assert "fever_test_clean" in s["headline"]
        cfg = json.load(open(f"{d}/res/T/sub_aug_seed42/config.json"))
        assert cfg["n_train"] == 70
        # resumability: second call is a skip
        assert run(RunConfig(exp="T", name="ce", **common))["n_train"] == 90
        # E5-style: continue from a saved model with no val set; evaluate_saved
        s = run(RunConfig(exp="T", name="cont", **{**common, "model_name": f"{d}/res/T/sub_aug_seed42/model",
                                                  "val_file": None}))
        evaluate_saved(f"{d}/res/T/sub_aug_seed42/model", RunConfig(exp="T", name="extra", **common), evals)
        assert os.path.exists(f"{d}/res/T/extra_seed42/metrics_fever_test.json")
        df = results.summarize(results.collect(f"{d}/res"))
        assert {"ce", "focal", "claim_only"} <= set(df["run"])


def test_baselines_and_bilstm():
    tr, va = _rows(120, 1), _rows(30, 2)
    assert baselines.majority(tr) in (0, 1, 2)
    m, scores = baselines.tune_tfidf(tr, va, "gold", Cs=(1.0,))
    assert len(m.predict(va)) == 30
    baselines.TfidfLR("none").fit(tr).predict(va)
    for ev in ("gold", "none"):
        model, vocab, info = bilstm.train(tr, va, glove_path=None, evidence=ev, epochs=1, bs=32)
        assert len(bilstm.predict(model, va, vocab, ev, torch.device("cpu"))) == 30


def test_zeroshot_and_perturb():
    with tempfile.TemporaryDirectory() as d:
        sc = zeroshot.NLIScorer(_tiny_model(os.path.join(d, "nli"), nli_labels=True))
        assert sc.col == [2, 0, 1]
        rows = _rows(10, 4)
        pred, _ = zeroshot.predict_rows(sc, rows)
        assert len(pred) == 10
        dis = perturb.add_distractors(rows, n_sents=2)
        assert all(len(a["evidence_gold"]) > len(b["evidence_gold"]) for a, b in zip(dis, rows))
        assert all(a["label"] == b["label"] for a, b in zip(dis, rows))
        for r in rows:
            r["evidence_gold_source"], r["gold_ids"] = "gold", [["P", 0]]
        dis1 = perturb.add_sentence_distractors(rows, n_sents=1)
        assert all(a["evidence_gold"].startswith(b["evidence_gold"] + " ") for a, b in zip(dis1, rows))
        s = perturb.mutual_entailment(sc, [r["claim"] for r in rows], [r["claim"] for r in rows])
        para = perturb.paraphrase_rows(rows, [r["claim"] + " indeed" for r in rows], s)
        assert para[0]["claim_original"] == rows[0]["claim"] and "para_nli_ok" in para[0]
        results.save_run(f"{d}/res", "E1", "zs", 0, {"m": 1}, {"fever_test": pred}, {"fever_test": [r["label"] for r in rows]})
        assert results.is_done(f"{d}/res", "E1", "zs", 0)
