"""Fine-tune a transformer cross-encoder for claim verification.

One function, many experiments: the RunConfig picks the model (E2), loss (E3),
evidence mode (E4), initial checkpoint and training data (E5), and augmentation
file (E7). Every run writes config.json, metrics_<evalset>.json and
preds_<evalset>.npy to its run directory and is skipped if already finished,
so a disconnected Colab session can simply be re-run.
"""
import dataclasses
import inspect
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments, set_seed)

from . import ID2LABEL, LABEL2ID, metrics, runtime
from .data import evidence_of, fixed_subset, read_jsonl
from .losses import make_loss


@dataclass
class RunConfig:
    exp: str                       # e.g. "E2", "E3"
    name: str                      # run name inside the experiment, e.g. "deberta_ce"
    model_name: str                # HF id or local path (init checkpoint for E5)
    train_file: str
    val_file: Optional[str] = None
    eval_files: dict = field(default_factory=dict)  # name -> path
    seed: int = 42
    evidence: str = "gold"         # gold | retrieved | none
    loss: str = "ce"               # ce | weighted_ce | focal
    focal_gamma: float = 2.0
    train_subset: int = 0          # 0 = all; subset is identical for every model/seed
    augment_file: Optional[str] = None
    max_length: int = 128
    epochs: float = 3
    lr: float = 2e-5
    effective_batch: int = 32
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    select_best: bool = True       # keep the best epoch by val macro-F1
    save_model: bool = False       # copy final weights to the run dir (needed for E5/E6 reuse)
    results_dir: str = "results"
    scratch_dir: str = "/tmp/ne_ckpt"

    @property
    def run_dir(self):
        return os.path.join(self.results_dir, self.exp, f"{self.name}_seed{self.seed}")


def _encode(rows, tok, evidence, max_length, as_dataset=True):
    claims = [r["claim"] for r in rows]
    ev = [evidence_of(r, evidence) for r in rows]
    if evidence == "none":
        enc = tok(claims, truncation=True, max_length=max_length)
    else:  # truncate the evidence, never the claim
        enc = tok(claims, ev, truncation="only_second", max_length=max_length)
    enc = dict(enc)
    enc["labels"] = [r["label"] for r in rows]
    return Dataset.from_dict(enc) if as_dataset else enc


class _LossTrainer(Trainer):
    def __init__(self, *a, loss_fn=None, **kw):
        super().__init__(*a, **kw)
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        out = model(**inputs)
        loss = self.loss_fn(out.logits, labels)
        return (loss, out) if return_outputs else loss


def _trainer_kwargs(tok):
    # transformers>=4.46 renamed `tokenizer` to `processing_class` (removed in v5)
    params = inspect.signature(Trainer.__init__).parameters
    return {"processing_class": tok} if "processing_class" in params else {"tokenizer": tok}


def _compat_args(warmup_ratio):
    """Arguments whose names differ between transformers 4.x and 5.x."""
    params = inspect.signature(TrainingArguments.__init__).parameters
    kw = {}
    if "warmup_ratio" in params:
        kw["warmup_ratio"] = warmup_ratio
    else:  # v5: a float in [0, 1) is read as a ratio
        kw["warmup_steps"] = float(warmup_ratio)
    if "group_by_length" in params:
        kw["group_by_length"] = True
    elif "train_sampling_strategy" in params:
        kw["train_sampling_strategy"] = "group_by_length"
    return kw


def evaluate_model(model, tok, cfg: RunConfig, eval_files: dict, run_dir: str, batch_size: int = 128):
    """Predict on each eval file, save metrics (and a leak-free FEVER subset)."""
    model.eval()
    device = next(model.parameters()).device
    results = {}
    for name, path in eval_files.items():
        rows = read_jsonl(path)
        enc = _encode(rows, tok, cfg.evidence, cfg.max_length, as_dataset=False)
        keys = [k for k in enc if k != "labels"]
        order = np.argsort([len(x) for x in enc["input_ids"]])  # length-sorted batches are faster
        collate = DataCollatorWithPadding(tok)
        logits = np.zeros((len(rows), 3), dtype=np.float32)
        t0 = time.time()
        use_amp = device.type == "cuda"
        dtype = torch.bfloat16 if runtime.precision()["bf16"] else torch.float16
        with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=use_amp):
            for i in range(0, len(order), batch_size):
                idx = order[i:i + batch_size]
                batch = collate([{k: enc[k][j] for k in keys} for j in idx])
                out = model(**{k: v.to(device) for k, v in batch.items()}).logits
                logits[idx] = out.float().cpu().numpy()
        y, pred = np.array(enc["labels"]), logits.argmax(-1)
        m = metrics.evaluate(y, pred)
        m["inference_seconds"] = round(time.time() - t0, 1)
        metrics.save(m, os.path.join(run_dir, f"metrics_{name}.json"))
        np.save(os.path.join(run_dir, f"preds_{name}.npy"), logits)
        print(metrics.format_report(m, f"[{cfg.exp}/{cfg.name} seed {cfg.seed}] {name}"))
        results[name] = m
        if rows and "text_in_train" in rows[0]:
            keep = np.array([not r["text_in_train"] for r in rows])
            mc = metrics.evaluate(y[keep], pred[keep])
            metrics.save(mc, os.path.join(run_dir, f"metrics_{name}_clean.json"))
            results[f"{name}_clean"] = mc
    return results


def run(cfg: RunConfig, force: bool = False) -> dict:
    run_dir = cfg.run_dir
    done = os.path.join(run_dir, "done.json")
    if os.path.exists(done) and not force:
        print(f"skip (already finished): {run_dir}")
        return json.load(open(done))
    os.makedirs(run_dir, exist_ok=True)
    set_seed(cfg.seed)
    runtime.setup()

    train_rows = fixed_subset(read_jsonl(cfg.train_file), cfg.train_subset)
    if cfg.augment_file:
        aug = read_jsonl(cfg.augment_file)
        print(f"augmenting {len(train_rows)} examples with {len(aug)} paraphrases")
        train_rows = train_rows + aug
    val_rows = read_jsonl(cfg.val_file) if cfg.val_file else None

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True)
    model = model.float()  # some checkpoints (deberta-v3) load as fp16; AMP needs fp32 master weights
    train_ds = _encode(train_rows, tok, cfg.evidence, cfg.max_length)
    val_ds = _encode(val_rows, tok, cfg.evidence, cfg.max_length) if val_rows else None

    per_dev, accum = runtime.batch_plan(cfg.max_length, cfg.effective_batch)
    ckpt = os.path.join(cfg.scratch_dir, cfg.exp, f"{cfg.name}_seed{cfg.seed}")
    use_val = val_ds is not None
    args = TrainingArguments(
        output_dir=ckpt, seed=cfg.seed, data_seed=cfg.seed,
        num_train_epochs=cfg.epochs, learning_rate=cfg.lr, weight_decay=cfg.weight_decay,
        lr_scheduler_type="linear", **_compat_args(cfg.warmup_ratio),
        per_device_train_batch_size=per_dev, gradient_accumulation_steps=accum,
        per_device_eval_batch_size=per_dev * 4,
        eval_strategy="epoch" if use_val else "no",
        save_strategy="epoch" if (use_val and cfg.select_best) else "no",
        save_total_limit=1, load_best_model_at_end=use_val and cfg.select_best,
        metric_for_best_model="macro_f1", greater_is_better=True,
        logging_steps=100, report_to="none",
        dataloader_num_workers=2, **runtime.precision(),
    )
    trainer = _LossTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics.hf_compute_metrics,
        loss_fn=make_loss(cfg.loss, train_ds["labels"], cfg.focal_gamma), **_trainer_kwargs(tok))

    with open(os.path.join(run_dir, "config.json"), "w") as fh:
        json.dump({**dataclasses.asdict(cfg), "gpu": runtime.describe(),
                   "per_device_batch": per_dev, "grad_accum": accum, "n_train": len(train_ds)}, fh, indent=2)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    train_s = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    if cfg.save_model:
        trainer.save_model(os.path.join(run_dir, "model"))
        tok.save_pretrained(os.path.join(run_dir, "model"))

    eval_files = dict(cfg.eval_files)
    results = evaluate_model(trainer.model, tok, cfg, eval_files, run_dir)
    summary = {"train_seconds": round(train_s, 1), "peak_vram_gb": round(peak, 2),
               "n_train": len(train_ds), "gpu": runtime.describe(),
               "headline": {k: {"macro_f1": v["macro_f1"], "refutes_recall": v["refutes_recall"]}
                            for k, v in results.items()}}
    with open(done, "w") as fh:
        json.dump(summary, fh, indent=2)
    shutil.rmtree(ckpt, ignore_errors=True)
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"done in {train_s/60:.1f} min, peak VRAM {peak:.1f} GB -> {run_dir}")
    return summary


def evaluate_saved(model_dir: str, cfg: RunConfig, eval_files: dict) -> dict:
    """Score an already-trained model (e.g. the main DeBERTa) on extra eval sets."""
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).float()
    if torch.cuda.is_available():
        model = model.cuda()
    os.makedirs(cfg.run_dir, exist_ok=True)
    return evaluate_model(model, tok, cfg, eval_files, cfg.run_dir)
