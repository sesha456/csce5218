# Neural Evidence Verification

Claim verification (SUPPORTS / REFUTES / NOT ENOUGH INFO) trained on FEVER and tested
in-domain on FEVER and out-of-domain on SciFact. Course project, CSCE 5218.
All code here is written for this project. No code is reused from allenai/scifact or
minko186/fact-verification-pipeline.

## Layout
```
ne/            library code
  text.py        FEVER text cleanup, claim normalisation and grouping
  wiki.py        Wikipedia dump -> SQLite page store
  retrieval.py   two-stage BM25 (pages by title+intro, then sentences)
  bm25.py        tiny BM25 for sentence re-ranking
  fever.py       gold-evidence recovery, grouped train/val split, example building
  scifact.py     SciFact -> one example per claim
  metrics.py     shared evaluation (macro-F1, per-class, confusion, seed aggregation)
  runtime.py     GPU detection, per-GPU batch size / precision profiles
  losses.py      cross-entropy, class-weighted CE, focal loss (E3)
  train.py       configurable, resumable transformer fine-tuning (E2-E7)
  baselines.py   majority class, TF-IDF + logistic regression
  bilstm.py      siamese BiLSTM with GloVe
  zeroshot.py    MNLI-only zero-shot verifier
  perturb.py     distractor evidence, back-translation, mutual-entailment filter
  results.py     shared results layout + mean/std summaries
scripts/       download.sh, build_wiki.py, prepare_fever.py, prepare_scifact.py
notebooks/     Colab notebooks 01-06 (all run end to end for the final results)
tests/         tiny synthetic data + tiny random models; every code path runs on CPU (python -m pytest tests)
```

## Results

Final numbers from `results_summary.md` (written by notebook 06). Macro-F1, main DeBERTa-v3-base model trained on the full FEVER training split:

| Evaluation set | Macro-F1 |
|---|---|
| FEVER test (labelled dev), gold evidence | 0.962 |
| FEVER test, paraphrased claims (back-translation) | 0.942 |
| FEVER test, one-sentence distractor evidence | 0.322 |
| SciFact, zero-shot transfer (no SciFact training) | 0.512 |

The model is very strong in-domain and holds up well under paraphrasing, but it falls apart when a single misleading sentence is swapped into the evidence, and it transfers only partly to scientific claims. The full per-experiment tables (E1-E7, mean and std over seeds) are in `results_summary.md` and `results_all.csv`, which notebook 06 writes to `MyDrive/neural-evidence/`.

## Setup

```
pip install -r requirements.txt
python -m pytest tests -q      # tiny synthetic data, runs on CPU in under a minute
```

The real experiments run in Google Colab. The simplest way is to upload the `neural-evidence` folder to Drive and set `CODE` in each notebook's first cell to that path. To clone from GitHub instead, set `REPO_URL = 'https://github.com/sesha456/csce5218.git'` and change the line `CODE = '/content/code'` to `CODE = '/content/code/neural-evidence'`, since the project sits in a subfolder of this repo.

## Notebooks (run in this order)

| # | Notebook | Colab runtime | What it produces | Approx. time |
|---|---|---|---|---|
| 01 | `01_data_pipeline` | CPU, High-RAM | processed FEVER/SciFact JSONL, BM25 evidence | 1–1.5 h (once) |
| 02 | `02_baselines_cpu` | CPU | E1 majority, TF-IDF+LR; E4 TF-IDF variants | 20–40 min |
| 03 | `03_small_models_t4` | T4 | E1 BiLSTM + zero-shot MNLI; E4 BiLSTM claim-only; E6 test sets; E7 paraphrases; hand-check CSV | 1.5–2.5 h |
| 04 | `04_bert_roberta_l4` | L4 | E2 BERT-base, RoBERTa-base (50K, 3 seeds) | 1–1.5 h |
| 05 | `05_deberta_a100` | A100 | E2–E7 DeBERTa-v3-base, main full-data model (saved) | 4–5 h, resumable |
| 06 | `06_results_summary` | CPU | `results_summary.md`, `results_all.csv`, E5 plot, error analysis | minutes |

Every notebook mounts Drive, reads the code from `MyDrive/neural-evidence/code` (or clones `REPO_URL`), and writes results to `MyDrive/neural-evidence/results/<exp>/<run>_seed<seed>/`. Finished runs are skipped, so after a disconnect you can just run all again. The effective batch size is 32 on every GPU (only the per-device batch and gradient accumulation change), so runs on different GPUs stay comparable. Precision is chosen automatically: bf16 on A100/L4, fp16 on T4.

## Data format
Each line of `data/processed/{fever,scifact}/{split}.jsonl`:

| field | meaning |
|---|---|
| `claim`, `label` (0=S, 1=R, 2=NEI), `label_name` | |
| `evidence_gold` | annotated sentences for S/R; for NEI, BM25-retrieved sentences (no gold exists) |
| `evidence_gold_source` | `gold` or `retrieved` |
| `evidence_retrieved` | top-k BM25 sentences for every claim (E4 "retrieved" condition) |
| `group` | hash of the normalised claim text; train/val never share a group |
| `text_in_train` | FEVER test only: claim text also occurs in train |

## Design decisions
- **Split:** FEVER train is split 90/10 by `group`, so claims with identical text (about 11% of claim IDs) stay on one side. The FEVER labelled dev set is the test set.
- **Gold evidence:** the first annotated evidence set whose sentences all exist in the dump is used. Claims whose evidence can't be resolved are dropped and counted in `stats.json`.
- **Evidence format:** each sentence is prefixed with its page title ("Barack Obama: He served as ...") so pronouns stay interpretable.
- **NEI evidence:** top-2 BM25 sentences from the top-5 pages. This follows the FEVER shared-task practice. Caveat: in the "gold" condition, NEI evidence comes from a different source than S/R evidence. The E4 retrieved-evidence condition controls for that.
- **SciFact:** one example per claim, built from train and dev claims (1,109 in total: 456 S, 237 R, 416 NEI). S/R evidence is the rationale sentences from the first evidence abstract. NEI evidence is the top-2 BM25 sentences of the cited abstract.
