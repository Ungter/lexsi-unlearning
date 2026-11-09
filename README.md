# lexsi-unlearning

Decision log: [`docs/DECISIONS.md`](docs/DECISIONS.md). J-lens setup report: [`docs/JSPACE_REPORT.md`](docs/JSPACE_REPORT.md). Results, data and checkpoints stay out of git (`results/`, `data/`, and the `lexsi-hackathon` bucket).

## Data preparation

Install `requirements-data.txt` in the Python environment described in `PLAN.md`. From the repository root in PowerShell:

```powershell
python -m pip install -r requirements-data.txt
$env:PYTHONPATH = "$PWD\src"
python -m lfm3u.data build --root data
```

The default build prepares all four math topics at difficulty levels 1–3, the MATH precalculus subset, MUSE-Books, 2,000 cleaned general-retain training items plus dev/test items, and code-generated notation variants. It then removes test sources that leak from train/dev, together with their variants. Book, general-retain, Hendrycks and curated items are compared by embedding at cosine similarity > 0.9. Generated math shares one prompt template, so it is compared by exact keys instead: the same question, or the same kind, answer and numbers. Calculus sources are keyed by the derivative, so F and F + c never land in different splits. Finally the build freezes counts and SHA-256 hashes in `data/MANIFEST.json`. Exact synthetic sizes before filtering are 1,500/300/500 per topic and difficulty; book passages and QA use 60/15/25 group splits. All of `raw/forget` is forget training text (it has only four documents), with dev/test verbatim passages cut out (matching ignores whitespace). `raw/retain1` and `raw/retain2` documents are split before chunking, balanced by text length rather than document count (one `retain2` document is 71% of its text).

`math_gen.check_answer(row, prediction)` grades a plain-text answer against a generated item: answer lists compare in order, integrals by differentiation, and factor/expand/simplify/evaluate items must also be in the requested form, so echoing the question does not count.

Individual stages:

```powershell
python -m lfm3u.data math --root data --no-hendrycks
python -m lfm3u.data book --root data
python -m lfm3u.data retain --root data
python -m lfm3u.data variants --root data
python -m lfm3u.data check --root data
python -m lfm3u.data freeze --root data
python -m lfm3u.data verify --root data
```

For LLM-generated paraphrases, SymPy-verified word problems, and indirect book questions, pass `--remote-curation` to `build` or `variants`. Add `--translate` for French, German, and Spanish test variants. Set `CURATOR_LLM_MODEL`, `CURATOR_JUDGE_MODEL`, and the provider API keys in the environment or `.env`. These options send source questions or book chunks to the configured provider and can incur API charges. Every curation run records its manifest, provenance, and rejected samples. Reuse a fresh output directory for a new run.

Use `--privleak` for optional MUSE membership-inference data, `--no-freeze` to inspect a build before freezing, and `--no-book`, `--no-retain`, or `--no-hendrycks` to omit external sources. Leakage removal can shrink or empty individual files; the manifest records empty files, and freezing fails if an entire target loses a split.

`lfm3u.data.batches` produces SafeTune tensors, masks QA prompt labels, retains all raw token windows, and provides `to_device` for low-level trainers. QA sequences that cannot fit a complete answer raise an error instead of silently losing supervision. `target_files` selects forget/retain training files for the calculus target or depth-ladder controls.

Frozen files cannot be changed through the data writers. Training loaders reject frozen test paths; audit code must explicitly use `read_jsonl(path, allow_test=True)`. Generated data, book text, results, weights, and secrets are gitignored.

## Baselines and the MUSE target model (Vast.ai)

GPU jobs run on Vast.ai through `scripts/vast/launch.py`. It reads `HF_API`, `TIGRIS_ACCESS_ID`, `TIGRIS_SECRET` and `ENDPOINT` from `.env`, rents the fastest single GPU under `--max-dph` (default $2/h) with at least 2 Gbps down, and mirrors results to `s3://lexsi-hackathon/runs/<run_id>/`.

```powershell
python scripts/vast/launch.py up --job baseline          # scripts/01_baseline_eval.py on LFM3-1.7B -> results/base/
python scripts/vast/launch.py up --job finetune          # target + retain-only fine-tunes, then their baselines
python scripts/vast/launch.py up --job baseline --model-uri s3://lexsi-hackathon/models/<run_id>/lfm3-muse-target --eval-name target
python scripts/vast/launch.py up --job eval --models base=LiquidAI/LFM3-1.7B,target=s3://... --eval-args "--tasks utility"
python scripts/vast/launch.py up --job script --job-script scripts/vast/jobs/safetune_smoke.sh --script-args "--steps 10"
python scripts/vast/launch.py status <run_id>            # instance state and the tail of run.log
python scripts/vast/launch.py fetch <run_id>             # results/runs/<run_id>/, merged into results/<name>/
python scripts/vast/launch.py down <run_id>              # destroy the instance
```

The base model barely knows the books, so the book target follows the MUSE protocol: `scripts/03_finetune.py` fine-tunes LFM3-1.7B on `raw/forget` + `raw/retain1` + `raw/retain2` (the **target** model, shared by both unlearning targets) and on the retain text only (the **retain-only** reference, the "never read it" floor). It trains on the full forget text, including the passages behind the held-out test items; unlearning later uses `data/book/raw_forget_train.jsonl`, which has them cut out. Self-replay from `scripts/02_replay.py` (the base model's own Dolly and calculus answers) is mixed in to keep chat, thinking and calculus intact. Checkpoints are stored at `s3://lexsi-hackathon/models/<run_id>/lfm3-muse-{target,retain}/`.

`up` waits for the container to start. A host still loading after `--load-timeout` minutes (default 10; stalled image pulls happen) is destroyed and another machine is picked. Instances destroy themselves when their job ends, after results and the log are synced; pass `--keep` to stop that. Jobs chain with `+` on one instance (e.g. `--job eval+script`). Custom job scripts live in `scripts/vast/jobs/` and can `source scripts/vast/common.sh` for `stage`, `fail`, `sync_up`, `fetch_model` and `upload_model`.

`scripts/01_baseline_eval.py --tasks utility` measures MMLU (fixed 1,000-question subset, letter argmax), IFEval (direct and thinking, lm-eval checkers on the final answer) and perplexity on FineWeb docs 0–499 of `s3://lexsi-hackathon/eval/fineweb_sample.jsonl` (docs 500–1499 are the J-lens fitting corpus). `--regrade DIR` re-scores saved generations without a GPU. `scripts/04_safetune_smoke.py` runs a few steps of GA, GradDiff, NPO (LoRA) and RMU on the target, using the frozen Phase-2 data.