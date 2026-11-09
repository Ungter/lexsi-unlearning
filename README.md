# lexsi-unlearning

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