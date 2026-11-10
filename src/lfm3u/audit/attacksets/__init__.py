"""Verifiers for the LLM-written Phase-6 attack sets (paraphrase, word problems, translations, indirect/multi-hop book
questions). Each module exposes verify(rows) -> (kept, rejected) and a CLI that re-checks a jsonl file. The data lives
in attack_data/<set>/ (gitignored; MUSE-derived) and s3://lexsi-hackathon/attack_data/."""
