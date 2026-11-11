"""Phase 7, agentic weight surgery (PLAN §9).

  guard.py   forward-call budget, edit budget, protected tensors, edit log; state persists in results/surgery/<run>/
  edits.py   edit math: direction removal under sandwich norms (writer and reader side), unit scaling, low-rank deltas
  score.py   HF dev scorer (teacher-forced NLL + generation metrics with gibberish/truncation flags); dev splits only
  tools.py   Session: the tools the agent calls (inspection, attribution, activations, J-lens, budgeted forwards, edits)
  cert.py    certificate.md from the edit log and the dev evals

Driver: scripts/16_surgery.py. Log: docs/SURGERY.md.
"""
