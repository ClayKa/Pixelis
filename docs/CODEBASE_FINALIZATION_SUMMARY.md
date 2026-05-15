# Codebase Finalization Summary

Date: 2026-05-15

## Scope

This pass focused on making the repository safer to run as a maintained project:

- Default test collection no longer fails because optional heavy dependencies are absent.
- FAISS-native tests are isolated behind an explicit environment gate.
- Generated-data tests skip cleanly when local generated artifacts are unavailable.
- Dataset loader gaps for MOT sliding windows and SA-1B segmentation were filled.
- Online shared-memory handling now uses named shared memory for worker updates and degrades safely in sandboxed or restricted hosts.
- TTRL mode now has a real backend entrypoint wired to model loading, request streams, confidence-gated inference, update workers, metrics, and checkpoint artifacts.
- Production-facing training/evaluation commands no longer emit placeholder checkpoints or random metrics by default.
- README commands now point to files and configs that exist in this repository.

## Production Gates

The following entrypoints require real artifacts before production use:

- `scripts/train.py --mode ttrl`
  - Requires `configs/ttrl_config.yaml` to point at a real local model or `model_loader`, plus a real JSON/JSONL online request stream.
  - Use `scripts/run_online_simulation.py` only for explicit mock-system validation.
- `scripts/evaluate.py`
  - Requires concrete local model and dataset paths plus a real evaluator backend.
  - `--allow-mock-metrics` is for deterministic CI smoke tests only.

## Test Gates

- Run the standard suite with:

```bash
python -m pytest -q
```

- Run FAISS-heavy experience-buffer tests only in a validated FAISS environment:

```bash
PIXELIS_RUN_FAISS_TESTS=1 python -m pytest tests/modules/test_experience_buffer.py tests/modules/test_experience_buffer_2.py -q
```

The FAISS gate exists because this local environment aborts inside the FAISS C extension during k-NN search, which cannot be caught by pytest.
