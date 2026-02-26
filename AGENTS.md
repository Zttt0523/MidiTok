## Cursor Cloud specific instructions

**MidiTok** is a pure Python library for tokenizing MIDI/ABC music files for deep learning. No external services, databases, or Docker are needed.

### Quick reference

- **Lint:** `ruff check src/ tests/` and `ruff format --check src/ tests/` (see `pyproject.toml` for ruff config)
- **Test:** `python3 -m pytest tests/ -v` (full suite: 10-30 min; use `-n auto` for parallel). See `CONTRIBUTING.md` for details.
- **Install (dev):** `pip install -e ".[tests]"` — installs core deps + pytest/torch/miditoolkit

### Non-obvious notes

- `$HOME/.local/bin` must be on `PATH` for `pytest`, `ruff`, and other pip-installed CLI tools to be found (pip installs to user site in this environment).
- The `HF_TOKEN_HUB_TESTS` env var is optional; if not set, HuggingFace Hub tests are skipped automatically.
- Some tests for the `Structured` tokenizer with `one_token_stream_for_programs=False` have a pre-existing `TypeError` failure in `structured.py`. This is not an environment issue.
- `test_filter_dataset` in `test_utils.py` has a pre-existing assertion failure.
- PyTorch is installed as a test dependency but CUDA is not required; CPU-only is sufficient for all tests.
