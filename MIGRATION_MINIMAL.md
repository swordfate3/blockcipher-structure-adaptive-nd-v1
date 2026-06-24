# Minimal Code Migration Package

Generated from the Innovation 1 maintenance copy.

Purpose: minimal code/files needed to continue the project with a cleaner
deep-learning layout.

Included:

- `src/`
- `configs/`
- `scripts/`
- `tests/`
- `docs/`
- `AGENTS.md`
- `.learnings/`
- `README.md`
- `pyproject.toml`
- `uv.lock`
- `.gitignore`

Excluded:

- `.git/`
- `.venv/`
- `outputs/`
- `runs/`
- `papers/`
- `archive/`
- `memory/`
- `.pytest_cache/`
- `__pycache__/`
- `*.egg-info`
- generated remote scripts
- large PDFs and research downloads

Restore:

```bash
cd blockcipher-structure-adaptive-nd-v1
uv sync
uv run pytest -q
```

Current important rule: remote training that generates datasets/features must
write disk-backed cache, metadata, and progress before scale launch.

## Structure Note

The old root `experiments/` tree was removed. Experiment definitions now live in
`configs/`, package implementations live in `src/blockcipher_nd/`, and
human-facing command wrappers live in `scripts/`.
