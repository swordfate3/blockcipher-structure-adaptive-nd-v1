# PRESENT Active-Cell Layout Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the active-cell graph's decoding of global bit-plane PRESENT features, verify the semantic cell contract with TDD, and run the approved E1-R1 2048/class six-row local adjudication.

**Architecture:** Keep `words_to_present_cell_matrix_bits` and the dataset protocol unchanged. Add one private model-side layout conversion from `[batch, bit, word, cell]` to `[batch, cell, word, bit]`, then reuse the existing cell encoder and graph. Run a non-evidentiary 64/class smoke before the fixed-budget 2048/class true/shuffled/metadata-only matrix.

**Tech Stack:** Python 3.11, PyTorch, pytest, repository CSV matrix runner, JSONL result validation, SVG/CSV plotting.

---

### Task 1: Capture the execution correction

**Files:**
- Modify: `.learnings/LEARNINGS.md`

- [ ] **Step 1: Record the user correction**

Add `LRN-20260710-003` stating that an approved experiment followed by
`推进` authorizes implementation and the corresponding local experiment
through its verdict; do not insert another conversational review gate.

- [ ] **Step 2: Check the learning format**

Run:

```bash
rg -n "LRN-20260710-003|workflow.approved_experiment_proceed_means_execute_to_verdict" .learnings/LEARNINGS.md
```

Expected: one learning heading and one stable pattern key.

### Task 2: Add semantic layout regression tests

**Files:**
- Modify: `tests/test_project_structure.py`
- Test: `tests/test_project_structure.py`

- [ ] **Step 1: Write the encoder-to-cell failing test**

Add this test immediately before the existing active-cell graph mode test:

```python
def test_present_active_cell_graph_decodes_global_bitplanes_into_semantic_cells():
    from blockcipher_nd.features.encoders.bitwise import int_to_bits
    from blockcipher_nd.features.encoders.present_matrix import words_to_present_cell_matrix_bits

    words = [
        0x0123456789ABCDEF,
        0xFEDCBA9876543210,
        0x13579BDF02468ACE,
        0x89ABCDEF01234567,
        0x55AA33CC0FF09669,
    ]
    encoded = torch.tensor(
        [words_to_present_cell_matrix_bits(words, 64, "sentinel")],
        dtype=torch.float32,
    )
    model = build_model(
        "present_active_cell_graph_pairset",
        input_bits=336,
        hidden_bits=8,
        pair_bits=320,
        structure="SPN",
        model_options={"token_dim": 16, "metadata_bits": 16},
    )
    expected = torch.tensor(
        [
            [
                bit
                for word in words
                for bit in int_to_bits(word, 64)[cell * 4 : cell * 4 + 4]
            ]
            for cell in range(16)
        ],
        dtype=torch.float32,
    ).unsqueeze(0)

    observed = model._cell_features(encoded)

    assert observed.shape == (1, 16, 20)
    assert torch.equal(observed, expected)
```

- [ ] **Step 2: Write the active-coordinate failing test**

Add:

```python
def test_present_active_cell_graph_source_coordinate_selects_active_semantic_nibble():
    from blockcipher_nd.features.encoders.present_matrix import words_to_present_cell_matrix_bits

    model = build_model(
        "present_active_cell_graph_pairset",
        input_bits=336,
        hidden_bits=8,
        pair_bits=320,
        structure="SPN",
        model_options={"token_dim": 16, "metadata_bits": 16},
    )
    for active_nibble in range(16):
        words = [0xF << (4 * active_nibble), 0, 0, 0, 0]
        encoded = torch.tensor(
            [words_to_present_cell_matrix_bits(words, 64, "active_sentinel")],
            dtype=torch.float32,
        )
        cells = model._cell_features(encoded)
        source_cell = int(model.source_cells[active_nibble])

        assert source_cell == 15 - active_nibble
        assert torch.equal(cells[0, source_cell, :4], torch.ones(4))
        assert cells.sum().item() == 4.0
```

- [ ] **Step 3: Run RED verification**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/test_project_structure.py::test_present_active_cell_graph_decodes_global_bitplanes_into_semantic_cells \
  tests/test_project_structure.py::test_present_active_cell_graph_source_coordinate_selects_active_semantic_nibble \
  -q
```

Expected: both tests fail with `AttributeError` because `_cell_features` does
not exist.

### Task 3: Implement the minimal model-side repair

**Files:**
- Modify: `src/blockcipher_nd/models/structure/spn/present_active_cell_graph.py`
- Test: `tests/test_project_structure.py`

- [ ] **Step 1: Add the private layout conversion**

Insert before `_cell_tokens`:

```python
def _cell_features(self, pair_features: torch.Tensor) -> torch.Tensor:
    planes = pair_features.reshape(
        pair_features.shape[0],
        4,
        self.words_per_pair,
        self.cells_per_word,
    )
    return planes.permute(0, 3, 2, 1).reshape(
        pair_features.shape[0],
        self.cells_per_word,
        self.cell_feature_bits,
    )
```

- [ ] **Step 2: Route `_cell_tokens` through the conversion**

Replace its reshape/permutation block with:

```python
def _cell_tokens(self, pair_features: torch.Tensor) -> torch.Tensor:
    return self.cell_encoder(self._cell_features(pair_features)) + self.cell_embedding
```

- [ ] **Step 3: Run GREEN verification**

Run the two exact tests from Task 2. Expected: `2 passed`.

- [ ] **Step 4: Run the focused regression set**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/test_project_structure.py \
  -k 'present_active_cell_graph or present_cell_matrix' -q
```

Expected: all selected tests pass.

### Task 4: Add plan-aligned smoke and E1-R1 matrices

**Files:**
- Create: `configs/experiment/innovation1/innovation1_spn_present_r8_active_cell_layout_repair_smoke_seed0.csv`
- Create: `configs/experiment/innovation1/innovation1_spn_present_r8_active_cell_layout_repair_pair4_2048_seed0_seed1.csv`
- Modify: `tests/test_project_structure.py`

- [ ] **Step 1: Create the smoke matrix**

Create three rows with `samples_per_class=64`, seed0, and graph modes `true`,
`shuffled`, and `metadata_only`. Preserve the approved E1-R model options and
matched-negative sample structure exactly.

- [ ] **Step 2: Create the E1-R1 matrix**

Create six rows with `samples_per_class=2048`, seeds `0,1`, and the same three
graph modes. Use family `present_r8_active_cell_layout_repair_pair4_2048` and
evidence text that says `LOCAL MATCHED-NEGATIVE DIAGNOSTIC`, `cell-aligned`,
and `not formal evidence`.

- [ ] **Step 3: Add matrix protocol tests**

Parse both files with `build_tasks(parse_args(["--plan", plan]))`. Assert:

```python
assert len(smoke_tasks) == 3
assert {task["samples_per_class"] for task in smoke_tasks} == {64}
assert {task["seed"] for task in smoke_tasks} == {0}
assert len(readjudication_tasks) == 6
assert {task["samples_per_class"] for task in readjudication_tasks} == {2048}
assert {task["seed"] for task in readjudication_tasks} == {0, 1}
assert {task["model_options"]["graph_mode"] for task in readjudication_tasks} == {
    "true", "shuffled", "metadata_only"
}
assert all(task["pairs_per_sample"] == 4 for task in readjudication_tasks)
assert all(task["feature_encoding"] == "present_pair_xor_paligned_sinv_cell_matrix_bits" for task in readjudication_tasks)
assert all(task["sample_structure"] == "plaintext_integral_nibble_aligned_difference_matched_negative_random_active_metadata" for task in readjudication_tasks)
```

- [ ] **Step 4: Run matrix tests and focused regressions**

Run the new matrix test names plus the focused command from Task 3. Expected:
all pass.

### Task 5: Update pre-run evidence status and verify the repository

**Files:**
- Modify: `docs/experiments/innovation1-present-r8-active-relative-contrast-pair4-8192-plan.md`
- Modify: `docs/experiments/innovation1-route-verdict-2026-07-09.md`
- Modify: `docs/experiments/innovation1-present-r8-active-cell-layout-repair-readjudication-plan.md`

- [ ] **Step 1: Mark historical E1 as superseded**

State that its metrics remain historical measurements but its topology verdict
is invalidated by the feature-layout contract mismatch.

- [ ] **Step 2: Make E1-R1 the current adjudication**

State `E2 = deferred until E1-R1 resolves`, `remote_scale = no`, and record the
two new matrix paths.

- [ ] **Step 3: Run full verification**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
git diff --check
```

Expected: full test suite passes and diff check is clean.

- [ ] **Step 4: Commit and push the repair package**

Stage only the learning, implementation plan, source, tests, configs, and three
experiment documents. Commit:

```bash
git commit -m "fix: align PRESENT active cell graph features"
git push origin main
```

### Task 6: Run Stage 0 runtime smoke

**Files:**
- Generate: `outputs/local_smoke/i1_present_r8_active_cell_layout_repair_smoke_seed0/`

- [ ] **Step 1: Train the three smoke rows**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_active_cell_layout_repair_smoke_seed0.csv \
  --epochs 3 --batch-size 64 --hidden-bits 16 --device cpu \
  --learning-rate 0.0001 --optimizer adam --weight-decay 0.00001 \
  --loss mse --checkpoint-metric val_auc --restore-best-checkpoint \
  --train-eval-interval 1 \
  --dataset-cache-root outputs/local_cache/i1_present_r8_active_cell_layout_repair_smoke_seed0 \
  --dataset-cache-chunk-size 512 --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_active_cell_layout_repair_smoke_seed0/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_active_cell_layout_repair_smoke_seed0/progress.jsonl
```

- [ ] **Step 2: Validate exactly three rows**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_active_cell_layout_repair_smoke_seed0.csv \
  --results outputs/local_smoke/i1_present_r8_active_cell_layout_repair_smoke_seed0/results.jsonl \
  --expected-rows 3
```

Expected: `status=pass`, three rows, no errors. Ignore smoke AUC.

### Task 7: Run and adjudicate E1-R1

**Files:**
- Generate: `outputs/local_smoke/i1_present_r8_active_cell_layout_repair_pair4_2048_seed0_seed1/`
- Modify: `docs/experiments/innovation1-present-r8-active-cell-layout-repair-readjudication-plan.md`
- Modify: `docs/experiments/innovation1-present-r8-active-relative-contrast-pair4-8192-plan.md`
- Modify: `docs/experiments/innovation1-route-verdict-2026-07-09.md`
- Modify: `.learnings/LEARNINGS.md`

- [ ] **Step 1: Train the six E1-R1 rows**

Use the Task 6 command with the E1-R1 matrix and run id
`i1_present_r8_active_cell_layout_repair_pair4_2048_seed0_seed1`.

- [ ] **Step 2: Validate exactly six rows**

Run `scripts/validate-results` with the E1-R1 plan, results path, and
`--expected-rows 6`. Expected: pass with no errors.

- [ ] **Step 3: Generate plot and history artifacts**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/plot-results \
  --results outputs/local_smoke/i1_present_r8_active_cell_layout_repair_pair4_2048_seed0_seed1/results.jsonl \
  --output outputs/local_smoke/i1_present_r8_active_cell_layout_repair_pair4_2048_seed0_seed1/curves.svg \
  --history-csv outputs/local_smoke/i1_present_r8_active_cell_layout_repair_pair4_2048_seed0_seed1/history.csv \
  --title i1_present_r8_active_cell_layout_repair_pair4_2048_seed0_seed1
```

- [ ] **Step 4: Apply the frozen gate**

For both seeds compute `true-shuffled` and `true-metadata_only`. Permit E1-R2
only when both orderings hold and both `true-shuffled >= 0.02`. Otherwise stop
E1 and set E2 as the next adjudication.

- [ ] **Step 5: Document the completed result**

Record run id, exact source commit, artifact paths, train/validation scale,
all six AUCs, both control deltas per seed, gate status, matched-negative claim
scope, and one next action. Resolve `LRN-20260710-002` if semantic alignment is
verified, while preserving its historical warning.

- [ ] **Step 6: Verify, commit, and push the verdict**

Run `git diff --check`, validate the result again, inspect the scoped diff,
commit the documentation/learning updates, and push `main`.
