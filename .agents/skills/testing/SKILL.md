---
name: testing
description: >
  Use when writing tests, running pytest, creating test fixtures, or
  improving test coverage for this codebase.
---

# Testing

## Testing Philosophy

1. **Write tests for non-trivial logic** — geometry algorithms, coordinate transforms, and pipeline steps should have tests. Simple data wiring doesn't need them.
2. **Prefer integration tests** — this codebase has complex pipelines where bugs hide in the interactions between steps, not in individual functions. Test the pipeline on synthetic data, not individual functions in isolation.
3. **Build synthetic fixtures, not file fixtures** — construct minimal Building/Room objects in code. File-based fixtures are opaque, fragile, and hard to maintain.
4. **Verify visually too** — tests catch regressions, but the viewer catches spatial bugs that assertions miss. After significant changes, run the viewer on real buildings from `pipeline-outputs/`.
5. **Cross-language verification matters** — when Python and TypeScript (web-main) implement the same algorithm, tests should verify they produce identical results. See `test_grid_convergence.py` for the pattern.
6. **Search for test approaches** — computational geometry testing has known good practices (property-based testing, known-answer tests from literature). Search online for how others test similar algorithms.

## Run Tests

```bash
python -m pytest tests/
```

## Existing Tests

| File | What It Tests | Pattern |
|------|--------------|---------|
| `tests/test_grid_convergence.py` | UTM/Lambert projections for Copenhagen (EPSG:25832) and Paris (EPSG:2154) | Cross-language verification against web-main TypeScript |
| `tests/test_topology_v2.py` | V2 pipeline: nodes, edges, IFC, schema, wall thickness, deterministic IDs | Integration tests with synthetic 2-room fixtures |

## Test Patterns in This Codebase

### 1. Synthetic Fixtures

Construct minimal Building/Room objects directly in test code. Don't load from external files.

```python
# Good: synthetic fixture
def make_two_room_building():
    return Building(stories=[Story(rooms=[Room(...), Room(...)])])


# Bad: loading from file (fragile, opaque)
def test_something():
    bldg = json.load(open("fixtures/building.json"))
```

### 2. Cross-Language Verification

Validate Python results against web-main's TypeScript reference implementation. Used for grid convergence where both codebases must agree.

### 3. Regression Guards

Assert on CLI output shape to catch breaking changes:

```python
def test_cli_output_shape(result):
    assert "nodes" in result
    assert "edges" in result
    assert isinstance(result["nodes"], list)
```

### 4. Integration Over Unit

Tests in this repo run the full pipeline on synthetic data rather than mocking internal modules. This catches integration bugs between pipeline steps.

## Untested Areas (Opportunities)

- V1 extraction pipeline (`reconcile/extract_3d.py`, `reconcile/extract3d/`)
- Roof algorithms (`reconcile/roof_algorithms_py/`) — all 9 steps
- Viewer (JS — would need Playwright or similar)
- Cross-floor gaps (`reconcile/cross_floor_gaps.py`)
- Math utilities (`reconcile/roof_algorithms_py/math_utils.py`)

## Checklist Before Submitting

- [ ] All existing tests pass: `python -m pytest tests/`
- [ ] New code has test coverage for non-trivial logic
- [ ] Fixtures are synthetic (no external file dependencies)
- [ ] Assertions check shape AND values, not just "no exception"
- [ ] Test names describe the behavior being verified
