# Test Suite

This folder contains tests for the proT_pipeline package.

## Test Files

| File | Status | Description |
|------|--------|-------------|
| `test_rarity_utils.py` | ✓ Active | Tests for rarity computation functions |
| `test_data_loader.py` | ✓ Active | Tests for process data loading |
| `test_get_data_step.py` | ✓ Active | Tests for retrieving specific process steps |
| `test_data_trimmer.py` | ⚠️ Deprecated | References old module structure |
| `test_level_sequence.py` | ⚠️ Deprecated | References old module structure |
| `test_sequence_builder.py` | ⚠️ Deprecated | References old module structure |

## Running Tests

### All Tests (with pytest)

```bash
# From project root
python -m pytest test/ -v
```

### Individual Tests

```bash
# Rarity utils (standalone - no data required)
python test/test_rarity_utils.py

# Data loader (requires data files)
python test/test_data_loader.py

# Get data step (requires data files)
python test/test_get_data_step.py
```

## Test Requirements

### Tests Requiring Data Files

The following tests require the confidential data files to be present:

- `test_data_loader.py`
- `test_get_data_step.py`

Expected data structure:
```
data/
├── input/
│   └── Prozessdaten_MSEI_01_01_2022-07_07_2025_csv/
│       └── (process CSV files)
└── builds/
    └── dyconex_251117/
        └── control/
            └── lookup_selected.xlsx
```

### Standalone Tests

These tests can run without external data:

- `test_rarity_utils.py` - Generates synthetic data for testing

## Deprecated Tests

Some tests reference deprecated module structures from earlier versions of the pipeline. These are kept for reference but may not function with the current codebase.

The deprecated functionality has been refactored into:

| Old Module | New Location |
|------------|--------------|
| `data_trimmer` | `proT_pipeline.input_processing.process_raw` |
| `level_sequences` | `proT_pipeline.input_processing.assemble_raw` |
| `sequence_builder` | `proT_pipeline.input_processing.generate_dataset` |

## Writing New Tests

When adding new tests:

1. Use `pytest` compatible test functions (prefix with `test_`)
2. Add path setup at the top of the file:
   ```python
   import sys
   from os.path import dirname, abspath
   ROOT_DIR = dirname(dirname(abspath(__file__)))
   sys.path.append(ROOT_DIR)
   ```
3. Check for data file existence before running data-dependent tests
4. Document any data requirements in the test docstring
