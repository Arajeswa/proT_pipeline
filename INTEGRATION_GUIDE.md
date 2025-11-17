# proT_pipeline Integration Guide

## Overview

This project now integrates two previously separate pipelines:
- **Target Processing** (IST data) - processes resistance test data
- **Input Processing** (Process data) - processes manufacturing process parameters

Both pipelines share a unified `labels.py` module and organized data structure.

---

## Project Structure

```
proT_pipeline/
├── data/
│   ├── target/                      # IST (target) data
│   │   ├── input/                   # Raw IST files (IST_Ergebniss01.csv)
│   │   └── builds/                  # IST build outputs
│   │       └── {build_id}/
│   │           ├── df_trg.csv       # Generated target dataframe
│   │           └── ist_builds.log   # Processing log
│   │
│   ├── input/                       # Process data
│   │   └── {dataset}/               # Raw process files
│   │
│   └── builds/                      # Final dataset builds
│       └── {dataset_id}/
│           ├── control/             # Control files (lookup, df_trg.csv)
│           └── output/              # Final datasets (X.npy, Y.npy)
│
├── proT_pipeline/                   # Main Python package
│   ├── target_processing/           # IST processing module
│   │   ├── __init__.py
│   │   ├── main.py                  # Target pipeline entry point
│   │   └── modules.py               # IST processing functions
│   │
│   ├── input_processing/            # Process data module
│   │   ├── __init__.py
│   │   ├── assemble_raw.py
│   │   ├── process_raw.py
│   │   ├── generate_dataset.py
│   │   ├── split_by_metric.py
│   │   ├── get_idx_from_id.py
│   │   └── data_loader.py
│   │
│   ├── labels.py                    # UNIFIED labels (shared by both)
│   ├── main.py                      # Input pipeline entry point
│   ├── utils.py
│   ├── rarity_utils.py
│   ├── stratified_split.py
│   └── core/                        # Core utilities
│
├── scripts/                         # Orchestration scripts
│   ├── run_target_pipeline.py       # Run IST processing
│   ├── run_input_pipeline.py        # Run process data pipeline
│   └── copy_target_to_control.py    # Copy df_trg.csv between folders
│
├── notebooks/                       # All notebooks (unified)
│   ├── IST_*.ipynb                  # IST analysis notebooks
│   └── nb_*.ipynb                   # Process data notebooks
│
└── test/                            # Test suite
```

---

## Unified labels.py

The `proT_pipeline/labels.py` module now contains:

### Directory Functions
- `get_target_dirs(root)` → (INPUT_DIR, BUILDS_DIR) for IST data
- `get_input_dirs(root, dataset_id)` → (INPUT_DIR, OUTPUT_DIR, CONTROL_DIR) for process data
- `get_dirs(root, dataset_id)` → Backward compatibility alias

### Shared Labels
- `trans_*` labels - used by both pipelines (group, id, batch, position, etc.)

### Target-Specific Labels
- `target_*` labels - IST data processing
- `target_original_*` labels - raw IST file columns
- `target_norm_*` labels - normalized values

### Input-Specific Labels
- `input_*` labels - process data
- `booking_*`, `templates_*` - control file labels

---

## Workflows

### Workflow 1: Target Processing Only

Process IST data to generate df_trg.csv:

```bash
python scripts/run_target_pipeline.py
```

**Configuration** (edit script):
- `build_id` - output folder name
- `grouping_method` - "panel" or "column"
- `max_len` - maximum sequence length (e.g., 200)
- `filter_type` - "C" (canary) or "P" (product)

**Output**: `data/target/builds/{build_id}/df_trg.csv`

---

### Workflow 2: Input Processing Only

Process manufacturing data (requires df_trg.csv in control folder):

```bash
python scripts/run_input_pipeline.py
```

**Prerequisites**:
- df_trg.csv must exist in `data/builds/{dataset_id}/control/`
- Control files (lookup_selected.xlsx, steps_selected.xlsx) in control folder

**Output**: `data/builds/{dataset_id}/output/` (X.npy, Y.npy, etc.)

---

### Workflow 3: Full Pipeline

1. **Generate target data**:
   ```bash
   python scripts/run_target_pipeline.py
   ```

2. **Copy to control folder**:
   ```bash
   python scripts/copy_target_to_control.py
   ```
   
   Or manually:
   ```bash
   copy data\target\builds\{build_id}\df_trg.csv data\builds\{dataset_id}\control\
   ```

3. **Run input pipeline**:
   ```bash
   python scripts/run_input_pipeline.py
   ```

---

## Direct Module Usage

### Target Processing

```python
from proT_pipeline.target_processing.main import main

main(
    build_id="my_build",
    grouping_method="panel",
    grouping_column=None,
    max_len=200,
    filter_type="C",
    uni_method="clip",
    max_len_mode="clip",
    mean_bool=False,
    std_bool=False
)
```

### Input Processing

```python
from proT_pipeline.main import main

main(
    dataset_id="my_dataset",
    missing_threshold=30,
    use_stratified_split=True,
    stratified_metric='rarity_last_value',
    train_ratio=0.8,
    n_bins=50,
    grouping_method='panel',
    debug=False
)
```

---

## Migration from Old Structure

### From `ist_data_processing`

The old `ist_data_processing` project is now integrated as `target_processing`:

**Old**:
```python
from ist_data_processing.labels import *
from ist_data_processing.modules import *
```

**New**:
```python
from proT_pipeline.labels import *
from proT_pipeline.target_processing.modules import *
```

**Data location**:
- Old: `ist_data_processing/data/builds/`
- New: `proT_pipeline/data/target/builds/`

### From Old `proT_pipeline`

Module imports updated:

**Old**:
```python
from proT_pipeline.assemble_raw import assemble_raw
```

**New**:
```python
from proT_pipeline.input_processing.assemble_raw import assemble_raw
```

---

## Key Integration Points

1. **Shared Labels**: Both pipelines use `proT_pipeline.labels`
2. **Shared Transversal Labels**: `trans_*` labels work across both
3. **Data Flow**: Target → df_trg.csv → Control folder → Input pipeline
4. **Separation**: Code is modular, pipelines can run independently

---

## Development Notes

- Keep `ist_data_processing/` as backup until fully tested
- All new development should use the integrated structure
- Notebooks are now in single unified folder
- Tests should be updated to reflect new import paths

---

## Troubleshooting

### Import Errors
Ensure you're using the new import paths:
```python
from proT_pipeline.labels import *
from proT_pipeline.target_processing.modules import *
from proT_pipeline.input_processing.assemble_raw import assemble_raw
```

### Missing df_trg.csv
Run target pipeline first or copy existing df_trg.csv to control folder.

### Path Issues
Use the new directory functions:
- `get_target_dirs(root)` for IST data
- `get_input_dirs(root, dataset_id)` for process data

---

## Next Steps

1. Test target pipeline with existing IST data
2. Test input pipeline with existing process data
3. Verify df_trg.csv compatibility
4. Update notebooks to use new structure
5. Archive old `ist_data_processing` when verified
