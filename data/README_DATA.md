# Data Setup Instructions

This document describes the data structure required for the proT_pipeline to function correctly.

**⚠️ IMPORTANT**: The actual data files are confidential and not included in this repository.

Data can be downloaded from [Polybox](https://polybox.ethz.ch/index.php/s/qr9XfqYRPYGMfSn), with authorization (contact [Francesco Scipione](fscipion@ethz.ch))

---

## Data Directory Structure

After obtaining the data, organize it as follows:

```
data/
├── README_DATA.md              # This file
│
├── input/                      # Process data (manufacturing parameters)
│   └── {dataset_name}/         # e.g., Prozessdaten_MSEI_01_01_2022-07_07_2025_csv/
│       ├── process_map.yaml    # Process file configuration
│       ├── laser.csv           # Laser process parameters
│       ├── plasma.csv          # Plasma process parameters
│       ├── galvanik.csv        # Galvanic process parameters
│       ├── multibond.csv       # Multibond process parameters
│       └── microetch.csv       # Microetch process parameters
│
├── target/                     # IST (resistance test) data
│   ├── input/
│   │   └── IST_Ergebniss01.csv # Raw IST test results
│   └── builds/                 # Processed IST outputs
│       └── {build_id}/
│           └── df_trg.csv      # Generated target dataframe
│
└── builds/                     # Dataset builds (output)
    ├── control/                # Shared control files (templates)
    │   ├── lookup_selected.xlsx
    │   └── steps_selected.xlsx
    │
    └── {dataset_id}/           # Specific dataset build
        ├── control/            # Build-specific control files
        │   ├── config.yaml
        │   ├── df_trg.csv
        │   ├── lookup_selected.xlsx
        │   ├── steps_selected.xlsx
        │   └── Prozessfolgen_MSEI.xlsx
        │
        └── output/             # Generated outputs
            ├── df_process_raw.csv
            ├── df_input.parquet
            ├── sample_metrics.parquet
            ├── *_vocabulary.json
            └── ds_{dataset_id}/
                ├── data.npz        # Full dataset
                ├── train_data.npz  # Training split
                └── test_data.npz   # Test split
```

---

## Creating a New Dataset Build

1. **Create the build folder**:
   ```bash
   mkdir -p data/builds/{your_dataset_id}/control
   mkdir -p data/builds/{your_dataset_id}/output
   ```

2. **Copy control files**:
   - Copy template control files from `data/builds/control/` or an existing build
   - Edit `config.yaml` to point to your input dataset folder

3. **Generate target data** (if not already available):
   ```bash
   python scripts/run_target_pipeline.py
   ```

4. **Copy target to control folder**:
   ```bash
   python scripts/copy_target_to_control.py
   ```
   Or manually copy `data/target/builds/{build_id}/df_trg.csv` to `data/builds/{dataset_id}/control/`

5. **Run the input pipeline**:
   ```bash
   python scripts/run_input_pipeline.py
   ```

---

## File Descriptions

### Input Files

See `data/input/README.md` for detailed schema of process data files.

### Target Files

See `data/target/input/README.md` for detailed schema of IST data files.

### Control Files

See `data/builds/template_build/control/README.md` for detailed schema of control files.

---

## Troubleshooting

### Common Issues

1. **"Builds directory doesn't exist"**
   - Create the build folder structure manually before running the pipeline

2. **"Target file not found"**
   - Run the target pipeline first, or copy `df_trg.csv` to the control folder

3. **"process_map.yaml not found"**
   - The pipeline will use default process configurations
   - For custom datasets, create a process_map.yaml in your input folder

4. **Missing columns in process files**
   - Check that your CSV files match the expected schema
   - The pipeline handles missing columns gracefully, but logs warnings
