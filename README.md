# proT_pipeline

## Overview

**proT_pipeline** is a data processing pipeline for manufacturing process data and quality test results. It transforms raw manufacturing data into structured datasets suitable for machine learning models, specifically transformer-based architectures for predictive quality analysis.

The pipeline processes two types of data:
- **Target Data (IST)**: Resistance test results from Insulation Stress Testing
- **Input Data (Process)**: Manufacturing process parameters (Laser, Plasma, Galvanic, Multibond, Microetch)

## Features

- Unified processing pipeline for manufacturing and quality data
- Configurable variable selection via Excel control files
- Automatic normalization and temporal ordering
- Support for panel-level and batch-level grouping
- Stratified train/test splitting based on sample metrics
- Generates compressed numpy datasets ready for ML training

## Installation

### Prerequisites

- Python 3.11+
- Git

### Setting Up

1. **Clone the repository**:
    ```bash
    git clone https://github.com/scipi1/proT_pipeline.git
    cd proT_pipeline
    ```

2. **Create a virtual environment**:
    ```bash
    python -m venv venv
    ```

3. **Activate the virtual environment**:
    - **Windows**:
        ```bash
        .\venv\Scripts\activate
        ```
    - **macOS/Linux**:
        ```bash
        source venv/bin/activate
        ```

4. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Setting Up Jupyter Notebook (for tutorials)

1. **Create a Jupyter kernel**:
    ```bash
    python -m ipykernel install --user --name=proT_pipeline_venv --display-name "proT Pipeline (venv)"
    ```

2. **Launch Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```

3. **Select the kernel**: Open a notebook → Kernel → Change kernel → "proT Pipeline (venv)"

## Data Setup

**⚠️ IMPORTANT**: The data is not included in this repository due to confidentiality.

Data can be downloaded from [polybox](https://polybox.ethz.ch/index.php/s/qr9XfqYRPYGMfSn) with authorization (contact [Francesco Scipione](fscipion@ethz.ch)).

Once you have access to the data, place the files according to the structure described in `data/README_DATA.md`.

## Quick Start

### Workflow 1: Generate Training Dataset

1. **Process IST (target) data**:
    ```bash
    python scripts/run_target_pipeline.py
    ```

2. **Copy target to build folder**:
    ```bash
    python scripts/copy_target_to_control.py
    ```

3. **Process input data**:
    ```bash
    python scripts/run_input_pipeline.py
    ```

**Output**: `data/builds/{dataset_id}/output/ds_{dataset_id}/data.npz`

### Workflow 2: Generate Prediction Dataset

See `tutorials/tutorial_02_prediction_dataset.ipynb` for detailed instructions on generating datasets for inference (without target data).

## Project Structure

```
proT_pipeline/
│
├── proT_pipeline/                   # Main Python package
│   ├── __init__.py
│   ├── main.py                      # Input pipeline entry point
│   ├── labels.py                    # Unified labels (shared by both pipelines)
│   ├── utils.py                     # Utility functions
│   ├── rarity_utils.py              # Sample metric calculations
│   ├── stratified_split.py          # Stratified splitting utilities
│   │
│   ├── core/                        # Core utilities
│   │   ├── modules.py               # Process class, data transformations
│   │   └── sequencer.py             # Sequence building utilities
│   │
│   ├── input_processing/            # Process data pipeline
│   │   ├── assemble_raw.py          # Assembles raw process data
│   │   ├── process_raw.py           # Normalizes and adds metadata
│   │   ├── generate_dataset.py      # Creates final numpy datasets
│   │   ├── data_loader.py           # Process file loaders
│   │   └── split_by_metric.py       # Train/test splitting
│   │
│   └── target_processing/           # IST data pipeline
│       ├── main.py                  # Target pipeline entry point
│       └── modules.py               # IST processing functions
│
├── scripts/                         # Orchestration scripts
│   ├── run_target_pipeline.py       # Run IST processing
│   ├── run_input_pipeline.py        # Run process data pipeline
│   └── copy_target_to_control.py    # Copy df_trg.csv between folders
│
├── tutorials/                       # Step-by-step tutorials
│   ├── tutorial_01_training_dataset.ipynb
│   └── tutorial_02_prediction_dataset.ipynb
│
├── notebooks/                       # Analysis notebooks
│   └── ...
│
├── test/                            # Test suite
│   └── ...
│
├── data/                            # Data directory (not in repo)
│   ├── README_DATA.md               # Data setup instructions
│   ├── input/                       # Raw process files
│   ├── target/                      # IST data
│   └── builds/                      # Dataset builds
│
├── config/                          # Configuration files
│   └── config.yaml
│
├── docs/                            # Documentation
│   └── ...
│
├── requirements.txt                 # Python dependencies
├── LICENSE                          # MIT License
├── INTEGRATION_GUIDE.md             # Technical integration guide
└── README.md                        # This file
```

## Configuration

### Build Configuration

Each dataset build requires a `control/` folder with:

| File | Description |
|------|-------------|
| `config.yaml` | Points to input dataset folder |
| `df_trg.csv` | Processed target data (from IST pipeline) |
| `lookup_selected.xlsx` | Variable selection per process |
| `steps_selected.xlsx` | Process steps to include |
| `Prozessfolgen_MSEI.xlsx` | Layer/occurrence mapping |

See `data/builds/template_build/control/README.md` for detailed schemas.

### Pipeline Parameters

**Target Pipeline** (`run_target_pipeline.py`):
- `build_id`: Output folder name
- `grouping_method`: "panel" or "column"
- `max_len`: Maximum sequence length
- `filter_type`: "C" (canary) or "P" (product)

**Input Pipeline** (`run_input_pipeline.py`):
- `dataset_id`: Build folder name
- `missing_threshold`: Max % missing values per variable
- `use_stratified_split`: Enable stratified train/test split
- `train_ratio`: Train set proportion (default: 0.8)

## Output Format

The pipeline generates compressed numpy archives (`.npz`) with:

- **X** (input): Shape `[n_samples, max_len, n_features]`
  - Features: group_id, process, occurrence, step, variable, value, order, time components
  
- **Y** (target): Shape `[n_samples, max_len, n_features]`
  - Features: group_id, position, variable, value, time components

## Tutorials

| Tutorial | Description |
|----------|-------------|
| [Tutorial 1: Training Dataset](tutorials/tutorial_01_training_dataset.ipynb) | Complete walkthrough of generating training data |
| [Tutorial 2: Prediction Dataset](tutorials/tutorial_02_prediction_dataset.ipynb) | Generate datasets for inference (no target) |

## Testing

Run the test suite:
```bash
python -m pytest test/
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this pipeline in your research, please cite:

```
[Citation information to be added]
```
