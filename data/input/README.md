# Input Data (Process Parameters)

This folder contains the raw manufacturing process data files.

**⚠️ CONFIDENTIAL**: These files are not included in the repository.

---

## Folder Structure

```
input/
├── README.md                   # This file
└── {dataset_name}/             # Dataset folder (name configured in control/config.yaml)
    ├── process_map.yaml        # Process configuration (optional but recommended)
    ├── laser.csv               # Laser drilling parameters
    ├── plasma.csv              # Plasma treatment parameters
    ├── galvanik.csv            # Galvanic plating parameters
    ├── multibond.csv           # Multibond lamination parameters
    └── microetch.csv           # Microetch surface treatment parameters
```

---

## Process Map Configuration

The `process_map.yaml` file defines how to read each process file. Example:

```yaml
"laser":
    "process_label" : "Laser"           # Display name
    "hidden_label"  : "Process_1"       # Internal identifier
    "machine_label" : "Machine"         # Machine ID column (or null)
    "WA_label"      : "WA"              # Work order (batch) column
    "panel_label"   : "PanelNr"         # Panel number column (or null)
    "PaPos_label"   : "PaPosNr"         # Process position column
    "date_label"    : ["TimeStamp"]     # Timestamp column(s)
    "date_format"   : "%m/%d/%y %I:%M %p"  # Datetime format
    "prefix"        : "las"             # Variable prefix (e.g., las_1, las_2)
    "filename"      : "laser.csv"       # Input filename
    "sep"           : ","               # CSV separator
    "header"        : 0                 # Header row number

# Similar entries for: plasma, galvanic, multibond, microetch
```

---

## Expected File Schemas

### Common Columns (all process files)

| Column | Type | Description |
|--------|------|-------------|
| `WA` | string | Work order / batch identifier (e.g., "453828B") |
| `PaPosNr` or `Position` | int | Process position in recipe sequence |
| Timestamp column | datetime | When the process was executed |

### Process-Specific Columns

**Laser (`laser.csv`)**
| Column | Type | Description |
|--------|------|-------------|
| `PanelNr` | int | Panel number within batch |
| `Machine` | string | Machine identifier |
| `las_1`, `las_2`, ... | float | Laser parameters (proprietary names) |

**Plasma (`plasma.csv`)**
| Column | Type | Description |
|--------|------|-------------|
| `PanelNummer` | int | Panel number within batch |
| `Machine` | string | Machine identifier |
| `pla_1`, `pla_2`, ... | float | Plasma parameters |

**Galvanic (`galvanik.csv`)**
| Column | Type | Description |
|--------|------|-------------|
| `Panelnr` | int | Panel number within batch |
| `gal_1`, `gal_2`, ... | float | Galvanic plating parameters |

**Multibond (`multibond.csv`)**
| Column | Type | Description |
|--------|------|-------------|
| (no panel column) | - | Processes entire batch at once |
| `mul_1`, `mul_2`, ... | float | Multibond parameters |

**Microetch (`microetch.csv`)**
| Column | Type | Description |
|--------|------|-------------|
| (no panel column) | - | Processes entire batch at once |
| `mic_1`, `mic_2`, ... | float | Microetch parameters |

---

## Variable Selection

Not all columns from the process files are used. The `lookup_selected.xlsx` control file determines:
- Which columns to include
- Mapping from raw column names to standardized variable names

---

## Notes

- The pipeline handles missing columns gracefully
- Processes without panel-level data (Multibond, Microetch) are expanded to match panel-level granularity
- Timestamp formats may vary; the pipeline attempts multiple format parsers
