# Target Data (IST Results)

This folder contains the raw IST (Insulation Stress Test) resistance measurement data.

**⚠️ CONFIDENTIAL**: These files are not included in the repository.

---

## Expected File

```
target/input/
└── IST_Ergebniss01.csv    # Raw IST test results
```

---

## File Schema

The IST file contains resistance measurements over thermal cycling tests.

### Key Columns

| Column | Type | Description |
|--------|------|-------------|
| `SapNummer` | int | SAP design number (product identifier) |
| `Version` | string | Design version |
| `WA` | string | Work order / batch identifier (e.g., "453828B") |
| `Name` | string | Coupon identifier (format: "Panel_Letter") |
| `couponID` | string | Unique coupon identifier |
| `CreateDate_1` | datetime | Measurement timestamp (format: "%m/%d/%y %I:%M %p") |
| `Zyklus` | int | Thermal cycle number (1 to ~200) |
| `AnzahlZyklen_2` | int | Total number of cycles |
| `Temperatur_2` | string | Temperature setting ("high" or "low") |
| `WiderstandSenseA` | float | Absolute resistance value - Sense A (Ohms) |
| `WiderstandSenseB` | float | Absolute resistance value - Sense B (Ohms) |
| `DeltaSenseA` | float | Delta resistance - Sense A (calculated) |
| `DeltaSenseB` | float | Delta resistance - Sense B (calculated) |

### Separator

The file uses **semicolon (`;`)** as the column separator.

---

## Data Processing

The target pipeline (`scripts/run_target_pipeline.py`) processes this file to:

1. **Filter** by temperature (typically "high")
2. **Calculate delta** resistance changes from initial values
3. **Normalize** values (typically by max value of 10)
4. **Group** measurements by panel or batch
5. **Handle duplicates** (select dominant ID if multiple per group)
6. **Apply max length** (clip or pad to uniform sequence length)
7. **Export** as `df_trg.csv` in long format

---

## Output Format

The processed `df_trg.csv` file has the following structure:

| Column | Description |
|--------|-------------|
| `group` | Group identifier (e.g., "453828B_1" for panel-level) |
| `position` | Cycle number (1 to max_len) |
| `date` | Measurement date |
| `variable` | "delta_A_norm" or "delta_B_norm" |
| `value` | Normalized resistance delta value |
| `design` | SAP design number |
| `version` | Design version |

---

## Coupon Types

The pipeline distinguishes between:
- **Canary coupons (C)**: Test structures for process monitoring
- **Product coupons (P)**: Actual product samples

Filter type is configured in the target pipeline parameters.

---

## Notes

- Values exceeding ±10 are treated as anomalies and set to NaN
- The IST test is typically stopped at 10% resistance change
- Default max sequence length is 200 cycles
