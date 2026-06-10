# Data Directory

This directory documents the expected local data layout for the thesis experiments.

The full datasets are not included in this repository. Place the required files locally before running the training and evaluation scripts.

## Expected files

```text
data/
|-- README.md
|-- dev.json
`-- extension/
    |-- gender_extension_clean.jsonl
    |-- race_extension_clean.jsonl
    `-- religion_extension_clean.jsonl
```

## Files

* `dev.json`
  StereoSet development file used for StereoSet evaluation.

* `extension/gender_extension_clean.jsonl`
  Generated and cleaned gender-axis preference data used for gender-specific alignment.

* `extension/race_extension_clean.jsonl`
  Generated and cleaned race-axis preference data used for race-specific alignment.

* `extension/religion_extension_clean.jsonl`
  Generated and cleaned religion-axis preference data used for religion-specific alignment.

## Notes

The generated extension datasets are not included in this repository. They were used for the thesis experiments but are omitted from the public repository together with large checkpoints, logs, generated outputs, and full result folders.

The original BiasDPO data is loaded or referenced by the relevant training scripts where applicable.
