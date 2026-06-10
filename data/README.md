# Data Directory

This directory contains the data files included with the repository and documents the expected local data layout for the thesis experiments.

## Included files

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

The repository includes the StereoSet development file and the cleaned gender, race, and religion extension files used in the thesis experiments.

Large model checkpoints, LoRA adapters, logs, generated outputs, full benchmark outputs, generated plots, heatmaps, and result folders are not included because they are large, machine-specific, or generated during experiment execution.

The original BiasDPO data is loaded or referenced by the relevant training scripts where applicable.
