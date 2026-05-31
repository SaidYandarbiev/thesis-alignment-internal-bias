@'
# Effect of Alignment on Internal Associations in Large Language Models

This repository contains the code used for the master thesis **"Effect of Alignment on Internal Associations in Large Language Models."** The project studies whether alignment procedures reduce social bias only at the level of observable behavior, or whether they also change internal bias-related representations.

The experiments are based on **Mistral-7B-v0.3** and compare **Supervised Fine-Tuning (SFT)** with **Direct Preference Optimization (DPO)**. Models are trained under both parallel single-axis and sequential alignment settings across gender, race, and religion bias dimensions.

## What is included

This repository contains:

- training scripts for SFT and DPO alignment;
- evaluation code for StereoSet and CrowS-Pairs;
- internal analysis code for:
  - stereotype-axis probing;
  - layerwise cosine similarity analysis;
  - word association network analysis;
- configuration files for the internal analysis pipeline;
- aggregation scripts for external and internal results.

Large model checkpoints, generated outputs, logs, and full result folders are not included in the repository.

## Repository structure

```text
.
|-- training/                  # SFT, DPO, and sequential alignment scripts
|-- evaluation/                # StereoSet and CrowS-Pairs evaluation code
|-- internal_analysis/         # probing, cosine, and network analysis code
|-- internal_analysis/configs/ # checkpoint configuration files for analysis
|-- data/                      # data documentation only; full datasets not included
|-- aggregate_external.py      # aggregate behavioral benchmark results
|-- aggregate_internal.py      # aggregate internal analysis results
|-- requirements.txt
`-- README.md
```

## Setup

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

The experiments were run on GPU hardware with bfloat16 support. Some runs require substantial GPU memory because they load Mistral-7B-v0.3 and LoRA adapters.

## Data and checkpoints

The full training data, model checkpoints, logs, and generated result folders are not included in this repository.

Expected local directories when reproducing the full pipeline:

```text
data/
checkpoints/
results/
external_results/
aggregated_results/
analysis/
logs/
```

The `data/README.md` file describes the expected data layout.

## Notes

This repository is intended to document and support the experiments reported in the thesis. Because the full checkpoints and generated result artifacts are not included, the repository should be treated as a code and reproducibility package rather than a fully self-contained archive.
