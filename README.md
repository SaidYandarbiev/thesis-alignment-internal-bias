# Effect of Alignment on Internal Associations in Large Language Models

This repository contains the code used for the master thesis **"Effect of Alignment on Internal Associations in Large Language Models."** The project studies whether alignment procedures reduce social bias only at the level of observable model behavior, or whether they also change internal bias-related representations.

The experiments are based on **Mistral-7B-v0.3** and compare **Supervised Fine-Tuning (SFT)** with **Direct Preference Optimization (DPO)**. Alignment is evaluated under both **parallel single-axis** and **sequential** training settings across gender, race, and religion bias dimensions.

## Thesis overview

The project evaluates aligned model checkpoints at two levels.

1. **Behavioral bias evaluation**
   - StereoSet
   - CrowS-Pairs
   - BBQ through `lm-evaluation-harness`

2. **Internal representational analysis**
   - stereotype-axis probing
   - leave-one-bucket-out probing
   - layerwise cosine similarity analysis
   - word association network analysis

The goal is to test whether alignment primarily suppresses biased outputs, or whether it also modifies the internal geometry and topology of stereotype-related associations.

## Repository structure

```text
.
|-- training/                  # SFT, DPO, and sequential alignment scripts
|-- evaluation/                # StereoSet evaluation code
|-- internal_analysis/         # probing, cosine, and network analysis code
|-- internal_analysis/configs/ # checkpoint configuration files for analysis
|-- data/                      # data documentation only; full datasets not included
|-- aggregate_external.py      # aggregate behavioral benchmark results
|-- aggregate_internal.py      # aggregate internal analysis results
|-- requirements.txt
`-- README.md
```

## What is included

This repository includes:

- training scripts for general, parallel single-axis, and sequential SFT/DPO alignment;
- custom StereoSet scoring code for causal language models;
- CrowS-Pairs and BBQ were evaluated using lm-evaluation-harness; this repository includes the main training, StereoSet evaluation, internal analysis, and result aggregation code used for the thesis.
- internal analysis scripts for probing, cosine similarity, and word association networks;
- checkpoint configuration files for the internal analysis pipeline;
- aggregation scripts used to summarize external and internal results.

## What is not included

The following files and folders are intentionally excluded from the repository:

- model checkpoints and LoRA adapters;
- generated result folders;
- logs;
- full benchmark outputs;
- generated plots and heatmaps;
- cluster-specific SLURM job scripts;
- full generated training datasets.

These files are excluded because they are either large, machine-specific, or generated during experiment execution.

## Setup

Create a Python environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

The experiments were run on GPU hardware with bfloat16 support. Running the full pipeline requires substantial GPU memory because the scripts load **Mistral-7B-v0.3** and LoRA adapters.

Optional acceleration packages such as `flash-attn` may be installed separately depending on the GPU environment.

## Data layout

The repository does not include the full training or evaluation datasets. The expected local layout is:

```text
data/
|-- README.md
|-- dev.json
`-- extension/
    |-- gender_extension_clean.jsonl
    |-- race_extension_clean.jsonl
    `-- religion_extension_clean.jsonl
```

`dev.json` refers to the StereoSet development file. The `extension/` files are the generated axis-isolated preference datasets used for gender, race, and religion alignment.

The original BiasDPO dataset is loaded through the relevant training scripts where applicable.

## Checkpoint layout

The training and analysis scripts expect local checkpoint directories. A typical local layout is:

```text
checkpoints/
|-- mistral7b_biasdpo_sft/
|-- mistral7b_biasdpo_dpo/
|-- mistral7b_biasdpo_gender_extended_sft/
|-- mistral7b_biasdpo_gender_extended_dpo/
|-- mistral7b_biasdpo_race_extended_sft/
|-- mistral7b_biasdpo_race_extended_dpo/
|-- mistral7b_biasdpo_religion_extended_sft/
|-- mistral7b_biasdpo_religion_extended_dpo/
`-- seq_.../
```

The checkpoint folders are not included in this repository.

## Training

Training scripts are located in `training/`. They are intended to be run from inside the `training/` directory because several scripts use relative paths such as `../data/` and `../checkpoints/`.

```bash
cd training
```

General BiasDPO alignment:

```bash
python mistral7b_biasdpo_sft.py
python mistral7b_biasdpo_dpo.py
```

Parallel single-axis alignment:

```bash
python mistral7b_biasdpo_gender_extended_sft.py
python mistral7b_biasdpo_gender_extended_dpo.py

python mistral7b_biasdpo_race_extended_sft.py
python mistral7b_biasdpo_race_extended_dpo.py

python mistral7b_biasdpo_religion_extended_sft.py
python mistral7b_biasdpo_religion_extended_dpo.py
```

Sequential alignment scripts are also included in `training/` and follow the naming pattern:

```text
seq_<stage1_axis>_<method>_to_<stage2_axis>_<method>.py
```

For example:

```bash
python seq_gender_dpo_to_race_dpo.py
```

Sequential scripts assume that the stage 1 checkpoint already exists locally.

## Behavioral evaluation

### StereoSet

StereoSet is evaluated using the custom causal-language-model scorer. The command below is an example for the unaligned baseline and can be adapted by replacing `--model-id` with a local aligned checkpoint path.

```bash
python evaluation/stereoset/stereoset_scorer.py \
  --model-id mistralai/Mistral-7B-v0.3 \
  --input-file data/dev.json \
  --output-file results/stereoset/predictions_baseline.json
```

The StereoSet evaluator in `evaluation/stereoset/evaluation.py` can then be used to compute LM Score, Stereotype Score, and ICAT.

### CrowS-Pairs

CrowS-Pairs was evaluated using `lm-evaluation-harness`.

Example task names used in the thesis:

```text
crows_pairs_english_gender
crows_pairs_english_race_color
crows_pairs_english_religion
crows_pairs_english_socioeconomic
```

### BBQ

BBQ was evaluated through `lm-evaluation-harness` using:

```text
bbq_ambig
bbq_disambig
```

The cluster-specific submission scripts used for these evaluations are not included in this repository.

## Internal analysis

The internal analysis code is located in `internal_analysis/`. These commands are intended to be run from inside that directory because the configuration files use relative checkpoint paths.

```bash
cd internal_analysis
```

Each analysis script takes a checkpoint configuration file and writes outputs to a selected output directory.

### Stereotype-axis probing

```bash
python probing_analysis.py \
  --checkpoints-config configs/biasdpo_full.json \
  --output-dir ../analysis/internal/biasdpo_full/probing \
  --probe-type stereotype \
  --pooling last_token
```

### Leave-one-bucket-out probing

```bash
python probing_analysis.py \
  --checkpoints-config configs/biasdpo_full.json \
  --output-dir ../analysis/internal/biasdpo_full/probing_lobo \
  --probe-type stereotype_lobo \
  --pooling last_token
```

### Cosine similarity analysis

```bash
python cosine_analysis.py \
  --checkpoints-config configs/biasdpo_full.json \
  --output-dir ../analysis/internal/biasdpo_full/cosine \
  --pooling last_token
```

### Word association network analysis

```bash
python network_analysis.py \
  --checkpoints-config configs/biasdpo_full.json \
  --output-dir ../analysis/internal/biasdpo_full/network \
  --pooling last_token
```

Additional checkpoint configurations are available in `internal_analysis/configs/`.

## Aggregation

After running evaluations and internal analyses, the aggregation scripts can be used to summarize results:

```bash
python aggregate_external.py
python aggregate_internal.py
```

These scripts expect the generated result folders to exist locally.

## Reproducibility note

This repository is intended to document and support the experiments reported in the thesis. It is not a fully self-contained archive because the full checkpoints, generated outputs, and large datasets are not included.

To reproduce the full pipeline, users must provide the required datasets, create the checkpoint directories, and run the training, evaluation, and internal analysis scripts in the appropriate order.

## Citation

If you use this repository, please cite the accompanying thesis:

```text
Yandarbiev, S. (2026). Effect of Alignment on Internal Associations in Large Language Models. Master's thesis, University of Antwerp.
```
