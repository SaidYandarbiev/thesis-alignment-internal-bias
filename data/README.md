# Data Directory

Place the following files here before running the pipeline:

## Required files:

- `SBIC.v2.trn.csv` — SBIC training split
- `SBIC.v2.dev.csv` — SBIC development split  
- `dev.json` — StereoSet evaluation data

## Generated files (created by generate_neutral_responses.py):

- `neutral_responses_sbic.json` — Diverse neutral responses for SBIC groups
- `neutral_responses_toxigen.json` — Diverse neutral responses for ToxiGen groups

## How to get the data:

- **SBIC**: Download from https://maartensap.com/social-bias-frames/
- **StereoSet**: Download dev.json from https://github.com/moinnadeem/StereoSet
- **ToxiGen**: Downloaded automatically from HuggingFace when running training scripts
