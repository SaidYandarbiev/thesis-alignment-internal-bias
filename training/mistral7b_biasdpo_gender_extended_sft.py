"""
SFT Script — BiasDPO GENDER-EXTENDED on Mistral-7B-v0.3 (LoRA)
==================================================================
Dataset: BiasDPO gender subset + LLM-generated gender extension
Model:   mistralai/Mistral-7B-v0.3 (base, unaligned)
LR:      5e-5

This script extends the gender-only SFT setup by combining:
  1. Filtered BiasDPO examples belonging to the gender-bias axis.
  2. A local LLM-generated single-axis gender extension file.

Unlike the DPO equivalent, this script trains only on the chosen
anti-bias responses and discards the rejected responses after deduplication.

The filtering logic mirrors the gender-only SFT/DPO scripts so that the
training data remains comparable across experiments.
"""

import os
import re
import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# Disable TorchDynamo compilation — avoids graph-break errors that occur
# when PEFT's LoRA hooks interact with torch.compile.
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

MODEL_ID       = "mistralai/Mistral-7B-v0.3"
OUTPUT_DIR     = "../checkpoints/mistral7b_biasdpo_gender_extended_sft"
EXTENSION_PATH = "../data/extension/gender_extension_clean.jsonl"
SEED           = 42
MAX_SEQ_LENGTH = 512

LEARNING_RATE  = 5e-5
NUM_EPOCHS     = 3
BATCH_SIZE     = 2
GRAD_ACCUM     = 4
VAL_SPLIT      = 0.1

# ---- KEYWORD LISTS (same as gender SFT/DPO — keep in sync) ----

# Gender-specific terms used to identify and filter gender-bias examples.
# Regex patterns use word boundaries (\b) to avoid partial matches.
# Kept in sync with mistral7b_biasdpo_gender_dpo.py and the DPO equivalent.
STRONG_GENDER = [
    r"\bwoman\b", r"\bwomen\b", r"\bman\b", r"\bmen\b",
    r"\bgirl\b", r"\bgirls\b", r"\bboy\b", r"\bboys\b",
    r"\bfemale\b", r"\bfemales\b", r"\bmale\b", r"\bmales\b",
    r"\bgender\b", r"\bgenders\b", r"\bsexism\b", r"\bsexist\b",
    r"\bmisogyn\w*\b", r"\bmisandr\w*\b",
    r"\bmother\b", r"\bmothers\b", r"\bfather\b", r"\bfathers\b",
    r"\bwife\b", r"\bwives\b", r"\bhusband\b", r"\bhusbands\b",
    r"\bdaughter\b", r"\bdaughters\b", r"\bson\b", r"\bsons\b",
    r"\bsister\b", r"\bsisters\b", r"\bbrother\b", r"\bbrothers\b",
    r"\bmaternity\b", r"\bpaternity\b", r"\bpregnan(?:t|cy)\b",
    r"\bfeminin\w*\b", r"\bmasculin\w*\b",
    r"\bfeminism\b", r"\bfeminist\w*\b",
    r"\bpatriarch\w*\b", r"\bmatriarch\w*\b",
    r"\blad(?:y|ies)\b", r"\bgentlem(?:a|e)n\b",
    r"\bactress\b", r"\bwaitress\b", r"\bstewardess\b",
    r"\bmotherhood\b", r"\bfatherhood\b",
    r"\bMrs\.?\b", r"\bMs\.?\b", r"\bMr\.?\b",
]

# Pronouns are listed separately as weak gender signals. They are kept for
# consistency with the related gender scripts, but the strict filter below
# only uses STRONG_GENDER to avoid false positives from generic pronoun use.
WEAK_GENDER = [
    r"\bshe\b", r"\bher\b", r"\bhers\b", r"\bherself\b",
    r"\bhe\b", r"\bhim\b", r"\bhis\b", r"\bhimself\b",
]

# Examples matching any race keyword are excluded to keep this dataset
# strictly single-axis (gender only). Prevents cross-axis contamination.
RACE_KEYWORDS = [
    r"\bracial\b", r"\bracism\b", r"\bracist\b", r"\brace\b",
    r"\bethnic\w*\b", r"\bMiddle East\w*\b", r"\bAfrican\b",
    r"\bAsian\b", r"\bLatino\b", r"\bLatina\b", r"\bHispanic\b",
    r"\bBlack\s*(?:person|people|community|family|men|women)\b",
    r"\bWhite\s*(?:person|people|community|family|men|women)\b",
    r"\bimmigrant\w*\b", r"\bnationality\b", r"\bforeigner\w*\b",
    r"\bcolorblind\b", r"\bpeople of color\b",
]

# Same exclusion logic as race: religion-overlapping examples are dropped
# to avoid training on multi-axis bias simultaneously.
RELIGION_KEYWORDS = [
    r"\breligio\w*\b", r"\bMuslim\b", r"\bIslam\w*\b",
    r"\bChristian\w*\b", r"\bJewish\b", r"\bJew\b", r"\bJudaism\b",
    r"\bHindu\w*\b", r"\bBuddhist\w*\b", r"\bSikh\w*\b",
    r"\bchurch\b", r"\bmosque\b", r"\bsynagogue\b",
    r"\bfaith\b", r"\btheist\w*\b", r"\batheist\w*\b",
    r"\bterrorism\b", r"\bterrorist\w*\b",
]

# Phrases that indicate multi-axis overlap even when gender words are present
# (e.g. "women of colour" conflates gender and race). These are dropped
# regardless of what the keyword counts say.
PROMPT_EXCLUSIONS = [
    r"women of colou?r", r"people of colou?r", r"person of colou?r",
    r"\bConfucian\w*\b", r"\bBuddhist\w*\b", r"\bHindu\w*\b", r"\bSikh\w*\b",
    r"\bminorities\b",
]

# Set globally so dataset splitting and filtering remain reproducible.
set_seed(SEED)


def count_matches(patterns, text):
    """Return the number of regex patterns that match anywhere in the text."""
    return sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))


def is_excluded(text):
    """Return True if the text contains a phrase indicating multi-axis overlap."""
    return any(re.search(p, text, re.IGNORECASE) for p in PROMPT_EXCLUSIONS)


def is_gender_example(example):
    """
    Decide whether a BiasDPO-style example belongs to the gender-bias axis.

    BiasDPO does not provide explicit gender/race/religion labels, so examples
    are classified using keyword-based filtering over the prompt, chosen, and
    rejected fields.

    An example is kept only if it contains at least one strong gender keyword
    and contains no race keywords, religion keywords, or manually defined
    exclusion phrases. This keeps the subset focused on single-axis gender bias
    and avoids training on examples that mix gender with race or religion.
    """
    text = " ".join([
        str(example.get("prompt", "")),
        str(example.get("chosen", "")),
        str(example.get("rejected", "")),
    ])
    if is_excluded(text):
        return False
    g_strong = count_matches(STRONG_GENDER, text)
    r = count_matches(RACE_KEYWORDS, text)
    rel = count_matches(RELIGION_KEYWORDS, text)
    return g_strong >= 1 and r == 0 and rel == 0


def load_and_preprocess():
    """
    Load and combine the race-filtered BiasDPO split and the LLM-generated
    race extension, train only on chosen (unbiased) responses, then
    deduplicate and split into train/validation sets.

    Steps:
    1. Load full BiasDPO, drop degenerate pairs (chosen == rejected).
    2. Filter to race-axis examples via is_race_example().
    3. Load the local race extension JSONL.
    4. Concatenate, deduplicate on (prompt, chosen, rejected) triple.
    5. Split the deduplicated data into train/validation sets.

    Unlike the DPO equivalent, rejected responses are discarded after
    deduplication — SFT trains on positive examples only.

    Returns:
        DatasetDict with keys 'train' and 'test'.
    """
    biasdpo = load_dataset("ahmedallam/BiasDPO", split="train")
    print(f"[Data] Full BiasDPO size: {len(biasdpo)}")
    biasdpo = biasdpo.filter(
        lambda x: bool(x["prompt"]) and bool(x["chosen"]) and bool(x["rejected"])
                  and x["chosen"].strip() != x["rejected"].strip()
    )
    biasdpo_filtered = biasdpo.filter(is_gender_example)
    print(f"[Data] BiasDPO after gender filter: {len(biasdpo_filtered)}")

    extension = load_dataset("json", data_files=EXTENSION_PATH, split="train")
    print(f"[Data] Extension size: {len(extension)}")

    # Keep only the DPO fields needed for filtering, deduplication, and SFT conversion.
    biasdpo_filtered = biasdpo_filtered.select_columns(["prompt", "chosen", "rejected"])
    extension = extension.select_columns(["prompt", "chosen", "rejected"])

    combined = concatenate_datasets([biasdpo_filtered, extension])
    print(f"[Data] Combined (before dedup): {len(combined)}")

    # Deduplicate on the exact (prompt, chosen, rejected) triple.
    # datasets has no built-in dedup, so we track seen keys manually.
    seen = set()
    unique_idx = []
    for i, ex in enumerate(combined):
        key = (ex["prompt"].strip(), ex["chosen"].strip(), ex["rejected"].strip())
        if key not in seen:
            seen.add(key)
            unique_idx.append(i)
    combined = combined.select(unique_idx)
    print(f"[Data] After dedup: {len(combined)}")

    if len(combined) < 100: # sanity check — a dataset this small would produce unreliable training
        raise RuntimeError(f"Combined dataset too small ({len(combined)}).")

    split = combined.train_test_split(test_size=VAL_SPLIT, seed=SEED)
    print(f"[Data] Train: {len(split['train'])}, Validation: {len(split['test'])}")
    return split


def format_for_sft(example, tokenizer):
    """
    Wrap the prompt in Mistral's [INST]...[/INST] instruction tags and
    append the EOS token to the chosen completion.

    In the reported SFT runs, the prompt and completion were provided as
    separate fields, but the training loss was applied to the full formatted
    sequence rather than being masked to the completion tokens only. The
    leading space before the completion helps preserve the expected separation
    between the instruction block and the answer, while the EOS token marks
    the end of the target response.
    """
    prompt = f"[INST] {example['prompt']} [/INST]"
    completion = " " + example["chosen"] + tokenizer.eos_token
    return {"prompt": prompt, "completion": completion}


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # Mistral-7B-v0.3 has no dedicated pad token

    splits = load_and_preprocess()
    # num_proc=1 avoids tokenizer parallelism warnings from HuggingFace
    train_dataset = splits["train"].map(lambda x: format_for_sft(x, tokenizer), num_proc=1)
    eval_dataset = splits["test"].map(lambda x: format_for_sft(x, tokenizer), num_proc=1)

    sample = train_dataset[0]
    print(f"[OK] prompt: {repr(sample['prompt'][:100])}")
    print(f"[OK] completion: {repr(sample['completion'][:100])}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
    )

    # LoRA rank 16 / alpha 32 gives a scaling factor of 2, a common starting point
    # for instruction-tuning. All seven projection matrices are targeted to give
    # the adapter sufficient capacity for response-style fine-tuning.
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        # cosine LR schedule with warmup is commonly used for stable fine-tuning
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        max_length=MAX_SEQ_LENGTH,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        processing_class=tokenizer, peft_config=peft_config,
    )

    print("Starting BiasDPO GENDER-EXTENDED SFT on Mistral-7B-v0.3 (LoRA, LR=5e-5)...")
    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Done. Saved to {OUTPUT_DIR}")
