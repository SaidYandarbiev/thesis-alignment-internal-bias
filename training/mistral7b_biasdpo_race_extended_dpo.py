"""
DPO Script — BiasDPO RACE-EXTENDED on Mistral-7B-v0.3 (LoRA)
==================================================================
Dataset: BiasDPO race subset + LLM-generated race extension
Model:   mistralai/Mistral-7B-v0.3 (base, unaligned)
LR:      5e-5

This script extends the race-only DPO setup by combining:
  1. Filtered BiasDPO examples belonging to the race-bias axis.
  2. A local LLM-generated single-axis race extension file.

The filtering logic mirrors the race-only SFT/DPO scripts so that the
training data remains comparable across experiments.
"""


import os
import re
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig

# Disable TorchDynamo compilation — avoids graph-break errors that occur
# when PEFT's LoRA hooks interact with torch.compile.
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

MODEL_ID       = "mistralai/Mistral-7B-v0.3"
OUTPUT_DIR     = "../checkpoints/mistral7b_biasdpo_race_extended_dpo"
EXTENSION_PATH = "../data/extension/race_extension_clean.jsonl"
SEED           = 42
MAX_SEQ_LENGTH = 512

LEARNING_RATE  = 5e-5
NUM_EPOCHS     = 3
BATCH_SIZE     = 2
GRAD_ACCUM     = 4
BETA           = 0.1
VAL_SPLIT      = 0.1

# ---- KEYWORD LISTS FOR BIAS CLASSIFICATION ----

# Race/ethnicity terms used to identify race-bias examples.
# More granular than the gender list, includes specific nationalities and
# ethnicities to capture the wider range of race-bias framings in the data.
STRONG_RACE = [
    r"\bracial\b", r"\bracism\b", r"\bracist\b", r"\brace\b", r"\braces\b",
    r"\bethnic\w*\b",
    r"\bAfrican\s*American\w*\b", r"\bAsian\s*American\w*\b",
    r"\bIndian\s*American\w*\b", r"\bNative\s*American\w*\b",
    r"\bLatino\b", r"\bLatina\b", r"\bLatinos\b", r"\bLatinas\b",
    r"\bLatinx\b", r"\bHispanic\w*\b",
    r"\bAsian\b", r"\bAsians\b",
    r"\bBlack\b", r"\bBlacks\b",
    r"\bWhite\b", r"\bWhites\b",
    r"\bIndigenous\b", r"\bAboriginal\w*\b", r"\bFirst\s*Nations\b",
    r"\bArab\w*\b", r"\bMiddle\s*East\w*\b", r"\bPersian\w*\b",
    r"\bChinese\b", r"\bKorean\w*\b", r"\bVietnamese\b", r"\bJapanese\b",
    r"\bThai\b", r"\bFilipino\w*\b", r"\bMexican\w*\b",
    r"\bPacific\s*Islander\w*\b", r"\bHawaiian\w*\b",
    r"\bimmigrant\w*\b", r"\bnationality\b", r"\bforeigner\w*\b",
    r"\bcolorblind\b", r"\bpeople of colou?r\b",
    r"\bxenophob\w*\b",
    r"\bsegregat\w*\b", r"\baffirmative action\b",
]

# Examples matching any gender keyword are excluded to keep this dataset
# strictly single-axis (race only). Prevents cross-axis contamination.
GENDER_KEYWORDS = [
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
]

# Same exclusion logic as gender: religion-overlapping examples are dropped
# to avoid training on multi-axis bias simultaneously.
RELIGION_KEYWORDS = [
    r"\breligio\w*\b", r"\bMuslim\w*\b", r"\bIslam\w*\b",
    r"\bChristian\w*\b", r"\bJewish\b", r"\bJew\b", r"\bJews\b", r"\bJudaism\b",
    r"\bHindu\w*\b", r"\bBuddhist\w*\b", r"\bBuddhism\b", r"\bSikh\w*\b",
    r"\bchurch\w*\b", r"\bmosque\w*\b", r"\bsynagogue\w*\b",
    r"\bevangelical\w*\b", r"\bCatholic\w*\b", r"\bProtestant\w*\b",
    r"\bMormon\w*\b",
    r"\bfaith\b", r"\btheist\w*\b", r"\batheist\w*\b",
    r"\bterrorism\b", r"\bterrorist\w*\b",
]

# Phrases that explicitly combine race with another axis are excluded,
# even if they would otherwise pass the keyword counts.
# e.g. "Black women" conflates race and gender, dropped here but eligible
# for a future intersectional training run.
PROMPT_EXCLUSIONS = [
    r"women of colou?r", r"people of colou?r", r"person of colou?r",
    r"Black\s+(?:women|men|mothers|fathers)",
    r"Asian\s+(?:women|men|mothers|fathers)",
    r"Latino\s+(?:women|men|mothers|fathers)",
    r"Latina\s+(?:women|men|mothers|fathers)",
    r"Hispanic\s+(?:women|men|mothers|fathers)",
    r"Muslim\s+(?:woman|women|man|men)",
    r"Christian\s+(?:woman|women|man|men)",
    r"Jewish\s+(?:woman|women|man|men)",
]

# Set globally so dataset splitting and filtering remain reproducible.
set_seed(SEED)


def count_matches(patterns, text):
    """Return the number of regex patterns that match anywhere in the text."""
    return sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))


def is_excluded(text):
    """Return True if the text contains a phrase indicating multi-axis overlap."""
    return any(re.search(p, text, re.IGNORECASE) for p in PROMPT_EXCLUSIONS)


def is_race_example(example):
    """
    Decide whether a BiasDPO-style example belongs to the race-bias axis.

    BiasDPO does not provide explicit race/gender/religion labels, so examples
    are classified using keyword-based filtering over the prompt, chosen, and
    rejected fields.

    An example is kept only if it contains at least one strong race keyword
    and contains no gender keywords, religion keywords, or manually defined
    exclusion phrases. This keeps the subset focused on single-axis race bias
    and avoids training on examples that mix race with gender or religion.
    """
    text = " ".join([
        str(example.get("prompt", "")),
        str(example.get("chosen", "")),
        str(example.get("rejected", "")),
    ])
    if is_excluded(text):
        return False
    r_strong = count_matches(STRONG_RACE, text)
    g = count_matches(GENDER_KEYWORDS, text)
    rel = count_matches(RELIGION_KEYWORDS, text)
    return r_strong >= 1 and g == 0 and rel == 0


def load_and_preprocess():
    """
    Load and combine the race-filtered BiasDPO split and the LLM-generated
    race extension, then deduplicate and split into train/validation sets.

    Steps:
    1. Load full BiasDPO, drop degenerate pairs (chosen == rejected).
    2. Filter to race-axis examples via is_race_example().
    3. Load the local race extension JSONL.
    4. Concatenate, deduplicate on (prompt, chosen, rejected) triple.
    5. Guard against accidental near-empty datasets.

    Returns:
        DatasetDict with keys 'train' and 'test'.
    """
    biasdpo = load_dataset("ahmedallam/BiasDPO", split="train")
    print(f"[Data] Full BiasDPO size: {len(biasdpo)}")
    biasdpo = biasdpo.filter(
        lambda x: bool(x["prompt"]) and bool(x["chosen"]) and bool(x["rejected"])
                  and x["chosen"].strip() != x["rejected"].strip() # drop pairs with identical chosen/rejected — no DPO signal
    )
    biasdpo_filtered = biasdpo.filter(is_race_example)
    print(f"[Data] BiasDPO after race filter: {len(biasdpo_filtered)}")

    extension = load_dataset("json", data_files=EXTENSION_PATH, split="train")
    print(f"[Data] Extension size: {len(extension)}")

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


def format_for_dpo(example, tokenizer):
    """
    Wrap the prompt in Mistral's [INST]...[/INST] instruction tags and
    prepend a leading space to chosen/rejected responses.

    The leading space is kept consistent with the other DPO scripts and helps
    TRL tokenize the response boundary correctly.
    """
    prompt = f"[INST] {example['prompt']} [/INST]"
    return {
        "prompt": prompt,
        "chosen": " " + example["chosen"],
        "rejected": " " + example["rejected"],
    }


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # Mistral-7B-v0.3 has no dedicated pad token

    splits = load_and_preprocess()
    # num_proc=1 avoids tokenizer parallelism warnings from HuggingFace
    train_dataset = splits["train"].map(lambda x: format_for_dpo(x, tokenizer), num_proc=1)
    eval_dataset = splits["test"].map(lambda x: format_for_dpo(x, tokenizer), num_proc=1)

    sample = train_dataset[0]
    print(f"[OK] Prompt: {repr(sample['prompt'][:150])}")
    print(f"[OK] Chosen: {repr(sample['chosen'][:100])}")
    print(f"[OK] Rejected: {repr(sample['rejected'][:100])}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
    )

    # LoRA rank 16 / alpha 32 gives a scaling factor of 2, a common starting point
    # for instruction-tuning. All seven projection matrices are targeted to give
    # the adapter sufficient capacity to shift the model's preference distribution.
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        # cosine LR schedule with warmup is standard for DPO fine-tuning
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        # beta controls the KL penalty strength; 0.1 is a common DPO starting point
        beta=BETA,
        max_length=MAX_SEQ_LENGTH,
        max_prompt_length=256,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = DPOTrainer(
        # With PEFT/LoRA, ref_model=None lets TRL use the frozen base model
        # behaviour as the implicit DPO reference model, avoiding a second full
        # model copy in memory. 
        model=model, ref_model=None,
        args=training_args,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        processing_class=tokenizer, peft_config=peft_config,
    )

    print("Starting BiasDPO RACE-EXTENDED DPO on Mistral-7B-v0.3 (LoRA, LR=5e-5)...")
    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Done. Saved to {OUTPUT_DIR}")
