"""
SFT Script — Sequential: Race SFT → Gender SFT on Mistral-7B-v0.3 (LoRA)
=========================================================================
Implements sequential bias alignment: trains on gender bias data starting
from a model already fine-tuned on race bias, rather than from the base model.

Step 1: Load the pre-trained race SFT adapter.
Step 2: Merge the adapter into the base weights with merge_and_unload().
Step 3: Apply a fresh LoRA adapter and train on the gender-extended SFT dataset.

This tests whether sequential SFT alignment causes cross-bias interference —
a key research question of the thesis.
"""

import os
import re
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# Disable TorchDynamo compilation — avoids graph-break errors that occur
# when PEFT's LoRA hooks interact with torch.compile.
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

MODEL_ID       = "mistralai/Mistral-7B-v0.3"
OUTPUT_DIR     = "checkpoints/seq_race_sft_to_gender_sft"
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
# Kept in sync with the gender-extended SFT/DPO scripts.
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

# Examples matching any race keyword are excluded to keep this second-stage
# dataset strictly single-axis gender. This helps isolate whether the prior
# race alignment is retained or disrupted during gender training.
RACE_KEYWORDS = [
    r"\bracial\b", r"\bracism\b", r"\bracist\b", r"\brace\b",
    r"\bethnic\w*\b", r"\bMiddle East\w*\b", r"\bAfrican\b",
    r"\bAsian\b", r"\bLatino\b", r"\bLatina\b", r"\bHispanic\b",
    r"\bBlack\s*(?:person|people|community|family|men|women)\b",
    r"\bWhite\s*(?:person|people|community|family|men|women)\b",
    r"\bimmigrant\w*\b", r"\bnationality\b", r"\bforeigner\w*\b",
    r"\bcolorblind\b", r"\bpeople of color\b",
]

# Religion-overlapping examples are dropped to avoid introducing a third bias
# axis during the second stage of sequential alignment.
RELIGION_KEYWORDS = [
    r"\breligio\w*\b", r"\bMuslim\b", r"\bIslam\w*\b",
    r"\bChristian\w*\b", r"\bJewish\b", r"\bJew\b", r"\bJudaism\b",
    r"\bHindu\w*\b", r"\bBuddhist\w*\b", r"\bSikh\w*\b",
    r"\bchurch\b", r"\bmosque\b", r"\bsynagogue\b",
    r"\bfaith\b", r"\btheist\w*\b", r"\batheist\w*\b",
    r"\bterrorism\b", r"\bterrorist\w*\b",
]

# Phrases that indicate multi-axis overlap even when gender words are present
# are dropped regardless of keyword counts.
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
    exclusion phrases. This keeps the second-stage dataset focused on
    single-axis gender bias after the initial race-SFT alignment.
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
    Load and combine the gender-filtered BiasDPO split and the LLM-generated
    gender extension for the second stage of sequential SFT training.

    Steps:
    1. Load full BiasDPO, drop degenerate pairs (chosen == rejected).
    2. Filter to gender-axis examples via is_gender_example().
    3. Load the local gender extension JSONL.
    4. Concatenate, deduplicate on (prompt, chosen, rejected) triple.
    5. Split the deduplicated data into train/validation sets.

    Unlike DPO, the final SFT training uses only the chosen responses as
    positive examples.

    Returns:
        DatasetDict with keys 'train' and 'test'.
    """
    biasdpo = load_dataset("ahmedallam/BiasDPO", split="train")
    print(f"[Data] Full BiasDPO size: {len(biasdpo)}")
    biasdpo = biasdpo.filter(
        lambda x: bool(x["prompt"]) and bool(x["chosen"]) and bool(x["rejected"])
                  and x["chosen"].strip() != x["rejected"].strip() # drop pairs with identical chosen/rejected — no DPO signal
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

    The prompt and completion are returned as separate fields for TRL's SFT
    trainer. In these SFT runs, completion_only_loss was not enabled, so the
    loss is applied to the full formatted sequence rather than being masked
    to the completion tokens only.

    The leading space before the completion preserves the expected separation
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

    # First-stage adapter: model previously aligned on the race-extended SFT data.
    # This adapter is merged into the base model before starting gender-SFT training.
    ADAPTER_ID = "../checkpoints/mistral7b_biasdpo_race_extended_sft"
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    
    from peft import PeftModel
    print(f"Loading and merging previous adapter: {ADAPTER_ID}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
    # Merge the race-SFT LoRA adapter into the model weights so the next LoRA
    # adapter starts from the race-aligned model rather than from the base model.
    model = model.merge_and_unload()
    print("Adapter merged successfully! Starting sequential training.")

    # A fresh LoRA adapter is trained for the second stage so that gender
    # alignment is learned on top of the merged race-aligned model.
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

    print("Starting sequential Race-SFT → Gender-SFT training on Mistral-7B-v0.3 (LoRA, LR=5e-5)...")
    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Done. Saved to {OUTPUT_DIR}")
