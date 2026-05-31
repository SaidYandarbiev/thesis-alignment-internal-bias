"""
DPO Script — Sequential: Race DPO → Religion DPO on Mistral-7B-v0.3 (LoRA)
============================================================================
Implements sequential bias alignment: trains on religion bias data starting
from a model already fine-tuned on race bias, rather than from the base model.

Step 1: Load the pre-trained race DPO adapter.
Step 2: Merge the adapter into the base weights with merge_and_unload().
Step 3: Apply a fresh LoRA adapter and train on the religion-extended dataset.

This tests whether sequential alignment causes cross-bias interference —
a key research question of the thesis.
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
OUTPUT_DIR     = "checkpoints/seq_race_dpo_to_religion_dpo"
EXTENSION_PATH = "../data/extension/religion_extension_clean.jsonl"
SEED           = 42
MAX_SEQ_LENGTH = 512

LEARNING_RATE  = 5e-5
NUM_EPOCHS     = 3
BATCH_SIZE     = 2
GRAD_ACCUM     = 4
BETA           = 0.1
VAL_SPLIT      = 0.1

# ---- KEYWORD LISTS FOR BIAS CLASSIFICATION ----

# Religion-specific terms used to identify and filter religion-bias examples.
# Regex patterns use word boundaries (\b) to avoid partial matches.
# Kept in sync with the religion-extended SFT/DPO scripts.
STRONG_RELIGION = [
    r"\breligio\w*\b", r"\bfaith\b", r"\bfaiths\b",
    r"\btheist\w*\b", r"\batheist\w*\b", r"\bagnostic\w*\b",
    r"\bMuslim\b", r"\bMuslims\b", r"\bIslam\w*\b",
    r"\bChristian\b", r"\bChristians\b", r"\bChristianity\b",
    r"\bJewish\b", r"\bJew\b", r"\bJews\b", r"\bJudaism\b",
    r"\bHindu\b", r"\bHindus\b", r"\bHinduism\b",
    r"\bBuddhist\b", r"\bBuddhists\b", r"\bBuddhism\b",
    r"\bSikh\b", r"\bSikhs\b", r"\bSikhism\b",
    r"\bChurch\b", r"\bchurches\b",
    r"\bmosque\b", r"\bmosques\b",
    r"\bsynagogue\b", r"\bsynagogues\b",
    r"\btemple\b", r"\btemples\b",
    r"\bevangelical\w*\b", r"\bCatholic\w*\b", r"\bProtestant\w*\b",
    r"\bMormon\w*\b", r"\bLDS\b", r"\bLatter[- ]day\b",
    r"\bBaptist\w*\b", r"\bMethodist\w*\b", r"\bPentecostal\w*\b",
    r"\bQuaker\w*\b", r"\bJain\w*\b", r"\bBaha[i\u2019\u2018\u02bc\u0027]\w*\b",
    r"\bZoroastri\w*\b", r"\bRastafar\w*\b",
    r"\bShinto\w*\b", r"\bWicca\w*\b",
    r"\bRamadan\b", r"\bChristmas\b", r"\bDiwali\b",
    r"\bHanukkah\b", r"\bEid\b", r"\bPassover\b",
    r"\bSharia\b", r"\bQuran\b", r"\bKoran\b",
    r"\bBible\b", r"\bbiblical\b", r"\bGospel\w*\b",
    r"\bTorah\b", r"\btheolog\w*\b",
    r"\bdenomination\w*\b",
    r"\bworship\w*\b", r"\bpray(?:er|ers|ing|ed)?\b",
    r"\bmissionar\w*\b", r"\bproselytiz\w*\b",
    r"\bclergy\b", r"\bpriest\w*\b", r"\bimam\w*\b", r"\brabbi\w*\b",
    r"\bmonk\w*\b", r"\bnun\w*\b",
    r"\bsecular\w*\b", r"\bspiritual(?:ly|ity)?\b",
    r"\bislamopho\w*\b", r"\bantisemit\w*\b",
    r"\bterrorism\b", r"\bterrorist\w*\b",
]

# Examples matching any gender keyword are excluded to keep this second-stage
# dataset strictly single-axis religion. This helps isolate whether the prior
# race alignment is retained or disrupted during religion training.
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

# Race-overlapping examples are dropped to avoid mixing the previous race
# alignment axis with the current religion training data.
RACE_KEYWORDS = [
    r"\bracial\b", r"\bracism\b", r"\bracist\b",
    r"\bAfrican\s*American\w*\b", r"\bAsian\s*American\w*\b",
    r"\bLatino\b", r"\bLatina\b", r"\bLatinx\b", r"\bHispanic\w*\b",
    r"\bAsian\b", r"\bAsians\b",
    r"\bBlack\b", r"\bBlacks\b",
    r"\bWhite\b", r"\bWhites\b",
    r"\bIndigenous\b", r"\bNative\s*American\w*\b",
    r"\bArab\w*\b", r"\bMiddle\s*East\w*\b",
    r"\bChinese\b", r"\bKorean\w*\b", r"\bVietnamese\b", r"\bJapanese\b",
    r"\bMexican\w*\b", r"\bPacific\s*Islander\w*\b",
    r"\bimmigrant\w*\b", r"\bnationality\b", r"\bforeigner\w*\b",
    r"\bcolorblind\b", r"\bpeople of colou?r\b",
]

# Phrases that indicate multi-axis overlap even when religion words are present
# are dropped regardless of keyword counts.
PROMPT_EXCLUSIONS = [
    r"women of colou?r", r"people of colou?r", r"person of colou?r",
    r"Muslim\s+(?:woman|women|man|men|mother|father)",
    r"Christian\s+(?:woman|women|man|men|mother|father)",
    r"Jewish\s+(?:woman|women|man|men|mother|father)",
    r"Hindu\s+(?:woman|women|man|men|mother|father)",
    r"Black\s+(?:Muslims?|Christians?|Jews?)",
    r"White\s+(?:Muslims?|Christians?|Jews?)",
    r"Asian\s+(?:Muslims?|Christians?|Jews?|Buddhists?|Hindus?)",
]

# Set globally so dataset splitting and filtering remain reproducible.
set_seed(SEED)


def count_matches(patterns, text):
    """Return the number of regex patterns that match anywhere in the text."""
    return sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))


def is_excluded(text):
    """Return True if the text contains a phrase indicating multi-axis overlap."""
    return any(re.search(p, text, re.IGNORECASE) for p in PROMPT_EXCLUSIONS)


def is_religion_example(example):
    """
    Decide whether a BiasDPO-style example belongs to the religion-bias axis.

    BiasDPO does not provide explicit religion/gender/race labels, so examples
    are classified using keyword-based filtering over the prompt, chosen, and
    rejected fields.

    An example is kept only if it contains at least one strong religion keyword
    and contains no gender keywords, race keywords, or manually defined
    exclusion phrases. This keeps the second-stage dataset focused on
    single-axis religion bias after the initial race-DPO alignment.
    """
    text = " ".join([
        str(example.get("prompt", "")),
        str(example.get("chosen", "")),
        str(example.get("rejected", "")),
    ])

    if is_excluded(text):
        return False

    rel_strong = count_matches(STRONG_RELIGION, text)
    g = count_matches(GENDER_KEYWORDS, text)
    r = count_matches(RACE_KEYWORDS, text)

    return rel_strong >= 1 and g == 0 and r == 0


def load_and_preprocess():
    """
    Load and combine the religion-filtered BiasDPO split and the LLM-generated
    religion extension for the second stage of sequential DPO training.

    Steps:
    1. Load full BiasDPO, drop degenerate pairs (chosen == rejected).
    2. Filter to religion-axis examples via is_religion_example().
    3. Load the local religion extension JSONL.
    4. Concatenate, deduplicate on (prompt, chosen, rejected) triple.
    5. Split the deduplicated data into train/validation sets.

    Returns:
        DatasetDict with keys 'train' and 'test'.
    """
    biasdpo = load_dataset("ahmedallam/BiasDPO", split="train")
    print(f"[Data] Full BiasDPO size: {len(biasdpo)}")
    biasdpo = biasdpo.filter(
        lambda x: bool(x["prompt"]) and bool(x["chosen"]) and bool(x["rejected"])
                  and x["chosen"].strip() != x["rejected"].strip() # drop pairs with identical chosen/rejected — no DPO signal
    )
    biasdpo_filtered = biasdpo.filter(is_religion_example)
    print(f"[Data] BiasDPO after religion filter: {len(biasdpo_filtered)}")

    extension = load_dataset("json", data_files=EXTENSION_PATH, split="train")
    print(f"[Data] Extension size: {len(extension)}")

    # Keep only the DPO fields needed for training and deduplication.
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

    # First-stage adapter: model previously aligned on the race-extended DPO data.
    # This adapter is merged into the base model before starting religion-DPO training.
    ADAPTER_ID = "../checkpoints/mistral7b_biasdpo_race_extended_dpo"
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    
    from peft import PeftModel
    print(f"Loading and merging previous adapter: {ADAPTER_ID}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
    # Merge the race-DPO LoRA adapter into the model weights so the next LoRA
    # adapter starts from the race-aligned model rather than from the base model.
    model = model.merge_and_unload()
    print("Adapter merged successfully! Starting sequential training.")

    # A fresh LoRA adapter is trained for the second stage so that religion
    # alignment is learned on top of the merged race-aligned model.
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
        # With PEFT/LoRA, ref_model=None lets TRL use the frozen pre-DPO model
        # behaviour as the implicit reference model, avoiding a second full model
        # copy in memory.
        model=model, ref_model=None, args=training_args,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        processing_class=tokenizer, peft_config=peft_config,
    )

    print("Starting sequential Race-DPO → Religion-DPO training on Mistral-7B-v0.3 (LoRA, LR=5e-5)...")
    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Done. Saved to {OUTPUT_DIR}")
