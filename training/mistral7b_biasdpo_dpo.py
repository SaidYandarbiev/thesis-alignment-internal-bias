"""
DPO Script — BiasDPO on Mistral-7B-v0.3 (LoRA)
=================================================
Dataset: ahmedallam/BiasDPO (1,145 examples)
Model:   mistralai/Mistral-7B-v0.3 (base, unaligned)
LR:      5e-5 
"""

import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig

# Disable TorchDynamo compilation — avoids graph-break errors that occur
# when PEFT's LoRA hooks interact with torch.compile.
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

MODEL_ID       = "mistralai/Mistral-7B-v0.3"
OUTPUT_DIR     = "../checkpoints/mistral7b_biasdpo_dpo"
SEED           = 42
MAX_SEQ_LENGTH = 512

LEARNING_RATE  = 5e-5
NUM_EPOCHS     = 3
BATCH_SIZE     = 2
GRAD_ACCUM     = 4
BETA           = 0.1
VAL_SPLIT      = 0.1

# Set globally so filter functions produce deterministic results
set_seed(SEED)


def load_and_preprocess():
    """
    Load the BiasDPO dataset, remove degenerate pairs, and split into
    train/validation sets.

    Filters out any example where the chosen and rejected responses are
    identical (after stripping whitespace), as these provide no learning
    signal for DPO.

    Returns:
        DatasetDict with keys 'train' and 'test'.
    """
    dataset = load_dataset("ahmedallam/BiasDPO", split="train")
    dataset = dataset.filter(
        lambda x: bool(x["prompt"]) and bool(x["chosen"]) and bool(x["rejected"])
                  and x["chosen"].strip() != x["rejected"].strip() # drop pairs with identical chosen/rejected — no DPO signal
    )
    split = dataset.train_test_split(test_size=VAL_SPLIT, seed=SEED)
    print(f"[Data] Train: {len(split['train'])}, Validation: {len(split['test'])}")
    return split


def format_for_dpo(example, tokenizer):
    """
    Wrap the prompt in Mistral's [INST]...[/INST] instruction tags and
    prepend a leading space to the chosen/rejected responses.

    The leading space is required by TRL's DPOTrainer to correctly
    tokenize the response boundary.
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
        # beta controls the KL penalty strength; 0.1 is the standard DPO starting point
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

    print("Starting BiasDPO DPO on Mistral-7B-v0.3 (LoRA, LR=5e-5)...")
    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model and tokenizer saved to {OUTPUT_DIR}")
