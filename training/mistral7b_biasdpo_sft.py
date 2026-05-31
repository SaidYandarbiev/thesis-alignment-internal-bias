"""
SFT Script — BiasDPO on Mistral-7B-v0.3 (LoRA)
=================================================
Dataset: ahmedallam/BiasDPO (1,145 examples, chosen responses)
Model:   mistralai/Mistral-7B-v0.3 (base, unaligned)
LR:      5e-5
"""

import os
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# Disable TorchDynamo compilation — avoids graph-break errors that occur
# when PEFT's LoRA hooks interact with torch.compile.
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

MODEL_ID       = "mistralai/Mistral-7B-v0.3"
OUTPUT_DIR     = "../checkpoints/mistral7b_biasdpo_sft"
SEED           = 42
MAX_SEQ_LENGTH = 512

LEARNING_RATE  = 5e-5
NUM_EPOCHS     = 3
BATCH_SIZE     = 4
GRAD_ACCUM     = 4
VAL_SPLIT      = 0.1

# Set globally so filter functions produce deterministic results
set_seed(SEED)


def load_and_preprocess():
    """
    Load the BiasDPO dataset, extract only the chosen (unbiased) responses,
    and split into train/validation sets.

    Unlike the DPO script, only the chosen response is kept — the rejected
    response is discarded because SFT trains on positive examples only.
    The filter requires both prompt and chosen to have more than 5 characters
    to remove near-empty entries.

    Returns:
        DatasetDict with keys 'train' and 'test'.
    """
    dataset = load_dataset("ahmedallam/BiasDPO", split="train")
    dataset = dataset.filter(
        lambda x: bool(x["prompt"]) and bool(x["chosen"]) and len(x["prompt"]) > 5 and len(x["chosen"]) > 5 # drop near-empty entries
    )

    prompts = [ex["prompt"] for ex in dataset]
    completions = [ex["chosen"] for ex in dataset]

    result = Dataset.from_dict({"prompt_text": prompts, "completion_text": completions})
    split = result.train_test_split(test_size=VAL_SPLIT, seed=SEED)
    print(f"[Data] Train: {len(split['train'])}, Validation: {len(split['test'])}")
    return split


def format_for_sft(example, tokenizer):
    """
    Wrap the prompt in Mistral's [INST]...[/INST] instruction tags and
    append the EOS token to the completion.

    The leading space before the completion and the trailing EOS token are
    required for SFTTrainer to correctly identify the response boundary
    when using completion_only_loss=True.
    """
    prompt = f"[INST] {example['prompt_text']} [/INST]"
    completion = " " + example["completion_text"] + tokenizer.eos_token
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
    print(f"[OK] Prompt: {repr(sample['prompt'][:150])}")
    print(f"[OK] Completion: {repr(sample['completion'][:150])}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
    )

    # LoRA rank 16 / alpha 32 gives a scaling factor of 2, a common starting
    # point for instruction-tuning. All seven projection matrices are targeted
    # to give the adapter sufficient capacity for response-style fine-tuning.
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
        seed=SEED,
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
        completion_only_loss=True, # compute loss only on the response tokens, not the [INST] prompt
    )

    trainer = SFTTrainer(
        model=model, processing_class=tokenizer, args=training_args,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    print("Starting BiasDPO SFT on Mistral-7B-v0.3 (LoRA, LR=5e-5)...")
    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model and tokenizer saved to {OUTPUT_DIR}")
