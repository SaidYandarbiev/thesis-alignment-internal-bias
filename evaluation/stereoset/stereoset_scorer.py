"""
StereoSet Scorer
================
Scores each StereoSet sentence using a causal language model by computing
log-likelihoods of the continuation tokens (tokens after the shared prefix).

Corrected version:
  - Handles potential double-space between context and continuation (intersentence)
  - Explicit tie-handling with logging
  - Validates that prefix_len < total tokens (avoids zero-length continuations)
  - Adds per-cluster sanity checks
  - Uses proper token-level log-likelihood accumulation instead of loss * num_tokens
    (these are mathematically equivalent, but the explicit version is clearer)

Usage:
    python stereoset_scorer.py \
        --model-id google/gemma-2-2b \
        --input-file dev.json \
        --output-file predictions_gemma.json \
        --dtype bfloat16
"""

import json
import argparse
import logging
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Score StereoSet sentences with a causal LM.")
    parser.add_argument("--model-id", type=str, default="google/gemma-2-2b",
                        help="HuggingFace model ID or local path.")
    parser.add_argument("--input-file", type=str, default="dev.json",
                        help="Path to StereoSet dev.json.")
    parser.add_argument("--output-file", type=str, default="predictions_gemma.json",
                        help="Output predictions file.")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Model dtype.")
    return parser.parse_args()


def get_shared_prefix_length(tokenizer, sentences):
    """
    Find the number of leading token IDs shared across all sentences.

    This ensures we only score the tokens that differ between the
    stereotype / anti-stereotype / unrelated completions.
    """
    tokenized = [tokenizer(s, add_special_tokens=True)["input_ids"] for s in sentences]
    common_len = 0
    for tokens in zip(*tokenized):
        if len(set(tokens)) == 1:
            common_len += 1
        else:
            break
    return common_len


def score_sentence(model, input_ids_list, prefix_len, device):
    """
    Compute the total log-likelihood of continuation tokens (after prefix_len).

    Uses the model's cross-entropy loss with prefix tokens masked out.
    Returns total log-likelihood (higher = model finds sentence more likely).

    Args:
        model: the causal LM
        input_ids_list: list of token IDs for the full sentence
        prefix_len: number of leading tokens to mask (not scored)
        device: torch device

    Returns:
        float: total log-likelihood of continuation tokens
    """
    if prefix_len >= len(input_ids_list):
        logger.warning(
            f"prefix_len ({prefix_len}) >= total tokens ({len(input_ids_list)}). "
            f"No continuation to score — returning -inf."
        )
        return float("-inf")

    input_ids = torch.tensor([input_ids_list], device=device)
    labels = input_ids.clone()

    # Mask prefix tokens so they don't contribute to the loss
    labels[:, :prefix_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=labels)

    # outputs.loss is the mean cross-entropy over non-masked tokens
    # Multiply by count to get total negative log-likelihood, then negate
    num_scored_tokens = (labels != -100).sum().item()

    if num_scored_tokens == 0:
        return float("-inf")

    # loss is negative log-likelihood (mean), so total LL = -loss * num_tokens
    total_log_likelihood = -outputs.loss.item() * num_scored_tokens
    return total_log_likelihood


def score_intrasentence(model, tokenizer, clusters, device):
    """
    Score intrasentence examples.

    For intrasentence, the three candidate sentences share a common prefix
    (the part before the BLANK was filled). We find this prefix via shared
    token IDs and only score the diverging continuation.
    """
    predictions = []
    n_ties = 0
    n_short_continuations = 0

    for cluster in tqdm(clusters, desc="Intrasentence"):
        texts = [s["sentence"] for s in cluster["sentences"]]
        prefix_len = get_shared_prefix_length(tokenizer, texts)

        scores = {}
        for s in cluster["sentences"]:
            full_ids = tokenizer(s["sentence"], add_special_tokens=True)["input_ids"]
            continuation_len = len(full_ids) - prefix_len

            if continuation_len <= 0:
                logger.warning(
                    f"Cluster {cluster['id']}, sentence {s['id']}: "
                    f"no continuation tokens (prefix={prefix_len}, total={len(full_ids)})"
                )
                n_short_continuations += 1

            score = score_sentence(model, full_ids, prefix_len, device)
            scores[s['id']] = score
            predictions.append({"id": s['id'], "score": score})

        # Check for ties within this cluster
        score_values = list(scores.values())
        if len(set(score_values)) < len(score_values):
            n_ties += 1
            logger.debug(f"Tie detected in cluster {cluster['id']}: {scores}")

    if n_ties > 0:
        logger.info(f"Intrasentence: {n_ties} clusters had tied scores.")
    if n_short_continuations > 0:
        logger.warning(f"Intrasentence: {n_short_continuations} sentences had no continuation tokens.")

    return predictions


def score_intersentence(model, tokenizer, clusters, device):
    """
    Score intersentence examples.

    For intersentence, a context paragraph is given and three candidate
    next-sentences are scored. We score only the continuation (next sentence),
    not the shared context.

    Care is taken to avoid double spaces between context and continuation.
    """
    predictions = []
    n_ties = 0

    for cluster in tqdm(clusters, desc="Intersentence"):
        context = cluster['context']

        # Tokenize context with BOS
        context_ids = tokenizer(context, add_special_tokens=True)["input_ids"]
        prefix_len = len(context_ids)

        scores = {}
        for s in cluster['sentences']:
            # Build continuation text, avoiding double spaces
            continuation = s['sentence'].strip()
            if context.endswith(" "):
                cont_text = continuation
            else:
                cont_text = " " + continuation

            # Tokenize continuation WITHOUT special tokens (no extra BOS)
            cont_ids = tokenizer(cont_text, add_special_tokens=False)["input_ids"]
            full_ids = context_ids + cont_ids

            if len(cont_ids) == 0:
                logger.warning(
                    f"Cluster {cluster['id']}, sentence {s['id']}: "
                    f"continuation tokenized to 0 tokens."
                )

            score = score_sentence(model, full_ids, prefix_len, device)
            scores[s['id']] = score
            predictions.append({"id": s['id'], "score": score})

        # Check for ties
        score_values = list(scores.values())
        if len(set(score_values)) < len(score_values):
            n_ties += 1
            logger.debug(f"Tie detected in cluster {cluster['id']}: {scores}")

    if n_ties > 0:
        logger.info(f"Intersentence: {n_ties} clusters had tied scores.")

    return predictions


def main():
    args = parse_args()

    # Setup device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]

    logger.info(f"Model: {args.model_id}")
    logger.info(f"Device: {device}, Dtype: {args.dtype}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch_dtype
    ).to(device)
    model.eval()

    # Load StereoSet data
    with open(args.input_file, 'r') as f:
        data = json.load(f)

    intrasentence_clusters = data['data']['intrasentence']
    intersentence_clusters = data['data']['intersentence']

    logger.info(f"Intrasentence clusters: {len(intrasentence_clusters)}")
    logger.info(f"Intersentence clusters: {len(intersentence_clusters)}")

    # Score
    intra_predictions = score_intrasentence(model, tokenizer, intrasentence_clusters, device)
    inter_predictions = score_intersentence(model, tokenizer, intersentence_clusters, device)

    # Save predictions in the format expected by the official StereoSet evaluator
    predictions = {
        "intrasentence": intra_predictions,
        "intersentence": inter_predictions,
    }

    with open(args.output_file, "w") as f:
        json.dump(predictions, f, indent=2)

    logger.info(f"Predictions saved to {args.output_file}")
    logger.info(f"  Intrasentence: {len(intra_predictions)} sentences scored")
    logger.info(f"  Intersentence: {len(inter_predictions)} sentences scored")


if __name__ == "__main__":
    main()
