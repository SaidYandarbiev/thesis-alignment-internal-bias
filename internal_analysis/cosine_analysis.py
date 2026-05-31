"""
Internal Association Analysis — Method 2: Cosine Similarity
=============================================================
Measures the geometric association between social group representations
and stereotype attribute representations at each layer.

Core question answered:
    "How close are representations of [group X] to representations of
     [stereotypical attribute Y] in the model's hidden state space at layer L?"

Methodology:
    1. For each social group term and attribute term, generate a context sentence
    2. Extract hidden states at all layers
    3. Compute cosine similarity between group and attribute representations
    4. Aggregate and compare across checkpoints

Output:
    - Layerwise association strength plots
    - Association change heatmaps (aligned vs baseline)
    - Per-group association profiles

Usage:
    python cosine_analysis.py \
        --checkpoints-config configs/biasdpo_full.json \
        --output-dir ./analysis/internal/cosine
"""

import os
import json
import argparse
import logging
import numpy as np
import matplotlib
# Use non-interactive backend so plots can be saved on headless servers
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

from internal_utils import (
    load_checkpoint, unload_model, extract_hidden_states,
    get_social_groups, get_stereotype_attributes, get_context_sentence,
    load_checkpoints_config, get_checkpoints, get_palette, get_model_labels,
    LAYERS, CATEGORIES, NUM_LAYERS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =====================================================
# REPRESENTATION EXTRACTION
# =====================================================

def extract_concept_representations(
    model, tokenizer, word_list: list, pooling: str = "last_token"
) -> dict:
    """
    Extract layerwise representations for a list of words/phrases.
    Each word is placed in a fixed neutral frame ("This sentence is about X.")
    to ensure representations reflect the word itself rather than topic-specific context.
    Returns: {layer_idx: np.array of shape (n_words, hidden_dim)}
    """
    reps_by_layer = defaultdict(list)

    for word in word_list:
        sentence = get_context_sentence(word)
        hs = extract_hidden_states(model, tokenizer, sentence, layers=LAYERS, pooling=pooling)
        for layer_idx, vec in hs.items():
            reps_by_layer[layer_idx].append(vec)

    return {layer: np.array(vecs) for layer, vecs in reps_by_layer.items()}


def compute_group_attribute_similarity(group_reps: dict, attribute_reps: dict) -> dict:
    """
    Compute mean cosine similarity between a group's representations
    and an attribute set's representations at each layer.
    """
    result = {}
    for layer in LAYERS:
        if layer not in group_reps or layer not in attribute_reps:
            continue
        g = group_reps[layer]
        a = attribute_reps[layer]
        sim_matrix = cosine_similarity(g, a)
        result[layer] = float(np.mean(sim_matrix))
    return result


# =====================================================
# FULL COSINE ANALYSIS PIPELINE
# =====================================================

def run_cosine_analysis(checkpoints: list, output_dir: str,
                        pooling: str = "last_token") -> dict:
    """
    Run layerwise cosine similarity analysis for all checkpoints and categories.
    Returns a nested dict: {checkpoint_name: {category: {group: {attribute: {layer: float}}}}}
    """
    social_groups = get_social_groups()
    stereotype_attrs = get_stereotype_attributes()
    all_results = {}

    for checkpoint in checkpoints:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {checkpoint['label']}")
        logger.info(f"{'='*60}")

        model, tokenizer = load_checkpoint(checkpoint)
        checkpoint_results = {}

        for category in CATEGORIES:
            logger.info(f"\n  Category: {category}")

            groups = social_groups.get(category, {})
            attrs = stereotype_attrs.get(category, {})

            # Skip categories with no defined groups or attributes in the config
            if not groups or not attrs:
                continue

            # Extract attribute representations once and reuse across all groups in this category
            attr_reps = {}
            for attr_name, attr_words in attrs.items():
                logger.info(f"    Extracting attribute: {attr_name} ({len(attr_words)} words)")
                attr_reps[attr_name] = extract_concept_representations(
                    model, tokenizer, attr_words, pooling
                )

            # Extract group representations and compute similarities
            cat_results = {}
            for group_name, group_words in groups.items():
                logger.info(f"    Extracting group: {group_name} ({len(group_words)} words)")
                group_rep = extract_concept_representations(
                    model, tokenizer, group_words, pooling
                )

                group_sims = {}
                for attr_name, attr_rep in attr_reps.items():
                    sim = compute_group_attribute_similarity(group_rep, attr_rep)
                    group_sims[attr_name] = sim

                cat_results[group_name] = group_sims

            checkpoint_results[category] = cat_results

        all_results[checkpoint["name"]] = checkpoint_results
        
        # Explicitly free GPU memory before loading the next checkpoint
        unload_model(model)
        del tokenizer

    return all_results


# =====================================================
# DERIVED METRICS
# =====================================================

def compute_bias_direction_score(results: dict, category: str, checkpoint: str) -> dict:
    """
    Compute a directional bias score at each layer.
    Positive = associations align with stereotypes; negative = counter-stereotypical.

    For gender/socioeconomic: averages (own-stereotype - cross-stereotype) similarity.
    For race/religion: averages (negative - neutral) similarity across all groups.
    Returns: {layer_idx: float}
    """
    cat_results = results.get(checkpoint, {}).get(category, {})
    if not cat_results:
        return {}

    if category == "gender":
        scores = {}
        for layer in LAYERS:
            try:
                f_stereo = cat_results["female"]["female_stereotypes"].get(layer, 0)
                f_counter = cat_results["female"]["male_stereotypes"].get(layer, 0)
                m_stereo = cat_results["male"]["male_stereotypes"].get(layer, 0)
                m_counter = cat_results["male"]["female_stereotypes"].get(layer, 0)
                # Bias score = how much more a group associates with its own stereotypes
                # than with the opposing group's stereotypes
                scores[layer] = ((f_stereo - f_counter) + (m_stereo - m_counter)) / 2
            except (KeyError, TypeError):
                scores[layer] = 0.0
        return scores

    
    elif category == "race":
        scores = {}
        for layer in LAYERS:
            layer_scores = []
            for group in cat_results:
                try:
                    neg = cat_results[group]["negative"].get(layer, 0)
                    neu = cat_results[group]["neutral"].get(layer, 0)
                    layer_scores.append(neg - neu)
                except (KeyError, TypeError):
                    pass
            scores[layer] = float(np.mean(layer_scores)) if layer_scores else 0.0
        return scores

    # Same logic as race: average (negative - neutral) association across all groups
    elif category == "religion":
        scores = {}
        for layer in LAYERS:
            layer_scores = []
            for group in cat_results:
                try:
                    neg = cat_results[group]["negative"].get(layer, 0)
                    neu = cat_results[group]["neutral"].get(layer, 0)
                    layer_scores.append(neg - neu)
                except (KeyError, TypeError):
                    pass
            scores[layer] = float(np.mean(layer_scores)) if layer_scores else 0.0
        return scores

    elif category == "socioeconomic":
        scores = {}
        for layer in LAYERS:
            try:
                p_stereo = cat_results["poor"]["poor_stereotypes"].get(layer, 0)
                p_neutral = cat_results["poor"]["neutral"].get(layer, 0)
                w_stereo = cat_results["wealthy"]["wealthy_stereotypes"].get(layer, 0)
                w_neutral = cat_results["wealthy"]["neutral"].get(layer, 0)
                # Score = how much more each class associates with its own stereotypes than neutral terms
                scores[layer] = ((p_stereo - p_neutral) + (w_stereo - w_neutral)) / 2
            except (KeyError, TypeError):
                scores[layer] = 0.0
        return scores

    return {}


# =====================================================
# REPORTING & SAVING
# =====================================================

def save_results(results: dict, output_dir: str) -> None:
    """Save results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "cosine_similarity_results.json")

    serializable = {}
    for model, categories in results.items():
        serializable[model] = {}
        for cat, groups in categories.items():
            serializable[model][cat] = {}
            for group, attrs in groups.items():
                serializable[model][cat][group] = {}
                for attr, layers in attrs.items():
                    # JSON requires string keys; convert integer layer indices accordingly
                    serializable[model][cat][group][attr] = {
                        str(k): v for k, v in layers.items()
                    }

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Saved → {path}")


# =====================================================
# VISUALISATION
# =====================================================

def plot_bias_direction_by_layer(results: dict, checkpoints: list,
                                 palette: dict, model_labels: dict,
                                 output_dir: str) -> None:
    """
    Line plot: directional bias score across layers for each checkpoint.
    """
    os.makedirs(output_dir, exist_ok=True)

    n_cats = len(CATEGORIES)
    fig, axes = plt.subplots(1, n_cats, figsize=(5 * n_cats, 5), sharey=False)
    # plt.subplots returns a single Axes object when n_cats == 1; wrap it for uniform iteration
    if n_cats == 1:
        axes = [axes]

    for col, category in enumerate(CATEGORIES):
        ax = axes[col]

        for checkpoint in checkpoints:
            model_name = checkpoint["name"]
            scores = compute_bias_direction_score(results, category, model_name)
            if not scores:
                continue

            layers_sorted = sorted(scores.keys())
            vals = [scores[l] for l in layers_sorted]

            ax.plot(layers_sorted, vals,
                    label=model_labels.get(model_name, model_name),
                    color=palette.get(model_name, "#999"),
                    linewidth=2, alpha=0.85)

        ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_title(category.capitalize(), fontsize=12)
        ax.set_xlabel("Layer", fontsize=10)
        if col == 0:
            ax.set_ylabel("Bias Direction Score\n(positive = stereotypical)", fontsize=10)
        ax.set_xlim(0, NUM_LAYERS - 1)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].legend(loc="best", fontsize=8, framealpha=0.9)
    fig.suptitle("Layerwise Stereotype Association Strength (Cosine Similarity)",
                 fontsize=13, y=1.02)
    plt.tight_layout()

    path = os.path.join(output_dir, "cosine_bias_direction.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {path}")


def plot_association_delta_heatmap(results: dict, checkpoints: list,
                                   model_labels: dict, output_dir: str) -> None:
    """
    Heatmap: change in bias direction score vs baseline.
    """
    os.makedirs(output_dir, exist_ok=True)

    if "baseline" not in results:
        # Delta heatmap requires a baseline to compare against; skip if absent
        return

    aligned_models = [c["name"] for c in checkpoints
                      if c["name"] != "baseline" and c["name"] in results]

    for category in CATEGORIES:
        baseline_scores = compute_bias_direction_score(results, category, "baseline")
        if not baseline_scores:
            continue

        matrix = np.full((len(aligned_models), NUM_LAYERS), np.nan)

        for i, model_name in enumerate(aligned_models):
            model_scores = compute_bias_direction_score(results, category, model_name)
            for layer in LAYERS:
                if layer in model_scores and layer in baseline_scores:
                    matrix[i, layer] = model_scores[layer] - baseline_scores[layer]

        fig, ax = plt.subplots(figsize=(14, 3))
        vmax = np.nanmax(np.abs(matrix)) if not np.all(np.isnan(matrix)) else 0.01
        # Guard against a zero-range colormap if all deltas are exactly zero
        if vmax == 0:
            vmax = 0.01
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")

        ax.set_xticks(range(0, NUM_LAYERS, 2))
        ax.set_xticklabels(range(0, NUM_LAYERS, 2), fontsize=8)
        ax.set_xlabel("Layer", fontsize=10)
        ax.set_yticks(range(len(aligned_models)))
        ax.set_yticklabels([model_labels.get(m, m) for m in aligned_models], fontsize=10)

        plt.colorbar(im, ax=ax, label="Δ Bias Direction Score vs Baseline")
        ax.set_title(
            f"Stereotype Association Change After Alignment — {category.capitalize()}",
            fontsize=12, pad=10,
        )
        plt.tight_layout()

        path = os.path.join(output_dir, f"cosine_delta_heatmap_{category}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved → {path}")


# =====================================================
# MAIN
# =====================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Cosine similarity analysis for internal bias associations."
    )
    parser.add_argument("--checkpoints-config", required=True,
                        help="Path to JSON config file describing the checkpoints "
                             "to analyze (e.g., configs/biasdpo_full.json).")
    parser.add_argument("--output-dir", default="./analysis/internal/cosine")
    parser.add_argument("--pooling", default="last_token",
                        choices=["last_token", "mean", "first_token"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = load_checkpoints_config(args.checkpoints_config)
    checkpoints = get_checkpoints(cfg)
    palette = get_palette(cfg)
    model_labels = get_model_labels(cfg)
    logger.info(f"Experiment: {cfg['experiment_name']}")
    logger.info(f"Checkpoints: {[c['name'] for c in checkpoints]}")

    results = run_cosine_analysis(checkpoints, args.output_dir, pooling=args.pooling)

    save_results(results, args.output_dir)
    plot_bias_direction_by_layer(results, checkpoints, palette, model_labels,
                                 args.output_dir)
    plot_association_delta_heatmap(results, checkpoints, model_labels,
                                   args.output_dir)

    logger.info(f"Done. All outputs in: {args.output_dir}")
