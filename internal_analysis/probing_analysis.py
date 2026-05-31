"""
Internal Association Analysis — Method 1: Probing Classifiers
==============================================================
Trains lightweight linear classifiers on frozen hidden states to test
whether bias-relevant information is encoded at each layer of the model.

Two probe modes (selected via --probe-type):

  demographic  (original):
    Sentences contain explicit demographic group words (man/woman, Black/White,
    Christian/Muslim, wealthy/poor). Probe asks: "Can the model distinguish
    demographic groups in its representations?"

  stereotype  (added Apr 28, 2026):
    Sentences contain NO demographic group words — only stereotype-associated
    traits and roles. Probe asks: "Does the model still encode the stereotype
    axis itself, even with no demographic label to latch onto?"

The stereotype probe is the stronger test for the silenced-bias hypothesis:
above-chance accuracy here means the stereotype association is a learnable
internal axis, regardless of whether surface bias outputs were aligned away.

Methodology:
    1. Feed probe sentences through each checkpoint
    2. Extract hidden states at all layers
    3. Train a logistic regression probe per layer per category
    4. Compare probe accuracy across checkpoints and layers
    5. Statistical testing for significant differences

Output:
    - Layerwise probe accuracy plots (per category, per checkpoint)
    - Accuracy difference heatmaps (aligned vs baseline)
    - JSON metrics for thesis tables

Usage:
    python probing_analysis.py \
        --output-dir ./analysis/internal/probing \
        --pooling last_token \
        --probe-type stereotype
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from collections import defaultdict

from internal_utils import (
    load_checkpoint, unload_model, extract_hidden_states,
    get_probe_sentences, get_stereotype_probe_sentences,
    get_stereotype_holdout_buckets,
    load_checkpoints_config, get_checkpoints,
    get_palette, get_model_labels,
    LAYERS, CATEGORIES, NUM_LAYERS,
)
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =====================================================
# PROBE-TYPE DISPATCH
# =====================================================

def get_sentences_for_probe_type(probe_type: str) -> dict:
    """Return the sentence dictionary for the requested probe type."""
    if probe_type == "demographic":
        return get_probe_sentences()
    elif probe_type in ("stereotype", "stereotype_lobo"):
        # LOBO uses the same sentences as the stereotype probe; only the CV differs.
        return get_stereotype_probe_sentences()
    else:
        raise ValueError(f"Unknown probe_type: {probe_type!r} "
                         f"(expected 'demographic', 'stereotype', or 'stereotype_lobo')")


def assign_sentence_to_bucket(sentence: str, buckets: list) -> int:
    """For LOBO: find the index of the bucket containing the first stereotype
    word that appears (word-boundary, case-insensitive) in the sentence.

    Returns -1 if no stereotype word from any bucket matches — caller should
    treat such sentences as un-assignable (this should never happen if the
    sentence dataset has been audited; we raise downstream if it does).
    """
    for bucket_idx, words in enumerate(buckets):
        for w in words:
            pattern = r'\b' + re.escape(w) + r'\b'
            if re.search(pattern, sentence, re.IGNORECASE):
                return bucket_idx
    return -1


# =====================================================
# HIDDEN STATE EXTRACTION FOR PROBING
# =====================================================

def extract_probe_data(model, tokenizer, category: str, pooling: str,
                       probe_type: str = "stereotype") -> tuple:
    """
    Extract hidden states for all probe sentences in a category.

    Returns:
        X: dict of {layer_idx: np.array of shape (n_sentences, hidden_dim)}
        y: np.array of string labels (group names)
        fold_ids: np.array of fold-assignment indices for LOBO; only populated
                  when probe_type == 'stereotype_lobo', else returns None.
                  Sentences from different poles may share fold indices —
                  i.e. fold-k holds out bucket-k from BOTH poles, so the
                  test set has sentences from both poles.
    """
    probe_sentences = get_sentences_for_probe_type(probe_type)
    cat_data = probe_sentences.get(category, {})

    if not cat_data:
        logger.warning(f"No probe sentences for category '{category}' "
                       f"under probe_type='{probe_type}'")
        return {}, np.array([]), None

    # If LOBO, prepare bucket lookups
    bucket_lookup = None
    if probe_type == "stereotype_lobo":
        all_buckets = get_stereotype_holdout_buckets()
        cat_buckets = all_buckets.get(category)
        if cat_buckets is None:
            raise ValueError(f"No LOBO buckets defined for category {category!r}")
        bucket_lookup = cat_buckets  # {pole_name: [[words], [words], ...]}

    all_sentences = []
    all_labels = []
    all_fold_ids = []

    for group_name, sentences in cat_data.items():
        for sent in sentences:
            all_sentences.append(sent)
            all_labels.append(group_name)
            if bucket_lookup is not None:
                pole_buckets = bucket_lookup.get(group_name)
                if pole_buckets is None:
                    raise ValueError(
                        f"LOBO buckets missing for {category}.{group_name}"
                    )
                fold_id = assign_sentence_to_bucket(sent, pole_buckets)
                if fold_id < 0:
                    raise ValueError(
                        f"LOBO: sentence has no matching stereotype word in any "
                        f"bucket for {category}.{group_name}: {sent!r}"
                    )
                all_fold_ids.append(fold_id)

    y = np.array(all_labels)
    fold_ids = np.array(all_fold_ids) if bucket_lookup is not None else None

    # Extract hidden states for all sentences
    X_by_layer = defaultdict(list)

    for i, sent in enumerate(all_sentences):
        if i % 10 == 0:
            logger.info(f"    Extracting sentence {i+1}/{len(all_sentences)}")

        hs = extract_hidden_states(model, tokenizer, sent, layers=LAYERS, pooling=pooling)

        for layer_idx, vec in hs.items():
            X_by_layer[layer_idx].append(vec)

    X = {layer: np.array(vecs) for layer, vecs in X_by_layer.items()}

    return X, y, fold_ids


# =====================================================
# PROBING WITH CROSS-VALIDATION
# =====================================================

def probe_layer(X: np.ndarray, y: np.ndarray, n_folds: int = 5, seed: int = 42) -> dict:
    """
    Train and evaluate a linear probe using stratified k-fold cross-validation.

    Uses logistic regression (linear classifier) — this is intentional.
    A linear probe tests whether information is *linearly accessible*
    in the representation, which is the standard for probing studies.
    Using a more powerful classifier would muddy the interpretation.

    Returns:
        {"accuracy_mean": float, "accuracy_std": float, "fold_accuracies": list}
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Cannot train a classifier with only one class; return zero accuracy
    if len(np.unique(y_encoded)) < 2:
        return {"accuracy_mean": 0.0, "accuracy_std": 0.0, "fold_accuracies": []}

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_accuracies = []

    for train_idx, test_idx in skf.split(X, y_encoded):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        # L2-regularized logistic regression; lbfgs handles multiclass natively, C=1.0 is standard
        clf = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            C=1.0,
            random_state=seed,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        fold_accuracies.append(accuracy_score(y_test, y_pred))

    return {
        "accuracy_mean": float(np.mean(fold_accuracies)),
        "accuracy_std": float(np.std(fold_accuracies)),
        "fold_accuracies": [float(a) for a in fold_accuracies],
    }


def probe_layer_lobo(X: np.ndarray, y: np.ndarray, fold_ids: np.ndarray,
                     seed: int = 42) -> dict:
    """
    Train and evaluate a linear probe using leave-one-bucket-out CV.

    Each unique value in fold_ids defines one fold: that fold holds out all
    sentences with that fold_id (drawn from BOTH poles — one bucket per pole),
    trains on the remainder, and evaluates on the heldout test set.

    Probing under LOBO tests whether the model encodes the abstract stereotype
    axis or merely memorizes specific lexical items, because at test time the
    probe sees sentences containing words it was never trained on.

    Returns:
        {"accuracy_mean", "accuracy_std", "fold_accuracies", "n_folds"}
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    if len(np.unique(y_encoded)) < 2:
        return {"accuracy_mean": 0.0, "accuracy_std": 0.0,
                "fold_accuracies": [], "n_folds": 0}

    unique_folds = sorted(np.unique(fold_ids).tolist())
    fold_accuracies = []

    for fold_k in unique_folds:
        test_mask = fold_ids == fold_k
        train_mask = ~test_mask

        # Skip fold if test set has no examples, or if train set lacks both classes
        if test_mask.sum() == 0:
            continue
        if len(np.unique(y_encoded[train_mask])) < 2:
            logger.warning(f"  LOBO fold {fold_k}: training set has only one class, skipping.")
            continue
        
        # lbfgs handles multiclass natively; C=1.0 is standard regularisation strength
        clf = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            C=1.0,
            random_state=seed,
        )
        clf.fit(X[train_mask], y_encoded[train_mask])
        y_pred = clf.predict(X[test_mask])
        fold_accuracies.append(accuracy_score(y_encoded[test_mask], y_pred))

    if not fold_accuracies:
        return {"accuracy_mean": 0.0, "accuracy_std": 0.0,
                "fold_accuracies": [], "n_folds": 0}

    return {
        "accuracy_mean": float(np.mean(fold_accuracies)),
        "accuracy_std": float(np.std(fold_accuracies)),
        "fold_accuracies": [float(a) for a in fold_accuracies],
        "n_folds": len(fold_accuracies),
    }


# =====================================================
# FULL PROBING PIPELINE
# =====================================================

def run_probing_analysis(checkpoints: list, output_dir: str,
                         pooling: str = "last_token",
                         probe_type: str = "stereotype") -> dict:
    """
    Run the complete probing analysis across all checkpoints, categories, and layers.

    Args:
        checkpoints: list of checkpoint dicts (from a loaded config)
        output_dir: where to write outputs
        pooling: hidden-state pooling strategy
        probe_type: 'demographic', 'stereotype', or 'stereotype_lobo'

    Returns:
        {checkpoint_name: {category: {layer_idx: {accuracy_mean, accuracy_std, ...}}}}
    """
    all_results = {}

    for checkpoint in checkpoints:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing checkpoint: {checkpoint['label']}")
        logger.info(f"Probe type: {probe_type}")
        logger.info(f"{'='*60}")

        model, tokenizer = load_checkpoint(checkpoint)

        checkpoint_results = {}

        for category in CATEGORIES:
            logger.info(f"\n  Category: {category}")

            # Extract hidden states (returns fold_ids only for LOBO mode)
            X, y, fold_ids = extract_probe_data(model, tokenizer, category, pooling, probe_type)

            if len(y) == 0:
                logger.warning(f"  No data for {category}, skipping.")
                continue

            logger.info(f"  Extracted {len(y)} sentences, {len(np.unique(y))} groups")
            if fold_ids is not None:
                # Log how many sentences fall into each fold to catch imbalanced buckets
                fold_dist = {int(k): int(v) for k, v in
                             zip(*np.unique(fold_ids, return_counts=True))}
                logger.info(f"  LOBO fold distribution: {fold_dist}")

            # Probe at each layer
            layer_results = {}
            for layer_idx in LAYERS:
                if probe_type == "stereotype_lobo":
                    result = probe_layer_lobo(X[layer_idx], y, fold_ids)
                else:
                    result = probe_layer(X[layer_idx], y)
                layer_results[layer_idx] = result

                # Log every 5th layer to avoid flooding output across 32 layers
                if layer_idx % 5 == 0:
                    logger.info(
                        f"    Layer {layer_idx:2d}: "
                        f"accuracy = {result['accuracy_mean']:.3f} "
                        f"(±{result['accuracy_std']:.3f})"
                    )

            checkpoint_results[category] = layer_results

        all_results[checkpoint["name"]] = checkpoint_results

        unload_model(model)
        del tokenizer

    return all_results


# =====================================================
# REPORTING
# =====================================================

def print_summary(results: dict, checkpoints: list,
                  probe_type: str = "stereotype") -> None:
    """Print a summary table of peak probe accuracies."""
    print("\n" + "=" * 80)
    print(f"Probing Analysis Summary ({probe_type}): Peak Accuracy per Category")
    print("=" * 80)

    header = f"{'Model':<22} {'Category':<16} {'Peak Acc':>9} {'Peak Layer':>11} {'Chance':>8}"
    print(header)
    print("-" * 80)

    probe_sentences = get_sentences_for_probe_type(probe_type)

    for model_name in [c["name"] for c in checkpoints]:
        if model_name not in results:
            continue
        for category in CATEGORIES:
            if category not in results[model_name]:
                continue

            layer_results = results[model_name][category]
            best_layer = max(layer_results, key=lambda l: layer_results[l]["accuracy_mean"])
            best_acc = layer_results[best_layer]["accuracy_mean"]

            # Chance level = 1 / num_groups
            n_groups = len(probe_sentences.get(category, {}))
            chance = 1.0 / n_groups if n_groups > 0 else 0.0

            label = model_name if category == CATEGORIES[0] else ""
            print(
                f"{label:<22} {category:<16} {best_acc:>9.3f} "
                f"{best_layer:>11d} {chance:>8.2f}"
            )

        print("-" * 80)


def save_results(results: dict, output_dir: str, probe_type: str = "stereotype") -> None:
    """Save full results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"probing_results_{probe_type}.json")

    # Convert numpy types for JSON serialization
    serializable = {}
    for model, categories in results.items():
        serializable[model] = {}
        for cat, layers in categories.items():
            serializable[model][cat] = {
                str(layer): metrics for layer, metrics in layers.items()
            }

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Saved results → {path}")


# =====================================================
# VISUALISATION
# =====================================================

def plot_layerwise_accuracy(results: dict, checkpoints: list,
                            palette: dict, model_labels: dict,
                            output_dir: str,
                            probe_type: str = "stereotype") -> None:
    """
    Line plot: probe accuracy across layers for each checkpoint.
    One subplot per category.
    """
    os.makedirs(output_dir, exist_ok=True)

    n_cats = len(CATEGORIES)
    fig, axes = plt.subplots(1, n_cats, figsize=(5 * n_cats, 5), sharey=True)
    if n_cats == 1:
        axes = [axes]

    probe_sentences = get_sentences_for_probe_type(probe_type)

    for col, category in enumerate(CATEGORIES):
        ax = axes[col]

        for model_name in [c["name"] for c in checkpoints]:
            if model_name not in results or category not in results[model_name]:
                continue

            layer_results = results[model_name][category]
            layers_sorted = sorted(layer_results.keys())
            accs = [layer_results[l]["accuracy_mean"] for l in layers_sorted]
            stds = [layer_results[l]["accuracy_std"] for l in layers_sorted]

            ax.plot(layers_sorted, accs,
                    label=model_labels.get(model_name, model_name),
                    color=palette.get(model_name, "#999"),
                    linewidth=2, alpha=0.85)
            ax.fill_between(layers_sorted,
                            np.array(accs) - np.array(stds),
                            np.array(accs) + np.array(stds),
                            alpha=0.1, color=palette.get(model_name, "#999"))

        # Chance line
        n_groups = len(probe_sentences.get(category, {}))
        if n_groups > 0:
            ax.axhline(1.0 / n_groups, color="black", linestyle="--",
                       linewidth=1, alpha=0.5, label="Chance")

        ax.set_title(category.capitalize(), fontsize=12)
        ax.set_xlabel("Layer", fontsize=10)
        if col == 0:
            cv_label = ("LOBO CV" if probe_type == "stereotype_lobo" else "5-fold CV")
            ax.set_ylabel(f"Probe Accuracy ({cv_label})", fontsize=10)
        ax.set_xlim(0, NUM_LAYERS - 1)
        ax.set_ylim(0, 1.05)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].legend(loc="upper right", fontsize=8, framealpha=0.9)

    title_map = {
        "demographic": "Layerwise Probe Accuracy: Demographic Detection",
        "stereotype":  "Layerwise Probe Accuracy: Stereotype-Axis Encoding",
        "stereotype_lobo": "Layerwise Probe Accuracy: Stereotype-Axis (Leave-One-Bucket-Out)",
    }
    title = title_map.get(probe_type, f"Layerwise Probe Accuracy ({probe_type})")
    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()

    path = os.path.join(output_dir, f"probe_layerwise_accuracy_{probe_type}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {path}")


def plot_accuracy_delta_heatmap(results: dict, checkpoints: list,
                                model_labels: dict, output_dir: str,
                                probe_type: str = "stereotype") -> None:
    """
    Heatmap: change in probe accuracy vs baseline at each layer.
    Rows = aligned models, Columns = layers.
    One heatmap per category.
    Blue = accuracy decreased (alignment removed information).
    Red = accuracy increased (alignment added/strengthened encoding).
    """
    os.makedirs(output_dir, exist_ok=True)

    if "baseline" not in results:
        logger.warning("No baseline results — skipping delta heatmap.")
        return

    aligned_models = [c["name"] for c in checkpoints if c["name"] != "baseline" and c["name"] in results]

    for category in CATEGORIES:
        if category not in results.get("baseline", {}):
            continue

        baseline_accs = results["baseline"][category]

        matrix = np.full((len(aligned_models), NUM_LAYERS), np.nan)

        for i, model_name in enumerate(aligned_models):
            if category not in results.get(model_name, {}):
                continue
            model_accs = results[model_name][category]
            for layer in LAYERS:
                if layer in model_accs and layer in baseline_accs:
                    delta = model_accs[layer]["accuracy_mean"] - baseline_accs[layer]["accuracy_mean"]
                    matrix[i, layer] = delta

        fig, ax = plt.subplots(figsize=(14, 3))
        vmax = np.nanmax(np.abs(matrix)) if not np.all(np.isnan(matrix)) else 0.1
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")

        ax.set_xticks(range(0, NUM_LAYERS, 2))
        ax.set_xticklabels(range(0, NUM_LAYERS, 2), fontsize=8)
        ax.set_xlabel("Layer", fontsize=10)
        ax.set_yticks(range(len(aligned_models)))
        ax.set_yticklabels([model_labels.get(m, m) for m in aligned_models], fontsize=10)

        plt.colorbar(im, ax=ax, label="Δ Probe Accuracy vs Baseline")
        suffix_map = {
            "demographic": "Demographic Probe",
            "stereotype": "Stereotype Probe",
            "stereotype_lobo": "Stereotype LOBO Probe",
        }
        title_suffix = suffix_map.get(probe_type, probe_type)
        ax.set_title(f"Probe Accuracy Change After Alignment "
                     f"({title_suffix}) — {category.capitalize()}",
                     fontsize=12, pad=10)
        plt.tight_layout()

        path = os.path.join(output_dir, f"probe_delta_heatmap_{probe_type}_{category}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved → {path}")


# =====================================================
# MAIN
# =====================================================

#Define and parse CLI arguments.
def parse_args():
    parser = argparse.ArgumentParser(
        description="Probing classifier analysis for internal bias detection."
    )
    parser.add_argument("--checkpoints-config", required=True,
                        help="Path to JSON config file describing the checkpoints "
                             "to analyze (e.g., configs/biasdpo_full.json).")
    parser.add_argument("--output-dir", default="./analysis/internal/probing")
    parser.add_argument("--pooling", default="last_token",
                        choices=["last_token", "mean", "first_token"])
    parser.add_argument("--probe-type", default="stereotype",
                        choices=["demographic", "stereotype", "stereotype_lobo"],
                        help="Which probe sentence set / CV strategy to use. "
                             "'demographic' = explicit group words, random 5-fold CV; "
                             "'stereotype' = stereotype-associated traits with no "
                             "demographic words, random 5-fold CV; "
                             "'stereotype_lobo' = same sentences as 'stereotype' "
                             "but leave-one-bucket-out CV (probe must generalize "
                             "to held-out stereotype words). Default: stereotype.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = load_checkpoints_config(args.checkpoints_config)
    checkpoints = get_checkpoints(cfg)
    palette = get_palette(cfg)
    model_labels = get_model_labels(cfg)
    logger.info(f"Experiment: {cfg['experiment_name']}")
    logger.info(f"Checkpoints: {[c['name'] for c in checkpoints]}")

    # Append probe-type subdirectory so demographic & stereotype runs don't collide
    output_dir = os.path.join(args.output_dir, args.probe_type)
    logger.info(f"Probe type: {args.probe_type}")
    logger.info(f"Output directory: {output_dir}")

    results = run_probing_analysis(checkpoints, output_dir,
                                   pooling=args.pooling,
                                   probe_type=args.probe_type)

    print_summary(results, checkpoints, probe_type=args.probe_type)
    save_results(results, output_dir, probe_type=args.probe_type)
    plot_layerwise_accuracy(results, checkpoints, palette, model_labels,
                            output_dir, probe_type=args.probe_type)
    plot_accuracy_delta_heatmap(results, checkpoints, model_labels,
                                output_dir, probe_type=args.probe_type)

    logger.info(f"Done. All outputs in: {output_dir}")
