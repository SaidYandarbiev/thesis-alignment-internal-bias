"""
Internal Association Analysis — Method 3: Word Association Networks
====================================================================
Constructs networks of social concept associations from hidden state
representations and analyzes their topology to reveal bias structure.

Core question answered:
    "How are social groups, stereotypes, and attributes structurally
     connected in the model's internal representation space?"
    "Does alignment change the network topology of bias?"
    "Does reducing one type of bias affect the network structure of others?"
     (cross-bias interaction — the final research question)

Based on: Abramski et al. (2025) — "A Word Association Network Methodology
for Evaluating Implicit Biases in LLMs Compared to Humans"

Methodology:
    1. Extract representations of all social groups + attributes at each layer
    2. Compute pairwise cosine similarity → weighted adjacency matrix
    3. Construct a network: nodes = concepts, edges = similarity weights
    4. Analyze network properties:
       - Community structure (do stereotypical concepts cluster together?)
       - Group centrality (which groups are most central?)
       - Cross-category connectivity (are gender and race concepts linked?)
       - Modularity (how separated are different bias domains?)
    5. Compare networks across checkpoints to measure alignment effects

Output:
    - Network visualisations at key layers
    - Community structure analysis
    - Cross-bias connectivity metrics
    - Alignment effect comparisons

Usage:
    python network_analysis.py \
        --checkpoints-config configs/biasdpo_full.json \
        --output-dir ./analysis/internal/network
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
from itertools import combinations

from internal_utils import (
    load_checkpoint, unload_model, extract_hidden_states,
    get_social_groups, get_stereotype_attributes, get_context_sentence,
    load_checkpoints_config, get_checkpoints, get_palette, get_model_labels,
    LAYERS, CATEGORIES, NUM_LAYERS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# networkx is optional — required only for graph construction and metrics
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    logger.warning("networkx not installed. Install with: pip install networkx")

try:
    from community import community_louvain
    HAS_COMMUNITY = True
except ImportError:
    HAS_COMMUNITY = False
    logger.warning("python-louvain not installed. Install with: pip install python-louvain")


# =====================================================
# CONCEPT COLLECTION
# =====================================================

def collect_all_concepts() -> list:
    """
    Collect all concept words/phrases across all categories for network construction.
    """
    social_groups = get_social_groups()
    stereotype_attrs = get_stereotype_attributes()

    concepts = []

    for category in CATEGORIES:
        for group_name, words in social_groups.get(category, {}).items():
            for word in words:
                concepts.append({
                    "word": word,
                    "category": category,
                    "type": "group",
                    "subtype": group_name,
                })

        for attr_name, words in stereotype_attrs.get(category, {}).items():
            for word in words:
                # Skip if this word already appears for this category (groups and attributes can overlap)
                if not any(c["word"] == word and c["category"] == category for c in concepts):
                    concepts.append({
                        "word": word,
                        "category": category,
                        "type": "attribute",
                        "subtype": attr_name,
                    })

    logger.info(f"Collected {len(concepts)} unique concepts across {len(CATEGORIES)} categories.")
    return concepts


# =====================================================
# NETWORK CONSTRUCTION
# =====================================================

def build_similarity_matrix(
    model, tokenizer, concepts: list, layer: int, pooling: str = "last_token"
) -> np.ndarray:
    """
    Construct a weighted undirected graph from the similarity matrix.

    Args:
        threshold: Minimum cosine similarity to create an edge. Default 0.0
                includes all positively similar pairs; raise to sparsify the graph.
    """
    representations = []

    for concept in concepts:
        sentence = get_context_sentence(concept["word"])
        hs = extract_hidden_states(model, tokenizer, sentence, layers=[layer], pooling=pooling)
        representations.append(hs[layer])

    rep_matrix = np.array(representations)
    sim_matrix = cosine_similarity(rep_matrix)

    return sim_matrix


def build_network(sim_matrix: np.ndarray, concepts: list, threshold: float = 0.0):
    """Construct a weighted undirected graph from the similarity matrix."""
    if not HAS_NETWORKX:
        raise ImportError("networkx required. Install with: pip install networkx")

    G = nx.Graph()

    for i, concept in enumerate(concepts):
        G.add_node(i,
                    word=concept["word"],
                    category=concept["category"],
                    node_type=concept["type"],
                    subtype=concept["subtype"])

    n = len(concepts)
    for i in range(n):
        for j in range(i + 1, n):
            weight = float(sim_matrix[i, j])
            if weight > threshold:
                G.add_edge(i, j, weight=weight)

    return G


# =====================================================
# NETWORK METRICS
# =====================================================

def compute_network_metrics(G, concepts: list) -> dict:
    """
    Compute network metrics that capture bias structure at a given layer.

    Metrics returned:
        - mean_intra_category_sim: avg edge weight within a bias category (cohesion)
        - mean_inter_category_sim: avg edge weight across categories (cross-bias entanglement)
        - modularity: how well concepts separate into distinct communities (via Louvain)
        - n_communities: number of detected communities
        - cross_bias_connectivity: pairwise inter-category similarity for all category pairs
        - stereotype_bias_strength: mean_group_stereotype_sim − mean_group_neutral_sim
    """
    metrics = {}

    intra_sims = []
    inter_sims = []

    for i, j, data in G.edges(data=True):
        w = data["weight"]
        if concepts[i]["category"] == concepts[j]["category"]:
            intra_sims.append(w)
        else:
            inter_sims.append(w)

    metrics["mean_intra_category_sim"] = float(np.mean(intra_sims)) if intra_sims else 0.0
    metrics["mean_inter_category_sim"] = float(np.mean(inter_sims)) if inter_sims else 0.0

    if HAS_COMMUNITY and len(G.edges) > 0:
        try:
            partition = community_louvain.best_partition(G, weight="weight", random_state=42)
            metrics["modularity"] = community_louvain.modularity(partition, G, weight="weight")
            metrics["n_communities"] = len(set(partition.values()))
        except Exception as e:
            logger.warning(f"Community detection failed: {e}")
            metrics["modularity"] = None
            metrics["n_communities"] = None
    else:
        metrics["modularity"] = None
        metrics["n_communities"] = None

    cross_bias = {}
    for cat1, cat2 in combinations(CATEGORIES, 2):
        pair_sims = []
        for i, j, data in G.edges(data=True):
            ci, cj = concepts[i]["category"], concepts[j]["category"]
            if (ci == cat1 and cj == cat2) or (ci == cat2 and cj == cat1):
                pair_sims.append(data["weight"])
        key = f"{cat1}_x_{cat2}"
        cross_bias[key] = float(np.mean(pair_sims)) if pair_sims else 0.0

    metrics["cross_bias_connectivity"] = cross_bias

    group_stereo_sims = []
    group_neutral_sims = []

    for i, j, data in G.edges(data=True):
        ci, cj = concepts[i], concepts[j]

        if ci["type"] != cj["type"] and ci["category"] == cj["category"]:
            if ci["type"] == "group":
                group_concept, attr_concept = ci, cj
            else:
                group_concept, attr_concept = cj, ci

            if "neutral" in attr_concept["subtype"]:
                group_neutral_sims.append(data["weight"])
            else:
                group_stereo_sims.append(data["weight"])

    metrics["mean_group_stereotype_sim"] = float(np.mean(group_stereo_sims)) if group_stereo_sims else 0.0
    metrics["mean_group_neutral_sim"] = float(np.mean(group_neutral_sims)) if group_neutral_sims else 0.0
    # Bias strength = how much closer groups are to stereotypical attributes
    # than to neutral ones; positive = stereotypical, zero = no preference
    metrics["stereotype_bias_strength"] = metrics["mean_group_stereotype_sim"] - metrics["mean_group_neutral_sim"]

    return metrics


# =====================================================
# FULL ANALYSIS PIPELINE
# =====================================================

def run_network_analysis(
    checkpoints: list,
    output_dir: str,
    analysis_layers: list = None,
    pooling: str = "last_token",
    threshold: float = 0.0,
) -> tuple:
    """Run network analysis at selected layers for all checkpoints."""
    if not HAS_NETWORKX:
        logger.error("networkx required. Install with: pip install networkx")
        return {}, []

    if analysis_layers is None:
        # Sample layers at regular intervals rather than all 32; balances coverage vs compute cost
        analysis_layers = sorted(set([0, 5, 10, 15, 20, 25] + [NUM_LAYERS - 1]))
        analysis_layers = [l for l in analysis_layers if l < NUM_LAYERS]

    concepts = collect_all_concepts()
    all_results = {}

    for checkpoint in checkpoints:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {checkpoint['label']}")
        logger.info(f"{'='*60}")

        model, tokenizer = load_checkpoint(checkpoint)

        checkpoint_results = {}

        for layer in analysis_layers:
            logger.info(f"  Building network at layer {layer}...")
            # Store as list for JSON serialisation; excluded from saved metrics in save_results()
            sim_matrix = build_similarity_matrix(model, tokenizer, concepts, layer, pooling)
            G = build_network(sim_matrix, concepts, threshold)
            metrics = compute_network_metrics(G, concepts)
            metrics["layer"] = layer

            checkpoint_results[layer] = {
                "metrics": metrics,
                "sim_matrix": sim_matrix.tolist(),
            }

            logger.info(
                f"    Intra-cat sim: {metrics['mean_intra_category_sim']:.4f}, "
                f"Inter-cat sim: {metrics['mean_inter_category_sim']:.4f}, "
                f"Stereotype bias: {metrics['stereotype_bias_strength']:.4f}"
            )

        all_results[checkpoint["name"]] = checkpoint_results

        unload_model(model)
        del tokenizer

    return all_results, concepts


# =====================================================
# REPORTING
# =====================================================

def print_network_summary(all_results: dict, checkpoints: list) -> None:
    """Print summary of key network metrics across checkpoints."""
    print("\n" + "=" * 90)
    print("Word Association Network Analysis Summary")
    print("=" * 90)

    header = (
        f"{'Model':<22} {'Layer':>6} {'Intra-Cat':>10} {'Inter-Cat':>10} "
        f"{'Modularity':>11} {'Stereo Bias':>12} {'Communities':>12}"
    )
    print(header)
    print("-" * 90)

    for checkpoint in checkpoints:
        model_name = checkpoint["name"]
        if model_name not in all_results:
            continue

        layers = sorted(all_results[model_name].keys())
        for i, layer in enumerate(layers):
            m = all_results[model_name][layer]["metrics"]
            label = model_name if i == 0 else ""
            mod_str = f"{m['modularity']:.4f}" if m['modularity'] is not None else "N/A"
            comm_str = str(m['n_communities']) if m['n_communities'] is not None else "N/A"

            print(
                f"{label:<22} {layer:>6} "
                f"{m['mean_intra_category_sim']:>10.4f} "
                f"{m['mean_inter_category_sim']:>10.4f} "
                f"{mod_str:>11} "
                f"{m['stereotype_bias_strength']:>12.4f} "
                f"{comm_str:>12}"
            )

        print("-" * 90)


def save_results(all_results: dict, concepts: list, output_dir: str) -> None:
    """Save results to JSON (without large similarity matrices)."""
    os.makedirs(output_dir, exist_ok=True)

    metrics_only = {}
    for model, layers in all_results.items():
        metrics_only[model] = {}
        for layer, data in layers.items():
            metrics_only[model][str(layer)] = data["metrics"]

    path = os.path.join(output_dir, "network_metrics.json")
    with open(path, "w") as f:
        json.dump(metrics_only, f, indent=2)
    logger.info(f"Saved metrics → {path}")

    concepts_path = os.path.join(output_dir, "concepts.json")
    with open(concepts_path, "w") as f:
        json.dump(concepts, f, indent=2)
    logger.info(f"Saved concepts → {concepts_path}")


# =====================================================
# VISUALISATION
# =====================================================

def plot_stereotype_bias_by_layer(all_results: dict, checkpoints: list,
                                  palette: dict, model_labels: dict,
                                  output_dir: str) -> None:
    """Line plot: stereotype bias strength across layers for each checkpoint."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    for checkpoint in checkpoints:
        model_name = checkpoint["name"]
        if model_name not in all_results:
            continue

        layers = sorted(all_results[model_name].keys())
        bias_vals = [
            all_results[model_name][l]["metrics"]["stereotype_bias_strength"]
            for l in layers
        ]

        ax.plot(layers, bias_vals,
                label=model_labels.get(model_name, model_name),
                color=palette.get(model_name, "#999"),
                linewidth=2, marker="o", markersize=4, alpha=0.85)

    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Stereotype Bias Strength\n(stereo_sim − neutral_sim)", fontsize=11)
    ax.set_title("Network-Based Stereotype Bias Across Layers", fontsize=13, pad=10)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    path = os.path.join(output_dir, "network_stereotype_bias.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {path}")


def plot_cross_bias_connectivity(all_results: dict, checkpoints: list,
                                 model_labels: dict, output_dir: str) -> None:
    """Heatmap: cross-bias connectivity for each checkpoint at a selected layer."""
    os.makedirs(output_dir, exist_ok=True)

    # Snapshot the network at the middle layer as a representative cross-checkpoint comparison point
    target_layer = NUM_LAYERS // 2
    for layer in sorted(all_results.get("baseline", {}).keys()):
        if layer >= target_layer:
            target_layer = layer
            break

    models = [c["name"] for c in checkpoints if c["name"] in all_results]
    cat_pairs = list(combinations(CATEGORIES, 2))
    pair_labels = [f"{c1[:3]}-{c2[:3]}" for c1, c2 in cat_pairs]

    matrix = np.full((len(models), len(cat_pairs)), np.nan)

    for i, model_name in enumerate(models):
        if target_layer not in all_results[model_name]:
            continue
        cross_bias = all_results[model_name][target_layer]["metrics"]["cross_bias_connectivity"]
        for j, (c1, c2) in enumerate(cat_pairs):
            key = f"{c1}_x_{c2}"
            matrix[i, j] = cross_bias.get(key, np.nan)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(pair_labels)))
    ax.set_xticklabels(pair_labels, fontsize=9, rotation=45, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([model_labels.get(m, m) for m in models], fontsize=10)

    for i in range(len(models)):
        for j in range(len(pair_labels)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax, label="Mean Cross-Category Similarity")
    ax.set_title(
        f"Cross-Bias Connectivity at Layer {target_layer} (Higher = More Entangled)",
        fontsize=12, pad=10,
    )
    plt.tight_layout()

    path = os.path.join(output_dir, "network_cross_bias_connectivity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {path}")


def plot_modularity_by_layer(all_results: dict, checkpoints: list,
                             palette: dict, model_labels: dict,
                             output_dir: str) -> None:
    """Line plot: network modularity across layers."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    for checkpoint in checkpoints:
        model_name = checkpoint["name"]
        if model_name not in all_results:
            continue

        layers = sorted(all_results[model_name].keys())
        mod_vals = []
        valid_layers = []
        for l in layers:
            m = all_results[model_name][l]["metrics"].get("modularity")
            if m is not None:
                mod_vals.append(m)
                valid_layers.append(l)

        if valid_layers:
            ax.plot(valid_layers, mod_vals,
                    label=model_labels.get(model_name, model_name),
                    color=palette.get(model_name, "#999"),
                    linewidth=2, marker="o", markersize=4, alpha=0.85)

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Network Modularity", fontsize=11)
    ax.set_title("Modularity of Social Concept Networks Across Layers", fontsize=13, pad=10)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    path = os.path.join(output_dir, "network_modularity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {path}")


# =====================================================
# MAIN
# =====================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Word association network analysis for internal bias."
    )
    parser.add_argument("--checkpoints-config", required=True,
                        help="Path to JSON config file describing the checkpoints "
                             "to analyze (e.g., configs/biasdpo_full.json).")
    parser.add_argument("--output-dir", default="./analysis/internal/network")
    parser.add_argument("--pooling", default="last_token",
                        choices=["last_token", "mean", "first_token"])
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Minimum similarity to create network edge.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = load_checkpoints_config(args.checkpoints_config)
    checkpoints = get_checkpoints(cfg)
    palette = get_palette(cfg)
    model_labels = get_model_labels(cfg)
    logger.info(f"Experiment: {cfg['experiment_name']}")
    logger.info(f"Checkpoints: {[c['name'] for c in checkpoints]}")

    all_results, concepts = run_network_analysis(
        checkpoints, args.output_dir,
        pooling=args.pooling, threshold=args.threshold,
    )

    if all_results:
        print_network_summary(all_results, checkpoints)
        save_results(all_results, concepts, args.output_dir)
        plot_stereotype_bias_by_layer(all_results, checkpoints, palette, model_labels,
                                      args.output_dir)
        plot_cross_bias_connectivity(all_results, checkpoints, model_labels,
                                     args.output_dir)
        plot_modularity_by_layer(all_results, checkpoints, palette, model_labels,
                                 args.output_dir)

    logger.info(f"Done. All outputs in: {args.output_dir}")
