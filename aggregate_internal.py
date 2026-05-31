"""
aggregate_internal.py
=====================
Reads all internal analysis JSON files and outputs clean aggregated tables
ready for AI interpretation or pasting into a thesis chat.

Run from your thesis_project root:
    python3 aggregate_internal.py --base-dir analysis/internal --output-dir aggregated_results

Outputs (in --output-dir):
    01_probing_peak_accuracy.md         — peak probe accuracy per model per category
    02_probing_lobo_peak_accuracy.md    — same for LOBO probe
    03_cosine_bias_direction.md         — directional bias score at layers 0,10,20,31
    04_network_metrics.md               — stereotype_bias_strength, modularity, cross_bias at layers 0,15,31
    05_sequential_stage1_vs_stage2.md   — stage1 vs final for sequential models (cross-bias transfer)
    summary.md                          — all tables combined in one file
"""

import os
import json
import argparse
import math
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS — must match internal_utils.py exactly
# ──────────────────────────────────────────────────────────────────────────────

LAYERS = list(range(32))
CATEGORIES = ["gender", "race", "religion", "socioeconomic"]
NETWORK_LAYERS = [0, 5, 10, 15, 20, 25, 31]
REPORT_LAYERS_COSINE = [0, 10, 20, 31]
REPORT_LAYERS_NETWORK = [0, 15, 31]

# All 16 experiment folders and their display names
EXPERIMENTS = {
    # General alignment
    "biasdpo_full": {
        "display": "General",
        "trained_keys": {"sft": "biasdpo_sft", "dpo": "biasdpo_dpo"},
        "type": "general",
    },
    # Parallel alignment
    "biasdpo_gender_extended": {
        "display": "Parallel Gender",
        "trained_keys": {"sft": "biasdpo_gender_extended_sft", "dpo": "biasdpo_gender_extended_dpo"},
        "type": "parallel",
        "axis": "gender",
    },
    "biasdpo_race_extended": {
        "display": "Parallel Race",
        "trained_keys": {"sft": "biasdpo_race_extended_sft", "dpo": "biasdpo_race_extended_dpo"},
        "type": "parallel",
        "axis": "race",
    },
    "biasdpo_religion_extended": {
        "display": "Parallel Religion",
        "trained_keys": {"sft": "biasdpo_religion_extended_sft", "dpo": "biasdpo_religion_extended_dpo"},
        "type": "parallel",
        "axis": "religion",
    },
    # Sequential alignment — SFT
    "seq_gender_sft_to_race_sft": {
        "display": "Seq Gender→Race (SFT)",
        "trained_keys": {"final": "seq_gender_sft_to_race_sft"},
        "stage1_key": "stage1_gender_sft",
        "type": "sequential", "method": "sft",
        "axis1": "gender", "axis2": "race",
    },
    "seq_gender_sft_to_religion_sft": {
        "display": "Seq Gender→Religion (SFT)",
        "trained_keys": {"final": "seq_gender_sft_to_religion_sft"},
        "stage1_key": "stage1_gender_sft",
        "type": "sequential", "method": "sft",
        "axis1": "gender", "axis2": "religion",
    },
    "seq_race_sft_to_gender_sft": {
        "display": "Seq Race→Gender (SFT)",
        "trained_keys": {"final": "seq_race_sft_to_gender_sft"},
        "stage1_key": "stage1_race_sft",
        "type": "sequential", "method": "sft",
        "axis1": "race", "axis2": "gender",
    },
    "seq_race_sft_to_religion_sft": {
        "display": "Seq Race→Religion (SFT)",
        "trained_keys": {"final": "seq_race_sft_to_religion_sft"},
        "stage1_key": "stage1_race_sft",
        "type": "sequential", "method": "sft",
        "axis1": "race", "axis2": "religion",
    },
    "seq_religion_sft_to_gender_sft": {
        "display": "Seq Religion→Gender (SFT)",
        "trained_keys": {"final": "seq_religion_sft_to_gender_sft"},
        "stage1_key": "stage1_religion_sft",
        "type": "sequential", "method": "sft",
        "axis1": "religion", "axis2": "gender",
    },
    "seq_religion_sft_to_race_sft": {
        "display": "Seq Religion→Race (SFT)",
        "trained_keys": {"final": "seq_religion_sft_to_race_sft"},
        "stage1_key": "stage1_religion_sft",
        "type": "sequential", "method": "sft",
        "axis1": "religion", "axis2": "race",
    },
    # Sequential alignment — DPO
    "seq_gender_dpo_to_race_dpo": {
        "display": "Seq Gender→Race (DPO)",
        "trained_keys": {"final": "seq_gender_dpo_to_race_dpo"},
        "stage1_key": "stage1_gender_dpo",
        "type": "sequential", "method": "dpo",
        "axis1": "gender", "axis2": "race",
    },
    "seq_gender_dpo_to_religion_dpo": {
        "display": "Seq Gender→Religion (DPO)",
        "trained_keys": {"final": "seq_gender_dpo_to_religion_dpo"},
        "stage1_key": "stage1_gender_dpo",
        "type": "sequential", "method": "dpo",
        "axis1": "gender", "axis2": "religion",
    },
    "seq_race_dpo_to_gender_dpo": {
        "display": "Seq Race→Gender (DPO)",
        "trained_keys": {"final": "seq_race_dpo_to_gender_dpo"},
        "stage1_key": "stage1_race_dpo",
        "type": "sequential", "method": "dpo",
        "axis1": "race", "axis2": "gender",
    },
    "seq_race_dpo_to_religion_dpo": {
        "display": "Seq Race→Religion (DPO)",
        "trained_keys": {"final": "seq_race_dpo_to_religion_dpo"},
        "stage1_key": "stage1_race_dpo",
        "type": "sequential", "method": "dpo",
        "axis1": "race", "axis2": "religion",
    },
    "seq_religion_dpo_to_gender_dpo": {
        "display": "Seq Religion→Gender (DPO)",
        "trained_keys": {"final": "seq_religion_dpo_to_gender_dpo"},
        "stage1_key": "stage1_religion_dpo",
        "type": "sequential", "method": "dpo",
        "axis1": "religion", "axis2": "gender",
    },
    "seq_religion_dpo_to_race_dpo": {
        "display": "Seq Religion→Race (DPO)",
        "trained_keys": {"final": "seq_religion_dpo_to_race_dpo"},
        "stage1_key": "stage1_religion_dpo",
        "type": "sequential", "method": "dpo",
        "axis1": "religion", "axis2": "race",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def fmt(v, decimals=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:.{decimals}f}"

def peak_accuracy(data, model_key, category):
    """
    Returns the peak probe accuracy and the layer at which it occurs
    for a given model and bias category.
    Iterates over all 32 layers and returns the highest accuracy_mean found.
    Returns (None, None) if no data is available for the given model/category.
    """
    cat_data = data.get(model_key, {}).get(category, {})
    if not cat_data:
        return None, None
    best_acc, best_layer = -1, -1
    for layer_str, metrics in cat_data.items():
        acc = metrics.get("accuracy_mean", -1)
        if acc > best_acc:
            best_acc = acc
            best_layer = int(layer_str)
    return best_acc if best_acc >= 0 else None, best_layer if best_layer >= 0 else None

def compute_bias_direction(cosine_data, model_key, category, layer):
    """
    Computes the directional bias score at a specific layer for a given model and category.
    For gender: mean of (female→female_stereo minus female→male_stereo) and
    (male→male_stereo minus male→female_stereo). Positive = stereotypical direction.
    For race/religion/socioeconomic: mean of (group→negative minus group→neutral)
    across all groups. Positive = group representations closer to negative stereotypes.
    Mirrors the scoring logic in cosine_analysis.py to ensure consistency.
    """
    cat = cosine_data.get(model_key, {}).get(category, {})
    if not cat:
        return None
    l = str(layer)

    try:
        if category == "gender":
            f_stereo  = cat["female"]["female_stereotypes"].get(l, 0)
            f_counter = cat["female"]["male_stereotypes"].get(l, 0)
            m_stereo  = cat["male"]["male_stereotypes"].get(l, 0)
            m_counter = cat["male"]["female_stereotypes"].get(l, 0)
            return ((f_stereo - f_counter) + (m_stereo - m_counter)) / 2

        elif category in ("race", "religion"):
            scores = []
            for group in cat:
                neg = cat[group].get("negative", {}).get(l, 0)
                neu = cat[group].get("neutral", {}).get(l, 0)
                scores.append(neg - neu)
            return sum(scores) / len(scores) if scores else None

        elif category == "socioeconomic":
            p_stereo  = cat["poor"]["poor_stereotypes"].get(l, 0)
            p_neutral = cat["poor"]["neutral"].get(l, 0)
            w_stereo  = cat["wealthy"]["wealthy_stereotypes"].get(l, 0)
            w_neutral = cat["wealthy"]["neutral"].get(l, 0)
            return ((p_stereo - p_neutral) + (w_stereo - w_neutral)) / 2

    except (KeyError, TypeError):
        return None

    return None

def delta(trained_val, baseline_val):
    """Return trained - baseline, or None if either is missing."""
    if trained_val is None or baseline_val is None:
        return None
    return trained_val - baseline_val


# ──────────────────────────────────────────────────────────────────────────────
# TABLE 1 & 2 — Probing peak accuracy
# ──────────────────────────────────────────────────────────────────────────────

def make_probing_table(base_dir, experiments, probe_type="stereotype"):
    """
    probe_type: "stereotype" or "stereotype_lobo"
    Returns markdown string.
    """
    subdir = probe_type
    filename = f"probing_results_{probe_type}.json"

    rows = []
    baseline_peaks = {}  # category -> (acc, layer) from first file that has baseline

    # Collect baseline from biasdpo_full (same baseline in every file)
    ref_path = os.path.join(base_dir, "biasdpo_full", "probing", subdir, filename)
    ref_data = load_json(ref_path)
    if ref_data and "baseline" in ref_data:
        for cat in CATEGORIES:
            acc, layer = peak_accuracy(ref_data, "baseline", cat)
            baseline_peaks[cat] = (acc, layer)

    # Header
    cat_headers = " | ".join([f"{c.capitalize()} acc (Δ) @ layer" for c in CATEGORIES])
    header = f"| Model | Method | {cat_headers} |"
    sep = "|---|---|" + "|".join(["---" for _ in CATEGORIES]) + "|"

    lines = [
        f"## Probing — {'LOBO' if probe_type == 'stereotype_lobo' else 'Standard'} Peak Accuracy",
        "",
        "Δ = trained − baseline. Positive = probe accuracy increased (bias more linearly decodable). "
        "Negative = decreased (alignment reduced decodability).",
        "",
        header, sep,
    ]

    # Baseline row
    base_cells = []
    for cat in CATEGORIES:
        acc, layer = baseline_peaks.get(cat, (None, None))
        base_cells.append(f"{fmt(acc, 3)} @ L{layer}" if acc is not None else "N/A")
    lines.append(f"| **Baseline** | — | " + " | ".join(base_cells) + " |")

    # Trained model rows
    for folder, meta in experiments.items():
        path = os.path.join(base_dir, folder, "probing", subdir, filename)
        data = load_json(path)
        if data is None:
            continue

        trained_keys = meta["trained_keys"]
        exp_type = meta["type"]

        for method_label, tk in trained_keys.items():
            if tk not in data:
                continue

            cells = []
            for cat in CATEGORIES:
                acc, layer = peak_accuracy(data, tk, cat)
                base_acc, _ = baseline_peaks.get(cat, (None, None))
                d = delta(acc, base_acc)
                d_str = f" ({'+' if d and d>0 else ''}{fmt(d,3)})" if d is not None else ""
                cells.append(f"{fmt(acc,3)}{d_str} @ L{layer}" if acc is not None else "N/A")

            display = meta["display"]
            lines.append(f"| {display} | {method_label.upper()} | " + " | ".join(cells) + " |")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# TABLE 3 — Cosine bias direction scores
# ──────────────────────────────────────────────────────────────────────────────

def make_cosine_table(base_dir, experiments):
    sections = []

    # Get baseline values from biasdpo_full
    ref_path = os.path.join(base_dir, "biasdpo_full", "cosine", "cosine_similarity_results.json")
    ref_data = load_json(ref_path)
    baseline_scores = {}  # (category, layer) -> score
    if ref_data and "baseline" in ref_data:
        for cat in CATEGORIES:
            for layer in REPORT_LAYERS_COSINE:
                v = compute_bias_direction(ref_data, "baseline", cat, layer)
                baseline_scores[(cat, layer)] = v

    for cat in CATEGORIES:
        layer_headers = " | ".join([f"L{l} (Δ)" for l in REPORT_LAYERS_COSINE])
        header = f"| Model | Method | {layer_headers} |"
        sep = "|---|---|" + "|".join(["---" for _ in REPORT_LAYERS_COSINE]) + "|"

        lines = [
            f"### Cosine Bias Direction — {cat.capitalize()}",
            "",
            "Positive score = internal geometry aligns stereotypically. "
            "Δ = trained − baseline.",
            "",
            header, sep,
        ]

        # Baseline row
        base_cells = [fmt(baseline_scores.get((cat, l)), 5) for l in REPORT_LAYERS_COSINE]
        lines.append(f"| **Baseline** | — | " + " | ".join(base_cells) + " |")

        for folder, meta in experiments.items():
            path = os.path.join(base_dir, folder, "cosine", "cosine_similarity_results.json")
            data = load_json(path)
            if data is None:
                continue

            for method_label, tk in meta["trained_keys"].items():
                if tk not in data:
                    continue

                cells = []
                for layer in REPORT_LAYERS_COSINE:
                    v = compute_bias_direction(data, tk, cat, layer)
                    bv = baseline_scores.get((cat, layer))
                    d = delta(v, bv)
                    d_str = f" ({'+' if d and d>0 else ''}{fmt(d,5)})" if d is not None else ""
                    cells.append(f"{fmt(v,5)}{d_str}")

                lines.append(f"| {meta['display']} | {method_label.upper()} | " + " | ".join(cells) + " |")

        sections.append("\n".join(lines))

    header_block = "## Cosine Similarity — Directional Bias Scores\n\nOne sub-table per bias category.\n"
    return header_block + "\n\n".join(sections)


# ──────────────────────────────────────────────────────────────────────────────
# TABLE 4 — Network metrics
# ──────────────────────────────────────────────────────────────────────────────

def make_network_table(base_dir, experiments):
    # Get baseline from biasdpo_full
    ref_path = os.path.join(base_dir, "biasdpo_full", "network", "network_metrics.json")
    ref_data = load_json(ref_path)

    def get_net(data, key, layer, metric):
        """Extract a single network metric for a given model key and layer.
        Handles both flat and nested JSON structures and computes
        cross_bias_mean as the average across all category pairs."""
        
        layer_str = str(layer)
        val = data.get(key, {}).get(layer_str, {})
        if isinstance(val, dict) and "metrics" in val:
            val = val["metrics"]
        if metric == "cross_bias_mean":
            cb = val.get("cross_bias_connectivity", {})
            if cb:
                return sum(cb.values()) / len(cb)
            return None
        return val.get(metric)

    baseline_vals = {}
    if ref_data and "baseline" in ref_data:
        for layer in REPORT_LAYERS_NETWORK:
            for metric in ["stereotype_bias_strength", "modularity", "cross_bias_mean"]:
                v = get_net(ref_data, "baseline", layer, metric)
                baseline_vals[(layer, metric)] = v

    metrics_display = {
        "stereotype_bias_strength": "Stereo Bias Strength",
        "modularity": "Modularity",
        "cross_bias_mean": "Mean Cross-Bias Conn.",
    }

    col_headers = []
    for layer in REPORT_LAYERS_NETWORK:
        for metric, mlabel in metrics_display.items():
            col_headers.append(f"L{layer} {mlabel}")

    header = "| Model | Method | " + " | ".join(col_headers) + " |"
    sep = "|---|---|" + "|".join(["---"] * len(col_headers)) + "|"

    lines = [
        "## Network Analysis Metrics",
        "",
        "Stereo Bias Strength = mean(group→stereo edges) − mean(group→neutral edges). "
        "Higher modularity = bias axes more separated in representational space. "
        "Cross-Bias Connectivity = mean similarity between concepts from different bias categories.",
        "",
        header, sep,
    ]

    # Baseline row
    base_cells = []
    for layer in REPORT_LAYERS_NETWORK:
        for metric in metrics_display:
            v = baseline_vals.get((layer, metric))
            base_cells.append(fmt(v, 4))
    lines.append("| **Baseline** | — | " + " | ".join(base_cells) + " |")

    for folder, meta in experiments.items():
        path = os.path.join(base_dir, folder, "network", "network_metrics.json")
        data = load_json(path)
        if data is None:
            continue

        for method_label, tk in meta["trained_keys"].items():
            if tk not in data:
                continue

            cells = []
            for layer in REPORT_LAYERS_NETWORK:
                for metric in metrics_display:
                    v = get_net(data, tk, layer, metric)
                    bv = baseline_vals.get((layer, metric))
                    d = delta(v, bv)
                    d_str = f" ({'+' if d and d>0 else ''}{fmt(d,4)})" if d is not None else ""
                    cells.append(f"{fmt(v,4)}{d_str}")

            lines.append(f"| {meta['display']} | {method_label.upper()} | " + " | ".join(cells) + " |")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# TABLE 5 — Sequential: stage1 vs final (cross-bias transfer)
# ──────────────────────────────────────────────────────────────────────────────

def make_sequential_transfer_table(base_dir, experiments):
    """
    For sequential models only: compare stage1 vs final across all three
    analysis types. This directly addresses RQ2 (cross-bias transfer).
    """
    lines = [
        "## Sequential Models — Stage 1 vs Final (Cross-Bias Transfer)",
        "",
        "For each sequential model, shows the effect of stage 2 training on the",
        "axis that was targeted in stage 1 (i.e. did the second bandage undo the first?).",
        "Uses peak probe accuracy and cosine bias direction at layer 20 as summary metrics.",
        "Δ_stage2 = final − stage1.",
        "",
    ]

    seq_experiments = {k: v for k, v in experiments.items() if v["type"] == "sequential"}

    # Probing transfer table
    lines += [
        "### Probe Accuracy: Effect of Stage 2 on Stage 1 Axis",
        "",
        "| Model | Axis1 (stage1 target) | Baseline probe | Stage1 probe | Final probe | Δ_stage2 |",
        "|---|---|---|---|---|---|",
    ]

    ref_path = os.path.join(base_dir, "biasdpo_full", "probing", "stereotype",
                            "probing_results_stereotype.json")
    ref_data = load_json(ref_path)
    baseline_peaks = {}
    if ref_data and "baseline" in ref_data:
        for cat in CATEGORIES:
            acc, layer = peak_accuracy(ref_data, "baseline", cat)
            baseline_peaks[cat] = acc

    for folder, meta in seq_experiments.items():
        path = os.path.join(base_dir, folder, "probing", "stereotype",
                            "probing_results_stereotype.json")
        data = load_json(path)
        if data is None:
            continue

        axis1 = meta.get("axis1", "?")
        stage1_key = meta.get("stage1_key")
        final_key = list(meta["trained_keys"].values())[0]

        base_acc = baseline_peaks.get(axis1)
        s1_acc, _ = peak_accuracy(data, stage1_key, axis1) if stage1_key in (data or {}) else (None, None)
        fin_acc, _ = peak_accuracy(data, final_key, axis1) if final_key in (data or {}) else (None, None)
        d_s2 = delta(fin_acc, s1_acc)

        d_str = f"{'+' if d_s2 and d_s2>0 else ''}{fmt(d_s2,3)}" if d_s2 is not None else "N/A"
        lines.append(
            f"| {meta['display']} | {axis1} | {fmt(base_acc,3)} | {fmt(s1_acc,3)} | {fmt(fin_acc,3)} | {d_str} |"
        )

    lines.append("")

    # Cosine transfer table (layer 20)
    lines += [
        "### Cosine Bias Direction at Layer 20: Effect of Stage 2 on Stage 1 Axis",
        "",
        "| Model | Axis1 | Baseline | Stage1 | Final | Δ_stage2 |",
        "|---|---|---|---|---|---|",
    ]

    ref_cosine_path = os.path.join(base_dir, "biasdpo_full", "cosine",
                                   "cosine_similarity_results.json")
    ref_cosine = load_json(ref_cosine_path)
    baseline_cosine = {}
    if ref_cosine and "baseline" in ref_cosine:
        for cat in CATEGORIES:
            v = compute_bias_direction(ref_cosine, "baseline", cat, 20)
            baseline_cosine[cat] = v

    for folder, meta in seq_experiments.items():
        path = os.path.join(base_dir, folder, "cosine", "cosine_similarity_results.json")
        data = load_json(path)
        if data is None:
            continue

        axis1 = meta.get("axis1", "?")
        stage1_key = meta.get("stage1_key")
        final_key = list(meta["trained_keys"].values())[0]

        base_v = baseline_cosine.get(axis1)
        s1_v = compute_bias_direction(data, stage1_key, axis1, 20) if stage1_key else None
        fin_v = compute_bias_direction(data, final_key, axis1, 20)
        d_s2 = delta(fin_v, s1_v)

        d_str = f"{'+' if d_s2 and d_s2>0 else ''}{fmt(d_s2,5)}" if d_s2 is not None else "N/A"
        lines.append(
            f"| {meta['display']} | {axis1} | {fmt(base_v,5)} | {fmt(s1_v,5)} | {fmt(fin_v,5)} | {d_str} |"
        )

    lines.append("")

    # Also show effect on axis2 (the second target — was it successfully aligned?)
    lines += [
        "### Cosine Bias Direction at Layer 20: Effect on Stage 2 Axis",
        "",
        "| Model | Axis2 | Baseline | Stage1 | Final | Δ_stage2 |",
        "|---|---|---|---|---|---|",
    ]

    for folder, meta in seq_experiments.items():
        path = os.path.join(base_dir, folder, "cosine", "cosine_similarity_results.json")
        data = load_json(path)
        if data is None:
            continue

        axis2 = meta.get("axis2", "?")
        stage1_key = meta.get("stage1_key")
        final_key = list(meta["trained_keys"].values())[0]

        base_v = baseline_cosine.get(axis2)
        s1_v = compute_bias_direction(data, stage1_key, axis2, 20) if stage1_key else None
        fin_v = compute_bias_direction(data, final_key, axis2, 20)
        d_s2 = delta(fin_v, s1_v)

        d_str = f"{'+' if d_s2 and d_s2>0 else ''}{fmt(d_s2,5)}" if d_s2 is not None else "N/A"
        lines.append(
            f"| {meta['display']} | {axis2} | {fmt(base_v,5)} | {fmt(s1_v,5)} | {fmt(fin_v,5)} | {d_str} |"
        )

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Aggregate internal analysis results.")
    parser.add_argument("--base-dir", default="analysis/internal",
                        help="Path to analysis/internal directory")
    parser.add_argument("--output-dir", default="aggregated_results",
                        help="Directory to write markdown tables")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base_dir = args.base_dir

    print("Generating probing (standard) table...")
    t1 = make_probing_table(base_dir, EXPERIMENTS, "stereotype")

    print("Generating probing (LOBO) table...")
    t2 = make_probing_table(base_dir, EXPERIMENTS, "stereotype_lobo")

    print("Generating cosine bias direction table...")
    t3 = make_cosine_table(base_dir, EXPERIMENTS)

    print("Generating network metrics table...")
    t4 = make_network_table(base_dir, EXPERIMENTS)

    print("Generating sequential transfer table...")
    t5 = make_sequential_transfer_table(base_dir, EXPERIMENTS)

    tables = {
        "01_probing_peak_accuracy.md": t1,
        "02_probing_lobo_peak_accuracy.md": t2,
        "03_cosine_bias_direction.md": t3,
        "04_network_metrics.md": t4,
        "05_sequential_stage1_vs_stage2.md": t5,
    }

    all_parts = []
    for filename, content in tables.items():
        out_path = os.path.join(args.output_dir, filename)
        with open(out_path, "w") as f:
            f.write(content + "\n")
        print(f"  Written: {out_path}")
        all_parts.append(content)

    summary_path = os.path.join(args.output_dir, "summary.md")
    with open(summary_path, "w") as f:
        f.write("# Internal Analysis — Aggregated Results\n\n")
        f.write("\n\n---\n\n".join(all_parts))
    print(f"  Written: {summary_path}")

    print("\nDone. Paste the contents of aggregated_results/summary.md into the thesis chat.")


if __name__ == "__main__":
    main()
