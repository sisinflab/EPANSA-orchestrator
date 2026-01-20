"""Inter-Annotator Agreement (IAA) Metrics Calculator.

Computes pairwise agreement metrics between multiple evaluators/models,
including Cohen's Kappa, Gwet's AC1, Spearman correlation, and percentage agreement.
Outputs heatmaps and summary statistics for reproducibility analysis.
"""

import itertools

import krippendorff
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, confusion_matrix

# Configuration
INPUT_FILE = "Model_analysis_normalized.csv"
OUTPUT_TABLE_FILE = "Comparison_Summary_Table.csv"


def calculate_gwet_ac1(y1, y2):
    """Calculate Gwet's AC1 (Unweighted) for two raters.

    Gwet's AC1 is robust to the high-agreement paradox that affects Cohen's Kappa
    when marginal distributions are highly skewed.

    Args:
        y1: Ratings from first evaluator.
        y2: Ratings from second evaluator.

    Returns:
        float: Gwet's AC1 coefficient in range [-1, 1].
    """
    y1 = np.array(y1)
    y2 = np.array(y2)
    classes = np.unique(np.concatenate([y1, y2]))
    cm = confusion_matrix(y1, y2, labels=classes)
    n = np.sum(cm)
    q = len(classes)

    if q < 2:
        return 1.0

    # Observed agreement
    pa = np.trace(cm) / n

    # Expected agreement by chance
    pe = 0
    for i in range(q):
        pi = (np.sum(cm[i, :]) + np.sum(cm[:, i])) / (2 * n)
        pe += pi * (1 - pi)
    pe = pe / (q - 1)

    if pe == 1:
        return 1.0

    ac1 = (pa - pe) / (1 - pe)
    return ac1


def plot_heatmap(data, title, filename, vmin=None, vmax=None):
    """Generate and save a heatmap visualization.

    Args:
        data: DataFrame containing the matrix to visualize.
        title: Plot title.
        filename: Output file path.
        vmin: Minimum value for color scale.
        vmax: Maximum value for color scale.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        data, annot=True, fmt=".2f", cmap="RdYlGn", linewidths=0.5, vmin=vmin, vmax=vmax
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Heatmap saved: {filename}")


def calculate_percentage_agreement(y1, y2):
    """Calculate raw percentage agreement between two raters.

    Args:
        y1: Ratings from first evaluator.
        y2: Ratings from second evaluator.

    Returns:
        float: Percentage of exact matches (0-100).
    """
    return np.mean(np.array(y1) == np.array(y2)) * 100


def plot_stacked_bars(df, model_list, title, save_path):
    """Generate stacked bar chart showing score distributions.

    Args:
        df: DataFrame containing evaluation scores.
        model_list: List of model/evaluator column names.
        title: Plot title.
        save_path: Output file path.
    """
    print(f"Generating distribution plot: {save_path}")
    plot_data = df[model_list].apply(pd.Series.value_counts).fillna(0).T
    plot_data = plot_data.sort_index(axis=1)

    print(f"\nScore distribution ({title}):")
    print(plot_data.astype(int))

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    colors = sns.color_palette("viridis", len(plot_data.columns))

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_data.plot(
        kind="bar",
        stacked=True,
        color=colors,
        ax=ax,
        edgecolor="black",
        linewidth=0.8,
        width=0.75,
        rot=0,
    )

    ax.set_title(title, fontsize=16, weight="bold", pad=20)
    ax.set_xlabel("Models", fontsize=13, weight="bold")
    ax.set_ylabel("Response Frequency", fontsize=13)
    ax.yaxis.grid(True, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.xticks(rotation=45, ha="right")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        reversed(handles),
        reversed(labels),
        title="Score",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    sns.despine(left=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def analyze_top_4():
    """Analyze inter-annotator agreement for the top 4 most consistent evaluators.

    Selects the 4-model combination with highest Krippendorff's alpha,
    then computes pairwise agreement metrics and generates visualizations.
    """
    print(f"Loading data from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: File {INPUT_FILE} not found.")
        return

    cols_to_ignore = ["QUESTION"]
    all_cols = [c for c in df.columns if c not in cols_to_ignore]
    for col in all_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df_clean = df.dropna(subset=all_cols)

    # Step 1: Select top 4 models by Krippendorff's alpha
    if len(all_cols) < 4:
        best_models = all_cols
        best_alpha = -1
    else:
        results = []
        for combo in itertools.combinations(all_cols, 4):
            current_data = df_clean[list(combo)].T.to_numpy()
            try:
                alpha = krippendorff.alpha(
                    reliability_data=current_data, level_of_measurement="ordinal"
                )
            except Exception:
                alpha = -1.0
            results.append({"Models": list(combo), "Alpha": alpha})

        results_df = pd.DataFrame(results).sort_values(by="Alpha", ascending=False)
        best_row = results_df.iloc[0]
        best_models = best_row["Models"]
        best_alpha = best_row["Alpha"]

    print(f"\nBest model combination (Krippendorff's Alpha: {best_alpha:.4f}):")
    for m in best_models:
        print(f"  - {m}")

    # Step 2: Calculate pairwise agreement matrices
    df_top = df_clean[best_models]
    matrix_kappa = pd.DataFrame(index=best_models, columns=best_models, dtype=float)
    matrix_agreement = pd.DataFrame(index=best_models, columns=best_models, dtype=float)
    matrix_gwet = pd.DataFrame(index=best_models, columns=best_models, dtype=float)
    matrix_spearman = df_top.corr(method="spearman")

    print("\nCalculating pairwise metrics...")
    for m1 in best_models:
        for m2 in best_models:
            if m1 == m2:
                matrix_kappa.loc[m1, m2] = 1.0
                matrix_agreement.loc[m1, m2] = 100.0
                matrix_gwet.loc[m1, m2] = 1.0
            else:
                matrix_kappa.loc[m1, m2] = cohen_kappa_score(
                    df_top[m1], df_top[m2], weights="quadratic"
                )
                matrix_agreement.loc[m1, m2] = calculate_percentage_agreement(
                    df_top[m1], df_top[m2]
                )
                matrix_gwet.loc[m1, m2] = calculate_gwet_ac1(df_top[m1], df_top[m2])

    # Step 3: Generate visualizations
    plot_heatmap(
        matrix_kappa, "Cohen's Kappa (Quadratic)", "Figure/Heatmap_Kappa.png", 0, 1
    )
    plot_heatmap(
        matrix_agreement,
        "Percentage Agreement (%)",
        "Figure/Heatmap_Agreement.png",
        0,
        100,
    )
    plot_heatmap(
        matrix_spearman, "Spearman's Correlation", "Figure/Heatmap_Spearman.png", -1, 1
    )
    plot_heatmap(matrix_gwet, "Gwet's AC1 (Robust)", "Figure/Heatmap_Gwet.png", 0, 1)
    plot_stacked_bars(
        df_top, best_models, "Score Distribution", "Figure/Barplot_Distribution.png"
    )

    # Step 4: Generate summary comparison table
    print("\n" + "=" * 80)
    print("Pairwise Comparison Summary")
    print("=" * 80)

    summary_data = []
    for m1, m2 in itertools.combinations(best_models, 2):
        k_score = matrix_kappa.loc[m1, m2]
        gwet_score = matrix_gwet.loc[m1, m2]
        agree_score = matrix_agreement.loc[m1, m2]
        spearman_score = matrix_spearman.loc[m1, m2]
        delta_paradox = gwet_score - k_score

        summary_data.append(
            {
                "Model A": m1,
                "Model B": m2,
                "Agreement (%)": f"{agree_score:.2f}%",
                "Gwet's AC1": f"{gwet_score:.4f}",
                "Cohen's Kappa": f"{k_score:.4f}",
                "Spearman Rho": f"{spearman_score:.4f}",
                "Delta (Gwet-Kappa)": f"{delta_paradox:.4f}",
            }
        )

    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(OUTPUT_TABLE_FILE, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(df_summary.to_string(index=False))
    print("=" * 80)
    print(f"Summary table saved to: {OUTPUT_TABLE_FILE}")


if __name__ == "__main__":
    analyze_top_4()
