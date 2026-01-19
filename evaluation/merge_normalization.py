"""Score Normalization and Aggregation for Multi-Evaluator Analysis.

Merges evaluation results from multiple LLM judges and normalizes the 1-5
Likert scale to a simplified 0-2 ordinal scale for stricter comparison:
    - Scores 4-5 -> 2 (Correct)
    - Score 3    -> 1 (Partially Correct)
    - Scores 1-2 -> 0 (Incorrect)
"""

import glob
import os

import pandas as pd

# Configuration
FILE_PATTERN = "prometheus_eval_results_*.csv"
OUTPUT_MASTER_CSV = "Analisi_Comparativa_Modelli_normalized.csv"


def normalize_score(score):
    """Convert 1-5 Likert scale to 0-2 ordinal scale.

    Args:
        score: Original score (1-5) or NaN.

    Returns:
        int: Normalized score (0, 1, or 2) or original if NaN.
    """
    if pd.isna(score):
        return score

    if score >= 4:
        return 2
    elif score == 3:
        return 1
    elif score <= 2:
        return 0
    else:
        return score


def create_master_dataset():
    """Merge and normalize evaluation results from multiple LLM judges.

    Reads all prometheus_eval_results_*.csv files, extracts scores,
    normalizes them to the 0-2 scale, and creates a unified comparison dataset.
    """
    files = glob.glob(FILE_PATTERN)

    if not files:
        print("No result files found matching pattern: prometheus_eval_results_*.csv")
        return

    print(f"Found {len(files)} result files.")

    # Initialize master DataFrame from first file
    first_file = files[0]
    print(f"Using '{first_file}' as base for questions...")

    try:
        base_df = pd.read_csv(first_file)

        if "QUESTION" not in base_df.columns:
            print("Error: 'QUESTION' column not found in base file.")
            return

        master_df = base_df[["QUESTION"]].copy()
        print("Base DataFrame created.")

    except Exception as e:
        print(f"Error reading base file: {e}")
        return

    # Merge scores from all result files
    for file in files:
        try:
            model_name = (
                os.path.basename(file)
                .replace("prometheus_eval_results_", "")
                .replace(".csv", "")
            )

            print(f"  Processing: {model_name}...")

            current_df = pd.read_csv(file)

            if "PROMETHEUS_SCORE" in current_df.columns:
                subset = current_df[["QUESTION", "PROMETHEUS_SCORE"]].copy()
                subset["PROMETHEUS_SCORE"] = pd.to_numeric(
                    subset["PROMETHEUS_SCORE"], errors="coerce"
                )
                subset["PROMETHEUS_SCORE"] = subset["PROMETHEUS_SCORE"].apply(
                    normalize_score
                )
                subset = subset.rename(columns={"PROMETHEUS_SCORE": model_name})
                master_df = pd.merge(master_df, subset, on="QUESTION", how="left")
            else:
                print(f"  Warning: {file} missing 'PROMETHEUS_SCORE' column. Skipped.")

        except Exception as e:
            print(f"Error processing {file}: {e}")

    # Save master dataset
    print("\nSaving merged dataset...")
    master_df.to_csv(OUTPUT_MASTER_CSV, index=False)
    print(f"Saved: {OUTPUT_MASTER_CSV}")

    # Preview
    print("\nData preview (first 5 rows):")
    print(master_df.head())

    print("\nColumn types:")
    print(master_df.dtypes)


if __name__ == "__main__":
    create_master_dataset()
