"""LLM-as-a-Judge Evaluation using Prometheus Rubric.

Evaluates model responses against ground truth using a local LLM judge
with the Prometheus scoring framework (1-5 Likert scale).
"""

import re
import time

import pandas as pd
from ollama import Client

# Configuration
MODEL_NAME = "OLLAMA MODEL"
INPUT_CSV = "qa_dataset.csv"
OUTPUT_CSV = f"prometheus_eval_results_{MODEL_NAME.split('/')[-1]}.csv"

client = Client()

TASK_DESCRIPTION = """An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing an evaluation criterion is given.
1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Example: "[RESULT] 5"
5. Please do not generate any other opening, closing, and explanations."""

SCORE_RUBRIC = """[Quality of the response in terms of accuracy and helpfulness regarding personal assistant tasks.]
Score 1: The response is completely incorrect, irrelevant, or fails to address the instruction.
Score 2: The response is partially relevant but contains major inaccuracies or misses significant parts of the instruction.
Score 3: The response is relevant and mostly correct but has minor inaccuracies or lacks necessary detail.
Score 4: The response is correct, detailed, and follows the instructions well, with only very minor issues.
Score 5: The response is excellent, accurate, comprehensive, and perfectly addresses all aspects of the instruction."""

PROMPT_TEMPLATE = """###Task:
{task_description}

###Instruction:
{instruction}

###Response:
{response}

###Reference (Score 5):
{reference_answer}

###Rubric:
{score_rubric}

###Feedback:"""


def evaluate_with_llm(instruction, response, reference):
    """Evaluate a single response using the LLM judge.

    Args:
        instruction: The original question/instruction.
        response: The model's generated response.
        reference: The ground truth reference answer.

    Returns:
        tuple: (feedback_text, score) where score is 1-5 or None if parsing failed.
    """
    prompt = PROMPT_TEMPLATE.format(
        task_description=TASK_DESCRIPTION,
        instruction=instruction,
        response=response,
        reference_answer=reference,
        score_rubric=SCORE_RUBRIC,
    )

    try:
        completion = client.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.1,
                "num_predict": 4096,
            },
        )

        full_output = completion["message"]["content"]
        score_match = re.search(r"\[RESULT\]\s*\(?(\d)\)?", full_output)
        score = int(score_match.group(1)) if score_match else None

        return full_output, score
    except Exception as e:
        print(f"API Error: {e}")
        return str(e), None


def print_score_distribution(df):
    """Print summary statistics of evaluation scores."""
    print("\n" + "=" * 50)
    print(f"Evaluation Results Summary ({MODEL_NAME})")
    print("=" * 50)
    valid_scores = df["PROMETHEUS_SCORE"].dropna()
    total_valid = len(valid_scores)
    if total_valid == 0:
        return
    counts = valid_scores.value_counts()
    print(f"Total valid scores: {total_valid}\n")
    print(f"{'Score':<10} | {'Count':<10} | {'Percentage':<10}")
    print("-" * 36)
    for score in [5, 4, 3, 2, 1]:
        count = counts.get(score, 0)
        percentage = (count / total_valid) * 100
        print(f"{score:<10} | {count:<10} | {percentage:.2f}%")
    print("=" * 50 + "\n")


def run_evaluation():
    """Run the full evaluation pipeline."""
    print(f"Loading {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV, sep=";", skiprows=1)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    results = []
    print(f"Starting evaluation of {len(df)} rows using {MODEL_NAME}...")

    for index, row in df.iterrows():
        print(f"Processing row {index + 1}/{len(df)}...")

        feedback, score = evaluate_with_llm(
            instruction=row["QUESTION"],
            response=row["EpisTwin ANSWER"],
            reference=row["TARGET ANSWER"],
        )

        results.append(
            {
                "QUESTION": row["QUESTION"],
                "ACTUAL_ANSWER": row["EpisTwin ANSWER"],
                "PROMETHEUS_FEEDBACK": feedback,
                "PROMETHEUS_SCORE": score,
            }
        )

        time.sleep(0.5)

    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Evaluation complete. Results saved to {OUTPUT_CSV}")

    print_score_distribution(output_df)


if __name__ == "__main__":
    run_evaluation()
