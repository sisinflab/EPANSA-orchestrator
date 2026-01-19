# EpisTwin Evaluation Benchmark

## Supplementary Material for IJCAI 2026 Submission

---

## 1. Overview

This directory contains the evaluation dataset, knowledge base, and evaluation scripts for the paper **"EpisTwin: Neuro-Symbolic Personal Knowledge Graphs for Trustworthy Personal AI"**.

The benchmark focuses on **Personal Information Management (PIM)**, challenging models to reason over heterogeneous, multi-modal, and temporally distributed data to answer natural language questions about a user's digital life. The dataset simulates a comprehensive digital persona (set in 2025) with data spanning calendars, logs, photos, calls, and documents.

---

## 2. Repository Structure

The project is organized into the Knowledge Base (the context data), the QA Dataset (the benchmark), and the Evaluation Scripts.

```text
├── knowledge_base/                # The Personal Knowledge Graph source data
│   ├── epistwin_docs/             # PDF documents (e.g., tickets, receipts, manuals)
│   │   ├── boats.pdf
│   │   ├── cars.pdf
│   │   ├── Flight_Milan.pdf
│   │   └── ...
│   ├── epistwin_images/           # Visual data (Photos from trips, documents, etc.)
│   │   ├── photo_20250601.JPG
│   │   ├── photo_20250602.JPG
│   │   └── ...
│   └── epistwin_jsontxt/          # Metadata & Events in JSON-formatted .txt files
│       ├── alarm_2.txt
│       ├── contact_LucasSmith_888.txt
│       ├── phoneCall_3.txt
│       ├── recurrentEvent_3.txt
│       └── ...
│
├── evalu.py                       # Main script for LLM-based evaluation (Prometheus)
├── merge_normalization.py         # Script to normalize and merge evaluation scores
├── all_metrics_normalized.py      # Script for calculating IAA (Inter-Annotator Agreement) metrics
├── IJCAI_Test_Dataset.csv         # The Question-Answer Benchmark
├── Analisi_Comparativa_Modelli.csv # (Generated) aggregated results
└── README.md
```

---

## 3. Knowledge Base Schema

The core context is stored in `knowledge_base/epistwin_jsontxt/`. Unlike standard text logs, these files contain structured **JSON objects** representing individual nodes or events in the personal graph.

### Data Types Supported:

| Source App | Filename Ex. | Content Description |
|:---|:---|:---|
| **Calendar** | `event_8.txt` | Single occurrences (meetings, lunches) or recurring events (gym, football). |
| **Communication** | `phoneCall_3.txt` | Call logs including duration, direction, and specific timestamps. |
| **Contacts** | `contact_LucasSmith.txt` | Address book entries linking names to phone numbers. |
| **Media (Meta)** | `photo_20250615.txt` | Metadata for images including EXIF data (location, creation time) and paths. |
| **System** | `alarm_2.txt` | Device states, alarms, and specific app usage logs. |
| **Documents** | `doc_1.txt` | Metadata linking to files in `epistwin_docs` (creation/modification dates). |
| **Notes** | `note_1_content.txt` | Unstructured text notes (diaries, to-do lists, thoughts). |

**Example Data Format (`phoneCall_3.txt`):**
```json
{
    "source_app": "phone",
    "call": "phoneCall_3",
    "metadata": {
        "date": "01-Sep-2025",
        "start_time": "10:30:00",
        "end_time": "10:45:00",
        "duration": "0h, 15min, 23sec",
        "call_direction": "incoming",
        "with_contact": "SarahGreen_521"
    }
}
```

---

## 4. Benchmark Dataset (`IJCAI_Test_Dataset.csv`)

The CSV file contains the ground truth for evaluation. It includes questions requiring reasoning across multiple data sources (e.g., correlating a flight ticket in `docs` with a calendar event and a phone call).

**Columns:**
*   `QUESTION`: The natural language query.
*   `DESIRED ANSWER`: The ground truth answer based on the knowledge base.
*   `ACTUAL ANSWER`: (Used during evaluation) The model's generated response.
*   `RATING`: Human or Synthetic score.

**Sample Scenarios:**
*   *Temporal:* "Did I wake up before the alarm last Sunday?"
*   *Cross-Modal:* "Where was I on June 1st at 20:00?" (Requires checking Photo Metadata).
*   *Reasoning:* "Why was I feeling stressed last Friday?" (Requires correlating Calendar meetings with Note entries).

---

## 5. Evaluation Pipeline

The repository includes a complete pipeline for **LLM-as-a-Judge** evaluation using the Prometheus scoring rubric.

### 1. Run Evaluation (`evalu.py`)
Uses an Ollama client (e.g., `Ministral-3-Reasoning`) to grade model responses against the ground truth.
*   **Input:** `IJCAI_Test_Dataset.csv`
*   **Output:** `prometheus_eval_results_{MODEL_NAME}.csv`
*   **Method:** Compares `ACTUAL ANSWER` vs `DESIRED ANSWER` using a 1-5 Rubric.

### 2. Normalization (`merge_normalization.py`)
Consolidates results from multiple models and normalizes the 1-5 Likert scale into a simplified 0-2 scale for stricter comparison:
*   Scores 4-5 $\to$ **2** (Correct)
*   Score 3 $\to$ **1** (Partially Correct)
*   Scores 1-2 $\to$ **0** (Incorrect)

### 3. Metrics Calculation (`all_metrics_normalized.py`)
Calculates Inter-Annotator Agreement (IAA) and performance metrics between different models or evaluators.
*   **Metrics:** Cohen's Kappa, Gwet's AC1, Percentage Agreement, Spearman's Correlation.
*   **Outputs:** Heatmaps (`Figure/`) and a summary CSV table.

---

## 6. Usage

### Prerequisites
```bash
pip install pandas numpy seaborn matplotlib scikit-learn krippendorff ollama
```

### Running the Pipeline
1.  **Configure:** Update `MODEL_NAME` in `evalu.py` to point to your local Ollama model.
2.  **Evaluate:**
    ```bash
    python evalu.py
    ```
3.  **Merge & Normalize:**
    ```bash
    python merge_normalization.py
    ```
4.  **Analyze Metrics:**
    ```bash
    python all_metrics_normalized.py
    ```

---

## 7. Citation

If you use this dataset or evaluation code in your research, please cite our paper:

```bibtex
@inproceedings{epistwin2026,
  title={EpisTwin: Neuro-Symbolic Personal Knowledge Graphs for Trustworthy Personal AI},
  author={},
  booktitle={},
  year={}
}
```