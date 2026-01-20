# PersonalQA-71-100: A Benchmark for Personal Knowledge Graph Reasoning

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXXX-blue)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![FAIR](https://img.shields.io/badge/FAIR-Compliant-green)](https://www.go-fair.org/fair-principles/)

---

## Dataset Summary

**PersonalQA-71-100** is a synthetic benchmark designed to evaluate Personal AI systems on their ability to retrieve, connect, and reason over multimodal personal data. The benchmark accompanies the paper *"EpisTwin: Neuro-Symbolic Personal Knowledge Graphs for Trustworthy Personal AI"* (IJCAI 2026).

| Property | Value |
|----------|-------|
| **Dataset Name** | PersonalQA-71-100 |
| **Version** | 1.0.0 |
| **Release Date** | 2026-01-19 |
| **Language** | English |
| **Task** | Question Answering, Multi-hop Reasoning |
| **Domain** | Personal Information Management (PIM) |
| **License** | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

---

## Table of Contents

1. [Dataset Description](#1-dataset-description)
2. [Data Composition](#2-data-composition)
3. [Collection Process](#3-collection-process)
4. [Data Format](#4-data-format)
5. [Intended Uses](#5-intended-uses)
6. [Evaluation Protocol](#6-evaluation-protocol)
7. [Limitations and Biases](#7-limitations-and-biases)
8. [Citation](#9-citation)

---

## 1. Dataset Description

### 1.1 Motivation

Personal AI systems must reason over heterogeneous, multi-modal, and temporally distributed data to answer natural language questions about a user's digital life. To the best of our knowledge, no prior benchmark exists that systematically evaluates these capabilities. **PersonalQA-71-100** fills this gap by providing:

- A synthetic but realistic digital persona simulating a user's data ecosystem
- Question-answer pairs requiring cross-source and temporal reasoning
- A controlled evaluation setting with fixed temporal context

### 1.2 Composition Overview

The benchmark comprises two synchronized collections:

| Collection | Description | Count |
|------------|-------------|-------|
| **Information Objects** | Synthetic personal data from 7 source applications | 71 objects |
| **QA Pairs** | Question-answer samples probing reasoning limits | 100 pairs |

### 1.3 Cognitive Dimensions Tested

The benchmark stress-tests architectures across three core cognitive dimensions:

| Dimension | Description | Example |
|-----------|-------------|---------|
| **Temporal Reasoning** | Resolution of indexical expressions and scheduling conflicts | *"Did I wake up before the alarm today?"* |
| **Cross-Source Reasoning** | Synthesis of information across heterogeneous applications | *"On my trip to Paris, did the plane land on time in Rome?"* |
| **Fact Retrieval** | Precision in extracting details or summarizing patterns | *"How long do incoming calls with Lucas Smith last on average?"* |

---

## 2. Data Composition

### 2.1 Information Objects Distribution

The 71 Information Objects are distributed across 7 source applications:

| Application | Count | Description |
|-------------|-------|-------------|
| **Calendar (Events)** | 20 | Single and recurring events (meetings, gym, travel) |
| **Photos (Images)** | 15 | Visual data with EXIF metadata (location, timestamp) |
| **Notes** | 15 | Unstructured text (mood logs, itineraries, to-do lists) |
| **Documents** | 9 | PDF files (tickets, receipts, product manuals) |
| **Phone (Calls)** | 6 | Call logs with duration, direction, timestamps |
| **Alarms** | 4 | Device alarm configurations |
| **Contacts** | 2 | Address book entries |

### 2.2 Cross-Source Reasoning Distribution

Questions are designed to require reasoning across varying numbers of data sources:

| Sources Required | Question Count | Percentage |
|------------------|----------------|------------|
| 1 Application | 63 | 63% |
| 2 Applications | 32 | 32% |
| 3 Applications | 4 | 4% |
| 4 Applications | 1 | 1% |

### 2.3 Temporal Context

To guarantee robust assessment of temporal logic:

- **Global Query Time**: $T_{global} = \text{2025-Sep-01 at 13:00}$
- **Data Temporal Range**: All information objects have creation dates prior to $T_{global}$
- **Temporal Expressions**: Questions include indexical references ("today", "last week", "next weekend")

---

## 3. Collection Process

### 3.1 Methodology

The dataset was **synthetically constructed** by domain experts to ensure:

1. **Controlled Complexity**: Systematic coverage of reasoning types
2. **Temporal Consistency**: All dates and events align with the fixed query time
3. **Cross-Source Dependencies**: Deliberate information fragmentation across applications
4. **Ground Truth Verifiability**: Each answer is deterministically derivable from the data

### 3.2 Annotation Process

- **Authors**: Question-answer pairs were authored by the research team
- **Validation**: Each QA pair was independently verified against the knowledge base
- **Format**: Triplets $(q, a_{target}, a_{sys})$ where $a_{target}$ contains only strictly necessary information

### 3.3 Quality Assurance

- Manual review of all 100 QA pairs for correctness
- Verification that ground truth answers are uniquely derivable
- Consistency checks for temporal and spatial references

---

## 4. Data Format

### 4.1 Directory Structure

```
evaluation/
├── README.md                      # This file (dataset documentation)
├── qa_dataset.csv                 # QA benchmark (100 pairs)
├── knowledge_base/                # Source data for PKG construction
│   ├── epistwin_docs/             # PDF documents
│   ├── epistwin_images/           # Visual data (JPEG/PNG)
│   └── epistwin_jsontxt/          # Structured metadata (JSON)
├── evalu.py                       # LLM-as-Judge evaluation script
├── merge_normalization.py         # Score normalization script
└── all_metrics_normalized.py      # IAA metrics computation
```

### 4.2 QA Dataset Schema (`qa_dataset.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `QUESTION` | string | Natural language query |
| `TARGET ANSWER` | string | Ground truth answer (minimal necessary information) |
| `EpisTwin ANSWER` | string | System-generated response (populated during evaluation) |

**File Format**: CSV with semicolon (`;`) delimiter, UTF-8 encoding

**Header Row**: `[ DATE: 01-Sep-2025 | DAY: Monday | TIME: 13:00 ]` (temporal context)

### 4.3 Knowledge Base Schema

Information Objects are stored as JSON-formatted text files:

#### Calendar Event Example
```json
{
    "source_app": "calendar",
    "event": "event_8",
    "metadata": {
        "title": "Gym",
        "date": "01-Sep-2025",
        "start_time": "07:00:00",
        "end_time": "08:30:00",
        "recurrence": "weekly",
        "days": ["Monday", "Wednesday", "Friday"]
    }
}
```

#### Phone Call Example
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

#### Photo Metadata Example
```json
{
    "source_app": "photos",
    "photo": "photo_20250729",
    "metadata": {
        "creation_date": "29-Jul-2025",
        "creation_time": "23:00:00",
        "location": "Rome - Fiumicino Airport",
        "coordinates": {"lat": 41.8003, "lon": 12.2389},
        "file_path": "epistwin_images/photo_20250729.jpg"
    }
}
```

---

## 5. Intended Uses

### 5.1 Primary Use Cases

- **Benchmarking Personal AI Systems**: Evaluate QA capabilities over personal knowledge graphs
- **Multi-hop Reasoning Research**: Study cross-source information synthesis
- **Temporal Reasoning Evaluation**: Test resolution of time-dependent queries
- **GraphRAG Development**: Train and evaluate graph-augmented retrieval systems

### 5.2 Out-of-Scope Uses

- **Production Personal Assistants**: Data is synthetic and not representative of real user patterns
- **Privacy Research**: No real personal data is included
- **Multilingual Evaluation**: English only

---

## 6. Evaluation Protocol

### 6.1 LLM-as-a-Judge Framework

We employ the **Prometheus** evaluation rubric with LLM judges:

| Score | Label | Description |
|-------|-------|-------------|
| 5 | Excellent | Accurate, comprehensive, perfectly addresses all aspects |
| 4 | Good | Correct, detailed, with only minor issues |
| 3 | Acceptable | Relevant and mostly correct, minor inaccuracies |
| 2 | Poor | Partially relevant, major inaccuracies |
| 1 | Incorrect | Completely incorrect or irrelevant |

### 6.2 Score Normalization

For stricter comparison, scores are normalized to a 3-point ordinal scale:

| Original | Normalized | Interpretation |
|----------|------------|----------------|
| 4-5 | 2 | Correct |
| 3 | 1 | Partially Correct |
| 1-2 | 0 | Incorrect |

### 6.3 Inter-Annotator Agreement Metrics

- **Cohen's Kappa** (Quadratic weighted)
- **Gwet's AC1** (Robust to high-agreement paradox)
- **Spearman's Correlation**
- **Percentage Agreement**

### 6.4 Running the Evaluation

```bash
# Prerequisites
pip install pandas numpy seaborn matplotlib scikit-learn krippendorff ollama

# Step 1: Run LLM-as-Judge evaluation
python evalu.py

# Step 2: Normalize and merge results
python merge_normalization.py

# Step 3: Compute IAA metrics
python all_metrics_normalized.py
```

---

## 7. Limitations and Biases

### 7.1 Known Limitations

| Limitation | Description |
|------------|-------------|
| **Synthetic Data** | Does not capture real-world noise, inconsistencies, or privacy concerns |
| **Single Persona** | Represents one user profile; may not generalize to diverse populations |
| **English Only** | No multilingual coverage |
| **Fixed Temporal Context** | All queries evaluated at a single point in time |
| **Limited Scale** | 71 objects and 100 questions may not stress large-scale retrieval |

### 7.2 Potential Biases

- **Western-centric Activities**: Events reflect typical Western lifestyle patterns
- **Tech-savvy User**: Assumes familiarity with digital calendars, cloud storage, etc.
- **Limited Demographic Diversity**: Single synthetic persona

### 7.3 Ethical Considerations

- **No Real Personal Data**: All data is synthetically generated
- **No Identifiable Information**: Names and contacts are fictional
- **Research Use Only**: Not intended for commercial deployment

---

## 8. Citation

If you use this dataset in your research, please cite:

```bibtex
@inproceedings{epistwin2026,
  title     = {EpisTwin: Neuro-Symbolic Personal Knowledge Graphs for Trustworthy Personal AI},
  author    = {},
  booktitle = {},
  year      = {},
  publisher = {}
}
```

---

## Acknowledgments

This work was partially supported by [funding sources]. We thank the reviewers for their constructive feedback.

---

*Last updated: January 2026*
