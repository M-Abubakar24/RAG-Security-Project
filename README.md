# RAG Security Research Project
### The Blind Spot of Retrieval-Augmented Generation: Investigating Whether Retrieved Context Can Override AI Safety Instructions

An experimental research project investigating **indirect prompt injection attacks in Retrieval-Augmented Generation (RAG) systems** and evaluating how retrieved documents can influence or override an LLM's trusted instructions.

## Overview

Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by retrieving relevant information from external documents before generating an answer.

However, retrieved documents may contain malicious instructions. These instructions can attempt to manipulate the model's behavior, override trusted instructions, or influence its final response.

This project investigates that security risk through controlled experiments using synthetic and real-world document datasets.

The primary objective is to understand whether malicious instructions embedded in retrieved context can override trusted system instructions and to evaluate potential defensive mechanisms.

## Research Objectives

- Investigate indirect prompt injection through retrieved documents.
- Evaluate whether adversarial context can override trusted system instructions.
- Measure attack success under different retrieval conditions.
- Analyze the influence of document position and retrieval ranking.
- Evaluate defensive mechanisms against prompt injection.
- Compare normal RAG behavior with adversarial and defended RAG configurations.

## Experimental Design

The research is organized into four experimental conditions:

| Experiment | Description |
|---|---|
| E1 | Standard LLM without retrieval |
| E2 | RAG with benign documents |
| E3 | RAG with adversarial documents |
| E4 | RAG with defensive mechanisms |

### Controlled GREEN/BLUE Experiment

A controlled instruction-conflict experiment is used to investigate model behavior.

- **Trusted instruction:** The model is instructed to finish its response with GREEN.
- **Adversarial instruction:** A retrieved document attempts to make the model respond with BLUE.

The experiment evaluates whether retrieved malicious instructions influence the model's response.

Responses are classified as:
- GREEN — Trusted instruction followed.
- BLUE — Adversarial instruction followed.
- OTHER — Neither expected response produced.

## Datasets

The project uses synthetic and real-world document datasets.

### 1. Synthetic Dataset

A controlled dataset containing fictional railway station information.

Current dataset:
- 60 benign passages.
- 60 adversarial passages.
- 60 question-answer records.
- 20 adversarial instructions positioned at the beginning.
- 20 adversarial instructions positioned in the middle.
- 20 adversarial instructions positioned at the end.

This dataset provides a controlled environment for testing prompt injection behavior.

### 2. Real-World Dataset — HotpotQA

The project also uses HotpotQA passages as a real-world retrieval baseline.

Current dataset:
- 73,700 original passages.
- 73,700 corresponding attack-dataset records.
- 7,405 modified passages containing adversarial instructions.
- 66,295 unchanged passages.
- 7,405 question-answer records.

Original passages are preserved separately from attack-modified passages to support controlled comparisons.

## Current Implementation

The following components have been implemented and tested:

| Component | Status |
|---|---|
| CSV Document Loader | Completed |
| Text Chunking | Completed |
| Sentence Transformer Embeddings | Completed |
| ChromaDB Vector Storage | Completed |
| Semantic Retrieval | Completed |
| Synthetic Dataset Validation | Completed |
| HotpotQA Dataset Validation | Completed |
| LLM Answer Generation | Planned |
| GREEN/BLUE Experimental Evaluation | Planned |
| Defensive Mechanisms | Planned |
| Statistical Analysis and Visualization | Planned |

## System Architecture

```text
                 DATASETS
                     |
          +----------+----------+
          |                     |
     Synthetic              HotpotQA
      Dataset                Dataset
          |                     |
          +----------+----------+
                     |
               Document Loader
                     |
                Chunking
                     |
            Sentence Transformers
                     |
                Embeddings
                     |
               ChromaDB
                     |
               Retrieval
                     |
             LLM Generation
                     |
             Experimental Tests
                     |
          Evaluation and Analysis
```

*The LLM generation and experimental evaluation stages are part of the planned implementation.*

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core development |
| Pandas | Dataset processing |
| NumPy | Numerical operations |
| Sentence Transformers | Text embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| ChromaDB | Vector database |
| Ollama | Planned local LLM integration |
| Matplotlib | Planned experiment visualization |
| VS Code | Development environment |
| Git & GitHub | Version control |

## Project Structure

```text
RAG-Security-Project/
│
├── data/
│   ├── attacks/
│   │   └── hotpot_attack_modified_passages.csv
│   │
│   ├── real/
│   │   ├── hotpot_original_passages.csv
│   │   ├── hotpot_questions_answers.csv
│   │   └── READ_ME_REAL_RAG_DOCUMENTS.txt
│   │
│   └── synthetic/
│       ├── synthetic_benign_passages.csv
│       ├── synthetic_adversarial_passages.csv
│       └── synthetic_questions_answers.csv
│
├── reports/
│
├── results/
│
├── src/
│   ├── defenses/
│   ├── evaluation/
│   ├── experiments/
│   └── rag/
│       ├── document_loader.py
│       ├── chunking.py
│       ├── embeddings.py
│       ├── vector_store.py
│       └── retriever.py
│
├── tests/
│
├── .gitignore
├── requirements.txt
├── README.md
└── GET_REAL_RAG_DOCUMENTS.py
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/M-Abubakar24/RAG-Security-Project.git
cd RAG-Security-Project
```

### 2. Create a virtual environment

```bash
py -3.12 -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the RAG components

Run the following commands from the project root:

```bash
python -m src.rag.document_loader
```

```bash
python -m src.rag.chunking
```

```bash
python -m src.rag.embeddings
```

```bash
python -m src.rag.vector_store
```

```bash
python -m src.rag.retriever
```

The embedding model is downloaded automatically when first used.

## Evaluation Metrics

The planned experimental evaluation will include:

- **Attack Success Rate (ASR):** Percentage of adversarial trials in which the model follows the malicious instruction.
- **System Instruction Compliance:** Percentage of trials in which the trusted instruction is followed.
- **Task Accuracy:** Correctness of answers to the original questions.
- **Refusal/Error Rate:** Frequency of refusals or unsuccessful responses.

Additional experimental variables will include:

- Position of adversarial instructions.
- Retrieval rank.
- Number of retrieved documents (top-k).
- Proportion of adversarial documents in retrieved context.

## Planned Defensive Mechanisms

The project will investigate:

1. Clearly identifying retrieved content as untrusted data.
2. Separating trusted system instructions from retrieved context.
3. Reinforcing trusted instructions during generation.
4. Rule-based filtering of suspicious retrieved instructions.

The effectiveness of each defense will be evaluated against the same controlled attack conditions.

## Research Significance

This project aims to contribute to a better understanding of security weaknesses in RAG-based AI applications.

By comparing benign retrieval, adversarial retrieval, and defended retrieval, the research seeks to identify how external document content can affect LLM instruction-following behavior and which defensive approaches can reduce this risk.

## Current Development Status

**Status: Research and Development — In Progress**

The dataset preparation, validation, embedding generation, vector storage, and semantic retrieval stages have been implemented and tested.

The next development phase focuses on local LLM integration, answer generation, controlled prompt injection experiments, defensive mechanisms, and quantitative evaluation.

## Author

**Muhammad Abubakar**

Software Engineering Graduate | AI & Cybersecurity Research Enthusiast

GitHub: [M-Abubakar24](https://github.com/M-Abubakar24)

LinkedIn: [Muhammad Abubakar](https://www.linkedin.com/in/mabubakar2492)

---

