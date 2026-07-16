# AI Engineering Interview Roadmap

This document outlines the preparation strategy for passing top-tier AI Engineering, Machine Learning Engineering, and GenAI positions.

---

## 1. Core Technical Pillars

```
                     +---------------------------+
                     |  AI Engineering Interview |
                     +---------------------------+
                                   |
       +---------------------------+---------------------------+
       |                           |                           |
       v                           v                           v
+--------------+            +--------------+            +--------------+
| Coding & DSA |            | Classical ML |            | System Design|
| (Python/SQL) |            |   & Deep L.  |            |   (ML/RAG)   |
+--------------+            +--------------+            +--------------+
```

### Pillar 1: Coding & Data Structures
* **Focus**: Clean, readable Python. Matrix operations using NumPy.
* **Topics**: Lists, Hash Maps, Trees, Graphs, Matrix Transformations, and vectorized computations.

### Pillar 2: ML & Deep Learning Theory
* **Focus**: Knowing the **Why** behind every algorithm. Why does X exist? How does it improve on Y?
* **Topics**:
  - MSE vs. Cross-Entropy Loss
  - Optimization algorithms (Gradient Descent, Adam)
  - Transformers & Attention mechanism details (Self-Attention math, shapes)
  - Fine-tuning methodologies (LoRA, PEFT)

### Pillar 3: AI System Design
* **Focus**: Designing end-to-end production pipelines.
* **Topics**:
  - Designing a RAG system at scale (ingestion, vector search, latency, cost tradeoffs)
  - Designing recommendation feeds
  - Serving LLMs under high concurrency (caching, batching, vLLM routing)

---

## 2. High-Frequency Interview Questions

Check out the specific `interview.md` documents under each topic folder. They cover core conceptual questions:
- *Why is Sigmoid used in binary classification instead of a step function or linear regression?*
- *How does LoRA work, and why does it save GPU memory?*
- *What is the difference between Batch Normalization and Layer Normalization, and where are they used?*
