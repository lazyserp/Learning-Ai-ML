# AI Engineering Roadmap

Welcome to your roadmap to becoming a production-grade AI Engineer. This document maps out the essential phases of your learning journey, from mathematical foundations to scaling models in production.

## The Learning Journey

```
+--------------------------------------------------------+
|  Phase 1: ML Foundations (Classical Algorithms)        |
+--------------------------------------------------------+
                           |
                           v
+--------------------------------------------------------+
|  Phase 2: Deep Learning (Neural Networks, CNNs, LSTMs) |
+--------------------------------------------------------+
                           |
                           v
+--------------------------------------------------------+
|  Phase 3: Large Language Models & GenAI (Transformers)  |
+--------------------------------------------------------+
                           |
                           v
+--------------------------------------------------------+
|  Phase 4: Retrieval-Augmented Generation & Agents      |
+--------------------------------------------------------+
                           |
                           v
+--------------------------------------------------------+
|  Phase 5: MLOps (Model Deployment, Scaling, Monitoring) |
+--------------------------------------------------------+
```

---

## 1. Classical Machine Learning (Foundations)
* **Goal**: Understand optimization, gradient descent, classification, and regression.
* **Topics**: 
  - Linear & Multiple Regression
  - Polynomial Regression (non-linearity)
  - Logistic Regression (probability & cross-entropy)
  - Decision Trees, Random Forests, SVMs, KNN, Clustering

## 2. Deep Learning
* **Goal**: Move from manual feature engineering to automatic representation learning.
* **Topics**:
  - Neural Network Architectures (layers, activations, backpropagation)
  - Convolutional Neural Networks (CV)
  - Recurrent Neural Networks & LSTMs (Sequential data)
  - Attention Mechanisms & Transformers (The foundation of modern AI)

## 3. Large Language Models & GenAI
* **Goal**: Build and fine-tune systems powered by modern foundation models.
* **Topics**:
  - Tokenization & Embeddings
  - Prompt Engineering & Context Window Management
  - Fine-Tuning (SFT, LoRA, QLoRA)
  - Model Evaluation & Guardrails

## 4. RAG & Agentic Systems
* **Goal**: Connect LLMs to external data sources and give them tools to act.
* **Topics**:
  - Document parsing, chunking, and embedding pipelines
  - Vector Databases (Chroma, Pinecone, pgvector)
  - Retrieval techniques (hybrid search, re-ranking)
  - AI Agents (ReAct framework, tool calling, memory management)

## 5. MLOps
* **Goal**: Take models out of notebooks and serve them reliably in production.
* **Topics**:
  - Model Serving (TGI, vLLM, Ollama)
  - Inference optimization (quantization, pruning)
  - CI/CD for ML (GitHub Actions, MLflow)
  - Evaluation and observability (LangSmith, Phoenix)
