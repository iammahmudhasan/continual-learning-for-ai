# Continual Learning for AI

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Topic](https://img.shields.io/badge/Topic-Continual%20%2F%20Lifelong%20Learning-orange)](#)

> Implementations, architectures, and benchmarks for Continual Learning (Lifelong Learning) in Deep Learning and AI systems.

---

## 📌 Overview

**Continual Learning (CL)**—also known as Lifelong Learning or Incremental Learning—aims to enable machine learning models to continuously learn from a stream of tasks or data distributions over time without suffering from **Catastrophic Forgetting** (the tendency to abruptly forget previously learned knowledge upon learning new information).

## 🚀 Key CL Approaches

1. **Regularization-Based Methods**:
   - Penalize changes to parameters critical for past tasks (e.g., EWC, SI, MAS, LwF).
2. **Replay & Memory-Based Methods**:
   - Store or generate representations from previous tasks to rehearse during training (e.g., Experience Replay, GEM, A-GEM, Generative Replay).
3. **Architecture-Based / Parameter Isolation**:
   - Dynamically allocate new network modules or parameters for new tasks (e.g., Progressive Neural Networks, PackNet, HAT).

---

## 📂 Repository Structure

```text
continual-learning-for-ai/
├── data/              # Datasets and benchmarks loaders (Split-MNIST, CIFAR-100, etc.)
├── models/            # Base architectures and continual learning modules
├── methods/           # CL algorithms (regularization, replay, expansion)
├── experiments/       # Training and evaluation pipelines
├── notebooks/         # Exploratory notebooks and visualizations
├── requirements.txt   # Dependencies
└── README.md          # Project documentation
```

---

## ⚙️ Getting Started

### Installation

```bash
git clone https://github.com/iammahmudhasan/continual-learning-for-ai.git
cd continual-learning-for-ai
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 👤 Author

**Mahmud Hasan**
- GitHub: [@iammahmudhasan](https://github.com/iammahmudhasan)
