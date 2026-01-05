Sparse and Non-Stationary Gradients in Large-Scale Language Model Optimization

This research project investigates how different optimizers behave under sparse and non-stationary gradient regimes during the training of transformer-based language models (LMs). The aim is to build a diagnostic framework capable of revealing optimizer robustness, failure modes, and adaptation behaviour in realistic language modeling scenarios.

📌 Project Overview

Large-scale language models frequently encounter two challenging gradient properties:

Sparse gradients — Only a small subset of parameters are updated at each training step, especially in embedding layers and rare tokens.

Non-stationary gradients — Gradient statistics drift over time as the optimization landscape evolves, making historical momentum and variance estimates potentially stale.

Classical optimizers like SGD fail to cope with these dynamics. Adaptive optimizers such as AdamW, Adafactor, and Lion improve stability, but their behaviour under combined sparsity + drift has not been systematically analysed.
This project fills that gap.

🎯 Research Objectives

Quantify gradient sparsity across different Transformer layers

Measure gradient drift and historical gradient aging

Compare optimizer behaviour under identical training conditions

Identify which optimizers:

remain stable under sparse update regimes

adapt efficiently to shifting gradient distributions

accumulate stale momentum or variance estimates

🏗️ Project Structure
sparse_nonstationary_optimizers/
│
├── src/
│   ├── data/          # dataset loaders (WikiText-103)
│   ├── models/        # GPT-style transformer implementation
│   ├── optimizers/    # AdamW, Adafactor, SGD, Lion wrappers
│   ├── metrics/       # sparsity + non-stationarity diagnostic tools
│   └── utils/         # helper functions
│
├── configs/           # experiment configuration files
├── logs/              # raw metric output
├── results/           # processed experimental outputs
├── plots/             # visualizations and comparison graphs
└── train.py           # main training entry point

🚀 Getting Started
1️⃣ Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

2️⃣ Install dependencies
pip install torch transformers datasets sentencepiece accelerate wandb

3️⃣ Run a test experiment
python train.py


You should see a confirmation message indicating the experiment framework initialized successfully.

🧪 Experiments

This project evaluates optimizers along two core axes:

Property	Diagnostics
Sparsity	Update frequency, embedding sparsity, parameter activation rate
Non-stationarity	Gradient drift, vₜ statistics aging, layer-wise norm evolution
Target Optimizers

SGD + Momentum — baseline for failure under sparse gradients

AdamW — industry standard

Adafactor — memory-efficient and sparse-friendly

Lion (optional) — sign-based update behaviour

👤 Author
Ramzi Amira
Mohammed Abo Shukr
M2 MMVAI — Université Paris-Saclay
2025

📄 License

This repository is part of an academic research project. Redistribution is permitted with attribution.
