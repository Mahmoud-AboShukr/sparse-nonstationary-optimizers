\# Sparse and Non-Stationary Gradients in Large-Scale Language Model Optimization



This research project investigates how different optimizers behave under \*\*sparse\*\* and \*\*non-stationary gradient regimes\*\* during the training of transformer-based language models (LMs). The goal is to develop a diagnostic framework capable of revealing optimizer robustness, failure modes, and adaptation behaviour in realistic language modeling scenarios.



---



\## Project Overview



Modern optimizer research shows that large language models frequently encounter:

\- \*\*Sparse gradients\*\* (only a small subset of parameters updated each step, especially embeddings and rare tokens)

\- \*\*Non-stationary gradients\*\* (gradient statistics drift over time as the optimization landscape changes)



Classical optimizers like SGD struggle under these conditions. Adaptive optimizers such as \*\*AdamW\*\*, \*\*Adafactor\*\*, and \*\*Lion\*\* were designed to address some of these issues, but their behaviour under combined sparsity + drift has not been systematically studied.



This project aims to fill that gap.



---



\## Research Objectives



\- Quantify gradient sparsity across transformer layers

\- Measure \*\*gradient drift\*\* and historical gradient aging

\- Compare optimizer behaviour under controlled conditions

\- Identify which optimizers:

&nbsp; - remain stable under sparse updates

&nbsp; - adapt to rapidly shifting gradient distributions

&nbsp; - accumulate stale momentum or variance estimates



---



\## Project Structure



sparse\_nonstationary\_optimizers/

│

├── src/

│ ├── data/ # dataset loaders (WikiText-103)

│ ├── models/ # GPT-style transformer implementation

│ ├── optimizers/ # AdamW, Adafactor, SGD, Lion wrappers

│ ├── metrics/ # sparsity + non-stationarity diagnostic tools

│ └── utils/ # helper functions

│

├── configs/ # configs for experiment runs

├── logs/ # raw log files

├── results/ # processed experimental results

├── plots/ # figures and comparison graphs

└── train.py # entry point for training experiments



\## Getting Started



\### Create and activate a virtual environment



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1



\### Install dependencies



pip install torch transformers datasets sentencepiece accelerate wandb



\### Run a test experiment



python train.py



\### Experiments



The project evaluates optimizers along two axes:

| Property             | Diagnostics                                                     |

| -------------------- | --------------------------------------------------------------- |

| \*\*Sparsity\*\*         | update frequency, embedding sparsity, parameter activation rate |

| \*\*Non-stationarity\*\* | gradient drift, v\_t statistics aging, layer-wise norm evolution |



\### Target optimizers:



SGD + Momentum



AdamW



Adafactor



Lion (optional)



Outcomes include:



Perplexity curves



Stability/divegence events



Layer-wise sensitivity maps



\## Author



Mohammed Abo Shukr

M2 MMVAI — Université Paris-Saclay

2025







