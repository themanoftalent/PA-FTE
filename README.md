# PA-FTE: Privacy-Preserving Agentic Clinical Monitoring with Federated Temporal Encoding

**(COMPSAC/ESAS 2026 submission)**  
**Authors:** Mehmet Akif CIFCI (Bandirma Onyedi Eylul University)  
**Paper abstract & full description:** [link to paper PDF if available, or arXiv later]

This repository aims to provide reference implementation and reproduction materials for the PA-FTE architecture described in the paper.

## Key Components (as per paper)
- Federated causal-masked Transformer encoder (FedProx, only encoder shared)
- Local memory-augmented risk estimation (belief state M_t)
- Local Llama-3.1-8B agent (ReAct-style, schema-constrained JSON output via GBNF)
- Local tools: ACC/AHA guidelines SQLite + offline DrugBank
- Deterministic safety validator
- Proactive metrics: Early Detection Gain (EDG), Intervention Utility U
- Evaluated on MIMIC-IV ICU cohort (~32k patients, decompensation within 12h)

**Status:** Early development / skeleton phase (Feb 2026). Full code forthcoming.

## Requirements
- Python 3.10+
- PyTorch 2.1+ (with CUDA if available)
- Flower or custom FedAvg/Prox implementation
- llama.cpp (for local Llama-3.1-8B GGUF inference)
- SQLite3 (for local guideline DB)
- Access to MIMIC-IV v3.1 on PhysioNet (credentialed)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install flwr pandas numpy tqdm scikit-learn sqlite3
# llama.cpp: follow https://github.com/ggerganov/llama.cpp
# Download Llama-3.1-8B-Instruct Q4_K_M GGUF from HuggingFace
