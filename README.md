# NEO PDF AI-Optimizer

**AI-Guided PDF Structural Compression via Domain-Constrained Monte Carlo Tree Search (MCTS)**

Author: Hideyoshi Murakami  
X (Twitter): @nagisa7654321  

---

## Overview

NEO PDF AI-Optimizer is an experimental PDF compression system that treats PDF compression as a **global optimization problem**
rather than a preset-based task.

The system applies a **domain-specialized Monte Carlo Tree Search (MCTS)** to explore compression strategies under strict safety constraints.

Unlike conventional tools, each rollout executes a real recompression pipeline and is evaluated using a multi-objective scoring function.

---

## Key Concepts

- PDF as a structured document language
- Domain-constrained Monte Carlo Tree Search
- Heuristic-guided exploration
- Multi-objective, asymmetric evaluation function
- Ghostscript-free GS-emulation (pikepdf + Pillow)

---

## Quick Start (Local)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

---

## Deploy (Render)

This repository includes `render.yaml`.

1. Push this repo to GitHub
2. On Render: New > Blueprint > select your repo
3. Deploy

---

## Disclaimer

This software is provided **AS IS**. Always verify important documents before use.

---

## License

Apache-2.0 (see LICENSE).


## Docs
- Technical flowchart: `docs/flowchart.md`
- Japanese README: `README_ja.md`
