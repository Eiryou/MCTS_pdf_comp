# NEO PDF AI-Optimizer

**AI-Guided PDF Structural Compression via Domain-Constrained Monte Carlo Tree Search (MCTS)**

Author: Hideyoshi Murakami  
X (Twitter): @nagisa7654321  
https://x.com/nagisa7654321

## DOI (Zenodo)
10.5281/zenodo.18428396
https://doi.org/10.5281/zenodo.18428396


## DEMO URL 
https://mcts-pdf-comp.onrender.com/

I'm deploying using Render. I'm on the Starter plan, so I only have 512MB of memory and my site crashes sometimes. Please set the file size to less than 1MB and move the search frequency and thread count sliders to the left. This is experimental and is being run with limited funding.

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

##
Why so many clones but few stars?

A: This is a “try locally” type of project.
If this helped you, please consider starring ⭐ (it supports further development).

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

## Runtime profiles (FREE / STARTER)

Render's **Free plan** can be tight for CPU/RAM and may produce 502 timeouts on
large PDFs or aggressive MCTS settings.

This repo supports two internal presets that change safety limits **without
changing the UI**, controlled by an environment variable:

- `PDF_COMP_PROFILE=FREE` (default): conservative limits for public demos on Free.
- `PDF_COMP_PROFILE=STARTER`: higher limits for paid plans.

### Recommended settings

**FREE (public demo / stability first)**
- Lower caps: iterations/threads/runtime/page count/upload size
- Stronger concurrency & rate limits

**STARTER (paid plan / more throughput)**
- Higher caps: iterations/threads/runtime
- Higher upload/page limits

### How to switch

On Render:
1. Open your service
2. Settings → Environment
3. Add / change `PDF_COMP_PROFILE` to `FREE` or `STARTER`
4. Deploy (restart)

You can still override specific limits by setting env vars directly (they take
priority over the preset):

- `NEO_MAX_ITERATIONS`, `NEO_MAX_THREADS`, `NEO_MAX_RUNTIME_SECONDS`
- `NEO_MAX_UPLOAD_MB`, `NEO_MAX_PDF_PAGES`

See `profiles.py` for the exact values.

---

## Disclaimer

This software is provided **AS IS**. Always verify important documents before use.

---

## License

Apache License 2.0


## Docs
- Technical flowchart: `docs/flowchart.md`
- Japanese README: `README_ja.md`
- 
## Contact
For comments, work, and collaborations, please contact us here
murakami3tech6compres9sion@gmail.com


### Repo hygiene
- **Use ASCII-only filenames** in this repository (avoid Japanese filenames), especially under `assets/`.
  If you already have non-ASCII filenames, run: `python tools/rename_non_ascii.py`.
