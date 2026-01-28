# Technical Flowchart (MCTS PDF Compression)

This project treats PDF compression as a **search / optimization problem**.
Each MCTS rollout runs a *real* recompression pipeline (pikepdf + Pillow) and is
scored by a multi-objective function (size reduction × quality proxies × safety).

## High-level pipeline

```mermaid
flowchart TD
  A[User uploads PDF] --> B[Security guards
- upload size limit
- page count limit
- rate limit
- concurrency limit]
  B --> C[Session ID + cache key]
  C --> D[Detect document traits
- color presence
- profile: text / mixed / image
- image weight]
  D --> E[Set soft target reduction
(Auto or user selection)]
  E --> F[Extract representative sample images
(for scoring & preview)]

  F --> G[Seed evaluation
(hand-tuned initial states)]
  G --> H{Best state found?}
  H -->|Yes| I[MCTS Loop]
  H -->|No| I

  I --> J[Select node (UCB)]
  J --> K[Expand node (apply one action)]
  K --> L[Parallel rollouts
(ThreadPoolExecutor)]
  L --> M[Execute compression
(GS-Emulation → pike-only)]

  M --> N[Scoring
- size reduction vs target
- SSIM-like MSE proxy
- color penalty
- zoom robustness]
  N --> O[Backpropagate avg score]
  O --> P[Update best solution]
  P --> Q[Adaptive gate
(relax min quality threshold if stalled)]
  Q --> I

  P --> R[Final recompress using best state]
  R --> S[Return PDF + Download]
```

## Compression core (GS-Emulation, Ghostscript-free)

```mermaid
flowchart TD
  A[Input PDF bytes] --> B[Metadata strip]
  B --> C[pikepdf open]

  C --> D[Images pass
- traverse /XObject /Image (incl. /Form recursion)
- PdfImage → PIL]
  D --> E[Safety guards
- avoid SMask/Mask
- preserve color if original has color]
  E --> F[Downsample decision (GS-like)
- page inches × target DPI × threshold]
  F --> G[Encode decision
- JPEG (DCTDecode)
- or PDF-valid Flate image]
  G --> H[Replace only if smaller
(dedupe by SHA1)]

  C --> I[Streams pass
- recompress simple Flate streams
- contents & form streams
- dedupe]

  H --> J[pikepdf save
- compress_streams
- object streams]
  I --> J
  J --> K[Output PDF bytes]
```

## Why a flowchart file is useful in the repo

- Reviewers can understand the **search loop** and **what is actually executed**.
- It documents safety assumptions and the “Ghostscript-free” design.
- Helps users diagnose performance issues (Free plan timeouts) and tune profiles.
