# Technical Flowchart (Mermaid)

This project treats PDF compression as a **search/optimization** problem.
Each rollout runs a real recompression pipeline (pikepdf + Pillow) and is scored by a multi-objective function.

> GitHub renders Mermaid automatically in Markdown.

```mermaid
flowchart TD
  A[Upload PDF] --> B{Security guards}
  B -->|size <= limit| B1[Read bytes]
  B -->|pages <= limit| B2[Parse via pikepdf]
  B -->|rate ok| B3[Acquire job semaphore]

  B1 --> C[Session ID + Cache Key]
  B2 --> D[Analyze document]
  D --> D1[Detect color usage]
  D --> D2[Estimate profile: text / mixed / image]
  D2 --> D3[Set soft target reduction]

  C --> E[Extract sample images]
  E --> E1[Representative XObject image sampling]

  D3 --> F[Initialize seed states]
  F --> G[Evaluate seeds]

  G --> H[MCTS loop (iterations)]
  H --> I[Selection: UCB]
  I --> J[Expansion: pick untried action]
  J --> K[Parallel rollouts (ThreadPoolExecutor)]

  K --> L[Execute recompression pipeline]
  L --> L1[Strip metadata]
  L1 --> L2[Recompress images]
  L2 --> L2a[Decode PdfImage -> PIL]
  L2a --> L2b[Downsample (GS-emulation rules)]
  L2b --> L2c{Choose output kind}
  L2c -->|jpeg| L2d[Encode JPEG (quality/subsampling/progressive)]
  L2c -->|flate| L2e[Build PDF Image XObject (FlateDecode)]
  L2d --> L2f[Replace only if smaller]
  L2e --> L2f

  L2f --> L3[Recompress simple Flate streams]
  L3 --> L4[Save with object streams + compress_streams]

  L4 --> M[Score rollout]
  M --> M1[Size reduction term]
  M --> M2[Quality term (SSIM-like MSE proxy)]
  M --> M3[Color penalty]
  M --> M4[Zoom robustness coeff]
  M --> M5[Gate/anneal threshold]

  M --> N[Backpropagate average score]
  N --> O{Best improved?}
  O -->|yes| P[Update best state]
  O -->|no| Q[Stall count++ -> relax gate]

  P --> R[Render progress + preview]
  Q --> R
  R --> H

  H --> S[Finish / time budget]
  S --> T[Run best state once more]
  T --> U[Download optimized PDF]

  %% Cache
  C --> C1{Cache hit?}
  C1 -->|yes| U
  C1 -->|no| L
```

## Notes
- **Ghostscript-free**: even if GS exists on the host, this build does *not* execute it (GS-emulation only).
- **Safety guards**: designed to avoid common failures (e.g., accidental grayscale conversion on color documents).
- **Replace-only-if-smaller**: prevents “compression” that actually inflates the PDF.
