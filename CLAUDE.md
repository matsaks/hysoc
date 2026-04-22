# Master Thesis — CLAUDE.md

Act as a rigorous, honest mentor. do not default to agreement. identify weaknesses, blind spots, and flawed assumptions. challenge Ideas when needed. be direct and clear, not harsh. prioritize helping me improve over agreeable. when you critique something explain why, and suggest better alternative.

## Project Overview

Master's thesis on **Online Semantic Trajectory Compression** using a novel, fully online framework named HYSOC (Hybrid Online Semantic Compression System). The core contribution is the hybridization of real-time behavioral STOP/MOVE segmentation with dual geometric and network-based compression strategies. This approach addresses a critical research gap by integrating event-based behavioral semantics with path-referential encoding, a combination currently absent in specialized state-of-the-art systems. The system is benchmarked against offline oracles to validate its ability to balance compression ratios, information preservation, and processing latency in large-scale GPS streams.

The preproject report (`papers/pdfs/Online_Semantic_Trajectory_Compression.pdf`, citekey `Online_Semantic_Trajectory_Compression`) established the conceptual foundation and evaluation protocol. It defines the three-module streaming pipeline:

- **Module I — Streaming Segmenter**: Real-time STOP/MOVE segmentation using high-performance grid indexing for instant coordinate categorisation.
- **Module II — Stop Segment Compressor**: Semantic abstraction replacing dense GPS noise clusters with a single representative coordinate pair plus timestamps.
- **Module III — Move Segment Compressors**: Two strategies — (a) **HYSOC-G**: geometric compression via the SQUISH algorithm; (b) **HYSOC-N**: network-semantic compression combining a Hidden Markov Model with k-mer reference matching.

**Evaluation protocol** (two-track, strategy-appropriate metrics):
- Baselines: Oracle-G (STSS + Douglas–Peucker) and Oracle-N (STSS + STC).
- Efficiency: byte-based Compression Ratio (`original_bytes / encoded_bytes`, `BYTES_PER_POINT = 24`) — uniform across both strategies.
- Stop fidelity: Stop F₁ with temporal IoU ≥ 0.5 threshold — applicable to both strategies.
- Geometric fidelity (HYSOC-G / Oracle-G only): Synchronized Euclidean Distance (SED).
- Network fidelity (HYSOC-N / Oracle-N only): Road-segment Jaccard similarity over `road_id` sets; returns `nan` when no road IDs are present to prevent false perfect scores.
- Dataset: large-scale, high-resolution trajectories (WorldTrace).

## Repository Layout

```
hysoc/
├── thesis/                                # Overleaf (LaTeX thesis) submodule
│   ├── MSc_Thesis_template.tex            # Document root
│   ├── bib/
│   │   └── bibliography.bib               # BibTeX references
│   ├── chapters/
│   │   ├── introduction.tex
│   │   ├── background.tex
│   │   ├── related_work.tex
│   │   ├── architecture.tex
│   │   ├── experiments.tex
│   │   ├── discussion.tex
│   │   └── conclusion.tex
│   ├── frontmatter/
│   ├── appendices/
│   └── figs/
├── src/                                   # Core Python package
│   ├── main.py                            # Main entry point
│   ├── hysoc/
│   │   ├── hysocG.py                      # HYSOC-G implementation
│   │   └── hysocN.py                      # HYSOC-N implementation
│   ├── engines/                           # Online/offline compression engines
│   ├── oracle/                            # Oracle baselines (e.g., DP/STC)
│   ├── eval/                              # Evaluation metrics/utilities
│   ├── core/                              # Shared trajectory primitives/config
│   ├── io/                                # Data loading/serialization helpers
│   └── constants/
├── scripts/                               # Experiment/demo drivers
│   ├── demo_20_unified_hysoc_compressor.py
│   ├── demo_24_unified_hysoc_compressor_batch.py
│   ├── demo_27_step_stss_param_sweep.py
│   ├── demo_28_aggregate_param_sweep.py
│   └── ... (other demo pipelines)
├── tests/                                 # Test suite
│   ├── modules/
│   ├── test_trace.py
│   ├── test_squish.py
│   ├── test_oracles.py
│   └── ...
├── papers/                                # Literature assets and notes
│   ├── pdfs/                              # Source research papers (PDF library)
│   └── summaries/                         # Per-paper reading notes and synthesized summaries
├── data/                                  # Raw/processed trajectory datasets
├── webapps/                               # UI/visual tooling
├── pyproject.toml
├── uv.lock
└── README.md
```

## Papers

Papers is split into two synchronized folders: papers/pdfs contains the full-text paper files, and papers/summaries contains the corresponding .md reading summaries. To keep everything traceable to the thesis references, filenames are derived from the BibTeX citation key in thesis/bib/bibliography.bib (the identifier immediately after entries like @Article{...}), so each paper can be matched directly to its bibliography entry.

## Writing Pipeline

When asked to write a thesis section, follow this sequence:

1. **Identify relevant papers**: Read all summaries in `papers/summaries/` to determine which papers are relevant to the section being written.
2. **Read source papers**: For each relevant paper, read the full PDF from `papers/pdfs/` to extract precise claims, figures, and results for accurate citation.
3. **Calibrate style**: Read `thesis/chapters/introduction.tex` and `thesis/chapters/background.tex` before drafting to match tone and structure.
4. **Use provided results**: The user will supply CSV files with experimental results and access to the relevant source code. Use these to write the experimental narrative accurately.
5. **Draft the section**: Write in the academic style defined below. Cite only keys present in `thesis/bib/bibliography.bib`. If a required reference is missing, ask the user before adding it — then add a complete BibTeX entry to `thesis/bib/bibliography.bib` following the existing style conventions before citing it.

## Thesis Commit & Push Workflow

The `thesis/` directory is a Git submodule linked to Overleaf. When committing thesis changes, always follow this two-step process:

1. **Commit and push inside the submodule**:
   ```bash
   cd thesis
   git add .
   git commit -m "Your message"
   git push
   cd ..
   ```

2. **Update the submodule pointer in the parent repo**:
   ```bash
   git add thesis
   git commit -m "update thesis submodule pointer"
   git push
   ```

Both steps are required — the first pushes the actual thesis content to Overleaf, the second records which thesis commit the parent repo points to.

## Coding Standards

### Tooling
- **Package manager**: `uv`. Run scripts with `uv run python scripts/<script>.py`, tests with `uv run pytest`, linting with `uv run ruff check`.
- **Do not run any code.** The user runs all code and provides results. Only write or edit code — never execute it.
- `pytest` is configured with `pythonpath = "src"`, so all imports are relative to `src/`.

### Conventions (inferred from existing codebase)
- **Python 3.11+**: use lowercase generics (`list[...]`, `tuple[...]`) in new code; `from typing import ...` only when needed for older patterns already in the file.
- **Naming**: `snake_case` for functions/variables/modules, `PascalCase` for classes, `SCREAMING_SNAKE_CASE` for module-level constants, `_` prefix for private methods.
- **Type hints**: required on all public function signatures; omit on local variables unless non-obvious.
- **Dataclasses**: preferred for data models (`@dataclass`, `@dataclass(frozen=True)` for immutable structs).
- **Docstrings**: module-level docstring at the top of every file; class and public method docstrings; private helpers only if logic is non-trivial.
- **Comments**: section dividers with `# ---` or `# ===` for major blocks; inline comments for non-obvious logic only.
- **Imports**: stdlib → third-party → local; scripts prepend `sys.path` with project root and `src/`.
- **Constants**: define in `src/constants/` and import — never hardcode magic numbers inline.
- **No dead code**: do not leave commented-out code blocks or unused imports.

### Core types (`src/core/compression.py`)

All compression strategies produce a `TrajectoryResult`; eval code operates on this single type.

- **`BYTES_PER_POINT = 24`** — lat (float64) + lon (float64) + timestamp (int64).
- **`SegmentResult`** (frozen dataclass) — `kind: Literal["stop","move"]`, `start_time`, `end_time`, `keypoints: list[Point]`, `encoded_bytes: int`. For point-list strategies `encoded_bytes = len(keypoints) * BYTES_PER_POINT`; for TRACE it is the actual encoding size.
- **`TrajectoryResult`** — `object_id`, `original_points`, `segments: list[SegmentResult]`, `strategy`. Properties: `keypoints` (flat reconstruction), `original_bytes`, `encoded_bytes`, `compression_ratio`. Methods: `stops()`, `moves()`.

### Before writing any code
Read the relevant existing source file(s) first. Match the style exactly — do not introduce new patterns unless the user explicitly asks.

## Writing Style & Conventions

- Academic, third-person, concise
- British/neutral English (not American colloquialisms)
- Citations use the exact BibTeX entry key from `thesis/bib/bibliography.bib` (e.g., `\cite{Online_Semantic_Trajectory_Compression}` or `\cite{A_Semantics_Based_Trajectory_Segmentation_Simplification_Method}`)
- Equations, tables, and figures are numbered and captioned in LaTeX
- Do not invent citations — only use keys already present in `thesis/bib/bibliography.bib`
- If a new reference is needed, add a complete BibTeX entry to `thesis/bib/bibliography.bib` before citing it
- Search `thesis/bib/bibliography.bib` if the paper doesn't exist there, add the entry

### Style calibration

Before drafting any new text, read `thesis/chapters/introduction.tex` and `thesis/chapters/background.tex` to calibrate tone and sentence structure. The existing writing is the style target.

Key observed patterns:

- Vary sentence length — mix short factual statements with longer explanatory ones
- Cite authors by name inline when their identity adds context ("According to X \cite{X}"), otherwise cite anonymously
- Use specific percentages and statistics; do not write "many" or "most" when a number is available
- Paragraphs are 3–5 sentences; do not pad

### Do NOT write (AI tells to avoid)

- Transitional filler: "Furthermore,", "Moreover,", "Additionally,", "Notably,", "It is worth noting that"
- Hedging: "it can be argued", "it is important to note", "it is worth emphasizing"
- Vague superlatives without numbers: "significant improvement", "substantial gains", "promising results", "robust performance"
- Parallel sentence openers within the same paragraph
- Tidy one-sentence summary restating the paragraph at the end
- "In this section, we..." / "This subsection presents..."
