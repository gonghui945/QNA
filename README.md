# Quantum Network of Assets (QNA)

This repository contains the latest revised materials for:

**Quantum Network of Assets (QNA): A Density-Operator Framework for Market Dependence and Structural Risk Diagnostics**

The project develops an operator-based representation of cross-asset dependence
using normalized rolling multi-feature market states. The empirical application
uses a stable Nasdaq-100 panel over `2020-01-01` to `2025-12-31` and studies
how entropy, purity-based mixing, and event-aligned structural deviations
compare with more classical covariance-spectrum diagnostics.

## Repository layout

- `analysis/`
  - `src/`: data download, metric construction, robustness, and figure/table export scripts
  - `notebooks/`: the main exploratory notebook used to organize the revised empirical workflow
  - `requirements.txt`: minimal Python dependencies
- `data/`
  - `raw/market_data/`: daily local CSV files for the stable-panel market sample
  - `processed/`: processed metric outputs used in the revised manuscript
  - `reference/`: ticker universe and event-catalog reference files
- `paper/`
  - `manuscript/`: latest journal-style LaTeX manuscript, class file, and compiled PDF
  - `figures/`: final figures used in the paper
  - `tables/`: final LaTeX tables used in the paper
- `arxiv/`
  - an arXiv/preprint-ready version of the latest manuscript with journal branding removed

## What is included

- The latest revised manuscript in QFE journal style
- A separate arXiv-ready preprint version
- Reproducible analysis scripts and notebook
- Reference files and processed outputs
- The raw stable-panel daily market CSV files used for the current revision

## What is intentionally excluded

This GitHub package does **not** include:

- peer-review PDFs
- response-to-reviewers and cover-letter files
- submission-system packaging artifacts
- rendered preview thumbnails and LaTeX build byproducts

## Quick start

Create a Python environment and install the minimal dependencies:

```bash
pip install -r analysis/requirements.txt
```

Download or refresh the local market data:

```bash
python analysis/src/download_market_data.py --ticker-source wikipedia
```

Rebuild processed metrics, figures, and manuscript tables:

```bash
python analysis/src/build_revision_outputs.py
```

## Manuscript files

- Journal-style manuscript:
  - `paper/manuscript/QNA-QFE-revised.tex`
  - `paper/manuscript/QNA-QFE-revised.pdf`
- arXiv/preprint version:
  - `arxiv/main.tex`
  - `arxiv/main.pdf`

## Data note

The empirical design uses a **stable current-constituent Nasdaq-100 panel**
over `2020--2025`. This avoids composition breaks inside the rolling operator,
but it also implies survivorship bias. The paper discusses this design choice
explicitly as a limitation.

## Suggested GitHub setup

- Choose your preferred open-source or research-sharing license before making the repository public.
- If desired, add a short project description such as:
  - `Operator-based market dependence diagnostics with QNA, ERI, and QEWS`
