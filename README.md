# Speech DTW Recognizer

This repository implements a vowel recognition system based on 39-dimensional MFCC features and Dynamic Time Warping (DTW) with generalized templates.

## Prerequisites
- Python 3.12 (tracked in `.python-version`)
- [`uv`](https://docs.astral.sh/uv/) package manager (version 0.6.11 or newer)
- Audio dataset arranged under `data/train/` (templates) and `data/test/` (open-set speakers)

## Initial Setup
1. Install uv if it is not available:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   On Windows PowerShell:
   ```powershell
   irm https://astral.sh/uv/install.ps1 | iex
   ```
2. From the repository root, create the project environment and install dependencies:
   ```bash
   uv sync
   ```
   This creates `.venv/` using the interpreter specified in `.python-version` and installs the dependencies from `pyproject.toml`/`uv.lock`.

## Running Evaluations
- Baseline evaluation (templates built from indices 1-3, closed-set evaluated on indices 4-5):
  ```bash
  uv run speech-dtw
  ```
- Include additional template samples (e.g., indices 1-4):
  ```bash
  uv run speech-dtw --train-max-index 4
  ```
- Inspect all CLI options:
  ```bash
  uv run python -m speech_dtw.cli --help
  ```

Each run reports closed-set accuracy (same speakers as training), open-set accuracy (other speakers from `data/test`), and the mean of both metrics.
