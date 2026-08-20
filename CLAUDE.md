# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

A personal Python learning/practice repository. It is a collection of independent study zones rather than a single application — there is no build system, package manifest, or test framework configured at the root. Each subdirectory is a self-contained topic. Doc files and code comments are predominantly in Chinese.

## Running code

There is no build step. Scripts are run directly:

```bash
python <script.py>
```

The de-facto entry point for most scripts is the `if __name__ == '__main__':` block at the bottom of the file, which usually contains a small demo/assertion. To "test" a script, run it directly and read its stdout — there is no pytest configuration (despite `.pytest_cache/` existing from occasional `pytest` runs; treat those files as plain runnable scripts, not pytest suites).

For the two LLM applications under `tanxin/llm_app_dev/`, each has its own `requirements.txt` and expects its own conda environment — see those subdirectories' README files for the exact `conda create` + `pip install` sequence. The advanced-rag app pins `langchain==0.0.354` (with a documented `langchain-community==0.0.19` workaround for a `No module named 'pwd'` bug on Windows) — do not upgrade langchain blindly.

## Linting

Pylint is configured via `.vscode/settings.json` with lint-on-save enabled and the default problem cap raised to 300. Match the surrounding code's existing style rather than reformatting wholesale.

## Code conventions (from `doc/00_code_standard.md`)

- **Indent**: 4 spaces.
- **Encoding**: UTF-8; files start with `# -*- coding: utf-8 -*-` (or `#coding: utf-8`).
- **Imports**: one module per line; grouped as stdlib → third-party → local, with a blank line between groups. Place after the module docstring, before globals/constants. (`from sys import stdin, stdout` — multiple symbols from the same module on one line is allowed.)
- **Spacing**: spaces around binary operators (`a = b + c`); a blank line between independent blocks and after variable declarations.
- **Docstrings**: module docstring at file top (功能/版权); class docstring indented 4 spaces on the line after `class X:`; public-function docstrings indented 4 spaces after the signature, structured as 功能描述 / 参数 / 返回值 / 异常描述. Inline comments start with `# ` (hash + space).
- **LeetCode solutions** (`leetcode/src/`) follow the platform's `class Solution` signature convention; keep the `__main__` demo block.

## Layout / architecture

- `studying/src/` — numbered concept scripts (`00_basic_datatype.py` … `26_torch.py`, plus `300_*` ML/DL demos and `500_pytorch_nn_embedding.py`). Numbering approximates learning order; the `300_*` files are algorithm implementations (LightGBM, LSTM, BPE, GBDT, linear regression).
- `studying/high-compute/` — NPU/GPU training experiments and memory-profiling tooling. Contains `sparse_optimizer.py`, `memory_profiler.py`, `logger_utils.py`, and `test_a2a_local*.py` / `test_train*.py` experiment drivers alongside their `.txt` log outputs and an a2a training-flow analysis in `doc/`. Log files and shell scripts (`run_hccl.sh`) live next to the code that produces them.
- `leetcode/src/` — LeetCode solutions, one problem per file, named `<num>_<snake_case_problem>.py`.
- `tanxin/llm_app_dev/` — two real, dependency-pinned LLM apps, each with its own README + `requirements.txt` + `.env.example`:
  - `llm-developing-advanced-rag/` — notebooks `0.ipynb`→`3.2.ipynb` building simple→advanced RAG over PDFs, evaluated with Ragas. Uses BGE embeddings, BM25 + ensemble retrievers, and contextual compression (LLMChainExtractor / BGE Reranker). FAISS indices and eval xlsx outputs live under `data/`.
  - `llm-developing-newsgpt/` — a Qdrant-backed news Q&A app (`main.py`, `NewsGPT.py`, `db_qdrant.py`, `save_news_to_qdrant.py`); runs against a Docker Qdrant container and needs `.env` with `OPENAI_API_KEY`/proxy settings. GPU PyTorch is optional (see `tests/test_gpu.py`).
- `doc/` — Chinese study notes organized by topic: `00_*` (env/standards/compilation/cursor-usage), `high-compute/`, `RL/`, `agent/`. Images under `doc/assets/`.

## Environment notes

- Machine runs Windows; conda is the primary environment manager. App-specific envs are named in each README (e.g. `py38_NewsGPT`). `doc/00_python_environment.md` documents pip/conda mirror configuration (USTC/Tsinghua/Aliyun) and the `pip install --no-deps --no-build-isolation -e .` editable-install flags.
- Secrets and proxy settings belong in `.env` (or `pip.ini`/`.condarc`), which are never committed — only `.env.example` templates exist in the repo.
- `.gitignore` excludes `build/`, `*.log`, `.idea`, and `__pycache__` directories.

## When making changes

- This repo has no CI and no automated tests — verify by running the affected script directly. For LeetCode files, the `__main__` block is the smoke test.
- Preserve the numbered-file naming in `studying/src/` and `leetcode/src/` when adding new files in those zones.
- New ML/notebook work in `tanxin/llm_app_dev/` should carry its own `requirements.txt` and `.env.example`; do not hoist dependencies to the repo root (there is no root manifest to hoist into).
- Notes/docs are written in Chinese — match that language and the existing `[TOC]` + `#` heading structure when editing `doc/`.
