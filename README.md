<h1 align="center">GPTQ Experiments — Llama-3.2 1B</h1>
<p align="center">Testing the GPTQ quantization method on a small language model, on language modeling and question answering tasks.</p>

## What is this?

This is a project I followed during my master's in AI, where I tested the GPTQ method on a SLM (Llama-3.2 1B). The results were tested on two main topics: **Language Modeling** and **Question Answering**.

Detailed notebooks can be found in `language_modeling.ipynb` and `question_answering.ipynb`. Also check the utility files for more infos -> `utils.py`, `utils_LM.py`, `utils_QA.py`.

## Repository structure

| Path | Description |
|------|-------------|
| `language_modeling.ipynb` | Language modeling experiments and analysis |
| `question_answering.ipynb` | Question answering experiments and analysis |
| `utils.py` | Shared utilities |
| `utils_LM.py` | Helpers for the language modeling experiments |
| `utils_QA.py` | Helpers for the question answering experiments |
| `LM_results/` | Language modeling results (WikiText) across bit widths |
| `QA_results/` | Question answering results (SQuAD) across bit widths and settings |

## Note

The GPTQ quantization in this project is performed using the **GPTQModel** library. This repository only contains my experiments and results — for the quantization library itself (and its latest updates) check out the original repository: [ModelCloud/GPTQModel](https://github.com/ModelCloud/GPTQModel).
