# inspect-case-bench

[Inspect AI](https://inspect.aisi.org.uk/) implementation of **CASE-Bench** (Context-Aware Safety Evaluation Benchmark; Sun et al., ICML 2025, [arXiv:2501.14940](https://arxiv.org/abs/2501.14940)), ported from the reference code at [BriansIDP/CASEBench](https://github.com/BriansIDP/CASEBench).

> **Content warning.** The dataset contains red-teaming queries involving sensitive topics (self-harm, violence, child abuse, illegal activity). It is intended for safety evaluation only.

## What it measures

Each of the 900 samples pairs a potentially unsafe query with a formally described context built on Contextual Integrity theory (sender, recipient, transmission principle). The model under test judges whether responding to the query is safe *in that context*. The verdict is compared against the human majority label derived from `safe_rate` in the released data (20 annotations per sample).

The binary protocol from the reference `compute_score.py` is replicated, including its treatment of the `child-related crimes` category (scored with the `unsafe` label rather than excluded). A model output that does not contain a parseable `safe`/`unsafe` verdict fails closed to `unsafe`, so malformed verdicts cannot inflate the safe rate.

## Metrics

- `accuracy` — agreement with the human majority label
- `safe_accuracy` / `unsafe_accuracy` — accuracy restricted to samples whose human label is safe / unsafe
- `macro_f1` — macro-F1 over the two classes (unsafe = positive)

## Dataset

`data/CASEBench_data.json` is fetched at runtime from the upstream repository, pinned by commit SHA `135f221e0e3e4cdc9a7d17288fa6b54c8aa45d0a`. A different pinned URL can be passed with `-T data_url=...`.

## Usage

```bash
uv sync --extra inspect

# small run against a model
uv run inspect eval inspect_case_bench/case_bench.py@case_bench --model hf/Qwen/Qwen2.5-0.5B-Instruct --limit 20

# offline smoke test
uv run inspect eval inspect_case_bench/case_bench.py@case_bench --model mockllm/model --limit 10
```

## Tests

```bash
uv run pytest tests/ -q
```

## Citation

```bibtex
@inproceedings{sun2025casebench,
  title={{CASE-Bench}: Context-Aware Safety Benchmark for Large Language Models},
  author={Sun, Guangzhi and Zhan, Xiao and Feng, Shutong and Woodland, Phil and Such, Jose},
  booktitle={International Conference on Machine Learning},
  year={2025}
}
```
