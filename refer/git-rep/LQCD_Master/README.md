# LQCD Master

<div align="center">

**Agentic Scientific Computing for Lattice QCD - Natural Language to GPU Computation**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

LQCD Master translates natural-language physics requests into executable PyQUDA programs and Slurm submission scripts. It combines a **Planner** agent (physics plan generation with critique–rewrite) and an **Executor** agent (code generation with static analysis and auto-debugging) into a reproducible, human-supervised pipeline.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure your environment
cp .env.template .env
# Edit .env: fill in your LLM API key and Serper key

# Run a task
python run.py --task "Calculate the pion two-point correlator" --test --non-interactive
```

---

## Pipeline

```
Natural-language task
       │
       ▼
┌─────────────────┐
│     Planner      │  Physics plan (observable, propagators, contractions)
│  solve/critique/ │  → plan.yaml + summary.md
│     rewrite      │
└────────┬─────────┘
         │ [Optional: Human checkpoint]
         ▼
┌─────────────────┐
│    Executor      │  Code generation + Auto-debugging
│ generate/critique│  → Code artifacts
│    /rewrite      │
└────────┬─────────┘
         │ [Optional: Human checkpoint]
         ▼
┌─────────────────┐
│  Slurm Submit    │  GPU computation → result
└─────────────────┘
```

**Key design decisions:**

- **Human in the loop.** Every stage has a checkpoint — the system never submits to the cluster without confirmation.
- **Auto-debugging.** If static analysis detects `errors` in generated code, the executor rewrites automatically (capped at `executor_static_check_rounds`). Warnings are shown but don't block.
- **Skill routing.** Domain knowledge (correlator theory, PyQUDA API, analysis methods) is injected into LLM prompts per stage, selected by relevance to the current task.

---

## Project Structure

```
.
├── run.py                       # CLI entry point
├── core_architecture/            # Planner → Executor → Submit pipeline
│   ├── orchestrator.py
│   ├── planner.py
│   └── executor.py
├── utils/                        # LLM client, tools, I/O, skill system
│   ├── generate_einsum/          # Wick contraction & einsum code generation
│   └── ...
├── prompts/                      # Prompt templates per stage
│   ├── planner/                  # solve / critique / rewrite
│   └── executor/                 # generate / critique / rewrite
├── configs/                       # Cluster, ensemble, skill configuration
├── skills/                        # Domain knowledge (correlators, PyQUDA, analysis)
├── benchmark/                     # 70 validation tasks + standard-method reference data
└── experiments/                   # Organized experiment results with per-model summaries
```

---

## CLI Reference

```bash
python run.py [OPTIONS]
```

| Flag                   | Description                                    |
| ---------------------- | ---------------------------------------------- |
| `--task "<text>"`      | Task as inline text (skips interactive prompt) |
| `--task <file>`        | Read task from a file                          |
| `--test`               | Generate code only; skip Slurm submission      |
| `--run-dir <path>`     | Output directory (default: `runs/<timestamp>`) |
| `--non-interactive`    | Auto-accept all checkpoints                    |
| `--dotenv-path <path>` | Custom `.env` path                             |
| `--list-skills`        | List discoverable skills and exit              |

---

## Benchmark

The [benchmark/](benchmark/) directory defines **70 independent validation tasks** across 5 observable classes, all evaluated on the C24P29 ensemble (cfg 10000, point source, zero momentum).

| Observable       | Tasks | Examples                                                         |
| ---------------- | ----- | ---------------------------------------------------------------- |
| **2pt local**    | 20    | π, K, ρ, D, Dₛ, J/ψ, p, Λ, Ξ, Σ, Λ_c, Ξ_c, Ω_c, …                |
| **2pt nonlocal** | 10    | π, K, ρ, D, Dₛ, J/ψ, D\*, Dₛ\* (Wilson-line shift z = 0…10)      |
| **Wilson loop**  | 12    | W(R, T) for R, T ∈ {1, 2, 3, 4}, averaged over XT, YT, ZT planes |
| **3pt meson**    | 13    | D→K/π, B→D/K/π, D→K\*, B→K\*, Dₛ→φ, K→π, B_c→J/ψ                 |
| **3pt baryon**   | 15    | p→p, Λ→Λ, Λ→p, Λ_c→Λ, Ξ_c→Ξ, Λ_b→Λ_c/Λ/p, Ξ→Λ, Ξ_cc→Ξ_c          |

Each category includes hand-written PyQUDA reference implementations with numerical outputs for cross-validation.

---

## Experiment Results

Cross-validated against hand-written standard-method benchmarks on the full 70-task suite.

| Experiment                                                          | Backbone        | Exact  | Sign Flip | Failure | Accuracy  |
| ------------------------------------------------------------------- | --------------- | ------ | --------- | ------- | --------- |
| [LQCD Master GPT-5.4](experiments/LQCD_Master_GPT_5.4/summary.md)   | GPT-5.4         | **63** | 3         | 4       | **90.0%** |
| [LQCD Master DeepSeek](experiments/LQCD_Master_DeepSeek/summary.md) | DeepSeek V4 Pro | **56** | 2         | 12      | **80.0%** |

> **Exact:** matches standard-method at machine precision (|Δ| < 10⁻¹²) or relative error < 10⁻³.  
> **Sign flip:** global γ₅-Hermiticity phase convention — physics unaffected, counted separately.  
> **Failure:** genuine code-generation or execution errors.

See `summary.md` in each experiment subfolder for per-observable breakdowns, per-task failure tables, and timing statistics.

---

## Configuration

### Environment (`.env`)

```bash
OPENAI_API_KEY=sk-...         # LLM API key (required)
OPENAI_BASE_URL=...           # API endpoint (default: api.deepseek.com/v1)
OPENAI_MODEL=...              # Model name (default: deepseek-v4-pro)
SERPER_API_KEY=...            # Serper.dev key for web_search tool
```

### Cluster & Ensemble (`configs/`)

[`configs/config.yaml`](configs/config.yaml) defines Slurm partition, module loads, MPI launch, and QUDA environment paths. [`configs/ensemble_presets.yaml`](configs/ensemble_presets.yaml) defines 7 lattice ensembles with gauge config paths, quark masses, clover coefficients, and multigrid parameters. Both files must be customized for your cluster.

### Skill Routing (`configs/skills.yaml`)

Controls which domain-knowledge modules are injected at each pipeline stage. Five built-in skills cover correlator theory, spectrum analysis, PyQUDA API usage, gauge fixing, and data analysis.

---

## Dependencies

- **Python 3.10+**
- **pip packages:** `openai`, `prompt_toolkit`, `PyYAML`, `numpy`, `opt_einsum`
- **GPU backend:** `cupy` (CUDA or ROCm, matching your toolchain)
- **Lattice QCD:** PyQUDA + QUDA (installed from source on your cluster)
- **Cluster:** Slurm with GPU partition
- **External APIs:** OpenAI-compatible LLM endpoint + Serper.dev (web search)

---

## Contributing

1. Customize `configs/config.yaml` and `configs/ensemble_presets.yaml` for your cluster
2. Add new skills under `skills/` for domain-specific knowledge
3. Extend `benchmark/` with additional validation tasks and standard-method references
4. Run cross-validation against standard methods to benchmark new LLM backbones

---

## License

MIT — see [LICENSE](LICENSE) for details.
