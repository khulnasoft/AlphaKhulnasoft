# AlphaKhulnasoft v2

## Overview
AlphaKhulnasoft is an advanced AI Code Fixer and Competitive Programming engine. It uses "Flow Engineering" to iteratively repair broken code, moving beyond simple one-shot prompting.

## Core Components

### `alpha_repair.py`
The core engine driving the repair loop: `Analyze → Plan → Generate → Test → Root Cause → Fix`. Now integrated with real LLM providers.

### `llm.py`
Flexible LLM wrapper using `litellm`. Supports OpenAI, Anthropic, and other providers via environment variables.

### `data_loader.py`
Ingestion script for datasets like CodeContests and RealWorldBugs. Includes local loading and mock data support.

### `evaluator.py`
The "Leaderboard" engine. Calculates Pass@k, Efficiency Scores, and iteration depths for comparative benchmarking.

### `sandbox.py`
A production-grade execution engine. It runs generated code in isolated subprocesses, enforces time limits, and captures standard I/O for precise feedback loops.

### `visualizer.py`
The "Proof" engine. Generates research-grade charts (Repair Trajectory and Efficiency Matrix) to visualize system performance.

### `dataset_gen.py`
The "Challenge" engine. Uses an LLM to bootstrap a "Golden Dataset" of hard, competitive programming problems to stress-test the repair loop.

## Setup

1. Install dependencies:
   ```bash
   uv sync
   ```
2. Configure environment:
   ```bash
   cp .env.example .env
   # Add your API keys to .env
   ```

## Usage

Run the full "Go-Live" pipeline:
```bash
# 1. Generate the Challenge (The Golden Dataset)
uv run python -m alphakhulnasoft.dataset_gen

# 2. Run the Gauntlet (The Benchmark)
uv run python -m alphakhulnasoft.benchmark data/hard_mode.jsonl

# 3. Generate the Proof (The Visualization)
uv run python -m alphakhulnasoft.visualizer results_YYYYMMDD_HHMMSS.json
```

Or run the core loop directly in Python:
```python
from alphakhulnasoft.alpha_repair import AlphaRepairAgent

agent = AlphaRepairAgent(model_name="gpt-4o")
problem = "Write a function to double an integer, but return 0 for negatives."
result = agent.run_flow(problem)

print(result['metrics'])
```

## Next Steps
- Implement Secure Sandbox (Docker/nsjail) for code execution.
- Add support for Multi-Turn Reasoning in `alpha_repair.py`.
- Integrate with Hugging Face for model fine-tuning data collection.
