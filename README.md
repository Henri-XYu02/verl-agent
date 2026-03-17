# verl-agent + FIRE-Bench

This fork of [verl-agent](https://github.com/langfengQ/verl-agent) adds a **FIRE-Bench** environment, enabling RL training of LLM research agents that reproduce experiments from academic papers by writing and executing code in a sandboxed Docker environment.

For the original verl-agent documentation (GiGPO, ALFWorld, WebShop, etc.) see [README_verlagent.md](./README_verlagent.md).


The agent has access to three tools:

| Tool | Description |
|------|-------------|
| `<function=bash><parameter=command>...</parameter></function>` | Run a shell command |
| `<function=python><parameter=code>...</parameter></function>` | Execute a Python code block |
| `<function=write_file><parameter=path>...</parameter><parameter=content>...</parameter></function>` | Write a file to `/workspace/` |
| `<function=finish><parameter=conclusion>...</parameter></function>` | Submit final conclusion |

The agent must enclose all reasoning in `<think>...</think>` before each tool call.

---

## Setup


### 2. Install Python dependencies

I simply reused the verl-agent-webshop environment, but any verl-agent should work.

### 3. Build the Docker sandbox image

The sandbox image provides a clean Ubuntu 22.04 + CUDA 12.1 environment with common ML libraries pre-installed.

I simply reuse the Dockerfile in FireBench openhands.

```bash
docker build -t fire-bench-sandbox:latest ./
```

Pre-installed packages: `numpy pandas datasets transformers openai anthropic scikit-learn matplotlib scipy python-dotenv torch torchvision`

### 4. Prepare the dataset

```bash
python tools/build_firebench_dataset.py --output_dir ~/data/fire_bench
# Creates train.parquet and val.parquet
```

### 5. Set environment variables

```bash
export FIRE_BENCH_ROOT=/home/xinle/FIRE-Bench
export FIRE_BENCH_IMAGE=fire-bench-sandbox:latest
export FIRE_BENCH_INSTRUCTION_TYPE=user   # or "gt" for ground-truth instructions
export OPENAI_API_KEY=...                 # passed into the sandbox
export HF_TOKEN=...                       # for HuggingFace model downloads
```

---

## Running

### Smoke test (no GPU required)

Verifies the Docker sandbox works: bash, python, write_file (relative and absolute paths), and dotenv import.

```bash
python tools/smoke_test_fire_bench.py
```

### RL Training — Qwen2.5-7B-Instruct (3× A100 80GB)

```bash
bash examples/gigpo_trainer/run_firebench.sh
```

Key hyperparameters (edit at top of script):

| Parameter | Value | Notes |
|-----------|-------|-------|
| `data.max_prompt_length` | 8192 | Prompt truncated left if exceeded |
| `data.max_response_length` | 2048 | Max tokens per LLM response |
| `env.max_steps` | 20 | Max sandbox steps per episode |
| `env.rollout.n` | 2 | Group size for GiGPO |
| `trainer.n_gpus_per_node` | 3 | GPUs used |

### RL Training — Qwen3-Coder-30B-A3B-Instruct (3× A100 80GB)

Haven't been tested it.
```bash
bash examples/gigpo_trainer/run_firebench_qwen3_coder_30b.sh
```

### VSCode Debugger

Two launch configs are provided in `.vscode/launch.json`:

- **FireBench trainer (debug) - Qwen2.5-7B** — uses `RAY_LOCAL_MODE=1` to run everything in a single process so breakpoints work in the rollout loop, env manager, etc.
- **FireBench trainer (debug) - Qwen3-Coder-30B-A3B** — same, for the larger model.

---

## Architecture

```
agent_system/
  environments/
    env_package/fire_bench/
      envs.py          # FireBenchWorker (Ray actor) + FireBenchEnvs
      sandbox.py       # DockerSandbox: exec_bash / exec_python / write_file
      projection.py    # Parse LLM output → action dict
    env_manager.py     # FireBenchEnvironmentManager (reset/step/build_text_obs)
    prompts/
      fire_bench.py    # System prompt + few-shot examples + templates
  multi_turn_rollout/
    rollout_loop.py    # vanilla_multi_turn_loop → log_trajectories
```

Each episode:
1. `FireBenchWorker.reset(task_id)` starts a Docker container, copies `benchmark/papers/<task_id>/data/` into `/workspace/`
2. The rollout loop calls the LLM, parses the tool call via `projection.py`, and dispatches to `DockerSandbox`
3. At episode end (max steps or `finish` tool), trajectories are logged to `{default_local_dir}/traj_logs/{experiment_name}/`

---

## Trajectory Logs

After each rollout, one `.txt` file per environment is written to:

```
<default_local_dir>/traj_logs/<experiment_name>/
  step00005_20260317_120000_llm_racial_bias_in_medicine_steps20.txt
```

Each file contains the full task instruction, followed by every raw LLM action and sandbox observation, useful for debugging agent behavior.

Still collecting those txt file to check if there is something wrong with firebench env impl.

---

## Known Issues / Notes

- **Disk space**: The `/workspace` volume is carved from the host's `/data` partition. If it fills up (`df -h /workspace` shows 100%), the sandbox cannot write files. Check disk usage before long runs.
- **`sudo` not available** inside the sandbox container — the agent runs as root so `sudo` is not needed, but the command itself is absent.


Previous Problem: `dotenv` not in docker image and `write_file` permission denied. Just applied fix but didn't verify if they work.
- **Absolute paths in `write_file`**: The implementation strips `/workspace/` prefixes so both `foo.py` and `/workspace/foo.py` work correctly.
- **`python-dotenv`**: Pre-installed in the Docker image. Agents can use `from dotenv import load_dotenv` directly.
