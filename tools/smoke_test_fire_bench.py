"""
Smoke test for the FireBench environment.

Runs one episode on a single task using FireBenchWorker directly
(no Ray, no verl trainer) — step through with the VSCode debugger.

Usage:
    python tools/smoke_test_fire_bench.py

VSCode launch config (add to .vscode/launch.json):
    {
        "name": "FireBench smoke test",
        "type": "debugpy",
        "request": "launch",
        "program": "${workspaceFolder}/tools/smoke_test_fire_bench.py",
        "console": "integratedTerminal",
        "cwd": "${workspaceFolder}"
    }
"""

import sys
import os

# Make sure the repo root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_system.environments.env_package.fire_bench.envs import FireBenchWorker
from agent_system.environments.env_package.fire_bench.projection import fire_bench_projection

# ── Config ──────────────────────────────────────────────────────────────────
TASK_ID   = "activation_control"   # one real FIRE-Bench task
MAX_STEPS = 10                     # enough for 8 scripted steps

# A short scripted episode to exercise each action type:
SCRIPTED_ACTIONS = [
    # 1. bash: check what's in /workspace
    "<function=bash><parameter=command>ls /workspace</parameter></function>",
    # 2. python: run a trivial computation
    "<function=python><parameter=code>print('hello from python:', 2 + 2)</parameter></function>",
    # 3. write_file with relative path (original behaviour)
    "<function=write_file><parameter=path>hello.txt</parameter><parameter=content>smoke test content\n</parameter></function>",
    # 4. bash: verify the relative-path file was written
    "<function=bash><parameter=command>cat /workspace/hello.txt</parameter></function>",
    # 5. write_file with absolute /workspace/ path — this was the bug causing Permission denied
    "<function=write_file><parameter=path>/workspace/abs_path_test.py</parameter><parameter=content>print('absolute path write works')\n</parameter></function>",
    # 6. bash: verify the absolute-path file and run it
    "<function=bash><parameter=command>cat /workspace/abs_path_test.py && python3 /workspace/abs_path_test.py</parameter></function>",
    # 7. python: verify python-dotenv is installed in the Docker image
    "<function=python><parameter=code>from dotenv import load_dotenv; print('python-dotenv import OK')</parameter></function>",
    # 8. finish: submit a conclusion
    "<function=finish><parameter=conclusion>Smoke test passed. write_file (relative + absolute) and python-dotenv all work.</parameter></function>",
]
# ────────────────────────────────────────────────────────────────────────────


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    section("1. Create worker")
    worker = FireBenchWorker(worker_id=0, max_steps=MAX_STEPS)
    print(f"  worker_id={worker.worker_id}, max_steps={worker.max_steps}")

    # ── reset ────────────────────────────────────────────────────────────────
    section("2. Reset worker (starts Docker container)")
    instruction, info = worker.reset(TASK_ID)
    print(f"  task_id  : {info['task_id']}")
    print(f"  instruction (first 300 chars):\n{instruction[:300]}")

    # ── scripted steps ───────────────────────────────────────────────────────
    for step_idx, raw_action in enumerate(SCRIPTED_ACTIONS):
        section(f"3.{step_idx+1}  Step {step_idx+1} — parse + execute")

        # Parse the LLM-style action string exactly as env_manager does
        parsed_actions, valids = fire_bench_projection([raw_action])
        action = parsed_actions[0]
        valid  = valids[0]

        print(f"  raw     : {raw_action[:80]}")
        print(f"  parsed  : {action}")
        print(f"  valid   : {valid}")

        obs, reward, done, info = worker.step(action)

        print(f"  obs     : {obs[:300]}")
        print(f"  reward  : {reward}")
        print(f"  done    : {done}")
        print(f"  info    : {info}")

        if done:
            print("\n  Episode finished early (finish tool or max_steps).")
            break

    # ── cleanup (in case finish was not reached) ─────────────────────────────
    section("4. Cleanup")
    worker.close()
    print("  Worker closed. Docker container removed.")



if __name__ == "__main__":
    main()
