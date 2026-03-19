import os
import re
import numpy as np
import ray
from pathlib import Path
import json
import sys

from .sandbox import DockerSandbox, FIRE_BENCH_ROOT, SANDBOX_IMAGE

INSTRUCTION_TYPE = os.environ.get("FIRE_BENCH_INSTRUCTION_TYPE", "user")  # "gt" or "user"

from dotenv import load_dotenv
load_dotenv()  # Loads .env file

# ---------------------------------------------------------------------------
# RAGChecker imports — optional; falls back to binary reward if unavailable
# ---------------------------------------------------------------------------
try:
    sys.path.insert(0, str(Path(FIRE_BENCH_ROOT).parent / "FIRE-Bench" / "eval" / "RAGChecker"))
    from ragchecker import RAGResults, RAGChecker
    from ragchecker.metrics import overall_metrics
    _RAGCHECKER_AVAILABLE = True
except Exception:
    _RAGCHECKER_AVAILABLE = False

# Ground-truth answers keyed by task_id, loaded lazily from FIRE-Bench utils.py
_GT_CACHE: dict[str, str] | None = None
_QUERY_CACHE: dict[str, str] | None = None


def _load_gt_and_query() -> tuple[dict, dict]:
    global _GT_CACHE, _QUERY_CACHE
    if _GT_CACHE is not None:
        return _GT_CACHE, _QUERY_CACHE
    utils_path = Path(FIRE_BENCH_ROOT).parent / "FIRE-Bench" / "eval" / "RAGChecker" / "utils.py"
    namespace: dict = {}
    exec(utils_path.read_text(), namespace)
    _GT_CACHE = namespace.get("gt", {})
    _QUERY_CACHE = namespace.get("query", {})
    return _GT_CACHE, _QUERY_CACHE


class FireBenchWorker:
    """
    Ray Actor that manages one DockerSandbox for one paper task episode.

    reset(task_id) → starts a fresh sandbox, returns (instruction_text, info)
    step(action)   → executes action dict, returns (obs, reward, done, info)
    close()        → stops the sandbox
    """

    def __init__(self, worker_id: int, max_steps: int, image: str = SANDBOX_IMAGE):
        self.worker_id = worker_id
        self.max_steps = max_steps
        self.image = image
        self.sandbox: DockerSandbox | None = None
        self.task_id: str | None = None
        self.step_count: int = 0
        self.instruction_gt: str = ""

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, task_id: str) -> tuple[str, dict]:
        # Clean up previous episode
        if self.sandbox is not None:
            try:
                self.sandbox.stop()
            except Exception:
                pass

        self.task_id = task_id
        self.step_count = 0

        # Read instruction
        papers_dir = Path(FIRE_BENCH_ROOT) / "benchmark" / "papers" / task_id
        instruction_path = papers_dir / "instruction" / f"instruction_{INSTRUCTION_TYPE}.txt"
        if not instruction_path.exists():
            instruction_path = papers_dir / "instruction" / "instruction.txt"
        self.instruction_gt_path = papers_dir / "instruction" / "instruction_gt.txt"
        self.instruction_gt = self.instruction_gt_path.read_text() if self.instruction_gt_path.exists() else ""

        instruction_text = instruction_path.read_text()

        # Start Docker sandbox
        self.sandbox = DockerSandbox(task_id, self.worker_id, image=self.image)
        self.sandbox.start()

        return instruction_text, {"task_id": task_id}

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: dict) -> tuple[str, float, bool, dict]:
        """
        action = {
            "type": "bash" | "python" | "write_file" | "finish",
            "content": str,          # bash/python: the command/code; finish: conclusion text
            "path":    str,          # write_file only
        }
        """
        self.step_count += 1
        action_type = action.get("type", "bash")

        if action_type == "finish":
            conclusion = action.get("content", "").strip() or self.sandbox.get_full_log()
            reward = self._compute_reward(conclusion)
            try:
                self.sandbox.stop()
            except Exception:
                pass
            return (
                f"Task finished. Conclusion submitted ({len(conclusion)} chars).",
                reward,
                True,
                {"won": reward > 0.5, "conclusion": conclusion, "task_id": self.task_id},
            )

        # Execute action in sandbox
        obs = self._execute_action(action)

        # Force done at max_steps — run evaluation on whatever was produced
        done = self.step_count >= self.max_steps
        reward = 0.0
        info = {"won": False, "task_id": self.task_id}

        if done:
            conclusion = self.sandbox.get_full_log()
            reward = self._compute_reward(conclusion)
            info["won"] = reward > 5.0
            info["conclusion"] = conclusion
            try:
                self.sandbox.stop()
            except Exception:
                pass

        return obs, reward, done, info

    def _execute_action(self, action: dict) -> str:
        action_type = action.get("type", "bash")
        content = action.get("content", "")

        try:
            if action_type == "bash":
                stdout, stderr, rc = self.sandbox.exec_bash(content)
                obs = f"exit_code={rc}\n"
                if stdout:
                    obs += f"stdout:\n{stdout}\n"
                if stderr:
                    obs += f"stderr:\n{stderr}\n"
                return obs.strip()

            elif action_type == "python":
                stdout, stderr, rc = self.sandbox.exec_python(content)
                obs = f"exit_code={rc}\n"
                if stdout:
                    obs += f"stdout:\n{stdout}\n"
                if stderr:
                    obs += f"stderr:\n{stderr}\n"
                return obs.strip()

            elif action_type == "write_file":
                path = action.get("path", "output.txt")
                msg = self.sandbox.write_file(path, content)
                return msg

            else:
                return f"Unknown action type: {action_type}"

        except Exception as e:
            return f"Error executing {action_type}: {e}"

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, conclusion: str) -> float:
        """
        Compute reward using RAGChecker overall F1 score (0–100 scale → 0.0–1.0).

        The agent's conclusion is compared against the ground-truth answer for
        the current task using RAGChecker precision + recall → F1.  If RAGChecker
        is unavailable or evaluation fails, falls back to 1.0 / 0.0 binary reward.
        """
        if not conclusion.strip():
            return 0.0

        if not _RAGCHECKER_AVAILABLE or not self.task_id:
            return 1.0

        try:
            gt_map, query_map = _load_gt_and_query()
            gt_answer = gt_map.get(self.task_id, "")
            query = query_map.get(self.task_id, self.task_id)
            if not gt_answer:
                return 1.0  # no ground truth available — binary fallback

            rag_input = {
                "results": [
                    {
                        "query_id": "000",
                        "query": query,
                        "gt_answer": gt_answer,
                        "response": conclusion,
                        "retrieved_context": [],
                    }
                ]
            }
            evaluator = RAGChecker(
                extractor_name="openai/gpt-4o-mini",
                checker_name="openai/gpt-4o-mini",
                batch_size_extractor=8,
                batch_size_checker=8,
            )
            rag_results = RAGResults.from_json(json.dumps(rag_input))
            metrics = evaluator.evaluate(rag_results, overall_metrics)
            f1 = metrics.get(overall_metrics, {}).get("f1", 0.0)
            # RAGChecker reports scores on a 0–100 scale; normalize to 0–10
            return float(f1) / 100.0
        except Exception as e:
            # Evaluation error — fall back to binary reward
            return 1.0 if conclusion.strip() else 0.0

    def close(self):
        if self.sandbox is not None:
            try:
                self.sandbox.stop()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Vectorised wrapper — mirrors AppWorldEnvs interface
# ---------------------------------------------------------------------------

class FireBenchEnvs:
    """
    Ray-based distributed wrapper for FireBenchWorker.

    Creates env_num * group_n Ray actors.
    reset() samples env_num tasks, repeats each group_n times.
    step(actions) dispatches one action per actor in parallel.
    """

    def __init__(
        self,
        task_ids: list[str],
        max_steps: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        image: str = SANDBOX_IMAGE,
    ):
        self.task_ids = task_ids
        self.max_steps = max_steps
        self.env_num = env_num
        self.group_n = group_n
        self.num_processes = env_num * group_n
        self.image = image

        if not ray.is_initialized():
            ray.init()

        RemoteWorker = ray.remote(**resources_per_worker)(FireBenchWorker)
        self.workers = [
            RemoteWorker.remote(
                worker_id=i,
                max_steps=max_steps,
                image=image,
            )
            for i in range(self.num_processes)
        ]

    def reset(self) -> tuple[list[str], list[dict]]:
        # Sample env_num distinct tasks, repeat each group_n times
        chosen = np.random.choice(self.task_ids, size=self.env_num, replace=len(self.task_ids) < self.env_num)
        task_id_list = np.repeat(chosen, self.group_n).tolist()

        futures = [w.reset.remote(tid) for w, tid in zip(self.workers, task_id_list)]
        results = ray.get(futures)

        obs_list = [r[0] for r in results]
        info_list = [r[1] for r in results]
        return obs_list, info_list

    def step(self, actions: list[dict]) -> tuple[list[str], list[float], list[bool], list[dict]]:
        assert len(actions) == self.num_processes
        futures = [w.step.remote(a) for w, a in zip(self.workers, actions)]
        results = ray.get(futures)

        obs_list, reward_list, done_list, info_list = zip(*results)
        return list(obs_list), list(reward_list), list(done_list), list(info_list)

    def close(self):
        futures = [w.close.remote() for w in self.workers]
        ray.get(futures)
        for w in self.workers:
            ray.kill(w)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _load_task_ids() -> list[str]:
    papers_dir = Path(FIRE_BENCH_ROOT) / "benchmark" / "papers"
    task_ids = sorted(
        d.name for d in papers_dir.iterdir()
        if d.is_dir() and (d / "instruction" / "instruction_gt.txt").exists()
    )
    if not task_ids:
        raise ValueError(f"No tasks found under {papers_dir}")
    return task_ids


def build_fire_bench_envs(
    task_ids: list[str] | None = None,
    max_steps: int = 50,
    env_num: int = 1,
    group_n: int = 1,
    resources_per_worker: dict | None = None,
    image: str = SANDBOX_IMAGE,
) -> FireBenchEnvs:
    if task_ids is None:
        task_ids = _load_task_ids()
    if resources_per_worker is None:
        resources_per_worker = {"num_cpus": 0.1}

    return FireBenchEnvs(
        task_ids=task_ids,
        max_steps=max_steps,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        image=image,
    )
