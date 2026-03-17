import os
import shutil
import subprocess
import time
import random
import tempfile
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  # Loads .env file

FIRE_BENCH_ROOT = os.environ.get("FIRE_BENCH_ROOT", "/home/xinle/FIRE-Bench")
SANDBOX_IMAGE = os.environ.get("FIRE_BENCH_IMAGE", "fire-bench-sandbox:latest")
MAX_OUTPUT_LEN = 8000  # chars — truncate long stdout to avoid context explosion


class DockerSandbox:
    """
    Manages one Docker container per episode.

    Lifecycle:
        sandbox = DockerSandbox(task_id, worker_id)
        sandbox.start()
        stdout, stderr, rc = sandbox.exec_bash("ls /workspace")
        sandbox.stop()
    """

    def __init__(self, task_id: str, worker_id: int, image: str = SANDBOX_IMAGE):
        self.task_id = task_id
        self.image = image

        rd = random.randint(10000, 99999)
        timestamp = time.strftime("%Y%m%d%H%M%S")
        self.container_name = f"fire-bench-{worker_id}-{timestamp}-{rd}"

        # Workspace directory on the host, mounted into /workspace in the container
        runs_dir = Path(FIRE_BENCH_ROOT) / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        self.sandbox_dir = runs_dir / f"{task_id}_{timestamp}_{rd}"

        self._log_parts: list[str] = []
        self._running = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _prepare_workspace(self):
        """Copy task data and utils into the sandbox directory."""
        papers_dir = Path(FIRE_BENCH_ROOT) / "benchmark" / "papers" / self.task_id

        # Copy contents of data/ directly into sandbox root (matching original run.py behaviour),
        # so files are available at /workspace/<file> rather than /workspace/data/<file>.
        data_src = papers_dir / "data"
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        if data_src.exists():
            shutil.copytree(data_src, self.sandbox_dir, dirs_exist_ok=True)


        # Copy utils/
        utils_src = Path(FIRE_BENCH_ROOT) / "utils"
        if utils_src.exists():
            shutil.copytree(utils_src, self.sandbox_dir / "utils")

        # Write .env with API keys
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", os.environ.get("CLAUDE_API_KEY", ""))
        google_key = os.environ.get("GOOGLE_API_KEY", "")
        hf_token = os.environ.get("HF_TOKEN", "")
        with open(self.sandbox_dir / ".env", "w") as f:
            f.write(f"OPENAI_API_KEY={openai_key}\n")
            f.write(f"ANTHROPIC_API_KEY={anthropic_key}\n")
            f.write(f"CLAUDE_API_KEY={anthropic_key}\n")
            f.write(f"GOOGLE_API_KEY={google_key}\n")
            f.write(f"HF_TOKEN={hf_token}\n")

    def start(self):
        """Prepare workspace and start the Docker container."""
        self._prepare_workspace()

        cmd = [
            "docker", "run", "-d",
            "--name", self.container_name,
            "-v", f"{self.sandbox_dir}:/workspace",
            "-e", f"OPENAI_API_KEY={os.environ.get('OPENAI_API_KEY', '')}",
            "-e", f"ANTHROPIC_API_KEY={os.environ.get('ANTHROPIC_API_KEY', os.environ.get('CLAUDE_API_KEY', ''))}",
            "-e", f"GOOGLE_API_KEY={os.environ.get('GOOGLE_API_KEY', '')}",
            "-e", f"HF_TOKEN={os.environ.get('HF_TOKEN', '')}",
            self.image,
            "sleep", "infinity",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to start Docker container: {result.stderr.strip()}"
            )
        self._running = True

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def exec_bash(self, cmd: str, timeout: int = 120) -> tuple[str, str, int]:
        """Run a shell command inside the container. Returns (stdout, stderr, exit_code)."""
        if not self._running:
            raise RuntimeError("Sandbox not started. Call start() first.")

        result = subprocess.run(
            ["docker", "exec", self.container_name, "bash", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = _truncate(result.stdout, MAX_OUTPUT_LEN)
        stderr = _truncate(result.stderr, MAX_OUTPUT_LEN // 2)
        self._log_parts.append(f"$ {cmd}\n{stdout}{stderr}")
        return stdout, stderr, result.returncode

    def exec_python(self, code: str, timeout: int = 120) -> tuple[str, str, int]:
        """Write code to a temp file and run it inside the container."""
        # Write code to a temp file visible to the container via the shared volume
        script_path = self.sandbox_dir / "_tmp_script.py"
        script_path.write_text(code)
        return self.exec_bash(f"cd /workspace && python3 /workspace/_tmp_script.py", timeout=timeout)

    def write_file(self, path: str, content: str) -> str:
        """Write content to /workspace/{path} inside the container."""
        # Strip leading /workspace/ or / so that absolute paths from the LLM
        # (e.g. "/workspace/foo.py" or "/foo.py") are resolved correctly.
        rel_path = path.lstrip("/")
        if rel_path.startswith("workspace/"):
            rel_path = rel_path[len("workspace/"):]
        abs_path = self.sandbox_dir / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(content)
        return f"Written: /workspace/{rel_path}"

    def get_full_log(self) -> str:
        """Return the concatenated stdout of all exec calls so far."""
        return "\n".join(self._log_parts)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def stop(self):
        """Stop and remove the Docker container."""
        if not self._running:
            return
        try:
            subprocess.run(
                ["docker", "stop", self.container_name],
                capture_output=True, timeout=30,
            )
            subprocess.run(
                ["docker", "rm", self.container_name],
                capture_output=True, timeout=30,
            )
        except Exception:
            pass
        self._running = False


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return f"...[truncated {len(text) - max_len} chars]...\n" + text[-max_len:]
