import re
from typing import List


def fire_bench_projection(actions: List[str]):
    """
    Parse LLM outputs using SkyRL-style function-call XML format:

      <function=bash><parameter=command>ls /workspace</parameter></function>
      <function=python><parameter=code>print("hi")</parameter></function>
      <function=write_file><parameter=path>out.py</parameter><parameter=content>...</parameter></function>
      <function=finish><parameter=conclusion>...</parameter></function>

    Falls back to accepting legacy <bash>...</bash> / <python>...</python> /
    <finish>...</finish> tags for compatibility.

    Returns:
        parsed_actions: List[dict]  — each dict has 'type' + relevant keys
        valids:         List[int]   — 1 if the action parsed cleanly, 0 otherwise
    """
    parsed_actions = []
    valids = []

    for raw in actions:
        action, valid = _parse_single(raw)
        parsed_actions.append(action)
        valids.append(valid)

    return parsed_actions, valids


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_FUNC_RE = re.compile(
    r"<function=(\w+)>(.*?)</function>",
    re.DOTALL,
)
_PARAM_RE = re.compile(
    r"<parameter=(\w+)>(.*?)</parameter>",
    re.DOTALL,
)

# Legacy tag patterns
_LEGACY = {
    "bash":       re.compile(r"<bash>(.*?)</bash>", re.DOTALL),
    "python":     re.compile(r"<python>(.*?)</python>", re.DOTALL),
    "write_file": re.compile(r'<write_file\s+path="([^"]*)">(.*?)</write_file>', re.DOTALL),
    "finish":     re.compile(r"<finish>(.*?)</finish>", re.DOTALL),
}


def _parse_single(raw: str):
    # --- Try function-call XML format first ---
    func_match = _FUNC_RE.search(raw)
    if func_match:
        func_name = func_match.group(1).lower()
        params_str = func_match.group(2)
        params = {m.group(1): m.group(2) for m in _PARAM_RE.finditer(params_str)}

        if func_name == "bash":
            cmd = params.get("command", "").strip()
            if cmd:
                return {"type": "bash", "content": cmd}, 1
        elif func_name == "python":
            code = params.get("code", "").strip()
            if code:
                return {"type": "python", "content": code}, 1
        elif func_name == "write_file":
            path = params.get("path", "output.txt").strip()
            content = params.get("content", "")
            return {"type": "write_file", "path": path, "content": content}, 1
        elif func_name == "finish":
            conclusion = params.get("conclusion", "").strip()
            return {"type": "finish", "content": conclusion}, 1
        else:
            # Unknown function name — treat as invalid bash fallback
            return {"type": "bash", "content": raw[-200:]}, 0

    # --- Try legacy XML tags ---
    for tag, pattern in _LEGACY.items():
        m = pattern.search(raw)
        if m:
            if tag == "write_file":
                return {"type": "write_file", "path": m.group(1), "content": m.group(2)}, 1
            else:
                return {"type": tag, "content": m.group(1).strip()}, 1

    # --- Nothing matched — invalid ---
    return {"type": "bash", "content": raw[-200:]}, 0
