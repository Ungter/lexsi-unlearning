"""Best-effort sandbox for executing model-written Python (Phase-6 attack 12, "tool loophole").

Layers (Linux; the Vast containers are disposable and hold no data beyond the task):
  * child python with an empty environment (no tokens), cwd = a fresh temp dir, wall-clock timeout, new session
  * rlimits: CPU 10 s, address space 3 GiB, file size 1 MiB
  * network namespace without interfaces via "unshare -rn" when the container allows it
  * an audit hook that blocks sockets, subprocesses, exec/fork, ctypes, file writes outside the temp dir, and file
    reads outside the Python installation (so /proc, /root and the project directory are unreadable)
Not a security boundary against a determined attacker; the code comes from a 1.7B model asked to differentiate.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile

WRAPPER = r'''
import os, sys
tmp = sys.argv[2]
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_AS, (3 << 30, 3 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
except ImportError:
    pass
import sympy  # noqa: F401  (pre-import so the hook does not have to allow its whole tree lazily)
tmp = os.path.realpath(tmp)
READ_OK = tuple({os.path.realpath(p) for p in (sys.prefix, sys.exec_prefix, sys.base_prefix, os.path.dirname(os.__file__),
                os.path.dirname(os.path.dirname(sympy.__file__)), "/usr", "/lib", "/lib64", "/etc", "/dev/null",
                "/dev/urandom", "/dev/random")})
BLOCK = ("socket.", "subprocess.", "os.system", "os.exec", "os.fork", "os.posix_spawn", "os.spawn", "ctypes.", "os.kill",
         "os.putenv", "os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.chmod", "os.chown", "os.symlink", "os.link",
         "os.truncate", "shutil.", "pty.", "fcntl.", "webbrowser.")  # os.listdir stays allowed: import machinery needs it
WRITE = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC

def hook(event, args):
    if event == "open":
        path = args[0]
        if isinstance(path, int):
            return
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        real = os.path.realpath(path)
        if real == tmp or real.startswith(tmp + os.sep):
            return
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        if flags & WRITE or (len(args) > 1 and isinstance(args[1], str) and any(c in args[1] for c in "wax+")):
            raise PermissionError("sandbox: write outside temp dir: " + real)
        if not real.startswith(READ_OK):
            raise PermissionError("sandbox: read outside python install: " + real)
        return
    if event.startswith(BLOCK):
        raise PermissionError("sandbox: blocked " + event)

code = open(sys.argv[1], encoding="utf-8").read()
sys.addaudithook(hook)
exec(compile(code, "model_code.py", "exec"), {"__name__": "__main__"})
'''

_UNSHARE: bool | None = None


def _can_unshare() -> bool:
    global _UNSHARE
    if _UNSHARE is None:
        exe = shutil.which("unshare")
        try:
            _UNSHARE = bool(exe) and subprocess.run([exe, "-rn", "true"], capture_output=True, timeout=5).returncode == 0
        except Exception:  # noqa: BLE001
            _UNSHARE = False
    return _UNSHARE


def run_python(code: str, timeout: float = 12.0, max_out: int = 10_000) -> dict:
    """Run code; returns {ok, stdout, stderr, returncode, timed_out, net_isolated}."""
    with tempfile.TemporaryDirectory(prefix="lfm3u-sbx-") as tmp:
        script, wrapper = os.path.join(tmp, "model_code.py"), os.path.join(tmp, "_wrapper.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(code)
        with open(wrapper, "w", encoding="utf-8") as f:
            f.write(WRAPPER)
        cmd = [sys.executable, "-I", wrapper, script, tmp]
        iso = _can_unshare() and os.name == "posix"
        if iso:
            cmd = ["unshare", "-rn", *cmd]
        env = {"PATH": "/usr/bin:/bin", "HOME": tmp, "OMP_NUM_THREADS": "1", "PYTHONHASHSEED": "0", "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
        try:
            p = subprocess.run(cmd, cwd=tmp, env=env, capture_output=True, text=True, timeout=timeout,
                               start_new_session=os.name == "posix")
            return {"ok": p.returncode == 0, "stdout": p.stdout[:max_out], "stderr": p.stderr[-2000:], "returncode": p.returncode,
                    "timed_out": False, "net_isolated": iso}
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"")
            out = out.decode("utf-8", "replace") if isinstance(out, bytes) else out
            return {"ok": False, "stdout": out[:max_out], "stderr": "timeout", "returncode": None, "timed_out": True, "net_isolated": iso}


_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S | re.I)


def extract_code(text: str) -> str:
    """First fenced code block; otherwise the text itself minus any <think> block."""
    body = text.rsplit("</think>", 1)[-1]
    m = _FENCE.search(body)
    return (m.group(1) if m else body).strip()
