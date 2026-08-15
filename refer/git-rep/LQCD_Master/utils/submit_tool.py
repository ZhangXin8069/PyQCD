from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


class SlurmSubmitTool:
    """Submit slurm jobs via tool-style API."""

    name = "slurm_submit"

    def submit(self, script_path: str) -> dict[str, Any]:
        script = Path(script_path)
        command = f"sbatch {script}"
        if not script.exists():
            return {
                "tool": self.name,
                "ok": False,
                "command": command,
                "error": f"Submission script does not exist: {script}",
            }

        if shutil.which("sbatch") is None:
            return {
                "tool": self.name,
                "ok": False,
                "command": command,
                "error": "sbatch was not found in PATH",
            }

        proc = subprocess.run(
            ["sbatch", str(script)],
            check=False,
            capture_output=True,
            text=True,
        )
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        match = re.search(r"Submitted batch job\s+(\d+)", stdout)
        job_id = match.group(1) if match else None

        return {
            "tool": self.name,
            "ok": proc.returncode == 0,
            "command": command,
            "returncode": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "job_id": job_id,
        }
