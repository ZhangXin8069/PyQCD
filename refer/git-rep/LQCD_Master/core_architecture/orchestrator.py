from __future__ import annotations

import getpass
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .executor import ExecutorAgent
from utils.io_utils import read_text, write_json, write_text
from .planner import PlannerAgent


class WorkflowOrchestrator:
    _TERMINAL_SLURM_STATES = {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
    }

    def __init__(
        self,
        *,
        task: str,
        run_dir: Path,
        llm_client: Any,
        tool_client: Any,
        submit_tool: Any,
        test_mode: bool = False,
        non_interactive: bool = False,
        max_revision_rounds: int = 3,
    ):
        self.task = task
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.test_mode = test_mode
        self.non_interactive = non_interactive
        self.max_revision_rounds = max_revision_rounds

        self.planner = PlannerAgent(
            llm_client=llm_client,
            tool_client=tool_client,
            run_dir=run_dir,
        )
        self.executor = ExecutorAgent(
            llm_client=llm_client,
            submit_tool=submit_tool,
            run_dir=run_dir,
            tool_client=tool_client,
        )

        self.state: dict[str, Any] = {
            "mode": "test" if test_mode else "normal",
            "task": task,
            "run_dir": str(run_dir.resolve()),
            "started_at": datetime.now().isoformat(),
            "planner": {},
            "executor": {},
            "test_submission": {},
            "full_submission": {},
        }
        self._save_state()

    def run(self) -> dict[str, Any]:
        print("==================== [ Planner ] ====================")
        planner_output, planner_version = self._run_planner_stage()
        print()
        print("==================== [ Executor ] ====================")
        bundle, exec_version = self._run_executor_stage(planner_output)
        if self.test_mode:
            self.state["test_submission"] = {
                "approved": False,
                "result": {"skipped": True, "reason": "test_mode"},
            }
            self.state["full_submission"] = {
                "approved": False,
                "result": {"skipped": True, "reason": "not_submitted_in_main_workflow"},
            }
            self._save_state()
        else:
            self._run_test_submission_stage(bundle, exec_version, planner_output)

        self.state["finished_at"] = datetime.now().isoformat()
        self.state["status"] = "completed"
        self._save_state()
        return self.state

    def _run_planner_stage(self) -> tuple[dict[str, Any], int]:
        planner_output = self.planner.run(self.task)
        planner_version = 1
        self.state["planner"] = {
            "version": planner_version,
            "output": planner_output,
            "checkpoints": [],
            "debug": self.planner.debug_state(),
        }
        self._save_state()

        while True:
            approved, feedback = self._checkpoint_planner(planner_output, planner_version)
            self.state["planner"]["checkpoints"].append(
                {"version": planner_version, "approved": approved, "feedback": feedback}
            )
            self._save_state()

            if approved:
                break

            planner_version += 1
            if planner_version > self.max_revision_rounds + 1:
                raise RuntimeError("Planner rewrite exceeded max_revision_rounds")

            planner_output = self.planner.rewrite_with_feedback(
                task=self.task,
                current=planner_output,
                feedback=feedback,
                version=planner_version,
            )
            self.state["planner"]["version"] = planner_version
            self.state["planner"]["output"] = planner_output
            self._save_state()

        return planner_output, planner_version

    def _run_executor_stage(self, planner_output: dict[str, Any]) -> tuple[dict[str, Any], int]:
        exec_version = 1
        try:
            bundle = self.executor.generate_bundle(
                task=self.task,
                planner_output=planner_output,
                version=exec_version,
            )
        except Exception as e:
            self.state["executor"] = {
                "version": exec_version,
                "error": f"{type(e).__name__}: {e}",
                "status": "failed",
            }
            self._save_state()
            raise
        self.state["executor"] = {
            "version": exec_version,
            "bundle": bundle,
            "critique": bundle.get("critique", {}),
            "mode": "generate_only",
            "checkpoints": [],
        }
        self._save_state()

        while True:
            approved, feedback = self._checkpoint_executor(bundle, exec_version)
            self.state["executor"]["checkpoints"].append(
                {"version": exec_version, "approved": approved, "feedback": feedback}
            )
            self._save_state()

            if approved:
                break

            current_exec_dir = self.executor._version_dir(exec_version)
            write_text(current_exec_dir / "feedback.txt", feedback)
            exec_version += 1
            if exec_version > self.max_revision_rounds + 1:
                raise RuntimeError("Executor revise loop exceeded max_revision_rounds")

            next_exec_dir = self.executor._version_dir(exec_version)
            write_text(next_exec_dir / "feedback.txt", feedback)
            try:
                bundle = self.executor.generate_bundle(
                    task=self.task,
                    planner_output=planner_output,
                    version=exec_version,
                    feedback=feedback,
                    previous_bundle=bundle,
                )
            except Exception as e:
                self.state["executor"]["version"] = exec_version
                self.state["executor"]["error"] = f"{type(e).__name__}: {e}"
                self.state["executor"]["status"] = "failed"
                self._save_state()
                raise

            self.state["executor"]["version"] = exec_version
            self.state["executor"]["bundle"] = bundle
            self.state["executor"]["critique"] = bundle.get("critique", {})
            self.state["executor"]["mode"] = "generate_only"
            self._save_state()

        return bundle, exec_version

    def _run_test_submission_stage(
        self,
        bundle: dict[str, Any],
        version: int,
        planner_output: dict[str, Any],
    ) -> None:
        current_bundle = bundle
        current_version = version
        max_dynamic_test_rounds = self._resolve_executor_dynamic_test_rounds()
        attempts: list[dict[str, Any]] = []
        dynamic_test_round = 0

        while True:
            version_dir = self.executor._version_dir(current_version)
            result = self.executor.submit_test(bundle=current_bundle, version=current_version)
            print("[Queue] Waiting for test job output...")
            monitor_status = self._wait_for_test_job_output(
                bundle=current_bundle,
                submitted_job_id=result.get("job_id"),
            )
            write_json(version_dir / "test_monitor.json", monitor_status)
            queue_status = self._inspect_slurm_queue(
                bundle=current_bundle,
                submitted_job_id=result.get("job_id"),
            )
            write_json(version_dir / "queue_status.json", queue_status)

            attempt = {
                "version": current_version,
                "submit_result": result,
                "monitor_status": monitor_status,
                "queue_status": queue_status,
            }
            attempts.append(attempt)

            self.state["test_submission"] = {
                "approved": True,
                "version": current_version,
                "result": result,
                "monitor_status": monitor_status,
                "queue_status": queue_status,
                "attempts": attempts,
            }
            self.state["full_submission"] = {
                "approved": False,
                "result": {"skipped": True, "reason": "main_workflow_stops_after_test_submission"},
            }
            self._save_state()

            if not self._test_monitor_has_error(monitor_status):
                return

            if dynamic_test_round >= max_dynamic_test_rounds:
                print("[Executor] Test stderr detected, but executor_dynamic_test_rounds has been exhausted")
                return

            dynamic_test_round += 1
            feedback = self._build_test_failure_feedback(monitor_status)
            write_text(version_dir / "test_feedback.txt", feedback)
            current_version += 1
            next_exec_dir = self.executor._version_dir(current_version)
            write_text(next_exec_dir / "test_feedback.txt", feedback)
            try:
                current_bundle = self.executor.generate_bundle(
                    task=self.task,
                    planner_output=planner_output,
                    version=current_version,
                    feedback=feedback,
                    previous_bundle=current_bundle,
                )
            except Exception as e:
                self.state["executor"]["version"] = current_version
                self.state["executor"]["error"] = f"{type(e).__name__}: {e}"
                self.state["executor"]["status"] = "failed"
                self._save_state()
                raise
            self.state["executor"]["version"] = current_version
            self.state["executor"]["bundle"] = current_bundle
            self.state["executor"]["critique"] = current_bundle.get("critique", {})
            self.state["executor"]["mode"] = "dynamic_test_rewrite"
            self._save_state()

    def _inspect_slurm_queue(self, bundle: dict[str, Any], submitted_job_id: str | None) -> dict[str, Any]:
        user = getpass.getuser().strip()
        command = ["squeue", "-u", user]
        output_hint = self._extract_output_hint(bundle)
        if not user:
            status = {
                "ok": False,
                "command": "squeue -u $USER",
                "error": "Unable to determine the current user",
            }
            self._print_queue_progress(status)
            return status

        if shutil.which("squeue") is None:
            status = {
                "ok": False,
                "command": f"squeue -u {user}",
                "error": "squeue was not found in PATH",
            }
            self._print_queue_progress(status)
            return status

        proc = subprocess.run(command, check=False, capture_output=True, text=True)
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        lines = [line for line in stdout.splitlines() if line.strip()]
        parsed_jobs = self._parse_squeue_output(lines)
        matched_job = self._find_job_in_queue(parsed_jobs, submitted_job_id)
        status = {
            "ok": proc.returncode == 0,
            "command": " ".join(command),
            "returncode": proc.returncode,
            "stderr": stderr,
            "user": user,
            "job_count": len(parsed_jobs),
            "job_found": matched_job is not None,
            "submitted_job_id": submitted_job_id,
            "squeue_raw": {
                "stdout": stdout,
                "stderr": stderr,
            },
            "squeue_jobs": parsed_jobs,
            "matched_job": matched_job,
            "output_hint": output_hint,
        }
        if matched_job and shutil.which("scontrol") is not None:
            scontrol_status = self._inspect_slurm_job(job_id=str(matched_job.get("job_id", "")))
            status["scontrol"] = scontrol_status
            status["summary"] = self._build_queue_summary(
                matched_job=matched_job,
                scontrol_status=scontrol_status,
                output_hint=output_hint,
            )
        else:
            status["summary"] = self._build_queue_summary(
                matched_job=matched_job,
                scontrol_status=None,
                output_hint=output_hint,
            )
        self._print_queue_progress(status)
        return status

    def _inspect_slurm_job(self, job_id: str) -> dict[str, Any]:
        command = ["scontrol", "show", "job", job_id]
        proc = subprocess.run(command, check=False, capture_output=True, text=True)
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        parsed = self._parse_scontrol_output(stdout)
        return {
            "ok": proc.returncode == 0,
            "command": " ".join(command),
            "returncode": proc.returncode,
            "raw": {
                "stdout": stdout,
                "stderr": stderr,
            },
            "parsed": parsed,
        }

    def _extract_output_hint(self, bundle: dict[str, Any]) -> dict[str, Any]:
        script_path = Path(str(bundle.get("test_submit_path", "")))
        main_program_path = Path(str(bundle.get("main_program_path", "")))
        script_text = read_text(script_path)
        out_match = re.search(r"--out\s+([^\s]+)", script_text)
        relative_out = out_match.group(1) if out_match else ""
        script_dir = script_path.parent if script_path.exists() else self.run_dir
        output_path = ""
        if relative_out:
            output_path = str((script_dir / relative_out).resolve())
        return {
            "submit_script_path": str(script_path),
            "main_program_path": str(main_program_path),
            "relative_output_path": relative_out,
            "absolute_output_path": output_path,
        }

    def _resolve_executor_dynamic_test_rounds(self) -> int:
        workflow_cfg = self.executor.submit_config.get("workflow", {})
        if not isinstance(workflow_cfg, dict):
            return 0
        raw = workflow_cfg.get("executor_dynamic_test_rounds")
        try:
            if raw is None:
                return 0
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 0

    def _wait_for_test_job_output(
        self,
        *,
        bundle: dict[str, Any],
        submitted_job_id: str | None,
        poll_interval_seconds: int = 5,
        timeout_seconds: int = 1800,
    ) -> dict[str, Any]:
        log_paths = self._resolve_test_log_paths(bundle=bundle, submitted_job_id=submitted_job_id)
        stdout_path = log_paths.get("stdout_path") or ""
        stderr_path = log_paths.get("stderr_path") or ""
        if not submitted_job_id:
            return {
                "ok": False,
                "job_id": submitted_job_id,
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
                "stdout_has_output": False,
                "stderr_has_output": False,
                "stdout_excerpt": "",
                "stderr_excerpt": "",
                "error": "missing_submitted_job_id",
            }
        if not stdout_path and not stderr_path:
            return {
                "ok": False,
                "job_id": submitted_job_id,
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
                "stdout_has_output": False,
                "stderr_has_output": False,
                "stdout_excerpt": "",
                "stderr_excerpt": "",
                "error": "unable_to_resolve_test_log_paths",
            }
        started_at = time.time()

        while time.time() - started_at < timeout_seconds:
            stdout_text = read_text(Path(stdout_path)) if stdout_path else ""
            stderr_text = read_text(Path(stderr_path)) if stderr_path else ""
            slurm_status = self._query_test_job_status(submitted_job_id)
            job_state = str(slurm_status.get("job_state") or "").strip().upper()
            exit_code = str(slurm_status.get("exit_code") or "").strip()
            terminal = job_state in self._TERMINAL_SLURM_STATES
            failed = self._slurm_status_indicates_failure(job_state=job_state, exit_code=exit_code)

            if terminal:
                return {
                    "ok": not failed,
                    "job_id": submitted_job_id,
                    "stdout_path": stdout_path,
                    "stderr_path": stderr_path,
                    "stdout_exists": bool(stdout_path and Path(stdout_path).exists()),
                    "stderr_exists": bool(stderr_path and Path(stderr_path).exists()),
                    "stdout_has_output": bool(stdout_text.strip()),
                    "stderr_has_output": bool(stderr_text.strip()),
                    "stdout_excerpt": self._truncate_text_tail(stdout_text),
                    "stderr_excerpt": self._truncate_text_tail(stderr_text),
                    "job_state": job_state or None,
                    "exit_code": exit_code or None,
                    "slurm_status": slurm_status,
                }
            time.sleep(poll_interval_seconds)

        return {
            "ok": False,
            "job_id": submitted_job_id,
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "stdout_exists": bool(stdout_path and Path(stdout_path).exists()),
            "stderr_exists": bool(stderr_path and Path(stderr_path).exists()),
            "stdout_has_output": False,
            "stderr_has_output": False,
            "stdout_excerpt": self._truncate_text_tail(read_text(Path(stdout_path)) if stdout_path else ""),
            "stderr_excerpt": self._truncate_text_tail(read_text(Path(stderr_path)) if stderr_path else ""),
            "error": f"timeout_after_{timeout_seconds}s_waiting_for_test_output",
        }

    def _resolve_test_log_paths(
        self,
        *,
        bundle: dict[str, Any],
        submitted_job_id: str | None,
    ) -> dict[str, str]:
        script_path = Path(str(bundle.get("test_submit_path", "")))
        script_text = read_text(script_path)
        stdout_match = re.search(r"^#SBATCH --output=(.+)$", script_text, flags=re.MULTILINE)
        stderr_match = re.search(r"^#SBATCH --error=(.+)$", script_text, flags=re.MULTILINE)
        stdout_path = stdout_match.group(1).strip() if stdout_match else ""
        stderr_path = stderr_match.group(1).strip() if stderr_match else ""
        script_dir = script_path.parent if script_path.exists() else self.run_dir
        job_id = str(submitted_job_id or "").strip()
        if job_id:
            stdout_path = stdout_path.replace("%j", job_id)
            stderr_path = stderr_path.replace("%j", job_id)
        if stdout_path and not Path(stdout_path).is_absolute():
            stdout_path = str((script_dir / stdout_path).resolve())
        if stderr_path and not Path(stderr_path).is_absolute():
            stderr_path = str((script_dir / stderr_path).resolve())
        return {
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
        }

    def _test_monitor_has_error(self, monitor_status: dict[str, Any]) -> bool:
        stderr_excerpt = str(monitor_status.get("stderr_excerpt", "")).strip()
        job_state = str(monitor_status.get("job_state", "")).strip().upper()
        exit_code = str(monitor_status.get("exit_code", "")).strip()
        error = str(monitor_status.get("error", "")).strip()
        return bool(stderr_excerpt) or self._slurm_status_indicates_failure(job_state=job_state, exit_code=exit_code) or bool(error)

    def _build_test_failure_feedback(self, monitor_status: dict[str, Any]) -> str:
        stderr_excerpt = str(monitor_status.get("stderr_excerpt", "")).strip()
        stdout_excerpt = str(monitor_status.get("stdout_excerpt", "")).strip()
        job_state = str(monitor_status.get("job_state", "")).strip()
        exit_code = str(monitor_status.get("exit_code", "")).strip()
        monitor_error = str(monitor_status.get("error", "")).strip()
        lines = [
            "The submitted test job failed during dynamic testing. Rewrite the implementation to fix the failing test run.",
        ]
        if job_state or exit_code:
            lines.extend(
                [
                    "",
                    "Observed Slurm status:",
                    f"JobState={job_state or '(unknown)'} ExitCode={exit_code or '(unknown)'}",
                ]
            )
        if monitor_error:
            lines.extend(["", "Monitor error:", monitor_error])
        lines.extend(
            [
            "",
            "Observed error:",
            stderr_excerpt or "(none)",
            ]
        )
        if stdout_excerpt:
            lines.extend(["", "Observed output:", stdout_excerpt])
        return "\n".join(lines).strip()

    def _query_test_job_status(self, submitted_job_id: str | None) -> dict[str, Any]:
        job_id = str(submitted_job_id or "").strip()
        if not job_id:
            return {"ok": False, "error": "missing_submitted_job_id"}

        sacct_status = self._inspect_sacct_job(job_id)
        if sacct_status.get("ok") and str(sacct_status.get("job_state") or "").strip():
            return sacct_status

        if shutil.which("scontrol") is not None:
            scontrol_status = self._inspect_slurm_job(job_id=job_id)
            parsed = scontrol_status.get("parsed", {}) if scontrol_status.get("ok") else {}
            return {
                "ok": bool(scontrol_status.get("ok")),
                "source": "scontrol",
                "job_id": job_id,
                "job_state": parsed.get("JobState"),
                "exit_code": parsed.get("ExitCode"),
                "raw": scontrol_status,
            }

        return {"ok": False, "job_id": job_id, "error": "no_slurm_status_command_available"}

    def _inspect_sacct_job(self, job_id: str) -> dict[str, Any]:
        if shutil.which("sacct") is None:
            return {"ok": False, "job_id": job_id, "error": "sacct_not_found"}

        command = ["sacct", "-j", job_id, "--format=JobIDRaw,State,ExitCode", "-P", "-n"]
        proc = subprocess.run(command, check=False, capture_output=True, text=True)
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        entries: list[dict[str, str]] = []
        for line in stdout.splitlines():
            parts = [part.strip() for part in line.split("|")]
            if len(parts) < 3:
                continue
            if parts[0] != str(job_id):
                continue
            entries.append(
                {
                    "job_id": parts[0],
                    "job_state": parts[1],
                    "exit_code": parts[2],
                }
            )

        primary = entries[0] if entries else {}
        return {
            "ok": proc.returncode == 0,
            "source": "sacct",
            "command": " ".join(command),
            "returncode": proc.returncode,
            "job_id": job_id,
            "job_state": primary.get("job_state"),
            "exit_code": primary.get("exit_code"),
            "entries": entries,
            "raw": {
                "stdout": stdout,
                "stderr": stderr,
            },
        }

    def _slurm_status_indicates_failure(self, *, job_state: str, exit_code: str) -> bool:
        normalized_state = str(job_state or "").strip().upper()
        normalized_exit = str(exit_code or "").strip()
        if normalized_state and normalized_state != "COMPLETED":
            return normalized_state in self._TERMINAL_SLURM_STATES
        if not normalized_exit:
            return False
        primary = normalized_exit.split(":", 1)[0].strip()
        if not primary.isdigit():
            return False
        return int(primary) != 0

    def _truncate_text_tail(self, text: str, max_chars: int = 4000) -> str:
        clean = (text or "").strip()
        if len(clean) <= max_chars:
            return clean
        return clean[-max_chars:]

    def _parse_squeue_output(self, lines: list[str]) -> list[dict[str, Any]]:
        if len(lines) < 2:
            return []
        jobs: list[dict[str, Any]] = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 8:
                continue
            reason = parts[7]
            if reason.startswith("(") and reason.endswith(")"):
                reason = reason[1:-1]
            jobs.append(
                {
                    "job_id": parts[0],
                    "partition": parts[1],
                    "name": parts[2],
                    "user": parts[3],
                    "state": parts[4],
                    "time": parts[5],
                    "nodes": parts[6],
                    "nodelist_or_reason": parts[7],
                    "reason": reason,
                }
            )
        return jobs

    def _find_job_in_queue(self, jobs: list[dict[str, Any]], submitted_job_id: str | None) -> dict[str, Any] | None:
        if not submitted_job_id:
            return None
        for job in jobs:
            if str(job.get("job_id")) == str(submitted_job_id):
                return job
        return None

    def _parse_scontrol_output(self, text: str) -> dict[str, str]:
        parsed: dict[str, str] = {}
        for key, value in re.findall(r"([A-Za-z][A-Za-z0-9:_/-]*)=([^\s]+)", text):
            parsed[key] = value
        return parsed

    def _build_queue_summary(
        self,
        *,
        matched_job: dict[str, Any] | None,
        scontrol_status: dict[str, Any] | None,
        output_hint: dict[str, Any],
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "state": None,
            "reason": None,
            "expected_start_time": None,
            "scheduled_node": None,
            "stdout_path": None,
            "stderr_path": None,
            "data_output_path": output_hint.get("absolute_output_path") or None,
            "job_id": matched_job.get("job_id") if matched_job else None,
        }
        if matched_job:
            summary["state"] = matched_job.get("state")
            summary["reason"] = matched_job.get("reason")
        if scontrol_status and scontrol_status.get("ok"):
            parsed = scontrol_status.get("parsed", {})
            summary["state"] = parsed.get("JobState", summary["state"])
            summary["reason"] = parsed.get("Reason", summary["reason"])
            summary["expected_start_time"] = parsed.get("StartTime")
            summary["scheduled_node"] = parsed.get("SchedNodeList") or parsed.get("NodeList")
            summary["stdout_path"] = parsed.get("StdOut")
            summary["stderr_path"] = parsed.get("StdErr")
        return summary

    def _to_run_relative_path(self, path_str: str | None) -> str | None:
        if not path_str:
            return None
        path = Path(path_str)
        try:
            return str(path.relative_to(self.run_dir.resolve()))
        except ValueError:
            return str(path)

    def _format_slurm_time(self, value: str | None) -> str | None:
        if not value or value in {"N/A", "Unknown"}:
            return None
        return value.replace("T", " ")

    def _print_queue_progress(self, queue_status: dict[str, Any]) -> None:
        print()
        print("==================== [ Slurm Queue ] ====================")

        if not queue_status.get("ok"):
            print(f"[Queue] Query failed: {queue_status.get('error') or queue_status.get('stderr') or 'unknown error'}")
            return

        summary = queue_status.get("summary", {})
        if not queue_status.get("job_found"):
            submitted_job_id = queue_status.get("submitted_job_id")
            if submitted_job_id:
                print(f"Current state: submitted, but Job {submitted_job_id} is not yet visible in this `squeue` query")
            else:
                print("Current state: the submission result did not include a job id")
            return

        print(f"Current state: {summary.get('state') or 'UNKNOWN'}")
        if summary.get("reason"):
            print(f"Reason: {summary['reason']}")
        formatted_start_time = self._format_slurm_time(summary.get("expected_start_time"))
        if formatted_start_time:
            print(f"Expected start time: {formatted_start_time}")
        if summary.get("scheduled_node"):
            print(f"Allocated node: {summary['scheduled_node']}")
        stdout_path = self._to_run_relative_path(summary.get("stdout_path"))
        stderr_path = self._to_run_relative_path(summary.get("stderr_path"))
        data_output_path = self._to_run_relative_path(summary.get("data_output_path"))
        if stdout_path or stderr_path:
            print("Log output paths:")
            if stdout_path:
                print(stdout_path)
            if stderr_path:
                print(stderr_path)
        if data_output_path:
            print("Data output path:")
            print(data_output_path)

        if queue_status.get("scontrol") and not queue_status["scontrol"].get("ok"):
            print(f"[Queue] `scontrol show job` failed: {queue_status['scontrol']['raw'].get('stderr') or 'unknown error'}")

    def _checkpoint_planner(self, output: dict[str, Any], version: int) -> tuple[bool, str]:
        print("\n" + "=" * 80)
        print(f"[Planner Checkpoint v{version}] Computational plan")
        print("-" * 80)
        print(str(output.get("plan_yaml", "")))
        print("-" * 80)
        print(str(output.get("summary_md", "")))
        print("-" * 80)
        print(f"Citations: {output.get('citations', [])}")
        print("=" * 80)

        if self.non_interactive:
            return True, ""

        try:
            from prompt_toolkit import prompt
            raw = prompt("[Planner Checkpoint] Press Enter to accept the plan, or provide revision feedback: ")
        except ImportError:
            raw = input("[Planner Checkpoint] Press Enter to accept the plan, or provide revision feedback: ")
        feedback = raw.strip()
        if not feedback:
            return True, ""
        return False, feedback

    def _checkpoint_executor(self, bundle: dict[str, Any], version: int) -> tuple[bool, str]:
        shared_code = read_text(Path(str(bundle.get("main_program_path", ""))))
        test_submit = read_text(Path(str(bundle.get("test_submit_path", ""))))
        full_submit = read_text(Path(str(bundle.get("full_submit_path", ""))))
        critique = bundle.get("critique", {})
        summary = str(critique.get("summary", "")).strip()
        errors = critique.get("errors", [])
        warnings = critique.get("warnings", [])
        if not isinstance(errors, list):
            errors = []
        if not isinstance(warnings, list):
            warnings = []
        checkpoint_title = "Scripts generated. Approval will trigger the test submission and queue inspection"
        prompt_text = "[Executor Checkpoint] Press Enter to submit the test job, or provide revision feedback: "
        if self.test_mode:
            checkpoint_title = "Scripts generated. Test mode is active, so no job will be submitted"
            prompt_text = "[Executor Checkpoint] Press Enter to finish, or provide revision feedback: "

        print("\n" + "=" * 80)
        print(f"[Executor Checkpoint v{version}] {checkpoint_title}")
        print("-" * 80)
        print("PYQUDA CODE")
        print(shared_code)
        print("-" * 80)
        print("TEST SUBMIT SCRIPT")
        print(test_submit)
        print("-" * 80)
        print("FULL SUBMIT SCRIPT")
        print(full_submit)
        print("-" * 80)
        print("PLAN SUMMARY")
        print(summary or "(none)")
        print("-" * 80)
        print("WARNINGS")
        print("\n".join(f"- {item}" for item in warnings) if warnings else "(none)")
        if errors:
            print("-" * 80)
            print("EXECUTOR CRITIQUE ERRORS")
            print("\n".join(f"- {item}" for item in errors))
        print("=" * 80)

        if self.non_interactive:
            return True, ""

        try:
            from prompt_toolkit import prompt
            raw = prompt(prompt_text)
        except ImportError:
            raw = input(prompt_text)
        feedback = raw.strip()
        if not feedback:
            return True, ""
        return False, feedback

    def _save_state(self) -> None:
        write_json(self.run_dir / "trajectory_full.json", self.state)
