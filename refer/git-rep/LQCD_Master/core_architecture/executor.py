from __future__ import annotations

import json
import math
import re
import shlex
import struct
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from utils.config_utils import build_prompt_submit_config_yaml, build_runtime_launch, load_yaml_mapping, resolve_submit_config_presets
from utils.io_utils import extract_json_from_text, read_text, write_json, write_text
from utils.prompt_loader import PromptLoader
from utils.static_check import analyze_bundle_static
from utils.skill_utils import (
    SkillContext,
    SkillMessageAssembler,
    SkillRegistry,
    SkillRouter,
    SkillRunner,
    ToolRegistry,
)

class ExecutorAgent:
    def __init__(
        self,
        llm_client: Any,
        submit_tool: Any,
        run_dir: Path,
        tool_client: Any | None = None,
        max_static_check_rounds: int | None = None,
    ):
        self.llm = llm_client
        self.tools = tool_client
        self.submit_tool = submit_tool
        self.run_dir = run_dir
        self.exec_dir = run_dir / "executor_v1"
        self.exec_dir.mkdir(parents=True, exist_ok=True)
        self.submit_config_path = Path("configs/config.yaml")
        self.ensemble_preset_path = Path("configs/ensemble_presets.yaml")
        self.raw_submit_config = load_yaml_mapping(self.submit_config_path)
        self.ensemble_presets = load_yaml_mapping(self.ensemble_preset_path)
        self.submit_config = self._load_submit_config(self.submit_config_path)
        self.prompt_loader = PromptLoader(Path("prompts"))
        self.skill_registry = SkillRegistry(Path("skills"))
        self.skill_registry.load()
        self.skill_router = SkillRouter(self.skill_registry, Path("configs/skills.yaml"))
        self.skill_runner = SkillRunner(
            llm_client=self.llm,
            registry=self.skill_registry,
            router=self.skill_router,
            assembler=SkillMessageAssembler(),
            tool_registry=ToolRegistry(self.tools),
        )
        self.max_static_check_rounds = self._resolve_max_static_check_rounds(max_static_check_rounds)
        self.solve_payload: dict[str, Any] = {}
        self.static_analysis_payload: dict[str, Any] = {}
        self.critique_payload: dict[str, Any] = {}
        self.rewrite_payload: dict[str, Any] = {}
        self.stage_traces: dict[str, list[dict[str, Any]]] = {}
        self.trajectory: dict[str, Any] = {"iterations": []}
        self._active_stage: str = ""
        self._tool_count_by_stage: dict[str, int] = {}

    def _version_dir(self, version: int) -> Path:
        version = max(1, int(version))
        return self.run_dir / f"executor_v{version}"

    def generate_and_submit_test(
        self,
        task: str,
        planner_output: dict[str, Any],
        version: int,
        feedback: str | None = None,
        previous_bundle: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._print_stage_start("generate_and_submit_test", version=version)
        bundle = self.generate_bundle(
            task=task,
            planner_output=planner_output,
            version=version,
            feedback=feedback,
            previous_bundle=previous_bundle,
        )
        version_dir = self._version_dir(version)
        test_submit_path = Path(str(bundle["test_submit_path"]))

        submit_result = self._submit_test_script(test_submit_path)
        write_json(version_dir / "submit_test_result.json", submit_result)
        self._print_stage_done(
            "generate_and_submit_test",
            version=version,
            extra=f"submit_ok={submit_result.get('ok')} | job_id={submit_result.get('job_id')}",
        )

        return bundle, submit_result

    def generate_bundle(
        self,
        *,
        task: str,
        planner_output: dict[str, Any],
        version: int,
        feedback: str | None = None,
        previous_bundle: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        version_dir = self._version_dir(version)
        version_dir.mkdir(parents=True, exist_ok=True)

        previous_code = self._load_previous_code(previous_bundle)
        input_facts = self._collect_input_facts()
        plan_yaml = str(planner_output.get("plan_yaml", ""))
        summary_md = str(planner_output.get("summary_md", ""))
        submit_config_yaml = build_prompt_submit_config_yaml(self.raw_submit_config, self.ensemble_presets)
        input_facts_yaml = yaml.safe_dump(
            input_facts,
            allow_unicode=True,
            sort_keys=False,
        ).strip()

        iteration: dict[str, Any] = {
            "version": f"v{version}",
            "started_at": datetime.now().isoformat(),
            "feedback": feedback or "",
            "static_check_rewrite_rounds": [],
        }

        if previous_code and feedback:
            self._print_stage_start("executor_rewriter", version=version, extra="source=user_feedback")
            payload, solve_trace = self._run_rewriter_stage(
                task=task,
                plan_yaml=plan_yaml,
                summary_md=summary_md,
                submit_config_yaml=submit_config_yaml,
                input_facts_yaml=input_facts_yaml,
                executor_dir=str(version_dir),
                current_code=previous_code,
                critique_summary="",
                errors=[],
                warnings=[],
                feedback=feedback,
            )
            iteration["initial_stage"] = "executor_rewriter"
        else:
            self._print_stage_start("executor", version=version)
            payload, solve_trace = self._run_executor_stage(
                task=task,
                plan_yaml=plan_yaml,
                summary_md=summary_md,
                submit_config_yaml=submit_config_yaml,
                input_facts_yaml=input_facts_yaml,
                executor_dir=str(version_dir),
            )
            iteration["initial_stage"] = "executor"
        iteration["solve_trace"] = solve_trace

        final_payload = payload
        print(f"[Executor] [executor] LLM output keys: {list(final_payload.keys())}; main_program length: {len(final_payload.get('main_program',''))}")
        # Suggestion 1: catch ValueError (missing required fields), build feedback, retry via rewriter
        max_validate_retries = 2
        for validate_attempt in range(max_validate_retries + 1):
            try:
                final_bundle_data = self._build_bundle_data_from_payload(final_payload, version_dir)
                break
            except ValueError as e:
                error_msg = str(e)
                if validate_attempt < max_validate_retries and feedback:
                    # Already in a feedback retry, do NOT retry validation again
                    # (we're in the previous_bundle/feedback path); re-raise
                    raise
                if validate_attempt >= max_validate_retries:
                    self._print_stage_info(
                        "executor", version=version,
                        message=f"validation failed after {max_validate_retries + 1} attempts: {error_msg}"
                    )
                    raise
                self._print_stage_info(
                    "executor", version=version,
                    message=f"validation failed (attempt {validate_attempt + 1}): {error_msg}; retrying with feedback"
                )
                missing_fields = {}
                for key in ["main_program", "test_submit", "full_submit"]:
                    val = final_payload.get(key, None)
                    if isinstance(val, str) and val.strip():
                        missing_fields[key] = val
                    elif isinstance(val, dict):
                        missing_fields[key] = json.dumps(val, indent=2)
                retry_feedback = (
                    f"Your previous output was incomplete: {error_msg}.\n"
                    f"You MUST include ALL 4 fields: main_program (complete Python code), "
                    f"test_submit (SLURM sbatch spec), full_submit (SLURM sbatch spec), "
                    f"and notes (optional). Please regenerate with the missing field(s)."
                )
                rewrite_payload, rewrite_trace = self._run_rewriter_stage(
                    task=task,
                    plan_yaml=plan_yaml,
                    summary_md=summary_md,
                    submit_config_yaml=submit_config_yaml,
                    input_facts_yaml=input_facts_yaml,
                    executor_dir=str(version_dir),
                    current_code=missing_fields if missing_fields else None,
                    critique_summary="",
                    errors=[error_msg],
                    warnings=[],
                    feedback=retry_feedback,
                )
                final_payload = rewrite_payload
                print(f"[Executor] [executor] rewrite attempt keys: {list(final_payload.keys())}; main_program length: {len(final_payload.get('main_program',''))}")
        final_bundle_data["main_program"] = self._write_sink_files(solve_trace, version_dir, final_bundle_data["main_program"])
        static_report = self._run_static_analysis_stage(
            main_program=final_bundle_data["main_program"],
            test_submit_script=final_bundle_data["test_submit"],
            full_submit_script=final_bundle_data["full_submit"],
        )
        critique_report, critique_trace = self._run_critique_stage(
            task=task,
            plan_yaml=plan_yaml,
            summary_md=summary_md,
            main_program=final_bundle_data["main_program"],
            test_submit_script=final_bundle_data["test_submit"],
            full_submit_script=final_bundle_data["full_submit"],
            static_analysis_report=static_report,
        )
        critique_report = self._merge_static_analysis_into_critique(critique_report, static_report)
        iteration["critique_trace"] = critique_trace

        static_check_round = 0
        while critique_report["errors"] and static_check_round < self.max_static_check_rounds:
            static_check_round += 1
            self._print_stage_info(
                "executor_critique",
                version=version,
                message=(
                    f"found {len(critique_report['errors'])} error(s); "
                    f"trigger executor_rewriter static-check round {static_check_round}"
                ),
            )
            rewrite_payload, rewrite_trace = self._run_rewriter_stage(
                task=task,
                plan_yaml=plan_yaml,
                summary_md=summary_md,
                submit_config_yaml=submit_config_yaml,
                input_facts_yaml=input_facts_yaml,
                executor_dir=str(version_dir),
                current_code={
                    "main_program": final_bundle_data["main_program"],
                    "test_submit": final_bundle_data["test_submit"],
                    "full_submit": final_bundle_data["full_submit"],
                },
                critique_summary=critique_report["summary"],
                errors=critique_report["errors"],
                warnings=critique_report["warnings"],
                feedback=feedback,
            )
            final_payload = rewrite_payload
            # Suggestion 1 (static check loop): catch ValueError from rewriter, raise critique error to abort loop
            try:
                final_bundle_data = self._build_bundle_data_from_payload(final_payload, version_dir)
            except ValueError as e:
                self._print_stage_info(
                    "executor_rewrite", version=version,
                    message=f"rewrite produced incomplete output: {e}; aborting static check loop"
                )
                # Force critque_report to have no errors so the while loop terminates
                critique_report["errors"] = []
                critique_report["summary"] = f"Rewrite failed: {e}. Using previous valid bundle."
                break
            final_bundle_data["main_program"] = self._write_sink_files(rewrite_trace, version_dir, final_bundle_data["main_program"])
            static_report = self._run_static_analysis_stage(
                main_program=final_bundle_data["main_program"],
                test_submit_script=final_bundle_data["test_submit"],
                full_submit_script=final_bundle_data["full_submit"],
            )
            critique_report, critique_trace = self._run_critique_stage(
                task=task,
                plan_yaml=plan_yaml,
                summary_md=summary_md,
                main_program=final_bundle_data["main_program"],
                test_submit_script=final_bundle_data["test_submit"],
                full_submit_script=final_bundle_data["full_submit"],
                static_analysis_report=static_report,
            )
            critique_report = self._merge_static_analysis_into_critique(critique_report, static_report)
            iteration["static_check_rewrite_rounds"].append(
                {
                    "round": static_check_round,
                    "rewrite_trace": rewrite_trace,
                    "critique_trace": critique_trace,
                    "static_analysis": static_report,
                    "critique": critique_report,
                }
            )

        if critique_report["errors"]:
            self._print_stage_info(
                "executor_critique",
                version=version,
                message=(
                    f"remaining_errors={len(critique_report['errors'])}; "
                    "pass_to_checkpoint_for_human_review"
                ),
            )

        main_program_path = version_dir / "main.py"
        test_submit_path = version_dir / "submit_test.sh"
        full_submit_path = version_dir / "submit_full.sh"
        static_analysis_report_path = version_dir / "static_analysis.json"
        critique_report_path = version_dir / "critique.json"

        write_text(main_program_path, final_bundle_data["main_program"])
        write_text(test_submit_path, final_bundle_data["test_submit"])
        write_text(full_submit_path, final_bundle_data["full_submit"])
        write_json(static_analysis_report_path, static_report)
        write_json(critique_report_path, critique_report)
        self._print_stage_info("executor", version=version, message=f"artifacts_saved={version_dir}")

        bundle = {
            "main_program_path": str(main_program_path.resolve()),
            "test_submit_path": str(test_submit_path.resolve()),
            "full_submit_path": str(full_submit_path.resolve()),
            "static_analysis_report_path": str(static_analysis_report_path.resolve()),
            "critique_report_path": str(critique_report_path.resolve()),
            "static_analysis": static_report,
            "critique": critique_report,
        }

        iteration["finished_at"] = datetime.now().isoformat()
        iteration["final_bundle"] = bundle
        iteration["final_critique"] = critique_report
        self.trajectory["iterations"].append(iteration)

        write_json(
            version_dir / "generation.json",
            {
                "submit_config_path": str(self.submit_config_path.resolve()),
                "input_facts": input_facts,
                "feedback": feedback or "",
                "solve_payload": self.solve_payload,
                "static_analysis_payload": self.static_analysis_payload,
                "critique_payload": self.critique_payload,
                "rewrite_payload": self.rewrite_payload,
                "stage_traces": self.stage_traces,
                "final_submit_specs": {
                    "test_submit": final_bundle_data["test_submit_spec"],
                    "full_submit": final_bundle_data["full_submit_spec"],
                },
                "bundle": bundle,
                "trajectory": self.trajectory,
            },
        )
        return bundle

    def submit_full(self, bundle: dict[str, Any], version: int) -> dict[str, Any]:
        self._print_stage_start("submit_full", version=version)
        version_dir = self._version_dir(version)
        script_path = str(bundle.get("full_submit_path", ""))
        result = self.submit_tool.submit(script_path)
        write_json(version_dir / "submit_full_result.json", result)
        self._print_stage_done(
            "submit_full",
            version=version,
            extra=(
                f"ok={result.get('ok')} | job_id={result.get('job_id')}"
            ),
        )
        return result

    def submit_test(self, bundle: dict[str, Any], version: int) -> dict[str, Any]:
        self._print_stage_start("submit_test", version=version)
        version_dir = self._version_dir(version)
        script_path = Path(str(bundle.get("test_submit_path", "")))
        result = self._submit_test_script(script_path)
        write_json(version_dir / "submit_test_result.json", result)
        self._print_stage_done(
            "submit_test",
            version=version,
            extra=f"ok={result.get('ok')} | job_id={result.get('job_id')}",
        )
        return result

    def collect_evidence_paths(self, bundle: dict[str, Any], version: int) -> list[str]:
        version_dir = self._version_dir(version)
        files: list[str] = []
        if version_dir.exists():
            for p in sorted(version_dir.rglob("*")):
                if p.is_file() and p.suffix in {".npy", ".out", ".err", ".json", ".log", ".txt"}:
                    files.append(str(p.resolve()))
        for key in ["main_program_path", "test_submit_path", "full_submit_path"]:
            p = str(bundle.get(key, ""))
            if p and p not in files:
                files.append(p)
        return files

    def _run_executor_stage(
        self,
        *,
        task: str,
        plan_yaml: str,
        summary_md: str,
        submit_config_yaml: str,
        input_facts_yaml: str,
        executor_dir: str = "",
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        self._active_stage = "executor"
        self._tool_count_by_stage.setdefault(self._active_stage, 0)
        messages = self._build_executor_user_message(
            stage="executor_generate",
            payload={
                "task": task,
                "plan_yaml": plan_yaml,
                "summary_md": summary_md,
                "submit_config_yaml": submit_config_yaml,
                "executor_dir": executor_dir,
                "input_facts_yaml": input_facts_yaml,
            },
        )
        payload, trace = self._chat_json_autonomous(messages, stage="executor_generate", task=task)
        payload = self._normalize_payload(payload)
        self.solve_payload = payload
        self.stage_traces["executor"] = trace
        self._print_stage_done("executor", tool_calls=len(trace))
        return payload, trace

    def _run_critique_stage(
        self,
        *,
        task: str,
        plan_yaml: str,
        summary_md: str,
        main_program: str,
        test_submit_script: str,
        full_submit_script: str,
        static_analysis_report: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        self._active_stage = "executor_critique"
        self._tool_count_by_stage.setdefault(self._active_stage, 0)
        messages = self._build_executor_user_message(
            stage="executor_critique",
            payload={
                "task": task,
                "plan_yaml": plan_yaml,
                "summary_md": summary_md,
                "main_program": main_program,
                "test_submit_script": test_submit_script,
                "full_submit_script": full_submit_script,
                "static_analysis_report": static_analysis_report,
            },
        )
        payload, trace = self._chat_json_autonomous(messages, stage="executor_critique", task=task)
        report = self._normalize_critique_payload(payload)
        self.critique_payload = report
        self.stage_traces["executor_critique"] = trace
        self._print_stage_done("executor_critique", tool_calls=len(trace))
        return report, trace

    def _run_rewriter_stage(
        self,
        *,
        task: str,
        plan_yaml: str,
        summary_md: str,
        submit_config_yaml: str,
        input_facts_yaml: str,
        executor_dir: str = "",
        current_code: dict[str, str] | None,
        critique_summary: str,
        errors: list[str],
        warnings: list[str],
        feedback: str | None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        self._active_stage = "executor_rewriter"
        self._tool_count_by_stage.setdefault(self._active_stage, 0)
        messages = self._build_executor_user_message(
            stage="executor_rewrite",
            payload={
                "task": task,
                "plan_yaml": plan_yaml,
                "summary_md": summary_md,
                "submit_config_yaml": submit_config_yaml,
                "input_facts_yaml": input_facts_yaml,
                "executor_dir": executor_dir,
                "current_code": current_code,
                "critique_summary": critique_summary,
                "errors": errors,
                "warnings": warnings,
                "static_analysis_report": self.static_analysis_payload,
                "feedback": feedback,
            },
        )
        payload, trace = self._chat_json_autonomous(messages, stage="executor_rewrite", task=task)
        payload = self._normalize_payload(payload)
        self.rewrite_payload = payload
        self.stage_traces["executor_rewriter"] = trace
        self._print_stage_done("executor_rewriter", tool_calls=len(trace))
        return payload, trace

    def _normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            payload = {}
        if "raw_text" in payload:
            payload = extract_json_from_text(str(payload.get("raw_text", "")))
        return payload if isinstance(payload, dict) else {}

    def _normalize_critique_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = self._normalize_payload(payload)
        summary = str(payload.get("summary", "")).strip()
        errors = payload.get("errors")
        warnings = payload.get("warnings")
        if not isinstance(errors, list):
            errors = []
        if not isinstance(warnings, list):
            warnings = []
        return {
            "summary": summary,
            "errors": [str(item).strip() for item in errors if str(item).strip()],
            "warnings": [str(item).strip() for item in warnings if str(item).strip()],
            "notes": str(payload.get("notes", "")).strip(),
        }

    def _run_static_analysis_stage(
        self,
        *,
        main_program: str,
        test_submit_script: str,
        full_submit_script: str,
    ) -> dict[str, Any]:
        self._print_stage_start("executor_static_analysis")
        report = self._analyze_bundle_static(
            main_program=main_program,
            test_submit_script=test_submit_script,
            full_submit_script=full_submit_script,
        )
        self.static_analysis_payload = report
        self.stage_traces["executor_static_analysis"] = []
        self._print_stage_done(
            "executor_static_analysis",
            extra=f"errors={len(self._flatten_static_analysis_errors(report))}",
        )
        return report

    def _merge_static_analysis_into_critique(
        self,
        critique_report: dict[str, Any],
        static_report: dict[str, Any],
    ) -> dict[str, Any]:
        merged = {
            "summary": str(critique_report.get("summary", "")).strip(),
            "errors": list(critique_report.get("errors", []) or []),
            "warnings": list(critique_report.get("warnings", []) or []),
            "notes": str(critique_report.get("notes", "")).strip(),
        }

        for item in self._flatten_static_analysis_errors(static_report):
            text = f"[static] {str(item).strip()}"
            if text not in merged["errors"]:
                merged["errors"].append(text)
        return merged

    def _flatten_static_analysis_errors(self, static_report: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        for key in ("main_errors", "test_sh_errors", "full_sh_errors"):
            for item in static_report.get(key, []) or []:
                text = str(item).strip()
                if text:
                    errors.append(text)
        return errors

    def _write_sink_files(self, trace: list[dict[str, Any]], version_dir: Path, main_program: str) -> str:
        """Copy sink file from tool's write location to version_dir, fix sink_path in main_program.
        Use just the filename (not full path) since the job runs from executor_v1 directory."""
        import re, shutil
        for entry in trace:
            if isinstance(entry, dict):
                for result_key in ("tool_result", "result", "output"):
                    result = entry.get(result_key) or {}
                    if isinstance(result, dict) and "sink_path" in result:
                        src_path = result["sink_path"]
                        sink_file = result.get("sink_file", "sink_output.py")
                        dst_path = str(version_dir / sink_file)
                        version_dir.mkdir(parents=True, exist_ok=True)
                        if Path(src_path).exists():
                            shutil.move(src_path, dst_path)
                            print(f"[Executor] moved sink file {src_path} -> {dst_path}")
                        # Replace any sink_path assignment with just the filename
                        main_program = re.sub(
                            r"sink_path\s*=\s*['\"].*?['\"]",
                            f"sink_path = '{sink_file}'",
                            main_program
                        )
        return main_program

    def _build_bundle_data_from_payload(self, payload: dict[str, Any], version_dir: Path) -> dict[str, Any]:
        main_program = self._require_payload_field(payload, "main_program")
        test_submit_spec = self._require_submit_spec(payload, "test_submit", expected_program="main.py")
        full_submit_spec = self._require_submit_spec(payload, "full_submit", expected_program="main.py")
        test_submit_spec = self._normalize_submit_spec_paths(test_submit_spec, version_dir, stem="test")
        full_submit_spec = self._normalize_submit_spec_paths(full_submit_spec, version_dir, stem="full")
        test_submit = self._render_submit_script(test_submit_spec, script_dir=version_dir)
        full_submit = self._render_submit_script(full_submit_spec, script_dir=version_dir)
        return {
            "main_program": main_program,
            "test_submit": test_submit,
            "full_submit": full_submit,
            "test_submit_spec": test_submit_spec,
            "full_submit_spec": full_submit_spec,
        }

    def _load_previous_code(self, previous_bundle: dict[str, Any] | None) -> dict[str, str] | None:
        if not previous_bundle:
            return None
        def _safe_read(key: str) -> str:
            raw = str(previous_bundle.get(key, "")).strip()
            if not raw:
                return ""
            path = Path(raw)
            if not path.exists() or not path.is_file():
                return ""
            return read_text(path)
        return {
            "main_program": _safe_read("main_program_path"),
            "test_submit": _safe_read("test_submit_path"),
            "full_submit": _safe_read("full_submit_path"),
        }

    def _submit_test_script(self, script_path: Path) -> dict[str, Any]:
        self._print_stage_info("submit_test", message=f"submit_script={script_path}")
        return self.submit_tool.submit(str(script_path))

    def _collect_input_facts(self) -> dict[str, Any]:
        ensemble_cfg = self.raw_submit_config.get("ensemble", {})
        preset_name = str(ensemble_cfg.get("preset", "")).strip()
        preset = self.ensemble_presets.get(preset_name) if preset_name else {}
        if not isinstance(preset, dict):
            preset = {}

        cfg_path_template = str(preset.get("cfg_path", "")).strip()
        cfg_dir = Path(cfg_path_template).expanduser().parent if cfg_path_template else Path("")
        cfg_nums = ensemble_cfg.get("cfg_num", [])

        facts: dict[str, Any] = {
            "preset": preset_name,
            "cfg_num": cfg_nums if isinstance(cfg_nums, list) else [],
            "cfg_path_template": cfg_path_template,
            "cfg_directory": str(cfg_dir) if cfg_path_template else "",
            "cfg_directory_exists": bool(cfg_path_template) and cfg_dir.exists(),
            "configured_samples": [],
            "discovered_cfg_samples": [],
        }

        if isinstance(cfg_nums, list):
            for cfg in cfg_nums[:8]:
                sample = self._build_cfg_sample(cfg_path_template, str(cfg))
                facts["configured_samples"].append(sample)

        discovered = self._discover_cfg_samples(cfg_path_template)
        facts["discovered_cfg_samples"] = discovered["samples"]
        facts["discovered_cfg_count"] = discovered["count"]
        facts["directory_entries_sample"] = discovered["entries"]
        header_sample = self._select_header_sample(facts["configured_samples"], discovered["samples"])
        facts["gauge_header_probe"] = self._probe_gauge_header(header_sample)
        return facts

    def _select_header_sample(
        self,
        configured_samples: list[dict[str, Any]],
        discovered_samples: list[dict[str, Any]],
    ) -> str:
        for sample in configured_samples:
            path = str(sample.get("path", ""))
            if path:
                return path
        for sample in discovered_samples:
            path = str(sample.get("path", ""))
            if path:
                return path
        return ""

    def _build_cfg_sample(self, cfg_path_template: str, cfg: str) -> dict[str, Any]:
        if not cfg_path_template:
            return {"cfg": cfg, "error": "missing cfg_path_template"}
        path_str = self._format_cfg_path(cfg_path_template, cfg)
        if not path_str:
            return {"cfg": cfg, "error": "cfg_path_template format failed"}
        path = Path(path_str).expanduser()
        sample = {
            "cfg": cfg,
            "path": str(path),
            "exists": path.exists(),
        }
        if path.exists():
            sample["filename"] = path.name
        if path.exists() and path.is_file():
            sample["size_bytes"] = path.stat().st_size
        return sample

    def _discover_cfg_samples(self, cfg_path_template: str) -> dict[str, Any]:
        if not cfg_path_template:
            return {"count": 0, "samples": [], "entries": []}

        template_path = Path(cfg_path_template).expanduser()
        base_path = template_path.parent
        if not base_path.exists() or not base_path.is_dir():
            return {"count": 0, "samples": [], "entries": []}

        entries = [p.name for p in sorted(base_path.iterdir())[:20]]
        pattern_regex = self._cfg_path_template_to_regex(template_path.name)
        if not pattern_regex:
            return {"count": 0, "samples": [], "entries": entries}

        samples: list[dict[str, Any]] = []
        count = 0
        for path in sorted(base_path.iterdir()):
            if not path.is_file():
                continue
            match = pattern_regex.match(path.name)
            if not match:
                continue
            count += 1
            if len(samples) < 8:
                sample: dict[str, Any] = {
                    "cfg": match.group("cfg"),
                    "filename": path.name,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                }
                samples.append(sample)
        return {"count": count, "samples": samples, "entries": entries}

    def _probe_gauge_header(self, sample_path: str) -> dict[str, Any]:
        if not sample_path:
            return {"ok": False, "error": "no_sample_path_available"}
        path = Path(sample_path)
        if not path.exists() or not path.is_file():
            return {"ok": False, "path": sample_path, "error": "sample_file_not_found"}

        lime_probe = self._probe_lime_header(path)
        if lime_probe.get("ok"):
            return lime_probe
        return {
            "ok": False,
            "path": sample_path,
            "error": "unsupported_or_unparsed_header",
            "details": lime_probe,
        }

    def _probe_lime_header(self, path: Path) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        geometry: list[int] | None = None
        detected_format: str | None = None
        try:
            with path.open("rb") as f:
                for _ in range(8):
                    header = f.read(144)
                    if len(header) < 144:
                        break
                    magic, version, flags, data_length, type_bytes = struct.unpack(">IHHQ128s", header)
                    if magic != 0x456789AB:
                        return {
                            "ok": False,
                            "path": str(path),
                            "error": f"not_lime_magic:{hex(magic)}",
                        }
                    record_type = type_bytes.split(b"\x00", 1)[0].decode("ascii", errors="ignore")
                    data = f.read(data_length)
                    pad = (-data_length) % 8
                    if pad:
                        f.read(pad)
                    preview = data[:256]
                    preview_text = preview.decode("utf-8", errors="ignore")
                    records.append(
                        {
                            "type": record_type,
                            "length": data_length,
                            "preview": preview_text,
                        }
                    )
                    fmt = self._detect_format_from_record_type(record_type)
                    if fmt and not detected_format:
                        detected_format = fmt
                    parsed_geometry = self._extract_geometry_from_payload(data)
                    if parsed_geometry and not geometry:
                        geometry = parsed_geometry
                    if geometry and detected_format:
                        break
        except OSError as exc:
            return {"ok": False, "path": str(path), "error": f"read_failed:{exc}"}

        return {
            "ok": True,
            "path": str(path),
            "container": "lime",
            "format": detected_format or "unknown",
            "geometry": geometry,
            "records": records,
        }

    def _detect_format_from_record_type(self, record_type: str) -> str | None:
        lower = record_type.lower()
        if "ildg" in lower:
            return "ildg"
        if "scidac" in lower or "qio" in lower:
            return "chroma_qio"
        return None

    def _extract_geometry_from_payload(self, payload: bytes) -> list[int] | None:
        text = payload.decode("utf-8", errors="ignore").strip()
        if not text or "<" not in text:
            return None
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            return self._extract_geometry_by_regex(text)

        dims = self._extract_geometry_from_xml_tree(root)
        if dims:
            return dims
        return self._extract_geometry_by_regex(text)

    def _extract_geometry_from_xml_tree(self, root: ET.Element) -> list[int] | None:
        for elem in root.iter():
            tag = elem.tag.split("}", 1)[-1].lower()
            text = (elem.text or "").strip()
            if tag in {"dims", "nrow"} and text:
                nums = [int(x) for x in re.findall(r"-?\d+", text)]
                if len(nums) == 4:
                    return nums
        values: dict[str, int] = {}
        for elem in root.iter():
            tag = elem.tag.split("}", 1)[-1].lower()
            text = (elem.text or "").strip()
            if tag in {"lx", "ly", "lz", "lt"} and re.fullmatch(r"-?\d+", text):
                values[tag] = int(text)
        if all(key in values for key in ("lx", "ly", "lz", "lt")):
            return [values["lx"], values["ly"], values["lz"], values["lt"]]
        return None

    def _extract_geometry_by_regex(self, text: str) -> list[int] | None:
        dims_match = re.search(r"<(?:dims|nrow)>\s*([0-9\s]+)\s*</(?:dims|nrow)>", text, flags=re.IGNORECASE)
        if dims_match:
            nums = [int(x) for x in re.findall(r"\d+", dims_match.group(1))]
            if len(nums) == 4:
                return nums

        tags = {}
        for key in ("lx", "ly", "lz", "lt"):
            match = re.search(rf"<{key}>\s*(\d+)\s*</{key}>", text, flags=re.IGNORECASE)
            if match:
                tags[key] = int(match.group(1))
        if all(key in tags for key in ("lx", "ly", "lz", "lt")):
            return [tags["lx"], tags["ly"], tags["lz"], tags["lt"]]
        return None

    def _format_cfg_path(self, cfg_path_template: str, cfg: str) -> str:
        if "{n_cfg}" in cfg_path_template:
            return cfg_path_template.replace("{n_cfg}", cfg)
        if "{cfg}" in cfg_path_template:
            return cfg_path_template.replace("{cfg}", cfg)
        return cfg_path_template

    def _cfg_path_template_to_regex(self, pattern: str) -> re.Pattern[str] | None:
        if not pattern or ("{cfg}" not in pattern and "{n_cfg}" not in pattern):
            return None
        escaped = re.escape(pattern)
        escaped = escaped.replace(re.escape("{cfg}"), r"(?P<cfg>[^/]+)")
        escaped = escaped.replace(re.escape("{n_cfg}"), r"(?P<cfg>[^/]+)")
        try:
            return re.compile(rf"^{escaped}$")
        except re.error:
            return None

    def _load_submit_config(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"submit config not found: {path}")
        data = load_yaml_mapping(path)
        if not data:
            raise ValueError(f"submit config must be a YAML mapping: {path}")
        data = resolve_submit_config_presets(data, self.ensemble_presets)
        required_mapping_sections = [
            "script",
            "slurm",
            "runtime",
            "ensemble",
            "measurement",
            "solver",
            "output",
        ]
        for key in required_mapping_sections:
            value = data.get(key)
            if value is None:
                continue
            if not isinstance(value, dict):
                raise ValueError(f"submit config section must be a YAML mapping: {key}")
        return data

    def _resolve_max_static_check_rounds(self, explicit: int | None) -> int:
        if explicit is not None:
            return max(1, int(explicit))

        workflow_cfg = self.submit_config.get("workflow", {})
        if isinstance(workflow_cfg, dict):
            raw = workflow_cfg.get("executor_static_check_rounds")
            try:
                if raw is not None:
                    return max(1, int(raw))
            except (TypeError, ValueError):
                pass
        return 5

    def _chat_json_autonomous(
        self,
        user_message: str,
        *,
        stage: str,
        task: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        context = SkillContext(
            stage=stage,
            task=task,
            payload={},
        )
        payload, trace, _ = self.skill_runner.run(
            context,
            self._load_system_prompt(stage),
            user_message,
#            fallback_tools=["web_search", "web_parse"],
            fallback_tools=[ ],
        )
        if not isinstance(payload, dict):
            payload = {}
        if "raw_text" in payload:
            payload = extract_json_from_text(str(payload.get("raw_text", "")))
        return payload, trace

    def _load_system_prompt(self, stage: str) -> str:
        mapping = {
            "executor_generate": "executor/generate_system.md",
            "executor_critique": "executor/critique_system.md",
            "executor_rewrite": "executor/rewrite_system.md",
        }
        return self.prompt_loader.load(mapping[stage])

    def _build_executor_user_message(self, *, stage: str, payload: dict[str, Any]) -> str:
        if stage == "executor_generate":
            return (
                f"{self.prompt_loader.load('executor/generate_user.md')}\n\n"
                f"Original task:\n{payload.get('task', '')}\n\n"
                f"Approved plan YAML:\n{payload.get('plan_yaml', '')}\n\n"
                f"Plan summary:\n{payload.get('summary_md', '')}\n\n"
                "Fixed submission configuration (do not override existing settings):\n"
                f"{payload.get('submit_config_yaml') or '(none)'}\n\n"
                "Executor directory:\n"
                f"{payload.get('executor_dir', '(unknown)')}\n\n"
                "Observed input facts from the filesystem:\n"
                f"{payload.get('input_facts_yaml') or '(none)'}\n"
            )
        if stage == "executor_critique":
            return (
                f"{self.prompt_loader.load('executor/critique_user.md')}\n\n"
                f"Original task:\n{payload.get('task', '')}\n\n"
                f"Approved plan YAML:\n{payload.get('plan_yaml', '')}\n\n"
                f"Plan summary:\n{payload.get('summary_md', '')}\n\n"
                "Current main program:\n"
                f"{payload.get('main_program', '')}\n\n"
                "Current test submission script:\n"
                f"{payload.get('test_submit_script', '')}\n\n"
                "Current full submission script:\n"
                f"{payload.get('full_submit_script', '')}\n\n"
                "Static analysis report:\n"
                f"{json.dumps(payload.get('static_analysis_report') or {}, ensure_ascii=False, indent=2)}\n"
            )
        if stage == "executor_rewrite":
            feedback = str(payload.get("feedback") or "").strip()
            return (
                f"{self.prompt_loader.load('executor/rewrite_user.md')}\n\n"
                f"Original task:\n{payload.get('task', '')}\n\n"
                f"Approved plan YAML:\n{payload.get('plan_yaml', '')}\n\n"
                f"Plan summary:\n{payload.get('summary_md', '')}\n\n"
                "Fixed submission configuration (do not override existing settings):\n"
                f"{payload.get('submit_config_yaml') or '(none)'}\n\n"
                "Observed input facts from the filesystem:\n"
                f"{payload.get('input_facts_yaml') or '(none)'}\n\n"
                "Current implementation bundle:\n"
                f"{json.dumps(payload.get('current_code') or {}, ensure_ascii=False, indent=2)}\n"
                f"Critique summary:\n{payload.get('critique_summary') or '(none)'}\n\n"
                f"Critique errors:\n{json.dumps(payload.get('errors') or [], ensure_ascii=False, indent=2)}\n"
                f"Critique warnings:\n{json.dumps(payload.get('warnings') or [], ensure_ascii=False, indent=2)}\n"
                "Static analysis report:\n"
                f"{json.dumps(payload.get('static_analysis_report') or {}, ensure_ascii=False, indent=2)}\n"
                f"User feedback:\n{feedback or '(none)'}\n"
            )
        raise ValueError(f"unknown executor stage: {stage}")

    def _analyze_bundle_static(
        self,
        *,
        main_program: str,
        test_submit_script: str,
        full_submit_script: str,
    ) -> dict[str, Any]:
        return analyze_bundle_static(
            main_program=main_program,
            test_submit_script=test_submit_script,
            full_submit_script=full_submit_script,
        )

    def _print_stage_start(self, stage: str, *, version: int | None = None, extra: str = "") -> None:
        label = stage if version is None else f"{stage}:v{version}"
        tail = f" | {extra}" if extra else ""
        print(f"[Executor] [{label}] stage start! 🚀{tail}")

    def _print_stage_done(
        self,
        stage: str,
        *,
        version: int | None = None,
        tool_calls: int | None = None,
        extra: str = "",
    ) -> None:
        label = stage if version is None else f"{stage}:v{version}"
        parts: list[str] = []
        if tool_calls is not None:
            parts.append(f"tool_calls={tool_calls}")
        if extra:
            parts.append(extra)
        tail = f" | {' | '.join(parts)}" if parts else ""
        print(f"[Executor] [{label}] stage done! ✅{tail}")

    def _print_stage_info(self, stage: str, *, version: int | None = None, message: str) -> None:
        label = stage if version is None else f"{stage}:v{version}"
        print(f"[Executor] [{label}] {message}")

    def _print_tool_call(self, stage: str, tool_name: str) -> None:
        print(f"[Executor] [{stage}] tool call: {tool_name} 🔧")

    def _require_payload_field(self, payload: dict[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"executor model output missing required field: {key}")
        return value

    def _require_submit_spec(
        self,
        payload: dict[str, Any],
        key: str,
        *,
        expected_program: str,
    ) -> dict[str, Any]:
        value = payload.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"executor model output missing required submit spec: {key}")

        job = value.get("job")
        run = value.get("run")
        if not isinstance(job, dict) or not isinstance(run, dict):
            raise ValueError(f"executor model submit spec must contain job/run objects: {key}")

        required_job_keys = ["name", "output", "error", "time"]
        for field in required_job_keys:
            field_val = job.get(field)
            if not isinstance(field_val, str) or not field_val.strip():
                raise ValueError(f"executor model submit spec missing job.{field}: {key}")

        program = run.get("program")
        if not isinstance(program, str) or program.strip() != expected_program:
            raise ValueError(
                f"executor model submit spec must use run.program={expected_program!r}: {key}"
            )

        args = run.get("args")
        if not isinstance(args, (list, str)):
            raise ValueError(f"executor model submit spec must use run.args as list[str] or str: {key}")
        if isinstance(args, list):
            args = [str(item) for item in args if str(item).strip()]

        array = job.get("array")
        if array is None:
            array = ""
        elif not isinstance(array, str):
            array = str(array)

        return {
            "job": {
                "name": str(job["name"]).strip(),
                "output": str(job["output"]).strip(),
                "error": str(job["error"]).strip(),
                "array": array.strip(),
                "time": str(job["time"]).strip(),
            },
            "run": {
                "program": expected_program,
                "args": args,
            },
        }


    def _normalize_submit_spec_paths(
        self,
        submit_spec: dict[str, Any],
        version_dir: Path,
        *,
        stem: str,
    ) -> dict[str, Any]:
        job = dict(submit_spec["job"])
        run = dict(submit_spec["run"])
        job["output"] = str((version_dir / f"{stem}_%j.out").resolve())
        job["error"] = str((version_dir / f"{stem}_%j.err").resolve())
        return {"job": job, "run": run}

    def _render_submit_script(self, submit_spec: dict[str, Any], *, script_dir: Path) -> str:
        script_cfg = self.submit_config.get("script", {})
        slurm_cfg = self.submit_config.get("slurm", {})
        runtime_cfg = self.submit_config.get("runtime", {})
        ensemble_cfg = self.submit_config.get("ensemble", {})

        shell = str(script_cfg.get("shell", "/bin/bash")).strip() or "/bin/bash"
        job = submit_spec["job"]
        run = submit_spec["run"]
        process_grid = self._require_process_grid(ensemble_cfg)
        ntasks = math.prod(process_grid)

        lines = [f"#!{shell}"]
        lines.append(f"#SBATCH -J {job['name']}")
        lines.append(f"#SBATCH -p {self._require_config_value(slurm_cfg, 'partition')}")
        lines.append(f"#SBATCH --output={job['output']}")
        lines.append(f"#SBATCH --error={job['error']}")
        array = str(job.get("array", "")).strip()
        if array:
            lines.append(f"#SBATCH --array={array}")
        lines.append(f"#SBATCH --nodes={self._require_config_value(slurm_cfg, 'nodes')}")
        lines.append(f"#SBATCH -n {ntasks}")
        lines.append(f"#SBATCH --gres={self._require_config_value(slurm_cfg, 'gres')}")
        exclude = str(slurm_cfg.get("exclude", "")).strip()
        if exclude:
            lines.append(f'#SBATCH --exclude="{exclude}"')
        lines.append(f"#SBATCH --time={job['time']}")
        lines.append(f"#SBATCH --ntasks-per-node={self._require_config_value(slurm_cfg, 'ntasks_per_node')}")
        lines.append(
            f"#SBATCH --ntasks-per-socket={self._require_config_value(slurm_cfg, 'ntasks_per_socket')}"
        )
        if bool(slurm_cfg.get("exclusive", False)):
            lines.append("#SBATCH --exclusive")

        lines.extend(["", "set -euo pipefail", ""])

        module_purge = runtime_cfg.get("module_purge")
        if module_purge is None or bool(module_purge):
            lines.append("module purge")

        for module in runtime_cfg.get("modules", []) or []:
            mod = str(module).strip()
            if mod:
                lines.append(f"module load {mod}")

        for key, value in (runtime_cfg.get("exports", {}) or {}).items():
            lines.append(f"export {key}={shlex.quote(str(value))}")

        activate = str(runtime_cfg.get("activate", "")).strip()
        if activate:
            lines.append(activate)

        for command in runtime_cfg.get("source", []) or []:
            cmd = str(command).strip()
            if cmd:
                lines.append(cmd)

        log_dirs = self._collect_parent_dirs(job["output"], job["error"])
        for directory in log_dirs:
            lines.append(f"mkdir -p {shlex.quote(directory)}")

        lines.append(f"cd {shlex.quote(str(script_dir.resolve()))}")

        launch = build_runtime_launch(runtime_cfg)
        if not launch:
            raise ValueError("submit config missing required runtime launch settings")
        args = run.get("args", [])
        if isinstance(args, list):
            arg_text = " ".join([str(item) for item in args])
#            arg_text = shlex.join([str(item) for item in args])
        else:
            arg_text = str(args).strip()
        command_parts = [launch, run["program"]]
        if arg_text:
            command_parts.append(arg_text)
        lines.extend(["", " ".join(command_parts).rstrip(), ""])
        return "\n".join(lines)

    def _require_config_value(self, section: dict[str, Any], key: str) -> str:
        value = section.get(key)
        if value is None or str(value).strip() == "":
            raise ValueError(f"submit config missing required value: {key}")
        return str(value).strip()

    def _require_process_grid(self, ensemble_cfg: dict[str, Any]) -> list[int]:
        value = ensemble_cfg.get("process_grid")
        if not isinstance(value, list) or len(value) == 0:
            raise ValueError("submit config missing required ensemble.process_grid")

        process_grid: list[int] = []
        for raw in value:
            try:
                dim = int(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"ensemble.process_grid must contain integers: {value}") from exc
            if dim <= 0:
                raise ValueError(f"ensemble.process_grid must contain positive integers: {value}")
            process_grid.append(dim)
        return process_grid

    def _collect_parent_dirs(self, *paths: str) -> list[str]:
        seen: list[str] = []
        for raw in paths:
            parent = str(Path(str(raw)).parent)
            if parent and parent != "." and parent not in seen:
                seen.append(parent)
        return seen
