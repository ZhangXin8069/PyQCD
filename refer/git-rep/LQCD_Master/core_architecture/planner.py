from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import json
import yaml

from utils.config_utils import build_prompt_ensemble_yaml, load_yaml_mapping, resolve_submit_config_presets
from utils.io_utils import extract_json_from_text, write_json, write_text
from utils.prompt_loader import PromptLoader
from utils.skill_utils import (
    SkillContext,
    SkillMessageAssembler,
    SkillRegistry,
    SkillRouter,
    SkillRunner,
    ToolRegistry,
)

REQUIRED_PLAN_KEYS = [
    "task:",
    "ensemble:",
    "physics:",
    "measurement:",
    "output:",
    "extras:",
]


class PlannerAgent:
    def __init__(
        self,
        llm_client: Any,
        tool_client: Any,
        run_dir: Path,
    ):
        self.llm = llm_client
        self.tools = tool_client
        self.run_dir = run_dir
        self.planner_dir = run_dir / "planner"
        self.planner_dir.mkdir(parents=True, exist_ok=True)
        self.submit_config_path = Path("configs/config.yaml")
        self.ensemble_preset_path = Path("configs/ensemble_presets.yaml")
        self.raw_submit_config = load_yaml_mapping(self.submit_config_path)
        self.ensemble_presets = load_yaml_mapping(self.ensemble_preset_path)
        self.submit_config = self._load_config(self.submit_config_path)
        self.fixed_facts_yaml = self._build_fixed_facts_yaml()

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

        self.solve_payload: dict[str, Any] = {}
        self.critique_payload: dict[str, Any] = {}
        self.rewrite_payload: dict[str, Any] = {}
        self.stage_traces: dict[str, list[dict[str, Any]]] = {}
        self._active_stage: str = ""
        self._tool_count_by_stage: dict[str, int] = {}
        self.trajectory: dict[str, Any] = {"iterations": []}
        self._current_iteration: dict[str, Any] | None = None

    def run(self, task: str) -> dict[str, Any]:
        self._begin_iteration(version_label="v1", kind="initial", feedback=None)
        self.solve_payload = self._run_stage(
            stage="solve",
            build_messages=lambda: self._build_stage_user_message(
                stage="planner_solve",
                payload={"task": task, "fixed_facts_yaml": self.fixed_facts_yaml},
                task=task,
            ),
            task=task,
        )

        self.critique_payload = self._run_stage(
            stage="critique",
            build_messages=lambda: self._build_stage_user_message(
                stage="planner_critique",
                payload={"task": task, "solve_payload": self.solve_payload},
                task=task,
            ),
            task=task,
        )
        self.critique_payload = self._normalize_critique_payload(self.critique_payload)

        self.rewrite_payload = self._run_stage(
            stage="rewrite",
            build_messages=lambda: self._build_stage_user_message(
                stage="planner_rewrite",
                payload={
                    "task": task,
                    "solve_payload": self.solve_payload,
                    "critique_payload": self.critique_payload,
                    "fixed_facts_yaml": self.fixed_facts_yaml,
                },
                task=task,
            ),
            task=task,
        )

        output = self._to_planner_output(self.rewrite_payload, fallback=self.solve_payload)
        self._persist_final(output)
        return output

    def rewrite_with_feedback(self, task: str, current: dict[str, Any], feedback: str, version: int) -> dict[str, Any]:
        self._begin_iteration(version_label=f"v{version}", kind="feedback_rewrite", feedback=feedback)

        rewritten_payload = self._run_stage(
            stage=f"rewrite_feedback_v{version}",
            build_messages=lambda: self._build_stage_user_message(
                stage="planner_rewrite",
                payload={
                    "task": task,
                    "solve_payload": {
                        "plan_yaml": current.get("plan_yaml", ""),
                        "summary_md": current.get("summary_md", ""),
                        "citations": current.get("citations", []),
                    },
                    "critique_payload": self.critique_payload,
                    "fixed_facts_yaml": self.fixed_facts_yaml,
                    "feedback": feedback,
                },
                task=task,
            ),
            task=task,
        )
        self.critique_payload = self._normalize_critique_payload(self.critique_payload)
        output = self._to_planner_output(rewritten_payload, fallback={
            "plan_yaml": current.get("plan_yaml", ""),
            "summary_md": current.get("summary_md", ""),
            "citations": current.get("citations", []),
        })
        self.rewrite_payload = rewritten_payload
        self._persist_final(output)
        return output

    def _persist_final(self, output: dict[str, Any]) -> None:
        write_text(self.planner_dir / "plan.yaml", str(output.get("plan_yaml", "")))
        write_text(self.planner_dir / "summary.md", str(output.get("summary_md", "")))
        self._save_trajectory(final_output=output)

    def validate_plan_coverage(self, plan_yaml: str) -> dict[str, Any]:
        missing = [k for k in REQUIRED_PLAN_KEYS if k not in (plan_yaml or "")]
        return {"ok": not missing, "missing": missing}

    def _to_planner_output(self, payload: dict[str, Any], fallback: dict[str, Any] | None = None) -> dict[str, Any]:
        fallback = fallback or {}
        plan_yaml = str(payload.get("plan_yaml") or fallback.get("plan_yaml") or "").strip()
        summary_md = str(payload.get("summary_md") or fallback.get("summary_md") or "").strip()

        citations = payload.get("citations")
        if citations is None:
            citations = fallback.get("citations", [])
        elif not isinstance(citations, list):
            citations = []

        parsed_urls = self._collect_verified_citation_urls()
        filtered_citations: list[str] = []
        for raw in citations:
            url = str(raw).strip()
            if url and url in parsed_urls and url not in filtered_citations:
                filtered_citations.append(url)
        
        return {
            "plan_yaml": plan_yaml,
            "summary_md": summary_md,
            "citations": filtered_citations,
        }

    def _collect_verified_citation_urls(self) -> set[str]:
        urls: set[str] = set()
        for trace in self.stage_traces.values():
            if not isinstance(trace, list):
                continue
            for item in trace:
                if not isinstance(item, dict):
                    continue
                if str(item.get("tool_name", "")) != "web_parse":
                    continue
                result = item.get("tool_result", {})
                if not isinstance(result, dict):
                    continue
                if result.get("error"):
                    continue
                url = str(result.get("url", "")).strip()
                if url:
                    urls.add(url)
        return urls

    def _normalize_critique_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        issues = payload.get("issues")
        if not isinstance(issues, list):
            issues = []

        risks = payload.get("risks")
        if not isinstance(risks, list):
            risks = []

        revision_instructions = payload.get("revision_instructions")
        if not isinstance(revision_instructions, list):
            revision_instructions = []

        return {
            **payload,
            "issues": [str(item) for item in issues if item],
            "risks": [str(item) for item in risks if item],
            "revision_instructions": [str(item) for item in revision_instructions if item],
        }

    def _load_config(self, path: Path) -> dict[str, Any]:
        data = load_yaml_mapping(path)
        return resolve_submit_config_presets(data, self.ensemble_presets)

    def _build_fixed_facts_yaml(self) -> str:
        return build_prompt_ensemble_yaml(self.raw_submit_config, self.ensemble_presets)

    def _run_stage(
        self,
        *,
        stage: str,
        build_messages: Any,
        task: str,
    ) -> dict[str, Any]:
        self._active_stage = stage
        self._tool_count_by_stage.setdefault(stage, 0)
        self._print_stage_start(stage)

        user_message = build_messages()
        payload, trace = self._chat_json_autonomous(user_message, stage=stage, task=task)
        self.stage_traces[stage] = trace
        self._append_stage_record(stage, payload, trace, mode="autonomous_tools", search_payload={})
        self._print_stage_done(stage, len(trace))
        return payload

    def _chat_json_autonomous(
        self,
        user_message: str,
        *,
        stage: str,
        task: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        context = SkillContext(
            stage=self._resolve_route_stage(stage),
            task=task,
            payload={},
        )
        payload, trace, _ = self.skill_runner.run(
            context,
            self._load_system_prompt(self._resolve_route_stage(stage)),
            user_message,
            fallback_tools=["web_search", "web_parse"],
        )
        if not isinstance(payload, dict):
            payload = {}
        if "raw_text" in payload:
            payload = extract_json_from_text(str(payload.get("raw_text", "")))

        return payload, trace

    def _load_system_prompt(self, stage: str) -> str:
        mapping = {
            "planner_solve": "planner/solve_system.md",
            "planner_critique": "planner/critique_system.md",
            "planner_rewrite": "planner/rewrite_system.md",
        }
        return self.prompt_loader.load(mapping[stage])

    def _resolve_route_stage(self, stage: str) -> str:
        if stage == "solve":
            return "planner_solve"
        if stage == "critique":
            return "planner_critique"
        if stage.startswith("rewrite"):
            return "planner_rewrite"
        raise ValueError(f"unknown planner route stage: {stage}")

    def _build_stage_user_message(self, *, stage: str, payload: dict[str, Any], task: str) -> str:
        del task
        if stage == "planner_solve":
            return (
                f"{self.prompt_loader.load('planner/solve_user.md')}\n\n"
                f"Natural-language task:\n{payload.get('task', '')}\n\n"
                "Ensemble information (must be written into the `ensemble:` section of `plan_yaml` using these values directly where applicable):\n"
                f"{payload.get('fixed_facts_yaml') or '(none)'}\n\n"
                "Plan template (the ensemble block is intentionally omitted here; you must insert it from the provided ensemble information):\n"
                f"{self.prompt_loader.load('planner/plan_template.yaml')}\n"
            )
        if stage == "planner_critique":
            return (
                f"{self.prompt_loader.load('planner/critique_user.md')}\n\n"
                f"Original task:\n{payload.get('task', '')}\n\n"
                "Solve-stage output:\n"
                f"{json.dumps(payload.get('solve_payload', {}), ensure_ascii=False, indent=2)}\n"
            )
        if stage == "planner_rewrite":
            feedback = str(payload.get("feedback") or "").strip()
            feedback_block = f"\nUser feedback:\n{feedback}\n" if feedback else ""
            return (
                f"{self.prompt_loader.load('planner/rewrite_user.md')}\n\n"
                f"Original task:\n{payload.get('task', '')}\n\n"
                "Previous plan:\n"
                f"{json.dumps(payload.get('solve_payload', {}), ensure_ascii=False, indent=2)}\n\n"
                "Critique output:\n"
                f"{json.dumps(payload.get('critique_payload', {}), ensure_ascii=False, indent=2)}\n"
                f"{feedback_block}\n"
                "Ensemble information (the `ensemble:` section of the revised `plan_yaml` must stay consistent with these values):\n"
                f"{payload.get('fixed_facts_yaml') or '(none)'}\n\n"
                "Plan template (the ensemble block is intentionally omitted here; you must keep or restore it from the provided ensemble information):\n"
                f"{self.prompt_loader.load('planner/plan_template.yaml')}\n"
            )
        raise ValueError(f"unknown planner stage: {stage}")

    def _print_stage_start(self, stage: str) -> None:
        print(f"[Planner] [{stage}] stage start! 🚀")

    def _print_stage_done(self, stage: str, tool_calls: int, fallback: bool = False) -> None:
        del fallback
        print(f"[Planner] [{stage}] stage done! ✅ | tool_calls={tool_calls}")

    def _print_tool_call(self, stage: str, tool_name: str) -> None:
        print(f"[Planner] [{stage}] tool call: {tool_name} 🔧")

    def _begin_iteration(self, *, version_label: str, kind: str, feedback: str | None) -> None:
        iteration = {
            "version": version_label,
            "kind": kind,
            "feedback": feedback or "",
            "started_at": datetime.now().isoformat(),
            "stages": [],
        }
        self.trajectory["iterations"].append(iteration)
        self._current_iteration = iteration

    def _append_stage_record(
        self,
        stage: str,
        payload: dict[str, Any],
        trace: list[dict[str, Any]],
        *,
        mode: str,
        search_payload: dict[str, Any],
    ) -> None:
        if self._current_iteration is None:
            self._begin_iteration(version_label="v0", kind="implicit", feedback=None)
        stage_record = {
            "stage": stage,
            "mode": mode,
            "tool_calls": len(trace),
            "tool_trace": trace,
            "search": search_payload,
            "output": payload,
            "timestamp": datetime.now().isoformat(),
        }
        self._current_iteration["stages"].append(stage_record)
        self._save_trajectory()

    def _save_trajectory(self, final_output: dict[str, Any] | None = None) -> None:
        data = dict(self.trajectory)
        if final_output is not None:
            data["latest_output"] = final_output
            data["updated_at"] = datetime.now().isoformat()
        write_json(self.planner_dir / "trajectory.json", data)

    def debug_state(self) -> dict[str, Any]:
        return {
            "solve_payload": self.solve_payload,
            "critique_payload": self.critique_payload,
            "rewrite_payload": self.rewrite_payload,
            "stage_traces": self.stage_traces,
        }
