from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from utils.io_utils import extract_json_from_text


class LLMAPIError(Exception):
    """Raised when the LLM API call fails after all retries."""
    def __init__(self, message: str, status_code: int | None = None, response: str = ""):
        self.status_code = status_code
        self.response = response
        super().__init__(message)


@dataclass(frozen=True)
class LoadedSkill:
    name: str
    description: str
    root_dir: Path
    skill_file: Path
    body: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def references_dir(self) -> Path:
        return self.root_dir / "reference"

    @property
    def scripts_dir(self) -> Path:
        return self.root_dir / "scripts"


@dataclass
class SkillSelection:
    knowledge_skills: list[str] = field(default_factory=list)
    route_mode: str = "unknown"
    raw_response: dict[str, Any] = field(default_factory=dict)

    @property
    def all_skill_names(self) -> list[str]:
        seen: list[str] = []
        for name in self.knowledge_skills:
            if name and name not in seen:
                seen.append(name)
        return seen


@dataclass
class SkillContext:
    stage: str
    task: str
    payload: dict[str, Any] = field(default_factory=dict)


class SkillRegistry:
    def __init__(self, skills_root: Path):
        self.skills_root = skills_root
        self._skills: dict[str, LoadedSkill] = {}

    def load(self) -> None:
        self._skills = {}
        if not self.skills_root.exists():
            return
        for skill_file in sorted(self.skills_root.rglob("SKILL.md")):
            skill = self._load_skill(skill_file)
            if skill.name in self._skills:
                existing = self._skills[skill.name].skill_file
                raise ValueError(f"duplicate skill name '{skill.name}': {existing} and {skill_file}")
            self._skills[skill.name] = skill

    def all_skills(self) -> list[LoadedSkill]:
        return [self._skills[name] for name in sorted(self._skills)]

    def get(self, name: str) -> LoadedSkill:
        if name not in self._skills:
            raise KeyError(f"skill not found: {name}")
        return self._skills[name]

    def has(self, name: str) -> bool:
        return name in self._skills

    def names(self) -> list[str]:
        return sorted(self._skills)

    def _load_skill(self, skill_file: Path) -> LoadedSkill:
        text = skill_file.read_text(encoding="utf-8")
        metadata, body = self._parse_skill_markdown(text)
        name = str(metadata.get("name", "")).strip()
        description = str(metadata.get("description", "")).strip()
        if not name:
            raise ValueError(f"skill missing frontmatter name: {skill_file}")
        if not description:
            raise ValueError(f"skill missing frontmatter description: {skill_file}")
        return LoadedSkill(
            name=name,
            description=description,
            root_dir=skill_file.parent,
            skill_file=skill_file,
            body=body.strip(),
            metadata=metadata,
        )

    def _parse_skill_markdown(self, text: str) -> tuple[dict[str, Any], str]:
        raw = text.lstrip()
        if not raw.startswith("---"):
            raise ValueError("SKILL.md must start with YAML frontmatter")
        lines = raw.splitlines()
        if not lines or lines[0].strip() != "---":
            raise ValueError("SKILL.md frontmatter is malformed")
        closing_index = None
        for index, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                closing_index = index
                break
        if closing_index is None:
            raise ValueError("SKILL.md frontmatter is malformed")
        frontmatter_text = "\n".join(lines[1:closing_index])
        body = "\n".join(lines[closing_index + 1 :])
        metadata = yaml.safe_load(frontmatter_text) or {}
        if not isinstance(metadata, dict):
            raise ValueError("SKILL.md frontmatter must be a mapping")
        return dict(metadata), body


class SkillRouter:
    def __init__(self, registry: SkillRegistry, config_path: Path):
        self.registry = registry
        self.config_path = config_path
        self.config = self._load_config(config_path)

    def reload(self) -> None:
        self.config = self._load_config(self.config_path)

    def candidate_skills(self, context: SkillContext) -> list[LoadedSkill]:
        disabled = set(self._sequence(self.config.get("disabled_skills")))
        candidates: list[LoadedSkill] = []
        for skill_name in self._candidate_skill_names(context):
            if not skill_name or skill_name in disabled:
                continue
            if not self.registry.has(skill_name):
                continue
            candidates.append(self.registry.get(skill_name))
        return candidates

    def tools_for_skills(self, skill_names: list[str], fallback_tools: list[str] | None = None) -> list[str]:
        tool_map = self._mapping("skill_tools")
        tools: list[str] = []
        for tool_name in fallback_tools or []:
            if tool_name not in tools:
                tools.append(tool_name)
        for skill_name in skill_names:
            for tool_name in self._sequence(tool_map.get(skill_name)):
                if tool_name not in tools:
                    tools.append(tool_name)
        return tools

    def forced_skills_for_stage(self, stage: str) -> list[str]:
        forced_map = self._mapping("stage_forced_skills")
        return self._sequence(forced_map.get(self._stage_group(stage)))

    def routing_guidance_for_stage(self, stage: str) -> str:
        stage_group = self._stage_group(stage)
        path = Path("prompts") / stage_group / "skill_routing.md"
        if not path.exists() or not path.is_file():
            return ""
        return path.read_text(encoding="utf-8").strip()

    def _candidate_skill_names(self, context: SkillContext) -> list[str]:
        configured_enabled = self._sequence(self.config.get("enabled_skills")) or self.registry.names()
        names = list(configured_enabled)
        for skill_name in self.forced_skills_for_stage(context.stage):
            if skill_name not in names:
                names.append(skill_name)
        return names

    def _stage_group(self, stage: str) -> str:
        if stage.startswith("planner"):
            return "planner"
        if stage.startswith("executor"):
            return "executor"
        return stage

    def _load_config(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return {}
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = yaml.safe_load(text) or {}
        return data if isinstance(data, dict) else {}

    def _mapping(self, key: str) -> dict[str, Any]:
        value = self.config.get(key)
        return value if isinstance(value, dict) else {}

    def _sequence(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []


class SkillMessageAssembler:
    def __init__(self, max_inline_reference_chars: int = 24000):
        self.max_inline_reference_chars = max_inline_reference_chars

    def build_system_message(self, base_system_prompt: str, knowledge_skills: list[LoadedSkill]) -> str:
        blocks = [base_system_prompt.strip()]
        for skill in knowledge_skills:
            blocks.append(self._skill_block("Knowledge skill", skill))
        return "\n\n".join(block for block in blocks if block).strip()

    def build_messages(
        self,
        *,
        base_system_prompt: str,
        knowledge_skills: list[LoadedSkill],
        user_message: str,
    ) -> list[dict[str, str]]:
        system = self.build_system_message(base_system_prompt, knowledge_skills)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message.strip()},
        ]

    def _skill_block(self, title: str, skill: LoadedSkill) -> str:
        sections = [
            f"[{title}] {skill.name}",
            f"Description: {skill.description}",
            skill.body.strip(),
        ]
        references_text = self._inline_references(skill.references_dir)
        if references_text:
            sections.append(references_text)
        return "\n\n".join(part for part in sections if part).strip()

    def _inline_references(self, references_dir: Path) -> str:
        if not references_dir.exists() or not references_dir.is_dir():
            return ""
        chunks: list[str] = []
        used = 0
        for path in sorted(references_dir.rglob("*")):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                continue
            snippet = text.strip()
            remaining = self.max_inline_reference_chars - used
            if remaining <= 0:
                break
            if len(snippet) > remaining:
                snippet = snippet[:remaining].rstrip() + "\n...[truncated]"
            chunks.append(f"[Reference: {path.name}]\n{snippet}")
            used += len(snippet)
        return "\n\n".join(chunks)


class ToolRegistry:
    def __init__(self, tool_client: Any | None):
        self.tool_client = tool_client

    def build_openai_tools(self, tool_names: list[str]) -> list[dict[str, Any]]:
        builders = {
            "web_search": self._web_search_schema,
            "web_parse": self._web_parse_schema,
            "execute": self._execute_schema,
            "generate_einsum": self._generate_einsum_schema,
        }
        tools: list[dict[str, Any]] = []
        for name in tool_names:
            builder = builders.get(name)
            if builder:
                tools.append(builder())
        return tools

    def handle(self, tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
        if not self.tool_client:
            return {"error": "tool_client_unavailable", "tool": tool_name}
        if tool_name == "web_search":
            query = str(tool_args.get("query", "")).strip()
            top_k_raw = tool_args.get("top_k", 5)
            top_k = top_k_raw if isinstance(top_k_raw, int) else (int(top_k_raw) if str(top_k_raw).isdigit() else 5)
            return self.tool_client.web_search(query, top_k=top_k)
        if tool_name == "web_parse":
            link = str(tool_args.get("link", "")).strip()
            user_prompt = str(tool_args.get("user_prompt", "")).strip()
            llm = str(tool_args.get("llm", "gpt-4o")).strip() or "gpt-4o"
            return self.tool_client.web_parse(link, user_prompt, llm=llm)
        if tool_name == "execute":
            code = str(tool_args.get("code", ""))
            timeout_raw = tool_args.get("timeout", 120)
            timeout = timeout_raw if isinstance(timeout_raw, int) else (int(timeout_raw) if str(timeout_raw).isdigit() else 120)
            session_id = str(tool_args.get("session_id", "local") or "local")
            return self.tool_client.execute(code=code, timeout=timeout, session_id=session_id)
        if tool_name == "generate_einsum":
            return self._handle_generate_einsum(tool_args)
        return {"error": "unknown_tool", "tool": tool_name, "args": tool_args}

    _GMAP = {
        "gamma1": "g1", "gamma2": "g2", "gamma3": "g3",
        "gamma4": "g4", "gamma5": "g5",
        "gamma_x": "g1", "gamma_y": "g2", "gamma_z": "g3", "gamma_t": "g4",
        "gx": "g1", "gy": "g2", "gz": "g3", "gt": "g4",
    }

    def _handle_generate_einsum(self, tool_args: dict[str, Any]) -> dict[str, Any]:
        from utils.generate_einsum.hadron_operator import meson_operator as _mk_meson
        from utils.generate_einsum.hadron_operator import baryon_operator as _mk_baryon
        from utils.generate_einsum.two_pt.contract import wick_contract_2pt as _wick
        from utils.generate_einsum.two_pt.codegen import pyquda_format_contract as _fmt_code
        etype = str(tool_args.get("type", "")).strip()
        try:
            if etype == "meson_2pt":
                antiquark = str(tool_args.get("antiquark", "u"))
                quark     = str(tool_args.get("quark", "d"))
                gamma_snk = self._GMAP.get(str(tool_args.get("gamma_snk", "gamma5")), "g5")
                gamma_src = self._GMAP.get(str(tool_args.get("gamma_src", "gamma5")), "g5")

                snk = _mk_meson(antiquark, quark, gamma_snk)
                src = _mk_meson(antiquark, quark, gamma_src)
                r = _wick(snk, src, "I")
                if not r.terms:
                    return {"error": "no contraction terms generated"}
                code_lines = [_fmt_code(t) for t in r.terms]
                terms_info = []
                for t in r.terms:
                    terms_info.append({
                        "sign": t.coefficient,
                        "subs": t.einsum_subs,
                        "operands": list(t.operands),
                    })
                return {
                    "einsum_type": "meson_2pt",
                    "n_terms": len(r.terms),
                    "terms": terms_info,
                    "code": "# FROM generate_einsum (meson_2pt)\n" + "\n".join(code_lines),
                }
            if etype == "baryon_2pt":
                quark_a = str(tool_args.get("quark_a", "u"))
                quark_b = str(tool_args.get("quark_b", "d"))
                quark_c = str(tool_args.get("quark_c", "u"))
                projector = str(tool_args.get("projector", "P_plus"))
                diquark_gamma = str(tool_args.get("diquark_gamma", "Cg5"))
                c_gamma = str(tool_args.get("c_gamma", "I4"))

                snk = _mk_baryon(quark_a, quark_b, quark_c, diquark_gamma, c_gamma)
                src = _mk_baryon(quark_a, quark_b, quark_c, diquark_gamma, c_gamma)
                r = _wick(snk, src, projector)
                code_lines = [_fmt_code(t) for t in r.terms]
                topos = {}
                for i, t in enumerate(r.terms):
                    props = [o for o in t.operands if "prop_" in o]
                    topos[f"2pt_topo{i}"] = {
                        "einsum": t.einsum_subs,
                        "sign": t.coefficient,
                        "args": [{"var": p, "dag": False} for p in props],
                    }
                return {"einsum_type": "baryon_2pt",
                        "topologies": topos, "n_topologies": len(topos),
                        "code": "# FROM generate_einsum (baryon_2pt)\n" + "\n".join(code_lines)}
            if etype == "meson_3pt":
                return self._handle_meson_3pt(tool_args)
            if etype == "baryon_3pt":
                return self._handle_baryon_3pt(tool_args)
            if etype == "multi_hadron_2pt":
                from utils.generate_einsum.two_pt.codegen_multi_hadron import (
                    gen_code_2pt as _gen_multi)
                specs = tool_args.get("specs", [])
                out_name = tool_args.get("out_name", "output")
                if not specs:
                    return {"error": "multi_hadron_2pt requires 'specs' list"}
                sink_code = _gen_multi(specs, specs, snk_name=out_name)
                sink_file = f"sink_{out_name}.py"
                sink_path = os.path.join(os.getcwd(), sink_file)
                with open(sink_path, "w") as f:
                    f.write(sink_code)
                return {
                    "einsum_type": "multi_hadron_2pt",
                    "n_hadrons": len(specs),
                    "n_topologies": sink_code.count("contract("),
                    "sink_file": sink_file,
                    "sink_path": sink_path,
                    "status": "sink_block_written",
                }
            return {"error": f"Unknown generate_einsum type: {etype}. "
                    f"Available: meson_2pt, baryon_2pt, multi_hadron_2pt, meson_3pt, baryon_3pt"}
        except Exception as e:
            return {"error": f"generate_einsum failed: {e}", "tool": "generate_einsum"}

    def _handle_meson_3pt(self, tool_args: dict) -> dict:
        """Generate complete meson 3pt PyQUDA code via codegen_sep."""
        from utils.generate_einsum.hadron_operator import meson_operator, current_operator
        from utils.generate_einsum import gen_meson_3pt_code

        src_quark = str(tool_args.get("src_quark", "c"))
        src_antiquark = str(tool_args.get("src_antiquark", "u"))
        snk_quark = str(tool_args.get("snk_quark", "s"))
        snk_antiquark = str(tool_args.get("sink_antiquark", "u"))
        cur_quark = str(tool_args.get("current_quark", "c"))
        cur_antiquark = str(tool_args.get("current_antiquark", "s"))

        gamma_snk = self._GMAP.get(str(tool_args.get("gamma_snk", "gamma5")), "g5")
        gamma_src = self._GMAP.get(str(tool_args.get("gamma_src", "gamma5")), "g5")
        gamma_cur = self._GMAP.get(str(tool_args.get("gamma_cur", "gamma1")), "g1")
        tseq = str(tool_args.get("tseq", "8"))

        src = meson_operator(src_antiquark, src_quark, gamma_src)
        snk = meson_operator(snk_antiquark, snk_quark, gamma_snk)
        cur = current_operator(cur_antiquark, cur_quark, gamma_cur)

        code = gen_meson_3pt_code(snk, src, cur, tseq)
        print(code)
        return {
            "ok": True,
            "einsum_type": "meson_3pt",
            "code": "# FROM generate_einsum (meson_3pt)\n" + code,
        }

    def _handle_baryon_3pt(self, tool_args: dict) -> dict:
        """Generate baryon 3pt sink-block + seq-source + final contraction."""
        from utils.generate_einsum.hadron_operator import baryon_operator, current_operator
        from utils.generate_einsum.three_pt.codegen_baryon import gen_baryon_3pt_code

        src_a = str(tool_args.get("src_a", "u"))
        src_b = str(tool_args.get("src_b", "d"))
        src_c = str(tool_args.get("src_c", "s"))
        snk_a = str(tool_args.get("snk_a", "u"))
        snk_b = str(tool_args.get("snk_b", "d"))
        snk_c = str(tool_args.get("snk_c", "s"))
        cur_quark = str(tool_args.get("current_quark", ""))
        cur_antiquark = str(tool_args.get("current_antiquark", ""))
        cur_gamma = self._GMAP.get(
            str(tool_args.get("current_gamma", "gamma1")), "g1")
        projector = str(tool_args.get("projector", "P_plus"))
        diquark_gamma_snk = str(tool_args.get("diquark_gamma_snk", "Cg5"))
        diquark_gamma_src = str(tool_args.get("diquark_gamma_src", "Cg5"))
        c_gamma_snk = str(tool_args.get("c_gamma_snk", "I4"))
        c_gamma_src = str(tool_args.get("c_gamma_src", "I4"))
        tseq = str(tool_args.get("tseq", "t_sep"))

        src = baryon_operator(src_a, src_b, src_c, diquark_gamma_src, c_gamma_src)
        snk = baryon_operator(snk_a, snk_b, snk_c, diquark_gamma_snk, c_gamma_snk)
        cur = current_operator(cur_antiquark, cur_quark, cur_gamma)

        code = gen_baryon_3pt_code(snk, src, cur, tseq, projector)

        return {
            "ok": True,
            "einsum_type": "baryon_3pt",
            "code": "# FROM generate_einsum (baryon_3pt)\n" + code,
        }

    def _web_search_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for relevant information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            },
        }

    def _web_parse_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "web_parse",
                "description": "Parse webpage content through built-in parser.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "link": {"type": "string"},
                        "user_prompt": {"type": "string"},
                        "llm": {"type": "string"},
                    },
                    "required": ["link", "user_prompt"],
                },
            },
        }

    def _generate_einsum_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "generate_einsum",
                "description": (
                    "Generate validated einsum contraction strings for hadronic"
                    " correlators. Use this INSTEAD of writing einsum by hand.\n"
                    "Types:\n"
                    "  meson_2pt - requires: antiquark, quark, gamma_snk, gamma_src\n"
                    "  baryon_2pt - requires: quark_a, quark_b, quark_c, projector,"
                    " diquark_gamma, c_gamma\n"
                    "  multi_hadron_2pt - requires: specs list (see 'specs' param)\n"
                    "  meson_3pt - requires: src_antiquark, src_quark,"
                    " sink_antiquark, snk_quark, current_quark, current_antiquark, tseq\n"
                    "  baryon_3pt - requires: src_a, src_b, src_c, snk_a, snk_b, snk_c,"
                    " current_quark, current_antiquark, tseq"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": [
                                "meson_2pt",
                                "baryon_2pt",
                                "multi_hadron_2pt",
                                "meson_3pt",
                                "baryon_3pt",
                            ],
                            "description": "Correlator type. For meson_2pt: gamma_snk AS-IS, gamma_src gets Dirac adjoint (from O_src^dag).",
                        },
                        "antiquark": {
                            "type": "string",
                            "description": "Antiquark flavor (for meson_2pt).",
                        },
                        "quark": {
                            "type": "string",
                            "description": "Quark flavor (for meson_2pt).",
                        },
                        "gamma_snk": {
                            "type": "string",
                            "description": "Gamma at SINK (t, spatial x): \\bar{q} * Gamma_snk * q. "
                                           "Default gamma5. Used AS-IS. "
                                           "e.g. gamma0, gamma5, gamma1.",
                        },
                        "gamma_src": {
                            "type": "string",
                            "description": "Gamma at SOURCE (t=0): \\bar{q} * Gamma_src * q. "
                                           "Default gamma5. Gets Dirac adjoint (gamma4 @ gamma_src^dag @ gamma4) "
                                           "because O_src^dag = \\bar{q} * (g4 @ G_src^dag @ g4) * q. "
                                           "e.g. gamma5, gamma0.",
                        },
                        "quark_a": {
                            "type": "string",
                            "description": "Baryon quark a flavor (for baryon_2pt).",
                        },
                        "quark_b": {
                            "type": "string",
                            "description": "Baryon quark b flavor (for baryon_2pt).",
                        },
                        "quark_c": {
                            "type": "string",
                            "description": "Baryon quark c flavor (for baryon_2pt).",
                        },
                        "projector": {
                            "type": "string",
                            "description": "Spin projector: P_plus or P_minus (for baryon_2pt).",
                        },
                        "diquark_gamma": {
                            "type": "string",
                            "description": "Diquark gamma matrix (for baryon_2pt). Default Cg5.",
                        },
                        "c_gamma": {
                            "type": "string",
                            "description": "c-quark gamma matrix (for baryon_2pt). Default I4.",
                        },
                        "src_a": {
                            "type": "string",
                            "description": "Source baryon quark a flavor (for baryon_3pt).",
                        },
                        "src_b": {
                            "type": "string",
                            "description": "Source baryon quark b flavor (for baryon_3pt).",
                        },
                        "src_c": {
                            "type": "string",
                            "description": "Source baryon quark c flavor (for baryon_3pt).",
                        },
                        "snk_a": {
                            "type": "string",
                            "description": "Sink baryon quark a flavor (for baryon_3pt).",
                        },
                        "snk_b": {
                            "type": "string",
                            "description": "Sink baryon quark b flavor (for baryon_3pt).",
                        },
                        "snk_c": {
                            "type": "string",
                            "description": "Sink baryon quark c flavor (for baryon_3pt).",
                        },
                        "current_quark": {
                            "type": "string",
                            "description": "Flavor annihilated by the current (for baryon_3pt).",
                        },
                        "current_antiquark": {
                            "type": "string",
                            "description": "Flavor created by the current (for baryon_3pt).",
                        },
                        "current_gamma": {
                            "type": "string",
                            "description": "Dirac matrix for current (default gamma1). Accepts gamma1-5, gamma_x/y/z/t, gx/gy/gz/gt.",
                        },
                        "projector": {
                            "type": "string",
                            "description": "Spin projector (for baryon_3pt). P_plus or P_minus.",
                        },
                        "diquark_gamma_snk": {
                            "type": "string",
                            "description": "Diquark gamma at sink (for baryon_3pt). Default Cg5.",
                        },
                        "diquark_gamma_src": {
                            "type": "string",
                            "description": "Diquark gamma at source (for baryon_3pt). Default Cg5.",
                        },
                        "c_gamma_snk": {
                            "type": "string",
                            "description": "c-quark gamma at sink (for baryon_3pt). Default I4.",
                        },
                        "c_gamma_src": {
                            "type": "string",
                            "description": "c-quark gamma at source (for baryon_3pt). Default I4.",
                        },
                        "tseq": {
                            "type": "string",
                            "description": "Time separation (for baryon_3pt).",
                        },
                        # ---- multi_hadron_2pt params ----
                        "specs": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "List of hadron specs for multi_hadron_2pt. Each spec: {type, flavors, gamma (for mesons), projector (for baryons)}. E.g. [{\"type\": \"baryon\", \"flavors\": [\"u\",\"d\",\"u\"], \"projector\": \"P_plus\"}].",
                        },
                        "out_name": {
                            "type": "string",
                            "description": "Correlator label (for multi_hadron_2pt). The tool writes sink_{out_name}.py to the sink_dir.",
                        },
                        "sink_dir": {
                            "type": "string",
                            "description": "Directory to write the sink file (for multi_hadron_2pt). Default: current working directory.",
                        },
                        # ---- meson_3pt params ----
                        "src_antiquark": {
                            "type": "string",
                            "description": "Source antiquark flavor (for meson_3pt).",
                        },
                        "src_quark": {
                            "type": "string",
                            "description": "Source quark flavor (for meson_3pt). The particle annihilated by the current.",
                        },
                        "sink_antiquark": {
                            "type": "string",
                            "description": "Sink antiquark flavor (for meson_3pt). Must match source antiquark (spectator).",
                        },
                        "snk_quark": {
                            "type": "string",
                            "description": "Sink quark flavor (for meson_3pt). The particle created by the current.",
                        },
                        "current_quark": {
                            "type": "string",
                            "description": "Flavor ANNIHILATED by the current (incoming). Must match src_quark. IMPORTANT: this is the OPPOSITE of the task description's q̄·Γ·q convention — e.g. for c->s transition: current_quark='c', current_antiquark='s'.",
                        },
                        "current_antiquark": {
                            "type": "string",
                            "description": "Flavor CREATED by the current (outgoing). Must match snk_quark. IMPORTANT: this is the OPPOSITE of the task description's q̄·Γ·q convention — e.g. for c->s transition: current_antiquark='s', current_quark='c'.",
                        },
                        "gamma_snk": {
                            "type": "string",
                            "description": "Gamma matrix at sink (for meson_3pt). Default gamma5.",
                        },
                        "gamma_src": {
                            "type": "string",
                            "description": "Gamma matrix at source (for meson_3pt). Default gamma5.",
                        },
                        "gamma_cur": {
                            "type": "string",
                            "description": "Dirac matrix for the current (for meson_3pt). Accepts gamma1-5, gamma_x/y/z/t, gx/gy/gz/gt.",
                        },
                        "tseq": {
                            "type": "string",
                            "description": "Time separation for sequential source (for meson_3pt). Default 8.",
                        },
                    },
                    "required": ["type"],
                },
            },
        }

    def _execute_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "execute",
                "description": "Run Python code for local evidence inspection.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "timeout": {"type": "integer"},
                        "session_id": {"type": "string"},
                    },
                    "required": ["code"],
                },
            },
        }


class SkillRunner:
    def __init__(
        self,
        *,
        llm_client: Any,
        registry: SkillRegistry,
        router: SkillRouter,
        assembler: SkillMessageAssembler,
        tool_registry: ToolRegistry,
    ):
        self.llm = llm_client
        self.registry = registry
        self.router = router
        self.assembler = assembler
        self.tool_registry = tool_registry

    def run(
        self,
        context: SkillContext,
        base_system_prompt: str,
        user_message: str,
        fallback_tools: list[str] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], SkillSelection]:
        if not getattr(self.llm, "enabled", False) or not getattr(self.llm, "_client", None):
            return {}, [], SkillSelection(route_mode="semantic_llm_unavailable", raw_response={"knowledge_skills": []})

        selection = self._select_skills(context)
        knowledge_skills = [self.registry.get(name) for name in selection.knowledge_skills if self.registry.has(name)]
        tool_names = self.router.tools_for_skills([skill.name for skill in knowledge_skills], fallback_tools=fallback_tools)
        self._print_skill_run(context, knowledge_skills, tool_names)

        system_message = self.assembler.build_system_message(base_system_prompt, knowledge_skills)
        conversation: list[dict[str, Any]] = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message.strip()},
        ]
        tool_trace: list[dict[str, Any]] = []

        for _ in range(8):
            tools = self.tool_registry.build_openai_tools(tool_names)
            msg = self._create_completion_message(conversation, tools)
            if msg is None:
                return {}, tool_trace, selection

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": msg.content or "",
            }
            if hasattr(msg, "reasoning_content") and msg.reasoning_content:
                assistant_msg["reasoning_content"] = msg.reasoning_content
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                assistant_msg["tool_calls"] = []
                for tc in tool_calls:
                    fn_name = tc.function.name if tc.function else ""
                    fn_args_str = tc.function.arguments if tc.function else "{}"
                    assistant_msg["tool_calls"].append(
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": fn_name,
                                "arguments": fn_args_str,
                            },
                        }
                    )
            conversation.append(assistant_msg)

            if not tool_calls:
                payload = self._parse_payload_from_text(msg.content or "")
                return payload, tool_trace, selection

            for tc in tool_calls:
                fn_name = tc.function.name if tc.function else ""
                fn_args_str = tc.function.arguments if tc.function else "{}"
                try:
                    fn_args = json.loads(fn_args_str) if fn_args_str else {}
                    if not isinstance(fn_args, dict):
                        fn_args = {}
                except json.JSONDecodeError:
                    fn_args = {}

                self._print_tool_call(context, fn_name)
                try:
                    result = self.tool_registry.handle(fn_name, fn_args)
                except Exception as e:
                    result = {"error": str(e), "tool": fn_name}

                tool_trace.append(
                    {
                        "tool_name": fn_name,
                        "tool_args": fn_args,
                        "tool_result": result,
                    }
                )
                def _json_fallback(o):
                    if isinstance(o, bytes):
                        return o.decode('utf-8', errors='replace')
                    return repr(o)
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False, default=_json_fallback),
                    }
                )

        return {"error": "max_turns_exceeded", "raw_text": ""}, tool_trace, selection

    def _print_skill_run(
        self,
        context: SkillContext,
        knowledge_skills: list[LoadedSkill],
        tool_names: list[str],
    ) -> None:
        del tool_names
        selected = [skill.name for skill in knowledge_skills]
        if not selected:
            return
        selected_text = ", ".join(selected)
        owner = self._stage_owner_label(context.stage)
        print(
            f"[{owner}] [{context.stage}] skills selected! ✅ | skills={selected_text}"
        )

    def _print_tool_call(self, context: SkillContext, tool_name: str) -> None:
        owner = self._stage_owner_label(context.stage)
        print(f"[{owner}] [{context.stage}] tool call: {tool_name} 🔧")

    def _stage_owner_label(self, stage: str) -> str:
        if stage.startswith("planner"):
            return "Planner"
        if stage.startswith("executor"):
            return "Executor"
        return "Skill"

    def _select_skills(self, context: SkillContext) -> SkillSelection:
        candidates = self.router.candidate_skills(context)
        if not candidates:
            return SkillSelection(
                knowledge_skills=self._filter_existing_skill_names(self.router.forced_skills_for_stage(context.stage)),
                route_mode="semantic_llm",
                raw_response={"knowledge_skills": []},
            )
        if not getattr(self.llm, "enabled", False):
            return SkillSelection(
                knowledge_skills=self._filter_existing_skill_names(self.router.forced_skills_for_stage(context.stage)),
                route_mode="semantic_llm_unavailable",
                raw_response={"knowledge_skills": []},
            )

        raw_response = self.llm.chat_json(
            self._build_selector_messages(context, candidates),
            stream=False,
            temperature=0,
            max_tokens=4000,
            model=self.llm.model,
        )
        selected_names, route_mode = self._normalize_selected_names(raw_response, candidates)
        forced_names = self._filter_existing_skill_names(self.router.forced_skills_for_stage(context.stage))
        for name in forced_names:
            if name not in selected_names:
                selected_names.append(name)
        return SkillSelection(
            knowledge_skills=selected_names,
            route_mode=route_mode,
            raw_response=raw_response if isinstance(raw_response, dict) else {},
        )

    def _build_selector_messages(
        self,
        context: SkillContext,
        candidates: list[LoadedSkill],
    ) -> list[dict[str, str]]:
        candidate_lines = [
            json.dumps({"name": skill.name, "description": skill.description}, ensure_ascii=False)
            for skill in candidates
        ]
        payload_json = json.dumps(context.payload, ensure_ascii=False, indent=2)
        stage_guidance = self.router.routing_guidance_for_stage(context.stage) or "(none)"
        forced_skills = self.router.forced_skills_for_stage(context.stage)
        user_content = (
            "Task context:\n"
            f"- stage: {context.stage}\n"
            f"- task: {context.task}\n"
            f"- payload: {payload_json}\n\n"
            "Stage-specific routing guidance:\n"
            f"{stage_guidance}\n\n"
            "Skills that will be force-enabled for this stage regardless of your selection:\n"
            f"{json.dumps(forced_skills, ensure_ascii=False)}\n\n"
            "Candidate skills:\n"
            + "\n".join(candidate_lines)
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are a skill router. Select the skills whose descriptions are materially relevant "
                    "to the task. Use semantic judgment, not keyword matching. Use the stage-specific routing "
                    "guidance to understand what this stage is trying to do. Return JSON only with the "
                    'shape {"knowledge_skills": ["skill-name", ...]}. Return an empty list if none apply. '
                    "Never output names that are not in the candidate list."
                ),
            },
            {"role": "user", "content": user_content},
        ]

    def _normalize_selected_names(
        self,
        raw_response: Any,
        candidates: list[LoadedSkill],
    ) -> tuple[list[str], str]:
        if not isinstance(raw_response, dict):
            return [], "semantic_llm_invalid"
        selected = raw_response.get("knowledge_skills")
        if not isinstance(selected, list):
            return [], "semantic_llm_invalid"
        allowed = {skill.name for skill in candidates}
        normalized: list[str] = []
        for item in selected:
            name = str(item).strip()
            if name and name in allowed and name not in normalized:
                normalized.append(name)
        return normalized, "semantic_llm"

    def _create_completion_message(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
        """Force streaming because API proxy only supports text/event-stream."""
        req: dict[str, Any] = {
            "model": self.llm.model,
            "messages": messages,
            "stream": True,
            "temperature": 0.3,
        }
        if tools:
            req["tools"] = tools
            req["tool_choice"] = "auto"
        max_api_retries = 3
        stream = None
        last_error = None
        for attempt in range(max_api_retries):
            try:
                stream = self.llm._client.chat.completions.create(**req)
                last_error = None
                break
            except Exception as e:
                last_error = e
                if attempt < max_api_retries - 1:
                    wait = 2 ** attempt
                    print(f"[SkillRunner] LLM API call failed (attempt {attempt+1}/{max_api_retries}), "
                          f"retrying in {wait}s: {type(e).__name__}: {e}")
                    time.sleep(wait)
        if stream is None:
            status_code = None
            resp_text = ""
            if hasattr(last_error, "response"):
                if hasattr(last_error.response, "status_code"):
                    status_code = last_error.response.status_code
                if hasattr(last_error.response, "text"):
                    resp_text = last_error.response.text
            raise LLMAPIError(
                f"LLM API call failed after {max_api_retries} attempts: {last_error}",
                status_code=status_code,
                response=resp_text,
            )

        # Reconstruct a non-streaming response from SSE chunks
        content_parts: list[str] = []
        tool_call_deltas: dict[int, dict[str, Any]] = {}
        stream_read_ok = False
        last_stream_error = None
        for stream_attempt in range(max_api_retries):
            try:
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta is None:
                        continue
                    if delta.content:
                        content_parts.append(delta.content)
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_call_deltas:
                                tool_call_deltas[idx] = {"id": "", "function": {"name": "", "arguments": ""}}
                            if tc_delta.id:
                                tool_call_deltas[idx]["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    tool_call_deltas[idx]["function"]["name"] = tc_delta.function.name
                                if tc_delta.function.arguments:
                                    tool_call_deltas[idx]["function"]["arguments"] += tc_delta.function.arguments
                stream_read_ok = True
                break
            except Exception as e:
                last_stream_error = e
                if stream_attempt < max_api_retries - 1:
                    wait = 2 ** stream_attempt
                    print(f"[SkillRunner] LLM stream read failed (attempt {stream_attempt+1}/{max_api_retries}), "
                          f"retrying in {wait}s: {type(e).__name__}: {e}")
                    time.sleep(wait)
        if not stream_read_ok:
            raise LLMAPIError(
                f"LLM stream read failed after {max_api_retries} attempts: {last_stream_error}",
            )

        # Build a minimal mock Message with the needed attributes
        content = "".join(content_parts)

        class _MockToolCall:
            pass

        class _MockFunction:
            pass

        tool_calls_list = None
        if tool_call_deltas:
            tool_calls_list = []
            for idx in sorted(tool_call_deltas.keys()):
                d = tool_call_deltas[idx]
                tc = _MockToolCall()
                tc.id = d["id"]
                tc.function = _MockFunction()
                tc.function.name = d["function"]["name"]
                tc.function.arguments = d["function"]["arguments"]
                tool_calls_list.append(tc)

        class _MockMessage:
            def __init__(self, content, tool_calls):
                self.content = content
                self.tool_calls = tool_calls

        return _MockMessage(content, tool_calls_list)

    def _parse_payload_from_text(self, text: str) -> dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return {}
        text = self.llm._strip_code_fence(text)
        try:
            payload = json.loads(text)
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            extracted = self.llm._extract_json_obj(text)
            if extracted is not None:
                return extracted
            return extract_json_from_text(text)

    def _filter_existing_skill_names(self, names: list[str]) -> list[str]:
        normalized: list[str] = []
        for name in names:
            if self.registry.has(name) and name not in normalized:
                normalized.append(name)
        return normalized
