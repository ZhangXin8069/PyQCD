from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from utils.skill_utils import SkillRegistry


def _prompt(msg: str) -> str:
    """Interactive prompt (only imported when needed)."""
    from prompt_toolkit import prompt as pt_prompt
    return pt_prompt(msg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LQCD Planner/Executor workflow")
    parser.add_argument("--test", action="store_true", help="Run in test mode: generate artifacts and stop before job submission")
    parser.add_argument("--run-dir", type=str, default="", help="Run directory, default: runs/<timestamp>")
    parser.add_argument("--dotenv-path", type=str, default=".env", help="Path to the .env file")
    parser.add_argument("--task", type=str, default="", help="Provide the task description directly, or path to a txt file  and skip interactive input")
    parser.add_argument("--list-skills", action="store_true", help="List locally available skills and exit")
    parser.add_argument("--non-interactive", action="store_true", help="Skip manual checkpoints and accept generated results automatically")
    return parser


def resolve_run_dir(user_run_dir: str) -> Path:
    if user_run_dir:
        return Path(user_run_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / ts


def print_banner() -> None:
    banner = [
        "██╗      ██████╗  ██████╗ ██████╗     ███╗   ███╗ █████╗ ███████╗████████╗███████╗██████╗ ",
        "██║     ██╔═══██╗██╔════╝██╔═══██╗    ████╗ ████║██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔══██╗",
        "██║     ██║   ██║██║     ██║   ██║    ██╔████╔██║███████║███████╗   ██║   █████╗  ██████╔╝",
        "██║     ██║▄▄ ██║██║     ██║   ██║    ██║╚██╔╝██║██╔══██║╚════██║   ██║   ██╔══╝  ██╔══██╗",
        "███████╗╚██████╔╝╚██████╗╚██████╔╝    ██║ ╚═╝ ██║██║  ██║███████║   ██║   ███████╗██║  ██║",
        "╚══════╝ ╚══▀▀═╝  ╚═════╝ ╚═════╝     ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝",
    ]

    subtitle = "                  【 An Autonomous AI Scientist for Lattice QCD 】"
    reset = "\033[0m"

    def color_line(i: int, total: int) -> str:
        r = 30
        g = int(100 + 80 * (i / (total - 1)))
        b = 220
        return f"\033[38;2;{r};{g};{b}m"

    print()
    for i, line in enumerate(banner):
        print(f"{color_line(i, len(banner))}{line}{reset}")
    print()
    print(f"\033[38;2;80;180;255m{subtitle}{reset}")
    print()

def prompt_task(prefilled: str = "") -> str:
    if prefilled.strip():
        return prefilled.strip()
    while True:
        try:
            task = _prompt("[LQCD Master] Enter task description: ").strip()
        except ImportError:
            task = input("[LQCD Master] Enter task description: ").strip()
        if task:
            return task
        print("[LQCD Master] Task description cannot be empty. Please try again.")


def list_skills() -> list[str]:
    registry = SkillRegistry(Path("skills"))
    registry.load()
    return registry.names()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    print_banner()

    if args.list_skills:
        for name in list_skills():
            print(name)
        return

    from core_architecture.orchestrator import WorkflowOrchestrator
    from utils.llm_client import LQCDLLMClient
    from utils.submit_tool import SlurmSubmitTool
    from utils.tool_client import BuiltinToolClient

    if args.task.endswith(".txt"):
        with open(args.task, "r") as f:
            task = f.read().strip()
        args.task = task  

    task = prompt_task(args.task)

    run_dir = resolve_run_dir(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    llm = LQCDLLMClient(dotenv_path=args.dotenv_path)
    tool_client = BuiltinToolClient(dotenv_path=args.dotenv_path)
    submit_tool = SlurmSubmitTool()
    orchestrator = WorkflowOrchestrator(
        task=task,
        run_dir=run_dir,
        llm_client=llm,
        tool_client=tool_client,
        submit_tool=submit_tool,
        test_mode=args.test,
        non_interactive=args.non_interactive,
    )
    final_state = orchestrator.run()

    print()
    print("========== [ Workflow Completed ] ==========")
    print(f"[Run dir] Path: {run_dir.resolve()}")
    print(f"[Status] Completed")
    print(f"[Trajectory] File saved to: {(run_dir / 'trajectory_full.json').resolve()}")
    print(json.dumps({
        "planner_version": final_state.get("planner", {}).get("version"),
        "executor_version": final_state.get("executor", {}).get("version"),
        "test_submission": final_state.get("test_submission", {}),
        "full_submission": final_state.get("full_submission", {}),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
