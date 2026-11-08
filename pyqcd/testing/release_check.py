"""One-shot release evidence check.

Everything that must be true before tagging a release, in one command:

    python -m pyqcd.testing.release_check            # full, needs network
    python -m pyqcd.testing.release_check --offline   # skip PyPI/RTD/git remotes
    python -m pyqcd.testing.release_check --with-tests # also run the test suite

Each check prints `PASS`/`FAIL` with a one-line evidence string and the process
exits non-zero if anything fails, so it can gate a tag.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGES = ("overview", "usage", "performance", "install")
BLOCK = re.compile(r"```[a-z]*\n(.*?)```", re.S)

PYPI_JSON = "https://pypi.org/pypi/pyqcd/json"
RTD = {
    "en": "https://py-qcd.readthedocs.io/en/latest/index.html",
    "zh": "https://py-qcd.readthedocs.io/zh-cn/latest/index.html",
}


def _run(command: list[str], cwd: Path | None = None) -> tuple[int, str]:
    proc = subprocess.run(command, cwd=str(cwd or ROOT), capture_output=True,
                          text=True, timeout=1800)
    return proc.returncode, (proc.stdout + proc.stderr)


def _fetch(url: str, timeout: int = 30) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "pyqcd-release-check"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, evidence: str) -> None:
        self.rows.append((name, ok, evidence))
        print(f"  {'PASS' if ok else 'FAIL'}  {name:28s} {evidence}")

    @property
    def failed(self) -> list[tuple[str, bool, str]]:
        return [row for row in self.rows if not row[1]]


def check_versions(report: Report) -> str:
    import tomllib

    import pyqcd
    import pyqcd.einsum._version as version_module

    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    declared = project["version"]
    ok = (declared == pyqcd.__version__ == version_module.__version__
          and project["name"] == "pyqcd")
    report.add("versions agree", ok,
               f"pyproject={declared} pyqcd={pyqcd.__version__} "
               f"einsum={version_module.__version__}")
    return declared


def check_sdist(report: Report, version: str) -> None:
    sdist = ROOT / "dist" / f"pyqcd-{version}.tar.gz"
    if not sdist.is_file():
        code, output = _run([sys.executable, "-m", "build", "--sdist"])
        report.add("sdist builds", code == 0, output.strip().splitlines()[-1][:80])
    sdist = ROOT / "dist" / f"pyqcd-{version}.tar.gz"
    if not sdist.is_file():
        report.add("sdist present", False, "missing dist/*.tar.gz")
        return
    report.add("sdist present", True, sdist.name)

    code, output = _run([sys.executable, "-m", "twine", "check", str(sdist)])
    report.add("twine check", code == 0,
               next((line for line in output.splitlines() if "PASSED" in line), "")[:80])

    listing = _run(["tar", "-tzf", str(sdist)])[1].splitlines()
    names = [line for line in listing if line and not line.endswith("/")]
    problems = []
    if any("pyqcd/testing" in name for name in names):
        problems.append("pyqcd/testing shipped")
    if any(name.endswith(".so") for name in names):
        problems.append(".so shipped")
    if not any(name.endswith("pyqcd/py.typed") for name in names):
        problems.append("py.typed missing")
    if any("AGENTS.md" in name for name in names):
        problems.append("AGENTS.md shipped")
    report.add("sdist contents", not problems,
               f"{len(names)} files; " + ("clean" if not problems else ", ".join(problems)))

    sources = _run(["tar", "-xzOf", str(sdist),
                    f"pyqcd-{version}/pyqcd.egg-info/SOURCES.txt"])[1]
    absolute = [line for line in sources.splitlines() if "/home/" in line]
    report.add("sdist has no absolute paths", not absolute,
               f"{len(absolute)} offending lines")


def check_docs(report: Report) -> None:
    code, output = _run([sys.executable, "-m", "pyqcd.testing.docs_examples"])
    report.add("docs examples run", code == 0,
               output.strip().splitlines()[-1][:80] if output.strip() else "")

    mismatched = []
    for page in PAGES:
        en = BLOCK.findall((ROOT / "docs" / "en" / f"{page}.md").read_text(encoding="utf-8"))
        zh = BLOCK.findall((ROOT / "docs" / "zh" / f"{page}.md").read_text(encoding="utf-8"))
        if len(en) != len(zh) or any(a != b for a, b in zip(en, zh)):
            mismatched.append(page)
    report.add("en/zh code blocks", not mismatched,
               "identical" if not mismatched else f"differ: {mismatched}")

    for lang in ("en", "zh"):
        code, _ = _run([sys.executable, "-m", "sphinx", "-b", "html",
                        "-d", f"/tmp/relchk/{lang}/doctrees",
                        "-W", f"docs/{lang}", f"/tmp/relchk/{lang}/html"])
        report.add(f"sphinx -W ({lang})", code == 0, f"exit={code}")


def check_pypi(report: Report, version: str) -> None:
    try:
        payload = json.loads(_fetch(PYPI_JSON))
    except (urllib.error.URLError, TimeoutError) as error:
        report.add("pypi reachable", False, str(error)[:70])
        return
    info = payload["info"]
    report.add("pypi latest version", info["version"] == version,
               f"pypi={info['version']} local={version}")

    files = [item["filename"] for item in payload.get("urls", [])]
    report.add("pypi ships sdist only", all(name.endswith(".tar.gz") for name in files),
               ", ".join(files)[:80] or "(no files)")

    description = info.get("description", "")
    # 探针只挑"每版都应存在"的稳定片段，避免数字随版本更新而失效
    probes = {"Repositories": "## Repositories", "docs link": "py-qcd.readthedocs.io",
              "cli": "python -m pyqcd", "torch": "PyTorch"}
    missing = [label for label, probe in probes.items() if probe not in description]
    report.add("pypi page content", not missing,
               "all probes present" if not missing else f"missing: {missing}")


def check_rtd(report: Report, version: str) -> None:
    for lang, url in RTD.items():
        try:
            page = _fetch(url)
        except (urllib.error.URLError, TimeoutError) as error:
            report.add(f"rtd reachable ({lang})", False, str(error)[:70])
            continue
        report.add(f"rtd rebuilt ({lang})", version in page,
                   f"{'contains' if version in page else 'missing'} {version}")


def check_remotes(report: Report) -> None:
    head = _run(["git", "rev-parse", "HEAD"])[1].strip()
    targets = {
        "gitlab": ["git", "ls-remote", "origin", "main"],
        "gitee": ["git", "ls-remote", "git@gitee.com:zhangxin8069/PyQCD.git", "main"],
    }
    for name, command in targets.items():
        code, output = _run(command)
        remote = output.split()[0] if code == 0 and output.strip() else ""
        report.add(f"remote in sync ({name})", remote == head,
                   f"local={head[:8]} {name}={remote[:8] or 'unreachable'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--offline", action="store_true",
                        help="skip PyPI, Read the Docs and git remote checks")
    parser.add_argument("--with-tests", action="store_true",
                        help="also run the full test suite")
    args = parser.parse_args(argv)

    report = Report()
    print("release check")
    version = check_versions(report)
    check_sdist(report, version)
    check_docs(report)
    if not args.offline:
        check_pypi(report, version)
        check_rtd(report, version)
        check_remotes(report)
    if args.with_tests:
        code, output = _run([sys.executable, "-m", "pyqcd.testing"])
        summary = next((line for line in reversed(output.splitlines())
                        if "passed" in line), "")
        report.add("test suite", code == 0 and "0 failed" in summary, summary[:80])

    failed = report.failed
    print(f"\n{len(report.rows) - len(failed)}/{len(report.rows)} checks passed")
    for name, _ok, evidence in failed:
        print(f"  FAILED: {name}: {evidence}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
