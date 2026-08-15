from __future__ import annotations

import json
import os
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _load_dotenv(dotenv_path: str | Path = ".env") -> None:
    p = Path(dotenv_path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


class BuiltinToolClient:
    def __init__(self, dotenv_path: str | Path = ".env"):
        _load_dotenv(dotenv_path)
        self.serper_api_key = os.getenv("SERPER_API_KEY", "").strip()

    def web_search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        if not self.serper_api_key:
            return {"error": "SERPER_API_KEY is not configured"}

        payload = json.dumps({"q": query, "num": top_k}).encode("utf-8")
        req = urllib.request.Request(
            "https://google.serper.dev/search",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "X-API-KEY": self.serper_api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            return {"error": str(e), "source": "serper"}
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse JSON response: {e}", "source": "serper"}

        if not isinstance(data, dict):
            return {"error": f"Serper returned an unexpected response: {data}", "source": "serper"}

        organic = data.get("organic")
        if not isinstance(organic, list):
            organic = []
        return {
            **data,
            "organic": organic,
            "source": "serper",
        }

    def web_parse(self, link: str, user_prompt: str, llm: str = "gpt-4o") -> dict[str, Any]:
        clean_link = link.strip()
        if not clean_link:
            return {"error": "The link argument cannot be empty", "source": "jina"}
        reader_url = f"https://r.jina.ai/{clean_link}"
        req = urllib.request.Request(
            reader_url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "text/plain",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read().decode("utf-8", errors="ignore")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            return {"error": str(e), "source": "jina", "url": clean_link}

        lines = [line.strip() for line in content.splitlines() if line.strip()]
        title = lines[0] if lines else ""
        return {
            "source": "jina",
            "url": clean_link,
            "reader_url": reader_url,
            "title": title,
            "content": content,
            "excerpt": content[:4000],
            "user_prompt": user_prompt,
            "llm": llm,
        }

    def execute(self, code: str, timeout: int = 60, session_id: str = "local") -> dict[str, Any]:
        del session_id
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            script_path = f.name
        try:
            proc = subprocess.run(
                ["python3", script_path],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        except subprocess.TimeoutExpired as e:
            return {
                "ok": False,
                "error": f"execute timeout after {timeout}s",
                "stdout": e.stdout or "",
                "stderr": e.stderr or "",
            }
        finally:
            try:
                Path(script_path).unlink(missing_ok=True)
            except OSError:
                pass
