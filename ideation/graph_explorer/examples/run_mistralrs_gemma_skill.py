#!/usr/bin/env python3
"""Run graphml-deep-analysis with local mistral.rs and Gemma.

This example is local-only:

- no OpenAI SDK
- no OpenAI cloud API
- no OPENAI_API_KEY requirement

It talks directly to a local mistral.rs OpenAI-compatible server.

The script expects these files next to itself by default:

    graphml-deep-analysis.zip
    iter_0002.graphml

Start mistral.rs yourself:

    cd /Users/mbuehler/LOCALCODES/graph-preflexor-grpo/ideation
    mistralrs from-config -f models.toml

Then run:

    cd /Users/mbuehler/LOCALCODES/graph-preflexor-grpo/ideation/graph_explorer/examples
    python run_mistralrs_gemma_skill.py

Or let this script start mistral.rs from the repo TOML:

    python ideation/graph_explorer/examples/run_mistralrs_gemma_skill.py --start-server
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


BASE_URL = "http://localhost:1234/v1"
MODEL = "google/gemma-4-E4B-it"
QUESTION = (
    "Which concepts are acting as the most surprising bridges between distant "
    "thematic modules, and what are three testable research hypotheses that "
    "follow from those bridge paths?"
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def examples_dir() -> Path:
    return Path(__file__).resolve().parent


def default_graphml() -> Path:
    candidates = [
        examples_dir() / "iter_0002.graphml",
        repo_root() / "ideation/runs/explorer_run/graphml/iter_0002.graphml",
        repo_root() / "ideation/runs/explorer_run/graph.graphml",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No default GraphML found next to {Path(__file__).name}. "
        "Expected iter_0002.graphml, or pass --graphml."
    )


def default_skill_zip() -> Optional[Path]:
    path = examples_dir() / "graphml-deep-analysis.zip"
    return path if path.exists() else None


def default_skill_dir() -> Path:
    return repo_root() / "ideation/graph_explorer/skills/graphml-deep-analysis"


def default_toml() -> Path:
    return repo_root() / "ideation/models.toml"


def make_skill_zip(skill_dir: Path, out_dir: Path) -> Path:
    if not (skill_dir / "SKILL.md").exists():
        raise FileNotFoundError(f"No SKILL.md found in {skill_dir}")
    zip_path = out_dir / f"{skill_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(skill_dir.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                archive.write(path, (Path(skill_dir.name) / path.relative_to(skill_dir)).as_posix())
    return zip_path


def request_json(
    method: str,
    url: str,
    *,
    body: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 120,
) -> Dict[str, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer not-used",
            **(headers or {}),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed: HTTP {exc.code}\n{detail}") from exc
    return json.loads(payload)


def upload_skill_zip(base_url: str, zip_path: Path, timeout: int = 120) -> Dict[str, Any]:
    """Upload a Skill ZIP to local mistral.rs.

    mistral.rs examples use multipart field name `file`. The primary endpoint
    is `/v1/skills`; this also tries `/skills` if `/v1/skills` is not present.
    """
    boundary = f"----mistralrs-skill-{uuid.uuid4().hex}"
    payload = zip_path.read_bytes()
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            (
                'Content-Disposition: form-data; name="file"; '
                f'filename="{zip_path.name}"\r\n'
            ).encode(),
            b"Content-Type: application/zip\r\n\r\n",
            payload,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )

    urls = [f"{base_url.rstrip('/')}/skills"]
    parsed = urllib.parse.urlparse(base_url)
    if parsed.path.rstrip("/") == "/v1":
        root = urllib.parse.urlunparse(parsed._replace(path="", params="", query="", fragment="")).rstrip("/")
        urls.append(f"{root}/skills")

    errors = []
    for url in urls:
        print(f"Uploading skill to {url}")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": "Bearer not-used",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            errors.append(f"{url}: HTTP {exc.code}\n{detail}")
            if exc.code != 404:
                break

    joined = "\n\n".join(errors)
    raise RuntimeError(
        "Skill upload failed.\n"
        f"{joined}\n\n"
        "A 404 on both /v1/skills and /skills means the running process on this "
        "port does not have Skills routes registered. Restart mistral.rs from "
        "ideation/models.toml, make sure [runtime] has agent = true, and verify "
        "you are not talking to an older/stale server process on port 1234."
    )


def extract_skill_id(skill: Dict[str, Any]) -> str:
    skill_id = skill.get("id") or skill.get("skill_id")
    if not isinstance(skill_id, str) or not skill_id:
        raise RuntimeError(f"Upload response has no skill id:\n{json.dumps(skill, indent=2)}")
    return skill_id


def ask_with_skill(base_url: str, model: str, skill_id: str, graphml_path: Path, question: str) -> Dict[str, Any]:
    graph_b64 = base64.b64encode(graphml_path.read_bytes()).decode("utf-8")
    filename = graphml_path.name
    prompt = f"""
Use the graphml-deep-analysis skill.

Analyze the attached GraphML file named `{filename}`. Run the bundled analyzer
first, then answer using concrete graph evidence: module ids, bridge nodes,
relation chains, and caveats.

Research question:
{question}
""".strip()

    body = {
        "model": model,
        "tools": [
            {
                "type": "shell",
                "environment": {
                    "type": "container_auto",
                    "skills": [
                        {
                            "type": "skill_reference",
                            "skill_id": skill_id,
                            "version": "latest",
                        }
                    ],
                },
            }
        ],
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "filename": filename,
                        "file_data": f"data:application/xml;base64,{graph_b64}",
                    },
                    {"type": "input_text", "text": prompt},
                ],
            }
        ],
    }
    return request_json("POST", f"{base_url.rstrip('/')}/responses", body=body, timeout=300)


def output_text(response: Dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    chunks = []
    for item in response.get("output", []) or []:
        for content in item.get("content", []) or []:
            if isinstance(content, dict) and content.get("type") in {"output_text", "text"}:
                chunks.append(str(content.get("text", "")))
    return "\n".join(chunks).strip() or json.dumps(response, indent=2)


def start_server(toml_path: Path) -> subprocess.Popen[str]:
    print(f"Starting mistral.rs from {toml_path}")
    return subprocess.Popen(
        ["mistralrs", "from-config", "-f", str(toml_path)],
        cwd=str(toml_path.parent),
        text=True,
    )


def wait_for_server(base_url: str, timeout_s: int = 180) -> None:
    deadline = time.time() + timeout_s
    url = f"{base_url.rstrip('/')}/models"
    last_error = None
    while time.time() < deadline:
        try:
            models = request_json("GET", url, timeout=5)
            ids = [item.get("id") for item in models.get("data", []) if isinstance(item, dict)]
            print(f"mistral.rs is up. Models: {', '.join(ids) if ids else '(none listed)'}")
            return
        except Exception as exc:
            last_error = exc
            time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {url}: {last_error}")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--graphml", type=Path, default=None)
    parser.add_argument("--skill-zip", type=Path, default=default_skill_zip())
    parser.add_argument("--skill-dir", type=Path, default=default_skill_dir())
    parser.add_argument("--question", default=QUESTION)
    parser.add_argument("--start-server", action="store_true", help="Start mistralrs from --toml before calling it")
    parser.add_argument("--toml", type=Path, default=default_toml(), help="mistral.rs TOML used with --start-server")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    server: Optional[subprocess.Popen[str]] = None
    if args.start_server:
        server = start_server(args.toml.resolve())
        wait_for_server(args.base_url)

    graphml_path = (args.graphml or default_graphml()).resolve()
    if not graphml_path.exists():
        raise FileNotFoundError(graphml_path)

    with tempfile.TemporaryDirectory(prefix="mistralrs-graphml-skill-") as tmp:
        skill_zip = args.skill_zip.resolve() if args.skill_zip else make_skill_zip(args.skill_dir.resolve(), Path(tmp))
        if not skill_zip.exists():
            skill_zip = make_skill_zip(args.skill_dir.resolve(), Path(tmp))

        print(f"Skill ZIP: {skill_zip}")
        print(f"GraphML: {graphml_path}")
        print(f"Model: {args.model}")
        skill = upload_skill_zip(args.base_url, skill_zip)
        skill_id = extract_skill_id(skill)
        print(f"Uploaded skill_id: {skill_id}")

        response = ask_with_skill(args.base_url, args.model, skill_id, graphml_path, args.question)

    print("\n=== Answer ===\n")
    print(output_text(response))
    print("\n=== Raw response ===\n")
    print(json.dumps(response, indent=2))

    if server:
        print("\nServer was started by this script and is still running.")
        print("Stop it with Ctrl-C in this terminal, or kill the mistralrs process.")
        try:
            server.wait()
        except KeyboardInterrupt:
            server.terminate()
            server.wait(timeout=10)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
