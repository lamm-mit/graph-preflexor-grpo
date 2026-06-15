#!/usr/bin/env python3
"""Upload the graphml-deep-analysis skill, then ask a question about GraphML.

This is a complete OpenAI API example for Graph Explorer:

1. Zip `ideation/graph_explorer/skills/graphml-deep-analysis`.
2. Upload the ZIP to `/v1/skills`.
3. Send a GraphML snapshot to the Responses API as base64 `file_data`.
4. Mount the uploaded skill in a hosted shell container and ask a graph question.

Install:
    python -m pip install --upgrade openai

Run from the repository root:
    export OPENAI_API_KEY="sk-..."
    python ideation/graph_explorer/examples/upload_graphml_skill_and_ask.py

Use your own graph:
    python ideation/graph_explorer/examples/upload_graphml_skill_and_ask.py \
      --graphml ideation/runs/my_run/graphml/iter_0042.graphml

Note: the GraphML attachment uses the Responses `input_file` content item with
base64 `file_data`. This does not use the Files API or a separate file upload.

This one-shot example intentionally uses `container_auto`, not
`container_reference`. Use `container_reference` only when you pre-create a
container or want a follow-up request to reuse the same container filesystem.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import tempfile
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_MODEL = "gpt-5.5"
DEFAULT_QUESTION = (
    "Which concepts are acting as the most surprising bridges between distant "
    "thematic modules, and what are three testable research hypotheses that "
    "follow from those bridge paths?"
)


def repo_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[3]


def default_graphml_path(repo_root: Path) -> Path:
    candidates = [
        repo_root / "ideation/runs/explorer_run/graphml/iter_0002.graphml",
        repo_root / "ideation/runs/explorer_run/graph.graphml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No default GraphML example found. Pass --graphml /path/to/file.graphml."
    )


def make_skill_zip(skill_dir: Path, out_dir: Path) -> Path:
    """Create a ZIP whose top-level folder is the skill folder name."""
    if not skill_dir.exists():
        raise FileNotFoundError(f"Skill directory not found: {skill_dir}")
    if not (skill_dir / "SKILL.md").exists():
        raise FileNotFoundError(f"Skill directory has no SKILL.md: {skill_dir}")

    zip_path = out_dir / f"{skill_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(skill_dir.rglob("*")):
            if path.is_file():
                arcname = Path(skill_dir.name) / path.relative_to(skill_dir)
                zf.write(path, arcname.as_posix())
    return zip_path


def multipart_upload_skill_zip(
    *,
    zip_path: Path,
    api_key: str,
    base_url: str,
    timeout: int = 120,
) -> Dict[str, Any]:
    """Upload a skill ZIP to POST /v1/skills using only the Python stdlib."""
    boundary = f"----graphml-skill-{uuid.uuid4().hex}"
    file_bytes = zip_path.read_bytes()
    body = b"".join(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                'Content-Disposition: form-data; name="files"; '
                f'filename="{zip_path.name}"\r\n'
            ).encode("utf-8"),
            b"Content-Type: application/zip\r\n\r\n",
            file_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )

    url = f"{base_url.rstrip('/')}/skills"
    request = urllib.request.Request(
        url=url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Skill upload failed: HTTP {exc.code}\n{detail}") from exc

    return json.loads(payload)


def extract_skill_id(upload_response: Dict[str, Any]) -> str:
    skill_id = upload_response.get("id") or upload_response.get("skill_id")
    if isinstance(skill_id, str) and skill_id:
        return skill_id
    raise RuntimeError(
        "Could not find a skill id in the upload response:\n"
        + json.dumps(upload_response, indent=2)
    )


def openai_client(api_key: str, base_url: Optional[str]):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install the OpenAI Python SDK with "
            "`python -m pip install --upgrade openai`."
        ) from exc

    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def ask_with_skill(
    *,
    api_key: str,
    sdk_base_url: Optional[str],
    model: str,
    skill_id: str,
    graphml_path: Path,
    question: str,
) -> Any:
    client = openai_client(api_key, sdk_base_url)
    graph_b64 = base64.b64encode(graphml_path.read_bytes()).decode("utf-8")
    filename = graphml_path.name

    prompt = f"""
Use the graphml-deep-analysis skill.

Analyze the attached GraphML file named `{filename}`. Run the bundled analyzer
first if useful, then answer the research question using concrete graph
evidence: module ids, bridge nodes, relation chains, and any important caveats.

Research question:
{question}
""".strip()

    return client.responses.create(
        model=model,
        tools=[
            {
                "type": "shell",
                "environment": {
                    # For uploaded skills, a one-shot request can mount the skill
                    # directly in an automatically managed hosted shell container.
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
        input=[
            {
                "role": "user",
                "content": [
                    {
                        # Base64 file_data still uses the input_file content type.
                        "type": "input_file",
                        "filename": filename,
                        "file_data": f"data:application/xml;base64,{graph_b64}",
                    },
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                ],
            }
        ],
    )


def response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text
    if hasattr(response, "model_dump_json"):
        return response.model_dump_json(indent=2)
    return str(response)


def parse_args() -> argparse.Namespace:
    repo_root = repo_root_from_this_file()
    skill_dir = repo_root / "ideation/graph_explorer/skills/graphml-deep-analysis"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill-dir", default=str(skill_dir), help="Path to graphml-deep-analysis skill folder")
    parser.add_argument("--graphml", default=None, help="GraphML file to ask about")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="Question to ask about the GraphML graph")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Responses model")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI API base URL for raw skill upload",
    )
    parser.add_argument(
        "--sdk-base-url",
        default=os.environ.get("OPENAI_BASE_URL"),
        help="Optional base URL passed to the OpenAI SDK",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the generated skill ZIP in the system temp directory and print its path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY before running this example.")

    repo_root = repo_root_from_this_file()
    skill_dir = Path(args.skill_dir).expanduser().resolve()
    graphml_path = (
        Path(args.graphml).expanduser().resolve()
        if args.graphml
        else default_graphml_path(repo_root).resolve()
    )
    if not graphml_path.exists():
        raise FileNotFoundError(f"GraphML file not found: {graphml_path}")

    with tempfile.TemporaryDirectory(prefix="graphml-skill-upload-") as tmp:
        tmp_dir = Path(tmp)
        zip_path = make_skill_zip(skill_dir, tmp_dir)
        if args.keep_zip:
            persistent_zip = Path(tempfile.gettempdir()) / zip_path.name
            persistent_zip.write_bytes(zip_path.read_bytes())
            print(f"Skill ZIP: {persistent_zip}")

        print(f"Uploading skill ZIP: {zip_path.name}")
        upload_response = multipart_upload_skill_zip(
            zip_path=zip_path,
            api_key=api_key,
            base_url=args.base_url,
        )
        skill_id = extract_skill_id(upload_response)
        print(f"Uploaded skill_id: {skill_id}")

        print(f"Asking about GraphML: {graphml_path}")
        print(f"Question: {args.question}")
        response = ask_with_skill(
            api_key=api_key,
            sdk_base_url=args.sdk_base_url,
            model=args.model,
            skill_id=skill_id,
            graphml_path=graphml_path,
            question=args.question,
        )

    print("\n=== Model answer ===\n")
    print(response_text(response))


if __name__ == "__main__":
    main()
