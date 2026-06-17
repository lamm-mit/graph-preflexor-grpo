#!/usr/bin/env python3
"""Simple local mistral.rs Skill CLI.

No OpenAI SDK and no OpenAI cloud API are used. This script talks directly to a
local mistral.rs OpenAI-compatible server, defaulting to:

    http://localhost:1234/v1
    google/gemma-4-E4B-it

Example:

    python mistralrs_skill_cli.py ../skills/graphml-deep-analysis \
      --file iter_0002.graphml \
      --query "Which nodes bridge the major communities?"

By default, each `--file` is uploaded to `/v1/files` with `purpose=user_data`,
then attached to the Responses request as `{"type": "input_file", "file_id": ...}`.
Use `--file-mode inline` to send base64 `file_data` in the request body instead.

The script writes local run artifacts to `skill_cli_runs/<timestamp>_<skill>/`:

- uploaded skill ZIP
- input file upload metadata
- request.json
- response.json
- answer.md
- RESULTS.md
- run_manifest.json
- any first-class response files, discoverable shell artifact directories,
  plus optional ZIP artifact bundles printed by the skill
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import mimetypes
import os
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "google/gemma-4-E4B-it"
ARTIFACT_BEGIN = "BEGIN_MISTRALRS_ARTIFACT_ZIP"
ARTIFACT_END = "END_MISTRALRS_ARTIFACT_ZIP"


def slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return value.strip("-") or "skill"


def safe_filename(value: str, fallback: str = "artifact") -> str:
    name = Path(value or fallback).name
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip(".-")
    return name or fallback


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for idx in range(2, 10000):
        candidate = path.with_name(f"{stem}-{idx}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find a free filename near {path}")


def read_skill_name(skill_dir: Path) -> str:
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        raise FileNotFoundError(f"No SKILL.md found in {skill_dir}")
    text = skill_md.read_text(encoding="utf-8")
    match = re.search(r"^---\s*\n(.*?)\n---", text, flags=re.S)
    if not match:
        raise ValueError(f"{skill_md} has no YAML frontmatter")
    name_match = re.search(r"^name:\s*[\"']?([^\"'\n]+)[\"']?\s*$", match.group(1), flags=re.M)
    if not name_match:
        raise ValueError(f"{skill_md} frontmatter has no name field")
    return name_match.group(1).strip()


def validate_skill_dir(skill_dir: Path) -> str:
    skill_dir = skill_dir.resolve()
    skill_name = read_skill_name(skill_dir)
    if skill_name != skill_dir.name:
        raise ValueError(
            "Skill name must match directory name for this CLI.\n"
            f"  directory: {skill_dir.name}\n"
            f"  SKILL.md name: {skill_name}"
        )
    return skill_name


def make_skill_zip(skill_dir: Path, out_dir: Path) -> Path:
    skill_name = validate_skill_dir(skill_dir)
    zip_path = out_dir / f"{skill_name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(skill_dir.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                archive_name = (Path(skill_dir.name) / path.relative_to(skill_dir)).as_posix()
                archive.write(path, archive_name)
    return zip_path


def http_json(
    method: str,
    url: str,
    *,
    body: Optional[Dict[str, Any]] = None,
    timeout: int = 300,
) -> Dict[str, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": "Bearer not-used",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed: HTTP {exc.code}\n{detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"{method} {url} timed out after {timeout} seconds") from exc
    return json.loads(payload)


def http_bytes(method: str, url: str, *, timeout: int = 300) -> bytes:
    request = urllib.request.Request(
        url,
        method=method,
        headers={"Authorization": "Bearer not-used"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed: HTTP {exc.code}\n{detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"{method} {url} timed out after {timeout} seconds") from exc


def upload_skill(base_url: str, zip_path: Path, *, timeout: int = 300) -> Dict[str, Any]:
    boundary = f"----mistralrs-skill-{uuid.uuid4().hex}"
    data = zip_path.read_bytes()
    body = b"".join(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                'Content-Disposition: form-data; name="file"; '
                f'filename="{zip_path.name}"\r\n'
            ).encode("utf-8"),
            b"Content-Type: application/zip\r\n\r\n",
            data,
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )

    urls = [f"{base_url.rstrip('/')}/skills"]
    parsed = urllib.parse.urlparse(base_url)
    if parsed.path.rstrip("/") == "/v1":
        root = urllib.parse.urlunparse(parsed._replace(path="", params="", query="", fragment="")).rstrip("/")
        urls.append(f"{root}/skills")

    errors: List[str] = []
    for url in urls:
        print(f"Uploading skill ZIP to {url}")
        request = urllib.request.Request(
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
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            errors.append(f"{url}: HTTP {exc.code}\n{detail}")
            if exc.code != 404:
                break
        except urllib.error.URLError as exc:
            errors.append(f"{url}: {exc.reason}")
        except TimeoutError:
            errors.append(f"{url}: timed out after {timeout} seconds")
    raise RuntimeError("Skill upload failed:\n" + "\n\n".join(errors))


def upload_user_file(base_url: str, path: Path, *, timeout: int = 300) -> Dict[str, Any]:
    boundary = f"----mistralrs-user-file-{uuid.uuid4().hex}"
    mime, _encoding = mimetypes.guess_type(path.name)
    mime = mime or "application/octet-stream"
    filename = path.name.replace('"', "_")
    body = b"".join(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            b'Content-Disposition: form-data; name="purpose"\r\n\r\n',
            b"user_data\r\n",
            f"--{boundary}\r\n".encode("utf-8"),
            (
                'Content-Disposition: form-data; name="file"; '
                f'filename="{filename}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {mime}\r\n\r\n".encode("utf-8"),
            path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    url = f"{base_url.rstrip('/')}/files"
    request = urllib.request.Request(
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
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"File upload failed for {path}: HTTP {exc.code}\n{detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"File upload failed for {path}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"File upload timed out for {path} after {timeout} seconds") from exc


def extract_skill_id(upload_response: Dict[str, Any]) -> str:
    skill_id = upload_response.get("id") or upload_response.get("skill_id")
    if not isinstance(skill_id, str) or not skill_id:
        raise RuntimeError("Upload response has no skill id:\n" + json.dumps(upload_response, indent=2))
    return skill_id


def extract_file_id(upload_response: Dict[str, Any], path: Path) -> str:
    file_id = upload_response.get("id") or upload_response.get("file_id")
    if not isinstance(file_id, str) or not file_id:
        raise RuntimeError(f"Upload response for {path} has no file id:\n" + json.dumps(upload_response, indent=2))
    return file_id


def input_file_content(path: Path) -> Dict[str, str]:
    mime, _encoding = mimetypes.guess_type(path.name)
    mime = mime or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return {
        "type": "input_file",
        "filename": path.name,
        "file_data": f"data:{mime};base64,{encoded}",
    }


def build_file_content_parts(
    *,
    base_url: str,
    files: Iterable[Path],
    mode: str,
    timeout: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    content_parts: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []
    for path in files:
        resolved = path.resolve()
        if mode == "upload":
            print(f"Uploading input file to /v1/files: {resolved}")
            upload_response = upload_user_file(base_url, resolved, timeout=timeout)
            file_id = extract_file_id(upload_response, resolved)
            content_part = {"type": "input_file", "file_id": file_id}
            records.append(
                {
                    "path": str(resolved),
                    "mode": "upload",
                    "file_id": file_id,
                    "upload_response": upload_response,
                    "content_part": content_part,
                }
            )
        elif mode == "inline":
            content_part = input_file_content(resolved)
            records.append(
                {
                    "path": str(resolved),
                    "mode": "inline",
                    "content_part": {
                        "type": "input_file",
                        "filename": content_part["filename"],
                        "file_data": "<base64 omitted>",
                    },
                }
            )
        else:
            raise ValueError(f"Unknown file mode: {mode}")
        content_parts.append(content_part)
    return content_parts, records


def build_prompt(
    skill_name: str,
    query: str,
    output_label: str,
    files: Iterable[Path],
    artifact_bundle: bool,
) -> str:
    attached_files = [path.name for path in files]
    file_text = (
        "Attached input_file filenames: " + ", ".join(attached_files)
        if attached_files
        else "No user data files were attached."
    )
    bundle_text = ""
    if artifact_bundle:
        bundle_text = f"""

If you create analysis artifacts, put them in `{output_label}/`. At the end,
write `{output_label}/RESULTS.md` with a short index of the files and
`{output_label}/manifest.json` with machine-readable artifact paths. Also create
`{output_label}.zip` from that directory. If the ZIP is small and the runtime
does not surface it as a first-class response file, print exactly this fallback
block so the client can download and extract it. Do not print base64 for large
HTML/PDF/media artifacts; report their shell paths instead:

{ARTIFACT_BEGIN} {output_label}.zip
<base64 contents of {output_label}.zip>
{ARTIFACT_END}
""".rstrip()

    return f"""
Use the {skill_name} skill.

{file_text}

Run the skill workflow needed to answer the query. The files above were attached
as OpenAI-compatible `input_file` content parts, so use the filenames directly
from the session working directory when a file is needed.
If the skill creates files or analysis artifacts, create or use an output
directory named `{output_label}` in the shell working directory. In the final
answer, state the exact path to every artifact you created or inspected so a
human or another agent can find it.
Before writing the final answer, verify that the artifacts actually exist by
running `find {output_label} -maxdepth 2 -type f -print` or an equivalent
directory listing. If no files are listed, continue the tool workflow and create
the missing files instead of claiming success.
{bundle_text}

Query:
{query}
""".strip()


def build_tools(
    *,
    skill_id: str,
    enable_code: bool,
    enable_search: bool,
    search_context_size: str,
) -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = [
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
    ]
    if enable_code:
        tools.append({"type": "code_interpreter", "container": {"type": "auto"}})
    if enable_search:
        tools.append(
            {
                "type": "web_search",
                "search_context_size": search_context_size,
                "return_token_budget": "default",
            }
        )
    return tools


def build_request(
    *,
    model: str,
    skill_id: str,
    skill_name: str,
    query: str,
    files: Iterable[Path],
    file_content_parts: Iterable[Dict[str, Any]],
    output_label: str,
    require_tool: bool,
    enable_code: bool,
    enable_search: bool,
    search_context_size: str,
    max_tool_rounds: int,
    artifact_bundle: bool,
) -> Dict[str, Any]:
    file_list = list(files)
    content: List[Dict[str, Any]] = list(file_content_parts)
    content.append(
        {
            "type": "input_text",
            "text": build_prompt(skill_name, query, output_label, file_list, artifact_bundle),
        }
    )
    request: Dict[str, Any] = {
        "model": model,
        "tools": build_tools(
            skill_id=skill_id,
            enable_code=enable_code,
            enable_search=enable_search,
            search_context_size=search_context_size,
        ),
        "input": [{"role": "user", "content": content}],
        "max_tool_rounds": max_tool_rounds,
    }
    if require_tool:
        request["tool_choice"] = "required"
    return request


def iter_string_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from iter_string_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_string_values(child)


def is_file_object(value: Dict[str, Any]) -> bool:
    has_file_name = isinstance(value.get("name"), str) or isinstance(value.get("filename"), str)
    has_file_id = isinstance(value.get("id"), str) and value["id"].startswith("file")
    return (
        (has_file_name or value.get("object") == "file")
        and (
            has_file_id
            or "text" in value
            or "data_base64" in value
            or "mime_type" in value
            or "bytes" in value
            or "code" in value
        )
    )


def find_response_files(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if is_file_object(value):
                key = (
                    str(value.get("id") or ""),
                    str(value.get("name") or ""),
                    str(value.get("bytes") or ""),
                )
                if key not in seen:
                    seen.add(key)
                    found.append(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(response.get("files", []))
    visit(response.get("output", []))
    visit(response.get("agentic_tool_calls", []))
    return found


def safe_extract_zip(zip_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    root = extract_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (root / member.filename).resolve()
            if target != root and not str(target).startswith(str(root) + os.sep):
                raise RuntimeError(f"Refusing unsafe zip member: {member.filename}")
            archive.extract(member, root)


def maybe_extract_zip(path: Path, extract_root: Path) -> Optional[Path]:
    if path.suffix.lower() != ".zip" or not zipfile.is_zipfile(path):
        return None
    extract_dir = unique_path(extract_root / path.stem)
    safe_extract_zip(path, extract_dir)
    return extract_dir


def write_file_object(
    file_obj: Dict[str, Any],
    base_url: str,
    download_dir: Path,
    *,
    timeout: int,
) -> Optional[Path]:
    download_dir.mkdir(parents=True, exist_ok=True)
    name = safe_filename(str(file_obj.get("filename") or file_obj.get("name") or file_obj.get("id") or "file"))
    dest = unique_path(download_dir / name)

    if file_obj.get("code") or file_obj.get("message"):
        dest = dest.with_suffix(dest.suffix + ".error.json")
        dest.write_text(json.dumps(file_obj, indent=2), encoding="utf-8")
        return dest

    if isinstance(file_obj.get("text"), str):
        dest.write_text(file_obj["text"], encoding="utf-8")
        return dest

    if isinstance(file_obj.get("data_base64"), str):
        dest.write_bytes(base64.b64decode(file_obj["data_base64"]))
        return dest

    file_id = file_obj.get("id")
    if isinstance(file_id, str) and file_id:
        url = f"{base_url.rstrip('/')}/files/{urllib.parse.quote(file_id)}/content"
        dest.write_bytes(http_bytes("GET", url, timeout=timeout))
        return dest

    dest = dest.with_suffix(dest.suffix + ".metadata.json")
    dest.write_text(json.dumps(file_obj, indent=2), encoding="utf-8")
    return dest


def save_first_class_files(
    response: Dict[str, Any],
    *,
    base_url: str,
    out_dir: Path,
    timeout: int,
) -> List[Path]:
    files = find_response_files(response)
    download_dir = out_dir / "downloaded_files"
    metadata_dir = out_dir / "downloaded_file_metadata"
    extract_root = out_dir / "extracted_artifacts"
    saved: List[Path] = []

    for idx, file_obj in enumerate(files, start=1):
        metadata_dir.mkdir(parents=True, exist_ok=True)
        try:
            saved_path = write_file_object(file_obj, base_url, download_dir, timeout=timeout)
            if not saved_path:
                continue
            saved.append(saved_path)
            metadata_path = metadata_dir / f"{idx:03d}_{safe_filename(saved_path.name)}.json"
            metadata_path.write_text(json.dumps(file_obj, indent=2), encoding="utf-8")
            try:
                extracted = maybe_extract_zip(saved_path, extract_root)
            except Exception as exc:
                error_path = metadata_dir / f"{idx:03d}_{safe_filename(saved_path.name)}_extract_error.json"
                error_path.write_text(json.dumps({"error": str(exc)}, indent=2), encoding="utf-8")
                saved.append(error_path)
            else:
                if extracted:
                    saved.append(extracted)
        except Exception as exc:
            error_path = metadata_dir / f"{idx:03d}_download_error.json"
            error_path.write_text(
                json.dumps({"error": str(exc), "file": file_obj}, indent=2),
                encoding="utf-8",
            )
            saved.append(error_path)
    return saved


def save_marker_artifact_bundles(response: Dict[str, Any], out_dir: Path) -> List[Path]:
    text = "\n".join(iter_string_values(response))
    pattern = re.compile(
        rf"{re.escape(ARTIFACT_BEGIN)}\s+([^\r\n]+)\s+(.*?){re.escape(ARTIFACT_END)}",
        flags=re.S,
    )
    download_dir = out_dir / "downloaded_files"
    extract_root = out_dir / "extracted_artifacts"
    saved: List[Path] = []

    for match in pattern.finditer(text):
        filename = safe_filename(match.group(1).strip(), "artifact.zip")
        if not filename.lower().endswith(".zip"):
            filename += ".zip"
        payload = re.sub(r"\s+", "", match.group(2))
        try:
            data = base64.b64decode(payload, validate=True)
        except ValueError:
            continue
        download_dir.mkdir(parents=True, exist_ok=True)
        zip_path = unique_path(download_dir / filename)
        zip_path.write_bytes(data)
        saved.append(zip_path)
        try:
            extracted = maybe_extract_zip(zip_path, extract_root)
        except Exception as exc:
            error_path = unique_path((out_dir / "downloaded_file_metadata") / f"{zip_path.stem}_extract_error.json")
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(json.dumps({"error": str(exc)}, indent=2), encoding="utf-8")
            saved.append(error_path)
        else:
            if extracted:
                saved.append(extracted)
    return saved


def nearest_named_directory(path: Path, directory_name: str) -> Optional[Path]:
    """Return the nearest existing parent named directory_name."""
    candidates = [path if path.is_dir() else path.parent, *path.parents]
    for candidate in candidates:
        if candidate.name == directory_name and candidate.exists() and candidate.is_dir():
            return candidate.resolve()
    return None


def likely_shell_search_roots() -> List[Path]:
    roots: List[Path] = []
    raw_roots = [
        os.environ.get("TMPDIR"),
        os.environ.get("MISTRALRS_SHELL_WORKDIR"),
        os.environ.get("SHELL_WORKDIR"),
        os.getcwd(),
        "/tmp",
        "/private/tmp",
    ]

    def add(path: Path) -> None:
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            return
        if resolved.exists() and resolved.is_dir() and resolved not in roots:
            roots.append(resolved)

    for raw in raw_roots:
        if raw:
            add(Path(raw))

    for root in list(roots):
        for pattern in ("mistralrs-code-*", "mistralrs-shell-*", "mistralrs-*"):
            try:
                matches = root.glob(pattern)
                for match in matches:
                    add(match)
            except OSError:
                continue

    return roots


def find_shell_artifact_dirs(response: Dict[str, Any], output_label: str) -> List[Path]:
    found: List[Path] = []
    seen: set[Path] = set()
    escaped_label = re.escape(output_label)
    path_patterns = [
        re.compile(rf"(/[^\s`'\"<>]*{escaped_label}(?:/[^\s`'\"<>)]*)?)"),
        re.compile(rf"((?:\./)?{escaped_label}(?:/[^\s`'\"<>)]*)?)"),
    ]

    def add_candidate(path: Path) -> None:
        directory = nearest_named_directory(path, output_label)
        if not directory:
            return
        if directory not in seen:
            seen.add(directory)
            found.append(directory)

    for text in iter_string_values(response):
        if output_label not in text:
            continue
        for pattern in path_patterns:
            for match in pattern.finditer(text):
                raw_path = match.group(1).strip().rstrip(".,;:")
                if not raw_path:
                    continue
                path = Path(raw_path)
                if path.is_absolute():
                    add_candidate(path)
                else:
                    add_candidate((Path.cwd() / path).resolve())

    for root in likely_shell_search_roots():
        candidates = [root / output_label]
        try:
            candidates.extend(root.glob(f"*/{output_label}"))
        except OSError:
            pass
        for candidate in candidates:
            add_candidate(candidate)

    return found


def copy_shell_artifact_dir(source_dir: Path, out_dir: Path) -> Optional[Path]:
    dest_root = out_dir / "downloaded_files" / "shell_artifacts"
    dest_root.mkdir(parents=True, exist_ok=True)
    dest_dir = unique_path(dest_root / safe_filename(source_dir.name, "shell_artifacts"))
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied_files: List[str] = []
    for source in sorted(source_dir.rglob("*")):
        if source.is_symlink() or not source.is_file():
            continue
        rel = source.relative_to(source_dir)
        target = dest_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied_files.append(str(rel))

    metadata = {
        "source_directory": str(source_dir),
        "copied_directory": str(dest_dir),
        "copied_files": copied_files,
    }
    metadata_path = dest_dir / "shell_artifact_source.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if not copied_files:
        warnings_dir = out_dir / "downloaded_file_metadata"
        warnings_dir.mkdir(parents=True, exist_ok=True)
        warning_path = unique_path(warnings_dir / f"{safe_filename(source_dir.name)}_empty_shell_dir.json")
        warning_path.write_text(
            json.dumps(
                {
                    "warning": "The model reported or created the requested shell artifact directory, but it contained no files.",
                    "source_directory": str(source_dir),
                    "copied_directory": str(dest_dir),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return None
    return dest_dir


def save_shell_artifact_dirs(response: Dict[str, Any], out_dir: Path, output_label: str) -> List[Path]:
    saved: List[Path] = []
    metadata_dir = out_dir / "downloaded_file_metadata"
    for source_dir in find_shell_artifact_dirs(response, output_label):
        try:
            copied = copy_shell_artifact_dir(source_dir, out_dir)
            if copied:
                saved.append(copied)
        except Exception as exc:
            metadata_dir.mkdir(parents=True, exist_ok=True)
            error_path = unique_path(metadata_dir / f"{safe_filename(source_dir.name)}_shell_copy_error.json")
            error_path.write_text(
                json.dumps({"source_directory": str(source_dir), "error": str(exc)}, indent=2),
                encoding="utf-8",
            )
            saved.append(error_path)
    return saved


def save_response_artifacts(
    response: Dict[str, Any],
    *,
    base_url: str,
    out_dir: Path,
    timeout: int,
    output_label: str,
) -> List[Path]:
    saved = save_first_class_files(response, base_url=base_url, out_dir=out_dir, timeout=timeout)
    saved.extend(save_marker_artifact_bundles(response, out_dir))
    saved.extend(save_shell_artifact_dirs(response, out_dir, output_label))
    summary = {
        "saved_paths": [str(path) for path in saved],
        "note": (
            "First-class response files are saved from response files[] and fetched "
            "from /v1/files/{id}/content when needed. ZIP marker bundles are parsed "
            f"from {ARTIFACT_BEGIN}/{ARTIFACT_END} blocks in tool output. If the "
            "model reports shell paths but mistral.rs does not expose files[], this "
            "CLI also looks for the requested skill_output_* directory in local "
            "mistral.rs shell/temp workdirs and copies it into downloaded_files/."
        ),
    }
    (out_dir / "artifact_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return saved


def write_run_results(
    *,
    out_dir: Path,
    answer: str,
    request_body: Dict[str, Any],
    response: Dict[str, Any],
    skill_name: str,
    skill_id: str,
    skill_zip: Path,
    input_file_records: List[Dict[str, Any]],
    saved_artifacts: List[Path],
    output_label: str,
) -> Tuple[Path, Path]:
    manifest = {
        "skill_name": skill_name,
        "skill_id": skill_id,
        "skill_zip": str(skill_zip),
        "response_id": response.get("id"),
        "response_status": response.get("status"),
        "model": response.get("model") or request_body.get("model"),
        "local_run_directory": str(out_dir),
        "answer": str(out_dir / "answer.md"),
        "answer_preview": answer[:2000],
        "raw_response": str(out_dir / "response.json"),
        "request": str(out_dir / "request.json"),
        "skill_upload": str(out_dir / "skill_upload.json"),
        "input_file_uploads": str(out_dir / "input_file_uploads.json"),
        "artifact_summary": str(out_dir / "artifact_summary.json"),
        "downloaded_files_directory": str(out_dir / "downloaded_files"),
        "extracted_artifacts_directory": str(out_dir / "extracted_artifacts"),
        "requested_shell_artifact_directory": output_label,
        "input_files": input_file_records,
        "saved_artifacts": [str(path) for path in saved_artifacts],
        "tools": request_body.get("tools", []),
    }

    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    saved_lines = "\n".join(f"- `{path}`" for path in saved_artifacts) or "- None found."
    input_lines = "\n".join(
        f"- `{record.get('path')}` via `{record.get('mode')}`"
        + (f" as `{record.get('file_id')}`" if record.get("file_id") else "")
        for record in input_file_records
    ) or "- None."
    results_md = f"""# Skill Run Results

## Answer

See `answer.md` for the final model answer.

## Local Files

- Run directory: `{out_dir}`
- Human-readable answer: `{out_dir / "answer.md"}`
- Machine-readable manifest: `{manifest_path}`
- Raw request: `{out_dir / "request.json"}`
- Raw response: `{out_dir / "response.json"}`
- Artifact summary: `{out_dir / "artifact_summary.json"}`

## Inputs

{input_lines}

## Downloaded Or Extracted Artifacts

{saved_lines}

## Shell Artifact Directory

The request asked the skill to write artifacts inside the shell working directory
as `{output_label}` and, when possible, bundle that directory as `{output_label}.zip`.

This CLI saves first-class `/v1/files` outputs, decodes optional artifact ZIP
markers, and also tries to copy the requested `{output_label}` directory from
local mistral.rs shell/temp workdirs when the server only reports shell paths.
If no downloaded artifacts are listed, inspect `answer.md` and `response.json`.
For the most reliable shell artifact retrieval, start mistral.rs with an
explicit `--shell-workdir <path>`.
"""
    results_path = out_dir / "RESULTS.md"
    results_path.write_text(results_md, encoding="utf-8")
    return results_path, manifest_path


def output_text(response: Dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    chunks: List[str] = []
    for item in response.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if isinstance(content, dict) and content.get("type") in {"output_text", "text"}:
                chunks.append(str(content.get("text", "")))
    return "\n".join(chunks).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("skill_dir", type=Path, help="Skill directory. Directory name must match SKILL.md name.")
    parser.add_argument("--query", "-q", required=True, help="Query to run with the uploaded skill.")
    parser.add_argument(
        "--file",
        "-f",
        action="append",
        type=Path,
        default=[],
        help="Optional input file to attach as type='input_file'. Repeatable.",
    )
    parser.add_argument(
        "--file-mode",
        choices=["upload", "inline"],
        default="upload",
        help="How to attach --file inputs: upload to /v1/files and use file_id, or inline base64 file_data.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=Path("skill_cli_runs"))
    parser.add_argument("--max-tool-rounds", type=int, default=8)
    parser.add_argument("--response-timeout", type=int, default=1800, help="Seconds to wait for POST /responses.")
    parser.add_argument("--upload-timeout", type=int, default=300, help="Seconds to wait for skill and file uploads.")
    parser.add_argument("--download-timeout", type=int, default=300, help="Seconds to wait for response file downloads.")
    parser.add_argument("--no-code", action="store_true", help="Do not include the code_interpreter tool.")
    parser.add_argument("--no-search", action="store_true", help="Do not include the web_search tool.")
    parser.add_argument("--search-context-size", choices=["low", "medium", "high"], default="medium")
    parser.add_argument(
        "--artifact-bundle",
        action="store_true",
        help="Ask the skill to print a base64 ZIP fallback for small shell artifacts.",
    )
    parser.add_argument(
        "--no-artifact-bundle",
        action="store_true",
        help="Deprecated compatibility flag. Artifact bundle fallback is off unless --artifact-bundle is set.",
    )
    parser.add_argument("--require-tool", action="store_true", help="Send tool_choice='required'.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    skill_dir = args.skill_dir.resolve()
    skill_name = validate_skill_dir(skill_dir)
    for path in args.file:
        if not path.exists():
            raise FileNotFoundError(path)

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_label = f"skill_output_{run_id}_{slug(skill_name)}"
    out_dir = (args.out / f"{run_id}_{slug(skill_name)}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = make_skill_zip(skill_dir, out_dir)
    print(f"Skill: {skill_name}")
    print(f"Skill directory: {skill_dir}")
    print(f"Skill ZIP: {zip_path}")
    print(f"Local output directory: {out_dir}")
    print(f"Input file mode: {args.file_mode}")

    upload_response = upload_skill(args.base_url, zip_path, timeout=args.upload_timeout)
    (out_dir / "skill_upload.json").write_text(json.dumps(upload_response, indent=2), encoding="utf-8")
    skill_id = extract_skill_id(upload_response)
    print(f"Uploaded skill_id: {skill_id}")
    file_content_parts, file_records = build_file_content_parts(
        base_url=args.base_url,
        files=args.file,
        mode=args.file_mode,
        timeout=args.upload_timeout,
    )
    (out_dir / "input_file_uploads.json").write_text(json.dumps(file_records, indent=2), encoding="utf-8")

    request_body = build_request(
        model=args.model,
        skill_id=skill_id,
        skill_name=skill_name,
        query=args.query,
        files=args.file,
        file_content_parts=file_content_parts,
        output_label=output_label,
        require_tool=args.require_tool,
        enable_code=not args.no_code,
        enable_search=not args.no_search,
        search_context_size=args.search_context_size,
        max_tool_rounds=args.max_tool_rounds,
        artifact_bundle=args.artifact_bundle and not args.no_artifact_bundle,
    )
    (out_dir / "request.json").write_text(json.dumps(request_body, indent=2), encoding="utf-8")

    try:
        response = http_json(
            "POST",
            f"{args.base_url.rstrip('/')}/responses",
            body=request_body,
            timeout=args.response_timeout,
        )
    except Exception as exc:
        error = {
            "error": str(exc),
            "base_url": args.base_url,
            "model": args.model,
            "response_timeout": args.response_timeout,
            "local_run_directory": str(out_dir),
            "request": str(out_dir / "request.json"),
            "skill_upload": str(out_dir / "skill_upload.json"),
            "input_file_uploads": str(out_dir / "input_file_uploads.json"),
            "hint": (
                "The skill and input files were uploaded and request.json was saved. "
                "For D3/LaTeX artifact generation, try increasing --response-timeout, "
                "increasing the server shell timeout, or simplifying the requested artifact."
            ),
        }
        error_path = out_dir / "response_error.json"
        error_path.write_text(json.dumps(error, indent=2), encoding="utf-8")
        print("\n=== Response Error ===")
        print(error["error"])
        print("\n=== Saved Partial Run ===")
        print(f"Local run directory: {out_dir}")
        print(f"Request: {out_dir / 'request.json'}")
        print(f"Skill upload response: {out_dir / 'skill_upload.json'}")
        print(f"Input file upload metadata: {out_dir / 'input_file_uploads.json'}")
        print(f"Response error: {error_path}")
        raise SystemExit(1) from exc
    (out_dir / "response.json").write_text(json.dumps(response, indent=2), encoding="utf-8")

    answer = output_text(response) or json.dumps(response, indent=2)
    (out_dir / "answer.md").write_text(answer, encoding="utf-8")
    saved_artifacts = save_response_artifacts(
        response,
        base_url=args.base_url,
        out_dir=out_dir,
        timeout=args.download_timeout,
        output_label=output_label,
    )
    results_path, manifest_path = write_run_results(
        out_dir=out_dir,
        answer=answer,
        request_body=request_body,
        response=response,
        skill_name=skill_name,
        skill_id=skill_id,
        skill_zip=zip_path,
        input_file_records=file_records,
        saved_artifacts=saved_artifacts,
        output_label=output_label,
    )

    print("\n=== Answer ===\n")
    print(answer)
    print("\n=== Saved Outputs ===")
    print(f"Local run directory: {out_dir}")
    print(f"Results index: {results_path}")
    print(f"Run manifest: {manifest_path}")
    print(f"Answer: {out_dir / 'answer.md'}")
    print(f"Raw response: {out_dir / 'response.json'}")
    print(f"Request: {out_dir / 'request.json'}")
    print(f"Skill upload response: {out_dir / 'skill_upload.json'}")
    print(f"Input file upload metadata: {out_dir / 'input_file_uploads.json'}")
    print(f"Artifact summary: {out_dir / 'artifact_summary.json'}")
    print(f"Uploaded ZIP copy: {zip_path}")
    print(f"Skill artifact directory requested inside shell: {output_label}")
    if saved_artifacts:
        print("Downloaded/extracted artifacts:")
        for path in saved_artifacts:
            print(f"  {path}")
    else:
        print(
            "Downloaded/extracted artifacts: none found in response files[], "
            "artifact ZIP markers, or discoverable shell artifact directories."
        )


if __name__ == "__main__":
    main()
