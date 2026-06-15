import { useQuery } from "@tanstack/react-query";
import { CircleStop, ExternalLink, FileText, Loader2, Network, Play } from "lucide-react";
import React, { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { formatNumber } from "../graph-utils";
import { useExplorerStore } from "../store";
import type { GraphFileSummary, GraphPayload, ProfileArtifacts, ProfileJobStatus, ProfileOptions, ProfileReportPayload } from "../types";
import { cx, Drawer, formatRunTime, HelpTip, IconButton } from "../components/common";

const REPORT_STUDIO_STORAGE_KEY = "graph-preflexor-explorer.report-studio.v1";

type StoredReportStudio = ProfileOptions & {
  job?: ProfileJobStatus | null;
  jobId?: string;
  activeReportOut?: string;
};

const defaultProfileOptions: ProfileOptions = {
  run: "runs/exp_leap",
  graph: "",
  out: "runs/exp_leap/profile_gpt55",
  embed_model: "auto",
  top_nodes: 25,
  max_modules: 30,
  profile_preset: "full",
  llm: true,
  llm_modules: 12,
  backend: "responses",
  model: "gpt-5.5",
  base_url: "",
  temperature: 0.2,
  max_summary_tokens: 1600,
  deep_pass_tokens: 5000,
  deep_dive_tokens: 12000,
  reasoning_effort: "high",
  llm_deep_passes: 4,
  llm_report_review: true,
  report_review_tokens: 10000,
  report_review_max_chunks: 0,
  report_review_chunk_chars: 0,
  report_review_memo_chars: 0,
  dtype: "auto",
  pdf: true,
};

export function readReportStudioStorage(): StoredReportStudio {
  if (typeof window === "undefined") return { ...defaultProfileOptions };
  try {
    const stored = JSON.parse(window.localStorage.getItem(REPORT_STUDIO_STORAGE_KEY) || "{}") as StoredReportStudio;
    const merged = { ...defaultProfileOptions, ...stored };
    if (!merged.run && !merged.graph) merged.run = defaultProfileOptions.run;
    if (!merged.out) merged.out = defaultProfileOptions.out;
    return merged;
  } catch {
    return { ...defaultProfileOptions };
  }
}

function writeReportStudioStorage(value: StoredReportStudio) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(REPORT_STUDIO_STORAGE_KEY, JSON.stringify({ ...value, savedAt: Date.now() }));
}

function inferRunFromGraph(graph: GraphPayload | null) {
  if (!graph?.path) return "";
  const match = graph.path.match(/(?:^|\/)(runs\/[^/]+)/);
  return match?.[1] || "";
}

function modelSlug(model: string) {
  return (
    (model || "profile")
      .toLowerCase()
      .replace(/^.*\//, "")
      .replace(/[^a-z0-9]+/g, "")
      .slice(0, 18) || "profile"
  );
}

function defaultProfileOut(run: string, model: string, suffix = "") {
  return `${run || "runs/explorer_run"}/profile_${modelSlug(model)}${suffix}`;
}

const profilePresetHelp = {
  research:
    "Highest-quality report generation. Uses the Responses backend with gpt-5.5, high reasoning, more module summaries, four deep passes, LLM report review, and PDF output.",
  light:
    "Fast LLM report preset. Uses --profile-preset light, six modules, one deep pass, medium reasoning, smaller token budgets, no LLM report review, and PDF output.",
  local:
    "Local draft path for privacy/cost-sensitive iteration. Uses the Responses backend on localhost:8000 and skips PDF by default.",
  fast:
    "Metrics-only audit. Disables LLM reporting and PDF generation; useful to inspect graph statistics quickly before spending model calls.",
} as const;

function graphFileLabel(file: GraphFileSummary) {
  const iter = file.iter == null ? "final" : `iter ${file.iter}`;
  const run = file.run_name || file.run || "graph";
  return `${run} | ${iter} | ${file.name}`;
}

export function ReportStudio({
  onReportReady,
  defaultOpen = false,
}: {
  onReportReady: (out: string) => void;
  defaultOpen?: boolean;
}) {
  const graph = useExplorerStore((state) => state.graph);
  const storedRef = useRef<StoredReportStudio | null>(null);
  if (!storedRef.current) storedRef.current = readReportStudioStorage();
  const stored = storedRef.current;
  const [sourceMode, setSourceMode] = useState<"run" | "graph">(stored.graph ? "graph" : "run");
  const [options, setOptions] = useState<ProfileOptions>({ ...defaultProfileOptions, ...stored });
  const [job, setJob] = useState<ProfileJobStatus | null>(stored.job || null);
  const [status, setStatus] = useState(stored.job ? "Restored saved profile job." : "");
  const [busy, setBusy] = useState(false);

  const runsQuery = useQuery({
    queryKey: ["runs"],
    queryFn: api.runs,
    refetchInterval: 7000,
  });
  const graphmlQuery = useQuery({
    queryKey: ["graphml-files"],
    queryFn: api.graphmlFiles,
    refetchInterval: 9000,
  });
  const reportsQuery = useQuery({
    queryKey: ["profile-reports", options.run],
    queryFn: () => api.profileReports(options.run || ""),
    enabled: Boolean(options.run),
    refetchInterval: 7000,
  });

  useEffect(() => {
    writeReportStudioStorage({ ...options, job, jobId: job?.id, activeReportOut: job?.artifacts?.out || options.out });
  }, [job, options]);

  useEffect(() => {
    const inferred = inferRunFromGraph(graph);
    if (!inferred || (options.run && options.run !== defaultProfileOptions.run)) return;
    patchOptions({ run: inferred, out: defaultProfileOut(inferred, options.model) });
  }, [graph]);

  useEffect(() => {
    const restoredJobId = stored.jobId || stored.job?.id;
    if (!restoredJobId) return undefined;
    let cancelled = false;
    api.profileJob(restoredJobId)
      .then((next) => {
        if (cancelled) return;
        setJob(next);
        if (next.artifacts?.out) onReportReady(next.artifacts.out);
        setStatus(`Reconnected to profile job ${restoredJobId}.`);
      })
      .catch((error) => {
        if (!cancelled) {
          setJob(null);
          setStatus(`Saved profile job is not active: ${error instanceof Error ? error.message : String(error)}`);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!job || !["running", "stopping"].includes(job.status)) return undefined;
    const timer = window.setInterval(async () => {
      try {
        const next = await api.profileJob(job.id);
        setJob(next);
        if (next.artifacts?.out) onReportReady(next.artifacts.out);
        if (next.status === "done") void reportsQuery.refetch();
      } catch (error) {
        setStatus(error instanceof Error ? error.message : String(error));
      }
    }, 2600);
    return () => window.clearInterval(timer);
  }, [job, onReportReady, reportsQuery]);

  function patchOptions(patch: Partial<ProfileOptions>) {
    setOptions((state) => ({ ...state, ...patch }));
  }

  function useCurrentRun() {
    const inferred = inferRunFromGraph(graph);
    if (!inferred) return;
    patchOptions({ run: inferred, out: defaultProfileOut(inferred, options.model) });
  }

  function useCurrentGraphPath() {
    if (!graph?.path) return;
    patchOptions({ graph: graph.path, out: defaultProfileOut(inferRunFromGraph(graph) || "runs/explorer_run", options.model, "_graph") });
  }

  function setPreset(kind: "research" | "light" | "local" | "fast") {
    const run = options.run || inferRunFromGraph(graph) || defaultProfileOptions.run || "runs/exp_leap";
    if (kind === "research") {
      patchOptions({
        profile_preset: "full",
        llm: true,
        backend: "responses",
        model: "gpt-5.5",
        base_url: "",
        reasoning_effort: "high",
        embed_model: "auto",
        llm_modules: 12,
        llm_deep_passes: 4,
        max_summary_tokens: 1600,
        deep_pass_tokens: 5000,
        deep_dive_tokens: 12000,
        report_review_tokens: 10000,
        llm_report_review: true,
        pdf: true,
        out: defaultProfileOut(run, "gpt-5.5"),
      });
    } else if (kind === "light") {
      patchOptions({
        profile_preset: "light",
        llm: true,
        backend: "responses",
        model: "gpt-5.5",
        base_url: "",
        reasoning_effort: "medium",
        embed_model: "auto",
        llm_modules: 6,
        llm_deep_passes: 1,
        max_summary_tokens: 900,
        deep_pass_tokens: 2500,
        deep_dive_tokens: 6000,
        llm_report_review: false,
        pdf: true,
        out: defaultProfileOut(run, "gpt-5.5", "_light"),
      });
    } else if (kind === "local") {
      patchOptions({
        profile_preset: "full",
        llm: true,
        backend: "responses",
        model: "meta-llama/Llama-3.2-3B-Instruct",
        base_url: "http://localhost:8000/v1",
        reasoning_effort: "medium",
        max_summary_tokens: 1200,
        deep_pass_tokens: 3500,
        deep_dive_tokens: 7000,
        report_review_tokens: 7000,
        pdf: false,
        out: defaultProfileOut(run, "llama"),
      });
    } else {
      patchOptions({
        profile_preset: "full",
        llm: false,
        backend: "responses",
        model: "gpt-5.5",
        embed_model: "",
        llm_report_review: false,
        pdf: false,
        out: defaultProfileOut(run, "fast"),
      });
    }
  }

  async function start() {
    const payload: ProfileOptions = { ...options };
    if (sourceMode === "run") {
      payload.graph = "";
    } else {
      payload.run = "";
    }
    if (!payload.out.trim()) {
      payload.out = defaultProfileOut(payload.run || inferRunFromGraph(graph), payload.model);
    }
    setBusy(true);
    setStatus("");
    try {
      const next = await api.profileGraph(payload);
      setJob(next);
      onReportReady(next.artifacts?.out || next.out);
      setOptions(payload);
      setStatus(`Started profile job ${next.id}.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  async function stop() {
    if (!job) return;
    const next = await api.stopProfileJob(job.id);
    setJob(next);
    setStatus(`Stop requested for profile job ${next.id}.`);
  }

  const progress = job?.status === "done" ? 100 : Math.round((job?.progress?.percent || 0) * 100);
  const canStart = sourceMode === "run" ? Boolean(options.run?.trim()) : Boolean(options.graph?.trim());

  return (
    <Drawer
      defaultOpen={defaultOpen}
      description="Generate and review graph profile reports. It runs profile_graph.py, tracks progress from logs, lists report folders, renders report.md, and exposes PDF/JSON artifacts when present."
      icon={<FileText size={14} />}
      note={job?.status || "profile"}
      title="Report Studio"
    >
      <div className="segmented-control">
        <button className={cx(sourceMode === "run" && "active")} onClick={() => setSourceMode("run")} type="button">
          Run Folder
        </button>
        <button className={cx(sourceMode === "graph" && "active")} onClick={() => setSourceMode("graph")} type="button">
          GraphML
        </button>
      </div>
      <div className="source-help">
        {sourceMode === "run" ? (
          <>
            <strong>Run folder source</strong>
            <span>
              Profiles the latest GraphML snapshot in the selected run folder. Existing report folders for that run are listed below.
            </span>
          </>
        ) : (
          <>
            <strong>GraphML source</strong>
            <span>
              Profiles one exact server-visible GraphML file. Browser uploads are in memory only unless the file also exists on disk.
            </span>
          </>
        )}
      </div>
      {sourceMode === "run" ? (
        <div className="source-picker">
          <label>
            Available runs
            <select
              onChange={(event) => {
                const run = event.target.value;
                if (run) patchOptions({ run, out: defaultProfileOut(run, options.model) });
              }}
              value={runsQuery.data?.runs.some((run) => run.path === options.run) ? options.run || "" : ""}
            >
              <option value="">Choose a run folder...</option>
              {(runsQuery.data?.runs || []).map((run) => (
                <option disabled={!run.graph_ready} key={run.path} value={run.path}>
                  {run.name} {run.snapshot_count ? `| ${run.snapshot_count} snapshots` : ""} {run.graph_ready ? "" : "| no graph"}
                </option>
              ))}
            </select>
          </label>
          <input
            onChange={(event) => patchOptions({ run: event.target.value })}
            placeholder="runs/exp_leap"
            value={options.run || ""}
          />
          <IconButton disabled={!inferRunFromGraph(graph)} icon={<Network size={14} />} label="Current" onClick={useCurrentRun} />
        </div>
      ) : (
        <div className="source-picker">
          <label>
            Available GraphML
            <select
              onChange={(event) => {
                const graphPath = event.target.value;
                if (graphPath) patchOptions({ graph: graphPath, out: defaultProfileOut(options.run || "runs/explorer_run", options.model, "_graph") });
              }}
              value={graphmlQuery.data?.graphs.some((item) => item.path === options.graph || item.absolute_path === options.graph) ? options.graph || "" : ""}
            >
              <option value="">Choose a GraphML file...</option>
              {(graphmlQuery.data?.graphs || []).map((file) => (
                <option key={file.absolute_path || file.path} value={file.path}>
                  {graphFileLabel(file)}
                </option>
              ))}
            </select>
          </label>
          <input
            onChange={(event) => patchOptions({ graph: event.target.value })}
            placeholder="/path/to/file.graphml"
            value={options.graph || ""}
          />
          <IconButton disabled={!graph?.path} icon={<Network size={14} />} label="Current" onClick={useCurrentGraphPath} />
        </div>
      )}
      <label>
        Output
        <input
          onChange={(event) => patchOptions({ out: event.target.value })}
          placeholder="runs/exp_leap/profile_gpt55"
          value={options.out}
        />
      </label>

      <div className="preset-grid">
        <button onClick={() => setPreset("research")} title={profilePresetHelp.research} type="button">
          <strong>GPT-5.5 Deep <HelpTip text={profilePresetHelp.research} /></strong>
          <span>responses | high effort</span>
        </button>
        <button onClick={() => setPreset("light")} title={profilePresetHelp.light} type="button">
          <strong>GPT-5.5 Light <HelpTip text={profilePresetHelp.light} /></strong>
          <span>profile-preset light</span>
        </button>
        <button onClick={() => setPreset("local")} title={profilePresetHelp.local} type="button">
          <strong>Local Draft <HelpTip text={profilePresetHelp.local} /></strong>
          <span>chat | localhost:8000</span>
        </button>
        <button onClick={() => setPreset("fast")} title={profilePresetHelp.fast} type="button">
          <strong>Fast Audit <HelpTip text={profilePresetHelp.fast} /></strong>
          <span>metrics only</span>
        </button>
      </div>

      <div className="profile-progress">
        <div>
          <b>{progress}%</b>
          <span>{job?.progress?.message || "Idle"}</span>
        </div>
        <progress max={100} value={progress} />
        {job?.progress?.detail ? <span>{job.progress.detail}</span> : null}
      </div>
      <div className="button-row">
        <IconButton
          disabled={busy || !canStart}
          description="Start profile_graph.py with the selected source, preset, model, and output folder."
          icon={busy ? <Loader2 className="spin" size={14} /> : <Play size={14} />}
          label="Generate"
          onClick={start}
          tone="primary"
        />
        <IconButton
          disabled={!job || job.status !== "running"}
          description="Stop the active profile report job and keep any partial output."
          icon={<CircleStop size={14} />}
          label="Stop"
          onClick={stop}
          tone="danger"
        />
      </div>

      <details className="settings-block">
        <summary>Model and LLM</summary>
        <div className="control-grid">
          <label>
            LLM
            <select value={options.llm ? "on" : "off"} onChange={(event) => patchOptions({ llm: event.target.value === "on" })}>
              <option value="on">On</option>
              <option value="off">Off</option>
            </select>
          </label>
          <label>
            Backend
            <select value={options.backend} onChange={(event) => patchOptions({ backend: event.target.value as ProfileOptions["backend"] })}>
              <option value="responses">Responses</option>
            </select>
          </label>
          <label>
            Model
            <input value={options.model} onChange={(event) => patchOptions({ model: event.target.value })} />
          </label>
          <label>
            Base URL
            <input value={options.base_url || ""} onChange={(event) => patchOptions({ base_url: event.target.value })} />
          </label>
          <label>
            Effort
            <select
              value={options.reasoning_effort || "high"}
              onChange={(event) => patchOptions({ reasoning_effort: event.target.value as ProfileOptions["reasoning_effort"] })}
            >
              <option value="minimal">Minimal</option>
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
            </select>
          </label>
          <label>
            Temperature
            <input
              max={1}
              min={0}
              onChange={(event) => patchOptions({ temperature: Number(event.target.value) })}
              step="0.05"
              type="number"
              value={options.temperature ?? 0.2}
            />
          </label>
        </div>
      </details>

      <details className="settings-block">
        <summary>Report Options</summary>
        <div className="control-grid">
          <label>
            Profile preset
            <select
              value={options.profile_preset || "full"}
              onChange={(event) => {
                if (event.target.value === "light") setPreset("light");
                else patchOptions({ profile_preset: "full" });
              }}
            >
              <option value="full">Full</option>
              <option value="light">Light</option>
            </select>
          </label>
          <label>
            Embed model
            <input value={options.embed_model || ""} onChange={(event) => patchOptions({ embed_model: event.target.value })} />
          </label>
          <label>
            Modules
            <input min={1} onChange={(event) => patchOptions({ llm_modules: Number(event.target.value) })} type="number" value={options.llm_modules ?? 12} />
          </label>
          <label>
            Deep passes
            <input min={0} onChange={(event) => patchOptions({ llm_deep_passes: Number(event.target.value) })} type="number" value={options.llm_deep_passes ?? 4} />
          </label>
          <label>
            Summary tokens
            <input min={100} onChange={(event) => patchOptions({ max_summary_tokens: Number(event.target.value) })} type="number" value={options.max_summary_tokens ?? 1600} />
          </label>
          <label>
            Pass tokens
            <input min={100} onChange={(event) => patchOptions({ deep_pass_tokens: Number(event.target.value) })} type="number" value={options.deep_pass_tokens ?? 5000} />
          </label>
          <label>
            Deep dive tokens
            <input min={100} onChange={(event) => patchOptions({ deep_dive_tokens: Number(event.target.value) })} type="number" value={options.deep_dive_tokens ?? 12000} />
          </label>
          <label>
            Review tokens
            <input min={100} onChange={(event) => patchOptions({ report_review_tokens: Number(event.target.value) })} type="number" value={options.report_review_tokens ?? 10000} />
          </label>
          <label>
            PDF
            <select value={options.pdf === false ? "off" : "on"} onChange={(event) => patchOptions({ pdf: event.target.value === "on" })}>
              <option value="on">On</option>
              <option value="off">Off</option>
            </select>
          </label>
        </div>
      </details>
      {status ? <div className="status-box">{status}</div> : null}
      {job?.cmd?.length ? <pre className="code-preview">{job.cmd.join(" ")}</pre> : null}
      {job?.log_tail ? <pre className="run-log">{job.log_tail}</pre> : null}
    </Drawer>
  );
}

function isMarkdownTableLine(line: string) {
  return line.trim().startsWith("|") && line.trim().endsWith("|");
}

function renderInline(text: string) {
  const tokens = text.split(/(`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\]\([^)]+\))/g).filter(Boolean);
  return tokens.map((token, index) => {
    if (token.startsWith("`") && token.endsWith("`")) return <code key={index}>{token.slice(1, -1)}</code>;
    if (token.startsWith("**") && token.endsWith("**")) return <strong key={index}>{token.slice(2, -2)}</strong>;
    const link = token.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
    if (link) {
      return (
        <a href={link[2]} key={index} rel="noreferrer" target="_blank">
          {link[1]}
        </a>
      );
    }
    return <React.Fragment key={index}>{token}</React.Fragment>;
  });
}

export function MarkdownReport({ markdown, out, assetUrl }: { markdown: string; out: string; assetUrl?: (file: string) => string }) {
  const blocks: React.ReactNode[] = [];
  const lines = markdown.split(/\r?\n/);
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();
    if (!trimmed) {
      i += 1;
      continue;
    }
    if (trimmed.startsWith("```")) {
      const code: string[] = [];
      i += 1;
      while (i < lines.length && !lines[i].trim().startsWith("```")) {
        code.push(lines[i]);
        i += 1;
      }
      i += 1;
      blocks.push(<pre key={blocks.length}>{code.join("\n")}</pre>);
      continue;
    }
    const heading = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      if (heading[1].length === 1) blocks.push(<h1 key={blocks.length}>{renderInline(heading[2])}</h1>);
      else if (heading[1].length === 2) blocks.push(<h2 key={blocks.length}>{renderInline(heading[2])}</h2>);
      else if (heading[1].length === 3) blocks.push(<h3 key={blocks.length}>{renderInline(heading[2])}</h3>);
      else blocks.push(<h4 key={blocks.length}>{renderInline(heading[2])}</h4>);
      i += 1;
      continue;
    }
    const image = trimmed.match(/^!\[([^\]]*)\]\(([^)]+)\)$/);
    if (image) {
      const file = image[2].replace(/^\.?\//, "");
      const src = /^https?:\/\//.test(image[2]) ? image[2] : assetUrl ? assetUrl(file) : api.reportAssetUrl(out, file);
      blocks.push(<img alt={image[1]} key={blocks.length} src={src} />);
      i += 1;
      continue;
    }
    if (isMarkdownTableLine(trimmed)) {
      const rows: string[][] = [];
      while (i < lines.length && isMarkdownTableLine(lines[i])) {
        const cells = lines[i].trim().slice(1, -1).split("|").map((cell) => cell.trim());
        if (!cells.every((cell) => /^:?-{3,}:?$/.test(cell))) rows.push(cells);
        i += 1;
      }
      const [head, ...body] = rows;
      blocks.push(
        <div className="report-table-wrap" key={blocks.length}>
          <table>
            {head ? (
              <thead>
                <tr>{head.map((cell, index) => <th key={index}>{renderInline(cell)}</th>)}</tr>
              </thead>
            ) : null}
            <tbody>
              {body.map((row, rowIndex) => (
                <tr key={rowIndex}>{row.map((cell, cellIndex) => <td key={cellIndex}>{renderInline(cell)}</td>)}</tr>
              ))}
            </tbody>
          </table>
        </div>,
      );
      continue;
    }
    if (/^[-*]\s+/.test(trimmed)) {
      const items: string[] = [];
      while (i < lines.length && /^[-*]\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^[-*]\s+/, ""));
        i += 1;
      }
      blocks.push(<ul key={blocks.length}>{items.map((item, index) => <li key={index}>{renderInline(item)}</li>)}</ul>);
      continue;
    }
    if (trimmed.startsWith(">")) {
      const quote: string[] = [];
      while (i < lines.length && lines[i].trim().startsWith(">")) {
        quote.push(lines[i].trim().replace(/^>\s?/, ""));
        i += 1;
      }
      blocks.push(<blockquote key={blocks.length}>{quote.map((item, index) => <p key={index}>{renderInline(item)}</p>)}</blockquote>);
      continue;
    }
    const paragraph = [trimmed];
    i += 1;
    while (
      i < lines.length &&
      lines[i].trim() &&
      !lines[i].trim().startsWith("#") &&
      !lines[i].trim().startsWith("```") &&
      !lines[i].trim().startsWith("!") &&
      !isMarkdownTableLine(lines[i]) &&
      !/^[-*]\s+/.test(lines[i].trim())
    ) {
      paragraph.push(lines[i].trim());
      i += 1;
    }
    blocks.push(<p key={blocks.length}>{renderInline(paragraph.join(" "))}</p>);
  }
  return <article className="markdown-report">{blocks}</article>;
}

function inferRunFromReportOut(out = "") {
  const parts = out.split("/").filter(Boolean);
  const profileIndex = parts.findIndex((part) => part.startsWith("profile"));
  if (profileIndex > 0) return parts.slice(0, profileIndex).join("/");
  const match = out.match(/(?:^|\/)(runs\/[^/]+)/);
  return match?.[1] || "";
}

function profileLabel(report: ProfileArtifacts) {
  return report.out.split("/").pop() || report.out;
}

function ProfilePicker({
  reports,
  activeOut,
  onSelect,
}: {
  reports: ProfileArtifacts[];
  activeOut: string;
  onSelect: (out: string) => void;
}) {
  if (!reports.length) return null;
  return (
    <div className="profile-picker">
      <div className="profile-picker-head">
        <strong>Existing profiles</strong>
        <span>{reports.length} found</span>
      </div>
      <div className="profile-picker-grid">
        {reports.map((report) => (
          <button className={cx(report.out === activeOut && "active")} key={report.out} onClick={() => onSelect(report.out)} type="button">
            <strong>{profileLabel(report)}</strong>
            <span>
              {formatRunTime(report.updated_at)} | {formatNumber(report.summary.nodes || 0)} nodes |{" "}
              {formatNumber(report.summary.modules || 0)} modules
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}

export function ReportStage({
  out,
  run,
  onOpenReports,
  onReportReady,
}: {
  out: string;
  run: string;
  onOpenReports: () => void;
  onReportReady: (out: string) => void;
}) {
  const reportRun = run || inferRunFromReportOut(out);
  const reportQuery = useQuery<ProfileReportPayload>({
    queryKey: ["profile-report", out],
    queryFn: () => api.profileReport(out),
    enabled: Boolean(out),
    refetchInterval: (query) => (query.state.data?.artifacts.ready ? false : 5000),
  });
  const reportsQuery = useQuery({
    queryKey: ["profile-reports-stage", reportRun],
    queryFn: () => api.profileReports(reportRun),
    enabled: Boolean(reportRun),
    refetchInterval: 8000,
  });
  const artifacts = reportQuery.data?.artifacts;
  const summary = artifacts?.summary || {};
  const reports = reportsQuery.data?.reports || [];
  const markdown = reportQuery.data?.markdown || "";
  const hasMarkdown = Boolean(markdown);
  const selectedName = out ? out.split("/").pop() || out : "";

  if (!hasMarkdown) {
    return (
      <section className="report-stage">
        <div className="report-stage-empty-toolbar">
          <div>
            <strong>Profile reports</strong>
            <span>{reportRun || "No run selected"}</span>
          </div>
          <button onClick={onOpenReports} type="button">
            Generate report
          </button>
        </div>
        <div className="report-scroll">
          <div className="report-empty">
            <FileText size={23} />
            <strong>{out ? "No rendered report.md yet" : "No profile report selected"}</strong>
            <span>
              {out
                ? `Selected ${selectedName}, but report.md is not available yet.`
                : reports.length
                  ? "Choose an existing profile below or generate a new one."
                  : "Generate a profile report to render report.md here."}
            </span>
            {reportQuery.isLoading ? (
              <span className="inline-status">
                <Loader2 className="spin" size={13} /> Checking selected report...
              </span>
            ) : null}
            {reportQuery.error ? <span className="inline-status">{String(reportQuery.error)}</span> : null}
            <button onClick={onOpenReports} type="button">
              Generate report
            </button>
          </div>
          <ProfilePicker activeOut={out} onSelect={onReportReady} reports={reports} />
        </div>
      </section>
    );
  }

  return (
    <section className="report-stage">
      <div className="artifact-toolbar">
        <div>
          <strong>{summary.topic || "Profile report"}</strong>
          <span>{out}</span>
        </div>
        <div className="artifact-actions">
          <button onClick={() => void reportQuery.refetch()} type="button">
            Refresh
          </button>
          {artifacts?.pdf_path ? (
            <a href={api.reportAssetUrl(out, "report.pdf")} rel="noreferrer" target="_blank">
              <ExternalLink size={13} /> PDF
            </a>
          ) : null}
          {artifacts?.profile_path ? (
            <a href={api.reportAssetUrl(out, "profile.json")} rel="noreferrer" target="_blank">
              <ExternalLink size={13} /> JSON
            </a>
          ) : null}
          <button onClick={onOpenReports} type="button">
            Controls
          </button>
        </div>
      </div>
      <div className="report-scroll">
        <ProfilePicker activeOut={out} onSelect={onReportReady} reports={reports} />
        {reportQuery.isLoading ? <div className="status-box">Loading report...</div> : null}
        {reportQuery.error ? <div className="status-box">{String(reportQuery.error)}</div> : null}
        {hasMarkdown ? (
          <MarkdownReport markdown={markdown} out={out} />
        ) : (
          <div className="status-box">
            {artifacts?.ready ? "Report markdown is empty." : "Report job is still preparing artifacts."}
          </div>
        )}
      </div>
    </section>
  );
}
