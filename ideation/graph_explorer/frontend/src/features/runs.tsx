import { useQuery } from "@tanstack/react-query";
import { CircleStop, FolderOpen, Loader2, Play, Rocket, RotateCcw, Sparkles, Upload } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { cx, Drawer, formatRunTime, IconButton } from "../components/common";
import { formatNumber } from "../graph-utils";
import { useExplorerStore } from "../store";
import type { GraphFileSummary, GraphPayload, JobStatus, RunSummary } from "../types";

const RUN_MONITOR_STORAGE_KEY = "graph-preflexor-explorer.run-monitor.v1";
const IDEATION_STRATEGIES = ["frontier", "node", "answer", "edge", "novelty", "leap", "converse", "mixed"] as const;
type IdeationStrategy = (typeof IDEATION_STRATEGIES)[number];

function normalizeStrategy(value: string | undefined): IdeationStrategy {
  return IDEATION_STRATEGIES.includes(value as IdeationStrategy) ? (value as IdeationStrategy) : "frontier";
}

type StoredRunMonitor = {
  topic?: string;
  strategy?: string;
  calls?: number;
  iters?: number;
  out?: string;
  job?: JobStatus | null;
  jobId?: string;
};

function readRunMonitorStorage(): StoredRunMonitor {
  if (typeof window === "undefined") return {};
  try {
    return JSON.parse(window.localStorage.getItem(RUN_MONITOR_STORAGE_KEY) || "{}") as StoredRunMonitor;
  } catch {
    return {};
  }
}

function writeRunMonitorStorage(value: StoredRunMonitor) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(RUN_MONITOR_STORAGE_KEY, JSON.stringify({ ...value, savedAt: Date.now() }));
}

export function RunExplorer({
  onLoadRun,
  onGraphLoaded,
  defaultOpen = false,
}: {
  onLoadRun: (run: string) => Promise<void>;
  onGraphLoaded?: (graph: GraphPayload) => void;
  defaultOpen?: boolean;
}) {
  const graph = useExplorerStore((state) => state.graph);
  const [filter, setFilter] = useState("");
  const [status, setStatus] = useState("");
  const [loadingRun, setLoadingRun] = useState("");
  const [manualPath, setManualPath] = useState("");
  const [uploading, setUploading] = useState(false);
  const runsQuery = useQuery({
    queryKey: ["runs"],
    queryFn: api.runs,
    refetchInterval: 5000,
  });
  const graphmlQuery = useQuery({
    queryKey: ["graphml-files"],
    queryFn: api.graphmlFiles,
    refetchInterval: 10000,
  });
  const runs = useMemo(() => {
    const query = filter.trim().toLowerCase();
    const items = runsQuery.data?.runs || [];
    if (!query) return items;
    return items.filter((run) =>
      [run.name, run.path, run.topic, run.strategy].some((value) => String(value || "").toLowerCase().includes(query)),
    );
  }, [filter, runsQuery.data?.runs]);

  async function loadRun(run: RunSummary) {
    setLoadingRun(run.path);
    setStatus("");
    try {
      await onLoadRun(run.path);
      setStatus(`Loaded ${run.name}.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setLoadingRun("");
    }
  }

  async function loadManualPath() {
    const path = manualPath.trim();
    if (!path) return;
    setLoadingRun(path);
    setStatus("");
    try {
      await onLoadRun(path);
      setStatus(`Loaded ${path}.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setLoadingRun("");
    }
  }

  async function uploadGraphml(file: File | undefined) {
    if (!file) return;
    setUploading(true);
    setStatus("");
    try {
      const text = await file.text();
      const next = await api.uploadGraphml(file.name, text);
      if (onGraphLoaded) onGraphLoaded(next);
      setStatus(`Uploaded ${file.name}.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setUploading(false);
    }
  }

  async function loadGraphFile(file: GraphFileSummary) {
    const path = file.absolute_path || file.path || "";
    if (!path) return;
    setLoadingRun(path);
    setStatus("");
    try {
      setManualPath(file.path || path);
      await onLoadRun(path);
      setStatus(`Loaded ${file.name}.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setLoadingRun("");
    }
  }

  return (
    <Drawer
      defaultOpen={defaultOpen}
      description="Load existing run folders, individual GraphML paths, or uploaded GraphML files. Loaded graphs become the active viewer, search, and chat context while you stay in this Runs tool."
      icon={<FolderOpen size={14} />}
      note={`${runsQuery.data?.runs.length || 0} folders`}
      title="Run Explorer"
    >
      <div className="run-load-card">
        <label>
          Run folder or GraphML path
          <div className="run-path-row">
            <input
              onChange={(event) => setManualPath(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") void loadManualPath();
              }}
              placeholder="runs/exp_leap or /path/to/file.graphml"
              value={manualPath}
            />
            <IconButton
              disabled={!manualPath.trim() || Boolean(loadingRun)}
              description="Load a previous run folder or a single GraphML file path."
              icon={loadingRun === manualPath.trim() ? <Loader2 className="spin" size={14} /> : <FolderOpen size={14} />}
              label="Load"
              onClick={loadManualPath}
            />
          </div>
        </label>
        <label className="file-button run-upload-button" title="Upload a GraphML/XML file from this machine and make it the active graph context.">
          {uploading ? <Loader2 className="spin" size={14} /> : <Upload size={14} />}
          Upload GraphML
          <input accept=".graphml,.xml" disabled={uploading} onChange={(event) => void uploadGraphml(event.target.files?.[0])} type="file" />
        </label>
      </div>
      <div className="row run-filter-row">
        <input
          onChange={(event) => setFilter(event.target.value)}
          placeholder="filter runs, topics, strategies"
          value={filter}
        />
        <IconButton
          disabled={runsQuery.isFetching}
          description="Rescan run folders and refresh graph readiness, node counts, and status metadata."
          icon={runsQuery.isFetching ? <Loader2 className="spin" size={14} /> : <RotateCcw size={14} />}
          label="Refresh"
          onClick={() => void runsQuery.refetch()}
        />
      </div>
      {graphmlQuery.data?.graphs?.length ? (
        <div className="snapshot-browser compact">
          <div className="snapshot-head">
            <div>
              <strong>Available GraphML</strong>
              <span>{graphmlQuery.data.graphs.length} files under ideation/runs</span>
            </div>
            <IconButton
              disabled={graphmlQuery.isFetching}
              description="Refresh discovered GraphML files."
              icon={graphmlQuery.isFetching ? <Loader2 className="spin" size={14} /> : <RotateCcw size={14} />}
              label="Refresh"
              onClick={() => void graphmlQuery.refetch()}
            />
          </div>
          <div className="snapshot-list compact">
            {graphmlQuery.data.graphs.slice(0, 10).map((item) => {
              const path = item.absolute_path || item.path;
              return (
                <button
                  className={cx(graph?.path === path && "active")}
                  disabled={loadingRun === path}
                  key={path}
                  onClick={() => void loadGraphFile(item)}
                  title="Load this GraphML file into the active graph context."
                  type="button"
                >
                  <strong>{item.run_name ? `${item.run_name} / ${item.name}` : item.name}</strong>
                  <span>{item.path}</span>
                </button>
              );
            })}
          </div>
        </div>
      ) : null}
      <div className="run-list">
        {runs.map((run) => {
          const isCurrent = Boolean(graph?.path && (graph.path.includes(`/${run.name}/`) || graph.path.endsWith(`${run.name}/graph.graphml`)));
          return (
            <button
              className={cx("run-item", isCurrent && "active")}
              disabled={!run.graph_ready || loadingRun === run.path}
              key={run.path}
              onClick={() => loadRun(run)}
              title={run.graph_ready ? `Load ${run.path}` : "No graph snapshot is available yet"}
              type="button"
            >
              <div className="run-row">
                <strong>{run.name}</strong>
                <span>{formatRunTime(run.updated_at)}</span>
              </div>
              <span className="run-topic">{run.topic || run.path}</span>
              <div className="run-metrics">
                <span>{formatNumber(run.nodes)} nodes</span>
                <span>{formatNumber(run.edges)} edges</span>
                {run.snapshot_count ? <span>{formatNumber(run.snapshot_count)} snapshots</span> : null}
                <span>{run.strategy || "strategy?"}</span>
                <span>{run.graph_ready ? "ready" : "pending"}</span>
              </div>
            </button>
          );
        })}
        {!runs.length ? <div className="status-box">{runsQuery.error ? String(runsQuery.error) : "No runs found."}</div> : null}
      </div>
      {status ? <div className="status-box">{status}</div> : null}
    </Drawer>
  );
}

export function RunMonitor({
  onRunGraphReady,
  onRunStart,
  defaultOpen = false,
}: {
  onRunGraphReady: (run: string) => Promise<void>;
  onRunStart?: () => Promise<void> | void;
  defaultOpen?: boolean;
}) {
  const storedRunRef = useRef<StoredRunMonitor | null>(null);
  if (!storedRunRef.current) storedRunRef.current = readRunMonitorStorage();
  const storedRun = storedRunRef.current;
  const roles = useExplorerStore((state) => state.roles);
  const chatRole = useExplorerStore((state) => state.chatRole);
  const [topic, setTopic] = useState(storedRun.topic || "");
  const [strategy, setStrategy] = useState<IdeationStrategy>(normalizeStrategy(storedRun.strategy));
  const [calls, setCalls] = useState(storedRun.calls || 50);
  const [iters, setIters] = useState(storedRun.iters || 50);
  const [out, setOut] = useState(storedRun.out || "runs/explorer_run");
  const [job, setJob] = useState<JobStatus | null>(storedRun.job || null);
  const [monitorStatus, setMonitorStatus] = useState(storedRun.job ? "Restored saved exploration run." : "");
  const [busy, setBusy] = useState(false);
  const [suggestingOut, setSuggestingOut] = useState(false);
  const lastSnapshotRef = useRef<string>("");
  const outTouchedRef = useRef(false);
  const autoOutRef = useRef(out);
  const suggestSeqRef = useRef(0);
  const namingRole = roles[chatRole] || roles.chat || roles.questioner || roles.graph_qa;

  useEffect(() => {
    writeRunMonitorStorage({ topic, strategy, calls, iters, out, job, jobId: job?.id });
  }, [calls, iters, job, out, strategy, topic]);

  async function suggestOutName(force = false) {
    const task = topic.trim();
    if (!task || task.length < 8) return;
    if (!force && outTouchedRef.current && out !== autoOutRef.current) return;
    const seq = suggestSeqRef.current + 1;
    suggestSeqRef.current = seq;
    setSuggestingOut(true);
    try {
      const suggestion = await api.suggestRunOut({ topic: task, strategy, model_config: namingRole });
      if (seq !== suggestSeqRef.current) return;
      autoOutRef.current = suggestion.out;
      setOut(suggestion.out);
      if (force) {
        setMonitorStatus(
          suggestion.source === "model"
            ? `Suggested output folder: ${suggestion.out}`
            : `Suggested output folder from fallback slug: ${suggestion.out}`,
        );
      }
    } catch (error) {
      if (force) setMonitorStatus(error instanceof Error ? error.message : String(error));
    } finally {
      if (seq === suggestSeqRef.current) setSuggestingOut(false);
    }
  }

  useEffect(() => {
    const task = topic.trim();
    if (!task || task.length < 8) return undefined;
    if (outTouchedRef.current && out !== autoOutRef.current) return undefined;
    const timer = window.setTimeout(() => {
      void suggestOutName(false);
    }, 1100);
    return () => window.clearTimeout(timer);
  }, [topic, strategy]);

  useEffect(() => {
    const restoredJobId = storedRun.jobId || storedRun.job?.id;
    if (!restoredJobId) return undefined;
    let cancelled = false;
    api.job(restoredJobId)
      .then((next) => {
        if (!cancelled) {
          setJob(next);
          setMonitorStatus(`Reconnected to run ${restoredJobId}.`);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setJob(null);
          setMonitorStatus(`Saved run ${restoredJobId} is not active on this server: ${error instanceof Error ? error.message : String(error)}`);
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
        const next = await api.job(job.id);
        setJob(next);
        setMonitorStatus("");
        void syncRunGraph(next);
      } catch (error) {
        setMonitorStatus(error instanceof Error ? error.message : String(error));
      }
    }, 2200);
    return () => window.clearInterval(timer);
  }, [job]);

  async function syncRunGraph(next: JobStatus) {
    const snapshot = next.snapshot_id || (next.graph_ready ? `${next.graph_path || next.out}:ready` : "");
    if (!next.graph_ready || !next.out || !snapshot || snapshot === lastSnapshotRef.current) return;
    lastSnapshotRef.current = snapshot;
    try {
      await onRunGraphReady(next.out);
    } catch (error) {
      setMonitorStatus(error instanceof Error ? error.message : String(error));
    }
  }

  async function start() {
    if (!topic.trim()) return;
    setBusy(true);
    try {
      setJob(null);
      setMonitorStatus("Clearing current graph and starting run...");
      lastSnapshotRef.current = "";
      try {
        await onRunStart?.();
      } catch (error) {
        setMonitorStatus(error instanceof Error ? error.message : String(error));
      }
      const next = await api.ideate({ topic, strategy, budget_calls: calls, max_iters: iters, out, clear_output: true });
      setJob(next);
      if (next.out) setOut(next.out);
      setMonitorStatus(`Started run ${next.id}. Active graph will follow ${next.out} as snapshots appear.`);
      void syncRunGraph(next);
    } finally {
      setBusy(false);
    }
  }

  async function stop() {
    if (!job) return;
    const next = await api.stopJob(job.id);
    setJob(next);
    setMonitorStatus(`Stop requested for run ${next.id}.`);
    void syncRunGraph(next);
  }

  const progress = job?.status === "done" ? 100 : Math.round((job?.progress?.percent || 0) * 100);

  return (
    <Drawer
      defaultOpen={defaultOpen}
      description="Launch a new ideation run, stop it, and monitor progress from the run logs. The monitor persists so navigation within the app will not lose the run."
      icon={<Rocket size={14} />}
      note={job?.status || "idle"}
      title="New Exploration Run"
    >
      <textarea onChange={(event) => setTopic(event.target.value)} placeholder="topic or benchmark task" rows={3} value={topic} />
      <div className="control-grid">
        <label>
          Strategy
          <select onChange={(event) => setStrategy(event.target.value as IdeationStrategy)} value={strategy}>
            {IDEATION_STRATEGIES.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </label>
        <label>
          Calls
          <input min={1} onChange={(event) => setCalls(Number(event.target.value))} type="number" value={calls} />
        </label>
        <label>
          Iters
          <input min={1} onChange={(event) => setIters(Number(event.target.value))} type="number" value={iters} />
        </label>
        <label>
          Output folder
          <div className="run-path-row">
            <input
              onChange={(event) => {
                outTouchedRef.current = true;
                setOut(event.target.value);
              }}
              value={out}
            />
            <button
              aria-label="Suggest output folder"
              className="icon-field-button"
              disabled={!topic.trim() || suggestingOut}
              onClick={() => {
                outTouchedRef.current = false;
                void suggestOutName(true);
              }}
              title="Suggest a short valid run folder from the topic using the configured model, with a deterministic fallback."
              type="button"
            >
              {suggestingOut ? <Loader2 className="spin" size={14} /> : <Sparkles size={14} />}
            </button>
          </div>
        </label>
      </div>
      <div className="progress-card">
        <div>
          <b>{progress}%</b>
          <span>
            {formatNumber(job?.progress?.nodes || 0)} nodes | {formatNumber(job?.progress?.edges || 0)} edges
          </span>
        </div>
        <progress max={100} value={progress} />
      </div>
      <div className="button-row">
        <IconButton
          disabled={busy}
          description="Start a new ideation run with the selected strategy, call budget, iteration count, and output folder."
          icon={busy ? <Loader2 className="spin" size={14} /> : <Play size={14} />}
          label="Start"
          onClick={start}
          tone="primary"
        />
        <IconButton
          disabled={!job || job.status !== "running"}
          description="Ask the running process to stop and keep the partial output folder."
          icon={<CircleStop size={14} />}
          label="Stop"
          onClick={stop}
          tone="danger"
        />
      </div>
      {(monitorStatus || job?.log_tail) ? (
        <details className="run-status-log">
          <summary>
            <span>Run status & log</span>
            <em>{job?.status || "idle"}</em>
          </summary>
          {monitorStatus ? <div className="status-box">{monitorStatus}</div> : null}
          {job?.log_tail ? <pre className="run-log">{job.log_tail}</pre> : null}
        </details>
      ) : null}
    </Drawer>
  );
}
