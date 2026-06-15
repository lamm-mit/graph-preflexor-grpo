import { useQuery } from "@tanstack/react-query";
import { Activity, BarChart3, Braces, FileText, GitBranch, Image as ImageIcon, Loader2, Network, RotateCcw, Route, Square, Zap } from "lucide-react";
import React, { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { formatRunTime } from "../components/common";
import { MarkdownReport } from "./reporting";
import { formatNumber } from "../graph-utils";
import type { RunAnalysisFigure, RunAnalysisJson, RunAnalysisJobStatus, RunDashboard, RunDashboardInsight } from "../types";

type XY = { x: number; y: number };
type Series = { label: string; color: string; points: XY[] };

function extent(values: number[]) {
  const clean = values.filter((value) => Number.isFinite(value));
  if (!clean.length) return [0, 1] as const;
  const min = Math.min(...clean);
  const max = Math.max(...clean);
  if (min === max) return [Math.min(0, min), max + 1] as const;
  return [min, max] as const;
}

function pathFor(points: XY[], xMin: number, xMax: number, yMin: number, yMax: number) {
  const w = 520;
  const h = 154;
  const pad = { l: 34, r: 12, t: 12, b: 24 };
  const sx = (x: number) => pad.l + ((x - xMin) / Math.max(1e-9, xMax - xMin)) * (w - pad.l - pad.r);
  const sy = (y: number) => h - pad.b - ((y - yMin) / Math.max(1e-9, yMax - yMin)) * (h - pad.t - pad.b);
  return points.map((point, index) => `${index ? "L" : "M"}${sx(point.x).toFixed(2)},${sy(point.y).toFixed(2)}`).join(" ");
}

function LineChart({ series, yHint, xHint = "iter" }: { series: Series[]; yHint?: string; xHint?: string }) {
  const all = series.flatMap((item) => item.points);
  if (!all.length) return <div className="plot-empty">No time-series data yet.</div>;
  const [xMin, xMax] = extent(all.map((point) => point.x));
  const [yMin, yMax] = extent(all.map((point) => point.y));
  return (
    <div className="native-chart">
      <svg role="img" viewBox="0 0 520 154">
        <line className="axis" x1="34" x2="508" y1="130" y2="130" />
        <line className="axis" x1="34" x2="34" y1="12" y2="130" />
        {[0, 0.5, 1].map((tick) => {
          const y = 130 - tick * 118;
          const value = yMin + tick * (yMax - yMin);
          return (
            <g key={tick}>
              <line className="grid" x1="34" x2="508" y1={y} y2={y} />
              <text x="4" y={y + 4}>
                {formatNumber(value, value < 10 ? 2 : 0)}
              </text>
            </g>
          );
        })}
        {series.map((item) => (
          <path d={pathFor(item.points, xMin, xMax, yMin, yMax)} fill="none" key={item.label} stroke={item.color} strokeWidth="2.4" />
        ))}
        <text x="34" y="149">
          {xHint} {formatNumber(xMin)} to {formatNumber(xMax)}
        </text>
        {yHint ? (
          <text className="chart-hint" x="508" y="149">
            {yHint}
          </text>
        ) : null}
      </svg>
      <div className="chart-legend">
        {series.map((item) => (
          <span key={item.label}>
            <i style={{ background: item.color }} />
            {item.label}
          </span>
        ))}
      </div>
    </div>
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function numberArray(value: unknown): number[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => Number(item)).filter((item) => Number.isFinite(item));
}

function stringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => String(item)).filter(Boolean);
}

function numberValue(value: unknown, fallback = 0) {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
}

function JsonTree({ value, name, level = 0 }: { value: unknown; name?: string; level?: number }) {
  if (Array.isArray(value)) {
    return (
      <details className="json-node" open={level < 1}>
        <summary>
          <span>{name || "array"}</span>
          <em>{value.length} items</em>
        </summary>
        <div>
          {value.map((item, index) => (
            <JsonTree key={index} level={level + 1} name={`[${index}]`} value={item} />
          ))}
        </div>
      </details>
    );
  }
  if (isRecord(value)) {
    const entries = Object.entries(value);
    return (
      <details className="json-node" open={level < 1}>
        <summary>
          <span>{name || "object"}</span>
          <em>{entries.length} keys</em>
        </summary>
        <div>
          {entries.map(([key, item]) => (
            <JsonTree key={key} level={level + 1} name={key} value={item} />
          ))}
        </div>
      </details>
    );
  }
  return (
    <div className="json-leaf">
      {name ? <span>{name}</span> : null}
      <code>{value === null ? "null" : JSON.stringify(value)}</code>
    </div>
  );
}

function JsonArtifactViewer({ artifacts }: { artifacts: RunAnalysisJson[] }) {
  if (!artifacts.length) return <div className="plot-empty">No analysis JSON artifacts yet.</div>;
  return (
    <div className="json-artifact-list">
      {artifacts.map((artifact) => (
        <details className="json-artifact" key={artifact.path}>
          <summary>
            <span>
              <Braces size={13} />
              <strong>{artifact.title}</strong>
            </span>
            <em>{artifact.note}</em>
          </summary>
          {artifact.error ? <div className="plot-empty">{artifact.error}</div> : <JsonTree name={artifact.name} value={artifact.data} />}
        </details>
      ))}
    </div>
  );
}

function FigureGallery({ figures, run }: { figures: RunAnalysisFigure[]; run: string }) {
  if (!figures.length) return <div className="plot-empty">No generated figures yet. Use Recompute analysis to create them.</div>;
  const imagePreference = ["png", "svg", "gif", "webp", "jpg", "jpeg"];
  return (
    <div className="figure-gallery">
      {figures.map((figure) => {
        const displayPath = imagePreference.map((ext) => figure.formats[ext]).find(Boolean) || "";
        return (
          <div className="figure-card" key={figure.key}>
            <div className="figure-card-head">
              <div>
                <strong>{figure.title}</strong>
                <span>{figure.note}</span>
              </div>
              <div>
                {Object.entries(figure.formats).map(([ext, path]) => (
                  <a href={api.runAssetUrl(run, path)} key={ext} rel="noreferrer" target="_blank">
                    {ext}
                  </a>
                ))}
              </div>
            </div>
            {displayPath ? (
              <img alt={figure.title} loading="lazy" src={api.runAssetUrl(run, displayPath)} />
            ) : (
              <a className="figure-pdf-only" href={api.runAssetUrl(run, figure.default_path)} rel="noreferrer" target="_blank">
                Open PDF artifact
              </a>
            )}
          </div>
        );
      })}
    </div>
  );
}

function insightFallback(insights: RunDashboardInsight[]) {
  if (!insights.length) return <div className="plot-empty">No insights.md or insights.json yet. Use Recompute analysis to run the miners.</div>;
  return (
    <div className="insight-preview-list">
      {insights.map((item) => (
        <div key={`${item.kind}-${item.title}`}>
          <strong>{item.title || item.kind}</strong>
          <span>
            {item.kind} | score {formatNumber(item.score, 2)}
          </span>
        </div>
      ))}
    </div>
  );
}

type LongRangePath = {
  title: string;
  path: string[];
  relations: string[];
  hops: number;
  distance: number;
  actionability: number;
};

function longRangePaths(artifacts: RunAnalysisJson[]): LongRangePath[] {
  const insights = artifacts.find((item) => item.name === "insights.json");
  if (!insights || !isRecord(insights.data) || !isRecord(insights.data.miners)) return [];
  const bridges = insights.data.miners.conceptual_bridge;
  if (!Array.isArray(bridges)) return [];
  return bridges
    .filter(isRecord)
    .map((item) => ({
      title: String(item.title || "Conceptual bridge"),
      path: stringArray(item.path),
      relations: stringArray(item.relations),
      hops: numberValue(item.hops),
      distance: numberValue(item.embed_distance),
      actionability: numberValue(item.actionability),
    }))
    .filter((item) => item.path.length >= 2)
    .sort((a, b) => b.hops - a.hops || b.actionability - a.actionability || b.distance - a.distance)
    .slice(0, 8);
}

function LongRangePathPanel({ artifacts }: { artifacts: RunAnalysisJson[] }) {
  const paths = longRangePaths(artifacts);
  if (!paths.length) return <div className="plot-empty">No conceptual_bridge paths yet. Run insights.py to mine long-range routes.</div>;
  return (
    <div className="long-path-list">
      {paths.map((item, index) => (
        <div className="long-path-card" key={`${item.title}-${index}`}>
          <div className="long-path-head">
            <strong>{item.title}</strong>
            <span>
              {item.hops} hops | distance {formatNumber(item.distance, 2)} | action {formatNumber(item.actionability, 2)}
            </span>
          </div>
          <div className="long-path-route" title={item.path.join(" -> ")}>
            {item.path.map((node, nodeIndex) => (
              <React.Fragment key={`${node}-${nodeIndex}`}>
                <b>{node}</b>
                {nodeIndex < item.path.length - 1 ? <em>{item.relations[nodeIndex] || "links"}</em> : null}
              </React.Fragment>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

type ScalingCurve = {
  label: string;
  x: number[];
  xHint: string;
  ideas: number[];
  spread: number[];
  reach: number[];
  recomb: number[];
};

function scalingCurves(artifact: RunAnalysisJson | undefined): ScalingCurve[] {
  if (!artifact || !isRecord(artifact.data)) return [];
  return Object.entries(artifact.data)
    .map(([label, raw]) => {
      if (!isRecord(raw)) return null;
      const tokens = numberArray(raw.x_tokens);
      const iters = numberArray(raw.iters);
      const x = tokens.length ? tokens : iters;
      return {
        label,
        x,
        xHint: tokens.length ? "tokens" : "iter",
        ideas: numberArray(raw.ideas),
        spread: numberArray(raw.spread),
        reach: numberArray(raw.reach),
        recomb: numberArray(raw.recomb),
      };
    })
    .filter((item): item is ScalingCurve => Boolean(item && item.x.length));
}

function ScalingAnalysis({ artifacts }: { artifacts: RunAnalysisJson[] }) {
  const artifact =
    artifacts.find((item) => item.name === "scaling_scaling.json") ||
    artifacts.find((item) => item.key.includes("scaling_scaling"));
  const curves = scalingCurves(artifact);
  if (!curves.length) return <div className="plot-empty">No scaling_scaling.json curves yet.</div>;
  const colors = ["#2f7d5d", "#6b84c7", "#c56f42", "#8a6fb4", "#348a9a"];
  const metrics: Array<{ key: keyof Pick<ScalingCurve, "ideas" | "spread" | "reach" | "recomb">; title: string; yHint: string }> = [
    { key: "ideas", title: "Distinct ideas", yHint: "cumulative" },
    { key: "spread", title: "Idea-space expansion", yHint: "variance" },
    { key: "reach", title: "Frontier reach", yHint: "distance" },
    { key: "recomb", title: "Surprising recombinations", yHint: "count" },
  ];
  return (
    <div className="scaling-chart-grid">
      {metrics.map((metric) => (
        <div className="scaling-chart-card" key={metric.key}>
          <strong>{metric.title}</strong>
          <LineChart
            series={curves.map((curve, index) => ({
              label: curve.label,
              color: colors[index % colors.length],
              points: curve.x.map((x, pointIndex) => ({ x, y: curve[metric.key][pointIndex] ?? 0 })),
            }))}
            xHint={curves[0]?.xHint || "iter"}
            yHint={metric.yHint}
          />
        </div>
      ))}
    </div>
  );
}

function BarChart({
  bars,
  valueKey,
  labelKey,
  color = "#3f7f6b",
}: {
  bars: Array<Record<string, number | string>>;
  valueKey: string;
  labelKey: string;
  color?: string;
}) {
  const data = bars.filter((item) => Number.isFinite(Number(item[valueKey]))).slice(0, 16);
  if (!data.length) return <div className="plot-empty">No categorical data yet.</div>;
  const max = Math.max(1, ...data.map((item) => Number(item[valueKey])));
  return (
    <div className="bar-chart">
      {data.map((item) => {
        const value = Number(item[valueKey]);
        const label = String(item[labelKey]);
        return (
          <div className="bar-row" key={label}>
            <span title={label}>{label}</span>
            <div>
              <i style={{ width: `${Math.max(2, (value / max) * 100)}%`, background: color }} />
            </div>
            <b>{formatNumber(value, value < 10 ? 2 : 0)}</b>
          </div>
        );
      })}
    </div>
  );
}

function PlotSection({
  title,
  note,
  icon,
  children,
  open = false,
}: {
  title: string;
  note: string;
  icon: React.ReactNode;
  children: React.ReactNode;
  open?: boolean;
}) {
  return (
    <details className="run-plot" open={open}>
      <summary>
        <span>
          {icon}
          <strong>{title}</strong>
        </span>
        <em>{note}</em>
      </summary>
      <div className="run-plot-body">{children}</div>
    </details>
  );
}

function growthSeries(data: RunDashboard | undefined) {
  const growth = data?.growth || [];
  if (growth.length) {
    return growth.map((row) => ({
      iter: row.iter,
      nodes: row.nodes,
      edges: row.edges,
      new_nodes: row.new_nodes,
      new_edges: 0,
      tokens: row.tokens,
      cum_tokens: row.cum_tokens,
      diversity: row.diversity,
    }));
  }
  return (data?.graph_series || []).map((row) => ({ ...row, tokens: 0, cum_tokens: 0, diversity: 0 }));
}

export function RunDashboardPanel({ run }: { run: string }) {
  const [job, setJob] = useState<RunAnalysisJobStatus | null>(null);
  const dashboard = useQuery({
    queryKey: ["run-dashboard", run],
    queryFn: () => api.runDashboard(run),
    enabled: Boolean(run),
    refetchInterval: job?.status === "running" ? 4000 : false,
  });
  const jobQuery = useQuery({
    queryKey: ["run-analysis-job", job?.id],
    queryFn: () => api.runAnalysisJob(job?.id || ""),
    enabled: Boolean(job?.id && ["running", "stopping"].includes(job.status)),
    refetchInterval: 1500,
  });

  useEffect(() => {
    if (!jobQuery.data) return;
    setJob(jobQuery.data);
    if (["done", "failed", "stopped"].includes(jobQuery.data.status)) void dashboard.refetch();
  }, [dashboard.refetch, jobQuery.data?.id, jobQuery.data?.status, jobQuery.data]);

  const data = dashboard.data;
  const series = useMemo(() => growthSeries(data), [data]);
  const graphSeries = data?.graph_series || [];
  const latest = series[series.length - 1];
  const analysisRunning = Boolean(job && ["running", "stopping"].includes(job.status));
  const analysisStatus = job?.progress;

  async function recompute() {
    if (!run || analysisRunning) return;
    const next = await api.runAnalysis({ run });
    setJob(next);
  }

  async function stop() {
    if (!job?.id) return;
    setJob(await api.stopRunAnalysis(job.id));
  }

  if (!run) {
    return (
      <div className="run-dashboard empty">
        <Network size={15} />
        <span>Load a run folder to populate growth, dynamics, novelty, and analysis panels.</span>
      </div>
    );
  }

  if (dashboard.isLoading) {
    return (
      <div className="run-dashboard empty">
        <Loader2 className="spin" size={15} />
        <span>Reading run artifacts...</span>
      </div>
    );
  }

  if (dashboard.error || !data) {
    return (
      <div className="run-dashboard empty">
        <FileText size={15} />
        <span>{dashboard.error instanceof Error ? dashboard.error.message : "Run dashboard unavailable."}</span>
      </div>
    );
  }

  const markdownArtifacts = data.analysis_markdown || [];
  const jsonArtifacts = data.analysis_json || [];
  const figureArtifacts = data.analysis_figures || [];
  const insightsMarkdown = markdownArtifacts.find((item) => item.name === "insights.md") || markdownArtifacts[0];

  return (
    <div className="run-dashboard">
      <div className="run-dashboard-head">
        <div>
          <strong>Run analysis</strong>
          <span>
            {data.path} | updated {formatRunTime(data.updated_at)} | {data.analysis_artifacts.length} analysis files
          </span>
        </div>
        <div className="run-dashboard-actions">
          <button disabled={analysisRunning} onClick={() => void dashboard.refetch()} type="button" title="Refresh dashboard from raw run files.">
            <RotateCcw size={13} />
            Refresh
          </button>
          {analysisRunning ? (
            <button onClick={() => void stop()} type="button" title="Stop the active analysis recompute job.">
              <Square size={12} />
              Stop
            </button>
          ) : (
            <button onClick={() => void recompute()} type="button" title="Run insights.py, plot_ideation.py, novelty.py, scaling.py, and dynamics.py for this run.">
              <Zap size={13} />
              Recompute analysis
            </button>
          )}
        </div>
      </div>

      {analysisStatus ? (
        <div className="analysis-progress">
          <span>
            {job?.status} | {analysisStatus.message || "analysis"} | {analysisStatus.current}/{analysisStatus.total}
          </span>
          <progress max={1} value={analysisStatus.percent || 0} />
        </div>
      ) : null}

      <div className="run-dashboard-metrics">
        <div>
          <span>Iterations</span>
          <strong>{formatNumber(latest?.iter ?? data.iters ?? 0)}</strong>
        </div>
        <div>
          <span>Nodes / edges</span>
          <strong>
            {formatNumber(latest?.nodes ?? data.nodes)} / {formatNumber(latest?.edges ?? data.edges)}
          </strong>
        </div>
        <div>
          <span>Tokens</span>
          <strong>{formatNumber(latest?.cum_tokens || 0)}</strong>
        </div>
        <div>
          <span>Snapshots</span>
          <strong>{formatNumber(data.snapshots.length || data.snapshot_count || 0)}</strong>
        </div>
      </div>

      <div className="run-plot-grid">
        <PlotSection icon={<Activity size={13} />} note="growth.csv" open title="Growth over iterations">
          <LineChart
            series={[
              { label: "nodes", color: "#2f7d5d", points: series.map((row) => ({ x: row.iter, y: row.nodes })) },
              { label: "edges", color: "#6b84c7", points: series.map((row) => ({ x: row.iter, y: row.edges })) },
            ]}
            yHint="cumulative"
          />
        </PlotSection>
        <PlotSection icon={<BarChart3 size={13} />} note="yield rate" open title="Per-iteration yield">
          <LineChart
            series={[
              { label: "new nodes", color: "#c56f42", points: series.map((row) => ({ x: row.iter, y: row.new_nodes })) },
              { label: "new edges", color: "#8a6fb4", points: graphSeries.map((row) => ({ x: row.iter, y: row.new_edges })) },
            ]}
            yHint="new per step"
          />
        </PlotSection>
        <PlotSection icon={<Zap size={13} />} note="compute trace" title="Compute and diversity">
          <LineChart
            series={[
              { label: "tokens", color: "#b8842f", points: series.map((row) => ({ x: row.iter, y: row.tokens })) },
              { label: "diversity", color: "#348a9a", points: series.map((row) => ({ x: row.iter, y: row.diversity })) },
            ]}
            yHint="raw"
          />
        </PlotSection>
        <PlotSection icon={<GitBranch size={13} />} note="GraphML provenance" title="Depth expansion">
          <BarChart bars={data.depth_series} color="#638cbd" labelKey="depth" valueKey="nodes" />
        </PlotSection>
        <PlotSection icon={<Network size={13} />} note="edge labels" title="Relation mix">
          <BarChart bars={data.relations} color="#7b9b62" labelKey="relation" valueKey="count" />
        </PlotSection>
        <PlotSection icon={<FileText size={13} />} note="insights.md" open title="Mined insights report">
          {insightsMarkdown?.markdown ? (
            <div className="run-markdown-scroll compact">
              <MarkdownReport
                assetUrl={(file) => api.runAssetUrl(data.path, file)}
                markdown={insightsMarkdown.markdown}
                out={data.path}
              />
            </div>
          ) : (
            insightFallback(data.insights)
          )}
        </PlotSection>
        <PlotSection icon={<Route size={13} />} note="conceptual_bridge" open title="Long-range paths">
          <LongRangePathPanel artifacts={jsonArtifacts} />
        </PlotSection>
        <PlotSection icon={<ImageIcon size={13} />} note="png / svg / pdf" open={Boolean(figureArtifacts.length)} title="Generated artifact figures">
          <FigureGallery figures={figureArtifacts} run={data.path} />
        </PlotSection>
        <PlotSection icon={<BarChart3 size={13} />} note="scaling_scaling.json" title="Scaling analysis">
          <ScalingAnalysis artifacts={jsonArtifacts} />
        </PlotSection>
        <PlotSection icon={<Braces size={13} />} note="expandable JSON" title="Structured analysis data">
          <JsonArtifactViewer artifacts={jsonArtifacts} />
        </PlotSection>
      </div>
    </div>
  );
}
