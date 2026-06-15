import { useQuery } from "@tanstack/react-query";
import { BarChart3, Braces, FileText, GitBranch, Image as ImageIcon, Loader2, Network, RotateCcw, Route, Square, Zap } from "lucide-react";
import React, { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { formatRunTime } from "../components/common";
import { MarkdownReport } from "./reporting";
import { formatNumber } from "../graph-utils";
import type { RunAnalysisFigure, RunAnalysisJson, RunAnalysisJobStatus, RunDashboard, RunDashboardInsight } from "../types";

type XY = { x: number; y: number };
type Series = { label: string; color: string; points: XY[] };
type GrowthRow = {
  iter: number;
  nodes: number;
  edges: number;
  new_nodes: number;
  new_edges: number;
  tokens: number;
  cum_tokens: number;
  diversity: number;
};

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
  const [preview, setPreview] = useState<{ figure: RunAnalysisFigure; path: string } | null>(null);
  const visibleFigures = figures.filter((figure) => !figure.key.includes("ideation_bars") && figure.title !== "Final metrics");
  if (!visibleFigures.length) return <div className="plot-empty">No generated figures yet. Run analysis to create them.</div>;
  const imagePreference = ["png", "svg", "gif", "webp", "jpg", "jpeg"];
  return (
    <>
      <div className="figure-gallery">
        {visibleFigures.map((figure) => {
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
                <button
                  className="figure-preview-button"
                  onClick={() => setPreview({ figure, path: displayPath })}
                  title={`Preview ${figure.title}`}
                  type="button"
                >
                  <img alt={figure.title} loading="lazy" src={api.runAssetUrl(run, displayPath)} />
                </button>
              ) : (
                <a className="figure-pdf-only" href={api.runAssetUrl(run, figure.default_path)} rel="noreferrer" target="_blank">
                  Open PDF artifact
                </a>
              )}
            </div>
          );
        })}
      </div>
      {preview ? (
        <div className="model-modal-backdrop figure-preview-backdrop" role="presentation">
          <div aria-label="Figure preview" aria-modal="true" className="figure-preview-modal" role="dialog">
            <div className="figure-preview-head">
              <div>
                <strong>{preview.figure.title}</strong>
                <span>{preview.figure.note}</span>
              </div>
              <div>
                {Object.entries(preview.figure.formats).map(([ext, path]) => (
                  <a href={api.runAssetUrl(run, path)} key={ext} rel="noreferrer" target="_blank">
                    {ext}
                  </a>
                ))}
                <button aria-label="Close figure preview" onClick={() => setPreview(null)} type="button">
                  Close
                </button>
              </div>
            </div>
            <img alt={preview.figure.title} src={api.runAssetUrl(run, preview.path)} />
          </div>
        </div>
      ) : null}
    </>
  );
}

function insightFallback(insights: RunDashboardInsight[]) {
  if (!insights.length) return <div className="plot-empty">No mined insights yet. Run analysis to mine graph structure, routes, and candidate leads.</div>;
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
  if (!paths.length) return <div className="plot-empty">No long-range routes yet. Run analysis to mine multi-step routes through the graph.</div>;
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
  wide = false,
}: {
  title: string;
  note: string;
  icon: React.ReactNode;
  children: React.ReactNode;
  open?: boolean;
  wide?: boolean;
}) {
  return (
    <details className={`run-plot${wide ? " wide" : ""}`} open={open}>
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

function growthSeries(data: RunDashboard | undefined): GrowthRow[] {
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

function GrowthHeadline({ graphSeries, series }: { graphSeries: RunDashboard["graph_series"]; series: GrowthRow[] }) {
  if (!series.length) {
    return (
      <section className="growth-headline">
        <div className="plot-empty">No growth trace yet. Start or load a run with growth.csv to populate the session trajectory.</div>
      </section>
    );
  }
  const edgeYield = new Map(graphSeries.map((row) => [row.iter, row.new_edges]));
  const enriched = series.map((row) => ({ ...row, new_edges: edgeYield.get(row.iter) ?? row.new_edges }));
  const first = enriched[0];
  const latest = enriched[enriched.length - 1];
  const peak = enriched.reduce((best, row) => (row.new_nodes + row.new_edges > best.new_nodes + best.new_edges ? row : best), enriched[0]);
  const nodeGain = Math.max(0, latest.nodes - first.nodes);
  const edgeGain = Math.max(0, latest.edges - first.edges);
  return (
    <section className="growth-headline">
      <div className="growth-headline-copy">
        <span>Run trajectory</span>
        <strong>Growth over iterations</strong>
        <p>
          {formatNumber(nodeGain)} new nodes and {formatNumber(edgeGain)} new edges across {formatNumber(enriched.length)} recorded steps.
        </p>
      </div>
      <div className="growth-headline-chart">
        <LineChart
          series={[
            { label: "nodes", color: "#2f7d5d", points: enriched.map((row) => ({ x: row.iter, y: row.nodes })) },
            { label: "edges", color: "#6b84c7", points: enriched.map((row) => ({ x: row.iter, y: row.edges })) },
            { label: "new nodes", color: "#c56f42", points: enriched.map((row) => ({ x: row.iter, y: row.new_nodes })) },
          ]}
          yHint="count"
        />
      </div>
      <div className="growth-headline-stats">
        <div>
          <span>Peak yield</span>
          <strong>{formatNumber(peak.new_nodes + peak.new_edges)}</strong>
          <em>iter {formatNumber(peak.iter)}</em>
        </div>
        <div>
          <span>Token total</span>
          <strong>{formatNumber(latest.cum_tokens || 0)}</strong>
          <em>{formatNumber(latest.tokens || 0)} last step</em>
        </div>
        <div>
          <span>Diversity</span>
          <strong>{formatNumber(latest.diversity || 0, 3)}</strong>
          <em>latest</em>
        </div>
      </div>
    </section>
  );
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

  const jsonArtifacts = data.analysis_json || [];
  const figureArtifacts = data.analysis_figures || [];
  const visibleFigureCount = figureArtifacts.filter((figure) => !figure.key.includes("ideation_bars") && figure.title !== "Final metrics").length;
  const analysisButtonLabel = data.analysis_artifacts.length ? "Recompute analysis" : "Run analysis";

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
            <button onClick={() => void recompute()} type="button" title="Recompute graph growth, mined insights, routes, novelty, scaling, and dynamics for this run.">
              <Zap size={13} />
              {analysisButtonLabel}
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

      <GrowthHeadline graphSeries={graphSeries} series={series} />

      <div className="run-plot-grid">
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
        <PlotSection icon={<FileText size={13} />} note="insight preview" open title="Top mined leads">
          {insightFallback(data.insights)}
        </PlotSection>
        <PlotSection icon={<Route size={13} />} note="route miner" open title="Long-range paths">
          <LongRangePathPanel artifacts={jsonArtifacts} />
        </PlotSection>
        <PlotSection icon={<ImageIcon size={13} />} note="figures" open={Boolean(visibleFigureCount)} title="Generated figures" wide>
          <FigureGallery figures={figureArtifacts} run={data.path} />
        </PlotSection>
        <PlotSection icon={<Braces size={13} />} note="expandable JSON" title="Structured analysis data" wide>
          <JsonArtifactViewer artifacts={jsonArtifacts} />
        </PlotSection>
      </div>
    </div>
  );
}
