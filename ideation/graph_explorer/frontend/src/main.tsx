import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import Graph from "graphology";
import Sigma from "sigma";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import {
  Activity,
  BrainCircuit,
  Command,
  Download,
  FileText,
  FolderOpen,
  Loader2,
  Network,
  PanelLeft,
  Play,
  Plus,
  RotateCcw,
  Search,
  Send,
  Settings2,
  SlidersHorizontal,
  Upload,
  X,
} from "lucide-react";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { api } from "./api";
import { cx, Drawer, HelpTip, IconButton, SidebarHeader } from "./components/common";
import { ReportStage, ReportStudio, readReportStudioStorage } from "./features/reporting";
import { RunExplorer, RunMonitor } from "./features/runs";
import {
  colorScale,
  colorPalettes,
  contextSummary,
  edgeColor,
  edgeKey,
  formatNumber,
  categoryColor as paletteCategoryColor,
  layoutNode,
  layoutNode3D,
  metricRange,
  nodeMetric,
  nodeSize,
  pathEdgeSet,
  pathNodeSet,
} from "./graph-utils";
import { useExplorerStore } from "./store";
import type {
  BridgeIdea,
  EmbeddingStatus,
  GraphAskContextNode,
  GraphNode,
  GraphPayload,
  ModelProbe,
  ModelRole,
  PathConnector,
  SearchResult,
} from "./types";
import "./styles.css";

type CoreWorkspaceMode = "chat" | "graph" | "search" | "runs" | "reports" | "models";
type OptionalToolMode = "graphrag";
type WorkspaceMode = CoreWorkspaceMode | OptionalToolMode;

const TOOL_RAIL_STORAGE_KEY = "graph-preflexor-explorer.tool-rail.v1";
const SESSION_REPORTS_STORAGE_KEY = "graph-preflexor-explorer.session-reports.v1";
const OPTIONAL_TOOL_IDS = ["graphrag"] as const;

type SessionReport = {
  out: string;
  label: string;
  seenAt: number;
};

function isOptionalTool(value: string): value is OptionalToolMode {
  return (OPTIONAL_TOOL_IDS as readonly string[]).includes(value);
}

function readToolRailStorage(): OptionalToolMode[] {
  if (typeof window === "undefined") return [];
  try {
    const stored = JSON.parse(window.localStorage.getItem(TOOL_RAIL_STORAGE_KEY) || "[]");
    if (!Array.isArray(stored)) return [];
    return stored.filter((item): item is OptionalToolMode => typeof item === "string" && isOptionalTool(item));
  } catch {
    return [];
  }
}

function writeToolRailStorage(tools: OptionalToolMode[]) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(TOOL_RAIL_STORAGE_KEY, JSON.stringify(tools));
}

function reportContextLabel(out: string) {
  return out.split("/").filter(Boolean).pop() || out || "report";
}

function readSessionReports(): SessionReport[] {
  if (typeof window === "undefined") return [];
  try {
    const stored = JSON.parse(window.sessionStorage.getItem(SESSION_REPORTS_STORAGE_KEY) || "[]");
    if (!Array.isArray(stored)) return [];
    return stored
      .filter((item): item is SessionReport => Boolean(item && typeof item.out === "string" && item.out))
      .map((item) => ({ out: item.out, label: item.label || reportContextLabel(item.out), seenAt: Number(item.seenAt) || Date.now() }));
  } catch {
    return [];
  }
}

function writeSessionReports(reports: SessionReport[]) {
  if (typeof window === "undefined") return;
  window.sessionStorage.setItem(SESSION_REPORTS_STORAGE_KEY, JSON.stringify(reports));
}

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
      refetchOnWindowFocus: false,
    },
  },
});

function graphAgentRole(roles: Record<string, ModelRole>, chatRole: string) {
  return roles.graph_qa?.model ? roles.graph_qa : roles.questioner?.model ? roles.questioner : roles[chatRole];
}

function chatModelRole(roles: Record<string, ModelRole>) {
  return roles.chat?.model ? roles.chat : roles.questioner?.model ? roles.questioner : roles.graph_qa;
}

function contextNodesToSearchResults(nodes: GraphAskContextNode[]): SearchResult[] {
  return nodes.map((node) => ({
    id: node.id,
    label: node.label,
    degree: node.degree,
    pagerank: node.pagerank,
    core: node.core,
    iter: node.iter || 0,
    score: node.score || 0,
  }));
}

function nodeLabel(graph: GraphPayload | null, id: string) {
  return graph?.nodes.find((node) => node.id === id)?.label || id;
}

function nodeLookupOptions(graph: GraphPayload | null, query: string, selectedNodes: string[] = []) {
  if (!graph) return [];
  const q = query.trim().toLowerCase();
  const selected = selectedNodes
    .map((id) => graph.nodes.find((node) => node.id === id))
    .filter(Boolean) as GraphNode[];
  const matches = q
    ? graph.nodes
        .filter((node) => {
          const label = node.label.toLowerCase();
          const id = node.id.toLowerCase();
          return label === q || id === q || label.includes(q) || id.includes(q);
        })
        .sort((a, b) => {
          const aq = a.label.toLowerCase() === q || a.id.toLowerCase() === q ? 1 : 0;
          const bq = b.label.toLowerCase() === q || b.id.toLowerCase() === q ? 1 : 0;
          return bq - aq || b.degree - a.degree || b.pagerank - a.pagerank;
        })
    : selected;
  const seen = new Set<string>();
  return [...selected, ...matches]
    .filter((node) => {
      if (seen.has(node.id)) return false;
      seen.add(node.id);
      return true;
    })
    .slice(0, 5);
}

function parseGeneratedPrompts(answer: string) {
  return answer
    .split(/\r?\n/)
    .map((line) => line.replace(/^\s*(?:[-*]|\d+[.)])\s*/, "").trim())
    .filter((line) => line.length > 8)
    .slice(0, 4);
}

function isLocalModelProbeIssue(probe: ModelProbe) {
  return Boolean(
    probe.local &&
      ["connection", "model_missing", "completion_error", "status_error"].includes(probe.category),
  );
}

function modelProbeStatus(probe: ModelProbe) {
  if (probe.ok) {
    return `Probe OK: ${probe.message}${probe.sample ? ` Sample: ${probe.sample}` : ""}`;
  }
  if (probe.category === "api_key") {
    return `API key needed: ${probe.message}`;
  }
  if (probe.category === "model_missing") {
    return `Model not served: ${probe.message}`;
  }
  if (probe.category === "connection") {
    return `Server unavailable: ${probe.message}`;
  }
  return `Probe failed at ${probe.stage}: ${probe.message}`;
}

function localServePort(baseUrl?: string) {
  try {
    const url = new URL(baseUrl || "http://localhost:1234/v1");
    return url.port || (url.protocol === "https:" ? "443" : "80");
  } catch {
    return "1234";
  }
}

function serveCommands(probe: ModelProbe) {
  const model = probe.model || "<model-id>";
  const port = localServePort(probe.base_url);
  const mistralNote =
    port === "1234"
      ? "# models.toml already serves the default local graph/questioner models on port 1234"
      : `# set [server].port = ${port} in models.toml, then add this model if it is missing`;
  return {
    mistral: `cd ideation\n${mistralNote}\nmistralrs from-config -f models.toml`,
    vllm: `vllm serve ${model} --host 127.0.0.1 --port ${port} --served-model-name ${model}`,
  };
}

function CopyCommandButton({ text }: { text: string }) {
  return (
    <button
      onClick={() => {
        if (navigator.clipboard) void navigator.clipboard.writeText(text);
      }}
      type="button"
    >
      Copy
    </button>
  );
}

function ServeModelModal({ probe, onClose }: { probe: ModelProbe; onClose: () => void }) {
  const commands = serveCommands(probe);
  return (
    <div className="model-modal-backdrop" role="presentation">
      <section aria-modal="true" className="serve-modal" role="dialog">
        <div className="serve-modal-head">
          <div>
            <span className="eyebrow">Local model check</span>
            <h3>Model is not available</h3>
          </div>
          <button aria-label="Close model help" onClick={onClose} type="button">
            <X size={16} />
          </button>
        </div>
        <div className="serve-meta">
          <span>Model: {probe.model || "not set"}</span>
          <span>Endpoint: {probe.base_url || "not set"}</span>
          <span>Failure: {probe.message}</span>
        </div>
        <p className="serve-note">
          Start a local OpenAI-compatible server, then run Test again. If this is a new model, add it to
          ideation/models.toml for the mistral.rs path.
        </p>
        <div className="command-card">
          <div>
            <strong>mistral.rs</strong>
            <CopyCommandButton text={commands.mistral} />
          </div>
          <pre>{commands.mistral}</pre>
        </div>
        <div className="command-card">
          <div>
            <strong>vLLM</strong>
            <CopyCommandButton text={commands.vllm} />
          </div>
          <pre>{commands.vllm}</pre>
        </div>
      </section>
    </div>
  );
}

const MODEL_PRESETS: Array<{ label: string; values: Partial<ModelRole> }> = [
  {
    label: "Local Gemma chat/questioner",
    values: {
      provider: "openai",
      model: "google/gemma-4-E4B",
      base_url: "http://localhost:1234/v1",
      api_key_env: "",
      temperature: 0.3,
      max_tokens: 1800,
      reasoning_effort: "",
    },
  },
  {
    label: "OpenAI gpt-5.5",
    values: {
      provider: "openai",
      model: "gpt-5.5",
      base_url: "https://api.openai.com/v1",
      api_key_env: "OPENAI_API_KEY",
      temperature: 0.3,
      max_tokens: 4000,
      reasoning_effort: "medium",
    },
  },
  {
    label: "OpenAI gpt-5.5-mini",
    values: {
      provider: "openai",
      model: "gpt-5.5-mini",
      base_url: "https://api.openai.com/v1",
      api_key_env: "OPENAI_API_KEY",
      temperature: 0.3,
      max_tokens: 3000,
      reasoning_effort: "medium",
    },
  },
  {
    label: "OpenAI gpt-5.5-nano",
    values: {
      provider: "openai",
      model: "gpt-5.5-nano",
      base_url: "https://api.openai.com/v1",
      api_key_env: "OPENAI_API_KEY",
      temperature: 0.3,
      max_tokens: 1800,
      reasoning_effort: "low",
    },
  },
  {
    label: "OpenAI gpt-5.5 high reasoning",
    values: {
      provider: "openai",
      model: "gpt-5.5",
      base_url: "https://api.openai.com/v1",
      api_key_env: "OPENAI_API_KEY",
      temperature: 0.2,
      max_tokens: 6000,
      reasoning_effort: "high",
    },
  },
  {
    label: "Local Llama on 8000",
    values: {
      provider: "openai",
      model: "meta-llama/Llama-3.2-3B-Instruct",
      base_url: "http://localhost:8000/v1",
      api_key_env: "",
      temperature: 0.3,
      max_tokens: 1800,
      reasoning_effort: "",
    },
  },
  {
    label: "Graph-PRefLexOR generator",
    values: {
      provider: "openai",
      model: "lamm-mit/Graph-Preflexor-3b_08012026",
      base_url: "http://localhost:1234/v1",
      api_key_env: "",
      temperature: 0.7,
      max_tokens: 1800,
      reasoning_effort: "",
    },
  },
];

function Header({
  onLoadRun,
  onGraphLoaded,
}: {
  onLoadRun: (run: string) => Promise<void>;
  onGraphLoaded: (graph: GraphPayload) => void;
}) {
  const graph = useExplorerStore((state) => state.graph);
  const [run, setRun] = useState("");
  const [busy, setBusy] = useState(false);

  async function upload(file: File | undefined) {
    if (!file) return;
    setBusy(true);
    try {
      const text = await file.text();
      onGraphLoaded(await api.uploadGraphml(file.name, text));
    } finally {
      setBusy(false);
    }
  }

  return (
    <header className="topbar">
      <div className="brand">
        <div className="brand-mark">GP</div>
        <div className="brand-copy">
          <strong>Graph-PRefLexOR Explorer</strong>
          <span>
            {graph
              ? `${graph.name || "graph"} | ${formatNumber(graph.stats.nodes)} nodes, ${formatNumber(graph.stats.edges)} edges`
              : "No graph loaded"}
          </span>
        </div>
      </div>
      <div className="top-actions">
        <label className="file-button" title="Upload a single GraphML/XML file from your machine. To load an existing run folder, use the path field or Run Explorer.">
          <Upload size={14} />
          Upload GraphML
          <input accept=".graphml,.xml" onChange={(event) => upload(event.target.files?.[0])} type="file" />
        </label>
        <input
          className="path-input"
          onChange={(event) => setRun(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") void onLoadRun(run);
          }}
          placeholder="run folder, e.g. runs/exp_leap, or /path/to/file.graphml"
          title="Enter a previous run folder or a GraphML file path, then press Enter or Load."
          value={run}
        />
        <IconButton
          disabled={busy || !run.trim()}
          icon={busy ? <Loader2 className="spin" size={14} /> : <FolderOpen size={14} />}
          label="Load"
          onClick={() => onLoadRun(run)}
        />
      </div>
    </header>
  );
}

function Overview() {
  const graph = useExplorerStore((state) => state.graph);
  const stats = graph?.stats;
  const items = [
    ["Nodes", stats?.nodes],
    ["Edges", stats?.edges],
    ["Components", stats?.components],
    ["Communities", stats?.communities],
    ["Largest", stats?.largest_component],
    ["Avg degree", stats?.avg_degree, 2],
    ["Max degree", stats?.max_degree],
    ["Max iter", stats?.max_iter],
    ["Density", stats?.density, 5],
  ] as const;
  return (
    <section className="panel-card">
      <div className="section-head">
        <h2>Graph Overview</h2>
        <span>{graph ? graph.topic || graph.path || "Loaded" : "Idle"}</span>
      </div>
      <div className="stats-grid">
        {items.map(([label, value, digits]) => (
          <div className="stat" key={label}>
            <b>{formatNumber(value ?? 0, digits ?? 0)}</b>
            <span>{label}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

function VisualControls({ defaultOpen = false }: { defaultOpen?: boolean }) {
  const graph = useExplorerStore((state) => state.graph);
  const visual = useExplorerStore((state) => state.visual);
  const setVisual = useExplorerStore((state) => state.setVisual);
  const colorRange = graph ? metricRange(graph.nodes, visual.colorBy) : { min: 0, max: 1 };
  const paletteInfo = colorPalettes[visual.colorPalette] || colorPalettes.atlas;
  const categorical = visual.colorBy === "component" || visual.colorBy === "community";
  const colorLabel = visual.colorBy === "pagerank" ? "PageRank" : visual.colorBy[0].toUpperCase() + visual.colorBy.slice(1);
  return (
    <Drawer
      defaultOpen={defaultOpen}
      description="Control how the loaded graph is drawn: renderer, layout, node color metric, node size metric, and edge opacity. These controls change the current visual encoding, not the graph data."
      icon={<SlidersHorizontal size={14} />}
      note="layout, color, size"
      title="Visual Mapping"
    >
      <div className="control-grid">
        <label>
          Renderer
          <select
            value={visual.viewMode}
            onChange={(event) => setVisual({ viewMode: event.target.value as typeof visual.viewMode })}
          >
            <option value="3d">Atlas 3D</option>
            <option value="2d">2D map</option>
          </select>
        </label>
        <label>
          Layout
          <select
            value={visual.layout}
            onChange={(event) => setVisual({ layout: event.target.value as typeof visual.layout })}
          >
            <option value="force">Force</option>
            <option value="component">Component</option>
            <option value="community">Community</option>
            <option value="degree">Degree radial</option>
            <option value="timeline">Timeline</option>
          </select>
        </label>
        <label>
          Color
          <select
            value={visual.colorBy}
            onChange={(event) => setVisual({ colorBy: event.target.value as typeof visual.colorBy })}
          >
            <option value="degree">Degree</option>
            <option value="component">Component</option>
            <option value="community">Community</option>
            <option value="pagerank">PageRank</option>
            <option value="core">Core</option>
            <option value="iter">Iteration</option>
            <option value="depth">Depth</option>
          </select>
        </label>
        <label>
          Palette
          <select
            value={visual.colorPalette}
            onChange={(event) => setVisual({ colorPalette: event.target.value as typeof visual.colorPalette })}
          >
            {Object.entries(colorPalettes).map(([id, option]) => (
              <option key={id} value={id}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          Size
          <select
            value={visual.sizeBy}
            onChange={(event) => setVisual({ sizeBy: event.target.value as typeof visual.sizeBy })}
          >
            <option value="degree">Degree</option>
            <option value="pagerank">PageRank</option>
            <option value="core">Core</option>
            <option value="constant">Constant</option>
          </select>
        </label>
        <label>
          Edge opacity
          <input
            max="0.8"
            min="0.02"
            onChange={(event) => setVisual({ edgeOpacity: Number(event.target.value) })}
            step="0.01"
            type="range"
            value={visual.edgeOpacity}
          />
        </label>
      </div>
      <div className="visual-legend">
        <div>
          <strong>{colorLabel}</strong>
          <span>
            {categorical
              ? `${paletteInfo.label}; colors repeat across ${visual.colorBy} groups`
              : `${formatNumber(colorRange.min, 3)} to ${formatNumber(colorRange.max, 3)}`}
          </span>
        </div>
        {categorical ? (
          <div className="legend-swatches" aria-label={`${visual.colorBy} category colors`}>
            {paletteInfo.colors.slice(0, 10).map((color, index) => (
              <i key={`${color}-${index}`} style={{ background: color }} title={`${visual.colorBy} group color`} />
            ))}
          </div>
        ) : (
          <div
            className="legend-gradient"
            style={{ background: `linear-gradient(90deg, ${paletteInfo.colors.join(", ")})` }}
            aria-label={`${paletteInfo.label} low to high gradient`}
          />
        )}
      </div>
    </Drawer>
  );
}

function SearchPanel({ defaultOpen = false }: { defaultOpen?: boolean }) {
  const results = useExplorerStore((state) => state.searchResults);
  const setSearchResults = useExplorerStore((state) => state.setSearchResults);
  const setSelectedNode = useExplorerStore((state) => state.setSelectedNode);
  const setHighlightedPaths = useExplorerStore((state) => state.setHighlightedPaths);
  const selectedNodes = useExplorerStore((state) => state.selectedNodes);
  const graph = useExplorerStore((state) => state.graph);
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("Click a result to select it in the viewer. Shift-click graph nodes to build a multi-node selection.");

  async function runSearch() {
    if (!query.trim()) return;
    setBusy(true);
    try {
      const next = (await api.search(query)).results;
      setSearchResults(next);
      setStatus(next.length ? `${next.length} close graph matches. Click one to select; use Focus Tools to route between selections.` : "No close matches found.");
    } finally {
      setBusy(false);
    }
  }

  function showHubs() {
    if (!graph) return;
    const hubs: SearchResult[] = [...graph.nodes]
      .sort((a, b) => b.degree - a.degree)
      .slice(0, 30)
      .map((node, index) => ({
        id: node.id,
        label: `${index + 1}. ${node.label}`,
        degree: node.degree,
        pagerank: node.pagerank,
        iter: node.iter,
        score: node.pagerank,
      }));
    setSearchResults(hubs);
    setStatus("Showing highest-degree hubs. Click a hub to select it in the viewer.");
  }

  return (
    <Drawer
      defaultOpen={defaultOpen}
      description="Search node ids, labels, and attributes. Clicking a result selects it in the graph; hold Shift while clicking graph nodes to add more selections."
      icon={<Search size={14} />}
      note={`${results.length} results`}
      title="Search & Select"
    >
      <div className="row">
        <input
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") void runSearch();
          }}
          placeholder="concept, mechanism, material"
          value={query}
        />
        <IconButton
          disabled={busy}
          description="Find close node matches by id, label, and recorded graph attributes."
          icon={busy ? <Loader2 className="spin" size={14} /> : <Search size={14} />}
          label="Search"
          onClick={runSearch}
        />
      </div>
      <div className="button-row">
        <IconButton description="List the most connected nodes by degree." icon={<Network size={14} />} label="Top Hubs" onClick={showHubs} />
        <IconButton
          description="Clear search results and the temporary visual highlight."
          icon={<X size={14} />}
          label="Clear"
          onClick={() => {
            setSearchResults([]);
            setHighlightedPaths([]);
            setStatus("Search results cleared.");
          }}
        />
      </div>
      {status ? <div className="micro-help">{status}</div> : null}
      <div className="result-list">
        {results.map((result) => (
          <button
            className={cx("result-item", selectedNodes.includes(result.id) && "active")}
            key={result.id}
            onClick={(event) => {
              setSelectedNode(result.id, event.shiftKey);
              setHighlightedPaths([[result.id]]);
              setStatus(`Selected ${result.label}. Open Focus Tools to use it as a path source or target.`);
            }}
            title="Select this node in the center graph. Shift-click graph nodes to build a path anchor set."
            type="button"
          >
            <strong>{result.label}</strong>
            <span>
              degree {formatNumber(result.degree)} | iter {formatNumber(result.iter)} | score {formatNumber(result.score, 2)}
              {result.semantic_score !== undefined ? ` | semantic ${formatNumber(result.semantic_score, 2)}` : ""}
            </span>
          </button>
        ))}
      </div>
    </Drawer>
  );
}

function SigmaGraphCanvas() {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const rendererRef = useRef<Sigma | null>(null);
  const graph = useExplorerStore((state) => state.graph);
  const visual = useExplorerStore((state) => state.visual);
  const selectedNodes = useExplorerStore((state) => state.selectedNodes);
  const highlightedPaths = useExplorerStore((state) => state.highlightedPaths);
  const setSelectedNode = useExplorerStore((state) => state.setSelectedNode);
  const [hoverNode, setHoverNode] = useState<GraphNode | null>(null);

  useEffect(() => {
    if (!containerRef.current || !graph) return undefined;
    rendererRef.current?.kill();

    const sigmaGraph = new Graph();
    const colorRange = metricRange(graph.nodes, visual.colorBy);
    const selected = new Set(selectedNodes);
    const highlightedNodes = pathNodeSet(highlightedPaths);
    const highlightedEdges = pathEdgeSet(highlightedPaths);

    for (const node of graph.nodes) {
      const pos = layoutNode(node, graph, visual);
      const isHighlighted = highlightedNodes.has(node.id);
      const nodeColor =
        visual.colorBy === "component" || visual.colorBy === "community"
          ? paletteCategoryColor(nodeMetric(node, visual.colorBy), visual)
          : colorScale(nodeMetric(node, visual.colorBy), colorRange, visual.colorPalette);
      sigmaGraph.addNode(node.id, {
        ...pos,
        label: node.label,
        size: selected.has(node.id) || isHighlighted ? nodeSize(node, graph, visual) * 1.9 : nodeSize(node, graph, visual),
        color: selected.has(node.id) ? "#ffffff" : isHighlighted ? "#ffd166" : nodeColor,
        borderColor: selected.has(node.id) ? "#37d49a" : isHighlighted ? "#ffef9f" : nodeColor,
      });
    }

    const maxEdges = graph.edges.length > 45000 ? 45000 : graph.edges.length;
    for (let i = 0; i < maxEdges; i++) {
      const edge = graph.edges[i];
      if (sigmaGraph.hasNode(edge.source) && sigmaGraph.hasNode(edge.target) && !sigmaGraph.hasEdge(edge.id)) {
        const isHighlighted = highlightedEdges.has(edgeKey(edge.source, edge.target));
        sigmaGraph.addEdgeWithKey(edge.id, edge.source, edge.target, {
          size: isHighlighted ? 2.4 : graph.edges.length > 12000 ? 0.35 : 0.55,
          color: isHighlighted ? "rgba(255,209,102,0.92)" : edgeColor(edge, visual.edgeOpacity),
        });
      }
    }

    const renderer = new Sigma(sigmaGraph, containerRef.current, {
      renderLabels: false,
      renderEdgeLabels: false,
      defaultEdgeColor: "rgba(94,112,132,0.12)",
      allowInvalidContainer: true,
      labelRenderedSizeThreshold: 12,
      minCameraRatio: 0.05,
      maxCameraRatio: 12,
    });
    renderer.on("clickNode", ({ node }) => setSelectedNode(node));
    renderer.on("enterNode", ({ node }) => setHoverNode(graph.nodes.find((item) => item.id === node) || null));
    renderer.on("leaveNode", () => setHoverNode(null));
    rendererRef.current = renderer;

    return () => renderer.kill();
  }, [graph, highlightedPaths, selectedNodes, setSelectedNode, visual]);

  const selectedLabel = useMemo(() => {
    if (!graph || !selectedNodes.length) return "No selection";
    const names = selectedNodes.slice(0, 2).map((id) => graph.nodes.find((node) => node.id === id)?.label || id);
    return `Selected: ${names.join(", ")}${selectedNodes.length > 2 ? ` +${selectedNodes.length - 2}` : ""}`;
  }, [graph, selectedNodes]);

  return (
    <section className="graph-shell">
      <div className="graph-overlay top-left">{contextSummary(graph, selectedNodes)}</div>
      <div className="graph-overlay top-right">{selectedLabel}</div>
      <div className="graph-canvas" ref={containerRef} />
      {hoverNode ? (
        <div className="node-card">
          <strong>{hoverNode.label}</strong>
          <span>
            degree {formatNumber(hoverNode.degree)} | PageRank {formatNumber(hoverNode.pagerank, 4)}
          </span>
          <span>
            component {hoverNode.component} | community {hoverNode.community}
          </span>
          <span>iter {hoverNode.iter} | depth {hoverNode.depth}</span>
        </div>
      ) : null}
    </section>
  );
}

function ThreeGraphCanvas() {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const graph = useExplorerStore((state) => state.graph);
  const visual = useExplorerStore((state) => state.visual);
  const selectedNodes = useExplorerStore((state) => state.selectedNodes);
  const highlightedPaths = useExplorerStore((state) => state.highlightedPaths);
  const setSelectedNode = useExplorerStore((state) => state.setSelectedNode);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || !graph) return undefined;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#f8f7f2");
    scene.fog = new THREE.FogExp2("#f8f7f2", 0.0019);

    const width = Math.max(320, container.clientWidth || 900);
    const height = Math.max(320, container.clientHeight || 640);
    const camera = new THREE.PerspectiveCamera(48, width / height, 0.1, 2200);
    camera.position.set(0, -18, 230);

    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance", preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.autoRotate = highlightedPaths.length === 0;
    controls.autoRotateSpeed = 0.35;

    const selected = new Set(selectedNodes);
    const highlightedNodes = pathNodeSet(highlightedPaths);
    const highlightedEdges = pathEdgeSet(highlightedPaths);
    const colorRange = metricRange(graph.nodes, visual.colorBy);
    const raw = graph.nodes.map((node) => [node, layoutNode3D(node, graph, visual)] as const);
    const maxRadius = Math.max(
      1,
      ...raw.map(([, p]) => Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z)),
    );
    const scale = 172 / maxRadius;
    const coords = new Map<string, THREE.Vector3>();

    const positions: number[] = [];
    const colors: number[] = [];
    const sizes: number[] = [];
    const nodeIds: string[] = [];
    for (const [node, pos] of raw) {
      const point = new THREE.Vector3(pos.x * scale, pos.y * scale, pos.z * scale);
      coords.set(node.id, point);
      nodeIds.push(node.id);
      positions.push(point.x, point.y, point.z);
      const nodeColor =
        visual.colorBy === "component" || visual.colorBy === "community"
          ? paletteCategoryColor(nodeMetric(node, visual.colorBy), visual)
          : colorScale(nodeMetric(node, visual.colorBy), colorRange, visual.colorPalette);
      const color = new THREE.Color(
        selected.has(node.id) ? "#163c3a" : highlightedNodes.has(node.id) ? "#b56b16" : nodeColor,
      );
      colors.push(color.r, color.g, color.b);
      sizes.push((selected.has(node.id) || highlightedNodes.has(node.id) ? 13.5 : 6.2 + Math.sqrt(node.degree || 1) * 0.95));
    }

    const graphGroup = new THREE.Group();
    scene.add(graphGroup);

    const nodeGeometry = new THREE.BufferGeometry();
    nodeGeometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    nodeGeometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    nodeGeometry.setAttribute("pointSize", new THREE.Float32BufferAttribute(sizes, 1));
    const nodeMaterial = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.NormalBlending,
      vertexColors: true,
      vertexShader: `
        attribute float pointSize;
        varying vec3 vColor;
        void main() {
          vColor = color;
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = pointSize * (360.0 / max(80.0, -mvPosition.z));
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        varying vec3 vColor;
        void main() {
          vec2 delta = gl_PointCoord - vec2(0.5);
          float dist = dot(delta, delta);
          if (dist > 0.25) discard;
          float alpha = smoothstep(0.25, 0.12, dist) * 0.92;
          gl_FragColor = vec4(vColor, alpha);
        }
      `,
    });
    const pointCloud = new THREE.Points(nodeGeometry, nodeMaterial);
    graphGroup.add(pointCloud);

    const edgePositions: number[] = [];
    const edgeColors: number[] = [];
    const maxEdges = Math.min(graph.edges.length, 42000);
    for (let i = 0; i < maxEdges; i++) {
      const edge = graph.edges[i];
      const a = coords.get(edge.source);
      const b = coords.get(edge.target);
      if (!a || !b || highlightedEdges.has(edgeKey(edge.source, edge.target))) continue;
      edgePositions.push(a.x, a.y, a.z, b.x, b.y, b.z);
      const color = new THREE.Color(edgeColor(edge, 1));
      edgeColors.push(color.r, color.g, color.b, color.r, color.g, color.b);
    }
    const edgeGeometry = new THREE.BufferGeometry();
    edgeGeometry.setAttribute("position", new THREE.Float32BufferAttribute(edgePositions, 3));
    edgeGeometry.setAttribute("color", new THREE.Float32BufferAttribute(edgeColors, 3));
    graphGroup.add(
      new THREE.LineSegments(
        edgeGeometry,
        new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: Math.max(0.18, visual.edgeOpacity * 1.35) }),
      ),
    );

    const pathMaterial = new THREE.MeshBasicMaterial({ color: "#c77b1f", transparent: true, opacity: 0.9 });
    for (const path of highlightedPaths.slice(0, 18)) {
      const points = path.map((id) => coords.get(id)).filter(Boolean) as THREE.Vector3[];
      if (points.length < 2) continue;
      const curve = new THREE.CatmullRomCurve3(points);
      graphGroup.add(new THREE.Mesh(new THREE.TubeGeometry(curve, Math.max(8, points.length * 8), 0.42, 8, false), pathMaterial));
    }

    const sphereGeometry = new THREE.SphereGeometry(1.8, 16, 12);
    const anchorMaterial = new THREE.MeshBasicMaterial({ color: "#163c3a" });
    const bridgeMaterial = new THREE.MeshBasicMaterial({ color: "#c77b1f" });
    for (const nodeId of new Set([...selectedNodes, ...Array.from(highlightedNodes)])) {
      const point = coords.get(nodeId);
      if (!point) continue;
      const sphere = new THREE.Mesh(sphereGeometry, selected.has(nodeId) ? anchorMaterial : bridgeMaterial);
      sphere.position.copy(point);
      sphere.scale.setScalar(selected.has(nodeId) ? 1.55 : 1.15);
      graphGroup.add(sphere);
    }

    const resize = () => {
      const nextWidth = Math.max(320, container.clientWidth || width);
      const nextHeight = Math.max(320, container.clientHeight || height);
      camera.aspect = nextWidth / nextHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(nextWidth, nextHeight);
    };
    const observer = new ResizeObserver(resize);
    observer.observe(container);

    const raycaster = new THREE.Raycaster();
    raycaster.params.Points = { threshold: 3.5 };
    const pointer = new THREE.Vector2();
    const onPointerDown = (event: PointerEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObject(pointCloud, false)[0];
      if (hit?.index !== undefined && nodeIds[hit.index]) {
        setSelectedNode(nodeIds[hit.index], event.shiftKey);
      }
    };
    renderer.domElement.addEventListener("pointerdown", onPointerDown);

    let frame = 0;
    let stopped = false;
    const animate = () => {
      if (stopped) return;
      frame = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      stopped = true;
      cancelAnimationFrame(frame);
      observer.disconnect();
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      controls.dispose();
      renderer.dispose();
      nodeGeometry.dispose();
      nodeMaterial.dispose();
      edgeGeometry.dispose();
      sphereGeometry.dispose();
      pathMaterial.dispose();
      anchorMaterial.dispose();
      bridgeMaterial.dispose();
      if (renderer.domElement.parentElement === container) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [graph, highlightedPaths, selectedNodes, setSelectedNode, visual]);

  const selectedLabel = useMemo(() => {
    if (!graph || !selectedNodes.length) return "Atlas 3D";
    const names = selectedNodes.slice(0, 2).map((id) => graph.nodes.find((node) => node.id === id)?.label || id);
    return `Selected: ${names.join(", ")}${selectedNodes.length > 2 ? ` +${selectedNodes.length - 2}` : ""}`;
  }, [graph, selectedNodes]);

  return (
    <section className="graph-shell">
      <div className="graph-overlay top-left">
        {contextSummary(graph, selectedNodes)}
        {highlightedPaths.length ? ` | ${highlightedPaths.length} paths highlighted` : ""}
      </div>
      <div className="graph-overlay top-right">{selectedLabel}</div>
      <div className="graph-canvas" ref={containerRef} />
    </section>
  );
}

function GraphCanvas() {
  const viewMode = useExplorerStore((state) => state.visual.viewMode);
  return viewMode === "3d" ? <ThreeGraphCanvas /> : <SigmaGraphCanvas />;
}

function EmbeddingStatusBar({
  status,
  onRebuild,
}: {
  status: EmbeddingStatus | null;
  onRebuild: () => void;
}) {
  if (!status || status.status === "idle") return null;
  const percent = status.status === "done" ? 100 : Math.round((status.progress?.percent || 0) * 100);
  const label = status.ready
    ? `${status.cached ? "Cached " : ""}Semantic index ready: ${formatNumber(status.nodes)} nodes`
    : status.status === "failed"
      ? "Semantic index failed"
      : status.progress?.message || "Building semantic index";
  return (
    <div className={cx("embedding-status", status.status === "failed" && "failed", status.ready && "ready")}>
      <div className="embedding-copy">
        <strong>{label}</strong>
        <span>
          {status.model || "embedding model"}
          {status.dimension ? ` | ${formatNumber(status.dimension)} dims` : ""}
          {status.progress?.detail ? ` | ${status.progress.detail}` : ""}
        </span>
      </div>
      <div className="embedding-meter">
        <span>{percent}%</span>
        <progress max={100} value={percent} />
      </div>
      <button onClick={onRebuild} title="Force rebuild the semantic node embedding index." type="button">
        Rebuild
      </button>
    </div>
  );
}

function ChatPanel({ sessionReports }: { sessionReports: SessionReport[] }) {
  const graph = useExplorerStore((state) => state.graph);
  const selectedNodes = useExplorerStore((state) => state.selectedNodes);
  const setSelectedNode = useExplorerStore((state) => state.setSelectedNode);
  const setSearchResults = useExplorerStore((state) => state.setSearchResults);
  const messages = useExplorerStore((state) => state.chatMessages);
  const addChatMessage = useExplorerStore((state) => state.addChatMessage);
  const updateChatMessage = useExplorerStore((state) => state.updateChatMessage);
  const setChatMessages = useExplorerStore((state) => state.setChatMessages);
  const resetChat = useExplorerStore((state) => state.resetChat);
  const roles = useExplorerStore((state) => state.roles);
  const [question, setQuestion] = useState("");
  const [agentMode, setAgentMode] = useState<"focused" | "graph_rag">("graph_rag");
  const [contextQuery, setContextQuery] = useState("");
  const [contextNodes, setContextNodes] = useState(220);
  const [selectedReportOut, setSelectedReportOut] = useState("");
  const [reportMenuOpen, setReportMenuOpen] = useState(false);
  const [reportMaxChars, setReportMaxChars] = useState(14000);
  const [lastRagNodes, setLastRagNodes] = useState<GraphAskContextNode[]>([]);
  const [followups, setFollowups] = useState<string[]>([
    "What are the strongest bridge concepts here?",
    "Which gaps look most experimentally useful?",
    "What should I inspect next in this graph?",
    "Find a surprising path between selected nodes.",
  ]);
  const [busy, setBusy] = useState(false);
  const [ideaBusy, setIdeaBusy] = useState(false);
  const chatEndRef = useRef<HTMLDivElement | null>(null);
  const chatHydratedRef = useRef(false);
  const contextLabel = contextSummary(graph, selectedNodes);
  const activeChatRole = chatModelRole(roles);
  const chatRoleName = activeChatRole?.role || "chat";
  const selectedReport = sessionReports.find((report) => report.out === selectedReportOut);
  const commandQuery = question.startsWith("/") ? question.slice(1).trim().toLowerCase() : "";
  const commandItems = [
    { command: "/clear", label: "Clear chat", detail: "Reset this chat thread.", action: () => resetChat() },
    { command: "/followups", label: "Generate follow-ups", detail: "Ask the active graph model for next query ideas.", action: () => void suggestFollowups() },
    { command: "/rag", label: "Graph-RAG mode", detail: "Retrieve broader semantic neighborhoods and path connectors.", action: () => setAgentMode("graph_rag") },
    { command: "/focus", label: "Focused mode", detail: "Use only selected nodes, focus query, and compact neighborhoods.", action: () => setAgentMode("focused") },
    { command: "/nodes 160", label: "Context nodes", detail: "Set the graph context size to 160 nodes.", action: () => setContextNodes(160) },
    { command: "/help", label: "Show help", detail: "Insert a compact command reference.", action: () => addCommandHelp() },
  ].filter((item) => !commandQuery || item.command.toLowerCase().includes(commandQuery) || item.label.toLowerCase().includes(commandQuery));

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ block: "end" });
  }, [messages]);

  useEffect(() => {
    if (chatHydratedRef.current || typeof window === "undefined") return;
    chatHydratedRef.current = true;
    try {
      const saved = JSON.parse(window.localStorage.getItem("graph-preflexor-explorer.chat.v1") || "[]");
      if (Array.isArray(saved) && saved.length && !messages.length) setChatMessages(saved.slice(-80));
    } catch {
      // Ignore corrupt local storage and start a new chat.
    }
  }, [messages.length, setChatMessages]);

  useEffect(() => {
    if (!chatHydratedRef.current || typeof window === "undefined") return;
    window.localStorage.setItem("graph-preflexor-explorer.chat.v1", JSON.stringify(messages.slice(-80)));
  }, [messages]);

  useEffect(() => {
    if (selectedReportOut && !sessionReports.some((report) => report.out === selectedReportOut)) {
      setSelectedReportOut("");
    }
  }, [selectedReportOut, sessionReports]);

  function recentHistory() {
    return messages
      .filter((message) => message.role === "user" || message.role === "assistant")
      .slice(-8)
      .map((message) => ({ role: message.role as "user" | "assistant", content: message.content }));
  }

  function addCommandHelp() {
    addChatMessage({
      role: "system",
      meta: "commands",
      content:
        "/clear resets the chat.\n/rag switches to graph-RAG retrieval.\n/focus switches to selected/focused context.\n/nodes 160 changes context size.\n/followups asks the chat model for next query ideas.\nUse Model Settings to change the chat model role.",
    });
  }

  function executeCommand(raw: string) {
    const value = raw.trim();
    if (!value.startsWith("/")) return false;
    const [command, ...rest] = value.slice(1).split(/\s+/);
    if (command === "clear") {
      resetChat();
      setQuestion("");
      return true;
    }
    if (command === "followups") {
      setQuestion("");
      void suggestFollowups();
      return true;
    }
    if (command === "rag") {
      setAgentMode("graph_rag");
      addChatMessage({ role: "system", content: "Switched to Graph-RAG agent mode.", meta: "command" });
      setQuestion("");
      return true;
    }
    if (command === "focus") {
      setAgentMode("focused");
      addChatMessage({ role: "system", content: "Switched to focused context mode.", meta: "command" });
      setQuestion("");
      return true;
    }
    if (command === "nodes") {
      const next = Number(rest[0]);
      if (Number.isFinite(next)) setContextNodes(Math.max(20, Math.min(900, next)));
      setQuestion("");
      return true;
    }
    if (command === "help") {
      addCommandHelp();
      setQuestion("");
      return true;
    }
    return false;
  }

  async function ask() {
    if (!question.trim() || !graph) return;
    if (executeCommand(question)) return;
    const role = activeChatRole;
    if (!role?.model) {
      addChatMessage({ role: "assistant", content: "Configure the chat model under Model Settings before asking the graph.", meta: "configuration" });
      return;
    }
    addChatMessage({ role: "user", content: question, meta: contextSummary(graph, selectedNodes) });
    const pending = addChatMessage({ role: "assistant", content: "Thinking...", meta: chatRoleName });
    setBusy(true);
    try {
      const res = await api.ask({
        question,
        selected_nodes: selectedNodes,
        query: contextQuery,
        depth: 1,
        max_nodes: contextNodes,
        max_edges: agentMode === "graph_rag" ? 520 : 160,
        context_mode: agentMode,
        report_context: selectedReport ? { out: selectedReport.out, max_chars: reportMaxChars } : null,
        model_config: role,
        history: recentHistory(),
      });
      const retrievedNodes = res.context.nodes || [];
      setLastRagNodes(agentMode === "graph_rag" ? retrievedNodes : []);
      if (agentMode === "graph_rag" && retrievedNodes.length) {
        setSearchResults(contextNodesToSearchResults(retrievedNodes));
      }
      updateChatMessage(pending, {
        content: res.answer || "(empty response)",
        meta: `${res.context.mode === "graph_rag" ? "graph-RAG" : "focused"} ${res.context.node_count}n/${res.context.edge_count}e${res.context.report_context ? ` | report ${res.context.report_context.title}` : ""}`,
      });
      setQuestion("");
    } catch (error) {
      updateChatMessage(pending, { content: error instanceof Error ? error.message : String(error), meta: "error" });
    } finally {
      setBusy(false);
    }
  }

  async function suggestFollowups() {
    if (!graph) return;
    const role = activeChatRole;
    if (!role?.model) {
      addChatMessage({ role: "assistant", content: "Configure the chat model under Model Settings before generating follow-ups.", meta: "configuration" });
      return;
    }
    const prompt =
      "Generate exactly four concise follow-up graph questions for the current user. Make them actionable for this graph explorer. Avoid numbering unless necessary.";
    setIdeaBusy(true);
    try {
      const res = await api.ask({
        question: prompt,
        selected_nodes: selectedNodes,
        query: contextQuery,
        depth: 1,
        max_nodes: Math.min(contextNodes, 120),
        max_edges: 160,
        context_mode: agentMode,
        report_context: selectedReport ? { out: selectedReport.out, max_chars: Math.min(reportMaxChars, 10000) } : null,
        model_config: role,
        history: recentHistory(),
      });
      const prompts = parseGeneratedPrompts(res.answer);
      if (prompts.length) setFollowups(prompts);
    } catch (error) {
      addChatMessage({ role: "assistant", content: error instanceof Error ? error.message : String(error), meta: "follow-up error" });
    } finally {
      setIdeaBusy(false);
    }
  }

  return (
    <section className="chat-panel">
      <div className="chat-head">
        <div>
          <h2>Assistant</h2>
          <span>{contextLabel} | {agentMode === "graph_rag" ? "Graph-RAG" : "Focused"} | chat model: {activeChatRole?.model || "not configured"}</span>
        </div>
        <div className="chat-actions">
          <span className="model-badge" title="Change this under Model Settings.">chat</span>
          <IconButton description="Clear the current browser-side chat thread." icon={<RotateCcw size={14} />} label="Reset" onClick={resetChat} />
        </div>
      </div>
      <div className="chat-log">
        {messages.length ? (
          messages.map((message) => (
            <div className={cx("chat-message", message.role)} key={message.id}>
              <div className="chat-avatar">{message.role === "user" ? "You" : "AI"}</div>
              <div className="chat-message-body">
                <div className="chat-meta">
                  {message.role}
                  {message.meta ? ` | ${message.meta}` : ""}
                </div>
                <div className="chat-bubble">{message.content}</div>
              </div>
            </div>
          ))
        ) : (
          <div className="empty-chat">
            {graph
              ? "Ask about selected nodes, graph structure, gaps, mechanisms, bridge paths, or the current run."
              : "Load a run, upload GraphML, or start a run to begin."}
          </div>
        )}
        <div ref={chatEndRef} />
      </div>
      <div className="chat-composer">
        <div className="prompt-row">
          {followups.map((prompt, index) => (
            <button key={prompt} onClick={() => setQuestion(prompt)} title={prompt} type="button">
              <span>{index === 0 ? "Next" : `Idea ${index + 1}`}</span>
              {prompt}
            </button>
          ))}
          <button disabled={ideaBusy || !graph} onClick={() => void suggestFollowups()} title="Generate follow-up ideas with the active graph model." type="button">
            {ideaBusy ? "Generating" : "Generate"}
          </button>
        </div>
        <div className="report-context-bar">
          <div>
            <FileText size={13} />
            <span>
              {selectedReport
                ? `Report context: ${selectedReport.label}`
                : sessionReports.length
                  ? "No report context attached"
                  : "No reports generated or opened in this session"}
            </span>
          </div>
          <button
            disabled={!sessionReports.length}
            onClick={() => setReportMenuOpen((value) => !value)}
            title="Attach a generated/opened report as additional chat context."
            type="button"
          >
            {selectedReport ? "Change" : "Attach"}
          </button>
          {selectedReport ? (
            <button onClick={() => setSelectedReportOut("")} title="Clear attached report context." type="button">
              Clear
            </button>
          ) : null}
        </div>
        {reportMenuOpen ? (
          <div className="report-context-menu">
            <div>
              <strong>Attach report context</strong>
              <span>Reports generated or opened during this browser session. The selected report is added to the next chat request.</span>
            </div>
            <button
              className={!selectedReportOut ? "active" : ""}
              onClick={() => {
                setSelectedReportOut("");
                setReportMenuOpen(false);
              }}
              type="button"
            >
              <strong>No report</strong>
              <span>Use only graph, selection, and retrieval context.</span>
            </button>
            {sessionReports.map((report, index) => (
              <button
                className={selectedReportOut === report.out ? "active" : ""}
                key={report.out}
                onClick={() => {
                  setSelectedReportOut(report.out);
                  setReportMenuOpen(false);
                }}
                type="button"
              >
                <strong>{index === 0 ? `Most recent: ${report.label}` : report.label}</strong>
                <span>{report.out}</span>
              </button>
            ))}
            <label>
              Max report chars
              <input
                max={50000}
                min={1000}
                onChange={(event) => setReportMaxChars(Number(event.target.value))}
                step={1000}
                type="number"
                value={reportMaxChars}
              />
            </label>
          </div>
        ) : null}
        {lastRagNodes.length ? (
          <div className="rag-context-strip">
            <div>
              <strong>Retrieved context</strong>
              <span>{lastRagNodes.length} nodes surfaced by Graph-RAG</span>
            </div>
            <div className="rag-node-chips">
              {lastRagNodes.slice(0, 18).map((node) => (
                <button
                  key={node.id}
                  onClick={() => setSelectedNode(node.id)}
                  title={`Select ${node.label} in the graph viewer`}
                  type="button"
                >
                  {node.label}
                </button>
              ))}
            </div>
          </div>
        ) : null}
        {question.startsWith("/") ? (
          <div className="command-menu">
            {commandItems.map((item) => (
              <button
                key={item.command}
                onClick={() => {
                  item.action();
                  setQuestion("");
                }}
                type="button"
              >
                <strong>{item.command}</strong>
                <span>{item.label} | {item.detail}</span>
              </button>
            ))}
          </div>
        ) : null}
        <textarea
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={(event) => {
            if ((event.metaKey || event.ctrlKey) && event.key === "Enter") void ask();
          }}
          placeholder={graph ? "Message the graph, or type / for commands..." : "Load a graph to chat..."}
          rows={4}
          value={question}
        />
        <div className="composer-tools">
          <label className="agent-mode">
            <span>
              Agent
              <HelpTip text="Focused sends selected nodes plus a compact neighborhood. Graph-RAG retrieves broader semantic/text matches, neighborhoods, bridge paths, and central nodes before answering." />
            </span>
            <select value={agentMode} onChange={(event) => setAgentMode(event.target.value as "focused" | "graph_rag")}>
              <option value="graph_rag">Graph-RAG</option>
              <option value="focused">Focused</option>
            </select>
          </label>
          <label className="context-query">
            <span>
              Focus query
              <HelpTip text="Optional search term used to pull matching nodes into the graph context packet. In Graph-RAG mode it is combined with your question for broader retrieval." />
            </span>
            <input
              onChange={(event) => setContextQuery(event.target.value)}
              placeholder="optional concept filter"
              value={contextQuery}
            />
          </label>
          <label className="context-count">
            <span>
              Max nodes
              <HelpTip text="Maximum retrieved nodes sent with each chat request. Graph-RAG can use a larger budget; the backend still caps it to protect the browser and model context." />
            </span>
            <input
              min={20}
              max={900}
              onChange={(event) => setContextNodes(Number(event.target.value))}
              type="number"
              value={contextNodes}
            />
          </label>
          <IconButton
            disabled={busy || !graph}
            icon={busy ? <Loader2 className="spin" size={14} /> : <Send size={14} />}
            label="Send"
            onClick={ask}
            tone="primary"
          />
        </div>
      </div>
    </section>
  );
}

function GraphRagExplorerTool({ defaultOpen = false }: { defaultOpen?: boolean }) {
  const graph = useExplorerStore((state) => state.graph);
  const selectedNodes = useExplorerStore((state) => state.selectedNodes);
  const setSelectedNode = useExplorerStore((state) => state.setSelectedNode);
  const setSearchResults = useExplorerStore((state) => state.setSearchResults);
  const messages = useExplorerStore((state) => state.chatMessages);
  const addChatMessage = useExplorerStore((state) => state.addChatMessage);
  const updateChatMessage = useExplorerStore((state) => state.updateChatMessage);
  const roles = useExplorerStore((state) => state.roles);
  const [query, setQuery] = useState("");
  const [maxNodes, setMaxNodes] = useState(260);
  const [nodes, setNodes] = useState<GraphAskContextNode[]>([]);
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);
  const [agentBusy, setAgentBusy] = useState(false);
  const activeChatRole = chatModelRole(roles);

  function promptText() {
    return (
      query.trim() ||
      (selectedNodes.length
        ? `Explore graph-RAG context around selected nodes: ${selectedNodes.map((id) => nodeLabel(graph, id)).join(", ")}`
        : "Explore the graph and surface the most relevant neighborhoods, bridge concepts, and structural gaps.")
    );
  }

  function recentHistory() {
    return messages
      .filter((message) => message.role === "user" || message.role === "assistant")
      .slice(-8)
      .map((message) => ({ role: message.role as "user" | "assistant", content: message.content }));
  }

  function applyContext(nextNodes: GraphAskContextNode[]) {
    setNodes(nextNodes);
    setSearchResults(contextNodesToSearchResults(nextNodes));
  }

  async function retrieve() {
    if (!graph) return;
    setBusy(true);
    setStatus("Retrieving graph-RAG context...");
    try {
      const res = await api.graphRagContext({
        question: promptText(),
        selected_nodes: selectedNodes,
        query,
        depth: 1,
        max_nodes: maxNodes,
        max_edges: 620,
      });
      applyContext(res.context.nodes || []);
      setStatus(`Retrieved ${formatNumber(res.context.node_count)} nodes and ${formatNumber(res.context.edge_count)} edges. Results are also available in Search.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  async function askAgent() {
    if (!graph) return;
    const role = activeChatRole;
    if (!role?.model) {
      setStatus("Configure the chat model under Model Settings before asking the Graph-RAG agent.");
      return;
    }
    const prompt = promptText();
    addChatMessage({ role: "user", content: prompt, meta: "Graph-RAG Explorer" });
    const pending = addChatMessage({ role: "assistant", content: "Retrieving graph context...", meta: "graph-RAG" });
    setAgentBusy(true);
    setStatus("Asking Graph-RAG agent...");
    try {
      const res = await api.ask({
        question: prompt,
        selected_nodes: selectedNodes,
        query,
        depth: 1,
        max_nodes: maxNodes,
        max_edges: 620,
        context_mode: "graph_rag",
        model_config: role,
        history: recentHistory(),
      });
      applyContext(res.context.nodes || []);
      updateChatMessage(pending, {
        content: res.answer || "(empty response)",
        meta: `graph-RAG ${res.context.node_count}n/${res.context.edge_count}e`,
      });
      setStatus(`Agent used ${formatNumber(res.context.node_count)} nodes and ${formatNumber(res.context.edge_count)} edges.`);
    } catch (error) {
      updateChatMessage(pending, { content: error instanceof Error ? error.message : String(error), meta: "graph-RAG error" });
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setAgentBusy(false);
    }
  }

  return (
    <Drawer
      defaultOpen={defaultOpen}
      description="Retrieve broad graph-RAG context from semantic/text matches, local neighborhoods, bridge paths, and centrality anchors. Use Retrieve for node surfacing or Ask Agent to post a stateful answer into chat."
      icon={<BrainCircuit size={14} />}
      note="retrieval agent"
      title="Graph-RAG Explorer"
    >
      <div className="tool-intro">
        <strong>{graph ? graph.name : "No graph loaded"}</strong>
        <span>{contextSummary(graph, selectedNodes)}</span>
      </div>
      <textarea
        onChange={(event) => setQuery(event.target.value)}
        placeholder="retrieval goal, question, or concept filter"
        rows={3}
        value={query}
      />
      <div className="control-grid">
        <label>
          Max nodes
          <input min={60} max={900} onChange={(event) => setMaxNodes(Number(event.target.value))} type="number" value={maxNodes} />
        </label>
        <label>
          Selection
          <input readOnly value={selectedNodes.length ? `${selectedNodes.length} selected` : "whole graph"} />
        </label>
      </div>
      <div className="button-row">
        <IconButton
          disabled={!graph || busy}
          description="Retrieve Graph-RAG context without calling the chat model. The surfaced nodes are sent to Search."
          icon={busy ? <Loader2 className="spin" size={14} /> : <Search size={14} />}
          label="Retrieve"
          onClick={retrieve}
          tone="primary"
        />
        <IconButton
          disabled={!graph || agentBusy}
          description="Ask the active chat model to interpret the retrieved graph-RAG context and post the answer into the assistant thread."
          icon={agentBusy ? <Loader2 className="spin" size={14} /> : <BrainCircuit size={14} />}
          label="Ask Agent"
          onClick={askAgent}
        />
      </div>
      {nodes.length ? (
        <div className="rag-tool-results">
          <div>
            <strong>Surfaced nodes</strong>
            <span>{formatNumber(nodes.length)} shown here; full list is in Search</span>
          </div>
          <div className="rag-node-chips">
            {nodes.slice(0, 36).map((node) => (
              <button key={node.id} onClick={() => setSelectedNode(node.id)} title={`Select ${node.label}`} type="button">
                {node.label}
              </button>
            ))}
          </div>
        </div>
      ) : null}
      {status ? <div className="status-box">{status}</div> : null}
    </Drawer>
  );
}

function ModelSettings({ defaultOpen = false }: { defaultOpen?: boolean }) {
  const roles = useExplorerStore((state) => state.roles);
  const updateRole = useExplorerStore((state) => state.updateRole);
  const setRoles = useExplorerStore((state) => state.setRoles);
  const [active, setActive] = useState("chat");
  const [status, setStatus] = useState("No server checked.");
  const [preview, setPreview] = useState("");
  const [probeBusy, setProbeBusy] = useState(false);
  const [serveHelp, setServeHelp] = useState<ModelProbe | null>(null);
  const role =
    roles[active] ||
    roles.chat ||
    roles.questioner ||
    Object.values(roles)[0] || {
      role: active,
      provider: "openai",
      model: "",
      base_url: "",
      api_key_env: "",
    };

  function roleWithPatch(patch: Partial<ModelRole>) {
    return { ...role, ...patch, role: active };
  }

  function patchRole(patch: Partial<ModelRole>) {
    updateRole(active, roleWithPatch(patch));
  }

  async function runProbe(nextRole = role) {
    setProbeBusy(true);
    setServeHelp(null);
    setStatus(`Testing ${nextRole.model || "selected model"}...`);
    try {
      const res = await api.modelProbe(nextRole);
      setStatus(modelProbeStatus(res));
      if (isLocalModelProbeIssue(res)) setServeHelp(res);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setProbeBusy(false);
    }
  }

  function applyPreset(label: string) {
    const preset = MODEL_PRESETS.find((item) => item.label === label);
    if (!preset) return;
    const nextRole = roleWithPatch(preset.values);
    updateRole(active, nextRole);
    setStatus(`Applied ${label} to ${active}. Testing...`);
    void runProbe(nextRole);
  }

  async function loadPreview(write = false) {
    const res = write ? await api.saveConfig(roles) : await api.configPreview(roles);
    setPreview(res.config);
    if ("path" in res) setStatus(`Wrote ${res.path}`);
  }

  return (
    <Drawer
      defaultOpen={defaultOpen}
      description="Configure model roles used by chat, graph QA, generation, judging, and local services. Test checks /models and a tiny completion; Write updates ideation/config.yaml."
      icon={<Settings2 size={14} />}
      note="roles, presets, config"
      title="Model Settings"
    >
      <div className="control-grid">
        <label>
          Role
          <select value={active} onChange={(event) => setActive(event.target.value)}>
            {Object.keys(roles).map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </label>
        <label>
          Preset
          <select value="" onChange={(event) => applyPreset(event.target.value)}>
            <option value="">Choose a preset...</option>
            {MODEL_PRESETS.map((preset) => (
              <option key={preset.label} value={preset.label}>
                {preset.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          Provider
          <select value={role.provider || "openai"} onChange={(event) => patchRole({ provider: event.target.value })}>
            <option value="openai">OpenAI / compatible API</option>
            <option value="hf">Local HF</option>
            <option value="embedding">Embedding</option>
          </select>
        </label>
        <label>
          Model
          <input value={role.model || ""} onChange={(event) => patchRole({ model: event.target.value })} />
        </label>
        <label>
          Base URL
          <input value={role.base_url || ""} onChange={(event) => patchRole({ base_url: event.target.value })} />
        </label>
        <label>
          API env
          <input value={role.api_key_env || ""} onChange={(event) => patchRole({ api_key_env: event.target.value })} />
        </label>
        <label>
          Temp
          <input value={role.temperature ?? ""} onChange={(event) => patchRole({ temperature: event.target.value })} />
        </label>
      </div>
      <div className="button-row">
        <IconButton
          description="Call /models and run a tiny completion against the selected model, surfacing auth and local serving errors."
          disabled={probeBusy}
          icon={probeBusy ? <Loader2 className="spin" size={14} /> : <Activity size={14} />}
          label="Test"
          onClick={() => void runProbe(role)}
        />
        <IconButton description="Render the config.yaml that would be written from these role settings." icon={<Command size={14} />} label="Preview" onClick={() => loadPreview(false)} />
        <IconButton description="Write these role settings to ideation/config.yaml." icon={<Download size={14} />} label="Write" onClick={() => loadPreview(true)} />
        <IconButton description="Reload roles from ideation/config.yaml." icon={<RotateCcw size={14} />} label="Reload" onClick={() => api.config().then((cfg) => setRoles(cfg.roles))} />
      </div>
      <div className="status-box">{status}</div>
      {preview ? <pre className="code-preview">{preview}</pre> : null}
      {serveHelp ? <ServeModelModal probe={serveHelp} onClose={() => setServeHelp(null)} /> : null}
    </Drawer>
  );
}

function Inspector({ defaultOpen = false }: { defaultOpen?: boolean }) {
  const graph = useExplorerStore((state) => state.graph);
  const selectedNode = useExplorerStore((state) => state.selectedNode);
  const node = graph?.nodes.find((item) => item.id === selectedNode);
  return (
    <Drawer
      defaultOpen={defaultOpen}
      description="Inspect the currently selected node. Select a node from search results or by clicking in the graph viewer."
      icon={<PanelLeft size={14} />}
      note="selected node"
      title="Selection Inspector"
    >
      {node ? (
        <div className="inspector">
          <strong>{node.label}</strong>
          <span>id: {node.id}</span>
          <span>degree: {formatNumber(node.degree)}</span>
          <span>pagerank: {formatNumber(node.pagerank, 5)}</span>
          <span>core: {formatNumber(node.core)}</span>
          <span>
            component: {formatNumber(node.component)} | community: {formatNumber(node.community)}
          </span>
          <span>
            iter: {formatNumber(node.iter)} | depth: {formatNumber(node.depth)}
          </span>
        </div>
      ) : (
        <div className="status-box">No node selected.</div>
      )}
    </Drawer>
  );
}

function FocusTools({ defaultOpen = false }: { defaultOpen?: boolean }) {
  const graph = useExplorerStore((state) => state.graph);
  const selectedNode = useExplorerStore((state) => state.selectedNode);
  const selectedNodes = useExplorerStore((state) => state.selectedNodes);
  const setGraph = useExplorerStore((state) => state.setGraph);
  const setHighlightedPaths = useExplorerStore((state) => state.setHighlightedPaths);
  const highlightedPaths = useExplorerStore((state) => state.highlightedPaths);
  const roles = useExplorerStore((state) => state.roles);
  const chatRole = useExplorerStore((state) => state.chatRole);
  const addChatMessage = useExplorerStore((state) => state.addChatMessage);
  const updateChatMessage = useExplorerStore((state) => state.updateChatMessage);
  const [depth, setDepth] = useState(1);
  const [limit, setLimit] = useState(400);
  const [source, setSource] = useState("");
  const [target, setTarget] = useState("");
  const [concepts, setConcepts] = useState("");
  const [pathMode, setPathMode] = useState<"pairwise" | "sequence" | "stochastic">("pairwise");
  const [pathCutoff, setPathCutoff] = useState(8);
  const [sampleCount, setSampleCount] = useState(28);
  const [connectors, setConnectors] = useState<PathConnector[]>([]);
  const [ideas, setIdeas] = useState<BridgeIdea[]>([]);
  const [status, setStatus] = useState("");
  const [pathBusy, setPathBusy] = useState(false);
  const [agentBusy, setAgentBusy] = useState(false);
  const [ideaBusy, setIdeaBusy] = useState(false);
  const ideaGraphRef = useRef("");
  const seed = selectedNodes.length ? selectedNodes : selectedNode ? [selectedNode] : [];
  const sourceMatches = useMemo(() => nodeLookupOptions(graph, source, selectedNodes), [graph, selectedNodes, source]);
  const targetMatches = useMemo(() => nodeLookupOptions(graph, target, selectedNodes), [graph, selectedNodes, target]);

  useEffect(() => {
    if (selectedNode && !source) setSource(selectedNode);
  }, [selectedNode, source]);

  useEffect(() => {
    const key = graph?.graph_id || graph?.path || graph?.name || "";
    if (!graph || !key || ideaGraphRef.current === key || concepts.trim()) return;
    ideaGraphRef.current = key;
    void suggestBridgeIdeas(true);
  }, [graph]);

  async function focusNeighborhood() {
    if (!seed.length) return;
    const next = await api.neighborhood({ nodes: seed, depth, limit });
    setGraph(next);
    setStatus(`${formatNumber(next.stats.nodes)} nodes | ${formatNumber(next.stats.edges)} edges`);
  }

  async function showPath() {
    if (!source.trim() || !target.trim()) return;
    setPathBusy(true);
    try {
      const next = await api.path({ source: source.trim(), target: target.trim(), k: 5, cutoff: pathCutoff });
      setGraph(next);
      setHighlightedPaths(next.paths || []);
      setConnectors([]);
      if (next.resolved_source) setSource(next.resolved_source);
      if (next.resolved_target) setTarget(next.resolved_target);
      setStatus(
        `${formatNumber(next.paths?.length || 0)} paths between ${nodeLabel(next, next.resolved_source || source)} and ${nodeLabel(next, next.resolved_target || target)}.`,
      );
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setPathBusy(false);
    }
  }

  async function findBridgeNetwork() {
    const nodes = seed.length >= 2 && !concepts.trim() ? seed : [];
    setPathBusy(true);
    try {
      const next = await api.multipath({
        nodes,
        query: concepts,
        mode: pathMode,
        cutoff: pathCutoff,
        anchor_limit: 8,
        sample_count: sampleCount,
      });
      setGraph(next);
      setHighlightedPaths(next.paths || []);
      setConnectors(next.connectors || []);
      setStatus(
        `${formatNumber(next.paths?.length || 0)} ${pathMode === "stochastic" ? "sampled " : ""}paths | ${formatNumber(next.connectors?.length || 0)} connector nodes. Highlighted paths are now drawn in the viewer.`,
      );
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setPathBusy(false);
    }
  }

  async function suggestBridgeIdeas(auto = false) {
    if (!graph) return;
    setIdeaBusy(true);
    try {
      const res = await api.bridgeSuggestions({ selected_nodes: seed, limit: 5 });
      setIdeas(res.ideas || []);
      if (res.ideas?.[0]?.query && (auto || !concepts.trim())) {
        setConcepts(res.ideas[0].query);
        setPathMode("pairwise");
      }
      setStatus(res.ideas?.length ? `${res.ideas.length} bridge ideas loaded. Click one, then run Find Bridges.` : "No bridge ideas found for this graph.");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setIdeaBusy(false);
    }
  }

  async function askPathAgent() {
    if (!graph) return;
    const role = graphAgentRole(roles, chatRole);
    if (!role?.model) {
      setStatus("Configure graph_qa or questioner before asking the graph agent.");
      return;
    }
    const connectorText = connectors.map((item) => `${item.label} (degree ${item.degree}, seen ${item.count}x)`).join(", ");
    const ideaText = ideas.map((idea) => `${idea.title}: ${idea.query} (${idea.rationale})`).join("\n");
    const hasPaths = highlightedPaths.length > 0;
    const pendingQuestion = hasPaths
      ? `Analyze the highlighted bridge paths. Identify the key connector concepts, explain why the route matters, and suggest the next graph queries. Connectors: ${connectorText || "none ranked"}.`
      : `Suggest high-value bridge searches for this graph before path highlighting. Use the current concept query if useful: ${concepts || "(none)"}. Candidate seeds:\n${ideaText || "(none generated yet)"}. Return concise bridge queries and why they are worth testing.`;
    addChatMessage({ role: "user", content: pendingQuestion, meta: hasPaths ? `${highlightedPaths.length} highlighted paths` : "bridge planning" });
    const pending = addChatMessage({ role: "assistant", content: hasPaths ? "Inspecting highlighted paths..." : "Planning bridge searches...", meta: "graph_qa" });
    setAgentBusy(true);
    try {
      const res = await api.ask({
        question: pendingQuestion,
        selected_nodes: hasPaths ? Array.from(new Set(highlightedPaths.flat())) : seed,
        query: concepts,
        depth: 1,
        max_nodes: 120,
        max_edges: 220,
        model_config: role,
      });
      updateChatMessage(pending, {
        content: res.answer || "(empty response)",
        meta: `path agent | ${res.context.node_count}n/${res.context.edge_count}e`,
      });
      setStatus("Agent response added to chat.");
    } catch (error) {
      updateChatMessage(pending, { content: error instanceof Error ? error.message : String(error), meta: "error" });
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setAgentBusy(false);
    }
  }

  async function restoreGraph() {
    setGraph(await api.graph());
    setHighlightedPaths([]);
    setConnectors([]);
    setStatus("");
  }

  return (
    <Drawer
      defaultOpen={defaultOpen}
      description="Build graph focus views. Use selected nodes for local neighborhoods, source-target path finding, or bridge networks across multiple concepts."
      icon={<Network size={14} />}
      note="paths, neighborhoods"
      title="Focus Tools"
    >
      <div className="tool-section">
        <div className="tool-section-head">
          <strong>Selection Focus</strong>
          <HelpTip text="Click a node in the center graph or a search result first. Neighborhood keeps nearby nodes; Whole Graph restores the full loaded graph." />
        </div>
      <div className="control-grid">
        <label>
          Depth
          <input min={0} max={4} onChange={(event) => setDepth(Number(event.target.value))} type="number" value={depth} />
        </label>
        <label>
          Limit
          <input min={50} max={2000} onChange={(event) => setLimit(Number(event.target.value))} type="number" value={limit} />
        </label>
      </div>
      <div className="button-row">
        <IconButton
          description="Replace the viewer with the selected node(s) plus nearby neighbors up to the chosen depth and limit."
          disabled={!graph || !seed.length}
          icon={<Network size={14} />}
          label="Neighborhood"
          onClick={focusNeighborhood}
        />
        <IconButton
          description="Restore the full currently loaded graph and clear highlighted paths."
          disabled={!graph}
          icon={<RotateCcw size={14} />}
          label="Whole Graph"
          onClick={restoreGraph}
        />
      </div>
      </div>
      <div className="tool-section">
        <div className="tool-section-head">
          <strong>Source-Target Path</strong>
          <HelpTip text="Type a node id or label, choose a close match, or select nodes in the graph and capture them as source/target. Find Paths highlights short routes in the viewer." />
        </div>
        <div className="selection-strip">
          <span>{seed.length ? `${seed.length} selected` : "No selected nodes"}</span>
          {seed.slice(0, 3).map((id) => (
            <button key={id} onClick={() => setSource(id)} type="button" title="Use this selected node as the source">
              {nodeLabel(graph, id)}
            </button>
          ))}
        </div>
      <div className="control-grid">
        <label>
          Source
          <input
            onChange={(event) => setSource(event.target.value)}
            placeholder="node id or label"
            title="Type a node id, exact label, or close label/attribute match."
            value={source}
          />
        </label>
        <label>
          Target
          <input
            onChange={(event) => setTarget(event.target.value)}
            placeholder="node id or label"
            title="Type a node id, exact label, or close label/attribute match."
            value={target}
          />
        </label>
      </div>
      <div className="lookup-grid">
        <div className="node-suggestions">
          {sourceMatches.map((node) => (
            <button key={`source-${node.id}`} onClick={() => setSource(node.id)} type="button" title="Use as source">
              {node.label}
            </button>
          ))}
        </div>
        <div className="node-suggestions">
          {targetMatches.map((node) => (
            <button key={`target-${node.id}`} onClick={() => setTarget(node.id)} type="button" title="Use as target">
              {node.label}
            </button>
          ))}
        </div>
      </div>
      <div className="button-row">
        <IconButton
          description="Use the most recent selected node as source."
          disabled={!seed.length}
          icon={<PanelLeft size={14} />}
          label="Set Source"
          onClick={() => setSource(seed[0] || "")}
        />
        <IconButton
          description="Use the most recent second selected node, or current selected node, as target."
          disabled={!seed.length}
          icon={<PanelLeft size={14} />}
          label="Set Target"
          onClick={() => setTarget(seed[1] || seed[0] || "")}
        />
        <IconButton
          description="Find short source-target paths and highlight them in the graph."
          disabled={!graph || pathBusy || !source.trim() || !target.trim()}
          icon={pathBusy ? <Loader2 className="spin" size={14} /> : <Network size={14} />}
          label="Find Paths"
          onClick={showPath}
          tone="primary"
        />
      </div>
      </div>
      <div className="tool-section">
        <div className="tool-section-head">
          <strong>Multi-Concept Bridge</strong>
          <HelpTip text="A bridge network resolves two or more concept labels/nodes, finds routes among them, and ranks interior connector nodes that repeatedly sit between the concepts." />
        </div>
      <label>
        Multi-concept bridge
        <textarea
          onChange={(event) => setConcepts(event.target.value)}
          placeholder="paste concept ids or labels, separated by commas or lines"
          rows={3}
          value={concepts}
        />
      </label>
      {ideas.length ? (
        <div className="bridge-ideas">
          {ideas.map((idea) => (
            <button
              key={`${idea.title}-${idea.query}`}
              onClick={() => {
                setConcepts(idea.query);
                setPathMode("pairwise");
                setStatus(`${idea.rationale} Run Find Bridges to draw the routes.`);
              }}
              type="button"
            >
              <strong>{idea.title}</strong>
              <span>{idea.rationale}</span>
            </button>
          ))}
        </div>
      ) : null}
      <div className="control-grid">
        <label>
          Mode <HelpTip text="Pairwise connects every anchor pair. Ordered route connects concepts in the order listed. Stochastic samples alternative routes with randomized edge weights to expose less obvious connectors." />
          <select value={pathMode} onChange={(event) => setPathMode(event.target.value as typeof pathMode)}>
            <option value="pairwise">Pairwise bridge</option>
            <option value="sequence">Ordered route</option>
            <option value="stochastic">Stochastic sampler</option>
          </select>
        </label>
        <label>
          Max hops
          <input min={2} max={16} onChange={(event) => setPathCutoff(Number(event.target.value))} type="number" value={pathCutoff} />
        </label>
        {pathMode === "stochastic" ? (
          <label>
            Samples
            <input min={4} max={80} onChange={(event) => setSampleCount(Number(event.target.value))} type="number" value={sampleCount} />
          </label>
        ) : null}
      </div>
      <div className="button-row">
        <IconButton
          disabled={!graph || pathBusy || (!concepts.trim() && seed.length < 2)}
          description="Resolve the listed concepts or current selected nodes, compute bridge paths, and highlight the resulting route network."
          icon={pathBusy ? <Loader2 className="spin" size={14} /> : <Network size={14} />}
          label="Find Bridges"
          onClick={findBridgeNetwork}
          tone="primary"
        />
        <IconButton
          disabled={!graph || ideaBusy}
          description="Generate structural bridge candidates from selected nodes and high-centrality graph anchors. Click an idea to load it."
          icon={ideaBusy ? <Loader2 className="spin" size={14} /> : <BrainCircuit size={14} />}
          label="Suggest Ideas"
          onClick={() => suggestBridgeIdeas(false)}
        />
        <IconButton
          disabled={!graph || agentBusy}
          description="Ask the configured graph_qa/questioner model to interpret highlighted paths or propose bridge queries."
          icon={agentBusy ? <Loader2 className="spin" size={14} /> : <BrainCircuit size={14} />}
          label="Ask Agent"
          onClick={askPathAgent}
        />
      </div>
      </div>
      {connectors.length ? (
        <div className="connector-list">
          {connectors.slice(0, 6).map((item) => (
            <button key={item.id} onClick={() => useExplorerStore.getState().setSelectedNode(item.id)} type="button">
              <strong>{item.label}</strong>
              <span>{item.count} paths | degree {formatNumber(item.degree)}</span>
            </button>
          ))}
        </div>
      ) : null}
      {status ? <div className="status-box">{status}</div> : null}
    </Drawer>
  );
}

function ChatSidebar({
  onLoadRun,
  onRunGraphReady,
  onRunStart,
  onReportReady,
}: {
  onLoadRun: (run: string) => Promise<void>;
  onRunGraphReady: (run: string) => Promise<void>;
  onRunStart: () => Promise<void>;
  onReportReady: (out: string) => void;
}) {
  const graph = useExplorerStore((state) => state.graph);
  const selectedNodes = useExplorerStore((state) => state.selectedNodes);
  return (
    <>
      <SidebarHeader title="Workspace" subtitle="left rail switches tools; assistant stays fixed" />
      <section className="panel-card">
        <div className="artifact-card">
          <div>
            <strong>{graph?.name || "No graph loaded"}</strong>
            <span>{contextSummary(graph, selectedNodes)}</span>
          </div>
          <Network size={16} />
        </div>
      </section>
      <SearchPanel />
      <RunExplorer onLoadRun={onLoadRun} />
      <RunMonitor onRunGraphReady={onRunGraphReady} onRunStart={onRunStart} />
      <ReportStudio onReportReady={onReportReady} />
    </>
  );
}

function SideRail({
  activeMode,
  onLoadRun,
  onRunGraphReady,
  onRunStart,
  onReportReady,
}: {
  activeMode: WorkspaceMode;
  onLoadRun: (run: string) => Promise<void>;
  onRunGraphReady: (run: string) => Promise<void>;
  onRunStart: () => Promise<void>;
  onReportReady: (out: string) => void;
}) {
  return (
    <aside className="side-panel">
      {activeMode === "chat" ? (
        <ChatSidebar onLoadRun={onLoadRun} onRunGraphReady={onRunGraphReady} onRunStart={onRunStart} onReportReady={onReportReady} />
      ) : null}
      {activeMode === "graph" ? (
        <>
          <SidebarHeader title="Graph" subtitle="viewer controls for the center artifact" />
          <Overview />
          <VisualControls defaultOpen />
          <Inspector defaultOpen />
        </>
      ) : null}
      {activeMode === "search" ? (
        <>
          <SidebarHeader title="Search" subtitle="find/select nodes, then route or focus" />
          <SearchPanel defaultOpen />
          <FocusTools defaultOpen />
        </>
      ) : null}
      {activeMode === "runs" ? (
        <>
          <SidebarHeader title="Runs" subtitle="load previous folders or launch new runs" />
          <RunExplorer defaultOpen onLoadRun={onLoadRun} />
          <RunMonitor defaultOpen onRunGraphReady={onRunGraphReady} onRunStart={onRunStart} />
        </>
      ) : null}
      {activeMode === "reports" ? (
        <>
          <SidebarHeader title="Reports" subtitle="generate reports and open artifacts" />
          <ReportStudio defaultOpen onReportReady={onReportReady} />
        </>
      ) : null}
      {activeMode === "graphrag" ? (
        <>
          <SidebarHeader title="Graph-RAG" subtitle="retrieve broad graph context" />
          <GraphRagExplorerTool defaultOpen />
        </>
      ) : null}
      {activeMode === "models" ? (
        <>
          <SidebarHeader title="Models" subtitle="roles, endpoints, and config" />
          <ModelSettings defaultOpen />
        </>
      ) : null}
    </aside>
  );
}

function ActivityRail({
  activeMode,
  onModeChange,
}: {
  activeMode: WorkspaceMode;
  onModeChange: (mode: WorkspaceMode) => void;
}) {
  const [addedTools, setAddedTools] = useState<OptionalToolMode[]>(readToolRailStorage);
  const [catalogOpen, setCatalogOpen] = useState(false);
  const coreModes: Array<{ id: CoreWorkspaceMode; label: string; icon: React.ReactNode }> = [
    { id: "chat", label: "Chat", icon: <BrainCircuit size={17} /> },
    { id: "graph", label: "Graph", icon: <Network size={17} /> },
    { id: "search", label: "Search", icon: <Search size={17} /> },
    { id: "runs", label: "Runs", icon: <Play size={17} /> },
    { id: "reports", label: "Reports", icon: <FileText size={17} /> },
    { id: "models", label: "Models", icon: <Settings2 size={17} /> },
  ];
  const optionalModes: Array<{ id: OptionalToolMode; label: string; icon: React.ReactNode; description: string }> = [
    {
      id: "graphrag",
      label: "Graph-RAG Explorer",
      icon: <BrainCircuit size={17} />,
      description: "Retrieve broad graph context, surface relevant nodes, and ask the graph-RAG agent.",
    },
  ];
  const modes: Array<{ id: WorkspaceMode; label: string; icon: React.ReactNode }> = [
    ...coreModes,
    ...optionalModes.filter((mode) => addedTools.includes(mode.id)),
  ];

  function addTool(id: OptionalToolMode) {
    const next = Array.from(new Set([...addedTools, id]));
    setAddedTools(next);
    writeToolRailStorage(next);
    setCatalogOpen(false);
    onModeChange(id);
  }

  return (
    <nav className="activity-rail" aria-label="Explorer modes">
      <span className="rail-kicker" title="These buttons switch the left tool panel. The assistant stays fixed on the right.">Tools</span>
      {modes.map((mode) => (
        <button
          aria-label={mode.label}
          aria-pressed={activeMode === mode.id}
          className={cx("rail-button", activeMode === mode.id && "active")}
          key={mode.id}
          onClick={() => onModeChange(mode.id)}
          title={mode.label}
          type="button"
        >
          {mode.icon}
        </button>
      ))}
      <div className="tool-add-wrap">
        <button
          aria-expanded={catalogOpen}
          aria-label="Add tool"
          className={cx("rail-button", "rail-add-button", catalogOpen && "active")}
          onClick={() => setCatalogOpen((value) => !value)}
          title="Add tool"
          type="button"
        >
          <Plus size={17} />
        </button>
        {catalogOpen ? (
          <div className="tool-catalog" role="menu">
            <div className="tool-catalog-head">
              <strong>Add tool</strong>
              <span>Optional rail tools</span>
            </div>
            {optionalModes.map((tool) => {
              const added = addedTools.includes(tool.id);
              return (
                <button
                  key={tool.id}
                  onClick={() => {
                    if (added) {
                      setCatalogOpen(false);
                      onModeChange(tool.id);
                    } else {
                      addTool(tool.id);
                    }
                  }}
                  type="button"
                >
                  <span>{tool.icon}</span>
                  <div>
                    <strong>{tool.label}</strong>
                    <small>{tool.description}</small>
                  </div>
                  <em>{added ? "Open" : "Add"}</em>
                </button>
              );
            })}
          </div>
        ) : null}
      </div>
    </nav>
  );
}

function ThreadStage({
  onOpenGraph,
  onOpenSearch,
  onOpenRuns,
  onOpenReports,
  embeddingStatus,
  onRebuildEmbeddings,
}: {
  onOpenGraph: () => void;
  onOpenSearch: () => void;
  onOpenRuns: () => void;
  onOpenReports: () => void;
  embeddingStatus: EmbeddingStatus | null;
  onRebuildEmbeddings: () => void;
}) {
  const graph = useExplorerStore((state) => state.graph);
  const selectedNodes = useExplorerStore((state) => state.selectedNodes);
  const stats = graph?.stats;

  return (
    <section className="thread-stage">
      <div className="thread-inner">
        <div className="thread-titlebar">
          <div>
            <strong>Current graph workspace</strong>
            <span>{graph?.topic || "Ask questions, inspect runs, and open graph artifacts as needed."}</span>
          </div>
          <button type="button" onClick={onOpenRuns}>
            Runs
          </button>
          <button type="button" onClick={onOpenReports}>
            Reports
          </button>
        </div>

        <div className="thread-artifact">
          <div className="artifact-icon">
            <Network size={18} />
          </div>
          <div className="artifact-copy">
            <strong>{graph?.name || "No graph loaded"}</strong>
            <span>{contextSummary(graph, selectedNodes)}</span>
          </div>
          <div className="artifact-metrics">
            <span>{formatNumber(stats?.components || 0)} components</span>
            <span>{formatNumber(stats?.communities || 0)} communities</span>
            <span>{formatNumber(stats?.avg_degree || 0, 2)} avg degree</span>
          </div>
          <div className="thread-actions">
            <button type="button" onClick={onOpenGraph}>
              Open graph
            </button>
            <button type="button" onClick={onOpenSearch}>
              Search nodes
            </button>
            <button type="button" onClick={onOpenReports}>
              Profile
            </button>
          </div>
        </div>

        <div className="thread-context-strip" title="Compact status for the current graph workspace. These are indicators, not controls.">
          <div>
            <span>Selection</span>
            <strong>{selectedNodes.length ? `${formatNumber(selectedNodes.length)} nodes` : "None"}</strong>
          </div>
          <div>
            <span>Graph</span>
            <strong>{graph ? `${formatNumber(stats?.nodes || 0)} / ${formatNumber(stats?.edges || 0)}` : "Not loaded"}</strong>
          </div>
          <div>
            <span>Run</span>
            <strong>{graph?.path ? graph.path.split("/runs/").pop()?.split("/")[0] || graph.name : "No session"}</strong>
          </div>
        </div>
        <EmbeddingStatusBar status={embeddingStatus} onRebuild={onRebuildEmbeddings} />
      </div>
    </section>
  );
}

function App() {
  const setGraph = useExplorerStore((state) => state.setGraph);
  const setRoles = useExplorerStore((state) => state.setRoles);
  const graph = useExplorerStore((state) => state.graph);
  const selectedNodes = useExplorerStore((state) => state.selectedNodes);
  const [activeMode, setActiveMode] = useState<WorkspaceMode>("chat");
  const [activeReportOut, setActiveReportOut] = useState(() => readReportStudioStorage().activeReportOut || "");
  const [sessionReports, setSessionReports] = useState<SessionReport[]>(readSessionReports);
  const [embeddingStatus, setEmbeddingStatus] = useState<EmbeddingStatus | null>(null);
  const { data: initialGraph } = useQuery({ queryKey: ["graph"], queryFn: api.graph, retry: false });
  const { data: config } = useQuery({ queryKey: ["config"], queryFn: api.config });

  const showGraph = React.useCallback(
    (next: GraphPayload) => {
      setGraph(next);
      setActiveMode("graph");
    },
    [setGraph],
  );

  useEffect(() => {
    if (initialGraph) showGraph(initialGraph);
  }, [initialGraph, showGraph]);

  useEffect(() => {
    if (config?.roles) setRoles(config.roles);
  }, [config, setRoles]);

  const startEmbeddingIndex = React.useCallback(async (force = false) => {
    try {
      setEmbeddingStatus(await api.startEmbeddings({ model: "auto", force }));
    } catch (error) {
      setEmbeddingStatus({
        status: "failed",
        ready: false,
        nodes: 0,
        dimension: 0,
        error: error instanceof Error ? error.message : String(error),
        progress: {
          percent: 0,
          current: 0,
          total: 0,
          message: "Embedding failed",
          detail: error instanceof Error ? error.message : String(error),
        },
      });
    }
  }, []);

  useEffect(() => {
    if (!graph?.graph_id) {
      setEmbeddingStatus(null);
      return undefined;
    }
    let cancelled = false;
    api.startEmbeddings({ model: "auto" })
      .then((status) => {
        if (!cancelled) setEmbeddingStatus(status);
      })
      .catch((error) => {
        if (!cancelled) {
          setEmbeddingStatus({
            status: "failed",
            ready: false,
            nodes: 0,
            dimension: 0,
            error: error instanceof Error ? error.message : String(error),
            progress: {
              percent: 0,
              current: 0,
              total: 0,
              message: "Embedding failed",
              detail: error instanceof Error ? error.message : String(error),
            },
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [graph?.graph_id]);

  useEffect(() => {
    if (!embeddingStatus || !["running"].includes(embeddingStatus.status)) return undefined;
    const timer = window.setInterval(async () => {
      try {
        setEmbeddingStatus(await api.embeddingStatus());
      } catch (error) {
        setEmbeddingStatus((state) => ({
          ...(state || {
            status: "failed",
            ready: false,
            nodes: 0,
            dimension: 0,
            progress: { percent: 0, current: 0, total: 0, message: "", detail: "" },
          }),
          status: "failed",
          ready: false,
          error: error instanceof Error ? error.message : String(error),
          progress: {
            ...(state?.progress || { percent: 0, current: 0, total: 0, message: "", detail: "" }),
            message: "Embedding status unavailable",
            detail: error instanceof Error ? error.message : String(error),
          },
        }));
      }
    }, 1800);
    return () => window.clearInterval(timer);
  }, [embeddingStatus]);

  const loadRun = React.useCallback(
    async (run: string) => {
      if (!run.trim()) return;
      showGraph(await api.loadRun(run));
    },
    [showGraph],
  );

  const refreshRunGraph = React.useCallback(
    async (run: string) => {
      if (!run.trim()) return;
      setGraph(await api.loadRun(run));
    },
    [setGraph],
  );

  const clearGraphForRun = React.useCallback(async () => {
    setGraph(null);
    await api.clearGraph();
  }, [setGraph]);

  const rememberReport = React.useCallback((out: string) => {
    if (!out.trim()) return;
    setSessionReports((state) => {
      const next = [
        { out, label: reportContextLabel(out), seenAt: Date.now() },
        ...state.filter((report) => report.out !== out),
      ].slice(0, 24);
      writeSessionReports(next);
      return next;
    });
  }, []);

  useEffect(() => {
    if (activeReportOut) rememberReport(activeReportOut);
  }, [activeReportOut, rememberReport]);

  const handleReportReady = React.useCallback(
    (out: string) => {
      if (!out.trim()) return;
      rememberReport(out);
      setActiveReportOut(out);
      setActiveMode("reports");
    },
    [rememberReport],
  );

  const graphArtifactOpen = activeMode === "graph" || activeMode === "search";
  const reportArtifactOpen = activeMode === "reports";

  return (
    <div className="app">
      <Header onGraphLoaded={showGraph} onLoadRun={loadRun} />
      <main className="workspace">
        <ActivityRail activeMode={activeMode} onModeChange={setActiveMode} />
        <SideRail
          activeMode={activeMode}
          onLoadRun={loadRun}
          onRunGraphReady={refreshRunGraph}
          onRunStart={clearGraphForRun}
          onReportReady={handleReportReady}
        />
        {reportArtifactOpen ? (
          <ReportStage out={activeReportOut} onOpenReports={() => setActiveMode("reports")} />
        ) : graphArtifactOpen ? (
          <section className="artifact-stage">
            <div className="artifact-toolbar">
              <div>
                <strong>Graph artifact</strong>
                <span>{contextSummary(graph, selectedNodes)}</span>
              </div>
              <div className="artifact-actions">
                <button type="button" onClick={() => setActiveMode("chat")}>
                  Back to chat
                </button>
                <button type="button" onClick={() => setActiveMode("graph")}>
                  Controls
                </button>
                <button type="button" onClick={() => setActiveMode("search")}>
                  Search
                </button>
              </div>
            </div>
            <EmbeddingStatusBar status={embeddingStatus} onRebuild={() => void startEmbeddingIndex(true)} />
            <GraphCanvas />
          </section>
        ) : (
          <ThreadStage
            embeddingStatus={embeddingStatus}
            onRebuildEmbeddings={() => void startEmbeddingIndex(true)}
            onOpenGraph={() => setActiveMode("graph")}
            onOpenReports={() => setActiveMode("reports")}
            onOpenRuns={() => setActiveMode("runs")}
            onOpenSearch={() => setActiveMode("search")}
          />
        )}
        <aside className="assistant-panel">
          <ChatPanel sessionReports={sessionReports} />
        </aside>
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </React.StrictMode>,
);
