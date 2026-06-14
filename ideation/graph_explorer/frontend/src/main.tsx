import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import * as Collapsible from "@radix-ui/react-collapsible";
import Graph from "graphology";
import Sigma from "sigma";
import {
  Activity,
  BrainCircuit,
  ChevronRight,
  CircleStop,
  Command,
  Download,
  FolderOpen,
  Loader2,
  Network,
  PanelLeft,
  Play,
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
import {
  colorScale,
  contextSummary,
  edgeColor,
  formatNumber,
  layoutNode,
  metricRange,
  nodeMetric,
  nodeSize,
  palette,
} from "./graph-utils";
import { useExplorerStore } from "./store";
import type { GraphNode, JobStatus, ModelRole, SearchResult } from "./types";
import "./styles.css";

type WorkspaceMode = "chat" | "graph" | "search" | "runs" | "models";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
      refetchOnWindowFocus: false,
    },
  },
});

function cx(...classes: Array<string | false | undefined>) {
  return classes.filter(Boolean).join(" ");
}

function IconButton({
  icon,
  label,
  onClick,
  disabled,
  tone = "default",
}: {
  icon: React.ReactNode;
  label: string;
  onClick?: () => void;
  disabled?: boolean;
  tone?: "default" | "primary" | "danger";
}) {
  return (
    <button
      className={cx("btn", tone === "primary" && "btn-primary", tone === "danger" && "btn-danger")}
      disabled={disabled}
      onClick={onClick}
      type="button"
      title={label}
    >
      {icon}
      <span>{label}</span>
    </button>
  );
}

function Drawer({
  title,
  note,
  icon,
  children,
  defaultOpen = false,
}: {
  title: string;
  note?: string;
  icon?: React.ReactNode;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  return (
    <Collapsible.Root defaultOpen={defaultOpen} className="drawer">
      <Collapsible.Trigger className="drawer-trigger">
        <span className="drawer-title">
          <ChevronRight className="drawer-chevron" size={14} />
          {icon}
          {title}
        </span>
        {note ? <span className="drawer-note">{note}</span> : null}
      </Collapsible.Trigger>
      <Collapsible.Content className="drawer-content">{children}</Collapsible.Content>
    </Collapsible.Root>
  );
}

function Header({ onLoadRun }: { onLoadRun: (run: string) => Promise<void> }) {
  const graph = useExplorerStore((state) => state.graph);
  const setGraph = useExplorerStore((state) => state.setGraph);
  const [run, setRun] = useState("");
  const [busy, setBusy] = useState(false);

  async function upload(file: File | undefined) {
    if (!file) return;
    setBusy(true);
    try {
      const text = await file.text();
      setGraph(await api.uploadGraphml(file.name, text));
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
        <label className="file-button">
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
          placeholder="runs/exp_leap or /path/to/file.graphml"
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

function SidebarHeader({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <div className="sidebar-header">
      <h2>{title}</h2>
      <span>{subtitle}</span>
    </div>
  );
}

function VisualControls({ defaultOpen = false }: { defaultOpen?: boolean }) {
  const visual = useExplorerStore((state) => state.visual);
  const setVisual = useExplorerStore((state) => state.setVisual);
  return (
    <Drawer defaultOpen={defaultOpen} icon={<SlidersHorizontal size={14} />} note="layout, color, size" title="Visual Mapping">
      <div className="control-grid">
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
      <div className="legend-row">
        {palette.slice(0, 8).map((color, index) => (
          <span key={color}>
            <i style={{ background: color }} />
            {index}
          </span>
        ))}
      </div>
    </Drawer>
  );
}

function SearchPanel({ defaultOpen = false }: { defaultOpen?: boolean }) {
  const results = useExplorerStore((state) => state.searchResults);
  const setSearchResults = useExplorerStore((state) => state.setSearchResults);
  const setSelectedNode = useExplorerStore((state) => state.setSelectedNode);
  const graph = useExplorerStore((state) => state.graph);
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState(false);

  async function runSearch() {
    if (!query.trim()) return;
    setBusy(true);
    try {
      setSearchResults((await api.search(query)).results);
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
  }

  return (
    <Drawer defaultOpen={defaultOpen} icon={<Search size={14} />} note={`${results.length} results`} title="Search & Select">
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
          icon={busy ? <Loader2 className="spin" size={14} /> : <Search size={14} />}
          label="Search"
          onClick={runSearch}
        />
      </div>
      <div className="button-row">
        <IconButton icon={<Network size={14} />} label="Top Hubs" onClick={showHubs} />
        <IconButton icon={<X size={14} />} label="Clear" onClick={() => setSearchResults([])} />
      </div>
      <div className="result-list">
        {results.map((result) => (
          <button className="result-item" key={result.id} onClick={() => setSelectedNode(result.id)} type="button">
            <strong>{result.label}</strong>
            <span>
              degree {formatNumber(result.degree)} | iter {formatNumber(result.iter)} | score {formatNumber(result.score, 2)}
            </span>
          </button>
        ))}
      </div>
    </Drawer>
  );
}

function GraphCanvas() {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const rendererRef = useRef<Sigma | null>(null);
  const graph = useExplorerStore((state) => state.graph);
  const visual = useExplorerStore((state) => state.visual);
  const selectedNodes = useExplorerStore((state) => state.selectedNodes);
  const setSelectedNode = useExplorerStore((state) => state.setSelectedNode);
  const [hoverNode, setHoverNode] = useState<GraphNode | null>(null);

  useEffect(() => {
    if (!containerRef.current || !graph) return undefined;
    rendererRef.current?.kill();

    const sigmaGraph = new Graph();
    const colorRange = metricRange(graph.nodes, visual.colorBy);
    const selected = new Set(selectedNodes);

    for (const node of graph.nodes) {
      const pos = layoutNode(node, graph, visual);
      const categoryColor =
        visual.colorBy === "component" || visual.colorBy === "community"
          ? palette[Math.abs(Math.floor(nodeMetric(node, visual.colorBy))) % palette.length]
          : colorScale(nodeMetric(node, visual.colorBy), colorRange);
      sigmaGraph.addNode(node.id, {
        ...pos,
        label: node.label,
        size: selected.has(node.id) ? nodeSize(node, graph, visual) * 1.8 : nodeSize(node, graph, visual),
        color: selected.has(node.id) ? "#ffffff" : categoryColor,
        borderColor: selected.has(node.id) ? "#37d49a" : categoryColor,
      });
    }

    const maxEdges = graph.edges.length > 45000 ? 45000 : graph.edges.length;
    for (let i = 0; i < maxEdges; i++) {
      const edge = graph.edges[i];
      if (sigmaGraph.hasNode(edge.source) && sigmaGraph.hasNode(edge.target) && !sigmaGraph.hasEdge(edge.id)) {
        sigmaGraph.addEdgeWithKey(edge.id, edge.source, edge.target, {
          size: graph.edges.length > 12000 ? 0.35 : 0.55,
          color: edgeColor(edge, visual.edgeOpacity),
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
  }, [graph, selectedNodes, setSelectedNode, visual]);

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

function ChatPanel() {
  const graph = useExplorerStore((state) => state.graph);
  const selectedNodes = useExplorerStore((state) => state.selectedNodes);
  const messages = useExplorerStore((state) => state.chatMessages);
  const addChatMessage = useExplorerStore((state) => state.addChatMessage);
  const updateChatMessage = useExplorerStore((state) => state.updateChatMessage);
  const resetChat = useExplorerStore((state) => state.resetChat);
  const roles = useExplorerStore((state) => state.roles);
  const chatRole = useExplorerStore((state) => state.chatRole);
  const setChatRole = useExplorerStore((state) => state.setChatRole);
  const [question, setQuestion] = useState("");
  const [contextQuery, setContextQuery] = useState("");
  const [contextNodes, setContextNodes] = useState(90);
  const [busy, setBusy] = useState(false);

  async function ask() {
    if (!question.trim() || !graph) return;
    const role = roles[chatRole];
    addChatMessage({ role: "user", content: question, meta: contextSummary(graph, selectedNodes) });
    const pending = addChatMessage({ role: "assistant", content: "Thinking...", meta: chatRole });
    setBusy(true);
    try {
      const res = await api.ask({
        question,
        selected_nodes: selectedNodes,
        query: contextQuery,
        depth: 1,
        max_nodes: contextNodes,
        max_edges: 160,
        model_config: role,
      });
      updateChatMessage(pending, {
        content: res.answer || "(empty response)",
        meta: `context ${res.context.node_count}n/${res.context.edge_count}e`,
      });
      setQuestion("");
    } catch (error) {
      updateChatMessage(pending, { content: error instanceof Error ? error.message : String(error), meta: "error" });
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="chat-panel">
      <div className="chat-head">
        <div>
          <h2>Graph Chat</h2>
          <span>{contextSummary(graph, selectedNodes)}</span>
        </div>
        <div className="chat-actions">
          <select value={chatRole} onChange={(event) => setChatRole(event.target.value)}>
            {Object.keys(roles).map((role) => (
              <option key={role} value={role}>
                {role}
              </option>
            ))}
          </select>
          <IconButton icon={<RotateCcw size={14} />} label="Reset" onClick={resetChat} />
        </div>
      </div>
      <div className="chat-log">
        {messages.length ? (
          messages.map((message) => (
            <div className={cx("chat-message", message.role)} key={message.id}>
              <div className="chat-meta">
                {message.role}
                {message.meta ? ` | ${message.meta}` : ""}
              </div>
              <div className="chat-bubble">{message.content}</div>
            </div>
          ))
        ) : (
          <div className="empty-chat">
            Ask about selected nodes, visible graph structure, gaps, mechanisms, or the current run.
          </div>
        )}
      </div>
      <div className="prompt-row">
        {[
          "Summarize this graph focus.",
          "Find gaps and weak links.",
          "Suggest bridge experiments.",
          "Rank the strongest hubs.",
        ].map((prompt) => (
          <button key={prompt} onClick={() => setQuestion(prompt)} type="button">
            {prompt.split(" ")[0]}
          </button>
        ))}
      </div>
      <input
        onChange={(event) => setContextQuery(event.target.value)}
        placeholder="optional extra context search"
        value={contextQuery}
      />
      <textarea
        onChange={(event) => setQuestion(event.target.value)}
        onKeyDown={(event) => {
          if ((event.metaKey || event.ctrlKey) && event.key === "Enter") void ask();
        }}
        placeholder="Ask about the graph..."
        rows={4}
        value={question}
      />
      <div className="composer-row">
        <label>
          Context nodes
          <input min={20} max={400} onChange={(event) => setContextNodes(Number(event.target.value))} type="number" value={contextNodes} />
        </label>
        <IconButton
          disabled={busy || !graph}
          icon={busy ? <Loader2 className="spin" size={14} /> : <Send size={14} />}
          label="Send"
          onClick={ask}
          tone="primary"
        />
      </div>
    </section>
  );
}

function ModelSettings({ defaultOpen = false }: { defaultOpen?: boolean }) {
  const roles = useExplorerStore((state) => state.roles);
  const updateRole = useExplorerStore((state) => state.updateRole);
  const setRoles = useExplorerStore((state) => state.setRoles);
  const [active, setActive] = useState("graph_qa");
  const [status, setStatus] = useState("No server checked.");
  const [preview, setPreview] = useState("");
  const role = roles[active];

  function patchRole(patch: Partial<ModelRole>) {
    updateRole(active, { ...role, ...patch, role: active });
  }

  async function check() {
    setStatus("Checking...");
    try {
      const res = await api.modelStatus(role);
      setStatus(`${res.ok ? "OK" : "Unavailable"} ${res.url || ""} ${res.message || ""}`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    }
  }

  async function loadPreview(write = false) {
    const res = write ? await api.saveConfig(roles) : await api.configPreview(roles);
    setPreview(res.config);
    if ("path" in res) setStatus(`Wrote ${res.path}`);
  }

  return (
    <Drawer defaultOpen={defaultOpen} icon={<Settings2 size={14} />} note="roles, presets, config" title="Model Settings">
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
          Provider
          <select value={role.provider} onChange={(event) => patchRole({ provider: event.target.value })}>
            <option value="openai">OpenAI-compatible</option>
            <option value="hf">Local HF</option>
            <option value="embedding">Embedding</option>
          </select>
        </label>
        <label>
          Model
          <input value={role.model} onChange={(event) => patchRole({ model: event.target.value })} />
        </label>
        <label>
          Base URL
          <input value={role.base_url} onChange={(event) => patchRole({ base_url: event.target.value })} />
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
        <IconButton icon={<Activity size={14} />} label="Check" onClick={check} />
        <IconButton icon={<Command size={14} />} label="Preview" onClick={() => loadPreview(false)} />
        <IconButton icon={<Download size={14} />} label="Write" onClick={() => loadPreview(true)} />
        <IconButton icon={<RotateCcw size={14} />} label="Reload" onClick={() => api.config().then((cfg) => setRoles(cfg.roles))} />
      </div>
      <div className="status-box">{status}</div>
      {preview ? <pre className="code-preview">{preview}</pre> : null}
    </Drawer>
  );
}

function RunMonitor({ onRunGraphReady, defaultOpen = false }: { onRunGraphReady: (run: string) => Promise<void>; defaultOpen?: boolean }) {
  const [topic, setTopic] = useState("");
  const [strategy, setStrategy] = useState("frontier");
  const [calls, setCalls] = useState(50);
  const [iters, setIters] = useState(50);
  const [out, setOut] = useState("runs/explorer_run");
  const [job, setJob] = useState<JobStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const lastSnapshotRef = useRef<string>("");

  useEffect(() => {
    if (!job || !["running", "stopping"].includes(job.status)) return undefined;
    const timer = window.setInterval(async () => {
      const next = await api.job(job.id);
      setJob(next);
      const snapshot = next.snapshot_id || (next.graph_ready ? `${next.graph_path || next.out}:ready` : "");
      if (next.graph_ready && next.out && snapshot && snapshot !== lastSnapshotRef.current) {
        lastSnapshotRef.current = snapshot;
        void onRunGraphReady(next.out);
      }
    }, 2200);
    return () => window.clearInterval(timer);
  }, [job, onRunGraphReady]);

  async function start() {
    if (!topic.trim()) return;
    setBusy(true);
    try {
      lastSnapshotRef.current = "";
      setJob(await api.ideate({ topic, strategy, budget_calls: calls, max_iters: iters, out }));
    } finally {
      setBusy(false);
    }
  }

  async function stop() {
    if (!job) return;
    setJob(await api.stopJob(job.id));
  }

  const progress = job?.status === "done" ? 100 : Math.round((job?.progress?.percent || 0) * 100);

  return (
    <Drawer defaultOpen={defaultOpen} icon={<Play size={14} />} note={job?.status || "idle"} title="Run Monitor">
      <textarea onChange={(event) => setTopic(event.target.value)} placeholder="topic or benchmark task" rows={3} value={topic} />
      <div className="control-grid">
        <label>
          Strategy
          <input onChange={(event) => setStrategy(event.target.value)} value={strategy} />
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
          Out
          <input onChange={(event) => setOut(event.target.value)} value={out} />
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
          icon={busy ? <Loader2 className="spin" size={14} /> : <Play size={14} />}
          label="Start"
          onClick={start}
          tone="primary"
        />
        <IconButton
          disabled={!job || job.status !== "running"}
          icon={<CircleStop size={14} />}
          label="Stop"
          onClick={stop}
          tone="danger"
        />
      </div>
      {job?.log_tail ? <pre className="run-log">{job.log_tail}</pre> : null}
    </Drawer>
  );
}

function Inspector({ defaultOpen = false }: { defaultOpen?: boolean }) {
  const graph = useExplorerStore((state) => state.graph);
  const selectedNode = useExplorerStore((state) => state.selectedNode);
  const node = graph?.nodes.find((item) => item.id === selectedNode);
  return (
    <Drawer defaultOpen={defaultOpen} icon={<PanelLeft size={14} />} note="selected node" title="Selection Inspector">
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
  const [depth, setDepth] = useState(1);
  const [limit, setLimit] = useState(400);
  const [source, setSource] = useState("");
  const [target, setTarget] = useState("");
  const [status, setStatus] = useState("");
  const seed = selectedNodes.length ? selectedNodes : selectedNode ? [selectedNode] : [];

  useEffect(() => {
    if (selectedNode && !source) setSource(selectedNode);
  }, [selectedNode, source]);

  async function focusNeighborhood() {
    if (!seed.length) return;
    const next = await api.neighborhood({ nodes: seed, depth, limit });
    setGraph(next);
    setStatus(`${formatNumber(next.stats.nodes)} nodes | ${formatNumber(next.stats.edges)} edges`);
  }

  async function showPath() {
    if (!source.trim() || !target.trim()) return;
    const next = await api.path({ source: source.trim(), target: target.trim(), k: 5, cutoff: 8 });
    setGraph(next);
    setStatus(`${formatNumber(next.paths?.length || 0)} paths`);
  }

  async function restoreGraph() {
    setGraph(await api.graph());
    setStatus("");
  }

  return (
    <Drawer defaultOpen={defaultOpen} icon={<Network size={14} />} note="paths, neighborhoods" title="Focus Tools">
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
        <IconButton disabled={!graph || !seed.length} icon={<Network size={14} />} label="Neighborhood" onClick={focusNeighborhood} />
        <IconButton disabled={!graph} icon={<RotateCcw size={14} />} label="Whole Graph" onClick={restoreGraph} />
      </div>
      <div className="control-grid">
        <label>
          Source
          <input onChange={(event) => setSource(event.target.value)} placeholder="source node id" value={source} />
        </label>
        <label>
          Target
          <input onChange={(event) => setTarget(event.target.value)} placeholder="target node id" value={target} />
        </label>
      </div>
      <div className="button-row">
        <IconButton disabled={!graph || !source.trim() || !target.trim()} icon={<Network size={14} />} label="Paths" onClick={showPath} />
      </div>
      {status ? <div className="status-box">{status}</div> : null}
    </Drawer>
  );
}

function ChatSidebar({ onRunGraphReady }: { onRunGraphReady: (run: string) => Promise<void> }) {
  const graph = useExplorerStore((state) => state.graph);
  const selectedNodes = useExplorerStore((state) => state.selectedNodes);
  return (
    <>
      <SidebarHeader title="Workspace" subtitle="chat-first graph exploration" />
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
      <RunMonitor onRunGraphReady={onRunGraphReady} />
    </>
  );
}

function SideRail({ activeMode, onRunGraphReady }: { activeMode: WorkspaceMode; onRunGraphReady: (run: string) => Promise<void> }) {
  return (
    <aside className="side-panel">
      {activeMode === "chat" ? <ChatSidebar onRunGraphReady={onRunGraphReady} /> : null}
      {activeMode === "graph" ? (
        <>
          <SidebarHeader title="Graph" subtitle="artifact controls and inspection" />
          <Overview />
          <VisualControls defaultOpen />
          <FocusTools defaultOpen />
          <Inspector defaultOpen />
        </>
      ) : null}
      {activeMode === "search" ? (
        <>
          <SidebarHeader title="Search" subtitle="find nodes and focus concepts" />
          <SearchPanel defaultOpen />
          <FocusTools defaultOpen />
        </>
      ) : null}
      {activeMode === "runs" ? (
        <>
          <SidebarHeader title="Runs" subtitle="launch, monitor, and refresh" />
          <RunMonitor defaultOpen onRunGraphReady={onRunGraphReady} />
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
  const modes: Array<{ id: WorkspaceMode; label: string; icon: React.ReactNode }> = [
    { id: "chat", label: "Chat", icon: <BrainCircuit size={17} /> },
    { id: "graph", label: "Graph", icon: <Network size={17} /> },
    { id: "search", label: "Search", icon: <Search size={17} /> },
    { id: "runs", label: "Runs", icon: <Play size={17} /> },
    { id: "models", label: "Models", icon: <Settings2 size={17} /> },
  ];
  return (
    <nav className="activity-rail" aria-label="Explorer modes">
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
    </nav>
  );
}

function App() {
  const setGraph = useExplorerStore((state) => state.setGraph);
  const setRoles = useExplorerStore((state) => state.setRoles);
  const graph = useExplorerStore((state) => state.graph);
  const selectedNodes = useExplorerStore((state) => state.selectedNodes);
  const [activeMode, setActiveMode] = useState<WorkspaceMode>("chat");
  const { data: initialGraph } = useQuery({ queryKey: ["graph"], queryFn: api.graph, retry: false });
  const { data: config } = useQuery({ queryKey: ["config"], queryFn: api.config });

  useEffect(() => {
    if (initialGraph) setGraph(initialGraph);
  }, [initialGraph, setGraph]);

  useEffect(() => {
    if (config?.roles) setRoles(config.roles);
  }, [config, setRoles]);

  const loadRun = React.useCallback(
    async (run: string) => {
      if (!run.trim()) return;
      setGraph(await api.loadRun(run));
    },
    [setGraph],
  );

  return (
    <div className="app">
      <Header onLoadRun={loadRun} />
      <main className="workspace">
        <ActivityRail activeMode={activeMode} onModeChange={setActiveMode} />
        <SideRail activeMode={activeMode} onRunGraphReady={loadRun} />
        <section className="artifact-stage">
          <div className="artifact-toolbar">
            <div>
              <strong>Graph artifact</strong>
              <span>{contextSummary(graph, selectedNodes)}</span>
            </div>
            <div className="artifact-actions">
              <button type="button" onClick={() => setActiveMode("graph")}>Controls</button>
              <button type="button" onClick={() => setActiveMode("search")}>Search</button>
            </div>
          </div>
          <GraphCanvas />
        </section>
        <aside className="assistant-panel">
          <ChatPanel />
          {!graph ? (
            <div className="empty-state">
              <BrainCircuit size={18} />
              Load a run, upload GraphML, or start a CLI run to begin.
            </div>
          ) : null}
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
