import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import Graph from "graphology";
import Sigma from "sigma";
import { EdgeArrowProgram, EdgeLineProgram } from "sigma/rendering";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import {
  Activity,
  CircleAlert,
  CircleCheck,
  BrainCircuit,
  ChevronLeft,
  ChevronRight,
  Command,
  Download,
  FileText,
  FolderOpen,
  Info,
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
  Sparkles,
  X,
} from "lucide-react";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { api } from "./api";
import { cx, Drawer, HelpTip, IconButton, SidebarHeader } from "./components/common";
import { MarkdownReport, ReportStage, ReportStudio, readReportStudioStorage } from "./features/reporting";
import { RunDashboardPanel } from "./features/run-dashboard";
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
  ChatMessage,
  EmbeddingStatus,
  GraphFileSummary,
  GraphAskContextNode,
  GraphNode,
  GraphPayload,
  ModelProbe,
  ModelRole,
  PathConnector,
  SearchResult,
} from "./types";
import "./styles.css";

const logoUrl = "/api/logo";

type CoreWorkspaceMode = "chat" | "graph" | "search" | "runs" | "reports" | "models";
type OptionalToolMode = "graphrag";
type WorkspaceMode = CoreWorkspaceMode | OptionalToolMode;
type ChatContextMode = "none" | "focused" | "graph_rag";
type ReportContextPart = "report" | "profile";

const TOOL_RAIL_STORAGE_KEY = "graph-preflexor-explorer.tool-rail.v1";
const SESSION_REPORTS_STORAGE_KEY = "graph-preflexor-explorer.session-reports.v1";
const PANEL_WIDTH_STORAGE_KEY = "graph-preflexor-explorer.panel-widths.v1";
const OPTIONAL_TOOL_IDS = ["graphrag"] as const;
const DEFAULT_CHAT_MAX_OUTPUT_TOKENS = 20000;
const HIGH_CHAT_MAX_OUTPUT_TOKENS = 20000;
const GENERATOR_MAX_OUTPUT_TOKENS = 8000;

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

function chatContextLabel(mode: ChatContextMode) {
  if (mode === "graph_rag") return "Graph-RAG retrieval";
  if (mode === "focused") return "Focused selection";
  return "None";
}

function reportPartsLabel(parts: ReportContextPart[]) {
  const labels = [];
  if (parts.includes("report")) labels.push("report.md");
  if (parts.includes("profile")) labels.push("profile.json");
  return labels.join(" + ") || "no files";
}

function reportPartsFromIncluded(included?: string[]): ReportContextPart[] {
  const parts: ReportContextPart[] = [];
  if (!included?.length || included.includes("report.md")) parts.push("report");
  if (included?.includes("profile.json")) parts.push("profile");
  return parts;
}

function clampNumber(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function readPanelWidths() {
  if (typeof window === "undefined") return { left: 282, right: 388 };
  try {
    const stored = JSON.parse(window.localStorage.getItem(PANEL_WIDTH_STORAGE_KEY) || "{}");
    return {
      left: clampNumber(Number(stored.left) || 282, 220, 560),
      right: clampNumber(Number(stored.right) || 388, 300, 680),
    };
  } catch {
    return { left: 282, right: 388 };
  }
}

function writePanelWidths(widths: { left: number; right: number }) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(PANEL_WIDTH_STORAGE_KEY, JSON.stringify(widths));
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

function normalizedBackend(role?: ModelRole) {
  if (!role || role.provider === "hf" || role.provider === "embedding") return role?.backend || "";
  const backend = (role.backend || "responses").toLowerCase().replace("_", "-");
  return ["chat", "chat-completions", "chat.completions"].includes(backend) ? "chat" : "responses";
}

function isResponsesRole(role?: ModelRole) {
  return normalizedBackend(role).toLowerCase() === "responses";
}

function previousResponseId(messages: ChatMessage[], role?: ModelRole) {
  if (!isResponsesRole(role)) return "";
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (
      message.role === "assistant" &&
      message.response_id &&
      message.response_model === role?.model &&
      message.response_backend === normalizedBackend(role) &&
      message.response_base_url === (role?.base_url || "")
    ) {
      return message.response_id;
    }
  }
  return "";
}

function responseStateMeta(role: ModelRole, responseId?: string) {
  return responseId
    ? {
        response_id: responseId,
        response_model: role.model,
        response_backend: normalizedBackend(role),
        response_base_url: role.base_url || "",
      }
    : {};
}

function chatStateMeta(res: { stateful?: boolean; state_mode?: string }) {
  if (res.state_mode === "responses_previous_response_id") return " | Responses state";
  if (res.state_mode === "history_replay") return " | history replay";
  if (res.stateful) return " | stateful";
  return "";
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

function inferRunFromGraphPath(path = "") {
  const match = path.match(/(?:^|\/)(runs\/[^/]+)/);
  return match?.[1] || "";
}

function snapshotLabel(snapshot: GraphFileSummary) {
  return snapshot.iter == null ? "Final graph" : `Iteration ${snapshot.iter}`;
}

function snapshotPath(snapshot: GraphFileSummary) {
  return snapshot.absolute_path || snapshot.path || "";
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

function splitCommandArgs(input: string) {
  const out: string[] = [];
  input.replace(/"([^"]*)"|'([^']*)'|(\S+)/g, (_, dq, sq, bare) => {
    out.push(dq ?? sq ?? bare ?? "");
    return "";
  });
  return out;
}

function parseSynthesizeCommand(input: string) {
  const args = splitCommandArgs(input);
  const task: string[] = [];
  const opts: {
    task: string;
    style?: string;
    backend?: string;
    model?: string;
    base_url?: string;
    max_leads?: number;
    max_tokens?: number;
    temperature?: number;
    mine?: boolean;
    no_insights?: boolean;
  } = { task: "" };
  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    const next = args[i + 1];
    if (arg === "--style" && next) {
      opts.style = next;
      i += 1;
    } else if (arg === "--backend" && next) {
      opts.backend = next;
      i += 1;
    } else if (arg === "--hf" && next) {
      opts.backend = "hf";
      opts.model = next;
      i += 1;
    } else if (arg === "--model" && next) {
      opts.model = next;
      i += 1;
    } else if (arg === "--base-url" && next) {
      opts.base_url = next;
      i += 1;
    } else if ((arg === "--leads" || arg === "--max-leads") && next) {
      opts.max_leads = Number(next);
      i += 1;
    } else if ((arg === "--tokens" || arg === "--max-tokens") && next) {
      opts.max_tokens = Number(next);
      i += 1;
    } else if (arg === "--temperature" && next) {
      opts.temperature = Number(next);
      i += 1;
    } else if (arg === "--mine") {
      opts.mine = true;
    } else if (arg === "--baseline" || arg === "--no-insights") {
      opts.no_insights = true;
    } else {
      task.push(arg);
    }
  }
  opts.task = task.join(" ").trim();
  return opts;
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
  const toml = `# Save as models.${model.replace(/[^a-zA-Z0-9_.-]+/g, "_")}.toml, then run:
#   mistralrs from-config -f models.${model.replace(/[^a-zA-Z0-9_.-]+/g, "_")}.toml
command = "serve"
default_model_id = "${model}"

[server]
host = "127.0.0.1"
port = ${port}

[[models]]
kind = "auto"
model_id = "${model}"
[models.quantization]
in_situ_quant = "8"
`;
  return {
    mistral: `mistralrs serve -m ${model} --host 127.0.0.1 --port ${port}`,
    mistralToml: toml,
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

function DownloadTextButton({ fileName, text }: { fileName: string; text: string }) {
  return (
    <button
      onClick={() => {
        const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = fileName;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
      }}
      type="button"
    >
      Download
    </button>
  );
}

function ServeModelModal({ probe, onClose }: { probe: ModelProbe; onClose: () => void }) {
  const commands = serveCommands(probe);
  const tomlName = `models.${(probe.model || "local_model").replace(/[^a-zA-Z0-9_.-]+/g, "_")}.toml`;
  return (
    <div className="model-modal-backdrop" role="presentation">
      <section aria-modal="true" className="serve-modal" role="dialog">
        <div className="serve-modal-head">
          <div>
            <span className="eyebrow">Local model check</span>
            <h3>Model probe needs attention</h3>
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
          Start or adjust the local OpenAI-compatible server, then run Test again. The explorer expects
          <strong> OpenAI Responses</strong> at /v1/responses for local and hosted models.
        </p>
        <div className="command-card">
          <div>
            <strong>mistral.rs single model</strong>
            <CopyCommandButton text={commands.mistral} />
          </div>
          <pre>{commands.mistral}</pre>
        </div>
        <div className="command-card">
          <div>
            <strong>mistral.rs TOML option</strong>
            <span className="command-actions">
              <CopyCommandButton text={commands.mistralToml} />
              <DownloadTextButton fileName={tomlName} text={commands.mistralToml} />
            </span>
          </div>
          <pre>{commands.mistralToml}</pre>
        </div>
        <div className="command-card">
          <div>
            <strong>vLLM single model</strong>
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
    label: "Local Gemma on 1234",
    values: {
      provider: "openai",
      model: "google/gemma-4-E4B-it",
      base_url: "http://localhost:1234/v1",
      backend: "responses",
      api_key_env: "",
      temperature: 0.3,
      max_tokens: DEFAULT_CHAT_MAX_OUTPUT_TOKENS,
      reasoning_effort: "",
    },
  },
  {
    label: "Local Gemma on 8000",
    values: {
      provider: "openai",
      model: "google/gemma-4-E4B-it",
      base_url: "http://localhost:8000/v1",
      backend: "responses",
      api_key_env: "",
      temperature: 0.3,
      max_tokens: DEFAULT_CHAT_MAX_OUTPUT_TOKENS,
      reasoning_effort: "",
    },
  },
  {
    label: "Local Qwen3-4B on 1234",
    values: {
      provider: "openai",
      model: "Qwen/Qwen3-4B",
      base_url: "http://localhost:1234/v1",
      backend: "responses",
      api_key_env: "",
      temperature: 0.3,
      max_tokens: DEFAULT_CHAT_MAX_OUTPUT_TOKENS,
      reasoning_effort: "",
    },
  },
  {
    label: "Local Qwen3-4B on 8000",
    values: {
      provider: "openai",
      model: "Qwen/Qwen3-4B",
      base_url: "http://localhost:8000/v1",
      backend: "responses",
      api_key_env: "",
      temperature: 0.3,
      max_tokens: DEFAULT_CHAT_MAX_OUTPUT_TOKENS,
      reasoning_effort: "",
    },
  },
  {
    label: "OpenAI gpt-5.5",
    values: {
      provider: "openai",
      model: "gpt-5.5",
      base_url: "https://api.openai.com/v1",
      backend: "responses",
      api_key_env: "OPENAI_API_KEY",
      temperature: 0.3,
      max_tokens: DEFAULT_CHAT_MAX_OUTPUT_TOKENS,
      reasoning_effort: "medium",
    },
  },
  {
    label: "OpenAI gpt-5.5-mini",
    values: {
      provider: "openai",
      model: "gpt-5.5-mini",
      base_url: "https://api.openai.com/v1",
      backend: "responses",
      api_key_env: "OPENAI_API_KEY",
      temperature: 0.3,
      max_tokens: DEFAULT_CHAT_MAX_OUTPUT_TOKENS,
      reasoning_effort: "medium",
    },
  },
  {
    label: "OpenAI gpt-5.5-nano",
    values: {
      provider: "openai",
      model: "gpt-5.5-nano",
      base_url: "https://api.openai.com/v1",
      backend: "responses",
      api_key_env: "OPENAI_API_KEY",
      temperature: 0.3,
      max_tokens: DEFAULT_CHAT_MAX_OUTPUT_TOKENS,
      reasoning_effort: "low",
    },
  },
  {
    label: "OpenAI gpt-5.5 high reasoning",
    values: {
      provider: "openai",
      model: "gpt-5.5",
      base_url: "https://api.openai.com/v1",
      backend: "responses",
      api_key_env: "OPENAI_API_KEY",
      temperature: 0.2,
      max_tokens: HIGH_CHAT_MAX_OUTPUT_TOKENS,
      reasoning_effort: "high",
    },
  },
  {
    label: "Local Llama on 1234",
    values: {
      provider: "openai",
      model: "meta-llama/Llama-3.2-3B-Instruct",
      base_url: "http://localhost:1234/v1",
      backend: "responses",
      api_key_env: "",
      temperature: 0.3,
      max_tokens: DEFAULT_CHAT_MAX_OUTPUT_TOKENS,
      reasoning_effort: "",
    },
  },
  {
    label: "Local Llama on 8000",
    values: {
      provider: "openai",
      model: "meta-llama/Llama-3.2-3B-Instruct",
      base_url: "http://localhost:8000/v1",
      backend: "responses",
      api_key_env: "",
      temperature: 0.3,
      max_tokens: DEFAULT_CHAT_MAX_OUTPUT_TOKENS,
      reasoning_effort: "",
    },
  },
  {
    label: "Graph-PRefLexOR Generator 1.7b",
    values: {
      provider: "openai",
      model: "lamm-mit/Graph-Preflexor-1.7b_08012026",
      base_url: "http://localhost:1234/v1",
      backend: "responses",
      api_key_env: "",
      temperature: 0.1,
      max_tokens: GENERATOR_MAX_OUTPUT_TOKENS,
      reasoning_effort: "",
    },
  },
  {
    label: "Graph-PRefLexOR Generator 3b",
    values: {
      provider: "openai",
      model: "lamm-mit/Graph-Preflexor-3b_08012026",
      base_url: "http://localhost:1234/v1",
      backend: "responses",
      api_key_env: "",
      temperature: 0.1,
      max_tokens: GENERATOR_MAX_OUTPUT_TOKENS,
      reasoning_effort: "",
    },
  },
  {
    label: "Graph-PRefLexOR Generator 8b",
    values: {
      provider: "openai",
      model: "lamm-mit/Graph-Preflexor-8b_12292025",
      base_url: "http://localhost:1234/v1",
      backend: "responses",
      api_key_env: "",
      temperature: 0.1,
      max_tokens: GENERATOR_MAX_OUTPUT_TOKENS,
      reasoning_effort: "",
    },
  },
];

function Header() {
  const graph = useExplorerStore((state) => state.graph);

  return (
    <header className="topbar">
      <div className="brand">
        <div className="brand-mark">
          <img alt="Graph-PRefLexOR" src={logoUrl} />
        </div>
        <div className="brand-copy">
          <strong>Graph-PRefLexOR Explorer</strong>
          <span>
            {graph
              ? `${graph.name || "graph"} | ${formatNumber(graph.stats.nodes)} nodes, ${formatNumber(graph.stats.edges)} edges`
              : "No graph loaded"}
          </span>
        </div>
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
          Canvas
          <select
            value={visual.canvasTheme}
            onChange={(event) => setVisual({ canvasTheme: event.target.value as typeof visual.canvasTheme })}
          >
            <option value="dark">Black canvas</option>
            <option value="light">White canvas</option>
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
          2D edge style
          <select
            value={visual.edgeStyle}
            onChange={(event) => setVisual({ edgeStyle: event.target.value as typeof visual.edgeStyle })}
          >
            <option value="straight">Straight</option>
            <option value="directed">Directed arrows</option>
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
        <label>
          Edge width
          <input
            max="3"
            min="0.4"
            onChange={(event) => setVisual({ edgeWidth: Number(event.target.value) })}
            step="0.1"
            type="range"
            value={visual.edgeWidth}
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
      <div className="micro-help">
        Edge color is relation-based, edge opacity and width are controlled here, and highlighted paths override both.
        Directed arrows apply in the 2D renderer. 3D highlighted paths are rendered as curved tubes.
      </div>
    </Drawer>
  );
}

function SearchPanel({ defaultOpen = false }: { defaultOpen?: boolean }) {
  const results = useExplorerStore((state) => state.searchResults);
  const setSearchResults = useExplorerStore((state) => state.setSearchResults);
  const setGraph = useExplorerStore((state) => state.setGraph);
  const setSelectedNode = useExplorerStore((state) => state.setSelectedNode);
  const setSelectedNodes = useExplorerStore((state) => state.setSelectedNodes);
  const clearSelection = useExplorerStore((state) => state.clearSelection);
  const setHighlightedPaths = useExplorerStore((state) => state.setHighlightedPaths);
  const selectedNodes = useExplorerStore((state) => state.selectedNodes);
  const graph = useExplorerStore((state) => state.graph);
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState(false);
  const [selectionBusy, setSelectionBusy] = useState(false);
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

  function selectVisibleNodes() {
    if (!graph) return;
    const ids = graph.nodes.map((node) => node.id);
    setSelectedNodes(ids);
    setHighlightedPaths([]);
    setStatus(`Selected ${formatNumber(ids.length)} visible nodes.`);
  }

  async function selectLoadedGraph() {
    if (!graph) return;
    setSelectionBusy(true);
    try {
      const fullGraph = await api.graph();
      const ids = fullGraph.nodes.map((node) => node.id);
      setGraph(fullGraph);
      setSelectedNodes(ids);
      setHighlightedPaths([]);
      setStatus(`Selected the full loaded graph: ${formatNumber(ids.length)} nodes.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setSelectionBusy(false);
    }
  }

  function invertVisibleSelection() {
    if (!graph) return;
    const selected = new Set(selectedNodes);
    const ids = graph.nodes.map((node) => node.id).filter((id) => !selected.has(id));
    setSelectedNodes(ids);
    setHighlightedPaths([]);
    setStatus(`Inverted visible selection: ${formatNumber(ids.length)} nodes selected.`);
  }

  function clearNodeSelection() {
    clearSelection();
    setHighlightedPaths([]);
    setStatus("Selection cleared.");
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
      <div className="mini-action-row" aria-label="Selection actions">
        <button disabled={!graph} onClick={selectVisibleNodes} title="Select every node currently visible in the viewer." type="button">
          <CircleCheck size={12} />
          Visible
        </button>
        <button disabled={!graph || selectionBusy} onClick={selectLoadedGraph} title="Restore the full loaded graph and select every node." type="button">
          {selectionBusy ? <Loader2 className="spin" size={12} /> : <Network size={12} />}
          All
        </button>
        <button disabled={!graph} onClick={invertVisibleSelection} title="Invert the selection across the currently visible nodes." type="button">
          <RotateCcw size={12} />
          Invert
        </button>
        <button disabled={!selectedNodes.length} onClick={clearNodeSelection} title="Clear the selected node set." type="button">
          <X size={12} />
          Clear
        </button>
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
  const sigmaGraphRef = useRef<Graph | null>(null);
  const appendSelectionRef = useRef(false);
  const draggedNodeRef = useRef<string | null>(null);
  const selectedNodesRef = useRef<string[]>([]);
  const graph = useExplorerStore((state) => state.graph);
  const visual = useExplorerStore((state) => state.visual);
  const selectedNodes = useExplorerStore((state) => state.selectedNodes);
  const highlightedPaths = useExplorerStore((state) => state.highlightedPaths);
  const setSelectedNode = useExplorerStore((state) => state.setSelectedNode);
  const [hoverNode, setHoverNode] = useState<GraphNode | null>(null);

  useEffect(() => {
    selectedNodesRef.current = selectedNodes;
  }, [selectedNodes]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || !graph) return undefined;
    rendererRef.current?.kill();
    const updateAppendSelection = (event: KeyboardEvent | PointerEvent | MouseEvent) => {
      appendSelectionRef.current = event.shiftKey;
    };
    window.addEventListener("keydown", updateAppendSelection);
    window.addEventListener("keyup", updateAppendSelection);
    container.addEventListener("pointerdown", updateAppendSelection, true);

    const sigmaGraph = new Graph();
    sigmaGraphRef.current = sigmaGraph;
    const colorRange = metricRange(graph.nodes, visual.colorBy);
    const selected = new Set(selectedNodesRef.current);
    const highlightedNodes = pathNodeSet(highlightedPaths);
    const highlightedEdges = pathEdgeSet(highlightedPaths);
    const darkCanvas = visual.canvasTheme === "dark";
    const edgeOpacity = darkCanvas ? Math.max(visual.edgeOpacity, 0.16) : visual.edgeOpacity;
    const selectedColor = darkCanvas ? "#ffffff" : "#133d35";
    const selectedBorder = darkCanvas ? "#4df0ab" : "#2f8d68";
    const highlightedColor = darkCanvas ? "#ffd166" : "#b56b16";

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
        color: selected.has(node.id) ? selectedColor : isHighlighted ? highlightedColor : nodeColor,
        borderColor: selected.has(node.id) ? selectedBorder : isHighlighted ? "#ffef9f" : nodeColor,
      });
    }

    const maxEdges = graph.edges.length > 45000 ? 45000 : graph.edges.length;
    for (let i = 0; i < maxEdges; i++) {
      const edge = graph.edges[i];
      if (sigmaGraph.hasNode(edge.source) && sigmaGraph.hasNode(edge.target) && !sigmaGraph.hasEdge(edge.id)) {
        const isHighlighted = highlightedEdges.has(edgeKey(edge.source, edge.target));
        const denseSize = graph.edges.length > 12000 ? 0.35 : 0.55;
        sigmaGraph.addEdgeWithKey(edge.id, edge.source, edge.target, {
          type: visual.edgeStyle === "directed" ? "arrow" : "line",
          size: (isHighlighted ? 2.4 : denseSize) * visual.edgeWidth,
          color: isHighlighted ? "rgba(255,209,102,0.92)" : edgeColor(edge, edgeOpacity),
        });
      }
    }

    const renderer = new Sigma(sigmaGraph, container, {
      renderLabels: false,
      renderEdgeLabels: false,
      defaultEdgeType: visual.edgeStyle === "directed" ? "arrow" : "line",
      defaultEdgeColor: darkCanvas ? "rgba(180,196,215,0.18)" : "rgba(94,112,132,0.12)",
      edgeProgramClasses: {
        line: EdgeLineProgram,
        arrow: EdgeArrowProgram,
      },
      allowInvalidContainer: true,
      labelRenderedSizeThreshold: 12,
      minCameraRatio: 0.05,
      maxCameraRatio: 12,
    });
    const mouseCaptor = renderer.getMouseCaptor();
    const stopDragging = () => {
      draggedNodeRef.current = null;
      container.classList.remove("dragging-node");
    };
    const dragNode = (event: { x: number; y: number; preventSigmaDefault?: () => void }) => {
      const node = draggedNodeRef.current;
      if (!node) return;
      event.preventSigmaDefault?.();
      const pos = renderer.viewportToGraph({ x: event.x, y: event.y });
      sigmaGraph.setNodeAttribute(node, "x", pos.x);
      sigmaGraph.setNodeAttribute(node, "y", pos.y);
      renderer.refresh({ partialGraph: { nodes: [node] }, skipIndexation: true });
    };
    mouseCaptor.on("mousemovebody", dragNode);
    mouseCaptor.on("mouseup", stopDragging);
    mouseCaptor.on("mouseleave", stopDragging);
    renderer.on("downNode", (payload) => {
      const eventPayload = payload as typeof payload & {
        event?: { original?: MouseEvent; originalEvent?: MouseEvent; shiftKey?: boolean };
      };
      const sourceEvent = eventPayload.event?.original || eventPayload.event?.originalEvent || eventPayload.event;
      draggedNodeRef.current = payload.node;
      container.classList.add("dragging-node");
      payload.preventSigmaDefault();
      setSelectedNode(payload.node, Boolean(sourceEvent?.shiftKey || appendSelectionRef.current));
    });
    renderer.on("clickNode", (payload) => {
      const eventPayload = payload as typeof payload & {
        event?: { original?: MouseEvent; originalEvent?: MouseEvent; shiftKey?: boolean };
      };
      const sourceEvent = eventPayload.event?.original || eventPayload.event?.originalEvent || eventPayload.event;
      setSelectedNode(payload.node, Boolean(sourceEvent?.shiftKey || appendSelectionRef.current));
    });
    renderer.on("enterNode", ({ node }) => setHoverNode(graph.nodes.find((item) => item.id === node) || null));
    renderer.on("leaveNode", () => setHoverNode(null));
    rendererRef.current = renderer;

    return () => {
      window.removeEventListener("keydown", updateAppendSelection);
      window.removeEventListener("keyup", updateAppendSelection);
      container.removeEventListener("pointerdown", updateAppendSelection, true);
      mouseCaptor.off("mousemovebody", dragNode);
      mouseCaptor.off("mouseup", stopDragging);
      mouseCaptor.off("mouseleave", stopDragging);
      draggedNodeRef.current = null;
      sigmaGraphRef.current = null;
      renderer.kill();
    };
  }, [graph, highlightedPaths, setSelectedNode, visual]);

  useEffect(() => {
    const sigmaGraph = sigmaGraphRef.current;
    const renderer = rendererRef.current;
    if (!sigmaGraph || !renderer || !graph) return;
    const selected = new Set(selectedNodes);
    const highlightedNodes = pathNodeSet(highlightedPaths);
    const colorRange = metricRange(graph.nodes, visual.colorBy);
    const darkCanvas = visual.canvasTheme === "dark";
    const selectedColor = darkCanvas ? "#ffffff" : "#133d35";
    const selectedBorder = darkCanvas ? "#4df0ab" : "#2f8d68";
    const highlightedColor = darkCanvas ? "#ffd166" : "#b56b16";
    const changedNodes: string[] = [];
    for (const node of graph.nodes) {
      if (!sigmaGraph.hasNode(node.id)) continue;
      const isHighlighted = highlightedNodes.has(node.id);
      const nodeColor =
        visual.colorBy === "component" || visual.colorBy === "community"
          ? paletteCategoryColor(nodeMetric(node, visual.colorBy), visual)
          : colorScale(nodeMetric(node, visual.colorBy), colorRange, visual.colorPalette);
      sigmaGraph.mergeNodeAttributes(node.id, {
        size: selected.has(node.id) || isHighlighted ? nodeSize(node, graph, visual) * 1.9 : nodeSize(node, graph, visual),
        color: selected.has(node.id) ? selectedColor : isHighlighted ? highlightedColor : nodeColor,
        borderColor: selected.has(node.id) ? selectedBorder : isHighlighted ? "#ffef9f" : nodeColor,
      });
      changedNodes.push(node.id);
    }
    if (changedNodes.length) renderer.refresh({ partialGraph: { nodes: changedNodes }, skipIndexation: true });
  }, [graph, highlightedPaths, selectedNodes, visual]);

  const selectedLabel = useMemo(() => {
    if (!graph || !selectedNodes.length) return "No selection";
    const names = selectedNodes.slice(0, 2).map((id) => graph.nodes.find((node) => node.id === id)?.label || id);
    return `Selected: ${names.join(", ")}${selectedNodes.length > 2 ? ` +${selectedNodes.length - 2}` : ""}`;
  }, [graph, selectedNodes]);

  return (
    <section className={cx("graph-shell", visual.canvasTheme === "dark" ? "graph-shell-dark" : "graph-shell-light")}>
      <div className="graph-top-dock">
        <div className="graph-overlay">{contextSummary(graph, selectedNodes)}</div>
        <div className="graph-overlay">{selectedLabel}</div>
      </div>
      <div className="graph-canvas" ref={containerRef} />
      <div className="graph-bottom-dock">
        <GraphIterationStepper />
      </div>
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

    const darkCanvas = visual.canvasTheme === "dark";
    const background = darkCanvas ? "#020407" : "#f8f7f2";
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(background);
    scene.fog = new THREE.FogExp2(background, darkCanvas ? 0.0014 : 0.0019);

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
        selected.has(node.id) ? (darkCanvas ? "#ffffff" : "#163c3a") : highlightedNodes.has(node.id) ? "#f4b24f" : nodeColor,
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
        new THREE.LineBasicMaterial({
          vertexColors: true,
          transparent: true,
          opacity: darkCanvas ? Math.max(0.2, visual.edgeOpacity * 1.5) : Math.max(0.18, visual.edgeOpacity * 1.35),
          linewidth: visual.edgeWidth,
        }),
      ),
    );

    const pathMaterial = new THREE.MeshBasicMaterial({ color: "#c77b1f", transparent: true, opacity: 0.9 });
    for (const path of highlightedPaths.slice(0, 18)) {
      const points = path.map((id) => coords.get(id)).filter(Boolean) as THREE.Vector3[];
      if (points.length < 2) continue;
      const curve = new THREE.CatmullRomCurve3(points);
      graphGroup.add(new THREE.Mesh(new THREE.TubeGeometry(curve, Math.max(8, points.length * 8), 0.42 * visual.edgeWidth, 8, false), pathMaterial));
    }

    const sphereGeometry = new THREE.SphereGeometry(1.8, 16, 12);
    const anchorMaterial = new THREE.MeshBasicMaterial({ color: darkCanvas ? "#ffffff" : "#163c3a" });
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
    <section className={cx("graph-shell", visual.canvasTheme === "dark" ? "graph-shell-dark" : "graph-shell-light")}>
      <div className="graph-top-dock">
        <div className="graph-overlay">
          {contextSummary(graph, selectedNodes)}
          {highlightedPaths.length ? ` | ${highlightedPaths.length} paths highlighted` : ""}
        </div>
        <div className="graph-overlay">{selectedLabel}</div>
      </div>
      <div className="graph-canvas" ref={containerRef} />
      <div className="graph-bottom-dock">
        <GraphIterationStepper />
      </div>
    </section>
  );
}

function GraphCanvas() {
  const viewMode = useExplorerStore((state) => state.visual.viewMode);
  return viewMode === "3d" ? <ThreeGraphCanvas /> : <SigmaGraphCanvas />;
}

function GraphIterationStepper() {
  const graph = useExplorerStore((state) => state.graph);
  const setGraph = useExplorerStore((state) => state.setGraph);
  const runPath = useMemo(() => inferRunFromGraphPath(graph?.path || ""), [graph?.path]);
  const [loadingPath, setLoadingPath] = useState("");
  const [status, setStatus] = useState("");
  const snapshotsQuery = useQuery({
    queryKey: ["graph-artifact-run-graphs", runPath],
    queryFn: () => api.runGraphs(runPath),
    enabled: Boolean(runPath),
    refetchInterval: 6000,
  });
  const orderedSnapshots = useMemo(() => {
    const items = [...(snapshotsQuery.data?.graphs || [])];
    return items.sort((a, b) => {
      const aIter = a.iter == null ? Number.POSITIVE_INFINITY : a.iter;
      const bIter = b.iter == null ? Number.POSITIVE_INFINITY : b.iter;
      if (aIter !== bIter) return aIter - bIter;
      return (a.updated_at || 0) - (b.updated_at || 0);
    });
  }, [snapshotsQuery.data?.graphs]);
  const currentSnapshotIndex = useMemo(() => {
    if (!orderedSnapshots.length || !graph?.path) return -1;
    return orderedSnapshots.findIndex((snapshot) => {
      const absolute = snapshot.absolute_path || "";
      const relative = snapshot.path || "";
      return graph.path === absolute || (relative && graph.path.endsWith(relative));
    });
  }, [graph?.path, orderedSnapshots]);
  const latestSnapshotIndex = orderedSnapshots.findIndex((snapshot) => snapshot.is_latest);
  const selectedSnapshotIndex =
    currentSnapshotIndex >= 0
      ? currentSnapshotIndex
      : latestSnapshotIndex >= 0
        ? latestSnapshotIndex
        : orderedSnapshots.length
          ? orderedSnapshots.length - 1
          : -1;
  const selectedSnapshot = selectedSnapshotIndex >= 0 ? orderedSnapshots[selectedSnapshotIndex] : null;

  async function loadSnapshotAt(index: number) {
    const snapshot = orderedSnapshots[index];
    const path = snapshot ? snapshotPath(snapshot) : "";
    if (!path) return;
    setLoadingPath(path);
    setStatus("");
    try {
      setGraph(await api.loadRun(path));
      setStatus(`Loaded ${snapshotLabel(snapshot)}.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setLoadingPath("");
    }
  }

  if (!runPath) return null;

  const busy = Boolean(loadingPath);
  const canStep = orderedSnapshots.length > 0 && !busy;

  return (
    <div className={cx("iteration-navigator", "graph-iteration", runPath && "active")}>
      <div className="iteration-stepper">
        <IconButton
          disabled={!canStep || selectedSnapshotIndex <= 0}
          description="Load the previous generated graph snapshot."
          icon={<ChevronLeft size={14} />}
          label="Previous"
          onClick={() => void loadSnapshotAt(selectedSnapshotIndex - 1)}
        />
        <label className="iteration-select">
          <span>
            {orderedSnapshots.length
              ? `Iterations ${selectedSnapshotIndex + 1}/${orderedSnapshots.length}`
              : snapshotsQuery.isFetching
                ? "Scanning iterations..."
                : "No iterations"}
          </span>
          <select
            disabled={!canStep}
            onChange={(event) => void loadSnapshotAt(Number(event.target.value))}
            value={selectedSnapshotIndex >= 0 ? String(selectedSnapshotIndex) : ""}
          >
            <option value="">Choose iteration...</option>
            {orderedSnapshots.map((snapshot, index) => (
              <option key={snapshotPath(snapshot) || index} value={index}>
                {snapshotLabel(snapshot)}
                {snapshot.is_latest ? " (latest)" : ""}
              </option>
            ))}
          </select>
        </label>
        <IconButton
          disabled={!canStep || selectedSnapshotIndex < 0 || selectedSnapshotIndex >= orderedSnapshots.length - 1}
          description="Load the next generated graph snapshot."
          icon={<ChevronRight size={14} />}
          label="Next"
          onClick={() => void loadSnapshotAt(selectedSnapshotIndex + 1)}
        />
      </div>
      <span className="iteration-meta">
        {status ||
          (selectedSnapshot
            ? `${snapshotLabel(selectedSnapshot)}${selectedSnapshot.is_latest ? " | latest" : ""}`
            : "No snapshot selected")}
      </span>
      {snapshotsQuery.isFetching ? <Loader2 className="spin iteration-loader" size={13} /> : null}
      {snapshotsQuery.isError ? <span className="iteration-error">{String(snapshotsQuery.error)}</span> : null}
    </div>
  );
}

function EmbeddingStatusBadge({
  status,
  onRebuild,
}: {
  status: EmbeddingStatus | null;
  onRebuild: () => void;
}) {
  const [open, setOpen] = useState(false);
  if (!status || status.status === "idle") return null;
  const percent = status.status === "done" ? 100 : Math.round((status.progress?.percent || 0) * 100);
  const label = status.ready
    ? `${status.cached ? "Cached " : ""}Semantic index ready: ${formatNumber(status.nodes)} nodes`
    : status.status === "failed"
      ? "Semantic index failed"
      : status.progress?.message || "Building semantic index";
  const detail = [
    status.model || "embedding model",
    status.dimension ? `${formatNumber(status.dimension)} dims` : "",
    status.nodes ? `${formatNumber(status.nodes)} nodes` : "",
    status.progress?.detail || "",
  ]
    .filter(Boolean)
    .join(" | ");
  return (
    <div className="embedding-badge-wrap">
      <button
        aria-expanded={open}
        className={cx("embedding-badge", status.status === "failed" && "failed", status.ready && "ready")}
        onClick={() => setOpen((value) => !value)}
        title={`${label}. Click for semantic index details and rebuild controls.`}
        type="button"
      >
        {status.ready ? (
          <CircleCheck size={15} />
        ) : status.status === "failed" ? (
          <CircleAlert size={15} />
        ) : (
          <Loader2 className="spin" size={15} />
        )}
        <span>Semantic index</span>
      </button>
      {open ? (
        <div className={cx("embedding-popover", status.status === "failed" && "failed", status.ready && "ready")} role="dialog">
          <div className="embedding-copy">
            <strong>{label}</strong>
            <span>{detail || "No semantic index details yet."}</span>
          </div>
          <div className="embedding-meter">
            <span>{percent}%</span>
            <progress max={100} value={percent} />
          </div>
          <div className="embedding-popover-actions">
            <button onClick={onRebuild} title="Force rebuild the semantic node embedding index." type="button">
              Rebuild
            </button>
            <button onClick={() => setOpen(false)} type="button">
              Close
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function safeUrl(url: string) {
  return /^https?:\/\//i.test(url) || /^mailto:/i.test(url) ? url : "#";
}

function trimInlineFence(value: string, left: string, right = left) {
  return value.slice(left.length, value.length - right.length);
}

function renderInlineMarkdown(text: string, keyPrefix: string) {
  const pattern = /(`[^`]*`|\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\$[^$\n]+\$|\*\*[^*]+\*\*|\[[^\]]+\]\((?:https?:\/\/|mailto:)[^)\s]+\))/g;
  const nodes: React.ReactNode[] = [];
  let last = 0;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(text))) {
    if (match.index > last) nodes.push(text.slice(last, match.index));
    const token = match[0];
    const key = `${keyPrefix}-${match.index}`;
    if (token.startsWith("`")) {
      nodes.push(<code key={key}>{trimInlineFence(token, "`")}</code>);
    } else if (token.startsWith("$$")) {
      nodes.push(<span className="math-inline" key={key}>{trimInlineFence(token, "$$")}</span>);
    } else if (token.startsWith("\\[")) {
      nodes.push(<span className="math-inline" key={key}>{trimInlineFence(token, "\\[", "\\]")}</span>);
    } else if (token.startsWith("\\(")) {
      nodes.push(<span className="math-inline" key={key}>{trimInlineFence(token, "\\(", "\\)")}</span>);
    } else if (token.startsWith("$")) {
      nodes.push(<span className="math-inline" key={key}>{trimInlineFence(token, "$")}</span>);
    } else if (token.startsWith("**")) {
      nodes.push(<strong key={key}>{renderInlineMarkdown(trimInlineFence(token, "**"), `${key}-strong`)}</strong>);
    } else {
      const link = token.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
      if (link) {
        nodes.push(
          <a href={safeUrl(link[2])} key={key} rel="noreferrer" target="_blank">
            {link[1]}
          </a>,
        );
      } else {
        nodes.push(token);
      }
    }
    last = pattern.lastIndex;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes;
}

function isMarkdownBlockStart(line: string) {
  const trimmed = line.trim();
  return (
    !trimmed ||
    trimmed.startsWith("```") ||
    trimmed.startsWith(">") ||
    trimmed.startsWith("$$") ||
    trimmed.startsWith("\\[") ||
    /^#{1,4}\s+/.test(trimmed) ||
    /^[-*]\s+/.test(trimmed) ||
    /^\d+\.\s+/.test(trimmed)
  );
}

function parseMarkdownTable(lines: string[]) {
  return lines.map((line) =>
    line
      .trim()
      .replace(/^\|/, "")
      .replace(/\|$/, "")
      .split("|")
      .map((cell) => cell.trim()),
  );
}

function MarkdownMessage({ content }: { content: string }) {
  const lines = content.replace(/\r\n/g, "\n").split("\n");
  const blocks: React.ReactNode[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();
    const key = `md-${i}`;
    if (!trimmed) {
      i += 1;
      continue;
    }
    if (trimmed.startsWith("```")) {
      const language = trimmed.slice(3).trim();
      const code: string[] = [];
      i += 1;
      while (i < lines.length && !lines[i].trim().startsWith("```")) {
        code.push(lines[i]);
        i += 1;
      }
      i += 1;
      blocks.push(
        <pre className="chat-code-block" key={key}>
          {language ? <span>{language}</span> : null}
          <code>{code.join("\n")}</code>
        </pre>,
      );
      continue;
    }
    if (trimmed.startsWith("$$") || trimmed.startsWith("\\[")) {
      const close = trimmed.startsWith("$$") ? "$$" : "\\]";
      const open = trimmed.startsWith("$$") ? "$$" : "\\[";
      const math = [trimmed.replace(open, "")];
      i += 1;
      while (i < lines.length && !lines[i].trim().endsWith(close)) {
        math.push(lines[i]);
        i += 1;
      }
      if (i < lines.length) {
        math.push(lines[i].trim().replace(close, ""));
        i += 1;
      }
      blocks.push(<div className="math-block" key={key}>{math.join("\n").trim()}</div>);
      continue;
    }
    if (/^#{1,4}\s+/.test(trimmed)) {
      const level = Math.min(4, trimmed.match(/^#+/)?.[0].length || 2);
      const text = trimmed.replace(/^#{1,4}\s+/, "");
      if (level === 1) blocks.push(<h1 key={key}>{renderInlineMarkdown(text, key)}</h1>);
      else if (level === 2) blocks.push(<h2 key={key}>{renderInlineMarkdown(text, key)}</h2>);
      else if (level === 3) blocks.push(<h3 key={key}>{renderInlineMarkdown(text, key)}</h3>);
      else blocks.push(<h4 key={key}>{renderInlineMarkdown(text, key)}</h4>);
      i += 1;
      continue;
    }
    if (trimmed.startsWith(">")) {
      const quote: string[] = [];
      while (i < lines.length && lines[i].trim().startsWith(">")) {
        quote.push(lines[i].trim().replace(/^>\s?/, ""));
        i += 1;
      }
      blocks.push(<blockquote key={key}>{renderInlineMarkdown(quote.join(" "), key)}</blockquote>);
      continue;
    }
    if (/^[-*]\s+/.test(trimmed)) {
      const items: string[] = [];
      while (i < lines.length && /^[-*]\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^[-*]\s+/, ""));
        i += 1;
      }
      blocks.push(
        <ul key={key}>
          {items.map((item, index) => (
            <li key={`${key}-${index}`}>{renderInlineMarkdown(item, `${key}-${index}`)}</li>
          ))}
        </ul>,
      );
      continue;
    }
    if (/^\d+\.\s+/.test(trimmed)) {
      const items: string[] = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^\d+\.\s+/, ""));
        i += 1;
      }
      blocks.push(
        <ol key={key}>
          {items.map((item, index) => (
            <li key={`${key}-${index}`}>{renderInlineMarkdown(item, `${key}-${index}`)}</li>
          ))}
        </ol>,
      );
      continue;
    }
    if (line.includes("|") && i + 1 < lines.length && /^\s*\|?\s*:?-{3,}/.test(lines[i + 1])) {
      const tableLines = [line];
      i += 2;
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) {
        tableLines.push(lines[i]);
        i += 1;
      }
      const [head, ...body] = parseMarkdownTable(tableLines);
      blocks.push(
        <table key={key}>
          <thead>
            <tr>{head.map((cell, index) => <th key={`${key}-h-${index}`}>{renderInlineMarkdown(cell, `${key}-h-${index}`)}</th>)}</tr>
          </thead>
          <tbody>
            {body.map((row, rowIndex) => (
              <tr key={`${key}-r-${rowIndex}`}>
                {row.map((cell, cellIndex) => <td key={`${key}-r-${rowIndex}-${cellIndex}`}>{renderInlineMarkdown(cell, `${key}-r-${rowIndex}-${cellIndex}`)}</td>)}
              </tr>
            ))}
          </tbody>
        </table>,
      );
      continue;
    }
    const paragraph = [trimmed];
    i += 1;
    while (i < lines.length && !isMarkdownBlockStart(lines[i])) {
      paragraph.push(lines[i].trim());
      i += 1;
    }
    blocks.push(<p key={key}>{renderInlineMarkdown(paragraph.join(" "), key)}</p>);
  }
  return <div className="chat-markdown">{blocks.length ? blocks : <p>{content}</p>}</div>;
}

function ReportPreviewModal({
  report,
  contextParts,
  onClose,
  onOpenReports,
}: {
  report: SessionReport;
  contextParts: ReportContextPart[];
  onClose: () => void;
  onOpenReports: () => void;
}) {
  const reportQuery = useQuery({
    queryKey: ["profile-report", report.out],
    queryFn: () => api.profileReport(report.out),
    enabled: Boolean(report.out),
    refetchInterval: (query) => (query.state.data?.artifacts.ready ? false : 5000),
  });
  const artifacts = reportQuery.data?.artifacts;
  const summary = artifacts?.summary || {};
  const includeReport = contextParts.includes("report");
  const includeProfile = contextParts.includes("profile");
  const profileJson = reportQuery.data?.profile_json || "";
  const profileJsonPreview =
    profileJson.length > 50000 ? `${profileJson.slice(0, 50000)}\n\n... truncated in preview ...` : profileJson;
  return (
    <div className="model-modal-backdrop report-preview-backdrop" role="presentation">
      <div aria-label="Attached report preview" aria-modal="true" className="serve-modal report-preview-modal" role="dialog">
        <div className="serve-modal-head report-preview-head">
          <div>
            <span>Attached report context</span>
            <h3>{summary.topic || report.label}</h3>
            <p>{report.out}</p>
          </div>
          <button aria-label="Close report preview" onClick={onClose} type="button">
            <X size={15} />
          </button>
        </div>
        <div className="report-preview-actions">
          <button onClick={() => void reportQuery.refetch()} type="button">
            Refresh
          </button>
          <button
            onClick={() => {
              onClose();
              onOpenReports();
            }}
            type="button"
          >
            Open in Reports
          </button>
          {artifacts?.pdf_path ? (
            <a href={api.reportAssetUrl(report.out, "report.pdf")} rel="noreferrer" target="_blank">
              <Download size={13} /> PDF
            </a>
          ) : null}
        </div>
        <div className="report-preview-context">
          Selected context parts: <strong>{reportPartsLabel(contextParts)}</strong>. They are attached as plain prompt
          context only when you send a chat request with this profile selected.
        </div>
        <div className="report-preview-body">
          {reportQuery.isLoading ? <div className="status-box">Loading report...</div> : null}
          {reportQuery.error ? <div className="status-box">{String(reportQuery.error)}</div> : null}
          {includeReport ? (
            reportQuery.data?.markdown ? (
              <section>
                <h4>report.md</h4>
                <MarkdownReport markdown={reportQuery.data.markdown} out={report.out} />
              </section>
            ) : (
              <div className="status-box">
                {artifacts?.ready ? "Report markdown is empty or unavailable." : "Report job is still preparing artifacts."}
              </div>
            )
          ) : null}
          {includeProfile ? (
            profileJson ? (
              <section>
                <h4>profile.json</h4>
                <pre className="report-json-preview">{profileJsonPreview}</pre>
              </section>
            ) : (
              <div className="status-box">
                {artifacts?.ready ? "profile.json is empty or unavailable." : "Profile job is still preparing artifacts."}
              </div>
            )
          ) : null}
        </div>
      </div>
    </div>
  );
}

function ChatContextInfoModal({
  body,
  selectedLabels,
  onClose,
}: {
  body: Parameters<typeof api.chatContextPreview>[0];
  selectedLabels: string[];
  onClose: () => void;
}) {
  const previewQuery = useQuery({
    queryKey: ["chat-context-preview", body],
    queryFn: () => api.chatContextPreview(body),
  });
  const preview = previewQuery.data;
  return (
    <div className="model-modal-backdrop chat-context-backdrop" role="presentation">
      <div aria-label="LLM context preview" aria-modal="true" className="serve-modal chat-context-modal" role="dialog">
        <div className="serve-modal-head report-preview-head">
          <div>
            <span>LLM transmission preview</span>
            <h3>Graph Assistant context</h3>
            <p>Shows the payload built for the next chat turn before the model is called.</p>
          </div>
          <button aria-label="Close LLM context preview" onClick={onClose} type="button">
            <X size={15} />
          </button>
        </div>
        <div className="chat-context-summary">
          <div>
            <span>Backend</span>
            <strong>{preview?.backend || normalizedBackend(body.model_config)}</strong>
          </div>
          <div>
            <span>State</span>
            <strong>{preview?.state_mode || (body.previous_response_id ? "responses_previous_response_id" : "ready_for_multiturn")}</strong>
          </div>
          <div>
            <span>Selection</span>
            <strong>{selectedLabels.length ? `${selectedLabels.length} nodes` : "none"}</strong>
          </div>
          <div>
            <span>Report</span>
            <strong>
              {body.report_context
                ? reportPartsLabel([
                    ...(body.report_context.include_report === false ? [] : (["report"] as ReportContextPart[])),
                    ...(body.report_context.include_profile ? (["profile"] as ReportContextPart[]) : []),
                  ])
                : "none"}
            </strong>
          </div>
        </div>
        {selectedLabels.length ? (
          <div className="chat-context-chips">
            {selectedLabels.slice(0, 18).map((label) => (
              <span key={label}>{label}</span>
            ))}
          </div>
        ) : null}
        <div className="report-preview-context">
          Selection is transmitted as node IDs in <code>selected_nodes</code>. The backend expands those IDs into a graph
          context packet, adds the Graph-PRefLexOR developer/system instruction, and then sends the messages below.
        </div>
        <div className="chat-context-body">
          {previewQuery.isLoading ? <div className="status-box">Building preview...</div> : null}
          {previewQuery.error ? <div className="status-box">{String(previewQuery.error)}</div> : null}
          {preview ? (
            <>
              <section>
                <h4>Request Body</h4>
                <pre>{JSON.stringify(preview.request, null, 2)}</pre>
              </section>
              <section>
                <h4>Messages Sent To Model</h4>
                <pre>{JSON.stringify(preview.messages, null, 2)}</pre>
              </section>
              {preview.fallback_messages.length ? (
                <section>
                  <h4>Fallback Messages</h4>
                  <pre>{JSON.stringify(preview.fallback_messages, null, 2)}</pre>
                </section>
              ) : null}
            </>
          ) : null}
        </div>
      </div>
    </div>
  );
}

function ChatPanel({
  sessionReports,
  onOpenRuns,
  onOpenReports,
}: {
  sessionReports: SessionReport[];
  onOpenRuns: () => void;
  onOpenReports: () => void;
}) {
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
  const [agentMode, setAgentMode] = useState<ChatContextMode>("none");
  const [contextQuery, setContextQuery] = useState("");
  const [contextNodes, setContextNodes] = useState(220);
  const [selectedReportOut, setSelectedReportOut] = useState("");
  const [reportContextParts, setReportContextParts] = useState<ReportContextPart[]>(["report"]);
  const [reportMenuOpen, setReportMenuOpen] = useState(false);
  const [reportPreviewOpen, setReportPreviewOpen] = useState(false);
  const [contextInfoOpen, setContextInfoOpen] = useState(false);
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
  const activeRun = inferRunFromGraphPath(graph?.path || "");
  const activeChatRole = chatModelRole(roles);
  const chatRoleName = activeChatRole?.role || "chat";
  const selectedReport = sessionReports.find((report) => report.out === selectedReportOut);
  const includeReportContext = reportContextParts.includes("report");
  const includeProfileContext = reportContextParts.includes("profile");
  const reportContextSelection =
    selectedReport && (includeReportContext || includeProfileContext)
      ? {
          out: selectedReport.out,
          max_chars: reportMaxChars,
          include_report: includeReportContext,
          include_profile: includeProfileContext,
        }
      : null;
  const selectedLabels = selectedNodes.map((id) => nodeLabel(graph, id));
  const commandQuery = question.startsWith("/") ? question.slice(1).trim().toLowerCase() : "";
  const commandItems = [
    { command: "/clear", label: "Clear chat", detail: "Reset this chat thread.", action: () => resetChat() },
    { command: "/followups", label: "Generate follow-ups", detail: "Ask the active graph model for next query ideas.", action: () => void suggestFollowups() },
    { command: "/synthesize <task>", label: "Run synthesis", detail: "Run synthesize.py on the active run and post answer.md here.", action: () => setQuestion("/synthesize ") },
    { command: "/none", label: "No retrieval", detail: "Use only selected nodes; with no selection this is regular chat.", action: () => setAgentMode("none") },
    { command: "/rag", label: "Graph-RAG retrieval", detail: "Retrieve broader semantic neighborhoods and path connectors.", action: () => setAgentMode("graph_rag") },
    { command: "/focus", label: "Focused selection", detail: "Use selected nodes, an optional focus query, and compact neighborhoods.", action: () => setAgentMode("focused") },
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
      setReportPreviewOpen(false);
    }
  }, [selectedReportOut, sessionReports]);

  function recentHistory() {
    return messages
      .filter((message) => message.role === "user" || message.role === "assistant")
      .slice(-8)
      .map((message) => ({ role: message.role as "user" | "assistant", content: message.content }));
  }

  function buildChatBody(role: ModelRole, text = question || "(next user message)") {
    return {
      question: text,
      selected_nodes: graph ? selectedNodes : [],
      query: contextQuery,
      depth: 1,
      max_nodes: contextNodes,
      max_edges: agentMode === "graph_rag" ? 520 : 160,
      context_mode: agentMode,
      report_context: reportContextSelection,
      model_config: role,
      history: recentHistory(),
      previous_response_id: previousResponseId(messages, role) || undefined,
    };
  }

  function toggleReportContextPart(part: ReportContextPart) {
    setReportContextParts((parts) => {
      if (parts.includes(part)) {
        return parts.length === 1 ? parts : parts.filter((item) => item !== part);
      }
      return [...parts, part];
    });
  }

  function addCommandHelp() {
    addChatMessage({
      role: "system",
      meta: "commands",
      content:
        "/clear resets the chat.\n/none disables graph retrieval and uses only selected nodes.\n/rag switches to Graph-RAG retrieval.\n/focus switches to focused selection context.\n/nodes 160 changes context size.\n/followups asks the chat model for next query ideas.\n/synthesize <task> runs synthesize.py on the active run and posts answer.md here.\n/synthesize --style hypotheses --mine \"Rank three falsifiable hypotheses\" changes style and mines insights if needed.\n/synthesize --hf meta-llama/Llama-3.2-3B-Instruct \"...\" uses the Hugging Face backend.\nUse Model Settings to change the default chat/synthesis model role.",
    });
  }

  async function pollSynthesis(jobId: string, pendingId: string) {
    let last = await api.synthesisJob(jobId);
    while (["running", "stopping"].includes(last.status)) {
      const progress = last.progress;
      updateChatMessage(pendingId, {
        content: `Synthesizing answer...\n\n${progress?.detail || "Waiting for synthesize.py output."}`,
        meta: `synthesize ${Math.round((progress?.percent || 0) * 100)}% | ${last.model || "model"}`,
      });
      await new Promise((resolve) => window.setTimeout(resolve, 1800));
      last = await api.synthesisJob(jobId);
    }
    if (last.status === "done") {
      updateChatMessage(pendingId, {
        content: `${last.answer_markdown || "(synthesis completed but answer.md was empty)"}\n\n---\n\nSaved to \`${last.out}\`.`,
        meta: `synthesize | ${last.model || "model"} | ${last.out}`,
      });
    } else {
      updateChatMessage(pendingId, {
        content: `${last.status}: synthesize.py did not complete.\n\n${last.log_tail || ""}`.trim(),
        meta: "synthesize error",
      });
    }
  }

  async function startSynthesis(rawArgs: string) {
    if (!activeRun) {
      addChatMessage({ role: "assistant", content: "Load a run folder before using `/synthesize`; the tool needs a run with graph and insight artifacts.", meta: "synthesize" });
      return;
    }
    const role = activeChatRole;
    const options = parseSynthesizeCommand(rawArgs);
    const backend = options.backend || role?.backend || "responses";
    const model = options.model || role?.model || "";
    if (!model) {
      addChatMessage({ role: "assistant", content: "Configure a chat model under Model Settings, or pass `--model` / `--hf <repo>` to `/synthesize`.", meta: "synthesize" });
      return;
    }
    const task = options.task;
    const taskLabel = task || `(style preset: ${options.style || "report"})`;
    addChatMessage({ role: "user", content: `/synthesize ${rawArgs}`.trim(), meta: activeRun });
    const pending = addChatMessage({
      role: "assistant",
      content: `Starting synthesize.py for \`${activeRun}\`...\n\nTask: ${taskLabel}`,
      meta: `synthesize | ${model}`,
    });
    try {
      const started = await api.synthesize({
        run: activeRun,
        task: task || undefined,
        style: options.style || "report",
        backend,
        model,
        base_url: options.base_url || role?.base_url,
        api_key_env: role?.api_key_env,
        model_config: role,
        temperature: options.temperature ?? role?.temperature ?? 0.7,
        max_tokens: options.max_tokens ?? role?.max_tokens ?? 8000,
        max_leads: options.max_leads ?? 8,
        mine: options.mine,
        no_insights: options.no_insights,
      });
      await pollSynthesis(started.id, pending);
    } catch (error) {
      updateChatMessage(pending, { content: error instanceof Error ? error.message : String(error), meta: "synthesize error" });
    }
  }

  function executeCommand(raw: string) {
    const value = raw.trim();
    if (!value.startsWith("/")) return false;
    const match = value.match(/^\/(\S+)\s*(.*)$/);
    const command = match?.[1] || "";
    const restText = match?.[2] || "";
    const rest = restText.split(/\s+/).filter(Boolean);
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
    if (command === "synthesize") {
      setQuestion("");
      void startSynthesis(restText);
      return true;
    }
    if (command === "rag") {
      setAgentMode("graph_rag");
      addChatMessage({ role: "system", content: "Switched to Graph-RAG agent mode.", meta: "command" });
      setQuestion("");
      return true;
    }
    if (command === "none") {
      setAgentMode("none");
      addChatMessage({ role: "system", content: "Switched to no retrieval. Selected nodes are included if present; otherwise this is regular chat.", meta: "command" });
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
    if (!question.trim()) return;
    if (executeCommand(question)) return;
    if (!graph && agentMode !== "none") {
      addChatMessage({ role: "assistant", content: "Load a graph before using Focused selection or Graph-RAG retrieval. Use Context: None for regular chat without graph context.", meta: "context" });
      return;
    }
    const role = activeChatRole;
    if (!role?.model) {
      addChatMessage({ role: "assistant", content: "Configure the chat model under Model Settings before asking the graph.", meta: "configuration" });
      return;
    }
    const priorResponseId = previousResponseId(messages, role);
    addChatMessage({ role: "user", content: question, meta: contextSummary(graph, selectedNodes) });
    const pending = addChatMessage({ role: "assistant", content: "Thinking...", meta: chatRoleName });
    setBusy(true);
    try {
      const res = await api.ask({ ...buildChatBody(role, question), previous_response_id: priorResponseId || undefined });
      const retrievedNodes = res.context.nodes || [];
      setLastRagNodes(agentMode === "graph_rag" ? retrievedNodes : []);
      if (agentMode === "graph_rag" && retrievedNodes.length) {
        setSearchResults(contextNodesToSearchResults(retrievedNodes));
      }
      updateChatMessage(pending, {
        content: res.answer || "(empty response)",
        meta: `${chatContextLabel((res.context.mode || agentMode) as ChatContextMode)} ${res.context.node_count}n/${res.context.edge_count}e${chatStateMeta(res)}${res.context.report_context ? ` | ${reportPartsLabel(reportPartsFromIncluded(res.context.report_context.included))} ${res.context.report_context.title}` : ""}`,
        ...responseStateMeta(role, res.response_id),
      });
      setQuestion("");
    } catch (error) {
      updateChatMessage(pending, { content: error instanceof Error ? error.message : String(error), meta: "error" });
    } finally {
      setBusy(false);
    }
  }

  async function suggestFollowups() {
    if (!graph && agentMode !== "none") return;
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
        selected_nodes: graph ? selectedNodes : [],
        query: contextQuery,
        depth: 1,
        max_nodes: Math.min(contextNodes, 120),
        max_edges: 160,
        context_mode: agentMode,
        report_context: reportContextSelection
          ? { ...reportContextSelection, max_chars: Math.min(reportMaxChars, 10000) }
          : null,
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
        <div className="chat-title-block">
          <h2><Sparkles size={15} /> Graph Assistant</h2>
          <span className="chat-context-line">{contextLabel} | {chatContextLabel(agentMode)} | chat model: {activeChatRole?.model || "not configured"}</span>
        </div>
        <div className="chat-actions">
          <button
            className="context-info-button"
            disabled={!activeChatRole?.model}
            onClick={() => setContextInfoOpen(true)}
            title="Preview the exact LLM context and transmission payload for the next chat turn."
            type="button"
          >
            <Info size={14} />
          </button>
          <span className="model-badge" title="Change this under Model Settings.">chat</span>
          <IconButton description="Clear the current browser-side chat thread." icon={<RotateCcw size={14} />} label="Reset" onClick={resetChat} />
        </div>
      </div>
      {contextInfoOpen && activeChatRole?.model ? (
        <ChatContextInfoModal body={buildChatBody(activeChatRole)} onClose={() => setContextInfoOpen(false)} selectedLabels={selectedLabels} />
      ) : null}
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
                <div className="chat-bubble">
                  <MarkdownMessage content={message.content} />
                </div>
              </div>
            </div>
          ))
        ) : (
          <div className="empty-chat">
            {graph ? (
              "Ask normally, select nodes for selected-only context, or switch to Focused / Graph-RAG when you want retrieval."
            ) : (
              <>
                <strong>No graph context loaded.</strong>
                <span>Context: None works as regular chat. Open Runs when you want graph-aware context from a run folder, GraphML snapshot, upload, or new ideation run.</span>
                <button onClick={onOpenRuns} type="button">Open Runs</button>
              </>
            )}
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
          <button
            className="report-context-chip"
            disabled={!selectedReport}
            onClick={() => {
              if (selectedReport) setReportPreviewOpen(true);
            }}
            title={selectedReport ? "Preview the attached report context before sending." : "Attach a report to preview it here."}
            type="button"
          >
            <FileText size={13} />
            <span>
              {selectedReport
                ? `Profile context: ${selectedReport.label} | ${reportPartsLabel(reportContextParts)}`
                : sessionReports.length
                  ? "No profile context attached"
                  : "No reports generated or opened in this session"}
            </span>
          </button>
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
              <span>Reports generated or opened during this browser session. Select the output folder and which artifacts to include in the next chat request.</span>
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
            <div className="report-context-options">
              <span>Artifacts to include</span>
              <label>
                <input
                  checked={includeReportContext}
                  onChange={() => toggleReportContextPart("report")}
                  type="checkbox"
                />
                report.md
              </label>
              <label>
                <input
                  checked={includeProfileContext}
                  onChange={() => toggleReportContextPart("profile")}
                  type="checkbox"
                />
                profile.json
              </label>
            </div>
            <label>
              Max context chars
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
        {selectedReport && reportPreviewOpen ? (
          <ReportPreviewModal
            onClose={() => setReportPreviewOpen(false)}
            onOpenReports={onOpenReports}
            contextParts={reportContextParts}
            report={selectedReport}
          />
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
                  if (!item.command.startsWith("/synthesize")) setQuestion("");
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
            if (event.key !== "Enter") return;
            if (event.shiftKey) return;
            event.preventDefault();
            void ask();
          }}
          placeholder={graph ? "Message the graph, or type / for commands..." : "Regular chat, or load a graph in Runs for graph context..."}
          rows={4}
          value={question}
        />
        <div className="composer-tools">
          <label className="agent-mode">
            <span>
              Context
              <HelpTip text="None is the default: it sends only explicitly selected nodes; with no selected nodes it is regular chat. Focused adds compact neighborhoods and an optional focus query. Graph-RAG retrieves semantic/text matches, neighborhoods, bridge paths, and central nodes." />
            </span>
            <select value={agentMode} onChange={(event) => setAgentMode(event.target.value as ChatContextMode)}>
              <option value="none">None</option>
              <option value="graph_rag">Graph-RAG retrieval</option>
              <option value="focused">Focused selection</option>
            </select>
          </label>
          <label className="context-query">
            <span>
              Focus query
              <HelpTip text="Optional search term used to pull matching nodes into the graph context packet. In Graph-RAG mode it is combined with your question for broader retrieval." />
            </span>
            <input
              disabled={agentMode === "none"}
              onChange={(event) => setContextQuery(event.target.value)}
              placeholder={agentMode === "none" ? "disabled in None mode" : "optional concept filter"}
              value={contextQuery}
            />
          </label>
          <label className="context-count">
            <span>
              Max nodes
              <HelpTip text="Maximum retrieved nodes sent with each chat request. Graph-RAG can use a larger budget; the backend still caps it to protect the browser and model context." />
            </span>
            <input
              disabled={agentMode === "none"}
              min={20}
              max={900}
              onChange={(event) => setContextNodes(Number(event.target.value))}
              type="number"
              value={contextNodes}
            />
          </label>
          <IconButton
            disabled={busy || (!graph && agentMode !== "none")}
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
    const priorResponseId = previousResponseId(messages, role);
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
        previous_response_id: priorResponseId || undefined,
      });
      applyContext(res.context.nodes || []);
      updateChatMessage(pending, {
        content: res.answer || "(empty response)",
        meta: `graph-RAG ${res.context.node_count}n/${res.context.edge_count}e${chatStateMeta(res)}`,
        ...responseStateMeta(role, res.response_id),
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
  const [active, setActive] = useState("generator");
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
          API mode
          <select value={role.backend || "responses"} onChange={(event) => patchRole({ backend: event.target.value })}>
            <option value="responses">OpenAI Responses (native state)</option>
            <option value="chat">Chat completions (history replay)</option>
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
        <label>
          Max output
          <input
            max={131072}
            min={256}
            step={512}
            type="number"
            value={role.max_tokens ?? ""}
            onChange={(event) => patchRole({ max_tokens: event.target.value ? Number(event.target.value) : "" })}
          />
        </label>
      </div>
      <div className="micro-help">
        Max output is the requested assistant answer budget. The backend requests a high limit by default and retries with lower limits if the selected provider rejects the cap.
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
  const messages = useExplorerStore((state) => state.chatMessages);
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
    const priorResponseId = previousResponseId(messages, role);
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
        previous_response_id: priorResponseId || undefined,
      });
      updateChatMessage(pending, {
        content: res.answer || "(empty response)",
        meta: `path agent | ${res.context.node_count}n/${res.context.edge_count}e${chatStateMeta(res)}`,
        ...responseStateMeta(role, res.response_id),
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
  onOpenGraph,
  onOpenSearch,
  onOpenRuns,
  onOpenReports,
  onOpenModels,
}: {
  onOpenGraph: () => void;
  onOpenSearch: () => void;
  onOpenRuns: () => void;
  onOpenReports: () => void;
  onOpenModels: () => void;
}) {
  const graph = useExplorerStore((state) => state.graph);
  const selectedNodes = useExplorerStore((state) => state.selectedNodes);
  return (
    <>
      <SidebarHeader title="Home" subtitle="session status; tools live in the left rail" />
      <section className="panel-card">
        <div className="artifact-card">
          <div>
            <strong>{graph?.name || "No graph loaded"}</strong>
            <span>{contextSummary(graph, selectedNodes)}</span>
          </div>
          <Network size={16} />
        </div>
      </section>
      <section className="panel-card">
        <div className="home-actions">
          <button onClick={onOpenRuns} type="button">
            <FolderOpen size={14} />
            <div>
              <strong>Load or launch runs</strong>
              <span>Run folders, GraphML files, snapshots, and live jobs.</span>
            </div>
          </button>
          <button disabled={!graph} onClick={onOpenGraph} type="button">
            <Network size={14} />
            <div>
              <strong>Open graph display</strong>
              <span>Visual layout, color, size, and graph statistics.</span>
            </div>
          </button>
          <button disabled={!graph} onClick={onOpenSearch} type="button">
            <Search size={14} />
            <div>
              <strong>Search and focus</strong>
              <span>Find nodes, select concepts, and compute bridge paths.</span>
            </div>
          </button>
          <button onClick={onOpenReports} type="button">
            <FileText size={14} />
            <div>
              <strong>Report Studio</strong>
              <span>Generate, view, and attach graph reports to chat.</span>
            </div>
          </button>
          <button onClick={onOpenModels} type="button">
            <Settings2 size={14} />
            <div>
              <strong>Model Settings</strong>
              <span>Configure chat, questioner, local, and OpenAI roles.</span>
            </div>
          </button>
        </div>
      </section>
    </>
  );
}

function SideRail({
  activeMode,
  onLoadRun,
  onRunGraphReady,
  onRunStart,
  onReportReady,
  onModeChange,
}: {
  activeMode: WorkspaceMode;
  onLoadRun: (run: string) => Promise<void>;
  onRunGraphReady: (run: string) => Promise<void>;
  onRunStart: () => Promise<void>;
  onReportReady: (out: string) => void;
  onModeChange: (mode: WorkspaceMode) => void;
}) {
  return (
    <aside className="side-panel">
      {activeMode === "chat" ? (
        <ChatSidebar
          onOpenGraph={() => onModeChange("graph")}
          onOpenModels={() => onModeChange("models")}
          onOpenReports={() => onModeChange("reports")}
          onOpenRuns={() => onModeChange("runs")}
          onOpenSearch={() => onModeChange("search")}
        />
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
          <RunExplorer onGraphLoaded={(next) => useExplorerStore.getState().setGraph(next)} onLoadRun={onLoadRun} />
          <RunMonitor onRunGraphReady={onRunGraphReady} onRunStart={onRunStart} />
        </>
      ) : null}
      {activeMode === "reports" ? (
        <>
          <SidebarHeader title="Reports" subtitle="generate reports and open artifacts" />
          <ReportStudio onReportReady={onReportReady} />
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
    { id: "chat", label: "Home", icon: <BrainCircuit size={17} /> },
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
  const activeRun = inferRunFromGraphPath(graph?.path || "");

  return (
    <section className="thread-stage">
      <div className="thread-inner">
        <div className="thread-titlebar">
          <div>
            <strong>Session overview</strong>
            <span>{graph?.topic || "Load a run folder to inspect growth, graph snapshots, and analysis outputs."}</span>
          </div>
          <div className="thread-title-actions">
            <button type="button" onClick={onOpenRuns} title="Open run loader and launch controls">
              <Play size={13} />
              Runs
            </button>
            <button type="button" onClick={onOpenReports} title="Open report studio">
              <FileText size={13} />
              Reports
            </button>
          </div>
        </div>

        <div className="thread-artifact">
          <div className="artifact-icon">
            <Network size={18} />
          </div>
          <div className="artifact-copy">
            <div className="artifact-title-row">
              <strong>{graph?.name || "No graph loaded"}</strong>
              <EmbeddingStatusBadge status={embeddingStatus} onRebuild={onRebuildEmbeddings} />
            </div>
            <span>{contextSummary(graph, selectedNodes)}</span>
          </div>
          <div className="artifact-metrics">
            <span title="Connected components">{formatNumber(stats?.components || 0)} comp</span>
            <span title="Detected communities">{formatNumber(stats?.communities || 0)} comm</span>
            <span title="Average degree">{formatNumber(stats?.avg_degree || 0, 2)} deg</span>
          </div>
          <div className="thread-actions">
            <button type="button" onClick={onOpenGraph} title="Open the graph viewer">
              <Network size={13} />
              Graph
            </button>
            <button type="button" onClick={onOpenSearch} title="Search and select nodes">
              <Search size={13} />
              Search
            </button>
            <button type="button" onClick={onOpenReports} title="Generate or open graph reports">
              <FileText size={13} />
              Profile
            </button>
          </div>
        </div>

        <div className="thread-context-strip" title="Compact status for the current session. These are indicators, not controls.">
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

        <RunDashboardPanel run={activeRun} />
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
  const initialPanelWidths = useMemo(readPanelWidths, []);
  const [leftPanelWidth, setLeftPanelWidth] = useState(initialPanelWidths.left);
  const [rightPanelWidth, setRightPanelWidth] = useState(initialPanelWidths.right);
  const { data: initialGraph } = useQuery({ queryKey: ["graph"], queryFn: api.graph, retry: false });
  const { data: config } = useQuery({ queryKey: ["config"], queryFn: api.config });

  useEffect(() => {
    if (initialGraph) setGraph(initialGraph);
  }, [initialGraph, setGraph]);

  useEffect(() => {
    if (config?.roles) setRoles(config.roles);
  }, [config, setRoles]);

  useEffect(() => {
    writePanelWidths({ left: leftPanelWidth, right: rightPanelWidth });
  }, [leftPanelWidth, rightPanelWidth]);

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
      setGraph(await api.loadRun(run));
    },
    [setGraph],
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

  const activeRun = inferRunFromGraphPath(graph?.path || "");
  const graphArtifactOpen = activeMode === "graph" || activeMode === "search";
  const reportArtifactOpen = activeMode === "reports";
  const workspaceStyle = {
    "--left-panel-width": `${leftPanelWidth}px`,
    "--right-panel-width": `${rightPanelWidth}px`,
  } as React.CSSProperties;

  function startPanelResize(side: "left" | "right") {
    return (event: React.PointerEvent<HTMLButtonElement>) => {
      event.preventDefault();
      const onMove = (moveEvent: PointerEvent) => {
        if (side === "left") {
          setLeftPanelWidth(clampNumber(moveEvent.clientX - 44, 220, 560));
        } else {
          setRightPanelWidth(clampNumber(window.innerWidth - moveEvent.clientX, 300, 680));
        }
      };
      const onUp = () => {
        document.body.classList.remove("resizing-panels");
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
      };
      document.body.classList.add("resizing-panels");
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
    };
  }

  return (
    <div className="app">
      <Header />
      <main className="workspace" style={workspaceStyle}>
        <ActivityRail activeMode={activeMode} onModeChange={setActiveMode} />
        <SideRail
          activeMode={activeMode}
          onModeChange={setActiveMode}
          onLoadRun={loadRun}
          onRunGraphReady={refreshRunGraph}
          onRunStart={clearGraphForRun}
          onReportReady={handleReportReady}
        />
        <button aria-label="Resize tool panel" className="panel-resizer left" onPointerDown={startPanelResize("left")} title="Resize tool panel" type="button" />
        {reportArtifactOpen ? (
          <ReportStage
            out={activeReportOut}
            run={activeRun}
            onOpenReports={() => setActiveMode("reports")}
            onReportReady={handleReportReady}
          />
        ) : graphArtifactOpen ? (
          <section className="artifact-stage">
            <div className="artifact-toolbar">
              <div>
                <div className="artifact-title-row">
                  <strong>Graph artifact</strong>
                  <EmbeddingStatusBadge status={embeddingStatus} onRebuild={() => void startEmbeddingIndex(true)} />
                </div>
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
        <button aria-label="Resize assistant panel" className="panel-resizer right" onPointerDown={startPanelResize("right")} title="Resize assistant panel" type="button" />
        <aside className="assistant-panel">
          <ChatPanel
            onOpenReports={() => setActiveMode("reports")}
            onOpenRuns={() => setActiveMode("runs")}
            sessionReports={sessionReports}
          />
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
