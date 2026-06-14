import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import * as Collapsible from "@radix-ui/react-collapsible";
import Graph from "graphology";
import Sigma from "sigma";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
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
  edgeKey,
  formatNumber,
  layoutNode,
  layoutNode3D,
  metricRange,
  nodeMetric,
  nodeSize,
  palette,
  pathEdgeSet,
  pathNodeSet,
} from "./graph-utils";
import { useExplorerStore } from "./store";
import type { BridgeIdea, GraphNode, GraphPayload, JobStatus, ModelRole, PathConnector, RunSummary, SearchResult } from "./types";
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

function graphAgentRole(roles: Record<string, ModelRole>, chatRole: string) {
  return roles.graph_qa?.model ? roles.graph_qa : roles.questioner?.model ? roles.questioner : roles[chatRole];
}

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

function cx(...classes: Array<string | false | undefined>) {
  return classes.filter(Boolean).join(" ");
}

function formatRunTime(seconds: number) {
  if (!seconds) return "unknown";
  return new Date(seconds * 1000).toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
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
      const categoryColor =
        visual.colorBy === "component" || visual.colorBy === "community"
          ? palette[Math.abs(Math.floor(nodeMetric(node, visual.colorBy))) % palette.length]
          : colorScale(nodeMetric(node, visual.colorBy), colorRange);
      sigmaGraph.addNode(node.id, {
        ...pos,
        label: node.label,
        size: selected.has(node.id) || isHighlighted ? nodeSize(node, graph, visual) * 1.9 : nodeSize(node, graph, visual),
        color: selected.has(node.id) ? "#ffffff" : isHighlighted ? "#ffd166" : categoryColor,
        borderColor: selected.has(node.id) ? "#37d49a" : isHighlighted ? "#ffef9f" : categoryColor,
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
      const categoryColor =
        visual.colorBy === "component" || visual.colorBy === "community"
          ? palette[Math.abs(Math.floor(nodeMetric(node, visual.colorBy))) % palette.length]
          : colorScale(nodeMetric(node, visual.colorBy), colorRange);
      const color = new THREE.Color(
        selected.has(node.id) ? "#163c3a" : highlightedNodes.has(node.id) ? "#b56b16" : categoryColor,
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
  const chatEndRef = useRef<HTMLDivElement | null>(null);
  const contextLabel = contextSummary(graph, selectedNodes);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ block: "end" });
  }, [messages]);

  async function ask() {
    if (!question.trim() || !graph) return;
    const role = roles[chatRole] || graphAgentRole(roles, chatRole);
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
          <h2>Assistant</h2>
          <span>{contextLabel}</span>
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
          {[
            ["Summary", "Summarize this graph focus."],
            ["Gaps", "Find gaps and weak links."],
            ["Bridges", "Suggest bridge experiments."],
            ["Hubs", "Rank the strongest hubs."],
          ].map(([label, prompt]) => (
            <button key={prompt} onClick={() => setQuestion(prompt)} type="button">
              {label}
            </button>
          ))}
        </div>
        <textarea
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={(event) => {
            if ((event.metaKey || event.ctrlKey) && event.key === "Enter") void ask();
          }}
          placeholder={graph ? "Message the graph..." : "Load a graph to chat..."}
          rows={4}
          value={question}
        />
        <div className="composer-tools">
          <input
            onChange={(event) => setContextQuery(event.target.value)}
            placeholder="context search"
            value={contextQuery}
          />
          <label className="context-count">
            <span>Nodes</span>
            <input
              min={20}
              max={400}
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

function RunExplorer({
  onLoadRun,
  defaultOpen = false,
}: {
  onLoadRun: (run: string) => Promise<void>;
  defaultOpen?: boolean;
}) {
  const graph = useExplorerStore((state) => state.graph);
  const [filter, setFilter] = useState("");
  const [status, setStatus] = useState("");
  const [loadingRun, setLoadingRun] = useState("");
  const runsQuery = useQuery({
    queryKey: ["runs"],
    queryFn: api.runs,
    refetchInterval: 5000,
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

  return (
    <Drawer
      defaultOpen={defaultOpen}
      icon={<FolderOpen size={14} />}
      note={`${runsQuery.data?.runs.length || 0} folders`}
      title="Run Explorer"
    >
      <div className="row run-filter-row">
        <input
          onChange={(event) => setFilter(event.target.value)}
          placeholder="filter runs, topics, strategies"
          value={filter}
        />
        <IconButton
          disabled={runsQuery.isFetching}
          icon={runsQuery.isFetching ? <Loader2 className="spin" size={14} /> : <RotateCcw size={14} />}
          label="Refresh"
          onClick={() => void runsQuery.refetch()}
        />
      </div>
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

function RunMonitor({
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
  const [topic, setTopic] = useState(storedRun.topic || "");
  const [strategy, setStrategy] = useState<IdeationStrategy>(normalizeStrategy(storedRun.strategy));
  const [calls, setCalls] = useState(storedRun.calls || 50);
  const [iters, setIters] = useState(storedRun.iters || 50);
  const [out, setOut] = useState(storedRun.out || "runs/explorer_run");
  const [job, setJob] = useState<JobStatus | null>(storedRun.job || null);
  const [monitorStatus, setMonitorStatus] = useState(storedRun.job ? "Restored saved run monitor." : "");
  const [busy, setBusy] = useState(false);
  const lastSnapshotRef = useRef<string>("");

  useEffect(() => {
    writeRunMonitorStorage({ topic, strategy, calls, iters, out, job, jobId: job?.id });
  }, [calls, iters, job, out, strategy, topic]);

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
        const snapshot = next.snapshot_id || (next.graph_ready ? `${next.graph_path || next.out}:ready` : "");
        if (next.graph_ready && next.out && snapshot && snapshot !== lastSnapshotRef.current) {
          lastSnapshotRef.current = snapshot;
          void onRunGraphReady(next.out);
        }
      } catch (error) {
        setMonitorStatus(error instanceof Error ? error.message : String(error));
      }
    }, 2200);
    return () => window.clearInterval(timer);
  }, [job, onRunGraphReady]);

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
      const next = await api.ideate({ topic, strategy, budget_calls: calls, max_iters: iters, out });
      setJob(next);
      setMonitorStatus(`Started run ${next.id}.`);
    } finally {
      setBusy(false);
    }
  }

  async function stop() {
    if (!job) return;
    const next = await api.stopJob(job.id);
    setJob(next);
    setMonitorStatus(`Stop requested for run ${next.id}.`);
  }

  const progress = job?.status === "done" ? 100 : Math.round((job?.progress?.percent || 0) * 100);

  return (
    <Drawer defaultOpen={defaultOpen} icon={<Play size={14} />} note={job?.status || "idle"} title="Run Monitor">
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
      {monitorStatus ? <div className="status-box">{monitorStatus}</div> : null}
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
  const [pathMode, setPathMode] = useState<"pairwise" | "sequence">("pairwise");
  const [pathCutoff, setPathCutoff] = useState(8);
  const [connectors, setConnectors] = useState<PathConnector[]>([]);
  const [ideas, setIdeas] = useState<BridgeIdea[]>([]);
  const [status, setStatus] = useState("");
  const [pathBusy, setPathBusy] = useState(false);
  const [agentBusy, setAgentBusy] = useState(false);
  const [ideaBusy, setIdeaBusy] = useState(false);
  const ideaGraphRef = useRef("");
  const seed = selectedNodes.length ? selectedNodes : selectedNode ? [selectedNode] : [];

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
    const next = await api.path({ source: source.trim(), target: target.trim(), k: 5, cutoff: 8 });
    setGraph(next);
    setHighlightedPaths(next.paths || []);
    setConnectors([]);
    setStatus(`${formatNumber(next.paths?.length || 0)} paths`);
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
      });
      setGraph(next);
      setHighlightedPaths(next.paths || []);
      setConnectors(next.connectors || []);
      setStatus(`${formatNumber(next.paths?.length || 0)} paths | ${formatNumber(next.connectors?.length || 0)} connector nodes`);
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
      setStatus(res.ideas?.length ? `${res.ideas.length} bridge ideas loaded` : "No bridge ideas found for this graph.");
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
                setStatus(idea.rationale);
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
          Mode
          <select value={pathMode} onChange={(event) => setPathMode(event.target.value as typeof pathMode)}>
            <option value="pairwise">Pairwise bridge</option>
            <option value="sequence">Ordered route</option>
          </select>
        </label>
        <label>
          Max hops
          <input min={2} max={16} onChange={(event) => setPathCutoff(Number(event.target.value))} type="number" value={pathCutoff} />
        </label>
      </div>
      <div className="button-row">
        <IconButton
          disabled={!graph || pathBusy || (!concepts.trim() && seed.length < 2)}
          icon={pathBusy ? <Loader2 className="spin" size={14} /> : <Network size={14} />}
          label="Find Bridges"
          onClick={findBridgeNetwork}
          tone="primary"
        />
        <IconButton
          disabled={!graph || ideaBusy}
          icon={ideaBusy ? <Loader2 className="spin" size={14} /> : <BrainCircuit size={14} />}
          label="Suggest Ideas"
          onClick={() => suggestBridgeIdeas(false)}
        />
        <IconButton
          disabled={!graph || agentBusy}
          icon={agentBusy ? <Loader2 className="spin" size={14} /> : <BrainCircuit size={14} />}
          label="Ask Agent"
          onClick={askPathAgent}
        />
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
}: {
  onLoadRun: (run: string) => Promise<void>;
  onRunGraphReady: (run: string) => Promise<void>;
  onRunStart: () => Promise<void>;
}) {
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
      <RunExplorer onLoadRun={onLoadRun} />
      <RunMonitor onRunGraphReady={onRunGraphReady} onRunStart={onRunStart} />
    </>
  );
}

function SideRail({
  activeMode,
  onLoadRun,
  onRunGraphReady,
  onRunStart,
}: {
  activeMode: WorkspaceMode;
  onLoadRun: (run: string) => Promise<void>;
  onRunGraphReady: (run: string) => Promise<void>;
  onRunStart: () => Promise<void>;
}) {
  return (
    <aside className="side-panel">
      {activeMode === "chat" ? (
        <ChatSidebar onLoadRun={onLoadRun} onRunGraphReady={onRunGraphReady} onRunStart={onRunStart} />
      ) : null}
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
          <SidebarHeader title="Runs" subtitle="sessions, launch, and monitor" />
          <RunExplorer defaultOpen onLoadRun={onLoadRun} />
          <RunMonitor defaultOpen onRunGraphReady={onRunGraphReady} onRunStart={onRunStart} />
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

function ThreadStage({
  onOpenGraph,
  onOpenSearch,
  onOpenRuns,
}: {
  onOpenGraph: () => void;
  onOpenSearch: () => void;
  onOpenRuns: () => void;
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
          </div>
        </div>

        <div className="thread-session-grid">
          <div className="session-card">
            <span>Selection</span>
            <strong>{selectedNodes.length ? `${formatNumber(selectedNodes.length)} nodes` : "None"}</strong>
          </div>
          <div className="session-card">
            <span>Graph</span>
            <strong>{graph ? `${formatNumber(stats?.nodes || 0)} / ${formatNumber(stats?.edges || 0)}` : "Not loaded"}</strong>
          </div>
          <div className="session-card">
            <span>Run</span>
            <strong>{graph?.path ? graph.path.split("/runs/").pop()?.split("/")[0] || graph.name : "No session"}</strong>
          </div>
        </div>
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

  const graphArtifactOpen = activeMode === "graph" || activeMode === "search";

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
        />
        {graphArtifactOpen ? (
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
            <GraphCanvas />
          </section>
        ) : (
          <ThreadStage
            onOpenGraph={() => setActiveMode("graph")}
            onOpenRuns={() => setActiveMode("runs")}
            onOpenSearch={() => setActiveMode("search")}
          />
        )}
        <aside className="assistant-panel">
          <ChatPanel />
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
