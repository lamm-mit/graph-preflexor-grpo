import type { GraphEdge, GraphNode, GraphPayload, VisualState } from "./types";

export const palette = [
  "#37d49a",
  "#f4b24f",
  "#38bdf8",
  "#ef6c73",
  "#a78bfa",
  "#5eead4",
  "#d4d46a",
  "#fb923c",
  "#818cf8",
  "#84cc16",
  "#e879f9",
  "#b7c4d3",
];

export function formatNumber(value: number | undefined, digits = 0): string {
  if (value === undefined || Number.isNaN(value)) return "";
  return value.toLocaleString(undefined, { maximumFractionDigits: digits });
}

export function nodeMetric(node: GraphNode, key: VisualState["colorBy"] | VisualState["sizeBy"]): number {
  if (key === "constant") return 1;
  const value = node[key as keyof GraphNode];
  return typeof value === "number" ? value : 0;
}

export function metricRange(nodes: GraphNode[], key: VisualState["colorBy"] | VisualState["sizeBy"]) {
  if (!nodes.length || key === "constant") return { min: 0, max: 1 };
  const values = nodes.map((node) => nodeMetric(node, key)).filter(Number.isFinite).sort((a, b) => a - b);
  if (!values.length) return { min: 0, max: 1 };
  const lo = values[Math.floor(values.length * 0.02)] ?? values[0];
  const hi = values[Math.max(0, Math.ceil(values.length * 0.98) - 1)] ?? values[values.length - 1];
  return lo === hi ? { min: values[0], max: values[values.length - 1] || values[0] + 1 } : { min: lo, max: hi };
}

export function normalize(value: number, range: { min: number; max: number }) {
  if (range.max === range.min) return 0.5;
  return Math.max(0, Math.min(1, (value - range.min) / (range.max - range.min)));
}

export function colorForNode(node: GraphNode, visual: VisualState, selected: boolean): string {
  if (selected) return "#ffffff";
  if (visual.colorBy === "component" || visual.colorBy === "community") {
    return palette[Math.abs(Math.floor(nodeMetric(node, visual.colorBy))) % palette.length];
  }
  const range = metricRange([node], visual.colorBy);
  const metric = nodeMetric(node, visual.colorBy);
  if (range.max === range.min) {
    if (visual.colorBy === "degree") return "#37d49a";
    if (visual.colorBy === "pagerank") return "#38bdf8";
    return "#f4b24f";
  }
  return metric >= (range.min + range.max) / 2 ? "#f4b24f" : "#38bdf8";
}

export function colorScale(value: number, range: { min: number; max: number }) {
  const t = normalize(value, range);
  const low = [56, 189, 248];
  const high = [244, 178, 79];
  const rgb = low.map((v, i) => Math.round(v + (high[i] - v) * t));
  return `rgb(${rgb.join(",")})`;
}

export function hash01(input: string, salt = 0) {
  let h = 2166136261 + salt;
  for (let i = 0; i < input.length; i++) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return ((h >>> 0) % 100000) / 100000;
}

export function layoutNode(node: GraphNode, graph: GraphPayload, visual: VisualState) {
  const comp = Math.max(0, node.component || 0);
  const community = Math.max(0, node.community || 0);
  const degreeRange = metricRange(graph.nodes, "degree");
  const degreeT = normalize(node.degree, degreeRange);
  const local = hash01(node.id, 17) * Math.PI * 2;
  const radius = 8 + Math.sqrt(Math.max(1, node.degree)) * 3.2 + hash01(node.id, 31) * 22;

  if (visual.layout === "component") {
    const angle = comp * 2.3999632297;
    const cr = 24 + Math.sqrt(comp + 1) * 20;
    return { x: Math.cos(angle) * cr + Math.cos(local) * radius, y: Math.sin(angle) * cr + Math.sin(local) * radius };
  }
  if (visual.layout === "community") {
    const angle = community * 2.3999632297;
    const cr = 30 + Math.sqrt(community + 1) * 18;
    return { x: Math.cos(angle) * cr + Math.cos(local) * radius, y: Math.sin(angle) * cr + Math.sin(local) * radius };
  }
  if (visual.layout === "degree") {
    const r = 18 + (1 - degreeT) * 120;
    return { x: Math.cos(local) * r, y: Math.sin(local) * r };
  }
  if (visual.layout === "timeline") {
    const maxIter = Math.max(1, graph.stats.max_iter);
    return { x: (node.iter / maxIter - 0.5) * 260, y: (comp - graph.stats.components / 2) * 8 + Math.sin(local) * 12 };
  }
  const angle = comp * 2.3999632297;
  const cr = Math.sqrt(comp + 1) * 8;
  return { x: Math.cos(angle) * cr + Math.cos(local) * radius, y: Math.sin(angle) * cr + Math.sin(local) * radius };
}

export function nodeSize(node: GraphNode, graph: GraphPayload, visual: VisualState) {
  if (visual.sizeBy === "constant") return 3.8;
  const range = metricRange(graph.nodes, visual.sizeBy);
  return 2.4 + Math.pow(normalize(nodeMetric(node, visual.sizeBy), range), 0.7) * 9;
}

function hexToRgb(hex: string) {
  const clean = hex.replace("#", "");
  return {
    r: parseInt(clean.slice(0, 2), 16),
    g: parseInt(clean.slice(2, 4), 16),
    b: parseInt(clean.slice(4, 6), 16),
  };
}

export function edgeColor(edge: GraphEdge, opacity = 0.16) {
  if (!edge.relation) return `rgba(110,126,145,${opacity})`;
  const index = Math.floor(hash01(edge.relation) * palette.length) % palette.length;
  const rgb = hexToRgb(palette[index]);
  return `rgba(${rgb.r},${rgb.g},${rgb.b},${opacity})`;
}

export function contextSummary(graph: GraphPayload | null, selectedNodes: string[]) {
  if (!graph) return "No graph loaded.";
  const bits = [`${formatNumber(graph.stats.nodes)} nodes`, `${formatNumber(graph.stats.edges)} edges`];
  if (selectedNodes.length) bits.push(`${selectedNodes.length} selected`);
  return bits.join(" | ");
}
