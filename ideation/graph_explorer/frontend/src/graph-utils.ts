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

export const colorPalettes: Record<VisualState["colorPalette"], { label: string; colors: string[] }> = {
  atlas: {
    label: "Atlas blue-gold",
    colors: ["#27648f", "#3b8e8a", "#7aa95c", "#d5a23f", "#be6b3f"],
  },
  viridis: {
    label: "Viridis",
    colors: ["#440154", "#31688e", "#35b779", "#fde725"],
  },
  plasma: {
    label: "Plasma",
    colors: ["#0d0887", "#7e03a8", "#cc4778", "#f89540", "#f0f921"],
  },
  graphite: {
    label: "Graphite",
    colors: ["#5f6874", "#87909a", "#b2a273", "#d6a044"],
  },
  categorical: {
    label: "Categorical",
    colors: palette,
  },
};

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

function hexToRgb(hex: string) {
  const clean = hex.replace("#", "");
  return {
    r: parseInt(clean.slice(0, 2), 16),
    g: parseInt(clean.slice(2, 4), 16),
    b: parseInt(clean.slice(4, 6), 16),
  };
}

function interpolateColors(colors: string[], t: number) {
  const stops = colors.length ? colors : colorPalettes.atlas.colors;
  if (stops.length === 1) return stops[0];
  const scaled = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  const lowIndex = Math.floor(scaled);
  const highIndex = Math.min(stops.length - 1, lowIndex + 1);
  const localT = scaled - lowIndex;
  const low = hexToRgb(stops[lowIndex]);
  const high = hexToRgb(stops[highIndex]);
  const rgb = [low.r, low.g, low.b].map((value, index) => {
    const target = [high.r, high.g, high.b][index];
    return Math.round(value + (target - value) * localT);
  });
  return `rgb(${rgb.join(",")})`;
}

export function categoryColor(value: number, visual: VisualState) {
  const colors = colorPalettes[visual.colorPalette]?.colors || colorPalettes.categorical.colors;
  return colors[Math.abs(Math.floor(value)) % colors.length];
}

export function colorForNode(node: GraphNode, visual: VisualState, selected: boolean): string {
  if (selected) return "#ffffff";
  if (visual.colorBy === "component" || visual.colorBy === "community") {
    return categoryColor(nodeMetric(node, visual.colorBy), visual);
  }
  const range = metricRange([node], visual.colorBy);
  const metric = nodeMetric(node, visual.colorBy);
  if (range.max === range.min) {
    if (visual.colorBy === "degree") return "#37d49a";
    if (visual.colorBy === "pagerank") return "#38bdf8";
    return "#f4b24f";
  }
  return metric >= (range.min + range.max) / 2
    ? colorPalettes[visual.colorPalette]?.colors.at(-1) || "#f4b24f"
    : colorPalettes[visual.colorPalette]?.colors[0] || "#38bdf8";
}

export function colorScale(value: number, range: { min: number; max: number }, paletteName: VisualState["colorPalette"] = "atlas") {
  const t = normalize(value, range);
  return interpolateColors(colorPalettes[paletteName]?.colors || colorPalettes.atlas.colors, t);
}

export function hash01(input: string, salt = 0) {
  let h = 2166136261 + salt;
  for (let i = 0; i < input.length; i++) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return ((h >>> 0) % 100000) / 100000;
}

export function pathNodeSet(paths: string[][]) {
  return new Set(paths.flat());
}

export function edgeKey(source: string, target: string) {
  return source <= target ? `${source}::${target}` : `${target}::${source}`;
}

export function pathEdgeSet(paths: string[][]) {
  const out = new Set<string>();
  for (const path of paths) {
    for (let i = 0; i < path.length - 1; i++) {
      out.add(edgeKey(path[i], path[i + 1]));
    }
  }
  return out;
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

export function layoutNode3D(node: GraphNode, graph: GraphPayload, visual: VisualState) {
  const base = layoutNode(node, graph, visual);
  const zMetric = visual.colorBy === "component" || visual.colorBy === "community" ? "pagerank" : visual.colorBy;
  const range = metricRange(graph.nodes, zMetric);
  const zT = normalize(nodeMetric(node, zMetric), range);
  const jitter = (hash01(node.id, 101) - 0.5) * 42;
  return {
    x: base.x,
    y: base.y,
    z: (zT - 0.5) * 95 + jitter,
  };
}

export function nodeSize(node: GraphNode, graph: GraphPayload, visual: VisualState) {
  if (visual.sizeBy === "constant") return 3.8;
  const range = metricRange(graph.nodes, visual.sizeBy);
  return 2.4 + Math.pow(normalize(nodeMetric(node, visual.sizeBy), range), 0.7) * 9;
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
