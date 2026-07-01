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

type NumericRange = { min: number; max: number };

type CoordinatePair = {
  xKey: string;
  yKey: string;
  xRange: NumericRange;
  yRange: NumericRange;
};

type LayoutCache = {
  bridgeScore: Map<string, number>;
  maxIter: number;
  maxDepth: number;
  componentCount: number;
  communityCount: number;
  degreeRange: NumericRange;
  pagerankRange: NumericRange;
  coreRange: NumericRange;
  semanticCoordinates: CoordinatePair | null;
};

const layoutCache = new WeakMap<GraphPayload, LayoutCache>();

const coordinateCandidates = [
  ["umap_x", "umap_y"],
  ["umap_0", "umap_1"],
  ["umap1", "umap2"],
  ["embedding_x", "embedding_y"],
  ["embed_x", "embed_y"],
  ["semantic_x", "semantic_y"],
  ["pca_x", "pca_y"],
  ["pca_0", "pca_1"],
  ["pca1", "pca2"],
  ["tsne_x", "tsne_y"],
  ["tsne_0", "tsne_1"],
  ["tsne1", "tsne2"],
  ["layout_x", "layout_y"],
  ["pos_x", "pos_y"],
  ["x", "y"],
] as const;

function finiteNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
}

function attrNumber(node: GraphNode, key: string): number | null {
  return finiteNumber(node.attrs?.[key]);
}

function valuesRange(values: number[]): NumericRange {
  const finite = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!finite.length) return { min: 0, max: 1 };
  const lo = finite[Math.floor(finite.length * 0.02)] ?? finite[0];
  const hi = finite[Math.max(0, Math.ceil(finite.length * 0.98) - 1)] ?? finite[finite.length - 1];
  if (lo !== hi) return { min: lo, max: hi };
  const value = finite[0];
  return { min: value - 1, max: value + 1 };
}

function detectCoordinatePair(nodes: GraphNode[]): CoordinatePair | null {
  for (const [xKey, yKey] of coordinateCandidates) {
    const xs: number[] = [];
    const ys: number[] = [];
    for (const node of nodes) {
      const x = attrNumber(node, xKey);
      const y = attrNumber(node, yKey);
      if (x === null || y === null) continue;
      xs.push(x);
      ys.push(y);
    }
    if (xs.length >= Math.min(3, nodes.length)) {
      return { xKey, yKey, xRange: valuesRange(xs), yRange: valuesRange(ys) };
    }
  }
  return null;
}

function countDistinct(nodes: GraphNode[], key: "component" | "community") {
  const values = new Set<number>();
  for (const node of nodes) {
    const value = node[key];
    if (Number.isFinite(value) && value >= 0) values.add(value);
  }
  return Math.max(1, values.size);
}

function getLayoutCache(graph: GraphPayload): LayoutCache {
  const cached = layoutCache.get(graph);
  if (cached) return cached;

  const nodesById = new Map(graph.nodes.map((node) => [node.id, node] as const));
  const totals = new Map<string, number>();
  const crossCommunity = new Map<string, number>();
  const componentCount = Math.max(1, graph.stats.components || countDistinct(graph.nodes, "component"));
  const communityCount = Math.max(1, graph.stats.communities || countDistinct(graph.nodes, "community"));

  for (const edge of graph.edges) {
    const source = nodesById.get(edge.source);
    const target = nodesById.get(edge.target);
    if (!source || !target) continue;
    totals.set(source.id, (totals.get(source.id) || 0) + 1);
    totals.set(target.id, (totals.get(target.id) || 0) + 1);
    const crossesCommunity =
      source.community !== target.community || (communityCount <= 1 && source.component !== target.component);
    if (crossesCommunity) {
      crossCommunity.set(source.id, (crossCommunity.get(source.id) || 0) + 1);
      crossCommunity.set(target.id, (crossCommunity.get(target.id) || 0) + 1);
    }
  }

  const degreeRange = valuesRange(graph.nodes.map((node) => node.degree));
  const pagerankRange = valuesRange(graph.nodes.map((node) => node.pagerank));
  const betweennessRange = valuesRange(graph.nodes.map((node) => node.betweenness));
  const coreRange = valuesRange(graph.nodes.map((node) => node.core));
  const bridgeScore = new Map<string, number>();

  for (const node of graph.nodes) {
    const total = totals.get(node.id) || node.degree || 0;
    const cross = crossCommunity.get(node.id) || 0;
    const crossRatio = total > 0 ? Math.min(1, cross / total) : 0;
    const betweennessT = normalize(node.betweenness, betweennessRange);
    const pagerankT = normalize(node.pagerank, pagerankRange);
    const clusteringGap = 1 - Math.max(0, Math.min(1, node.clustering || 0));
    const score = Math.max(
      0,
      Math.min(1, betweennessT * 0.42 + crossRatio * 0.34 + clusteringGap * 0.16 + pagerankT * 0.08),
    );
    bridgeScore.set(node.id, score);
  }

  const next: LayoutCache = {
    bridgeScore,
    maxIter: Math.max(1, graph.stats.max_iter || 0, ...graph.nodes.map((node) => node.iter || 0)),
    maxDepth: Math.max(1, ...graph.nodes.map((node) => node.depth || 0)),
    componentCount,
    communityCount,
    degreeRange,
    pagerankRange,
    coreRange,
    semanticCoordinates: detectCoordinatePair(graph.nodes),
  };
  layoutCache.set(graph, next);
  return next;
}

function compactLane(index: number, count: number, span = 190) {
  if (count <= 1) return 0;
  const laneCount = Math.min(count, 34);
  const lane = count > laneCount ? Math.floor(hash01(String(index), 823) * laneCount) : Math.abs(index) % laneCount;
  return (lane / Math.max(1, laneCount - 1) - 0.5) * span;
}

function preferredLane(node: GraphNode, cache: LayoutCache) {
  if (cache.communityCount > 1 && cache.communityCount <= 42) {
    return { index: Math.max(0, node.community || 0), count: cache.communityCount };
  }
  return { index: Math.max(0, node.component || 0), count: cache.componentCount };
}

function textProjection(node: GraphNode) {
  const source = `${node.label || ""} ${node.id || ""}`.toLowerCase();
  const tokens = source.match(/[a-z0-9][a-z0-9_-]{1,}/g) || [source || node.id];
  let x = 0;
  let y = 0;
  let weightSum = 0;
  for (const token of tokens.slice(0, 18)) {
    const weight = 0.65 + hash01(token, 503) * 0.85;
    const angle = hash01(token, 557) * Math.PI * 2;
    const radius = 0.45 + hash01(token, 563) * 0.7;
    x += Math.cos(angle) * radius * weight;
    y += Math.sin(angle) * radius * weight;
    weightSum += weight;
  }
  if (!weightSum) {
    const angle = hash01(node.id, 557) * Math.PI * 2;
    return { x: Math.cos(angle), y: Math.sin(angle) };
  }
  return { x: x / weightSum, y: y / weightSum };
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
  const cache = getLayoutCache(graph);
  const comp = Math.max(0, node.component || 0);
  const community = Math.max(0, node.community || 0);
  const degreeT = normalize(node.degree, cache.degreeRange);
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
    return { x: (node.iter / cache.maxIter - 0.5) * 260, y: (comp - cache.componentCount / 2) * 8 + Math.sin(local) * 12 };
  }
  if (visual.layout === "timeline_community") {
    const lane = preferredLane(node, cache);
    const x = (node.iter / cache.maxIter - 0.5) * 320 + (hash01(node.id, 71) - 0.5) * 8;
    const y = compactLane(lane.index, lane.count, 210) + (degreeT - 0.5) * 10 + Math.sin(local) * 5;
    return { x, y };
  }
  if (visual.layout === "semantic_map") {
    if (cache.semanticCoordinates) {
      const { xKey, yKey, xRange, yRange } = cache.semanticCoordinates;
      const x = attrNumber(node, xKey);
      const y = attrNumber(node, yKey);
      if (x !== null && y !== null) {
        return {
          x: (normalize(x, xRange) - 0.5) * 310 + (hash01(node.id, 811) - 0.5) * 5,
          y: (normalize(y, yRange) - 0.5) * 220 + (hash01(node.id, 821) - 0.5) * 5,
        };
      }
    }
    const semantic = textProjection(node);
    const lane = preferredLane(node, cache);
    return {
      x: semantic.x * 255 + (hash01(String(lane.index), 607) - 0.5) * 32,
      y: semantic.y * 185 + (hash01(String(lane.index), 613) - 0.5) * 28,
    };
  }
  if (visual.layout === "lineage_tree") {
    const level = cache.maxDepth > 1 ? Math.max(0, node.depth || 0) : Math.max(0, node.iter || 0);
    const maxLevel = Math.max(1, cache.maxDepth > 1 ? cache.maxDepth : cache.maxIter);
    const lane = preferredLane(node, cache);
    const progress = level / maxLevel;
    const branch = (hash01(`${node.id}:branch`, 409) - 0.5) * (18 + progress * 92);
    return {
      x: (progress - 0.5) * 315,
      y: compactLane(lane.index, lane.count, 175) + branch + Math.sin(local) * (4 + degreeT * 8),
    };
  }
  if (visual.layout === "unexpected_bridges") {
    const score = cache.bridgeScore.get(node.id) || 0;
    const lane = preferredLane(node, cache);
    const angle = lane.index * 2.3999632297 + (hash01(node.id, 577) - 0.5) * 0.42;
    const r = 16 + Math.pow(1 - score, 1.55) * 165 + hash01(node.id, 587) * 14;
    return { x: Math.cos(angle) * r, y: Math.sin(angle) * r };
  }
  if (visual.layout === "core_periphery") {
    const coreT = Math.max(normalize(node.core, cache.coreRange), normalize(node.pagerank, cache.pagerankRange), degreeT * 0.72);
    const lane = preferredLane(node, cache);
    const angle = lane.index * 2.3999632297 + (local - Math.PI) * 0.34;
    const r = 18 + Math.pow(1 - coreT, 1.25) * 150 + hash01(node.id, 911) * 12;
    return { x: Math.cos(angle) * r, y: Math.sin(angle) * r };
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
