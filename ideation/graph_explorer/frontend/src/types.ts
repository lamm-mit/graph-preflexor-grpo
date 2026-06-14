export type GraphStats = {
  nodes: number;
  edges: number;
  directed: boolean;
  components: number;
  communities: number;
  largest_component: number;
  density: number;
  max_degree: number;
  avg_degree: number;
  max_iter: number;
  component_sizes?: number[];
  community_sizes?: number[];
};

export type GraphNode = {
  id: string;
  label: string;
  degree: number;
  pagerank: number;
  core: number;
  closeness: number;
  betweenness: number;
  clustering: number;
  eigenvector: number;
  component: number;
  community: number;
  iter: number;
  depth: number;
  attrs: Record<string, unknown>;
};

export type GraphEdge = {
  id: string;
  source: string;
  target: string;
  relation: string;
  iter: number;
  depth: number;
  attrs: Record<string, unknown>;
};

export type GraphPayload = {
  graph_id?: string;
  name: string;
  path: string;
  topic: string;
  stats: GraphStats;
  nodes: GraphNode[];
  edges: GraphEdge[];
};

export type PathConnector = {
  id: string;
  label: string;
  count: number;
  degree: number;
  pagerank: number;
  core: number;
};

export type SearchResult = {
  id: string;
  label: string;
  degree: number;
  pagerank?: number;
  core?: number;
  iter: number;
  score: number;
};

export type ModelRole = {
  role?: string;
  provider: string;
  model: string;
  base_url: string;
  api_key_env?: string;
  temperature?: number | string;
  max_tokens?: number | string;
  reasoning_effort?: string;
};

export type ConfigPayload = {
  exists: boolean;
  path: string;
  roles: Record<string, ModelRole>;
};

export type JobProgress = {
  percent: number;
  calls: number;
  total_calls: number;
  iter: number;
  total_iters: number;
  nodes: number;
  edges: number;
  new_nodes: number;
  tokens: number;
  cum_tokens: number;
  max_tokens: number;
  diversity: number;
  growth_tail: Array<{
    iter: number;
    nodes: number;
    edges: number;
    new_nodes: number;
    cum_tokens: number;
    diversity: number;
  }>;
};

export type JobStatus = {
  id: string;
  cmd: string[];
  out: string;
  status: "running" | "stopping" | "stopped" | "done" | "failed";
  returncode: number | null;
  started_at: number;
  ended_at: number | null;
  log_tail?: string;
  graph_ready?: boolean;
  graph_path?: string;
  snapshot_id?: string;
  snapshot_count?: number;
  snapshot_iter?: number | null;
  progress?: JobProgress;
};

export type ChatMessage = {
  id: string;
  role: "system" | "user" | "assistant";
  content: string;
  meta?: string;
};

export type VisualState = {
  viewMode: "3d" | "2d";
  layout: "force" | "component" | "community" | "degree" | "timeline";
  colorBy: "component" | "community" | "degree" | "pagerank" | "core" | "iter" | "depth";
  sizeBy: "degree" | "pagerank" | "core" | "constant";
  edgeOpacity: number;
};
