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

export type BridgeIdea = {
  title: string;
  concepts: string[];
  query: string;
  rationale: string;
  connectors: string[];
};

export type SearchResult = {
  id: string;
  label: string;
  degree: number;
  pagerank?: number;
  core?: number;
  iter: number;
  score: number;
  semantic_score?: number;
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

export type EmbeddingProgress = {
  percent: number;
  current: number;
  total: number;
  message: string;
  detail: string;
};

export type EmbeddingStatus = {
  id?: string;
  graph_id?: string;
  graph_name?: string;
  model?: string;
  status: "idle" | "running" | "done" | "failed";
  ready: boolean;
  nodes: number;
  dimension: number;
  started_at?: number | null;
  ended_at?: number | null;
  error?: string;
  progress: EmbeddingProgress;
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

export type ProfileSummary = {
  topic?: string;
  generated_at?: string;
  nodes?: number;
  edges?: number;
  density?: number;
  components?: number;
  modules?: number;
  modularity?: number;
  embed_model?: string;
  embed_error?: string;
  llm_model?: string;
  llm_backend?: string;
  pdf_error?: string;
  source?: string;
};

export type ProfileArtifacts = {
  out: string;
  absolute_out?: string;
  ready: boolean;
  report_path?: string;
  profile_path?: string;
  pdf_path?: string;
  figures: string[];
  summary: ProfileSummary;
  updated_at: number;
  error?: string;
};

export type ProfileProgress = {
  percent: number;
  current: number;
  total: number;
  message: string;
  detail: string;
};

export type ProfileOptions = {
  run?: string;
  graph?: string;
  out: string;
  embed_model?: string;
  top_nodes?: number;
  max_modules?: number;
  profile_preset?: "full" | "light";
  llm: boolean;
  llm_modules?: number;
  backend: "responses" | "openai" | "chat" | "hf";
  model: string;
  base_url?: string;
  temperature?: number;
  max_summary_tokens?: number;
  deep_pass_tokens?: number;
  deep_dive_tokens?: number;
  reasoning_effort?: "minimal" | "low" | "medium" | "high";
  llm_deep_passes?: number;
  llm_report_review?: boolean;
  report_review_tokens?: number;
  report_review_max_chunks?: number;
  report_review_chunk_chars?: number;
  report_review_memo_chars?: number;
  device?: string;
  dtype?: "auto" | "float16" | "bfloat16" | "float32";
  pdf?: boolean;
};

export type ProfileJobStatus = {
  id: string;
  cmd: string[];
  run?: string;
  graph?: string;
  out: string;
  status: "running" | "stopping" | "stopped" | "done" | "failed";
  returncode: number | null;
  started_at: number;
  ended_at: number | null;
  log_tail?: string;
  progress?: ProfileProgress;
  artifacts?: ProfileArtifacts;
};

export type ProfileReportPayload = {
  artifacts: ProfileArtifacts;
  markdown: string;
};

export type RunSummary = {
  name: string;
  path: string;
  absolute_path?: string;
  topic: string;
  strategy: string;
  updated_at: number;
  graph_ready: boolean;
  graph_path: string;
  snapshot_count: number;
  snapshot_iter: number | null;
  nodes: number;
  edges: number;
  calls: number;
  iters: number;
  status: string;
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
