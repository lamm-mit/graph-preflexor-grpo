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
  backend?: "responses" | string;
  api_key_env?: string;
  temperature?: number | string;
  max_tokens?: number | string;
  reasoning_effort?: string;
};

export type ModelProbe = {
  ok: boolean;
  category: "ok" | "missing_model" | "api_key" | "connection" | "model_missing" | "completion_error" | "status_error" | "not_http";
  stage: "config" | "models" | "completion";
  message: string;
  model: string;
  base_url?: string;
  api_key_env?: string;
  local?: boolean;
  url?: string;
  models?: string[];
  sample?: string;
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
  cached?: boolean;
  cache_key?: string;
  cache_path?: string;
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

export type RunAnalysisArtifact = {
  path: string;
  name: string;
  updated_at: number;
  size: number;
};

export type RunDashboardGrowthPoint = {
  iter: number;
  depth?: number;
  nodes: number;
  edges: number;
  new_nodes: number;
  tokens: number;
  cum_tokens: number;
  diversity: number;
};

export type RunDashboardGraphPoint = {
  iter: number;
  nodes: number;
  edges: number;
  new_nodes: number;
  new_edges: number;
  avg_degree: number;
};

export type RunDashboardDepthPoint = {
  depth: number;
  nodes: number;
  edges: number;
  cumulative_nodes: number;
  cumulative_edges: number;
};

export type RunDashboardInsight = {
  kind: string;
  title: string;
  score: number;
  detail: string;
};

export type RunAnalysisMarkdown = RunAnalysisArtifact & {
  key: string;
  title: string;
  note: string;
  markdown: string;
};

export type RunAnalysisJson = RunAnalysisArtifact & {
  key: string;
  title: string;
  note: string;
  data: unknown;
  error?: string;
};

export type RunAnalysisFigure = {
  key: string;
  title: string;
  note: string;
  formats: Record<string, string>;
  default_path: string;
  updated_at: number;
  size: number;
};

export type RunDashboard = RunSummary & {
  summary: Record<string, unknown>;
  growth: RunDashboardGrowthPoint[];
  graph_series: RunDashboardGraphPoint[];
  depth_series: RunDashboardDepthPoint[];
  relations: Array<{ relation: string; count: number }>;
  transcript: { turns: number; last_question: string; last_iter: number | null };
  snapshots: GraphFileSummary[];
  analysis_artifacts: RunAnalysisArtifact[];
  analysis_markdown: RunAnalysisMarkdown[];
  analysis_json: RunAnalysisJson[];
  analysis_figures: RunAnalysisFigure[];
  insights: RunDashboardInsight[];
  graph_error?: string;
};

export type RunAnalysisProgress = {
  percent: number;
  current: number;
  total: number;
  message: string;
  detail: string;
};

export type RunAnalysisJobStatus = {
  id: string;
  cmd: string[][];
  run: string;
  status: "running" | "stopping" | "stopped" | "done" | "failed";
  returncode: number | null;
  started_at: number;
  ended_at: number | null;
  log_tail?: string;
  progress?: RunAnalysisProgress;
  analysis_artifacts?: RunAnalysisArtifact[];
};

export type SynthesisJobStatus = {
  id: string;
  cmd: string[];
  run: string;
  out: string;
  absolute_out?: string;
  task?: string;
  style?: string;
  backend?: string;
  model?: string;
  status: "running" | "stopping" | "stopped" | "done" | "failed";
  returncode: number | null;
  started_at: number;
  ended_at: number | null;
  log_tail?: string;
  progress?: RunAnalysisProgress;
  answer_markdown?: string;
  answer_error?: string;
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
  backend: "responses";
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
  profile_json?: string;
};

export type GraphFileSummary = {
  name: string;
  path: string;
  absolute_path?: string;
  run?: string;
  run_name?: string;
  iter?: number | null;
  updated_at: number;
  size: number;
  is_latest?: boolean;
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

export type GraphAskContextNode = {
  id: string;
  label: string;
  degree: number;
  pagerank?: number;
  core?: number;
  iter?: number;
  score?: number;
};

export type GraphAskContext = {
  mode?: "none" | "focused" | "graph_rag";
  query?: string;
  node_count: number;
  edge_count: number;
  node_ids?: string[];
  nodes?: GraphAskContextNode[];
  report_context?: {
    out: string;
    title: string;
    chars: number;
    total_chars: number;
    truncated: boolean;
    included?: string[];
  };
  skill_context?: {
    skills: SkillSummary[];
    chars: number;
    total_requested: number;
    truncated: boolean;
    include_instructions: boolean;
  };
};

export type ChatMessage = {
  id: string;
  role: "system" | "user" | "assistant";
  content: string;
  meta?: string;
  response_id?: string;
  response_model?: string;
  response_backend?: string;
  response_base_url?: string;
  images?: ChatImage[];
  files?: ChatFile[];
  created_at?: number;
};

export type ChatImage = {
  id: string;
  file: string;
  url: string;
  mime: string;
  format: string;
  prompt: string;
  revised_prompt?: string;
  tool_call_id?: string;
  size?: number;
  requested_size?: string;
  quality?: string;
};

export type ChatFile = {
  id: string;
  file: string;
  filename: string;
  url?: string;
  mime: string;
  size: number;
  error?: string;
  source?: {
    type?: string;
    container_id?: string;
    file_id?: string;
  };
};

export type ChatSessionSummary = {
  id: string;
  title: string;
  run: string;
  run_name: string;
  created_at: number;
  updated_at: number;
  message_count: number;
  image_count: number;
  file_count?: number;
  last_message: string;
  archived?: boolean;
};

export type ChatSessionPayload = {
  run: string;
  run_name: string;
  root: string;
  session: ChatSessionSummary;
  messages: ChatMessage[];
  state: Record<string, unknown>;
};

export type ChatSessionsPayload = {
  run: string;
  run_name: string;
  root: string;
  sessions: ChatSessionSummary[];
};

export type ImageGenerationResponse = {
  answer: string;
  response_id?: string;
  images: ChatImage[];
  chat_id: string;
  run: string;
  model: string;
  tool: Record<string, unknown>;
};

export type VisualState = {
  viewMode: "3d" | "2d";
  canvasTheme: "dark" | "light";
  layout: "force" | "component" | "community" | "degree" | "timeline";
  colorBy: "component" | "community" | "degree" | "pagerank" | "core" | "iter" | "depth";
  colorPalette: "atlas" | "viridis" | "plasma" | "graphite" | "categorical";
  sizeBy: "degree" | "pagerank" | "core" | "constant";
  edgeOpacity: number;
  edgeWidth: number;
  edgeStyle: "straight" | "directed";
};

export type SkillSummary = {
  id: string;
  name: string;
  title: string;
  description: string;
  relative_path: string;
  resource_dirs: string[];
  tags: string[];
  safety: "context" | "tool-capable" | "external-capable" | string;
  updated_at: number;
};

export type SkillRegistryPayload = {
  root: string;
  count: number;
  categories: Record<string, number>;
  skills: SkillSummary[];
};

export type SkillDetailPayload = {
  skill: SkillSummary;
  skill_md: string;
  body: string;
  truncated: boolean;
  files: Array<{ path: string; size: number; kind: string }>;
};

export type SkillChatContext = {
  ids: string[];
  include_instructions?: boolean;
  max_chars?: number;
};
