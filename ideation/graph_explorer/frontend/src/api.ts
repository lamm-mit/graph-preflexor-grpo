import type {
  BridgeIdea,
  ChatMessage,
  ChatSessionPayload,
  ChatSessionsPayload,
  ConfigPayload,
  EmbeddingStatus,
  GraphAskContext,
  GraphFileSummary,
  GraphPayload,
  ImageGenerationResponse,
  JobStatus,
  ModelProbe,
  ModelRole,
  PathConnector,
  ProfileJobStatus,
  ProfileOptions,
  ProfileReportPayload,
  ProfileArtifacts,
  RunAnalysisJobStatus,
  RunDashboard,
  RunSummary,
  SearchResult,
  SkillChatContext,
  SkillDetailPayload,
  SkillRegistryPayload,
  SynthesisJobStatus,
} from "./types";

async function request<T>(path: string, body?: unknown): Promise<T> {
  const init =
    body === undefined
      ? undefined
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        };
  const res = await fetch(path, init);
  const data = (await res.json()) as T & { error?: string };
  if (!res.ok || data.error) {
    throw new Error(data.error || `HTTP ${res.status}`);
  }
  return data;
}

export const api = {
  graph: () => request<GraphPayload>("/api/graph"),
  runs: () => request<{ root: string; runs: RunSummary[] }>("/api/runs"),
  config: () => request<ConfigPayload>("/api/config"),
  loadRun: (run: string) => request<GraphPayload>("/api/load_run", { run }),
  runDashboard: (run: string) => request<RunDashboard>("/api/run_dashboard", { run }),
  runGraphs: (run: string) => request<{ run: string; graphs: GraphFileSummary[] }>("/api/run_graphs", { run }),
  graphmlFiles: () => request<{ graphs: GraphFileSummary[] }>("/api/graphml_files", {}),
  skills: (query = "") => request<SkillRegistryPayload>("/api/skills", query ? { query } : {}),
  skillDetail: (id: string, max_chars = 30000) => request<SkillDetailPayload>("/api/skill", { id, max_chars }),
  chatSessions: (run: string) => request<ChatSessionsPayload>("/api/chat_sessions", { run }),
  createChatSession: (body: { run: string; title?: string; state?: Record<string, unknown>; messages?: ChatMessage[] }) =>
    request<ChatSessionPayload>("/api/chat_session_create", body),
  loadChatSession: (run: string, id: string) => request<ChatSessionPayload>("/api/chat_session_load", { run, id }),
  saveChatSession: (body: {
    run: string;
    id: string;
    title?: string;
    messages: ChatMessage[];
    state?: Record<string, unknown>;
  }) => request<ChatSessionPayload>("/api/chat_session_save", body),
  renameChatSession: (run: string, id: string, title: string) =>
    request<ChatSessionPayload>("/api/chat_session_rename", { run, id, title }),
  chatAssetUrl: (run: string, chat_id: string, file: string) =>
    `/api/chat_asset?run=${encodeURIComponent(run || "__workspace__")}&chat_id=${encodeURIComponent(chat_id)}&file=${encodeURIComponent(file)}`,
  suggestRunOut: (body: { topic: string; strategy?: string; model_config?: ModelRole }) =>
    request<{ out: string; slug: string; source: "model" | "fallback"; reason?: string; model_text?: string }>("/api/suggest_run_out", body),
  clearGraph: () => request<{ ok: boolean }>("/api/clear_graph", {}),
  uploadGraphml: (name: string, graphml: string) =>
    request<GraphPayload>("/api/load_graphml", { name, graphml }),
  search: (query: string, limit = 40) =>
    request<{ results: SearchResult[] }>("/api/search", { query, limit }),
  embeddingStatus: () => request<EmbeddingStatus>("/api/embedding_status"),
  startEmbeddings: (body?: { model?: string; force?: boolean; batch_size?: number }) =>
    request<EmbeddingStatus>("/api/embedding_index", body || { model: "auto" }),
  ask: (body: {
    run?: string;
    chat_id?: string;
    question: string;
    selected_nodes: string[];
    query: string;
    depth: number;
    max_nodes: number;
    max_edges: number;
    context_mode?: "none" | "focused" | "graph_rag";
    report_context?: { out: string; max_chars?: number; include_report?: boolean; include_profile?: boolean } | null;
    skill_context?: SkillChatContext | null;
    enable_code_interpreter?: boolean;
    code_interpreter_memory?: string;
    model_config: ModelRole & { api_key?: string };
    history?: Array<{ role: "user" | "assistant"; content: string }>;
    previous_response_id?: string;
  }) => request<{ answer: string; context: GraphAskContext; response_id?: string; files?: ChatMessage["files"]; stateful?: boolean; state_mode?: string; backend?: string }>("/api/ask", body),
  chatContextPreview: (body: {
    run?: string;
    chat_id?: string;
    question: string;
    selected_nodes: string[];
    query: string;
    depth: number;
    max_nodes: number;
    max_edges: number;
    context_mode?: "none" | "focused" | "graph_rag";
    report_context?: { out: string; max_chars?: number; include_report?: boolean; include_profile?: boolean } | null;
    skill_context?: SkillChatContext | null;
    enable_code_interpreter?: boolean;
    code_interpreter_memory?: string;
    model_config: ModelRole & { api_key?: string };
    history?: Array<{ role: "user" | "assistant"; content: string }>;
    previous_response_id?: string;
  }) =>
    request<{
      backend: string;
      state_mode: string;
      instruction_role: string;
      assistant_instruction: string;
      user_prompt: string;
      messages: Array<{ role: string; content: string }>;
      fallback_messages: Array<{ role: string; content: string }>;
      context: GraphAskContext;
      request: Record<string, unknown>;
    }>("/api/chat_context_preview", body),
  graphRagContext: (body: {
    question: string;
    selected_nodes: string[];
    query: string;
    depth: number;
    max_nodes: number;
    max_edges: number;
  }) => request<{ context: GraphAskContext }>("/api/graph_rag_context", body),
  generateImage: (body: {
    run: string;
    chat_id: string;
    prompt: string;
    model_config: ModelRole & { api_key?: string };
    previous_response_id?: string;
    action?: "auto" | "generate" | "edit";
    size?: string;
    quality?: "auto" | "low" | "medium" | "high" | string;
    output_format?: "png" | "jpeg" | "webp" | string;
    background?: "auto" | "opaque" | "transparent" | string;
    partial_images?: number;
  }) => request<ImageGenerationResponse>("/api/generate_image", body),
  neighborhood: (body: { nodes: string[]; depth: number; limit: number }) =>
    request<GraphPayload & { focus_nodes?: string[] }>("/api/neighborhood", body),
  path: (body: { source: string; target: string; k: number; cutoff: number }) =>
    request<GraphPayload & { paths?: string[][]; resolved_source?: string; resolved_target?: string }>("/api/path", body),
  multipath: (body: {
    nodes?: string[];
    query?: string;
    mode: "pairwise" | "sequence" | "stochastic";
    cutoff: number;
    anchor_limit?: number;
    sample_count?: number;
  }) =>
    request<GraphPayload & { anchors?: string[]; paths?: string[][]; connectors?: PathConnector[] }>("/api/multipath", body),
  bridgeSuggestions: (body: { selected_nodes?: string[]; limit?: number }) =>
    request<{ ideas: BridgeIdea[] }>("/api/bridge_suggestions", body),
  ideate: (body: {
    topic: string;
    strategy: string;
    budget_calls: number;
    max_iters: number;
    out: string;
    clear_output?: boolean;
  }) => request<JobStatus>("/api/ideate", body),
  job: (id: string) => request<JobStatus>(`/api/job?id=${encodeURIComponent(id)}`),
  stopJob: (id: string) => request<JobStatus>("/api/stop_job", { id }),
  profileGraph: (body: ProfileOptions) => request<ProfileJobStatus>("/api/profile_graph", body),
  profileJob: (id: string) => request<ProfileJobStatus>(`/api/profile_job?id=${encodeURIComponent(id)}`),
  stopProfileJob: (id: string) => request<ProfileJobStatus>("/api/stop_profile_job", { id }),
  runAnalysis: (body: { run: string; analyses?: string[]; embed_model?: string }) =>
    request<RunAnalysisJobStatus>("/api/run_analysis", body),
  runAnalysisJob: (id: string) => request<RunAnalysisJobStatus>(`/api/run_analysis_job?id=${encodeURIComponent(id)}`),
  stopRunAnalysis: (id: string) => request<RunAnalysisJobStatus>("/api/stop_run_analysis", { id }),
  synthesize: (body: {
    run: string;
    task?: string;
    style?: string;
    backend?: string;
    model?: string;
    base_url?: string;
    api_key_env?: string;
    model_config?: ModelRole;
    temperature?: number | string;
    max_tokens?: number | string;
    max_leads?: number | string;
    mine?: boolean;
    no_insights?: boolean;
    out?: string;
  }) => request<SynthesisJobStatus>("/api/synthesize", body),
  synthesisJob: (id: string) => request<SynthesisJobStatus>(`/api/synthesis_job?id=${encodeURIComponent(id)}`),
  stopSynthesis: (id: string) => request<SynthesisJobStatus>("/api/stop_synthesis", { id }),
  profileReports: (run: string) => request<{ run: string; reports: ProfileArtifacts[] }>("/api/profile_reports", { run }),
  profileReport: (out: string) => request<ProfileReportPayload>("/api/profile_report", { out }),
  reportAssetUrl: (out: string, file: string) =>
    `/api/report_asset?out=${encodeURIComponent(out)}&file=${encodeURIComponent(file)}`,
  runAssetUrl: (run: string, file: string) =>
    `/api/run_asset?run=${encodeURIComponent(run)}&file=${encodeURIComponent(file)}`,
  modelStatus: (role: ModelRole) =>
    request<{ ok: boolean; url?: string; message?: string; models?: string[] }>("/api/model_status", { role }),
  modelProbe: (role: ModelRole) => request<ModelProbe>("/api/model_probe", { role }),
  configPreview: (roles: Record<string, ModelRole>) =>
    request<{ config: string }>("/api/config_preview", { roles }),
  saveConfig: (roles: Record<string, ModelRole>) =>
    request<{ config: string; path: string }>("/api/save_config", { roles }),
};
