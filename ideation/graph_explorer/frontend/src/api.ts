import type {
  BridgeIdea,
  ConfigPayload,
  EmbeddingStatus,
  GraphAskContext,
  GraphFileSummary,
  GraphPayload,
  JobStatus,
  ModelProbe,
  ModelRole,
  PathConnector,
  ProfileJobStatus,
  ProfileOptions,
  ProfileReportPayload,
  ProfileArtifacts,
  RunSummary,
  SearchResult,
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
  runGraphs: (run: string) => request<{ run: string; graphs: GraphFileSummary[] }>("/api/run_graphs", { run }),
  graphmlFiles: () => request<{ graphs: GraphFileSummary[] }>("/api/graphml_files", {}),
  clearGraph: () => request<{ ok: boolean }>("/api/clear_graph", {}),
  uploadGraphml: (name: string, graphml: string) =>
    request<GraphPayload>("/api/load_graphml", { name, graphml }),
  search: (query: string, limit = 40) =>
    request<{ results: SearchResult[] }>("/api/search", { query, limit }),
  embeddingStatus: () => request<EmbeddingStatus>("/api/embedding_status"),
  startEmbeddings: (body?: { model?: string; force?: boolean; batch_size?: number }) =>
    request<EmbeddingStatus>("/api/embedding_index", body || { model: "auto" }),
  ask: (body: {
    question: string;
    selected_nodes: string[];
    query: string;
    depth: number;
    max_nodes: number;
    max_edges: number;
    context_mode?: "focused" | "graph_rag";
    report_context?: { out: string; max_chars?: number } | null;
    model_config: ModelRole & { api_key?: string };
    history?: Array<{ role: "user" | "assistant"; content: string }>;
  }) => request<{ answer: string; context: GraphAskContext }>("/api/ask", body),
  graphRagContext: (body: {
    question: string;
    selected_nodes: string[];
    query: string;
    depth: number;
    max_nodes: number;
    max_edges: number;
  }) => request<{ context: GraphAskContext }>("/api/graph_rag_context", body),
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
  }) => request<JobStatus>("/api/ideate", body),
  job: (id: string) => request<JobStatus>(`/api/job?id=${encodeURIComponent(id)}`),
  stopJob: (id: string) => request<JobStatus>("/api/stop_job", { id }),
  profileGraph: (body: ProfileOptions) => request<ProfileJobStatus>("/api/profile_graph", body),
  profileJob: (id: string) => request<ProfileJobStatus>(`/api/profile_job?id=${encodeURIComponent(id)}`),
  stopProfileJob: (id: string) => request<ProfileJobStatus>("/api/stop_profile_job", { id }),
  profileReports: (run: string) => request<{ run: string; reports: ProfileArtifacts[] }>("/api/profile_reports", { run }),
  profileReport: (out: string) => request<ProfileReportPayload>("/api/profile_report", { out }),
  reportAssetUrl: (out: string, file: string) =>
    `/api/report_asset?out=${encodeURIComponent(out)}&file=${encodeURIComponent(file)}`,
  modelStatus: (role: ModelRole) =>
    request<{ ok: boolean; url?: string; message?: string; models?: string[] }>("/api/model_status", { role }),
  modelProbe: (role: ModelRole) => request<ModelProbe>("/api/model_probe", { role }),
  configPreview: (roles: Record<string, ModelRole>) =>
    request<{ config: string }>("/api/config_preview", { roles }),
  saveConfig: (roles: Record<string, ModelRole>) =>
    request<{ config: string; path: string }>("/api/save_config", { roles }),
};
