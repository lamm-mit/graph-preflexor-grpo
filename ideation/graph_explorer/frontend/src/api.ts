import type { ConfigPayload, GraphPayload, JobStatus, ModelRole, PathConnector, SearchResult } from "./types";

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
  config: () => request<ConfigPayload>("/api/config"),
  loadRun: (run: string) => request<GraphPayload>("/api/load_run", { run }),
  uploadGraphml: (name: string, graphml: string) =>
    request<GraphPayload>("/api/load_graphml", { name, graphml }),
  search: (query: string, limit = 40) =>
    request<{ results: SearchResult[] }>("/api/search", { query, limit }),
  ask: (body: {
    question: string;
    selected_nodes: string[];
    query: string;
    depth: number;
    max_nodes: number;
    max_edges: number;
    model_config: ModelRole & { api_key?: string };
  }) => request<{ answer: string; context: { node_count: number; edge_count: number } }>("/api/ask", body),
  neighborhood: (body: { nodes: string[]; depth: number; limit: number }) =>
    request<GraphPayload & { focus_nodes?: string[] }>("/api/neighborhood", body),
  path: (body: { source: string; target: string; k: number; cutoff: number }) =>
    request<GraphPayload & { paths?: string[][] }>("/api/path", body),
  multipath: (body: { nodes?: string[]; query?: string; mode: "pairwise" | "sequence"; cutoff: number; anchor_limit?: number }) =>
    request<GraphPayload & { anchors?: string[]; paths?: string[][]; connectors?: PathConnector[] }>("/api/multipath", body),
  ideate: (body: {
    topic: string;
    strategy: string;
    budget_calls: number;
    max_iters: number;
    out: string;
  }) => request<JobStatus>("/api/ideate", body),
  job: (id: string) => request<JobStatus>(`/api/job?id=${encodeURIComponent(id)}`),
  stopJob: (id: string) => request<JobStatus>("/api/stop_job", { id }),
  modelStatus: (role: ModelRole) =>
    request<{ ok: boolean; url?: string; message?: string; models?: string[] }>("/api/model_status", { role }),
  configPreview: (roles: Record<string, ModelRole>) =>
    request<{ config: string }>("/api/config_preview", { roles }),
  saveConfig: (roles: Record<string, ModelRole>) =>
    request<{ config: string; path: string }>("/api/save_config", { roles }),
};
