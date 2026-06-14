import { create } from "zustand";
import type { ChatMessage, GraphPayload, ModelRole, SearchResult, VisualState } from "./types";

const defaultRole: ModelRole = {
  provider: "openai",
  model: "meta-llama/Llama-3.2-3B-Instruct",
  base_url: "http://localhost:8000/v1",
  api_key_env: "",
  temperature: 0.3,
  max_tokens: 1800,
  reasoning_effort: "",
};

const questionerRole: ModelRole = { ...defaultRole, model: "Qwen/Qwen3-0.6B", base_url: "http://localhost:1234/v1" };

export type ExplorerState = {
  graph: GraphPayload | null;
  selectedNode: string | null;
  selectedNodes: string[];
  searchResults: SearchResult[];
  highlightedPaths: string[][];
  visual: VisualState;
  roles: Record<string, ModelRole>;
  chatRole: string;
  chatMessages: ChatMessage[];
  setGraph: (graph: GraphPayload | null) => void;
  setSelectedNode: (id: string | null, append?: boolean) => void;
  clearSelection: () => void;
  setSearchResults: (results: SearchResult[]) => void;
  setHighlightedPaths: (paths: string[][]) => void;
  setVisual: (visual: Partial<VisualState>) => void;
  setRoles: (roles: Record<string, ModelRole>) => void;
  updateRole: (name: string, role: ModelRole) => void;
  setChatRole: (role: string) => void;
  addChatMessage: (message: Omit<ChatMessage, "id">) => string;
  updateChatMessage: (id: string, message: Partial<ChatMessage>) => void;
  resetChat: () => void;
};

export const useExplorerStore = create<ExplorerState>((set) => ({
  graph: null,
  selectedNode: null,
  selectedNodes: [],
  searchResults: [],
  highlightedPaths: [],
  visual: {
    viewMode: "3d",
    layout: "force",
    colorBy: "degree",
    sizeBy: "degree",
    edgeOpacity: 0.12,
  },
  roles: {
    graph_qa: { ...questionerRole },
    generator: {
      ...defaultRole,
      model: "lamm-mit/Graph-Preflexor-3b_08012026",
      base_url: "http://localhost:1234/v1",
    },
    questioner: questionerRole,
    judge: {
      ...defaultRole,
      model: "gpt-4o",
      base_url: "https://api.openai.com/v1",
      api_key_env: "OPENAI_API_KEY",
      temperature: 0,
      max_tokens: 4000,
    },
    baseline: defaultRole,
  },
  chatRole: "graph_qa",
  chatMessages: [],
  setGraph: (graph) =>
    set({
      graph,
      selectedNode: null,
      selectedNodes: [],
      searchResults: [],
      highlightedPaths: [],
    }),
  setSelectedNode: (id, append = false) =>
    set((state) => {
      if (!id) return { selectedNode: null, selectedNodes: [] };
      const selectedNodes = append
        ? Array.from(new Set([...state.selectedNodes, id]))
        : [id];
      return { selectedNode: id, selectedNodes };
    }),
  clearSelection: () => set({ selectedNode: null, selectedNodes: [] }),
  setSearchResults: (searchResults) => set({ searchResults }),
  setHighlightedPaths: (highlightedPaths) => set({ highlightedPaths }),
  setVisual: (visual) => set((state) => ({ visual: { ...state.visual, ...visual } })),
  setRoles: (roles) =>
    set((state) => ({
      roles: {
        ...state.roles,
        ...roles,
      },
    })),
  updateRole: (name, role) => set((state) => ({ roles: { ...state.roles, [name]: role } })),
  setChatRole: (chatRole) => set({ chatRole }),
  addChatMessage: (message) => {
    const id = crypto.randomUUID();
    set((state) => ({ chatMessages: [...state.chatMessages, { ...message, id }] }));
    return id;
  },
  updateChatMessage: (id, message) =>
    set((state) => ({
      chatMessages: state.chatMessages.map((item) => (item.id === id ? { ...item, ...message } : item)),
    })),
  resetChat: () => set({ chatMessages: [] }),
}));
