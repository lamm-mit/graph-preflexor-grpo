import { create } from "zustand";
import type { ChatMessage, GraphPayload, ModelRole, SearchResult, VisualState } from "./types";

const GEMMA_CHAT_MODEL = "google/gemma-4-E4B-it";

const defaultRole: ModelRole = {
  provider: "openai",
  model: GEMMA_CHAT_MODEL,
  base_url: "http://localhost:1234/v1",
  backend: "responses",
  api_key_env: "",
  temperature: 0.3,
  max_tokens: 20000,
  reasoning_effort: "",
};

const questionerRole: ModelRole = { ...defaultRole, model: GEMMA_CHAT_MODEL, base_url: "http://localhost:1234/v1" };

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
  setSelectedNodes: (ids: string[]) => void;
  clearSelection: () => void;
  setSearchResults: (results: SearchResult[]) => void;
  setHighlightedPaths: (paths: string[][]) => void;
  setVisual: (visual: Partial<VisualState>) => void;
  setRoles: (roles: Record<string, ModelRole>) => void;
  updateRole: (name: string, role: ModelRole) => void;
  setChatRole: (role: string) => void;
  addChatMessage: (message: Omit<ChatMessage, "id">) => string;
  updateChatMessage: (id: string, message: Partial<ChatMessage>) => void;
  setChatMessages: (messages: ChatMessage[]) => void;
  resetChat: () => void;
};

export const useExplorerStore = create<ExplorerState>((set) => ({
  graph: null,
  selectedNode: null,
  selectedNodes: [],
  searchResults: [],
  highlightedPaths: [],
  visual: {
    viewMode: "2d",
    canvasTheme: "dark",
    layout: "force",
    colorBy: "degree",
    colorPalette: "atlas",
    sizeBy: "degree",
    edgeOpacity: 0.18,
    edgeWidth: 1,
    edgeStyle: "straight",
  },
  roles: {
    chat: { ...questionerRole, role: "chat" },
    graph_qa: { ...questionerRole },
    generator: {
      ...defaultRole,
      model: "lamm-mit/Graph-Preflexor-3b_08012026",
      base_url: "http://localhost:1234/v1",
      temperature: 0.1,
      max_tokens: 8000,
    },
    questioner: questionerRole,
    judge: {
      ...defaultRole,
      model: "gpt-4o",
      base_url: "https://api.openai.com/v1",
      api_key_env: "OPENAI_API_KEY",
      temperature: 0,
      max_tokens: 20000,
    },
    baseline: defaultRole,
  },
  chatRole: "chat",
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
  setSelectedNodes: (ids) => {
    const selectedNodes = Array.from(new Set(ids.filter(Boolean)));
    set({ selectedNode: selectedNodes[0] || null, selectedNodes });
  },
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
    set((state) => ({ chatMessages: [...state.chatMessages, { ...message, id, created_at: message.created_at || Date.now() / 1000 }] }));
    return id;
  },
  updateChatMessage: (id, message) =>
    set((state) => ({
      chatMessages: state.chatMessages.map((item) => (item.id === id ? { ...item, ...message } : item)),
    })),
  setChatMessages: (chatMessages) => set({ chatMessages }),
  resetChat: () => set({ chatMessages: [] }),
}));
