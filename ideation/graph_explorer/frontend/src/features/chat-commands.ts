export type ChatCommandId =
  | "clear"
  | "followups"
  | "insights"
  | "run"
  | "synthesize"
  | "skills"
  | "none"
  | "rag"
  | "focus"
  | "nodes"
  | "help";

export type ChatCommandSpec = {
  id: ChatCommandId;
  command: string;
  label: string;
  detail: string;
  insert?: string;
};

export const CHAT_COMMANDS: ChatCommandSpec[] = [
  { id: "clear", command: "/clear", label: "Clear chat", detail: "Reset this browser-side chat thread." },
  { id: "followups", command: "/followups", label: "Generate follow-ups", detail: "Ask the active graph model for next query ideas." },
  { id: "insights", command: "/insights", label: "Summarize insights", detail: "Mine or summarize the active run's structural insights.", insert: "/insights " },
  { id: "run", command: "/run <topic>", label: "New exploration run", detail: "Start a guided exploration run from chat.", insert: "/run " },
  { id: "synthesize", command: "/synthesize <task>", label: "Run synthesis", detail: "Generate a synthesis answer from the active run and post it here.", insert: "/synthesize " },
  { id: "skills", command: "/skills", label: "Browse skills", detail: "Search local skills, preview instructions, and attach one to the next chat turn.", insert: "/skills " },
  { id: "none", command: "/none", label: "No retrieval", detail: "Use only selected nodes; with no selection this is regular chat." },
  { id: "rag", command: "/rag", label: "Graph-RAG retrieval", detail: "Retrieve broader semantic neighborhoods and path connectors." },
  { id: "focus", command: "/focus", label: "Focused selection", detail: "Use selected nodes, an optional focus query, and compact neighborhoods." },
  { id: "nodes", command: "/nodes 160", label: "Context nodes", detail: "Set the graph context size to 160 nodes.", insert: "/nodes 160" },
  { id: "help", command: "/help", label: "Show help", detail: "Insert a compact command reference." },
];

export function filterChatCommands(query: string) {
  const needle = query.trim().toLowerCase();
  if (!needle) return CHAT_COMMANDS;
  return CHAT_COMMANDS.filter(
    (item) =>
      item.command.toLowerCase().includes(needle) ||
      item.label.toLowerCase().includes(needle) ||
      item.detail.toLowerCase().includes(needle),
  );
}

export function commandHelpText() {
  return CHAT_COMMANDS.map((item) => `${item.command} - ${item.detail}`).join("\n");
}
