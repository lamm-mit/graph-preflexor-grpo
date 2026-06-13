import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (id) => document.getElementById(id);

const palette = [
  0x4fb477, 0xe1a13a, 0x64a7d9, 0xd96868, 0xb58ad7, 0x6fc7bd,
  0xc0c46a, 0xe58f6e, 0x8fa8ff, 0x78a65a, 0xcf79a5, 0x9bb0bd
];

const metricLabels = {
  component: "Component",
  community: "Community",
  degree: "Degree",
  pagerank: "PageRank",
  core: "Core",
  betweenness: "Betweenness",
  closeness: "Closeness",
  eigenvector: "Eigenvector",
  clustering: "Clustering",
  iter: "Iteration",
  depth: "Depth",
  constant: "Constant",
};

const state = {
  fullGraph: null,
  activeGraph: null,
  nodeById: new Map(),
  edgeById: new Map(),
  layout: new Map(),
  visual: {
    layoutMode: "force3d",
    colorBy: "component",
    sizeBy: "degree",
    edgeColorBy: "constant",
    lowColor: "#64a7d9",
    highColor: "#e1a13a",
    edgeColor: "#46505c",
    backgroundColor: "#090b0d",
    nodeScale: 1,
    edgeOpacity: 0.34,
    physicsTicks: 90,
    springLength: 18,
    repulsion: 42,
    gravity: 0.004,
  },
  selectedNodes: new Set(),
  pinnedNodes: new Set(),
  pinnedPositions: new Map(),
  boxSelectMode: false,
  boxDrag: null,
  timelineTimer: null,
  modelRoles: {},
  source: null,
  target: null,
  hover: null,
  mode: "whole graph",
  currentJob: null,
};

let scene, camera, renderer, controls, raycaster, pointer;
let nodeMesh = null;
let edgeLines = null;
let animationStarted = false;

const modelPresets = {
  graph1234: {
    provider: "openai",
    model: "lamm-mit/Graph-Preflexor-3b_08012026",
    base_url: "http://localhost:1234/v1",
    api_key_env: "",
    temperature: 0.3,
    max_tokens: 8000,
    reasoning_effort: "",
  },
  qwen1234: {
    provider: "openai",
    model: "Qwen/Qwen3-0.6B",
    base_url: "http://localhost:1234/v1",
    api_key_env: "",
    temperature: 0.3,
    max_tokens: 8000,
    reasoning_effort: "",
  },
  llama8000: {
    provider: "openai",
    model: "meta-llama/Llama-3.2-3B-Instruct",
    base_url: "http://localhost:8000/v1",
    api_key_env: "",
    temperature: 0.3,
    max_tokens: 1800,
    reasoning_effort: "",
  },
  openaiJudge: {
    provider: "openai",
    model: "gpt-4o",
    base_url: "https://api.openai.com/v1",
    api_key_env: "OPENAI_API_KEY",
    temperature: 0,
    max_tokens: 4000,
    reasoning_effort: "",
  },
  embeddingGemma: {
    provider: "embedding",
    model: "google/embeddinggemma-300m",
    base_url: "",
    api_key_env: "",
    temperature: "",
    max_tokens: "",
    reasoning_effort: "",
  },
};

const defaultModelRoles = {
  generator: { ...modelPresets.graph1234, role: "generator" },
  questioner: { ...modelPresets.qwen1234, role: "questioner" },
  graph_qa: { ...modelPresets.llama8000, role: "graph_qa" },
  judge: { ...modelPresets.openaiJudge, role: "judge" },
  baseline: { ...modelPresets.llama8000, role: "baseline" },
  embedder: { ...modelPresets.embeddingGemma, role: "embedder" },
};

function initScene() {
  const viewport = $("viewport");
  scene = new THREE.Scene();
  scene.background = new THREE.Color(state.visual.backgroundColor);
  camera = new THREE.PerspectiveCamera(58, viewport.clientWidth / viewport.clientHeight, 0.1, 6000);
  camera.position.set(0, 0, 260);
  renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(viewport.clientWidth, viewport.clientHeight);
  viewport.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.rotateSpeed = 0.55;
  controls.zoomSpeed = 0.9;

  raycaster = new THREE.Raycaster();
  pointer = new THREE.Vector2();

  const hemi = new THREE.HemisphereLight(0xffffff, 0x1b2028, 1.7);
  scene.add(hemi);
  const key = new THREE.DirectionalLight(0xffffff, 1.8);
  key.position.set(80, 110, 140);
  scene.add(key);

  window.addEventListener("resize", resize);
  renderer.domElement.addEventListener("pointermove", onPointerMove);
  renderer.domElement.addEventListener("pointerdown", onPointerDown);
  renderer.domElement.addEventListener("pointerup", onPointerUp);
  renderer.domElement.addEventListener("dblclick", () => focusNeighborhood());

  if (!animationStarted) {
    animationStarted = true;
    animate();
  }
}

function resize() {
  const viewport = $("viewport");
  if (!viewport || !renderer) return;
  const w = Math.max(1, viewport.clientWidth);
  const h = Math.max(1, viewport.clientHeight);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

async function api(path, body = null) {
  const opts = body === null
    ? {}
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const res = await fetch(path, opts);
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

function saveModelPrefs() {
  const prefs = {
    provider: $("providerInput").value,
    model: $("modelInput").value,
    base_url: $("baseUrlInput").value,
    effort: $("effortInput").value,
    temperature: $("temperatureInput").value,
    max_tokens: $("maxTokensInput").value,
  };
  localStorage.setItem("graphExplorerModelPrefs", JSON.stringify(prefs));
}

function loadModelPrefs() {
  try {
    const prefs = JSON.parse(localStorage.getItem("graphExplorerModelPrefs") || "{}");
    if (prefs.provider) $("providerInput").value = prefs.provider;
    if (prefs.model) $("modelInput").value = prefs.model;
    if (prefs.base_url) $("baseUrlInput").value = prefs.base_url;
    if (prefs.effort) $("effortInput").value = prefs.effort;
    if (prefs.temperature) $("temperatureInput").value = prefs.temperature;
    if (prefs.max_tokens) $("maxTokensInput").value = prefs.max_tokens;
  } catch (_) {
    /* ignore malformed localStorage */
  }
}

function saveModelRoles() {
  localStorage.setItem("graphExplorerModelRoles", JSON.stringify(state.modelRoles));
}

function loadModelRoles() {
  try {
    state.modelRoles = JSON.parse(localStorage.getItem("graphExplorerModelRoles") || "{}");
  } catch (_) {
    state.modelRoles = {};
  }
  state.modelRoles = { ...defaultModelRoles, ...state.modelRoles };
  syncRoleFields();
}

async function importConfigRoles() {
  try {
    const data = await api("/api/config");
    if (data.roles) {
      state.modelRoles = { ...state.modelRoles, ...data.roles };
      saveModelRoles();
      syncRoleFields();
      $("modelStatus").textContent = data.exists
        ? `Imported roles from ${data.path}`
        : `No config.yaml found at ${data.path}; using defaults.`;
    }
  } catch (err) {
    $("modelStatus").textContent = err.message;
  }
}

function currentRoleName() {
  return $("modelRoleInput")?.value || "graph_qa";
}

function readRoleFields() {
  const role = currentRoleName();
  state.modelRoles[role] = {
    role,
    provider: $("roleProviderInput").value,
    model: $("roleModelInput").value.trim(),
    base_url: $("roleBaseUrlInput").value.trim(),
    api_key_env: $("roleApiKeyEnvInput").value.trim(),
    temperature: $("roleTempInput").value,
    max_tokens: $("roleTokensInput").value,
    reasoning_effort: $("roleEffortInput").value.trim(),
  };
  saveModelRoles();
  return state.modelRoles[role];
}

function syncRoleFields() {
  const role = currentRoleName();
  const cfg = state.modelRoles[role] || defaultModelRoles[role] || {};
  if (!$("roleProviderInput")) return;
  $("roleProviderInput").value = cfg.provider || "openai";
  $("roleModelInput").value = cfg.model || "";
  $("roleBaseUrlInput").value = cfg.base_url || "";
  $("roleApiKeyEnvInput").value = cfg.api_key_env || "";
  $("roleTempInput").value = cfg.temperature ?? "";
  $("roleTokensInput").value = cfg.max_tokens ?? "";
  $("roleEffortInput").value = cfg.reasoning_effort || "";
}

function applyPresetToRole() {
  const preset = modelPresets[$("modelPresetInput").value];
  if (!preset) return;
  const role = currentRoleName();
  state.modelRoles[role] = { ...preset, role };
  saveModelRoles();
  syncRoleFields();
  $("modelPresetInput").value = "";
}

function applyRoleToAsk() {
  const cfg = readRoleFields();
  $("providerInput").value = cfg.provider === "hf" ? "hf" : "openai";
  $("modelInput").value = cfg.model || "";
  $("baseUrlInput").value = cfg.base_url || "";
  $("effortInput").value = cfg.reasoning_effort || "";
  if (cfg.temperature !== "") $("temperatureInput").value = cfg.temperature;
  if (cfg.max_tokens !== "") $("maxTokensInput").value = cfg.max_tokens;
  saveModelPrefs();
}

async function checkModelServer() {
  const cfg = readRoleFields();
  $("modelStatus").textContent = "Checking...";
  try {
    const data = await api("/api/model_status", { role: cfg, timeout: 2.0 });
    const models = (data.models || []).slice(0, 12).join("\n");
    $("modelStatus").textContent = [
      data.ok ? "OK" : "Unavailable",
      data.url || cfg.provider,
      data.message || "",
      models ? `\nModels:\n${models}` : "",
    ].filter(Boolean).join(" | ");
  } catch (err) {
    $("modelStatus").textContent = err.message;
  }
}

async function previewConfig(write = false) {
  readRoleFields();
  $("configPreview").textContent = write ? "Writing config.yaml..." : "Building preview...";
  try {
    const data = await api(write ? "/api/save_config" : "/api/config_preview", { roles: state.modelRoles });
    $("configPreview").textContent = data.config || "";
    if (write) $("modelStatus").textContent = `Wrote ${data.path}`;
  } catch (err) {
    $("configPreview").textContent = err.message;
  }
}

function visualControlMap() {
  return {
    layoutMode: "layoutModeInput",
    colorBy: "colorByInput",
    sizeBy: "sizeByInput",
    edgeColorBy: "edgeColorByInput",
    lowColor: "lowColorInput",
    highColor: "highColorInput",
    edgeColor: "edgeColorInput",
    backgroundColor: "backgroundColorInput",
    nodeScale: "nodeScaleInput",
    edgeOpacity: "edgeOpacityInput",
    physicsTicks: "physicsTicksInput",
    springLength: "springLengthInput",
    repulsion: "repulsionInput",
    gravity: "gravityInput",
  };
}

function readVisualControls() {
  const map = visualControlMap();
  for (const [key, id] of Object.entries(map)) {
    const el = $(id);
    if (!el) continue;
    const numeric = ["nodeScale", "edgeOpacity", "physicsTicks", "springLength", "repulsion", "gravity"].includes(key);
    state.visual[key] = numeric ? Number(el.value || state.visual[key]) : el.value;
  }
  state.visual.nodeScale = Math.max(0.2, state.visual.nodeScale || 1);
  state.visual.edgeOpacity = Math.min(1, Math.max(0.02, state.visual.edgeOpacity || 0.34));
  state.visual.physicsTicks = Math.max(0, Math.floor(state.visual.physicsTicks || 0));
  state.visual.springLength = Math.max(4, state.visual.springLength || 18);
  state.visual.repulsion = Math.max(0, state.visual.repulsion || 42);
  state.visual.gravity = Math.max(0, state.visual.gravity || 0.004);
}

function saveVisualPrefs() {
  readVisualControls();
  localStorage.setItem("graphExplorerVisualPrefs", JSON.stringify(state.visual));
}

function loadVisualPrefs() {
  try {
    const prefs = JSON.parse(localStorage.getItem("graphExplorerVisualPrefs") || "{}");
    const map = visualControlMap();
    for (const [key, id] of Object.entries(map)) {
      if (prefs[key] === undefined || !$(id)) continue;
      $(id).value = prefs[key];
    }
  } catch (_) {
    /* ignore malformed localStorage */
  }
  readVisualControls();
  if (scene) scene.background = new THREE.Color(state.visual.backgroundColor);
}

function fmt(n, digits = 0) {
  if (n === undefined || n === null || Number.isNaN(Number(n))) return "";
  return Number(n).toLocaleString(undefined, { maximumFractionDigits: digits });
}

function nodeLabel(node) {
  return node?.label || node?.id || "";
}

function numberValue(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function metricValue(node, key) {
  if (!node || key === "constant") return 1;
  if (key === "component") return numberValue(node.component, 0);
  if (key === "community") return numberValue(node.community, 0);
  return numberValue(node[key], numberValue(node.attrs?.[key], 0));
}

function metricRange(graph, key) {
  if (!graph?.nodes?.length || key === "constant") return { min: 0, max: 1 };
  const values = graph.nodes
    .map((node) => metricValue(node, key))
    .filter((v) => Number.isFinite(v))
    .sort((a, b) => a - b);
  if (!values.length) return { min: 0, max: 1 };
  const lo = values[Math.floor(values.length * 0.02)];
  const hi = values[Math.max(0, Math.ceil(values.length * 0.98) - 1)];
  return lo === hi ? { min: values[0], max: values[values.length - 1] || values[0] + 1 } : { min: lo, max: hi };
}

function normalized(value, range) {
  if (!range || range.max === range.min) return 0.5;
  return Math.min(1, Math.max(0, (value - range.min) / (range.max - range.min)));
}

function colorInt(hex) {
  return new THREE.Color(hex).getHex();
}

function sequentialColor(value, range) {
  const t = normalized(value, range);
  const low = new THREE.Color(state.visual.lowColor);
  const high = new THREE.Color(state.visual.highColor);
  return low.lerp(high, t).getHex();
}

function categoryColor(value) {
  return palette[Math.abs(Math.floor(numberValue(value, 0))) % palette.length];
}

function hashCategoryColor(text) {
  return palette[Math.floor(hash01(String(text), 31) * palette.length) % palette.length];
}

function colorForNode(node, range) {
  if (node.id === state.source) return 0x4fb477;
  if (node.id === state.target) return 0xe1a13a;
  if (state.selectedNodes.has(node.id)) return 0x8fa8ff;
  if (node.id === state.hover) return 0xffffff;
  const key = state.visual.colorBy;
  if (key === "component" || key === "community") return categoryColor(metricValue(node, key));
  return sequentialColor(metricValue(node, key), range);
}

function sizeForNode(node, range) {
  const key = state.visual.sizeBy;
  const base = key === "constant" ? 0.5 : normalized(metricValue(node, key), range);
  const size = 1.15 + Math.pow(base, 0.72) * 3.25;
  return size * state.visual.nodeScale;
}

function colorForEdge(edge, ranges) {
  if ((edge.source === state.source && edge.target === state.target) ||
      (edge.source === state.target && edge.target === state.source)) {
    return 0xe1a13a;
  }
  if (state.visual.edgeColorBy === "relation") return hashCategoryColor(edge.relation);
  if (state.visual.edgeColorBy === "iter" || state.visual.edgeColorBy === "depth") {
    return sequentialColor(numberValue(edge[state.visual.edgeColorBy], 0), ranges.edge);
  }
  return colorInt(state.visual.edgeColor);
}

function renderVisualLegend(graph) {
  const legend = $("visualLegend");
  if (!legend) return;
  const colorKey = state.visual.colorBy;
  const sizeKey = state.visual.sizeBy;
  const range = metricRange(graph, colorKey);
  if (colorKey === "component" || colorKey === "community") {
    legend.innerHTML = `<div class="legend-row"><span>${metricLabels[colorKey]}</span>${palette.slice(0, 8).map((c, i) =>
      `<i style="background:#${c.toString(16).padStart(6, "0")}"></i><small>${i}</small>`).join("")}</div>
      <div class="legend-note">size: ${metricLabels[sizeKey] || sizeKey}</div>`;
    return;
  }
  legend.innerHTML = `
    <div class="legend-gradient" style="background:linear-gradient(90deg, ${state.visual.lowColor}, ${state.visual.highColor})"></div>
    <div class="legend-range"><span>${metricLabels[colorKey] || colorKey}: ${fmt(range.min, 3)}</span><span>${fmt(range.max, 3)}</span></div>
    <div class="legend-note">size: ${metricLabels[sizeKey] || sizeKey}</div>
  `;
}

function setGraph(payload, mode = "whole graph", isFull = false) {
  if (isFull) state.fullGraph = payload;
  state.activeGraph = payload;
  state.mode = mode;
  state.nodeById = new Map(payload.nodes.map((n) => [n.id, n]));
  state.edgeById = new Map(payload.edges.map((e) => [e.id, e]));
  state.selectedNodes = new Set([...state.selectedNodes].filter((id) => state.nodeById.has(id)));
  state.pinnedNodes = new Set([...state.pinnedNodes].filter((id) => state.nodeById.has(id) || state.fullGraph?.nodes.some((n) => n.id === id)));
  $("graphName").textContent = `${payload.name || "graph"} | ${fmt(payload.stats.nodes)} nodes, ${fmt(payload.stats.edges)} edges`;
  $("renderMode").textContent = mode;
  renderStats(payload.stats);
  if (isFull) updateTimelineBounds(payload);
  buildLayout(payload);
  rebuildScene(true);
  updateSelectionHud();
}

function renderStats(stats) {
  const rows = [
    ["Nodes", fmt(stats.nodes)],
    ["Edges", fmt(stats.edges)],
    ["Components", fmt(stats.components)],
    ["Communities", fmt(stats.communities)],
    ["Largest", fmt(stats.largest_component)],
    ["Avg degree", fmt(stats.avg_degree, 2)],
    ["Max degree", fmt(stats.max_degree)],
    ["Max iter", fmt(stats.max_iter)],
    ["Density", fmt(stats.density, 5)],
  ];
  $("stats").innerHTML = rows.map(([k, v]) => `<div class="stat"><b>${v}</b><span>${k}</span></div>`).join("");
}

function inducedGraph(nodeIds, name = "filtered graph") {
  if (!state.fullGraph) return null;
  const keep = new Set(nodeIds);
  const nodes = state.fullGraph.nodes.filter((n) => keep.has(n.id));
  const edges = state.fullGraph.edges.filter((e) => keep.has(e.source) && keep.has(e.target));
  return {
    ...state.fullGraph,
    name,
    stats: computeStats(nodes, edges),
    nodes,
    edges,
  };
}

function computeStats(nodes, edges) {
  const ids = new Set(nodes.map((n) => n.id));
  const adj = new Map(nodes.map((n) => [n.id, new Set()]));
  for (const edge of edges) {
    if (!ids.has(edge.source) || !ids.has(edge.target)) continue;
    adj.get(edge.source).add(edge.target);
    adj.get(edge.target).add(edge.source);
  }
  const seen = new Set();
  const compSizes = [];
  for (const node of nodes) {
    if (seen.has(node.id)) continue;
    const q = [node.id];
    seen.add(node.id);
    let size = 0;
    while (q.length) {
      const cur = q.pop();
      size++;
      for (const nxt of adj.get(cur) || []) {
        if (!seen.has(nxt)) {
          seen.add(nxt);
          q.push(nxt);
        }
      }
    }
    compSizes.push(size);
  }
  compSizes.sort((a, b) => b - a);
  const degrees = nodes.map((n) => adj.get(n.id)?.size || 0);
  const communities = new Map();
  for (const node of nodes) {
    const c = numberValue(node.community, -1);
    communities.set(c, (communities.get(c) || 0) + 1);
  }
  const communitySizes = [...communities.values()].sort((a, b) => b - a);
  return {
    nodes: nodes.length,
    edges: edges.length,
    directed: Boolean(state.fullGraph?.stats?.directed),
    components: compSizes.length,
    largest_component: compSizes[0] || 0,
    density: nodes.length > 1 ? (2 * edges.length) / (nodes.length * (nodes.length - 1)) : 0,
    max_degree: Math.max(0, ...degrees),
    avg_degree: degrees.reduce((a, b) => a + b, 0) / Math.max(1, degrees.length),
    max_iter: Math.max(0, ...nodes.map((n) => numberValue(n.iter, 0))),
    component_sizes: compSizes.slice(0, 40),
    communities: communitySizes.length,
    community_sizes: communitySizes.slice(0, 40),
  };
}

function setDerivedGraph(payload, mode) {
  if (!payload) return;
  state.activeGraph = payload;
  state.mode = mode;
  state.nodeById = new Map(payload.nodes.map((n) => [n.id, n]));
  state.edgeById = new Map(payload.edges.map((e) => [e.id, e]));
  $("renderMode").textContent = mode;
  renderStats(payload.stats);
  buildLayout(payload);
  rebuildScene(true);
  updateSelectionHud();
}

function updateTimelineBounds(payload) {
  const input = $("filterIterInput");
  if (!input || !payload?.stats) return;
  const maxIter = Math.max(0, Number(payload.stats.max_iter || 0));
  input.max = String(maxIter);
  input.value = String(maxIter);
}

function hash01(str, salt = 0) {
  let h = 2166136261 + salt;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return ((h >>> 0) % 100000) / 100000;
}

function buildLayout(graph) {
  state.layout = new Map();
  const maxIter = Math.max(1, graph.stats.max_iter || 1);
  const maxDepth = Math.max(1, ...graph.nodes.map((node) => numberValue(node.depth, 0)));
  const maxCore = Math.max(1, ...graph.nodes.map((node) => numberValue(node.core, 0)));
  const degreeRange = metricRange(graph, "degree");
  const compSizes = graph.stats.component_sizes || [];
  const compCount = Math.max(1, compSizes.length || Math.max(...graph.nodes.map((node) => numberValue(node.component, 0))) + 1);
  const communityCount = Math.max(1, graph.stats.communities || Math.max(...graph.nodes.map((node) => numberValue(node.community, 0))) + 1);
  const gridCols = Math.ceil(Math.sqrt(compCount));
  for (const node of graph.nodes) {
    const comp = Math.max(0, numberValue(node.component, 0));
    const community = Math.max(0, numberValue(node.community, 0));
    const compAngle = comp * 2.3999632297;
    const communityAngle = community * 2.3999632297;
    const compRadius = Math.sqrt(comp + 1) * 26;
    const localAngle = hash01(node.id, 11) * Math.PI * 2;
    const degreeT = normalized(metricValue(node, "degree"), degreeRange);
    const localRadius = 18 + Math.sqrt(Math.max(1, node.degree || 1)) * 8 + hash01(node.id, 17) * 60;
    const zByIter = ((node.iter || 0) / maxIter - 0.5) * 150;
    const zNoise = (hash01(node.id, 23) - 0.5) * 90;
    let x, y, z;
    if (state.visual.layoutMode === "component") {
      x = Math.cos(compAngle) * (90 + compRadius) + Math.cos(localAngle) * localRadius * 0.9;
      y = Math.sin(compAngle) * (90 + compRadius) + Math.sin(localAngle) * localRadius * 0.9;
      z = zNoise * 0.45;
    } else if (state.visual.layoutMode === "community") {
      const communityRadius = 80 + Math.sqrt(community + 1) * 32;
      x = Math.cos(communityAngle) * communityRadius + Math.cos(localAngle) * localRadius * 0.85;
      y = Math.sin(communityAngle) * communityRadius + Math.sin(localAngle) * localRadius * 0.85;
      z = (community - communityCount / 2) * 12 + zNoise * 0.35;
    } else if (state.visual.layoutMode === "degreeRadial") {
      const radius = 38 + (1 - degreeT) * 230;
      x = Math.cos(localAngle + comp * 0.2) * radius;
      y = Math.sin(localAngle + comp * 0.2) * radius;
      z = (comp - compCount / 2) * 18 + zNoise * 0.18;
    } else if (state.visual.layoutMode === "coreShell") {
      const coreT = normalized(metricValue(node, "core"), { min: 0, max: maxCore });
      const radius = 42 + (1 - coreT) * 230;
      x = Math.cos(localAngle) * radius;
      y = Math.sin(localAngle) * radius;
      z = (coreT - 0.5) * 180 + zNoise * 0.12;
    } else if (state.visual.layoutMode === "timeline") {
      x = ((numberValue(node.iter, 0) / maxIter) - 0.5) * 520;
      y = (comp - compCount / 2) * 34 + Math.sin(localAngle) * 16;
      z = (degreeT - 0.5) * 170 + Math.cos(localAngle) * 16;
    } else if (state.visual.layoutMode === "depthBands") {
      const depth = numberValue(node.depth, 0);
      x = ((depth / maxDepth) - 0.5) * 460;
      y = Math.cos(localAngle) * (42 + degreeT * 90) + (comp % 3 - 1) * 42;
      z = Math.sin(localAngle) * (42 + degreeT * 90);
    } else if (state.visual.layoutMode === "grid") {
      const col = comp % gridCols;
      const row = Math.floor(comp / gridCols);
      x = (col - (gridCols - 1) / 2) * 180 + Math.cos(localAngle) * localRadius * 0.55;
      y = (row - Math.floor((compCount - 1) / gridCols) / 2) * 155 + Math.sin(localAngle) * localRadius * 0.55;
      z = zNoise * 0.4;
    } else {
      x = Math.cos(compAngle) * compRadius + Math.cos(localAngle) * localRadius;
      y = Math.sin(compAngle) * compRadius + Math.sin(localAngle) * localRadius;
      z = zByIter + zNoise;
    }
    const pinned = state.pinnedPositions.get(node.id);
    state.layout.set(node.id, pinned && state.visual.layoutMode === "force3d" ? {
      ...pinned,
      vx: 0,
      vy: 0,
      vz: 0,
    } : {
      x,
      y,
      z,
      vx: 0,
      vy: 0,
      vz: 0,
    });
  }
  if (state.visual.layoutMode === "force3d" && graph.nodes.length <= 900 && state.visual.physicsTicks > 0) {
    relaxLayout(graph, graph.nodes.length < 140 ? Math.max(state.visual.physicsTicks, 160) : state.visual.physicsTicks);
  }
}

function relaxLayout(graph, ticks) {
  const nodes = graph.nodes;
  const edges = graph.edges;
  const n = nodes.length;
  if (n > 700) return;
  const pos = nodes.map((node) => state.layout.get(node.id));
  const index = new Map(nodes.map((node, i) => [node.id, i]));
  const springs = edges
    .map((e) => [index.get(e.source), index.get(e.target)])
    .filter(([a, b]) => a !== undefined && b !== undefined);

  for (let t = 0; t < ticks; t++) {
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const a = pos[i], b = pos[j];
        let dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
        let d2 = dx * dx + dy * dy + dz * dz + 1.0;
        if (d2 > 18000) continue;
        const f = state.visual.repulsion / d2;
        if (!state.pinnedNodes.has(nodes[i].id)) {
          a.vx += dx * f; a.vy += dy * f; a.vz += dz * f;
        }
        if (!state.pinnedNodes.has(nodes[j].id)) {
          b.vx -= dx * f; b.vy -= dy * f; b.vz -= dz * f;
        }
      }
    }
    for (const [ia, ib] of springs) {
      const a = pos[ia], b = pos[ib];
      let dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) + 0.001;
      const rest = state.visual.springLength;
      const f = (dist - rest) * 0.018;
      dx /= dist; dy /= dist; dz /= dist;
      if (!state.pinnedNodes.has(nodes[ia].id)) {
        a.vx += dx * f; a.vy += dy * f; a.vz += dz * f;
      }
      if (!state.pinnedNodes.has(nodes[ib].id)) {
        b.vx -= dx * f; b.vy -= dy * f; b.vz -= dz * f;
      }
    }
    for (let i = 0; i < pos.length; i++) {
      const p = pos[i];
      if (state.pinnedNodes.has(nodes[i].id)) {
        p.vx = 0; p.vy = 0; p.vz = 0;
        continue;
      }
      p.vx += -p.x * state.visual.gravity;
      p.vy += -p.y * state.visual.gravity;
      p.vz += -p.z * state.visual.gravity;
      p.x += p.vx;
      p.y += p.vy;
      p.z += p.vz;
      p.vx *= 0.72;
      p.vy *= 0.72;
      p.vz *= 0.72;
    }
  }
}

function clearObjects() {
  for (const obj of [nodeMesh, edgeLines]) {
    if (!obj) continue;
    scene.remove(obj);
    if (obj.geometry) obj.geometry.dispose();
    if (obj.material) obj.material.dispose();
  }
  nodeMesh = null;
  edgeLines = null;
}

function rebuildScene(shouldFrame = true) {
  if (!state.activeGraph) return;
  readVisualControls();
  if (scene) scene.background = new THREE.Color(state.visual.backgroundColor);
  clearObjects();
  const graph = state.activeGraph;
  const nodeIndex = new Map(graph.nodes.map((n, i) => [n.id, i]));
  const colorRange = metricRange(graph, state.visual.colorBy);
  const sizeRange = metricRange(graph, state.visual.sizeBy);
  const edgeRange = state.visual.edgeColorBy === "depth" || state.visual.edgeColorBy === "iter"
    ? metricRange({ nodes: graph.edges }, state.visual.edgeColorBy)
    : { min: 0, max: 1 };

  const edgePositions = new Float32Array(graph.edges.length * 2 * 3);
  const edgeColors = new Float32Array(graph.edges.length * 2 * 3);
  const edgeColor = new THREE.Color();
  let ep = 0, ec = 0;
  for (const edge of graph.edges) {
    const a = state.layout.get(edge.source);
    const b = state.layout.get(edge.target);
    if (!a || !b) continue;
    edgePositions[ep++] = a.x; edgePositions[ep++] = a.y; edgePositions[ep++] = a.z;
    edgePositions[ep++] = b.x; edgePositions[ep++] = b.y; edgePositions[ep++] = b.z;
    edgeColor.setHex(colorForEdge(edge, { edge: edgeRange }));
    for (let i = 0; i < 2; i++) {
      edgeColors[ec++] = edgeColor.r; edgeColors[ec++] = edgeColor.g; edgeColors[ec++] = edgeColor.b;
    }
  }
  const edgeGeom = new THREE.BufferGeometry();
  edgeGeom.setAttribute("position", new THREE.BufferAttribute(edgePositions, 3));
  edgeGeom.setAttribute("color", new THREE.BufferAttribute(edgeColors, 3));
  edgeLines = new THREE.LineSegments(edgeGeom, new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: graph.edges.length > 10000 ? Math.min(state.visual.edgeOpacity, 0.18) : state.visual.edgeOpacity,
  }));
  scene.add(edgeLines);

  const geom = new THREE.SphereGeometry(1, 12, 8);
  const mat = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.65,
    metalness: 0.05,
  });
  nodeMesh = new THREE.InstancedMesh(geom, mat, graph.nodes.length);
  nodeMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  nodeMesh.userData.nodeIndex = nodeIndex;

  const dummy = new THREE.Object3D();
  const color = new THREE.Color();
  graph.nodes.forEach((node, i) => {
    const p = state.layout.get(node.id);
    const size = sizeForNode(node, sizeRange);
    dummy.position.set(p.x, p.y, p.z);
    dummy.scale.setScalar(node.id === state.source || node.id === state.target ? size * 1.75 : size);
    dummy.updateMatrix();
    nodeMesh.setMatrixAt(i, dummy.matrix);
    color.setHex(colorForNode(node, colorRange));
    nodeMesh.setColorAt(i, color);
  });
  nodeMesh.instanceMatrix.needsUpdate = true;
  if (nodeMesh.instanceColor) nodeMesh.instanceColor.needsUpdate = true;
  scene.add(nodeMesh);

  if (shouldFrame) frameGraph(graph);
  renderVisualLegend(graph);
}

function frameGraph(graph) {
  if (!graph.nodes.length) return;
  const box = new THREE.Box3();
  for (const node of graph.nodes) {
    const p = state.layout.get(node.id);
    if (p) box.expandByPoint(new THREE.Vector3(p.x, p.y, p.z));
  }
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);
  controls.target.copy(center);
  const radius = Math.max(size.x, size.y, size.z, 30);
  const direction = new THREE.Vector3(0.2, -0.1, 1).normalize();
  camera.position.copy(center).add(direction.multiplyScalar(radius * 1.35 + 70));
  camera.near = Math.max(0.1, radius / 1000);
  camera.far = Math.max(2000, radius * 8);
  camera.updateProjectionMatrix();
}

function updatePointer(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
}

function pickNode(event) {
  if (!nodeMesh || !state.activeGraph) return null;
  updatePointer(event);
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObject(nodeMesh, false)[0];
  if (!hit || hit.instanceId === undefined) return null;
  return state.activeGraph.nodes[hit.instanceId] || null;
}

function onPointerMove(event) {
  if (state.boxDrag) {
    updateSelectionBox(event);
    return;
  }
  const node = pickNode(event);
  const tooltip = $("tooltip");
  if (!node) {
    state.hover = null;
    tooltip.classList.add("hidden");
    return;
  }
  state.hover = node.id;
  tooltip.classList.remove("hidden");
  tooltip.style.left = `${event.clientX + 12}px`;
  tooltip.style.top = `${event.clientY + 12}px`;
  tooltip.innerHTML = `<strong>${escapeHtml(nodeLabel(node))}</strong><br>` +
    `degree ${fmt(node.degree)} | PageRank ${fmt(node.pagerank, 4)} | core ${fmt(node.core)}<br>` +
    `between ${fmt(node.betweenness, 4)} | close ${fmt(node.closeness, 4)} | cluster ${fmt(node.clustering, 3)}<br>` +
    `iter ${fmt(node.iter)} | depth ${fmt(node.depth)} | component ${fmt(node.component)} | community ${fmt(node.community)}`;
}

function onPointerDown(event) {
  if (state.boxSelectMode) {
    beginBoxSelect(event);
    return;
  }
  const node = pickNode(event);
  if (!node) return;
  if (event.metaKey || event.ctrlKey || event.altKey) addSelectedNode(node.id);
  else selectNode(node.id, event.shiftKey);
}

function onPointerUp(event) {
  if (state.boxDrag) finishBoxSelect(event);
}

function selectNode(id, asTarget = false) {
  if (asTarget) state.target = id;
  else state.source = id;
  state.selectedNodes.add(id);
  if (!state.target && state.source && state.activeGraph?.nodes.length === 1) state.target = state.source;
  updateSelectionHud();
  renderSelectionDetails();
  rebuildScene(false);
}

function addSelectedNode(id) {
  state.selectedNodes.add(id);
  if (!state.source) state.source = id;
  updateSelectionHud();
  renderSelectionDetails();
  rebuildScene(false);
}

function clearSelection() {
  state.source = null;
  state.target = null;
  state.selectedNodes.clear();
  $("pathDetails").textContent = "No path loaded.";
  updateSelectionHud();
  renderSelectionDetails();
  rebuildScene(false);
}

function updateSelectionHud() {
  const source = state.source ? nodeLabel(findNode(state.source)) : "none";
  const target = state.target ? nodeLabel(findNode(state.target)) : "none";
  $("selectionHud").textContent = `source: ${source} | target: ${target} | selected: ${state.selectedNodes.size} | pinned: ${state.pinnedNodes.size}`;
}

function findNode(id) {
  return state.nodeById.get(id) || state.fullGraph?.nodes.find((n) => n.id === id) || null;
}

function beginBoxSelect(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  state.boxDrag = {
    x0: event.clientX - rect.left,
    y0: event.clientY - rect.top,
    x1: event.clientX - rect.left,
    y1: event.clientY - rect.top,
  };
  controls.enabled = false;
  updateSelectionBox(event);
}

function updateSelectionBox(event) {
  if (!state.boxDrag) return;
  const rect = renderer.domElement.getBoundingClientRect();
  state.boxDrag.x1 = event.clientX - rect.left;
  state.boxDrag.y1 = event.clientY - rect.top;
  const box = $("selectionBox");
  const x = Math.min(state.boxDrag.x0, state.boxDrag.x1);
  const y = Math.min(state.boxDrag.y0, state.boxDrag.y1);
  const w = Math.abs(state.boxDrag.x1 - state.boxDrag.x0);
  const h = Math.abs(state.boxDrag.y1 - state.boxDrag.y0);
  box.style.left = `${x}px`;
  box.style.top = `${y}px`;
  box.style.width = `${w}px`;
  box.style.height = `${h}px`;
  box.classList.remove("hidden");
}

function finishBoxSelect() {
  if (!state.boxDrag || !state.activeGraph) return;
  const box = $("selectionBox");
  box.classList.add("hidden");
  const x0 = Math.min(state.boxDrag.x0, state.boxDrag.x1);
  const x1 = Math.max(state.boxDrag.x0, state.boxDrag.x1);
  const y0 = Math.min(state.boxDrag.y0, state.boxDrag.y1);
  const y1 = Math.max(state.boxDrag.y0, state.boxDrag.y1);
  const rect = renderer.domElement.getBoundingClientRect();
  const v = new THREE.Vector3();
  let added = 0;
  for (const node of state.activeGraph.nodes) {
    const p = state.layout.get(node.id);
    if (!p) continue;
    v.set(p.x, p.y, p.z).project(camera);
    const sx = (v.x * 0.5 + 0.5) * rect.width;
    const sy = (-v.y * 0.5 + 0.5) * rect.height;
    if (sx >= x0 && sx <= x1 && sy >= y0 && sy <= y1) {
      state.selectedNodes.add(node.id);
      if (!state.source) state.source = node.id;
      added++;
    }
  }
  state.boxDrag = null;
  controls.enabled = true;
  $("filterSummary").textContent = `Box selected ${added} visible nodes.`;
  updateSelectionHud();
  renderSelectionDetails();
  rebuildScene(false);
}

function renderSelectionDetails() {
  const node = findNode(state.source);
  if (!node) {
    $("selectionDetails").textContent = state.selectedNodes.size
      ? `${state.selectedNodes.size} nodes selected.`
      : "No node selected.";
    return;
  }
  const attrs = Object.entries(node.attrs || {})
    .filter(([k]) => !["label"].includes(k))
    .slice(0, 16)
    .map(([k, v]) => `${k}: ${String(v).slice(0, 240)}`)
    .join("\n");
  $("selectionDetails").textContent = [
    nodeLabel(node),
    `id: ${node.id}`,
    `degree: ${node.degree}`,
    `pagerank: ${Number(node.pagerank || 0).toPrecision(3)}`,
    `core: ${node.core}`,
    `betweenness: ${Number(node.betweenness || 0).toPrecision(3)}`,
    `closeness: ${Number(node.closeness || 0).toPrecision(3)}`,
    `eigenvector: ${Number(node.eigenvector || 0).toPrecision(3)}`,
    `clustering: ${Number(node.clustering || 0).toPrecision(3)}`,
    `component: ${node.component}`,
    `community: ${node.community}`,
    `iter: ${node.iter}`,
    `depth: ${node.depth}`,
    `selected nodes: ${state.selectedNodes.size}`,
    `pinned nodes: ${state.pinnedNodes.size}`,
    attrs ? `\n${attrs}` : "",
  ].join("\n");
}

async function loadRun() {
  const run = $("runPath").value.trim();
  if (!run) return;
  setBusy("loadRunBtn", true);
  try {
    const data = await api("/api/load_run", { run });
    state.source = null;
    state.target = null;
    state.selectedNodes.clear();
    state.pinnedNodes.clear();
    state.pinnedPositions.clear();
    setGraph(data, "whole graph", true);
    $("pathDetails").textContent = "No path loaded.";
  } catch (err) {
    alert(err.message);
  } finally {
    setBusy("loadRunBtn", false);
  }
}

async function uploadGraph(file) {
  if (!file) return;
  const text = await file.text();
  const data = await api("/api/load_graphml", { name: file.name, graphml: text });
  state.source = null;
  state.target = null;
  state.selectedNodes.clear();
  state.pinnedNodes.clear();
  state.pinnedPositions.clear();
  setGraph(data, "whole graph", true);
}

async function search() {
  const query = $("searchInput").value.trim();
  if (!query) return;
  const data = await api("/api/search", { query, limit: 50 });
  renderSearchResults(data.results || []);
}

function renderSearchResults(results) {
  $("searchResults").innerHTML = results.map((r) => `
    <div class="result-item" data-node="${escapeAttr(r.id)}">
      <strong>${escapeHtml(r.label)}</strong>
      <span>degree ${fmt(r.degree)} | iter ${fmt(r.iter)} | score ${fmt(r.score, 2)}</span>
    </div>
  `).join("");
  for (const el of $("searchResults").querySelectorAll(".result-item")) {
    el.addEventListener("click", () => {
      const id = el.getAttribute("data-node");
      selectNode(id, false);
      flyToNode(id);
    });
  }
}

function showHubs() {
  if (!state.fullGraph) return;
  const hubs = [...state.fullGraph.nodes].sort((a, b) => (b.degree || 0) - (a.degree || 0)).slice(0, 40);
  renderSearchResults(hubs.map((n, i) => ({
    id: n.id,
    label: `${i + 1}. ${nodeLabel(n)}`,
    degree: n.degree,
    iter: n.iter,
    score: n.pagerank || 0,
  })));
}

async function focusNeighborhood() {
  if (!state.source) return;
  setBusy("neighborhoodBtn", true);
  try {
    const data = await api("/api/neighborhood", {
      nodes: [state.source],
      depth: Number($("depthInput").value || 1),
      limit: Number($("limitInput").value || 400),
    });
    setGraph(data, `neighborhood of ${nodeLabel(findNode(state.source))}`, false);
  } catch (err) {
    alert(err.message);
  } finally {
    setBusy("neighborhoodBtn", false);
  }
}

async function focusPath() {
  if (!state.source || !state.target) {
    alert("Select a source node, then shift-click a target node.");
    return;
  }
  setBusy("pathBtn", true);
  try {
    const data = await api("/api/path", {
      source: state.source,
      target: state.target,
      k: Number($("pathKInput").value || 5),
      cutoff: Number($("pathCutoffInput").value || 6),
    });
    setGraph(data, `paths: ${nodeLabel(findNode(state.source))} to ${nodeLabel(findNode(state.target))}`, false);
    renderPaths(data.paths || []);
  } catch (err) {
    alert(err.message);
  } finally {
    setBusy("pathBtn", false);
  }
}

function renderPaths(paths) {
  if (!paths.length) {
    $("pathDetails").textContent = "No path found within cutoff.";
    return;
  }
  $("pathDetails").textContent = paths.map((p, i) => {
    const labels = p.map((id) => nodeLabel(findNode(id)) || id);
    return `${i + 1}. ${labels.join(" -> ")}`;
  }).join("\n\n");
}

function showAll() {
  if (!state.fullGraph) return;
  stopTimeline();
  setGraph(state.fullGraph, "whole graph", false);
  $("filterSummary").textContent = "Showing full graph.";
}

function parseIdList(text) {
  return new Set(String(text || "").split(",").map((x) => x.trim()).filter(Boolean));
}

function nodeMatchesText(node, text) {
  const q = String(text || "").trim().toLowerCase();
  if (!q) return true;
  const hay = [
    node.id,
    node.label,
    ...Object.entries(node.attrs || {}).map(([k, v]) => `${k} ${v}`),
  ].join(" ").toLowerCase();
  return q.split(/\s+/).every((term) => hay.includes(term));
}

function applyFilter(stopPlayback = true) {
  if (!state.fullGraph) return;
  if (stopPlayback) stopTimeline(false);
  const metric = $("filterMetricInput").value;
  const minRaw = $("filterMinInput").value;
  const maxRaw = $("filterMaxInput").value;
  const min = minRaw === "" ? -Infinity : Number(minRaw);
  const max = maxRaw === "" ? Infinity : Number(maxRaw);
  const comps = parseIdList($("filterComponentInput").value);
  const communities = parseIdList($("filterCommunityInput").value);
  const rel = $("filterRelationInput").value.trim().toLowerCase();
  const text = $("filterTextInput").value;
  const maxIter = Number($("filterIterInput").value || state.fullGraph.stats.max_iter || 0);
  const byRelation = new Set();
  if (rel) {
    for (const edge of state.fullGraph.edges) {
      if (String(edge.relation || "").toLowerCase().includes(rel)) {
        byRelation.add(edge.source);
        byRelation.add(edge.target);
      }
    }
  }
  const keep = state.fullGraph.nodes.filter((node) => {
    const value = metricValue(node, metric);
    if (value < min || value > max) return false;
    if (numberValue(node.iter, 0) > maxIter) return false;
    if (comps.size && !comps.has(String(node.component))) return false;
    if (communities.size && !communities.has(String(node.community))) return false;
    if (rel && !byRelation.has(node.id)) return false;
    return nodeMatchesText(node, text);
  }).map((node) => node.id);
  const payload = inducedGraph(keep, `filter ${metric}`);
  setDerivedGraph(payload, `filter: ${fmt(payload.stats.nodes)} nodes`);
  $("filterSummary").textContent = `${payload.stats.nodes} nodes, ${payload.stats.edges} edges after filter.`;
}

function clearFilter() {
  if (!state.fullGraph) return;
  $("filterMinInput").value = "";
  $("filterMaxInput").value = "";
  $("filterComponentInput").value = "";
  $("filterCommunityInput").value = "";
  $("filterRelationInput").value = "";
  $("filterTextInput").value = "";
  updateTimelineBounds(state.fullGraph);
  showAll();
}

function playTimeline() {
  if (!state.fullGraph) return;
  stopTimeline(false);
  const input = $("filterIterInput");
  const max = Number(input.max || state.fullGraph.stats.max_iter || 0);
  if (max <= 0) return;
  input.value = "0";
  state.timelineTimer = window.setInterval(() => {
    const next = Number(input.value || 0) + 1;
    input.value = String(next);
    applyFilter(false);
    if (next >= max) stopTimeline(false);
  }, 650);
}

function stopTimeline(updateText = true) {
  if (state.timelineTimer) window.clearInterval(state.timelineTimer);
  state.timelineTimer = null;
  if (updateText) $("filterSummary").textContent = "Timeline stopped.";
}

function focusSelectedSubgraph() {
  if (!state.selectedNodes.size) return;
  const payload = inducedGraph(state.selectedNodes, "selected subgraph");
  setDerivedGraph(payload, `selected: ${payload.stats.nodes} nodes`);
}

function pinSelected() {
  const ids = state.selectedNodes.size ? [...state.selectedNodes] : [state.source].filter(Boolean);
  for (const id of ids) {
    const p = state.layout.get(id);
    if (!p) continue;
    state.pinnedNodes.add(id);
    state.pinnedPositions.set(id, { x: p.x, y: p.y, z: p.z });
  }
  updateSelectionHud();
  renderSelectionDetails();
}

function clearPins() {
  state.pinnedNodes.clear();
  state.pinnedPositions.clear();
  updateSelectionHud();
  renderSelectionDetails();
}

function reheatPhysics() {
  if (!state.activeGraph) return;
  $("layoutModeInput").value = "force3d";
  readVisualControls();
  relaxLayout(state.activeGraph, Math.max(60, state.visual.physicsTicks));
  rebuildScene(true);
}

function explainCluster() {
  const selected = [...state.selectedNodes];
  if (!selected.length && state.source) selected.push(state.source);
  if (!selected.length) return;
  $("questionInput").value = `Explain this selected graph cluster. Identify the major mechanisms, bridges, gaps, and the next experiments or search queries.`;
  askGraph();
}

function downloadText(filename, text, type = "application/json") {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function exportJson(selectedOnly = false) {
  const graph = selectedOnly && state.selectedNodes.size
    ? inducedGraph(state.selectedNodes, "selected export")
    : state.activeGraph;
  if (!graph) return;
  downloadText("graph_explorer_view.json", JSON.stringify({
    graph,
    visual: state.visual,
    selected_nodes: [...state.selectedNodes],
    pinned_nodes: [...state.pinnedNodes],
  }, null, 2));
}

function exportPng() {
  if (!renderer) return;
  const a = document.createElement("a");
  a.href = renderer.domElement.toDataURL("image/png");
  a.download = "graph_explorer_view.png";
  document.body.appendChild(a);
  a.click();
  a.remove();
}

function exportSvg() {
  if (!state.activeGraph) return;
  const graph = state.activeGraph;
  const points = graph.nodes.map((n) => state.layout.get(n.id)).filter(Boolean);
  const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
  const minX = Math.min(...xs, -100), maxX = Math.max(...xs, 100);
  const minY = Math.min(...ys, -100), maxY = Math.max(...ys, 100);
  const pad = 40;
  const colorRange = metricRange(graph, state.visual.colorBy);
  const sizeRange = metricRange(graph, state.visual.sizeBy);
  const edgeEls = graph.edges.map((e) => {
    const a = state.layout.get(e.source), b = state.layout.get(e.target);
    if (!a || !b) return "";
    return `<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke="#${colorForEdge(e, { edge: { min: 0, max: 1 } }).toString(16).padStart(6, "0")}" stroke-opacity="${state.visual.edgeOpacity}" stroke-width="1" />`;
  }).join("\n");
  const nodeEls = graph.nodes.map((n) => {
    const p = state.layout.get(n.id);
    if (!p) return "";
    const r = sizeForNode(n, sizeRange) * 1.7;
    const c = colorForNode(n, colorRange).toString(16).padStart(6, "0");
    return `<circle cx="${p.x}" cy="${p.y}" r="${r}" fill="#${c}"><title>${escapeHtml(nodeLabel(n))}</title></circle>`;
  }).join("\n");
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${minX - pad} ${minY - pad} ${maxX - minX + pad * 2} ${maxY - minY + pad * 2}">\n<rect x="${minX - pad}" y="${minY - pad}" width="${maxX - minX + pad * 2}" height="${maxY - minY + pad * 2}" fill="${state.visual.backgroundColor}"/>\n${edgeEls}\n${nodeEls}\n</svg>\n`;
  downloadText("graph_explorer_view.svg", svg, "image/svg+xml");
}

function saveView() {
  const name = $("viewNameInput").value.trim();
  if (!name) return;
  const views = loadViews();
  views[name] = {
    visual: { ...state.visual },
    filters: readFilterState(),
    selected_nodes: [...state.selectedNodes],
    pinned_nodes: [...state.pinnedNodes],
  };
  localStorage.setItem("graphExplorerSavedViews", JSON.stringify(views));
  renderSavedViews();
}

function loadViews() {
  try {
    return JSON.parse(localStorage.getItem("graphExplorerSavedViews") || "{}");
  } catch (_) {
    return {};
  }
}

function renderSavedViews() {
  const select = $("savedViewsInput");
  if (!select) return;
  const views = loadViews();
  select.innerHTML = Object.keys(views).sort().map((name) => `<option value="${escapeAttr(name)}">${escapeHtml(name)}</option>`).join("");
}

function readFilterState() {
  return {
    metric: $("filterMetricInput").value,
    min: $("filterMinInput").value,
    max: $("filterMaxInput").value,
    component: $("filterComponentInput").value,
    community: $("filterCommunityInput").value,
    relation: $("filterRelationInput").value,
    text: $("filterTextInput").value,
    iter: $("filterIterInput").value,
  };
}

function writeFilterState(filters = {}) {
  $("filterMetricInput").value = filters.metric || "degree";
  $("filterMinInput").value = filters.min || "";
  $("filterMaxInput").value = filters.max || "";
  $("filterComponentInput").value = filters.component || "";
  $("filterCommunityInput").value = filters.community || "";
  $("filterRelationInput").value = filters.relation || "";
  $("filterTextInput").value = filters.text || "";
  if (filters.iter !== undefined) $("filterIterInput").value = filters.iter;
}

function loadSelectedView() {
  const name = $("savedViewsInput").value;
  const view = loadViews()[name];
  if (!view) return;
  for (const [key, id] of Object.entries(visualControlMap())) {
    if (view.visual?.[key] !== undefined && $(id)) $(id).value = view.visual[key];
  }
  readVisualControls();
  writeFilterState(view.filters || {});
  state.selectedNodes = new Set(view.selected_nodes || []);
  state.pinnedNodes = new Set(view.pinned_nodes || []);
  for (const id of state.pinnedNodes) {
    const p = state.layout.get(id);
    if (p) state.pinnedPositions.set(id, { x: p.x, y: p.y, z: p.z });
  }
  applyFilter();
}

function deleteSelectedView() {
  const name = $("savedViewsInput").value;
  const views = loadViews();
  delete views[name];
  localStorage.setItem("graphExplorerSavedViews", JSON.stringify(views));
  renderSavedViews();
}

async function compareRun() {
  const run = $("compareRunInput").value.trim();
  if (!run) return;
  $("compareDetails").textContent = "Comparing...";
  try {
    const data = await api("/api/compare_run", { run });
    const topOther = (data.other_only.top_nodes || []).map((n) => `+ ${n.label} (${n.degree})`).join("\n");
    const topCurrent = (data.current_only.top_nodes || []).map((n) => `- ${n.label} (${n.degree})`).join("\n");
    $("compareDetails").textContent = [
      `${data.current.name || "current"}: ${data.current.nodes}n/${data.current.edges}e`,
      `${data.other.name}: ${data.other.nodes}n/${data.other.edges}e`,
      `shared: ${data.shared.nodes} nodes, ${data.shared.edges} edges`,
      `current only: ${data.current_only.nodes} nodes, ${data.current_only.edges} edges`,
      topCurrent,
      `other only: ${data.other_only.nodes} nodes, ${data.other_only.edges} edges`,
      topOther,
    ].filter(Boolean).join("\n");
  } catch (err) {
    $("compareDetails").textContent = err.message;
  }
}

function applyVisualSettings(layoutChanged = false) {
  saveVisualPrefs();
  if (scene) scene.background = new THREE.Color(state.visual.backgroundColor);
  if (!state.activeGraph) return;
  if (layoutChanged) buildLayout(state.activeGraph);
  rebuildScene(layoutChanged);
}

function flyToNode(id) {
  const p = state.layout.get(id);
  if (!p) return;
  const target = new THREE.Vector3(p.x, p.y, p.z);
  controls.target.copy(target);
  camera.position.copy(target).add(new THREE.Vector3(0, 0, 80));
}

async function askGraph() {
  if (!state.fullGraph && !state.activeGraph) return;
  const question = $("questionInput").value.trim();
  if (!question) return;
  saveModelPrefs();
  $("answerBox").textContent = "Thinking...";
  setBusy("askBtn", true);
  try {
    const selected = [...new Set([...state.selectedNodes, state.source, state.target].filter(Boolean))];
    const modelConfig = {
      provider: $("providerInput").value,
      model: $("modelInput").value.trim(),
      base_url: $("baseUrlInput").value.trim(),
      api_key: $("apiKeyInput").value,
      reasoning_effort: $("effortInput").value.trim(),
      temperature: Number($("temperatureInput").value || 0.3),
      max_tokens: Number($("maxTokensInput").value || 1800),
    };
    const data = await api("/api/ask", {
      question,
      selected_nodes: selected,
      query: $("contextQueryInput").value.trim(),
      depth: Number($("depthInput").value || 1),
      max_nodes: Number($("contextNodesInput").value || 90),
      max_edges: 160,
      model_config: modelConfig,
    });
    $("answerBox").textContent = `${data.answer}\n\nContext: ${data.context.node_count} nodes, ${data.context.edge_count} edges`;
  } catch (err) {
    $("answerBox").textContent = err.message;
  } finally {
    setBusy("askBtn", false);
  }
}

async function startRun() {
  const topic = $("topicInput").value.trim();
  if (!topic) return;
  setBusy("startRunBtn", true);
  $("jobLog").textContent = "Starting...";
  try {
    const body = {
      topic,
      strategy: $("strategyInput").value.trim() || "frontier",
      budget_calls: Number($("callsInput").value || 50),
      max_iters: Number($("itersInput").value || 50),
      out: $("outInput").value.trim(),
    };
    const job = await api("/api/ideate", body);
    state.currentJob = job.id;
    pollJob(job.id);
  } catch (err) {
    $("jobLog").textContent = err.message;
    setBusy("startRunBtn", false);
  }
}

async function pollJob(jobId) {
  if (!jobId || state.currentJob !== jobId) return;
  try {
    const job = await api(`/api/job?id=${encodeURIComponent(jobId)}`);
    $("jobLog").textContent = [
      `${job.status} | ${job.cmd.join(" ")}`,
      "",
      job.log_tail || "",
    ].join("\n");
    $("jobLog").scrollTop = $("jobLog").scrollHeight;
    if (job.status === "running") {
      setTimeout(() => pollJob(jobId), 2200);
      return;
    }
    setBusy("startRunBtn", false);
    if (job.graph_ready) {
      $("runPath").value = job.out;
      await loadRun();
    }
  } catch (err) {
    $("jobLog").textContent = err.message;
    setBusy("startRunBtn", false);
  }
}

function setBusy(id, busy) {
  const el = $(id);
  if (!el) return;
  el.disabled = busy;
}

function escapeHtml(text) {
  return String(text ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function escapeAttr(text) {
  return escapeHtml(text).replaceAll("'", "&#39;");
}

function wireEvents() {
  $("fileInput").addEventListener("change", (e) => uploadGraph(e.target.files[0]).catch((err) => alert(err.message)));
  $("loadRunBtn").addEventListener("click", loadRun);
  $("runPath").addEventListener("keydown", (e) => { if (e.key === "Enter") loadRun(); });
  $("searchBtn").addEventListener("click", search);
  $("searchInput").addEventListener("keydown", (e) => { if (e.key === "Enter") search(); });
  $("neighborhoodBtn").addEventListener("click", focusNeighborhood);
  $("pathBtn").addEventListener("click", focusPath);
  $("hubsBtn").addEventListener("click", showHubs);
  $("showAllBtn").addEventListener("click", showAll);
  $("selectedSubgraphBtn").addEventListener("click", focusSelectedSubgraph);
  $("boxSelectBtn").addEventListener("click", () => {
    state.boxSelectMode = !state.boxSelectMode;
    $("boxSelectBtn").classList.toggle("active", state.boxSelectMode);
    $("filterSummary").textContent = state.boxSelectMode ? "Box Select enabled." : "Box Select disabled.";
  });
  $("pinSelectedBtn").addEventListener("click", pinSelected);
  $("reheatBtn").addEventListener("click", reheatPhysics);
  $("clearSelectionBtn").addEventListener("click", clearSelection);
  $("clearPinsBtn").addEventListener("click", clearPins);
  $("explainClusterBtn").addEventListener("click", explainCluster);
  $("exportSelectionBtn").addEventListener("click", () => exportJson(true));
  $("applyFilterBtn").addEventListener("click", applyFilter);
  $("clearFilterBtn").addEventListener("click", clearFilter);
  $("playTimelineBtn").addEventListener("click", playTimeline);
  $("stopTimelineBtn").addEventListener("click", () => stopTimeline(true));
  $("filterIterInput").addEventListener("input", applyFilter);
  $("saveViewBtn").addEventListener("click", saveView);
  $("loadViewBtn").addEventListener("click", loadSelectedView);
  $("deleteViewBtn").addEventListener("click", deleteSelectedView);
  $("compareRunBtn").addEventListener("click", compareRun);
  $("exportPngBtn").addEventListener("click", exportPng);
  $("exportSvgBtn").addEventListener("click", exportSvg);
  $("exportJsonBtn").addEventListener("click", () => exportJson(false));
  $("resetViewBtn").addEventListener("click", () => state.activeGraph && frameGraph(state.activeGraph));
  $("askBtn").addEventListener("click", askGraph);
  $("startRunBtn").addEventListener("click", startRun);
  $("modelRoleInput").addEventListener("change", syncRoleFields);
  $("modelPresetInput").addEventListener("change", applyPresetToRole);
  for (const id of ["roleProviderInput", "roleModelInput", "roleBaseUrlInput", "roleApiKeyEnvInput", "roleTempInput", "roleTokensInput", "roleEffortInput"]) {
    $(id).addEventListener("change", readRoleFields);
  }
  $("applyRoleToAskBtn").addEventListener("click", applyRoleToAsk);
  $("checkModelBtn").addEventListener("click", checkModelServer);
  $("previewConfigBtn").addEventListener("click", () => previewConfig(false));
  $("writeConfigBtn").addEventListener("click", () => previewConfig(true));
  for (const id of ["providerInput", "modelInput", "baseUrlInput", "effortInput", "temperatureInput", "maxTokensInput"]) {
    $(id).addEventListener("change", saveModelPrefs);
  }
  const layoutControlIds = new Set(["layoutModeInput", "physicsTicksInput", "springLengthInput", "repulsionInput", "gravityInput"]);
  const visualIds = Object.values(visualControlMap());
  for (const id of visualIds) {
    const el = $(id);
    if (!el) continue;
    const eventName = el.type === "range" || el.type === "color" ? "input" : "change";
    el.addEventListener(eventName, () => applyVisualSettings(layoutControlIds.has(id)));
  }
  $("applyVisualsBtn").addEventListener("click", () => applyVisualSettings(true));
}

async function loadExistingGraphIfAny() {
  try {
    const data = await api("/api/graph");
    setGraph(data, "whole graph", true);
  } catch (_) {
    renderStats({ nodes: 0, edges: 0, components: 0, communities: 0, largest_component: 0, avg_degree: 0, max_degree: 0, max_iter: 0, density: 0 });
  }
}

initScene();
wireEvents();
loadModelPrefs();
loadModelRoles();
renderSavedViews();
loadVisualPrefs();
loadExistingGraphIfAny();
importConfigRoles();
window.graphExplorerReady = true;
