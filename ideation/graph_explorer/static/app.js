import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (id) => document.getElementById(id);

const palette = [
  0x4fb477, 0xe1a13a, 0x64a7d9, 0xd96868, 0xb58ad7, 0x6fc7bd,
  0xc0c46a, 0xe58f6e, 0x8fa8ff, 0x78a65a, 0xcf79a5, 0x9bb0bd
];

const state = {
  fullGraph: null,
  activeGraph: null,
  nodeById: new Map(),
  edgeById: new Map(),
  layout: new Map(),
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

function initScene() {
  const viewport = $("viewport");
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x090b0d);
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

function fmt(n, digits = 0) {
  if (n === undefined || n === null || Number.isNaN(Number(n))) return "";
  return Number(n).toLocaleString(undefined, { maximumFractionDigits: digits });
}

function nodeLabel(node) {
  return node?.label || node?.id || "";
}

function colorForNode(node) {
  if (node.id === state.source) return 0x4fb477;
  if (node.id === state.target) return 0xe1a13a;
  if (node.id === state.hover) return 0xffffff;
  return palette[Math.abs(node.component || 0) % palette.length];
}

function setGraph(payload, mode = "whole graph", isFull = false) {
  if (isFull) state.fullGraph = payload;
  state.activeGraph = payload;
  state.mode = mode;
  state.nodeById = new Map(payload.nodes.map((n) => [n.id, n]));
  state.edgeById = new Map(payload.edges.map((e) => [e.id, e]));
  $("graphName").textContent = `${payload.name || "graph"} | ${fmt(payload.stats.nodes)} nodes, ${fmt(payload.stats.edges)} edges`;
  $("renderMode").textContent = mode;
  renderStats(payload.stats);
  ensureLayout(payload);
  if (payload.nodes.length <= 650) relaxLayout(payload, payload.nodes.length < 140 ? 180 : 90);
  rebuildScene(true);
  updateSelectionHud();
}

function renderStats(stats) {
  const rows = [
    ["Nodes", fmt(stats.nodes)],
    ["Edges", fmt(stats.edges)],
    ["Components", fmt(stats.components)],
    ["Largest", fmt(stats.largest_component)],
    ["Avg degree", fmt(stats.avg_degree, 2)],
    ["Max degree", fmt(stats.max_degree)],
    ["Max iter", fmt(stats.max_iter)],
    ["Density", fmt(stats.density, 5)],
  ];
  $("stats").innerHTML = rows.map(([k, v]) => `<div class="stat"><b>${v}</b><span>${k}</span></div>`).join("");
}

function hash01(str, salt = 0) {
  let h = 2166136261 + salt;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return ((h >>> 0) % 100000) / 100000;
}

function ensureLayout(graph) {
  const maxIter = Math.max(1, graph.stats.max_iter || 1);
  const compSizes = graph.stats.component_sizes || [];
  const compRank = new Map(compSizes.map((_, i) => [i, i]));
  for (const node of graph.nodes) {
    if (state.layout.has(node.id)) continue;
    const comp = compRank.get(node.component) ?? node.component ?? 0;
    const compAngle = comp * 2.3999632297;
    const compRadius = Math.sqrt(comp + 1) * 26;
    const localAngle = hash01(node.id, 11) * Math.PI * 2;
    const localRadius = 18 + Math.sqrt(Math.max(1, node.degree || 1)) * 8 + hash01(node.id, 17) * 60;
    const zByIter = ((node.iter || 0) / maxIter - 0.5) * 150;
    const zNoise = (hash01(node.id, 23) - 0.5) * 90;
    state.layout.set(node.id, {
      x: Math.cos(compAngle) * compRadius + Math.cos(localAngle) * localRadius,
      y: Math.sin(compAngle) * compRadius + Math.sin(localAngle) * localRadius,
      z: zByIter + zNoise,
      vx: 0,
      vy: 0,
      vz: 0,
    });
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
        const f = 42 / d2;
        a.vx += dx * f; a.vy += dy * f; a.vz += dz * f;
        b.vx -= dx * f; b.vy -= dy * f; b.vz -= dz * f;
      }
    }
    for (const [ia, ib] of springs) {
      const a = pos[ia], b = pos[ib];
      let dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) + 0.001;
      const rest = 18;
      const f = (dist - rest) * 0.018;
      dx /= dist; dy /= dist; dz /= dist;
      a.vx += dx * f; a.vy += dy * f; a.vz += dz * f;
      b.vx -= dx * f; b.vy -= dy * f; b.vz -= dz * f;
    }
    for (const p of pos) {
      p.vx += -p.x * 0.004;
      p.vy += -p.y * 0.004;
      p.vz += -p.z * 0.004;
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
  clearObjects();
  const graph = state.activeGraph;
  const nodeIndex = new Map(graph.nodes.map((n, i) => [n.id, i]));

  const edgePositions = new Float32Array(graph.edges.length * 2 * 3);
  const edgeColors = new Float32Array(graph.edges.length * 2 * 3);
  const edgeColor = new THREE.Color(0x46505c);
  const pathColor = new THREE.Color(0xe1a13a);
  let ep = 0, ec = 0;
  for (const edge of graph.edges) {
    const a = state.layout.get(edge.source);
    const b = state.layout.get(edge.target);
    if (!a || !b) continue;
    edgePositions[ep++] = a.x; edgePositions[ep++] = a.y; edgePositions[ep++] = a.z;
    edgePositions[ep++] = b.x; edgePositions[ep++] = b.y; edgePositions[ep++] = b.z;
    const selected = (edge.source === state.source && edge.target === state.target) ||
      (edge.source === state.target && edge.target === state.source);
    const c = selected ? pathColor : edgeColor;
    for (let i = 0; i < 2; i++) {
      edgeColors[ec++] = c.r; edgeColors[ec++] = c.g; edgeColors[ec++] = c.b;
    }
  }
  const edgeGeom = new THREE.BufferGeometry();
  edgeGeom.setAttribute("position", new THREE.BufferAttribute(edgePositions, 3));
  edgeGeom.setAttribute("color", new THREE.BufferAttribute(edgeColors, 3));
  edgeLines = new THREE.LineSegments(edgeGeom, new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: graph.edges.length > 10000 ? 0.18 : 0.34,
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
    const size = 1.35 + Math.log1p(node.degree || 0) * 0.34 + Math.log1p(node.core || 0) * 0.09;
    dummy.position.set(p.x, p.y, p.z);
    dummy.scale.setScalar(node.id === state.source || node.id === state.target ? size * 1.75 : size);
    dummy.updateMatrix();
    nodeMesh.setMatrixAt(i, dummy.matrix);
    color.setHex(colorForNode(node));
    nodeMesh.setColorAt(i, color);
  });
  nodeMesh.instanceMatrix.needsUpdate = true;
  if (nodeMesh.instanceColor) nodeMesh.instanceColor.needsUpdate = true;
  scene.add(nodeMesh);

  if (shouldFrame) frameGraph(graph);
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
  tooltip.innerHTML = `<strong>${escapeHtml(nodeLabel(node))}</strong><br>degree ${fmt(node.degree)} | iter ${fmt(node.iter)} | component ${fmt(node.component)}`;
}

function onPointerDown(event) {
  const node = pickNode(event);
  if (!node) return;
  selectNode(node.id, event.shiftKey);
}

function selectNode(id, asTarget = false) {
  if (asTarget) state.target = id;
  else state.source = id;
  if (!state.target && state.source && state.activeGraph?.nodes.length === 1) state.target = state.source;
  updateSelectionHud();
  renderSelectionDetails();
  rebuildScene(false);
}

function updateSelectionHud() {
  const source = state.source ? nodeLabel(findNode(state.source)) : "none";
  const target = state.target ? nodeLabel(findNode(state.target)) : "none";
  $("selectionHud").textContent = `source: ${source} | target: ${target}`;
}

function findNode(id) {
  return state.nodeById.get(id) || state.fullGraph?.nodes.find((n) => n.id === id) || null;
}

function renderSelectionDetails() {
  const node = findNode(state.source);
  if (!node) {
    $("selectionDetails").textContent = "No node selected.";
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
    `component: ${node.component}`,
    `iter: ${node.iter}`,
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
  setGraph(state.fullGraph, "whole graph", false);
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
    const selected = [state.source, state.target].filter(Boolean);
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
  $("resetViewBtn").addEventListener("click", () => state.activeGraph && frameGraph(state.activeGraph));
  $("askBtn").addEventListener("click", askGraph);
  $("startRunBtn").addEventListener("click", startRun);
  for (const id of ["providerInput", "modelInput", "baseUrlInput", "effortInput", "temperatureInput", "maxTokensInput"]) {
    $(id).addEventListener("change", saveModelPrefs);
  }
}

async function loadExistingGraphIfAny() {
  try {
    const data = await api("/api/graph");
    setGraph(data, "whole graph", true);
  } catch (_) {
    renderStats({ nodes: 0, edges: 0, components: 0, largest_component: 0, avg_degree: 0, max_degree: 0, max_iter: 0, density: 0 });
  }
}

initScene();
wireEvents();
loadModelPrefs();
loadExistingGraphIfAny();
window.graphExplorerReady = true;
