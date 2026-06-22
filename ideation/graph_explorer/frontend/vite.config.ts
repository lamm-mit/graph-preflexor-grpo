import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const apiPort = Number(process.env.GRAPH_EXPLORER_API_PORT || 8765);

export default defineConfig({
  plugins: [react()],
  server: {
    port: Number(process.env.GRAPH_EXPLORER_VITE_PORT || 5177),
    strictPort: true,
    proxy: {
      "/api": {
        target: `http://127.0.0.1:${apiPort}`,
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
