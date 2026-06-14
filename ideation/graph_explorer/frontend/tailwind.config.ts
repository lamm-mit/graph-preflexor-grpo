import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        shell: {
          bg: "#090b10",
          panel: "#111821",
          panel2: "#151e2a",
          field: "#080d14",
          line: "#263243",
          text: "#edf3fa",
          muted: "#94a3b5",
          accent: "#37d49a",
          blue: "#38bdf8",
          amber: "#f4b24f",
          danger: "#ef6c73",
        },
      },
      boxShadow: {
        panel: "0 18px 44px rgba(0, 0, 0, 0.24)",
      },
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "sans-serif",
        ],
      },
    },
  },
  plugins: [],
};

export default config;
