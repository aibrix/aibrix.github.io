import { resolve } from "path";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  base: "/tools/",
  build: {
    outDir: resolve(__dirname, "../static/tools"),
    emptyOutDir: true,
    rollupOptions: {
      input: {
        "batch-cost-calculator": resolve(
          __dirname,
          "batch-cost-calculator/index.html"
        ),
      },
    },
  },
});
