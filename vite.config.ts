import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { VitePWA } from "vite-plugin-pwa";
import path from "node:path";

// https://vite.dev/config/
export default defineConfig(({ command, mode }) => {
  const experimental = command === "serve" || loadEnv(mode, process.cwd(), "VITE_").VITE_EXPERIMENTAL_PROCESSOR === "1";
  return {
  // LibRaw creates its own module Worker and loads a companion WASM binary.
  // Pre-bundling it rewrites the worker URL to Vite's optimizer cache, where
  // that companion file does not exist. Keep it as a normal dependency in
  // development; Rollup still emits the worker/WASM pair in production.
  optimizeDeps: {
    exclude: ["libraw-wasm"],
  },
  plugins: [
    react(),
    tailwindcss(),
    VitePWA({
      registerType: "autoUpdate",
      includeAssets: ["icons/*.png"],
      manifest: {
        name: "FujiApp — Film Simulation Recipe Previewer",
        short_name: "FujiApp",
        description: "Apply Fuji X Weekly-style film simulation recipes to your photos, entirely on-device.",
        theme_color: "#0a0a0a",
        background_color: "#0a0a0a",
        display: "standalone",
        orientation: "portrait",
        start_url: "/",
        icons: [
          { src: "/icons/icon-192.png", sizes: "192x192", type: "image/png" },
          { src: "/icons/icon-512.png", sizes: "512x512", type: "image/png" },
          { src: "/icons/icon-512-maskable.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
        ],
      },
      workbox: {
        // Precache every built asset (JS/CSS/HTML/LUT PNGs/icons) so the app
        // — including the film-simulation LUTs — works fully offline once
        // installed, matching the "zero-server, client-side-only" design.
        globPatterns: ["**/*.{js,css,html,png,svg,ico}"],
        // Cache large local inference assets after first use. Native builds
        // bundle them directly; this also enables repeat offline PWA tests.
        runtimeCaching: [{
          urlPattern: ({ url, sameOrigin }) => sameOrigin && /\.(?:wasm|mjs|onnx)$/.test(url.pathname),
          handler: "CacheFirst",
          options: { cacheName: "fuji-local-inference", expiration: { maxEntries: 12, maxAgeSeconds: 60 * 60 * 24 * 30 } },
        }],
      },
    }),
  ],
  resolve: {
    // Removing a lazy React branch alone does not stop Vite from emitting
    // its imported binary assets. Resolve the entry to a harmless stub in
    // ordinary release builds, before the real module is ever scanned.
    alias: [
      { find: "@/components/viewer/ExperimentalRawPreview", replacement: path.resolve(__dirname, experimental
        ? "./src/components/viewer/ExperimentalRawPreview.tsx" : "./src/components/viewer/ExperimentalProcessorUnavailable.tsx") },
      { find: "@", replacement: path.resolve(__dirname, "./src") },
    ],
  },
  };
});
