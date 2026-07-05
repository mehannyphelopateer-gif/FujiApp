import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { VitePWA } from "vite-plugin-pwa";
import path from "node:path";

// https://vite.dev/config/
export default defineConfig({
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
      },
    }),
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
});
