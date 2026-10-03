import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    headers: {
      // Firebase's Google popup flow uses window.closed/window.close.
      // Keep the local development page compatible with that popup flow.
      "Cross-Origin-Opener-Policy": "unsafe-none",
    },
  },
});
