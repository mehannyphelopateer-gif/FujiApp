import { useCallback, useEffect, useState } from "react";

/**
 * Minimal client-side router — the app only has two "pages" (the previewer
 * and the recipe catalog), so a full router library would be more machinery
 * than the app needs. Tracks window.location.pathname and exposes a
 * pushState-based navigate().
 */
export function useRoute() {
  const [path, setPath] = useState(() => window.location.pathname);

  useEffect(() => {
    const onPopState = () => setPath(window.location.pathname);
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const navigate = useCallback((to: string) => {
    window.history.pushState({}, "", to);
    setPath(to);
  }, []);

  return { path, navigate };
}
