import { useCallback, useEffect, useState } from "react";

const STORAGE_KEY = "fujiapp:recipe-preview-overrides";

function loadOverrides(): Record<string, string> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Record<string, string>) : {};
  } catch {
    return {};
  }
}

/**
 * Lets a user replace any recipe's cover photo (built-in or custom) with one
 * of their own — separate from Recipe.previewImage, which only exists for
 * custom recipes. Keyed by recipe id so it works for either kind uniformly.
 */
export function useRecipePreviewOverrides() {
  const [overrides, setOverrides] = useState<Record<string, string>>(loadOverrides);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(overrides));
  }, [overrides]);

  const setOverride = useCallback((id: string, dataUrl: string) => {
    setOverrides((prev) => ({ ...prev, [id]: dataUrl }));
  }, []);

  const clearOverride = useCallback((id: string) => {
    setOverrides((prev) => {
      const next = { ...prev };
      delete next[id];
      return next;
    });
  }, []);

  return { overrides, setOverride, clearOverride };
}
