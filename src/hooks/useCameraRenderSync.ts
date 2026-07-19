import { useEffect } from "react";
import { useAppState } from "@/context/AppStateContext";
import { useCameraLink } from "@/context/CameraLinkContext";

/**
 * Mounted once (in AppShell). Keeps the camera-rendered preview in sync with
 * whatever the user is actually looking at — every time render mode is on
 * and a recipe or RAF changes, this re-triggers a real camera conversion.
 * RecipeGrid/RecipeCard stay untouched; they only ever set the selected
 * recipe id, same as the WebGL preview path.
 */
export function useCameraRenderSync() {
  const { selectedRecipe, originalRawFile } = useAppState();
  const { isCameraRenderMode, convertWithRecipe } = useCameraLink();

  useEffect(() => {
    if (!isCameraRenderMode || !originalRawFile) return;
    void convertWithRecipe(selectedRecipe, originalRawFile);
  }, [isCameraRenderMode, selectedRecipe, originalRawFile, convertWithRecipe]);
}
