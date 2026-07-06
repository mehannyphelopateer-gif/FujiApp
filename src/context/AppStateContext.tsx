import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import type { Recipe } from "@/types/recipe";
import type { DetectedSettings } from "@/types/exif";
import { recipes as builtInRecipes } from "@/lib/recipes/loadRecipes";
import { computeRecipeAdjustment, type RecipeAdjustment } from "@/lib/recipes/neutralize";
import { extractDetectedSettings } from "@/lib/exif/parseFujiMakerNotes";
import { mapCameraModelToSensorGeneration } from "@/lib/exif/sensorGenerations";
import { useCustomRecipes } from "@/hooks/useCustomRecipes";

interface AppState {
  recipes: Recipe[];
  /** Recipes compatible with the uploaded photo's detected sensor generation. All recipes if unknown. */
  compatibleRecipes: Recipe[];
  selectedFile: File | null;
  previewUrl: string | null;
  selectedRecipeId: string;
  selectedRecipe: Recipe;
  detectedSettings: DetectedSettings | null;
  sensorGeneration: string | null;
  recipeAdjustment: RecipeAdjustment;
  setSelectedFile: (file: File | null) => void;
  setSelectedRecipeId: (id: string) => void;
  saveCustomRecipe: (recipe: Recipe) => void;
  deleteCustomRecipe: (id: string) => void;
  captureCustomRecipePreview: (id: string) => void;
}

const AppStateContext = createContext<AppState | null>(null);

export function AppStateProvider({ children }: { children: ReactNode }) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedRecipeId, setSelectedRecipeId] = useState<string>(builtInRecipes[0].id);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [detectedSettings, setDetectedSettings] = useState<DetectedSettings | null>(null);
  const { customRecipes, saveRecipe, deleteRecipe, setPreviewImage } = useCustomRecipes();

  const recipes = useMemo(() => [...builtInRecipes, ...customRecipes], [customRecipes]);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [selectedFile]);

  useEffect(() => {
    if (!selectedFile) {
      setDetectedSettings(null);
      return;
    }
    let cancelled = false;
    extractDetectedSettings(selectedFile)
      .then((settings) => {
        if (!cancelled) setDetectedSettings(settings);
      })
      .catch(() => {
        if (!cancelled) setDetectedSettings(null);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedFile]);

  const sensorGeneration = useMemo(
    () => mapCameraModelToSensorGeneration(detectedSettings?.cameraModel ?? null),
    [detectedSettings],
  );

  const compatibleRecipes = useMemo(() => {
    if (!sensorGeneration) return recipes; // fail open: unknown camera shows everything
    return recipes.filter((recipe) => recipe.compatibleSensors.includes(sensorGeneration));
  }, [recipes, sensorGeneration]);

  // If the currently-selected recipe falls outside the newly-compatible set
  // (e.g. a photo from an older body was just uploaded), fall back to the
  // first compatible recipe instead of silently keeping an incompatible one.
  useEffect(() => {
    if (compatibleRecipes.length === 0) return;
    if (!compatibleRecipes.some((recipe) => recipe.id === selectedRecipeId)) {
      setSelectedRecipeId(compatibleRecipes[0].id);
    }
  }, [compatibleRecipes, selectedRecipeId]);

  const selectedRecipe = useMemo(
    () => recipes.find((recipe) => recipe.id === selectedRecipeId) ?? recipes[0],
    [recipes, selectedRecipeId],
  );

  const recipeAdjustment = useMemo(
    () => computeRecipeAdjustment(detectedSettings, selectedRecipe),
    [detectedSettings, selectedRecipe],
  );

  // Custom recipes have no place in the pre-rendered /recipe-previews/ build
  // step, so the only way to get an accurate thumbnail is to actually select
  // the recipe (letting the real WebGL pipeline render it against whatever
  // photo is currently loaded) and grab the canvas a moment later.
  function captureCustomRecipePreview(id: string) {
    if (!previewUrl) return;
    setSelectedRecipeId(id);
    setTimeout(() => {
      const canvas = document.querySelector("canvas");
      if (!canvas) return;
      try {
        const dataUrl = canvas.toDataURL("image/jpeg", 0.6);
        setPreviewImage(id, dataUrl);
      } catch {
        // Canvas may be tainted/unready — thumbnail just stays unset, not fatal.
      }
    }, 300);
  }

  const value: AppState = {
    recipes,
    compatibleRecipes,
    selectedFile,
    previewUrl,
    selectedRecipeId,
    selectedRecipe,
    detectedSettings,
    sensorGeneration,
    recipeAdjustment,
    setSelectedFile,
    setSelectedRecipeId,
    saveCustomRecipe: saveRecipe,
    deleteCustomRecipe: deleteRecipe,
    captureCustomRecipePreview,
  };

  return <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>;
}

export function useAppState(): AppState {
  const context = useContext(AppStateContext);
  if (!context) {
    throw new Error("useAppState must be used within an AppStateProvider");
  }
  return context;
}
