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
import { useRecipePreviewOverrides } from "@/hooks/useRecipePreviewOverrides";

// Small — this only ever backs a thumbnail-sized cover photo, not an export.
const COVER_PHOTO_MAX_DIMENSION = 640;

// Shown in the Preview tab before the user uploads anything, so recipe
// switching and the before/after slider are usable immediately. Same photo
// for every visitor — swapped out entirely once a real file is selected.
const DEFAULT_PREVIEW_PHOTO_URL = "/sample-photo/base.jpg";

// Cover photos are real photos already shot with that recipe in-camera —
// running them back through the WebGL engine would apply the recipe a
// second time (extra grain on top of grain, color chrome pushed twice,
// etc.), so this only resizes/compresses via a plain 2D canvas. No recipe
// math involved.
function resizeToDataUrl(file: File, maxDimension: number): Promise<string> {
  return new Promise((resolve, reject) => {
    const objectUrl = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(objectUrl);
      const scale = Math.min(1, maxDimension / Math.max(image.naturalWidth, image.naturalHeight));
      const canvas = document.createElement("canvas");
      canvas.width = Math.round(image.naturalWidth * scale);
      canvas.height = Math.round(image.naturalHeight * scale);
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        reject(new Error("Canvas 2D context unavailable."));
        return;
      }
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      resolve(canvas.toDataURL("image/jpeg", 0.85));
    };
    image.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      reject(new Error("Failed to read image."));
    };
    image.src = objectUrl;
  });
}

interface AppState {
  recipes: Recipe[];
  /** Recipes compatible with the uploaded photo's detected sensor generation. All recipes if unknown. */
  compatibleRecipes: Recipe[];
  selectedFile: File | null;
  previewUrl: string | null;
  selectedRecipeId: string;
  selectedRecipe: Recipe;
  detectedSettings: DetectedSettings | null;
  /** True once a native RAW decode (see setNeutralRenderFile) is standing in for the uploaded photo. */
  isNeutralPreview: boolean;
  sensorGeneration: string | null;
  recipeAdjustment: RecipeAdjustment;
  setSelectedFile: (file: File | null) => void;
  /**
   * True RAW-demosaiced version of the current .RAF upload (native iOS
   * only) — see decodeNeutralRaf's doc comment. When set, it replaces
   * selectedFile as the actual WebGL base image, and recipeAdjustment
   * treats the baseline as neutral rather than subtracting the original
   * photo's detected settings (that photo's baked-in look was rendered by
   * the camera's JPEG engine, which this decode never went through).
   */
  setNeutralRenderFile: (file: File | null) => void;
  setSelectedRecipeId: (id: string) => void;
  saveCustomRecipe: (recipe: Recipe) => void;
  deleteCustomRecipe: (id: string) => void;
  captureCustomRecipePreview: (id: string) => void;
  /** Applies `recipe` to `photo` and sets the result as that recipe's cover photo everywhere in the app. */
  setRecipeCoverPhoto: (recipeId: string, photo: File) => Promise<void>;
  clearRecipeCoverPhoto: (recipeId: string) => void;
}

const AppStateContext = createContext<AppState | null>(null);

export function AppStateProvider({ children }: { children: ReactNode }) {
  const [selectedFile, setSelectedFileState] = useState<File | null>(null);
  const [neutralRenderFile, setNeutralRenderFile] = useState<File | null>(null);
  const [selectedRecipeId, setSelectedRecipeId] = useState<string>(builtInRecipes[0].id);
  const [previewUrl, setPreviewUrl] = useState<string | null>(DEFAULT_PREVIEW_PHOTO_URL);
  const [detectedSettings, setDetectedSettings] = useState<DetectedSettings | null>(null);

  // A newly-selected photo can't have a neutral render carried over from
  // whatever was uploaded before it — onNeutralFile (if any) arrives
  // asynchronously afterward and re-sets this itself.
  const setSelectedFile = (file: File | null) => {
    setNeutralRenderFile(null);
    setSelectedFileState(file);
  };
  const { customRecipes, saveRecipe, deleteRecipe, setPreviewImage } = useCustomRecipes();
  const { overrides, setOverride, clearOverride } = useRecipePreviewOverrides();

  const recipes = useMemo(() => {
    const merged = [...builtInRecipes, ...customRecipes];
    if (Object.keys(overrides).length === 0) return merged;
    return merged.map((recipe) => (overrides[recipe.id] ? { ...recipe, previewImage: overrides[recipe.id] } : recipe));
  }, [customRecipes, overrides]);

  useEffect(() => {
    const renderFile = neutralRenderFile ?? selectedFile;
    if (!renderFile) {
      setPreviewUrl(DEFAULT_PREVIEW_PHOTO_URL);
      return;
    }
    const url = URL.createObjectURL(renderFile);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [selectedFile, neutralRenderFile]);

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
    // A neutral render has no baked-in look to subtract against — passing
    // the original photo's detectedSettings here would incorrectly try to
    // neutralize a baseline that was never actually applied to these pixels.
    () => computeRecipeAdjustment(neutralRenderFile ? null : detectedSettings, selectedRecipe),
    [detectedSettings, neutralRenderFile, selectedRecipe],
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

  // Lets a user pick a photo of their own — already shot with that recipe,
  // typically — and use it directly as that recipe's cover. For built-in
  // recipes too, not just custom ones, since this is about personalizing
  // the library rather than editing a recipe's settings.
  async function setRecipeCoverPhoto(recipeId: string, photo: File) {
    const dataUrl = await resizeToDataUrl(photo, COVER_PHOTO_MAX_DIMENSION);
    setOverride(recipeId, dataUrl);
  }

  const value: AppState = {
    recipes,
    compatibleRecipes,
    selectedFile,
    previewUrl,
    selectedRecipeId,
    selectedRecipe,
    detectedSettings,
    isNeutralPreview: neutralRenderFile !== null,
    sensorGeneration,
    recipeAdjustment,
    setSelectedFile,
    setNeutralRenderFile,
    setSelectedRecipeId,
    saveCustomRecipe: saveRecipe,
    deleteCustomRecipe: deleteRecipe,
    captureCustomRecipePreview,
    setRecipeCoverPhoto,
    clearRecipeCoverPhoto: clearOverride,
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
