// Matches Claude's efficient vision resolution — the scene's overall mood and
// lighting don't need full sensor resolution to classify, and downscaling
// keeps the request fast and cheap.
const MAX_DIMENSION = 1568;

export interface RecipeRecommendation {
  recipeId: string;
  reasoning: string;
}

export interface RecommendationResult {
  sceneDescription: string;
  recommendations: RecipeRecommendation[];
}

export async function recommendRecipeForPhoto(file: File): Promise<RecommendationResult> {
  const { base64, mimeType } = await downscaleToBase64(file);

  const response = await fetch("/api/recommend-recipe", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ imageBase64: base64, mimeType }),
  });

  if (!response.ok) {
    const errorBody = (await response.json().catch(() => null)) as { error?: string } | null;
    throw new Error(errorBody?.error ?? `Request failed with status ${response.status}`);
  }

  return response.json();
}

function downscaleToBase64(file: File): Promise<{ base64: string; mimeType: string }> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    const objectUrl = URL.createObjectURL(file);

    image.onload = () => {
      URL.revokeObjectURL(objectUrl);

      const scale = Math.min(1, MAX_DIMENSION / Math.max(image.naturalWidth, image.naturalHeight));
      const canvas = document.createElement("canvas");
      canvas.width = Math.round(image.naturalWidth * scale);
      canvas.height = Math.round(image.naturalHeight * scale);

      const ctx = canvas.getContext("2d");
      if (!ctx) {
        reject(new Error("Canvas 2D context unavailable."));
        return;
      }
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

      const dataUrl = canvas.toDataURL("image/jpeg", 0.85);
      resolve({ base64: dataUrl.slice(dataUrl.indexOf(",") + 1), mimeType: "image/jpeg" });
    };
    image.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      reject(new Error("Failed to load the image for analysis."));
    };
    image.src = objectUrl;
  });
}
