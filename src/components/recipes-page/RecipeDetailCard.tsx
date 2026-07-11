import { useRef, useState } from "react";
import type { Recipe } from "@/types/recipe";
import { RecipeParameterList } from "@/components/recipes/RecipeParameterList";
import { RecipeEditor } from "@/components/recipes/RecipeEditor";
import { DialInGuide } from "@/components/recipes-page/DialInGuide";
import { ApplyToCameraModal } from "@/components/camera/ApplyToCameraModal";
import { useAppState } from "@/context/AppStateContext";

interface RecipeDetailCardProps {
  recipe: Recipe;
  isFavorite: boolean;
  onToggleFavorite: (id: string) => void;
}

export function RecipeDetailCard({ recipe, isFavorite, onToggleFavorite }: RecipeDetailCardProps) {
  const [expanded, setExpanded] = useState(false);
  const [editing, setEditing] = useState(false);
  const [dialingIn, setDialingIn] = useState(false);
  const [applyingToCamera, setApplyingToCamera] = useState(false);
  const [isUpdatingCover, setIsUpdatingCover] = useState(false);
  const coverInputRef = useRef<HTMLInputElement>(null);
  const { deleteCustomRecipe, setRecipeCoverPhoto, clearRecipeCoverPhoto } = useAppState();

  const thumbnailSrc = recipe.previewImage ?? `/recipe-previews/${recipe.id}.jpg`;

  async function handleCoverPhotoChosen(file: File | undefined) {
    if (!file) return;
    setIsUpdatingCover(true);
    try {
      await setRecipeCoverPhoto(recipe.id, file);
    } finally {
      setIsUpdatingCover(false);
    }
  }

  return (
    <div className="group overflow-hidden rounded-lg border border-ink-800 bg-ink-900 transition-colors hover:border-gold-700/60">
      <div className="relative aspect-[3/2] w-full overflow-hidden bg-black">
        {recipe.isCustom && !recipe.previewImage ? (
          <div className="flex h-full w-full items-center justify-center bg-ink-800 text-ink-600">
            <svg viewBox="0 0 24 24" fill="none" strokeWidth="1.4" stroke="currentColor" className="h-10 w-10">
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 5h16v14H4z" />
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 15l4-4 4 4 3-3 5 5" />
            </svg>
          </div>
        ) : (
          <img
            src={thumbnailSrc}
            alt={`Example photo with the ${recipe.name} recipe applied`}
            loading="lazy"
            className="h-full w-full object-cover transition-transform duration-500 ease-out group-hover:scale-105"
          />
        )}
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-black/85 via-black/10 to-transparent" />

        {isUpdatingCover && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/70">
            <svg className="h-6 w-6 animate-spin text-gold-400" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
            </svg>
          </div>
        )}

        {recipe.isCustom && (
          <span className="absolute left-2.5 top-2.5 rounded bg-gold-500/90 px-1.5 py-0.5 text-[9px] font-black uppercase tracking-wide text-ink-950">
            Custom
          </span>
        )}

        <button
          type="button"
          aria-pressed={isFavorite}
          aria-label={isFavorite ? `Remove ${recipe.name} from favorites` : `Add ${recipe.name} to favorites`}
          onClick={() => onToggleFavorite(recipe.id)}
          className={`absolute right-2.5 top-2.5 flex h-7 w-7 items-center justify-center rounded-full text-sm backdrop-blur-md transition-all ${
            isFavorite ? "bg-gold-500 text-ink-950" : "bg-black/50 text-white hover:bg-black/70"
          }`}
        >
          {isFavorite ? "★" : "☆"}
        </button>

        <div className="absolute inset-x-0 bottom-0 p-3.5">
          <p className="text-base font-black uppercase leading-tight tracking-tight text-white">{recipe.name}</p>
          <p className="font-mono text-[10px] font-semibold uppercase tracking-wider text-gold-400">
            {recipe.baseFilmSimulation}
          </p>
        </div>
      </div>

      <div className="p-3.5">
        {recipe.description && (
          <p className="line-clamp-2 text-xs leading-relaxed text-ink-400">{recipe.description}</p>
        )}

        <div className="mt-3 flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={() => setExpanded((prev) => !prev)}
            className="flex items-center gap-1 text-xs font-bold uppercase tracking-wide text-gold-400 transition-colors hover:text-gold-300"
          >
            {expanded ? "Hide settings" : "Show settings"}
            <svg
              viewBox="0 0 24 24"
              fill="none"
              strokeWidth="2.5"
              stroke="currentColor"
              className={`h-3 w-3 transition-transform ${expanded ? "rotate-180" : ""}`}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 9l6 6 6-6" />
            </svg>
          </button>

          <button
            type="button"
            onClick={() => setEditing(true)}
            className="text-xs font-bold uppercase tracking-wide text-ink-400 transition-colors hover:text-ink-100"
          >
            {recipe.isCustom ? "Edit" : "Duplicate & Edit"}
          </button>

          <button
            type="button"
            onClick={() => setDialingIn(true)}
            className="text-xs font-bold uppercase tracking-wide text-gold-400 transition-colors hover:text-gold-300"
          >
            Dial In
          </button>

          <button
            type="button"
            onClick={() => setApplyingToCamera(true)}
            className="text-xs font-bold uppercase tracking-wide text-gold-400 transition-colors hover:text-gold-300"
          >
            Apply to Camera
          </button>

          <button
            type="button"
            onClick={() => coverInputRef.current?.click()}
            className="text-xs font-bold uppercase tracking-wide text-ink-400 transition-colors hover:text-ink-100"
          >
            My Photo
          </button>
          <input
            ref={coverInputRef}
            type="file"
            accept="image/jpeg"
            className="hidden"
            onChange={(event) => {
              void handleCoverPhotoChosen(event.target.files?.[0]);
              event.target.value = "";
            }}
          />

          {recipe.previewImage && !recipe.isCustom && (
            <button
              type="button"
              onClick={() => clearRecipeCoverPhoto(recipe.id)}
              className="text-xs font-bold uppercase tracking-wide text-ink-500 transition-colors hover:text-ink-300"
            >
              Reset Photo
            </button>
          )}

          {recipe.isCustom && (
            <button
              type="button"
              onClick={() => deleteCustomRecipe(recipe.id)}
              className="text-xs font-bold uppercase tracking-wide text-red-500/80 transition-colors hover:text-red-400"
            >
              Delete
            </button>
          )}
        </div>

        {expanded && (
          <div className="mt-4 space-y-4 border-t border-ink-800 pt-4">
            {recipe.isCustom && !recipe.previewImage ? (
              <div className="flex aspect-[3/2] w-full items-center justify-center rounded-md bg-ink-800 text-ink-600">
                <svg viewBox="0 0 24 24" fill="none" strokeWidth="1.4" stroke="currentColor" className="h-10 w-10">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4 5h16v14H4z" />
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4 15l4-4 4 4 3-3 5 5" />
                </svg>
              </div>
            ) : (
              <img
                src={thumbnailSrc}
                alt={`Full example photo with the ${recipe.name} recipe applied`}
                className="w-full rounded-md bg-black object-contain"
              />
            )}
            <RecipeParameterList recipe={recipe} />
          </div>
        )}
      </div>

      {editing && <RecipeEditor baseRecipe={recipe} onClose={() => setEditing(false)} />}
      {dialingIn && <DialInGuide recipe={recipe} onClose={() => setDialingIn(false)} />}
      {applyingToCamera && <ApplyToCameraModal recipe={recipe} onClose={() => setApplyingToCamera(false)} />}
    </div>
  );
}
