import { useMemo, useState } from "react";
import { useAppState } from "@/context/AppStateContext";
import { CompareSlot } from "@/components/compare/CompareSlot";

const MAX_COMPARE = 4;

export function CompareView() {
  const { recipes, previewUrl, detectedSettings, compatibleRecipes } = useAppState();
  const [search, setSearch] = useState("");
  const [selectedIds, setSelectedIds] = useState<string[]>([]);

  const filteredRecipes = useMemo(() => {
    const query = search.trim().toLowerCase();
    if (!query) return compatibleRecipes;
    return compatibleRecipes.filter((recipe) => recipe.name.toLowerCase().includes(query));
  }, [compatibleRecipes, search]);

  function toggleRecipe(id: string) {
    setSelectedIds((prev) => {
      if (prev.includes(id)) return prev.filter((existing) => existing !== id);
      if (prev.length >= MAX_COMPARE) return prev;
      return [...prev, id];
    });
  }

  const selectedRecipes = selectedIds
    .map((id) => recipes.find((recipe) => recipe.id === id))
    .filter((recipe): recipe is NonNullable<typeof recipe> => recipe !== undefined);

  return (
    <div className="h-full w-full overflow-y-auto bg-ink-950 text-ink-50">
      <header className="border-b border-ink-800 px-4 py-6 [padding-top:calc(1.5rem+env(safe-area-inset-top))]">
        <div className="mb-1.5 flex items-center gap-2 text-[11px] font-bold uppercase tracking-[0.15em] text-gold-500">
          <span className="h-px w-3 bg-gold-600" />
          Side by Side
        </div>
        <h1 className="text-4xl font-black uppercase leading-[0.95] tracking-tight text-ink-50">
          Compare<span className="text-gold-400">.</span>
        </h1>
        <p className="mt-2 text-xs text-ink-400">Pick up to {MAX_COMPARE} recipes to render side by side.</p>
      </header>

      {!previewUrl ? (
        <div className="flex flex-col items-center gap-2 px-4 py-16 text-center">
          <p className="text-sm text-ink-400">Upload a photo on the Preview tab first.</p>
        </div>
      ) : (
        <>
          <div className="border-b border-ink-800 px-4 py-3">
            <input
              type="search"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search recipes to add…"
              className="w-full rounded-md border border-ink-700 bg-ink-900 px-3 py-2 text-sm text-ink-50 placeholder:text-ink-500 focus:border-gold-500 focus:outline-none"
            />
            <div className="mt-3 flex flex-wrap gap-2">
              {filteredRecipes.slice(0, 24).map((recipe) => {
                const isSelected = selectedIds.includes(recipe.id);
                return (
                  <button
                    key={recipe.id}
                    type="button"
                    onClick={() => toggleRecipe(recipe.id)}
                    disabled={!isSelected && selectedIds.length >= MAX_COMPARE}
                    className={`rounded-full border px-3 py-1.5 text-xs font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
                      isSelected
                        ? "border-gold-500 bg-gold-500/10 text-gold-400"
                        : "border-ink-700 text-ink-400 hover:border-ink-500 hover:text-ink-100"
                    }`}
                  >
                    {recipe.name}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4 p-4 lg:grid-cols-4">
            {selectedRecipes.map((recipe) => (
              <CompareSlot
                key={recipe.id}
                recipe={recipe}
                previewUrl={previewUrl}
                detectedSettings={detectedSettings}
                onRemove={() => toggleRecipe(recipe.id)}
              />
            ))}
            {selectedRecipes.length === 0 && (
              <p className="col-span-full py-12 text-center text-sm text-ink-500">
                Select recipes above to compare them side by side.
              </p>
            )}
          </div>
        </>
      )}
    </div>
  );
}
