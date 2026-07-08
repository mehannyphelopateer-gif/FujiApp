import { useMemo, useRef, useState } from "react";
import { useAppState } from "@/context/AppStateContext";
import { processBatch, type BatchResult } from "@/lib/batch/processBatch";
import { createZip } from "@/lib/zip/createZip";

export function BatchView() {
  const { recipes } = useAppState();
  const inputRef = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [recipeSearch, setRecipeSearch] = useState("");
  const [selectedRecipeId, setSelectedRecipeId] = useState<string | null>(null);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [results, setResults] = useState<BatchResult[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const filteredRecipes = useMemo(() => {
    const query = recipeSearch.trim().toLowerCase();
    if (!query) return recipes;
    return recipes.filter((r) => r.name.toLowerCase().includes(query));
  }, [recipes, recipeSearch]);

  const selectedRecipe = recipes.find((r) => r.id === selectedRecipeId) ?? null;
  const isProcessing = progress !== null && progress.done < progress.total;

  function handleFilesChosen(fileList: FileList | null) {
    if (!fileList) return;
    const jpegs = [...fileList].filter((f) => f.type === "image/jpeg");
    setFiles(jpegs);
    setResults(null);
    setError(null);
  }

  async function handleProcess() {
    if (!selectedRecipe || files.length === 0) return;
    setError(null);
    setResults(null);
    setProgress({ done: 0, total: files.length });
    try {
      const batchResults = await processBatch(files, selectedRecipe, (done, total) => setProgress({ done, total }));
      setResults(batchResults);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Batch processing failed.");
    } finally {
      setProgress(null);
    }
  }

  async function handleDownloadZip() {
    if (!results) return;
    const entries = await Promise.all(
      results.map(async (r) => ({ name: r.name, data: new Uint8Array(await r.blob.arrayBuffer()) })),
    );
    const zip = createZip(entries);
    const url = URL.createObjectURL(zip);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${selectedRecipe?.name.toLowerCase().replace(/[^a-z0-9]+/g, "-") ?? "batch"}-export.zip`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  return (
    <div className="h-full w-full overflow-y-auto bg-ink-950 text-ink-50">
      <header className="border-b border-ink-800 px-4 py-6 [padding-top:calc(1.5rem+env(safe-area-inset-top))]">
        <div className="mb-1.5 flex items-center gap-2 text-[11px] font-bold uppercase tracking-[0.15em] text-gold-500">
          <span className="h-px w-3 bg-gold-600" />
          Bulk Export
        </div>
        <h1 className="text-4xl font-black uppercase leading-[0.95] tracking-tight text-ink-50">
          Batch<span className="text-gold-400">.</span>
        </h1>
        <p className="mt-2 text-xs text-ink-400">Apply one recipe to many photos at once, then download a zip.</p>
      </header>

      <div className="space-y-6 p-4">
        <div>
          <p className="mb-2 text-xs font-bold uppercase tracking-wide text-ink-400">1. Choose photos</p>
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            className="w-full rounded-md border-2 border-dashed border-ink-700 bg-ink-900 py-6 text-sm font-bold text-ink-200 hover:border-ink-500"
          >
            {files.length > 0 ? `${files.length} photo${files.length === 1 ? "" : "s"} selected` : "Select JPEGs"}
          </button>
          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg"
            multiple
            className="hidden"
            onChange={(event) => handleFilesChosen(event.target.files)}
          />
        </div>

        <div>
          <div className="mb-2 flex items-center justify-between">
            <p className="text-xs font-bold uppercase tracking-wide text-ink-400">2. Choose one recipe</p>
            <p className="text-[11px] font-semibold text-ink-500">{filteredRecipes.length} available</p>
          </div>
          <input
            type="search"
            value={recipeSearch}
            onChange={(event) => setRecipeSearch(event.target.value)}
            placeholder="Search recipes…"
            className="mb-2 w-full rounded-md border border-ink-700 bg-ink-900 px-3 py-2 text-sm text-ink-50 placeholder:text-ink-500 focus:border-gold-500 focus:outline-none"
          />
          <div className="flex max-h-72 flex-wrap gap-2 overflow-y-auto rounded-md border border-ink-800 bg-ink-900/50 p-2.5">
            {filteredRecipes.map((recipe) => (
              <button
                key={recipe.id}
                type="button"
                onClick={() => setSelectedRecipeId(recipe.id)}
                className={`rounded-full border px-3 py-1.5 text-xs font-semibold transition-colors ${
                  selectedRecipeId === recipe.id
                    ? "border-gold-500 bg-gold-500/10 text-gold-400"
                    : "border-ink-700 text-ink-400 hover:border-ink-500 hover:text-ink-100"
                }`}
              >
                {recipe.name}
              </button>
            ))}
            {filteredRecipes.length === 0 && (
              <p className="w-full py-4 text-center text-xs text-ink-500">No recipes match "{recipeSearch}".</p>
            )}
          </div>
        </div>

        <div>
          <p className="mb-2 text-xs font-bold uppercase tracking-wide text-ink-400">3. Process</p>
          <button
            type="button"
            onClick={handleProcess}
            disabled={!selectedRecipe || files.length === 0 || isProcessing}
            className="w-full rounded-md bg-gold-500 py-3 text-sm font-bold uppercase tracking-wide text-ink-950 hover:bg-gold-400 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {isProcessing
              ? `Processing ${progress?.done ?? 0} / ${progress?.total ?? 0}…`
              : `Apply "${selectedRecipe?.name ?? "…"}" to ${files.length} photo${files.length === 1 ? "" : "s"}`}
          </button>
          {error && <p className="mt-2 rounded-md bg-red-500/10 px-3 py-2 text-xs text-red-400">{error}</p>}
        </div>

        {results && (
          <div className="rounded-md border border-ink-800 bg-ink-900 p-4">
            <p className="mb-3 text-sm font-bold text-ink-50">{results.length} photos processed</p>
            <button
              type="button"
              onClick={() => void handleDownloadZip()}
              className="w-full rounded-md bg-gold-500 py-2.5 text-xs font-bold uppercase tracking-wide text-ink-950 hover:bg-gold-400"
            >
              Download All (.zip)
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
