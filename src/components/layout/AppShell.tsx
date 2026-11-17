import { Dropzone } from "@/components/upload/Dropzone";
import { ImageViewport } from "@/components/viewer/ImageViewport";
import { CameraRenderToggle } from "@/components/viewer/CameraRenderToggle";
import { RecipeGrid } from "@/components/recipes/RecipeGrid";
import { RecipeParameterList } from "@/components/recipes/RecipeParameterList";
import { DetectedSettingsPanel } from "@/components/exif/DetectedSettingsPanel";
import { useAppState } from "@/context/AppStateContext";
import { useCameraRenderSync } from "@/hooks/useCameraRenderSync";
import { lazy, Suspense, useState } from "react";

const experimentalEnabled = import.meta.env.DEV || import.meta.env.VITE_EXPERIMENTAL_PROCESSOR === "1";
const ExperimentalRawPreview = experimentalEnabled ? lazy(() => import("@/components/viewer/ExperimentalRawPreview")) : null;

export function AppShell() {
  const [experimental, setExperimental] = useState(false);
  if (!ExperimentalRawPreview) return <StandardPreviewShell />;
  return (
    <div className="flex h-full min-h-0 flex-col">
      <div role="group" aria-label="Preview processor" className="flex shrink-0 flex-wrap gap-2 border-b border-ink-800 px-4 py-2">
        <button type="button" aria-pressed={!experimental} onClick={() => setExperimental(false)}
          className={`rounded px-3 py-2 text-xs font-bold ${!experimental ? "bg-gold-500 text-ink-950" : "text-ink-300"}`}>Standard Preview</button>
        <button type="button" aria-pressed={experimental} onClick={() => setExperimental(true)}
          className={`rounded px-3 py-2 text-xs font-bold ${experimental ? "bg-gold-500 text-ink-950" : "text-ink-300"}`}>Experimental Provia</button>
      </div>
      <div className="min-h-0 flex-1">{experimental ? <Suspense fallback={<p className="p-4 text-sm text-ink-400">Loading experimental processor…</p>}><ExperimentalRawPreview /></Suspense> : <StandardPreviewShell />}</div>
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="mb-3 flex items-center gap-2 text-[11px] font-bold uppercase tracking-[0.15em] text-ink-500">
      <span className="h-px w-3 bg-ink-700" />
      {children}
    </h2>
  );
}

function StandardPreviewShell() {
  const { selectedRecipe } = useAppState();
  useCameraRenderSync();

  return (
    // Below lg, this scrolls as a normal page within its flex-1 slot (the
    // parent <App> reserves the rest of the screen height for the bottom
    // tab bar) — with 100+ recipe cards the stacked mobile layout is much
    // taller than one screen. At lg+ we switch to the fixed-height 3-pane
    // app shell, where each pane scrolls independently instead of the whole
    // page.
    <div className="flex h-full w-full flex-col overflow-y-auto bg-ink-950 text-ink-50 lg:overflow-hidden">
      <header className="flex shrink-0 items-center gap-2.5 border-b border-ink-800 px-4 py-3.5 [padding-top:calc(0.875rem+env(safe-area-inset-top))]">
        <span className="flex h-7 w-7 items-center justify-center rounded-full border border-gold-500/60">
          <svg viewBox="0 0 24 24" fill="none" className="h-3.5 w-3.5 text-gold-400">
            <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.6" />
            <path
              stroke="currentColor"
              strokeWidth="1.4"
              strokeLinejoin="round"
              d="M12 5.5 15 12l-3 6.5-3-6.5 3-6.5Z"
            />
          </svg>
        </span>
        <p className="text-sm font-black uppercase tracking-wide text-ink-50">
          Fuji<span className="ml-1 font-medium tracking-widest text-ink-400">App</span>
        </p>
      </header>

      <div className="flex flex-1 flex-col lg:flex-row lg:overflow-hidden">
        <aside className="w-full shrink-0 space-y-6 border-b border-ink-800 p-4 lg:h-full lg:w-72 lg:overflow-y-auto lg:border-b-0 lg:border-r">
          <div>
            <SectionLabel>Upload</SectionLabel>
            <Dropzone />
          </div>

          <div>
            <SectionLabel>Current Image Settings</SectionLabel>
            <DetectedSettingsPanel />
          </div>
        </aside>

        <main className="flex min-h-[40vh] flex-1 flex-col items-center justify-center gap-3 bg-black/30 p-4 lg:min-h-0 lg:p-6">
          <div className="flex w-full shrink-0 justify-end">
            <CameraRenderToggle />
          </div>
          <div className="min-h-0 w-full flex-1">
            <ImageViewport />
          </div>
        </main>

        <aside className="w-full shrink-0 space-y-6 border-t border-ink-800 p-4 lg:h-full lg:w-80 lg:overflow-y-auto lg:border-t-0 lg:border-l">
          <div>
            <SectionLabel>Recipes</SectionLabel>
            <RecipeGrid />
          </div>

          <div>
            <SectionLabel>Recipe Parameters</SectionLabel>
            <RecipeParameterList recipe={selectedRecipe} />
          </div>
        </aside>
      </div>
    </div>
  );
}
