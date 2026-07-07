import { useState } from "react";
import { SceneAnalysisPanel } from "@/components/ai/SceneAnalysisPanel";
import { TripPlannerChat } from "@/components/ai/TripPlannerChat";

type AIMode = "scene" | "trip";

export function AIPage() {
  const [mode, setMode] = useState<AIMode>("scene");

  return (
    <div className="flex h-full w-full flex-col bg-ink-950 text-ink-50">
      <header className="shrink-0 border-b border-ink-800 px-4 py-5 [padding-top:calc(1.25rem+env(safe-area-inset-top))]">
        <div className="mb-1.5 flex items-center gap-2 text-[11px] font-bold uppercase tracking-[0.15em] text-gold-500">
          <span className="h-px w-3 bg-gold-600" />
          AI Assistant
        </div>
        <h1 className="text-3xl font-black uppercase leading-[0.95] tracking-tight text-ink-50">
          Recipe AI<span className="text-gold-400">.</span>
        </h1>

        <div className="mt-4 flex rounded-md border border-ink-700 bg-ink-900 p-1">
          <button
            type="button"
            onClick={() => setMode("scene")}
            className={`flex-1 rounded px-3 py-1.5 text-xs font-bold uppercase tracking-wide transition-all ${
              mode === "scene" ? "bg-gold-500 text-ink-950" : "text-ink-400 hover:text-ink-100"
            }`}
          >
            Scene Suggestion
          </button>
          <button
            type="button"
            onClick={() => setMode("trip")}
            className={`flex-1 rounded px-3 py-1.5 text-xs font-bold uppercase tracking-wide transition-all ${
              mode === "trip" ? "bg-gold-500 text-ink-950" : "text-ink-400 hover:text-ink-100"
            }`}
          >
            Trip Planner
          </button>
        </div>
      </header>

      <div className="min-h-0 flex-1">
        {mode === "scene" ? (
          <div className="h-full overflow-y-auto p-4">
            <SceneAnalysisPanel />
          </div>
        ) : (
          <TripPlannerChat />
        )}
      </div>
    </div>
  );
}
