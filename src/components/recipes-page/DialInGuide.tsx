import { useMemo, useState } from "react";
import type { Recipe } from "@/types/recipe";
import { buildDialInSteps } from "@/lib/recipes/dialInSteps";

interface DialInGuideProps {
  recipe: Recipe;
  onClose: () => void;
}

/**
 * Zero-hardware fallback for programming a recipe onto the camera: a
 * checklist mirroring the camera's own IMAGE QUALITY SETTING menu order, so
 * the user can tick items off while dialing them in by hand. Independent of
 * any camera-link feature — works today, on every model, no cable needed.
 */
export function DialInGuide({ recipe, onClose }: DialInGuideProps) {
  const steps = useMemo(() => buildDialInSteps(recipe), [recipe]);
  const [checked, setChecked] = useState<Record<string, boolean>>({});

  const doneCount = steps.filter((step) => checked[step.id]).length;
  const allDone = doneCount === steps.length;

  function toggle(id: string) {
    setChecked((prev) => ({ ...prev, [id]: !prev[id] }));
  }

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/70 sm:items-center sm:p-4">
      <div className="flex max-h-[90vh] w-full flex-col rounded-t-lg border border-ink-700 bg-ink-950 sm:max-w-lg sm:rounded-lg">
        <div className="flex shrink-0 items-center justify-between border-b border-ink-800 px-4 py-3.5">
          <div>
            <h2 className="text-sm font-black uppercase tracking-wide text-ink-50">Dial In: {recipe.name}</h2>
            <p className="text-[11px] text-ink-500">
              {doneCount} of {steps.length} set on the camera
            </p>
          </div>
          <button type="button" onClick={onClose} className="text-ink-500 hover:text-ink-200">
            ✕
          </button>
        </div>

        <div className="h-1 shrink-0 bg-ink-900">
          <div
            className="h-1 bg-gold-500 transition-all"
            style={{ width: `${steps.length === 0 ? 0 : (doneCount / steps.length) * 100}%` }}
          />
        </div>

        <div className="flex-1 space-y-2 overflow-y-auto p-4">
          {steps.map((step, index) => {
            const isChecked = Boolean(checked[step.id]);
            return (
              <button
                key={step.id}
                type="button"
                onClick={() => toggle(step.id)}
                className={`flex w-full items-start gap-3 rounded-md border px-3 py-2.5 text-left transition-colors ${
                  isChecked ? "border-ink-800 bg-ink-900/50 opacity-60" : "border-ink-700 bg-ink-900 hover:border-gold-700/60"
                }`}
              >
                <span
                  className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded border text-[11px] font-bold ${
                    isChecked ? "border-gold-500 bg-gold-500 text-ink-950" : "border-ink-600 text-ink-600"
                  }`}
                >
                  {isChecked ? "✓" : index + 1}
                </span>
                <span className="min-w-0 flex-1">
                  <span className={`block text-sm font-bold ${isChecked ? "text-ink-400 line-through" : "text-ink-50"}`}>
                    {step.label}
                  </span>
                  <span className="block font-mono text-[10px] uppercase tracking-wide text-ink-500">{step.menuPath}</span>
                </span>
                <span className="mt-0.5 shrink-0 rounded bg-ink-800 px-2 py-1 font-mono text-xs font-medium text-gold-300">
                  {step.value}
                </span>
              </button>
            );
          })}
        </div>

        <div className="flex shrink-0 items-center justify-between gap-2 border-t border-ink-800 px-4 py-3">
          <button
            type="button"
            onClick={() => setChecked({})}
            className="text-xs font-bold uppercase tracking-wide text-ink-500 hover:text-ink-300"
          >
            Reset
          </button>
          <button
            type="button"
            onClick={onClose}
            className={`rounded-md px-4 py-2 text-xs font-bold uppercase tracking-wide transition-colors ${
              allDone ? "bg-gold-500 text-ink-950 hover:bg-gold-400" : "border border-ink-700 text-ink-300 hover:text-ink-100"
            }`}
          >
            {allDone ? "Done" : "Close"}
          </button>
        </div>
      </div>
    </div>
  );
}
