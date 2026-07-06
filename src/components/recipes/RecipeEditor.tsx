import { useState } from "react";
import type {
  BaseFilmSimulation,
  DynamicRange,
  EffectStrength,
  GrainSize,
  Recipe,
  WhiteBalanceMode,
} from "@/types/recipe";
import { useAppState } from "@/context/AppStateContext";

const FILM_SIMULATIONS: BaseFilmSimulation[] = [
  "Provia",
  "Velvia",
  "Astia",
  "Classic Chrome",
  "Pro Neg Hi",
  "Pro Neg Std",
  "Classic Negative",
  "Eterna",
  "Eterna Bleach Bypass",
  "Acros",
  "Monochrome",
  "Sepia",
  "Nostalgic Neg",
  "Reala Ace",
];
const DYNAMIC_RANGES: DynamicRange[] = ["DR-AUTO", "DR100", "DR200", "DR400"];
const WB_MODES: WhiteBalanceMode[] = [
  "Auto",
  "Daylight",
  "Shade",
  "Fluorescent1",
  "Fluorescent2",
  "Fluorescent3",
  "Incandescent",
  "Underwater",
  "Kelvin",
];
const STRENGTHS: EffectStrength[] = ["Off", "Weak", "Strong"];
const GRAIN_SIZES: GrainSize[] = ["Small", "Large"];
const BROAD_SENSORS = ["X-Trans V", "X-Trans IV", "X-Trans III", "X-Trans II", "X-Trans I"];

const NEUTRAL_DEFAULTS: Recipe = {
  id: "",
  name: "",
  baseFilmSimulation: "Provia",
  dynamicRange: "DR-AUTO",
  whiteBalance: { mode: "Auto", shift: { red: 0, blue: 0 } },
  highlightTone: 0,
  shadowTone: 0,
  color: 0,
  sharpness: 0,
  colorChromeEffect: "Off",
  colorChromeFxBlue: "Off",
  grainEffect: "Off",
  compatibleSensors: BROAD_SENSORS,
};

interface RecipeEditorProps {
  /** Pass an existing recipe to edit (custom) or duplicate-and-customize (built-in). Omit to start fresh. */
  baseRecipe?: Recipe;
  onClose: () => void;
}

function SegmentedField<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: T[];
  onChange: (value: T) => void;
}) {
  return (
    <div className="space-y-1.5">
      <span className="text-xs font-medium uppercase tracking-wide text-ink-400">{label}</span>
      <div className="flex rounded-md border border-ink-700 bg-ink-900 p-0.5">
        {options.map((option) => (
          <button
            key={option}
            type="button"
            onClick={() => onChange(option)}
            className={`flex-1 rounded px-2 py-1 text-xs font-semibold transition-colors ${
              value === option ? "bg-gold-500 text-ink-950" : "text-ink-400 hover:text-ink-100"
            }`}
          >
            {option}
          </button>
        ))}
      </div>
    </div>
  );
}

function SliderField({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="block space-y-1.5">
      <span className="flex items-center justify-between text-xs font-medium uppercase tracking-wide text-ink-400">
        {label}
        <span className="font-mono normal-case text-gold-300">{value}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(parseFloat(event.target.value))}
        className="w-full accent-gold-500"
      />
    </label>
  );
}

function SelectField<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: T[];
  onChange: (value: T) => void;
}) {
  return (
    <label className="block space-y-1.5">
      <span className="text-xs font-medium uppercase tracking-wide text-ink-400">{label}</span>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value as T)}
        className="w-full rounded-md border border-ink-700 bg-ink-900 px-2.5 py-2 text-sm text-ink-50 focus:border-gold-500 focus:outline-none"
      >
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}

export function RecipeEditor({ baseRecipe, onClose }: RecipeEditorProps) {
  const { saveCustomRecipe, captureCustomRecipePreview, previewUrl } = useAppState();

  const isEditingExistingCustom = baseRecipe?.isCustom === true;

  const [draft, setDraft] = useState<Recipe>(() => {
    if (!baseRecipe) return { ...NEUTRAL_DEFAULTS, name: "My Recipe" };
    if (isEditingExistingCustom) return { ...baseRecipe };
    return { ...baseRecipe, id: "", name: `${baseRecipe.name} (Custom)`, isCustom: true, previewImage: undefined };
  });

  function update<K extends keyof Recipe>(key: K, value: Recipe[K]) {
    setDraft((prev) => ({ ...prev, [key]: value }));
  }

  function handleSave() {
    if (!draft.name.trim()) return;
    const id = isEditingExistingCustom && draft.id ? draft.id : `custom-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const finalRecipe: Recipe = { ...draft, id, isCustom: true, compatibleSensors: BROAD_SENSORS };
    saveCustomRecipe(finalRecipe);
    if (previewUrl) captureCustomRecipePreview(id);
    onClose();
  }

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/70 sm:items-center sm:p-4">
      <div className="flex max-h-[90vh] w-full flex-col rounded-t-lg border border-ink-700 bg-ink-950 sm:max-w-lg sm:rounded-lg">
        <div className="flex shrink-0 items-center justify-between border-b border-ink-800 px-4 py-3.5">
          <h2 className="text-sm font-black uppercase tracking-wide text-ink-50">
            {isEditingExistingCustom ? "Edit Recipe" : "New Custom Recipe"}
          </h2>
          <button type="button" onClick={onClose} className="text-ink-500 hover:text-ink-200">
            ✕
          </button>
        </div>

        <div className="flex-1 space-y-4 overflow-y-auto p-4">
          <label className="block space-y-1.5">
            <span className="text-xs font-medium uppercase tracking-wide text-ink-400">Name</span>
            <input
              type="text"
              value={draft.name}
              onChange={(event) => update("name", event.target.value)}
              className="w-full rounded-md border border-ink-700 bg-ink-900 px-2.5 py-2 text-sm text-ink-50 focus:border-gold-500 focus:outline-none"
            />
          </label>

          <label className="block space-y-1.5">
            <span className="text-xs font-medium uppercase tracking-wide text-ink-400">Description (optional)</span>
            <textarea
              value={draft.description ?? ""}
              onChange={(event) => update("description", event.target.value)}
              rows={2}
              className="w-full resize-none rounded-md border border-ink-700 bg-ink-900 px-2.5 py-2 text-sm text-ink-50 focus:border-gold-500 focus:outline-none"
            />
          </label>

          <SelectField
            label="Film Simulation"
            value={draft.baseFilmSimulation}
            options={FILM_SIMULATIONS}
            onChange={(value) => update("baseFilmSimulation", value)}
          />
          <SelectField
            label="Dynamic Range"
            value={draft.dynamicRange}
            options={DYNAMIC_RANGES}
            onChange={(value) => update("dynamicRange", value)}
          />
          <SelectField
            label="White Balance Mode"
            value={draft.whiteBalance.mode}
            options={WB_MODES}
            onChange={(mode) => update("whiteBalance", { ...draft.whiteBalance, mode })}
          />
          {draft.whiteBalance.mode === "Kelvin" && (
            <SliderField
              label="Kelvin"
              value={draft.whiteBalance.kelvin ?? 5000}
              min={2500}
              max={10000}
              step={50}
              onChange={(kelvin) => update("whiteBalance", { ...draft.whiteBalance, kelvin })}
            />
          )}
          <SliderField
            label="WB Shift — Red"
            value={draft.whiteBalance.shift.red}
            min={-9}
            max={9}
            onChange={(red) => update("whiteBalance", { ...draft.whiteBalance, shift: { ...draft.whiteBalance.shift, red } })}
          />
          <SliderField
            label="WB Shift — Blue"
            value={draft.whiteBalance.shift.blue}
            min={-9}
            max={9}
            onChange={(blue) => update("whiteBalance", { ...draft.whiteBalance, shift: { ...draft.whiteBalance.shift, blue } })}
          />

          <SliderField label="Highlight Tone" value={draft.highlightTone} min={-2} max={4} step={0.5} onChange={(v) => update("highlightTone", v)} />
          <SliderField label="Shadow Tone" value={draft.shadowTone} min={-2} max={4} step={0.5} onChange={(v) => update("shadowTone", v)} />
          <SliderField label="Color" value={draft.color} min={-4} max={4} onChange={(v) => update("color", v)} />
          <SliderField label="Sharpness" value={draft.sharpness} min={-4} max={4} onChange={(v) => update("sharpness", v)} />

          <SegmentedField
            label="Color Chrome Effect"
            value={draft.colorChromeEffect}
            options={STRENGTHS}
            onChange={(v) => update("colorChromeEffect", v)}
          />
          <SegmentedField
            label="Color Chrome FX Blue"
            value={draft.colorChromeFxBlue}
            options={STRENGTHS}
            onChange={(v) => update("colorChromeFxBlue", v)}
          />
          <SegmentedField label="Grain Effect" value={draft.grainEffect} options={STRENGTHS} onChange={(v) => update("grainEffect", v)} />
          {draft.grainEffect !== "Off" && (
            <SegmentedField
              label="Grain Size"
              value={draft.grainSize ?? "Small"}
              options={GRAIN_SIZES}
              onChange={(v) => update("grainSize", v)}
            />
          )}

          <SliderField
            label="Clarity"
            value={draft.clarity ?? 0}
            min={-5}
            max={5}
            onChange={(v) => update("clarity", v)}
          />

          {!previewUrl && (
            <p className="rounded-md border border-ink-800 bg-ink-900 px-3 py-2 text-xs text-ink-500">
              Upload a photo on the Preview tab first if you want an accurate thumbnail generated for this recipe.
            </p>
          )}
        </div>

        <div className="flex shrink-0 gap-2 border-t border-ink-800 p-4">
          <button
            type="button"
            onClick={onClose}
            className="flex-1 rounded-md border border-ink-700 py-2.5 text-xs font-bold uppercase tracking-wide text-ink-300 hover:bg-ink-900"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleSave}
            disabled={!draft.name.trim()}
            className="flex-1 rounded-md bg-gold-500 py-2.5 text-xs font-bold uppercase tracking-wide text-ink-950 hover:bg-gold-400 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Save Recipe
          </button>
        </div>
      </div>
    </div>
  );
}
