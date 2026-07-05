type ParameterVariant = "slider" | "badge" | "segments";

interface ParameterReadoutProps {
  label: string;
  variant: ParameterVariant;
  value: number | string;
  /** Required when variant === "slider". */
  min?: number;
  /** Required when variant === "slider". */
  max?: number;
  unit?: string;
}

const SEGMENT_COUNT = 3;
const STRENGTH_TO_SEGMENTS: Record<string, number> = { Off: 0, Weak: 1, Strong: 2 };

export function ParameterReadout({ label, variant, value, min, max, unit }: ParameterReadoutProps) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-xs">
        <span className="font-medium uppercase tracking-wide text-ink-400">{label}</span>
        {variant !== "badge" && (
          <span className="rounded bg-ink-800 px-1.5 py-0.5 font-mono text-[11px] text-gold-300">
            {value}
            {unit ?? ""}
          </span>
        )}
      </div>

      {variant === "badge" && (
        <span className="inline-block rounded bg-ink-800 px-2.5 py-1 font-mono text-xs font-medium text-ink-100">
          {value}
        </span>
      )}

      {variant === "slider" && typeof value === "number" && min !== undefined && max !== undefined && (
        <div className="relative h-1 w-full rounded-full bg-ink-800">
          {(() => {
            const zeroPercent = ((0 - min) / (max - min)) * 100;
            const valuePercent = ((value - min) / (max - min)) * 100;
            const left = Math.min(zeroPercent, valuePercent);
            const width = Math.abs(valuePercent - zeroPercent);
            return (
              <>
                <div className="absolute top-0 h-1 w-px bg-ink-600" style={{ left: `${zeroPercent}%` }} />
                <div className="absolute top-0 h-1 rounded-full bg-gold-500" style={{ left: `${left}%`, width: `${width}%` }} />
              </>
            );
          })()}
        </div>
      )}

      {variant === "segments" && typeof value === "string" && (
        <div className="flex gap-1">
          {Array.from({ length: SEGMENT_COUNT }).map((_, index) => (
            <span
              key={index}
              className={`h-1 flex-1 rounded-full transition-colors ${
                index < (STRENGTH_TO_SEGMENTS[value] ?? 0) ? "bg-gold-500" : "bg-ink-800"
              }`}
            />
          ))}
        </div>
      )}
    </div>
  );
}
