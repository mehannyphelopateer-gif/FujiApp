import { useEffect, useState } from "react";
import type { Recipe } from "@/types/recipe";
import { useCameraLink, type WriteResult } from "@/context/CameraLinkContext";

interface ApplyToCameraModalProps {
  recipe: Recipe;
  onClose: () => void;
}

const SLOT_NUMBERS = [1, 2, 3, 4, 5, 6, 7];

export function ApplyToCameraModal({ recipe, onClose }: ApplyToCameraModalProps) {
  const { isNative, status, deviceName, error, slots, isScanning, isWriting, connect, scanSlots, writeRecipeToSlot, clearError } =
    useCameraLink();
  const [selectedSlot, setSelectedSlot] = useState<number | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [result, setResult] = useState<WriteResult | null>(null);

  // Auto-scan once, right after connecting, so slot names are available to pick from.
  useEffect(() => {
    if (status === "connected" && !slots && !isScanning) {
      void scanSlots();
    }
  }, [status, slots, isScanning, scanSlots]);

  function slotLabel(slot: number): string {
    const found = slots?.find((s) => s.slot === slot);
    return found ? found.name || `(C${slot})` : "Unknown — scan first";
  }

  async function handleConfirmWrite() {
    if (selectedSlot === null) return;
    clearError();
    setResult(null);
    const writeResult = await writeRecipeToSlot(recipe, selectedSlot);
    setResult(writeResult);
    if (writeResult.ok) {
      void scanSlots(); // refresh cached names/values for the slot that just changed
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/70 sm:items-center sm:p-4">
      <div className="flex max-h-[90vh] w-full flex-col rounded-t-lg border border-ink-700 bg-ink-950 sm:max-w-lg sm:rounded-lg">
        <div className="flex shrink-0 items-center justify-between border-b border-ink-800 px-4 py-3.5">
          <h2 className="text-sm font-black uppercase tracking-wide text-ink-50">Apply to Camera: {recipe.name}</h2>
          <button type="button" onClick={onClose} className="text-ink-500 hover:text-ink-200">
            ✕
          </button>
        </div>

        <div className="flex-1 space-y-4 overflow-y-auto p-4">
          {!isNative && (
            <p className="rounded-md border border-ink-800 bg-ink-900 px-3 py-2.5 text-xs text-ink-400">
              Applying to camera only works in the native iOS app, not in a browser.
            </p>
          )}

          <div className="flex items-center gap-2">
            <span
              className={`h-2 w-2 rounded-full ${
                status === "connected" ? "bg-green-400" : status === "error" ? "bg-red-400" : "bg-ink-600"
              }`}
            />
            <span className="text-xs font-bold uppercase tracking-wide text-ink-400">
              {status === "connected" ? `Connected — ${deviceName}` : status}
            </span>
          </div>

          {status !== "connected" && (
            <button
              type="button"
              onClick={connect}
              disabled={!isNative || status === "connecting"}
              className="w-full rounded-md bg-gold-500 px-4 py-2.5 text-xs font-bold uppercase tracking-wide text-ink-950 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {status === "connecting" ? "Connecting…" : "Connect to Camera"}
            </button>
          )}

          {status === "connected" && !result && (
            <div className="space-y-2">
              <p className="text-xs font-bold uppercase tracking-wide text-ink-400">
                {isScanning ? "Reading camera slots…" : "Choose a slot to overwrite"}
              </p>
              <div className="grid grid-cols-1 gap-2">
                {SLOT_NUMBERS.map((slot) => (
                  <button
                    key={slot}
                    type="button"
                    onClick={() => {
                      setSelectedSlot(slot);
                      setConfirming(true);
                    }}
                    disabled={isScanning || isWriting}
                    className="flex items-center justify-between rounded-md border border-ink-700 bg-ink-900 px-3 py-2.5 text-left text-sm transition-colors hover:border-gold-700/60 disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    <span className="font-bold text-ink-50">C{slot}</span>
                    <span className="truncate pl-3 text-xs text-ink-400">{slotLabel(slot)}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {confirming && selectedSlot !== null && !result && (
            <div className="space-y-3 rounded-md border border-gold-700/50 bg-gold-500/5 p-3.5">
              <p className="text-xs text-ink-200">
                This will overwrite <span className="font-bold text-gold-300">Camera Slot C{selectedSlot}</span>{" "}
                (currently <span className="italic">{slotLabel(selectedSlot)}</span>) with{" "}
                <span className="font-bold text-gold-300">{recipe.name}</span>. This can't be undone from the app.
              </p>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => {
                    setConfirming(false);
                    setSelectedSlot(null);
                  }}
                  disabled={isWriting}
                  className="flex-1 rounded-md border border-ink-700 px-4 py-2 text-xs font-bold uppercase tracking-wide text-ink-300 disabled:opacity-40"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={handleConfirmWrite}
                  disabled={isWriting}
                  className="flex-1 rounded-md bg-gold-500 px-4 py-2 text-xs font-bold uppercase tracking-wide text-ink-950 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {isWriting ? "Writing…" : `Write to C${selectedSlot}`}
                </button>
              </div>
            </div>
          )}

          {result && (
            <div
              className={`space-y-2 rounded-md border p-3.5 ${
                result.ok ? "border-green-700/50 bg-green-500/5" : "border-red-700/50 bg-red-500/5"
              }`}
            >
              <p className={`text-sm font-bold ${result.ok ? "text-green-400" : "text-red-400"}`}>
                {result.ok ? `Written to C${selectedSlot}` : "Write failed"}
              </p>
              {result.warnings.length > 0 && (
                <ul className="space-y-1 text-[11px] text-ink-400">
                  {result.warnings.map((w, i) => (
                    <li key={i}>{w}</li>
                  ))}
                </ul>
              )}
              {result.ok && (
                <p className="text-[11px] text-ink-500">
                  Double-check on the camera's own screen (IMAGE QUALITY SETTING &gt; EDIT/SAVE CUSTOM SETTING &gt; C
                  {selectedSlot}) that it looks right.
                </p>
              )}
            </div>
          )}

          {error && (
            <pre className="whitespace-pre-wrap break-all rounded-md bg-red-500/10 px-3 py-2 text-[11px] text-red-400">
              {error}
            </pre>
          )}
        </div>

        <div className="flex shrink-0 items-center justify-end gap-2 border-t border-ink-800 px-4 py-3">
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-ink-700 px-4 py-2 text-xs font-bold uppercase tracking-wide text-ink-300 hover:text-ink-100"
          >
            {result ? "Done" : "Close"}
          </button>
        </div>
      </div>
    </div>
  );
}
