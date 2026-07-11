import { useState } from "react";
import { useCameraLink } from "@/context/CameraLinkContext";
import { decodeCameraSlot } from "@/lib/camera/decodeSlot";

/**
 * Phase 2 debug harness — not the polished Phase 4 UI. Purpose: let the
 * camera's real behavior (does connect work, do decoded values match what's
 * actually dialed into each slot on the camera's own screen) get verified
 * against real hardware before any further investment. Shows both the
 * decoded Recipe-shaped values AND the raw property dump side by side so a
 * mismatch is easy to diagnose. Connection state itself lives in
 * CameraLinkContext so it survives switching away from this tab.
 */
export function CameraDebugPage() {
  const { isNative, status, deviceName, error, slots, isScanning, connect, disconnect, scanSlots, getDeviceInfo, clearError } =
    useCameraLink();
  const [deviceInfo, setDeviceInfo] = useState<string | null>(null);
  const [isFetchingInfo, setIsFetchingInfo] = useState(false);

  async function handleGetDeviceInfo() {
    setIsFetchingInfo(true);
    clearError();
    setDeviceInfo(null);
    try {
      const result = await getDeviceInfo();
      setDeviceInfo(`${result.model}\n${result.raw}`);
    } catch (err) {
      setDeviceInfo(null);
      // getDeviceInfo throws directly (not routed through context's error state) — show inline instead.
      setDeviceInfo(err instanceof Error ? `Error: ${err.message}` : "Failed to get device info.");
    } finally {
      setIsFetchingInfo(false);
    }
  }

  return (
    <div className="h-full w-full overflow-y-auto bg-ink-950 p-4 text-ink-50">
      <h1 className="mb-1 text-[11px] font-bold uppercase tracking-[0.15em] text-gold-400">Camera Link — Debug</h1>
      <p className="mb-6 text-2xl font-black uppercase tracking-tight text-ink-50">
        Phase 2<span className="text-gold-400">.</span>
      </p>

      {!isNative && (
        <p className="rounded-md border border-ink-800 bg-ink-900 px-3 py-2.5 text-xs text-ink-400">
          Camera Link only works in the native iOS app (needs ImageCaptureCore, not available in a browser). You're
          viewing this in a regular browser/PWA context.
        </p>
      )}

      <div className="space-y-3">
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

        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={connect}
            disabled={!isNative || status === "connecting" || status === "connected"}
            className="rounded-md bg-gold-500 px-4 py-2 text-xs font-bold uppercase tracking-wide text-ink-950 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Connect
          </button>
          <button
            type="button"
            onClick={disconnect}
            disabled={status !== "connected"}
            className="rounded-md border border-ink-700 px-4 py-2 text-xs font-bold uppercase tracking-wide text-ink-300 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Disconnect
          </button>
          <button
            type="button"
            onClick={handleGetDeviceInfo}
            disabled={status !== "connected" || isFetchingInfo}
            className="rounded-md border border-ink-700 px-4 py-2 text-xs font-bold uppercase tracking-wide text-ink-300 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {isFetchingInfo ? "Checking…" : "Get Device Info"}
          </button>
          <button
            type="button"
            onClick={scanSlots}
            disabled={status !== "connected" || isScanning}
            className="rounded-md border border-gold-600 bg-gold-500/10 px-4 py-2 text-xs font-bold uppercase tracking-wide text-gold-400 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {isScanning ? "Scanning…" : "Scan 7 Slots"}
          </button>
        </div>

        {deviceInfo && (
          <pre className="whitespace-pre-wrap break-all rounded-md border border-ink-800 bg-ink-900 px-3 py-2.5 text-[11px] text-ink-300">
            {deviceInfo}
          </pre>
        )}

        {error && (
          <pre className="whitespace-pre-wrap break-all rounded-md bg-red-500/10 px-3 py-2 text-[11px] text-red-400">
            {error}
          </pre>
        )}

        {slots && (
          <div className="space-y-3 pt-2">
            {slots.map((slot) => {
              const decoded = decodeCameraSlot(slot);
              return (
                <details key={slot.slot} className="rounded-md border border-ink-800 bg-ink-900 p-3" open>
                  <summary className="cursor-pointer text-sm font-bold text-ink-50">
                    C{slot.slot} — {slot.name}
                  </summary>
                  <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 font-mono text-[11px] text-ink-300">
                    <span>Film Sim</span>
                    <span className="text-gold-300">{decoded.baseFilmSimulation}</span>
                    <span>Dynamic Range</span>
                    <span className="text-gold-300">{decoded.dynamicRange}</span>
                    <span>White Balance</span>
                    <span className="text-gold-300">
                      {decoded.whiteBalance.mode}
                      {decoded.whiteBalance.kelvin ? ` (${decoded.whiteBalance.kelvin}K)` : ""}
                    </span>
                    <span>WB Shift R/B</span>
                    <span className="text-gold-300">
                      {decoded.whiteBalance.shift.red} / {decoded.whiteBalance.shift.blue}
                    </span>
                    <span>Highlight / Shadow</span>
                    <span className="text-gold-300">
                      {decoded.highlightTone} / {decoded.shadowTone}
                    </span>
                    <span>Color / Sharpness</span>
                    <span className="text-gold-300">
                      {decoded.color} / {decoded.sharpness}
                    </span>
                    <span>Color Chrome / FX Blue</span>
                    <span className="text-gold-300">
                      {decoded.colorChromeEffect} / {decoded.colorChromeFxBlue}
                    </span>
                    <span>Grain</span>
                    <span className="text-gold-300">
                      {decoded.grainEffect}
                      {decoded.grainSize ? ` (${decoded.grainSize})` : ""}
                    </span>
                    <span>Clarity</span>
                    <span className="text-gold-300">{decoded.clarity}</span>
                  </div>
                  <details className="mt-3">
                    <summary className="cursor-pointer text-[10px] uppercase tracking-wide text-ink-500">
                      Raw properties
                    </summary>
                    <pre className="mt-2 overflow-x-auto text-[10px] text-ink-500">
                      {JSON.stringify(slot.properties, null, 2)}
                    </pre>
                  </details>
                </details>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
