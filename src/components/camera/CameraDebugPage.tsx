import { useState } from "react";
import { Capacitor } from "@capacitor/core";
import { CameraLink } from "@/lib/camera/cameraLinkPlugin";
import { decodeCameraSlot } from "@/lib/camera/decodeSlot";
import type { CameraSlotRaw } from "@/types/camera";

/**
 * Phase 2 debug harness — not the polished Phase 4 UI. Purpose: let the
 * camera's real behavior (does connect work, do decoded values match what's
 * actually dialed into each slot on the camera's own screen) get verified
 * against real hardware before any further investment. Shows both the
 * decoded Recipe-shaped values AND the raw property dump side by side so a
 * mismatch is easy to diagnose.
 */
export function CameraDebugPage() {
  const [status, setStatus] = useState<"idle" | "connecting" | "connected" | "error">("idle");
  const [deviceName, setDeviceName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [slots, setSlots] = useState<CameraSlotRaw[] | null>(null);
  const [isScanning, setIsScanning] = useState(false);

  const isNative = Capacitor.isNativePlatform();

  async function handleConnect() {
    setStatus("connecting");
    setError(null);
    try {
      const result = await CameraLink.connect();
      setDeviceName(result.deviceName);
      setStatus("connected");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to connect.");
      setStatus("error");
    }
  }

  async function handleDisconnect() {
    await CameraLink.disconnect().catch(() => {});
    setStatus("idle");
    setDeviceName(null);
    setSlots(null);
  }

  async function handleScan() {
    setIsScanning(true);
    setError(null);
    try {
      const result = await CameraLink.scanPresets();
      setSlots(result.slots);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to scan presets.");
    } finally {
      setIsScanning(false);
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

        <div className="flex gap-2">
          <button
            type="button"
            onClick={handleConnect}
            disabled={!isNative || status === "connecting" || status === "connected"}
            className="rounded-md bg-gold-500 px-4 py-2 text-xs font-bold uppercase tracking-wide text-ink-950 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Connect
          </button>
          <button
            type="button"
            onClick={handleDisconnect}
            disabled={status !== "connected"}
            className="rounded-md border border-ink-700 px-4 py-2 text-xs font-bold uppercase tracking-wide text-ink-300 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Disconnect
          </button>
          <button
            type="button"
            onClick={handleScan}
            disabled={status !== "connected" || isScanning}
            className="rounded-md border border-gold-600 bg-gold-500/10 px-4 py-2 text-xs font-bold uppercase tracking-wide text-gold-400 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {isScanning ? "Scanning…" : "Scan 7 Slots"}
          </button>
        </div>

        {error && <p className="rounded-md bg-red-500/10 px-3 py-2 text-xs text-red-400">{error}</p>}

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
