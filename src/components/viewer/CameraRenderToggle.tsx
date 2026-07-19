import { Capacitor } from "@capacitor/core";
import { useAppState } from "@/context/AppStateContext";
import { useCameraLink } from "@/context/CameraLinkContext";

/**
 * Only makes sense when there's a real .RAF to feed the camera (a plain
 * JPEG has nothing for the camera's own conversion engine to work from) and
 * on-device, where the camera-link plugin actually exists.
 */
export function CameraRenderToggle() {
  const { originalRawFile } = useAppState();
  const { status, connect, isCameraRenderMode, setCameraRenderMode } = useCameraLink();

  if (!Capacitor.isNativePlatform() || !originalRawFile) return null;

  if (status !== "connected") {
    return (
      <button
        type="button"
        onClick={() => void connect()}
        disabled={status === "connecting"}
        className="rounded-md border border-ink-700 bg-ink-900 px-3.5 py-1.5 text-xs font-bold uppercase tracking-wide text-ink-300 transition-colors hover:border-ink-500 hover:text-ink-100 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {status === "connecting" ? "Connecting…" : "Connect Camera"}
      </button>
    );
  }

  return (
    <div className="flex items-center gap-2 rounded-md border border-ink-700 bg-ink-900 px-3 py-1.5">
      <span className="text-xs font-bold uppercase tracking-wide text-ink-300">Render with Camera</span>
      <button
        type="button"
        role="switch"
        aria-checked={isCameraRenderMode}
        onClick={() => setCameraRenderMode(!isCameraRenderMode)}
        className={`relative h-5 w-9 shrink-0 rounded-full transition-colors ${
          isCameraRenderMode ? "bg-gold-500" : "bg-ink-700"
        }`}
      >
        <span
          className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-transform ${
            isCameraRenderMode ? "translate-x-4" : "translate-x-0.5"
          }`}
        />
      </button>
    </div>
  );
}
