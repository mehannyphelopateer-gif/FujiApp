import { createContext, useCallback, useContext, useRef, useState, type ReactNode } from "react";
import { Capacitor } from "@capacitor/core";
import { CameraLink } from "@/lib/camera/cameraLinkPlugin";
import type { Recipe } from "@/types/recipe";
import type { CameraSlotRaw } from "@/types/camera";
import { encodeRecipe, sanitizeCameraName } from "@/lib/camera/encodeRecipe";
import { patchRawProfile } from "@/lib/camera/patchRawProfile";
import { base64ToBlob, base64ToUint8Array, fileToBase64, uint8ArrayToBase64 } from "@/lib/camera/base64";

type ConnectionStatus = "idle" | "connecting" | "connected" | "error";

export interface WriteResult {
  ok: boolean;
  warnings: string[];
}

export interface ConversionResult {
  ok: boolean;
  imageUrl?: string;
  error?: string;
}

// ~1s between polls, capped at 45 tries — matches the plan's Phase 3 Go/No-Go
// window (real hardware conversions have taken 1-2s in testing).
const POLL_INTERVAL_MS = 1000;
const POLL_MAX_ATTEMPTS = 45;
// A freshly-detected object handle has occasionally not been immediately
// downloadable on real hardware (InvalidObjectHandle) — one retry after a
// short wait rather than surfacing that as a hard failure straight away.
const DOWNLOAD_RETRY_DELAY_MS = 3000;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

interface CameraLinkState {
  isNative: boolean;
  status: ConnectionStatus;
  deviceName: string | null;
  error: string | null;
  slots: CameraSlotRaw[] | null;
  isScanning: boolean;
  isWriting: boolean;
  connect: () => Promise<void>;
  disconnect: () => Promise<void>;
  scanSlots: () => Promise<void>;
  getDeviceInfo: () => Promise<{ model: string; raw: string }>;
  writeRecipeToSlot: (recipe: Recipe, slot: number) => Promise<WriteResult>;
  clearError: () => void;

  // --- RAW conversion ("Render with Camera") ---
  /** Whether the Preview tab should be driving real camera conversions instead of the WebGL preview. */
  isCameraRenderMode: boolean;
  /** Turning this off discards any in-flight conversion rather than letting it land after the fact. */
  setCameraRenderMode: (on: boolean) => void;
  isConverting: boolean;
  /** Object URL for the most recent camera-converted JPEG. Caller does not need to revoke it — this context does, on the next conversion or unmount-equivalent turn-off. */
  convertedImageUrl: string | null;
  conversionError: string | null;
  /**
   * Uploads `rafFile` to the camera fresh (every call — real-hardware testing
   * showed reconverting an already-uploaded RAF without a fresh upload is
   * unreliable), patches the profile for `recipe`, triggers conversion, and
   * resolves once `convertedImageUrl` is updated. Superseded by a later call
   * (or `setCameraRenderMode(false)`) rather than racing it — check
   * `isConverting`/`conversionError` for outcome instead of relying on the
   * resolved value for live-preview UI. The resolved `ConversionResult` is
   * for callers that need a definite per-call outcome for their own bookkeeping
   * (e.g. a sequential QA sweep across many recipes) rather than the shared
   * single-slot `convertedImageUrl`/`conversionError` state.
   */
  convertWithRecipe: (recipe: Recipe, rafFile: File) => Promise<ConversionResult>;
}

const CameraLinkContext = createContext<CameraLinkState | null>(null);

/**
 * Lives above the tab router in App.tsx (not inside CameraDebugPage) so the
 * connection survives switching tabs — the native ImageCaptureCore session
 * itself is owned by the Swift plugin singleton and never actually drops on
 * navigation, but the JS-side status/slots state was getting reset to "idle"
 * every time the Camera tab's component unmounted, which is what looked like
 * a dropped connection from the UI's perspective.
 */
export function CameraLinkProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<ConnectionStatus>("idle");
  const [deviceName, setDeviceName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [slots, setSlots] = useState<CameraSlotRaw[] | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [isWriting, setIsWriting] = useState(false);

  const isNative = Capacitor.isNativePlatform();

  const connect = useCallback(async () => {
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
  }, []);

  const disconnect = useCallback(async () => {
    await CameraLink.disconnect().catch(() => {});
    setStatus("idle");
    setDeviceName(null);
    setSlots(null);
  }, []);

  const scanSlots = useCallback(async () => {
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
  }, []);

  const getDeviceInfo = useCallback(async () => {
    return CameraLink.getDeviceInfo();
  }, []);

  const writeRecipeToSlot = useCallback(async (recipe: Recipe, slot: number): Promise<WriteResult> => {
    setIsWriting(true);
    setError(null);
    try {
      const properties = encodeRecipe(recipe);
      const name = sanitizeCameraName(recipe.name).slice(0, 20);
      const result = await CameraLink.writeRecipeToSlot({ slot, name, properties });
      return result;
    } finally {
      setIsWriting(false);
    }
  }, []);

  const clearError = useCallback(() => setError(null), []);

  const [isCameraRenderMode, setIsCameraRenderMode] = useState(false);
  const [isConverting, setIsConverting] = useState(false);
  const [convertedImageUrl, setConvertedImageUrl] = useState<string | null>(null);
  const [conversionError, setConversionError] = useState<string | null>(null);
  // Bumped on every new convertWithRecipe call (and on turning render mode
  // off) so an in-flight conversion superseded by a newer one — or by the
  // user leaving camera-render mode — discards its result instead of
  // landing after the fact.
  const conversionGenerationRef = useRef(0);

  const setCameraRenderMode = useCallback((on: boolean) => {
    setIsCameraRenderMode(on);
    if (!on) {
      conversionGenerationRef.current += 1;
      setConversionError(null);
    }
  }, []);

  const convertWithRecipe = useCallback(async (recipe: Recipe, rafFile: File): Promise<ConversionResult> => {
    const myGeneration = ++conversionGenerationRef.current;
    const superseded: ConversionResult = { ok: false, error: "superseded" };
    setIsConverting(true);
    setConversionError(null);
    try {
      const rafBase64 = await fileToBase64(rafFile);
      await CameraLink.uploadRaf({ data: rafBase64 });
      if (myGeneration !== conversionGenerationRef.current) return superseded;

      const { profile } = await CameraLink.getRawProfile();
      const patched = patchRawProfile(base64ToUint8Array(profile), recipe, { forceWhiteBalance: true });
      await CameraLink.setRawProfile({ profile: uint8ArrayToBase64(patched) });
      if (myGeneration !== conversionGenerationRef.current) return superseded;

      const { handles: baseline } = await CameraLink.listObjectHandles();
      await CameraLink.startRawConversion();

      let newHandle: number | null = null;
      for (let attempt = 0; attempt < POLL_MAX_ATTEMPTS; attempt++) {
        if (myGeneration !== conversionGenerationRef.current) return superseded;
        const { handles } = await CameraLink.listObjectHandles();
        const fresh = handles.filter((h) => !baseline.includes(h));
        if (fresh.length > 0) {
          newHandle = fresh[0];
          break;
        }
        await sleep(POLL_INTERVAL_MS);
      }
      if (newHandle === null) {
        throw new Error("Timed out waiting for the camera to finish converting.");
      }
      if (myGeneration !== conversionGenerationRef.current) return superseded;

      let data: string;
      try {
        ({ data } = await CameraLink.downloadObject({ handle: newHandle }));
      } catch {
        await sleep(DOWNLOAD_RETRY_DELAY_MS);
        if (myGeneration !== conversionGenerationRef.current) return superseded;
        ({ data } = await CameraLink.downloadObject({ handle: newHandle }));
      }
      if (myGeneration !== conversionGenerationRef.current) return superseded;

      const url = URL.createObjectURL(base64ToBlob(data, "image/jpeg"));
      setConvertedImageUrl((previous) => {
        if (previous) URL.revokeObjectURL(previous);
        return url;
      });

      // Best-effort cleanup — a failure here shouldn't undo an otherwise-successful conversion.
      CameraLink.deleteObject({ handle: newHandle }).catch(() => {});
      return { ok: true, imageUrl: url };
    } catch (err) {
      const message = err instanceof Error ? err.message : "Conversion failed.";
      if (myGeneration === conversionGenerationRef.current) {
        setConversionError(message);
      }
      return { ok: false, error: message };
    } finally {
      if (myGeneration === conversionGenerationRef.current) {
        setIsConverting(false);
      }
    }
  }, []);

  const value: CameraLinkState = {
    isNative,
    status,
    deviceName,
    error,
    slots,
    isScanning,
    isWriting,
    connect,
    disconnect,
    scanSlots,
    getDeviceInfo,
    writeRecipeToSlot,
    clearError,
    isCameraRenderMode,
    setCameraRenderMode,
    isConverting,
    convertedImageUrl,
    conversionError,
    convertWithRecipe,
  };

  return <CameraLinkContext.Provider value={value}>{children}</CameraLinkContext.Provider>;
}

export function useCameraLink(): CameraLinkState {
  const context = useContext(CameraLinkContext);
  if (!context) {
    throw new Error("useCameraLink must be used within a CameraLinkProvider");
  }
  return context;
}
