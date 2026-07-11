import { createContext, useCallback, useContext, useState, type ReactNode } from "react";
import { Capacitor } from "@capacitor/core";
import { CameraLink } from "@/lib/camera/cameraLinkPlugin";
import type { Recipe } from "@/types/recipe";
import type { CameraSlotRaw } from "@/types/camera";
import { encodeRecipe } from "@/lib/camera/encodeRecipe";

type ConnectionStatus = "idle" | "connecting" | "connected" | "error";

export interface WriteResult {
  ok: boolean;
  warnings: string[];
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
      const result = await CameraLink.writeRecipeToSlot({ slot, name: recipe.name.slice(0, 20), properties });
      return result;
    } finally {
      setIsWriting(false);
    }
  }, []);

  const clearError = useCallback(() => setError(null), []);

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
