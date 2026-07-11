import { registerPlugin } from "@capacitor/core";
import type { CameraConnectionStatus, CameraSlotRaw } from "@/types/camera";
import type { EncodedProperty } from "@/lib/camera/encodeRecipe";

export interface CameraLinkPlugin {
  connect(): Promise<CameraConnectionStatus>;
  disconnect(): Promise<void>;
  getStatus(): Promise<CameraConnectionStatus>;
  /** Diagnostic-only plain GetDeviceInfo, isolated from Fuji-specific properties. */
  getDeviceInfo(): Promise<{ model: string; raw: string }>;
  /** Reads all 7 custom slots (C1-C7) in order. Read-only — Phase 2 scope. */
  scanPresets(): Promise<{ slots: CameraSlotRaw[] }>;
  /** Writes a recipe to a camera slot (1-7). See encodeRecipe.ts for the ordered property list. */
  writeRecipeToSlot(options: { slot: number; name: string; properties: EncodedProperty[] }): Promise<{
    ok: boolean;
    warnings: string[];
  }>;
}

/**
 * Native bridge to ios/App/App/CameraLink/CameraLinkPlugin.swift. Only
 * resolves on iOS inside the Capacitor shell — there is no web/PWA
 * implementation, since this needs ImageCaptureCore (see CLAUDE.md's iOS
 * section and the plan this was built from: no WebUSB equivalent exists in
 * iOS Safari, so this feature is native-app-only by necessity).
 */
export const CameraLink = registerPlugin<CameraLinkPlugin>("CameraLink");
