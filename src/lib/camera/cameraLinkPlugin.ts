import { registerPlugin } from "@capacitor/core";
import type { CameraConnectionStatus, CameraSlotRaw } from "@/types/camera";

export interface CameraLinkPlugin {
  connect(): Promise<CameraConnectionStatus>;
  disconnect(): Promise<void>;
  getStatus(): Promise<CameraConnectionStatus>;
  /** Reads all 7 custom slots (C1-C7) in order. Read-only — Phase 2 scope. */
  scanPresets(): Promise<{ slots: CameraSlotRaw[] }>;
}

/**
 * Native bridge to ios/App/App/CameraLink/CameraLinkPlugin.swift. Only
 * resolves on iOS inside the Capacitor shell — there is no web/PWA
 * implementation, since this needs ImageCaptureCore (see CLAUDE.md's iOS
 * section and the plan this was built from: no WebUSB equivalent exists in
 * iOS Safari, so this feature is native-app-only by necessity).
 */
export const CameraLink = registerPlugin<CameraLinkPlugin>("CameraLink");
