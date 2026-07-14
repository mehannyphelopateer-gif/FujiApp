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

  // --- RAW conversion (drives the camera's own conversion engine so any
  // app recipe, not just the 7 saved custom slots, can be applied to a RAW
  // file with real camera color science). See src/lib/camera/patchRawProfile.ts
  // for what the profile bytes mean — these methods just move bytes/numbers.

  /** Uploads a .RAF's full bytes to the camera. Call once per file. */
  uploadRaf(options: { data: string }): Promise<{ ok: boolean }>;
  /** Reads the camera's current RAW-conversion profile (0xD185) — only valid after uploadRaf. */
  getRawProfile(): Promise<{ profile: string; length: number }>;
  /** Writes a patched profile back (0xD185). */
  setRawProfile(options: { profile: string }): Promise<{ ok: boolean }>;
  /** Triggers conversion using whatever profile is currently set (0xD183). */
  startRawConversion(): Promise<{ ok: boolean }>;
  /** Lists every object handle on the camera/card — diff against a baseline to spot the new converted JPEG. */
  listObjectHandles(): Promise<{ handles: number[] }>;
  /** Downloads an object's full bytes (the converted JPEG) by handle. */
  downloadObject(options: { handle: number }): Promise<{ data: string }>;
  /** Deletes a temporary object after downloading it. Best-effort — treat failure as non-fatal. */
  deleteObject(options: { handle: number }): Promise<{ ok: boolean }>;
}

/**
 * Native bridge to ios/App/App/CameraLink/CameraLinkPlugin.swift. Only
 * resolves on iOS inside the Capacitor shell — there is no web/PWA
 * implementation, since this needs ImageCaptureCore (see CLAUDE.md's iOS
 * section and the plan this was built from: no WebUSB equivalent exists in
 * iOS Safari, so this feature is native-app-only by necessity).
 */
export const CameraLink = registerPlugin<CameraLinkPlugin>("CameraLink");
