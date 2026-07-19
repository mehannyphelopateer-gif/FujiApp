import { registerPlugin } from "@capacitor/core";

export interface PhotoSaverPlugin {
  /** `data` is a base64-encoded JPEG. Writes it directly to the Photos library. */
  saveImage(options: { data: string }): Promise<{ saved: boolean; colorProfile?: string }>;
  /**
   * Presents the system "Save to..." document picker (Files/iCloud Drive/
   * On My iPhone/etc.) rather than the share sheet — the share sheet's
   * activity list is OS-controlled and doesn't reliably include a Files
   * destination (confirmed missing entirely when running via "Designed for
   * iPad" on a Mac). `saved: false` means the user cancelled the picker,
   * not that anything failed.
   */
  saveToFiles(options: { data: string; filename: string }): Promise<{ saved: boolean }>;
}

/**
 * Native bridge to ios/App/App/PhotoSaver/PhotoSaverPlugin.swift. iOS-only:
 * writes straight to Photos via PHPhotoLibrary instead of going through
 * navigator.share(), which doesn't reliably offer a "Save Image" action when
 * triggered from inside a Capacitor WKWebView (see that file's doc comment).
 */
export const PhotoSaver = registerPlugin<PhotoSaverPlugin>("PhotoSaver");
