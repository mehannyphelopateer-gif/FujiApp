import { registerPlugin } from "@capacitor/core";

export interface PhotoSaverPlugin {
  /** `data` is a base64-encoded JPEG. Writes it directly to the Photos library. */
  saveImage(options: { data: string }): Promise<{ saved: boolean }>;
}

/**
 * Native bridge to ios/App/App/PhotoSaver/PhotoSaverPlugin.swift. iOS-only:
 * writes straight to Photos via PHPhotoLibrary instead of going through
 * navigator.share(), which doesn't reliably offer a "Save Image" action when
 * triggered from inside a Capacitor WKWebView (see that file's doc comment).
 */
export const PhotoSaver = registerPlugin<PhotoSaverPlugin>("PhotoSaver");
