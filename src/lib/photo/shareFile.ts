/**
 * Hands a file to the OS share sheet (Save to Files, AirDrop, Messages,
 * etc.) via the Web Share API — distinct from PhotoSaverPlugin's direct
 * PHPhotoLibrary write, which only ever targets the Photos library. Falls
 * back to a plain anchor-tag download if file sharing isn't available
 * (e.g. desktop browsers), same as the web/PWA path already used elsewhere.
 */
export async function shareOrDownloadFile(blob: Blob, filename: string): Promise<void> {
  const file = new File([blob], filename, { type: blob.type || "image/jpeg" });
  const canShareFiles = typeof navigator.canShare === "function" && navigator.canShare({ files: [file] });

  if (canShareFiles) {
    try {
      await navigator.share({ files: [file] });
      return;
    } catch (err) {
      // AbortError means the user dismissed the share sheet — leave it at
      // that rather than dropping into a surprise download. Any other
      // failure falls through to the download below.
      if (err instanceof Error && err.name === "AbortError") return;
    }
  }

  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
