import { useCallback, useRef, useState, type ChangeEvent, type DragEvent } from "react";
import { decodeNeutralRafWithDiagnostics, extractRafPreviewJpeg, isRafFile } from "@/lib/raw/rawService";

interface UseFileDropOptions {
  onFile: (file: File | null) => void;
  /**
   * Called after onFile for a .RAF upload with a true RAW-demosaiced (no
   * baked-in film simulation/grain) File. The browser uses LibRaw WASM and
   * native iOS uses CIRAWFilter; null means that decoder could not handle
   * the specific file. Not called at all for a plain JPEG upload.
   */
  onNeutralFile?: (file: File | null) => void;
  /**
   * Called with the original, untouched .RAF File (its real bytes, not the
   * extracted preview or the decoded-neutral derivative) — this is what
   * "Render with Camera" uploads to the camera for real conversion. Called
   * with null for a plain JPEG upload, mirroring onNeutralFile.
   */
  onOriginalRafFile?: (file: File | null) => void;
  accept?: string[];
}

interface UseFileDropResult {
  isDragging: boolean;
  error: string | null;
  isConverting: boolean;
  dropzoneProps: {
    onDragOver: (event: DragEvent<HTMLDivElement>) => void;
    onDragLeave: (event: DragEvent<HTMLDivElement>) => void;
    onDrop: (event: DragEvent<HTMLDivElement>) => void;
  };
  inputProps: {
    ref: React.RefObject<HTMLInputElement | null>;
    type: "file";
    accept: string;
    onChange: (event: ChangeEvent<HTMLInputElement>) => void;
  };
  openFileDialog: () => void;
}

function validateFile(file: File, accept: string[]): string | null {
  if (isRafFile(file)) return null; // validated separately after conversion
  if (!accept.includes(file.type)) {
    return `Unsupported file type "${file.type || "unknown"}". Please upload a JPEG or a Fujifilm .RAF file.`;
  }
  return null;
}

export function useFileDrop({
  onFile,
  onNeutralFile,
  onOriginalRafFile,
  accept = ["image/jpeg"],
}: UseFileDropOptions): UseFileDropResult {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isConverting, setIsConverting] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const processFile = useCallback(
    async (file: File | undefined) => {
      if (!file) return;
      const validationError = validateFile(file, accept);
      if (validationError) {
        setError(validationError);
        return;
      }
      setError(null);

      if (isRafFile(file)) {
        // Clear a previous image before starting the asynchronous decode. A
        // replacement RAF must never leave the old render visible/exportable
        // while its own pixels are still being prepared.
        onFile(null);
        onNeutralFile?.(null);
        onOriginalRafFile?.(null);
        setIsConverting(true);
        try {
          const jpegName = file.name.replace(/\.raf$/i, ".jpg");
          // Do not expose the RAF's baked JPEG while its sensor data is still
          // decoding. Doing so lets the user save an old-recipe preview that
          // looks finished even though the real offline RAW path has not
          // completed yet. A RAF preview is only considered ready once the
          // neutral decoder succeeds, or it explicitly reports why it cannot.
          const neutral = await decodeNeutralRafWithDiagnostics(file);
          if (neutral.blob) {
            const neutralFile = new File([neutral.blob], jpegName, { type: "image/jpeg" });
            onFile(neutralFile);
            onOriginalRafFile?.(file);
            onNeutralFile?.(neutralFile);
          } else {
            const previewBlob = await extractRafPreviewJpeg(file);
            onFile(new File([previewBlob], jpegName, { type: "image/jpeg" }));
            onOriginalRafFile?.(file);
            onNeutralFile?.(null);
            setError(`Local RAW decode failed; Preview is using the embedded JPEG instead. ${neutral.error}`);
          }
        } catch (err) {
          setError(err instanceof Error ? err.message : "Failed to read this .RAF file.");
        } finally {
          setIsConverting(false);
        }
        return;
      }

      onNeutralFile?.(null);
      onOriginalRafFile?.(null);
      onFile(file);
    },
    [accept, onFile, onNeutralFile, onOriginalRafFile],
  );

  const onDragOver = useCallback((event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(true);
  }, []);

  const onDragLeave = useCallback((event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
  }, []);

  const onDrop = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      setIsDragging(false);
      void processFile(event.dataTransfer.files[0]);
    },
    [processFile],
  );

  const onChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      void processFile(event.target.files?.[0]);
      event.target.value = "";
    },
    [processFile],
  );

  const openFileDialog = useCallback(() => {
    inputRef.current?.click();
  }, []);

  return {
    isDragging,
    error,
    isConverting,
    dropzoneProps: { onDragOver, onDragLeave, onDrop },
    inputProps: { ref: inputRef, type: "file", accept: accept.join(",") + ",.raf", onChange },
    openFileDialog,
  };
}
