import { useCallback, useRef, useState, type ChangeEvent, type DragEvent } from "react";
import { decodeNeutralRaf, extractRafPreviewJpeg, isRafFile } from "@/lib/raw/rawService";

interface UseFileDropOptions {
  onFile: (file: File) => void;
  /**
   * Called after onFile for a .RAF upload, with a true RAW-demosaiced (no
   * baked-in film simulation/grain) File on native iOS, or null on web /
   * on decode failure — see decodeNeutralRaf's doc comment. Not called at
   * all for a plain JPEG upload.
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
        setIsConverting(true);
        try {
          const previewBlob = await extractRafPreviewJpeg(file);
          const jpegName = file.name.replace(/\.raf$/i, ".jpg");
          onFile(new File([previewBlob], jpegName, { type: "image/jpeg" }));
          onOriginalRafFile?.(file);

          // Best-effort, iOS-only true RAW decode — runs after the preview
          // JPEG is already showing, so the photo appears immediately and
          // this just upgrades the base image underneath it once ready.
          const neutralBlob = await decodeNeutralRaf(file);
          onNeutralFile?.(neutralBlob ? new File([neutralBlob], jpegName, { type: "image/jpeg" }) : null);
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
