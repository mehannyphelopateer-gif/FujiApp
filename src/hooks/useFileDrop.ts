import { useCallback, useRef, useState, type ChangeEvent, type DragEvent } from "react";

interface UseFileDropOptions {
  onFile: (file: File) => void;
  accept?: string[];
}

interface UseFileDropResult {
  isDragging: boolean;
  error: string | null;
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
  if (!accept.includes(file.type)) {
    return `Unsupported file type "${file.type || "unknown"}". Please upload a JPEG image.`;
  }
  return null;
}

export function useFileDrop({ onFile, accept = ["image/jpeg"] }: UseFileDropOptions): UseFileDropResult {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const processFile = useCallback(
    (file: File | undefined) => {
      if (!file) return;
      const validationError = validateFile(file, accept);
      if (validationError) {
        setError(validationError);
        return;
      }
      setError(null);
      onFile(file);
    },
    [accept, onFile],
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
      processFile(event.dataTransfer.files[0]);
    },
    [processFile],
  );

  const onChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      processFile(event.target.files?.[0]);
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
    dropzoneProps: { onDragOver, onDragLeave, onDrop },
    inputProps: { ref: inputRef, type: "file", accept: accept.join(","), onChange },
    openFileDialog,
  };
}
