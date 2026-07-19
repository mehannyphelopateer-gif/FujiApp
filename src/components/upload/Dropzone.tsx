import { useAppState } from "@/context/AppStateContext";
import { useFileDrop } from "@/hooks/useFileDrop";

export function Dropzone() {
  const { selectedFile, setSelectedFile, setNeutralRenderFile, setOriginalRawFile } = useAppState();
  const { isDragging, error, isConverting, dropzoneProps, inputProps, openFileDialog } = useFileDrop({
    onFile: setSelectedFile,
    onNeutralFile: setNeutralRenderFile,
    onOriginalRafFile: setOriginalRawFile,
  });

  return (
    <div className="space-y-3">
      <div
        {...dropzoneProps}
        onClick={openFileDialog}
        role="button"
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") openFileDialog();
        }}
        className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-md border-2 border-dashed p-7 text-center transition-all ${
          isDragging ? "border-gold-500 bg-gold-500/10" : "border-ink-700 bg-ink-900 hover:border-ink-500"
        }`}
      >
        {isConverting ? (
          <svg className="h-8 w-8 animate-spin text-gold-400" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
          </svg>
        ) : (
          <svg
            viewBox="0 0 24 24"
            fill="none"
            strokeWidth="1.6"
            stroke="currentColor"
            className={`h-8 w-8 ${isDragging ? "text-gold-400" : "text-ink-500"}`}
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 16V4m0 0l-4 4m4-4l4 4" />
            <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v3a2 2 0 002 2h12a2 2 0 002-2v-3" />
          </svg>
        )}
        <p className="text-sm font-bold text-ink-100">
          {isConverting ? "Reading RAF preview…" : "Drop a JPEG or .RAF here"}
        </p>
        <p className="text-xs text-ink-500">or click to browse</p>
        <input {...inputProps} className="hidden" />
      </div>

      {error && <p className="rounded-md bg-red-500/10 px-3 py-2 text-xs text-red-400">{error}</p>}

      {selectedFile && (
        <div className="flex items-center justify-between rounded-md border border-ink-800 bg-ink-900 px-3 py-2.5 text-xs">
          <span className="truncate text-ink-300">{selectedFile.name}</span>
          <button
            type="button"
            onClick={openFileDialog}
            className="ml-2 shrink-0 font-bold uppercase tracking-wide text-gold-400 hover:text-gold-300"
          >
            Replace
          </button>
        </div>
      )}
    </div>
  );
}
