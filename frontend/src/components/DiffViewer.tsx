import { useState, useMemo } from "react";
import { diffLines, type Change } from "diff";

interface Props {
  oldContent: string;
  newContent: string;
  oldLabel?: string;
  newLabel?: string;
}

export default function DiffViewer({
  oldContent,
  newContent,
  oldLabel = "Previous",
  newLabel = "Current",
}: Props) {
  const [viewMode, setViewMode] = useState<"unified" | "split">("unified");
  const changes = useMemo(
    () => diffLines(oldContent, newContent),
    [oldContent, newContent],
  );

  return (
    <div className="bg-white rounded-lg border border-surface-border overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-surface-border bg-gray-50">
        <div className="flex items-center gap-4 text-sm text-gray-600">
          <span>{oldLabel}</span>
          <span className="text-gray-300">vs</span>
          <span>{newLabel}</span>
        </div>
        <div className="flex rounded-md border border-gray-300 overflow-hidden">
          <button
            onClick={() => setViewMode("unified")}
            className={`px-3 py-1 text-xs font-medium transition-colors ${
              viewMode === "unified"
                ? "bg-accent text-white"
                : "bg-white text-gray-600 hover:bg-gray-50"
            }`}
          >
            Unified
          </button>
          <button
            onClick={() => setViewMode("split")}
            className={`px-3 py-1 text-xs font-medium transition-colors ${
              viewMode === "split"
                ? "bg-accent text-white"
                : "bg-white text-gray-600 hover:bg-gray-50"
            }`}
          >
            Split
          </button>
        </div>
      </div>

      {/* Content */}
      {viewMode === "unified" ? (
        <UnifiedView changes={changes} />
      ) : (
        <SplitView oldContent={oldContent} newContent={newContent} changes={changes} />
      )}
    </div>
  );
}

function UnifiedView({ changes }: { changes: Change[] }) {
  let lineNum = 0;
  return (
    <div className="overflow-auto max-h-[500px]">
      {changes.map((change, i) => {
        const lines = change.value.split("\n").filter((l, idx, arr) =>
          idx < arr.length - 1 || l !== "",
        );
        return lines.map((line, j) => {
          if (!change.added) lineNum++;
          let cls = "diff-unchanged";
          let prefix = " ";
          if (change.added) {
            cls = "diff-added";
            prefix = "+";
          } else if (change.removed) {
            cls = "diff-removed";
            prefix = "-";
          }
          return (
            <div
              key={`${i}-${j}`}
              className={`${cls} px-4 py-0.5 font-mono text-sm`}
            >
              <span className="text-gray-400 mr-3 select-none inline-block w-8 text-right">
                {!change.added ? lineNum : ""}
              </span>
              <span className="text-gray-400 mr-2 select-none">{prefix}</span>
              {line}
            </div>
          );
        });
      })}
    </div>
  );
}

function SplitView({
  oldContent,
  newContent,
  changes,
}: {
  oldContent: string;
  newContent: string;
  changes: Change[];
}) {
  // Build left/right lines
  const left: { text: string; type: "removed" | "unchanged" | "empty" }[] = [];
  const right: { text: string; type: "added" | "unchanged" | "empty" }[] = [];

  for (const change of changes) {
    const lines = change.value.split("\n").filter((l, idx, arr) =>
      idx < arr.length - 1 || l !== "",
    );
    if (change.added) {
      for (const line of lines) {
        left.push({ text: "", type: "empty" });
        right.push({ text: line, type: "added" });
      }
    } else if (change.removed) {
      for (const line of lines) {
        left.push({ text: line, type: "removed" });
        right.push({ text: "", type: "empty" });
      }
    } else {
      for (const line of lines) {
        left.push({ text: line, type: "unchanged" });
        right.push({ text: line, type: "unchanged" });
      }
    }
  }

  // Suppress unused variable warnings
  void oldContent;
  void newContent;

  const colorMap = {
    removed: "bg-red-50",
    added: "bg-green-50",
    unchanged: "bg-white",
    empty: "bg-gray-50",
  };

  return (
    <div className="overflow-auto max-h-[500px] grid grid-cols-2 divide-x divide-surface-border">
      <div>
        {left.map((l, i) => (
          <div
            key={i}
            className={`${colorMap[l.type]} px-3 py-0.5 font-mono text-sm min-h-[1.5rem]`}
          >
            {l.text}
          </div>
        ))}
      </div>
      <div>
        {right.map((l, i) => (
          <div
            key={i}
            className={`${colorMap[l.type]} px-3 py-0.5 font-mono text-sm min-h-[1.5rem]`}
          >
            {l.text}
          </div>
        ))}
      </div>
    </div>
  );
}
