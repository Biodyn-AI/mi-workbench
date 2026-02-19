import { useState } from "react";
import { Copy, Check, FileText } from "lucide-react";

interface Props {
  content: string;
  name: string;
  artifactType?: string;
  runId?: string | null;
  iteration?: number | null;
}

export default function ArtifactViewer({
  content,
  name,
  artifactType,
  runId,
  iteration,
}: Props) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="bg-white rounded-lg border border-surface-border overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-surface-border bg-gray-50">
        <div className="flex items-center gap-3">
          <FileText className="w-4 h-4 text-gray-400" />
          <span className="font-medium text-sm text-gray-800">{name}</span>
          {artifactType && (
            <span className="px-2 py-0.5 bg-indigo-100 text-indigo-700 text-xs rounded font-medium">
              {artifactType}
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          {runId && (
            <span className="text-xs text-gray-400">Run: {runId}</span>
          )}
          {iteration != null && (
            <span className="text-xs text-gray-400">Iter: {iteration}</span>
          )}
          <button
            onClick={handleCopy}
            className="p-1.5 rounded hover:bg-gray-200 transition-colors"
            title="Copy content"
          >
            {copied ? (
              <Check className="w-4 h-4 text-green-500" />
            ) : (
              <Copy className="w-4 h-4 text-gray-400" />
            )}
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="p-4 overflow-auto max-h-[600px]">
        <pre className="text-sm font-mono text-gray-700 whitespace-pre-wrap leading-relaxed">
          {content}
        </pre>
      </div>
    </div>
  );
}
