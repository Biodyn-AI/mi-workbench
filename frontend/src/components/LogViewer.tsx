import { useEffect, useRef } from "react";
import { Terminal } from "lucide-react";

interface LogEntry {
  timestamp: string;
  role: string;
  message: string;
}

interface Props {
  logs: LogEntry[];
  autoScroll?: boolean;
  maxHeight?: string;
}

const roleClasses: Record<string, string> = {
  executor: "log-line-executor",
  reviewer: "log-line-reviewer",
  adversarial_reviewer: "log-line-adversarial",
  adversarial: "log-line-adversarial",
  system: "log-line-system",
};

function formatTime(ts: string): string {
  try {
    const d = new Date(ts);
    return d.toLocaleTimeString("en-US", { hour12: false });
  } catch {
    return ts;
  }
}

export default function LogViewer({
  logs,
  autoScroll = true,
  maxHeight = "400px",
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (autoScroll && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [logs, autoScroll]);

  return (
    <div className="bg-gray-900 rounded-lg overflow-hidden border border-gray-700">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-2 bg-gray-800 border-b border-gray-700">
        <Terminal className="w-4 h-4 text-gray-400" />
        <span className="text-sm text-gray-300 font-medium">Logs</span>
        <span className="text-xs text-gray-500 ml-auto">
          {logs.length} entries
        </span>
      </div>

      {/* Log content */}
      <div
        ref={containerRef}
        className="overflow-auto p-4 space-y-0.5"
        style={{ maxHeight }}
      >
        {logs.length === 0 ? (
          <div className="text-gray-500 text-sm italic">No log entries yet.</div>
        ) : (
          logs.map((entry, i) => {
            const roleClass = roleClasses[entry.role] ?? "text-gray-400";
            return (
              <div key={i} className="log-line flex">
                <span className="text-gray-600 mr-3 flex-shrink-0 select-none">
                  {formatTime(entry.timestamp)}
                </span>
                <span
                  className={`font-semibold mr-2 flex-shrink-0 w-24 text-right ${roleClass}`}
                >
                  [{entry.role}]
                </span>
                <span className="text-gray-300 break-all">{entry.message}</span>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
