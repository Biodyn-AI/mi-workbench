import type { RunStatus as RunStatusType } from "../api/client";

const statusConfig: Record<
  RunStatusType,
  { bg: string; text: string; dot?: string }
> = {
  pending: { bg: "bg-gray-100", text: "text-gray-600" },
  running: {
    bg: "bg-blue-100",
    text: "text-blue-700",
    dot: "bg-blue-500 animate-pulse-dot",
  },
  paused: { bg: "bg-yellow-100", text: "text-yellow-700" },
  stopped: { bg: "bg-red-100", text: "text-red-600" },
  completed: { bg: "bg-green-100", text: "text-green-700" },
  failed: { bg: "bg-red-100", text: "text-red-700" },
};

interface Props {
  status: RunStatusType;
  className?: string;
}

export default function RunStatusBadge({ status, className = "" }: Props) {
  const config = statusConfig[status] ?? statusConfig.pending;

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${config.bg} ${config.text} ${className}`}
    >
      {config.dot ? (
        <span className={`w-1.5 h-1.5 rounded-full ${config.dot}`} />
      ) : null}
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
}
