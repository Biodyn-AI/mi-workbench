import { useParams, useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  Square,
  Play,
  Clock,
  DollarSign,
  Layers,
  FileText,
  CheckCircle,
  XCircle,
  Loader,
} from "lucide-react";
import { useApi } from "../hooks/useApi";
import { useWebSocket } from "../hooks/useWebSocket";
import { getRun, stopRun, resumeRun, type IterationResult } from "../api/client";
import RunStatusBadge from "../components/RunStatus";
import LogViewer from "../components/LogViewer";
import { useState, useCallback, useMemo } from "react";

export default function RunDetail() {
  const { runId } = useParams<{ runId: string }>();
  const navigate = useNavigate();
  const { data: run, loading, refresh } = useApi(
    () => getRun(runId!),
    [runId],
    3000,
  );
  const { messages, connected } = useWebSocket(
    run?.status === "running" ? runId : undefined,
  );
  const [actionLoading, setActionLoading] = useState(false);

  const handleStop = useCallback(async () => {
    if (!runId) return;
    setActionLoading(true);
    try {
      await stopRun(runId);
      refresh();
    } finally {
      setActionLoading(false);
    }
  }, [runId, refresh]);

  const handleResume = useCallback(async () => {
    if (!runId) return;
    setActionLoading(true);
    try {
      await resumeRun(runId);
      refresh();
    } finally {
      setActionLoading(false);
    }
  }, [runId, refresh]);

  const logEntries = useMemo(() => {
    // Combine iteration logs + WebSocket messages
    const entries: { timestamp: string; role: string; message: string }[] = [];

    if (run) {
      for (const iter of run.iterations) {
        for (const log of iter.logs) {
          entries.push({
            timestamp: iter.started_at ?? run.created_at,
            role: iter.role,
            message: log,
          });
        }
      }
    }

    for (const msg of messages) {
      entries.push({
        timestamp: msg.timestamp,
        role: (msg.data as { role?: string })?.role ?? "system",
        message:
          typeof msg.data === "string"
            ? msg.data
            : (msg.data as { message?: string })?.message ?? JSON.stringify(msg.data),
      });
    }

    return entries;
  }, [run, messages]);

  if (loading && !run) {
    return <div className="text-center text-gray-400 py-20">Loading...</div>;
  }

  if (!run) {
    return (
      <div className="text-center py-20">
        <p className="text-gray-500">Run not found.</p>
        <button
          onClick={() => navigate("/runs")}
          className="mt-4 text-accent hover:underline"
        >
          Back to Runs
        </button>
      </div>
    );
  }

  const canStop = run.status === "running" || run.status === "paused";
  const canResume = run.status === "paused" || run.status === "stopped";

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <button
            onClick={() => navigate("/runs")}
            className="flex items-center gap-1 text-sm text-gray-500 hover:text-gray-700 mb-2 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Runs
          </button>
          <div className="flex items-center gap-4">
            <h2 className="text-2xl font-bold text-gray-900 font-mono">
              {run.run_id}
            </h2>
            <RunStatusBadge status={run.status} />
            {run.status === "running" && connected && (
              <span className="flex items-center gap-1 text-xs text-green-600">
                <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse-dot" />
                Live
              </span>
            )}
          </div>
          <p className="text-gray-600 mt-1 max-w-2xl">{run.task || "(no task)"}</p>
        </div>
        <div className="flex gap-2">
          {canStop && (
            <button
              onClick={handleStop}
              disabled={actionLoading}
              className="flex items-center gap-2 px-4 py-2 bg-red-500 text-white text-sm font-medium rounded-lg hover:bg-red-600 disabled:opacity-50 transition-colors"
            >
              <Square className="w-4 h-4" />
              Stop
            </button>
          )}
          {canResume && (
            <button
              onClick={handleResume}
              disabled={actionLoading}
              className="flex items-center gap-2 px-4 py-2 bg-green-500 text-white text-sm font-medium rounded-lg hover:bg-green-600 disabled:opacity-50 transition-colors"
            >
              <Play className="w-4 h-4" />
              Resume
            </button>
          )}
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MiniStat
          icon={<Layers className="w-4 h-4" />}
          label="Iterations"
          value={`${run.current_iteration} / ${run.max_iterations}`}
        />
        <MiniStat
          icon={<Clock className="w-4 h-4" />}
          label="Provider"
          value={run.provider}
        />
        <MiniStat
          icon={<DollarSign className="w-4 h-4" />}
          label="Cost"
          value={`$${run.total_cost.toFixed(2)}`}
        />
        <MiniStat
          icon={<FileText className="w-4 h-4" />}
          label="Preset"
          value={run.loop_preset}
        />
      </div>

      {/* Two-column layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Iteration timeline */}
        <div className="lg:col-span-1">
          <div className="bg-white rounded-xl border border-surface-border">
            <div className="px-5 py-4 border-b border-surface-border">
              <h3 className="font-semibold text-gray-800">Iterations</h3>
            </div>
            <div className="p-4 space-y-0 max-h-[500px] overflow-y-auto">
              {run.iterations.length === 0 ? (
                <p className="text-sm text-gray-400 text-center py-4">
                  No iterations yet.
                </p>
              ) : (
                run.iterations.map((iter) => (
                  <IterationStep key={iter.iteration_id} iteration={iter} />
                ))
              )}
            </div>
          </div>
        </div>

        {/* Logs + Artifacts */}
        <div className="lg:col-span-2 space-y-6">
          <LogViewer logs={logEntries} maxHeight="400px" />

          {/* Artifacts per iteration */}
          {run.iterations.some((it) => it.artifacts_produced.length > 0) && (
            <div className="bg-white rounded-xl border border-surface-border">
              <div className="px-5 py-4 border-b border-surface-border">
                <h3 className="font-semibold text-gray-800">Artifacts</h3>
              </div>
              <div className="p-4 space-y-3">
                {run.iterations
                  .filter((it) => it.artifacts_produced.length > 0)
                  .map((iter) => (
                    <div key={iter.iteration_id}>
                      <p className="text-xs text-gray-500 font-medium mb-1">
                        Iteration {iter.iteration_number} ({iter.role})
                      </p>
                      <div className="flex flex-wrap gap-2">
                        {iter.artifacts_produced.map((a) => (
                          <span
                            key={a}
                            className="inline-flex items-center gap-1 px-2.5 py-1 bg-gray-100 text-gray-700 text-xs rounded-md"
                          >
                            <FileText className="w-3 h-3" />
                            {a.split("/").pop()}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          )}

          {/* Error display */}
          {run.error && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-xl text-sm text-red-700">
              <span className="font-semibold">Error:</span> {run.error}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function MiniStat({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="bg-white rounded-lg border border-surface-border p-4 flex items-center gap-3">
      <div className="text-gray-400">{icon}</div>
      <div>
        <p className="text-xs text-gray-500">{label}</p>
        <p className="text-sm font-semibold text-gray-800">{value}</p>
      </div>
    </div>
  );
}

function IterationStep({ iteration }: { iteration: IterationResult }) {
  const statusIcons: Record<string, React.ReactNode> = {
    completed: <CheckCircle className="w-4 h-4 text-green-500" />,
    failed: <XCircle className="w-4 h-4 text-red-500" />,
    running: <Loader className="w-4 h-4 text-blue-500 animate-spin" />,
    pending: <Clock className="w-4 h-4 text-gray-400" />,
  };

  const roleColors: Record<string, string> = {
    executor: "border-blue-400",
    reviewer: "border-amber-400",
    adversarial_reviewer: "border-red-400",
    idea_generator: "border-purple-400",
  };

  return (
    <div className="flex gap-3 py-3 border-b border-surface-border last:border-b-0">
      <div className="flex flex-col items-center">
        {statusIcons[iteration.status] ?? statusIcons.pending}
        <div className={`w-0.5 flex-1 mt-1 ${roleColors[iteration.role] ? "bg-gray-200" : "bg-gray-200"}`} />
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-gray-800">
            #{iteration.iteration_number}
          </span>
          <span
            className={`text-xs px-1.5 py-0.5 rounded font-medium ${
              roleColors[iteration.role]
                ? `border ${roleColors[iteration.role]} text-gray-600`
                : "bg-gray-100 text-gray-600"
            }`}
          >
            {iteration.role}
          </span>
        </div>
        {iteration.output_summary && (
          <p className="text-xs text-gray-500 mt-1 line-clamp-2">
            {iteration.output_summary}
          </p>
        )}
        {iteration.error && (
          <p className="text-xs text-red-500 mt-1">{iteration.error}</p>
        )}
        <div className="flex items-center gap-3 mt-1 text-xs text-gray-400">
          {iteration.token_usage > 0 && (
            <span>{iteration.token_usage.toLocaleString()} tokens</span>
          )}
          {iteration.cost_estimate > 0 && (
            <span>${iteration.cost_estimate.toFixed(3)}</span>
          )}
        </div>
      </div>
    </div>
  );
}
