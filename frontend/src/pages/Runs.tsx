import { useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { Plus, X, Search } from "lucide-react";
import { useApi } from "../hooks/useApi";
import {
  listRuns,
  listWorkspaces,
  createRun,
  type RunStatus,
  type RunSummary,
  type RunCreate,
  type ProviderName,
} from "../api/client";
import RunStatusBadge from "../components/RunStatus";

const STATUS_OPTIONS: (RunStatus | "all")[] = [
  "all",
  "pending",
  "running",
  "paused",
  "completed",
  "stopped",
  "failed",
];

export default function Runs() {
  const navigate = useNavigate();
  const [statusFilter, setStatusFilter] = useState<RunStatus | "all">("all");
  const [wsFilter, setWsFilter] = useState("");
  const [dialogOpen, setDialogOpen] = useState(false);

  const { data: runs, loading, refresh } = useApi(
    () => listRuns(wsFilter || undefined, statusFilter === "all" ? undefined : statusFilter),
    [wsFilter, statusFilter],
    5000,
  );
  const { data: workspaces } = useApi(listWorkspaces);

  const filtered = runs ?? [];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">Runs</h2>
        <button
          onClick={() => setDialogOpen(true)}
          className="flex items-center gap-2 px-4 py-2 bg-accent text-white text-sm font-medium rounded-lg hover:bg-accent-dark transition-colors"
        >
          <Plus className="w-4 h-4" />
          New Run
        </button>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-4">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <select
            value={wsFilter}
            onChange={(e) => setWsFilter(e.target.value)}
            className="pl-9 pr-4 py-2 text-sm border border-surface-border rounded-lg bg-white focus:ring-2 focus:ring-accent focus:border-transparent outline-none appearance-none"
          >
            <option value="">All Workspaces</option>
            {workspaces?.map((ws) => (
              <option key={ws.id} value={ws.id}>
                {ws.name}
              </option>
            ))}
          </select>
        </div>
        <div className="flex rounded-lg border border-surface-border overflow-hidden">
          {STATUS_OPTIONS.map((s) => (
            <button
              key={s}
              onClick={() => setStatusFilter(s)}
              className={`px-3 py-2 text-xs font-medium transition-colors ${
                statusFilter === s
                  ? "bg-accent text-white"
                  : "bg-white text-gray-600 hover:bg-gray-50"
              }`}
            >
              {s === "all" ? "All" : s.charAt(0).toUpperCase() + s.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Table */}
      <div className="bg-white rounded-xl border border-surface-border overflow-x-auto">
        {loading ? (
          <div className="p-8 text-center text-gray-400">Loading...</div>
        ) : filtered.length === 0 ? (
          <div className="p-8 text-center text-gray-400">No runs found.</div>
        ) : (
          <table className="w-full">
            <thead>
              <tr className="text-xs text-gray-500 uppercase tracking-wider border-b border-surface-border">
                <th className="text-left px-6 py-3 font-medium">Run ID</th>
                <th className="text-left px-6 py-3 font-medium">Task</th>
                <th className="text-left px-6 py-3 font-medium">Preset</th>
                <th className="text-left px-6 py-3 font-medium">Status</th>
                <th className="text-left px-6 py-3 font-medium">Provider</th>
                <th className="text-right px-6 py-3 font-medium">Iters</th>
                <th className="text-right px-6 py-3 font-medium">Created</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-surface-border">
              {filtered.map((run: RunSummary) => (
                <tr
                  key={run.run_id}
                  className="hover:bg-gray-50 cursor-pointer transition-colors"
                  onClick={() => navigate(`/runs/${run.run_id}`)}
                >
                  <td className="px-6 py-3 text-sm font-mono text-accent">
                    {run.run_id}
                  </td>
                  <td className="px-6 py-3 text-sm text-gray-700 max-w-xs truncate">
                    {run.task || "(no task)"}
                  </td>
                  <td className="px-6 py-3 text-sm text-gray-600">
                    {run.loop_preset}
                  </td>
                  <td className="px-6 py-3">
                    <RunStatusBadge status={run.status} />
                  </td>
                  <td className="px-6 py-3 text-sm text-gray-600">
                    {run.provider}
                  </td>
                  <td className="px-6 py-3 text-sm text-gray-600 text-right">
                    {run.current_iteration}
                  </td>
                  <td className="px-6 py-3 text-sm text-gray-500 text-right">
                    {new Date(run.created_at).toLocaleDateString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* New Run Dialog */}
      {dialogOpen && (
        <NewRunDialog
          workspaces={workspaces ?? []}
          onClose={() => setDialogOpen(false)}
          onCreated={() => {
            setDialogOpen(false);
            refresh();
          }}
        />
      )}
    </div>
  );
}

function NewRunDialog({
  workspaces,
  onClose,
  onCreated,
}: {
  workspaces: { id: string; name: string }[];
  onClose: () => void;
  onCreated: () => void;
}) {
  const [form, setForm] = useState<RunCreate>({
    workspace_id: workspaces[0]?.id ?? "",
    task: "",
    loop_preset: "executor_reviewer",
    provider: "mock",
    max_iterations: 50,
  });
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      setSubmitting(true);
      setError(null);
      try {
        const run = await createRun(form);
        onCreated();
        navigate(`/runs/${run.run_id}`);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to create run");
      } finally {
        setSubmitting(false);
      }
    },
    [form, onCreated, navigate],
  );

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-lg mx-4">
        <div className="flex items-center justify-between px-6 py-4 border-b border-surface-border">
          <h3 className="font-semibold text-gray-800">New Run</h3>
          <button onClick={onClose} className="p-1 hover:bg-gray-100 rounded">
            <X className="w-5 h-5 text-gray-400" />
          </button>
        </div>
        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Workspace
            </label>
            <select
              value={form.workspace_id}
              onChange={(e) => setForm({ ...form, workspace_id: e.target.value })}
              className="w-full px-3 py-2 text-sm border border-surface-border rounded-lg focus:ring-2 focus:ring-accent focus:border-transparent outline-none"
            >
              {workspaces.map((ws) => (
                <option key={ws.id} value={ws.id}>
                  {ws.name}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Task
            </label>
            <textarea
              value={form.task}
              onChange={(e) => setForm({ ...form, task: e.target.value })}
              placeholder="Describe the research task..."
              rows={3}
              className="w-full px-3 py-2 text-sm border border-surface-border rounded-lg focus:ring-2 focus:ring-accent focus:border-transparent outline-none resize-y"
            />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Loop Preset
              </label>
              <select
                value={form.loop_preset}
                onChange={(e) =>
                  setForm({ ...form, loop_preset: e.target.value })
                }
                className="w-full px-3 py-2 text-sm border border-surface-border rounded-lg focus:ring-2 focus:ring-accent focus:border-transparent outline-none"
              >
                <option value="executor_reviewer">Executor + Reviewer</option>
                <option value="adversarial_review">Adversarial Review</option>
                <option value="idea_generation">Idea Generation</option>
                <option value="full_pipeline">Full Pipeline</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Provider
              </label>
              <select
                value={form.provider}
                onChange={(e) =>
                  setForm({ ...form, provider: e.target.value as ProviderName })
                }
                className="w-full px-3 py-2 text-sm border border-surface-border rounded-lg focus:ring-2 focus:ring-accent focus:border-transparent outline-none"
              >
                <option value="mock">Mock</option>
                <option value="claude_code">Claude Code</option>
                <option value="codex_cli">Codex CLI</option>
                <option value="gemini_cli">Gemini CLI</option>
              </select>
            </div>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Max Iterations
            </label>
            <input
              type="number"
              value={form.max_iterations}
              onChange={(e) =>
                setForm({ ...form, max_iterations: parseInt(e.target.value) || 50 })
              }
              min={1}
              max={500}
              className="w-full px-3 py-2 text-sm border border-surface-border rounded-lg focus:ring-2 focus:ring-accent focus:border-transparent outline-none"
            />
          </div>
          {error && (
            <div className="p-3 bg-red-50 text-red-700 text-sm rounded-lg">
              {error}
            </div>
          )}
          <div className="flex justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-sm text-gray-600 hover:bg-gray-50 rounded-lg transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={submitting || !form.task.trim()}
              className="px-4 py-2 bg-accent text-white text-sm font-medium rounded-lg hover:bg-accent-dark disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {submitting ? "Creating..." : "Create Run"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
