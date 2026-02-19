import { useNavigate } from "react-router-dom";
import {
  Play,
  FileText,
  Activity,
  DollarSign,
  Clock,
  Plus,
  ArrowRight,
} from "lucide-react";
import { useApi } from "../hooks/useApi";
import { listRuns, listWorkspaces } from "../api/client";
import type { RunSummary } from "../api/client";
import RunStatusBadge from "../components/RunStatus";

export default function Overview() {
  const { data: runs, loading: runsLoading } = useApi(listRuns, [], 10000);
  const { data: workspaces } = useApi(listWorkspaces);
  const navigate = useNavigate();

  const activeRuns = runs?.filter((r) => r.status === "running").length ?? 0;
  const totalIterations = runs?.reduce((s, r) => s + r.current_iteration, 0) ?? 0;
  const totalCost = runs?.reduce((s, r) => s + r.total_cost, 0) ?? 0;
  const recentRuns = (runs ?? []).slice(0, 8);

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">Dashboard</h2>
        <div className="flex gap-3">
          <button
            onClick={() => navigate("/runs")}
            className="flex items-center gap-2 px-4 py-2 bg-accent text-white text-sm font-medium rounded-lg hover:bg-accent-dark transition-colors"
          >
            <Plus className="w-4 h-4" />
            New Run
          </button>
          <button
            onClick={() => navigate("/artifacts")}
            className="flex items-center gap-2 px-4 py-2 border border-surface-border text-gray-700 text-sm font-medium rounded-lg hover:bg-gray-50 transition-colors"
          >
            <FileText className="w-4 h-4" />
            Browse Artifacts
          </button>
        </div>
      </div>

      {/* Stats cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
        <StatCard
          icon={<Play className="w-5 h-5" />}
          label="Active Runs"
          value={activeRuns}
          color="blue"
        />
        <StatCard
          icon={<Activity className="w-5 h-5" />}
          label="Total Iterations"
          value={totalIterations}
          color="indigo"
        />
        <StatCard
          icon={<DollarSign className="w-5 h-5" />}
          label="Total Cost"
          value={`$${totalCost.toFixed(2)}`}
          color="green"
        />
        <StatCard
          icon={<Clock className="w-5 h-5" />}
          label="Workspaces"
          value={workspaces?.length ?? 0}
          color="purple"
        />
      </div>

      {/* Recent runs */}
      <div className="bg-white rounded-xl border border-surface-border">
        <div className="flex items-center justify-between px-6 py-4 border-b border-surface-border">
          <h3 className="font-semibold text-gray-800">Recent Runs</h3>
          <button
            onClick={() => navigate("/runs")}
            className="flex items-center gap-1 text-sm text-accent hover:text-accent-dark transition-colors"
          >
            View all
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
        <div className="overflow-x-auto">
          {runsLoading ? (
            <div className="p-8 text-center text-gray-400">Loading...</div>
          ) : recentRuns.length === 0 ? (
            <div className="p-8 text-center text-gray-400">
              No runs yet. Create one to get started.
            </div>
          ) : (
            <table className="w-full">
              <thead>
                <tr className="text-xs text-gray-500 uppercase tracking-wider">
                  <th className="text-left px-6 py-3 font-medium">Run ID</th>
                  <th className="text-left px-6 py-3 font-medium">Task</th>
                  <th className="text-left px-6 py-3 font-medium">Preset</th>
                  <th className="text-left px-6 py-3 font-medium">Status</th>
                  <th className="text-left px-6 py-3 font-medium">Provider</th>
                  <th className="text-right px-6 py-3 font-medium">Iters</th>
                  <th className="text-right px-6 py-3 font-medium">Cost</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-surface-border">
                {recentRuns.map((run: RunSummary) => (
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
                    <td className="px-6 py-3 text-sm text-gray-600 text-right">
                      ${run.total_cost.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}

function StatCard({
  icon,
  label,
  value,
  color,
}: {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  color: string;
}) {
  const colorMap: Record<string, string> = {
    blue: "bg-blue-50 text-blue-600",
    indigo: "bg-indigo-50 text-indigo-600",
    green: "bg-green-50 text-green-600",
    purple: "bg-purple-50 text-purple-600",
  };
  const iconClass = colorMap[color] ?? colorMap.blue;

  return (
    <div className="bg-white rounded-xl border border-surface-border p-5">
      <div className="flex items-center gap-3">
        <div className={`p-2.5 rounded-lg ${iconClass}`}>{icon}</div>
        <div>
          <p className="text-sm text-gray-500">{label}</p>
          <p className="text-2xl font-bold text-gray-900">{value}</p>
        </div>
      </div>
    </div>
  );
}
