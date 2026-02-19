import { useEffect, useState } from "react";

const API = "/api/telemetry";

interface Summary {
  total_calls: number;
  total_tokens: number;
  total_cost: number;
  avg_latency: number;
  success_rate: number;
}

interface RoleBreakdown {
  [role: string]: {
    calls: number;
    total_tokens: number;
    cost: number;
    avg_latency: number;
    success_rate: number;
  };
}

interface LatencyPercentiles {
  p50: number;
  p90: number;
  p95: number;
  p99: number;
}

function Card({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-card rounded-xl p-5 border border-white/10">
      <p className="text-xs text-indigo-300 uppercase tracking-wide mb-1">
        {label}
      </p>
      <p className="text-2xl font-bold text-white">{value}</p>
    </div>
  );
}

function BarChart({
  data,
}: {
  data: { label: string; value: number; color: string }[];
}) {
  const maxVal = Math.max(...data.map((d) => d.value), 0.001);
  return (
    <div className="space-y-2">
      {data.map((d) => (
        <div key={d.label} className="flex items-center gap-3">
          <span className="text-xs text-indigo-300 w-28 truncate text-right">
            {d.label}
          </span>
          <div className="flex-1 h-6 bg-white/5 rounded overflow-hidden">
            <div
              className="h-full rounded"
              style={{
                width: `${(d.value / maxVal) * 100}%`,
                backgroundColor: d.color,
              }}
            />
          </div>
          <span className="text-xs text-white w-20 text-right">
            ${d.value.toFixed(4)}
          </span>
        </div>
      ))}
    </div>
  );
}

const ROLE_COLORS: Record<string, string> = {
  executor: "#6366f1",
  reviewer: "#8b5cf6",
  critic: "#a78bfa",
  planner: "#c4b5fd",
};

function colorForRole(role: string, idx: number): string {
  if (ROLE_COLORS[role]) return ROLE_COLORS[role];
  const hues = [210, 260, 310, 30, 160];
  return `hsl(${hues[idx % hues.length]}, 60%, 60%)`;
}

export default function Telemetry() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [byRole, setByRole] = useState<RoleBreakdown>({});
  const [byAdapter, setByAdapter] = useState<RoleBreakdown>({});
  const [latency, setLatency] = useState<LatencyPercentiles | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/summary`).then((r) => r.json()),
      fetch(`${API}/by-role`).then((r) => r.json()),
      fetch(`${API}/by-adapter`).then((r) => r.json()),
      fetch(`${API}/latency`).then((r) => r.json()),
    ])
      .then(([s, r, a, l]) => {
        setSummary(s);
        setByRole(r);
        setByAdapter(a);
        setLatency(l);
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-indigo-300">Loading telemetry...</p>
      </div>
    );
  }

  const roleChartData = Object.entries(byRole).map(
    ([role, stats], idx) => ({
      label: role,
      value: stats.cost,
      color: colorForRole(role, idx),
    })
  );

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-white mb-1">Telemetry</h2>
        <p className="text-indigo-300 text-sm">
          Cost, latency, and token usage across all runs
        </p>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card
          label="Total Cost"
          value={`$${(summary?.total_cost ?? 0).toFixed(4)}`}
        />
        <Card
          label="Total Tokens"
          value={(summary?.total_tokens ?? 0).toLocaleString()}
        />
        <Card
          label="Avg Latency"
          value={`${(summary?.avg_latency ?? 0).toFixed(2)}s`}
        />
        <Card
          label="Success Rate"
          value={`${((summary?.success_rate ?? 0) * 100).toFixed(1)}%`}
        />
      </div>

      {/* Cost by role bar chart */}
      {roleChartData.length > 0 && (
        <div className="bg-card rounded-xl p-5 border border-white/10">
          <h3 className="text-sm font-semibold text-white mb-4">
            Cost by Role
          </h3>
          <BarChart data={roleChartData} />
        </div>
      )}

      {/* Per-role table */}
      {Object.keys(byRole).length > 0 && (
        <div className="bg-card rounded-xl p-5 border border-white/10 overflow-x-auto">
          <h3 className="text-sm font-semibold text-white mb-4">
            Breakdown by Role
          </h3>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-indigo-300 text-xs uppercase tracking-wide border-b border-white/10">
                <th className="text-left py-2 pr-4">Role</th>
                <th className="text-right py-2 px-4">Calls</th>
                <th className="text-right py-2 px-4">Tokens</th>
                <th className="text-right py-2 px-4">Cost</th>
                <th className="text-right py-2 px-4">Avg Latency</th>
                <th className="text-right py-2 pl-4">Success</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(byRole).map(([role, stats]) => (
                <tr
                  key={role}
                  className="border-b border-white/5 text-white"
                >
                  <td className="py-2 pr-4 font-medium">{role}</td>
                  <td className="text-right py-2 px-4">{stats.calls}</td>
                  <td className="text-right py-2 px-4">
                    {stats.total_tokens.toLocaleString()}
                  </td>
                  <td className="text-right py-2 px-4">
                    ${stats.cost.toFixed(4)}
                  </td>
                  <td className="text-right py-2 px-4">
                    {stats.avg_latency.toFixed(2)}s
                  </td>
                  <td className="text-right py-2 pl-4">
                    {(stats.success_rate * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Per-adapter table */}
      {Object.keys(byAdapter).length > 0 && (
        <div className="bg-card rounded-xl p-5 border border-white/10 overflow-x-auto">
          <h3 className="text-sm font-semibold text-white mb-4">
            Breakdown by Adapter
          </h3>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-indigo-300 text-xs uppercase tracking-wide border-b border-white/10">
                <th className="text-left py-2 pr-4">Adapter</th>
                <th className="text-right py-2 px-4">Calls</th>
                <th className="text-right py-2 px-4">Tokens</th>
                <th className="text-right py-2 px-4">Cost</th>
                <th className="text-right py-2 px-4">Avg Latency</th>
                <th className="text-right py-2 pl-4">Success</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(byAdapter).map(([adapter, stats]) => (
                <tr
                  key={adapter}
                  className="border-b border-white/5 text-white"
                >
                  <td className="py-2 pr-4 font-medium">{adapter}</td>
                  <td className="text-right py-2 px-4">{stats.calls}</td>
                  <td className="text-right py-2 px-4">
                    {stats.total_tokens.toLocaleString()}
                  </td>
                  <td className="text-right py-2 px-4">
                    ${stats.cost.toFixed(4)}
                  </td>
                  <td className="text-right py-2 px-4">
                    {stats.avg_latency.toFixed(2)}s
                  </td>
                  <td className="text-right py-2 pl-4">
                    {(stats.success_rate * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Latency percentiles */}
      {latency && (
        <div className="bg-card rounded-xl p-5 border border-white/10">
          <h3 className="text-sm font-semibold text-white mb-4">
            Latency Percentiles
          </h3>
          <div className="grid grid-cols-4 gap-4">
            <div>
              <p className="text-xs text-indigo-300">p50</p>
              <p className="text-lg font-bold text-white">
                {latency.p50.toFixed(2)}s
              </p>
            </div>
            <div>
              <p className="text-xs text-indigo-300">p90</p>
              <p className="text-lg font-bold text-white">
                {latency.p90.toFixed(2)}s
              </p>
            </div>
            <div>
              <p className="text-xs text-indigo-300">p95</p>
              <p className="text-lg font-bold text-white">
                {latency.p95.toFixed(2)}s
              </p>
            </div>
            <div>
              <p className="text-xs text-indigo-300">p99</p>
              <p className="text-lg font-bold text-white">
                {latency.p99.toFixed(2)}s
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Empty state */}
      {(summary?.total_calls ?? 0) === 0 && (
        <div className="text-center py-16 text-indigo-300">
          <p className="text-lg font-medium">No telemetry data yet</p>
          <p className="text-sm mt-2">
            Run an experiment to start collecting cost and latency metrics
          </p>
        </div>
      )}
    </div>
  );
}
