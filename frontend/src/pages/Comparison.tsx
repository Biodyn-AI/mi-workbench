import { useEffect, useState } from "react";
import {
  listRuns,
  compareRuns,
  type RunSummary,
  type CompareResult,
} from "../api/client";
import RunStatusBadge from "../components/RunStatus";

const GRADE_COLORS: Record<string, string> = {
  A: "#22c55e",
  B: "#3b82f6",
  C: "#eab308",
  D: "#f97316",
  F: "#ef4444",
};

function gradeColor(grade: string): string {
  return GRADE_COLORS[grade.charAt(0).toUpperCase()] ?? "#6b7280";
}

function similarityColor(value: number): string {
  // 0 = transparent, 1 = full indigo
  const alpha = Math.round(value * 255)
    .toString(16)
    .padStart(2, "0");
  return `#6366f1${alpha}`;
}

export default function Comparison() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [result, setResult] = useState<CompareResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [comparing, setComparing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listRuns()
      .then(setRuns)
      .finally(() => setLoading(false));
  }, []);

  const toggleRun = (runId: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(runId)) {
        next.delete(runId);
      } else if (next.size < 5) {
        next.add(runId);
      }
      return next;
    });
  };

  const handleCompare = async () => {
    if (selected.size < 2) return;
    setComparing(true);
    setError(null);
    try {
      const data = await compareRuns(Array.from(selected), "summary");
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Comparison failed");
    } finally {
      setComparing(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-indigo-300">Loading runs...</p>
      </div>
    );
  }

  // Find best values for highlighting
  const bestCost = result?.metrics?.best_by_cost;
  const bestTokens = result?.metrics?.best_by_tokens;

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-white mb-1">Compare Runs</h2>
        <p className="text-indigo-300 text-sm">
          Select 2-5 runs to compare metrics, quality, and output similarity
        </p>
      </div>

      {/* Run selector */}
      <div className="bg-card rounded-xl p-5 border border-white/10">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold text-white">
            Select Runs ({selected.size}/5)
          </h3>
          <button
            onClick={handleCompare}
            disabled={selected.size < 2 || comparing}
            className="px-4 py-2 bg-accent text-white text-sm font-medium rounded-lg hover:bg-accent-dark disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {comparing ? "Comparing..." : "Compare"}
          </button>
        </div>
        <div className="max-h-64 overflow-y-auto space-y-1">
          {runs.length === 0 ? (
            <p className="text-indigo-300 text-sm py-4 text-center">
              No runs available
            </p>
          ) : (
            runs.map((run) => (
              <label
                key={run.run_id}
                className={`flex items-center gap-3 px-3 py-2 rounded-lg cursor-pointer transition-colors ${
                  selected.has(run.run_id)
                    ? "bg-white/10"
                    : "hover:bg-white/5"
                }`}
              >
                <input
                  type="checkbox"
                  checked={selected.has(run.run_id)}
                  onChange={() => toggleRun(run.run_id)}
                  disabled={!selected.has(run.run_id) && selected.size >= 5}
                  className="accent-indigo-500"
                />
                <span className="text-sm font-mono text-accent-light truncate w-24">
                  {run.run_id}
                </span>
                <span className="text-sm text-indigo-200 truncate flex-1">
                  {run.task
                    ? run.task.length > 50
                      ? run.task.slice(0, 50) + "..."
                      : run.task
                    : "(no task)"}
                </span>
                <RunStatusBadge status={run.status} />
                <span className="text-xs text-indigo-300 w-20 text-right">
                  ${run.total_cost.toFixed(4)}
                </span>
              </label>
            ))
          )}
        </div>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 text-red-300 text-sm">
          {error}
        </div>
      )}

      {/* Metrics comparison table */}
      {result?.metrics && result.metrics.runs.length > 0 && (
        <div className="bg-card rounded-xl p-5 border border-white/10 overflow-x-auto">
          <h3 className="text-sm font-semibold text-white mb-4">
            Metrics Comparison
          </h3>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-indigo-300 text-xs uppercase tracking-wide border-b border-white/10">
                <th className="text-left py-2 pr-4">Run ID</th>
                <th className="text-right py-2 px-4">Iterations</th>
                <th className="text-right py-2 px-4">Tokens</th>
                <th className="text-right py-2 px-4">Cost</th>
                <th className="text-right py-2 px-4">Duration</th>
                <th className="text-left py-2 pl-4">Status</th>
              </tr>
            </thead>
            <tbody>
              {result.metrics.runs.map((m) => (
                <tr
                  key={m.run_id}
                  className="border-b border-white/5 text-white"
                >
                  <td className="py-2 pr-4 font-mono text-accent-light">
                    {m.run_id}
                  </td>
                  <td className="text-right py-2 px-4">{m.iterations}</td>
                  <td
                    className={`text-right py-2 px-4 ${
                      m.run_id === bestTokens ? "text-green-400 font-semibold" : ""
                    }`}
                  >
                    {m.tokens.toLocaleString()}
                  </td>
                  <td
                    className={`text-right py-2 px-4 ${
                      m.run_id === bestCost ? "text-green-400 font-semibold" : ""
                    }`}
                  >
                    ${m.cost.toFixed(4)}
                  </td>
                  <td className="text-right py-2 px-4">
                    {m.duration_s.toFixed(1)}s
                  </td>
                  <td className="py-2 pl-4">
                    <RunStatusBadge status={m.status as any} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Quality comparison */}
      {result?.quality && result.quality.runs.length > 0 && (
        <div className="bg-card rounded-xl p-5 border border-white/10">
          <h3 className="text-sm font-semibold text-white mb-4">
            Quality Comparison
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
            {result.quality.runs.map((q) => (
              <div
                key={q.run_id}
                className={`rounded-lg p-4 border ${
                  q.run_id === result.quality?.best_by_grade
                    ? "border-green-500/40 bg-green-500/5"
                    : "border-white/10 bg-white/5"
                }`}
              >
                <p className="text-xs font-mono text-indigo-300 mb-3 truncate">
                  {q.run_id}
                </p>
                {/* Grade circle */}
                <div className="flex items-center gap-3 mb-3">
                  <div
                    className="w-10 h-10 rounded-full flex items-center justify-center text-lg font-bold text-white"
                    style={{ backgroundColor: gradeColor(q.final_grade) }}
                  >
                    {q.final_grade}
                  </div>
                  <span className="text-xs text-indigo-300">
                    {q.total_critiques} critique{q.total_critiques !== 1 ? "s" : ""}
                  </span>
                </div>
                {/* Critique bar segments */}
                {q.total_critiques > 0 && (
                  <div className="flex h-2 rounded-full overflow-hidden bg-white/5">
                    {q.critical_count > 0 && (
                      <div
                        className="bg-red-500"
                        style={{
                          width: `${(q.critical_count / q.total_critiques) * 100}%`,
                        }}
                        title={`${q.critical_count} critical`}
                      />
                    )}
                    {q.high_count > 0 && (
                      <div
                        className="bg-orange-500"
                        style={{
                          width: `${(q.high_count / q.total_critiques) * 100}%`,
                        }}
                        title={`${q.high_count} high`}
                      />
                    )}
                    {q.total_critiques - q.critical_count - q.high_count > 0 && (
                      <div
                        className="bg-gray-500"
                        style={{
                          width: `${((q.total_critiques - q.critical_count - q.high_count) / q.total_critiques) * 100}%`,
                        }}
                        title="medium/low"
                      />
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Output similarity heatmap */}
      {result?.outputs &&
        Object.keys(result.outputs.artifact_comparisons).length > 0 && (
          <div className="bg-card rounded-xl p-5 border border-white/10">
            <h3 className="text-sm font-semibold text-white mb-4">
              Output Similarity
            </h3>
            <div className="space-y-6">
              {Object.entries(result.outputs.artifact_comparisons).map(
                ([artifact, data]) => (
                  <div key={artifact}>
                    <p className="text-xs text-indigo-300 font-medium mb-2">
                      {artifact}
                    </p>
                    <div className="inline-block">
                      {/* Column headers */}
                      <div className="flex">
                        <div className="w-20" />
                        {data.run_ids.map((id) => (
                          <div
                            key={id}
                            className="w-14 text-center text-[10px] font-mono text-indigo-300 truncate px-0.5"
                            title={id}
                          >
                            {id.slice(0, 6)}
                          </div>
                        ))}
                      </div>
                      {/* Matrix rows */}
                      {data.pairwise_similarity.map((row, i) => (
                        <div key={data.run_ids[i]} className="flex items-center">
                          <div
                            className="w-20 text-[10px] font-mono text-indigo-300 truncate pr-1 text-right"
                            title={data.run_ids[i]}
                          >
                            {data.run_ids[i].slice(0, 8)}
                          </div>
                          {row.map((val, j) => (
                            <div
                              key={j}
                              className="w-14 h-10 border border-white/5 flex items-center justify-center text-[10px] text-white"
                              style={{
                                backgroundColor: similarityColor(val),
                              }}
                              title={`${data.run_ids[i]} vs ${data.run_ids[j]}: ${(val * 100).toFixed(0)}%`}
                            >
                              {(val * 100).toFixed(0)}%
                            </div>
                          ))}
                        </div>
                      ))}
                    </div>
                  </div>
                ),
              )}
            </div>
          </div>
        )}

      {/* Empty state after comparison */}
      {result && !result.metrics && !result.quality && !result.outputs && (
        <div className="text-center py-16 text-indigo-300">
          <p className="text-lg font-medium">No comparison data available</p>
          <p className="text-sm mt-2">
            Selected runs may not have enough data to compare
          </p>
        </div>
      )}
    </div>
  );
}
