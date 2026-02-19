import { useEffect, useState } from "react";
import type {
  KnowledgeClaim,
  KnowledgeFact,
  KnowledgeLink,
  KnowledgeSummary,
  KnowledgeGraph,
  RunSummary,
} from "../api/client";

const API = "/api";

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

function UncertaintyBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color =
    value < 0.33 ? "#22c55e" : value < 0.66 ? "#eab308" : "#ef4444";
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-2 bg-white/10 rounded overflow-hidden">
        <div
          className="h-full rounded"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      <span className="text-xs text-indigo-300 w-10 text-right">{pct}%</span>
    </div>
  );
}

function StrengthBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color =
    value > 0.66 ? "#22c55e" : value > 0.33 ? "#eab308" : "#ef4444";
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-2 bg-white/10 rounded overflow-hidden">
        <div
          className="h-full rounded"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      <span className="text-xs text-indigo-300 w-10 text-right">{pct}%</span>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    active: "bg-green-500/20 text-green-300",
    archived: "bg-gray-500/20 text-gray-300",
    superseded: "bg-yellow-500/20 text-yellow-300",
    refuted: "bg-red-500/20 text-red-300",
  };
  return (
    <span
      className={`text-xs px-2 py-0.5 rounded-full ${colors[status] ?? "bg-indigo-500/20 text-indigo-300"}`}
    >
      {status}
    </span>
  );
}

function nodeColor(uncertainty: number): string {
  if (uncertainty < 0.33) return "#22c55e";
  if (uncertainty < 0.66) return "#eab308";
  return "#ef4444";
}

function GraphView({ graph }: { graph: KnowledgeGraph }) {
  if (graph.nodes.length === 0) {
    return (
      <div className="text-center py-12 text-indigo-300">
        <p className="text-sm">No graph data available</p>
      </div>
    );
  }

  const xs = graph.nodes.map((n) => n.position.x);
  const ys = graph.nodes.map((n) => n.position.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;
  const pad = 60;
  const w = 800;
  const h = 500;

  function scaleX(x: number) {
    return pad + ((x - minX) / rangeX) * (w - 2 * pad);
  }
  function scaleY(y: number) {
    return pad + ((y - minY) / rangeY) * (h - 2 * pad);
  }

  const nodeMap = new Map(graph.nodes.map((n) => [n.id, n]));

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full" style={{ maxHeight: 500 }}>
      {graph.edges.map((e) => {
        const src = nodeMap.get(e.source);
        const tgt = nodeMap.get(e.target);
        if (!src || !tgt) return null;
        return (
          <line
            key={e.id}
            x1={scaleX(src.position.x)}
            y1={scaleY(src.position.y)}
            x2={scaleX(tgt.position.x)}
            y2={scaleY(tgt.position.y)}
            stroke="rgba(99,102,241,0.4)"
            strokeWidth={1.5}
          />
        );
      })}
      {graph.edges.map((e) => {
        const src = nodeMap.get(e.source);
        const tgt = nodeMap.get(e.target);
        if (!src || !tgt || !e.label) return null;
        const mx = (scaleX(src.position.x) + scaleX(tgt.position.x)) / 2;
        const my = (scaleY(src.position.y) + scaleY(tgt.position.y)) / 2;
        return (
          <text
            key={`label-${e.id}`}
            x={mx}
            y={my - 6}
            textAnchor="middle"
            className="text-[9px]"
            fill="rgba(165,180,252,0.7)"
          >
            {e.label}
          </text>
        );
      })}
      {graph.nodes.map((n) => {
        const cx = scaleX(n.position.x);
        const cy = scaleY(n.position.y);
        const r = 10 + n.data.strength * 10;
        return (
          <g key={n.id}>
            <circle
              cx={cx}
              cy={cy}
              r={r}
              fill={nodeColor(n.data.uncertainty)}
              fillOpacity={0.7}
              stroke={nodeColor(n.data.uncertainty)}
              strokeWidth={1.5}
            />
            <text
              x={cx}
              y={cy + r + 14}
              textAnchor="middle"
              fill="white"
              className="text-[10px]"
            >
              {n.data.label.length > 20
                ? n.data.label.slice(0, 20) + "..."
                : n.data.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function ClaimModal({
  claim,
  facts,
  links,
  allClaims,
  onClose,
}: {
  claim: KnowledgeClaim;
  facts: KnowledgeFact[];
  links: KnowledgeLink[];
  allClaims: KnowledgeClaim[];
  onClose: () => void;
}) {
  const claimFacts = facts.filter((f) => f.claim_id === claim.claim_id);
  const claimLinks = links.filter(
    (l) =>
      l.source_claim_id === claim.claim_id ||
      l.target_claim_id === claim.claim_id
  );
  const claimMap = new Map(allClaims.map((c) => [c.claim_id, c]));

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
      onClick={onClose}
    >
      <div
        className="bg-card border border-white/10 rounded-xl p-6 max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between mb-4">
          <h3 className="text-lg font-bold text-white pr-4">Claim Detail</h3>
          <button
            onClick={onClose}
            className="text-indigo-300 hover:text-white text-xl leading-none"
          >
            x
          </button>
        </div>

        <div className="space-y-4">
          <div>
            <p className="text-xs text-indigo-300 uppercase tracking-wide mb-1">
              Claim Text
            </p>
            <p className="text-sm text-white">{claim.claim_text}</p>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-xs text-indigo-300 uppercase tracking-wide mb-1">
                Uncertainty
              </p>
              <UncertaintyBar value={claim.uncertainty} />
            </div>
            <div>
              <p className="text-xs text-indigo-300 uppercase tracking-wide mb-1">
                Strength
              </p>
              <StrengthBar value={claim.strength} />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-xs text-indigo-300 uppercase tracking-wide mb-1">
                Source Artifact
              </p>
              <p className="text-sm text-white">{claim.source_artifact}</p>
            </div>
            <div>
              <p className="text-xs text-indigo-300 uppercase tracking-wide mb-1">
                Status
              </p>
              <StatusBadge status={claim.status} />
            </div>
          </div>

          {claim.evidence_pointers.length > 0 && (
            <div>
              <p className="text-xs text-indigo-300 uppercase tracking-wide mb-1">
                Evidence Pointers
              </p>
              <ul className="text-sm text-white space-y-1">
                {claim.evidence_pointers.map((ep, i) => (
                  <li key={i} className="flex gap-2">
                    <span className="text-indigo-400">-</span>
                    <span>{ep}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {claim.falsification_tests.length > 0 && (
            <div>
              <p className="text-xs text-indigo-300 uppercase tracking-wide mb-1">
                Falsification Tests
              </p>
              <ul className="text-sm text-white space-y-1">
                {claim.falsification_tests.map((ft, i) => (
                  <li key={i} className="flex gap-2">
                    <span className="text-indigo-400">-</span>
                    <span>{ft}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {claimFacts.length > 0 && (
            <div>
              <p className="text-xs text-indigo-300 uppercase tracking-wide mb-1">
                Linked Facts ({claimFacts.length})
              </p>
              <div className="space-y-2">
                {claimFacts.map((f) => (
                  <div
                    key={f.fact_id}
                    className="bg-white/5 rounded-lg p-3 text-sm text-white"
                  >
                    <p>{f.fact_text}</p>
                    <p className="text-xs text-indigo-300 mt-1">
                      Source: {f.source} | Confidence:{" "}
                      {(f.confidence * 100).toFixed(0)}%
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {claimLinks.length > 0 && (
            <div>
              <p className="text-xs text-indigo-300 uppercase tracking-wide mb-1">
                Linked Claims ({claimLinks.length})
              </p>
              <div className="space-y-2">
                {claimLinks.map((l) => {
                  const otherId =
                    l.source_claim_id === claim.claim_id
                      ? l.target_claim_id
                      : l.source_claim_id;
                  const other = claimMap.get(otherId);
                  return (
                    <div
                      key={l.link_id}
                      className="bg-white/5 rounded-lg p-3 text-sm text-white"
                    >
                      <p className="text-xs text-indigo-400 mb-1">
                        {l.link_type} (weight: {l.weight.toFixed(2)})
                      </p>
                      <p>
                        {other
                          ? other.claim_text.length > 100
                            ? other.claim_text.slice(0, 100) + "..."
                            : other.claim_text
                          : otherId}
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          <div className="text-xs text-indigo-400 pt-2 border-t border-white/10">
            Created: {new Date(claim.created_at).toLocaleString()} | Updated:{" "}
            {new Date(claim.updated_at).toLocaleString()} | ID:{" "}
            {claim.claim_id}
          </div>
        </div>
      </div>
    </div>
  );
}

type SortKey = "claim_text" | "uncertainty" | "strength" | "source_artifact" | "status" | "created_at";

export default function Knowledge() {
  const [summary, setSummary] = useState<KnowledgeSummary | null>(null);
  const [claims, setClaims] = useState<KnowledgeClaim[]>([]);
  const [facts, setFacts] = useState<KnowledgeFact[]>([]);
  const [links, setLinks] = useState<KnowledgeLink[]>([]);
  const [graph, setGraph] = useState<KnowledgeGraph | null>(null);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedClaim, setSelectedClaim] = useState<KnowledgeClaim | null>(null);
  const [filterRunId, setFilterRunId] = useState<string>("");
  const [filterStatus, setFilterStatus] = useState<string>("all");
  const [sortKey, setSortKey] = useState<SortKey>("created_at");
  const [sortAsc, setSortAsc] = useState(false);

  useEffect(() => {
    const runParams = filterRunId ? `?run_id=${filterRunId}` : "";
    Promise.all([
      fetch(`${API}/knowledge/summary`).then((r) => r.json()),
      fetch(`${API}/knowledge/claims${runParams}`).then((r) => r.json()),
      fetch(`${API}/knowledge/facts`).then((r) => r.json()),
      fetch(`${API}/knowledge/links`).then((r) => r.json()),
      fetch(`${API}/knowledge/graph`).then((r) => r.json()),
      fetch(`${API}/runs`).then((r) => r.json()),
    ])
      .then(([s, c, f, l, g, r]) => {
        setSummary(s);
        setClaims(c);
        setFacts(f);
        setLinks(l);
        setGraph(g);
        setRuns(r);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [filterRunId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-indigo-300">Loading knowledge graph...</p>
      </div>
    );
  }

  const filteredClaims = claims.filter((c) => {
    if (filterStatus !== "all" && c.status !== filterStatus) return false;
    return true;
  });

  const sortedClaims = [...filteredClaims].sort((a, b) => {
    let cmp = 0;
    switch (sortKey) {
      case "claim_text":
        cmp = a.claim_text.localeCompare(b.claim_text);
        break;
      case "uncertainty":
        cmp = a.uncertainty - b.uncertainty;
        break;
      case "strength":
        cmp = a.strength - b.strength;
        break;
      case "source_artifact":
        cmp = a.source_artifact.localeCompare(b.source_artifact);
        break;
      case "status":
        cmp = a.status.localeCompare(b.status);
        break;
      case "created_at":
        cmp = a.created_at.localeCompare(b.created_at);
        break;
    }
    return sortAsc ? cmp : -cmp;
  });

  function handleSort(key: SortKey) {
    if (sortKey === key) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(key);
      setSortAsc(true);
    }
  }

  function sortIndicator(key: SortKey) {
    if (sortKey !== key) return "";
    return sortAsc ? " ^" : " v";
  }

  const statuses = [...new Set(claims.map((c) => c.status))];

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-white mb-1">Knowledge Graph</h2>
        <p className="text-indigo-300 text-sm">
          Claims, facts, and relationships extracted from runs
        </p>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card
          label="Total Claims"
          value={(summary?.total_claims ?? 0).toLocaleString()}
        />
        <Card
          label="Total Facts"
          value={(summary?.total_facts ?? 0).toLocaleString()}
        />
        <Card
          label="Avg Uncertainty"
          value={`${((summary?.avg_uncertainty ?? 0) * 100).toFixed(1)}%`}
        />
        <Card
          label="Avg Strength"
          value={`${((summary?.avg_strength ?? 0) * 100).toFixed(1)}%`}
        />
      </div>

      {/* Filter controls */}
      <div className="flex flex-wrap gap-4">
        <div>
          <label className="text-xs text-indigo-300 uppercase tracking-wide block mb-1">
            Filter by Run
          </label>
          <select
            value={filterRunId}
            onChange={(e) => setFilterRunId(e.target.value)}
            className="bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white"
          >
            <option value="">All runs</option>
            {runs.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {r.task.length > 40 ? r.task.slice(0, 40) + "..." : r.task} (
                {r.run_id.slice(0, 8)})
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="text-xs text-indigo-300 uppercase tracking-wide block mb-1">
            Filter by Status
          </label>
          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className="bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white"
          >
            <option value="all">All statuses</option>
            {statuses.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Graph visualization */}
      {graph && (
        <div className="bg-card rounded-xl p-5 border border-white/10">
          <h3 className="text-sm font-semibold text-white mb-4">
            Claim Graph
          </h3>
          <div className="flex gap-4 mb-3 text-xs text-indigo-300">
            <span className="flex items-center gap-1">
              <span
                className="inline-block w-3 h-3 rounded-full"
                style={{ backgroundColor: "#22c55e" }}
              />
              Low uncertainty
            </span>
            <span className="flex items-center gap-1">
              <span
                className="inline-block w-3 h-3 rounded-full"
                style={{ backgroundColor: "#eab308" }}
              />
              Medium
            </span>
            <span className="flex items-center gap-1">
              <span
                className="inline-block w-3 h-3 rounded-full"
                style={{ backgroundColor: "#ef4444" }}
              />
              High uncertainty
            </span>
          </div>
          <GraphView graph={graph} />
        </div>
      )}

      {/* Claims table */}
      {sortedClaims.length > 0 && (
        <div className="bg-card rounded-xl p-5 border border-white/10 overflow-x-auto">
          <h3 className="text-sm font-semibold text-white mb-4">
            Claims ({sortedClaims.length})
          </h3>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-indigo-300 text-xs uppercase tracking-wide border-b border-white/10">
                <th
                  className="text-left py-2 pr-4 cursor-pointer select-none"
                  onClick={() => handleSort("claim_text")}
                >
                  Claim{sortIndicator("claim_text")}
                </th>
                <th
                  className="text-left py-2 px-4 cursor-pointer select-none w-32"
                  onClick={() => handleSort("uncertainty")}
                >
                  Uncertainty{sortIndicator("uncertainty")}
                </th>
                <th
                  className="text-left py-2 px-4 cursor-pointer select-none w-32"
                  onClick={() => handleSort("strength")}
                >
                  Strength{sortIndicator("strength")}
                </th>
                <th
                  className="text-left py-2 px-4 cursor-pointer select-none"
                  onClick={() => handleSort("source_artifact")}
                >
                  Source{sortIndicator("source_artifact")}
                </th>
                <th
                  className="text-left py-2 px-4 cursor-pointer select-none"
                  onClick={() => handleSort("status")}
                >
                  Status{sortIndicator("status")}
                </th>
                <th
                  className="text-right py-2 pl-4 cursor-pointer select-none"
                  onClick={() => handleSort("created_at")}
                >
                  Created{sortIndicator("created_at")}
                </th>
              </tr>
            </thead>
            <tbody>
              {sortedClaims.map((c) => (
                <tr
                  key={c.claim_id}
                  className="border-b border-white/5 text-white hover:bg-white/5 cursor-pointer transition-colors"
                  onClick={() => setSelectedClaim(c)}
                >
                  <td className="py-2 pr-4">
                    {c.claim_text.length > 80
                      ? c.claim_text.slice(0, 80) + "..."
                      : c.claim_text}
                  </td>
                  <td className="py-2 px-4">
                    <UncertaintyBar value={c.uncertainty} />
                  </td>
                  <td className="py-2 px-4">
                    <StrengthBar value={c.strength} />
                  </td>
                  <td className="py-2 px-4 text-indigo-300 text-xs">
                    {c.source_artifact}
                  </td>
                  <td className="py-2 px-4">
                    <StatusBadge status={c.status} />
                  </td>
                  <td className="text-right py-2 pl-4 text-xs text-indigo-300">
                    {new Date(c.created_at).toLocaleDateString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Empty state */}
      {(summary?.total_claims ?? 0) === 0 && (
        <div className="text-center py-16 text-indigo-300">
          <p className="text-lg font-medium">No knowledge data yet</p>
          <p className="text-sm mt-2">
            Run an experiment to start building the knowledge graph
          </p>
        </div>
      )}

      {/* Claim detail modal */}
      {selectedClaim && (
        <ClaimModal
          claim={selectedClaim}
          facts={facts}
          links={links}
          allClaims={claims}
          onClose={() => setSelectedClaim(null)}
        />
      )}
    </div>
  );
}
