import { useState, useCallback } from "react";
import {
  FolderOpen,
  FileText,
  ChevronRight,
  Filter,
} from "lucide-react";
import { useApi } from "../hooks/useApi";
import {
  listWorkspaces,
  browseArtifacts,
  getArtifact,
  type ArtifactInfo,
} from "../api/client";
import ArtifactViewer from "../components/ArtifactViewer";

const ARTIFACT_TYPES = ["ALL", "MECH", "EVAL", "XP", "METHOD"];

export default function Artifacts() {
  const { data: workspaces } = useApi(listWorkspaces);
  const [wsId, setWsId] = useState("");
  const [typeFilter, setTypeFilter] = useState("ALL");
  const [selectedArtifact, setSelectedArtifact] = useState<ArtifactInfo | null>(null);
  const [content, setContent] = useState<string | null>(null);
  const [contentLoading, setContentLoading] = useState(false);

  const effectiveWsId = wsId || workspaces?.[0]?.id || "";

  const { data: artifacts, loading } = useApi(
    () =>
      effectiveWsId
        ? browseArtifacts(
            effectiveWsId,
            undefined,
            typeFilter === "ALL" ? undefined : typeFilter,
          )
        : Promise.resolve([]),
    [effectiveWsId, typeFilter],
  );

  const handleSelect = useCallback(async (artifact: ArtifactInfo) => {
    setSelectedArtifact(artifact);
    setContentLoading(true);
    try {
      const text = await getArtifact(artifact.path);
      setContent(text);
    } catch {
      setContent("(Failed to load artifact content)");
    } finally {
      setContentLoading(false);
    }
  }, []);

  // Group artifacts by run_id
  const grouped = (artifacts ?? []).reduce<Record<string, ArtifactInfo[]>>(
    (acc, a) => {
      const key = a.run_id ?? "unassigned";
      (acc[key] ??= []).push(a);
      return acc;
    },
    {},
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">Artifacts</h2>
        <div className="flex items-center gap-3">
          <select
            value={wsId}
            onChange={(e) => setWsId(e.target.value)}
            className="px-3 py-2 text-sm border border-surface-border rounded-lg bg-white focus:ring-2 focus:ring-accent focus:border-transparent outline-none"
          >
            {workspaces?.map((ws) => (
              <option key={ws.id} value={ws.id}>
                {ws.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Type filter */}
      <div className="flex items-center gap-2">
        <Filter className="w-4 h-4 text-gray-400" />
        <div className="flex rounded-lg border border-surface-border overflow-hidden">
          {ARTIFACT_TYPES.map((t) => (
            <button
              key={t}
              onClick={() => setTypeFilter(t)}
              className={`px-3 py-1.5 text-xs font-medium transition-colors ${
                typeFilter === t
                  ? "bg-accent text-white"
                  : "bg-white text-gray-600 hover:bg-gray-50"
              }`}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      {/* Split view */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* File tree */}
        <div className="lg:col-span-1 bg-white rounded-xl border border-surface-border overflow-hidden">
          <div className="px-4 py-3 border-b border-surface-border bg-gray-50">
            <h3 className="text-sm font-semibold text-gray-700">File Tree</h3>
          </div>
          <div className="overflow-y-auto max-h-[600px]">
            {loading ? (
              <div className="p-4 text-sm text-gray-400">Loading...</div>
            ) : Object.keys(grouped).length === 0 ? (
              <div className="p-4 text-sm text-gray-400">No artifacts found.</div>
            ) : (
              Object.entries(grouped).map(([runId, items]) => (
                <div key={runId}>
                  <div className="flex items-center gap-2 px-4 py-2 bg-gray-50 border-b border-surface-border">
                    <FolderOpen className="w-4 h-4 text-amber-500" />
                    <span className="text-xs font-medium text-gray-600 font-mono">
                      {runId}
                    </span>
                    <span className="text-xs text-gray-400 ml-auto">
                      {items.length}
                    </span>
                  </div>
                  {items.map((a) => (
                    <button
                      key={a.path}
                      onClick={() => handleSelect(a)}
                      className={`w-full flex items-center gap-2 px-6 py-2 text-left hover:bg-blue-50 transition-colors ${
                        selectedArtifact?.path === a.path
                          ? "bg-blue-50 border-l-2 border-accent"
                          : ""
                      }`}
                    >
                      <FileText className="w-3.5 h-3.5 text-gray-400 flex-shrink-0" />
                      <span className="text-sm text-gray-700 truncate">
                        {a.name}
                      </span>
                      <ChevronRight className="w-3 h-3 text-gray-300 ml-auto flex-shrink-0" />
                    </button>
                  ))}
                </div>
              ))
            )}
          </div>
        </div>

        {/* Viewer */}
        <div className="lg:col-span-2">
          {contentLoading ? (
            <div className="bg-white rounded-xl border border-surface-border p-8 text-center text-gray-400">
              Loading content...
            </div>
          ) : selectedArtifact && content !== null ? (
            <ArtifactViewer
              content={content}
              name={selectedArtifact.name}
              artifactType={selectedArtifact.artifact_type}
              runId={selectedArtifact.run_id}
              iteration={selectedArtifact.iteration}
            />
          ) : (
            <div className="bg-white rounded-xl border border-surface-border p-8 text-center text-gray-400">
              Select an artifact to view its contents.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
