import { useState, useCallback } from "react";
import { Save, Upload, Code, GitFork } from "lucide-react";
import { useApi } from "../hooks/useApi";
import { listLoops, saveLoop, type LoopDefinition } from "../api/client";
import GraphEditor from "../components/GraphEditor";

export default function Loops() {
  const { data: loops, loading, refresh } = useApi(listLoops);
  const [selected, setSelected] = useState<LoopDefinition | null>(null);
  const [editMode, setEditMode] = useState<"graph" | "yaml">("graph");
  const [yamlContent, setYamlContent] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSelect = useCallback((loop: LoopDefinition) => {
    setSelected(loop);
    setYamlContent(JSON.stringify(loop, null, 2));
    setError(null);
  }, []);

  const handleGraphChange = useCallback((updated: LoopDefinition) => {
    setSelected(updated);
    setYamlContent(JSON.stringify(updated, null, 2));
  }, []);

  const handleSave = useCallback(async () => {
    if (!selected) return;
    setSaving(true);
    setError(null);
    try {
      let toSave = selected;
      if (editMode === "yaml") {
        toSave = JSON.parse(yamlContent) as LoopDefinition;
      }
      await saveLoop(toSave);
      refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  }, [selected, editMode, yamlContent, refresh]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">Loop Editor</h2>
        <div className="flex items-center gap-3">
          {/* View toggle */}
          <div className="flex rounded-lg border border-surface-border overflow-hidden">
            <button
              onClick={() => setEditMode("graph")}
              className={`flex items-center gap-1.5 px-3 py-2 text-xs font-medium transition-colors ${
                editMode === "graph"
                  ? "bg-accent text-white"
                  : "bg-white text-gray-600 hover:bg-gray-50"
              }`}
            >
              <GitFork className="w-3.5 h-3.5" />
              Graph
            </button>
            <button
              onClick={() => setEditMode("yaml")}
              className={`flex items-center gap-1.5 px-3 py-2 text-xs font-medium transition-colors ${
                editMode === "yaml"
                  ? "bg-accent text-white"
                  : "bg-white text-gray-600 hover:bg-gray-50"
              }`}
            >
              <Code className="w-3.5 h-3.5" />
              DSL
            </button>
          </div>
          <button
            onClick={handleSave}
            disabled={!selected || saving}
            className="flex items-center gap-2 px-4 py-2 bg-accent text-white text-sm font-medium rounded-lg hover:bg-accent-dark disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Save className="w-4 h-4" />
            {saving ? "Saving..." : "Save"}
          </button>
        </div>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 text-sm rounded-lg">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Loop list */}
        <div className="lg:col-span-1">
          <div className="bg-white rounded-xl border border-surface-border overflow-hidden">
            <div className="px-4 py-3 border-b border-surface-border bg-gray-50">
              <h3 className="text-sm font-semibold text-gray-700">
                Presets & Custom
              </h3>
            </div>
            <div className="divide-y divide-surface-border">
              {loading ? (
                <div className="p-4 text-sm text-gray-400">Loading...</div>
              ) : (loops ?? []).length === 0 ? (
                <div className="p-4 text-sm text-gray-400">No loops defined.</div>
              ) : (
                (loops ?? []).map((loop) => (
                  <button
                    key={loop.name}
                    onClick={() => handleSelect(loop)}
                    className={`w-full text-left px-4 py-3 hover:bg-gray-50 transition-colors ${
                      selected?.name === loop.name ? "bg-blue-50 border-l-2 border-accent" : ""
                    }`}
                  >
                    <div className="text-sm font-medium text-gray-800">
                      {loop.name}
                    </div>
                    <div className="text-xs text-gray-500 mt-0.5">
                      {loop.nodes.length} nodes / v{loop.version}
                    </div>
                    {loop.description && (
                      <div className="text-xs text-gray-400 mt-0.5 truncate">
                        {loop.description}
                      </div>
                    )}
                  </button>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Editor */}
        <div className="lg:col-span-3">
          {selected ? (
            editMode === "graph" ? (
              <GraphEditor loop={selected} onChange={handleGraphChange} />
            ) : (
              <div className="space-y-3">
                <textarea
                  value={yamlContent}
                  onChange={(e) => setYamlContent(e.target.value)}
                  className="w-full h-[550px] px-4 py-3 font-mono text-sm border border-surface-border rounded-lg focus:ring-2 focus:ring-accent focus:border-transparent outline-none resize-y bg-gray-50"
                  spellCheck={false}
                />
                <div className="flex items-center gap-2 text-xs text-gray-500">
                  <Upload className="w-3.5 h-3.5" />
                  Edit the JSON definition directly. Changes are applied on save.
                </div>
              </div>
            )
          ) : (
            <div className="bg-white rounded-xl border border-surface-border p-12 text-center text-gray-400">
              Select a loop preset to edit it.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
