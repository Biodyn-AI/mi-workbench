import { useState, useCallback } from "react";
import { Tag, History } from "lucide-react";
import { useApi } from "../hooks/useApi";
import { listPrompts, type PromptTemplate } from "../api/client";
import PromptEditor from "../components/PromptEditor";
import DiffViewer from "../components/DiffViewer";

export default function Prompts() {
  const { data: prompts, loading, refresh } = useApi(listPrompts);
  const [selected, setSelected] = useState<PromptTemplate | null>(null);
  const [showDiff, setShowDiff] = useState(false);
  const [prevVersion, setPrevVersion] = useState<string>("");

  const handleSelect = useCallback((prompt: PromptTemplate) => {
    // Store previous version for diff comparison
    if (selected && selected.metadata.role === prompt.metadata.role) {
      setPrevVersion(selected.system_prompt);
    }
    setSelected(prompt);
    setShowDiff(false);
  }, [selected]);

  const handleSaved = useCallback(
    (updated: PromptTemplate) => {
      if (selected) {
        setPrevVersion(selected.system_prompt);
      }
      setSelected(updated);
      refresh();
    },
    [selected, refresh],
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">Prompt Registry</h2>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Role list */}
        <div className="lg:col-span-1">
          <div className="bg-white rounded-xl border border-surface-border overflow-hidden">
            <div className="px-4 py-3 border-b border-surface-border bg-gray-50">
              <h3 className="text-sm font-semibold text-gray-700">Roles</h3>
            </div>
            <div className="divide-y divide-surface-border">
              {loading ? (
                <div className="p-4 text-sm text-gray-400">Loading...</div>
              ) : (prompts ?? []).length === 0 ? (
                <div className="p-4 text-sm text-gray-400">No prompts found.</div>
              ) : (
                (prompts ?? []).map((p) => (
                  <button
                    key={p.metadata.role}
                    onClick={() => handleSelect(p)}
                    className={`w-full text-left px-4 py-3 hover:bg-gray-50 transition-colors ${
                      selected?.metadata.role === p.metadata.role
                        ? "bg-blue-50 border-l-2 border-accent"
                        : ""
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-gray-800">
                        {p.metadata.name}
                      </span>
                      <span className="px-1.5 py-0.5 bg-indigo-100 text-indigo-600 text-xs rounded font-mono">
                        v{p.metadata.version}
                      </span>
                    </div>
                    <div className="text-xs text-gray-500 mt-0.5">
                      {p.metadata.role}
                    </div>
                    {p.metadata.tags.length > 0 && (
                      <div className="flex gap-1 mt-1.5">
                        {p.metadata.tags.slice(0, 3).map((tag) => (
                          <span
                            key={tag}
                            className="inline-flex items-center gap-0.5 px-1.5 py-0.5 bg-gray-100 text-gray-500 text-xs rounded"
                          >
                            <Tag className="w-2.5 h-2.5" />
                            {tag}
                          </span>
                        ))}
                      </div>
                    )}
                  </button>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Editor panel */}
        <div className="lg:col-span-3 space-y-4">
          {selected ? (
            <>
              {/* Changelog */}
              {selected.metadata.changelog.length > 0 && (
                <div className="bg-white rounded-xl border border-surface-border p-4">
                  <button
                    onClick={() => setShowDiff(!showDiff)}
                    className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-800 transition-colors"
                  >
                    <History className="w-4 h-4" />
                    Changelog ({selected.metadata.changelog.length} entries)
                  </button>
                  {showDiff && (
                    <div className="mt-3 space-y-2">
                      {selected.metadata.changelog.map((entry, i) => (
                        <div
                          key={i}
                          className="text-xs text-gray-500 pl-6 border-l-2 border-gray-200 py-0.5"
                        >
                          {entry}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Diff view */}
              {prevVersion && prevVersion !== selected.system_prompt && (
                <DiffViewer
                  oldContent={prevVersion}
                  newContent={selected.system_prompt}
                  oldLabel="Previous"
                  newLabel="Current"
                />
              )}

              {/* Editor */}
              <div className="bg-white rounded-xl border border-surface-border p-6">
                <PromptEditor prompt={selected} onSaved={handleSaved} />
              </div>

              {/* Variables & Output schema */}
              {(selected.variables.length > 0 ||
                selected.output_schema.length > 0) && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {selected.variables.length > 0 && (
                    <div className="bg-white rounded-xl border border-surface-border p-4">
                      <h4 className="text-sm font-semibold text-gray-700 mb-3">
                        Variables
                      </h4>
                      <div className="space-y-2">
                        {selected.variables.map((v) => (
                          <div key={v.name} className="text-sm">
                            <span className="font-mono text-amber-700">
                              {"{{"}
                              {v.name}
                              {"}}"}
                            </span>
                            {v.required && (
                              <span className="text-red-400 ml-1">*</span>
                            )}
                            <p className="text-xs text-gray-500">
                              {v.description}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  {selected.output_schema.length > 0 && (
                    <div className="bg-white rounded-xl border border-surface-border p-4">
                      <h4 className="text-sm font-semibold text-gray-700 mb-3">
                        Output Schema
                      </h4>
                      <div className="space-y-2">
                        {selected.output_schema.map((f) => (
                          <div key={f.name} className="text-sm">
                            <span className="font-mono text-blue-700">
                              {f.name}
                            </span>
                            {f.required && (
                              <span className="text-red-400 ml-1">*</span>
                            )}
                            <p className="text-xs text-gray-500">
                              {f.description}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </>
          ) : (
            <div className="bg-white rounded-xl border border-surface-border p-12 text-center text-gray-400">
              Select a prompt role to view and edit.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
