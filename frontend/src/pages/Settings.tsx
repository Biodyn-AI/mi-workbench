import { useState, useCallback, useEffect } from "react";
import { Save, Plus, Trash2 } from "lucide-react";
import { useApi } from "../hooks/useApi";
import {
  listWorkspaces,
  getWorkspace,
  updateSettings,
  createWorkspace,
  deleteWorkspace,
  type WorkspaceConfig,
  type WorkspaceCreate,
  type GitMode,
  type ProviderName,
} from "../api/client";

const GIT_MODES: { value: GitMode; label: string }[] = [
  { value: "none", label: "None" },
  { value: "commit_per_run", label: "Commit per run" },
  { value: "commit_per_iteration", label: "Commit per iteration" },
  { value: "branch_per_run", label: "Branch per run" },
];

const PROVIDERS: { value: ProviderName; label: string }[] = [
  { value: "claude_code", label: "Claude Code" },
  { value: "codex_cli", label: "Codex CLI" },
  { value: "gemini_cli", label: "Gemini CLI" },
  { value: "mock", label: "Mock" },
];

export default function Settings() {
  const { data: workspaces, refresh: refreshWorkspaces } = useApi(listWorkspaces);
  const [selectedWsId, setSelectedWsId] = useState<string>("");
  const [config, setConfig] = useState<WorkspaceConfig | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [showNewWs, setShowNewWs] = useState(false);

  // Auto-select first workspace
  useEffect(() => {
    if (workspaces?.length && !selectedWsId) {
      setSelectedWsId(workspaces[0].id);
    }
  }, [workspaces, selectedWsId]);

  // Load workspace config when selection changes
  useEffect(() => {
    if (!selectedWsId) return;
    getWorkspace(selectedWsId)
      .then(setConfig)
      .catch(() => setConfig(null));
  }, [selectedWsId]);

  const handleSave = useCallback(async () => {
    if (!config) return;
    setSaving(true);
    setError(null);
    setSuccess(false);
    try {
      await updateSettings(config);
      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  }, [config]);

  const handleDelete = useCallback(async () => {
    if (!selectedWsId) return;
    try {
      await deleteWorkspace(selectedWsId);
      setSelectedWsId("");
      setConfig(null);
      refreshWorkspaces();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed");
    }
  }, [selectedWsId, refreshWorkspaces]);

  const handleCreateWorkspace = useCallback(
    async (data: WorkspaceCreate) => {
      try {
        const ws = await createWorkspace(data);
        setSelectedWsId(ws.id);
        setShowNewWs(false);
        refreshWorkspaces();
      } catch (err) {
        setError(err instanceof Error ? err.message : "Create failed");
      }
    },
    [refreshWorkspaces],
  );

  return (
    <div className="space-y-6 max-w-3xl">
      <h2 className="text-2xl font-bold text-gray-900">Settings</h2>

      {/* Workspace selector */}
      <div className="bg-white rounded-xl border border-surface-border p-6 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-gray-800">Workspace</h3>
          <button
            onClick={() => setShowNewWs(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium border border-surface-border rounded-lg hover:bg-gray-50 transition-colors"
          >
            <Plus className="w-4 h-4" />
            New
          </button>
        </div>
        <select
          value={selectedWsId}
          onChange={(e) => setSelectedWsId(e.target.value)}
          className="w-full px-3 py-2 text-sm border border-surface-border rounded-lg focus:ring-2 focus:ring-accent focus:border-transparent outline-none"
        >
          <option value="">Select a workspace...</option>
          {workspaces?.map((ws) => (
            <option key={ws.id} value={ws.id}>
              {ws.name} ({ws.path})
            </option>
          ))}
        </select>
      </div>

      {config && (
        <>
          {/* Budget controls */}
          <div className="bg-white rounded-xl border border-surface-border p-6 space-y-4">
            <h3 className="font-semibold text-gray-800">Budget Controls</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Max Tokens per Run
                </label>
                <input
                  type="number"
                  value={config.budget_max_tokens_per_run}
                  onChange={(e) =>
                    setConfig({
                      ...config,
                      budget_max_tokens_per_run: parseInt(e.target.value) || 0,
                    })
                  }
                  className="w-full px-3 py-2 text-sm border border-surface-border rounded-lg focus:ring-2 focus:ring-accent focus:border-transparent outline-none"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Max Cost per Run ($)
                </label>
                <input
                  type="number"
                  step="0.01"
                  value={config.budget_max_cost_per_run}
                  onChange={(e) =>
                    setConfig({
                      ...config,
                      budget_max_cost_per_run: parseFloat(e.target.value) || 0,
                    })
                  }
                  className="w-full px-3 py-2 text-sm border border-surface-border rounded-lg focus:ring-2 focus:ring-accent focus:border-transparent outline-none"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Max Iterations
                </label>
                <input
                  type="number"
                  value={config.budget_max_iterations}
                  onChange={(e) =>
                    setConfig({
                      ...config,
                      budget_max_iterations: parseInt(e.target.value) || 0,
                    })
                  }
                  className="w-full px-3 py-2 text-sm border border-surface-border rounded-lg focus:ring-2 focus:ring-accent focus:border-transparent outline-none"
                />
              </div>
            </div>
          </div>

          {/* Git mode */}
          <div className="bg-white rounded-xl border border-surface-border p-6 space-y-4">
            <h3 className="font-semibold text-gray-800">Git Mode</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {GIT_MODES.map(({ value, label }) => (
                <button
                  key={value}
                  onClick={() => setConfig({ ...config, git_mode: value })}
                  className={`px-4 py-3 rounded-lg border-2 text-sm font-medium text-center transition-colors ${
                    config.git_mode === value
                      ? "border-accent bg-blue-50 text-accent"
                      : "border-surface-border text-gray-600 hover:border-gray-300"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          {/* Provider & Safety */}
          <div className="bg-white rounded-xl border border-surface-border p-6 space-y-4">
            <h3 className="font-semibold text-gray-800">Provider & Safety</h3>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Default Provider
              </label>
              <select
                value={config.default_provider}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    default_provider: e.target.value as ProviderName,
                  })
                }
                className="w-full px-3 py-2 text-sm border border-surface-border rounded-lg focus:ring-2 focus:ring-accent focus:border-transparent outline-none"
              >
                {PROVIDERS.map(({ value, label }) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Enabled Providers
              </label>
              <div className="flex flex-wrap gap-3">
                {PROVIDERS.map(({ value, label }) => {
                  const enabled = config.providers_enabled.includes(value);
                  return (
                    <label
                      key={value}
                      className="flex items-center gap-2 cursor-pointer"
                    >
                      <input
                        type="checkbox"
                        checked={enabled}
                        onChange={(e) => {
                          const next = e.target.checked
                            ? [...config.providers_enabled, value]
                            : config.providers_enabled.filter(
                                (p) => p !== value,
                              );
                          setConfig({ ...config, providers_enabled: next });
                        }}
                        className="rounded border-gray-300 text-accent focus:ring-accent"
                      />
                      <span className="text-sm text-gray-700">{label}</span>
                    </label>
                  );
                })}
              </div>
            </div>
            <div className="flex items-center justify-between pt-2 border-t border-surface-border">
              <div>
                <p className="text-sm font-medium text-gray-700">
                  Deny Destructive Commands
                </p>
                <p className="text-xs text-gray-500">
                  Prevent rm -rf, force pushes, and other destructive operations
                </p>
              </div>
              <button
                onClick={() =>
                  setConfig({
                    ...config,
                    safety_deny_destructive_commands:
                      !config.safety_deny_destructive_commands,
                  })
                }
                className={`relative w-11 h-6 rounded-full transition-colors ${
                  config.safety_deny_destructive_commands
                    ? "bg-accent"
                    : "bg-gray-300"
                }`}
              >
                <span
                  className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform ${
                    config.safety_deny_destructive_commands
                      ? "translate-x-5"
                      : ""
                  }`}
                />
              </button>
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-3">
            <button
              onClick={handleSave}
              disabled={saving}
              className="flex items-center gap-2 px-4 py-2 bg-accent text-white text-sm font-medium rounded-lg hover:bg-accent-dark disabled:opacity-50 transition-colors"
            >
              <Save className="w-4 h-4" />
              {saving ? "Saving..." : "Save Settings"}
            </button>
            <button
              onClick={handleDelete}
              className="flex items-center gap-2 px-4 py-2 text-red-600 border border-red-200 text-sm font-medium rounded-lg hover:bg-red-50 transition-colors"
            >
              <Trash2 className="w-4 h-4" />
              Delete Workspace
            </button>
            {success && (
              <span className="text-sm text-green-600">Saved successfully.</span>
            )}
            {error && <span className="text-sm text-red-600">{error}</span>}
          </div>
        </>
      )}

      {/* New workspace dialog */}
      {showNewWs && (
        <NewWorkspaceDialog
          onClose={() => setShowNewWs(false)}
          onCreate={handleCreateWorkspace}
        />
      )}
    </div>
  );
}

function NewWorkspaceDialog({
  onClose,
  onCreate,
}: {
  onClose: () => void;
  onCreate: (data: WorkspaceCreate) => void;
}) {
  const [name, setName] = useState("");
  const [path, setPath] = useState("");

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-md mx-4 p-6 space-y-4">
        <h3 className="font-semibold text-gray-800">New Workspace</h3>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Name
          </label>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="my-research"
            className="w-full px-3 py-2 text-sm border border-surface-border rounded-lg focus:ring-2 focus:ring-accent focus:border-transparent outline-none"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Path
          </label>
          <input
            value={path}
            onChange={(e) => setPath(e.target.value)}
            placeholder="/path/to/workspace"
            className="w-full px-3 py-2 text-sm border border-surface-border rounded-lg focus:ring-2 focus:ring-accent focus:border-transparent outline-none"
          />
        </div>
        <div className="flex justify-end gap-3 pt-2">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm text-gray-600 hover:bg-gray-50 rounded-lg transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={() => onCreate({ name, path })}
            disabled={!name.trim() || !path.trim()}
            className="px-4 py-2 bg-accent text-white text-sm font-medium rounded-lg hover:bg-accent-dark disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Create
          </button>
        </div>
      </div>
    </div>
  );
}
