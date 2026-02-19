import { useState, useCallback } from "react";
import { Save, CheckCircle, AlertCircle, RefreshCw } from "lucide-react";
import type { PromptTemplate, PromptUpdate, PromptValidationResult } from "../api/client";
import { updatePrompt, validatePrompt } from "../api/client";

interface Props {
  prompt: PromptTemplate;
  onSaved?: (updated: PromptTemplate) => void;
}

export default function PromptEditor({ prompt, onSaved }: Props) {
  const [systemPrompt, setSystemPrompt] = useState(prompt.system_prompt);
  const [devPrompt, setDevPrompt] = useState(prompt.developer_prompt);
  const [changelogEntry, setChangelogEntry] = useState("");
  const [saving, setSaving] = useState(false);
  const [validation, setValidation] = useState<PromptValidationResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const hasChanges =
    systemPrompt !== prompt.system_prompt || devPrompt !== prompt.developer_prompt;

  const highlightVariables = (text: string): React.ReactNode[] => {
    const parts = text.split(/({{[^}]+}})/g);
    return parts.map((part, i) =>
      part.startsWith("{{") ? (
        <span key={i} className="bg-amber-100 text-amber-800 rounded px-0.5">
          {part}
        </span>
      ) : (
        <span key={i}>{part}</span>
      ),
    );
  };

  const handleSave = useCallback(async () => {
    setSaving(true);
    setError(null);
    try {
      const data: PromptUpdate = {};
      if (systemPrompt !== prompt.system_prompt) data.system_prompt = systemPrompt;
      if (devPrompt !== prompt.developer_prompt) data.developer_prompt = devPrompt;
      if (changelogEntry) data.changelog_entry = changelogEntry;

      const updated = await updatePrompt(prompt.metadata.role, data);
      setChangelogEntry("");
      onSaved?.(updated);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  }, [systemPrompt, devPrompt, changelogEntry, prompt, onSaved]);

  const handleValidate = useCallback(async () => {
    try {
      const result = await validatePrompt(prompt.metadata.role);
      setValidation(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Validation failed");
    }
  }, [prompt.metadata.role]);

  return (
    <div className="space-y-4">
      {/* Metadata bar */}
      <div className="flex items-center gap-4 text-sm">
        <span className="font-medium text-gray-800">
          {prompt.metadata.name}
        </span>
        <span className="px-2 py-0.5 bg-indigo-100 text-indigo-700 rounded text-xs font-medium">
          {prompt.metadata.role}
        </span>
        <span className="text-gray-400">v{prompt.metadata.version}</span>
        {prompt.metadata.tags.map((tag) => (
          <span
            key={tag}
            className="px-2 py-0.5 bg-gray-100 text-gray-600 rounded text-xs"
          >
            {tag}
          </span>
        ))}
      </div>

      {/* System prompt editor */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          System Prompt
        </label>
        <textarea
          value={systemPrompt}
          onChange={(e) => setSystemPrompt(e.target.value)}
          className="w-full h-64 px-4 py-3 font-mono text-sm border border-surface-border rounded-lg focus:ring-2 focus:ring-accent focus:border-transparent outline-none resize-y bg-gray-50"
          spellCheck={false}
        />
        {/* Variable preview */}
        {systemPrompt.includes("{{") && (
          <div className="mt-1 text-xs text-gray-500">
            Variables: {highlightVariables(systemPrompt.match(/{{[^}]+}}/g)?.join(", ") || "")}
          </div>
        )}
      </div>

      {/* Developer prompt editor */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Developer Prompt
        </label>
        <textarea
          value={devPrompt}
          onChange={(e) => setDevPrompt(e.target.value)}
          className="w-full h-32 px-4 py-3 font-mono text-sm border border-surface-border rounded-lg focus:ring-2 focus:ring-accent focus:border-transparent outline-none resize-y bg-gray-50"
          spellCheck={false}
        />
      </div>

      {/* Changelog entry */}
      {hasChanges && (
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Changelog Entry
          </label>
          <input
            type="text"
            value={changelogEntry}
            onChange={(e) => setChangelogEntry(e.target.value)}
            placeholder="Describe your changes..."
            className="w-full px-3 py-2 text-sm border border-surface-border rounded-lg focus:ring-2 focus:ring-accent focus:border-transparent outline-none"
          />
        </div>
      )}

      {/* Validation feedback */}
      {validation && (
        <div
          className={`p-3 rounded-lg text-sm ${
            validation.valid
              ? "bg-green-50 text-green-700"
              : "bg-red-50 text-red-700"
          }`}
        >
          <div className="flex items-center gap-2 font-medium mb-1">
            {validation.valid ? (
              <CheckCircle className="w-4 h-4" />
            ) : (
              <AlertCircle className="w-4 h-4" />
            )}
            {validation.valid ? "Valid" : "Invalid"}
          </div>
          {validation.errors.map((e, i) => (
            <div key={i} className="ml-6">
              {e}
            </div>
          ))}
          {validation.warnings.map((w, i) => (
            <div key={i} className="ml-6 text-yellow-700">
              {w}
            </div>
          ))}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="p-3 rounded-lg bg-red-50 text-red-700 text-sm">
          {error}
        </div>
      )}

      {/* Actions */}
      <div className="flex items-center gap-3">
        <button
          onClick={handleSave}
          disabled={!hasChanges || saving}
          className="flex items-center gap-2 px-4 py-2 bg-accent text-white text-sm font-medium rounded-lg hover:bg-accent-dark disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <Save className="w-4 h-4" />
          {saving ? "Saving..." : "Save"}
        </button>
        <button
          onClick={handleValidate}
          className="flex items-center gap-2 px-4 py-2 border border-surface-border text-gray-700 text-sm font-medium rounded-lg hover:bg-gray-50 transition-colors"
        >
          <RefreshCw className="w-4 h-4" />
          Validate
        </button>
      </div>
    </div>
  );
}
