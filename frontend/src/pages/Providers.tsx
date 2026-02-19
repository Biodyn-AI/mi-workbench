import { useState, useCallback } from "react";
import {
  Server,
  CheckCircle,
  XCircle,
  Loader,
  Zap,
} from "lucide-react";
import { useApi } from "../hooks/useApi";
import {
  listProviders,
  smoketestProvider,
  type ProviderInfo,
  type ProviderName,
  type SmokeTestResult,
} from "../api/client";

const providerMeta: Record<
  ProviderName,
  { label: string; description: string; color: string }
> = {
  claude_code: {
    label: "Claude Code",
    description: "Anthropic Claude via claude CLI with code execution capabilities",
    color: "bg-indigo-500",
  },
  codex_cli: {
    label: "Codex CLI",
    description: "OpenAI Codex CLI for code generation and editing",
    color: "bg-green-500",
  },
  gemini_cli: {
    label: "Gemini CLI",
    description: "Google Gemini CLI for code understanding and generation",
    color: "bg-blue-500",
  },
  mock: {
    label: "Mock Provider",
    description: "Mock provider for testing and development",
    color: "bg-gray-500",
  },
};

export default function Providers() {
  const { data: providers, loading } = useApi(listProviders);
  const [testResults, setTestResults] = useState<
    Record<string, { loading: boolean; result?: SmokeTestResult }>
  >({});

  const handleSmokeTest = useCallback(async (name: ProviderName) => {
    setTestResults((prev) => ({
      ...prev,
      [name]: { loading: true },
    }));
    try {
      const result = await smoketestProvider(name);
      setTestResults((prev) => ({
        ...prev,
        [name]: { loading: false, result },
      }));
    } catch (err) {
      setTestResults((prev) => ({
        ...prev,
        [name]: {
          loading: false,
          result: {
            success: false,
            message: err instanceof Error ? err.message : "Test failed",
            duration_ms: 0,
          },
        },
      }));
    }
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">Providers</h2>
      </div>

      {loading ? (
        <div className="text-center text-gray-400 py-12">Loading providers...</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {(providers ?? []).map((provider) => (
            <ProviderCard
              key={provider.name}
              provider={provider}
              testState={testResults[provider.name]}
              onTest={handleSmokeTest}
            />
          ))}
          {/* Show known providers even if not returned from API */}
          {(Object.keys(providerMeta) as ProviderName[])
            .filter((name) => !(providers ?? []).find((p) => p.name === name))
            .map((name) => (
              <ProviderCard
                key={name}
                provider={{
                  name,
                  available: false,
                  default_model: "",
                  models: [],
                }}
                testState={testResults[name]}
                onTest={handleSmokeTest}
              />
            ))}
        </div>
      )}
    </div>
  );
}

function ProviderCard({
  provider,
  testState,
  onTest,
}: {
  provider: ProviderInfo;
  testState?: { loading: boolean; result?: SmokeTestResult };
  onTest: (name: ProviderName) => void;
}) {
  const meta = providerMeta[provider.name] ?? {
    label: provider.name,
    description: "",
    color: "bg-gray-500",
  };

  return (
    <div className="bg-white rounded-xl border border-surface-border p-6 space-y-4">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className={`p-2.5 rounded-lg ${meta.color} bg-opacity-10`}>
            <Server className={`w-5 h-5 ${meta.color.replace("bg-", "text-")}`} />
          </div>
          <div>
            <h3 className="font-semibold text-gray-800">{meta.label}</h3>
            <p className="text-xs text-gray-500 mt-0.5">{meta.description}</p>
          </div>
        </div>
        <StatusIndicator available={provider.available} />
      </div>

      {/* Model info */}
      {provider.default_model && (
        <div className="text-sm">
          <span className="text-gray-500">Default model: </span>
          <span className="font-mono text-gray-700">{provider.default_model}</span>
        </div>
      )}
      {provider.models.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {provider.models.map((m) => (
            <span
              key={m}
              className="px-2 py-0.5 bg-gray-100 text-gray-600 text-xs rounded font-mono"
            >
              {m}
            </span>
          ))}
        </div>
      )}

      {/* Smoke test */}
      <div className="pt-2 border-t border-surface-border">
        <button
          onClick={() => onTest(provider.name)}
          disabled={testState?.loading}
          className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium border border-surface-border rounded-lg hover:bg-gray-50 disabled:opacity-50 transition-colors"
        >
          {testState?.loading ? (
            <Loader className="w-4 h-4 animate-spin text-gray-400" />
          ) : (
            <Zap className="w-4 h-4 text-amber-500" />
          )}
          {testState?.loading ? "Testing..." : "Run Smoke Test"}
        </button>
        {testState?.result && (
          <div
            className={`mt-3 p-3 rounded-lg text-sm ${
              testState.result.success
                ? "bg-green-50 text-green-700"
                : "bg-red-50 text-red-700"
            }`}
          >
            <div className="flex items-center gap-2 font-medium">
              {testState.result.success ? (
                <CheckCircle className="w-4 h-4" />
              ) : (
                <XCircle className="w-4 h-4" />
              )}
              {testState.result.success ? "Passed" : "Failed"}
              {testState.result.duration_ms > 0 && (
                <span className="font-normal text-xs ml-auto">
                  {testState.result.duration_ms}ms
                </span>
              )}
            </div>
            <p className="mt-1 text-xs">{testState.result.message}</p>
          </div>
        )}
      </div>
    </div>
  );
}

function StatusIndicator({ available }: { available: boolean }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium ${
        available
          ? "bg-green-100 text-green-700"
          : "bg-gray-100 text-gray-500"
      }`}
    >
      <span
        className={`w-1.5 h-1.5 rounded-full ${
          available ? "bg-green-500" : "bg-gray-400"
        }`}
      />
      {available ? "Available" : "Unavailable"}
    </span>
  );
}
