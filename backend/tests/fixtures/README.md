# CLI output fixtures for adapter parsing tests

Real samples (captured on 2026-09-30/10-01 on the development machine):

- `codex_sample.jsonl`: `codex exec --skip-git-repo-check --ephemeral --json --sandbox read-only -m gpt-5.5 -c model_reasoning_effort=low -` (codex-cli 0.145.0), prompt "Reply with the single word OK".
- `codex_sample_scout_gpt-5.6-sol.jsonl`: the same prompt with `-m gpt-5.6-sol` (earlier probe).
- `codex_failure_sample.jsonl`: the same command without `-m`; the config default model `gpt-6-astra` is rejected by codex-cli 0.145.0 (HTTP 400, exit 1).
- `claude_auth_error_sample.json`: `claude -p --output-format json --max-turns 1 --model sonnet --effort low --tools ""` (claude 2.1.212) with an expired OAuth session (`is_error: true`, exit 1).
- `gemini_auth_error_stderr.txt`: gemini 0.26.0 stderr when the account needs `GOOGLE_CLOUD_PROJECT` (exit 1, no JSON).

Synthetic samples (no successful call was possible on this machine; structure follows the CLI source/docs):

- `claude_success_synthetic.json`: Claude Code `--output-format json` result object (`result`, `usage`, `total_cost_usd`, `modelUsage`).
- `gemini_success_synthetic.json`: gemini-cli `-o json` output (`JsonFormatter.format` + `UiTelemetryService` metrics in gemini-cli-core 0.26.0).
- `gemini_error_synthetic.json`: gemini-cli `JsonFormatter.formatError` output.
