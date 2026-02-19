"""Adapter registry — maps ProviderName to adapter instances."""
from backend.models import ProviderName
from backend.adapters.base import BaseAdapter
from backend.adapters.mock import MockAdapter
from backend.adapters.claude_code import ClaudeCodeAdapter
from backend.adapters.codex_cli import CodexCliAdapter
from backend.adapters.gemini_cli import GeminiCliAdapter


_ADAPTERS: dict[ProviderName, BaseAdapter] = {
    ProviderName.MOCK: MockAdapter(),
    ProviderName.CLAUDE_CODE: ClaudeCodeAdapter(),
    ProviderName.CODEX_CLI: CodexCliAdapter(),
    ProviderName.GEMINI_CLI: GeminiCliAdapter(),
}


def get_adapter(provider_name: ProviderName) -> BaseAdapter:
    """Return the adapter instance for a given provider name."""
    adapter = _ADAPTERS.get(provider_name)
    if adapter is None:
        raise ValueError(f"Unknown provider: {provider_name}")
    return adapter


def list_adapters() -> dict[str, bool]:
    """Return a dict of adapter name -> availability status."""
    return {name.value: adapter.is_available() for name, adapter in _ADAPTERS.items()}
