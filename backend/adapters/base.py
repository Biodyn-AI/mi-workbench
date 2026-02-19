"""Abstract base adapter interface for MI-Workbench providers."""
from abc import ABC, abstractmethod

from backend.models import AdapterRunRequest, AdapterRunResult


class BaseAdapter(ABC):
    name: str

    @abstractmethod
    async def run(self, request: AdapterRunRequest) -> AdapterRunResult:
        """Execute a prompt bundle and return results."""
        pass

    @abstractmethod
    async def smoke_test(self) -> dict:
        """Quick health check returning status info."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the adapter binary/API is available."""
        pass
