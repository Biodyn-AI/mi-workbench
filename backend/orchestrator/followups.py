"""Parse idea generator output and auto-create follow-up runs."""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from backend.models import FollowUpPolicy, FollowUpProposal, ProviderName, RunCreate

logger = logging.getLogger(__name__)


class FollowUpExecutor:
    """Parse follow-up proposals and create new runs."""

    def parse_proposals(self, output: str) -> list[FollowUpProposal]:
        """Parse idea generator output into structured proposals.

        Detects proposals in formats:
        1. Numbered list: "1. **Title** — description" or "1. **Title** - description"
        2. Markdown headers: "## Proposal: Title" (with body as description)
        3. JSON blocks with title/description/effort fields
        """
        if not output or not output.strip():
            return []

        proposals: list[FollowUpProposal] = []

        # Try JSON blocks first
        json_proposals = self._parse_json_blocks(output)
        if json_proposals:
            return json_proposals

        # Try markdown header format: ## Proposal: Title
        header_proposals = self._parse_markdown_headers(output)
        if header_proposals:
            return header_proposals

        # Try numbered list format: 1. **Title** — description
        numbered_proposals = self._parse_numbered_list(output)
        if numbered_proposals:
            return numbered_proposals

        return proposals

    def _parse_json_blocks(self, output: str) -> list[FollowUpProposal]:
        """Extract proposals from JSON code blocks."""
        proposals: list[FollowUpProposal] = []
        # Match ```json ... ``` blocks
        json_pattern = re.compile(r"```json\s*\n(.*?)```", re.DOTALL)
        for match in json_pattern.finditer(output):
            try:
                data = json.loads(match.group(1))
                items = data if isinstance(data, list) else [data]
                for i, item in enumerate(items):
                    if isinstance(item, dict) and "title" in item:
                        proposals.append(FollowUpProposal(
                            title=item["title"],
                            description=item.get("description", ""),
                            expected_value=item.get("expected_value", ""),
                            effort=item.get("effort", "medium"),
                            dependencies=item.get("dependencies", []),
                            priority=item.get("priority", i + 1),
                        ))
            except (json.JSONDecodeError, TypeError):
                continue
        return proposals

    def _parse_markdown_headers(self, output: str) -> list[FollowUpProposal]:
        """Extract proposals from ## Proposal: Title headers."""
        proposals: list[FollowUpProposal] = []
        # Match "## Proposal: Title" or "## Proposal N: Title"
        header_pattern = re.compile(
            r"^##\s+Proposal\s*(?:\d+\s*)?[:\-]\s*(.+)$", re.MULTILINE
        )
        matches = list(header_pattern.finditer(output))
        if not matches:
            return []

        for i, match in enumerate(matches):
            title = match.group(1).strip()
            # Description is text between this header and the next one (or end)
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(output)
            description = output[start:end].strip()
            # Extract effort if mentioned
            effort = "medium"
            effort_match = re.search(r"[Ee]ffort:\s*(low|medium|high)", description)
            if effort_match:
                effort = effort_match.group(1)
            proposals.append(FollowUpProposal(
                title=title,
                description=description,
                effort=effort,
                priority=i + 1,
            ))
        return proposals

    def _parse_numbered_list(self, output: str) -> list[FollowUpProposal]:
        """Extract proposals from numbered lists like '1. **Title** -- description'."""
        proposals: list[FollowUpProposal] = []
        # Match "N. **Title** — description" or "N. **Title** - description"
        pattern = re.compile(
            r"^\d+\.\s+\*\*(.+?)\*\*\s*[\u2014\-]+\s*(.+?)$",
            re.MULTILINE,
        )
        lines = output.split("\n")
        i = 0
        priority = 0
        while i < len(lines):
            match = pattern.match(lines[i])
            if match:
                priority += 1
                title = match.group(1).strip()
                description = match.group(2).strip()
                # Collect continuation lines (indented or sub-items)
                j = i + 1
                while j < len(lines) and lines[j].strip() and not pattern.match(lines[j]):
                    description += " " + lines[j].strip()
                    j += 1
                effort = "medium"
                effort_match = re.search(r"[Ee]ffort:\s*(low|medium|high)", description)
                if effort_match:
                    effort = effort_match.group(1)
                proposals.append(FollowUpProposal(
                    title=title,
                    description=description,
                    effort=effort,
                    priority=priority,
                ))
                i = j
            else:
                i += 1
        return proposals

    def select_followups(
        self,
        proposals: list[FollowUpProposal],
        policy: FollowUpPolicy,
        max_n: int = 3,
    ) -> list[FollowUpProposal]:
        """Select which proposals to execute based on policy.

        - AUTO_RUN_TOP_1: return top priority only
        - AUTO_RUN_TOP_N: return top max_n by priority
        - ASK_APPROVAL: return empty (UI handles approval)
        """
        if policy == FollowUpPolicy.ASK_APPROVAL:
            return []
        if not proposals:
            return []

        sorted_proposals = sorted(proposals, key=lambda p: p.priority)

        if policy == FollowUpPolicy.AUTO_RUN_TOP_1:
            return sorted_proposals[:1]
        if policy == FollowUpPolicy.AUTO_RUN_TOP_N:
            return sorted_proposals[:max_n]

        return []

    def create_run_from_proposal(
        self,
        proposal: FollowUpProposal,
        parent_workspace_id: str,
        parent_run_id: str,
        provider: ProviderName = ProviderName.MOCK,
    ) -> RunCreate:
        """Create a RunCreate from a follow-up proposal.

        The task description includes the original proposal title and description,
        a reference to the parent run for context, and expected value/effort estimates.
        """
        task_parts = [
            f"Follow-up from run {parent_run_id}: {proposal.title}",
            "",
            proposal.description,
        ]
        if proposal.expected_value:
            task_parts.append(f"\nExpected value: {proposal.expected_value}")
        if proposal.effort:
            task_parts.append(f"Effort: {proposal.effort}")

        return RunCreate(
            workspace_id=parent_workspace_id,
            loop_preset="executor_reviewer",
            task="\n".join(task_parts),
            provider=provider,
            config_overrides={
                "parent_run_id": parent_run_id,
                "follow_up_proposal_id": proposal.id,
            },
        )
