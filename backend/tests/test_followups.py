"""Tests for follow-up proposal parsing and execution."""
import pytest

from backend.models import FollowUpPolicy, FollowUpProposal, ProviderName, RunCreate
from backend.orchestrator.followups import FollowUpExecutor


@pytest.fixture
def executor():
    return FollowUpExecutor()


# ── Parsing tests ────────────────────────────────────────────────────


class TestParseNumberedProposals:
    def test_parse_numbered_proposals(self, executor):
        output = (
            "Here are the follow-up experiments:\n\n"
            "1. **Cross-context validation** - Test attention on CRISPRa data\n"
            "2. **Head ablation study** - Ablate top-5 heads and measure drop\n"
            "3. **Scaling analysis** - Vary N from 100 to 5000\n"
        )
        proposals = executor.parse_proposals(output)
        assert len(proposals) == 3
        assert proposals[0].title == "Cross-context validation"
        assert proposals[1].title == "Head ablation study"
        assert proposals[2].title == "Scaling analysis"
        assert "CRISPRa" in proposals[0].description

    def test_parse_numbered_with_em_dash(self, executor):
        output = "1. **Title One** \u2014 description of first\n2. **Title Two** \u2014 description of second\n"
        proposals = executor.parse_proposals(output)
        assert len(proposals) == 2
        assert proposals[0].title == "Title One"


class TestParseMarkdownProposals:
    def test_parse_markdown_proposals(self, executor):
        output = (
            "## Proposal 1: Cross-context validation\n"
            "Test attention on Adamson CRISPRa data.\n"
            "Effort: low\n\n"
            "## Proposal 2: Head ablation\n"
            "Ablate top-5 heads and measure perturbation prediction.\n"
            "Effort: high\n"
        )
        proposals = executor.parse_proposals(output)
        assert len(proposals) == 2
        assert proposals[0].title == "Cross-context validation"
        assert proposals[0].effort == "low"
        assert proposals[1].title == "Head ablation"
        assert proposals[1].effort == "high"

    def test_parse_markdown_without_number(self, executor):
        output = (
            "## Proposal: Single experiment\n"
            "A standalone proposal description.\n"
        )
        proposals = executor.parse_proposals(output)
        assert len(proposals) == 1
        assert proposals[0].title == "Single experiment"


class TestParseJsonProposals:
    def test_parse_json_block(self, executor):
        output = (
            "Here are my proposals:\n\n"
            '```json\n'
            '[{"title": "Exp A", "description": "Do A", "effort": "low", "priority": 1},'
            ' {"title": "Exp B", "description": "Do B", "effort": "high", "priority": 2}]\n'
            '```\n'
        )
        proposals = executor.parse_proposals(output)
        assert len(proposals) == 2
        assert proposals[0].title == "Exp A"
        assert proposals[0].effort == "low"
        assert proposals[1].title == "Exp B"


class TestParseEmptyOutput:
    def test_parse_empty_output(self, executor):
        assert executor.parse_proposals("") == []
        assert executor.parse_proposals("   ") == []
        assert executor.parse_proposals("No proposals here.") == []


class TestProposalHasPriorityOrdering:
    def test_proposal_has_priority_ordering(self, executor):
        output = (
            "1. **Third** - should be priority 1\n"
            "2. **First** - should be priority 2\n"
            "3. **Second** - should be priority 3\n"
        )
        proposals = executor.parse_proposals(output)
        assert len(proposals) == 3
        # Priority follows the ordering in the text
        assert proposals[0].priority == 1
        assert proposals[1].priority == 2
        assert proposals[2].priority == 3


# ── Selection tests ──────────────────────────────────────────────────


def _make_proposals(n: int = 5) -> list[FollowUpProposal]:
    return [
        FollowUpProposal(
            title=f"Proposal {i}",
            description=f"Description {i}",
            priority=i,
        )
        for i in range(1, n + 1)
    ]


class TestSelectTop1:
    def test_select_top_1(self, executor):
        proposals = _make_proposals(5)
        selected = executor.select_followups(proposals, FollowUpPolicy.AUTO_RUN_TOP_1)
        assert len(selected) == 1
        assert selected[0].priority == 1

    def test_select_top_1_empty(self, executor):
        selected = executor.select_followups([], FollowUpPolicy.AUTO_RUN_TOP_1)
        assert selected == []


class TestSelectTopN:
    def test_select_top_n(self, executor):
        proposals = _make_proposals(5)
        selected = executor.select_followups(proposals, FollowUpPolicy.AUTO_RUN_TOP_N, max_n=3)
        assert len(selected) == 3
        assert [p.priority for p in selected] == [1, 2, 3]

    def test_select_top_n_fewer_than_max(self, executor):
        proposals = _make_proposals(2)
        selected = executor.select_followups(proposals, FollowUpPolicy.AUTO_RUN_TOP_N, max_n=5)
        assert len(selected) == 2


class TestSelectAskApproval:
    def test_select_ask_approval_returns_empty(self, executor):
        proposals = _make_proposals(5)
        selected = executor.select_followups(proposals, FollowUpPolicy.ASK_APPROVAL)
        assert selected == []


# ── Run creation test ────────────────────────────────────────────────


class TestCreateRunFromProposal:
    def test_create_run_from_proposal(self, executor):
        proposal = FollowUpProposal(
            title="Cross-context validation",
            description="Test on CRISPRa data",
            expected_value="Validate attention mechanism",
            effort="medium",
        )
        run = executor.create_run_from_proposal(
            proposal,
            parent_workspace_id="ws123",
            parent_run_id="run456",
            provider=ProviderName.MOCK,
        )
        assert isinstance(run, RunCreate)
        assert run.workspace_id == "ws123"
        assert run.provider == ProviderName.MOCK
        assert "run456" in run.task
        assert "Cross-context validation" in run.task
        assert "CRISPRa" in run.task
        assert run.config_overrides["parent_run_id"] == "run456"
        assert "follow_up_proposal_id" in run.config_overrides
