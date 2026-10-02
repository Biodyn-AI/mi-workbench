"""E1: reviewer-lens prompts share one machine-readable output contract, and the
combined-checklist prompt is the item-by-item union of the three lenses."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from backend.orchestrator.consensus import parse_lens_output
from backend.registry.prompts import PromptRegistry

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
LENSES = {
    "rigour": ("reviewer", "mi_reviewer"),
    "adversarial": ("adversarial_reviewer", "adversarial_reviewer"),
    "bio": ("biological_plausibility", "bio_plausibility_checker"),
}
COMBINED = ("reviewer", "mi_reviewer_combined")
ALL = list(LENSES.values()) + [COMBINED]
# Section that holds each lens's substantive checklist.
CHECKLIST_HEADERS = {
    "rigour": "REVIEW DIMENSIONS:",
    "adversarial": "THREAT MODEL",
    "bio": "EVALUATION DIMENSIONS:",
}


@pytest.fixture(scope="module")
def registry():
    return PromptRegistry(str(PROMPTS_DIR))


def _norm(text: str) -> str:
    return " ".join(text.split())


def _checklist_items(system_prompt: str, header: str) -> list[str]:
    """Bullet ('- ...') and 'Attack:' items of the checklist section, with
    their continuation lines joined."""
    start = system_prompt.index(header)
    end = system_prompt.index("REVIEW PRACTICE:", start)
    items: list[str] = []
    for line in system_prompt[start:end].splitlines():
        s = line.strip()
        if s.startswith("- ") or s.startswith("Attack:"):
            items.append(s)
        elif s and items and not re.match(r"^\d+\.\s", s) and not s.isupper():
            items[-1] += " " + s
    return [_norm(i) for i in items]


@pytest.mark.parametrize("role,name", ALL)
def test_prompt_has_single_required_json_contract(registry, role, name):
    t = registry.load_prompt(role, name)
    sp = t.system_prompt
    assert "REQUIRED OUTPUT FORMAT" in sp
    assert sp.count("```json") == 1
    for key in ('"critiques"', '"severity"', '"category"', '"description"',
                '"required_fix"', '"overall_assessment"'):
        assert key in sp
    assert "critical|high|medium|low|info" in sp
    # The contract is the final section of the prompt.
    assert sp.rstrip().endswith("Strict JSON: double quotes, no trailing commas, no comments.")
    # No placeholders the registry would try to substitute.
    assert "{{" not in sp and "{{" not in t.developer_prompt
    assert [o.name for o in t.output_schema] == ["critiques", "overall_assessment"]


@pytest.mark.parametrize("role,name", list(LENSES.values()))
def test_conflicting_legacy_output_formats_removed(registry, role, name):
    sp = registry.load_prompt(role, name).system_prompt
    for legacy in ("\n  OUTPUT FORMAT:", "OUTPUT FORMAT:\n", "adversarial_grade", "plausibility_grade",
                   "List of (severity", "overall_grade:", "attack_plan:", "concrete_counterexample:"):
        assert legacy not in sp, legacy


@pytest.mark.parametrize("role,name", list(LENSES.values()))
def test_lens_versions_bumped(registry, role, name):
    t = registry.load_prompt(role, name)
    assert t.metadata.version == "2.0.0"
    assert any(c.startswith("2.0.0") for c in t.metadata.changelog)


def test_combined_prompt_metadata(registry):
    t = registry.load_prompt(*COMBINED)
    assert t.metadata.role == "reviewer"
    assert t.metadata.name == "mi_reviewer_combined"
    assert ("reviewer", "mi_reviewer_combined", "1.0.0") in registry.list_prompts()


@pytest.mark.parametrize("lens", list(LENSES))
def test_combined_contains_every_lens_checklist_item_verbatim(registry, lens):
    lens_sp = registry.load_prompt(*LENSES[lens]).system_prompt
    combined = _norm(registry.load_prompt(*COMBINED).system_prompt)
    items = _checklist_items(lens_sp, CHECKLIST_HEADERS[lens])
    assert len(items) >= 15
    missing = [i for i in items if i not in combined]
    assert not missing, missing


@pytest.mark.parametrize("lens", list(LENSES))
def test_combined_contains_lens_practice_items(registry, lens):
    """Review-practice bullets of each lens appear in the combined prompt
    (the ordering instruction is merged into the combined TONE section)."""
    lens_sp = registry.load_prompt(*LENSES[lens]).system_prompt
    combined = _norm(registry.load_prompt(*COMBINED).system_prompt)
    start = lens_sp.index("REVIEW PRACTICE:")
    end = lens_sp.index("SEVERITY LEVELS", start)
    practice = []
    for line in lens_sp[start:end].splitlines()[1:]:
        s = line.strip()
        if s.startswith("- "):
            practice.append(s)
        elif s and practice:
            practice[-1] += " " + s
    practice = [_norm(p) for p in practice if "most to least damaging" not in p]
    assert practice
    missing = [p for p in practice if p not in combined]
    assert not missing, missing


def test_combined_severity_scale_merges_lens_definitions(registry):
    combined = _norm(registry.load_prompt(*COMBINED).system_prompt)
    for phrase in (
        "Invalidates a core claim",
        "overturns a core conclusion",
        "contradicts established biology",
        "unsupported in the specific experimental context studied",
        "relies on generic database annotations",
        "Observation or clarification request",
    ):
        assert phrase in combined, phrase


@pytest.mark.parametrize("role,name", ALL)
def test_contract_example_itself_parses_to_no_critiques(registry, role, name):
    """A model that echoes the template verbatim must not create critiques,
    and must not count as a clean review either (unparsed lens, no grade A)."""
    sp = registry.load_prompt(role, name).system_prompt
    block = sp[sp.index("```json"): sp.index("```", sp.index("```json") + 3) + 3]
    parsed = parse_lens_output(block)
    assert parsed.method == "none"
    assert parsed.invalid_reason == "all_items_unreadable"
    assert parsed.critiques == []
