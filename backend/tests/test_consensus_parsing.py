"""E1: critique output contract parsing (JSON -> YAML -> legacy regex).

Covers strict and lenient JSON (trailing commas, single quotes, comments,
missing closing fence, truncation, unescaped quotes), YAML-like blocks,
legacy formats, single counting per span, and that mock-adapter outputs parse
exactly as before.
"""
from __future__ import annotations

import pytest

from backend.adapters import mock as mock_mod
from backend.models import AdapterRunResult, SeverityLevel
from backend.orchestrator.consensus import ConsensusReviewer, parse_lens_output
from backend.orchestrator.feedback import (
    extract_json_critiques,
    extract_json_payloads,
    lenient_json_loads,
    normalize_severity,
    parse_yaml_critiques,
)
from backend.utils.output_cleaner import strip_thinking_traces

CONTRACT = (
    "Reasoning notes first.\n\n"
    "```json\n"
    "{\n"
    '  "critiques": [\n'
    '    {"severity": "critical", "category": "leakage",\n'
    '     "description": "TRRUST edges may be in the pretraining corpus.",\n'
    '     "required_fix": "Evaluate on a held-out perturbation network."},\n'
    '    {"severity": "medium", "category": "statistics",\n'
    '     "description": "No confidence interval for the AUROC difference.",\n'
    '     "required_fix": "Bootstrap a 95% CI."}\n'
    "  ],\n"
    '  "overall_assessment": "Promising but not yet convincing."\n'
    "}\n"
    "```\n"
)


def _descs(parsed):
    return [c.description for c in parsed.critiques]


class TestJsonContract:
    def test_strict_json_block(self):
        p = parse_lens_output(CONTRACT)
        assert p.method == "json" and p.detail == "strict"
        assert [c.severity for c in p.critiques] == [SeverityLevel.CRITICAL, SeverityLevel.MEDIUM]
        assert p.critiques[0].category == "leakage"
        assert p.critiques[0].required_fix == "Evaluate on a held-out perturbation network."
        assert p.overall_assessment == "Promising but not yet convincing."

    def test_explicit_empty_list_is_json_with_zero_critiques(self):
        p = parse_lens_output('```json\n{"critiques": [], "overall_assessment": "Sound."}\n```')
        assert p.method == "json"
        assert p.critiques == []
        assert p.overall_assessment == "Sound."

    def test_trailing_commas(self):
        text = '```json\n{"critiques": [{"severity": "HIGH", "description": "A",},' \
               '{"severity":"low","description":"B",},],}\n```'
        p = parse_lens_output(text)
        assert p.method == "json" and p.detail == "repaired"
        assert _descs(p) == ["A", "B"]
        assert [c.severity for c in p.critiques] == [SeverityLevel.HIGH, SeverityLevel.LOW]

    def test_single_quotes_python_literal(self):
        text = ("```json\n{'critiques': [{'severity': 'critical', 'description': "
                "'Leakage from TRRUST', 'required_fix': 'Hold out', 'flag': true}]}\n```")
        p = parse_lens_output(text)
        assert p.method == "json"
        assert _descs(p) == ["Leakage from TRRUST"]
        assert p.critiques[0].severity == SeverityLevel.CRITICAL

    def test_comments_are_ignored_but_urls_in_strings_kept(self):
        text = ('```json\n{\n // header comment\n "critiques": [ {"severity": "high", '
                '"description": "http://x.org is cited"} ] /* end */\n}\n```')
        p = parse_lens_output(text)
        assert _descs(p) == ["http://x.org is cited"]

    def test_missing_closing_fence(self):
        text = 'Notes\n```json\n{"critiques": [{"severity": "medium", "description": "No seeds."}]}\n'
        p = parse_lens_output(text)
        assert p.method == "json"
        assert _descs(p) == ["No seeds."]

    def test_truncated_output_keeps_complete_items(self):
        text = ('```json\n{"critiques": [{"severity": "high", "description": "A complete one."}, '
                '{"severity": "low", "description":')
        p = parse_lens_output(text)
        assert p.method == "json" and p.detail == "truncated"
        assert _descs(p) == ["A complete one."]

    def test_truncated_mid_string_closes_string(self):
        text = ('```json\n{"critiques": [{"severity": "high", "description": "One."}, '
                '{"severity": "low", "description": "Cut off mid sent')
        p = parse_lens_output(text)
        assert _descs(p) == ["One.", "Cut off mid sent"]

    def test_unescaped_inner_quotes_are_salvaged(self):
        text = ('```json\n{"critiques": [{"severity": "high", "description": '
                '"The "76%" metric is undefined", "required_fix": "Define it"}, '
                '{"severity": "low", "description": "Minor typo"}]}\n```')
        p = parse_lens_output(text)
        assert p.method == "json" and p.detail == "salvaged"
        assert _descs(p) == ['The "76%" metric is undefined', "Minor typo"]
        assert p.critiques[0].required_fix == "Define it"

    def test_bare_json_without_fence(self):
        text = 'Here: {"critiques": [{"severity": "info", "description": "x y z"}], ' \
               '"overall_assessment": "ok"} Thanks!'
        p = parse_lens_output(text)
        assert p.method == "json"
        assert _descs(p) == ["x y z"]
        assert p.overall_assessment == "ok"

    def test_last_contract_block_wins(self):
        text = ('```json\n{"critiques": [{"severity": "low", "description": "draft"}]}\n```\n'
                'Final answer:\n'
                '```json\n{"critiques": [{"severity": "high", "description": "final"}]}\n```')
        assert _descs(parse_lens_output(text)) == ["final"]

    def test_json_takes_precedence_over_prose_brackets_no_double_count(self):
        text = ('[HIGH] Effect size missing\n'
                '```json\n{"critiques": [{"severity": "high", "description": "Effect size missing"}]}\n```')
        p = parse_lens_output(text)
        assert p.method == "json"
        assert len(p.critiques) == 1

    def test_echoed_template_is_not_a_critique(self):
        text = ('```json\n{"critiques": [{"severity": "critical|high|medium|low|info", '
                '"category": "<short>", "description": "<one or two sentences identifying '
                'the specific problem>", "required_fix": "<concrete fix>"}], '
                '"overall_assessment": "<one sentence>"}\n```')
        p = parse_lens_output(text)
        # An echoed template creates no critique AND is not a clean review:
        # the lens is unparsed (retried / failed), never graded A.
        assert p.method == "none"
        assert p.critiques == []
        assert p.invalid_reason == "all_items_unreadable"
        assert p.detail == "invalid_contract:all_items_unreadable"
        assert p.n_raw_items == 1 and p.n_dropped == 1
        assert p.overall_assessment == ""

    def test_aliases_and_severity_synonyms(self):
        text = ('```json\n{"critiques": [{"level": "Major", "issue": "Hub genes dominate.", '
                '"fix": "Degree-matched null."}, {"severity": "weird", "description": "Unknown sev."}, '
                '{"severity": "minor", "description": "no issue detected"}]}\n```')
        p = parse_lens_output(text)
        assert [(c.severity, c.description, c.required_fix) for c in p.critiques] == [
            (SeverityLevel.HIGH, "Hub genes dominate.", "Degree-matched null."),
            (SeverityLevel.MEDIUM, "Unknown sev.", None),
        ]

    def test_top_level_list_of_critiques(self):
        text = '```json\n[{"severity": "low", "description": "a"}, {"severity": "info", "description": "b"}]\n```'
        assert _descs(parse_lens_output(text)) == ["a", "b"]

    def test_non_contract_json_is_ignored(self):
        text = '```json\n{"config": {"k": 5}}\n```\n[HIGH] Legacy critique line'
        p = parse_lens_output(text)
        assert p.method == "regex"
        assert _descs(p) == ["Legacy critique line"]

    def test_lenient_loader_details(self):
        assert lenient_json_loads('{"a": 1}') == ({"a": 1}, "strict")
        assert lenient_json_loads('{"a": 1,}') == ({"a": 1}, "repaired")
        assert lenient_json_loads('{"a": [1, 2') == ({"a": [1, 2]}, "truncated")
        with pytest.raises(ValueError):
            lenient_json_loads("not json at all")

    def test_extract_payloads_handles_python_blocks(self):
        text = "```python\nprint({'x': 1})\n```\n```json\n{\"groups\": [[0, 1]]}\n```"
        payloads = [p for p, _ in extract_json_payloads(text)]
        assert {"groups": [[0, 1]]} in payloads

    def test_extract_json_critiques_none_without_contract(self):
        assert extract_json_critiques("[HIGH] something") is None
        assert extract_json_critiques("") is None


class TestYamlLike:
    def test_yaml_list_of_mappings(self):
        text = (
            "critiques:\n"
            "  - severity: HIGH\n"
            "    category: stats\n"
            "    description: Effect sizes are missing\n"
            "      for the main comparison.\n"
            "    required_fix: Report Cohen's d.\n"
            "  - severity: low\n"
            "    description: >\n"
            "      Typo in table 2.\n"
        )
        p = parse_lens_output(text)
        assert p.method == "yaml"
        assert [(c.severity, c.category, c.description, c.required_fix) for c in p.critiques] == [
            (SeverityLevel.HIGH, "stats", "Effect sizes are missing for the main comparison.",
             "Report Cohen's d."),
            (SeverityLevel.LOW, "general", "Typo in table 2.", None),
        ]

    def test_yaml_bulleted_fields_old_adversarial_format(self):
        text = (
            "1. P-HACKING:\n"
            "   - finding: Layer 15 chosen post hoc\n"
            "   - severity: high\n"
            "   - concrete_counterexample: layers 10-14 give AUROC 0.6\n"
            "2. LEAKAGE:\n"
            "   - finding: no issue detected\n"
            "   - severity: low\n"
        )
        crits = parse_yaml_critiques(text)
        assert [(c.severity, c.description) for c in crits] == [
            (SeverityLevel.HIGH, "Layer 15 chosen post hoc"),
        ]

    def test_yaml_requires_recognised_severity(self):
        text = "- severity: unclear because the text is vague\n  description: something\n"
        assert parse_yaml_critiques(text) == []

    def test_one_line_structured_legacy_is_not_yaml(self):
        text = "Severity: critical, Category: statistics, Description: P-value uncorrected."
        p = parse_lens_output(text)
        assert p.method == "regex"
        assert p.critiques[0].category == "statistics"


class TestLegacyFormats:
    def test_line_matching_several_patterns_counts_once(self):
        text = "- [HIGH] Severity: high, Category: stats, Description: P-values uncorrected\n"
        p = parse_lens_output(text)
        assert p.method == "regex"
        assert len(p.critiques) == 1
        assert p.critiques[0].category == "stats"
        assert p.critiques[0].description == "P-values uncorrected"

    def test_markdown_line_with_nested_bracket_counts_once(self):
        text = "**Severity: MEDIUM** -- [LOW] nested bracket\n[CRITICAL] Core claim unsupported\n"
        p = parse_lens_output(text)
        assert [(c.severity, c.description) for c in p.critiques] == [
            (SeverityLevel.MEDIUM, "[LOW] nested bracket"),
            (SeverityLevel.CRITICAL, "Core claim unsupported"),
        ]

    def test_no_recognisable_format(self):
        p = parse_lens_output("All looks good, no issues.")
        assert p.method == "none" and p.critiques == []
        assert parse_lens_output("").method == "none"

    def test_normalize_severity(self):
        assert normalize_severity("CRITICAL") == SeverityLevel.CRITICAL
        assert normalize_severity("**High**") == SeverityLevel.HIGH
        assert normalize_severity("moderate") == SeverityLevel.MEDIUM
        assert normalize_severity("low (cosmetic)") == SeverityLevel.LOW
        assert normalize_severity("informational") == SeverityLevel.INFO
        assert normalize_severity("banana") is None
        assert normalize_severity(None) is None


class TestMockCompatibility:
    """The mock adapter emits bracket-format critiques; they must parse exactly
    as with the legacy parser (same severities, descriptions, order)."""

    EXPECTED_TIER1 = {
        "reviewer": [
            (SeverityLevel.CRITICAL, "Statistical power analysis is missing for the bootstrap test"),
            (SeverityLevel.HIGH, "Confidence intervals are not reported for the AUROC difference"),
            (SeverityLevel.HIGH, mock_mod._SHARED_EFFECT_SIZE),
        ],
        "adversarial": [
            (SeverityLevel.CRITICAL, "The central claim is unfalsifiable as stated"),
            (SeverityLevel.HIGH, mock_mod._SHARED_EFFECT_SIZE),
            (SeverityLevel.MEDIUM, "The residualization method needs validation on synthetic data"),
        ],
        "bio_plausibility": [
            (SeverityLevel.HIGH,
             "Cell-type composition confounds co-expression and the tissue is unspecified"),
            (SeverityLevel.MEDIUM,
             "Regulatory direction from transcription factor to target is not assessed"),
        ],
    }

    @pytest.mark.parametrize("role", ["reviewer", "adversarial", "bio_plausibility"])
    def test_mock_tier1_outputs_parse_as_regex(self, role):
        out = mock_mod._OUTPUT_BUILDERS[role](1)
        p = parse_lens_output(strip_thinking_traces(out))
        assert p.method == "regex"
        assert [(c.severity, c.description) for c in p.critiques] == self.EXPECTED_TIER1[role]
        assert all(c.category == "general" for c in p.critiques)

    @pytest.mark.parametrize("iteration", range(1, 7))
    @pytest.mark.parametrize("role", ["reviewer", "adversarial", "bio_plausibility"])
    def test_mock_outputs_one_critique_per_bracket_line(self, role, iteration):
        out = mock_mod._OUTPUT_BUILDERS[role](iteration)
        n_lines = sum(1 for line in out.splitlines() if line.startswith("["))
        crits = ConsensusReviewer()._parse_review_output(strip_thinking_traces(out))
        assert len(crits) == n_lines

    def test_mock_panel_merge_unchanged(self):
        """Tier-1 three-lens merge: 7 distinct, shared effect-size critique
        escalated HIGH -> CRITICAL, grade F (same as the legacy merger)."""
        outs = [
            AdapterRunResult(success=True, output=mock_mod._OUTPUT_BUILDERS[r](1))
            for r in ("reviewer", "adversarial", "bio_plausibility")
        ]
        merged = ConsensusReviewer().merge_critiques(outs)
        assert len(merged.critiques) == 7
        assert merged.overall_grade == "F"
        shared = [c for c in merged.critiques if mock_mod._SHARED_EFFECT_SIZE in c.description]
        assert len(shared) == 1
        assert shared[0].severity == SeverityLevel.CRITICAL
        assert shared[0].raised_by == ["adversarial_reviewer", "reviewer"]
        assert shared[0].description.startswith("[adversarial_reviewer, reviewer] ")
        meta = merged.consensus_meta
        assert meta["parse_methods"] == {
            "reviewer": "regex", "adversarial_reviewer": "regex", "bio_plausibility_checker": "regex",
        }
        assert meta["raw_critique_counts"] == {
            "reviewer": 3, "adversarial_reviewer": 3, "bio_plausibility_checker": 2,
        }
        assert meta["panel_failed"] is False and meta["failed_lenses"] == []
