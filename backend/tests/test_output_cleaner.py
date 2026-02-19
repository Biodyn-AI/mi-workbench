"""Tests for the output cleaner (thinking trace removal)."""
import pytest

from backend.utils.output_cleaner import strip_thinking_traces


class TestStripThinkingTraces:
    def test_removes_i_will_lines(self):
        text = (
            "I will read the PROTOCOL.md file to assess its state.\n"
            "I will list the contents of the knowledge directory.\n"
            "\n"
            "## ReviewResult\n"
            "\n"
            "**Overall Grade:** B\n"
        )
        result = strip_thinking_traces(text)
        assert "I will read" not in result
        assert "I will list" not in result
        assert "## ReviewResult" in result
        assert "**Overall Grade:** B" in result

    def test_removes_let_me_lines(self):
        text = (
            "Let me analyze the experimental protocol.\n"
            "Let me check the statistical framework.\n"
            "\n"
            "# Analysis\n"
            "The protocol is sound.\n"
        )
        result = strip_thinking_traces(text)
        assert "Let me" not in result
        assert "# Analysis" in result
        assert "The protocol is sound." in result

    def test_removes_ill_lines(self):
        text = "I'll search for relevant data.\n\nThe data shows clear trends."
        result = strip_thinking_traces(text)
        assert "I'll search" not in result
        assert "The data shows clear trends." in result

    def test_removes_i_need_to_lines(self):
        text = "I need to verify the results.\n\n## Results\nAUROC = 0.54"
        result = strip_thinking_traces(text)
        assert "I need to" not in result
        assert "## Results" in result

    def test_removes_im_going_to_lines(self):
        text = "I'm going to overwrite the file.\n\n# Protocol\nStep 1."
        result = strip_thinking_traces(text)
        assert "I'm going to" not in result
        assert "# Protocol" in result

    def test_removes_tool_narration(self):
        text = (
            "I will read the file to check for errors.\n"
            "I will search for any mentions of scPerturb.\n"
            "I'll overwrite the PROTOCOL.md file with updated content.\n"
            "\n"
            "# Experimental Protocol\n"
            "This protocol tests causal regulatory encoding.\n"
        )
        result = strip_thinking_traces(text)
        assert "I will read" not in result
        assert "I will search" not in result
        assert "I'll overwrite" not in result
        assert "# Experimental Protocol" in result

    def test_removes_sequencing_prefixes(self):
        text = (
            "First, I will read the current state.\n"
            "Next, I will update the protocol.\n"
            "Then I will verify the changes.\n"
            "Finally, I will commit.\n"
            "\n"
            "Done.\n"
        )
        result = strip_thinking_traces(text)
        assert "First, I will" not in result
        assert "Next, I will" not in result
        assert "Done." in result

    def test_removes_the_following_is(self):
        text = (
            "The following is the structured review of the protocol.\n"
            "\n"
            "## Review\n"
            "Grade: B\n"
        )
        result = strip_thinking_traces(text)
        assert "The following is" not in result
        assert "## Review" in result

    def test_preserves_code_blocks(self):
        text = (
            "I will analyze the code.\n"
            "\n"
            "```python\n"
            "# I will compute the AUROC\n"
            "auroc = compute_auroc(labels, scores)\n"
            "```\n"
            "\n"
            "The AUROC is 0.54.\n"
        )
        result = strip_thinking_traces(text)
        # First "I will" should be removed (outside code block)
        assert result.count("I will") == 1
        # "I will" inside code block should be preserved
        assert "# I will compute the AUROC" in result
        assert "auroc = compute_auroc" in result

    def test_preserves_markdown_headers(self):
        text = (
            "I will now present the results.\n"
            "\n"
            "# Results\n"
            "## Sub-results\n"
            "### Details\n"
            "The effect size is 0.3.\n"
        )
        result = strip_thinking_traces(text)
        assert "# Results" in result
        assert "## Sub-results" in result
        assert "### Details" in result

    def test_preserves_substantive_i_statements(self):
        """Lines that start with 'I' but are substantive analysis should be kept."""
        text = (
            "In this analysis, we examine the regulatory encoding.\n"
            "I hypothesize that attention captures co-expression.\n"
            "Indeed, the correlation is r=0.85.\n"
        )
        result = strip_thinking_traces(text)
        assert "In this analysis" in result
        assert "I hypothesize" in result
        assert "Indeed" in result

    def test_empty_input(self):
        assert strip_thinking_traces("") == ""

    def test_none_like_empty(self):
        assert strip_thinking_traces("") == ""

    def test_no_traces(self):
        text = (
            "# Protocol\n"
            "\n"
            "## Hypotheses\n"
            "H0: Attention does not encode causal regulation.\n"
        )
        result = strip_thinking_traces(text)
        assert result.strip() == text.strip()

    def test_all_traces(self):
        text = (
            "I will read the file.\n"
            "Let me check the contents.\n"
            "I need to verify the data.\n"
        )
        result = strip_thinking_traces(text)
        assert result.strip() == ""

    def test_collapses_excessive_blank_lines(self):
        text = (
            "I will read the file.\n"
            "\n"
            "\n"
            "\n"
            "\n"
            "# Content\n"
            "Real content here.\n"
        )
        result = strip_thinking_traces(text)
        # Should not have more than 2 consecutive blank lines
        assert "\n\n\n\n" not in result
        assert "# Content" in result

    def test_mixed_real_content(self):
        """Test with realistic mixed output from a Gemini run."""
        text = (
            "I will read the current content of `PROTOCOL.md` to assess its state.\n"
            "I will list the contents of the `knowledge` directory.\n"
            "I will search for any mentions of \"scPerturb\" in the current directory.\n"
            "I will overwrite the `PROTOCOL.md` file with a comprehensive protocol.\n"
            "\n"
            "# Experimental Protocol: Causal Regulatory Encoding\n"
            "\n"
            "**Date:** February 18, 2026\n"
            "**Objective:** Test if Geneformer encodes causal regulation.\n"
            "\n"
            "## 1. Hypotheses\n"
            "\n"
            "* **H0:** Attention weights are indistinguishable from co-expression.\n"
            "* **H1:** Regulatory pairs show significantly higher attention.\n"
        )
        result = strip_thinking_traces(text)
        # All trace lines should be removed
        assert "I will read" not in result
        assert "I will list" not in result
        assert "I will search" not in result
        assert "I will overwrite" not in result
        # All real content should be preserved
        assert "# Experimental Protocol" in result
        assert "**Date:** February 18, 2026" in result
        assert "## 1. Hypotheses" in result
        assert "**H0:**" in result
        assert "**H1:**" in result
