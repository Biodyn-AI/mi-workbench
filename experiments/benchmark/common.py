"""Shared helpers for the planted-flaw critique benchmark harness (X1/X2).

Everything here is pure standard library plus the platform's own ``backend``
package (pydantic, pyyaml), so the harness runs in the platform environment
(``mi_workbench``) and in the analysis environment alike.

Contract: ``experiments/benchmark/ANALYSIS_PLAN.md``. The reviewer system
prompts are the platform's prompt YAMLs resolved exactly as the engine does
(``PromptRegistry.load_prompt`` + ``resolve_template``, see
``LoopEngine._resolve_role_system_prompt``); critiques are parsed with
``backend.orchestrator.consensus.parse_lens_output`` after the same
``strip_thinking_traces`` pass the consensus panel applies, and conditions are
merged with ``merge_parsed_critiques``.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.models import (  # noqa: E402
    AdapterRunRequest,
    PromptBundle,
    ReviewCritique,
    SeverityLevel,
    WorkspaceContext,
)
from backend.orchestrator.consensus import (  # noqa: E402
    MergeOutcome,
    merge_parsed_critiques,
    parse_lens_output,
)
from backend.orchestrator import similarity  # noqa: E402
from backend.registry.prompts import PromptRegistry  # noqa: E402
from backend.utils.output_cleaner import strip_thinking_traces  # noqa: E402

DEFAULT_DATA_DIR = REPO_ROOT / "experiments_data" / "benchmark_runs"
MANIFEST_NAME = "benchmark_manifest.json"
RESULTS_DIR = BENCH_DIR / "results"

# Fixed task statement (ANALYSIS_PLAN.md, "Reviewer calls").
TASK_STATEMENT = (
    "Review the following mechanistic-interpretability analysis of a "
    "single-cell foundation model."
)

# Call tag -> (lens role name, prompt_ref). Order = canonical call order.
CALL_SPECS: dict[str, tuple[str, str]] = {
    "rig1": ("reviewer", "reviewer/mi_reviewer"),
    "rig2": ("reviewer", "reviewer/mi_reviewer"),
    "rig3": ("reviewer", "reviewer/mi_reviewer"),
    "cmb1": ("reviewer_combined", "reviewer/mi_reviewer_combined"),
    "cmb2": ("reviewer_combined", "reviewer/mi_reviewer_combined"),
    "cmb3": ("reviewer_combined", "reviewer/mi_reviewer_combined"),
    "adv": ("adversarial_reviewer", "adversarial_reviewer/adversarial_reviewer"),
    "bio": ("bio_plausibility_checker", "biological_plausibility/bio_plausibility_checker"),
}
ALL_CALLS: tuple[str, ...] = tuple(CALL_SPECS)

# Conditions (ANALYSIS_PLAN.md): calls each condition is built from.
CONDITIONS: dict[str, tuple[str, ...]] = {
    "C1": ("rig1",),
    "C2": ("cmb1",),
    "C3": ("rig1", "rig2", "rig3"),
    "C4": ("cmb1", "cmb2", "cmb3"),
    "C5": ("rig1", "adv", "bio"),
}
CONDITION_LABELS = {
    "C1": "single rigour",
    "C2": "single combined",
    "C3": "rigour x3",
    "C4": "combined x3",
    "C5": "panel (rigour+adversarial+bio)",
}
# Planned contrasts (A minus B).
PLANNED_CONTRASTS: tuple[tuple[str, str], ...] = (
    ("C5", "C1"), ("C5", "C3"), ("C5", "C4"), ("C4", "C2"), ("C3", "C1"),
)

FAMILY_OF_PREFIX = {
    "S": "statistics", "C": "confounding", "L": "causal_logic",
    "T": "technical", "B": "biological",
}
FLAW_TYPES = ("S1", "S2", "S3", "C1", "C2", "C3", "L1", "L2", "T1", "T2", "B1", "B2")
FLAWS_PER_TYPE = 4

# Merge of the pre-specified analyses: the platform merge with its default
# settings at freeze time (ANALYSIS_PLAN.md), i.e. legacy jaccard at 0.5. The
# platform defaults changed after the merge evaluation (LLM adjudication;
# calibrated jaccard / tfidf thresholds 0.1; results/merge_eval.json), so the
# harness pins the freeze-time values instead of following the platform. A
# threshold left as None resolves to the freeze-time default of the chosen
# method (similarity.UNCALIBRATED_THRESHOLDS: jaccard 0.5, tfidf 0.3,
# embedding 0.7), never to the current platform default.
DEFAULT_MERGE_METHOD = "jaccard"
DEFAULT_MERGE_THRESHOLD = 0.5
DEFAULT_CONSENSUS_THRESHOLD = 2
# Platform default (ConsensusReviewer / merge_parsed_critiques): merging is
# cross-lens (cross-call) only.
DEFAULT_SAME_LENS_MERGE = False

SEVERITY_SCORE = {"critical": 8, "high": 4, "medium": 2, "low": 1, "info": 0}
GRADE_ORDINAL = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}

REVIEW_SCHEMA = "miw-benchmark-review/1"
JUDGE_SCHEMA = "miw-benchmark-judgment/1"

_SYMMETRY_RE = re.compile(r"symmetr|A_ij\s*=\s*A_ji", re.IGNORECASE)


# ── small utilities ─────────────────────────────────────────────────────


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def atomic_write_json(path: Path, obj: Any) -> None:
    """Write JSON via a temp file + rename so a crash never leaves half a record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def stable_seed(*parts: Any) -> int:
    """64-bit seed from a hash of the parts (independent of PYTHONHASHSEED)."""
    return int(sha256_text("|".join(str(p) for p in parts))[:16], 16)


def visible_files(directory: Path) -> list[Path]:
    """Regular files in ``directory`` excluding dotfiles (e.g. macOS ``._*``)."""
    return sorted(
        p for p in Path(directory).iterdir()
        if p.is_file() and not p.name.startswith(".")
    )


def parse_csv_arg(value: Optional[str]) -> list[str]:
    if value is None:
        return []
    return [v.strip() for v in str(value).split(",") if v.strip()]


# ── benchmark items ─────────────────────────────────────────────────────


@dataclass
class Flaw:
    flaw_id: str
    type: str
    family: str
    quote: str
    description: str
    detection_criterion: str

    @property
    def is_symmetry_instance(self) -> bool:
        """T1 instances built on the false 'attention is symmetric' premise."""
        return self.type == "T1" and bool(
            _SYMMETRY_RE.search(self.quote) or _SYMMETRY_RE.search(self.description)
        )


@dataclass
class Item:
    item_id: str           # e.g. "A01-flawed"
    scenario: str          # e.g. "A01"
    version: str           # "flawed" | "clean"
    title: str             # scenario description from the ground truth
    artifact_path: Path
    gt_path: Path
    artifact_text: str
    flaws: list[Flaw] = field(default_factory=list)

    @property
    def is_flawed(self) -> bool:
        return self.version == "flawed"

    @property
    def flaw_ids(self) -> list[str]:
        return [f.flaw_id for f in self.flaws]


def load_items(bench_dir: Path = BENCH_DIR) -> dict[str, Item]:
    """Load every (artifact, ground truth) pair, keyed by item id, sorted."""
    bench_dir = Path(bench_dir)
    items: dict[str, Item] = {}
    for gt_path in visible_files(bench_dir / "ground_truth"):
        if gt_path.suffix != ".json":
            continue
        gt = read_json(gt_path)
        item_id = gt.get("artifact_id") or gt_path.stem
        artifact_path = bench_dir / "artifacts" / f"{item_id}.md"
        text = artifact_path.read_text(encoding="utf-8") if artifact_path.exists() else ""
        flaws = [
            Flaw(
                flaw_id=f["flaw_id"], type=f["type"], family=f["family"],
                quote=f["quote"], description=f["description"],
                detection_criterion=f["detection_criterion"],
            )
            for f in gt.get("flaws", [])
        ]
        items[item_id] = Item(
            item_id=item_id,
            scenario=item_id.split("-")[0],
            version=gt.get("version", "flawed" if flaws else "clean"),
            title=gt.get("scenario", ""),
            artifact_path=artifact_path,
            gt_path=gt_path,
            artifact_text=text,
            flaws=flaws,
        )
    return dict(sorted(items.items()))


def benchmark_files(bench_dir: Path = BENCH_DIR) -> list[Path]:
    """Files covered by the frozen manifest (artifacts, ground truth, plan)."""
    bench_dir = Path(bench_dir)
    files = visible_files(bench_dir / "artifacts") + visible_files(bench_dir / "ground_truth")
    files.append(bench_dir / "ANALYSIS_PLAN.md")
    return files


def current_file_hashes(bench_dir: Path = BENCH_DIR) -> dict[str, str]:
    bench_dir = Path(bench_dir)
    return {
        p.relative_to(bench_dir).as_posix(): sha256_file(p)
        for p in benchmark_files(bench_dir)
    }


class ManifestError(RuntimeError):
    pass


def verify_manifest(bench_dir: Path = BENCH_DIR) -> dict:
    """Check the frozen manifest against the files on disk; raise on mismatch."""
    bench_dir = Path(bench_dir)
    path = bench_dir / MANIFEST_NAME
    if not path.exists():
        raise ManifestError(f"{path} not found: run freeze.py first")
    manifest = read_json(path)
    frozen = manifest.get("files", {})
    now = current_file_hashes(bench_dir)
    problems = []
    for rel in sorted(set(frozen) | set(now)):
        if rel not in now:
            problems.append(f"missing: {rel}")
        elif rel not in frozen:
            problems.append(f"not in manifest: {rel}")
        elif frozen[rel] != now[rel]:
            problems.append(f"hash changed: {rel}")
    if problems:
        raise ManifestError("benchmark files differ from the frozen manifest: " + "; ".join(problems))
    return manifest


# ── prompts ─────────────────────────────────────────────────────────────


@dataclass
class PromptInfo:
    prompt_ref: str
    role: str
    text: str
    version: str
    sha256: str


_PROMPT_CACHE: dict[tuple[str, str, str], PromptInfo] = {}


def load_system_prompt(prompt_ref: str, role: str, prompts_dir: Optional[str] = None) -> PromptInfo:
    """Resolve a reviewer system prompt exactly as the engine does.

    ``LoopEngine._resolve_role_system_prompt`` loads ``role/name`` from the
    ``PromptRegistry`` and substitutes ``{{role}}``, ``{{iteration}}`` and
    ``{{task}}``. Unlike the engine, a failed lookup raises instead of falling
    back to an inline prompt, so a benchmark call can never silently run with
    the wrong instructions.
    """
    key = (prompt_ref, role, prompts_dir or "")
    if key in _PROMPT_CACHE:
        return _PROMPT_CACHE[key]
    registry = PromptRegistry(prompts_dir)
    group, name = prompt_ref.split("/", 1)
    template = registry.load_prompt(group, name)
    variables = {"role": role, "iteration": "1", "task": TASK_STATEMENT}
    text = registry.resolve_template(template, variables)
    if not text.strip():
        raise ValueError(f"prompt {prompt_ref} resolved to an empty system prompt")
    info = PromptInfo(
        prompt_ref=prompt_ref, role=role, text=text,
        version=template.metadata.version, sha256=sha256_text(text),
    )
    _PROMPT_CACHE[key] = info
    return info


def build_user_prompt(artifact_text: str) -> str:
    """Fixed task statement + the artifact text (nothing else)."""
    return (
        f"{TASK_STATEMENT}\n\n"
        "=== Analysis under review ===\n\n"
        f"{artifact_text.strip()}\n\n"
        "=== End of analysis ===\n"
    )


# ── adapters ────────────────────────────────────────────────────────────


def resolve_provider(model: str) -> tuple[str, str]:
    """``[provider:]model`` -> (provider, model). Provider inferred from the name."""
    if ":" in model:
        provider, name = model.split(":", 1)
        return provider.strip().lower(), name.strip()
    low = model.lower()
    if low.startswith(("gpt-", "o1", "o3", "o4", "codex")):
        return "codex", model
    if low.startswith(("claude", "opus", "sonnet", "haiku")):
        return "claude", model
    if low.startswith("gemini"):
        return "gemini", model
    raise ValueError(f"cannot infer provider for model {model!r}; use provider:model")


def model_dirname(model: str) -> str:
    name = model.split(":", 1)[1] if ":" in model else model
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


CODEX_WRAPPER = BENCH_DIR / "codex_wrapper.py"  # legacy shim, no longer used
# Codex tools-off arguments of the benchmark. ``--sandbox read-only`` (what
# the adapter passes for allow_tools=False) still leaves Codex's live web
# search tool enabled; the plan requires tools to be disabled. The frozen
# pilot condition ("web_only") disables web search and keeps the user config
# (read-only shell, user MCP servers / plugins remain available to the
# model); "strict" uses the backend adapter's full tools-off set (no shell,
# no apps / plugins / MCP, user config.toml ignored). The list is part of the
# cache signature (``cli_extra_args``), so records made under one condition
# are never reused under the other.
CODEX_NO_WEB_ARGS: tuple[str, ...] = ("-c", 'web_search="disabled"')
IGNORE_USER_CONFIG_FLAG = "--ignore-user-config"
CODEX_TOOLS_OFF_MODES = ("web_only", "strict")


def codex_extra_args(web_search: str = "disabled", tools_off: str = "web_only") -> list[str]:
    """Codex arguments added for the (tools-off) benchmark calls."""
    if tools_off == "strict":
        from backend.adapters.codex_cli import TOOLS_OFF_ARGS
        return list(TOOLS_OFF_ARGS) + [IGNORE_USER_CONFIG_FLAG]
    if tools_off != "web_only":
        raise ValueError(f"tools_off must be one of {CODEX_TOOLS_OFF_MODES}, got {tools_off!r}")
    if web_search == "disabled":
        return list(CODEX_NO_WEB_ARGS)
    if web_search == "default":
        return []
    raise ValueError(f"web_search must be 'disabled' or 'default', got {web_search!r}")


def make_adapter(model: str, effort: str, codex_home: Optional[str] = None,
                 codex_extra: Optional[Sequence[str]] = None):
    """Backend CLI adapter for ``model`` with tools disabled.

    For Codex, ``codex_extra`` (default: web search disabled, the frozen
    pilot condition) is exactly the set of tools-off arguments the backend
    adapter adds to its command (``--ignore-user-config`` in the list turns
    on the adapter's ignore-user-config option). The adapter builds the
    command itself; the old ``codex_wrapper.py`` shim is not needed.
    """
    provider, name = resolve_provider(model)
    if provider == "codex":
        from backend.adapters.codex_cli import CodexCliAdapter
        extra = list(CODEX_NO_WEB_ARGS) if codex_extra is None else list(codex_extra)
        ignore = IGNORE_USER_CONFIG_FLAG in extra
        tools_off = [a for a in extra if a != IGNORE_USER_CONFIG_FLAG]
        return CodexCliAdapter(binary="codex", model=name, reasoning_effort=effort,
                               allow_tools=False, codex_home=codex_home,
                               tools_off_args=tools_off, ignore_user_config=ignore)
    if provider == "claude":
        from backend.adapters.claude_code import ClaudeCodeAdapter
        return ClaudeCodeAdapter(model=name, reasoning_effort=effort, allow_tools=False)
    if provider == "gemini":
        from backend.adapters.gemini_cli import GeminiCliAdapter
        return GeminiCliAdapter(model=name, reasoning_effort=effort, allow_tools=False)
    raise ValueError(f"unknown provider {provider!r}")


def make_request(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str,
    effort: str,
    timeout_seconds: float,
    workspace: str,
    role: str,
) -> AdapterRunRequest:
    _, name = resolve_provider(model)
    return AdapterRunRequest(
        prompt_bundle=PromptBundle(
            system_prompt=system_prompt, user_prompt=user_prompt,
            variables={"role": role},
        ),
        workspace_context=WorkspaceContext(workspace_path=workspace),
        timeout_seconds=max(1, int(timeout_seconds)),
        model=name,
        reasoning_effort=effort,
        allow_tools=False,
    )


def make_workspace(root: Optional[Path] = None) -> Path:
    """Empty per-call working directory outside any repository.

    The CLI agent is started with ``--cd`` here so it sees no project files
    (no AGENTS.md, no benchmark ground truth in its working tree).
    """
    base = Path(root) if root else Path(tempfile.gettempdir()) / "miw_bench_ws"
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="ws_", dir=str(base)))


def remove_workspace(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def codex_env_fingerprint(codex_home: Optional[str] = None) -> dict:
    """What the Codex CLI adds on top of our prompt (recorded per call).

    Codex prepends its own harness instructions and tool definitions and the
    user-level ``$CODEX_HOME/AGENTS.md``; ``config.toml`` can add plugins and
    MCP servers. Only hashes are recorded.
    """
    home = Path(codex_home or os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    out: dict[str, Any] = {"codex_home": str(home)}
    for name in ("AGENTS.md", "AGENTS.override.md", "config.toml"):
        p = home / name
        out[name] = sha256_file(p) if p.is_file() else None
    return out


_CODEX_ENV_SIG = {
    "codex_agents_md_sha256": "AGENTS.md",
    "codex_agents_override_md_sha256": "AGENTS.override.md",
}


def codex_env_signature(provider: str, codex_home: Optional[str] = None) -> dict:
    """Cache-signature fields for what Codex injects into every prompt: the
    hashes of ``$CODEX_HOME/AGENTS.md`` / ``AGENTS.override.md`` (None for
    other providers). ``config.toml`` is not part of the signature: it holds
    volatile project-trust entries and is recorded in ``codex_env`` only."""
    if provider != "codex":
        return {k: None for k in _CODEX_ENV_SIG}
    fp = codex_env_fingerprint(codex_home)
    return {k: fp.get(name) for k, name in _CODEX_ENV_SIG.items()}


def signature_matches(request: dict, signature: dict) -> bool:
    """True if a cached record's ``request`` matches ``signature``.

    Records made before a signature field existed are compared on the value
    they recorded elsewhere (the AGENTS.md hashes in ``request.codex_env``),
    so old records stay reusable exactly when they were made under the same
    conditions."""
    request = request or {}
    for key, value in signature.items():
        if key in request:
            got = request.get(key)
        elif key in _CODEX_ENV_SIG:
            got = (request.get("codex_env") or {}).get(_CODEX_ENV_SIG[key])
        else:
            got = None
        if got != value:
            return False
    return True


def usage_tokens(result: Any) -> dict:
    raw = dict(getattr(result, "raw_usage", {}) or {})
    return {
        "input": int(getattr(result, "input_tokens", 0) or 0),
        "output": int(getattr(result, "output_tokens", 0) or 0),
        "cached_input": int(getattr(result, "cached_input_tokens", 0) or 0),
        "reasoning_output": int(raw.get("reasoning_output_tokens", 0) or 0),
        "raw_usage": raw,
    }


def count_tool_events(result: Any) -> int:
    """Number of agent tool events (codex ``item.started``) during a call."""
    so = getattr(result, "structured_output", {}) or {}
    return sum(1 for t in so.get("event_types", []) or [] if t == "item.started")


# ── CLI stdout capture (audit of agent tool use) ────────────────────────
#
# The adapters keep the raw CLI stdout only for failed calls. To audit what
# an agent did with its (read-only) tools, the harness wraps the module-level
# ``run_cli_process`` name the adapters call: the wrapper calls the original
# unchanged and stores the stdout in a per-call holder (a contextvar, so
# concurrent calls do not mix). Adapter logic is not modified.

_CLI_CAPTURE: contextvars.ContextVar = contextvars.ContextVar("miw_bench_cli_capture", default=None)
_ADAPTER_MODULES = ("backend.adapters.codex_cli", "backend.adapters.claude_code",
                    "backend.adapters.gemini_cli")
_NON_TOOL_ITEMS = {"agent_message", "assistant_message", "reasoning", "error"}


def install_cli_capture() -> None:
    import importlib

    for modname in _ADAPTER_MODULES:
        mod = importlib.import_module(modname)
        orig = getattr(mod, "run_cli_process", None)
        if orig is None or getattr(orig, "_miw_capture", False):
            continue

        async def wrapped(*args, __orig=orig, **kwargs):
            res = await __orig(*args, **kwargs)
            holder = _CLI_CAPTURE.get()
            if holder is not None:
                holder.append(getattr(res, "stdout", "") or "")
            return res

        wrapped._miw_capture = True  # type: ignore[attr-defined]
        mod.run_cli_process = wrapped


def begin_capture() -> list:
    holder: list = []
    _CLI_CAPTURE.set(holder)
    return holder


def extract_tool_items(stdouts: Sequence[str], limit: int = 2000) -> list[dict]:
    """Agent tool items (command executions, web searches, MCP calls, ...) from codex JSONL."""
    out: list[dict] = []
    for stdout in stdouts:
        for line in (stdout or "").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(ev, dict) or ev.get("type") != "item.completed":
                continue
            item = ev.get("item") if isinstance(ev.get("item"), dict) else {}
            itype = item.get("type") or item.get("item_type") or ""
            if itype in _NON_TOOL_ITEMS:
                continue
            slim = {}
            for k, v in item.items():
                if k == "id":
                    continue
                if isinstance(v, str) and len(v) > limit:
                    v = v[:limit] + f"...[{len(v) - limit} chars cut]"
                slim[k] = v
            out.append(slim)
    return out


# ── parsing / merging ──────────────────────────────────────────────────


def critique_to_dict(c: ReviewCritique) -> dict:
    return {
        "severity": c.severity.value if isinstance(c.severity, SeverityLevel) else str(c.severity),
        "category": c.category,
        "description": c.description,
        "required_fix": c.required_fix,
        "suggested_experiment": c.suggested_experiment,
    }


def dict_to_critique(d: dict) -> ReviewCritique:
    return ReviewCritique(
        severity=SeverityLevel(str(d.get("severity", "medium")).lower()),
        category=d.get("category") or "general",
        description=d.get("description") or "",
        required_fix=d.get("required_fix"),
        suggested_experiment=d.get("suggested_experiment"),
    )


def parse_output(raw_output: str) -> dict:
    """Parse a reviewer output the way the consensus panel does.

    Primary: ``parse_lens_output(strip_thinking_traces(raw))`` (what
    ``ConsensusReviewer.analyze_lenses`` does). The parse of the unstripped
    text is recorded alongside so stripping side effects can be audited.
    """
    stripped = strip_thinking_traces(raw_output or "")
    parsed = parse_lens_output(stripped)
    raw_parsed = parse_lens_output(raw_output or "")
    return {
        "method": parsed.method,
        "detail": parsed.detail,
        "n_critiques": len(parsed.critiques),
        # Contract health (see feedback.ParsedCritiques): entries in the
        # counted block, entries that could not be read, contract blocks
        # present (the contract allows one), distinct critiques found only in
        # earlier blocks, and why a block could not count as a review.
        "n_raw_items": parsed.n_raw_items,
        "n_dropped": parsed.n_dropped,
        "n_contract_blocks": parsed.n_contract_blocks,
        "dropped_earlier": parsed.dropped_earlier,
        "invalid_reason": parsed.invalid_reason,
        "overall_assessment": parsed.overall_assessment,
        "grade": parsed.grade,
        "critiques": [critique_to_dict(c) for c in parsed.critiques],
        "unstripped_method": raw_parsed.method,
        "unstripped_n_critiques": len(raw_parsed.critiques),
        "stripping_changed_parse": (
            raw_parsed.method != parsed.method or len(raw_parsed.critiques) != len(parsed.critiques)
        ),
    }


def freeze_time_threshold(method: str, threshold: Optional[float] = None) -> Optional[float]:
    """``threshold``, or the method's default at benchmark freeze time
    (jaccard 0.5, tfidf 0.3, embedding 0.7, llm None) when it is None."""
    if threshold is not None:
        return float(threshold)
    return similarity.UNCALIBRATED_THRESHOLDS[similarity.validate_method(method)]


def merge_calls(
    critiques_by_call: dict[str, list[dict]],
    calls: Sequence[str],
    *,
    method: str = DEFAULT_MERGE_METHOD,
    threshold: Optional[float] = None,
    consensus_threshold: int = DEFAULT_CONSENSUS_THRESHOLD,
    lens_identity: str = "tag",
    groups: Optional[Sequence[Sequence[int]]] = None,
    same_lens_merge: bool = DEFAULT_SAME_LENS_MERGE,
) -> tuple[MergeOutcome, list[tuple[str, int]]]:
    """Merge the critiques of ``calls`` with the platform merge.

    ``lens_identity='tag'`` treats every call as a distinct reviewer (so a
    self-ensemble can escalate exactly like a panel, = platform
    ``consensus_lens_identity="call"``); ``'role'`` uses the lens role name,
    as a platform panel with repeated roles does by default. The same-lens
    merge rule (``same_lens_merge``, platform default False) always applies
    per CALL, so critiques of one call never merge with each other while
    copies of one role can still merge across calls. ``threshold`` None means
    the method's freeze-time default (:func:`freeze_time_threshold`). Returns
    the outcome and the index ``pair -> (call tag, critique index)``.
    """
    threshold = freeze_time_threshold(method, threshold)
    pairs: list[tuple[str, ReviewCritique]] = []
    index: list[tuple[str, int]] = []
    block: list[str] = []
    for tag in calls:
        lens = tag if lens_identity == "tag" else CALL_SPECS[tag][0]
        for j, c in enumerate(critiques_by_call.get(tag, [])):
            pairs.append((lens, dict_to_critique(c)))
            index.append((tag, j))
            block.append(tag)
    outcome = merge_parsed_critiques(
        pairs, method=method, threshold=threshold,
        consensus_threshold=consensus_threshold, groups=groups,
        same_lens_merge=same_lens_merge, block_keys=block,
    )
    return outcome, index


# ── record paths ────────────────────────────────────────────────────────


def review_path(data_dir: Path, model: str, item: str, repeat: int, tag: str) -> Path:
    return Path(data_dir) / "reviews" / model_dirname(model) / item / f"r{repeat}" / f"{tag}.json"


def judgment_path(data_dir: Path, judge_model: str, model: str, item: str, repeat: int) -> Path:
    return (Path(data_dir) / "judgments" / model_dirname(judge_model)
            / model_dirname(model) / item / f"r{repeat}.json")


def load_review(data_dir: Path, model: str, item: str, repeat: int, tag: str) -> Optional[dict]:
    p = review_path(data_dir, model, item, repeat, tag)
    if not p.exists():
        return None
    try:
        return read_json(p)
    except (OSError, json.JSONDecodeError):
        return None


def discover_reviews(data_dir: Path) -> list[dict]:
    """All cached review records (any model / item / repeat / tag)."""
    root = Path(data_dir) / "reviews"
    out = []
    if not root.exists():
        return out
    for p in sorted(root.glob("*/*/r*/*.json")):
        if p.name.startswith("."):
            continue
        try:
            out.append(read_json(p))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def unit_records(data_dir: Path, model: str, item: str, repeat: int,
                 calls: Iterable[str] = ALL_CALLS) -> dict[str, Optional[dict]]:
    return {tag: load_review(data_dir, model, item, repeat, tag) for tag in calls}


def unit_complete(records: dict[str, Optional[dict]]) -> bool:
    return all(r is not None and r.get("success") for r in records.values())


def append_log(path: Path, line: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line.rstrip("\n") + "\n")
