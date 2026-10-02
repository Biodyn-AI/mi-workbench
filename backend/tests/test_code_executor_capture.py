"""Regression tests: results.json capture in the verified-execution report.

The executor prompt (prompts/executor/mi_executor_code.yaml) asks for every
reported number to be written to ``results.json`` and the reviewers are told
to check every number against the execution report.  Two defects made that
check impossible for realistic analyses:

1. File capture read files in sort order with one shared text budget, so
   text files whose names sort before ``results.json`` (``MECH.md``,
   ``per_head.csv``, ...) could exhaust the budget and ``results.json`` was
   reported as "text capture budget exhausted" (or truncated).
2. The per-file / total capture caps (4000 / 16000 bytes) could not be raised
   from the run config, so a results.json above 4 KB was always truncated.
"""
import asyncio
import json
import os

import pytest

from backend.orchestrator import code_executor as ce
from backend.orchestrator.code_executor import CodeExecutor
from backend.orchestrator.sandbox import capture as cap


def run(coro):
    return asyncio.run(coro)


def _write(path, text):
    with open(path, "w") as fh:
        fh.write(text)


def test_capture_files_reads_priority_file_first(tmp_path):
    for i in range(5):
        _write(tmp_path / f"a{i}_table.csv", "x," * 3000)  # 6000 bytes each
    _write(tmp_path / "MECH.md", "# notes\n" * 600)
    results = {"auroc_layer_3": 0.6734, "n_edges": 424}
    _write(tmp_path / "results.json", json.dumps(results))

    # Legacy behaviour (no priority): the budget is gone before results.json.
    files, _ = cap.capture_files(str(tmp_path), max_text_bytes=4000,
                                 max_total_text_bytes=16000)
    legacy = {f.path: f for f in files}
    assert legacy["results.json"].content is None
    assert legacy["results.json"].skipped == "text capture budget exhausted"

    files, _ = cap.capture_files(str(tmp_path), max_text_bytes=4000,
                                 max_total_text_bytes=16000,
                                 priority=("results.json",))
    assert files[0].path == "results.json"
    assert json.loads(files[0].content) == results
    assert files[0].truncated is False
    # The other files are still listed.
    assert {f.path for f in files} == {f.path for f in legacy.values()}


def test_capture_priority_applies_only_to_top_level(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    _write(sub / "results.json", "{}")
    _write(tmp_path / "b.txt", "b")
    files, _ = cap.capture_files(str(tmp_path), priority=("results.json",))
    assert [f.path for f in files] == ["b.txt", "sub/results.json"]


def test_capture_priority_respects_per_file_cap(tmp_path):
    _write(tmp_path / "results.json", json.dumps({"k%d" % i: i for i in range(2000)}))
    files, _ = cap.capture_files(str(tmp_path), max_text_bytes=1000,
                                 max_total_text_bytes=5000, priority=("results.json",))
    assert files[0].path == "results.json" and files[0].truncated is True
    assert len(files[0].content.encode()) <= 1000


def test_executor_report_shows_results_json_despite_other_files(tmp_path):
    code = (
        "```python\n"
        "import json\n"
        "for i in range(6):\n"
        "    open(f'a{i}_per_head.csv', 'w').write('0.1,' * 2000)\n"
        "json.dump({'auroc_best_layer': 0.6731, 'n_edges': 424}, open('results.json', 'w'))\n"
        "print('done')\n"
        "```\n"
    )
    ex = CodeExecutor(timeout_seconds=30)
    report = run(ex.run_artifact(code))
    assert report.passed == 1
    rj = [f for f in report.blocks[0].files if f.path == "results.json"][0]
    assert json.loads(rj.content) == {"auroc_best_layer": 0.6731, "n_edges": 424}
    fb = report.to_feedback()
    assert "--- results.json ---" in fb and "0.6731" in fb


def test_capture_caps_configurable_from_run_config():
    ex = CodeExecutor.from_run_config({
        "code_execution_capture_text_bytes": 24000,
        "code_execution_capture_total_bytes": 48000,
        "code_execution_capture_max_files": 80,
    })
    assert (ex.capture_text_bytes, ex.capture_total_bytes, ex.capture_max_files) == (
        24000, 48000, 80)
    cfg = ex.config_dict()
    assert cfg["capture_text_bytes"] == 24000 and cfg["capture_total_bytes"] == 48000
    # Defaults unchanged.
    d = CodeExecutor.from_run_config({})
    assert (d.capture_text_bytes, d.capture_total_bytes, d.capture_max_files) == (4000, 16000, 50)
    for bad in (-1, "many", True):
        with pytest.raises(ValueError):
            CodeExecutor.from_run_config({"code_execution_capture_text_bytes": bad})


def test_large_results_json_not_truncated_with_raised_cap():
    payload = {f"layer_{l}_head_{h}_auroc": round(0.5 + l * 0.001 + h * 0.0001, 6)
               for l in range(12) for h in range(12)}
    code = ("```python\nimport json\n"
            f"json.dump({payload!r}, open('results.json', 'w'), indent=1)\n```\n")
    small = run(CodeExecutor(timeout_seconds=30).run_artifact(code))
    rj = [f for f in small.blocks[0].files if f.path == "results.json"][0]
    assert rj.truncated is True  # default 4000-byte cap
    big = run(CodeExecutor.from_run_config({
        "code_execution_timeout": 30,
        "code_execution_capture_text_bytes": 24000,
        "code_execution_capture_total_bytes": 48000,
    }).run_artifact(code))
    rj = [f for f in big.blocks[0].files if f.path == "results.json"][0]
    assert rj.truncated is False and json.loads(rj.content) == payload


def test_new_keys_documented():
    for key in ("code_execution_capture_text_bytes", "code_execution_capture_total_bytes",
                "code_execution_capture_max_files"):
        assert key in CodeExecutor.RUN_CONFIG_KEYS
        assert f"``{key}``" in ce.__doc__


# ── Complete outputs (code_execution_persist_dir) ────────────────────────
#
# The report keeps only a head/tail of stdout and a capped preview of the
# written files, and the work directory is deleted afterwards, so the
# numbers a write-up reports could not be audited against the complete
# executed output. With code_execution_persist_dir every unit keeps them.


def _persist_dirs(root):
    return sorted(p for p in os.listdir(root) if not p.startswith("._"))


def test_persist_dir_keeps_complete_stdout_and_files(tmp_path):
    persist = tmp_path / "persist"
    payload = {f"k{i}": i / 7 for i in range(3000)}            # > 24 KB of JSON
    code = ("```python\nimport json\n"
            "for i in range(4000):\n    print('line', i, i / 3)\n"
            f"json.dump({payload!r}, open('results.json', 'w'))\n"
            "import os\nos.mkdir('sub')\nopen('sub/table.csv', 'w').write('a,b\\n1,2\\n')\n```\n")
    ex = CodeExecutor.from_run_config({"code_execution_timeout": 60,
                                       "code_execution_persist_dir": str(persist)})
    report = run(ex.run_artifact(code))
    b = report.blocks[0]
    assert b.success and b.stdout_truncated                     # the report is capped
    unit = b.persisted_to
    assert unit and os.path.dirname(unit) == os.path.realpath(persist)
    full = open(os.path.join(unit, "stdout.txt")).read()
    assert full.count("\n") == 4000 and "line 3999 1333.0" in full
    assert json.load(open(os.path.join(unit, "work", "results.json"))) == payload
    assert open(os.path.join(unit, "work", "sub", "table.csv")).read() == "a,b\n1,2\n"
    assert "for i in range(4000)" in open(os.path.join(unit, "code.py")).read()
    assert report.to_dict()["blocks"][0]["persisted_to"] == unit
    assert ex.config_dict()["persist_dir"] == os.path.realpath(persist)


def test_persist_never_follows_planted_symlinks(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET-CONTENT")
    persist = tmp_path / "persist"
    code = ("```python\nimport os\n"
            f"os.symlink({str(secret)!r}, 'leak.txt')\n"
            "open('ok.txt', 'w').write('fine')\n```\n")
    ex = CodeExecutor(timeout_seconds=30, persist_dir=str(persist))
    b = run(ex.run_artifact(code)).blocks[0]
    work = os.path.join(b.persisted_to, "work")
    assert [f for f in sorted(os.listdir(work)) if not f.startswith("._")] == ["ok.txt"]
    assert any("leak.txt" in n for n in b.notes)
    for dirpath, _, files in os.walk(b.persisted_to):
        for f in files:
            assert "TOP-SECRET-CONTENT" not in open(os.path.join(dirpath, f), errors="ignore").read()


def test_persist_byte_budget_is_enforced(tmp_path):
    persist = tmp_path / "persist"
    code = ("```python\nimport sys\nsys.stdout.write('x' * 3_000_000)\n"
            "open('big.bin', 'wb').write(b'y' * 3_000_000)\n```\n")
    ex = CodeExecutor(timeout_seconds=60, persist_dir=str(persist), persist_max_mb=1)
    b = run(ex.run_artifact(code)).blocks[0]
    assert os.path.getsize(os.path.join(b.persisted_to, "stdout.txt")) == 1024 * 1024
    assert os.path.getsize(os.path.join(b.persisted_to, "work", "big.bin")) == 1024 * 1024
    assert any("persisted stdout cut" in n for n in b.notes)
    assert any("big.bin cut" in n for n in b.notes)


def test_persist_off_by_default(tmp_path):
    b = run(CodeExecutor(timeout_seconds=30).run_artifact("```python\nprint(1)\n```")).blocks[0]
    assert b.persisted_to is None
    assert CodeExecutor().config_dict()["persist_dir"] is None
    assert "code_execution_persist_dir" in CodeExecutor.RUN_CONFIG_KEYS
    assert "``code_execution_persist_dir``" in ce.__doc__


def test_persist_works_under_sandbox_exec(tmp_path):
    from backend.orchestrator.sandbox import seatbelt
    ok, why = seatbelt.sandbox_exec_available()
    if not ok:
        pytest.skip(why)
    persist = tmp_path / "persist"
    ex = CodeExecutor(backend="sandbox_exec", timeout_seconds=30, persist_dir=str(persist))
    b = run(ex.run_artifact("```python\nprint('hello')\nopen('r.json','w').write('{}')\n```")).blocks[0]
    assert b.success
    assert open(os.path.join(b.persisted_to, "stdout.txt")).read() == "hello\n"
    assert os.path.isfile(os.path.join(b.persisted_to, "work", "r.json"))
    # The sandboxed code itself cannot write into the persist directory.
    code = f"```python\nopen({str(persist / 'evil.txt')!r}, 'w').write('x')\n```"
    b2 = run(ex.run_artifact(code)).blocks[0]
    assert not b2.success and not (persist / "evil.txt").exists()
