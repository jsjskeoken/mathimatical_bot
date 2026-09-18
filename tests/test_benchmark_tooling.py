"""Regression tests for the benchmark TOOLING itself (not production code).

The 51b16f4-era baseline extraction used subprocess(text=True), which decodes
git output with the Windows locale codec (cp1252) and crashed on the real
Windows run with:

    UnicodeDecodeError: 'charmap' codec can't decode byte 0x8f

because bot_core.py is UTF-8 and contains '←' / '→' (E2 86 90 / E2 86 92 —
bytes cp1252 leaves undefined). The same era leaked temp files into the
working tree: a crashed run left tmpXXXXXXXX_51b16f4.py (tempfile.mkstemp's
default 'tmp' + 8 random chars + the rev) sitting untracked in the repo
root. These tests pin both contracts:

  1. git output is captured as BYTES and decoded explicitly as UTF-8;
  2. genuinely mangled (non-UTF-8) content fails loudly with context —
     never benchmarked mangled, never silently replaced;
  3. genuine git failures (bad revision, missing object) surface with
     git's stderr instead of being swallowed;
  4. ALL benchmark temp state lives in a TemporaryDirectory OUTSIDE the
     repository, removed on success AND exceptions — no tmp*.py file can
     ever appear in the working tree, and every run is audited for it.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_PATH = Path(REPO).resolve()
sys.path.insert(0, REPO)

import conftest                      # noqa: E402  (platform stubs)
conftest._install_stubs()


def _load_benchmark_ocr():
    """Import benchmarks/benchmark_ocr.py as a module for direct testing."""
    if "benchmark_ocr" in sys.modules:
        return sys.modules["benchmark_ocr"]
    spec = importlib.util.spec_from_file_location(
        "benchmark_ocr",
        os.path.join(REPO, "benchmarks", "benchmark_ocr.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_ocr"] = mod
    spec.loader.exec_module(mod)
    return mod


bo = _load_benchmark_ocr()


def _repo_snapshot():
    """Every file under the repo (rel paths), skipping .git/ and
    __pycache__/ — the same scope the benchmark's leftover tripwire uses."""
    seen = set()
    for dirpath, dirnames, filenames in os.walk(REPO):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__")]
        for fn in filenames:
            seen.add(os.path.relpath(os.path.join(dirpath, fn), REPO))
    return seen


def test_decode_git_output_is_utf8_not_charmap():
    """'←'/'→' bytes are undefined in cp1252 — text=True crashed on them
    (the exact Windows failure). Explicit UTF-8 decoding round-trips them."""
    text = "[CORE] LUT ← '7+6' → 13  (7 total)"
    assert bo.decode_git_output(text.encode("utf-8"), "ctx") == text


def test_decode_git_output_rejects_mangled_bytes():
    """Non-UTF-8 bytes must fail loudly with context, never be mangled
    into a silently different baseline source."""
    with pytest.raises(RuntimeError, match="not valid UTF-8"):
        bo.decode_git_output(b"\x8f\x8f", "bot_core.py at deadbeef")


def _git_available() -> bool:
    """Forensic-audit PART 26: these two tests exercise the benchmark's
    baseline extraction, which legitimately drives `git show` against the
    repository. Outside a git checkout (extracted copy, zip download) they
    must SKIP — explicitly marked environment-specific — not fail."""
    import subprocess
    try:
        r = subprocess.run(["git", "rev-parse", "--git-dir"], cwd=REPO,
                           capture_output=True)
        return r.returncode == 0
    except OSError:
        return False


_requires_git = pytest.mark.skipif(
    not _git_available(), reason="requires a git checkout (benchmark tooling "
    "legitimately reads repository history)")


@_requires_git
def test_extract_baseline_source_is_strict_utf8():
    """Real extraction path: HEAD's bot_core.py comes back as a str with its
    UTF-8 punctuation intact (cp1252 would have crashed or mangled it)."""
    src = bo.extract_baseline_source("HEAD")
    assert "class BotCore" in src
    assert "LUT ←" in src              # the exact chars that broke cp1252


def test_extract_baseline_source_surfaces_git_failure():
    """A bad revision must raise with git's stderr — never an empty/None
    source reaching the file write."""
    with pytest.raises(RuntimeError, match="git show"):
        bo.extract_baseline_source("0000000000000000000000000000000000000000")


# ── temp-file hygiene: no tmp*.py may ever reach the working tree ───────────

def test_bench_dir_is_outside_repo_and_self_cleaning():
    """open_bench_dir() must resolve OUTSIDE the repository and remove its
    whole tree when the context exits — success or exception. The old
    mkdtemp/rmtree pair was only reached on the success path."""
    with bo.open_bench_dir() as bench_dir:
        p = Path(bench_dir).resolve()
        assert not p.is_relative_to(REPO_PATH), \
            "benchmark temp dir must never live inside the repo"
        assert p.name.startswith("mathbot_bench_")
        probe = p / "probe.py"
        probe.write_text("x = 1", encoding="utf-8")
    assert not probe.exists(), "TemporaryDirectory must clean up on exit"


@_requires_git
def test_load_baseline_core_writes_module_only_inside_bench_dir(tmp_path):
    """The extracted baseline module goes into the given bench dir (outside
    the repo) and the working tree stays byte-for-byte untouched."""
    before = _repo_snapshot()
    core, path = bo.load_baseline_core("HEAD", str(tmp_path))
    try:
        p = Path(path).resolve()
        assert p.is_relative_to(tmp_path.resolve())
        assert not p.is_relative_to(REPO_PATH)
        assert p.name == "baseline_bot_core_HEAD.py"
        assert p.exists()
    finally:
        bo.drain_lut_writes(core)      # no async LUT save outlives the test
    assert _repo_snapshot() == before, "repo working tree must be untouched"


def test_bad_baseline_rev_leaves_no_trace(tmp_path):
    """A failed extraction must not write ANY file — not in the bench dir,
    not in the repo. This is the crash path that used to leak tmp*.py."""
    before = _repo_snapshot()
    with pytest.raises(RuntimeError, match="git show"):
        bo.load_baseline_core(
            "0000000000000000000000000000000000000000", str(tmp_path))
    assert list(tmp_path.iterdir()) == [], "no temp file may survive a failure"
    assert _repo_snapshot() == before, "repo working tree must be untouched"


def test_scan_temp_leftovers_catches_exact_reported_leftover(tmp_path):
    """The detector recognises the EXACT leftover class the Windows run
    reported (tmpj39knlqs_51b16f4.py) plus stray mathbot_bench_* trees,
    skips .git/ and __pycache__/, ignores real files — and the REAL repo
    must be clean right now."""
    (tmp_path / "tmpj39knlqs_51b16f4.py").write_text("x = 1", encoding="utf-8")
    (tmp_path / "bot_core.py").write_text("# real file", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "tmpignored_51b16f4.py").write_text(
        "x = 1", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "tmpignored2_51b16f4.py").write_text(
        "x = 1", encoding="utf-8")
    (tmp_path / "mathbot_bench_zz9xqq77").mkdir()
    assert bo.scan_temp_leftovers(str(tmp_path)) == [
        "mathbot_bench_zz9xqq77", "tmpj39knlqs_51b16f4.py"]
    # the live repository must satisfy the same invariant the benchmark
    # enforces on itself after every run
    assert bo.scan_temp_leftovers() == [], (
        "repo contains benchmark temp leftovers — delete them; the fixed "
        "benchmark can no longer create them")
