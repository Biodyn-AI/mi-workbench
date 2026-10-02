"""Bounded capture of child output and of files the code wrote.

* Output streams are drained incrementally and kept within a fixed byte
  budget, so a child printing gigabytes cannot exhaust the backend's memory.
  A stream keeps its first ``head`` bytes and its last ``limit - head`` bytes
  (head + tail, with an omission marker in between), so the end of a
  traceback stays visible; a separate bounded rolling tail of the raw bytes
  (``DIAG_TAIL_BYTES``) is kept for failure classification.
* File capture never follows symlinks, never opens hard-linked or
  non-regular files (FIFOs, sockets, devices), and re-checks the opened file
  descriptor against the ``lstat`` result, so code confined by the
  ``sandbox_exec`` backend cannot use the (unconfined) backend to read files
  outside its sandbox via a link planted in its work directory.
"""
from __future__ import annotations

import asyncio
import os
import stat
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

TEXT_EXTENSIONS = {
    ".json", ".jsonl", ".txt", ".csv", ".tsv", ".md", ".log", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".py", ".r", ".tex", ".html", ".xml", ".svg",
}


#: Raw bytes kept from the end of every stream for failure classification
#: (exception type, errno, traceback line numbers), independent of the
#: display limit.
DIAG_TAIL_BYTES = 16384


def truncation_marker(total: int, kept: int) -> str:
    return f"\n...[truncated: kept {kept} of {total} bytes]"


def head_tail_marker(total: int, head: int, tail: int) -> str:
    return (f"\n...[truncated: {total - head - tail} bytes omitted; kept first {head} "
            f"and last {tail} of {total} bytes]...\n")


def utf8_suffix(data: bytes, limit: int) -> bytes:
    """Longest suffix of ``data`` of at most ``limit`` bytes that starts on a
    UTF-8 character boundary."""
    if limit <= 0:
        return b""
    if len(data) <= limit:
        tail = data
    else:
        tail = data[-limit:]
    i = 0
    while i < len(tail) and (tail[i] & 0xC0) == 0x80:
        i += 1
    return tail[i:]


def utf8_prefix(data: bytes, limit: int) -> bytes:
    """Longest prefix of ``data`` of at most ``limit`` bytes that does not
    split a UTF-8 multi-byte sequence."""
    if len(data) <= limit:
        return data
    cut = max(0, limit)
    # Back off over continuation bytes (10xxxxxx) to a character start.
    while cut > 0 and (data[cut] & 0xC0) == 0x80:
        cut -= 1
    return data[:cut]


def truncate_output(data: bytes, total: int, limit: int) -> tuple[str, bool]:
    """Decode the retained bytes; add a marker if ``total`` exceeded ``limit``.

    The decoded text (without the marker) is at most ``limit`` bytes when
    re-encoded as UTF-8 (undecodable bytes become U+FFFD, which may expand).
    """
    truncated = total > limit
    kept = utf8_prefix(data, limit) if truncated else data[:limit]
    text = kept.decode("utf-8", "replace")
    if truncated:
        text += truncation_marker(total, len(kept))
    return text, truncated


class CappedStream:
    """Drain an ``asyncio.StreamReader`` keeping at most ``limit`` bytes.

    ``head`` (default ``limit``: head only, the legacy behaviour) is how many
    of the ``limit`` bytes come from the start of the stream; the remaining
    ``limit - head`` come from its end. Memory use is bounded by
    ``limit + 1 + max(limit - head, DIAG_TAIL_BYTES)``.
    """

    def __init__(self, limit: int, head: Optional[int] = None,
                 diag_tail: int = DIAG_TAIL_BYTES, sink=None,
                 sink_limit: Optional[int] = None):
        # ``sink``: optional binary file that receives the raw stream (at most
        # ``sink_limit`` bytes; ``sink_truncated`` tells whether more came).
        self.sink = sink
        self.sink_limit = None if sink_limit is None else max(0, int(sink_limit))
        self.sink_written = 0
        self.sink_truncated = False
        self.sink_error: Optional[str] = None
        self.limit = max(0, int(limit))
        self.head = self.limit if head is None else max(0, min(int(head), self.limit))
        self.tail = self.limit - self.head
        # One byte of look-ahead so truncation can tell whether byte
        # ``limit`` starts a new UTF-8 character.
        self._keep = self.limit + 1
        self.buf = bytearray()
        self._tail_keep = max(self.tail, int(diag_tail))
        self._tail = bytearray()
        self.total = 0

    def _add(self, chunk: bytes) -> None:
        self.total += len(chunk)
        if self.sink is not None and self.sink_error is None:
            part = chunk
            if self.sink_limit is not None:
                room_sink = self.sink_limit - self.sink_written
                if room_sink < len(chunk):
                    self.sink_truncated = True
                part = chunk[:max(0, room_sink)]
            if part:
                try:
                    self.sink.write(part)
                    self.sink_written += len(part)
                except OSError as exc:  # never fail the execution over a copy
                    self.sink_error = str(exc)
        room = self._keep - len(self.buf)
        if room > 0:
            self.buf += chunk[:room]
        if self._tail_keep:
            self._tail += chunk[-self._tail_keep:]
            if len(self._tail) > self._tail_keep:
                del self._tail[:len(self._tail) - self._tail_keep]

    async def drain(self, reader: Optional[asyncio.StreamReader]) -> None:
        if reader is None:
            return
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                return
            self._add(chunk)

    def feed(self, data: bytes) -> None:
        """Synchronous variant used for synthetic output."""
        self._add(data)

    @property
    def truncated(self) -> bool:
        return self.total > self.limit

    def head_text(self) -> str:
        """The first (up to ``limit + 1``) bytes, decoded."""
        return bytes(self.buf).decode("utf-8", "replace")

    def diag_text(self) -> str:
        """The whole stream if it fit, else its last ``DIAG_TAIL_BYTES`` raw
        bytes (for classifying a failure from the end of a traceback)."""
        if self.total <= len(self.buf):
            return bytes(self.buf[:self.total]).decode("utf-8", "replace")
        return utf8_suffix(bytes(self._tail), self._tail_keep).decode("utf-8", "replace")

    def result(self) -> tuple[str, bool]:
        if self.tail == 0 or self.total <= self.limit:
            return truncate_output(bytes(self.buf), self.total, self.limit)
        head = utf8_prefix(bytes(self.buf), self.head) if self.head else b""
        tail = utf8_suffix(bytes(self._tail), self.tail)
        text = (head.decode("utf-8", "replace")
                + head_tail_marker(self.total, len(head), len(tail))
                + tail.decode("utf-8", "replace"))
        return text, True


@dataclass
class CapturedFile:
    """A file found in the work directory after execution."""
    path: str                      # relative to the work dir, '/'-separated
    size_bytes: int
    content: Optional[str] = None  # UTF-8 text, only for small text files
    truncated: bool = False        # content cut at the per-file cap
    skipped: Optional[str] = None  # why content was not read, if it was not

    def to_dict(self) -> dict:
        return asdict(self)


def _is_os_metadata(name: str, siblings: set) -> bool:
    """macOS Finder/AppleDouble files (``.DS_Store``; ``._X`` next to ``X``,
    created on non-HFS volumes such as exFAT external drives)."""
    if name == ".DS_Store":
        return True
    return name.startswith("._") and name[2:] in siblings


def _looks_text(sample: bytes, name: str) -> bool:
    if b"\x00" in sample:
        return False
    if os.path.splitext(name)[1].lower() in TEXT_EXTENSIONS:
        return True
    try:
        sample.decode("utf-8")
        return True
    except UnicodeDecodeError:
        # A multi-byte character may be cut at the sample boundary.
        try:
            utf8_prefix(sample, len(sample) - 1).decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False


def _read_regular(name: str, st: os.stat_result, cap: int,
                  dir_fd: Optional[int]) -> tuple[Optional[bytes], Optional[str]]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(name, flags, dir_fd=dir_fd) if dir_fd is not None else os.open(name, flags)
    except OSError as exc:
        return None, f"open failed: {exc.strerror}"
    try:
        fst = os.fstat(fd)
        if (not stat.S_ISREG(fst.st_mode) or fst.st_ino != st.st_ino
                or fst.st_dev != st.st_dev or fst.st_nlink != 1):
            return None, "file changed while capturing"
        chunks = []
        remaining = cap
        while remaining > 0:
            chunk = os.read(fd, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks), None
    finally:
        os.close(fd)


def _walk(root: str):
    """Yield ``(dirpath, dirnames, filenames, dir_fd)`` without following links.

    ``os.fwalk`` holds a descriptor for every directory it visits, and all
    file operations below are relative to that descriptor.  Swapping a
    directory for a symlink while the walk runs (for example by a process
    that escaped the process group) therefore cannot redirect a read
    outside the work directory.  Where ``fwalk`` is unavailable, it falls
    back to ``os.walk`` with path-based access.
    """
    if os.open in os.supports_dir_fd and hasattr(os, "fwalk"):
        yield from os.fwalk(root, follow_symlinks=False)
    else:  # pragma: no cover - platforms without *at() calls
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            yield dirpath, dirnames, filenames, None


def capture_files(
    work_dir: str,
    *,
    max_files: int = 50,
    max_text_bytes: int = 16384,
    max_total_text_bytes: int = 65536,
    max_entries: int = 2000,
    max_depth: int = 16,
    priority: Sequence[str] = (),
) -> tuple[list[CapturedFile], list[str]]:
    """List files under ``work_dir`` and read small text files.

    Returns ``(files, notes)``.  Symlinks, hard links and non-regular files
    are listed with ``skipped`` set and are never opened.  The walk holds
    one directory descriptor per level, so it never descends more than
    ``max_depth`` levels below ``work_dir``, and directories count towards
    ``max_entries`` (a deep or wide tree cannot exhaust the backend's
    descriptors or time).

    ``priority`` names files in ``work_dir`` itself (not in subdirectories)
    that are listed and read before every other file, so that files written
    earlier in sort order cannot exhaust ``max_total_text_bytes`` (or
    ``max_files``) before them; ``max_text_bytes`` still applies to them.
    """
    first = [str(p) for p in priority]
    files: list[CapturedFile] = []
    notes: list[str] = []
    total_text = 0
    seen = 0
    root = os.path.realpath(work_dir)
    depth_noted = False
    for dirpath, dirnames, filenames, dir_fd in _walk(root):
        dirnames.sort()
        names = set(filenames) | set(dirnames)
        rel_dir = os.path.relpath(dirpath, root)
        depth = 0 if rel_dir in (".", "") else rel_dir.count(os.sep) + 1
        seen += len(dirnames)
        if seen > max_entries:
            notes.append(f"file capture stopped after {max_entries} entries")
            return files, notes
        if depth >= max_depth and dirnames:
            if not depth_noted:
                notes.append(f"file capture did not descend below depth {max_depth}")
                depth_noted = True
            dirnames[:] = []

        def _lstat(entry: str) -> os.stat_result:
            if dir_fd is not None:
                return os.stat(entry, dir_fd=dir_fd, follow_symlinks=False)
            return os.lstat(os.path.join(dirpath, entry))

        linked_dirs = []
        for d in dirnames:
            try:
                if stat.S_ISLNK(_lstat(d).st_mode):
                    linked_dirs.append(d)
            except OSError:
                continue
        ordered = sorted(filenames)
        if depth == 0 and first:
            # Top-down walk: the root comes first, so these are read first.
            ordered = ([n for n in first if n in filenames]
                       + [n for n in ordered if n not in first])
        for name in ordered + linked_dirs:
            if _is_os_metadata(name, names):
                continue
            seen += 1
            if seen > max_entries:
                notes.append(f"file capture stopped after {max_entries} entries")
                return files, notes
            if len(files) >= max_files:
                notes.append(f"more than {max_files} files written; list truncated")
                return files, notes
            rel = os.path.relpath(os.path.join(dirpath, name), root).replace(os.sep, "/")
            try:
                st = _lstat(name)
            except OSError:
                continue
            if stat.S_ISLNK(st.st_mode):
                files.append(CapturedFile(rel, 0, skipped="symlink (not followed)"))
                continue
            if not stat.S_ISREG(st.st_mode):
                files.append(CapturedFile(rel, 0, skipped="not a regular file"))
                continue
            entry = CapturedFile(rel, int(st.st_size))
            files.append(entry)
            if st.st_nlink != 1:
                entry.skipped = "hard link (not read)"
                continue
            budget = min(max_text_bytes, max_total_text_bytes - total_text)
            if budget <= 0:
                entry.skipped = "text capture budget exhausted"
                continue
            target = name if dir_fd is not None else os.path.join(dirpath, name)
            data, err = _read_regular(target, st, budget + 1, dir_fd)
            if err:
                entry.skipped = err
                continue
            assert data is not None
            if not _looks_text(data[:8192], name):
                entry.skipped = "binary"
                continue
            if len(data) > budget:
                data = utf8_prefix(data, budget)
                entry.truncated = True
            entry.content = data.decode("utf-8", "replace")
            total_text += len(data)
    return files, notes


def persist_files(
    work_dir: str,
    dest_dir: str,
    *,
    max_total_bytes: int = 256 * 1024 * 1024,
    max_files: int = 2000,
    max_entries: int = 5000,
    max_depth: int = 16,
) -> tuple[int, int, list[str]]:
    """Copy the regular files written under ``work_dir`` to ``dest_dir``.

    Same safety rules as :func:`capture_files`: the walk holds a directory
    descriptor per level and never follows symlinks; symlinks, hard links and
    non-regular files are not copied (listed in the notes); every file is
    opened with ``O_NOFOLLOW`` relative to its directory descriptor and
    re-checked with ``fstat``. At most ``max_total_bytes`` are copied (a file
    that does not fit is cut and noted). Returns ``(files, bytes, notes)``.
    """
    notes: list[str] = []
    copied = total = seen = 0
    root = os.path.realpath(work_dir)
    os.makedirs(dest_dir, exist_ok=True)
    for dirpath, dirnames, filenames, dir_fd in _walk(root):
        dirnames.sort()
        names = set(filenames) | set(dirnames)
        rel_dir = os.path.relpath(dirpath, root)
        depth = 0 if rel_dir in (".", "") else rel_dir.count(os.sep) + 1
        seen += len(dirnames)
        if depth >= max_depth and dirnames:
            notes.append(f"persist: not descending below depth {max_depth}")
            dirnames[:] = []
        for name in sorted(filenames):
            if _is_os_metadata(name, names):
                continue
            seen += 1
            if seen > max_entries or copied >= max_files:
                notes.append(f"persist: stopped after {copied} files / {seen} entries")
                return copied, total, notes
            rel = os.path.relpath(os.path.join(dirpath, name), root)
            try:
                st = (os.stat(name, dir_fd=dir_fd, follow_symlinks=False) if dir_fd is not None
                      else os.lstat(os.path.join(dirpath, name)))
            except OSError:
                continue
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
                notes.append(f"persist: not copied (link or not a regular file): {rel}")
                continue
            budget = max_total_bytes - total
            if budget <= 0:
                notes.append(f"persist: byte budget exhausted at {rel}")
                return copied, total, notes
            target = name if dir_fd is not None else os.path.join(dirpath, name)
            data, err = _read_regular(target, st, min(budget, st.st_size + 1), dir_fd)
            if err:
                notes.append(f"persist: {rel}: {err}")
                continue
            assert data is not None
            if len(data) < st.st_size:
                notes.append(f"persist: {rel} cut at {len(data)} of {st.st_size} bytes")
            dest = os.path.join(dest_dir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as fh:
                fh.write(data)
            copied += 1
            total += len(data)
    return copied, total, notes
