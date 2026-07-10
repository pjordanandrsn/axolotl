# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Storage backends for expert-offload block homes.

The offload mechanism (``offload.py``) is agnostic to *where* an evicted block's packed
bytes live; it only needs, at stage time, a CPU tensor to copy onto the GPU. This module
provides that seam:

- :class:`RAMStore` — today's behavior, byte-for-byte: each block's packed tensors are
  homed as separate (pinned) CPU tensors held for the life of the model. ``fetch`` returns
  the home itself (zero-copy), and staging may alias it on CPU targets exactly as before.
- :class:`FileStore` — the homes are written once to a read-only packed file at install
  and dropped from host RAM; ``fetch`` reads a block back into a small set of reusable
  (pinned) staging buffers — via ``O_DIRECT`` when the filesystem supports it (bypassing
  the page cache, so timings are honest), else buffered reads followed by
  ``posix_fadvise(DONTNEED)``. Because staging buffers are reused, FileStore staging
  always copies (``copy_required``), and the H2D copy is synchronous — bytes identical,
  only earlier/later in time. The achieved read mode is recorded on ``FileStore.mode``.

Neither backend touches eviction, the single-resident-slot policy, the forward pre-hook,
or the recompute contract — the seam abstracts only the source of a block's packed bytes.
"""

from __future__ import annotations

import json
import os
import queue
import tempfile
import threading

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_ALIGN = 4096  # O_DIRECT alignment (offset, length, buffer address)
_O_DIRECT = getattr(os, "O_DIRECT", 0)  # Linux-only; 0 (no-op) elsewhere -> buffered fallback
_HAS_FADVISE = hasattr(os, "posix_fadvise")


def _pad(n: int) -> int:
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


def _u8(t: torch.Tensor) -> torch.Tensor:
    """Reinterpret a contiguous tensor's storage as uint8 (works for bf16 etc., which have
    no numpy dtype — the uint8 view is what crosses into ``os.preadv``/``write``)."""
    return t.contiguous().view(torch.uint8)


def _aligned_pinned_u8(nbytes: int, pin: bool) -> torch.Tensor:
    """A uint8 buffer of ``nbytes`` whose data pointer is ``_ALIGN``-aligned. Pinned when
    requested (cudaHostAlloc is page-aligned in practice, but alignment is asserted, with
    an over-allocate-and-slice fallback so O_DIRECT can never see a misaligned buffer)."""
    t = torch.empty(nbytes, dtype=torch.uint8, pin_memory=pin)
    if t.data_ptr() % _ALIGN == 0:
        return t
    t = torch.empty(nbytes + _ALIGN, dtype=torch.uint8, pin_memory=pin)
    off = (-t.data_ptr()) % _ALIGN
    return t[off : off + nbytes]


class _BufferSet:
    """One generation of per-slot staging buffers plus the CUDA event that says when the last
    H2D copies reading them completed.

    Buffers are keyed by **slot index**, never by byte size: a block's slots are read as a group
    and handed out together, so two same-size slots must not share one buffer (they would clobber
    each other before either was copied out). Reusing a set waits on its event first.
    """

    __slots__ = ("bufs", "event", "pinned")

    def __init__(self) -> None:
        self.bufs: dict[int, torch.Tensor] = {}
        self.event: torch.cuda.Event | None = None
        self.pinned = True

    def buffer(self, slot_idx: int, padded: int, pin: bool) -> torch.Tensor:
        buf = self.bufs.get(slot_idx)
        if buf is None or buf.numel() < padded:
            buf = _aligned_pinned_u8(padded, pin=pin)
            try:
                self.pinned = self.pinned and buf.is_pinned()
            except (RuntimeError, AssertionError):  # pragma: no cover
                self.pinned = False
            self.bufs[slot_idx] = buf
        return buf

    def wait(self) -> None:
        """Block the host until the H2D copies that last read these buffers have finished."""
        if self.event is not None:
            self.event.synchronize()
            self.event = None

    def record(self) -> None:
        """Mark the H2D copies just enqueued on the current stream as the guard for this set."""
        if torch.cuda.is_available():
            ev = torch.cuda.Event()
            ev.record()
            self.event = ev


class _Record:
    __slots__ = ("offset", "nbytes", "padded", "dtype", "shape")

    def __init__(self, offset: int, nbytes: int, dtype: torch.dtype, shape: tuple):
        self.offset = offset
        self.nbytes = nbytes
        self.padded = _pad(nbytes)
        self.dtype = dtype
        self.shape = shape


class RAMStore:
    """Pinned-CPU homes, exactly as before the seam existed."""

    copy_required = False  # homes are persistent; nothing to guard
    mode = "ram"

    def record_stage_event(self) -> None:
        """No-op: RAM homes are never recycled, so an in-flight H2D cannot be clobbered."""
        return None

    def __init__(self, pin: bool = True):
        self.pin = pin
        self._blocks: list[list[torch.Tensor]] = []
        self.pinned = True  # all homes pinned so far (see add_block)

    @staticmethod
    def _to_home(t: torch.Tensor, pin: bool) -> torch.Tensor:
        cpu = t.to("cpu")
        if pin:
            try:
                return cpu.pin_memory()
            except (RuntimeError, AssertionError):  # pragma: no cover
                pass
        return cpu

    def add_block(self, tensors: list[torch.Tensor]) -> int:
        homes = [self._to_home(t, self.pin) for t in tensors]
        try:
            self.pinned = self.pinned and all(t.is_pinned() for t in homes)
        except (RuntimeError, AssertionError):  # pragma: no cover
            self.pinned = False
        self._blocks.append(homes)
        return len(self._blocks) - 1

    def finalize(self) -> None:
        return None

    def fetch(self, block_idx: int, slot_idx: int) -> torch.Tensor:
        return self._blocks[block_idx][slot_idx]

    def state_tensor(self, block_idx: int, slot_idx: int) -> torch.Tensor:
        return self._blocks[block_idx][slot_idx]

    def block_nbytes(self, block_idx: int) -> int:
        return sum(t.numel() * t.element_size() for t in self._blocks[block_idx])

    def fetch_expert(self, block_idx: int, slot_idx: int, expert_id: int) -> torch.Tensor:
        """One expert's row (dim-0 slice) of a homed [num_experts, ...] tensor — for routed-subset
        staging, which copies only the routed experts' rows to the GPU."""
        return self._blocks[block_idx][slot_idx][expert_id]

    def n_experts(self, block_idx: int, slot_idx: int) -> int:
        return int(self._blocks[block_idx][slot_idx].shape[0])


class FileStore:
    """Packed experts on disk; small reusable (pinned) staging buffers in host RAM.

    Written once at install (read-only thereafter). Reads use ``O_DIRECT`` when the path
    supports it (``mode == "odirect"``); otherwise buffered reads with
    ``posix_fadvise(POSIX_FADV_DONTNEED)`` after each block (``mode == "buffered+fadvise"``)
    so repeated epochs cannot silently become page-cache reads. The mode is probed once at
    ``finalize`` and recorded — benchmark artifacts must carry it.
    """

    # Only meaningful for NON-CUDA targets: ``.to("cpu")`` returns the source tensor itself, so a
    # staged param would alias a reused staging buffer. CUDA staging is always a real copy.
    copy_required = True

    _N_SETS = 2  # double buffer: write set B while set A's copies are still in flight

    def __init__(self, store_dir: str | None = None, pin: bool = True):
        self.dir = store_dir or tempfile.mkdtemp(prefix="expert_store_")
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "expert_store.bin")
        self.pin = pin
        self.mode: str | None = None
        self._records: list[list[_Record]] = []
        self._sets = [_BufferSet() for _ in range(self._N_SETS)]
        self._cur = 0
        self._cur_block: int | None = None
        self._wfd: int | None = os.open(
            self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
        )
        self._rfd: int | None = None
        self._end = 0

    @property
    def pinned(self) -> bool:
        return all(s.pinned for s in self._sets) if self.pin else False

    def add_block(self, tensors: list[torch.Tensor]) -> int:
        assert self._wfd is not None, "add_block after finalize"
        recs: list[_Record] = []
        for t in tensors:
            cpu = t.detach().to("cpu")
            u8 = _u8(cpu)
            rec = _Record(self._end, u8.numel(), t.dtype, tuple(t.shape))
            os.pwrite(self._wfd, u8.numpy().tobytes(), rec.offset)
            self._end += rec.padded
            recs.append(rec)
        self._records.append(recs)
        return len(self._records) - 1

    def finalize(self) -> None:
        if self._wfd is None:
            return
        os.ftruncate(self._wfd, self._end)  # pad tail so the last O_DIRECT read is legal
        os.fsync(self._wfd)
        os.close(self._wfd)
        self._wfd = None
        with open(os.path.join(self.dir, "expert_store.json"), "w") as fh:
            json.dump(
                [
                    [
                        {
                            "offset": r.offset,
                            "nbytes": r.nbytes,
                            "dtype": str(r.dtype),
                            "shape": list(r.shape),
                        }
                        for r in block
                    ]
                    for block in self._records
                ],
                fh,
            )
        try:
            if not _O_DIRECT:
                raise OSError("O_DIRECT unavailable on this platform")
            self._rfd = os.open(self.path, os.O_RDONLY | _O_DIRECT)
            probe = _aligned_pinned_u8(_ALIGN, pin=False)
            os.preadv(self._rfd, [memoryview(probe.numpy())], 0)
            self.mode = "odirect"
        except OSError:
            if self._rfd is not None:
                os.close(self._rfd)
            self._rfd = os.open(self.path, os.O_RDONLY)
            self.mode = "buffered+fadvise" if _HAS_FADVISE else "buffered"
        LOG.info(
            f"expert_offload FileStore: {self._end / 1e9:.2f} GB at {self.path} "
            f"(read mode: {self.mode})"
        )

    def _set_for(self, block_idx: int) -> _BufferSet:
        """Rotate to the next buffer set when a new block starts, waiting for the H2D copies that
        last read it. Within one block every slot keeps its own buffer in the same set."""
        if block_idx != self._cur_block:
            self._cur = (self._cur + 1) % self._N_SETS
            self._sets[self._cur].wait()
            self._cur_block = block_idx
        return self._sets[self._cur]

    def record_stage_event(self) -> None:
        """Called after a block's H2D copies are enqueued: guard the set they read from."""
        self._sets[self._cur].record()

    def _read(self, rec: _Record, buf: torch.Tensor) -> None:
        got = os.preadv(self._rfd, [memoryview(buf.numpy())[: rec.padded]], rec.offset)
        if got < rec.nbytes:  # pragma: no cover - short read is a store bug
            raise IOError(f"short read: {got} < {rec.nbytes} at {rec.offset}")
        if self.mode == "buffered+fadvise" and _HAS_FADVISE:
            os.posix_fadvise(
                self._rfd, rec.offset, rec.padded, os.POSIX_FADV_DONTNEED
            )

    def fetch(self, block_idx: int, slot_idx: int) -> torch.Tensor:
        rec = self._records[block_idx][slot_idx]
        buf = self._set_for(block_idx).buffer(slot_idx, rec.padded, self.pin)
        self._read(rec, buf[: rec.padded])
        return buf[: rec.nbytes].view(rec.dtype).view(rec.shape)

    def n_experts(self, block_idx: int, slot_idx: int) -> int:
        return int(self._records[block_idx][slot_idx].shape[0])

    def fetch_expert(self, block_idx: int, slot_idx: int, expert_id: int) -> torch.Tensor:
        """One expert's row of a packed [num_experts, ...] tensor, read directly from disk — the
        routed-subset primitive. Reads only expert ``expert_id``'s byte range (an aligned O_DIRECT
        window when the mode requires it), so flash traffic is the routed subset, not the whole
        block. Returns a tensor shaped like one expert row (``record.shape[1:]``)."""
        rec = self._records[block_idx][slot_idx]
        E = rec.shape[0]
        per = rec.nbytes // E                      # bytes for one expert (dim-0 row)
        start = rec.offset + expert_id * per        # file offset of this expert
        # O_DIRECT needs offset+length+buffer all _ALIGN-aligned; read the covering aligned window.
        win_start = (start // _ALIGN) * _ALIGN
        win_end = ((start + per + _ALIGN - 1) // _ALIGN) * _ALIGN
        win_len = win_end - win_start
        buf = self._expert_buf(win_len)
        got = os.preadv(self._rfd, [memoryview(buf.numpy())[:win_len]], win_start)
        if got < (start - win_start) + per:  # pragma: no cover - short read is a store bug
            raise IOError(f"short expert read: {got} < {(start-win_start)+per}")
        if self.mode == "buffered+fadvise":
            if _HAS_FADVISE:
                os.posix_fadvise(self._rfd, win_start, win_len, os.POSIX_FADV_DONTNEED)
        off = start - win_start
        return buf[off : off + per].view(rec.dtype).view(rec.shape[1:])

    def _expert_buf(self, nbytes: int) -> torch.Tensor:
        b = getattr(self, "_ebuf", None)
        if b is None or b.numel() < nbytes:
            b = _aligned_pinned_u8(nbytes, pin=self.pin)
            self._ebuf = b
        return b

    def state_tensor(self, block_idx: int, slot_idx: int) -> torch.Tensor:
        # Rare path (full-model save): a fresh tensor, and its own buffer, so it never disturbs
        # the rotating staging sets (which an in-flight H2D may still be reading).
        rec = self._records[block_idx][slot_idx]
        buf = _aligned_pinned_u8(rec.padded, pin=False)
        self._read(rec, buf[: rec.padded])
        return buf[: rec.nbytes].view(rec.dtype).view(rec.shape).clone()

    def block_nbytes(self, block_idx: int) -> int:
        return sum(r.nbytes for r in self._records[block_idx])


def make_store(kind: str | None, store_dir: str | None, pin: bool):
    """Resolve the store backend: explicit arg > ``AXOLOTL_EXPERT_OFFLOAD_STORE`` env
    (benchmark/test hook — lets the existing suite run unmodified over FileStore) > ram."""
    kind = kind or os.environ.get("AXOLOTL_EXPERT_OFFLOAD_STORE") or "ram"
    store_dir = store_dir or os.environ.get("AXOLOTL_EXPERT_OFFLOAD_STORE_DIR")
    if kind == "ram":
        return RAMStore(pin=pin)
    if kind == "file":
        return FileStore(store_dir=store_dir, pin=pin)
    raise ValueError(f"unknown expert_offload store: {kind!r} (expected 'ram' or 'file')")


class PrefetchReader:
    """One background reader that loads block ``N±1``'s bytes into a second staging-buffer
    set while block ``N`` computes (Phase C of the store seam; ``expert_offload_prefetch``).

    Deterministic schedule, direction-aware: forward passes stage blocks in ascending
    order, but under ``use_reentrant=False`` gradient checkpointing the backward pass
    *recomputes* blocks in descending order — so the predictor follows the observed
    direction of the last two stagings. A prediction can only ever be wrong at the two
    turnaround points; a prefetch that does not match the block actually requested is
    discarded and that staging falls back to the synchronous path. Bytes are identical
    either way — prefetch moves them earlier in time, never changes them.

    The reader owns a SECOND buffer set (``buffers``); the store's own staging buffers
    remain the synchronous path's. Consumption hands the prefetched CPU tensor to the
    caller and swaps the set back for the next prediction, so at most one prefetched
    block exists at a time (host RAM: +1 block; GPU: unchanged — H2D still happens at
    stage time on the requesting thread).
    """

    _N_SETS = 2

    def __init__(self, store, n_blocks: int):
        self.store = store
        self.n_blocks = n_blocks
        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._ready: dict | None = None
        self._last: int | None = None
        self._direction = 1
        self._sets = [_BufferSet() for _ in range(self._N_SETS)]
        self._load_set = 0
        self._inflight_set: int | None = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def record_stage_event(self) -> None:
        """Guard the set whose buffers the just-enqueued H2D copies are reading, so the reader
        cannot overwrite them mid-flight."""
        if self._inflight_set is not None:
            self._sets[self._inflight_set].record()
            self._inflight_set = None

    def _run(self) -> None:
        while True:
            block_idx = self._q.get()
            if block_idx is None:  # pragma: no cover - shutdown
                return
            try:
                # Alternate sets and wait for any H2D still reading the one we are about to fill.
                self._load_set = (self._load_set + 1) % self._N_SETS
                bset = self._sets[self._load_set]
                bset.wait()
                recs = self.store._records[block_idx]
                tensors = []
                for slot_idx, rec in enumerate(recs):
                    buf = bset.buffer(slot_idx, rec.padded, getattr(self.store, "pin", False))
                    self.store._read(rec, buf[: rec.padded])
                    tensors.append(buf[: rec.nbytes].view(rec.dtype).view(rec.shape))
                with self._lock:
                    self._ready = {
                        "block": block_idx,
                        "tensors": tensors,
                        "set": self._load_set,
                    }
            except Exception:  # pragma: no cover - reader must never kill training
                with self._lock:
                    self._ready = None

    def observe_and_predict(self, block_idx: int) -> None:
        """Called after each staging: update direction, kick the next read."""
        if self._last is not None and block_idx != self._last:
            self._direction = 1 if block_idx > self._last else -1
        self._last = block_idx
        nxt = block_idx + self._direction
        if 0 <= nxt < self.n_blocks and self._q.empty():
            with self._lock:
                pending = self._ready
            if pending is None or pending.get("block") != nxt:
                try:
                    self._q.put_nowait(nxt)
                except queue.Full:  # pragma: no cover
                    pass

    def take(self, block_idx: int) -> list[torch.Tensor] | None:
        """The prefetched tensors for ``block_idx`` if (and only if) the prediction matched;
        None otherwise. Never blocks on the reader."""
        with self._lock:
            ready = self._ready
            if ready is not None and ready["block"] == block_idx:
                self._ready = None
                return ready["tensors"]
        return None
