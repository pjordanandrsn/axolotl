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
import tempfile

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_ALIGN = 4096  # O_DIRECT alignment (offset, length, buffer address)


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

    copy_required = False  # staging may alias the home (CPU target), as today
    mode = "ram"

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


class FileStore:
    """Packed experts on disk; small reusable (pinned) staging buffers in host RAM.

    Written once at install (read-only thereafter). Reads use ``O_DIRECT`` when the path
    supports it (``mode == "odirect"``); otherwise buffered reads with
    ``posix_fadvise(POSIX_FADV_DONTNEED)`` after each block (``mode == "buffered+fadvise"``)
    so repeated epochs cannot silently become page-cache reads. The mode is probed once at
    ``finalize`` and recorded — benchmark artifacts must carry it.
    """

    copy_required = True  # staging buffers are reused; staged params must never alias them

    def __init__(self, store_dir: str | None = None, pin: bool = True):
        self.dir = store_dir or tempfile.mkdtemp(prefix="expert_store_")
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "expert_store.bin")
        self.pin = pin
        self.pinned = pin
        self.mode: str | None = None
        self._records: list[list[_Record]] = []
        self._staging: dict[int, torch.Tensor] = {}  # padded nbytes -> aligned u8 buffer
        self._wfd: int | None = os.open(
            self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
        )
        self._rfd: int | None = None
        self._end = 0

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
            self._rfd = os.open(self.path, os.O_RDONLY | os.O_DIRECT)
            probe = _aligned_pinned_u8(_ALIGN, pin=False)
            os.preadv(self._rfd, [memoryview(probe.numpy())], 0)
            self.mode = "odirect"
        except OSError:
            if self._rfd is not None:
                os.close(self._rfd)
            self._rfd = os.open(self.path, os.O_RDONLY)
            self.mode = "buffered+fadvise"
        LOG.info(
            f"expert_offload FileStore: {self._end / 1e9:.2f} GB at {self.path} "
            f"(read mode: {self.mode})"
        )

    def _buffer(self, padded: int) -> torch.Tensor:
        buf = self._staging.get(padded)
        if buf is None:
            buf = _aligned_pinned_u8(padded, pin=self.pin)
            try:
                self.pinned = self.pinned and buf.is_pinned()
            except (RuntimeError, AssertionError):  # pragma: no cover
                self.pinned = False
            self._staging[padded] = buf
        return buf

    def _read(self, rec: _Record, buf: torch.Tensor) -> None:
        got = os.preadv(self._rfd, [memoryview(buf.numpy())[: rec.padded]], rec.offset)
        if got < rec.nbytes:  # pragma: no cover - short read is a store bug
            raise IOError(f"short read: {got} < {rec.nbytes} at {rec.offset}")
        if self.mode == "buffered+fadvise":
            os.posix_fadvise(
                self._rfd, rec.offset, rec.padded, os.POSIX_FADV_DONTNEED
            )

    def fetch(self, block_idx: int, slot_idx: int) -> torch.Tensor:
        rec = self._records[block_idx][slot_idx]
        buf = self._buffer(rec.padded)
        self._read(rec, buf)
        return buf[: rec.nbytes].view(rec.dtype).view(rec.shape)

    def state_tensor(self, block_idx: int, slot_idx: int) -> torch.Tensor:
        # Rare path (full-model save): a fresh tensor, never the reused staging buffer.
        return self.fetch(block_idx, slot_idx).clone()

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
