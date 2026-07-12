# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Expert-granularity CPU offload for 4-bit MoE QLoRA on a single GPU.

Only the *frozen 4-bit expert* weights are moved; attention, router/gate, norms and the
trainable LoRA adapters stay GPU-resident. Two expert layouts are supported:

- **Per-expert ``Linear4bit``** (an ``experts`` ``ModuleList``): each expert is a ``bitsandbytes``
  ``Linear4bit`` whose big packed tensor is ``weight.data`` (the ``quant_state`` scales are ~1/32
  the size and are left resident).
- **Grouped 3D stacks quantized via ``quantize_moe_experts``**: models whose experts are fused 3D
  ``nn.Parameter`` stacks (one ``[n_experts, out, in]`` tensor per projection — OLMoE / Qwen3-MoE
  style on current transformers). ``quantize_moe_experts: true`` quantizes each stack in place
  through ``bitsandbytes.nn.parametrize.replace_parameter_4bit``: the packed tensor becomes
  ``module.parametrizations[name].original`` and a ``Bnb4bitParametrization`` (carrying the small
  resident ``quant_state``) dequantizes it on access, under ``no_grad`` — so autograd only ever
  holds the *dequantized* activation, never the packed tensor, and eviction of the packed data is
  safe outside the module's forward.

In both layouts we home the packed tensor's ``.data`` in *pinned* CPU RAM, and a **forward
pre-hook** on the owning module copies that block's experts onto the GPU just before it runs.

Eviction is driven entirely by a **single-resident-slot** policy: staging a block first evicts the
previously-staged one. There is deliberately **no evict post-hook**. Under ``use_reentrant=False``
gradient checkpointing each decoder layer's forward is *recomputed* in the backward pass; the same
pre-hook re-stages the block's experts for that recompute, and because nothing evicts a block until
the *next* block stages — which, in backward (processed last-layer-first), only happens after the
current block's recomputed backward has finished — the staged weights are always present when the
recomputed backward reads them. So at most **one block's** experts are GPU-resident at any instant,
in forward and backward alike, without depending on exactly when PyTorch stops a recompute.

This lets a fused MoE whose 4-bit experts exceed VRAM QLoRA-train on a small card, at the cost of
one host->device expert transfer per block per pass — a memory-for-compute trade.

**Why gradient checkpointing is required (correctness *and* the memory win).**
``bnb.matmul_4bit``'s autograd ``Function`` re-reads the packed weight in its backward (to
re-dequantize for the input gradient) via ``save_for_backward``. Eviction repoints
``weight.data`` at a 0-element placeholder, so the saved reference would read that placeholder in a
backward that runs against the *initial* forward's graph. Gradient checkpointing discards the
initial-forward saved tensors and **recomputes** each layer in backward (re-staging via the
pre-hook and rebuilding the saved tensors from the staged weights), which is what makes eviction
both correct and actually memory-freeing rather than pinning every staged weight alive as a saved
tensor. ``gradient_checkpointing: true`` with an explicit ``use_reentrant: false`` is enforced at
config validation (the ``ExpertOffloadArgs`` schema validator).

One GPU **per replica**: plain DDP (multi-GPU data parallel) is supported — each rank is its own
process with a full replica, so each rank homes its own pinned copy of the experts (CPU RAM cost
scales with world size) and stages to its own device; the offloaded weights are registered on
DDP's ignore list so the initial module-state sync never touches the 0-element placeholders.
FSDP / DeepSpeed / expert-parallel move or shard these same weights and would race the
stage/evict swaps; the config schema refuses to enable under any of them.
"""

from __future__ import annotations

import os

from typing import NamedTuple

import torch
from torch import nn

from axolotl.utils.logging import get_logger

from .store import FileStore, PrefetchReader, make_store

LOG = get_logger(__name__)


class _Slot(NamedTuple):
    """One offloadable packed tensor: the frozen ``nn.Parameter`` whose ``.data`` is swapped
    between its pinned CPU home and the GPU, plus where it appears in ``state_dict``.

    ``keys`` are candidate keys relative to ``owner``'s prefix — the parametrized layout needs two
    because bitsandbytes' own state-dict post-hook renames ``parametrizations.<p>.original`` to the
    clean ``<p>`` (hook order puts ours after bnb's, but both spellings are covered regardless).
    """

    param: nn.Parameter
    owner: nn.Module
    keys: tuple[str, ...]


# 0-element GPU placeholders that an evicted expert's ``weight.data`` points at while offloaded.
# Shared across all offloaded experts (reads never mutate them) and cached per (device, dtype) —
# the 4-bit storage dtype varies with ``bnb_4bit_quant_storage`` (uint8 / bfloat16 / float32), so
# the placeholder must match the real tensor's dtype or a restage would change it. Keeping the real
# "home" data OFF the module — only a 0-element placeholder is registered while evicted — means a
# stray ``model.to(device)`` never drags the big expert tensors back onto the GPU.
_PLACEHOLDERS: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _placeholder(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    ph = _PLACEHOLDERS.get(key)
    if ph is None:
        ph = torch.empty(0, dtype=dtype, device=device)
        _PLACEHOLDERS[key] = ph
    return ph


def _copy_required_for(provider, block_idx: int) -> bool:
    """Whether a non-CUDA staging copy must be made out of ``provider``'s source buffers.

    Stores whose blocks live in different tiers (a fused RAM/flash store) answer per block via
    ``copy_required_for``; simple stores answer once via the ``copy_required`` class attribute.
    """
    per_block = getattr(provider, "copy_required_for", None)
    if per_block is not None:
        return bool(per_block(block_idx))
    return bool(getattr(provider, "copy_required", False))


def _is_pinned(t: torch.Tensor) -> bool:
    """Whether ``t`` is pinned (so a ``non_blocking`` H2D copy is truly async). Robust on hosts
    where ``is_pinned`` is unavailable/raises without CUDA."""
    try:
        return bool(t.is_pinned())
    except (RuntimeError, AssertionError):  # pragma: no cover - platform dependent
        return False


def _zero_decode_byte(slot: _Slot) -> int:
    """The fill byte whose decoded value is EXACTLY 0.0 for this slot's storage.

    4-bit packed (uint8) slots: both nibbles must hold the codebook index of 0.0 —
    nf4 index 7 (byte 0x77), fp4 index 0 (byte 0x00). Unpacked float slots: byte 0x00
    IS 0.0. Quant type is read from the owner's Bnb4bitParametrization when present
    (the ``quantize_moe_experts`` layout) or a ``quant_type``-ish attribute (kit
    layouts); defaults to nf4."""
    if slot.param.dtype != torch.uint8:
        return 0x00  # float/bf16 passthrough: zero bytes == 0.0
    qt = None
    plists = getattr(slot.owner, "parametrizations", None)
    if isinstance(plists, nn.ModuleDict):
        for plist in plists.values():
            for par in plist:
                if type(par).__name__ == "Bnb4bitParametrization":
                    qs = getattr(par, "quant_state", None)
                    qt = getattr(qs, "quant_type", None)
                    break
            if qt:
                break
    if qt is None:
        qt = getattr(slot.owner, "quant_type", None)
    return 0x00 if str(qt).lower() == "fp4" else 0x77  # nf4 (default): code 7 = 0.0


def _is_linear4bit(module: nn.Module) -> bool:
    """A ``bitsandbytes`` ``Linear4bit`` whose ``weight`` is a packed 4-bit ``Params4bit``.

    Matched structurally (by the ``Params4bit`` weight type) rather than by ``isinstance`` so we do
    not hard-import bitsandbytes here and so PEFT/other subclasses of ``Linear4bit`` still match.
    """
    weight = getattr(module, "weight", None)
    return weight is not None and type(weight).__name__ == "Params4bit"


def _base_layer(module: nn.Module) -> nn.Module:
    """Unwrap a PEFT adapter wrapper (``lora.Linear4bit`` etc.) to the frozen base ``Linear4bit``.

    PEFT wraps the quantized expert and delegates ``.weight`` to ``.base_layer``; the packed tensor
    we offload lives on that base, and the (tiny, trainable) LoRA ``A``/``B`` matrices are separate
    siblings that must stay GPU-resident. Returns ``module`` unchanged if it is not wrapped.
    """
    return getattr(module, "base_layer", module)


def find_moe_expert_blocks(
    model: nn.Module,
) -> list[tuple[str, nn.Module, list[nn.Module]]]:
    """Discover offloadable MoE blocks and their frozen 4-bit expert base layers.

    A block is any module exposing an ``experts`` ``ModuleList`` of length >= 2 whose leaves include
    ``Linear4bit`` weights — the classic per-expert layout (``block_sparse_moe.experts`` /
    ``mlp.experts``). Fused-experts layouts that pack all experts into one 3D parameter (OLMoE /
    Qwen3-MoE on current transformers, GPT-OSS, DBRX) are **not** matched here — raw 3D parameters
    are only 4-bit under ``quantize_moe_experts: true``, whose parametrized stacks are discovered by
    :func:`find_parametrized_expert_stacks` instead.

    Returns ``(block_name, block_module, expert_base_layers)`` triples. Deduplicates base layers so a
    projection shared across the list is homed once. The hook attaches to ``block_module`` because
    its ``forward`` runs the experts (and is what gradient checkpointing recomputes).
    """
    blocks: list[tuple[str, nn.Module, list[nn.Module]]] = []
    for name, block in model.named_modules():
        experts = getattr(block, "experts", None)
        if not isinstance(experts, nn.ModuleList) or len(experts) < 2:
            continue
        seen: set[int] = set()
        base_layers: list[nn.Module] = []
        for module in experts.modules():
            if not _is_linear4bit(module):
                continue
            base = _base_layer(module)
            if not _is_linear4bit(base) or id(base) in seen:
                continue
            seen.add(id(base))
            base_layers.append(base)
        if base_layers:
            blocks.append((name, block, base_layers))
    return blocks


def find_parametrized_expert_stacks(
    model: nn.Module,
) -> list[tuple[str, nn.Module, list[_Slot]]]:
    """Discover grouped 3D expert stacks quantized via ``quantize_moe_experts``.

    Matches any module carrying a ``Bnb4bitParametrization`` (by class name, mirroring
    ``_is_linear4bit``'s structural matching) — the layout ``quantize_moe_experts: true`` produces
    for fused-expert models (OLMoE / Qwen3-MoE style ``[n_experts, out, in]`` stacks). The packed
    tensor is ``module.parametrizations[<p>].original``; the parametrization's ``quant_state``
    stays resident. The module itself is the hook site: its ``forward`` dequantizes the stacks on
    access (and is what gradient checkpointing recomputes).
    """
    blocks: list[tuple[str, nn.Module, list[_Slot]]] = []
    for name, module in model.named_modules():
        plists = getattr(module, "parametrizations", None)
        if not isinstance(plists, nn.ModuleDict):
            continue
        slots: list[_Slot] = []
        for pname, plist in plists.items():
            if not any(type(p).__name__ == "Bnb4bitParametrization" for p in plist):
                continue
            packed = plist.original
            if not isinstance(packed, nn.Parameter) or packed.requires_grad:
                continue  # trainable parametrized params are not ours to move
            slots.append(
                _Slot(
                    param=packed,
                    owner=module,
                    keys=(pname, f"parametrizations.{pname}.original"),
                )
            )
        if slots:
            blocks.append((name, module, slots))
    return blocks


class _BlockOffload:
    """Owns the pinned-CPU home copies of one MoE block's expert ``weight.data`` tensors and streams
    them to ``device`` for the duration of each forward / gradient-checkpoint recompute.

    While evicted, each expert's ``weight.data`` holds a shared 0-element GPU placeholder, so nothing
    that walks the module tree (``.to()``, checkpoint-save) drags the offloaded data back onto the
    GPU. ``quant_state`` (the small NF4 scales) stays GPU-resident throughout, as do the LoRA
    adapters and everything outside ``experts``. A ``state_dict`` post-hook substitutes the CPU homes
    for the placeholders so a full-model save stays correct (adapter-only saves never touch the base
    keys, so they are unaffected).
    """

    # The single block whose experts are currently GPU-staged. Class-wide, so it assumes one
    # offloaded model per process (the training case — including DDP, where each rank is its own
    # process with its own replica). Under use_reentrant=False gradient checkpointing the backward
    # RECOMPUTE re-runs a layer's forward to rebuild its saved tensors; staging a new block first
    # evicts this previously-staged one, so at most one block is GPU-resident at any instant, in
    # forward AND backward.
    _resident: _BlockOffload | None = None

    def __init__(
        self, name: str, slots: list[_Slot], device, pin: bool = True, store=None
    ):
        self.name = name
        self.device = torch.device(device)
        self.slots = slots
        # Home each packed weight in the store BEFORE any placeholder swap. The source is on
        # the GPU at install time, so the store's capture is a real device->host copy that
        # decouples the home from the live parameter we then overwrite with a placeholder.
        # RAMStore keeps (pinned) CPU tensors exactly as before this seam existed; FileStore
        # writes the bytes to its packed file and holds no per-block host copy.
        if store is None:
            store = make_store(None, None, pin)
            store_owned = True
        else:
            store_owned = False
        self._store = store
        self._prefetch: PrefetchReader | None = None
        self._staging: str = "whole_layer"
        # REAL expert count from the owning module — NOT the packed tensor's shape[0], which is the
        # flat byte count for the bnb ``quantize_moe_experts`` layout (millions), not the experts.
        self._n_experts_real = next(
            (getattr(sl.owner, a) for sl in slots for a in
             ("num_experts", "n_experts", "num_local_experts") if getattr(sl.owner, a, None)),
            None,
        )
        self.block_idx = store.add_block([slot.param.data.detach() for slot in slots])
        if store_owned:
            store.finalize()
        self.staged = False
        for idx, slot in enumerate(slots):
            self._install_state_dict_hook(slot, idx)
        self._staged_sel = None  # routed-staged union; None = full/whole coverage
        self.evict()  # start evicted: experts hold placeholders, ~0 GPU footprint

    @property
    def params(self) -> list[nn.Parameter]:
        return [slot.param for slot in self.slots]

    @property
    def pinned(self) -> bool:
        return self._store.pinned

    def _install_state_dict_hook(self, slot: _Slot, idx: int) -> None:
        """Keep full-model ``state_dict()`` correct while evicted: substitute the (pinned) CPU home
        for the 0-element placeholder under any of the slot's candidate keys. References, not
        copies, so adapter-only saves stay cheap and while *staged* it is a no-op (the entry is the
        real GPU tensor)."""

        def hook(module, state_dict, prefix, local_metadata):
            for key in slot.keys:
                t = state_dict.get(prefix + key)
                if t is not None and t.numel() == 0:
                    state_dict[prefix + key] = self._store.state_tensor(
                        self.block_idx, idx
                    )

        register = getattr(slot.owner, "register_state_dict_post_hook", None)
        if (
            register is None
        ):  # older torch: private hook, same (mod, sd, prefix, meta) signature
            register = slot.owner._register_state_dict_hook
        register(hook)

    @property
    def bytes(self) -> int:
        return self._store.block_nbytes(self.block_idx)

    def stage(self, args=None) -> None:
        """Copy this block's packed expert weights onto ``device`` (idempotent), first evicting the
        previously staged block so at most one block's experts are GPU-resident.

        ``staging="routed"`` stages ONLY the experts this forward routes to (the distinct union of
        ``args[1]`` = ``top_k_index``): a full-size GPU weight is allocated but only the routed rows
        are filled from the store, so flash/H2D traffic is ``read_fraction`` x the layer. Un-routed
        rows are never indexed by the MoE forward, so the output is bit-identical to whole-layer."""
        if self.staged:
            if self._staging != "routed" or self._staged_sel is None:
                return  # whole-layer content is routing-independent -> plain idempotence
            # ROUTED staleness (the 104x-floor acceptance-FAIL mechanism, 2026-07-11): backward
            # walks blocks LAST->FIRST, so block 1 is the last thing staged in every backward and
            # the FIRST thing the next microbatch's forward needs -- its pre-hook used to see
            # ``staged`` and silently serve the PREVIOUS microbatch's expert union to NEW data.
            # Tokens routed outside that stale union read a zero-masked (or, legacy, wrong-expert)
            # row: coherent, fill-proportional corruption (never visible in eval, whose pass order
            # never leaves the first-needed block staged). Fix: the guard is routing-AWARE -- serve
            # the staged tensor only if this call's union is covered; otherwise TOP-UP just the
            # missing experts into the live resident tensor and record them.
            sel = self._routed_experts(args)
            if sel is None or self._staged_sel.issuperset(sel):
                return
            missing = sorted(set(sel) - self._staged_sel)
            if os.environ.get("DRIFT_LOG") == "1":
                type(self)._drift_events = getattr(type(self), "_drift_events", 0) + 1
                type(self)._drift_experts = getattr(type(self), "_drift_experts", 0) + len(missing)
                if type(self)._drift_events <= 200:
                    print(f"DRIFT block={self.name} +{len(missing)} experts "
                          f"(staged {len(self._staged_sel)} -> {len(self._staged_sel)+len(missing)})", flush=True)
            self._top_up_routed(missing)
            self._staged_sel.update(missing)
            return
        cls = type(self)
        if cls._resident is not None and cls._resident is not self:
            cls._resident.evict()  # single-slot: free the prior block before staging this one
        if self._staging == "routed":
            sel = self._routed_experts(args)
            if sel is not None:
                self._stage_routed(sel)
                self.staged = True
                cls._resident = self
                return
        prefetched = (
            self._prefetch.take(self.block_idx) if self._prefetch is not None else None
        )
        # Whoever owns the source buffers also owns the guard that protects them.
        provider = self._prefetch if prefetched is not None else self._store
        cuda = self.device.type == "cuda"
        for idx, slot in enumerate(self.slots):
            src = (
                prefetched[idx]
                if prefetched is not None
                else self._store.fetch(self.block_idx, idx)
            )
            if cuda:
                # A cpu->cuda ``.to()`` ALWAYS materializes a new device tensor, so this can
                # never alias ``src``. The only hazard is the host rewriting ``src`` while the
                # copy is in flight — handled by ``record_stage_event`` below, not by blocking.
                slot.param.data = src.to(self.device, non_blocking=True)
            elif _copy_required_for(provider, self.block_idx):
                # Non-CUDA target: ``.to("cpu")`` returns ``src`` itself, so a recycled staging
                # buffer really would alias. Copy out of it.
                slot.param.data = src.to(self.device, copy=True)
            else:
                slot.param.data = src.to(self.device, non_blocking=True)
        if cuda:
            # Guard the buffers the copies above are reading; the owner waits on this event
            # before refilling them. RAMStore's hook is a no-op (its homes are persistent).
            record = getattr(provider, "record_stage_event", None)
            if record is not None:
                record()
        if self._prefetch is not None:
            # Safe to start reading the predicted next block: the reader alternates buffer sets
            # and waits on the event just recorded before touching this one again.
            self._prefetch.observe_and_predict(self.block_idx)
        self.staged = True
        cls._resident = self

    @staticmethod
    def _routed_experts(args):
        """The distinct experts this forward routes to, from the experts module's
        ``(hidden, top_k_index, top_k_weights)`` args. Sorted list, or None to fall back to
        whole-layer (no integer routing tensor available)."""
        if not args or len(args) < 2:
            return None
        idx = args[1]
        if not isinstance(idx, torch.Tensor) or idx.dtype not in (
            torch.int32, torch.int64, torch.long,
        ):
            return None
        return torch.unique(idx).tolist()

    def _top_up_routed(self, missing) -> None:
        """Fetch ``missing`` experts' byte ranges into the ALREADY-STAGED live tensors (no
        realloc, no eviction). Called by the routing-aware guard when a new forward's union
        escapes the staged one -- the incremental form of "stage the outside experts"."""
        E = self._n_experts_real
        if not E:
            return
        cuda = self.device.type == "cuda"
        for idx, slot in enumerate(self.slots):
            _, _, nbytes = self._store.slot_meta(self.block_idx, idx)
            per = nbytes // E
            live_u8 = slot.param.data.view(torch.uint8).reshape(-1)[: E * per].view(E, per)
            for e in missing:
                live_u8[e].copy_(
                    self._store.fetch_expert_bytes(self.block_idx, idx, e, E).to(
                        self.device, non_blocking=cuda
                    )
                )

    def _stage_routed(self, sel) -> None:
        """Allocate each slot's FULL packed GPU tensor, fill only the routed experts' byte
        ranges from the store, and MASK un-routed regions with the zero-decode byte (they
        dequantize to exact 0.0 rows) -> bit-identical output AND an inert backward. Experts are the leading contiguous byte dimension, so
        expert ``e`` occupies ``[e*per : (e+1)*per]`` of the flat packed bytes — correct for both
        the flat bnb layout and the ``[num_experts, ...]`` kit layout, using the REAL expert count
        (``self._n_experts_real``), never ``shape[0]``."""
        E = self._n_experts_real
        if not E:  # can't identify the expert count -> safe fallback to whole-layer
            return self._stage_whole()
        self._staged_sel = set(sel)  # the routing-aware guard tops up against this
        if os.environ.get("ROUTED_PAD") == "full":
            # BISECTION CONTROL: stage ALL E experts (same bytes as whole-layer) while keeping the
            # routed ASSEMBLY path (torch.empty + per-expert copy loop). Isolates "is the residual
            # from the subset itself" (R128 lands at the floor) vs "from the assembly mechanics"
            # (R128 diverges like routed). No zero rows, no top-up (full set covers every union).
            sel = list(range(E))
        if os.environ.get("STAGED_COUNT_LOG") == "1" and not getattr(self, "_stagedcnt_logged", False):
            # one-shot per block: the measured-rf instrument used by the dose-response A/Bs
            self._stagedcnt_logged = True
            print(f"STAGEDCNT block={self.name} staged={len(sel)}/{E} rf={len(sel)/E:.3f}", flush=True)
        cuda = self.device.type == "cuda"
        legacy_fill = os.environ.get("AXOLOTL_EXPERT_OFFLOAD_ROUTED_FILL") == "real"
        for idx, slot in enumerate(self.slots):
            shape, dtype, nbytes = self._store.slot_meta(self.block_idx, idx)
            per = nbytes // E
            full = torch.empty(shape, dtype=dtype, device=self.device)
            flat_u8 = full.view(torch.uint8).reshape(-1)
            full_u8 = flat_u8[: E * per].view(E, per)
            if legacy_fill:
                # LEGACY (pre-mask-fix) fill, kept ONLY for reproducing the dose-response A/Bs and
                # the A4 attenuation arm: broadcast a real routed expert's bytes into un-routed
                # rows. The 2026-07-11 dose-response showed this leaks fill-proportional error
                # into training (gap 0.062/0.084/0.186 at fill 0.03/0.19/0.31, convex).
                row0 = self._store.fetch_expert_bytes(self.block_idx, idx, sel[0], E).to(
                    self.device, non_blocking=cuda
                )
                full_u8[:] = row0
                full_u8[sel[0]].copy_(row0)
                rest = sel[1:]
            else:
                # MASK FIX (2026-07-11, the pre-registered fix): fill un-routed regions with the
                # byte whose two 4-bit codes DECODE TO EXACTLY 0.0, so the whole-stack dequant
                # (forward AND the gradient-checkpoint recompute in backward) yields true zero
                # rows for every un-routed expert under ANY absmax — masked-dequant semantics
                # implemented at the packed level. Zero rows are inert in any linear read: the
                # sparse forward never indexes them, and whatever residual path touched un-routed
                # content (the dose-response mechanism) now reads exact zeros instead of a real
                # expert's weights. NB packed zero BYTES are NOT zero weights (nf4 code 0 decodes
                # to -1.0 -> the historical zeros-explosion); the zero-decode byte is quant-type
                # specific: nf4 code 7 = 0.0 -> 0x77; fp4 code 0 = 0.0 -> 0x00. Float (unpacked)
                # slots zero at byte 0x00. Also cheaper than the legacy fill: one memset, no
                # extra store fetch.
                flat_u8.fill_(_zero_decode_byte(slot))
                rest = sel
            for e in rest:
                full_u8[e].copy_(
                    self._store.fetch_expert_bytes(self.block_idx, idx, e, E).to(
                        self.device, non_blocking=cuda
                    )
                )
            slot.param.data = full
        if cuda:
            record = getattr(self._store, "record_stage_event", None)
            if record is not None:
                record()

    def _stage_whole(self) -> None:
        """Whole-layer staging body reused when routed can't identify the expert count."""
        cuda = self.device.type == "cuda"
        for idx, slot in enumerate(self.slots):
            src = self._store.fetch(self.block_idx, idx)
            slot.param.data = src.to(self.device, non_blocking=cuda) if cuda else src.to(self.device, copy=True)
        if cuda:
            record = getattr(self._store, "record_stage_event", None)
            if record is not None:
                record()

    def evict(self) -> None:
        """Point this block's expert weights back at shared 0-element placeholders (idempotent),
        dropping the GPU copies so the caching allocator can reuse the memory for the next block."""
        for slot in self.slots:
            slot.param.data = _placeholder(self.device, slot.param.data.dtype)
        self.staged = False
        self._staged_sel = None
        cls = type(self)
        if cls._resident is self:
            cls._resident = None


def install_expert_offload(
    model: nn.Module,
    device=None,
    pin: bool = True,
    store: str | None = None,
    store_dir: str | None = None,
    prefetch: bool | None = None,
    staging: str | None = None,
) -> list[_BlockOffload]:
    """Offload every discoverable MoE block's frozen 4-bit experts to (pinned) CPU RAM.

    For each block, homes its expert ``weight.data`` tensors on the CPU, evicts them from the GPU,
    and registers a forward pre-hook (stage) on the block module. The handles are stashed on
    ``model._expert_offload_handles`` so they live as long as the model. Returns the handles (empty
    if no offloadable MoE block was found).
    """
    slot_blocks: list[tuple[str, nn.Module, list[_Slot]]] = [
        (
            name,
            block,
            [
                _Slot(param=base.weight, owner=base, keys=("weight",))
                for base in base_layers
            ],
        )
        for name, block, base_layers in find_moe_expert_blocks(model)
    ]
    slot_blocks += find_parametrized_expert_stacks(model)
    if not slot_blocks:
        raise RuntimeError(
            "expert_offload is enabled but no 4-bit MoE expert weights were found. This "
            "integration offloads per-expert bitsandbytes Linear4bit weights (an ``experts`` "
            "ModuleList) or grouped 3D expert stacks quantized via ``quantize_moe_experts: "
            "true`` (OLMoE / Qwen3-MoE style fused layouts on current transformers). If your "
            "model stores experts as fused 3D parameters, set ``quantize_moe_experts: true`` — "
            "plain ``load_in_4bit`` leaves those stacks unquantized, so there is nothing 4-bit "
            "to offload. Otherwise disable expert_offload for this model."
        )

    if device is None:
        device = slot_blocks[0][2][0].param.data.device
    device = torch.device(device)

    if store is None or isinstance(store, str):
        store = make_store(store, store_dir, pin)
    handles: list[_BlockOffload] = []
    for name, block, slots in slot_blocks:
        handle = _BlockOffload(name, slots, device, pin=pin, store=store)
        block.register_forward_pre_hook(lambda module, args, h=handle: h.stage(args))
        handles.append(handle)
    store.finalize()
    if staging is None:
        staging = os.environ.get("AXOLOTL_EXPERT_OFFLOAD_STAGING", "") or "whole_layer"
    if staging not in ("whole_layer", "routed"):
        raise ValueError(f"expert_offload staging must be whole_layer|routed, got {staging!r}")
    if staging == "routed":
        # STATUS (2026-07-11): routed-subset stages ONLY the experts a forward routes to, filling
        # un-routed GPU rows with a real routed expert's bytes (deterministic, finite, discarded
        # by the sparse forward). Frozen FORWARD is bit-identical to whole-layer (real-OLMoE
        # step-0 loss 0.7038==0.7038) -- the decode/inference use case is sound. TRAINING is NOT,
        # and a pre-registered DOSE-RESPONSE (two Qwen3-30B legs, bracket whole/routed/whole,
        # 150 steps) pins WHY: the divergence scales with the un-routed FILL MASS (1-rf).
        #   seq 2048, rf 0.97 (~3% fill): routed gap 0.062 = 31x the |whole-whole| floor (0.002)
        #   seq  256, rf 0.688 (~31% fill): routed gap 0.186 = 186x the floor (0.001, 3 warm arms)
        # Fill x10 -> gap x3 (H_FILL confirmed; a fixed-per-step-corruption hypothesis, which
        # predicted a constant ~0.06 gap, is refuted). Horizon is NOT the driver -- the gap is
        # established by step 50 and stable-to-declining.
        #
        # THE ZERO-DECODE MASK (default fill) was the pre-registered fix: un-routed rows dequantize
        # to EXACTLY 0.0 (proven per-expert on real bnb). Its ACCEPTANCE RUN (2026-07-11, prereg
        # 411b57f, same seq256 point, 3-warm-arm floor 0.001) says it is CORRECT BUT INSUFFICIENT:
        #   routed_mask gap 0.104 = 104x floor (FAIL);  legacy control gap 0.182 = 182x (valid).
        # The mask removes ~44% of the divergence (the fill-CONTENT component) and leaves a ~104x
        # coherent residual. Two independent lines say the residual is a COHERENT/COMPOUNDING bias,
        # not independent noise: (1) the A4 arm -- doubling gradient averaging ga4->ga8 barely moved
        # the legacy gap (ratio 0.916, vs 0.71 predicted for quadrature noise); (2) the seq512
        # dose-response is CONVEX. Leading hypothesis for the residual: routing DRIFT across the
        # gradient-checkpoint recompute -- as the attention LoRA trains, the recomputed forward
        # routes some tokens to experts OUTSIDE the originally-staged union, which then read a zeroed
        # row instead of the correct expert (coherent, un-averageable, grows with fill). Content
        # masking cannot fix that; a drift-robust superset stager (or recompute-time restaging) is
        # the open path, and it converges toward whole_layer as drift grows. So: routed TRAINING is
        # still non-viable -- whole_layer is the only training-validated staging. Forward/decode is
        # bit-identical and unaffected.
        if os.environ.get("AXOLOTL_EXPERT_OFFLOAD_ROUTED") != "1" and os.environ.get("AXOLOTL_EXPERT_OFFLOAD_ROUTED_EXPERIMENTAL") != "1":
            raise RuntimeError(
                "expert_offload_staging='routed' is forward/decode-only for now. Its TRAINING "
                "divergence scales with un-routed fill mass (dose-response 31x->186x floor as rf "
                "0.97->0.69); the zero-decode mask (default) is CORRECT but only ~halves it "
                "(acceptance run: 104x floor, still FAIL) -- a coherent residual (routing drift "
                "across the checkpointed recompute) survives. Forward is bit-identical to "
                "whole-layer. Use whole_layer for training; set AXOLOTL_EXPERT_OFFLOAD_ROUTED=1 "
                "only for forward/decode use or to reproduce the A/Bs."
            )
        LOG.info("expert_offload_staging=routed: reads only the routed expert subset; forward "
                 "bit-identical. WARNING: TRAINING still diverges -- the zero-decode mask halves "
                 "the gap (104x floor, FAIL at acceptance) but a coherent routing-drift residual "
                 "survives. Forward/decode only.")
    for h in handles:
        h._staging = staging
    if prefetch is None:
        prefetch = os.environ.get("AXOLOTL_EXPERT_OFFLOAD_PREFETCH", "") == "1"
    if prefetch and staging == "routed":
        LOG.info("expert_offload: staging=routed disables prefetch (per-forward routing not "
                 "predictable a block ahead); routed-subset staging runs synchronous.")
        prefetch = False
    if prefetch and isinstance(store, FileStore):
        reader = PrefetchReader(store, len(handles))
        for handle in handles:
            handle._prefetch = reader
        model._expert_offload_prefetch_reader = reader  # keep-alive + test introspection
    elif prefetch:
        LOG.info(
            "expert_offload: prefetch requested but the store is RAM-backed — nothing to "
            "read ahead of the pinned homes; running without a reader."
        )

    model._expert_offload_handles = handles
    _register_ddp_ignore(model, handles)
    total_gb = sum(h.bytes for h in handles) / 1e9
    n_experts = sum(len(h.slots) for h in handles)
    pinned = "pinned" if all(h.pinned for h in handles) else "pageable (no async H2D)"
    rank = (
        f" [rank {torch.distributed.get_rank()}]"
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else ""
    )
    where = (
        f"{pinned} CPU RAM"
        if store.mode == "ram"
        else f"disk ({store.mode}; staging {pinned})"
    )
    LOG.info(
        f"expert_offload{rank}: homed {n_experts} expert layers across {len(handles)} MoE blocks "
        f"({total_gb:.2f} GB) to {where}; one block resident on {device} at a time."
    )
    model._expert_offload_handles = handles  # external harness (activation-diff) flips staging per pass
    return handles


def _register_ddp_ignore(model: nn.Module, handles: list[_BlockOffload]) -> None:
    """Register every offloaded expert weight on DDP's ignore list.

    While evicted, those weights are 0-element placeholders; DDP's initial module-state sync
    (and any buffer broadcast) must never touch them — broadcasting a placeholder is at best a
    no-op and at worst re-materializes state DDP has no business managing. The offloaded experts
    are frozen (``requires_grad=False``) so they never participate in gradient buckets either;
    the ignore list makes that contract explicit. ``_ddp_params_and_buffers_to_ignore`` is the
    mechanism behind ``DistributedDataParallel._set_params_and_buffers_to_ignore_for_model`` and
    is read off the module at DDP construction, which happens after ``post_model_load``.

    The frozen weight Parameter objects survive eviction (only ``.data`` is swapped), so identity
    matching against ``named_parameters`` yields their fully-qualified names.
    """
    offloaded_ids = {id(param) for handle in handles for param in handle.params}
    names = [
        name
        for name, param in model.named_parameters(remove_duplicate=False)
        if id(param) in offloaded_ids
    ]
    existing = list(getattr(model, "_ddp_params_and_buffers_to_ignore", []) or [])
    model._ddp_params_and_buffers_to_ignore = existing + [
        n for n in names if n not in existing
    ]
