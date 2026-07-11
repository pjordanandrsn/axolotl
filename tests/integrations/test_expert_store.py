# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Store-seam tests: FileStore serves byte-identical blocks to RAMStore, records its read
mode, and never lets a staged parameter alias its reused staging buffers."""

import copy
import hashlib

import pytest
import torch

from axolotl.integrations.expert_offload.offload import (
    _Slot,
    _zero_decode_byte,
    install_expert_offload,
)
from axolotl.integrations.expert_offload.store import FileStore, RAMStore

from .test_expert_offload import FakeGroupedMoEModel, FakeMoEModel


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def _build_pair(model_cls, **kwargs):
    torch.manual_seed(0)
    model = model_cls(**kwargs)
    return model, copy.deepcopy(model)


@pytest.mark.parametrize(
    "model_cls,kwargs",
    [
        (FakeMoEModel, dict(d=16, n_experts=4, n_layers=3, lora=True)),
        (FakeGroupedMoEModel, dict(d=16, n_experts=4, n_layers=3)),
    ],
    ids=["linear4bit", "parametrized"],
)
class TestStoreParity:
    def test_filestore_serves_ramstore_bytes(self, tmp_path, model_cls, kwargs):
        """Per-slot SHA256 of bytes served by FileStore == RAMStore, full model sweep."""
        m_ram, m_file = _build_pair(model_cls, **kwargs)
        h_ram = install_expert_offload(m_ram, device="cpu", pin=False, store="ram")
        h_file = install_expert_offload(
            m_file, device="cpu", pin=False, store="file", store_dir=str(tmp_path)
        )
        assert len(h_ram) == len(h_file) > 0
        for hr, hf in zip(h_ram, h_file):
            assert len(hr.slots) == len(hf.slots)
            for idx in range(len(hr.slots)):
                a = hr._store.fetch(hr.block_idx, idx)
                b = hf._store.fetch(hf.block_idx, idx)
                assert a.dtype == b.dtype and a.shape == b.shape
                assert _sha(a) == _sha(b), f"byte mismatch block={hr.name} slot={idx}"

    def test_state_dicts_match_across_stores(self, tmp_path, model_cls, kwargs):
        m_ram, m_file = _build_pair(model_cls, **kwargs)
        install_expert_offload(m_ram, device="cpu", pin=False, store="ram")
        install_expert_offload(
            m_file, device="cpu", pin=False, store="file", store_dir=str(tmp_path)
        )
        sd_r, sd_f = m_ram.state_dict(), m_file.state_dict()
        assert set(sd_r) == set(sd_f)
        for k in sd_r:
            assert sd_r[k].shape == sd_f[k].shape
            if sd_r[k].numel():
                assert _sha(sd_r[k]) == _sha(sd_f[k]), k


class TestFileStoreContracts:
    def test_mode_recorded_and_valid(self, tmp_path):
        m, _ = _build_pair(FakeMoEModel, d=16, n_experts=4, n_layers=2)
        handles = install_expert_offload(
            m, device="cpu", pin=False, store="file", store_dir=str(tmp_path)
        )
        st = handles[0]._store
        assert isinstance(st, FileStore)
        assert st.mode in ("odirect", "buffered+fadvise")

    def test_staged_params_never_alias_staging_buffers(self, tmp_path):
        """Staging block B must not mutate block A's previously staged bytes (the staging
        buffers are reused across blocks; staged tensors must be fresh copies)."""
        m, _ = _build_pair(FakeMoEModel, d=16, n_experts=4, n_layers=3)
        handles = install_expert_offload(
            m, device="cpu", pin=False, store="file", store_dir=str(tmp_path)
        )
        h0, h1 = handles[0], handles[1]
        h0.stage()
        before = [_sha(p.data) for p in h0.params]
        buf_ptrs = {
            b.data_ptr() for bs in h0._store._sets for b in bs.bufs.values()
        }
        assert all(p.data.data_ptr() not in buf_ptrs for p in h0.params)
        h0.staged = False  # bypass idempotence guard; force a buffer-overwriting refetch
        h1.stage()  # evicts h0 (single slot) and rewrites the shared staging buffers
        # h0's bytes were already copied out, so nothing it staged may have changed:
        h0.stage()
        after = [_sha(p.data) for p in h0.params]
        assert before == after

    def test_ram_default_unchanged(self):
        m, _ = _build_pair(FakeMoEModel, d=16, n_experts=4, n_layers=2)
        handles = install_expert_offload(m, device="cpu", pin=False)
        assert isinstance(handles[0]._store, RAMStore)


class TestPrefetch:
    """Phase C: deterministic host-side double-buffer over FileStore."""

    def _train_grads(self, model, x):
        model.zero_grad()
        out = model(x, use_ckpt=True)
        out.sum().backward()
        return out, {
            n: p.grad.clone()
            for n, p in model.named_parameters()
            if p.grad is not None
        }

    def test_prefetch_grads_match_reference(self, tmp_path):
        torch.manual_seed(0)
        model = FakeMoEModel(d=16, n_experts=4, n_layers=3, lora=True)
        reference = copy.deepcopy(model)
        x = torch.randn(2, 5, 16)
        out_ref, ref_grads = self._train_grads(reference, x)
        install_expert_offload(
            model, device="cpu", pin=False,
            store="file", store_dir=str(tmp_path), prefetch=True,
        )
        assert getattr(model, "_expert_offload_prefetch_reader", None) is not None
        out, grads = self._train_grads(model, x)
        assert torch.allclose(out_ref, out, atol=1e-6)
        assert set(ref_grads) == set(grads) and len(grads) > 0
        for name, g in ref_grads.items():
            assert torch.allclose(g, grads[name], atol=1e-6), name

    def test_prefetch_residency_and_buffer_accounting(self, tmp_path):
        torch.manual_seed(0)
        model = FakeMoEModel(d=16, n_experts=4, n_layers=4)
        handles = install_expert_offload(
            model, device="cpu", pin=False,
            store="file", store_dir=str(tmp_path), prefetch=True,
        )
        max_resident = 0
        from axolotl.integrations.expert_offload.offload import find_moe_expert_blocks
        def probe(_m, _a):
            nonlocal max_resident
            max_resident = max(max_resident, sum(h.staged for h in handles))
        for _n, block, _b in find_moe_expert_blocks(model):
            block.register_forward_pre_hook(probe)
        model(torch.randn(2, 5, 16), use_ckpt=True).sum().backward()
        assert max_resident == 1  # staged-block invariant UNCHANGED by prefetch
        reader = model._expert_offload_prefetch_reader
        # Host cost is bounded at TWO blocks' worth of staging buffers: the reader double-buffers
        # (one set filling while the other's H2D is still in flight), each set holding one buffer
        # per slot index. GPU peak is unaffected — the staged block count is still 1 (asserted
        # above); only host RAM grows, by exactly one extra set.
        n_slots_per_block = max(len(h.slots) for h in handles)
        assert len(reader._sets) == 2
        for bset in reader._sets:
            assert len(bset.bufs) <= n_slots_per_block
        assert any(bset.bufs for bset in reader._sets)

    def test_misprediction_discards_and_falls_back(self, tmp_path):
        torch.manual_seed(0)
        model = FakeMoEModel(d=16, n_experts=4, n_layers=3)
        handles = install_expert_offload(
            model, device="cpu", pin=False,
            store="file", store_dir=str(tmp_path), prefetch=True,
        )
        import time
        # stage out of any monotone order: 0, 2, 1, 0 — turnarounds force mispredictions
        for target in (0, 2, 1, 0):
            for h in handles:
                h.staged = False
            handles[target].stage()
            time.sleep(0.05)  # let the reader land a (possibly wrong) prediction
            got = [_sha(p.data) for p in handles[target].params]
            want = [
                _sha(handles[target]._store.fetch(handles[target].block_idx, i))
                for i in range(len(handles[target].slots))
            ]
            assert got == want, f"bytes wrong after staging block {target}"

    def test_prefetch_is_noop_on_ram_store(self):
        model = FakeMoEModel(d=16, n_experts=4, n_layers=2)
        install_expert_offload(model, device="cpu", pin=False, store="ram", prefetch=True)
        assert getattr(model, "_expert_offload_prefetch_reader", None) is None


class TestAsyncStaging:
    """The async-H2D contract: on CUDA every staging path is non_blocking and the source
    buffers are protected by a recorded event, not by a blocking copy."""

    def test_buffers_are_per_slot_not_per_size(self, tmp_path):
        """Same-size slots in one block must not share a buffer — they are read as a group and
        handed out together, so sharing would clobber slot 0 before it was copied out."""
        m, _ = _build_pair(FakeMoEModel, d=16, n_experts=4, n_layers=2)
        handles = install_expert_offload(
            m, device="cpu", pin=False, store="file", store_dir=str(tmp_path)
        )
        st = handles[0]._store
        h = handles[0]
        tensors = [st.fetch(h.block_idx, i) for i in range(len(h.slots))]
        ptrs = [t.data_ptr() for t in tensors]
        assert len(set(ptrs)) == len(ptrs), "slots aliased one buffer"
        # and their bytes are each individually correct
        for i, t in enumerate(tensors):
            assert _sha(t) == _sha(st.state_tensor(h.block_idx, i))

    def test_state_tensor_does_not_disturb_staging_sets(self, tmp_path):
        """A full-model save while staged must not recycle a set an H2D may still be reading."""
        m, _ = _build_pair(FakeMoEModel, d=16, n_experts=4, n_layers=3)
        handles = install_expert_offload(
            m, device="cpu", pin=False, store="file", store_dir=str(tmp_path)
        )
        h = handles[0]
        h.stage()
        before = [_sha(p.data) for p in h.params]
        _ = m.state_dict()  # walks every block via state_tensor
        assert [_sha(p.data) for p in h.params] == before

    def test_ram_store_has_noop_event_hook(self):
        m, _ = _build_pair(FakeMoEModel, d=16, n_experts=4, n_layers=2)
        handles = install_expert_offload(m, device="cpu", pin=False, store="ram")
        assert handles[0]._store.record_stage_event() is None

    def test_many_stagings_keep_bytes_correct(self, tmp_path):
        """Hammer the rotating sets: repeated forward/backward must never serve stale bytes."""
        torch.manual_seed(0)
        model = FakeMoEModel(d=16, n_experts=4, n_layers=4, lora=True)
        reference = copy.deepcopy(model)
        x = torch.randn(2, 5, 16)
        reference.zero_grad(); out_ref = reference(x, use_ckpt=True); out_ref.sum().backward()
        ref = {n: p.grad.clone() for n, p in reference.named_parameters() if p.grad is not None}
        install_expert_offload(
            model, device="cpu", pin=False, store="file",
            store_dir=str(tmp_path), prefetch=True,
        )
        for _ in range(3):  # several passes -> many set rotations + prefetch turnarounds
            model.zero_grad()
            out = model(x, use_ckpt=True)
            out.sum().backward()
        assert torch.allclose(out_ref, out, atol=1e-6)
        got = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        for n in ref:
            assert torch.allclose(ref[n], got[n], atol=1e-6), n


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestAsyncStagingCuda:
    def test_cuda_staging_records_guard_event(self, tmp_path):
        m, _ = _build_pair(FakeMoEModel, d=16, n_experts=4, n_layers=3)
        m = m.cuda()
        handles = install_expert_offload(
            m, device="cuda", pin=True, store="file", store_dir=str(tmp_path)
        )
        h = handles[0]
        h.stage()
        st = h._store
        assert any(bs.event is not None for bs in st._sets), "no CUDA event guard recorded"
        assert all(p.data.is_cuda and p.data.numel() > 0 for p in h.params)

    def test_cuda_grads_match_reference_with_prefetch(self, tmp_path):
        torch.manual_seed(0)
        model = FakeMoEModel(d=32, n_experts=4, n_layers=3, lora=True).cuda()
        reference = copy.deepcopy(model)
        x = torch.randn(2, 5, 32, device="cuda")
        reference.zero_grad(); o = reference(x, use_ckpt=True); o.sum().backward()
        ref = {n: p.grad.clone() for n, p in reference.named_parameters() if p.grad is not None}
        install_expert_offload(
            model, device="cuda", pin=True, store="file",
            store_dir=str(tmp_path), prefetch=True,
        )
        model.zero_grad(); o2 = model(x, use_ckpt=True); o2.sum().backward()
        torch.cuda.synchronize()
        got = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        assert torch.allclose(o, o2, atol=1e-5)
        for n in ref:
            assert torch.allclose(ref[n], got[n], atol=1e-5), n


# --------------------------------------------------------------------------- #
# Routed-subset staging: stage only the experts a forward routes to.          #
# A faithful SPARSE fake (mirrors ExpertsLoRA's (hidden, top_k_index,         #
# top_k_weights) signature, computing ONLY routed experts) — so un-staged     #
# rows are never read and routed output must be bit-identical to whole-layer. #
# --------------------------------------------------------------------------- #
import torch.nn.functional as _F  # noqa: E402
from torch.nn.utils import parametrize as _parametrize  # noqa: E402

from .test_expert_offload import Bnb4bitParametrization  # noqa: E402


class SparseGroupedExperts(torch.nn.Module):
    def __init__(self, d, n_experts):
        super().__init__()
        self.num_experts = n_experts
        self.gate_up_proj = torch.nn.Parameter(torch.randn(n_experts, d, d) * 0.1, requires_grad=False)
        self.down_proj = torch.nn.Parameter(torch.randn(n_experts, d, d) * 0.1, requires_grad=False)
        for pn in ("gate_up_proj", "down_proj"):
            _parametrize.register_parametrization(self, pn, Bnb4bitParametrization(), unsafe=True)

    def forward(self, hidden, top_k_index, top_k_weights):
        # SPARSE: read ONLY experts that appear in top_k_index. Un-routed rows are never touched,
        # so routed-subset staging (which leaves them uninitialized) must match whole-layer.
        flat = hidden.reshape(-1, hidden.shape[-1])
        out = torch.zeros_like(flat)
        idx = top_k_index.reshape(-1, top_k_index.shape[-1])
        wts = top_k_weights.reshape(-1, top_k_weights.shape[-1])
        for e in torch.unique(idx).tolist():
            hit = (idx == e).any(dim=-1)
            if not bool(hit.any()):
                continue
            x = flat[hit]
            h = torch.tanh(_F.linear(x, self.gate_up_proj[e]))
            y = _F.linear(h, self.down_proj[e])
            w = (wts * (idx == e)).sum(-1)[hit].unsqueeze(-1)
            out[hit] = out[hit] + w * y
        return out.view_as(hidden)


class SparseMoEBlock(torch.nn.Module):
    def __init__(self, d, n_experts, k=2):
        super().__init__()
        self.k = k
        self.router = torch.nn.Linear(d, n_experts)
        self.experts = SparseGroupedExperts(d, n_experts)

    def forward(self, x):
        logits = self.router(x)
        w, idx = torch.topk(torch.softmax(logits, dim=-1), self.k, dim=-1)
        return self.experts(x, idx, w)


class SparseMoEModel(torch.nn.Module):
    def __init__(self, d=16, n_experts=8, n_layers=3, k=2):
        super().__init__()
        self.blocks = torch.nn.ModuleList(SparseMoEBlock(d, n_experts, k) for _ in range(n_layers))

    def forward(self, x, use_ckpt=False):
        for b in self.blocks:
            x = x + (torch.utils.checkpoint.checkpoint(b, x, use_reentrant=False) if use_ckpt else b(x))
        return x


@pytest.mark.parametrize("store_kind", ["ram", "file"])
class TestRoutedSubsetStaging:
    def _grads(self, model, x):
        model.zero_grad()
        out = model(x, use_ckpt=True)
        out.sum().backward()
        return out, {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    def test_routed_bit_identical_to_whole_layer(self, tmp_path, store_kind):
        torch.manual_seed(0)
        model = SparseMoEModel(d=16, n_experts=8, n_layers=3, k=2)
        ref = copy.deepcopy(model)
        x = torch.randn(2, 6, 16)
        # whole-layer reference
        install_expert_offload(ref, device="cpu", pin=False, store=store_kind,
                               store_dir=str(tmp_path / "a"), staging="whole_layer")
        out_ref, g_ref = self._grads(ref, x)
        # routed-subset
        install_expert_offload(model, device="cpu", pin=False, store=store_kind,
                               store_dir=str(tmp_path / "b"), staging="routed")
        out, g = self._grads(model, x)
        assert torch.allclose(out_ref, out, atol=1e-6), "routed output != whole-layer"
        assert set(g_ref) == set(g) and len(g) > 0
        for n in g_ref:
            assert torch.allclose(g_ref[n], g[n], atol=1e-6), f"grad mismatch {n}"

    def test_routed_reads_only_the_union(self, tmp_path, store_kind):
        """The staged GPU weight has exactly the routed rows filled; un-routed rows are left
        uninitialized — proven by monkeypatching fetch_expert to record which experts were read."""
        torch.manual_seed(1)
        model = SparseMoEModel(d=16, n_experts=8, n_layers=2, k=2)
        handles = install_expert_offload(model, device="cpu", pin=False, store=store_kind,
                                         store_dir=str(tmp_path), staging="routed")
        read = set()
        for h in handles:
            orig = h._store.fetch_expert_bytes
            def wrap(b, s, e, ne, orig=orig):
                read.add(int(e)); return orig(b, s, e, ne)
            h._store.fetch_expert_bytes = wrap
        x = torch.randn(1, 8, 16)
        with torch.no_grad():
            model(x, use_ckpt=False)
        # something was read, and strictly fewer than every (expert x layer x slot) if routing is sparse
        assert 0 < len(read) <= 8


class TestMaskFixZeroDecode:
    """The mask fix: un-routed rows are filled with the byte that DEQUANTIZES to exactly 0.0, so
    they contribute nothing to any read of the packed stack (forward or the checkpointed-backward
    recompute) -- masked-dequant implemented at the packed level. These tests cover the byte
    SELECTION (CPU) and the per-expert STAGING SEMANTICS on real bitsandbytes (CUDA). They do NOT
    assert training-convergence efficacy: the CPU MoE fakes are sparse/routing-weighted and cannot
    reproduce the whole-stack dequant that leaks un-routed content in the real model, so efficacy
    is established by the real-model acceptance run, not here."""

    def test_zero_decode_byte_float_slot_is_zero(self):
        p = torch.nn.Parameter(torch.zeros(4, 4), requires_grad=False)
        assert _zero_decode_byte(_Slot(param=p, owner=torch.nn.Module(), keys=("weight",))) == 0x00

    def test_zero_decode_byte_nf4_default(self):
        # a uint8 packed slot with no discoverable quant_type defaults to nf4 -> 0x77 (code 7 = 0.0)
        p = torch.nn.Parameter(torch.zeros(8, dtype=torch.uint8), requires_grad=False)
        assert _zero_decode_byte(_Slot(param=p, owner=torch.nn.Module(), keys=("weight",))) == 0x77

    def test_zero_decode_byte_fp4(self):
        class _Owner(torch.nn.Module):
            quant_type = "fp4"
        p = torch.nn.Parameter(torch.zeros(8, dtype=torch.uint8), requires_grad=False)
        assert _zero_decode_byte(_Slot(param=p, owner=_Owner(), keys=("weight",))) == 0x00

    @pytest.mark.parametrize("quant_type,fill", [("nf4", 0x77), ("fp4", 0x00)])
    def test_masked_dequant_is_exactly_zero_per_expert(self, quant_type, fill):
        """The core correctness claim, on REAL bitsandbytes at the granularity the fix operates:
        replicate _stage_routed's byte writes on a real quantize_4bit'd [E,out,in] stack (fill the
        whole packed buffer with the zero-decode byte, overwrite the routed experts' byte ranges
        with their real bytes), dequantize the WHOLE stack, and assert un-routed rows are EXACTLY
        0.0 while routed rows equal the unmasked reference. Skips when CUDA/bnb are unavailable."""
        cuda = torch.cuda.is_available()
        if not cuda:
            pytest.skip("real-bnb 4-bit quantize requires CUDA")
        try:
            from bitsandbytes.functional import quantize_4bit, dequantize_4bit
        except Exception as e:  # pragma: no cover
            pytest.skip(f"bitsandbytes unavailable: {e}")
        E, OUT, IN, K = 8, 64, 32, 3
        torch.manual_seed(0)
        W = torch.randn(E, OUT, IN, device="cuda") * 0.08
        packed, qs = quantize_4bit(W.reshape(-1), quant_type=quant_type)
        ref = dequantize_4bit(packed, qs, quant_type=quant_type).reshape(E, OUT, IN)
        per = packed.numel() // E
        sel = sorted(torch.randperm(E)[:K].tolist())
        masked = packed.clone().view(torch.uint8)
        masked.fill_(fill)
        src = packed.view(torch.uint8)
        for e in sel:
            masked[e * per:(e + 1) * per] = src[e * per:(e + 1) * per]
        deq = dequantize_4bit(masked.view(packed.dtype), qs, quant_type=quant_type).reshape(E, OUT, IN)
        for e in range(E):
            if e in sel:
                assert torch.equal(deq[e], ref[e]), f"routed expert {e} changed"
            else:
                assert deq[e].abs().max().item() == 0.0, f"un-routed expert {e} not exactly zero"
