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

from axolotl.integrations.expert_offload.offload import install_expert_offload
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
        buf_ptrs = {b.data_ptr() for b in h0._store._staging.values()}
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
