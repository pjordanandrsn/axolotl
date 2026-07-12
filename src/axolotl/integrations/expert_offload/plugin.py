# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Expert-offload plugin for axolotl: per-replica CPU offload of frozen 4-bit MoE experts."""

from __future__ import annotations

from axolotl.integrations.base import BasePlugin


class ExpertOffloadPlugin(BasePlugin):
    """Stream frozen 4-bit MoE experts from pinned CPU RAM one block at a time.

    Surgical counterpart to ``layer_offloading``: it moves only the frozen 4-bit experts (the bulk
    of a MoE's parameters) while attention, router, norms and the trainable LoRA adapters stay
    GPU-resident — minimizing per-step PCIe traffic for the same peak-VRAM reduction. One GPU per
    replica (single-GPU or plain DDP; no FSDP / DeepSpeed / expert-parallel); the cross-field
    config requirements are enforced at config-parse time by the ``ExpertOffloadArgs`` schema
    validator. See ``offload.py`` for the mechanism and this integration's README.
    """

    def get_input_args(self):
        return "axolotl.integrations.expert_offload.ExpertOffloadArgs"


    def add_callbacks_pre_trainer(self, cfg, model):
        """DIAGNOSTIC (env-gated GRAD_DUMP=1): dump per-trainable-param grad norms at float64
        each step to GRAD_DUMP_PATH, to localize where a routed-vs-whole training signal first
        diverges. Off by default; zero cost unless GRAD_DUMP=1."""
        import os
        if os.environ.get("GRAD_DUMP") != "1":
            return []
        import json, torch
        from transformers import TrainerCallback

        path = os.environ.get("GRAD_DUMP_PATH", "/tmp/grad_dump.jsonl")
        max_steps = int(os.environ.get("GRAD_DUMP_STEPS", "3"))

        class _GradDump(TrainerCallback):
            def on_pre_optimizer_step(self, args, state, control, model=None, **kw):
                if model is None or state.global_step >= max_steps:
                    return
                rec = {"step": int(state.global_step), "loss_hp": None, "params": {}}
                with torch.no_grad():
                    for n, prm in model.named_parameters():
                        if prm.requires_grad and prm.grad is not None:
                            g = prm.grad.detach().double()
                            rec["params"][n] = [float(g.norm().item()), float(g.abs().max().item())]
                with open(path, "a") as f:
                    f.write(json.dumps(rec) + "\n")

        return [_GradDump()]

    def post_model_load(self, cfg, model):
        """Install the offload after the model is built, quantized, PEFT-wrapped and on the GPU."""
        if not getattr(cfg, "expert_offload", False):
            return
        from .offload import install_expert_offload

        install_expert_offload(
            model,
            pin=getattr(cfg, "expert_offload_pin_memory", True),
            store=getattr(cfg, "expert_offload_store", None),
            store_dir=getattr(cfg, "expert_offload_store_dir", None),
            prefetch=getattr(cfg, "expert_offload_prefetch", None),
            staging=getattr(cfg, "expert_offload_staging", None),
        )
