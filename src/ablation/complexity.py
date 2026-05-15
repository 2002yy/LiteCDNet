from __future__ import annotations

from types import SimpleNamespace

import torch
from thop import profile

from ablation.litecdnet_variants import build_ablation_model
from ablation.presets import get_ablation_config


def build_profile_args(case_name: str, n_class: int = 2):
    config = get_ablation_config(case_name)
    return SimpleNamespace(
        n_class=n_class,
        fusion_mode=config["fusion_mode"],
        context_mode=config["context_mode"],
        decoder_mode=config["decoder_mode"],
        deep_supervision=config["deep_supervision"],
        use_pretrained_backbone=False,
    )


def compute_case_complexity(case_name: str, img_size: int = 256, n_class: int = 2) -> dict[str, float]:
    args = build_profile_args(case_name, n_class=n_class)
    model = build_ablation_model(args)
    model.eval()
    dummy_t1 = torch.randn(1, 3, img_size, img_size)
    dummy_t2 = torch.randn(1, 3, img_size, img_size)
    macs, params = profile(model, inputs=(dummy_t1, dummy_t2), verbose=False)
    return {
        "params_m": params / 1e6,
        "macs_g": macs / 1e9,
        "flops_g": macs * 2 / 1e9,
    }


def compare_with_full(case_name: str, img_size: int = 256, n_class: int = 2) -> dict[str, float]:
    full = compute_case_complexity("full", img_size=img_size, n_class=n_class)
    current = full if case_name == "full" else compute_case_complexity(case_name, img_size=img_size, n_class=n_class)
    return {
        **current,
        "delta_params_m_vs_full": current["params_m"] - full["params_m"],
        "delta_flops_g_vs_full": current["flops_g"] - full["flops_g"],
        "ratio_params_vs_full": current["params_m"] / full["params_m"] if full["params_m"] > 0 else 1.0,
        "ratio_flops_vs_full": current["flops_g"] / full["flops_g"] if full["flops_g"] > 0 else 1.0,
        "params_reduction_pct_vs_full": (1.0 - current["params_m"] / full["params_m"]) * 100 if full["params_m"] > 0 else 0.0,
        "flops_reduction_pct_vs_full": (1.0 - current["flops_g"] / full["flops_g"]) * 100 if full["flops_g"] > 0 else 0.0,
    }
