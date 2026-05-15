from __future__ import annotations

from copy import deepcopy


PAPER_DEFAULT_BOUNDARY_WEIGHT = 0.3
PAPER_DEFAULT_LOSS_WEIGHTS = [1.0, 0.7, 0.5, 0.3]


BASE_PRESET = {
    "fusion_mode": "learnable",
    "context_mode": "lite",
    "decoder_mode": "add",
    "deep_supervision": True,
    "boundary_weight": PAPER_DEFAULT_BOUNDARY_WEIGHT,
    "loss_weights": PAPER_DEFAULT_LOSS_WEIGHTS,
}


ABLATION_PRESETS = {
    "full": {
        **BASE_PRESET,
        "description": "A0 Full LiteCDNet baseline",
    },
    "boundary_05": {
        **BASE_PRESET,
        "boundary_weight": 0.5,
        "description": "A6 Tune boundary loss weight to 0.5 while keeping other settings the same as full",
    },
    "abs_diff": {
        **BASE_PRESET,
        "fusion_mode": "abs_diff",
        "description": "A1 Replace learnable DiffFusion with fixed absolute difference",
    },
    "no_context": {
        **BASE_PRESET,
        "context_mode": "identity",
        "description": "A2 Remove LiteContextModule and keep identity mapping only",
    },
    "concat_decoder": {
        **BASE_PRESET,
        "decoder_mode": "concat",
        "description": "A3 Replace additive decoder with concatenation-based decoder",
    },
    "no_boundary": {
        **BASE_PRESET,
        "boundary_weight": 0.0,
        "description": "A4 Remove boundary loss and keep BCE-Dice only",
    },
    "no_deep_supervision": {
        **BASE_PRESET,
        "deep_supervision": False,
        "description": "A5 Disable deep supervision and keep final prediction only",
    },
    "loss_weights_080604": {
        **BASE_PRESET,
        "loss_weights": [1.0, 0.8, 0.6, 0.4],
        "description": "A7 Adjust multi-scale supervision weights from [1.0, 0.7, 0.5, 0.3] to [1.0, 0.8, 0.6, 0.4] while keeping other settings the same as full",
    },
    "c45_diff_absconcat": {
        **BASE_PRESET,
        "fusion_mode": "c45_abs_concat",
        "description": "E1 Upgrade DiffFusion on C4/C5 to concat + abs diff + 1x1 while keeping C1-C3 unchanged",
    },
    "c45_context_triple": {
        **BASE_PRESET,
        "context_mode": "c45_triple",
        "description": "E2 Upgrade LiteContext on C4/C5 to three branches with dilation 1/3/5 while keeping lower stages unchanged",
    },
}


def get_ablation_config(case_name: str) -> dict:
    if case_name not in ABLATION_PRESETS:
        raise KeyError(f"Unknown ablation case: {case_name}")
    return deepcopy(ABLATION_PRESETS[case_name])
