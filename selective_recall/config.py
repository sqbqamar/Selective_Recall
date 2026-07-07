"""Experiment configuration."""
from dataclasses import dataclass


@dataclass
class CFG:
    """All experiment knobs.

    Task:
        K          number of content tokens to recall (higher -> SSM fails harder)
        gap        filler tokens between queries (higher -> sparser recall)
        n_content  size of the content alphabet
        batch      batch size
    Model:
        d_model    model width
        n_layers   number of blocks
        n_heads    attention heads
        d_state    SSM state size (smaller -> saturates sooner, exposing the gap)
    Training:
        steps        optimization steps (GPU: 1500 is quick; CPU smoke: ~300)
        lr           peak learning rate
        weight_decay AdamW weight decay
        grad_clip    gradient clipping norm
        phase1_frac  fraction of steps with the gate forced open
        rho          target attending fraction (set at or above the recall density)
        lambda_max   peak budget-penalty weight
    Eval:
        eval_batches number of evaluation batches
        seed         random seed
    """
    # task
    K: int = 8
    gap: int = 3
    n_content: int = 40
    batch: int = 32

    # model
    d_model: int = 96
    n_layers: int = 2
    n_heads: int = 4
    d_state: int = 8

    # training
    steps: int = 1500
    lr: float = 2e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    phase1_frac: float = 0.4
    rho: float = 0.20
    lambda_max: float = 2.0

    # eval
    eval_batches: int = 8
    seed: int = 0
