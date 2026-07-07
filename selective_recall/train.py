"""Training and evaluation.

Two-phase schedule: the gate is held open at first so the attention pathway
learns to be useful, then a one-sided budget penalty anneals in to make the gate
sparse. This prevents the gate from collapsing before attention is trained.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import make_batch
from .models import build


def train_model(kind_name, cfg, device, verbose=False):
    """Train one model and return it."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = build(kind_name, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    p1 = int(cfg.phase1_frac * cfg.steps)
    warmup = max(1, int(0.1 * cfg.steps))

    def lr_at(step):
        if step < warmup:
            return cfg.lr * step / warmup
        prog = (step - warmup) / max(1, cfg.steps - warmup)
        return cfg.lr * 0.5 * (1 + math.cos(math.pi * prog))

    model.train()
    for step in range(cfg.steps):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        ids, tgt, _ = make_batch(cfg, device)
        gate_open = step < p1
        logits, gate = model(ids, gate_open=gate_open, return_gate=True)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
            ignore_index=-100)
        if (gate is not None) and (not gate_open):
            lam = cfg.lambda_max * (step - p1) / max(1, cfg.steps - p1)
            loss = loss + lam * F.relu(gate.mean() - cfg.rho)   # one-sided budget
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if verbose and step % max(1, cfg.steps // 6) == 0:
            print(f"    [{kind_name}] step {step:4d} loss {loss.item():.3f}")
    return model


@torch.no_grad()
def evaluate(model, cfg, device):
    """Return accuracy, and for gated models the attending fraction and the
    oracle gate precision/recall."""
    model.eval()
    accs, fracs, precs, recs = [], [], [], []
    density = None
    for _ in range(cfg.eval_batches):
        ids, tgt, rmask = make_batch(cfg, device)
        logits, gate = model(ids, gate_open=False, return_gate=True)
        pred = logits.argmax(-1)
        m = tgt != -100
        accs.append((pred[m] == tgt[m]).float().mean().item())
        if density is None:
            density = rmask.float().mean().item()
        if gate is not None:
            fires = gate.squeeze(-1) > 0.5
            fracs.append(fires.float().mean().item())
            tp = (fires & rmask).sum().float()
            precs.append((tp / fires.sum().clamp(min=1)).item())
            recs.append((tp / rmask.sum().clamp(min=1)).item())
    out = {"accuracy": float(np.mean(accs)), "recall_density": density}
    if fracs:
        out.update(attend_frac=float(np.mean(fracs)),
                   gate_precision=float(np.mean(precs)),
                   gate_recall=float(np.mean(recs)))
    return out
