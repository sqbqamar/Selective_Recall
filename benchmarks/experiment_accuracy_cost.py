"""
Accuracy-and-cost experiment for Selective Recall.

The decisive question the cost benchmark could NOT answer: does the *cheap*
bounded-memory mechanism stay accurate at long context? Cheapness is trivial if
you look at fewer tokens; the whole question is whether accuracy survives.

TASK (marked-token recall). A long sequence is mostly filler. K distinctive
content tokens sit at random positions. At the end, K query slots must reproduce
the content tokens in order of appearance. Recall is needed only at the query
slots, but the content to recall is scattered anywhere in the long context.

THREE MECHANISMS, identical everywhere except the attention module, at a MATCHED
attention budget (window width W = memory size M):

  DENSE        full causal attention over all L tokens. Accurate, but cost and
               KV memory grow with L.
  WINDOW       attention restricted to the last W tokens (sliding window). Cheap
               and bounded, but blind to content that fell outside the window.
  SR-MEMORY    a learned write score keeps the top-M tokens in a bounded memory;
               queries attend over those M. Cheap and bounded. Accurate ONLY IF
               the write score learns to keep the content tokens.

This is the fair test of learned vs naive bounded selection. If SR-MEMORY beats
WINDOW on accuracy at matched budget while staying far below DENSE in cost, the
mechanism is doing real work. If SR-MEMORY is no better than WINDOW, the bounded
memory is the bottleneck and we have learned that plainly.

Note: this isolates the attention/memory mechanism (no state-space backbone), so
it is fast enough to actually run at long context. The full model would add a
linear-in-L backbone term on top, as in the manuscript.
"""

import argparse
import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# CONFIG
# =====================================================================
class CFG:
    L = 256           # context length (raise on GPU: 1024, 4096, ...)
    K = 6             # number of content tokens to recall
    budget = 32       # W = M (matched attention budget)
    n_content = 40
    batch = 32

    d_model = 128
    n_heads = 4
    n_layers = 2

    steps = 1500
    lr = 2e-3
    grad_clip = 1.0
    eval_batches = 8
    seed = 0


# token ids: 0=PAD, 1=QUERY, 2=FILL, content = 3 .. 3+n_content-1
def vocab_size(cfg):
    return 3 + cfg.n_content


# =====================================================================
# DATA: marked-token recall
# =====================================================================
def make_batch(cfg, device):
    B, L, K = cfg.batch, cfg.L, cfg.K
    C0 = 3
    body = L - K                         # content+filler region; queries take last K
    assert body > K, "L too small for K"
    inp = np.full((B, L), 2, dtype=np.int64)       # FILL
    tgt = np.full((B, L), -100, dtype=np.int64)
    rec = np.zeros((B, L), dtype=bool)
    for b in range(B):
        pos = np.sort(np.random.choice(np.arange(0, body), size=K, replace=False))
        vals = np.random.randint(C0, C0 + cfg.n_content, size=K)
        inp[b, pos] = vals
        for j in range(K):
            qpos = body + j
            inp[b, qpos] = 1                        # QUERY marker
            tgt[b, qpos] = int(vals[j])            # must output j-th content token
            rec[b, qpos] = True
    return (torch.from_numpy(inp).to(device),
            torch.from_numpy(tgt).to(device),
            torch.from_numpy(rec).to(device))


# =====================================================================
# ATTENTION MECHANISMS
# =====================================================================
class DenseAttn(nn.Module):
    def __init__(self, d, h, **_):
        super().__init__()
        self.h, self.dk = h, d // h
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q = q.view(B, L, self.h, self.dk).transpose(1, 2)
        k = k.view(B, L, self.h, self.dk).transpose(1, 2)
        v = v.view(B, L, self.h, self.dk).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(o.transpose(1, 2).reshape(B, L, D))


class WindowAttn(nn.Module):
    """Causal attention restricted to the last W keys (sliding window)."""
    def __init__(self, d, h, budget=32, **_):
        super().__init__()
        self.h, self.dk, self.W = h, d // h, budget
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q = q.view(B, L, self.h, self.dk).transpose(1, 2)
        k = k.view(B, L, self.h, self.dk).transpose(1, 2)
        v = v.view(B, L, self.h, self.dk).transpose(1, 2)
        # banded causal mask: position t attends to [t-W+1, t]
        idx = torch.arange(L, device=x.device)
        diff = idx[:, None] - idx[None, :]
        allow = (diff >= 0) & (diff < self.W)          # (L, L)
        mask = torch.zeros(L, L, device=x.device, dtype=q.dtype)
        mask.masked_fill_(~allow, float("-inf"))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(o.transpose(1, 2).reshape(B, L, D))


class SRMemAttn(nn.Module):
    """Learned bounded memory: keep the top-M tokens by a learned write score;
    queries attend over those M. The soft gate on the score makes the write
    decision trainable."""
    def __init__(self, d, h, budget=32, **_):
        super().__init__()
        self.h, self.dk, self.M = h, d // h, budget
        self.write = nn.Linear(d, 1)
        self.q_proj = nn.Linear(d, d)
        self.kv_proj = nn.Linear(d, 2 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        B, L, D = x.shape
        M = min(self.M, L)
        s = self.write(x).squeeze(-1)                  # (B, L) salience
        topv, topi = s.topk(M, dim=1)                  # (B, M)
        order = topi.argsort(dim=1)                    # keep positional order
        topi = torch.gather(topi, 1, order)
        topv = torch.gather(topv, 1, order)
        gather_idx = topi.unsqueeze(-1).expand(B, M, D)
        mem_tok = torch.gather(x, 1, gather_idx)       # (B, M, D)
        gate = torch.sigmoid(topv).unsqueeze(-1)       # gradient to write score
        mem_tok = mem_tok * gate
        kv = self.kv_proj(mem_tok)
        km, vm = kv.chunk(2, -1)
        km = km.view(B, M, self.h, self.dk).transpose(1, 2)
        vm = vm.view(B, M, self.h, self.dk).transpose(1, 2)
        q = self.q_proj(x).view(B, L, self.h, self.dk).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, km, vm)  # queries over memory
        return self.proj(o.transpose(1, 2).reshape(B, L, D))


class SRCompressAttn(nn.Module):
    """Differentiable bounded memory. M learned slots attend over the full
    sequence to produce M memory vectors (a learned compression). Queries then
    attend over those M. Fully differentiable, so the memory gets a real
    gradient, unlike hard top-M selection. Cost is O(L*M): linear in L, bounded.
    """
    def __init__(self, d, h, budget=32, **_):
        super().__init__()
        self.h, self.dk, self.M = h, d // h, budget
        self.slots = nn.Parameter(0.02 * torch.randn(1, budget, d))
        self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d)
        self.q_proj = nn.Linear(d, d)
        self.mk = nn.Linear(d, d)
        self.mv = nn.Linear(d, d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        B, L, D = x.shape
        # build memory: M slots attend over the sequence
        sl = self.slots.expand(B, self.M, D)
        sq = sl.view(B, self.M, self.h, self.dk).transpose(1, 2)
        ck = self.wk(x).view(B, L, self.h, self.dk).transpose(1, 2)
        cv = self.wv(x).view(B, L, self.h, self.dk).transpose(1, 2)
        mem = F.scaled_dot_product_attention(sq, ck, cv)          # (B,h,M,dk)
        mem = mem.transpose(1, 2).reshape(B, self.M, D)
        # queries attend over the M memory vectors
        q = self.q_proj(x).view(B, L, self.h, self.dk).transpose(1, 2)
        km = self.mk(mem).view(B, self.M, self.h, self.dk).transpose(1, 2)
        vm = self.mv(mem).view(B, self.M, self.h, self.dk).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, km, vm)
        return self.proj(o.transpose(1, 2).reshape(B, L, D))


MECH = {"dense": DenseAttn, "window": WindowAttn, "sr_memory": SRMemAttn,
        "sr_compress": SRCompressAttn}


# =====================================================================
# MODEL
# =====================================================================
class Block(nn.Module):
    def __init__(self, d, h, mech, budget):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = MECH[mech](d, h, budget=budget)
        self.n2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.ffn(self.n2(x))
        return x


class Model(nn.Module):
    def __init__(self, vocab, cfg, mech):
        super().__init__()
        d = cfg.d_model
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(0.02 * torch.randn(1, cfg.L, d))
        self.blocks = nn.ModuleList(
            [Block(d, cfg.n_heads, mech, cfg.budget) for _ in range(cfg.n_layers)])
        self.nf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, ids):
        x = self.emb(ids) + self.pos[:, :ids.size(1)]
        for b in self.blocks:
            x = b(x)
        return self.head(self.nf(x))


def attention_cost(mech, cfg):
    """A simple per-layer attention-cost proxy: queries times keys attended."""
    L, W, M = cfg.L, cfg.budget, cfg.budget
    if mech == "dense":
        return L * L / 2.0            # causal
    if mech == "window":
        return L * W
    if mech == "sr_memory":
        return L * M + L              # attend over M, plus O(L) to score/select
    if mech == "sr_compress":
        return 2 * L * M              # build memory (L*M) + read (L*M)
    return float("nan")


# =====================================================================
# TRAIN / EVAL
# =====================================================================
def train(mech, cfg, device, verbose=False):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = Model(vocab_size(cfg), cfg, mech).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    warm = max(1, cfg.steps // 10)

    def lr_at(s):
        if s < warm:
            return cfg.lr * s / warm
        p = (s - warm) / max(1, cfg.steps - warm)
        return cfg.lr * 0.5 * (1 + math.cos(math.pi * p))

    model.train()
    for s in range(cfg.steps):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(s)
        ids, tgt, _ = make_batch(cfg, device)
        logits = model(ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               tgt.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if verbose and s % max(1, cfg.steps // 6) == 0:
            print(f"    [{mech}] step {s:4d} loss {loss.item():.3f}")
    return model


@torch.no_grad()
def evaluate(model, cfg, device):
    model.eval()
    accs = []
    for _ in range(cfg.eval_batches):
        ids, tgt, _ = make_batch(cfg, device)
        pred = model(ids).argmax(-1)
        m = tgt != -100
        accs.append((pred[m] == tgt[m]).float().mean().item())
    return float(np.mean(accs))


# =====================================================================
# DRIVER
# =====================================================================
def run(cfg, device, L_values, mechs=("dense", "window", "sr_compress"),
        verbose=False):
    out = {"L": list(L_values)}
    for m in mechs:
        out[m] = {"acc": [], "cost": []}
    for L in L_values:
        c = copy.copy(cfg)
        c.L = L
        for m in mechs:
            t0 = time.time()
            model = train(m, c, device, verbose=verbose)
            acc = evaluate(model, c, device)
            out[m]["acc"].append(acc)
            out[m]["cost"].append(attention_cost(m, c))
            print(f"  L={L:6d}  {m:10s} acc={acc:.3f}  "
                  f"cost~{attention_cost(m, c):.2e}  ({time.time()-t0:.0f}s)")
    return out


def plot(out, mechs=("dense", "window", "sr_compress"), path="acc_vs_len.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    color = {"dense": "#4d4d4d", "window": "#d95f0e", "sr_memory": "#6a51a3"}
    label = {"dense": "dense (full context)", "window": "fixed window",
             "sr_memory": "SR bounded memory (ours)"}
    marker = {"dense": "o", "window": "^", "sr_memory": "D"}
    plt.figure(figsize=(6.6, 4.6))
    for m in mechs:
        plt.plot(out["L"], out[m]["acc"], marker[m] + "-", color=color[m],
                 label=label[m])
    plt.xlabel("context length L")
    plt.ylabel("recall accuracy")
    plt.ylim(0, 1.05)
    plt.title("Accuracy vs context length at matched attention budget")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    print("saved", path)


def capacity_sweep(cfg, device, K_values, L_values, M=16, steps=3000,
                   mechs=("dense", "sr_compress"), verbose=False):
    """Stress test: sweep the number of tokens to recall (K) against context
    length (L), at fixed memory size M. dense is the control at every point.

    Returns grid[mech][L_index][K_index] = accuracy.
    Reading it: where sr_compress drops below dense (and dense is near 1.0), the
    bounded memory of size M has run out of capacity. Where dense is also low,
    that point is undertrained and inconclusive; raise steps.
    """
    grid = {m: [[None] * len(K_values) for _ in L_values] for m in mechs}
    for li, L in enumerate(L_values):
        for ki, K in enumerate(K_values):
            c = copy.copy(cfg)
            c.L, c.K, c.budget, c.steps = L, K, M, steps
            for m in mechs:
                t0 = time.time()
                model = train(m, c, device, verbose=verbose)
                acc = evaluate(model, c, device)
                grid[m][li][ki] = acc
                print(f"  L={L:6d} K={K:2d} M={M:3d}  {m:11s} acc={acc:.3f} "
                      f"({time.time()-t0:.0f}s)")
    return {"K": list(K_values), "L": list(L_values), "M": M, "grid": grid}


def plot_capacity(res, path="capacity.png"):
    """Accuracy vs number of recalled tokens K, one line per context length.
    sr_compress solid, dense dashed (control)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    K, L, grid = res["K"], res["L"], res["grid"]
    cmap = ["#6a51a3", "#2c7fb8", "#d95f0e", "#31a354", "#c51b8a"]
    plt.figure(figsize=(6.8, 4.8))
    for li, Lv in enumerate(L):
        c = cmap[li % len(cmap)]
        if "sr_compress" in grid:
            plt.plot(K, grid["sr_compress"][li], "D-", color=c,
                     label=f"SR memory, L={Lv}")
        if "dense" in grid:
            plt.plot(K, grid["dense"][li], "o--", color=c, alpha=0.5,
                     label=f"dense (control), L={Lv}")
    plt.xlabel(f"number of tokens to recall K   (memory size M={res['M']})")
    plt.ylabel("recall accuracy")
    plt.ylim(0, 1.05)
    plt.title("Where does the bounded memory run out of capacity?")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    print("saved", path)


def memory_scaling_sweep(cfg, device, M_values, K_values, L=512, steps=3000,
                         seeds=1, verbose=False):
    """Measure the capacity law: sweep memory size M against recall demand K at
    fixed context length L. dense is trained at every point as a control, and
    accuracy is averaged over `seeds` runs to reduce training variance.

    Returns grid[mech][M_index][K_index] = mean accuracy.
    """
    mechs = ("dense", "sr_compress")
    grid = {m: [[None] * len(K_values) for _ in M_values] for m in mechs}
    for mi, M in enumerate(M_values):
        for ki, K in enumerate(K_values):
            for m in mechs:
                accs = []
                for sd in range(seeds):
                    c = copy.copy(cfg)
                    c.L, c.K, c.budget, c.steps, c.seed = L, K, M, steps, sd
                    model = train(m, c, device, verbose=verbose)
                    accs.append(evaluate(model, c, device))
                grid[m][mi][ki] = float(np.mean(accs))
            d = grid["dense"][mi][ki]
            s = grid["sr_compress"][mi][ki]
            print(f"  M={M:3d} K={K:2d} (L={L})  dense={d:.3f}  "
                  f"sr_memory={s:.3f}  gap={d - s:+.3f}")
    return {"M": list(M_values), "K": list(K_values), "L": L,
            "steps": steps, "seeds": seeds, "grid": grid}


def extract_capacity(res, threshold=0.9):
    """For each memory size M, the usable capacity: the largest K (scanning up
    from the smallest) for which sr_compress reaches `threshold` while dense also
    does. The scan stops at the first K where the task is undertrained (dense
    below threshold, cannot judge) or where sr_compress falls short (a real
    ceiling). Returns a list aligned with res['M']."""
    caps = []
    for mi in range(len(res["M"])):
        cap = 0
        for ki, K in enumerate(res["K"]):
            d = res["grid"]["dense"][mi][ki]
            s = res["grid"]["sr_compress"][mi][ki]
            if d < threshold:
                break                     # undertrained; cannot determine further
            if s >= threshold:
                cap = K
            else:
                break                     # real capacity ceiling
        caps.append(cap)
    return caps


def plot_capacity_law(res, threshold=0.9, path="capacity_law.png"):
    """Two panels: (left) sr_compress accuracy heatmap over M x K; (right) the
    extracted usable capacity vs memory size M."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    M, K = res["M"], res["K"]
    acc = np.array(res["grid"]["sr_compress"], dtype=float)   # (len(M), len(K))
    caps = extract_capacity(res, threshold)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))

    im = ax[0].imshow(acc, origin="lower", aspect="auto", vmin=0, vmax=1,
                      cmap="viridis")
    ax[0].set_xticks(range(len(K)))
    ax[0].set_xticklabels(K)
    ax[0].set_yticks(range(len(M)))
    ax[0].set_yticklabels(M)
    ax[0].set_xlabel("tokens to recall K")
    ax[0].set_ylabel("memory size M")
    ax[0].set_title("SR bounded-memory accuracy")
    for i in range(len(M)):
        for j in range(len(K)):
            ax[0].text(j, i, f"{acc[i, j]:.2f}", ha="center", va="center",
                       color="white" if acc[i, j] < 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax[0], fraction=0.046)

    ax[1].plot(M, caps, "D-", color="#6a51a3", label="usable capacity")
    # reference: capacity proportional to M (fit through origin on the points)
    if any(caps):
        import numpy as np
        Ma = np.array(M, float)
        Ca = np.array(caps, float)
        good = Ca > 0
        if good.sum() >= 1:
            slope = (Ca[good] / Ma[good]).mean()
            ax[1].plot(Ma, slope * Ma, "k--", alpha=0.5,
                       label=f"linear ref (~{slope:.2f}*M)")
    ax[1].set_xlabel("memory size M")
    ax[1].set_ylabel(f"usable capacity  (largest K with acc>={threshold})")
    ax[1].set_title("Capacity law: does capacity scale with memory?")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=140)
    print("saved", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, nargs="+", default=[128, 256, 512])
    p.add_argument("--K", type=int, default=6)
    p.add_argument("--budget", type=int, default=32)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--no_plots", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and
                                     torch.cuda.is_available())
                          else ("cpu" if args.device in ("auto", "cpu")
                                else args.device))
    print("device:", device)
    cfg = CFG()
    cfg.K, cfg.budget, cfg.steps = args.K, args.budget, args.steps
    cfg.d_model, cfg.n_layers, cfg.seed = args.d_model, args.n_layers, args.seed

    out = run(cfg, device, args.L)
    if not args.no_plots:
        try:
            plot(out)
        except Exception as e:
            print("plot skipped:", e)

    print("\nInterpretation:")
    print("  GO signal: sr_compress accuracy stays high and tracks dense as L grows,")
    print("             while fixed window collapses. That shows the LEARNED bounded")
    print("             memory keeps the right tokens, which fixed budgets cannot.")
    print("  NO-GO:     sr_compress collapses with the window. Then the bounded memory")
    print("             is the bottleneck and cannot preserve distant content.")
    print("  Cost: dense ~ L^2, window ~ L*W, sr_compress ~ L*M. At long L the bounded")
    print("        methods are far cheaper; the question here is only accuracy.")


if __name__ == "__main__":
    main()
