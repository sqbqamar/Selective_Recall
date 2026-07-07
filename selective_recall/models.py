"""Model components for Selective Recall.

  - SelectiveSSM : compact selective state-space layer (Mamba-style S6 core)
  - CausalMHA    : causal multi-head attention
  - RecallGate   : per-token gate predicting whether a position needs recall
  - Block        : one block; kind in {"ssm", "attn", "selrec"}
  - SeqModel     : embedding + blocks + head
  - build        : construct one of the four experimental models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import seq_len, vocab_size


class SelectiveSSM(nn.Module):
    """Compact selective state-space layer (Mamba-style S6 core).

    Diagonal, input-dependent recurrence with an exact sequential scan. This is
    a readable reference implementation, not the CUDA hardware scan; for long
    sequences swap in the official ``mamba-ssm`` kernel (same interface). Order
    is carried by the recurrence, so no positional encoding is required here.
    """
    def __init__(self, d_model, d_state=8, d_conv=3):
        super().__init__()
        self.d_model, self.d_state = d_model, d_state
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.conv1d = nn.Conv1d(d_model, d_model, d_conv,
                                groups=d_model, padding=d_conv - 1)
        self.dt_proj = nn.Linear(d_model, d_model)
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        A_init = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
        self.A_log = nn.Parameter(A_init.repeat(d_model, 1))
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B_, L, _ = x.shape
        xin, z = self.in_proj(x).chunk(2, dim=-1)
        xc = self.conv1d(xin.transpose(1, 2))[..., :L].transpose(1, 2)
        xc = F.silu(xc)
        dt = F.softplus(self.dt_proj(xc))          # (B, L, D)
        Bt = self.B_proj(xc)                       # (B, L, N)
        Ct = self.C_proj(xc)                       # (B, L, N)
        A = -torch.exp(self.A_log)                 # (D, N)
        h = x.new_zeros(B_, self.d_model, self.d_state)
        ys = []
        for t in range(L):
            dt_t = dt[:, t, :]
            A_bar = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            Bx = dt_t.unsqueeze(-1) * Bt[:, t].unsqueeze(1) * xc[:, t].unsqueeze(-1)
            h = A_bar * h + Bx
            ys.append((h * Ct[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1)
        y = y + xc * self.D
        y = y * F.silu(z)
        return self.out_proj(y)


class CausalMHA(nn.Module):
    """Standard causal multi-head attention (positions supplied by the model)."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.h, self.dk).transpose(1, 2)
        k = k.view(B, L, self.h, self.dk).transpose(1, 2)
        v = v.view(B, L, self.h, self.dk).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).contiguous().view(B, L, D))


class RecallGate(nn.Module):
    """Per-token gate: reads the token and the SSM readout, outputs a
    probability that the position needs exact recall."""
    def __init__(self, d_model, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d_model, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x, state_readout):
        return torch.sigmoid(self.net(torch.cat([x, state_readout], dim=-1)))


def straight_through(soft):
    """Forward = hard 0/1 decision; backward = smooth sigmoid gradient."""
    hard = (soft > 0.5).float()
    return hard + soft - soft.detach()


class Block(nn.Module):
    """One block. kind:
        "ssm"    -> selective SSM only
        "attn"   -> causal attention only
        "selrec" -> SSM + gated attention pathway (Selective Recall)
    """
    def __init__(self, d_model, n_heads, d_state, kind):
        super().__init__()
        self.kind = kind
        self.norm1 = nn.LayerNorm(d_model)
        if kind in ("ssm", "selrec"):
            self.ssm = SelectiveSSM(d_model, d_state)
        if kind in ("attn", "selrec"):
            self.attn = CausalMHA(d_model, n_heads)
        if kind == "selrec":
            self.gate = RecallGate(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model))

    def forward(self, x, gate_open=False):
        h = self.norm1(x)
        g_out = None
        if self.kind == "ssm":
            mix = self.ssm(h)
        elif self.kind == "attn":
            mix = self.attn(h)
        else:  # selrec
            y_ssm = self.ssm(h)
            y_attn = self.attn(h)
            g = self.gate(h, y_ssm)
            g_eff = torch.ones_like(g) if gate_open else straight_through(g)
            mix = y_ssm + g_eff * y_attn
            g_out = g
        x = x + mix
        x = x + self.ffn(self.norm2(x))
        return x, g_out


class SeqModel(nn.Module):
    def __init__(self, vocab, d_model, kinds, n_heads, d_state, max_len):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = nn.Parameter(0.02 * torch.randn(1, max_len, d_model))
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_state, k) for k in kinds])
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, ids, gate_open=False, return_gate=False):
        x = self.emb(ids) + self.pos[:, :ids.size(1)]
        gates = []
        for blk in self.blocks:
            x, g = blk(x, gate_open=gate_open)
            if g is not None:
                gates.append(g)
        logits = self.head(self.norm_f(x))
        if return_gate:
            gate = torch.stack(gates, 0).mean(0) if gates else None
            return logits, gate
        return logits


def build(kind_name, cfg):
    """Construct one of: pure_attention, pure_ssm, fixed_hybrid, selective_recall."""
    n = cfg.n_layers
    if kind_name == "pure_attention":
        kinds = ["attn"] * n
    elif kind_name == "pure_ssm":
        kinds = ["ssm"] * n
    elif kind_name == "fixed_hybrid":
        kinds = ["ssm"] * (n - 1) + ["attn"]     # attention at the last layer
    elif kind_name == "selective_recall":
        kinds = ["selrec"] * n
    else:
        raise ValueError(f"unknown model: {kind_name}")
    return SeqModel(vocab_size(cfg), cfg.d_model, kinds,
                    cfg.n_heads, cfg.d_state, seq_len(cfg))
