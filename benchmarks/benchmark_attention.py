"""
Go/no-go benchmark for Selective Recall.

Question: does routing attention to only a fraction of tokens, over a bounded
memory, actually save wall-clock time and peak memory at long context, versus
dense full attention?

This measures the ATTENTION PATHWAY in isolation, because that is the only part
the gate changes. Two contenders, as a function of context length L:

  DENSE       standard causal multi-head attention over the full context of L
              tokens, using the efficient fused kernel. Keys and values grow
              with L. Compute grows with L squared.

  SPARSE      the Selective Recall pathway: only rho*L query tokens attend, and
              they attend over a bounded memory of M entries (M fixed, does not
              grow with L). Uses a real gather -> attend -> scatter. Compute
              grows like rho*L*M, i.e. linearly in L for fixed M.

Honest caveats, stated up front:
  1. The full Selective Recall model also runs a state-space layer over all L
     tokens. That cost is linear in L and is NOT included here; this benchmark
     isolates the attention pathway. A linear reference line is plotted so the
     added backbone term can be reasoned about.
  2. The bounded memory is an approximation of full context. Whether M entries
     preserve accuracy is a SEPARATE question (the recall experiments), not
     measured here. This benchmark measures COST only, and assumes the accuracy
     question is answered elsewhere.
  3. Real numbers require a GPU. On CPU the fused attention kernel is not used,
     so CPU timings do not reflect the GPU picture and memory profiling is
     unavailable. Run this on a Colab GPU for the decision.
"""

import argparse
import time
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# Attention pathways
# =====================================================================
class DenseAttention(nn.Module):
    """Standard causal multi-head attention over the full context."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.h, self.dk).transpose(1, 2)
        k = k.view(B, L, self.h, self.dk).transpose(1, 2)
        v = v.view(B, L, self.h, self.dk).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).reshape(B, L, D))


class SparseBoundedAttention(nn.Module):
    """Selective Recall attention pathway.

    A fraction rho of query positions are selected. Those queries are gathered,
    attend over a bounded memory of M entries, and the result is scattered back.
    The memory is the most recent M tokens (a stand-in for the compressed memory;
    its size, not its contents, drives cost).
    """
    def __init__(self, d_model, n_heads, memory_size):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.M = memory_size
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, sel_idx):
        # x: (B, L, D); sel_idx: (B, k) indices of the selected query positions
        B, L, D = x.shape
        k_sel = sel_idx.shape[1]

        # bounded memory: the last M tokens -> keys, values
        mem = x[:, -self.M:, :]                                  # (B, M, D)
        kv = self.kv_proj(mem)
        kmem, vmem = kv.chunk(2, dim=-1)
        kmem = kmem.view(B, self.M, self.h, self.dk).transpose(1, 2)
        vmem = vmem.view(B, self.M, self.h, self.dk).transpose(1, 2)

        # gather the selected queries: (B, k, D)
        gather_idx = sel_idx.unsqueeze(-1).expand(B, k_sel, D)
        q_sel = torch.gather(x, 1, gather_idx)
        q = self.q_proj(q_sel).view(B, k_sel, self.h, self.dk).transpose(1, 2)

        # selected queries attend over the bounded memory
        out = F.scaled_dot_product_attention(q, kmem, vmem)      # (B, h, k, dk)
        out = self.proj(out.transpose(1, 2).reshape(B, k_sel, D))

        # scatter back into a full-length output
        y = x.new_zeros(B, L, D)
        y.scatter_(1, gather_idx, out)
        return y


def make_selection(B, L, k, device):
    """Random distinct selected positions per row: (B, k) long."""
    idx = torch.stack([torch.randperm(L, device=device)[:k] for _ in range(B)])
    return idx.sort(dim=1).values


# =====================================================================
# Measurement
# =====================================================================
def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure_latency(fn, iters=10, warmup=3):
    """Median forward latency in milliseconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        _sync_dev()
        t0 = time.perf_counter()
        fn()
        _sync_dev()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


# module-level device for sync (set in run)
_DEVICE = torch.device("cpu")
def _sync_dev():
    if _DEVICE.type == "cuda":
        torch.cuda.synchronize()


def measure_peak_memory(fn):
    """Peak allocated memory in MB during fn (CUDA only)."""
    if _DEVICE.type != "cuda":
        return None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


# =====================================================================
# Benchmark driver
# =====================================================================
def run(L_values, rho_values, M, d_model, n_heads, batch, device, dtype,
        iters, warmup):
    global _DEVICE
    _DEVICE = device
    dense = DenseAttention(d_model, n_heads).to(device=device, dtype=dtype).eval()

    results = []
    print(f"\ndevice={device.type}  dtype={dtype}  d_model={d_model}  "
          f"heads={n_heads}  batch={batch}  memory M={M}")
    print(f"iters={iters} (warmup {warmup})\n")
    header = ["L", "dense_ms"] + [f"sparse_ms(r={r})" for r in rho_values]
    if device.type == "cuda":
        header += ["dense_MB"] + [f"sparse_MB(r={r})" for r in rho_values]
    print("  ".join(f"{h:>16s}" for h in header))

    with torch.no_grad():
        for L in L_values:
            x = torch.randn(batch, L, d_model, device=device, dtype=dtype)

            # dense
            try:
                dense_ms = measure_latency(lambda: dense(x), iters, warmup)
                dense_mb = measure_peak_memory(lambda: dense(x))
                dense_ok = True
            except RuntimeError as e:            # e.g. OOM at large L
                dense_ms, dense_mb, dense_ok = float("nan"), None, False
                torch.cuda.empty_cache() if device.type == "cuda" else None

            row = {"L": L, "dense_ms": dense_ms, "dense_MB": dense_mb}
            sparse_ms_list, sparse_mb_list = [], []
            for r in rho_values:
                k = max(1, int(r * L))
                k = min(k, L)
                sparse = SparseBoundedAttention(
                    d_model, n_heads, min(M, L)).to(device=device, dtype=dtype).eval()
                sel = make_selection(batch, L, k, device)
                sms = measure_latency(lambda: sparse(x, sel), iters, warmup)
                smb = measure_peak_memory(lambda: sparse(x, sel))
                sparse_ms_list.append(sms)
                sparse_mb_list.append(smb)
            row["sparse_ms"] = sparse_ms_list
            row["sparse_MB"] = sparse_mb_list
            results.append(row)

            cells = [f"{L:>16d}", f"{dense_ms:>16.2f}"]
            cells += [f"{s:>16.2f}" for s in sparse_ms_list]
            if device.type == "cuda":
                cells += [f"{(dense_mb or 0):>16.1f}"]
                cells += [f"{(s or 0):>16.1f}" for s in sparse_mb_list]
            print("  ".join(cells))
    return results


def plot(results, rho_values, device_type, path_prefix="bench"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ls = [r["L"] for r in results]
    # latency
    plt.figure(figsize=(6.4, 4.6))
    plt.plot(Ls, [r["dense_ms"] for r in results], "o-", color="#4d4d4d",
             label="dense full attention")
    colors = ["#6a51a3", "#2c7fb8", "#d95f0e", "#31a354"]
    for j, rho in enumerate(rho_values):
        plt.plot(Ls, [r["sparse_ms"][j] for r in results], "s-",
                 color=colors[j % len(colors)],
                 label=f"sparse+bounded (rho={rho})")
    plt.xlabel("context length L")
    plt.ylabel("forward latency (ms)")
    plt.title("Attention pathway latency vs context length")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{path_prefix}_latency.png", dpi=140)
    print(f"saved {path_prefix}_latency.png")

    # latency, log-log (to see quadratic vs linear slopes)
    plt.figure(figsize=(6.4, 4.6))
    plt.loglog(Ls, [r["dense_ms"] for r in results], "o-", color="#4d4d4d",
               label="dense full attention")
    for j, rho in enumerate(rho_values):
        plt.loglog(Ls, [r["sparse_ms"][j] for r in results], "s-",
                   color=colors[j % len(colors)], label=f"sparse (rho={rho})")
    plt.xlabel("context length L (log)")
    plt.ylabel("latency ms (log)")
    plt.title("Slopes: dense ~ quadratic, sparse+bounded ~ linear")
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(f"{path_prefix}_latency_loglog.png", dpi=140)
    print(f"saved {path_prefix}_latency_loglog.png")

    if device_type == "cuda" and results[0]["dense_MB"] is not None:
        plt.figure(figsize=(6.4, 4.6))
        plt.plot(Ls, [r["dense_MB"] for r in results], "o-", color="#4d4d4d",
                 label="dense full attention")
        for j, rho in enumerate(rho_values):
            plt.plot(Ls, [r["sparse_MB"][j] for r in results], "s-",
                     color=colors[j % len(colors)], label=f"sparse (rho={rho})")
        plt.xlabel("context length L")
        plt.ylabel("peak memory (MB)")
        plt.title("Attention pathway peak memory vs context length")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{path_prefix}_memory.png", dpi=140)
        print(f"saved {path_prefix}_memory.png")


def main():
    p = argparse.ArgumentParser(description="Selective Recall attention benchmark")
    p.add_argument("--L", type=int, nargs="+",
                   default=[512, 1024, 2048, 4096, 8192])
    p.add_argument("--rho", type=float, nargs="+", default=[0.1, 0.2, 0.5])
    p.add_argument("--M", type=int, default=256, help="bounded memory size")
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", default="auto", choices=["auto", "fp16", "fp32"])
    p.add_argument("--no_plots", action="store_true")
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if args.dtype == "auto":
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    else:
        dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    if device.type == "cpu":
        print("WARNING: running on CPU. The fused attention kernel is not used "
              "and memory profiling is unavailable, so these numbers do NOT "
              "reflect the GPU picture. Use a GPU for the go/no-go decision.")

    results = run(args.L, args.rho, args.M, args.d_model, args.n_heads,
                  args.batch, device, dtype, args.iters, args.warmup)

    if not args.no_plots:
        try:
            plot(results, args.rho, device.type)
        except Exception as e:
            print("plotting skipped:", e)

    # crossover summary
    print("\nInterpretation:")
    for j, r in enumerate(args.rho):
        crossed = None
        for row in results:
            d, s = row["dense_ms"], row["sparse_ms"][j]
            if s == s and d == d and s < d:      # not nan and sparse faster
                crossed = row["L"]
                break
        if crossed is not None:
            print(f"  rho={r}: sparse+bounded becomes faster than dense at "
                  f"L >= {crossed}")
        else:
            print(f"  rho={r}: sparse+bounded was NOT faster than dense at any "
                  f"tested L (dense kernel wins in this range)")
    print("\nReminder: this isolates the attention pathway. The full model adds "
          "a state-space term linear in L, and the bounded memory's effect on "
          "accuracy is a separate question measured by the recall experiments.")


if __name__ == "__main__":
    main()
