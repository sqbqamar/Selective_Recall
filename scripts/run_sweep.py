"""Sweep recall load (number of content tokens K) and plot accuracy for the
state-space, attention, and Selective Recall models.

Examples:
    python scripts/run_sweep.py
    python scripts/run_sweep.py --K 2 4 6 8 12 --steps 1000
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")

import torch

from selective_recall import CFG, difficulty_sweep, plot_sweep


def main():
    p = argparse.ArgumentParser(description="Selective Recall difficulty sweep")
    p.add_argument("--K", type=int, nargs="+", default=[2, 4, 6, 8],
                   help="content-token counts to sweep")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--d_model", type=int, default=96)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--d_state", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--out_dir", default="figures")
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    print("device:", device)

    cfg = CFG(d_model=args.d_model, n_layers=args.n_layers,
              d_state=args.d_state, seed=args.seed)
    sweep = difficulty_sweep(cfg, device, K_values=tuple(args.K), steps=args.steps)

    os.makedirs(args.out_dir, exist_ok=True)
    plot_sweep(sweep, path=os.path.join(args.out_dir, "fig_sweep.png"), show=False)
    print(f"figure written to {args.out_dir}/fig_sweep.png")


if __name__ == "__main__":
    main()
