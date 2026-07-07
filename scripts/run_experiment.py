"""Train all four models on the selective-copy task, print the comparison
table, and save the figures.

Examples:
    python scripts/run_experiment.py
    python scripts/run_experiment.py --steps 1500 --K 8 --gap 3
    python scripts/run_experiment.py --steps 300 --device cpu   # quick check
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")   # headless-safe; figures are saved to disk

import torch

from selective_recall import (
    CFG, run_all, print_table, plot_gate_activation, plot_frontier,
)


def main():
    p = argparse.ArgumentParser(description="Selective Recall experiment")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--K", type=int, default=8, help="content tokens (recall load)")
    p.add_argument("--gap", type=int, default=3, help="filler between queries")
    p.add_argument("--d_model", type=int, default=96)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--d_state", type=int, default=8)
    p.add_argument("--rho", type=float, default=0.20,
                   help="target attending fraction")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--out_dir", default="figures")
    p.add_argument("--no_plots", action="store_true")
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    print("device:", device)

    cfg = CFG(steps=args.steps, K=args.K, gap=args.gap, d_model=args.d_model,
              n_layers=args.n_layers, d_state=args.d_state, rho=args.rho,
              seed=args.seed)

    results = run_all(cfg, device)
    print_table(results)

    if not args.no_plots:
        os.makedirs(args.out_dir, exist_ok=True)
        plot_gate_activation(results["_sr_model"], cfg, device,
                             path=os.path.join(args.out_dir, "fig_gate_activation.png"),
                             show=False)
        plot_frontier(results, cfg,
                      path=os.path.join(args.out_dir, "fig_frontier.png"),
                      show=False)
        print(f"figures written to {args.out_dir}/")


if __name__ == "__main__":
    main()
