"""Plotting utilities. matplotlib is imported lazily so the package can be used
without it (for example in a headless training run)."""
import numpy as np
import torch

from .data import make_batch


def plot_gate_activation(sr_model, cfg, device, path="fig_gate_activation.png",
                         show=True):
    """Gate probability across positions for a few examples. Dashed orange lines
    mark the recall-critical query positions."""
    import matplotlib.pyplot as plt
    sr_model.eval()
    with torch.no_grad():
        ids, tgt, rmask = make_batch(cfg, device)
        _, gate = sr_model(ids, gate_open=False, return_gate=True)
    g = gate.squeeze(-1).cpu().numpy()
    r = rmask.cpu().numpy()
    n_show = min(4, g.shape[0])
    fig, axes = plt.subplots(n_show, 1, figsize=(9, 1.5 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]
    for i in range(n_show):
        axes[i].bar(np.arange(g.shape[1]), g[i], color="#6a51a3", width=0.9)
        for q in np.where(r[i])[0]:
            axes[i].axvline(q, color="#d95f0e", lw=1.0, ls="--", alpha=0.8)
        axes[i].set_ylim(0, 1.05)
        axes[i].set_ylabel(f"ex {i}")
    axes[-1].set_xlabel("token position  (dashed orange = recall-critical query)")
    axes[0].set_title("Learned gate fires on recall-critical positions")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    if show:
        plt.show()
    print("saved", path)


def plot_frontier(results, cfg, path="fig_frontier.png", show=True):
    """Accuracy vs a simple attention-cost proxy for the four models."""
    import matplotlib.pyplot as plt
    n = cfg.n_layers
    cost = {"pure_ssm": 0.0, "pure_attention": 1.0,
            "fixed_hybrid": 1.0 / n,
            "selective_recall": results["selective_recall"].get("attend_frac", 0.2)}
    color = {"pure_ssm": "#2c7fb8", "pure_attention": "#4d4d4d",
             "fixed_hybrid": "#d95f0e", "selective_recall": "#6a51a3"}
    plt.figure(figsize=(6.2, 4.4))
    for name in cost:
        plt.scatter(cost[name], results[name]["accuracy"], s=90,
                    color=color[name], zorder=3)
        plt.annotate(name, (cost[name], results[name]["accuracy"]),
                     textcoords="offset points", xytext=(6, 6), fontsize=8)
    plt.xlabel("relative attention cost  (fraction of tokens/layers attending)")
    plt.ylabel("recall task accuracy")
    plt.ylim(0, 1.05)
    plt.title("Accuracy vs attention cost")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    if show:
        plt.show()
    print("saved", path)


def plot_sweep(sweep, path="fig_sweep.png", show=True):
    """Accuracy vs recall load for state-space, attention, and Selective Recall."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6.2, 4.4))
    plt.plot(sweep["K"], sweep["pure_attention"], "o-", color="#4d4d4d",
             label="pure attention")
    plt.plot(sweep["K"], sweep["pure_ssm"], "s-", color="#2c7fb8",
             label="pure state-space")
    plt.plot(sweep["K"], sweep["selective_recall"], "D-", color="#6a51a3",
             label="Selective Recall (ours)")
    plt.xlabel("recall load  (number of content tokens K)")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.05)
    plt.title("Recall gap grows with load; Selective Recall tracks attention")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    if show:
        plt.show()
    print("saved", path)
