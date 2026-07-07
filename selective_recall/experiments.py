"""Experiment drivers: train all models, print the table, sweep difficulty."""
import copy
import time

from .train import train_model, evaluate

MODELS = ["pure_attention", "pure_ssm", "fixed_hybrid", "selective_recall"]


def run_all(cfg, device, verbose=False):
    """Train and evaluate all four models on the same data and budget.

    Returns a dict keyed by model name, plus "_sr_model" (the trained Selective
    Recall model, kept for the gate-activation plot).
    """
    results = {}
    for name in MODELS:
        t0 = time.time()
        model = train_model(name, cfg, device, verbose=verbose)
        res = evaluate(model, cfg, device)
        res["seconds"] = time.time() - t0
        results[name] = res
        if name == "selective_recall":
            results["_sr_model"] = model
        print(f"  trained {name:18s} acc={res['accuracy']:.3f} "
              f"({res['seconds']:.0f}s)")
    return results


def print_table(results):
    """Print a formatted comparison table."""
    print("\n" + "=" * 78)
    print(f"{'model':20s}{'accuracy':>10s}{'attend_frac':>13s}"
          f"{'gate_P':>9s}{'gate_R':>9s}{'sec':>7s}")
    print("-" * 78)
    for name in MODELS:
        r = results[name]
        af = r.get("attend_frac")
        gp = r.get("gate_precision")
        gr = r.get("gate_recall")
        af_s = (f"{af:.3f}" if af is not None
                else ("1.000" if name == "pure_attention" else "-"))
        print(f"{name:20s}{r['accuracy']:>10.3f}{af_s:>13s}"
              f"{(f'{gp:.3f}' if gp is not None else '-'):>9s}"
              f"{(f'{gr:.3f}' if gr is not None else '-'):>9s}"
              f"{r['seconds']:>7.0f}")
    print("=" * 78)
    print(f"(task recall density = "
          f"{results['pure_attention']['recall_density']:.3f}; "
          f"a random gate would score about this precision)")


def difficulty_sweep(cfg, device, K_values=(2, 4, 6, 8), steps=None):
    """Accuracy vs recall load (number of content tokens K) for the state-space,
    attention, and Selective Recall models."""
    out = {"K": list(K_values), "pure_ssm": [], "pure_attention": [],
           "selective_recall": []}
    for K in K_values:
        c = copy.copy(cfg)
        c.K = K
        if steps is not None:
            c.steps = steps
        for name in ["pure_ssm", "pure_attention", "selective_recall"]:
            m = train_model(name, c, device)
            out[name].append(evaluate(m, c, device)["accuracy"])
        print(f"  K={K:2d}  ssm={out['pure_ssm'][-1]:.3f}  "
              f"attn={out['pure_attention'][-1]:.3f}  "
              f"sr={out['selective_recall'][-1]:.3f}")
    return out
