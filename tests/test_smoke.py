"""Fast smoke tests. Run with:  pytest -q

They use a tiny model and few steps so they finish in seconds on CPU. They check
that the code runs, shapes are correct, and the gate diagnostic is produced.
"""
import torch

from selective_recall import CFG, make_batch, seq_len, vocab_size, run_all, MODELS


def _tiny_cfg():
    return CFG(steps=20, d_model=32, n_layers=2, n_heads=2, d_state=4,
              batch=8, eval_batches=2)


def test_batch_shapes():
    cfg = _tiny_cfg()
    ids, tgt, rmask = make_batch(cfg, "cpu")
    L = seq_len(cfg)
    assert ids.shape == (cfg.batch, L)
    assert tgt.shape == (cfg.batch, L)
    assert rmask.shape == (cfg.batch, L)
    # loss is defined exactly at recall positions
    assert int((tgt != -100).sum()) == int(rmask.sum())
    # every id is within the vocabulary
    assert int(ids.max()) < vocab_size(cfg)


def test_run_all_reports_metrics():
    cfg = _tiny_cfg()
    results = run_all(cfg, "cpu")
    for name in MODELS:
        assert "accuracy" in results[name]
        assert 0.0 <= results[name]["accuracy"] <= 1.0
    # only the gated model reports gate diagnostics
    sr = results["selective_recall"]
    for k in ("attend_frac", "gate_precision", "gate_recall"):
        assert k in sr
    assert "_sr_model" in results


def test_gate_is_bounded():
    cfg = _tiny_cfg()
    results = run_all(cfg, "cpu")
    af = results["selective_recall"]["attend_frac"]
    assert 0.0 <= af <= 1.0
