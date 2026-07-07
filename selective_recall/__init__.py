"""Selective Recall: learning when to attend in a linear-time backbone."""
from .config import CFG
from .data import make_batch, seq_len, vocab_size
from .models import (
    SelectiveSSM, CausalMHA, RecallGate, Block, SeqModel, build,
    straight_through,
)
from .train import train_model, evaluate
from .experiments import run_all, print_table, difficulty_sweep, MODELS
from .plots import plot_gate_activation, plot_frontier, plot_sweep

__version__ = "0.1.0"

__all__ = [
    "CFG",
    "make_batch", "seq_len", "vocab_size",
    "SelectiveSSM", "CausalMHA", "RecallGate", "Block", "SeqModel", "build",
    "straight_through",
    "train_model", "evaluate",
    "run_all", "print_table", "difficulty_sweep", "MODELS",
    "plot_gate_activation", "plot_frontier", "plot_sweep",
]
