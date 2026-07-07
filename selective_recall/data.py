"""Selective Copy with Filler (SCF) task.

The sequence starts with K content tokens, followed by a run of predictable
filler interleaved with query markers. At each query the model must reproduce
one content token from before the filler. Only the query positions require
recall, so recall is sparse.

  token ids: 0=PAD, 1=SEP, 2=QUERY, 3=FILL, content = 4 .. 4+n_content-1
"""
import numpy as np
import torch


def seq_len(cfg):
    """Length of the (LM-shifted) input sequence."""
    return (cfg.K + 1) + cfg.K * (cfg.gap + 2) - 1


def vocab_size(cfg):
    return 4 + cfg.n_content


def make_batch(cfg, device):
    """Generate one batch.

    Returns:
        input_ids   (batch, L) int64
        target_ids  (batch, L) int64, -100 except at query positions
        recall_mask (batch, L) bool, True exactly at query positions (oracle label)
    """
    L = seq_len(cfg)
    B = cfg.batch
    C0 = 4
    full = np.full((B, L + 1), 3, dtype=np.int64)      # default FILL
    rec = np.zeros((B, L + 1), dtype=bool)
    for b in range(B):
        content = np.random.randint(C0, C0 + cfg.n_content, size=cfg.K)
        seq = list(content) + [1]                       # content block + SEP
        for j in range(cfg.K):
            seq += [3] * cfg.gap + [2, int(content[j])] # filler, QUERY, value
        full[b, :len(seq)] = seq
        pos = cfg.K + 1
        for j in range(cfg.K):
            rec[b, pos + cfg.gap] = True                # mark the QUERY position
            pos += cfg.gap + 2
    full = torch.from_numpy(full).to(device)
    rec = torch.from_numpy(rec).to(device)
    input_ids = full[:, :-1].contiguous()
    target_ids = full[:, 1:].clone()                    # LM shift
    recall_mask = rec[:, :-1].contiguous()
    target_ids[~recall_mask] = -100                     # loss only at queries
    return input_ids, target_ids, recall_mask
