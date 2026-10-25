#!/usr/bin/env python3
"""Train and QAT-distill SPEL on a Colab GPU/CPU/MPS.

This script is a 1:1 PyTorch transliteration of model.c.  It has three stages:

  1. Synthetic + structured ASCII data (vocab=128), including associative
     NoteBank needle-in-a-haystack examples.
  2. Train the 12-layer FP32 target with next-token cross entropy.
  3. Initialize a 4-layer draft from teacher layers [0, 3, 6, 9], then
     QAT-distill it with q8 wqkv/wo, ternary MLP weights and dynamic int8
     activations.  Export a SPELW001 binary with a SPELV001 golden footer.

The C runtime owns the exact quantization and inference semantics.  This file
mirrors it deliberately:

  * RMSNorm: 1/sqrt(mean(x^2) + 1e-5), then multiply by the learned gain.
  * 3-tap causal conv: y[t] = x[t] + conv[0]*x[t-2] + conv[1]*x[t-1].
  * QKV layout: [q(d), k(d), v(d)], then each is [heads][head_dim].
  * RoPE: interleaved even/odd pairs, theta=10000, angle=pos*theta^(-2j/dh).
  * L2 q/k normalization: x/sqrt(sum(x^2)+1e-6).
  * GDN delta rule: S_t = a*S_{t-1} + beta*(v - S_{t-1}k)k^T.
  * Readout: S_t q / sqrt(head_dim).
  * MLP: down(silu(gate(x)) * up(x)).
  * q8 weights: row amax/127, round-half-away-from-zero, clamp [-127,127].
  * ternary weights: 3-step least-squares scale refinement, {-1,0,+1}.
  * dynamic activations: row amax/127, clamp [-127,127], same as tern_gemv.

Usage (Colab):
    !python train_colab.py --steps-target 1200 --steps-qat 1200 \
        --out weights/spel.bin --device auto

Then, in the C repo:
    make
    ./spel verify-weights weights/spel.bin
    SPEL_WEIGHTS=weights/spel.bin make bench
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import struct
import time
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SPEL_ROPE_THETA = 10000.0
RMS_EPS = 1e-5
L2_EPS = 1e-6
GOLDEN_LEN = 16


# --------------------------------------------------------------------- config


@dataclass(frozen=True)
class SpelConfig:
    d_model: int = 384
    n_layers: int = 12
    n_heads: int = 6
    d_ff: int = 1024
    vocab: int = 128
    max_seq: int = 512
    attn_period: int = 4
    conv_k: int = 3
    spec_k: int = 16
    draft_map: tuple[int, int, int, int] = (0, 3, 6, 9)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def is_linear(self, layer: int) -> bool:
        return (layer + 1) % self.attn_period != 0


CFG = SpelConfig()


# ------------------------------------------------------------------ numerics


def round_away(x: torch.Tensor) -> torch.Tensor:
    """C roundf: halfway cases go away from zero (not torch's ties-to-even)."""
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)


def q8_quantize_weight(w: torch.Tensor):
    """Exact q8_quantize_mat: per-row symmetric int8, scale=amax/127."""
    amax = w.abs().amax(dim=1, keepdim=True)
    scale = (amax / 127.0).clamp_min(1e-8)
    q = round_away(w / scale).clamp(-127.0, 127.0)
    return q * scale, q, scale


def q8_quantize_act(x: torch.Tensor) -> torch.Tensor:
    """Exact q8_gemv activation quantizer (per token vector)."""
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = (amax / 127.0).clamp_min(1e-12)
    q = round_away(x / scale).clamp(-127.0, 127.0)
    return q * scale


def ternary_quantize_weight(w: torch.Tensor):
    """Exact tern_quantize_row: 3 least-squares scale refinement steps."""
    n = w.shape[1]
    scale = (w.abs().sum(dim=1) / float(n)).clamp_min(1e-8)
    c = torch.zeros_like(w)
    for _ in range(3):
        c = round_away(w / scale[:, None]).clamp(-1.0, 1.0)
        num = (w * c).sum(dim=1)
        den = (c * c).sum(dim=1)
        scale = torch.where(den > 0, num / den, scale).clamp_min(1e-8)
    c = round_away(w / scale[:, None]).clamp(-1.0, 1.0)
    return c * scale[:, None], c, scale


def ste_weight_q8(w: torch.Tensor) -> torch.Tensor:
    dq, _, _ = q8_quantize_weight(w)
    return w + (dq - w).detach()


def ste_weight_ternary(w: torch.Tensor) -> torch.Tensor:
    dq, _, _ = ternary_quantize_weight(w)
    return w + (dq - w).detach()


def ste_act_q8(x: torch.Tensor) -> torch.Tensor:
    xq = q8_quantize_act(x)
    return x + (xq - x).detach()


# --------------------------------------------------------------------- model


def rmsnorm(x: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    inv = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + RMS_EPS)
    return x * inv * gain


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def rope(x: torch.Tensor, positions: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Interleaved even/odd RoPE, matching model.c's rope_apply exactly."""
    half = head_dim // 2
    j = torch.arange(half, device=x.device, dtype=torch.float32)
    freq = torch.pow(
        torch.tensor(SPEL_ROPE_THETA, device=x.device, dtype=torch.float32),
        -2.0 * j / float(head_dim),
    )
    ang = positions.to(torch.float32)[:, None] * freq[None, :]
    cos = torch.cos(ang)[None, :, None, :]
    sin = torch.sin(ang)[None, :, None, :]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = y1
    out[..., 1::2] = y2
    return out


def l2_normalize_heads(x: torch.Tensor) -> torch.Tensor:
    inv = torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True) + L2_EPS)
    return x * inv


def conv3(x: torch.Tensor, conv: torch.Tensor) -> torch.Tensor:
    """y[t] = x[t] + conv[0]*x[t-2] + conv[1]*x[t-1] (conv_k rows, last unused)."""
    y = x.clone()
    y[:, 2:, :] = y[:, 2:, :] + x[:, :-2, :] * conv[0][None, None, :]
    y[:, 1:, :] = y[:, 1:, :] + x[:, :-1, :] * conv[1][None, None, :]
    return y


class SpelLayer(nn.Module):
    """One target layer, or one distilled draft layer when quant=True."""

    def __init__(self, cfg: SpelConfig, is_linear: bool, quant: bool = False):
        super().__init__()
        self.cfg = cfg
        self.is_linear = is_linear
        self.quant = quant
        d, dff, h = cfg.d_model, cfg.d_ff, cfg.n_heads

        self.n1 = nn.Parameter(torch.ones(d))
        self.wqkv = nn.Parameter(torch.empty(3 * d, d))
        self.wo = nn.Parameter(torch.empty(d, d))
        self.n2 = nn.Parameter(torch.ones(d))
        self.w1 = nn.Parameter(torch.empty(dff, d))
        self.w3 = nn.Parameter(torch.empty(dff, d))
        self.w2 = nn.Parameter(torch.empty(d, dff))
        if is_linear:
            self.conv = nn.Parameter(torch.zeros(cfg.conv_k, d))
            self.wa = nn.Parameter(torch.empty(h))
            self.wb = nn.Parameter(torch.empty(h))
        else:
            self.register_parameter("conv", None)
            self.register_parameter("wa", None)
            self.register_parameter("wb", None)

    # q8 projection (wqkv/wo).  C uses a per-row int8 weight scale and a
    # dynamic per-token int8 activation scale.
    def _q8(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if not self.quant:
            return F.linear(x, w)
        return F.linear(ste_act_q8(x), ste_weight_q8(w))

    # ternary MLP projection (w1/w3/w2), same dynamic int8 activation path.
    def _tern(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if not self.quant:
            return F.linear(x, w)
        return F.linear(ste_act_q8(x), ste_weight_ternary(w))

    def _recurrent(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        B, T, D = x.shape
        H, Dh = cfg.n_heads, cfg.head_dim
        xn = rmsnorm(x, self.n1)
        cx = conv3(xn, self.conv)
        qkv = self._q8(cx, self.wqkv)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, T, H, Dh)
        k = k.view(B, T, H, Dh)
        v = v.view(B, T, H, Dh)
        pos = torch.arange(T, device=x.device)
        q = rope(q, pos, Dh)
        k = rope(k, pos, Dh)
        q = l2_normalize_heads(q)
        k = l2_normalize_heads(k)

        mean = cx.mean(dim=-1)                                  # [B,T]
        a = torch.sigmoid(self.wa[None, None, :] + 0.1 * mean[..., None])
        beta = torch.sigmoid(self.wb[None, None, :] + 0.1 * mean[..., None])

        S = torch.zeros(B, H, Dh, Dh, device=x.device, dtype=x.dtype)
        outs = []
        inv_sqrt = 1.0 / math.sqrt(float(Dh))
        for t in range(T):
            kt = k[:, t]                                        # [B,H,Dh]
            vt = v[:, t]
            delta = vt - torch.einsum("bhij,bhj->bhi", S, kt)
            S = (a[:, t, :, None, None] * S +
                 beta[:, t, :, None, None] * delta[..., None] * kt[..., None, :])
            outs.append(torch.einsum("bhij,bhj->bhi", S, q[:, t]) * inv_sqrt)
        o = torch.stack(outs, dim=1).reshape(B, T, D)
        return x + self._q8(o, self.wo)

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        B, T, D = x.shape
        H, Dh = cfg.n_heads, cfg.head_dim
        xn = rmsnorm(x, self.n1)
        qkv = self._q8(xn, self.wqkv)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, T, H, Dh).transpose(1, 2)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)
        pos = torch.arange(T, device=x.device)
        q = rope(q.transpose(1, 2), pos, Dh).transpose(1, 2)
        k = rope(k.transpose(1, 2), pos, Dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=1.0 / math.sqrt(float(Dh))
        )
        o = o.transpose(1, 2).reshape(B, T, D)
        return x + self._q8(o, self.wo)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._recurrent(x) if self.is_linear else self._attention(x)
        xn = rmsnorm(x, self.n2)
        gate = self._tern(xn, self.w1)
        up = self._tern(xn, self.w3)
        return x + self._tern(silu(gate) * up, self.w2)


class SpelModel(nn.Module):
    def __init__(self, cfg: SpelConfig, quant: bool = False,
                 layer_sources: tuple[int, ...] | None = None):
        super().__init__()
        self.cfg = cfg
        self.quant = quant
        self.layer_sources = tuple(range(cfg.n_layers)) if layer_sources is None else tuple(layer_sources)
        self.emb = nn.Parameter(torch.empty(cfg.vocab, cfg.d_model))
        self.layers = nn.ModuleList(
            [SpelLayer(cfg, cfg.is_linear(src), quant=quant)
             for src in self.layer_sources]
        )
        self.normf = nn.Parameter(torch.ones(cfg.d_model))
        self.wout = nn.Parameter(torch.empty(cfg.vocab, cfg.d_model))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.emb[ids]
        for layer in self.layers:
            x = layer(x)
        x = rmsnorm(x, self.normf)
        return F.linear(x, self.wout)


def init_target(model: SpelModel, seed: int) -> None:
    """Match model.c's initialization distribution."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    cfg = model.cfg
    d = cfg.d_model
    with torch.no_grad():
        model.emb.normal_(0.0, 1.0, generator=g)
        model.wout.normal_(0.0, 1.0 / math.sqrt(d), generator=g)
        for layer in model.layers:
            layer.n1.fill_(1.0)
            layer.n2.fill_(1.0)
            s = 1.0 / math.sqrt(d)
            layer.wqkv.normal_(0.0, s, generator=g)
            layer.wo.normal_(0.0, s, generator=g)
            layer.w1.normal_(0.0, s, generator=g)
            layer.w3.normal_(0.0, s, generator=g)
            layer.w2.normal_(0.0, 1.0 / math.sqrt(cfg.d_ff), generator=g)
            if layer.is_linear:
                layer.conv.zero_()
                layer.conv[0].normal_(0.0, 0.05, generator=g)
                layer.conv[1].normal_(0.0, 0.05, generator=g)
                layer.wa.normal_(2.0, 0.1, generator=g)
                layer.wb.normal_(0.0, 0.1, generator=g)


def make_student(teacher: SpelModel, cfg: SpelConfig) -> SpelModel:
    """Own layer tensors distilled from the teacher; shared emb/norm/head."""
    student = SpelModel(cfg, quant=True, layer_sources=cfg.draft_map)
    for i, src in enumerate(cfg.draft_map):
        student.layers[i].load_state_dict(teacher.layers[src].state_dict())
    student.emb = teacher.emb
    student.normf = teacher.normf
    student.wout = teacher.wout
    for p in (student.emb, student.normf, student.wout):
        p.requires_grad_(False)
    return student


# ------------------------------------------------------------------- dataset


WORDS = [
    "the", "a", "an", "quick", "slow", "brown", "green", "fox", "cat", "dog",
    "runs", "jumps", "walks", "waits", "near", "over", "under", "river", "stone",
    "garden", "window", "morning", "evening", "small", "large", "quiet", "bright",
    "user", "assistant", "hello", "please", "remember", "answer", "question",
    "context", "note", "value", "key", "bank", "chunk", "recall", "copy", "token",
    "we", "you", "they", "it", "is", "was", "and", "but", "then", "because",
]


def make_sequence(rng: random.Random, L: int, recall_prob: float = 0.75) -> list[int]:
    """One ASCII sequence.  The final query forces retrieval from a needle:
    the model sees ``?KEY_00=`` and must predict the value from the context.
    Later value characters can be copied from the already predicted prefix.
    """
    if rng.random() < recall_prob:
        chars = []
        while len(chars) < L:
            chars.extend(rng.choice(WORDS))
            chars.append(" ")
        chars = chars[:L]

        key = rng.randrange(16)
        value = f"{rng.randrange(10000):04d}"
        needle = f"KEY_{key:02d}={value};"
        # Three copies in three 128-token chunks (NoteBank-style multi-chunk
        # redundancy).  Positions leave room for the tail query.
        for base in (24, 152, 280):
            pos = base + rng.randrange(0, 12)
            if pos + len(needle) < L - 32:
                chars[pos:pos + len(needle)] = needle

        tail = f"?KEY_{key:02d}={value}."
        pos = L - len(tail)
        chars[pos:L] = tail
        return [ord(c) & 127 for c in chars]

    # Structured dialog / prose.  This keeps the model from collapsing to a
    # pure copy circuit and gives the target useful next-token statistics.
    lines = [
        "user: what is in the note bank?",
        "assistant: the note bank stores short keys and values.",
        "user: how should i recall a value?",
        "assistant: read the key, then copy the value from the matching note.",
        "user: where are the notes?",
        "assistant: they are spread across the context in small chunks.",
        "user: what happens at the end?",
        "assistant: the query asks for one key and the answer follows.",
    ]
    text = " ".join(rng.choice(lines) for _ in range(L))
    return [ord(c) & 127 for c in text[:L]]


def make_pool(n: int, L: int, seed: int, recall_prob: float = 0.75) -> torch.Tensor:
    rng = random.Random(seed)
    rows = [make_sequence(rng, L, recall_prob) for _ in range(n)]
    return torch.tensor(rows, dtype=torch.long)


def sample_batch(pool: torch.Tensor, batch_size: int, device: torch.device):
    idx = torch.randint(0, pool.shape[0], (batch_size,), device="cpu")
    x = pool[idx].to(device)
    y = torch.full_like(x, -100)
    y[:, :-1] = x[:, 1:]
    return x, y


# ------------------------------------------------------------------ training


def lr_at(step: int, steps: int, base_lr: float, warmup: int = 20) -> float:
    if step < warmup:
        return base_lr * float(step + 1) / float(warmup)
    t = float(step - warmup) / float(max(1, steps - warmup))
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * min(1.0, t)))


def train_target(model: SpelModel, pool: torch.Tensor, args, device: torch.device):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_target, weight_decay=0.01)
    steps = args.steps_target
    t0 = time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, args.lr_target)
        x, y = sample_batch(pool, args.batch_size, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, model.cfg.vocab),
                               y.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        opt.step()
        if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == steps:
            print(f"  [target] step {step+1:5d}/{steps}  ce={loss.item():.4f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}  "
                  f"{(time.time()-t0):.1f}s")
    print(f"  [target] done in {time.time()-t0:.1f}s")


def train_qat(teacher: SpelModel, student: SpelModel, pool: torch.Tensor,
              args, device: torch.device):
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student.train()
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr_qat, weight_decay=0.0)
    steps = args.steps_qat
    t0 = time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, args.lr_qat)
        x, y = sample_batch(pool, args.batch_size, device)
        with torch.no_grad():
            teacher_logits = teacher(x)
        student_logits = student(x)
        ce = F.cross_entropy(student_logits.reshape(-1, student.cfg.vocab),
                             y.reshape(-1), ignore_index=-100)
        T = args.temperature
        kl = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1).reshape(-1, student.cfg.vocab),
            F.softmax(teacher_logits / T, dim=-1).reshape(-1, student.cfg.vocab),
            reduction="batchmean",
        ) * (T * T)
        loss = ce + args.kl_weight * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        opt.step()
        if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == steps:
            print(f"  [qat] step {step+1:5d}/{steps}  ce={ce.item():.4f}  "
                  f"kl={kl.item():.4f}  loss={loss.item():.4f}  "
                  f"{(time.time()-t0):.1f}s")
    print(f"  [qat] done in {time.time()-t0:.1f}s")


@torch.no_grad()
def evaluate(teacher: SpelModel, student: SpelModel, pool: torch.Tensor,
             args, device: torch.device, batches: int = 8):
    teacher.eval()
    student.eval()
    V = student.cfg.vocab
    loss_sum = 0.0
    tok_correct = tok_total = 0
    copy_correct = copy_total = 0
    agree = agree_total = 0
    eq = ord("=")
    for _ in range(batches):
        x, y = sample_batch(pool, args.batch_size, device)
        lt = teacher(x)
        ls = student(x)
        loss = F.cross_entropy(ls.reshape(-1, V), y.reshape(-1),
                               ignore_index=-100, reduction="sum")
        loss_sum += float(loss)
        mask = y.ne(-100)
        tok_correct += int((ls.argmax(-1)[mask] == y[mask]).sum())
        tok_total += int(mask.sum())
        # Copy accuracy: first value digit after '='.  This is the position
        # where the model cannot copy from the query itself.
        prev = x[:, :-1]
        tgt = y[:, :-1]
        pred = ls.argmax(-1)[:, :-1]
        cm = (prev == eq) & (tgt >= ord("0")) & (tgt <= ord("9"))
        copy_correct += int((pred[cm] == tgt[cm]).sum())
        copy_total += int(cm.sum())
        am = mask & (lt.argmax(-1) == ls.argmax(-1))
        agree += int(am.sum())
        agree_total += int(mask.sum())
    return {
        "loss": loss_sum / max(1, tok_total),
        "token_acc": tok_correct / max(1, tok_total),
        "copy_acc": copy_correct / max(1, copy_total),
        "argmax_agree": agree / max(1, agree_total),
    }


# -------------------------------------------------------------------- export


def _f32_bytes(t: torch.Tensor) -> bytes:
    a = t.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return a.astype("<f4", copy=False).tobytes()


def _i32_bytes(a: Iterable[int]) -> bytes:
    return np.asarray(list(a), dtype="<i4").tobytes()


def _write_layer(f, layer: SpelLayer, cfg: SpelConfig, src_layer: int) -> None:
    # Exact SPELW001 per-layer order used by model.c save_layer/load_layer.
    f.write(_f32_bytes(layer.n1))
    f.write(_f32_bytes(layer.wqkv))
    f.write(_f32_bytes(layer.wo))
    if cfg.is_linear(src_layer):
        f.write(_f32_bytes(layer.conv))
        f.write(_f32_bytes(layer.wa))
        f.write(_f32_bytes(layer.wb))
    f.write(_f32_bytes(layer.n2))
    f.write(_f32_bytes(layer.w1))
    f.write(_f32_bytes(layer.w3))
    f.write(_f32_bytes(layer.w2))


def golden_logits(teacher: SpelModel, student: SpelModel, cfg: SpelConfig):
    """Run the same 16-token sequence the C verifier will use, from pos 0."""
    tokens = torch.tensor([(i * 29 + 7) % cfg.vocab for i in range(GOLDEN_LEN)],
                          dtype=torch.long)
    with torch.no_grad():
        lt = teacher(tokens[None])[0].to(device="cpu", dtype=torch.float32)
        ls = student(tokens[None])[0].to(device="cpu", dtype=torch.float32)
    return tokens, lt, ls


def export_weights(teacher: SpelModel, student: SpelModel, cfg: SpelConfig,
                   path: str, meta: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    teacher.eval()
    student.eval()
    # Golden is generated on CPU in float32 so the file is device-independent.
    teacher_cpu = teacher.to("cpu")
    student_cpu = student.to("cpu")
    tokens, gt, gs = golden_logits(teacher_cpu, student_cpu, cfg)

    # header: 8-byte magic + 14 u32 = 64 bytes.  The last u32 is
    # has_golden_footer, matching model.c's write_header/read_header.
    header = struct.pack(
        "<8s14I",
        b"SPELW001",
        1,                    # version
        cfg.d_model,
        cfg.n_layers,
        cfg.n_heads,
        cfg.head_dim,
        cfg.d_ff,
        cfg.vocab,
        cfg.attn_period,
        len(cfg.draft_map),   # n_draft_layers
        cfg.draft_map[0],
        cfg.draft_map[1],
        cfg.draft_map[2],
        cfg.draft_map[3],
        1,                    # has_golden_footer
    )
    assert len(header) == 64, len(header)

    with open(path, "wb") as f:
        f.write(header)
        # Shared token embedding.
        f.write(_f32_bytes(teacher_cpu.emb))
        # All target layers, exact save order.
        for l in range(cfg.n_layers):
            _write_layer(f, teacher_cpu.layers[l], cfg, l)
        # Final norm and LM head.
        f.write(_f32_bytes(teacher_cpu.normf))
        f.write(_f32_bytes(teacher_cpu.wout))
        # Distilled 4-layer draft (FP32; model.c quantizes on load).
        for l, src in enumerate(cfg.draft_map):
            _write_layer(f, student_cpu.layers[l], cfg, src)

        # SPELV001 golden footer.
        f.write(b"SPELV001")
        f.write(struct.pack("<I", GOLDEN_LEN))
        f.write(_i32_bytes(tokens.tolist()))
        f.write(_f32_bytes(gt))
        f.write(_f32_bytes(gs))

    size = os.path.getsize(path)
    meta = dict(meta)
    meta.update({
        "format": "SPELW001",
        "golden": "SPELV001",
        "out": path,
        "bytes": size,
        "config": asdict(cfg),
    })
    meta_path = os.path.splitext(path)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[export] wrote {path} ({size/1048576:.2f} MiB)")
    print(f"[export] wrote {meta_path}")
    print("[export] C check: ./spel verify-weights " + path)


# ---------------------------------------------------------------------- main


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    p = argparse.ArgumentParser(description="SPEL target training + QAT distillation")
    p.add_argument("--steps-target", type=int, default=1200)
    p.add_argument("--steps-qat", type=int, default=1200)
    p.add_argument("--out", type=str, default="weights/spel.bin")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=384)
    p.add_argument("--pool-size", type=int, default=2048)
    p.add_argument("--lr-target", type=float, default=3e-4)
    p.add_argument("--lr-qat", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--kl-weight", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--recall-prob", type=float, default=0.75)
    p.add_argument("--no-export", action="store_true")
    p.add_argument("--save-torch", type=str, default="",
                   help="optional path for a PyTorch state_dict checkpoint")
    return p.parse_args()


def main():
    args = parse_args()
    if args.seq_len < 32:
        raise SystemExit("--seq-len must be >= 32")
    device = pick_device(args.device)
    print(f"== SPEL training/QAT on {device} ==")
    print(f"   config: d={CFG.d_model} layers={CFG.n_layers} heads={CFG.n_heads} "
          f"d_ff={CFG.d_ff} vocab={CFG.vocab} seq={args.seq_len}")
    print(f"   draft: 4 layers {CFG.draft_map}, q8 wqkv/wo + ternary MLP + QAT")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("highest")

    # Stage 1: data.  A fixed pool keeps training fast and reproducible while
    # the generator still covers many key/value combinations.
    print("[1/3] synthetic + associative NoteBank dataset")
    train_pool = make_pool(args.pool_size, args.seq_len, args.seed,
                           args.recall_prob)
    val_pool = make_pool(64, args.seq_len, args.seed ^ 0x5EED,
                         args.recall_prob)
    sample = bytes(train_pool[0].tolist()).decode("ascii", errors="replace")
    print(f"   pool={tuple(train_pool.shape)}  val={tuple(val_pool.shape)}")
    print(f"   sample head: {sample[:120]!r}")
    print(f"   sample tail: {sample[-40:]!r}")

    # Stage 2: target.
    print("[2/3] target training")
    teacher = SpelModel(CFG, quant=False)
    init_target(teacher, args.seed)
    teacher = teacher.to(device)
    if args.steps_target > 0:
        train_target(teacher, train_pool, args, device)
    else:
        print("   skipped (--steps-target 0)")

    # Stage 3: QAT-distilled 4-layer draft.
    print("[3/3] 4-layer QAT distillation")
    student = make_student(teacher, CFG).to(device)
    if args.steps_qat > 0:
        train_qat(teacher, student, train_pool, args, device)
    else:
        print("   skipped (--steps-qat 0)")

    metrics = evaluate(teacher, student, val_pool, args, device,
                       batches=args.eval_batches)
    print("   validation: " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    if args.save_torch:
        torch.save({"target": teacher.state_dict(), "draft": student.state_dict(),
                    "config": asdict(CFG), "metrics": metrics}, args.save_torch)
        print(f"[export] wrote torch checkpoint {args.save_torch}")

    if not args.no_export:
        export_weights(teacher, student, CFG, args.out, {
            "metrics": metrics,
            "steps_target": args.steps_target,
            "steps_qat": args.steps_qat,
            "seed": args.seed,
            "device": str(device),
        })


if __name__ == "__main__":
    main()
