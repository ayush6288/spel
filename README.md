# SPEL — NoteBank + SubTernary research harness

A self-contained C11 harness for four things on Apple silicon:

1. **2-bit ternary GEMV** with ARM NEON (`{-1, 0, +1}`, per-row scale).
2. **A deterministic hybrid model** — Gated-DeltaNet-style recurrent layers plus
   RoPE attention — with batched verification and checkpoints.
3. **Transactional speculative decoding** with O(1) commit/rollback, a deferred
   correction token, and a governor that falls back to autoregressive (AR)
   decoding.
4. **A NoteBank-style writer, compact address bank, state branching and replay
   demo**, wired through to the speculative loop.

> **Status: research harness, not a model release.** The weights are randomly
> initialized and untrained, so generated text is meaningless and real
> acceptance is near the unigram rate. There is no trained checkpoint, no
> dataset, no tokenizer, no QAT run, and no llama.cpp/GGUF integration. The
> measured value here is the runtime, exactness machinery and cost model.
> The 2.7x speculative projection below is explicitly a cost-model projection at
> a hypothetical acceptance rate, not a measured generation speedup.

## Build and run

Requirements: macOS on ARM64 with NEON dot-product support, clang, and the
Accelerate framework. The Makefile targets the M5 with `-mcpu=apple-m5`.

```sh
git clone https://github.com/ayush6288/spel.git
cd spel
make                 # builds ./spel, links -framework Accelerate
make selftest        # correctness tests; must print PASS (0 failures)
make bench           # kernels + strict/fast verification + decode + rollback
make demo            # NoteBank branch/replay/spec + a 48-token spec run

# Compare speculative K on the same machine:
SPEL_BENCH_K=8  ./spel bench
SPEL_BENCH_K=16 ./spel bench
```

On a different ARM64 Mac, override the target:

```sh
make clean && make ARCH="-mcpu=apple-m1"
# or
make clean && make ARCH="-march=native"
```

## Strict mode and fast mode

Two arithmetic paths now exist:

- **Strict (default, `spel_use_accelerate = 0`).** Batched projections use the
  explicit NEON dot-product order used by single-token AR. Batched verification
  is bit-identical to sequential AR. All correctness tests run here.
- **Fast (`spel_use_accelerate = 1`).** Batched projections (`M > 1`) call
  `cblas_sgemm`, which uses Accelerate/AMX. This changes the floating-point
  summation order. It is **token-equivalent in the tested cases, not
  bit-identical**. On the 8-layer test model at K=16: 24/24 greedy tokens match
  strict AR, while 248/256 logits differ in their bit patterns (argmax differs
  at 0/4 positions).

The benchmark switches to fast mode only for the verification/spec phases; its
AR and draft-only baselines stay strict. There is no claim that fast mode
preserves bit-exactness, and no claim that it is correct for every input.

## What changed from the first version

- `cblas_sgemm` for target verification. Measured verification cost drops from
  **8.5–15.9x** one AR token (strict) to **2.2–2.8x** (fast) for M=9–17. It does
  **not** reach the plan's projected ~1.1x.
- The target's correction token is **deferred** instead of immediately forwarded.
  The next round feeds it as the first token of the target's `K+1` verification
  batch and consumes it in the draft. This removes the separate target `+1`
  pass; it does not remove the draft `+1` pass.
- The draft is a **4-of-12-layer subset** `[0, 3, 6, 9]`, with **int8 `wqkv`/
  `wo`** (`vdotq_s32`) and **ternary MLP**. It shares embeddings, output
  projection, norms and RoPE with the target. Measured draft cost is
  **c = 0.12–0.13x** one target token (was 0.58–0.61x for the 12-layer draft).
- K=16 is supported. `SPEL_MAX_BATCH = 16`, verification batches are up to
  M=17, and the recurrent ring is `R = K+1 = 17`. The short-conv history ring was
  increased to 32 because a 17-token batch can overwrite the 16-slot history
  needed after a truncated rewind. The K=16 selftest caught this; it now passes.
- NoteBank has an opt-in **per-channel integer voting** ranker
  (`NB_RANK_VOTES`) in addition to the default summed-similarity ranker.
- The demo branches `S_T`: address the query from one branch, restore `S_T`,
  replay the selected notes and query, and enter the speculative loop. Replay
  is still token re-prefill, not native state injection.

## Architecture / benchmark configuration

```text
target: d_model=384, layers=12, heads=6, head_dim=64
        d_ff=1024, vocab=128, max_seq=512 (4096 hard cap)
        attn_period=4 -> 9 recurrent + 3 attention layers
        ~21.35M FP32 parameters, 85.4 MB
draft:  4 layers [0,3,6,9], q8 wqkv/wo, ternary MLP
        shared embedding/output/norms/RoPE
spec:   SPEL_BENCH_K=16 by default (R=17, M up to 17)
```

Each layer is `RMSNorm -> mixer -> residual -> RMSNorm -> SwiGLU MLP ->
residual`.

- **Recurrent mixer:** short causal conv (3 taps), QKV projection, L2-normalized
  q/k, per-head decay and write rate, and the delta-rule update

  ```text
  S_t = a * S_{t-1} + beta * (v - S_{t-1} k) k^T
  o   = S_t q / sqrt(head_dim)
  ```

  The state is `[heads][head_dim][head_dim]` per layer and is ring-buffered in
  `spec_k+1` slots.
- **Attention mixer:** causal RoPE attention with a conventional KV cache.
- **Batched verification:** `spel_forward(..., M=K, ...)` or `M=K+1` computes the
  batch in one pass. In strict mode the result is bit-identical to the same
  number of one-token passes from the same checkpoint.

## Speculative decoding

A steady-state round works as follows, with a deferred correction token:

1. The draft consumes the pending correction token from the previous round, then
   proposes `K` greedy tokens.
2. The target verifies `[pending, proposal_0, ..., proposal_{K-1}]` in one batch
   of `K+1` tokens (the first round has no pending token and uses `K`).
3. The longest matching prefix `m` is accepted. The next correction token is
   `argmax(target_logits[m])`; it is emitted immediately but **not forwarded**.
4. Rollback is a position reset. Attention KV beyond the new position is masked
   by `pos`; the recurrent state is addressed by ring slot. The next round
   overwrites the stale slots.
5. The correction becomes `pending` for the next round. If generation stops with
   a pending token, it is forwarded once at the end so the final state includes
   every emitted token.

The target's extra correction pass is gone; the draft still pays one token to
consume the pending correction. The governor watches the rolling accepted/
proposed ratio over `window` rounds. If it falls below `alpha_threshold`, or a
round accepts zero proposals, drafting is suspended and the rest of the call
runs AR. It does not resume mid-call.

## Correctness invariants (`make selftest`)

1. **Kernel equivalence** — 50 random matrices: dotprod/addsub/scalar return
   identical int32 dot products; GEMV matches a dequantized reference.
2. **Batch equivalence** — 256 logits from a 4-token batched forward are
   bit-identical to four sequential forwards from the same checkpoint:
   `0 differ`.
3. **Speculative fidelity** — greedy output matches AR token-for-token:
   `40/40` plain (K=4), `32/32` governed (K=4), `32/32` K=16, and `32/32` K=16
   governed. A post-generation probe token produces bit-identical logits at the
   same position in every case, including K=16 with M=17 and a 17-slot ring.
4. **Governor** — forced threshold 1.0 trips, then emits AR rounds. K=16:
   alpha 0.031, 29 AR rounds, 1 trip.
5. **NoteBank** — all three planted needles are contained in retained notes;
   summed-similarity recall@2 is 3/3. The opt-in vote ranker gets 2/3 on these
   random weights.
6. **Fast mode** — strict AR vs fast K=16 speculative output: 24/24 tokens
   match; batched logits differ in 248/256 bit patterns, argmax at 0/4
   positions. This is evidence for the tested inputs, not a bit-exactness claim.

These are same-binary, same-flags guarantees (`-O3 -ffast-math`). Bit-exactness
across compilers, optimization levels or architectures is not claimed.

## Measured results

Apple M5, 10 cores, 16 GB, macOS. Single thread. `clang -O3 -mcpu=apple-m5
-ffast-math`. Ranges are across repeated runs; the decode rows are representative
single runs.

### Ternary GEMV vs FP32 (4096 x 1536)

| kernel | GMAC/s | packed GB/s | ms/GEMV |
|---|---:|---:|---:|
| neon-dotprod | 17.58–17.78 | 4.4 | 0.35–0.36 |
| neon-addsub | 10.58–10.75 | 2.6–2.7 | 0.59 |
| scalar | 2.49–2.52 | 0.6 | 2.50–2.52 |
| fp32-neon | 5.44–6.67 | 21.8–26.7 | — |

Packed weights are 1.50 MiB versus 24.00 MiB FP32. The dot-product kernel is
roughly 3x the FP32 per-MAC rate on this shape.

### Verification cost: strict vs fast

`v` is batch time divided by one AR token (2.6–2.8 ms). M is the number of tokens
in the target batch.

| M | strict v | fast v |
|---:|---:|---:|
| 4 | 3.9–4.0 | 1.9–2.0 |
| 5 | 4.8–5.0 | 2.0–2.1 |
| 8 | 7.6–7.9 | 2.2–2.3 |
| 9 | 8.5–8.8 | 2.2–2.3 |
| 16 | 14.7–15.0 | 2.65–2.70 |
| 17 | 15.6–15.9 | 2.69–2.77 |

The strict path scales almost linearly with M. Accelerate amortizes the weight
read and gives roughly a 4–6x reduction at M=17, but it is still **2.7x**, not
the plan's 1.1x. AMX is more efficient at M=16/17 than at M=4/5, which is why
K=16 is the better speculative configuration on this machine.

### Decode: 64 generated tokens, 4-layer q8/ternary draft

| path | K | tok/s | vs AR | notes |
|---|---:|---:|---:|---|
| AR target | — | 360–381 | 1.00x | 2.6–2.8 ms/token |
| draft 4L only | — | 2977–3014 | — | c = 0.12–0.13x target |
| spec raw | 8 | 124 | 0.34x | alpha 0.011, 1.08 tok/round |
| spec governed | 8 | 345–347 | 0.95–0.96x | 1 trip, 63 AR rounds |
| spec oracle | 8 | 266 | 0.73x | alpha 1.000, 8.00 tok/round |
| spec raw | 16 | 84–85 | 0.23x | alpha 0.005, 1.08 tok/round |
| spec governed | 16 | 335–339 | 0.92–1.00x | 1 trip, 63 AR rounds |
| spec oracle | 16 | 298–300 | 0.82–0.88x | alpha 1.000, 16.00 tok/round |

The raw rows are the honest untrained result: the 4-layer draft is almost a
unigram predictor on a random target, so five of 944 proposals are accepted at
K=16. The governor converts that into roughly AR throughput by falling back
after the first failed round. It is a safety mechanism, not a speedup.

The oracle row uses a full FP32 copy of the target as the draft. Even at 100%
acceptance it is **0.82–0.88x AR** at K=16, not 1.8–2.2x, because it still pays
about one full target-sized forward per proposal. Fixing verification alone
cannot make a full-target oracle draft faster than AR.

### Cost model and the projection

In steady state a round emits `1 + alpha*K` tokens on average. The draft pays
`K+1` token forwards (the pending correction plus K proposals) and the target
pays one `K+1` verification batch:

```text
speedup(alpha) = (1 + alpha*K) / ((K+1)*c + v_{K+1})
```

This is the corrected formula; the plan's `K*c + v` omitted the draft's pending
correction forward. With the measured K=16 values (`c=0.12`, `v=2.70`):

```text
alpha=.75: (1 + 12.0) / (17*0.12 + 2.70) = 13.0 / 4.74 = 2.74x
alpha=.80: (1 + 12.8) / (17*0.12 + 2.70) = 13.8 / 4.74 = 2.91x
```

Repeated runs with the same code gave **2.72–2.87x** at alpha=.75 and
**2.88–3.04x** at alpha=.80; the spread is run-to-run frequency/cache variation.
At K=8 (`c=0.12`, `v=2.13`): 2.17–2.18x at alpha=.75 and 2.29–2.31x at
alpha=.80. K=16 is better because the same correction/draft overhead is
amortized over more proposals and Accelerate is more efficient at M=17. To reach
2.5x at K=16 the draft needs alpha >= 0.68 with the currently measured c and v.
Those alpha values are **not measured**; they require training.

### Rollback

| operation | measured |
|---|---:|
| position reset (commit/rollback primitive) | ~0.5 ns |
| memcpy of 4.50 MiB KV | 108–115 us |
| ratio | ~2.4–2.5 x 10^5 |

The important property is that recurrent state is addressed by ring slot, so
rollback never walks or copies the 4.5 MiB attention cache.

### NoteBank demo

| quantity | value |
|---|---:|
| context / chunks / notes | 384 tokens / 3 / 6 |
| note spans | [28,52) [86,110) [158,182) [215,239) [294,318) [319,343) |
| writer time | 0.31–0.38 ms |
| address time (Q=4, R=2) | 0.019–0.023 ms |
| writer needle containment | 3/3 |
| summed-similarity recall@2 | 3/3 |
| vote-ranker recall@2 | 2/3 |
| note storage | 6,720 B |
| dense KV, all 8 layers | 3,145,728 B (468x) |
| hybrid attention KV, 2 layers | 786,432 B (117x) |

**Do not read the 468x/117x as end-to-end memory savings.** The accounting
excludes the normal model state and capture buffers (about 2.25 MiB for this
demo). Notes are not a replacement for attention KV; replay is a separate path.

The demo prints a NoteBank-to-spec run (`notes 0,3 -> ... alpha=0.000 trips=1
AR=7`). That proves the branch/replay/spec plumbing, not retrieval quality: the
random model has no learned retrieval circuit, and replay re-prefills tokens
rather than injecting native state.

## Why the plan's target is not reproduced yet

1. **Verification is 2.7x, not 1.1x.** Accelerate helps, but the target's
   recurrent state update, attention and normalization are serial. The measured
   best is v=2.77 at M=17.
2. **The draft is cheap, but not free.** 4 layers plus q8 projections gives
   c=0.12–0.13x, close to the plan's 0.12 target. That part worked.
3. **Acceptance is the blocker.** Random weights have no induction heads. The
   4-layer draft accepts 0.5–1.1% of proposals. The 2.72x projection assumes
   alpha=.75, which is not achievable without training/distillation.
4. **The oracle argument is wrong.** A full target copy as draft has c≈0.92–1.0;
   its ceiling is `(1+K)/((K+1)c+v)`, below 1x for K=16 even with fast
   verification. The measured oracle is 0.82–0.88x AR. Fixing v cannot produce
   the claimed 1.8–2.2x oracle.
5. **Training is not available in this environment.** `.venv` has Python 3.12.14
   but no `torch` (`ModuleNotFoundError: No module named 'torch'`). There is no
   dataset, tokenizer, pretrained checkpoint, or weight export/loader. The
   plan's "20 minutes on M5 MPS for 100M tokens, alpha 0.72–0.85" claim was not
   reproduced and should not be treated as a measured result.

To realize the projection, the repo still needs a real target/draft checkpoint
and a QAT-distilled 4-layer draft whose acceptance is at least ~0.68 at K=16.
That is a training/export project, not an inference tweak. The inference side is
now ready for it: the fast mode, deferred correction, q8 draft and K=16 ring are
in place.

## Limitations

- Random, untrained weights; no language-model quality claim.
- Fast mode is token-equivalent on tests, not bit-identical; strict mode remains
  the only bit-exact path.
- Single-threaded; no persistent thread pool, GPU, or ANE path.
- No QAT, distillation, training, or checkpoint format.
- No llama.cpp, GGUF, or pretrained hybrid checkpoint integration.
- `max_seq` is hard-capped at 4096; the benchmark uses 512.
- Attention is O(T) per token with a dense KV cache; recurrent layers are O(1)
  in context length but serial.
- The speculative loop falls back to AR when the remaining capacity is too tight
  for a `K+1` transient batch, and otherwise assumes `n_gen + K + 1` positions
  fit; it does not return a partial-generation error for a too-small `max_seq`.
- NoteBank is not a trained retrieval system; vote ranking is opt-in and loses to
  summed similarity on random weights.

## Files

| file | purpose |
|---|---|
| `spel.h` | configuration, model, ternary/q8, NoteBank and spec APIs |
| `ternary.c` | packing, quantization, NEON dotprod/addsub/scalar GEMV, q8 GEMV |
| `model.c` | hybrid forward, KV/state rings, checkpoints, spec loop, governor |
| `notebank.c` | chunked writer, BF16 addresses, addressing, vote/similarity ranking |
| `main.c` | selftests, benchmarks, NoteBank/spec demo |
| `Makefile` | build, `selftest`, `bench`, `demo`, `clean` |

The binary and object files are gitignored. Repository:
<https://github.com/ayush6288/spel>.
