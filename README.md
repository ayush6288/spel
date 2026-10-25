# SPEL — NoteBank + SubTernary research harness

A self-contained C11 harness for four things on Apple silicon:

1. **2-bit ternary GEMV** with ARM NEON (`{-1, 0, +1}`, per-row scale).
2. **A deterministic hybrid model** — Gated-DeltaNet-style recurrent layers plus
   RoPE attention — with batched verification and checkpoints.
3. **Transactional speculative decoding** with O(1) commit/rollback and a
   governor that falls back to autoregressive (AR) decoding.
4. **A NoteBank-style writer, compact address bank and replay demo**.

> **Status: research harness, not a model release.** The weights are randomly
> initialized and untrained, so generated text is meaningless. There is no
> pretrained checkpoint, no training, no QAT, and no llama.cpp/GGUF integration.
> The value here is the measured kernel/runtime behaviour and the exactness
> machinery, not language quality or a paper reproduction.

## Build and run

Requirements: macOS on ARM64 with NEON dot-product support, and clang.
The Makefile targets the M5 with `-mcpu=apple-m5`.

```sh
git clone https://github.com/ayush6288/spel.git
cd spel
make                 # builds ./spel
make selftest        # correctness tests; must print PASS (0 failures)
make bench           # kernels + decode + rollback
make demo            # NoteBank demo + a 48-token spec run
```

If you build on a different ARM64 Mac, override the target:

```sh
make clean && make ARCH="-mcpu=apple-m1"
# or
make clean && make ARCH="-march=native"
```

## What is implemented

### Ternary GEMV (`ternary.c`)

- 4 weights per byte, codes `0 -> 0`, `1 -> +1`, `3 -> -1`; code `2` decodes
  to 0.
- Per-row least-squares scale after a three-step sign/scale refinement.
- Activations are dynamically quantized to int8 with one scale per GEMV.
- Three interchangeable kernels:
  - `neon-dotprod`: `vdotq_s32` after a table-lookup unpack.
  - `neon-addsub`: mask/select/`vaddw_s8`, no multiply.
  - `scalar`: portable reference.
- The test asserts all three return **identical int32** dot products on random
  shapes; only the final float dequantization differs by rounding.

### Hybrid model (`model.c`)

Configuration used by the benchmarks:

```text
d_model = 384, layers = 12, heads = 6, head_dim = 64
d_ff = 1024, vocab = 128, max_seq = 512 (4096 hard cap)
attn_period = 4  -> 9 recurrent layers, 3 attention layers
K = 4
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
  `K+1` slots. L2 normalization is required; without it the state diverges.
- **Attention mixer:** causal RoPE attention with a conventional KV cache.
- **Weights:** about 21.35 M FP32 parameters (85.4 MB). The model is
  intentionally small enough to benchmark on a laptop.
- **Draft model:** same architecture and shared non-MLP weights; only the three
  MLP matrices per layer are replaced by ternary copies. This is the
  "MLP-only ternary" draft. Its incremental MLP storage is about 3.6 MB, but it
  still reads the shared 28.8 MB of FP32 non-MLP weights per token. This is why
  it is cheaper than the target but not tiny.
- **Batched verification:** `spel_forward(..., M=4, ...)` computes four tokens
  in one pass. The test proves the resulting logits are bit-identical to four
  one-token passes from the same checkpoint.

### Transactional speculative decoding (`model.c`)

- The draft proposes `K` greedy tokens from its own state.
- The target verifies all `K` in one batched forward.
- The longest matching prefix is accepted.
- **Rollback is a position reset**, not a cache copy:
  - attention KV entries beyond the new position are simply masked by `pos`;
  - recurrent state lives in `K+1` ring slots; the slot needed after rewind has
    not been overwritten because the verification touched at most `K` positions;
  - the correction token then overwrites the one stale KV entry and ring slot.
- `spel_ckpt` exists for tests and host-side save/restore; it copies only the
  recurrent state and `last_logits`, because attention history is masked by
  `pos`.
- **Governor:** when the rolling accepted/proposed ratio over the last two
  rounds is below 0.30, drafting is suspended and the rest of the call runs AR.
  It does not resume mid-call, because resuming would require replaying the
  accepted suffix into the draft.

`alpha` in the output is **accepted tokens / proposed tokens**, not first-token
acceptance. `tokens/round` is the more directly useful number.

### NoteBank (`notebank.c`)

- Chunked writer (chunk 128 in the demo, paper default is 512) that scores each
  position by key distinctiveness: mean over channels of
  `1 - cos(k_t, chunk_mean_k)`.
- Per chunk it selects the best non-overlapping window and the top `per_chunk`
  spans (2), with a span budget of 24 tokens in the demo.
- For each channel it stores a 16-coordinate BF16 address from the most
  distinctive position, using a fixed random sparse projection of the native
  q vector.
- Addressing scores every note by summed per-channel cosine similarity and
  returns the top R. The paper-style integer "channel vote" is computed but kept
  as a diagnostic; on random weights the summed similarity was needed to get
  3/3 recall@2.
- **Replay in this harness means re-prefilling the note tokens, not native state
  injection.** It verifies the plumbing; it is not the paper's native replay.

## Correctness invariants (what `make selftest` proves)

1. **Kernel equivalence** — for 50 random matrices, dotprod/addsub/scalar return
   identical int32 results; GEMV matches a dequantized reference.
2. **Batch equivalence** — 256 logits from a 4-token batched forward are
   bit-identical (`memcmp` on float bits) to four sequential forwards from the
   same checkpoint: `0 differ`.
3. **Speculative fidelity** — greedy output matches AR token-for-token
   (`40/40` plain, `32/32` governed), the final position matches, and a
   post-generation probe token produces bit-identical logits. That last check
   proves rollback left no stale state.
4. **Governor** — with a forced threshold of 1.0 it trips, then emits AR rounds
   (`alpha=0.000`, 31 AR rounds, 1 trip).
5. **NoteBank** — all three planted needles fall inside retained notes; address
   recall@2 is 3/3; storage is accounted for.

These are same-binary, same-flags guarantees (`-O3 -ffast-math`). Bit-exactness
across compilers, optimization levels or architectures is not claimed.

## Measured results

Apple M5, 10 cores, 16 GB, macOS. Single thread. `clang -O3 -mcpu=apple-m5
-ffast-math`. Three sequential runs; ranges below are across those runs.

### Ternary GEMV vs FP32 (4096 x 1536)

| kernel | GMAC/s | packed GB/s | ms/GEMV |
|---|---:|---:|---:|
| neon-dotprod | 17.67–17.75 | 4.4 | 0.35–0.36 |
| neon-addsub | 10.63–10.65 | 2.7 | 0.59 |
| scalar | 2.51 | 0.6 | 2.50–2.51 |
| fp32-neon | 5.4–6.6 | 21–26 | — |

Packed weights are 1.50 MiB versus 24.00 MiB FP32 (4x smaller). The dot-product
kernel is roughly 3x the FP32 per-MAC rate on this shape.

### Decode (12 layers, 9 recurrent + 3 attention, 64 generated tokens)

| path | tok/s | vs AR | notes |
|---|---:|---:|---|
| AR target | 360–362 | 1.00x | 2.77 ms/token |
| draft only | 609–617 | — | 0.58–0.59x target cost/token |
| spec K=4, raw | 58.8–58.9 | 0.16x | alpha 0.058, 1.23 tok/round |
| spec K=4, oracle draft | 183–185 | 0.51x | alpha 1.000, 5.00 tok/round |
| spec K=4, governed | 306–309 | 0.85–0.86x | 1 trip after 2 rounds, then AR |

The raw run was 52 rounds, 208 proposed, 12 accepted, 64 emitted. The oracle run
uses a full FP32 copy of the target as the draft, so it isolates runtime cost
from draft quality: even with 100% acceptance this implementation does not beat
AR.

### Rollback

| operation | measured |
|---|---:|
| position reset (commit/rollback primitive) | ~0.5 ns |
| memcpy of 4.50 MiB KV | 108–112 us |
| ratio | ~2.2–2.5 x 10^5 |

The important number is not the memcpy ratio; it is that the recurrent state is
addressed by ring slot, so rollback never walks or copies the 4.5 MiB cache.

### NoteBank demo (384 tokens, 128-token chunks, 8 layers, 2 attention layers)

| quantity | value |
|---|---:|
| chunks / notes | 3 / 6 |
| note spans | [28,52) [86,110) [158,182) [215,239) [294,318) [319,343) |
| writer time | 0.34–0.38 ms |
| address time (Q=4, R=2) | 0.019–0.022 ms |
| writer needle containment | 3/3 |
| address recall@2 | 3/3 |
| note storage | 6,720 B |
| dense KV, all 8 layers | 3,145,728 B (468x) |
| hybrid attention KV, 2 layers | 786,432 B (117x) |

**Do not read the 468x/117x as end-to-end memory savings.** The accounting
excludes the normal model state and the capture buffers used while writing
(about 2.25 MiB for this demo). The notes are not wired into the decoder as a
replacement for attention KV; replay is a separate demo path.

## Why the 4x / high-acceptance target is not reproduced

This is the central measured result, and it is a hardware/runtime result rather
than a tuning failure:

1. **The draft is not cheap enough.** It shares all non-MLP weights and still
   reads ~32 MB per token. Draft cost is 0.58x the target, so four proposals
   already cost 2.3 target-token units.
2. **Verification does not amortize on this CPU.** The target is compute-bound
   (about 7–8 GMAC/s effective), not bandwidth-bound. Batching four tokens reuses
   each weight row, but still performs four times the MACs, and the recurrent
   update is serial along the sequence. From the oracle run, verification of
   K=4 costs about 4.9 target-token units, essentially the same as four separate
   forwards. The 4.50 MiB KV copy figure is irrelevant here; decode is not
   memcpy-bound.
3. **The untrained draft has low acceptance.** Raw alpha is 0.058
   (12 accepted / 208 proposed). The measured per-round budget is:

   ```text
   emitted per round = 1 + m
   cost per round    = K*c + v + 1
   raw:      (1 + 0.23) / (4*0.58 + 4.9 + 1) = 1.23 / 8.22 = 0.15x  (measured 0.16x)
   oracle:   5          / (4*1.00 + 4.9 + 1) = 5    / 9.90 = 0.51x  (measured 0.51x)
   ```

   With K=4 the numerator is at most 5. A 4x speedup would require the
   denominator to be at most 1.25, i.e. near-zero draft cost **and** verification
   at a quarter of one token's cost. Neither is achievable with this serial
   scalar implementation.
4. **The governor is a safety mechanism, not a speedup.** It turns the 0.16x
   disaster into roughly AR speed (0.85x over 64 tokens, approaching 1.0x for
   longer runs) once it detects that acceptance is below threshold.

To approach the brief's numbers, the draft needs training/QAT and a much smaller
footprint, and verification needs to be parallel across heads/channels (or moved
to a GPU/ANE) so `v` stops scaling with K. See "Next steps".

## NoteBank caveats

- The value-rank experiment printed `no-replay == replay` for all three needles
  because the random model has no learned retrieval circuit. This is a **null
  result**, not evidence for or against NoteBank.
- The writer and addresser are simplified interpretations of the brief, not an
  exact reproduction. Addresses are random sparse projections, not trained.
- Notes currently store token IDs plus addresses; replay re-prefills the token
  spans. There is no state-injection kernel and no measured decode-time retrieval
  benefit.
- The storage comparison is against KV for the same 384-token context. It is a
  representation-size measurement, not a full decoder memory budget.

## Limitations

- Random, untrained weights; no language-model quality claim.
- Single-threaded; no persistent thread pool, GPU, or ANE path.
- No QAT, distillation, or training of any kind.
- No llama.cpp, GGUF, or pretrained hybrid checkpoint integration.
- `max_seq` is hard-capped at 4096; the benchmark uses 512. No 1M-context claim.
- Attention is O(T) per token with a dense KV cache; the recurrent layers are
  O(1) per token in context length but serial.
- Bit-exactness is within one build/flags; `-ffast-math` is used.
- The speculative loop assumes capacity for `n_gen + K` more positions; it does
  not return an error when `max_seq` would be exceeded.
- The NoteBank is not part of the speculative generation loop.

## Next steps

1. **Train or distill a draft** on target traces and apply QAT to the ternary
   MLP. Target per-position acceptance high enough that `K*c + v + 1` is
   competitive; with `c ~ 0.1` and `v ~ 1`, K=4 can plausibly approach 2–3x.
2. **Shrink the draft** beyond MLP-only ternary: fewer layers, smaller width, and
   ternary/int8 for projections, so `c` drops well below 0.2.
3. **Parallelize verification.** The recurrent update is currently serial per
   token. Split heads/channels across threads, use a parallel scan for the
   recurrence where possible, and batch GEMMs. The goal is `v` near 1 regardless
   of K.
4. **Port the runtime to a real hybrid checkpoint.** The rollback invariant maps
   directly to attention KV plus GDN recurrent state; a llama.cpp-style backend
   would make the measurements meaningful for a real model.
5. **Make NoteBank a decode path.** Implement state injection for native replay,
   train the writer/addresser, and measure retrieval quality and latency against
   a real long-context task.
6. Only then attempt long-context (1M) claims, with an explicit streaming/memory
   design.

## Files

| file | purpose |
|---|---|
| `spel.h` | configuration, model, ternary, NoteBank and spec APIs |
| `ternary.c` | packing, quantization, NEON dotprod/addsub/scalar GEMV |
| `model.c` | hybrid forward, KV/state rings, checkpoints, spec loop, governor |
| `notebank.c` | chunked writer, BF16 addresses, addressing, replay selection |
| `main.c` | selftests, benchmarks, demo |
| `Makefile` | build, `selftest`, `bench`, `demo`, `clean` |

The binary and object files are gitignored. Repository:
<https://github.com/ayush6288/spel>.
