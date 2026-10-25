# SPEL — NoteBank + SubTernary research harness

A self-contained C11 harness for four things on Apple silicon:

1. **2-bit ternary GEMV** with ARM NEON (`{-1, 0, +1}`, per-row scale).
2. **A deterministic hybrid model** — Gated-DeltaNet-style recurrent layers plus
   RoPE attention — with batched verification and checkpoints.
3. **Transactional speculative decoding** with O(1) commit/rollback, a deferred
   correction token, a governor that falls back to autoregressive (AR)
   decoding, and a forced-acceptance system oracle.
4. **A NoteBank-style writer, compact address bank, state branching and replay
   demo**, wired through to the speculative loop.

> **Status: research harness, not a model release.** The repository has a
> binary checkpoint format, a golden numerical verifier, a cheap-draft oracle,
> and a self-contained PyTorch training/QAT script. It does **not** ship the
> trained checkpoint used for the measurements below; that file is
> `weights/spel.bin` locally and is gitignored. Forced-acceptance rows are
> systems measurements, not real decoding quality.

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

## Binary weights and golden verification

`SPELW001` is a 64-byte little-endian header followed by raw FP32 tensors:

```text
offset  field
0       magic "SPELW001" (8 bytes)
8       u32 version = 1
12      u32 d_model
16      u32 n_layers
20      u32 n_heads
24      u32 head_dim
28      u32 d_ff
32      u32 vocab
36      u32 attn_period
40      u32 n_draft_layers = 4
44      u32 draft_map[4] = {0, 3, 6, 9}
60      u32 has_golden_footer
```

Target tensors, in file order:

```text
emb[V][d]
for each target layer 0..N-1:
    n1[d], wqkv[3d][d], wo[d][d],
    [conv[3][d], wa[H], wb[H] only for recurrent layers],
    n2[d], w1[d_ff][d], w3[d_ff][d], w2[d][d_ff]
normf[d]
wout[V][d]
```

The four draft layers follow in the same order. The draft holds independent
distilled FP32 parameters for layers `[0, 3, 6, 9]`; it shares the token
embedding, final RMSNorm and LM head with the target. On load, C quantizes each
draft `wqkv`/`wo` with its existing q8 quantizer and `w1`/`w3`/`w2` with the
existing 3-step ternary least-squares quantizer. The file keeps FP32 latents,
so QAT export is lossless with respect to the C quantizers. `spel_save_weights`
writes an attached distilled draft, or (when no draft is attached and the
target has at least 10 layers) a 4-layer copy using `cfg.draft_map`, whose
default is `[0,3,6,9]`; `spel_save_weights_pair` is the explicit distilled-draft
path.

An optional `SPELV001` footer follows the weights:

```text
magic "SPELV001" (8 bytes)
u32 seq_len = 16
i32 tokens[16]
f32 expected_target_logits[16][V]
f32 expected_draft_logits[16][V]
```

The fixed probe sequence is `token[i] = (i*29 + 7) % vocab`. C save generates
the footer from a reset state, restores the caller's state, and writes it back.

```sh
./spel verify-weights weights/spel.bin
```

This loads the default 384d/12-layer target and the 4-layer draft, resets both,
runs the 16 golden tokens, and prints target and draft max-abs logit error plus
greedy argmax agreement. Pass requires **target argmax 16/16 and target
`max_abs < SPEL_VERIFY_TOL`**, where the tolerance defaults to **1e-3** and can
be overridden only through `SPEL_VERIFY_TOL`. The command also prints the
runtime dimensions, the actual target/draft parameter counts, FP32 byte sizes,
and the checkpoint file size. Draft metrics are reported but not part of the
pass condition because the draft is quantized after loading. A C save/load
round-trip is also exercised by `make selftest` and currently reports
`max_abs=0` for both models.

## Phase 1 verification

The downloaded Colab checkpoint was checked without changing any math:

```sh
$ make selftest && make bench
$ ./spel verify-weights weights/spel.bin
  config: d_model=384 n_layers=12 n_heads=6 head_dim=64 d_ff=1024 vocab=128 max_seq=512 attn_period=4 spec_k=16 draft_map=[0,3,6,9]
  target params: 21352044  FP32: 85408176 bytes (81.45 MiB)
  draft params: 7084452  FP32: 28337808 bytes (27.03 MiB)
  file: 113762508 bytes (108.49 MiB)
  tolerance: 0.001
  target: max_abs=7.39098e-06  argmax=16/16
  draft : max_abs=0.0188234  argmax=16/16
```

The target passed the verifier at **16/16 argmax** with
**target `max_abs=7.39098e-06`**, below the required `1e-3`. The draft is
reported by the same command (`max_abs=0.0188234`, also 16/16 argmax), but only
the target is part of the pass condition. `make selftest` printed
`PASS (0 failures)`.

## Phase 2 verification: persistent draft GEMV pool

The draft now owns a persistent pthread pool. `SPEL_DRAFT_THREADS=1..6` selects
the worker count at draft construction; the default is **1**, which keeps the
original serial loops unchanged. `spel_model_set_threads()` is used by the
selftest and benchmark to switch the same loaded draft between 1 and 4 workers.
Rows are independent dot products, so no numerical reduction crosses threads.
Each worker handles `BLOCK = ceil(rows/(threads*6))` rows at a time with the
existing NEON q8/ternary kernels. The serial and pooled paths produce
bit-identical draft logits.

```sh
$ make selftest
...
[8] draft GEMV pool: threads=1 vs threads=4 bit-exact
  draft logits: 16 x 128 floats, 0 differ (threads 1 vs 4)

PASS (0 failures)

$ ./spel bench
```

Three measured runs of `./spel bench` with the downloaded checkpoint produced:

| path | threads | tok/s | c or alpha |
|---|---:|---:|---|
| AR target | — | 366.5–371.0 | 2.69–2.73 ms/token |
| draft 4L only | 1 | 2973.8–2981.9 | c=0.12x target |
| draft 4L only | 4 | 4394.7–4627.6 | c=0.08x target |
| spec raw K=16 | 1 | 142.0–144.1 | alpha=0.052 |
| spec raw K=16 | 4 | 168.7–171.0 | alpha=0.052 |
| spec governed K=16 | 1 | 341.5–344.0 | alpha=0.094 |
| spec governed K=16 | 4 | 356.2–361.0 | alpha=0.094 |

The measured draft-only speedup from 1 to 4 workers was **1.47–1.55x**. The
target verification cost was **v=2.69–2.72x** one AR token in both settings:
the target path is deliberately unchanged, so the pool changes `c`, not `v`.
Raw speculative throughput rose from `0.38–0.39x` to `0.46x` AR; governed
throughput rose from `0.92–0.94x` to `0.97x`. The forced oracle rows remain
threads=1 system ceilings and are reported separately below.

`bench` and `demo` auto-load weights from `SPEL_WEIGHTS`, or from
`weights/spel.bin` if that file exists in the current directory:

```sh
SPEL_WEIGHTS=weights/spel.bin ./spel bench
./spel bench                       # auto-loads weights/spel.bin if present
```

They print `[weights] loaded <path>` at startup. `demo` switches from its
128d/8-layer toy model to the 384d/12-layer weight configuration when a
checkpoint is present. Bench reloads the full-copy 12-layer oracle from the same
checkpoint, so the oracle comparison is valid rather than a random second model.

## Phase 3: 100M+ runtime configuration

The C runtime takes a `--config key=value,...` string or the matching
`SPEL_D_MODEL`, `SPEL_N_LAYERS`, `SPEL_N_HEADS`, `SPEL_HEAD_DIM`, `SPEL_D_FF`,
`SPEL_VOCAB`, `SPEL_MAX_SEQ`, `SPEL_ATTN_PERIOD`, `SPEL_SPEC_K`, and
`SPEL_DRAFT_MAP` environment variables. Defaults are still 384/12/6/1024, so
the existing selftests and benchmark reproduce the earlier configuration. The
recommended scaled shape is:

```text
d_model=1024  n_layers=16  n_heads=16  head_dim=64  d_ff=3072
vocab=128     attn_period=4  draft_map=[0,3,6,9]
```

The target is **218,436,992 parameters**, or **873,747,968 FP32 bytes
(833.27 MiB)**. The four-layer draft is 54,543,456 parameters
(218,173,824 bytes, 208.07 MiB). These are printed by the verifier, not
estimated.

### Scaled loader/verifier validation

A real trained scaled checkpoint was **not** available in this session. To
validate the scaled file format and runtime path without pretending to have a
trained model, a temporary C program called `spel_save_weights` on a randomly
initialized 1024/16 model. The resulting file is local, not committed, and
**not trained**:

```sh
$ ./spel verify-weights /tmp/spel_scaled_218M_synth.bin \
    --config d_model=1024,n_layers=16,n_heads=16,head_dim=64,d_ff=3072
  config: d_model=1024 n_layers=16 n_heads=16 head_dim=64 d_ff=3072 vocab=128 max_seq=512 attn_period=4 spec_k=16 draft_map=[0,3,6,9]
  target params: 218436992  FP32: 873747968 bytes (833.27 MiB)
  draft params: 54543456  FP32: 218173824 bytes (208.07 MiB)
  file: 1091938316 bytes (1041.35 MiB)
  tolerance: 0.001
  target: max_abs=0  argmax=16/16
  draft : max_abs=0  argmax=16/16
```

`SPEL_VERIFY_TOL=1e-3` was used. The zero error is expected because this file's
golden footer was generated by the same C path; it validates the loader,
dimensions, and footer format, not Python/C numerical agreement at scale.

### Scaled measured throughput (synthetic checkpoint)

```sh
$ SPEL_WEIGHTS=/tmp/spel_scaled_218M_synth.bin ./spel bench \
    --config d_model=1024,n_layers=16,n_heads=16,head_dim=64,d_ff=3072
```

Three runs produced:

| metric | measured range |
|---|---:|
| AR target | 29.5–29.9 tok/s (33.44–33.94 ms/token) |
| draft 4L, threads=1 | 333.3–354.0 tok/s, c=0.08–0.09 |
| draft 4L, threads=4 | 734.0–1000.2 tok/s, c=0.03–0.04 |
| verify K+1=17 fast, B2 M=17 | 49.97–55.16 ms, v=1.52–1.62 |
| verify K+1=17 fast, B3 | v=1.54–1.59 |
| spec raw K=16, threads=1 | 9.9–10.2 tok/s (0.34x AR), alpha=0.000 |
| spec raw K=16, threads=4 | 14.1–15.0 tok/s (0.48–0.50x AR), alpha=0.000 |
| spec governed K=16, threads=1 | 28.2–28.9 tok/s (0.96x AR), alpha=0.000 |
| spec governed K=16, threads=4 | 28.8–29.3 tok/s (0.97–0.98x AR), alpha=0.000 |
| forced oracle-4L alpha=0.75, threads=1 | 130.6–134.4 tok/s (4.39–4.50x AR) |
| forced oracle-4L alpha=1.00, threads=1 | 163.5–167.6 tok/s (5.50–5.61x AR) |
| forced oracle-16L full copy, threads=1 | 26.0–26.4 tok/s (0.88x AR) |

The synthetic random target has raw alpha=0.000, so the raw/governed rows say
nothing about learned draft acceptance. They only show the measured system
behavior when every proposal is rejected. The forced oracle rows are
benchmark-only systems ceilings and are not real decoding. The target is
DRAM-resident at this shape: measured AR fell from the 384d numbers above to
29.5–29.9 tok/s. These numbers are for this machine and this synthetic file
only; they are not generalized and are not a trained scaled-model result.

To train and measure the real scaled checkpoint, the opt-in Colab command is:

```sh
python train_colab.py --d-model 1024 --n-layers 16 --n-heads 16 \
  --head-dim 64 --d-ff 3072 --out weights/spel_scaled.bin
./spel verify-weights weights/spel_scaled.bin \
  --config d_model=1024,n_layers=16,n_heads=16,head_dim=64,d_ff=3072
SPEL_WEIGHTS=weights/spel_scaled.bin ./spel bench \
  --config d_model=1024,n_layers=16,n_heads=16,head_dim=64,d_ff=3072
```

Nothing scaled is committed: `weights/` is gitignored. The trained scaled
verification/quality claims remain pending until that file exists.

## Training and QAT distillation (`train_colab.py`)

`train_colab.py` is a 1:1 PyTorch transliteration of `model.c`, not a generic
transformer. It mirrors:

- RMSNorm `1/sqrt(mean(x^2) + 1e-5)` and the learned gain.
- The 3-tap causal conv used by recurrent layers:
  `y[t] = x[t] + conv[0]*x[t-2] + conv[1]*x[t-1]`.
- QKV layout `[q(d), k(d), v(d)]`, each split into `[heads][head_dim]`.
- Interleaved even/odd RoPE pairs, `theta=10000`, `angle=pos*theta^(-2j/dh)`.
- Per-head q/k L2 normalization with epsilon `1e-6`.
- The delta-rule recurrence
  `S_t = a*S_{t-1} + beta*(v - S_{t-1}k)k^T` and readout
  `S_t q / sqrt(head_dim)`.
- SwiGLU `down(silu(gate(x)) * up(x))`.
- C's exact q8 row quantizer, 3-step ternary scale refinement, and dynamic
  int8 activation quantizer, with straight-through estimators for QAT.
- C round-half-away-from-zero, not PyTorch's ties-to-even `round`.

The script has three stages:

1. **Synthetic + structured ASCII data (`vocab=128`).** A fixed pool mixes
   structured dialog/prose with associative NoteBank examples. Each recall
   example plants `KEY_xx=VAL_yyyy;` in three 128-token chunks and ends with
   `?KEY_xx=VAL_yyyy.`. The first value character after `=` cannot be copied
   from the query prefix, so next-token training must retrieve it from context.
2. **Target training.** Train the 12-layer FP32 target with next-token cross
   entropy, AdamW and a cosine learning-rate schedule.
3. **QAT distillation.** Initialize a 4-layer student from teacher layers
   `[0, 3, 6, 9]`, tie the embedding/final norm/LM head, and train with
   cross-entropy plus temperature-scaled KL to the frozen teacher while
   simulating q8 weights, ternary MLP weights and dynamic int8 activations.
4. **Export.** Write `SPELW001` with the four distilled draft layers and an
   `SPELV001` golden footer computed by the PyTorch models. A sibling
   `.json` file records config and validation metrics.

Typical Colab invocation:

```python
!git clone https://github.com/ayush6288/spel.git
%cd spel
!python train_colab.py --steps-target 1200 --steps-qat 1200 \
    --out weights/spel.bin --device auto
!make
!./spel verify-weights weights/spel.bin
```

Useful flags:

```text
--steps-target N     target training steps (default 1200)
--steps-qat N        QAT distillation steps (default 1200)
--out PATH           output checkpoint (default weights/spel.bin)
--device auto        auto -> cuda, mps, or cpu
--d-model N          target width (default 384; C limit 1024)
--n-layers N         target depth (default 12; C limit 16)
--n-heads N          heads (default 6; C limit 16)
--head-dim N         optional check: must equal d_model // n_heads
--d-ff N             SwiGLU width (default 1024; C limit 4096)
--vocab N            token vocabulary (default 128; C limit 1024)
--attn-period N      every Nth layer is attention (default 4)
--max-seq N          context capacity (default 512; C limit 4096)
--draft-map a,b,c,d  four target layers used by the draft (default 0,3,6,9)
--batch-size N       default 8
--seq-len N          default 384
--pool-size N        default 2048 fixed synthetic sequences
--lr-target / --lr-qat
--temperature        KL temperature (default 2.0)
--kl-weight          KL loss weight (default 0.5)
--save-torch PATH    optional PyTorch state_dict export
--no-export          train/evaluate only
```

The config flags are opt-in; omitting them keeps the original 384/12/1024
training command. `export_weights` prints the actual target parameter count and
FP32 byte size before writing the file.

A short CPU smoke run was used during development to validate export/load/QAT
plumbing and the transliteration. It is **not** evidence of useful language
quality or real speculative acceptance. The downloaded checkpoint used for the
Phase 1/2 measurements above is the actual trained artifact; its training log
and hyperparameters are not part of this repository.

## Cheap-draft system oracle

`./spel bench` keeps the old **full 12-layer FP32 copy** as a reference and adds
the actual cheap draft as a system oracle:

```text
oracle-4L (a=0.75)   real 4-layer q8/ternary draft forward for pending+K
                     proposals, real M=K+1 fast target verification,
                     forced 75% acceptance, real position/ring rollback
oracle-4L (a=1.00)   same, accepting all K proposals
oracle-12L           full target-sized FP32 copy (12L default; 16L scaled)
```

The 4-layer oracle is implemented with `spel_spec_cfg.force_alpha`. The forced
acceptance is applied **after** the real draft and target forwards, and the
fractional carry is preserved across rounds so the requested average is hit.
The target/ring state is really rolled back to `pos + m`. It measures the
runtime ceiling of this draft/verifier pair; it is not a decoding benchmark and
does not claim the draft would accept that many tokens in real generation.
Forced rows are the median of three runs.

Measured on the same M5 used for the other tables, K=16, with the downloaded
checkpoint. The oracle rows use the threads=1 draft, unchanged from Phase 1:

| path | tok/s | vs AR | real alpha |
|---|---:|---:|---:|
| AR target | 366.5–371.0 | 1.00x | — |
| draft 4L only | 2973.8–2981.9 | — | — |
| spec raw K=16 | 142.0–144.1 | 0.38–0.39x | 0.052 |
| spec governed K=16 | 341.5–344.0 | 0.92–0.94x | 0.094 |
| **oracle-4L a=.75** | **1016.6–1019.9** | **2.75–2.78x** | forced 0.750 |
| **oracle-4L a=1.00** | **1274.5–1290.8** | **3.45–3.52x** | forced 1.000 |
| oracle-12L full copy | 302.4–311.0 | 0.82–0.84x | 1.000 |

The full-copy oracle is still below AR because it pays nearly one target-sized
forward per proposal; fast verification alone cannot fix that. The 4-layer
oracle shows the actual systems ceiling of the current cheap draft: about
**2.75x** at 75% acceptance and **3.45x** at 100%, with the measured serial
draft cost `c ≈ 0.12x` and verification `v ≈ 2.72x` one AR token.

## Strict mode and fast mode

Two arithmetic paths exist:

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
AR, draft-only and strict tables stay strict. There is no claim that fast mode
preserves bit-exactness for every input.

## Architecture / benchmark configuration

```text
target: d_model=384, layers=12, heads=6, head_dim=64
        d_ff=1024, vocab=128, max_seq=512 (4096 hard cap)
        attn_period=4 -> 9 recurrent + 3 attention layers
        ~21.35M FP32 parameters, 85.4 MB
draft:  4 layers [0,3,6,9], independent FP32 layer parameters
        q8 wqkv/wo + ternary MLP after load
        shared embedding / final norm / LM head / RoPE
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
7. **Forced acceptance** — `force_alpha=0.75` gives 30/40 accepted proposals;
   `force_alpha=1.00` gives 32/32. Both emit 32 tokens token-identically to AR
   and roll back to the same final position. This is a benchmark path only.
8. **Weights round-trip** — save target+draft as `SPELW001`, load into fresh
   models, compare every target/draft tensor, then run the `SPELV001` verifier.
   `max_abs=0`, 16/16 argmax.
9. **Draft GEMV pool** — run the same 4-layer draft over the 16 golden tokens
   with `threads=1` and `threads=4`; all 16 x 128 logits are bit-identical
   (`0 differ`). This covers q8 `wqkv`/`wo` and ternary `w1`/`w3`/`w2`.

These are same-binary, same-flags guarantees (`-O3 -ffast-math`). Bit-exactness
across compilers, optimization levels or architectures is not claimed.

## Measured results

Apple M5, 10 cores, 16 GB, macOS. Single thread. `clang -O3 -mcpu=apple-m5
-ffast-math`. Ranges are across repeated runs; decode rows are representative
single runs unless marked as medians.

### Ternary GEMV vs FP32 (4096 x 1536)

Values from the three Phase 2 `./spel bench` runs:

| kernel | GMAC/s | packed GB/s | ms/GEMV |
|---|---:|---:|---:|
| neon-dotprod | 17.72–17.78 | 4.4 | 0.35–0.36 |
| neon-addsub | 10.67–10.68 | 2.7 | 0.59 |
| scalar | 2.51–2.52 | 0.6 | 2.50 |
| fp32-neon | 6.20–6.61 | 24.8–26.5 | — |

Packed weights are 1.50 MiB versus 24.00 MiB FP32. The dot-product kernel is
roughly 3x the FP32 per-MAC rate on this shape.

### Verification cost: strict vs fast

`v` is batch time divided by one AR token (2.6–2.8 ms). M is the number of tokens
in the target batch.

| M | strict v | fast v |
|---:|---:|---:|
| 4 | 3.93–3.96 | 1.94–1.97 |
| 5 | 4.82–4.88 | 1.98–2.02 |
| 8 | 7.48–7.58 | 2.16–2.19 |
| 9 | 8.39–8.50 | 2.21–2.26 |
| 16 | 14.61–14.84 | 2.64–2.69 |
| 17 | 15.54–15.74 | 2.69–2.74 |

The strict path scales almost linearly with M. Accelerate amortizes the weight
read and gives roughly a 4–6x reduction at M=17, but it is still **2.7x**, not
1.1x. AMX is more efficient at M=16/17 than at M=4/5, which is why K=16 is the
better speculative configuration on this machine.

### Decode: 64 generated tokens, 4-layer q8/ternary draft

| path | threads | K | tok/s | vs AR | notes |
|---|---:|---:|---:|---:|---|
| AR target | — | — | 366.5–371.0 | 1.00x | 2.69–2.73 ms/token |
| draft 4L only | 1 | — | 2973.8–2981.9 | — | c = 0.12x target |
| draft 4L only | 4 | — | 4394.7–4627.6 | — | c = 0.08x target |
| spec raw | 1 | 16 | 142.0–144.1 | 0.38–0.39x | alpha 0.052, 1.83 tok/round |
| spec raw | 4 | 16 | 168.7–171.0 | 0.46x | alpha 0.052, 1.83 tok/round |
| spec governed | 1 | 16 | 341.5–344.0 | 0.92–0.94x | alpha 0.094, 1 trip, 59 AR rounds |
| spec governed | 4 | 16 | 356.2–361.0 | 0.97x | alpha 0.094, 1 trip, 59 AR rounds |
| oracle-4L a=.75 | 1 | 16 | 1016.6–1019.9 | 2.75–2.78x | forced 0.750, 12.8 tok/round |
| oracle-4L a=1.00 | 1 | 16 | 1274.5–1290.8 | 3.45–3.52x | forced 1.000, 16.0 tok/round |
| oracle-12L full copy | 1 | 16 | 302.4–311.0 | 0.82–0.84x | same weights, alpha 1.000 |

With the downloaded checkpoint, raw K=16 accepts 29 of 560 proposals
(alpha=0.052) and runs at 0.38–0.39x AR. The governor trips after the first
round and falls back; it reaches 0.92–0.94x AR, so it is a safety mechanism, not
a speedup. The forced oracle rows remain benchmark-only system ceilings: the
draft and target forwards execute, but the accepted prefix is forced to
`alpha*K`. They are not real decoding quality.

The 12-layer full-copy oracle is retained as a reference: even at 100%
acceptance a target-sized draft is slower than AR because it pays roughly one
full target forward per proposal.

### Cost model

In steady state a round emits `1 + alpha*K` tokens on average. The draft pays
`K+1` token forwards (the pending correction plus K proposals) and the target
pays one `K+1` verification batch:

```text
speedup(alpha) = (1 + alpha*K) / ((K+1)*c + v_{K+1})
```

With the measured K=16 target values (`v=2.72`) and the Phase 2 draft costs:

```text
threads=1, c=0.12: alpha=.75 -> 13.0 / (17*0.12 + 2.72) = 2.69x
threads=1, c=0.12: alpha=1.00 -> 17.0 / (17*0.12 + 2.72) = 3.51x
threads=4, c=0.08: alpha=.75 -> 13.0 / (17*0.08 + 2.72) = 3.19x
threads=4, c=0.08: alpha=1.00 -> 17.0 / (17*0.08 + 2.72) = 4.17x
```

The measured forced oracle at the serial draft setting is close
(2.75–2.78x and 3.45–3.52x); the gap is draft/verify scheduling and measurement
overhead. These projections are formulas, not measured decoding results. Real
raw/governed throughput is alpha=0.052/0.094 as shown above.

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
| note storage (toy config) | 6,720 B |
| dense KV, all 8 layers | 3,145,728 B (468x) |
| hybrid attention KV, 2 layers | 786,432 B (117x) |

The 468x/117x ratios are for the toy 128d/8-layer demo and exclude model state
and capture buffers. Notes are not a replacement for attention KV; replay is a
separate path. When a 384d/12-layer checkpoint is present, `demo` uses that
configuration and prints larger storage numbers.

The demo prints a NoteBank-to-spec run (`notes 0,3 -> ... alpha=0.000 trips=1
AR=7`). That proves the branch/replay/spec plumbing, not retrieval quality. The
NoteBank path is still a reference implementation, not a trained memory system.

## What is still missing

1. **A committed trained checkpoint.** The repository does not commit the
   checkpoint used below. `weights/spel.bin` is local and gitignored; the
   training run log and exact hyperparameters are not included.
2. **A real speculative speedup.** On the downloaded checkpoint, measured
   acceptance is alpha=0.052 raw and 0.094 after the governor; raw reaches
   0.38–0.46x AR and governed 0.92–0.97x AR. The 2.75–3.52x oracle rows are
   forced-acceptance system ceilings, not real decoding.
3. **Scaled trained verification evidence.** Phase 3 adds the runtime
   configuration and validates a local random 1024/16 file through the scaled
   loader/verifier. A scaled **trained** checkpoint must still be exported and
   verified before any scaled quality or trained-throughput claim can be made.
4. **A tokenizer / pretrained model.** This remains an ASCII-token research
   harness, not a llama.cpp/GGUF integration.

## Limitations

- No committed trained weights; no language-model quality claim.
- Fast mode is token-equivalent on tests, not bit-identical; strict mode remains
  the only bit-exact path.
- Single-threaded C inference; no persistent thread pool, GPU, or ANE path.
- The training script uses a full-sequence PyTorch forward, not the C batched
  verifier. C inference is bit-identical to sequential AR; PyTorch reference
  logits match C to ~1e-5 for the target in smoke tests, not bit-for-bit.
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
| `spel.h` | configuration, model, ternary/q8, NoteBank, spec and `SPELW001` APIs |
| `ternary.c` | packing, quantization, NEON dotprod/addsub/scalar GEMV, q8 GEMV |
| `model.c` | hybrid forward, KV/state rings, checkpoints, spec loop, governor, weights I/O |
| `notebank.c` | chunked writer, BF16 addresses, addressing, vote/similarity ranking |
| `main.c` | selftests, benchmarks, NoteBank/spec demo, `verify-weights` |
| `train_colab.py` | PyTorch transliteration, synthetic data, target training, QAT distillation, export |
| `Makefile` | build, `selftest`, `bench`, `demo`, `clean` |

The binary and object files are gitignored. Repository:
<https://github.com/ayush6288/spel>.
