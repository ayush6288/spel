# SPEL: Low-Bit Speculative Execution & Episodic Memory on Apple Silicon

A bare-metal C11 research harness implementing and evaluating **Unified Speculative Decoding (SubTernary)** and **Sparse Episodic Memory (NoteBank)** on Apple Silicon (ARMv9-A / Apple M5).

SPEL investigates two core bottlenecks in edge inference:

1. **DRAM-resident autoregressive decode**, where each token streams the target weights and memory bandwidth is a candidate limiter. SPEL addresses this with 2-bit ternary draft proposals, a persistent draft-only thread pool, and O(1) transactional rollback.
2. **Context-memory growth in recurrent/attention hybrids**, addressed with chunked key-distinctiveness indexing, compact BF16 addresses, branch-isolated state snapshots, and replay scheduling.

> **Artifact status.** This is a research harness, not a model release. The 21.35M-parameter checkpoint is trained but is not committed (`weights/` is gitignored). The 218.44M-parameter checkpoint is a **C-generated random model** used only to validate the scaled runtime and measure DRAM-resident systems behavior; it has no language quality or learned acceptance. Forced-acceptance rows are benchmark-only system ceilings, not decoding results. Raw measurement captures are in [`results/`](results/).

**Measured code revisions:** `36a158e` (Phase 2: persistent draft pool, 21M steady-state decode, and NoteBank demo) and `cceae5b` (Phase 3: runtime configuration, 21M re-verification, scaled synthetic generation/verification, and final self-test). All measured numbers below are transcribed from the captures in `results/`; derived quantities are identified explicitly.

---

## 1. System Architecture & Kernel Innovations

SPEL is organized around four inference primitives plus the NoteBank memory subsystem.

### 1.1 ARM NEON 2-bit ternary GEMV (`ternary.c`)

- Symmetric 4-weight/byte packing with low bits first: code `0 -> 0`, `1 -> +1`, `3 -> -1`; code `2` decodes as zero.
- Per-row symmetric quantization with three least-squares scale-refinement iterations, followed by dynamic int8 activation quantization at GEMV time.
- Three interchangeable row kernels: `neon-dotprod` using `vdotq_s32`, `neon-addsub` using vector mask/add, and a portable scalar fallback. The two NEON kernels return identical integer dot products in the self-test; kernel selection is exposed through `tern_kernel_mode`.

Measured on the 4096 x 1536 kernel benchmark, single thread:

| kernel | GMAC/s | weight GB/s | ms/GEMV |
|---|---:|---:|---:|
| `neon-dotprod` | 17.72–17.87 | 4.4–4.5 | 0.35–0.36 |
| `neon-addsub` | 10.66–10.69 | 2.7 | 0.59 |
| scalar | 2.51–2.52 | 0.6 | 2.50 |
| FP32 NEON | 5.84–6.67 | 23.4–26.7 | — |

Packed weights are **1.50 MiB** versus **24.00 MiB** FP32. The dot-product kernel is **2.66–3.06x** the measured FP32 per-MAC rate on this shape.

### 1.2 Deterministic hybrid architecture (`model.c`)

The default model is a 12-layer, 384-wide hybrid with 6 heads, 64-wide head dimensions, a 1024-wide SwiGLU, and a 4-layer attention period: **9 Gated-DeltaNet recurrent layers + 3 causal RoPE attention layers**. A 16-layer, 1024-wide, 16-head scaled shape is also supported (12 recurrent + 4 attention layers).

Each layer is `RMSNorm -> mixer -> residual -> RMSNorm -> SwiGLU -> residual`. The recurrent mixer uses a 3-tap causal convolution, L2-normalized keys/queries, per-head decay and write rate, and the delta-rule update:

```text
S_t = a * S_{t-1} + beta * (v - S_{t-1} k) k^T
o   = S_t q / sqrt(head_dim)
```

Attention uses interleaved even/odd RoPE pairs, `theta=10000`, per-head q/k L2 normalization, and a conventional KV cache.

**Verification path.** `spel_forward(..., M, ...)` compiles the target batch into one pass. In strict mode, batched projections preserve the same NEON accumulation order as single-token AR and are bit-identical to sequential decoding. In fast mode, batched projections use Accelerate `cblas_sgemm`/AMX; this is token-equivalent on the tested inputs but not bit-identical. The speculative round defers the correction token: the target verifies `[pending, proposal_0, ..., proposal_{K-1}]` in one `K+1` batch, and the next round consumes the correction as its pending token, eliminating a separate `+1` target pass.

### 1.3 Transactional KV and recurrent-state isolation

- Attention rollback is a position reset; KV slots beyond `pos` are masked and reused. Accepted KV stays in place.
- Recurrent state lives in a `(K+1)`-slot ring per layer. A rollback only moves the read position; stale slots are overwritten by the next round.
- The measured commit/rollback primitive is **~0.5 ns**, at the benchmark timer floor. For comparison, copying the 21M model's 4.50 MiB attention KV takes **108.2–112.7 us** in the steady-state captures; the ratio is **2.40–2.50 x 10^5**. At 218M, copying the 16.00 MiB KV takes **446.1–561.8 us**.
- The **Dynamic Speculation Governor** tracks rolling accepted/proposed ratio over a window. It suspends drafting and runs the remainder of the call autoregressively when acceptance falls below `alpha_threshold`, or when a round accepts zero proposals. It does not resume drafting mid-call.

### 1.4 Persistent draft GEMV pool (`spel_gemv_pool`)

- A draft owns a persistent pthread pool created once per draft and reused across all draft GEMVs.
- Worker count is `SPEL_DRAFT_THREADS=1..6` (default **1**), or `spel_model_set_threads()` for tests. The target path is never parallelized.
- Work is row-parallel. Each worker takes `BLOCK = ceil(rows / (threads*6))` rows of the existing q8 or ternary kernel. No floating-point reduction crosses threads, so the pooled and serial draft logits are bit-identical.
- The pool uses hybrid atomic spin plus condition-variable dispatch to avoid per-GEMV thread creation and to keep the target from competing with idle draft workers.
- It lowers the draft cost `c` from **0.12x** to **0.08x** at 21M, and from **0.08–0.09x** to **0.03–0.04x** at 218M. It does not change target verification `v`.

Self-test `[8]` compares all `16 x 128` draft logits at 1 and 4 workers: **0 differ**.

### 1.5 NoteBank episodic memory (`notebank.c`)

- Chunked writer over captured native Q/K for every recurrent layer-head channel.
- Key distinctiveness is `1 - <k_hat, mu_hat>`, where `mu_hat` is the normalized chunk-mean key. The writer selects high-distinctiveness spans and stores **16 BF16 coordinates** per channel from a fixed sparse projection.
- Addressing scores each note per channel, then ranks by summed channel similarity. Per-channel integer voting is implemented as an opt-in diagnostic (`NB_RANK_VOTES`), which is not the default.
- State branching is a snapshot/restore of `S_T`: the demo captures `S_T`, runs the retrieval branch with activation capture, restores `S_T`, then runs the read/replay branch. `S_retrieval` and `S_read` are branch labels over the same checkpointed state, not separate model copies.
- Replay is scheduled by **token re-prefill**. Native decode-time recurrent-state injection is not implemented.

---

## 2. Experimental Outcomes & Regimes

### 2.1 Measurement protocol

Hardware: Apple M5 MacBook Air, 10 cores (4 performance + 6 efficiency), 16 GB unified memory, macOS. Build: `clang -O3 -mcpu=apple-m5 -ffast-math -std=gnu11 -Wall -Wextra -pthread`; links `-lm -framework Accelerate`.

- Decode benchmark: 64-token prompt, 8 warm tokens, 64 generated tokens, K=16 unless stated. Target AR is the baseline.
- Headline decode ranges span three steady-state process runs per regime: `phase2_bench2–4` for 21M and `phase3_scaled_bench1–3` for 218M. B1 is repeated inside every bench capture; its tabled range spans the captures in `results/`. B2/B3 verification-cost ranges use the same three steady-state runs. Forced-acceptance rows are the median of three internal repetitions per process run.
- `c` is draft time divided by AR target time. `v` is one `K+1` target verification divided by one AR token. These are system ratios, not FLOP-normalized efficiencies.
- Derived rates use verifier parameter counts and measured tok/s: one MAC per target parameter per token, and one FP32 target-weight read per token. The derivation is in [`results/derived_rates.txt`](results/derived_rates.txt).
- The 21.35M checkpoint is trained externally. The 218.44M checkpoint is random and synthetic. No training log or JSON sidecar is present in the repository.

Raw captures:

| measurement | capture |
|---|---|
| final self-test | [`results/selftest_final.log`](results/selftest_final.log) |
| 21M verified checkpoint | [`results/verify_trained_21m.log`](results/verify_trained_21m.log) |
| 21M steady-state decode runs | [`results/phase2_bench2.log`](results/phase2_bench2.log), [`results/phase2_bench3.log`](results/phase2_bench3.log), [`results/phase2_bench4.log`](results/phase2_bench4.log) |
| 21M early run with 4-thread outliers | [`results/phase2_bench1.log`](results/phase2_bench1.log) |
| 21M later sanity run (excluded from headline ranges) | [`results/phase3_default_bench.log`](results/phase3_default_bench.log) |
| 218M loader/verifier | [`results/verify_scaled_synth_218m.log`](results/verify_scaled_synth_218m.log) |
| 218M decode runs | [`results/phase3_scaled_bench1.log`](results/phase3_scaled_bench1.log), [`results/phase3_scaled_bench2.log`](results/phase3_scaled_bench2.log), [`results/phase3_scaled_bench3.log`](results/phase3_scaled_bench3.log) |
| NoteBank demo with trained checkpoint | [`results/phase2_demo.log`](results/phase2_demo.log) |
| derived rates | [`results/derived_rates.txt`](results/derived_rates.txt) |

### 2.2 Regime A: 21M Cache-Resident Model (Trained Checkpoint)

**Operational label.** The 21M model is the small-working-set regime. The full **81.45 MiB** target is not wholly resident in the M5 system-level cache (SLC); the cache-resident description refers to its **6.75–6.76 MiB per-layer weight slice**, which is cache-sized. The benchmark does not include cache-occupancy counters. See the correction in §3.

| property | measured |
|---|---:|
| target | 12 layers, d=384, 6 heads, head_dim=64, d_ff=1024, 9 GDN + 3 attention |
| target parameters | 21,352,044 |
| target FP32 size | 85,408,176 B = 85.41 MB = **81.45 MiB** |
| draft | 4 layers `[0,3,6,9]`, 7,084,452 params = 28,337,808 B = **27.03 MiB FP32** |
| draft packed compute | **3,602,432 B = 3.44 MiB** total (q8 2.27 MiB + ternary 1.16 MiB) |
| checkpoint file | 113,762,508 B = **108.49 MiB** |
| verification tolerance | 1e-3 |

The packed figure excludes retained FP32 draft latents and shared embedding, final norm, LM head, and RoPE tables. The ternary MLP alone is **1.16 MiB**; the 3.60 MB figure is the total packed q8+ternary footprint, not the ternary MLP alone.

**PyTorch/C equivalence.** The trained checkpoint verifies:

```text
target: max_abs=7.39098e-06  argmax=16/16
draft : max_abs=0.0188234    argmax=16/16
```

The target pass condition is `argmax=16/16` and `max_abs < 1e-3`; the draft is reported but not part of the pass condition because it is quantized after loading. A C save/load round-trip reports `max_abs=0` for both models.

**Microarchitectural result.** Single-token AR is **366.5–371.0 tok/s** (2.695–2.729 ms/token). The measured effective weight-read rate is **29.15–29.51 GiB/s**, and the implied target compute rate is **7.83–7.92 GMAC/s**. That is within about 1.2x of the top of the standalone FP32 NEON kernel range (**5.84–6.67 GMAC/s**) on this machine. This is consistent with compute-bound execution, not memory-bandwidth saturation, despite the 81.45 MiB total footprint; no hardware bandwidth counters were collected.

**Draft efficiency.**

| path | threads | tok/s | c |
|---|---:|---:|---:|
| draft 4L only | 1 | 2973.8–2981.9 | 0.12x |
| draft 4L only | 4 | 4394.7–4627.6 | 0.08x |

The 1-to-4-worker steady-state speedup is **1.47–1.55x**. One early 4-thread run measured 1514.2 tok/s (`c=0.24x`) and is retained in `results/phase2_bench1.log`; it is excluded from the steady-state range but is a measured variance/limitation, not a deleted result.

**Speculative decode.**

| path | threads | K | tok/s | vs AR | acceptance |
|---|---:|---:|---:|---:|---|
| AR target | — | — | 366.5–371.0 | 1.00x | — |
| raw | 1 | 16 | 142.0–144.1 | 0.38–0.39x | alpha=0.052; 29/560 proposals |
| raw | 4 | 16 | 168.7–171.0 | 0.46x | alpha=0.052; 29/560 proposals |
| governed | 1 | 16 | 341.5–344.0 | 0.92–0.94x | alpha=0.094; 1 trip; 59 AR rounds |
| governed | 4 | 16 | 356.2–361.0 | 0.96–0.97x | alpha=0.094; 1 trip; 59 AR rounds |

Raw speculation **loses** on this checkpoint. The governor trips early (one trip) and falls back to AR, reaching **0.96–0.97x** AR at 4 threads. It is a safety mechanism that recovers near-baseline speed; it is not a real speedup.

**Forced-acceptance system ceilings.** These rows are benchmark-only: the draft and target forwards execute for real, but the accepted prefix is forced after the fact. They are not decoding quality.

| path | threads | K | tok/s | vs AR | forced alpha |
|---|---:|---:|---:|---:|---:|
| oracle-4L | 1 | 16 | 1016.6–1019.9 | 2.75–2.78x | 0.750 |
| oracle-4L | 1 | 16 | 1274.5–1290.8 | 3.45–3.52x | 1.000 |
| oracle-12L full copy | 1 | 16 | 302.4–311.0 | 0.82–0.85x | 1.000 |

The full-copy oracle remains below AR even at 100% acceptance: a target-sized draft pays roughly one full target forward per proposal. The cheap-draft ceiling is real only under forced acceptance; real raw acceptance on this checkpoint is 0.052.

### 2.3 Regime B: 218M DRAM-Resident Model (Systems Scaling)

> **Checkpoint caveat.** This 218.44M model uses **random synthetic weights** generated through the C save path. It is not trained, not committed, and cannot support language-quality, learned-acceptance, or PyTorch/C equivalence claims. Its verifier result is a loader/format self-consistency check.

| property | measured |
|---|---:|
| target | 16 layers, d=1024, 16 heads, head_dim=64, d_ff=3072, 12 GDN + 4 attention |
| target parameters | 218,436,992 |
| target FP32 size | 873,747,968 B = 873.75 MB = **833.27 MiB** |
| draft | 4 layers `[0,3,6,9]`, 54,543,456 params = 218,173,824 B = **208.07 MiB FP32** |
| draft packed compute | **26,394,624 B = 25.17 MiB** (q8 16.06 MiB + ternary 9.11 MiB) |
| checkpoint file | 1,091,938,316 B = **1041.35 MiB** |

**Loader/verifier.** The C-generated file verifies at `max_abs=0`, `argmax=16/16` at tolerance 1e-3. This only proves that the scaled header, tensor order, draft map, quantizers, and golden footer are self-consistent. It does **not** establish Python/C numerical agreement at 1024-wide scale.

**Execution regime.** The full target is 833.27 MiB; each layer streams **52.01–52.02 MiB** of FP32 weights. The model is unambiguously DRAM-resident. Measured AR is **29.5–29.9 tok/s** (33.445–33.898 ms/token), an effective **24.01–24.33 GiB/s** weight-read rate and **6.44–6.53 GMAC/s**.

**Verification cost and relative amortization.**

| metric | 21M | 218M |
|---|---:|---:|
| AR token (B3 decode) | 2.695–2.729 ms | 33.445–33.898 ms |
| B2 M=17 strict | 41.93–42.10 ms; v=15.54–15.74 | 548.28–552.46 ms; v=16.23–16.69 |
| B2 M=17 fast | 7.28–7.32 ms; v=2.69–2.74 | 49.97–55.16 ms; v=1.52–1.62 |
| B3 verify K+1=17 fast | 7.34–7.36 ms; v=2.69–2.72 | 51.47–53.89 ms; v=1.54–1.59 |

The absolute fast M=17 batch time grows by **6.8–7.6x**, while AR grows by **12.4x**; strict mode grows by about **13x** because it does not use AMX. The relative fast verification ratio `v` therefore falls from ~2.7x to ~1.5x. This is amortization of a much more expensive AR token, not an absolute AMX speedup. `v` is computed within each B2/B3 capture against that capture's own AR token; the AR row shows the B3 decode baseline.

**Draft acceleration.**

| path | threads | tok/s | c |
|---|---:|---:|---:|
| draft 4L only | 1 | 333.3–354.0 | 0.08–0.09x |
| draft 4L only | 4 | 734.0–1000.2 | 0.03–0.04x |

The 4-thread spread is large, about 266 tok/s, which indicates memory/scheduling sensitivity at this working-set size. The pool still lowers `c` by roughly 2–3x.

**Speculative decode on random weights.**

| path | threads | K | tok/s | vs AR | acceptance |
|---|---:|---:|---:|---:|---|
| raw | 1 | 16 | 9.9–10.2 | 0.34x | alpha=0.000; 0/1024 proposals |
| raw | 4 | 16 | 14.1–15.0 | 0.48–0.50x | alpha=0.000; 0/1024 proposals |
| governed | 1 | 16 | 28.2–28.9 | 0.96x | alpha=0.000; 1 trip; 63 AR rounds |
| governed | 4 | 16 | 28.8–29.3 | 0.97–0.98x | alpha=0.000; 1 trip; 63 AR rounds |

Alpha is exactly zero because the target is random. These rows demonstrate rejection behavior and governor fallback, not learned speculation.

**Forced-acceptance system ceilings (synthetic weights).**

| path | threads | K | tok/s | vs AR | forced alpha |
|---|---:|---:|---:|---:|---:|
| oracle-4L | 1 | 16 | 130.6–134.4 | 4.39–4.50x | 0.750 |
| oracle-4L | 1 | 16 | 163.5–167.6 | 5.50–5.61x | 1.000 |
| oracle-16L full copy | 1 | 16 | 26.0–26.4 | 0.88x | 1.000 |

The forced ceiling rises with scale, but the real rows do not. The 218M experiment therefore validates a **systems ceiling**, not real SubTernary decoding.

### 2.4 Cross-regime synthesis

| metric | 21M trained | 218M synthetic |
|---|---:|---:|
| target parameters | 21,352,044 | 218,436,992 |
| target FP32 | 81.45 MiB | 833.27 MiB |
| per-layer FP32 weights | 6.75–6.76 MiB | 52.01–52.02 MiB |
| AR | 366.5–371.0 tok/s | 29.5–29.9 tok/s |
| AR effective weight rate | 29.15–29.51 GiB/s | 24.01–24.33 GiB/s |
| AR implied MAC rate | 7.83–7.92 GMAC/s | 6.44–6.53 GMAC/s |
| draft t1 / t4 | 2973.8–2981.9 / 4394.7–4627.6 | 333.3–354.0 / 734.0–1000.2 |
| c t1 / t4 | 0.12 / 0.08 | 0.08–0.09 / 0.03–0.04 |
| v at M=17 fast (B2) | 2.69–2.74 | 1.52–1.62 |
| raw alpha | 0.052 | 0.000 (random) |
| governed | 0.92–0.97x AR | 0.96–0.98x AR |
| forced 0.75 | 2.75–2.78x | 4.39–4.50x |
| forced 1.00 | 3.45–3.52x | 5.50–5.61x |

**What the scaling experiment does and does not show.** DRAM residency is verified by footprint: the 218M target is 10.23x larger and each layer is 7.7x larger than the 21M layer. AR is about 12.4x slower. But the 218M target still runs at an implied **6.44–6.53 GMAC/s**, inside the measured FP32 NEON kernel range, and its effective weight-read rate is **lower** than the 21M model's. The data therefore support a **compute-limited, DRAM-resident** interpretation, not a demonstrated bandwidth-saturation point. The ratio `v` improves because AR becomes more expensive, which raises the forced oracle ceiling; it does not by itself validate a real decoding thesis.

**Cost model.** In a steady-state round the accepted token count is `1 + alpha*K`; the draft pays `K+1` forwards and the target pays one `K+1` verification:

```text
speedup(alpha) = (1 + alpha*K) / ((K+1)*c + v)
```

Using the measured per-run `c` and `v` at K=16, the serial-draft projections are:

| regime | alpha | projected | measured forced |
|---|---:|---:|---:|
| 21M | 0.75 | 2.69–2.72x | 2.75–2.78x |
| 21M | 1.00 | 3.51–3.56x | 3.45–3.52x |
| 218M | 0.75 | 4.24–4.38x | 4.39–4.50x |
| 218M | 1.00 | 5.55–5.73x | 5.50–5.61x |

The forced rows track the model because they use the same real draft and target forwards with a forced prefix. They are not evidence that real acceptance will approach those values.

### 2.5 NoteBank evaluation

The writer, addressing, branch/restore, and replay path are measured on a toy 128d/8-layer model and on the trained 384d/12-layer checkpoint:

| quantity | toy 128d/8L | trained 384d/12L |
|---|---:|---:|
| context / chunks / notes | 384 / 3 / 6 | 384 / 3 / 6 |
| writer time | 0.41 ms | 1.64 ms |
| address time (Q=4, R=2) | 0.021–0.025 ms | 0.045–0.046 ms |
| writer needle containment | 3/3 | 3/3 |
| summed-similarity recall@2 | 3/3 | 3/3 |
| vote-ranker recall@2 | 2/3 | 2/3 |
| note storage | 6,720 B | 14,400 B |
| dense KV, all layers | 3,145,728 B (468x) | 14,155,776 B (983x) |
| hybrid attention KV | 786,432 B (117x) | 3,538,944 B (245.8x) |
| replay value rank | 73/92/6 unchanged | 117/79/49 unchanged |

Positive result: chunk-level distinctiveness contains all three planted needles, and summed-similarity addressing retrieves a needle-containing note at recall@2 of **3/3** in both configurations. The vote ranker loses (**2/3**) and remains opt-in.

Negative result: replay did not improve the rank of the planted value token in any measured case; `no-replay` and `replay` ranks are identical. The branch/restore path proves scheduling and state-isolation plumbing, not a retrieval benefit. Notes are not a replacement for attention KV, and replay is additional token re-prefill.

The demo's NoteBank-to-spec run on the trained checkpoint produced alpha=0.000, one governor trip, and 7 AR rounds. A separate 48-token spec run on the demo's freshly initialized random model produced 48/48 AR-matching tokens at alpha=0.036 and 1.14 tokens/round. These validate integration and greedy fidelity, not retrieval quality or trained acceptance.

### 2.6 Correctness invariants

`make selftest` exercises eight suites on the strict path:

| suite | measured result |
|---|---|
| `[1]` ternary pack/dot/GEMV | dotprod, addsub, scalar agree; GEMV matches dequantized reference |
| `[2]` batched vs sequential | 256 logits, **0 differ** |
| `[3]` speculative fidelity | plain 40/40; governed 32/32; K=16 32/32; K=16 governed 32/32 |
| `[4]` NoteBank | needles 3/3; similarity recall 3/3; votes 2/3; toy storage 6,720 B |
| `[5]` fast path | strict AR vs fast spec 24/24 tokens; 248/256 logit bits differ; argmax differs 0/4 |
| `[6]` forced acceptance | 30/40 at 0.75; 32/32 at 1.00; emitted 32/32; target/draft positions aligned |
| `[7]` weights round-trip | `max_abs=0`, 16/16 target and draft |
| `[8]` draft pool | 16 x 128 logits, threads 1 vs 4, **0 differ** |

Strict mode is the only bit-exact path. Fast mode is token-equivalent on the tested inputs, not bit-for-bit. All guarantees are same-binary, same-flags (`-O3 -ffast-math`); cross-compiler and cross-architecture bit-exactness are not claimed.

---

## 3. Drawbacks, Limitations & Negative Results

1. **The acceptance-rate gap is the dominant negative result.** The trained checkpoint reaches raw alpha **0.052** (29 accepted of 560 proposals) and governed alpha **0.094** on the benchmark prompt. Raw decoding runs at **0.38–0.46x AR**; the governor recovers **0.92–0.97x AR** by abandoning speculation. The repository contains no training log or sidecar metrics, so no KL-divergence reduction is reported. A claimed 98% KL reduction cannot be substantiated from this artifact and is omitted.

2. **Small-model speculation is not a speedup.** The 21M target already runs at 366.5–371.0 tok/s, and its measured MAC rate is consistent with compute-bound execution. Its raw speculation loses, and its governed path is approximately parity. The 2.75–3.52x figures are forced-acceptance system ceilings. Real decoding on this checkpoint does not approach them.

3. **DRAM residency is verified; bandwidth saturation is not.** The 218M target is 833.27 MiB and streams 52 MiB per layer, so it is DRAM-resident. However, its measured 6.44–6.53 GMAC/s is within the FP32 NEON kernel range, and its effective 24.0–24.3 GiB/s weight rate is lower than the 21M rate. The experiment does not establish a bandwidth-saturation ceiling. The forced 4.39–5.61x rows are systems ceilings with synthetic weights; the real rows are 0.34–0.50x raw and 0.96–0.98x governed.

4. **The “cache-resident” label is layer-local, not full-model.** The full 21M target is 81.45 MiB and is not wholly SLC-resident. Each layer is 6.75–6.76 MiB, which is cache-sized. No cache counters were collected. The full target does not fit in a single on-chip cache working set.

5. **Target verification remains single-threaded.** The draft pool lowers `c`, but target `v` is unchanged by threading because the target path is deliberately serial. At 218M, absolute M=17 verification is **49.97–55.16 ms** (fast) or **548.28–552.46 ms** (strict). The relative `v` drop is amortization, not a lower absolute verification cost.

6. **The scaled checkpoint is synthetic.** The 218M file is a C-generated random model used for loader and systems scaling. It cannot support language quality, trained acceptance, or Python/C numerical equivalence at 1024-wide scale. A trained scaled checkpoint is still required.

7. **NoteBank replay scope is limited.** `S_T` is snapshotted and restored around retrieval, but replay uses token re-prefill rather than native recurrent-state injection. Replay did not improve value rank, the vote ranker is 2/3, and notes do not replace KV. The memory path is a reference implementation, not a trained memory system.

8. **Run-to-run variance is material.** The early 4-thread 21M run was 1514.2 tok/s (`c=0.24x`), versus 4394.7–4627.6 tok/s in steady state. The 218M 4-thread draft spans 734.0–1000.2 tok/s. A later 21M sanity run measured governed 334.2/323.1 tok/s and forced-0.75 870.1 tok/s, outside the three-run steady-state table; the cause was not isolated. Headline decode and forced ranges exclude that later run, and the early outlier is retained in `results/` rather than deleted.

9. **Scope limits.** The tokenizer and data path are ASCII/synthetic; there is no pretrained language model, no committed trained checkpoint, no GPU/ANE path, and no multi-threaded target. Strict batched verification is bit-exact; fast AMX mode is not. `max_seq` is capped at 4096 and the benchmark uses 512.

---

## 4. Build, Test & Benchmark Quickstart

### 4.1 Build and run

Requirements: macOS on ARM64 with NEON dot-product support, clang, and Accelerate. The default Makefile target is `-mcpu=apple-m5`.

```sh
git clone https://github.com/ayush6288/spel.git
cd spel
make                 # builds ./spel, links -framework Accelerate
make selftest        # 8 invariant suites; must print PASS (0 failures)
make bench           # kernels + strict/fast verification + decode + rollback
make demo            # NoteBank branch -> retrieve -> replay -> spec demo

# Compare speculative depth on the same machine:
SPEL_BENCH_K=8  ./spel bench
SPEL_BENCH_K=16 ./spel bench
```

`make selftest` is self-contained. The decode numbers in §2 require the local trained 21M checkpoint at `weights/spel.bin`; without it, `bench` runs a randomly initialized model. `demo` uses the checkpoint for the NoteBank path when it is present and falls back to the toy configuration otherwise. The 48-token spec run is always a freshly initialized model and is a plumbing check.

On another ARM64 Mac:

```sh
make clean && make ARCH="-mcpu=apple-m1"
# or
make clean && make ARCH="-march=native"
```

### 4.2 Runtime configuration

The default shape is `d_model=384, n_layers=12, n_heads=6, head_dim=64, d_ff=1024, vocab=128`. The scaled shape is `d_model=1024, n_layers=16, n_heads=16, head_dim=64, d_ff=3072`.

CLI:

```sh
./spel bench --config d_model=1024,n_layers=16,n_heads=16,head_dim=64,d_ff=3072
./spel verify-weights weights/spel.bin
# When a scaled trained checkpoint exists:
./spel verify-weights weights/spel_scaled.bin \
  --config d_model=1024,n_layers=16,n_heads=16,head_dim=64,d_ff=3072
```

Environment overrides: `SPEL_CONFIG`, `SPEL_D_MODEL`, `SPEL_N_LAYERS`, `SPEL_N_HEADS`, `SPEL_HEAD_DIM`, `SPEL_D_FF`, `SPEL_VOCAB`, `SPEL_MAX_SEQ`, `SPEL_ATTN_PERIOD`, `SPEL_SPEC_K`, `SPEL_DRAFT_MAP`, `SPEL_DRAFT_THREADS`, `SPEL_BENCH_K`, `SPEL_VERIFY_TOL`, and `SPEL_WEIGHTS`.

Draft threading:

```sh
SPEL_DRAFT_THREADS=1 ./spel bench   # default
SPEL_DRAFT_THREADS=4 ./spel bench
```

The draft pool is draft-only; target verification is not parallelized.

### 4.3 Checkpoint verification

`./spel verify-weights` loads the default 384d/12-layer target and the 4-layer draft, runs 16 golden tokens from a reset state, and reports max-abs logit error and argmax agreement. Pass requires target argmax 16/16 and target `max_abs < SPEL_VERIFY_TOL` (default `1e-3`). It prints actual target/draft parameter counts, FP32 sizes, and file size.

For the scaled synthetic file used in this report:

```sh
./spel verify-weights /tmp/spel_scaled_218M_synth.bin \
  --config d_model=1024,n_layers=16,n_heads=16,head_dim=64,d_ff=3072
```

The file is not committed. The scaled training command, when a real checkpoint is available, is:

```sh
python train_colab.py --d-model 1024 --n-layers 16 --n-heads 16 \
  --head-dim 64 --d-ff 3072 --out weights/spel_scaled.bin
```

The 218M trained checkpoint, trained verification, and trained acceptance measurements remain pending.

### 4.4 Files and format

| file | purpose |
|---|---|
| `spel.h` | configuration, model, ternary/q8, NoteBank, spec, and `SPELW001` APIs |
| `ternary.c` | packing, q8/ternary quantization, NEON dotprod/addsub, scalar GEMV |
| `model.c` | hybrid forward, KV/state rings, checkpoints, spec loop, governor, weights I/O |
| `notebank.c` | chunked writer, BF16 addresses, addressing, vote/similarity ranking |
| `main.c` | selftests, benchmarks, NoteBank/spec demo, `verify-weights` |
| `train_colab.py` | PyTorch transliteration, synthetic data, target training, QAT distillation, export |
| `Makefile` | build, `selftest`, `bench`, `demo`, `clean` |
| `results/` | raw measurement captures cited in this report |

`SPELW001` is a 64-byte little-endian header followed by raw FP32 tensors and an optional `SPELV001` footer with 16 golden tokens and target/draft logits. The four draft layers hold independent distilled FP32 parameters for layers `[0,3,6,9]`; the draft shares the token embedding, final RMSNorm, LM head, and RoPE tables. On load, C quantizes each draft `wqkv`/`wo` with q8 and `w1`/`w3`/`w2` with ternary least-squares. The file keeps FP32 latents, so QAT export is lossless with respect to the C quantizers.

The repository does not commit `weights/`. The trained 21M checkpoint and the synthetic 218M file are local artifacts.

---

**Repository:** <https://github.com/ayush6288/spel>
