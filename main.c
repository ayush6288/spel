/* main.c — selftests, kernel/decoder benchmarks, NoteBank demo. */
#include "spel.h"
#include <arm_neon.h>
#include <time.h>

static int failures = 0;
#define CHECK(cond, ...) do { \
    if (!(cond)) { failures++; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } \
} while (0)

static void fill_tokens(int *t, int n, int vocab, uint64_t *rng) {
    for (int i = 0; i < n; i++) t[i] = (int)(spel_rng_next(rng) % (uint64_t)vocab);
}

static float frand(uint64_t *rng) { return (float)(spel_rng_double(rng) * 2.0 - 1.0); }

static uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

/* --------------------------------------------------------- ternary test */

static void test_ternary(void) {
    printf("[1] ternary pack / dot / GEMV\n");
    uint64_t rng = 0x1234;
    for (int trial = 0; trial < 50; trial++) {
        int nrows = 1 + (int)(spel_rng_next(&rng) % 8);
        int ncols = 1 + (int)(spel_rng_next(&rng) % 200);
        float *W = (float *)malloc((size_t)nrows * ncols * sizeof(float));
        float *x = (float *)malloc((size_t)ncols * sizeof(float));
        float *y = (float *)malloc((size_t)nrows * sizeof(float));
        tern_mat tm; tern_mat_init(&tm, nrows, ncols);
        for (int i = 0; i < nrows * ncols; i++) W[i] = frand(&rng) * (i % 7 == 0 ? 0.0f : 1.0f);
        for (int i = 0; i < ncols; i++) x[i] = frand(&rng) * 3.0f;
        tern_quantize_mat(&tm, W);
        tern_gemv(&tm, x, y);
        float xmax = 0.0f;
        for (int i = 0; i < ncols; i++) xmax = fmaxf(xmax, fabsf(x[i]));
        float xs = xmax / 127.0f; if (xs < 1e-12f) xs = 1e-12f;
        int8_t xq[512];
        for (int i = 0; i < ncols; i++) xq[i] = (int8_t)roundf(x[i] / xs);
        for (int r = 0; r < nrows; r++) {
            float *deq = (float *)malloc((size_t)ncols * sizeof(float));
            tern_dequant_row(tm.packed + (size_t)r * tm.packed_stride, tm.scale[r], ncols, deq);
            double ref = 0.0;
            for (int i = 0; i < ncols; i++) ref += (double)deq[i] * ((double)xq[i] * xs);
            CHECK(fabs((double)y[r] - ref) <= 1e-5 * (1.0 + fabs(ref)),
                  "gemv r=%d y=%.6f ref=%.6f", r, y[r], ref);
            free(deq);
        }
        /* integer kernels must agree exactly */
        for (int i = 0; i < ncols; i++) xq[i] = (int8_t)((int)(spel_rng_next(&rng) % 255) - 127);
        for (int r = 0; r < nrows; r++) {
            const uint8_t *p = tm.packed + (size_t)r * tm.packed_stride;
            int save = tern_kernel_mode;
            tern_kernel_mode = 2; int32_t a = tern_dot_i8(p, xq, ncols);
            tern_kernel_mode = 1; int32_t b = tern_dot_i8(p, xq, ncols);
            tern_kernel_mode = 0; int32_t c = tern_dot_i8(p, xq, ncols);
            tern_kernel_mode = save;
            CHECK(a == b && b == c, "kernel mismatch scalar=%d addsub=%d dot=%d", a, b, c);
        }
        tern_mat_free(&tm);
        free(W); free(x); free(y);
    }
    printf("  kernels: %s (dotprod/addsub/scalar agree)\n", tern_kernel_name());
}

/* ------------------------------------------------------ model bit tests */

static void test_batch_equiv(void) {
    printf("[2] batched verification == sequential AR (bitwise)\n");
    spel_cfg c;
    spel_cfg_init(&c, 128, 8, 4, 256, 64, 256, 4, 4);
    spel_model *m = spel_model_new(&c, 77);
    uint64_t rng = 9;
    int toks[32]; fill_tokens(toks, 32, c.vocab, &rng);
    for (int i = 0; i < 20; i++) spel_forward(m, &toks[i], 1, NULL);
    spel_ckpt *cp = spel_ckpt_new(m);
    float a[4 * 64], b[4 * 64];
    spel_forward(m, toks + 20, 4, a);
    spel_ckpt_load(m, cp);
    for (int i = 0; i < 4; i++) spel_forward(m, &toks[20 + i], 1, b + i * c.vocab);
    int diff = 0;
    for (int i = 0; i < 4 * c.vocab; i++) if (fbits(a[i]) != fbits(b[i])) diff++;
    CHECK(diff == 0, "%d/%d logits differ", diff, 4 * c.vocab);
    printf("  %d logits compared, %d differ\n", 4 * c.vocab, diff);
    spel_ckpt_free(cp);
    spel_model_free(m);
}

/* -------------------------------------------------------- spec decoding */

typedef struct {
    int matched;
    float probe_diff_bits;
    spel_spec_stats st;
    int n;
} spec_result;

static void run_spec_case(int d, int layers, int dff, int vocab, int max_seq,
                          int prompt_n, int gen_n, uint64_t seed,
                          int spec_k, int spec_K, int governor,
                          double threshold, spec_result *res) {
    spel_cfg c;
    spel_cfg_init(&c, d, layers, 4, dff, vocab, max_seq, 4, spec_k);
    spel_model *target = spel_model_new(&c, seed);
    spel_model *draft = spel_model_new_draft(target);
    uint64_t rng = seed ^ 0xDEADBEEF;
    int prompt[128];
    fill_tokens(prompt, prompt_n, vocab, &rng);

    spel_model_reset(target); spel_model_reset(draft);
    for (int i = 0; i < prompt_n; i += spec_k) {
        int n = prompt_n - i < spec_k ? prompt_n - i : spec_k;
        spel_forward(target, prompt + i, n, NULL);
        spel_forward(draft, prompt + i, n, NULL);
    }
    spel_ckpt *cp_t = spel_ckpt_new(target);
    spel_ckpt *cp_d = spel_ckpt_new(draft);

    int *ar = (int *)malloc((size_t)gen_n * sizeof(int));
    int *sp = (int *)malloc((size_t)gen_n * sizeof(int));
    for (int i = 0; i < gen_n; i++) {
        int t = spel_argmax(target->last_logits, vocab);
        ar[i] = t;
        spel_forward(target, &t, 1, NULL);
    }
    int probe = (int)(rng % (uint64_t)vocab);
    spel_forward(target, &probe, 1, NULL);
    float ar_probe[SPEL_MAX_VOCAB];
    memcpy(ar_probe, target->last_logits, (size_t)vocab * sizeof(float));
    int ar_pos = target->pos;

    spel_ckpt_load(target, cp_t);
    spel_ckpt_load(draft, cp_d);
    spel_spec_cfg sc; spel_spec_cfg_default(&sc, spec_K);
    sc.governor = governor; sc.alpha_threshold = threshold;
    spel_spec_stats st;
    spel_spec_generate(target, draft, gen_n, sp, &sc, &st);
    spel_forward(target, &probe, 1, NULL);
    float sp_probe[SPEL_MAX_VOCAB];
    memcpy(sp_probe, target->last_logits, (size_t)vocab * sizeof(float));
    int sp_pos = target->pos;

    int match = 0;
    for (int i = 0; i < gen_n; i++) if (ar[i] == sp[i]) match++;
    int pd = 0;
    for (int i = 0; i < vocab; i++) if (fbits(ar_probe[i]) != fbits(sp_probe[i])) pd++;

    res->matched = match;
    res->probe_diff_bits = (float)pd;
    res->st = st;
    res->n = gen_n;
    CHECK(match == gen_n, "token mismatch %d/%d", match, gen_n);
    CHECK(pd == 0, "post-generation state probe differs in %d/%d logits (pos %d vs %d)",
          pd, vocab, ar_pos, sp_pos);
    CHECK(ar_pos == sp_pos, "final pos differs");

    free(ar); free(sp);
    spel_ckpt_free(cp_t); spel_ckpt_free(cp_d);
    spel_model_free(target); spel_model_free(draft);
}

static void test_spec(void) {
    printf("[3] speculative decode: exact greedy fidelity + rollback\n");
    spec_result r;
    run_spec_case(128, 8, 256, 64, 256, 24, 40, 12345, 4, 4, 0, 0.0, &r);
    printf("  plain:   %d/%d tokens match, alpha=%.3f, trips=%ld\n",
           r.matched, r.n, r.st.alpha, r.st.governor_trips);
    CHECK(r.st.alpha >= 0.0 && r.st.alpha <= 1.0, "bad alpha");

    run_spec_case(128, 8, 256, 64, 256, 24, 32, 999, 4, 4, 1, 1.0, &r);
    printf("  governor: %d/%d match, alpha=%.3f, AR rounds=%ld, trips=%ld\n",
           r.matched, r.n, r.st.alpha, r.st.ar_rounds, r.st.governor_trips);
    CHECK(r.st.governor_trips > 0, "governor never tripped at threshold 1.0");
    CHECK(r.st.ar_rounds > 0, "governor emitted no AR fallback rounds");

    /* M=17 batch, ring R=17, deferred-correction path. */
    run_spec_case(128, 8, 256, 64, 256, 24, 32, 4242, 16, 16, 0, 0.0, &r);
    printf("  K=16:    %d/%d tokens match, alpha=%.3f, trips=%ld\n",
           r.matched, r.n, r.st.alpha, r.st.governor_trips);
    CHECK(r.st.alpha >= 0.0 && r.st.alpha <= 1.0, "bad alpha");

    run_spec_case(128, 8, 256, 64, 256, 24, 32, 4243, 16, 16, 1, 1.0, &r);
    printf("  K=16 gov:%d/%d match, alpha=%.3f, AR rounds=%ld, trips=%ld\n",
           r.matched, r.n, r.st.alpha, r.st.ar_rounds, r.st.governor_trips);
    CHECK(r.st.governor_trips > 0, "K=16 governor never tripped");
}

/* -------------------------------------------------------- fast math test */

static void test_fast_mode(void) {
    printf("[5] Accelerate fast path: token equivalence (not bitwise)\n");
    spel_cfg c;
    spel_cfg_init(&c, 128, 8, 4, 256, 64, 128, 4, 16);
    spel_model *target = spel_model_new(&c, 4321);
    spel_model *draft = spel_model_new_draft(target);
    uint64_t rng = 0xF00D;
    int prompt[24];
    fill_tokens(prompt, 24, c.vocab, &rng);
    for (int i = 0; i < 24; i += 4)
        spel_forward(target, prompt + i, 4, NULL);
    for (int i = 0; i < 24; i += 4)
        spel_forward(draft, prompt + i, 4, NULL);
    spel_ckpt *cp_t = spel_ckpt_new(target);
    spel_ckpt *cp_d = spel_ckpt_new(draft);

    const int n = 24;
    int ar[24], fast[24];

    spel_use_accelerate = 0;
    for (int i = 0; i < n; i++) {
        ar[i] = spel_argmax(target->last_logits, c.vocab);
        spel_forward(target, &ar[i], 1, NULL);
    }

    spel_ckpt_load(target, cp_t);
    spel_ckpt_load(draft, cp_d);
    spel_use_accelerate = 1;
    spel_spec_cfg sc; spel_spec_cfg_default(&sc, 16);
    sc.governor = 0;
    spel_spec_stats st;
    spel_spec_generate(target, draft, n, fast, &sc, &st);

    int match = 0;
    for (int i = 0; i < n; i++) if (ar[i] == fast[i]) match++;
    printf("  strict AR vs fast spec: %d/%d tokens match\n", match, n);
    CHECK(match == n, "fast spec token mismatch %d/%d", match, n);

    /* Quantify the bitwise difference on a verification-shaped batch. */
    int toks[4] = {3, 17, 41, 5};
    float a[4 * 64], b[4 * 64];
    spel_ckpt_load(target, cp_t);
    spel_use_accelerate = 0;
    spel_forward(target, toks, 4, a);
    spel_ckpt_load(target, cp_t);
    spel_use_accelerate = 1;
    spel_forward(target, toks, 4, b);
    int bits = 0, am = 0;
    for (int r = 0; r < 4; r++) {
        for (int i = 0; i < c.vocab; i++)
            if (fbits(a[r * c.vocab + i]) != fbits(b[r * c.vocab + i])) bits++;
        if (spel_argmax(a + r * c.vocab, c.vocab) !=
            spel_argmax(b + r * c.vocab, c.vocab)) am++;
    }
    printf("  batched logits: %d/%d bits differ, argmax differs at %d/4 positions\n",
           bits, 4 * c.vocab, am);

    spel_use_accelerate = 0;
    spel_ckpt_free(cp_t); spel_ckpt_free(cp_d);
    spel_model_free(target); spel_model_free(draft);
}

/* ------------------------------------------------------------- notebank */

static int note_contains(const nb_bank *b, int pos) {
    for (int i = 0; i < b->n_notes; i++) {
        if (pos >= b->notes[i].start && pos < b->notes[i].start + b->notes[i].n_tokens)
            return i;
    }
    return -1;
}

static int rank_of(const float *logits, int V, int tok) {
    int r = 0;
    for (int i = 0; i < V; i++) if (logits[i] > logits[tok]) r++;
    return r;
}

static void test_notebank(int with_spec) {
    printf("[4] NoteBank writer / addressing / replay\n");
    spel_cfg c;
    spel_cfg_init(&c, 128, 8, 4, 256, 128, 512, 4, 4);
    spel_model *m = spel_model_new(&c, 2024);
    spel_model_reset(m);
    spel_capture_alloc(m, c.max_seq);

    const int T = 384, C = 128, N_NEEDLE = 3;
    int *ctx = (int *)malloc((size_t)T * sizeof(int));
    uint64_t rng = 4242;
    for (int i = 0; i < T; i++) ctx[i] = 1 + (int)(spel_rng_next(&rng) % 8);
    int needles[3] = {40, 170, 300};
    int keys[3] = {100, 102, 104};
    int vals[3] = {80, 81, 82};
    for (int i = 0; i < N_NEEDLE; i++) {
        int p = needles[i];
        ctx[p + 0] = keys[i];
        ctx[p + 1] = vals[i];
        ctx[p + 2] = keys[i] + 1;
        ctx[p + 3] = vals[i] + 1;
        ctx[p + 4] = keys[i] + 2;
        ctx[p + 5] = vals[i] + 2;
    }
    m->cap_enabled = 1;
    for (int i = 0; i < T; i += 4) spel_forward(m, ctx + i, 4, NULL);
    m->cap_enabled = 0;

    nb_bank bank;
    nb_bank_init(&bank, &c, C, 12, 24, 2, 16, 777);
    double t0 = spel_now_ms();
    nb_write(&bank, &c, ctx, T, m->cap_k, m->cap_q);
    double tw = spel_now_ms() - t0;
    printf("  chunks=%d notes=%d write=%.2f ms\n", (T + C - 1) / C, bank.n_notes, tw);
    printf("  note spans:");
    for (int i = 0; i < bank.n_notes; i++) printf(" [%d,%d)", bank.notes[i].start,
                                                  bank.notes[i].start + bank.notes[i].n_tokens);
    printf("\n");

    for (int i = 0; i < N_NEEDLE; i++) {
        int ni = note_contains(&bank, needles[i]);
        CHECK(ni >= 0, "writer missed needle %d at %d", i, needles[i]);
        printf("  writer needle %d @%d -> note %d %s\n", i, needles[i], ni, ni >= 0 ? "ok" : "MISS");
    }

    spel_ckpt *cp = spel_ckpt_new(m);
    int sel[4];
    for (int i = 0; i < N_NEEDLE; i++) {
        spel_ckpt_load(m, cp);
        int q[4] = {keys[i], keys[i] + 1, keys[i] + 2, 127};
        m->cap_enabled = 1;
        spel_forward(m, q, 4, NULL);
        m->cap_enabled = 0;
        double ta = spel_now_ms();
        int ns = nb_address(&bank, &c, 4, m->cap_q, T, 2, sel);
        ta = spel_now_ms() - ta;
        int hit = 0;
        for (int j = 0; j < ns; j++)
            if (note_contains(&bank, needles[i]) == sel[j]) hit = 1;
        CHECK(hit, "address missed needle %d (selected %d,%d)", i,
              ns > 0 ? sel[0] : -1, ns > 1 ? sel[1] : -1);
        printf("  address needle %d -> notes %d,%d,%d,%d  hit=%d  %.3f ms\n", i,
               ns > 0 ? sel[0] : -1, ns > 1 ? sel[1] : -1,
               ns > 2 ? sel[2] : -1, ns > 3 ? sel[3] : -1, hit, ta);

        /* replay vs no replay: rank of the planted value token */
        spel_ckpt_load(m, cp);
        spel_forward(m, q, 4, NULL);
        int rank_no = rank_of(m->last_logits, c.vocab, vals[i]);
        spel_ckpt_load(m, cp);
        for (int j = 0; j < ns; j++) {
            nb_note *n = &bank.notes[sel[j]];
            spel_forward(m, n->tokens, n->n_tokens, NULL);
        }
        spel_forward(m, q, 4, NULL);
        int rank_yes = rank_of(m->last_logits, c.vocab, vals[i]);
        printf("      value rank: no-replay=%d replay=%d (V=%d)\n", rank_no, rank_yes, c.vocab);
    }

    /* Paper-style per-channel integer voting. Kept opt-in until a trained
     * model gives the channels meaningful preferences. */
    bank.rank_mode = NB_RANK_VOTES;
    int vote_hits = 0;
    for (int i = 0; i < N_NEEDLE; i++) {
        spel_ckpt_load(m, cp);
        int q[4] = {keys[i], keys[i] + 1, keys[i] + 2, 127};
        m->cap_enabled = 1;
        spel_forward(m, q, 4, NULL);
        m->cap_enabled = 0;
        int ns = nb_address(&bank, &c, 4, m->cap_q, T, 2, sel);
        int hit = 0;
        for (int j = 0; j < ns; j++)
            if (note_contains(&bank, needles[i]) == sel[j]) hit = 1;
        vote_hits += hit;
        printf("  votes needle %d -> notes %d,%d  hit=%d\n", i,
               ns > 0 ? sel[0] : -1, ns > 1 ? sel[1] : -1, hit);
    }
    bank.rank_mode = NB_RANK_SIM;
    printf("  vote recall: %d/%d (similarity ranking remains default)\n",
           vote_hits, N_NEEDLE);

    if (with_spec) {
        /* Branch S_T -> addressing -> restore S_T -> replay -> speculative
         * generation. Replay is still token re-prefill, not native state
         * injection; this only proves the integration path. */
        int map[4] = {0, 3, 5, 7};
        spel_model *draft = spel_model_new_draft_subset(m, map, 4);
        if (draft) {
            spel_model_reset(draft);
            for (int i = 0; i < T; i += c.spec_k) {
                int n = T - i < c.spec_k ? T - i : c.spec_k;
                spel_forward(draft, ctx + i, n, NULL);
            }
            spel_ckpt *cp_d = spel_ckpt_new(draft);
            spel_ckpt_load(m, cp);
            int q[4] = {keys[0], keys[0] + 1, keys[0] + 2, 127};
            m->cap_enabled = 1;
            spel_forward(m, q, 4, NULL);
            m->cap_enabled = 0;
            int ns = nb_address(&bank, &c, 4, m->cap_q, T, 2, sel);
            spel_ckpt_load(m, cp);
            spel_ckpt_load(draft, cp_d);
            for (int j = 0; j < ns; j++) {
                nb_note *n = &bank.notes[sel[j]];
                spel_forward(m, n->tokens, n->n_tokens, NULL);
                spel_forward(draft, n->tokens, n->n_tokens, NULL);
            }
            spel_forward(m, q, 4, NULL);
            spel_forward(draft, q, 4, NULL);

            spel_spec_cfg sc; spel_spec_cfg_default(&sc, 4);
            sc.governor = 1; sc.alpha_threshold = 0.30; sc.window = 2;
            int out[8];
            spel_spec_stats st;
            spel_spec_generate(m, draft, 8, out, &sc, &st);
            printf("  NoteBank->spec: notes %d,%d ->",
                   ns > 0 ? sel[0] : -1, ns > 1 ? sel[1] : -1);
            for (int i = 0; i < 8; i++) printf(" %d", out[i]);
            printf("  alpha=%.3f trips=%ld AR=%ld\n",
                   st.alpha, st.governor_trips, st.ar_rounds);
            spel_ckpt_free(cp_d);
            spel_model_free(draft);
        } else {
            printf("  NoteBank->spec: draft alloc failed\n");
        }
    }

    size_t nb_bytes = spel_note_bytes(&bank);
    size_t dense = 2 * (size_t)c.n_layers * T * c.d_model * 4;
    size_t actual = 2 * (size_t)c.n_attn * T * c.d_model * 4;
    printf("  storage: notes=%zu B  dense-KV=%zu B (%.0fx)  hybrid-KV=%zu B (%.1fx)\n",
           nb_bytes, dense, (double)dense / (double)nb_bytes,
           actual, (double)actual / (double)nb_bytes);
    CHECK(nb_bytes > 0, "empty notes");

    nb_bank_free(&bank);
    free(ctx);
    spel_ckpt_free(cp);
    spel_capture_free(m);
    spel_model_free(m);
}

/* ------------------------------------------------------------- benches */

static float dot_f32(const float *a, const float *b, int n) {
    float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
        s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    float s = vaddvq_f32(vaddq_f32(s0, s1));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

static void bench_ternary(void) {
    printf("[B1] ternary GEMV vs FP32 GEMV (M5, 1 thread)\n");
    const int N = 4096, K = 1536;
    float *W = (float *)malloc((size_t)N * K * sizeof(float));
    float *x = (float *)malloc((size_t)K * sizeof(float));
    float *y = (float *)malloc((size_t)N * sizeof(float));
    uint64_t rng = 55;
    for (int i = 0; i < N * K; i++) W[i] = frand(&rng) * (i % 11 == 0 ? 0.0f : 1.0f);
    for (int i = 0; i < K; i++) x[i] = frand(&rng);
    tern_mat tm; tern_mat_init(&tm, N, K);
    double tq = spel_now_ms();
    tern_quantize_mat(&tm, W);
    tq = spel_now_ms() - tq;
    printf("  quantize %dx%d: %.1f ms, packed=%.2f MiB, float=%.2f MiB\n",
           N, K, tq, (double)N * tm.packed_stride / 1048576.0,
           (double)N * K * 4 / 1048576.0);

    int iters = 200;
    for (int mode = 0; mode < 3; mode++) {
        tern_kernel_mode = mode;
        tern_gemv(&tm, x, y);
        double t0 = spel_now_ms();
        for (int it = 0; it < iters; it++) tern_gemv(&tm, x, y);
        double dt = (spel_now_ms() - t0) / 1000.0;
        double bytes = (double)N * tm.packed_stride * iters;
        double macs = (double)N * K * iters;
        printf("  %-13s %7.2f G MAC/s  %6.1f GB/s (packed)  %.2f ms/GEMV\n",
               mode == 0 ? "neon-dotprod" : mode == 1 ? "neon-addsub" : "scalar",
               macs / dt / 1e9, bytes / dt / 1e9, dt * 1000.0 / iters);
    }
    tern_kernel_mode = 0;

    /* FP32 baseline, same shape */
    double t0 = spel_now_ms();
    for (int it = 0; it < 20; it++) {
        for (int r = 0; r < N; r++) y[r] = dot_f32(W + (size_t)r * K, x, K);
    }
    double dt = (spel_now_ms() - t0) / 1000.0;
    printf("  %-13s %7.2f G MAC/s  %6.1f GB/s (fp32)\n",
           "fp32-neon", (double)N * K * 20 / dt / 1e9,
           (double)N * K * 4 * 20 / dt / 1e9);

    tern_mat_free(&tm);
    free(W); free(x); free(y);
}

static void prefill_chunks(spel_model *m, const int *toks, int n) {
    int chunk = m->cfg.spec_k;
    if (chunk < 1) chunk = 1;
    for (int i = 0; i < n; i += chunk) {
        int k = n - i < chunk ? n - i : chunk;
        spel_forward(m, toks + i, k, NULL);
    }
}

static void bench_verify(void) {
    printf("[B2] verification cost (d=384, 12 layers, 9 GDN + 3 attn)\n");
    spel_cfg c;
    spel_cfg_init(&c, 384, 12, 6, 1024, 128, 512, 4, 16);
    spel_model *t = spel_model_new(&c, 31337);
    uint64_t rng = 8;
    int prompt[64];
    fill_tokens(prompt, 64, c.vocab, &rng);
    spel_use_accelerate = 0;
    prefill_chunks(t, prompt, 64);
    spel_ckpt *cp = spel_ckpt_new(t);

    const int gen = 32;
    double t0 = spel_now_ms();
    for (int i = 0; i < gen; i++) {
        int tok = spel_argmax(t->last_logits, c.vocab);
        spel_forward(t, &tok, 1, NULL);
    }
    double ar_ms = (spel_now_ms() - t0) / gen;
    printf("  AR token: %.2f ms\n", ar_ms);

    int toks[17] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
    int ms_list[6] = {4, 5, 8, 9, 16, 17};
    for (int fi = 0; fi <= 1; fi++) {
        spel_use_accelerate = fi;
        for (int mi = 0; mi < 6; mi++) {
            int M = ms_list[mi];
            if (M > c.spec_k + 1) continue;
            spel_ckpt_load(t, cp);
            spel_forward(t, toks, M, NULL); /* warm */
            int iters = 24;
            spel_ckpt_load(t, cp);
            t0 = spel_now_ms();
            for (int it = 0; it < iters; it++) {
                spel_ckpt_load(t, cp);
                spel_forward(t, toks, M, NULL);
            }
            double ms = (spel_now_ms() - t0) / iters;
            printf("  M=%d %-6s: %6.2f ms  v=%.2fx\n", M,
                   fi ? "fast" : "strict", ms, ms / ar_ms);
        }
    }
    spel_use_accelerate = 0;
    spel_ckpt_free(cp);
    spel_model_free(t);
}

static void prepare_pair(spel_model *target, spel_model *draft,
                         const int *prompt, int n, int warm, int *warm_out) {
    spel_model_reset(target);
    spel_model_reset(draft);
    spel_use_accelerate = 0;
    prefill_chunks(target, prompt, n);
    prefill_chunks(draft, prompt, n);
    for (int i = 0; i < warm; i++) {
        int tok = spel_argmax(target->last_logits, target->cfg.vocab);
        warm_out[i] = tok;
        spel_forward(target, &tok, 1, NULL);
        spel_forward(draft, &tok, 1, NULL);
    }
}

static void bench_decode(void) {
    printf("[B3] decode throughput (12 layers, d=384, 9 GDN + 3 attn)\n");
    int K = 16;
    const char *ks = getenv("SPEL_BENCH_K");
    if (ks) {
        int v = atoi(ks);
        if (v >= 1 && v <= SPEL_MAX_BATCH) K = v;
    }
    spel_cfg c;
    spel_cfg_init(&c, 384, 12, 6, 1024, 128, 512, 4, K);
    spel_model *target = spel_model_new(&c, 31337);

    int map[4] = {0, 3, 6, 9}; /* 3 recurrent + 1 attention layer */
    double tq = spel_now_ms();
    spel_model *draft = spel_model_new_draft_subset(target, map, 4);
    tq = spel_now_ms() - tq;
    if (!draft) { printf("  draft alloc failed\n"); spel_model_free(target); return; }
    printf("  draft: 4 of 12 layers, MLP-only ternary (2-bit), built in %.1f ms\n", tq);

    uint64_t rng = 8;
    int prompt[64];
    fill_tokens(prompt, 64, c.vocab, &rng);
    const int gen = 64, warm_n = 8;
    int warm[8];
    int *out = (int *)malloc((size_t)gen * sizeof(int));
    double ar_ms, dr_ms, sp_ms, gov_ms, or_ms;

    prepare_pair(target, draft, prompt, 64, warm_n, warm);
    spel_ckpt *cp_t = spel_ckpt_new(target);
    spel_ckpt *cp_d = spel_ckpt_new(draft);

    /* AR target */
    double t0 = spel_now_ms();
    for (int i = 0; i < gen; i++) {
        int t = spel_argmax(target->last_logits, c.vocab);
        spel_forward(target, &t, 1, NULL);
    }
    ar_ms = spel_now_ms() - t0;

    /* draft-only (same prefix, then its own greedy stream) */
    spel_model_reset(draft);
    prefill_chunks(draft, prompt, 64);
    for (int i = 0; i < warm_n; i++) spel_forward(draft, &warm[i], 1, NULL);
    t0 = spel_now_ms();
    for (int i = 0; i < gen; i++) {
        int t = spel_argmax(draft->last_logits, c.vocab);
        spel_forward(draft, &t, 1, NULL);
    }
    dr_ms = spel_now_ms() - t0;

    /* Fast verification cost for the K+1 batch. */
    spel_ckpt_load(target, cp_t);
    int vtoks[17] = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2};
    spel_use_accelerate = 1;
    spel_forward(target, vtoks, K + 1, NULL); /* warm */
    int viters = 24;
    spel_ckpt_load(target, cp_t);
    t0 = spel_now_ms();
    for (int i = 0; i < viters; i++) {
        spel_ckpt_load(target, cp_t);
        spel_forward(target, vtoks, K + 1, NULL);
    }
    double v_ms = (spel_now_ms() - t0) / viters;

    /* Raw speculative decode, governor off, fast verification. */
    spel_ckpt_load(target, cp_t);
    spel_ckpt_load(draft, cp_d);
    spel_spec_cfg sc; spel_spec_cfg_default(&sc, K);
    sc.governor = 0;
    spel_spec_stats st, st_gov, st_or;
    spel_spec_generate(target, draft, warm_n, out, &sc, &st); /* warm caches */
    spel_ckpt_load(target, cp_t);
    spel_ckpt_load(draft, cp_d);
    t0 = spel_now_ms();
    spel_spec_generate(target, draft, gen, out, &sc, &st);
    sp_ms = spel_now_ms() - t0;

    /* Governed speculative decode. */
    sc.governor = 1;
    sc.alpha_threshold = 0.30;
    sc.window = 2;
    spel_ckpt_load(target, cp_t);
    spel_ckpt_load(draft, cp_d);
    spel_spec_generate(target, draft, warm_n, out, &sc, &st_gov); /* warm */
    spel_ckpt_load(target, cp_t);
    spel_ckpt_load(draft, cp_d);
    t0 = spel_now_ms();
    spel_spec_generate(target, draft, gen, out, &sc, &st_gov);
    gov_ms = spel_now_ms() - t0;

    /* Oracle: full FP32 copy of the target as draft. This is the runtime-cost
     * upper bound; it still pays one full target token per proposal. */
    spel_model *oracle = spel_model_new(&c, 31337);
    prepare_pair(target, oracle, prompt, 64, warm_n, warm);
    spel_ckpt_save(cp_t, target);
    spel_ckpt *cp_o = spel_ckpt_new(oracle);
    spel_use_accelerate = 1; /* oracle spec must use the fast verification path */
    sc.governor = 0;
    spel_spec_generate(target, oracle, warm_n, out, &sc, &st_or); /* warm */
    spel_ckpt_load(target, cp_t);
    spel_ckpt_load(oracle, cp_o);
    t0 = spel_now_ms();
    spel_spec_generate(target, oracle, gen, out, &sc, &st_or);
    or_ms = spel_now_ms() - t0;

    double ar_tps = gen / (ar_ms / 1000.0);
    double c_draft = dr_ms / ar_ms;
    double v = v_ms / (ar_ms / gen);
    printf("  AR target    : %7.1f tok/s  (%.2f ms/token)\n", ar_tps, ar_ms / gen);
    printf("  draft 4L only: %7.1f tok/s  (%.2f ms/token, c=%.2fx target)\n",
           gen / (dr_ms / 1000.0), dr_ms / gen, c_draft);
    printf("  verify K+1=%d : %6.2f ms  v=%.2fx AR token  (fast mode)\n",
           K + 1, v_ms, v);
    printf("  spec K=%d raw : %7.1f tok/s  (%.2fx AR)  alpha=%.3f  (%.2f tok/round)\n",
           K, gen / (sp_ms / 1000.0), (gen / (sp_ms / 1000.0)) / ar_tps,
           st.alpha, st.tokens_per_round);
    printf("  spec K=%d gov : %7.1f tok/s  (%.2fx AR)  alpha=%.3f  trips=%ld AR-rounds=%ld\n",
           K, gen / (gov_ms / 1000.0), (gen / (gov_ms / 1000.0)) / ar_tps,
           st_gov.alpha, st_gov.governor_trips, st_gov.ar_rounds);
    printf("  spec K=%d orac: %7.1f tok/s  (%.2fx AR)  alpha=%.3f  (%.2f tok/round)\n",
           K, gen / (or_ms / 1000.0), (gen / (or_ms / 1000.0)) / ar_tps,
           st_or.alpha, st_or.tokens_per_round);

    double proj75 = (1.0 + 0.75 * K) / ((K + 1) * c_draft + v);
    double proj80 = (1.0 + 0.80 * K) / ((K + 1) * c_draft + v);
    printf("  projected @ alpha=.75: %.2fx   @ alpha=.80: %.2fx  "
           "(K=%d, c=%.2f, v=%.2f)\n", proj75, proj80, K, c_draft, v);
    printf("  raw rounds=%ld proposed=%ld accepted=%ld\n",
           st.rounds, st.proposed, st.accepted);

    /* O(1) rollback vs KV memcpy */
    size_t kvb = spel_kv_bytes(&c);
    float *a = (float *)malloc(kvb), *bb = (float *)malloc(kvb);
    memset(a, 1, kvb); memset(bb, 2, kvb);
    t0 = spel_now_ms();
    for (int i = 0; i < 200; i++) memcpy(bb, a, kvb);
    volatile float checksum = 0.0f;
    for (size_t i = 0; i < kvb / sizeof(float); i += 1024) checksum += bb[i];
    double copy_us = (spel_now_ms() - t0) * 1000.0 / 200.0;
    volatile int ptr = 0;
    t0 = spel_now_ms();
    for (int i = 0; i < 10000000; i++) ptr = 64 - (i & 3);
    double reset_ns = (spel_now_ms() - t0) * 1e6 / 10000000.0;
    printf("  rollback     : pos reset ~%.1f ns, KV copy %.1f us (%.2f MiB), %.0fx\n",
           reset_ns, copy_us, (double)kvb / 1048576.0,
           (copy_us * 1000.0) / (reset_ns > 0 ? reset_ns : 1));
    if (checksum < -1e30f || ptr < -1) printf(" ");
    printf("  (commit moves a pointer; accepted KV stays in place)\n");

    spel_use_accelerate = 0;
    free(a); free(bb); free(out);
    spel_ckpt_free(cp_o);
    spel_model_free(oracle);
    spel_model_free(draft);
    spel_model_free(target);
    spel_ckpt_free(cp_t);
    spel_ckpt_free(cp_d);
}

static void run_bench(void) {
    printf("== SPEL benchmarks ==\n");
    bench_ternary();
    printf("\n");
    bench_verify();
    printf("\n");
    bench_decode();
}

int main(int argc, char **argv) {
    const char *cmd = argc > 1 ? argv[1] : "demo";
    if (!strcmp(cmd, "selftest")) {
        printf("== SPEL selftest ==\n");
        test_ternary();
        test_batch_equiv();
        test_spec();
        test_notebank(0);
        test_fast_mode();
        printf("\n%s (%d failures)\n", failures ? "FAILED" : "PASS", failures);
        return failures ? 1 : 0;
    } else if (!strcmp(cmd, "bench")) {
        run_bench();
        return 0;
    } else if (!strcmp(cmd, "demo")) {
        printf("== SPEL demo: NoteBank + SubTernary on Apple M5 ==\n");
        test_notebank(1);
        printf("\n");
        spec_result r;
        run_spec_case(384, 12, 1024, 128, 512, 48, 48, 2026, 4, 4, 0, 0.0, &r);
        printf("  48-token spec run: %d/%d tokens match AR, alpha=%.3f, %.2f tok/round\n",
               r.matched, r.n, r.st.alpha, r.st.tokens_per_round);
        return failures ? 1 : 0;
    } else {
        printf("usage: %s [selftest|bench|demo]\n", argv[0]);
        return 2;
    }
}
