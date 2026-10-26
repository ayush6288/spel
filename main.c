/* main.c — selftests, kernel/decoder benchmarks, NoteBank demo. */
#include "spel.h"
#include <arm_neon.h>
#include <time.h>
#include <errno.h>
#include <limits.h>
#include <sys/stat.h>

static int failures = 0;
#define CHECK(cond, ...) do { \
    if (!(cond)) { failures++; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } \
} while (0)

static void fill_tokens(int *t, int n, int vocab, uint64_t *rng) {
    for (int i = 0; i < n; i++) t[i] = (int)(spel_rng_next(rng) % (uint64_t)vocab);
}

static float frand(uint64_t *rng) { return (float)(spel_rng_double(rng) * 2.0 - 1.0); }

static uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

static int file_exists(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f) fclose(f);
    return f != NULL;
}

static const char *weights_path(void) {
    const char *p = getenv("SPEL_WEIGHTS");
    if (p && p[0]) return p;
    if (file_exists("weights/spel.bin")) return "weights/spel.bin";
    return NULL;
}

/* Runtime dimension config. Defaults are the historical 384/12/1024 model
 * with the adopted K=4 speculative depth. Environment variables and
 * --config key=value,... overrides; the C build remains deterministic when
 * none are set. */
typedef struct {
    int d_model, n_layers, n_heads, head_dim, d_ff, vocab;
    int max_seq, attn_period, spec_k;
    int draft_map[4];
} cfg_fields;

static void cfg_fields_default(cfg_fields *f) {
    f->d_model = 384;
    f->n_layers = 12;
    f->n_heads = 6;
    f->head_dim = 0;
    f->d_ff = 1024;
    f->vocab = 128;
    f->max_seq = 512;
    f->attn_period = 4;
    f->spec_k = SPEL_DEFAULT_SPEC_K;
    int map[4] = {0, 3, 6, 9};
    memcpy(f->draft_map, map, sizeof(map));
}

static int parse_int_str(const char *s, int *out) {
    if (!s || !s[0]) return -1;
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno || end == s || *end || v < INT_MIN || v > INT_MAX) return -1;
    *out = (int)v;
    return 0;
}

static int parse_int_list(const char *s, int *out, int n) {
    char buf[256];
    if (!s || strlen(s) >= sizeof(buf)) return -1;
    memcpy(buf, s, strlen(s) + 1);
    int count = 0;
    char *save = NULL;
    for (char *tok = strtok_r(buf, ":+, ", &save); tok;
         tok = strtok_r(NULL, ":+, ", &save)) {
        if (count >= n || parse_int_str(tok, &out[count])) return -1;
        count++;
    }
    return count == n ? 0 : -1;
}

static int env_int(const char *name, int *out) {
    const char *s = getenv(name);
    if (!s || !s[0]) return 0;
    return parse_int_str(s, out) ? -1 : 1;
}

static int apply_kv(cfg_fields *f, char *kv) {
    char *eq = strchr(kv, '=');
    if (!eq) return -1;
    *eq = '\0';
    const char *key = kv, *val = eq + 1;
    int *field = NULL;
    if (!strcmp(key, "d_model")) field = &f->d_model;
    else if (!strcmp(key, "n_layers")) field = &f->n_layers;
    else if (!strcmp(key, "n_heads")) field = &f->n_heads;
    else if (!strcmp(key, "head_dim")) field = &f->head_dim;
    else if (!strcmp(key, "d_ff")) field = &f->d_ff;
    else if (!strcmp(key, "vocab")) field = &f->vocab;
    else if (!strcmp(key, "max_seq")) field = &f->max_seq;
    else if (!strcmp(key, "attn_period")) field = &f->attn_period;
    else if (!strcmp(key, "spec_k")) field = &f->spec_k;
    else if (!strcmp(key, "draft_map")) return parse_int_list(val, f->draft_map, 4);
    else return -1;
    return parse_int_str(val, field);
}

static int apply_config_string(cfg_fields *f, const char *s) {
    char buf[512];
    if (!s || strlen(s) >= sizeof(buf)) return -1;
    memcpy(buf, s, strlen(s) + 1);
    char *save = NULL;
    for (char *tok = strtok_r(buf, ",", &save); tok;
         tok = strtok_r(NULL, ",", &save)) {
        if (apply_kv(f, tok)) return -1;
    }
    return 0;
}

static int apply_env_config(cfg_fields *f) {
    const char *all = getenv("SPEL_CONFIG");
    if (all && all[0] && apply_config_string(f, all)) return -1;
    if (env_int("SPEL_D_MODEL", &f->d_model) < 0) return -1;
    if (env_int("SPEL_N_LAYERS", &f->n_layers) < 0) return -1;
    if (env_int("SPEL_N_HEADS", &f->n_heads) < 0) return -1;
    if (env_int("SPEL_HEAD_DIM", &f->head_dim) < 0) return -1;
    if (env_int("SPEL_D_FF", &f->d_ff) < 0) return -1;
    if (env_int("SPEL_VOCAB", &f->vocab) < 0) return -1;
    if (env_int("SPEL_MAX_SEQ", &f->max_seq) < 0) return -1;
    if (env_int("SPEL_ATTN_PERIOD", &f->attn_period) < 0) return -1;
    if (env_int("SPEL_SPEC_K", &f->spec_k) < 0) return -1;
    const char *dm = getenv("SPEL_DRAFT_MAP");
    if (dm && dm[0] && parse_int_list(dm, f->draft_map, 4)) return -1;
    return 0;
}

static int build_run_cfg(int argc, char **argv, int first, spel_cfg *out) {
    cfg_fields f;
    cfg_fields_default(&f);
    if (apply_env_config(&f)) {
        printf("invalid configuration in environment\n");
        return -1;
    }
    for (int i = first; i < argc; i++) {
        if (!strcmp(argv[i], "--config")) {
            if (++i >= argc || apply_config_string(&f, argv[i])) {
                printf("invalid --config string\n");
                return -1;
            }
        } else {
            printf("unknown option: %s\n", argv[i]);
            return -1;
        }
    }
    if (f.head_dim && (f.n_heads <= 0 || f.head_dim != f.d_model / f.n_heads)) {
        printf("invalid config: head_dim must equal d_model / n_heads\n");
        return -1;
    }
    if (spel_cfg_init(out, f.d_model, f.n_layers, f.n_heads, f.d_ff,
                     f.vocab, f.max_seq, f.attn_period, f.spec_k)) {
        printf("invalid config: dimensions exceed C limits or are inconsistent\n");
        return -1;
    }
    for (int i = 0; i < 4; i++) {
        if (f.draft_map[i] < 0 || f.draft_map[i] >= out->n_layers) {
            printf("invalid config: draft_map[%d]=%d outside 0..%d\n",
                   i, f.draft_map[i], out->n_layers - 1);
            return -1;
        }
        out->draft_map[i] = f.draft_map[i];
    }
    return 0;
}

static void print_cfg(const spel_cfg *c) {
    printf("  config: d_model=%d n_layers=%d n_heads=%d head_dim=%d d_ff=%d "
           "vocab=%d max_seq=%d attn_period=%d spec_k=%d "
           "draft_map=[%d,%d,%d,%d]\n",
           c->d_model, c->n_layers, c->n_heads, c->head_dim, c->d_ff,
           c->vocab, c->max_seq, c->attn_period, c->spec_k,
           c->draft_map[0], c->draft_map[1], c->draft_map[2], c->draft_map[3]);
}

static void print_weight_info(const char *path, const spel_model *target,
                              const spel_model *draft) {
    unsigned long long tb = (unsigned long long)target->n_params * sizeof(float);
    printf("  target params: %zu  FP32: %llu bytes (%.2f MiB)\n",
           target->n_params, tb, (double)tb / 1048576.0);
    if (draft) {
        unsigned long long db = (unsigned long long)draft->n_params * sizeof(float);
        printf("  draft params: %zu  FP32: %llu bytes (%.2f MiB)\n",
               draft->n_params, db, (double)db / 1048576.0);
    }
    struct stat st;
    if (stat(path, &st) == 0)
        printf("  file: %lld bytes (%.2f MiB)\n",
               (long long)st.st_size, (double)st.st_size / 1048576.0);
}

/* Returns 1 if loaded, 0 if no file, -1 on failure. */
static int load_weights_pair_if_available(spel_model *target, spel_model *draft) {
    const char *p = weights_path();
    if (!p) return 0;
    if (spel_load_weights_pair(target, draft, p)) {
        printf("[weights] failed to load %s (config or magic mismatch)\n", p);
        return -1;
    }
    printf("[weights] loaded %s\n", p);
    return 1;
}

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

/* -------------------------------------------------------- forced bench */

static void test_forced_acceptance(void) {
    printf("[6] forced-acceptance benchmark path (not real decoding)\n");
    spel_cfg c;
    spel_cfg_init(&c, 128, 8, 4, 256, 64, 256, 4, 8);
    spel_model *target = spel_model_new(&c, 2026);
    int map[4] = {0, 2, 4, 6};
    spel_model *draft = spel_model_new_draft_subset(target, map, 4);
    if (!draft) { CHECK(0, "draft alloc failed"); spel_model_free(target); return; }

    uint64_t rng = 31;
    int prompt[16];
    fill_tokens(prompt, 16, c.vocab, &rng);
    spel_model_reset(target); spel_model_reset(draft);
    spel_forward(target, prompt, 16, NULL);
    spel_forward(draft, prompt, 16, NULL);
    const int start = target->pos;
    spel_ckpt *cp_t = spel_ckpt_new(target);
    spel_ckpt *cp_d = spel_ckpt_new(draft);

    const int gen = 32;
    int out[32];
    const double alphas[2] = {0.75, 1.00};
    for (int a = 0; a < 2; a++) {
        spel_ckpt_load(target, cp_t);
        spel_ckpt_load(draft, cp_d);
        spel_spec_cfg sc; spel_spec_cfg_default(&sc, 8);
        sc.governor = 0;
        sc.force_alpha = alphas[a];
        spel_spec_stats st;
        spel_spec_generate(target, draft, gen, out, &sc, &st);
        CHECK(st.emitted == gen, "alpha=%.2f emitted=%ld/%d", alphas[a], st.emitted, gen);
        CHECK(fabs(st.alpha - alphas[a]) < 1e-9,
              "alpha=%.2f stats alpha=%.3f", alphas[a], st.alpha);
        CHECK(target->pos == start + gen, "alpha=%.2f target pos=%d expected=%d",
              alphas[a], target->pos, start + gen);
        CHECK(draft->pos == target->pos, "alpha=%.2f draft pos=%d target pos=%d",
              alphas[a], draft->pos, target->pos);
        int bad = 0;
        for (int i = 0; i < gen; i++) if (out[i] < 0 || out[i] >= c.vocab) bad++;
        CHECK(bad == 0, "alpha=%.2f %d invalid output tokens", alphas[a], bad);
        printf("  alpha=%.2f: emitted=%ld accepted=%ld/%ld alpha=%.3f %.2f tok/round\n",
               alphas[a], st.emitted, st.accepted, st.proposed, st.alpha,
               st.tokens_per_round);
    }

    spel_ckpt_free(cp_t);
    spel_ckpt_free(cp_d);
    spel_model_free(draft);
    spel_model_free(target);
}


/* ------------------------------------------------------ weight round-trip */

static int layers_equal(const spel_model *a, const spel_model *b) {
    const spel_cfg *c = &a->cfg;
    if (c->n_layers != b->cfg.n_layers) return 0;
    for (int l = 0; l < c->n_layers; l++) {
        const spel_layer *x = &a->layer[l], *y = &b->layer[l];
        if (memcmp(x->n1, y->n1, (size_t)c->d_model * sizeof(float))) return 0;
        if (memcmp(x->n2, y->n2, (size_t)c->d_model * sizeof(float))) return 0;
        if (memcmp(x->wqkv, y->wqkv, (size_t)3 * c->d_model * c->d_model * sizeof(float))) return 0;
        if (memcmp(x->wo, y->wo, (size_t)c->d_model * c->d_model * sizeof(float))) return 0;
        if (memcmp(x->w1, y->w1, (size_t)c->d_model * c->d_ff * sizeof(float))) return 0;
        if (memcmp(x->w3, y->w3, (size_t)c->d_model * c->d_ff * sizeof(float))) return 0;
        if (memcmp(x->w2, y->w2, (size_t)c->d_ff * c->d_model * sizeof(float))) return 0;
        if (c->is_linear[l]) {
            if (memcmp(x->conv, y->conv, (size_t)c->conv_k * c->d_model * sizeof(float))) return 0;
            if (memcmp(x->wa, y->wa, (size_t)c->n_heads * sizeof(float))) return 0;
            if (memcmp(x->wb, y->wb, (size_t)c->n_heads * sizeof(float))) return 0;
        }
    }
    return 1;
}

static void test_weights_roundtrip(void) {
    printf("[7] SPELW001 save/load round-trip + SPELV001 golden verify\n");
    spel_cfg c;
    spel_cfg_init(&c, 128, 8, 4, 256, 128, 512, 4, 4);
    spel_model *t0 = spel_model_new(&c, 777);
    int map[4] = {0, 3, 5, 7};
    spel_model *d0 = spel_model_new_draft_subset(t0, map, 4);
    spel_model *t1 = spel_model_new(&c, 888);
    spel_model *d1 = spel_model_new_draft_subset(t1, map, 4);
    if (!t0 || !d0 || !t1 || !d1) {
        CHECK(0, "weight test allocation failed");
        spel_model_free(t0); spel_model_free(d0);
        spel_model_free(t1); spel_model_free(d1);
        return;
    }
    const char *tmp = getenv("TMPDIR");
    if (!tmp || !tmp[0]) tmp = ".";
    char path[512];
    snprintf(path, sizeof(path), "%s/spel_weights_selftest.bin", tmp);
    spel_model_attach_draft(t0, d0);
    spel_model_attach_draft(t1, d1);
    int saved = spel_save_weights(t0, path);
    CHECK(saved == 0, "save failed (%d) at %s", saved, path);
    if (saved == 0) {
        int loaded = spel_load_weights(t1, path);
        CHECK(loaded == 0, "load failed (%d)", loaded);
        if (loaded == 0) {
            CHECK(memcmp(t0->emb, t1->emb, (size_t)c.vocab * c.d_model * sizeof(float)) == 0,
                  "target embedding differs");
            CHECK(memcmp(t0->normf, t1->normf, (size_t)c.d_model * sizeof(float)) == 0,
                  "target final norm differs");
            CHECK(memcmp(t0->wout, t1->wout, (size_t)c.vocab * c.d_model * sizeof(float)) == 0,
                  "target LM head differs");
            CHECK(layers_equal(t0, t1), "target layer weights differ");
            CHECK(layers_equal(d0, d1), "draft layer weights differ");
            int v = spel_verify_weights(t1, d1, path);
            CHECK(v == 0, "golden verifier failed (%d)", v);
        }
    }
    remove(path);
    spel_model_free(t0); spel_model_free(d0);
    spel_model_free(t1); spel_model_free(d1);
}

static void test_draft_pool(void) {
    printf("[8] draft GEMV pool: threads=1 vs threads=4 bit-exact\n");
    spel_cfg c;
    spel_cfg_init(&c, 384, 12, 6, 1024, 128, 64, 4, 16);
    spel_model *target = spel_model_new(&c, 24680);
    int map[4] = {0, 3, 6, 9};
    spel_model *draft = spel_model_new_draft_subset(target, map, 4);
    if (!target || !draft) {
        CHECK(0, "pool test allocation failed");
        spel_model_free(draft); spel_model_free(target);
        return;
    }
    int n = SPEL_GOLDEN_LEN;
    float *serial = (float *)malloc((size_t)n * c.vocab * sizeof(float));
    float *pooled = (float *)malloc((size_t)n * c.vocab * sizeof(float));
    if (!serial || !pooled) {
        CHECK(0, "pool test logits allocation failed");
        free(serial); free(pooled);
        spel_model_free(draft); spel_model_free(target);
        return;
    }

    CHECK(spel_model_threads(draft) >= 1, "draft thread count invalid");
    CHECK(spel_model_set_threads(draft, 1) == 0, "set threads=1 failed");
    CHECK(spel_model_threads(draft) == 1, "draft did not switch to serial");
    spel_model_reset(draft);
    for (int i = 0; i < n; i++) {
        int tok = (i * 29 + 7) % c.vocab;
        spel_forward(draft, &tok, 1, NULL);
        memcpy(serial + (size_t)i * c.vocab, draft->last_logits,
               (size_t)c.vocab * sizeof(float));
    }

    CHECK(spel_model_set_threads(draft, 4) == 0, "set threads=4 failed");
    CHECK(spel_model_threads(draft) == 4, "draft thread count not 4");
    spel_model_reset(draft);
    for (int i = 0; i < n; i++) {
        int tok = (i * 29 + 7) % c.vocab;
        spel_forward(draft, &tok, 1, NULL);
        memcpy(pooled + (size_t)i * c.vocab, draft->last_logits,
               (size_t)c.vocab * sizeof(float));
    }

    int diff = 0;
    for (int i = 0; i < n * c.vocab; i++)
        if (memcmp(&serial[i], &pooled[i], sizeof(float))) diff++;
    printf("  draft logits: %d x %d floats, %d differ (threads 1 vs 4)\n",
           n, c.vocab, diff);
    CHECK(diff == 0, "pooled draft logits differ from serial");

    free(serial); free(pooled);
    spel_model_free(draft); spel_model_free(target);
}

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
    const char *wpath = with_spec ? weights_path() : NULL;
    spel_cfg c;
    if (wpath)
        spel_cfg_init(&c, 384, 12, 6, 1024, 128, 512, 4, 16);
    else
        spel_cfg_init(&c, 128, 8, 4, 256, 128, 512, 4, 4);
    spel_model *m = spel_model_new(&c, 2024);
    spel_model *draft = NULL;
    if (with_spec) {
        int map[4] = {0, 3, 5, 7};
        if (wpath) { map[0] = 0; map[1] = 3; map[2] = 6; map[3] = 9; }
        draft = spel_model_new_draft_subset(m, map, 4);
        if (draft && wpath) {
            if (spel_load_weights_pair(m, draft, wpath)) {
                printf("[weights] failed to load %s\n", wpath);
                spel_model_free(draft); spel_model_free(m);
                return;
            }
            printf("[weights] loaded %s\n", wpath);
        }
    }
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
    spel_model_free(draft);
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

static void bench_verify(const spel_cfg *base) {
    spel_cfg c = *base;
    printf("[B2] verification cost (d=%d, %d layers, %d GDN + %d attn)\n",
           c.d_model, c.n_layers, c.n_linear, c.n_attn);
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

static void bench_decode(const spel_cfg *base) {
    spel_cfg c = *base;
    int K = c.spec_k;
    const char *ks = getenv("SPEL_BENCH_K");
    if (ks) {
        int v = atoi(ks);
        if (v >= 1 && v <= SPEL_MAX_BATCH) K = v;
    }
    c.spec_k = K;
    printf("[B3] decode throughput (%d layers, d=%d, %d GDN + %d attn, K=%d)\n",
           c.n_layers, c.d_model, c.n_linear, c.n_attn, K);
    spel_model *target = spel_model_new(&c, 31337);

    int map[4];
    memcpy(map, c.draft_map, sizeof(map));
    double tq = spel_now_ms();
    spel_model *draft = spel_model_new_draft_subset(target, map, 4);
    tq = spel_now_ms() - tq;
    if (!draft) { printf("  draft alloc failed\n"); spel_model_free(target); return; }
    printf("  draft: 4 of %d layers, MLP-only ternary (2-bit), built in %.1f ms\n",
           c.n_layers, tq);
    if (load_weights_pair_if_available(target, draft) < 0) {
        spel_model_free(draft); spel_model_free(target); return;
    }
    int weights_loaded = weights_path() != NULL;

    uint64_t rng = 8;
    int prompt[64];
    fill_tokens(prompt, 64, c.vocab, &rng);
    const int gen = 64, warm_n = 8;
    int warm[8];
    int *out = (int *)malloc((size_t)gen * sizeof(int));
    double ar_ms, dr_ms[2], sp_ms[2], gov_ms[2];
    const int thread_counts[2] = {1, 4};

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

    /* Fast verification cost for the K+1 batch. The target path is unchanged
     * by the draft pool, so this v applies to both thread settings. */
    spel_ckpt_load(target, cp_t);
    int vtoks[SPEL_MAX_BATCH + 1];
    const int vseed[17] = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2};
    for (int i = 0; i < K + 1; i++) vtoks[i] = vseed[i];
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

    spel_spec_stats st[2], st_gov[2], st_or;
    spel_spec_cfg sc; spel_spec_cfg_default(&sc, K);
    for (int ti = 0; ti < 2; ti++) {
        int nt = thread_counts[ti];
        if (spel_model_set_threads(draft, nt)) {
            printf("  draft threads=%d unavailable\n", nt);
            continue;
        }

        /* draft-only (same prefix, then its own greedy stream) */
        spel_model_reset(draft);
        prefill_chunks(draft, prompt, 64);
        for (int i = 0; i < warm_n; i++) spel_forward(draft, &warm[i], 1, NULL);
        t0 = spel_now_ms();
        for (int i = 0; i < gen; i++) {
            int t = spel_argmax(draft->last_logits, c.vocab);
            spel_forward(draft, &t, 1, NULL);
        }
        dr_ms[ti] = spel_now_ms() - t0;

        /* Raw speculative decode, governor off, fast verification. */
        spel_ckpt_load(target, cp_t);
        spel_ckpt_load(draft, cp_d);
        sc.governor = 0;
        spel_spec_generate(target, draft, warm_n, out, &sc, &st[ti]); /* warm */
        spel_ckpt_load(target, cp_t);
        spel_ckpt_load(draft, cp_d);
        t0 = spel_now_ms();
        spel_spec_generate(target, draft, gen, out, &sc, &st[ti]);
        sp_ms[ti] = spel_now_ms() - t0;

        /* Governed speculative decode. */
        sc.governor = 1;
        sc.alpha_threshold = 0.30;
        sc.window = 2;
        spel_ckpt_load(target, cp_t);
        spel_ckpt_load(draft, cp_d);
        spel_spec_generate(target, draft, warm_n, out, &sc, &st_gov[ti]); /* warm */
        spel_ckpt_load(target, cp_t);
        spel_ckpt_load(draft, cp_d);
        t0 = spel_now_ms();
        spel_spec_generate(target, draft, gen, out, &sc, &st_gov[ti]);
        gov_ms[ti] = spel_now_ms() - t0;
    }

    /* The forced oracle remains a threads=1 draft measurement, unchanged from
     * before Phase 2, so its systems-ceiling rows stay comparable. */
    if (spel_model_set_threads(draft, 1)) {
        printf("  failed to restore draft threads=1\n");
        spel_model_free(draft); spel_model_free(target);
        spel_ckpt_free(cp_t); spel_ckpt_free(cp_d); free(out);
        return;
    }

    /* Benchmark-only forced-acceptance oracle. This is the same 4-layer
     * q8/ternary draft as the raw/governed rows and the same M=17 Accelerate
     * verification batch. Every draft and target forward still executes; only
     * the accepted-prefix decision m is forced to alpha*K. It measures systems
     * throughput at a hypothetical acceptance rate, not untrained model quality. */
    sc.governor = 0;
    const double forced_alpha[2] = {0.75, 1.00};
    double forced_ms[2] = {0.0, 0.0};
    spel_spec_stats forced_st[2];
    for (int a = 0; a < 2; a++) {
        sc.force_alpha = forced_alpha[a];
        spel_ckpt_load(target, cp_t);
        spel_ckpt_load(draft, cp_d);
        spel_spec_generate(target, draft, warm_n, out, &sc, &forced_st[a]); /* warm */
        double reps[3];
        for (int r = 0; r < 3; r++) {
            spel_ckpt_load(target, cp_t);
            spel_ckpt_load(draft, cp_d);
            t0 = spel_now_ms();
            spel_spec_generate(target, draft, gen, out, &sc, &forced_st[a]);
            reps[r] = spel_now_ms() - t0;
        }
        if (reps[0] > reps[1]) { double x = reps[0]; reps[0] = reps[1]; reps[1] = x; }
        if (reps[1] > reps[2]) { double x = reps[1]; reps[1] = reps[2]; reps[2] = x; }
        if (reps[0] > reps[1]) { double x = reps[0]; reps[0] = reps[1]; reps[1] = x; }
        forced_ms[a] = reps[1]; /* median of 3 */
    }
    sc.force_alpha = -1.0;

    double ar_tps = gen / (ar_ms / 1000.0);
    double v = v_ms / (ar_ms / gen);
    printf("  AR target    : %7.1f tok/s  (%.2f ms/token)\n", ar_tps, ar_ms / gen);
    for (int ti = 0; ti < 2; ti++) {
        double c_draft = dr_ms[ti] / ar_ms;
        printf("  draft 4L t=%d : %7.1f tok/s  (%.2f ms/token, c=%.2fx target)\n",
               thread_counts[ti], gen / (dr_ms[ti] / 1000.0),
               dr_ms[ti] / gen, c_draft);
    }
    printf("  verify K+1=%d : %6.2f ms  v=%.2fx AR token  (fast mode, target path)\n",
           K + 1, v_ms, v);
    for (int ti = 0; ti < 2; ti++) {
        double c_draft = dr_ms[ti] / ar_ms;
        printf("  spec K=%d raw t=%d: %7.1f tok/s  (%.2fx AR)  "
               "alpha=%.3f  c=%.2f  v=%.2f  (%.2f tok/round)\n",
               K, thread_counts[ti], gen / (sp_ms[ti] / 1000.0),
               (gen / (sp_ms[ti] / 1000.0)) / ar_tps,
               st[ti].alpha, c_draft, v, st[ti].tokens_per_round);
    }
    for (int ti = 0; ti < 2; ti++) {
        double c_draft = dr_ms[ti] / ar_ms;
        printf("  spec K=%d gov t=%d: %7.1f tok/s  (%.2fx AR)  "
               "alpha=%.3f  c=%.2f  v=%.2f  trips=%ld AR-rounds=%ld\n",
               K, thread_counts[ti], gen / (gov_ms[ti] / 1000.0),
               (gen / (gov_ms[ti] / 1000.0)) / ar_tps,
               st_gov[ti].alpha, c_draft, v,
               st_gov[ti].governor_trips, st_gov[ti].ar_rounds);
    }
    printf("  spec oracle-4L (alpha=0.75): %7.1f tok/s  (%.2fx AR)  alpha=%.3f  (%.2f tok/round)\n",
           gen / (forced_ms[0] / 1000.0),
           (gen / (forced_ms[0] / 1000.0)) / ar_tps,
           forced_st[0].alpha, forced_st[0].tokens_per_round);
    printf("  spec oracle-4L (alpha=1.00): %7.1f tok/s  (%.2fx AR)  alpha=%.3f  (%.2f tok/round)\n",
           gen / (forced_ms[1] / 1000.0),
           (gen / (forced_ms[1] / 1000.0)) / ar_tps,
           forced_st[1].alpha, forced_st[1].tokens_per_round);

    for (int ti = 0; ti < 2; ti++) {
        double c_draft = dr_ms[ti] / ar_ms;
        double proj75 = (1.0 + 0.75 * K) / ((K + 1) * c_draft + v);
        double proj100 = (1.0 + 1.00 * K) / ((K + 1) * c_draft + v);
        printf("  projected t=%d: a=.75 %.2fx  a=1.00 %.2fx  "
               "(K=%d, c=%.2f, v=%.2f)\n",
               thread_counts[ti], proj75, proj100, K, c_draft, v);
    }
    for (int ti = 0; ti < 2; ti++)
        printf("  raw t=%d rounds=%ld proposed=%ld accepted=%ld\n",
               thread_counts[ti], st[ti].rounds, st[ti].proposed, st[ti].accepted);

    /* Reference: full FP32 copy of the target as draft. This is not the
     * cheap-draft system ceiling; it shows why a target-sized draft cannot
     * speculatively draft for itself even at 100% acceptance. */
    spel_model *oracle = spel_model_new(&c, 31337);
    if (!oracle) { printf("  oracle alloc failed\n"); }
    else {
        if (weights_loaded) {
            const char *wp = weights_path();
            if (!wp || spel_load_weights(oracle, wp)) {
                printf("  oracle weight load failed\n");
                spel_model_free(oracle);
                oracle = NULL;
            }
        }
    }
    if (oracle) {
        prepare_pair(target, oracle, prompt, 64, warm_n, warm);
        spel_ckpt_save(cp_t, target);
        spel_ckpt *cp_o = spel_ckpt_new(oracle);
        spel_use_accelerate = 1;
        sc.governor = 0;
        sc.force_alpha = -1.0;
        spel_spec_generate(target, oracle, warm_n, out, &sc, &st_or); /* warm */
        spel_ckpt_load(target, cp_t);
        if (cp_o) spel_ckpt_load(oracle, cp_o);
        t0 = spel_now_ms();
        spel_spec_generate(target, oracle, gen, out, &sc, &st_or);
        double or_ms = spel_now_ms() - t0;
        printf("  spec oracle-%dL (full copy): %7.1f tok/s  (%.2fx AR)  alpha=%.3f  (%.2f tok/round)\n",
               c.n_layers, gen / (or_ms / 1000.0),
               (gen / (or_ms / 1000.0)) / ar_tps,
               st_or.alpha, st_or.tokens_per_round);
        spel_ckpt_free(cp_o);
        spel_model_free(oracle);
    }

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
    spel_model_free(draft);
    spel_model_free(target);
    spel_ckpt_free(cp_t);
    spel_ckpt_free(cp_d);
}

static void run_bench(const spel_cfg *cfg) {
    printf("== SPEL benchmarks ==\n");
    print_cfg(cfg);
    bench_ternary();
    printf("\n");
    bench_verify(cfg);
    printf("\n");
    bench_decode(cfg);
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
        test_forced_acceptance();
        test_weights_roundtrip();
        test_draft_pool();
        printf("\n%s (%d failures)\n", failures ? "FAILED" : "PASS", failures);
        return failures ? 1 : 0;
    } else if (!strcmp(cmd, "bench")) {
        spel_cfg c;
        if (build_run_cfg(argc, argv, 2, &c)) return 2;
        run_bench(&c);
        return 0;
    } else if (!strcmp(cmd, "verify-weights")) {
        if (argc < 3) {
            printf("usage: %s verify-weights <path> [--config key=value,...]\n", argv[0]);
            return 2;
        }
        spel_cfg c;
        if (build_run_cfg(argc, argv, 3, &c)) return 2;
        print_cfg(&c);
        spel_model *target = spel_model_new(&c, 31337);
        int map[4];
        memcpy(map, c.draft_map, sizeof(map));
        spel_model *draft = spel_model_new_draft_subset(target, map, 4);
        if (!target || !draft) {
            printf("model alloc failed\n");
            spel_model_free(target); spel_model_free(draft);
            return 1;
        }
        print_weight_info(argv[2], target, draft);
        int v = spel_verify_weights(target, draft, argv[2]);
        spel_model_free(draft); spel_model_free(target);
        return v == 0 ? 0 : 1;
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
        printf("usage: %s [selftest|bench [--config k=v,...]|demo|verify-weights <path> [--config k=v,...]]\n", argv[0]);
        return 2;
    }
}
