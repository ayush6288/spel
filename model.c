/* model.c — deterministic hybrid model (GDN-style recurrence + RoPE attention),
 * batched verification, checkpoints and greedy speculative decoding.
 *
 * Correctness invariant: a forward over M tokens produces bit-identical
 * per-token logits to M successive single-token forwards from the same state.
 * That is what makes the transactional commit/rollback auditable.
 */
#include "spel.h"
#include <arm_neon.h>
#include <time.h>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
/* cblas_sgemm is marked deprecated in the macOS 13.3+ SDK; the replacement
 * ILP64 interface is not needed for these int-sized GEMMs. */
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

int spel_use_accelerate = 0;

/* ---------------------------------------------------------------- rng */

uint64_t spel_rng_next(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

double spel_rng_double(uint64_t *s) {
    return (double)(spel_rng_next(s) >> 11) * (1.0 / 9007199254740992.0);
}

float spel_rng_normal(uint64_t *s) {
    double u1 = spel_rng_double(s), u2 = spel_rng_double(s);
    if (u1 < 1e-12) u1 = 1e-12;
    return (float)(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2));
}

double spel_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* ---------------------------------------------------------------- cfg */

int spel_cfg_init(spel_cfg *c, int d, int layers, int heads, int dff,
                  int vocab, int max_seq, int attn_period, int K) {
    memset(c, 0, sizeof(*c));
    c->d_model = d;
    c->n_layers = layers;
    c->n_heads = heads;
    c->head_dim = d / heads;
    c->d_ff = dff;
    c->vocab = vocab;
    c->max_seq = max_seq;
    c->attn_period = attn_period;
    c->conv_k = 3;
    c->spec_k = K;
    c->n_linear = c->n_attn = 0;
    for (int i = 0; i < layers; i++) {
        int is_attn = ((i + 1) % attn_period == 0);
        c->is_linear[i] = !is_attn;
        c->lin_index[i] = is_attn ? -1 : c->n_linear++;
        c->attn_index[i] = is_attn ? c->n_attn++ : -1;
    }
    c->n_channels = c->n_linear * heads;
    return spel_cfg_validate(c);
}

int spel_cfg_validate(const spel_cfg *c) {
    if (c->d_model <= 0 || c->d_model > SPEL_MAX_D) return -1;
    if (c->d_model % c->n_heads) return -1;
    if (c->n_heads > SPEL_MAX_HEADS || c->n_layers > SPEL_MAX_LAYERS) return -1;
    if (c->d_ff > SPEL_MAX_DFF || c->vocab > SPEL_MAX_VOCAB) return -1;
    if (c->max_seq > SPEL_MAX_SEQ || c->max_seq < 8) return -1;
    if (c->spec_k < 1 || c->spec_k > SPEL_MAX_BATCH) return -1;
    if (c->attn_period < 2) return -1;
    if (c->conv_k != 3) return -1;
    return 0;
}

/* -------------------------------------------------------------- math */

static inline float dotf(const float *a, const float *b, int n) {
    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
        s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    float s = vaddvq_f32(vaddq_f32(s0, s1));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

/* out[M][N] = x[M][K] * W[N][K]^T.  W is streamed once for all rows.
 * Strict mode keeps the explicit dot-product order used by AR, which is what
 * makes batched verification bit-identical. Fast mode routes M>1 through
 * Accelerate/AMX and is token-equivalent, not bit-identical. */
static void matmul_rows(float *out, const float *x, const float *W,
                        int M, int N, int K) {
#if defined(__APPLE__)
    if (spel_use_accelerate && M > 1) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    M, N, K, 1.0f, x, K, W, K, 0.0f, out, N);
        return;
    }
#endif
    for (int n = 0; n < N; n++) {
        const float *w = W + (size_t)n * K;
        for (int r = 0; r < M; r++)
            out[(size_t)r * N + n] = dotf(x + (size_t)r * K, w, K);
    }
}

static void rmsnorm_rows(const float *x, const float *g, float *y, int M, int d) {
    const float inv_d = 1.0f / (float)d;
    for (int r = 0; r < M; r++) {
        const float *xr = x + (size_t)r * d;
        float ss = 0.0f;
        for (int i = 0; i < d; i++) ss += xr[i] * xr[i];
        float rcp = 1.0f / sqrtf(ss * inv_d + 1e-5f);
        float *yr = y + (size_t)r * d;
        for (int i = 0; i < d; i++) yr[i] = xr[i] * rcp * g[i];
    }
}

static inline float silu(float x) { return x / (1.0f + expf(-x)); }
static inline float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

int spel_argmax(const float *v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

/* ------------------------------------------------------------- params */

static size_t param_count(const spel_cfg *c) {
    size_t n = (size_t)c->vocab * c->d_model * 2 + c->d_model; /* emb,wout,normf */
    size_t per = (size_t)3 * c->d_model * c->d_model
               + (size_t)c->d_model * c->d_model
               + (size_t)3 * c->d_model * c->d_ff
               + (size_t)2 * c->d_model;
    n += per * c->n_layers;
    n += (size_t)c->n_linear * (c->conv_k * c->d_model + 2 * c->n_heads);
    return n;
}

static float *palloc(float **p, size_t n) {
    float *r = *p;
    *p += n;
    return r;
}

static void fill_normal(float *p, size_t n, float sd, uint64_t *rng) {
    for (size_t i = 0; i < n; i++) p[i] = spel_rng_normal(rng) * sd;
}

static void alloc_runtime(spel_model *m) {
    const spel_cfg *c = &m->cfg;
    int d = c->d_model;
    for (int l = 0; l < c->n_layers; l++) {
        if (!c->is_linear[l]) {
            m->kv_k[l] = (float *)calloc((size_t)c->max_seq * d, sizeof(float));
            m->kv_v[l] = (float *)calloc((size_t)c->max_seq * d, sizeof(float));
        } else {
            size_t sd = (size_t)(c->spec_k + 1) * c->n_heads * c->head_dim * c->head_dim;
            m->S[l] = (float *)calloc(sd, sizeof(float));
            m->xh[l] = (float *)calloc((size_t)SPEL_XH_RING * d, sizeof(float));
        }
    }
    size_t bf = (size_t)(c->spec_k + 1) * (8 * (size_t)d + 3 * (size_t)c->d_ff) + 64;
    m->buf_floats = bf;
    m->buf = (float *)calloc(bf, sizeof(float));
    m->last_logits = (float *)calloc((size_t)c->vocab, sizeof(float));
}

spel_model *spel_model_new(const spel_cfg *cfg, uint64_t seed) {
    spel_model *m = (spel_model *)calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->cfg = *cfg;
    m->owns_params = 1;
    m->n_params = param_count(cfg);
    m->params = (float *)calloc(m->n_params, sizeof(float));
    if (!m->params) { free(m); return NULL; }
    uint64_t rng = seed ? seed : 0x123456789ABCDEFULL;
    float *p = m->params;
    const int d = cfg->d_model;
    m->emb     = palloc(&p, (size_t)cfg->vocab * d);
    m->wout    = palloc(&p, (size_t)cfg->vocab * d);
    m->normf   = palloc(&p, d);
    for (int l = 0; l < cfg->n_layers; l++) {
        spel_layer *L = &m->layer[l];
        L->wqkv = palloc(&p, (size_t)3 * d * d);
        L->wo   = palloc(&p, (size_t)d * d);
        L->w1   = palloc(&p, (size_t)d * cfg->d_ff);
        L->w3   = palloc(&p, (size_t)d * cfg->d_ff);
        L->w2   = palloc(&p, (size_t)cfg->d_ff * d);
        L->n1   = palloc(&p, d);
        L->n2   = palloc(&p, d);
        if (cfg->is_linear[l]) {
            L->conv = palloc(&p, (size_t)cfg->conv_k * d);
            L->wa   = palloc(&p, (size_t)cfg->n_heads);
            L->wb   = palloc(&p, (size_t)cfg->n_heads);
        }
    }
    /* init */
    fill_normal(m->emb, (size_t)cfg->vocab * d, 1.0f, &rng);
    fill_normal(m->wout, (size_t)cfg->vocab * d, 1.0f / sqrtf((float)d), &rng);
    for (int i = 0; i < d; i++) m->normf[i] = 1.0f;
    for (int l = 0; l < cfg->n_layers; l++) {
        spel_layer *L = &m->layer[l];
        float s = 1.0f / sqrtf((float)d);
        fill_normal(L->wqkv, (size_t)3 * d * d, s, &rng);
        fill_normal(L->wo, (size_t)d * d, s, &rng);
        fill_normal(L->w1, (size_t)d * cfg->d_ff, s, &rng);
        fill_normal(L->w3, (size_t)d * cfg->d_ff, s, &rng);
        fill_normal(L->w2, (size_t)cfg->d_ff * d, 1.0f / sqrtf((float)cfg->d_ff), &rng);
        for (int i = 0; i < d; i++) { L->n1[i] = 1.0f; L->n2[i] = 1.0f; }
        if (cfg->is_linear[l]) {
            fill_normal(L->conv, (size_t)cfg->conv_k * d, 0.05f, &rng);
            for (int h = 0; h < cfg->n_heads; h++) {
                L->wa[h] = 2.0f + 0.1f * spel_rng_normal(&rng);
                L->wb[h] = 0.1f * spel_rng_normal(&rng);
            }
        }
    }
    /* rope */
    int half = cfg->head_dim / 2;
    m->rope_cos = (float *)malloc((size_t)cfg->max_seq * half * sizeof(float));
    m->rope_sin = (float *)malloc((size_t)cfg->max_seq * half * sizeof(float));
    for (int pos = 0; pos < cfg->max_seq; pos++) {
        for (int j = 0; j < half; j++) {
            float freq = powf(SPEL_ROPE_THETA, -2.0f * (float)j / (float)cfg->head_dim);
            float ang = (float)pos * freq;
            m->rope_cos[(size_t)pos * half + j] = cosf(ang);
            m->rope_sin[(size_t)pos * half + j] = sinf(ang);
        }
    }
    alloc_runtime(m);
    m->rng = rng;
    m->pos = 0;
    if (!m->buf || !m->last_logits) { spel_model_free(m); return NULL; }
    return m;
}

static size_t layer_tensor_count(const spel_cfg *c, int is_linear) {
    size_t n = (size_t)3 * c->d_model * c->d_model   /* wqkv */
             + (size_t)c->d_model * c->d_model       /* wo   */
             + (size_t)3 * c->d_model * c->d_ff      /* w1,w3,w2 */
             + (size_t)2 * c->d_model;               /* n1,n2 */
    if (is_linear)
        n += (size_t)c->conv_k * c->d_model + (size_t)2 * c->n_heads;
    return n;
}

static int draft_threads_from_env(void) {
    const char *s = getenv("SPEL_DRAFT_THREADS");
    if (!s || !s[0]) return 1;
    int n = atoi(s);
    if (n < 1 || n > SPEL_POOL_MAX_THREADS) return 1;
    return n;
}

spel_model *spel_model_new_draft_subset(const spel_model *t,
                                        const int *layer_map, int n_layers) {
    if (!t || !layer_map || n_layers <= 0 ||
        n_layers > t->cfg.n_layers || n_layers > SPEL_MAX_LAYERS)
        return NULL;
    spel_model *m = (spel_model *)calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->cfg = t->cfg;
    m->cfg.n_layers = n_layers;
    m->cfg.n_linear = m->cfg.n_attn = 0;
    size_t np = 0;
    for (int l = 0; l < n_layers; l++) {
        int src = layer_map[l];
        if (src < 0 || src >= t->cfg.n_layers) { free(m); return NULL; }
        int is_attn = !t->cfg.is_linear[src];
        m->cfg.is_linear[l] = !is_attn;
        m->cfg.lin_index[l] = is_attn ? -1 : m->cfg.n_linear++;
        m->cfg.attn_index[l] = is_attn ? m->cfg.n_attn++ : -1;
        np += layer_tensor_count(&t->cfg, !is_attn);
    }
    m->cfg.n_channels = m->cfg.n_linear * m->cfg.n_heads;

    m->params = (float *)calloc(np, sizeof(float));
    if (!m->params) { free(m); return NULL; }
    m->n_params = np;
    m->owns_layer_params = 1;
    m->emb = t->emb; m->wout = t->wout; m->normf = t->normf;
    m->rope_cos = t->rope_cos; m->rope_sin = t->rope_sin;
    m->is_ternary = 1;
    m->rng = t->rng ^ 0xA5A5A5A55A5A5A5AULL;
    m->n_draft_layers = n_layers <= 4 ? n_layers : 0;
    for (int l = 0; l < m->n_draft_layers; l++) m->source_map[l] = layer_map[l];

    const spel_cfg *c = &m->cfg;
    const int d = c->d_model;
    float *p = m->params;
    for (int l = 0; l < n_layers; l++) {
        int src = layer_map[l];
        const spel_layer *S = &t->layer[src];
        spel_layer *L = &m->layer[l];
        L->wqkv = palloc(&p, (size_t)3 * d * d);
        L->wo   = palloc(&p, (size_t)d * d);
        L->w1   = palloc(&p, (size_t)d * c->d_ff);
        L->w3   = palloc(&p, (size_t)d * c->d_ff);
        L->w2   = palloc(&p, (size_t)c->d_ff * d);
        L->n1   = palloc(&p, (size_t)d);
        L->n2   = palloc(&p, (size_t)d);
        memcpy(L->wqkv, S->wqkv, (size_t)3 * d * d * sizeof(float));
        memcpy(L->wo, S->wo, (size_t)d * d * sizeof(float));
        memcpy(L->w1, S->w1, (size_t)d * c->d_ff * sizeof(float));
        memcpy(L->w3, S->w3, (size_t)d * c->d_ff * sizeof(float));
        memcpy(L->w2, S->w2, (size_t)c->d_ff * d * sizeof(float));
        memcpy(L->n1, S->n1, (size_t)d * sizeof(float));
        memcpy(L->n2, S->n2, (size_t)d * sizeof(float));
        if (c->is_linear[l]) {
            L->conv = palloc(&p, (size_t)c->conv_k * d);
            L->wa   = palloc(&p, (size_t)c->n_heads);
            L->wb   = palloc(&p, (size_t)c->n_heads);
            memcpy(L->conv, S->conv, (size_t)c->conv_k * d * sizeof(float));
            memcpy(L->wa, S->wa, (size_t)c->n_heads * sizeof(float));
            memcpy(L->wb, S->wb, (size_t)c->n_heads * sizeof(float));
        }
        if (tern_mat_init(&L->t1, c->d_ff, d) ||
            tern_mat_init(&L->t3, c->d_ff, d) ||
            tern_mat_init(&L->t2, d, c->d_ff) ||
            q8_mat_init(&L->q_wqkv, 3 * d, d) ||
            q8_mat_init(&L->q_wo, d, d)) {
            spel_model_free(m);
            return NULL;
        }
        tern_quantize_mat(&L->t1, L->w1);
        tern_quantize_mat(&L->t3, L->w3);
        tern_quantize_mat(&L->t2, L->w2);
        q8_quantize_mat(&L->q_wqkv, L->wqkv);
        q8_quantize_mat(&L->q_wo, L->wo);
    }
    alloc_runtime(m);
    if (!m->buf || !m->last_logits) { spel_model_free(m); return NULL; }
    int nt = draft_threads_from_env();
    if (nt > 1) {
        m->pool = spel_pool_create(nt);
        if (!m->pool) { spel_model_free(m); return NULL; }
    }
    m->draft_threads = nt;
    m->pos = 0;
    return m;
}

spel_model *spel_model_new_draft(const spel_model *t) {
    int map[SPEL_MAX_LAYERS];
    for (int i = 0; i < t->cfg.n_layers; i++) map[i] = i;
    return spel_model_new_draft_subset(t, map, t->cfg.n_layers);
}

int spel_model_set_threads(spel_model *m, int nthreads) {
    if (!m || !m->is_ternary ||
        nthreads < 1 || nthreads > SPEL_POOL_MAX_THREADS) return -1;
    if (nthreads == m->draft_threads) return 0;
    spel_gemv_pool *p = nthreads > 1 ? spel_pool_create(nthreads) : NULL;
    if (nthreads > 1 && !p) return -1;
    spel_pool_destroy(m->pool);
    m->pool = p;
    m->draft_threads = nthreads;
    return 0;
}

int spel_model_threads(const spel_model *m) {
    return m ? m->draft_threads : 0;
}

void spel_model_free(spel_model *m) {
    if (!m) return;
    spel_pool_destroy(m->pool);
    m->pool = NULL;
    for (int l = 0; l < m->cfg.n_layers; l++) {
        free(m->kv_k[l]); free(m->kv_v[l]);
        free(m->S[l]); free(m->xh[l]);
        tern_mat_free(&m->layer[l].t1);
        tern_mat_free(&m->layer[l].t3);
        tern_mat_free(&m->layer[l].t2);
        q8_mat_free(&m->layer[l].q_wqkv);
        q8_mat_free(&m->layer[l].q_wo);
    }
    if (m->owns_params) {
        free(m->params);
        free(m->rope_cos); free(m->rope_sin);
    } else if (m->owns_layer_params) {
        free(m->params);
    }
    free(m->buf); free(m->last_logits);
    free(m->cap_q); free(m->cap_k);
    free(m->golden_target_logits); free(m->golden_draft_logits);
    free(m);
}

void spel_model_reset(spel_model *m) {
    const spel_cfg *c = &m->cfg;
    int d = c->d_model;
    for (int l = 0; l < c->n_layers; l++) {
        if (!c->is_linear[l]) {
            memset(m->kv_k[l], 0, (size_t)c->max_seq * d * sizeof(float));
            memset(m->kv_v[l], 0, (size_t)c->max_seq * d * sizeof(float));
        } else {
            size_t sd = (size_t)(c->spec_k + 1) * c->n_heads * c->head_dim * c->head_dim;
            memset(m->S[l], 0, sd * sizeof(float));
            memset(m->xh[l], 0, (size_t)SPEL_XH_RING * d * sizeof(float));
        }
    }
    m->pos = 0;
    memset(m->last_logits, 0, (size_t)c->vocab * sizeof(float));
}

int spel_capture_alloc(spel_model *m, int max_pos) {
    size_t n = (size_t)max_pos * m->cfg.n_channels * m->cfg.head_dim;
    free(m->cap_q); free(m->cap_k);
    m->cap_q = (float *)calloc(n, sizeof(float));
    m->cap_k = (float *)calloc(n, sizeof(float));
    m->cap_len = max_pos;
    if (!m->cap_q || !m->cap_k) return -1;
    return 0;
}

void spel_capture_free(spel_model *m) {
    free(m->cap_q); free(m->cap_k);
    m->cap_q = m->cap_k = NULL;
    m->cap_len = 0;
    m->cap_enabled = 0;
}

/* ---------------------------------------------------------- attention */

static void rope_apply(float *v, int n_heads, int dh, int pos,
                       const float *cs, const float *sn) {
    int half = dh / 2;
    for (int h = 0; h < n_heads; h++) {
        float *x = v + (size_t)h * dh;
        for (int j = 0; j < half; j++) {
            float c = cs[(size_t)pos * half + j];
            float s = sn[(size_t)pos * half + j];
            float a = x[2 * j], b = x[2 * j + 1];
            x[2 * j]     = a * c - b * s;
            x[2 * j + 1] = a * s + b * c;
        }
    }
}

static void attention_rows(spel_model *m, int l, float *qkv, int M,
                           float *out) {
    const spel_cfg *c = &m->cfg;
    const int d = c->d_model, nh = c->n_heads, dh = c->head_dim;
    const int base = m->pos;
    const float scale = 1.0f / sqrtf((float)dh);
    float *kc = m->kv_k[l], *vc = m->kv_v[l];
    float scores[SPEL_MAX_SEQ];

    for (int r = 0; r < M; r++) {
        int p = base + r;
        float *q = qkv + (size_t)r * 3 * d;
        float *k = q + d;
        float *v = k + d;
        rope_apply(q, nh, dh, p, m->rope_cos, m->rope_sin);
        rope_apply(k, nh, dh, p, m->rope_cos, m->rope_sin);
        memcpy(kc + (size_t)p * d, k, (size_t)d * sizeof(float));
        memcpy(vc + (size_t)p * d, v, (size_t)d * sizeof(float));
    }
    for (int r = 0; r < M; r++) {
        int p = base + r;
        float *q = qkv + (size_t)r * 3 * d;
        float *o = out + (size_t)r * d;
        for (int h = 0; h < nh; h++) {
            const float *qh = q + (size_t)h * dh;
            for (int j = 0; j <= p; j++)
                scores[j] = dotf(qh, kc + (size_t)j * d + (size_t)h * dh, dh) * scale;
            float mx = scores[0];
            for (int j = 1; j <= p; j++) if (scores[j] > mx) mx = scores[j];
            float sum = 0.0f;
            for (int j = 0; j <= p; j++) { scores[j] = expf(scores[j] - mx); sum += scores[j]; }
            float inv = 1.0f / sum;
            for (int j = 0; j <= p; j++) scores[j] *= inv;
            float *oh = o + (size_t)h * dh;
            for (int t = 0; t < dh; t++) oh[t] = 0.0f;
            for (int j = 0; j <= p; j++) {
                const float *vj = vc + (size_t)j * d + (size_t)h * dh;
                float w = scores[j];
                for (int t = 0; t < dh; t++) oh[t] += w * vj[t];
            }
        }
    }
}

/* ----------------------------------------------------- recurrent layer */

static void draft_q8_gemv(spel_model *m, const q8_mat *mat,
                          const float *x, float *y) {
    if (m->pool) spel_pool_q8_gemv(m->pool, mat, x, y);
    else q8_gemv(mat, x, y);
}

static void draft_tern_gemv(spel_model *m, const tern_mat *mat,
                            const float *x, float *y) {
    if (m->pool) spel_pool_tern_gemv(m->pool, mat, x, y);
    else tern_gemv(mat, x, y);
}

static void proj_qkv(spel_model *m, spel_layer *L, const float *x, float *y,
                     int M, int d) {
    if (m->is_ternary && L->q_wqkv.q) {
        for (int r = 0; r < M; r++)
            draft_q8_gemv(m, &L->q_wqkv, x + (size_t)r * d,
                          y + (size_t)r * 3 * d);
    } else {
        matmul_rows(y, x, L->wqkv, M, 3 * d, d);
    }
}

static void proj_wo(spel_model *m, spel_layer *L, const float *x, float *y,
                    int M, int d) {
    if (m->is_ternary && L->q_wo.q) {
        for (int r = 0; r < M; r++)
            draft_q8_gemv(m, &L->q_wo, x + (size_t)r * d,
                          y + (size_t)r * d);
    } else {
        matmul_rows(y, x, L->wo, M, d, d);
    }
}

static void linear_rows(spel_model *m, int l, const float *xn, int M,
                        float *qkv, float *out, float *cx) {
    const spel_cfg *c = &m->cfg;
    const int d = c->d_model, nh = c->n_heads, dh = c->head_dim;
    const int base = m->pos, R = c->spec_k + 1;
    spel_layer *L = &m->layer[l];
    float *ring = m->S[l];
    float *xh = m->xh[l];

    /* short causal conv over the layer input, history indexed by abs position */
    for (int r = 0; r < M; r++) {
        int p = base + r;
        const float *x = xn + (size_t)r * d;
        float *y = cx + (size_t)r * d;
        for (int i = 0; i < d; i++) {
            float acc = x[i];
            for (int t = 0; t < c->conv_k - 1; t++) {
                int hp = p - (c->conv_k - 1) + t;
                if (hp >= 0) {
                    const float *hx = xh + (size_t)(hp % SPEL_XH_RING) * d;
                    acc += L->conv[(size_t)t * d + i] * hx[i];
                }
            }
            y[i] = acc;
        }
        memcpy(xh + (size_t)(p % SPEL_XH_RING) * d, x, (size_t)d * sizeof(float));
    }

    proj_qkv(m, L, cx, qkv, M, d);
    for (int r = 0; r < M; r++) {
        int p = base + r;
        float *qr = qkv + (size_t)r * 3 * d;
        rope_apply(qr, nh, dh, p, m->rope_cos, m->rope_sin);
        rope_apply(qr + d, nh, dh, p, m->rope_cos, m->rope_sin);
        float *o = out + (size_t)r * d;
        const float *cr = cx + (size_t)r * d;
        const int src = p % R, dst = (p + 1) % R;
        float mean = 0.0f;
        for (int i = 0; i < d; i++) mean += cr[i];
        mean /= (float)d;
        float *Ssrc = ring + (size_t)src * nh * dh * dh;
        float *Sdst = ring + (size_t)dst * nh * dh * dh;
        for (int h = 0; h < nh; h++) {
            float a = sigmoidf_(L->wa[h] + 0.1f * mean);
            float beta = sigmoidf_(L->wb[h] + 0.1f * mean);
            float *qh = qr + (size_t)h * dh;
            float *kh = qr + d + (size_t)h * dh;
            const float *vh = qr + 2 * d + (size_t)h * dh;
            float kn = 0.0f, qn = 0.0f;
            for (int j = 0; j < dh; j++) { kn += kh[j] * kh[j]; qn += qh[j] * qh[j]; }
            float kr = 1.0f / sqrtf(kn + 1e-6f);
            float qr_ = 1.0f / sqrtf(qn + 1e-6f);
            for (int j = 0; j < dh; j++) { kh[j] *= kr; qh[j] *= qr_; }
            float *Ss = Ssrc + (size_t)h * dh * dh;
            float *Sd = Sdst + (size_t)h * dh * dh;
            float delta[SPEL_MAX_D];
            for (int i = 0; i < dh; i++) {
                float t = 0.0f;
                const float *row = Ss + (size_t)i * dh;
                for (int j = 0; j < dh; j++) t += row[j] * kh[j];
                delta[i] = vh[i] - t;
            }
            for (int i = 0; i < dh; i++) {
                const float *row = Ss + (size_t)i * dh;
                float *dr = Sd + (size_t)i * dh;
                float dv = beta * delta[i];
                for (int j = 0; j < dh; j++) dr[j] = a * row[j] + dv * kh[j];
            }
            float *oh = o + (size_t)h * dh;
            for (int i = 0; i < dh; i++) {
                const float *row = Sd + (size_t)i * dh;
                float t = 0.0f;
                for (int j = 0; j < dh; j++) t += row[j] * qh[j];
                oh[i] = t * (1.0f / sqrtf((float)dh));
            }
        }
    }
}

/* ------------------------------------------------------------ forward */

static void mlp_rows(spel_model *m, int l, const float *xn, int M,
                     float *h1, float *h3, float *hh, float *out) {
    const spel_cfg *c = &m->cfg;
    spel_layer *L = &m->layer[l];
    if (!m->is_ternary) {
        matmul_rows(h1, xn, L->w1, M, c->d_ff, c->d_model);
        matmul_rows(h3, xn, L->w3, M, c->d_ff, c->d_model);
        for (int r = 0; r < M; r++) {
            float *a = h1 + (size_t)r * c->d_ff;
            float *b = h3 + (size_t)r * c->d_ff;
            float *y = hh + (size_t)r * c->d_ff;
            for (int i = 0; i < c->d_ff; i++) y[i] = silu(a[i]) * b[i];
        }
        matmul_rows(out, hh, L->w2, M, c->d_model, c->d_ff);
    } else {
        for (int r = 0; r < M; r++) {
            draft_tern_gemv(m, &L->t1, xn + (size_t)r * c->d_model,
                            h1 + (size_t)r * c->d_ff);
            draft_tern_gemv(m, &L->t3, xn + (size_t)r * c->d_model,
                            h3 + (size_t)r * c->d_ff);
            float *a = h1 + (size_t)r * c->d_ff;
            float *b = h3 + (size_t)r * c->d_ff;
            float *y = hh + (size_t)r * c->d_ff;
            for (int i = 0; i < c->d_ff; i++) y[i] = silu(a[i]) * b[i];
            draft_tern_gemv(m, &L->t2, y, out + (size_t)r * c->d_model);
        }
    }
}

void spel_forward(spel_model *m, const int *tokens, int M, float *logits_out) {
    const spel_cfg *c = &m->cfg;
    const int d = c->d_model, dff = c->d_ff;
    if (M <= 0 || m->pos + M > c->max_seq || M > c->spec_k + 1) return;

    float *x    = m->buf;
    float *xn   = x + (size_t)M * d;
    float *qkv  = xn + (size_t)M * d;
    float *tmp  = qkv + (size_t)M * 3 * d;
    float *h1   = tmp + (size_t)M * d;
    float *h3   = h1 + (size_t)M * dff;
    float *hh   = h3 + (size_t)M * dff;
    float *obuf = hh + (size_t)M * dff;
    const int base = m->pos;

    for (int r = 0; r < M; r++) {
        if (tokens[r] < 0 || tokens[r] >= c->vocab) return;
        memcpy(x + (size_t)r * d, m->emb + (size_t)tokens[r] * d,
               (size_t)d * sizeof(float));
    }

    for (int l = 0; l < c->n_layers; l++) {
        spel_layer *L = &m->layer[l];
        rmsnorm_rows(x, L->n1, xn, M, d);

        if (!c->is_linear[l]) {
            proj_qkv(m, L, xn, qkv, M, d);
            attention_rows(m, l, qkv, M, tmp);
            proj_wo(m, L, tmp, obuf, M, d);
        } else {
            linear_rows(m, l, xn, M, qkv, tmp, h1);
            if (m->cap_enabled && m->cap_q) {
                int chbase = c->lin_index[l] * c->n_heads;
                for (int r = 0; r < M; r++) {
                    int p = base + r;
                    if (p >= m->cap_len) continue;
                    for (int h = 0; h < c->n_heads; h++) {
                        size_t off = ((size_t)p * c->n_channels + chbase + h) * c->head_dim;
                        memcpy(m->cap_q + off, qkv + (size_t)r * 3 * d + (size_t)h * c->head_dim,
                               (size_t)c->head_dim * sizeof(float));
                        memcpy(m->cap_k + off, qkv + (size_t)r * 3 * d + d + (size_t)h * c->head_dim,
                               (size_t)c->head_dim * sizeof(float));
                    }
                }
            }
            proj_wo(m, L, tmp, obuf, M, d);
        }
        for (int r = 0; r < M; r++)
            for (int i = 0; i < d; i++) x[(size_t)r * d + i] += obuf[(size_t)r * d + i];

        rmsnorm_rows(x, L->n2, xn, M, d);
        mlp_rows(m, l, xn, M, h1, h3, hh, obuf);
        for (int r = 0; r < M; r++)
            for (int i = 0; i < d; i++) x[(size_t)r * d + i] += obuf[(size_t)r * d + i];
    }

    rmsnorm_rows(x, m->normf, xn, M, d);
    if (logits_out) {
        matmul_rows(logits_out, xn, m->wout, M, c->vocab, d);
        memcpy(m->last_logits, logits_out + (size_t)(M - 1) * c->vocab,
               (size_t)c->vocab * sizeof(float));
    } else {
        matmul_rows(m->last_logits, xn + (size_t)(M - 1) * d, m->wout, 1, c->vocab, d);
    }
    m->pos += M;
}

/* ---------------------------------------------------------- checkpoint */

struct spel_ckpt {
    int pos;
    int n_layers;
    float *S[SPEL_MAX_LAYERS];
    float *xh[SPEL_MAX_LAYERS];
    float *logits;
};

spel_ckpt *spel_ckpt_new(const spel_model *m) {
    spel_ckpt *cp = (spel_ckpt *)calloc(1, sizeof(*cp));
    if (!cp) return NULL;
    cp->n_layers = m->cfg.n_layers;
    cp->pos = m->pos;
    cp->logits = (float *)malloc((size_t)m->cfg.vocab * sizeof(float));
    if (!cp->logits) { spel_ckpt_free(cp); return NULL; }
    for (int l = 0; l < cp->n_layers; l++) {
        if (m->cfg.is_linear[l]) {
            size_t sd = (size_t)(m->cfg.spec_k + 1) * m->cfg.n_heads
                      * m->cfg.head_dim * m->cfg.head_dim;
            cp->S[l] = (float *)malloc(sd * sizeof(float));
            cp->xh[l] = (float *)malloc((size_t)SPEL_XH_RING * m->cfg.d_model * sizeof(float));
            if (!cp->S[l] || !cp->xh[l]) { spel_ckpt_free(cp); return NULL; }
        }
    }
    spel_ckpt_save(cp, m);
    return cp;
}

void spel_ckpt_save(spel_ckpt *cp, const spel_model *m) {
    cp->pos = m->pos;
    memcpy(cp->logits, m->last_logits, (size_t)m->cfg.vocab * sizeof(float));
    for (int l = 0; l < m->cfg.n_layers; l++) {
        if (!m->cfg.is_linear[l]) continue;
        size_t sd = (size_t)(m->cfg.spec_k + 1) * m->cfg.n_heads
                  * m->cfg.head_dim * m->cfg.head_dim;
        memcpy(cp->S[l], m->S[l], sd * sizeof(float));
        memcpy(cp->xh[l], m->xh[l], (size_t)SPEL_XH_RING * m->cfg.d_model * sizeof(float));
    }
}

void spel_ckpt_load(spel_model *m, const spel_ckpt *cp) {
    m->pos = cp->pos;
    memcpy(m->last_logits, cp->logits, (size_t)m->cfg.vocab * sizeof(float));
    for (int l = 0; l < m->cfg.n_layers; l++) {
        if (!m->cfg.is_linear[l]) continue;
        size_t sd = (size_t)(m->cfg.spec_k + 1) * m->cfg.n_heads
                  * m->cfg.head_dim * m->cfg.head_dim;
        memcpy(m->S[l], cp->S[l], sd * sizeof(float));
        memcpy(m->xh[l], cp->xh[l], (size_t)SPEL_XH_RING * m->cfg.d_model * sizeof(float));
    }
}

void spel_ckpt_free(spel_ckpt *cp) {
    if (!cp) return;
    for (int l = 0; l < cp->n_layers; l++) { free(cp->S[l]); free(cp->xh[l]); }
    free(cp->logits);
    free(cp);
}

/* ---------------------------------------------- binary weights SPELW001 */

void spel_model_attach_draft(spel_model *target, spel_model *draft) {
    if (target) target->draft = draft;
}

static int wr_bytes(FILE *f, const void *p, size_t n) {
    return fwrite(p, 1, n, f) == n ? 0 : -1;
}
static int rd_bytes(FILE *f, void *p, size_t n) {
    return fread(p, 1, n, f) == n ? 0 : -1;
}
static int wr_u32(FILE *f, uint32_t v) {
    uint8_t b[4] = {(uint8_t)v, (uint8_t)(v >> 8),
                    (uint8_t)(v >> 16), (uint8_t)(v >> 24)};
    return wr_bytes(f, b, 4);
}
static int rd_u32(FILE *f, uint32_t *v) {
    uint8_t b[4];
    if (rd_bytes(f, b, 4)) return -1;
    *v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
         ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return 0;
}
static int wr_i32(FILE *f, int32_t v) { return wr_u32(f, (uint32_t)v); }
static int rd_i32(FILE *f, int32_t *v) {
    uint32_t u;
    if (rd_u32(f, &u)) return -1;
    *v = (int32_t)u;
    return 0;
}
static int wr_f32s(FILE *f, const float *p, size_t n) {
    return fwrite(p, sizeof(float), n, f) == n ? 0 : -1;
}
static int rd_f32s(FILE *f, float *p, size_t n) {
    return fread(p, sizeof(float), n, f) == n ? 0 : -1;
}

/* Serialized per-layer order:
 * n1, wqkv, wo, [conv, wa, wb], n2, w1, w3, w2. */
static int save_layer(FILE *f, const spel_layer *L, const spel_cfg *c, int l) {
    const int d = c->d_model, dff = c->d_ff;
    if (wr_f32s(f, L->n1, (size_t)d)) return -1;
    if (wr_f32s(f, L->wqkv, (size_t)3 * d * d)) return -1;
    if (wr_f32s(f, L->wo, (size_t)d * d)) return -1;
    if (c->is_linear[l]) {
        if (wr_f32s(f, L->conv, (size_t)c->conv_k * d)) return -1;
        if (wr_f32s(f, L->wa, (size_t)c->n_heads)) return -1;
        if (wr_f32s(f, L->wb, (size_t)c->n_heads)) return -1;
    }
    if (wr_f32s(f, L->n2, (size_t)d)) return -1;
    if (wr_f32s(f, L->w1, (size_t)d * dff)) return -1;
    if (wr_f32s(f, L->w3, (size_t)d * dff)) return -1;
    if (wr_f32s(f, L->w2, (size_t)dff * d)) return -1;
    return 0;
}

static int load_layer(FILE *f, spel_layer *L, const spel_cfg *c, int l) {
    const int d = c->d_model, dff = c->d_ff;
    if (rd_f32s(f, L->n1, (size_t)d)) return -1;
    if (rd_f32s(f, L->wqkv, (size_t)3 * d * d)) return -1;
    if (rd_f32s(f, L->wo, (size_t)d * d)) return -1;
    if (c->is_linear[l]) {
        if (rd_f32s(f, L->conv, (size_t)c->conv_k * d)) return -1;
        if (rd_f32s(f, L->wa, (size_t)c->n_heads)) return -1;
        if (rd_f32s(f, L->wb, (size_t)c->n_heads)) return -1;
    }
    if (rd_f32s(f, L->n2, (size_t)d)) return -1;
    if (rd_f32s(f, L->w1, (size_t)d * dff)) return -1;
    if (rd_f32s(f, L->w3, (size_t)d * dff)) return -1;
    if (rd_f32s(f, L->w2, (size_t)dff * d)) return -1;
    return 0;
}

static size_t serialized_draft_floats(const spel_cfg *target, int n_layers,
                                      const int *map) {
    size_t n = 0;
    for (int l = 0; l < n_layers; l++) {
        int src = map[l];
        int linear = src >= 0 && src < target->n_layers
                  && target->is_linear[src];
        n += layer_tensor_count(target, linear);
    }
    return n;
}

static int forward_chunks(spel_model *m, const int32_t *tokens, int n,
                          float *logits) {
    int off = 0;
    while (off < n) {
        int chunk = m->cfg.spec_k;
        if (chunk < 1) return -1;
        if (chunk > n - off) chunk = n - off;
        spel_forward(m, tokens + off, chunk, logits + (size_t)off * m->cfg.vocab);
        off += chunk;
    }
    return 0;
}

/* Capture a deterministic SPELV001 footer from a fresh state, then restore
 * the caller's model state. */
static int make_golden(spel_model *target, spel_model *draft,
                       int32_t tokens[SPEL_GOLDEN_LEN],
                       float **out_t, float **out_d) {
    *out_t = NULL; *out_d = NULL;
    if (!target || target->cfg.max_seq < SPEL_GOLDEN_LEN) return -1;
    int V = target->cfg.vocab;
    for (int i = 0; i < SPEL_GOLDEN_LEN; i++)
        tokens[i] = (int32_t)((i * 29 + 7) % V);

    float *tl = (float *)malloc((size_t)SPEL_GOLDEN_LEN * V * sizeof(float));
    float *dl = draft ? (float *)malloc((size_t)SPEL_GOLDEN_LEN * V * sizeof(float)) : NULL;
    if (!tl || (draft && !dl)) { free(tl); free(dl); return -1; }

    spel_ckpt *ct = spel_ckpt_new(target);
    spel_ckpt *cd = draft ? spel_ckpt_new(draft) : NULL;
    if (!ct || (draft && !cd)) {
        spel_ckpt_free(ct); spel_ckpt_free(cd);
        free(tl); free(dl);
        return -1;
    }
    int save_acc = spel_use_accelerate;
    int cap_t = target->cap_enabled, cap_d = draft ? draft->cap_enabled : 0;
    target->cap_enabled = 0;
    if (draft) draft->cap_enabled = 0;
    spel_use_accelerate = 0;
    spel_model_reset(target);
    if (forward_chunks(target, tokens, SPEL_GOLDEN_LEN, tl)) goto fail;
    if (draft) {
        spel_model_reset(draft);
        if (forward_chunks(draft, tokens, SPEL_GOLDEN_LEN, dl)) goto fail;
    }
    spel_use_accelerate = save_acc;
    target->cap_enabled = cap_t;
    if (draft) draft->cap_enabled = cap_d;
    if (ct) spel_ckpt_load(target, ct);
    if (cd) spel_ckpt_load(draft, cd);
    spel_ckpt_free(ct); spel_ckpt_free(cd);
    *out_t = tl; *out_d = dl;
    return 0;

fail:
    spel_use_accelerate = save_acc;
    target->cap_enabled = cap_t;
    if (draft) draft->cap_enabled = cap_d;
    if (ct) spel_ckpt_load(target, ct);
    if (cd) spel_ckpt_load(draft, cd);
    spel_ckpt_free(ct); spel_ckpt_free(cd);
    free(tl); free(dl);
    return -1;
}

static int write_header(FILE *f, const spel_model *target,
                        const spel_model *draft, uint32_t has_footer) {
    uint8_t h[64];
    memset(h, 0, sizeof(h));
    memcpy(h, SPEL_WEIGHTS_MAGIC, 8);
    uint32_t vals[13] = {
        1u,
        (uint32_t)target->cfg.d_model,
        (uint32_t)target->cfg.n_layers,
        (uint32_t)target->cfg.n_heads,
        (uint32_t)target->cfg.head_dim,
        (uint32_t)target->cfg.d_ff,
        (uint32_t)target->cfg.vocab,
        (uint32_t)target->cfg.attn_period,
        draft ? (uint32_t)draft->cfg.n_layers : 0u,
        draft ? (uint32_t)draft->source_map[0] : 0u,
        draft ? (uint32_t)draft->source_map[1] : 0u,
        draft ? (uint32_t)draft->source_map[2] : 0u,
        draft ? (uint32_t)draft->source_map[3] : 0u
    };
    for (int i = 0; i < 13; i++)
        for (int b = 0; b < 4; b++)
            h[8 + i * 4 + b] = (uint8_t)(vals[i] >> (8 * b));
    for (int b = 0; b < 4; b++)
        h[60 + b] = (uint8_t)(has_footer >> (8 * b));
    return wr_bytes(f, h, sizeof(h));
}

typedef struct {
    uint32_t version, d_model, n_layers, n_heads, head_dim, d_ff, vocab;
    uint32_t attn_period, n_draft_layers, draft_map[4], has_footer;
} spel_wheader;

static int read_header(FILE *f, spel_wheader *h) {
    uint8_t b[64];
    if (rd_bytes(f, b, sizeof(b))) return -1;
    if (memcmp(b, SPEL_WEIGHTS_MAGIC, 8)) return -1;
    uint32_t v[13];
    for (int i = 0; i < 13; i++)
        v[i] = (uint32_t)b[8 + i * 4] | ((uint32_t)b[9 + i * 4] << 8) |
               ((uint32_t)b[10 + i * 4] << 16) | ((uint32_t)b[11 + i * 4] << 24);
    h->version = v[0]; h->d_model = v[1]; h->n_layers = v[2];
    h->n_heads = v[3]; h->head_dim = v[4]; h->d_ff = v[5];
    h->vocab = v[6]; h->attn_period = v[7]; h->n_draft_layers = v[8];
    for (int i = 0; i < 4; i++) h->draft_map[i] = v[9 + i];
    h->has_footer = (uint32_t)b[60] | ((uint32_t)b[61] << 8) |
                    ((uint32_t)b[62] << 16) | ((uint32_t)b[63] << 24);
    if (h->version != 1 || h->n_draft_layers > 4) return -1;
    for (uint32_t i = 0; i < h->n_draft_layers; i++)
        if (h->draft_map[i] >= h->n_layers) return -1;
    return 0;
}

int spel_save_weights_pair(const spel_model *target, const spel_model *draft,
                           const char *path) {
    if (!target || !path) return -1;
    if (draft && (draft->cfg.n_layers != 4 || draft->n_draft_layers != 4))
        return -2; /* SPELW001 carries exactly the four distilled layers */
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    int32_t gtokens[SPEL_GOLDEN_LEN];
    float *gt = NULL, *gd = NULL;
    int has_golden = make_golden((spel_model *)target, (spel_model *)draft,
                                 gtokens, &gt, &gd) == 0 && gt;
    if (write_header(f, target, draft, has_golden ? 1u : 0u)) goto fail;

    const spel_cfg *c = &target->cfg;
    const int d = c->d_model;
    if (wr_f32s(f, target->emb, (size_t)c->vocab * d)) goto fail;
    for (int l = 0; l < c->n_layers; l++)
        if (save_layer(f, &target->layer[l], c, l)) goto fail;
    if (wr_f32s(f, target->normf, (size_t)d)) goto fail;
    if (wr_f32s(f, target->wout, (size_t)c->vocab * d)) goto fail;

    if (draft) {
        for (int l = 0; l < 4; l++)
            if (save_layer(f, &draft->layer[l], &draft->cfg, l)) goto fail;
    }

    if (has_golden) {
        if (wr_bytes(f, SPEL_GOLDEN_MAGIC, 8)) goto fail;
        if (wr_u32(f, SPEL_GOLDEN_LEN)) goto fail;
        for (int i = 0; i < SPEL_GOLDEN_LEN; i++)
            if (wr_i32(f, gtokens[i])) goto fail;
        if (wr_f32s(f, gt, (size_t)SPEL_GOLDEN_LEN * c->vocab)) goto fail;
        if (gd && wr_f32s(f, gd, (size_t)SPEL_GOLDEN_LEN * c->vocab)) goto fail;
        if (!gd) {
            float *z = (float *)calloc((size_t)SPEL_GOLDEN_LEN * c->vocab, sizeof(float));
            int bad = !z || wr_f32s(f, z, (size_t)SPEL_GOLDEN_LEN * c->vocab);
            free(z);
            if (bad) goto fail;
        }
    }

    free(gt); free(gd);
    if (fclose(f)) return -1;
    return 0;
fail:
    free(gt); free(gd);
    fclose(f);
    return -1;
}

int spel_save_weights(const spel_model *m, const char *path) {
    if (!m) return -1;
    if (m->draft) return spel_save_weights_pair(m, m->draft, path);
    /* Default 12-layer targets get the conventional [0,3,6,9] cheap draft,
     * copied from the target and quantized on load. Callers that have a
     * distilled draft should attach it or use spel_save_weights_pair. */
    if (m->cfg.n_layers >= 10) {
        int map[4] = {0, 3, 6, 9};
        spel_model *tmp = spel_model_new_draft_subset(m, map, 4);
        if (!tmp) return -1;
        int r = spel_save_weights_pair(m, tmp, path);
        spel_model_free(tmp);
        return r;
    }
    return spel_save_weights_pair(m, NULL, path);
}

static int read_footer(FILE *f, spel_model *target, uint32_t has_footer) {
    free(target->golden_target_logits);
    free(target->golden_draft_logits);
    target->golden_target_logits = target->golden_draft_logits = NULL;
    target->golden_len = 0;
    if (!has_footer) return 0;
    char magic[8];
    uint32_t n = 0;
    if (rd_bytes(f, magic, 8) || memcmp(magic, SPEL_GOLDEN_MAGIC, 8)) return -1;
    if (rd_u32(f, &n) || n == 0 || n > SPEL_GOLDEN_LEN) return -1;
    int32_t toks[SPEL_GOLDEN_LEN];
    for (uint32_t i = 0; i < n; i++)
        if (rd_i32(f, &toks[i])) return -1;
    size_t count = (size_t)n * target->cfg.vocab;
    float *tl = (float *)malloc(count * sizeof(float));
    float *dl = (float *)malloc(count * sizeof(float));
    if (!tl || !dl) { free(tl); free(dl); return -1; }
    if (rd_f32s(f, tl, count) || rd_f32s(f, dl, count)) {
        free(tl); free(dl); return -1;
    }
    memcpy(target->golden_tokens, toks, (size_t)n * sizeof(int32_t));
    target->golden_len = (int)n;
    target->golden_target_logits = tl;
    target->golden_draft_logits = dl;
    return 0;
}

int spel_load_weights_pair(spel_model *target, spel_model *draft,
                           const char *path) {
    if (!target || !path) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    spel_wheader h;
    if (read_header(f, &h)) goto fail;
    const spel_cfg *c = &target->cfg;
    if ((int)h.d_model != c->d_model || (int)h.n_layers != c->n_layers ||
        (int)h.n_heads != c->n_heads || (int)h.head_dim != c->head_dim ||
        (int)h.d_ff != c->d_ff || (int)h.vocab != c->vocab ||
        (int)h.attn_period != c->attn_period) goto fail;
    if (draft) {
        if ((int)h.n_draft_layers != draft->cfg.n_layers) goto fail;
        for (int l = 0; l < 4; l++)
            if ((int)h.draft_map[l] != draft->source_map[l]) goto fail;
    }

    const int d = c->d_model;
    if (rd_f32s(f, target->emb, (size_t)c->vocab * d)) goto fail;
    for (int l = 0; l < c->n_layers; l++)
        if (load_layer(f, &target->layer[l], c, l)) goto fail;
    if (rd_f32s(f, target->normf, (size_t)d)) goto fail;
    if (rd_f32s(f, target->wout, (size_t)c->vocab * d)) goto fail;

    if ((int)h.n_draft_layers > 0) {
        if (draft) {
            if (draft->cfg.n_layers != (int)h.n_draft_layers) goto fail;
            for (int l = 0; l < draft->cfg.n_layers; l++) {
                if (load_layer(f, &draft->layer[l], &draft->cfg, l)) goto fail;
                tern_quantize_mat(&draft->layer[l].t1, draft->layer[l].w1);
                tern_quantize_mat(&draft->layer[l].t3, draft->layer[l].w3);
                tern_quantize_mat(&draft->layer[l].t2, draft->layer[l].w2);
                q8_quantize_mat(&draft->layer[l].q_wqkv, draft->layer[l].wqkv);
                q8_quantize_mat(&draft->layer[l].q_wo, draft->layer[l].wo);
            }
        } else {
            int map[4];
            for (int i = 0; i < 4; i++) map[i] = (int)h.draft_map[i];
            size_t skip = serialized_draft_floats(c, (int)h.n_draft_layers, map);
            if (fseek(f, (long)(skip * sizeof(float)), SEEK_CUR)) goto fail;
        }
    }

    if (read_footer(f, target, h.has_footer)) goto fail;
    fclose(f);
    spel_model_reset(target);
    if (draft) spel_model_reset(draft);
    return 0;
fail:
    fclose(f);
    free(target->golden_target_logits);
    free(target->golden_draft_logits);
    target->golden_target_logits = target->golden_draft_logits = NULL;
    target->golden_len = 0;
    return -1;
}

int spel_load_weights(spel_model *m, const char *path) {
    if (!m) return -1;
    return spel_load_weights_pair(m, m->draft, path);
}

int spel_verify_weights(spel_model *target, spel_model *draft, const char *path) {
    if (!target || !draft) return -1;
    if (spel_load_weights_pair(target, draft, path)) {
        printf("  failed to load %s\n", path);
        return -1;
    }
    if (target->golden_len != SPEL_GOLDEN_LEN || !target->golden_target_logits ||
        !target->golden_draft_logits) {
        printf("  no SPELV001 golden footer in %s\n", path);
        return -2;
    }
    int V = target->cfg.vocab, n = target->golden_len;
    float *tl = (float *)malloc((size_t)n * V * sizeof(float));
    float *dl = (float *)malloc((size_t)n * V * sizeof(float));
    if (!tl || !dl) { free(tl); free(dl); return -1; }
    int save_acc = spel_use_accelerate;
    spel_use_accelerate = 0;
    spel_model_reset(target);
    spel_model_reset(draft);
    int ok = forward_chunks(target, target->golden_tokens, n, tl) == 0 &&
             forward_chunks(draft, target->golden_tokens, n, dl) == 0;
    spel_use_accelerate = save_acc;
    if (!ok) { free(tl); free(dl); return -1; }

    double tmax = 0.0, dmax = 0.0;
    int targ = 0, darg = 0;
    for (int i = 0; i < n; i++) {
        const float *et = target->golden_target_logits + (size_t)i * V;
        const float *ed = target->golden_draft_logits + (size_t)i * V;
        const float *at = tl + (size_t)i * V;
        const float *ad = dl + (size_t)i * V;
        for (int j = 0; j < V; j++) {
            double a = fabs((double)at[j] - et[j]);
            double b = fabs((double)ad[j] - ed[j]);
            if (a > tmax) tmax = a;
            if (b > dmax) dmax = b;
        }
        if (spel_argmax(at, V) == spel_argmax(et, V)) targ++;
        if (spel_argmax(ad, V) == spel_argmax(ed, V)) darg++;
    }
    printf("  target: max_abs=%.6g  argmax=%d/%d\n", tmax, targ, n);
    printf("  draft : max_abs=%.6g  argmax=%d/%d\n", dmax, darg, n);
    free(tl); free(dl);
    return (targ == n && tmax < 1e-3) ? 0 : 1;
}


/* --------------------------------------------------------- spec decode */

void spel_spec_cfg_default(spel_spec_cfg *s, int K) {
    s->K = K;
    s->governor = 1;
    s->alpha_threshold = 0.30;
    s->window = 2;
    s->force_alpha = -1.0; /* real verification by default */
}

void spel_spec_generate(spel_model *target, spel_model *draft,
                        int n_gen, int *out, const spel_spec_cfg *sc,
                        spel_spec_stats *st) {
    if (!target || !draft || !sc || !st) return;
    const spel_cfg *c = &target->cfg;
    const int K = sc->K, V = c->vocab;
    memset(st, 0, sizeof(*st));
    if (K < 1 || K > c->spec_k || n_gen <= 0) return;
    if (target->pos + n_gen > c->max_seq || draft->pos + n_gen > c->max_seq)
        return;

    /* Speculative rounds need up to K+1 transient positions. If the context is
     * too tight, fall back to AR rather than silently truncating a forward. */
    if (target->pos + n_gen + K + 1 > c->max_seq) {
        for (int i = 0; i < n_gen; i++) {
            int tok = spel_argmax(target->last_logits, V);
            spel_forward(target, &tok, 1, NULL);
            spel_forward(draft, &tok, 1, NULL);
            out[i] = tok;
        }
        st->rounds = st->ar_rounds = st->emitted = n_gen;
        st->tokens_per_round = 1.0;
        return;
    }

    float *saved = (float *)malloc((size_t)V * sizeof(float));
    float *tlog = (float *)malloc((size_t)(K + 2) * V * sizeof(float));
    int *prop  = (int *)malloc((size_t)K * sizeof(int));
    if (!saved || !tlog || !prop) {
        free(saved); free(tlog); free(prop);
        return;
    }
    int alpha_win[16] = {0};
    int win_pos = 0, win_fill = 0;
    int window = sc->window > 0 ? sc->window : 1;
    if (window > 16) window = 16;
    int fallback = 0;
    int pending = -1;     /* emitted correction, not yet consumed by the models */
    int emitted = 0;
    double force_credit = 0.0; /* fractional forced-acceptance carry */

    while (emitted < n_gen) {
        if (fallback) {
            if (pending >= 0) {
                spel_forward(target, &pending, 1, NULL);
                spel_forward(draft,  &pending, 1, NULL);
                pending = -1;
            }
            int tok = spel_argmax(target->last_logits, V);
            spel_forward(target, &tok, 1, NULL);
            out[emitted++] = tok;
            st->rounds++;
            st->ar_rounds++;
            st->emitted++;
            continue;
        }

        const int base = target->pos;
        const int has_pending = pending >= 0;

        if (has_pending) {
            /* The correction token was emitted last round but deliberately not
             * forwarded. Consume it in the draft now; the target consumes it as
             * the first token of the verification batch. */
            spel_forward(draft, &pending, 1, NULL);
        } else {
            memcpy(saved, target->last_logits, (size_t)V * sizeof(float));
        }

        for (int j = 0; j < K; j++) {
            prop[j] = spel_argmax(draft->last_logits, V);
            spel_forward(draft, &prop[j], 1, NULL);
            st->proposed++;
        }

        int m = 0, corr = 0;
        if (has_pending) {
            int toks[SPEL_MAX_BATCH + 1];
            toks[0] = pending;
            for (int j = 0; j < K; j++) toks[j + 1] = prop[j];
            spel_forward(target, toks, K + 1, tlog);
            while (m < K && spel_argmax(tlog + (size_t)m * V, V) == prop[m]) m++;
        } else {
            memcpy(tlog, saved, (size_t)V * sizeof(float));
            spel_forward(target, prop, K, tlog + V);
            while (m < K && spel_argmax(tlog + (size_t)m * V, V) == prop[m]) m++;
        }

        /* Benchmark-only forced acceptance. Both model forwards above have
         * already run, including all K proposal draft steps and the pending
         * correction step, so the measured cost includes the real draft and
         * verification work. Only the accepted-prefix decision is synthetic. */
        if (sc->force_alpha >= 0.0) {
            force_credit += sc->force_alpha * (double)K;
            int forced = (int)force_credit;
            if (forced < 0) forced = 0;
            if (forced > K) forced = K;
            force_credit -= (double)forced;
            m = forced;
        }

        corr = spel_argmax(tlog + (size_t)m * V, V);
        if (has_pending) {
            target->pos = base + 1 + m;
            draft->pos  = base + 1 + m;
        } else {
            target->pos = base + m;
            draft->pos  = base + m;
        }

        /* Emit accepted proposals. If n_gen ends inside the accepted prefix,
         * leave both models exactly after the last emitted token. */
        int e = 0;
        while (e < m && emitted < n_gen) out[emitted++] = prop[e++];
        if (emitted >= n_gen) {
            int consumed = (has_pending ? 1 : 0) + e;
            target->pos = base + consumed;
            draft->pos  = base + consumed;
            memcpy(target->last_logits, tlog + (size_t)e * V,
                   (size_t)V * sizeof(float));
            pending = -1;
            st->accepted += m;
            st->rounds++;
            st->spec_rounds++;
            st->emitted += e;
            break;
        }

        out[emitted++] = corr;
        pending = corr;
        st->accepted += m;
        st->rounds++;
        st->spec_rounds++;
        st->emitted += m + 1;
        st->corr++;

        if (sc->governor) {
            int trip = (m == 0);
            if (!trip) {
                alpha_win[win_pos] = m;
                win_pos = (win_pos + 1) % window;
                if (win_fill < window) win_fill++;
                if (win_fill == window) {
                    int s = 0;
                    for (int i = 0; i < window; i++) s += alpha_win[i];
                    if ((double)s / (double)(window * K) < sc->alpha_threshold)
                        trip = 1;
                }
            }
            if (trip) {
                /* Suspend drafting for the rest of this call. The pending
                 * correction is consumed before the AR fallback so the target
                 * state stays aligned with the emitted stream. */
                fallback = 1;
                win_fill = 0;
                st->governor_trips++;
            }
        }
    }

    if (pending >= 0) {
        spel_forward(target, &pending, 1, NULL);
        spel_forward(draft,  &pending, 1, NULL);
    }
    st->alpha = st->proposed ? (double)st->accepted / (double)st->proposed : 0.0;
    st->tokens_per_round = st->rounds ? (double)st->emitted / (double)st->rounds : 0.0;
    free(saved); free(tlog); free(prop);
}

/* --------------------------------------------------------------- sizes */

size_t spel_kv_bytes(const spel_cfg *c) {
    return (size_t)c->n_attn * 2 * c->max_seq * c->d_model * sizeof(float);
}
