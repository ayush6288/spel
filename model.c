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
    for (int l = 0; l < n_layers; l++) {
        int src = layer_map[l];
        if (src < 0 || src >= t->cfg.n_layers) { free(m); return NULL; }
        m->layer[l] = t->layer[src];
        int is_attn = !t->cfg.is_linear[src];
        m->cfg.is_linear[l] = !is_attn;
        m->cfg.lin_index[l] = is_attn ? -1 : m->cfg.n_linear++;
        m->cfg.attn_index[l] = is_attn ? m->cfg.n_attn++ : -1;
    }
    m->cfg.n_channels = m->cfg.n_linear * m->cfg.n_heads;
    m->owns_params = 0;
    m->emb = t->emb; m->wout = t->wout; m->normf = t->normf;
    m->rope_cos = t->rope_cos; m->rope_sin = t->rope_sin;
    m->is_ternary = 1;
    m->rng = t->rng ^ 0xA5A5A5A55A5A5A5AULL;
    alloc_runtime(m);
    if (!m->buf || !m->last_logits) { spel_model_free(m); return NULL; }
    const spel_cfg *c = &m->cfg;
    for (int l = 0; l < n_layers; l++) {
        int src = layer_map[l];
        spel_layer *L = &m->layer[l];
        if (tern_mat_init(&L->t1, c->d_ff, c->d_model) ||
            tern_mat_init(&L->t3, c->d_ff, c->d_model) ||
            tern_mat_init(&L->t2, c->d_model, c->d_ff)) {
            spel_model_free(m);
            return NULL;
        }
        tern_quantize_mat(&L->t1, t->layer[src].w1);
        tern_quantize_mat(&L->t3, t->layer[src].w3);
        tern_quantize_mat(&L->t2, t->layer[src].w2);
        if (q8_mat_init(&L->q_wqkv, 3 * c->d_model, c->d_model) ||
            q8_mat_init(&L->q_wo, c->d_model, c->d_model)) {
            spel_model_free(m);
            return NULL;
        }
        q8_quantize_mat(&L->q_wqkv, t->layer[src].wqkv);
        q8_quantize_mat(&L->q_wo, t->layer[src].wo);
    }
    m->pos = 0;
    return m;
}

spel_model *spel_model_new_draft(const spel_model *t) {
    int map[SPEL_MAX_LAYERS];
    for (int i = 0; i < t->cfg.n_layers; i++) map[i] = i;
    return spel_model_new_draft_subset(t, map, t->cfg.n_layers);
}

void spel_model_free(spel_model *m) {
    if (!m) return;
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
    }
    free(m->buf); free(m->last_logits);
    free(m->cap_q); free(m->cap_k);
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

static void proj_qkv(spel_model *m, spel_layer *L, const float *x, float *y,
                     int M, int d) {
    if (m->is_ternary && L->q_wqkv.q) {
        for (int r = 0; r < M; r++)
            q8_gemv(&L->q_wqkv, x + (size_t)r * d, y + (size_t)r * 3 * d);
    } else {
        matmul_rows(y, x, L->wqkv, M, 3 * d, d);
    }
}

static void proj_wo(spel_model *m, spel_layer *L, const float *x, float *y,
                    int M, int d) {
    if (m->is_ternary && L->q_wo.q) {
        for (int r = 0; r < M; r++)
            q8_gemv(&L->q_wo, x + (size_t)r * d, y + (size_t)r * d);
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
            tern_gemv(&L->t1, xn + (size_t)r * c->d_model, h1 + (size_t)r * c->d_ff);
            tern_gemv(&L->t3, xn + (size_t)r * c->d_model, h3 + (size_t)r * c->d_ff);
            float *a = h1 + (size_t)r * c->d_ff;
            float *b = h3 + (size_t)r * c->d_ff;
            float *y = hh + (size_t)r * c->d_ff;
            for (int i = 0; i < c->d_ff; i++) y[i] = silu(a[i]) * b[i];
            tern_gemv(&L->t2, y, out + (size_t)r * c->d_model);
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

/* --------------------------------------------------------- spec decode */

void spel_spec_cfg_default(spel_spec_cfg *s, int K) {
    s->K = K;
    s->governor = 1;
    s->alpha_threshold = 0.30;
    s->window = 2;
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
            corr = spel_argmax(tlog + (size_t)m * V, V);
            target->pos = base + 1 + m;
            draft->pos  = base + 1 + m;
        } else {
            memcpy(tlog, saved, (size_t)V * sizeof(float));
            spel_forward(target, prop, K, tlog + V);
            while (m < K && spel_argmax(tlog + (size_t)m * V, V) == prop[m]) m++;
            corr = spel_argmax(tlog + (size_t)m * V, V);
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
