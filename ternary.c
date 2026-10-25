/* ternary.c — 2-bit {-1,0,+1} packing and ARM64 NEON GEMV.
 *
 * Layout: 4 weights per byte, low bits first:
 *   bits [1:0] = w0, [3:2] = w1, [5:4] = w2, [7:6] = w3
 *   code 0 -> 0, 1 -> +1, 3 -> -1  (code 2 is unused / decoded as 0)
 *
 * Two kernels:
 *   - dotprod:  unpack 64 weights into 4 int8x16 vectors, vdotq_s32
 *   - addsub:   unpack to int8 and accumulate with vaddw_s8 (no multiplier)
 * Both accumulate in int32/int16 exactly; results are integer-identical.
 */
#include "spel.h"
#include <arm_neon.h>
#include <sched.h>

int tern_kernel_mode = 0;

static const uint8_t kLut[16] = {0,1,0,0xFF, 0,0,0,0, 0,0,0,0, 0,0,0,0};

size_t tern_packed_stride(int ncols) { return (size_t)((ncols + 3) / 4); }

int tern_mat_init(tern_mat *m, int nrows, int ncols) {
    memset(m, 0, sizeof(*m));
    m->nrows = nrows;
    m->ncols = ncols;
    m->packed_stride = (int)tern_packed_stride(ncols);
    m->packed = (uint8_t *)calloc((size_t)nrows * m->packed_stride, 1);
    m->scale  = (float *)calloc((size_t)nrows, sizeof(float));
    if (!m->packed || !m->scale) { tern_mat_free(m); return -1; }
    for (int r = 0; r < nrows; r++) m->scale[r] = 1.0f;
    return 0;
}

void tern_mat_free(tern_mat *m) {
    if (!m) return;
    free(m->packed); free(m->scale);
    memset(m, 0, sizeof(*m));
}

/* Least-squares scale for a fixed ternary sign pattern. */
void tern_quantize_row(const float *w, int n, uint8_t *packed, float *scale) {
    memset(packed, 0, tern_packed_stride(n));
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += fabsf(w[i]);
    s /= (float)n;
    if (s < 1e-8f) s = 1e-8f;

    for (int it = 0; it < 3; it++) {
        float num = 0.0f, den = 0.0f;
        for (int i = 0; i < n; i++) {
            int c = (int)roundf(w[i] / s);
            if (c > 1) c = 1; else if (c < -1) c = -1;
            num += w[i] * (float)c;
            den += (float)(c * c);
        }
        if (den > 0.0f) s = num / den;
        if (s < 1e-8f) s = 1e-8f;
    }
    for (int i = 0; i < n; i++) {
        int c = (int)roundf(w[i] / s);
        if (c > 1) c = 1; else if (c < -1) c = -1;
        packed[i >> 2] |= (uint8_t)((c < 0 ? 3 : c > 0 ? 1 : 0) << ((i & 3) * 2));
    }
    *scale = s;
}

void tern_quantize_mat(tern_mat *m, const float *w) {
    for (int r = 0; r < m->nrows; r++)
        tern_quantize_row(w + (size_t)r * m->ncols, m->ncols,
                          m->packed + (size_t)r * m->packed_stride, &m->scale[r]);
}

static inline int8_t tern_decode(uint8_t c) {
    return c == 1 ? 1 : (c == 3 ? -1 : 0);
}

void tern_dequant_row(const uint8_t *packed, float scale, int n, float *out) {
    for (int i = 0; i < n; i++) {
        uint8_t c = (packed[i >> 2] >> ((i & 3) * 2)) & 3;
        out[i] = (float)tern_decode(c) * scale;
    }
}

/* ---------------- expansion helpers ---------------- */

/* Expand 16 packed bytes -> four int8x16 vectors (64 weights). */
static inline void tern_expand16(uint8x16_t p, uint8x16_t o[4]) {
    uint8x16_t c0 = vandq_u8(p, vdupq_n_u8(3));
    uint8x16_t c1 = vandq_u8(vshrq_n_u8(p, 2), vdupq_n_u8(3));
    uint8x16_t c2 = vandq_u8(vshrq_n_u8(p, 4), vdupq_n_u8(3));
    uint8x16_t c3 = vandq_u8(vshrq_n_u8(p, 6), vdupq_n_u8(3));
    uint8x16x2_t z01 = vzipq_u8(c0, c1);
    uint8x16x2_t z23 = vzipq_u8(c2, c3);
    uint16x8x2_t w0 = vzipq_u16(vreinterpretq_u16_u8(z01.val[0]),
                                vreinterpretq_u16_u8(z23.val[0]));
    uint16x8x2_t w1 = vzipq_u16(vreinterpretq_u16_u8(z01.val[1]),
                                vreinterpretq_u16_u8(z23.val[1]));
    uint8x16_t lut = vld1q_u8(kLut);
    o[0] = vqtbl1q_u8(lut, vreinterpretq_u8_u16(w0.val[0]));
    o[1] = vqtbl1q_u8(lut, vreinterpretq_u8_u16(w0.val[1]));
    o[2] = vqtbl1q_u8(lut, vreinterpretq_u8_u16(w1.val[0]));
    o[3] = vqtbl1q_u8(lut, vreinterpretq_u8_u16(w1.val[1]));
}

static int32_t tern_dot_scalar(const uint8_t *p, const int8_t *x, int n) {
    int32_t acc = 0;
    for (int i = 0; i < n; i++) {
        uint8_t c = (p[i >> 2] >> ((i & 3) * 2)) & 3;
        acc += (int32_t)tern_decode(c) * (int32_t)x[i];
    }
    return acc;
}

#if defined(__ARM_NEON)
static int32_t tern_dot_dotprod(const uint8_t *p, const int8_t *x, int n) {
    int32x4_t acc = vdupq_n_s32(0);
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        uint8x16_t packed = vld1q_u8(p + (i >> 2));
        uint8x16_t wv[4];
        tern_expand16(packed, wv);
        int8x16_t w0 = vreinterpretq_s8_u8(wv[0]);
        int8x16_t w1 = vreinterpretq_s8_u8(wv[1]);
        int8x16_t w2 = vreinterpretq_s8_u8(wv[2]);
        int8x16_t w3 = vreinterpretq_s8_u8(wv[3]);
#if defined(__ARM_FEATURE_DOTPROD)
        acc = vdotq_s32(acc, w0, vld1q_s8(x + i +  0));
        acc = vdotq_s32(acc, w1, vld1q_s8(x + i + 16));
        acc = vdotq_s32(acc, w2, vld1q_s8(x + i + 32));
        acc = vdotq_s32(acc, w3, vld1q_s8(x + i + 48));
#else
        int16x8_t s0 = vdupq_n_s16(0), s1 = vdupq_n_s16(0);
        s0 = vaddw_s8(s0, vget_low_s8(w0));  s1 = vaddw_s8(s1, vget_high_s8(w0));
        s0 = vaddw_s8(s0, vget_low_s8(w1));  s1 = vaddw_s8(s1, vget_high_s8(w1));
        s0 = vaddw_s8(s0, vget_low_s8(w2));  s1 = vaddw_s8(s1, vget_high_s8(w2));
        s0 = vaddw_s8(s0, vget_low_s8(w3));  s1 = vaddw_s8(s1, vget_high_s8(w3));
        int16x8_t x0 = vmovl_s8(vget_low_s8(vld1q_s8(x + i + 0)));
        int16x8_t x1 = vmovl_s8(vget_high_s8(vld1q_s8(x + i + 0)));
        (void)x0; (void)x1; /* not used: without dotprod we cannot mix cheaply */
        acc = vaddq_s32(acc, vmovl_s16(vget_low_s16(s0)));
        acc = vaddq_s32(acc, vmovl_s16(vget_high_s16(s0)));
        acc = vaddq_s32(acc, vmovl_s16(vget_low_s16(s1)));
        acc = vaddq_s32(acc, vmovl_s16(vget_high_s16(s1)));
#endif
    }
    int32_t r = vaddvq_s32(acc);
    for (; i < n; i++) {
        uint8_t c = (p[i >> 2] >> ((i & 3) * 2)) & 3;
        r += (int32_t)tern_decode(c) * (int32_t)x[i];
    }
    return r;
}

/* Pure add/sub: mask, select, widen-add. No multiply instructions. */
static int32_t tern_dot_addsub(const uint8_t *p, const int8_t *x, int n) {
    int32x4_t acc = vdupq_n_s32(0);
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        uint8x16_t packed = vld1q_u8(p + (i >> 2));
        uint8x16_t wv[4];
        tern_expand16(packed, wv);
        int16x8_t s0 = vdupq_n_s16(0), s1 = vdupq_n_s16(0);
        for (int k = 0; k < 4; k++) {
            int8x16_t xa = vld1q_s8(x + i + k * 16);
            /* Even lanes: (w==1)*x + (w==-1)*(-x); odd lanes likewise.
             * vbslq picks x or -x using the sign bit; zeros selected out. */
            uint8x16_t nz = vtstq_u8(wv[k], wv[k]);
            int8x16_t xneg = vnegq_s8(xa);
            uint8x16_t sign = vceqq_u8(vandq_u8(wv[k], vdupq_n_u8(2)), vdupq_n_u8(2));
            int8x16_t val = vbslq_s8(sign, xneg, xa);
            val = vandq_s8(val, vreinterpretq_s8_u8(nz));
            s0 = vaddw_s8(s0, vget_low_s8(val));
            s1 = vaddw_s8(s1, vget_high_s8(val));
        }
        acc = vaddq_s32(acc, vmovl_s16(vget_low_s16(s0)));
        acc = vaddq_s32(acc, vmovl_s16(vget_high_s16(s0)));
        acc = vaddq_s32(acc, vmovl_s16(vget_low_s16(s1)));
        acc = vaddq_s32(acc, vmovl_s16(vget_high_s16(s1)));
    }
    int32_t r = vaddvq_s32(acc);
    for (; i < n; i++) {
        uint8_t c = (p[i >> 2] >> ((i & 3) * 2)) & 3;
        r += (int32_t)tern_decode(c) * (int32_t)x[i];
    }
    return r;
}
#endif

int32_t tern_dot_i8(const uint8_t *packed, const int8_t *x, int n) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if (tern_kernel_mode != 2 && tern_kernel_mode != 1)
        return tern_dot_dotprod(packed, x, n);
    if (tern_kernel_mode == 1)
        return tern_dot_addsub(packed, x, n);
#endif
    return tern_dot_scalar(packed, x, n);
}

const char *tern_kernel_name(void) {
#if defined(__ARM_FEATURE_DOTPROD)
    if (tern_kernel_mode == 1) return "neon-addsub";
    if (tern_kernel_mode == 0) return "neon-dotprod";
#endif
    if (tern_kernel_mode == 1) return "scalar";
    return "scalar";
}

static float quantize_act(const float *x, int n, int8_t *xq) {
    float xmax = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > xmax) xmax = a;
    }
    float xs = xmax / 127.0f;
    if (xs < 1e-12f) xs = 1e-12f;
    for (int i = 0; i < n; i++) {
        float v = roundf(x[i] / xs);
        if (v > 127.0f) v = 127.0f;
        if (v < -127.0f) v = -127.0f;
        xq[i] = (int8_t)v;
    }
    return xs;
}

void tern_gemv(const tern_mat *m, const float *x, float *y) {
    int n = m->ncols;
    int8_t stackbuf[SPEL_MAX_DFF];
    int8_t *xq = stackbuf;
    if (n > SPEL_MAX_DFF) xq = (int8_t *)malloc((size_t)n);
    if (!xq) return;
    float xs = quantize_act(x, n, xq);
    for (int r = 0; r < m->nrows; r++) {
        int32_t acc = tern_dot_i8(m->packed + (size_t)r * m->packed_stride, xq, n);
        y[r] = (float)acc * m->scale[r] * xs;
    }
    if (xq != stackbuf) free(xq);
}

/* ------------------------------------------------------------- q8 int8 */

int q8_mat_init(q8_mat *m, int nrows, int ncols) {
    memset(m, 0, sizeof(*m));
    m->nrows = nrows;
    m->ncols = ncols;
    m->q = (int8_t *)calloc((size_t)nrows * ncols, 1);
    m->scale = (float *)calloc((size_t)nrows, sizeof(float));
    if (!m->q || !m->scale) { q8_mat_free(m); return -1; }
    for (int r = 0; r < nrows; r++) m->scale[r] = 1.0f;
    return 0;
}

void q8_mat_free(q8_mat *m) {
    if (!m) return;
    free(m->q); free(m->scale);
    memset(m, 0, sizeof(*m));
}

void q8_quantize_mat(q8_mat *m, const float *w) {
    for (int r = 0; r < m->nrows; r++) {
        const float *wr = w + (size_t)r * m->ncols;
        float amax = 0.0f;
        for (int i = 0; i < m->ncols; i++) {
            float a = fabsf(wr[i]);
            if (a > amax) amax = a;
        }
        float s = amax / 127.0f;
        if (s < 1e-8f) s = 1e-8f;
        m->scale[r] = s;
        int8_t *qr = m->q + (size_t)r * m->ncols;
        for (int i = 0; i < m->ncols; i++) {
            float v = roundf(wr[i] / s);
            if (v > 127.0f) v = 127.0f;
            if (v < -127.0f) v = -127.0f;
            qr[i] = (int8_t)v;
        }
    }
}

static int32_t q8_dot_i8(const int8_t *w, const int8_t *x, int n) {
#if defined(__ARM_FEATURE_DOTPROD)
    int32x4_t acc = vdupq_n_s32(0);
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        acc = vdotq_s32(acc, vld1q_s8(w + i),      vld1q_s8(x + i));
        acc = vdotq_s32(acc, vld1q_s8(w + i + 16), vld1q_s8(x + i + 16));
        acc = vdotq_s32(acc, vld1q_s8(w + i + 32), vld1q_s8(x + i + 32));
        acc = vdotq_s32(acc, vld1q_s8(w + i + 48), vld1q_s8(x + i + 48));
    }
    int32_t r = vaddvq_s32(acc);
    for (; i < n; i++) r += (int32_t)w[i] * (int32_t)x[i];
    return r;
#else
    int32_t r = 0;
    for (int i = 0; i < n; i++) r += (int32_t)w[i] * (int32_t)x[i];
    return r;
#endif
}

void q8_gemv(const q8_mat *m, const float *x, float *y) {
    int n = m->ncols;
    int8_t stackbuf[SPEL_MAX_DFF];
    int8_t *xq = stackbuf;
    if (n > SPEL_MAX_DFF) xq = (int8_t *)malloc((size_t)n);
    if (!xq) return;
    float xs = quantize_act(x, n, xq);
    for (int r = 0; r < m->nrows; r++) {
        int32_t acc = q8_dot_i8(m->q + (size_t)r * n, xq, n);
        y[r] = (float)acc * m->scale[r] * xs;
    }
    if (xq != stackbuf) free(xq);
}

/* ------------------------------------------------------- draft GEMV pool */

#define SPEL_POOL_SPIN 20000

static void *pool_worker(void *arg) {
    spel_gemv_pool *p = (spel_gemv_pool *)arg;
    int seen = 0;
    for (;;) {
        int got = 0;
        for (int i = 0; i < SPEL_POOL_SPIN; i++) {
            int g = atomic_load_explicit(&p->work_gen, memory_order_acquire);
            if (g != seen) { seen = g; got = 1; break; }
            if (atomic_load_explicit(&p->stop_flag, memory_order_acquire))
                return NULL;
            if ((i & 63) == 63) sched_yield();
        }
        if (!got) {
            pthread_mutex_lock(&p->mu);
            while (!p->stop && p->generation == seen)
                pthread_cond_wait(&p->start_cv, &p->mu);
            if (p->stop) {
                pthread_mutex_unlock(&p->mu);
                break;
            }
            seen = p->generation;
            pthread_mutex_unlock(&p->mu);
        }

        for (;;) {
            int r0 = atomic_fetch_add(&p->job.next_row, p->job.block);
            if (r0 >= p->job.rows) break;
            int r1 = r0 + p->job.block;
            if (r1 > p->job.rows) r1 = p->job.rows;
            if (p->job.kind == 0) {
                for (int r = r0; r < r1; r++) {
                    int32_t acc = q8_dot_i8(p->job.w + (size_t)r * p->job.row_stride,
                                            p->job.xq, p->job.cols);
                    p->job.y[r] = (float)acc * p->job.scales[r] * p->job.xs;
                }
            } else {
                for (int r = r0; r < r1; r++) {
                    int32_t acc = tern_dot_i8(p->job.packed + (size_t)r * p->job.row_stride,
                                              p->job.xq, p->job.cols);
                    p->job.y[r] = (float)acc * p->job.scales[r] * p->job.xs;
                }
            }
        }

        atomic_fetch_add_explicit(&p->job.done, 1, memory_order_release);
        pthread_mutex_lock(&p->mu);
        pthread_cond_broadcast(&p->done_cv);
        pthread_mutex_unlock(&p->mu);
    }
    return NULL;
}

spel_gemv_pool *spel_pool_create(int nthreads) {
    if (nthreads <= 1) return NULL;
    if (nthreads > SPEL_POOL_MAX_THREADS) nthreads = SPEL_POOL_MAX_THREADS;
    spel_gemv_pool *p = (spel_gemv_pool *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    atomic_init(&p->job.next_row, 0);
    atomic_init(&p->job.done, 0);
    atomic_init(&p->work_gen, 0);
    atomic_init(&p->stop_flag, 0);
    if (pthread_mutex_init(&p->mu, NULL)) { free(p); return NULL; }
    if (pthread_cond_init(&p->start_cv, NULL)) {
        pthread_mutex_destroy(&p->mu); free(p); return NULL;
    }
    if (pthread_cond_init(&p->done_cv, NULL)) {
        pthread_cond_destroy(&p->start_cv);
        pthread_mutex_destroy(&p->mu); free(p); return NULL;
    }
    for (int i = 0; i < nthreads; i++) {
        if (pthread_create(&p->workers[i], NULL, pool_worker, p)) {
            spel_pool_destroy(p);
            return NULL;
        }
        p->nthreads++;
    }
    return p;
}

void spel_pool_destroy(spel_gemv_pool *p) {
    if (!p) return;
    atomic_store_explicit(&p->stop_flag, 1, memory_order_release);
    pthread_mutex_lock(&p->mu);
    p->stop = 1;
    pthread_cond_broadcast(&p->start_cv);
    pthread_mutex_unlock(&p->mu);
    for (int i = 0; i < p->nthreads; i++) pthread_join(p->workers[i], NULL);
    pthread_cond_destroy(&p->done_cv);
    pthread_cond_destroy(&p->start_cv);
    pthread_mutex_destroy(&p->mu);
    free(p);
}

static void pool_dispatch(spel_gemv_pool *p, int kind,
                          const int8_t *w, const uint8_t *packed,
                          const float *scales, const int8_t *xq,
                          float *y, int rows, int cols, int row_stride,
                          float xs) {
    if (!p || p->nthreads <= 1) return;
    int nt = p->nthreads;
    int block = (rows + nt * 6 - 1) / (nt * 6);
    if (block < 1) block = 1;
    pthread_mutex_lock(&p->mu);
    p->job.kind = kind;
    p->job.w = w;
    p->job.packed = packed;
    p->job.scales = scales;
    p->job.xq = xq;
    p->job.y = y;
    p->job.rows = rows;
    p->job.cols = cols;
    p->job.row_stride = row_stride;
    p->job.xs = xs;
    p->job.block = block;
    atomic_store(&p->job.next_row, 0);
    atomic_store(&p->job.done, 0);
    p->generation++;
    atomic_store_explicit(&p->work_gen, p->generation, memory_order_release);
    pthread_cond_broadcast(&p->start_cv);
    pthread_mutex_unlock(&p->mu);

    /* GEMV dispatches are close together; a short acquire spin avoids paying
     * a condvar wake for every one of the five projections in each layer. */
    for (int i = 0; i < SPEL_POOL_SPIN; i++) {
        if (atomic_load_explicit(&p->job.done, memory_order_acquire) >= nt)
            return;
        if ((i & 63) == 63) sched_yield();
    }
    pthread_mutex_lock(&p->mu);
    while (atomic_load_explicit(&p->job.done, memory_order_acquire) < nt)
        pthread_cond_wait(&p->done_cv, &p->mu);
    pthread_mutex_unlock(&p->mu);
}

void spel_pool_q8_gemv(spel_gemv_pool *p, const q8_mat *m,
                       const float *x, float *y) {
    if (!p || p->nthreads <= 1) { q8_gemv(m, x, y); return; }
    int n = m->ncols;
    int8_t stackbuf[SPEL_MAX_DFF];
    int8_t *xq = stackbuf;
    if (n > SPEL_MAX_DFF) xq = (int8_t *)malloc((size_t)n);
    if (!xq) return;
    float xs = quantize_act(x, n, xq);
    pool_dispatch(p, 0, m->q, NULL, m->scale, xq, y,
                  m->nrows, n, n, xs);
    if (xq != stackbuf) free(xq);
}

void spel_pool_tern_gemv(spel_gemv_pool *p, const tern_mat *m,
                         const float *x, float *y) {
    if (!p || p->nthreads <= 1) { tern_gemv(m, x, y); return; }
    int n = m->ncols;
    int8_t stackbuf[SPEL_MAX_DFF];
    int8_t *xq = stackbuf;
    if (n > SPEL_MAX_DFF) xq = (int8_t *)malloc((size_t)n);
    if (!xq) return;
    float xs = quantize_act(x, n, xq);
    pool_dispatch(p, 1, NULL, m->packed, m->scale, xq, y,
                  m->nrows, n, m->packed_stride, xs);
    if (xq != stackbuf) free(xq);
}
