/* notebank.c — chunked key-distinctiveness writer, compact native addresses,
 * per-channel similarity scoring (with a vote diagnostic) and replay selection.
 *
 * Writer input is the captured native Q/K of every recurrent layer-head
 * (channel) at every context position. Addresses are 16 BF16 coordinates
 * selected by a fixed sparse projection, as in the NoteBank sketch.
 */
#include "spel.h"

void nb_bf16(float f, uint16_t *out) {
    uint32_t u;
    memcpy(&u, &f, 4);
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    *out = (uint16_t)(u >> 16);
}

float nb_bf16_to_f(uint16_t v) {
    uint32_t u = (uint32_t)v << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

void nb_bank_init(nb_bank *b, const spel_cfg *cfg, int chunk, int window,
                  int budget, int per_chunk, int addr_dim, uint64_t seed) {
    memset(b, 0, sizeof(*b));
    b->chunk = chunk;
    b->window = window;
    b->budget = budget;
    b->per_chunk = per_chunk;
    b->addr_dim = addr_dim;
    b->n_channels = cfg->n_channels;
    b->dh = cfg->head_dim;
    b->seed = seed ? seed : 0xC0FFEEULL;
    b->proj_idx = (int *)calloc((size_t)b->n_channels * addr_dim, sizeof(int));
    b->rho = (uint16_t *)calloc((size_t)NB_MAX_NOTES * b->n_channels * addr_dim,
                                sizeof(uint16_t));
    uint64_t rng = b->seed;
    for (int c = 0; c < b->n_channels; c++) {
        int *p = b->proj_idx + (size_t)c * addr_dim;
        for (int j = 0; j < addr_dim; j++) {
            int cand, dup;
            do {
                cand = (int)(spel_rng_next(&rng) % (uint64_t)b->dh);
                dup = 0;
                for (int k = 0; k < j; k++) if (p[k] == cand) dup = 1;
            } while (dup);
            p[j] = cand;
        }
    }
}

void nb_bank_free(nb_bank *b) {
    if (!b) return;
    for (int i = 0; i < b->n_notes; i++) free(b->notes[i].tokens);
    free(b->proj_idx);
    free(b->rho);
    memset(b, 0, sizeof(*b));
}

static inline void normalize(float *v, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += v[i] * v[i];
    float r = (s > 1e-20f) ? 1.0f / sqrtf(s) : 0.0f;
    for (int i = 0; i < n; i++) v[i] *= r;
}

int nb_write(nb_bank *b, const spel_cfg *cfg, const int *ctx, int T,
             const float *cap_k, const float *cap_q) {
    const int C = b->n_channels, dh = b->dh;
    if (T <= 0 || C <= 0 || T > cfg->max_seq) return b->n_notes;
    const int chunk = b->chunk > 0 ? b->chunk : 512;
    const int win = b->window > 0 ? b->window : 48;
    const int budget = b->budget > 0 ? b->budget : 64;
    const int per = b->per_chunk > 0 ? b->per_chunk : 2;
    const int ad = b->addr_dim;

    float *dist = (float *)calloc((size_t)T * C, sizeof(float));
    float *avg  = (float *)calloc((size_t)T, sizeof(float));
    float *khat = (float *)calloc((size_t)dh, sizeof(float));
    unsigned char *used = (unsigned char *)calloc((size_t)T, 1);
    if (!dist || !avg || !khat || !used) {
        free(dist); free(avg); free(khat); free(used);
        return b->n_notes;
    }

    for (int t0 = 0; t0 < T; t0 += chunk) {
        int t1 = t0 + chunk; if (t1 > T) t1 = T;
        int len = t1 - t0;
        if (len < 4) break;
        /* chunk mean of l2-normalised native keys, per channel */
        float *mu = (float *)calloc((size_t)C * dh, sizeof(float));
        for (int c = 0; c < C; c++) {
            float *m = mu + (size_t)c * dh;
            for (int t = t0; t < t1; t++) {
                const float *k = cap_k + ((size_t)t * C + c) * dh;
                memcpy(khat, k, (size_t)dh * sizeof(float));
                normalize(khat, dh);
                for (int j = 0; j < dh; j++) m[j] += khat[j];
            }
            normalize(m, dh);
        }
        for (int t = t0; t < t1; t++) {
            float acc = 0.0f;
            for (int c = 0; c < C; c++) {
                const float *k = cap_k + ((size_t)t * C + c) * dh;
                const float *m = mu + (size_t)c * dh;
                memcpy(khat, k, (size_t)dh * sizeof(float));
                normalize(khat, dh);
                float d = 1.0f;
                for (int j = 0; j < dh; j++) d -= khat[j] * m[j];
                dist[(size_t)t * C + c] = d;
                acc += d;
            }
            avg[t] = acc / (float)C;
        }
        free(mu);

        int win_len = win < len ? win : len;
        int span_len = budget < len ? budget : len;
        int pick = 0;
        while (pick < per) {
            int best = -1;
            float best_s = -1e30f;
            int nwin = len - win_len + 1;
            for (int w = 0; w < nwin; w++) {
                int center = t0 + w + win_len / 2;
                int s0 = center - span_len / 2;
                if (s0 < t0) s0 = t0;
                if (s0 + span_len > t1) s0 = t1 - span_len;
                int overlap = 0;
                for (int t = s0; t < s0 + span_len; t++) if (used[t]) { overlap = 1; break; }
                if (overlap) continue;
                float s = 0.0f;
                for (int j = 0; j < win_len; j++) s += avg[t0 + w + j];
                if (s > best_s) { best_s = s; best = w; }
            }
            if (best < 0) break;
            int center = t0 + best + win_len / 2;
            int s0 = center - span_len / 2;
            if (s0 < t0) s0 = t0;
            if (s0 + span_len > t1) s0 = t1 - span_len;
            int s1 = s0 + span_len;
            if (b->n_notes >= NB_MAX_NOTES) break;
            nb_note *n = &b->notes[b->n_notes];
            n->chunk = t0 / chunk;
            n->start = s0;
            n->n_tokens = s1 - s0;
            n->tokens = (int *)malloc((size_t)n->n_tokens * sizeof(int));
            memcpy(n->tokens, ctx + s0, (size_t)n->n_tokens * sizeof(int));
            n->used = 1;
            uint16_t *rho = b->rho + (size_t)b->n_notes * C * ad;
            for (int c = 0; c < C; c++) {
                int tstar = s0;
                float bestd = -1e30f;
                for (int t = s0; t < s1; t++) {
                    float d = dist[(size_t)t * C + c];
                    if (d > bestd) { bestd = d; tstar = t; }
                }
                const float *q = cap_q + ((size_t)tstar * C + c) * dh;
                for (int j = 0; j < ad; j++)
                    nb_bf16(q[b->proj_idx[(size_t)c * ad + j]],
                            &rho[(size_t)c * ad + j]);
            }
            for (int t = s0; t < s1; t++) used[t] = 1;
            b->n_notes++;
            pick++;
        }
    }
    free(dist); free(avg); free(khat); free(used);
    return b->n_notes;
}

int nb_address(const nb_bank *b, const spel_cfg *cfg, int Q, const float *cap_q,
               int q_base, int R, int *sel) {
    const int C = b->n_channels, dh = b->dh, ad = b->addr_dim;
    if (b->n_notes <= 0 || Q <= 0 || R <= 0) return 0;
    int *votes = (int *)calloc((size_t)b->n_notes, sizeof(int));
    float *rho_hat = (float *)malloc((size_t)ad * sizeof(float));
    float *u = (float *)malloc((size_t)ad * sizeof(float));
    float *best = (float *)malloc((size_t)b->n_notes * sizeof(float));
    float *score = (float *)calloc((size_t)b->n_notes, sizeof(float));
    if (!votes || !rho_hat || !u || !best || !score) {
        free(votes); free(rho_hat); free(u); free(best); free(score);
        return 0;
    }
    for (int c = 0; c < C; c++) {
        for (int i = 0; i < b->n_notes; i++) best[i] = -2.0f;
        for (int i = 0; i < b->n_notes; i++) {
            const uint16_t *rp = b->rho + ((size_t)i * C + c) * ad;
            for (int j = 0; j < ad; j++) rho_hat[j] = nb_bf16_to_f(rp[j]);
            normalize(rho_hat, ad);
            for (int q = 0; q < Q; q++) {
                int pos = q_base + q;
                if (pos < 0 || pos >= cfg->max_seq) continue;
                const float *qp = cap_q + ((size_t)pos * C + c) * dh;
                for (int j = 0; j < ad; j++) u[j] = qp[b->proj_idx[(size_t)c * ad + j]];
                normalize(u, ad);
                float s = 0.0f;
                for (int j = 0; j < ad; j++) s += u[j] * rho_hat[j];
                if (s > best[i]) best[i] = s;
            }
        }
        int arg = 0;
        for (int i = 1; i < b->n_notes; i++) if (best[i] > best[arg]) arg = i;
        votes[arg]++;
        for (int i = 0; i < b->n_notes; i++) score[i] += best[i];
    }
    /* Rank by summed channel similarity. The paper's integer channel vote is
     * kept as a diagnostic; with an untrained model the magnitudes are much
     * more discriminative than win counts. */
    int n = R < b->n_notes ? R : b->n_notes;
    for (int r = 0; r < n; r++) {
        int arg = -1;
        for (int i = 0; i < b->n_notes; i++) {
            if (score[i] <= -1e29f) continue;
            if (arg < 0 || score[i] > score[arg]) arg = i;
        }
        sel[r] = arg;
        if (arg >= 0) score[arg] = -1e30f;
    }
    free(votes); free(rho_hat); free(u); free(best); free(score);
    return n;
}

size_t spel_note_bytes(const nb_bank *b) {
    size_t n = (size_t)b->n_channels * b->addr_dim * sizeof(int);
    for (int i = 0; i < b->n_notes; i++) {
        n += (size_t)b->notes[i].n_tokens * sizeof(int);
        n += (size_t)b->n_channels * b->addr_dim * sizeof(uint16_t);
    }
    return n;
}
