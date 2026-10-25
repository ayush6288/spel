/* spel.h — hybrid linear-attention toy model + ternary GEMV + speculative runtime.
 *
 * This is a self-contained research harness, not a wrapper around llama.cpp.
 * Everything here is deterministic and testable on a laptop:
 *   - ternary {-1,0,+1} packing, NEON dotprod and add/sub GEMV kernels
 *   - hybrid model: Gated-DeltaNet-style recurrent layers + RoPE attention
 *   - O(1) transactional commit/rollback for KV cache and recurrent state
 *   - NoteBank-style chunked writer, address matching, replay demo
 *   - greedy speculative decoding that must be bit-identical to AR decoding
 */
#ifndef SPEL_H
#define SPEL_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define SPEL_MAX_LAYERS   16
#define SPEL_MAX_HEADS    16
#define SPEL_MAX_D        512
#define SPEL_MAX_DFF      1536
#define SPEL_MAX_SEQ      4096
#define SPEL_MAX_VOCAB    1024
#define SPEL_MAX_BATCH    16
#define SPEL_MAX_CHANNELS (SPEL_MAX_LAYERS * SPEL_MAX_HEADS)
#define SPEL_XH_RING      32
#define SPEL_ROPE_THETA   10000.0f

/* ------------------------------------------------------------------ cfg */

typedef struct {
    int d_model, n_layers, n_heads, head_dim, d_ff, vocab, max_seq;
    int attn_period;                 /* every attn_period-th layer is attention */
    int conv_k;                      /* short conv taps in recurrent layers */
    int spec_k;                      /* speculative depth K */
    int n_linear, n_attn, n_channels;/* n_channels = n_linear * n_heads */
    int is_linear[SPEL_MAX_LAYERS];
    int lin_index[SPEL_MAX_LAYERS];  /* -1 for attention layers */
    int attn_index[SPEL_MAX_LAYERS];
} spel_cfg;

int  spel_cfg_init(spel_cfg *c, int d, int layers, int heads, int dff,
                   int vocab, int max_seq, int attn_period, int spec_k);
int  spel_cfg_validate(const spel_cfg *c);

/* ------------------------------------------------------------- ternary */

typedef struct {
    uint8_t *packed;   /* nrows * packed_stride bytes */
    float   *scale;    /* nrows */
    int      nrows, ncols, packed_stride;
} tern_mat;

size_t tern_packed_stride(int ncols);
int    tern_mat_init(tern_mat *m, int nrows, int ncols);
void   tern_mat_free(tern_mat *m);
void   tern_quantize_row(const float *w, int ncols, uint8_t *packed, float *scale);
void   tern_quantize_mat(tern_mat *m, const float *w); /* w row-major [nrows][ncols] */
void   tern_dequant_row(const uint8_t *packed, float scale, int ncols, float *out);
int32_t tern_dot_i8(const uint8_t *packed, const int8_t *x, int ncols);
void   tern_gemv(const tern_mat *m, const float *x, float *y);
/* Kernel selection: 0 = auto (dotprod if available), 1 = add/sub, 2 = scalar */
extern int tern_kernel_mode;
const char *tern_kernel_name(void);

/* Fast math: when 1, batched (M>1) FP32 projections use Accelerate
 * cblas_sgemm. This changes floating-point summation order, so results are
 * token-equivalent in practice but not bit-identical to the strict path.
 * Default 0: strict, bit-identical batched vs sequential. */
extern int spel_use_accelerate;

/* ------------------------------------------------------------- q8 int8 */
/* Symmetric per-row int8 weights for the cheap draft's non-MLP projections.
 * Activations are quantized dynamically per GEMV, exactly like tern_gemv. */

typedef struct {
    int8_t *q;      /* nrows * ncols */
    float  *scale;  /* nrows */
    int     nrows, ncols;
} q8_mat;

int  q8_mat_init(q8_mat *m, int nrows, int ncols);
void q8_mat_free(q8_mat *m);
void q8_quantize_mat(q8_mat *m, const float *w);
void q8_gemv(const q8_mat *m, const float *x, float *y);

/* --------------------------------------------------------------- model */

typedef struct {
    float *wqkv;   /* [3d][d] */
    float *wo;     /* [d][d]  */
    float *w1;     /* [dff][d] */
    float *w3;     /* [dff][d] */
    float *w2;     /* [d][dff] */
    float *conv;   /* [conv_k][d] (linear only) */
    float *wa, *wb;/* [heads] (linear only) */
    float *n1, *n2;
    tern_mat t1, t2, t3;  /* optional ternary MLP (draft) */
    q8_mat q_wqkv, q_wo;  /* optional int8 projections (draft) */
} spel_layer;

typedef struct spel_model {
    spel_cfg cfg;
    int owns_params;        /* target: params holds shared + layer tensors */
    int owns_layer_params;  /* draft: params holds only this model's layers */
    float *params;
    size_t n_params;
    float *emb, *wout, *normf;      /* [vocab][d], [vocab][d], [d] */
    float *rope_cos, *rope_sin;     /* [max_seq][dh/2] */
    spel_layer layer[SPEL_MAX_LAYERS];

    /* draft provenance (serialized in SPELW001) */
    int n_draft_layers;
    int source_map[4];
    struct spel_model *draft;       /* non-owning, for save-by-pair convenience */

    /* optional golden footer from SPELW001/SPELV001 */
    int golden_len;
    int32_t golden_tokens[16];
    float *golden_target_logits;    /* [golden_len][vocab] */
    float *golden_draft_logits;

    /* runtime state (per model instance) */
    float *kv_k[SPEL_MAX_LAYERS], *kv_v[SPEL_MAX_LAYERS];
    float *S[SPEL_MAX_LAYERS];      /* (K+1) * heads * dh * dh */
    float *xh[SPEL_MAX_LAYERS];     /* SPEL_XH_RING * d */
    float *buf;                     /* forward scratch */
    size_t buf_floats;
    float *last_logits;             /* [vocab] */
    int pos;                        /* number of tokens processed */
    int is_ternary;

    /* optional activation capture for NoteBank (linear layers only) */
    float *cap_q, *cap_k;
    int cap_enabled;
    int cap_len;                    /* allocated positions */
    uint64_t rng;
} spel_model;

spel_model *spel_model_new(const spel_cfg *cfg, uint64_t seed);
spel_model *spel_model_new_draft(const spel_model *target);
/* Draft from a subset of target layers. layer_map[i] is the source layer for
 * draft layer i. The draft owns independent FP32 copies of those layers'
 * parameters (initialized from the target) and quantizes its q8 projections
 * and ternary MLPs. Embedding, final norm, LM head and RoPE are shared. */
spel_model *spel_model_new_draft_subset(const spel_model *target,
                                        const int *layer_map, int n_layers);
void        spel_model_free(spel_model *m);
void        spel_model_reset(spel_model *m);
/* Non-owning link used by spel_save_weights(m, path). */
void        spel_model_attach_draft(spel_model *target, spel_model *draft);

/* ------------------------------------------------ binary weights SPELW001 */
/* Header: 8-byte magic "SPELW001", u32 version=1, d_model, n_layers, n_heads,
 * head_dim, d_ff, vocab, attn_period, n_draft_layers, draft_map[4],
 * has_golden_footer. All integers/floats are little-endian. Tensor order:
 *   emb, target layer 0..N-1, normf, wout,
 *   4 draft layers, optional SPELV001 footer.
 * Per layer: n1, wqkv, wo, [conv, wa, wb if linear], n2, w1, w3, w2. */
#define SPEL_WEIGHTS_MAGIC "SPELW001"
#define SPEL_GOLDEN_MAGIC  "SPELV001"
#define SPEL_GOLDEN_LEN    16

int spel_save_weights(const spel_model *m, const char *path);
int spel_load_weights(spel_model *m, const char *path);
/* Explicit target+draft variants used by the CLI/bench loader. */
int spel_save_weights_pair(const spel_model *target, const spel_model *draft,
                           const char *path);
int spel_load_weights_pair(spel_model *target, spel_model *draft,
                           const char *path);
/* Load a pair, run the SPELV001 golden tokens, print max-abs and argmax
 * agreement. Returns 0 when target argmax is 100% and max_diff < 1e-3. */
int spel_verify_weights(spel_model *target, spel_model *draft, const char *path);

/* Capture buffers are indexed by absolute position: [pos][channel][dh]. */
int  spel_capture_alloc(spel_model *m, int max_pos);
void spel_capture_free(spel_model *m);

/* Forward M contiguous tokens starting at m->pos. logits may be NULL.
 * M may be up to cfg->spec_k + 1 (the deferred-correction batch). */
void spel_forward(spel_model *m, const int *tokens, int M, float *logits);
int  spel_argmax(const float *v, int n);

/* -------------------------------------------------------- checkpoints */

typedef struct spel_ckpt spel_ckpt;
spel_ckpt *spel_ckpt_new(const spel_model *m);
void       spel_ckpt_save(spel_ckpt *c, const spel_model *m);
void       spel_ckpt_load(spel_model *m, const spel_ckpt *c);
void       spel_ckpt_free(spel_ckpt *c);

/* ----------------------------------------------------------- note bank */

#define NB_MAX_NOTES 96
#define NB_RANK_SIM   0   /* summed per-channel cosine similarity (default) */
#define NB_RANK_VOTES 1   /* one integer vote per channel, paper-style */

typedef struct {
    int chunk;
    int start;         /* token start in source context */
    int n_tokens;
    int *tokens;
    int used;
} nb_note;

typedef struct {
    int chunk;         /* tokens per chunk (paper: 512) */
    int window;        /* scorer window (paper: 48) */
    int budget;        /* span budget  (paper: 64) */
    int per_chunk;     /* notes per chunk (paper: 2) */
    int addr_dim;      /* 16 */
    int rank_mode;     /* NB_RANK_SIM or NB_RANK_VOTES */
    int n_notes;
    int n_channels, dh;
    int *proj_idx;     /* [n_channels][addr_dim] coordinates */
    uint16_t *rho;     /* [n_notes][n_channels][addr_dim] bf16 */
    nb_note notes[NB_MAX_NOTES];
    uint64_t seed;
} nb_bank;

void nb_bank_init(nb_bank *b, const spel_cfg *cfg, int chunk, int window,
                  int budget, int per_chunk, int addr_dim, uint64_t seed);
void nb_bank_free(nb_bank *b);
/* cap_k/cap_q must be populated by a capture-enabled prefill over ctx. */
int  nb_write(nb_bank *b, const spel_cfg *cfg, const int *ctx, int T,
              const float *cap_k, const float *cap_q);
/* cap_q at absolute positions [q_base, q_base+Q). Returns notes selected. */
int  nb_address(const nb_bank *b, const spel_cfg *cfg, int Q, const float *cap_q,
                int q_base, int R, int *sel);
void nb_bf16(float f, uint16_t *out);
float nb_bf16_to_f(uint16_t v);

/* ------------------------------------------------------ speculative run */

typedef struct {
    int K;
    int governor;          /* enable dynamic governor */
    double alpha_threshold;/* default 0.30 */
    int window;            /* acceptance rolling window */
    /* Benchmark-only. If >= 0, ignore real draft/target matches and force the
     * accepted-prefix length to round(alpha*K) on average. The draft and target
     * forwards still execute normally, so this measures systems cost at a
     * chosen acceptance rate; it is not real speculative decoding. */
    double force_alpha;
} spel_spec_cfg;

typedef struct {
    long rounds, proposed, accepted, emitted, corr;
    long ar_rounds, spec_rounds, governor_trips;
    double alpha;          /* accepted / proposed */
    double tokens_per_round;
} spel_spec_stats;

void spel_spec_cfg_default(spel_spec_cfg *s, int K);
/* Both models must be at the same pos and hold last_logits for that pos. */
void spel_spec_generate(spel_model *target, spel_model *draft,
                        int n_gen, int *out, const spel_spec_cfg *sc,
                        spel_spec_stats *st);

/* ------------------------------------------------------------- utils */

uint64_t spel_rng_next(uint64_t *s);
double   spel_rng_double(uint64_t *s);
float    spel_rng_normal(uint64_t *s);
double   spel_now_ms(void);
size_t   spel_kv_bytes(const spel_cfg *c);
size_t   spel_note_bytes(const nb_bank *b);

#endif
