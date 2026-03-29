#ifndef TURBOQUANT_H
#define TURBOQUANT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Version */
#define TQ_VERSION_MAJOR 0
#define TQ_VERSION_MINOR 1
#define TQ_VERSION_PATCH 0

/* Error codes */
#define TQ_OK               0
#define TQ_ERR_INVALID_DIM  -1
#define TQ_ERR_OUT_OF_MEMORY -2
#define TQ_ERR_INVALID_DATA -3
#define TQ_ERR_NULL_PTR     -4

/* Opaque engine handle */
typedef void TqEngine;

/* Buffer for compressed data (caller must free via tq_free_buffer) */
typedef struct {
    uint8_t *data;
    size_t len;
} TqBuffer;

/* Runtime version check: returns (major << 16) | (minor << 8) | patch */
uint32_t tq_version(void);

/* Create engine. dim must be even and > 0. Returns NULL on failure. */
TqEngine *tq_engine_create(uint32_t dim, uint32_t seed);

/* Destroy engine. Null-safe (no-op on NULL). */
void tq_engine_destroy(TqEngine *engine);

/* Get engine dimension. Returns 0 on NULL. */
uint32_t tq_engine_dim(TqEngine *engine);

/* Compress float32 vector of length engine->dim.
 * Writes compressed data to out. Caller must call tq_free_buffer(out).
 * Returns TQ_OK on success, negative error code on failure. */
int tq_encode(TqEngine *engine, const float *data, TqBuffer *out);

/* Decompress into caller-provided float32 buffer of length engine->dim.
 * Returns TQ_OK on success, negative error code on failure. */
int tq_decode(TqEngine *engine, const uint8_t *compressed, size_t compressed_len,
              float *out);

/* Estimate dot product without decompression.
 * Returns 0.0 on error or NULL inputs. */
float tq_dot(TqEngine *engine, const float *query,
             const uint8_t *compressed, size_t compressed_len);

/* Free buffer returned by tq_encode. Null-safe. */
void tq_free_buffer(TqBuffer *buf);

#ifdef __cplusplus
}
#endif

#endif /* TURBOQUANT_H */
