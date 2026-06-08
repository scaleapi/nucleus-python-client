#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#define DEDUP_THRESHOLD_MAX 64
#define INDEX_MAX_THRESHOLD 11
#define PARTITION_COUNT 2
#define CHUNK_COUNT 6
#define ROTATED_PARTITION_BITS 8

static const int CHUNK_BITS[CHUNK_COUNT] = {11, 11, 11, 11, 10, 10};

typedef struct {
    Py_ssize_t *values;
    Py_ssize_t size;
    Py_ssize_t capacity;
} Bucket;

typedef struct {
    Bucket *buckets;
    Py_ssize_t bucket_count;
} ChunkIndex;

typedef struct {
    int threshold;
    int chunk_radius;
    uint64_t *hashes;
    uint8_t *candidate_marks;
    Py_ssize_t kept_count;
    Py_ssize_t hash_capacity;
    Py_ssize_t *touched_indexes;
    Py_ssize_t touched_count;
    Py_ssize_t touched_capacity;
    ChunkIndex indexes[PARTITION_COUNT][CHUNK_COUNT];
} HammingIndex;

static int popcount64(uint64_t value) {
#if defined(_MSC_VER)
    return (int)__popcnt64(value);
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(value);
#else
    int count = 0;
    while (value != 0) {
        value &= value - 1;
        count++;
    }
    return count;
#endif
}

static uint64_t rotate_right_64(uint64_t value, int bits) {
    return (value >> bits) | (value << (64 - bits));
}

static void partition_chunks(
    uint64_t phash_value, int partition_index, uint64_t chunks[CHUNK_COUNT]
) {
    int chunk_index;
    int shift = 64;

    if (partition_index == 1) {
        phash_value = rotate_right_64(phash_value, ROTATED_PARTITION_BITS);
    }

    for (chunk_index = 0; chunk_index < CHUNK_COUNT; chunk_index++) {
        shift -= CHUNK_BITS[chunk_index];
        chunks[chunk_index] =
            (phash_value >> shift) &
            ((((uint64_t)1) << CHUNK_BITS[chunk_index]) - 1);
    }
}

static int append_py_ssize(
    Py_ssize_t **values,
    Py_ssize_t *size,
    Py_ssize_t *capacity,
    Py_ssize_t value
) {
    Py_ssize_t new_capacity;
    Py_ssize_t *new_values;

    if (*size == *capacity) {
        new_capacity = *capacity == 0 ? 4 : *capacity * 2;
        if (new_capacity < *capacity) {
            PyErr_NoMemory();
            return -1;
        }
        new_values = (Py_ssize_t *)realloc(
            *values, (size_t)new_capacity * sizeof(Py_ssize_t)
        );
        if (new_values == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        *values = new_values;
        *capacity = new_capacity;
    }

    (*values)[*size] = value;
    *size += 1;
    return 0;
}

static int bucket_append(Bucket *bucket, Py_ssize_t kept_index) {
    return append_py_ssize(
        &bucket->values, &bucket->size, &bucket->capacity, kept_index
    );
}

static int touched_append(HammingIndex *index, Py_ssize_t kept_index) {
    return append_py_ssize(
        &index->touched_indexes,
        &index->touched_count,
        &index->touched_capacity,
        kept_index
    );
}

static void clear_candidate_marks(HammingIndex *index) {
    Py_ssize_t i;

    for (i = 0; i < index->touched_count; i++) {
        index->candidate_marks[index->touched_indexes[i]] = 0;
    }
    index->touched_count = 0;
}

static int hamming_index_init(
    HammingIndex *index, Py_ssize_t hash_capacity, int threshold
) {
    int partition_index;
    int chunk_index;
    Py_ssize_t bucket_count;
    Py_ssize_t allocation_capacity = hash_capacity > 0 ? hash_capacity : 1;

    memset(index, 0, sizeof(*index));
    index->threshold = threshold;
    index->chunk_radius = threshold / CHUNK_COUNT;
    index->hash_capacity = allocation_capacity;
    index->hashes = (uint64_t *)calloc(
        (size_t)allocation_capacity, sizeof(uint64_t)
    );
    index->candidate_marks = (uint8_t *)calloc(
        (size_t)allocation_capacity, sizeof(uint8_t)
    );
    if (index->hashes == NULL || index->candidate_marks == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    for (partition_index = 0; partition_index < PARTITION_COUNT;
         partition_index++) {
        for (chunk_index = 0; chunk_index < CHUNK_COUNT; chunk_index++) {
            bucket_count = ((Py_ssize_t)1) << CHUNK_BITS[chunk_index];
            index->indexes[partition_index][chunk_index].bucket_count =
                bucket_count;
            index->indexes[partition_index][chunk_index].buckets =
                (Bucket *)calloc((size_t)bucket_count, sizeof(Bucket));
            if (index->indexes[partition_index][chunk_index].buckets == NULL) {
                PyErr_NoMemory();
                return -1;
            }
        }
    }

    return 0;
}

static void hamming_index_free(HammingIndex *index) {
    int partition_index;
    int chunk_index;
    Py_ssize_t bucket_index;
    ChunkIndex *chunk_index_ptr;

    for (partition_index = 0; partition_index < PARTITION_COUNT;
         partition_index++) {
        for (chunk_index = 0; chunk_index < CHUNK_COUNT; chunk_index++) {
            chunk_index_ptr = &index->indexes[partition_index][chunk_index];
            if (chunk_index_ptr->buckets == NULL) {
                continue;
            }
            for (bucket_index = 0; bucket_index < chunk_index_ptr->bucket_count;
                 bucket_index++) {
                free(chunk_index_ptr->buckets[bucket_index].values);
            }
            free(chunk_index_ptr->buckets);
        }
    }

    free(index->hashes);
    free(index->candidate_marks);
    free(index->touched_indexes);
}

static int mark_bucket_candidates(HammingIndex *index, Bucket *bucket) {
    Py_ssize_t i;
    Py_ssize_t kept_index;

    for (i = 0; i < bucket->size; i++) {
        kept_index = bucket->values[i];
        if (index->candidate_marks[kept_index] == 0) {
            index->candidate_marks[kept_index] = 1;
            if (touched_append(index, kept_index) < 0) {
                return -1;
            }
        }
    }
    return 0;
}

static int mark_partition_zero_candidates(
    HammingIndex *index, uint64_t phash_value
) {
    uint64_t chunks[CHUNK_COUNT];
    int chunk_index;
    int bit_index;
    uint64_t chunk_value;
    uint64_t variant_value;
    ChunkIndex *chunk_index_ptr;

    partition_chunks(phash_value, 0, chunks);
    for (chunk_index = 0; chunk_index < CHUNK_COUNT; chunk_index++) {
        chunk_index_ptr = &index->indexes[0][chunk_index];
        chunk_value = chunks[chunk_index];
        if (mark_bucket_candidates(
                index, &chunk_index_ptr->buckets[chunk_value]
            ) < 0) {
            return -1;
        }

        if (index->chunk_radius == 0) {
            continue;
        }

        for (bit_index = 0; bit_index < CHUNK_BITS[chunk_index];
             bit_index++) {
            variant_value = chunk_value ^ (((uint64_t)1) << bit_index);
            if (mark_bucket_candidates(
                    index, &chunk_index_ptr->buckets[variant_value]
                ) < 0) {
                return -1;
            }
        }
    }
    return 0;
}

static int find_partition_one_duplicate(
    HammingIndex *index, uint64_t phash_value
) {
    uint64_t chunks[CHUNK_COUNT];
    int chunk_index;
    int bit_index;
    uint64_t chunk_value;
    uint64_t variant_value;
    ChunkIndex *chunk_index_ptr;
    Bucket *bucket;
    Py_ssize_t bucket_offset;
    Py_ssize_t kept_index;

    partition_chunks(phash_value, 1, chunks);
    for (chunk_index = 0; chunk_index < CHUNK_COUNT; chunk_index++) {
        chunk_index_ptr = &index->indexes[1][chunk_index];
        chunk_value = chunks[chunk_index];

        for (bit_index = -1; bit_index < CHUNK_BITS[chunk_index];
             bit_index++) {
            if (bit_index == -1) {
                variant_value = chunk_value;
            } else {
                if (index->chunk_radius == 0) {
                    break;
                }
                variant_value = chunk_value ^ (((uint64_t)1) << bit_index);
            }

            bucket = &chunk_index_ptr->buckets[variant_value];
            for (bucket_offset = 0; bucket_offset < bucket->size;
                 bucket_offset++) {
                kept_index = bucket->values[bucket_offset];
                if (index->candidate_marks[kept_index] != 1) {
                    continue;
                }
                if (popcount64(phash_value ^ index->hashes[kept_index]) <=
                    index->threshold) {
                    return 1;
                }
                index->candidate_marks[kept_index] = 2;
            }
        }
    }
    return 0;
}

static int hamming_index_find_duplicate(
    HammingIndex *index, uint64_t phash_value
) {
    int found_duplicate;

    if (index->kept_count == 0) {
        return 0;
    }

    index->touched_count = 0;
    if (mark_partition_zero_candidates(index, phash_value) < 0) {
        clear_candidate_marks(index);
        return -1;
    }

    if (index->touched_count == 0) {
        return 0;
    }

    found_duplicate = find_partition_one_duplicate(index, phash_value);
    clear_candidate_marks(index);
    return found_duplicate;
}

static int hamming_index_add(HammingIndex *index, uint64_t phash_value) {
    uint64_t chunks[CHUNK_COUNT];
    int partition_index;
    int chunk_index;
    Py_ssize_t kept_index = index->kept_count;
    ChunkIndex *chunk_index_ptr;

    if (kept_index >= index->hash_capacity) {
        PyErr_NoMemory();
        return -1;
    }

    index->hashes[kept_index] = phash_value;
    index->candidate_marks[kept_index] = 0;
    for (partition_index = 0; partition_index < PARTITION_COUNT;
         partition_index++) {
        partition_chunks(phash_value, partition_index, chunks);
        for (chunk_index = 0; chunk_index < CHUNK_COUNT; chunk_index++) {
            chunk_index_ptr = &index->indexes[partition_index][chunk_index];
            if (bucket_append(
                    &chunk_index_ptr->buckets[chunks[chunk_index]], kept_index
                ) < 0) {
                return -1;
            }
        }
    }

    index->kept_count += 1;
    return 0;
}

static PyObject *build_kept_index_list(
    const Py_ssize_t *kept_indexes, Py_ssize_t kept_count
) {
    PyObject *result = PyList_New(0);
    PyObject *index_obj;
    Py_ssize_t i;

    if (result == NULL) {
        return NULL;
    }

    for (i = 0; i < kept_count; i++) {
        index_obj = PyLong_FromSsize_t(kept_indexes[i]);
        if (index_obj == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        if (PyList_Append(result, index_obj) < 0) {
            Py_DECREF(index_obj);
            Py_DECREF(result);
            return NULL;
        }
        Py_DECREF(index_obj);
    }

    return result;
}

static PyObject *native_deduplicate_phashes(
    PyObject *self, PyObject *args
);

static PyObject *deduplicate_with_index(
    const uint64_t *phashes, Py_ssize_t input_count, int threshold
) {
    HammingIndex index;
    Py_ssize_t *kept_indexes = NULL;
    Py_ssize_t i;
    Py_ssize_t kept_count = 0;
    int is_duplicate;
    PyObject *result = NULL;

    memset(&index, 0, sizeof(index));

    if (input_count > 0) {
        kept_indexes = (Py_ssize_t *)malloc(
            (size_t)input_count * sizeof(Py_ssize_t)
        );
        if (kept_indexes == NULL) {
            PyErr_NoMemory();
            return NULL;
        }
    }

    if (hamming_index_init(&index, input_count, threshold) < 0) {
        goto cleanup;
    }

    for (i = 0; i < input_count; i++) {
        is_duplicate = hamming_index_find_duplicate(&index, phashes[i]);
        if (is_duplicate < 0) {
            goto cleanup;
        }
        if (!is_duplicate) {
            if (hamming_index_add(&index, phashes[i]) < 0) {
                goto cleanup;
            }
            kept_indexes[kept_count] = i;
            kept_count += 1;
        }

        if ((i & 0x3fff) == 0 && PyErr_CheckSignals() < 0) {
            goto cleanup;
        }
    }

    result = build_kept_index_list(kept_indexes, kept_count);

cleanup:
    hamming_index_free(&index);
    free(kept_indexes);
    return result;
}

static PyObject *deduplicate_with_linear_scan(
    const uint64_t *phashes, Py_ssize_t input_count, int threshold
) {
    Py_ssize_t *kept_indexes = NULL;
    Py_ssize_t i;
    Py_ssize_t kept_offset;
    Py_ssize_t kept_count = 0;
    int is_duplicate;
    PyObject *result;

    if (input_count > 0) {
        kept_indexes = (Py_ssize_t *)malloc(
            (size_t)input_count * sizeof(Py_ssize_t)
        );
        if (kept_indexes == NULL) {
            PyErr_NoMemory();
            return NULL;
        }
    }

    for (i = 0; i < input_count; i++) {
        is_duplicate = 0;
        for (kept_offset = 0; kept_offset < kept_count; kept_offset++) {
            if (popcount64(phashes[i] ^ phashes[kept_indexes[kept_offset]]) <=
                threshold) {
                is_duplicate = 1;
                break;
            }
        }

        if (!is_duplicate) {
            kept_indexes[kept_count] = i;
            kept_count += 1;
        }

        if ((i & 0x3fff) == 0 && PyErr_CheckSignals() < 0) {
            free(kept_indexes);
            return NULL;
        }
    }

    result = build_kept_index_list(kept_indexes, kept_count);
    free(kept_indexes);
    return result;
}

static PyObject *deduplicate_keep_first(Py_ssize_t input_count) {
    PyObject *result;
    PyObject *index_obj;

    result = PyList_New(0);
    if (result == NULL || input_count == 0) {
        return result;
    }

    index_obj = PyLong_FromLong(0);
    if (index_obj == NULL) {
        Py_DECREF(result);
        return NULL;
    }
    if (PyList_Append(result, index_obj) < 0) {
        Py_DECREF(index_obj);
        Py_DECREF(result);
        return NULL;
    }
    Py_DECREF(index_obj);
    return result;
}

static PyObject *native_deduplicate_phashes(
    PyObject *self, PyObject *args
) {
    PyObject *phashes_obj;
    PyObject *phashes_sequence;
    PyObject **phash_items;
    PyObject *result = NULL;
    uint64_t *phashes = NULL;
    Py_ssize_t input_count;
    Py_ssize_t i;
    int threshold;

    (void)self;

    if (!PyArg_ParseTuple(args, "Oi", &phashes_obj, &threshold)) {
        return NULL;
    }
    if (threshold < 0 || threshold > DEDUP_THRESHOLD_MAX) {
        PyErr_SetString(
            PyExc_ValueError,
            "native pHash deduplication supports thresholds between 0 and 64"
        );
        return NULL;
    }

    phashes_sequence = PySequence_Fast(
        phashes_obj, "phashes must be an iterable of 64-bit integer pHashes"
    );
    if (phashes_sequence == NULL) {
        return NULL;
    }

    input_count = PySequence_Fast_GET_SIZE(phashes_sequence);
    phash_items = PySequence_Fast_ITEMS(phashes_sequence);

    if (input_count > 0) {
        phashes = (uint64_t *)malloc((size_t)input_count * sizeof(uint64_t));
        if (phashes == NULL) {
            PyErr_NoMemory();
            goto cleanup;
        }
    }

    for (i = 0; i < input_count; i++) {
        phashes[i] = (uint64_t)PyLong_AsUnsignedLongLong(phash_items[i]);
        if (PyErr_Occurred()) {
            goto cleanup;
        }
    }

    if (threshold == DEDUP_THRESHOLD_MAX) {
        result = deduplicate_keep_first(input_count);
    } else if (threshold <= INDEX_MAX_THRESHOLD) {
        result = deduplicate_with_index(phashes, input_count, threshold);
    } else {
        result = deduplicate_with_linear_scan(phashes, input_count, threshold);
    }

cleanup:
    Py_DECREF(phashes_sequence);
    free(phashes);
    return result;
}

static PyMethodDef NativeDedupMethods[] = {
    {
        "deduplicate_phashes",
        native_deduplicate_phashes,
        METH_VARARGS,
        "Return indexes of pHashes kept by exact Hamming-distance deduplication.",
    },
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef NativeDedupModule = {
    PyModuleDef_HEAD_INIT,
    "_native_dedup",
    "Native pHash deduplication helpers.",
    -1,
    NativeDedupMethods,
};

PyMODINIT_FUNC PyInit__native_dedup(void) {
    return PyModule_Create(&NativeDedupModule);
}
