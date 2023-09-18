/*
Transformer function helper function.

Written by tomztyang,
2021/08/23
*/

#include <math.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 256
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
// #define DEBUG


__global__ void rpe_k_forward_v2(
    int b, int total_query_num, int local_size,
    int total_key_num, int nhead, int hdim, int l,
    const int *query_batch_cnt, const int *key_batch_cnt, const int* index_pair_batch,
    const int *index_pair,
    const float *relpos, const float* lookup_table, const float* key_features,
    float *output) {
    // dim3 blocks(total_query_num, nhead); dim3 threads(local_size);
    // params query_batch_cnt: [b]
    // params key_batch_cnt: [b]
    // params index_pair_batch: [total_query_num]
    // params index_pair: [total_query_num, local_size]
    // params relpos: [total_query_num, local_size]
    // params lookup_table: [l, nhead, hdim]
    // params key_features: [total_key_num, nhead, hdim]
    // params output: [total_query_num, local_size, nhead]

    int query_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int local_key_idx = threadIdx.x;

    if (query_idx >= total_query_num ||
        head_idx >= nhead ||
        local_key_idx >= local_size) return;

    int index = query_idx * local_size + local_key_idx;
    if (index_pair[index] == -1){
        // Ignore index.
        return;
    }

    // get real index of local keys.
    int batch_idx = index_pair_batch[query_idx];
    int key_start_idx = 0;
    for (int i = 0; i < batch_idx; i++){
        key_start_idx += key_batch_cnt[i];
    }
    key_start_idx += index_pair[index];
    key_features += key_start_idx * nhead * hdim + head_idx * hdim;

    // get quantized relative position.
    relpos += index;
    int quantize_relpos = min(max(int(floor(relpos[0])), 0), l - 1);
    lookup_table += quantize_relpos * nhead * hdim + head_idx * hdim;

    // Compute result.
    float attn_weight = 0;
    for (int i = 0; i < hdim; i++){
        attn_weight += key_features[i] * lookup_table[i];
    }
    output[index * nhead + head_idx] = attn_weight;
}


void rpe_k_launcher_v2(
    int b, int total_query_num, int local_size,
    int total_key_num, int nhead, int hdim, int l,
    const int *query_batch_cnt, const int *key_batch_cnt, const int* index_pair_batch,
    const int *index_pair,
    const float *relpos, const float* lookup_table, const float* key_features,
    float *output){
    // params query_batch_cnt: [b]
    // params key_batch_cnt: [b]
    // params index_pair_batch: [total_query_num]
    // params index_pair: [total_query_num, local_size]
    // params relpos: [total_query_num, local_size]
    // params lookup_table: [l, nhead, hdim]
    // params key_features: [total_key_num, nhead, hdim]
    // params output: [total_query_num, local_size, nhead]
    dim3 blocks(total_query_num, nhead);
    dim3 threads(local_size);

    rpe_k_forward_v2<<<blocks, threads>>>(
        b, total_query_num, local_size, total_key_num, nhead, hdim, l,
        query_batch_cnt, key_batch_cnt, index_pair_batch,
        index_pair, relpos, lookup_table, key_features,
        output);
}


__global__ void rpe_k_backward_v2(
    int b, int total_query_num, int local_size,
    int total_key_num, int nhead, int hdim, int l,
    const int *query_batch_cnt, const int *key_batch_cnt, const int* index_pair_batch,
    const int *index_pair,
    const float *relpos, const float* lookup_table, const float* key_features,
    float *grad_out, float * grad_lookup_table, float * grad_key_features) {
    // dim3 blocks(DIVUP(total_query_num * local_size, THREADS_PER_BLOCK), nhead, hdim);
    // params query_batch_cnt: [b]
    // params key_batch_cnt: [b]
    // params index_pair_batch: [total_query_num]
    // params index_pair: [total_query_num, local_size]
    // params relpos: [total_query_num, local_size]
    // params lookup_table: [l, nhead, hdim]
    // params key_features: [total_key_num, nhead, hdim]
    // params grad_out: [total_query_num, local_size, nhead]
    // params grad_lookup_table: [l, nhead, hdim]
    // params grad_key_features: [total_key_num, nhead, hdim]

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int head_idx = blockIdx.y;
    int hdim_idx = blockIdx.z;
    if (index >= total_query_num * local_size ||
        head_idx >= nhead ||
        hdim_idx >= hdim) return;

    if (index_pair[index] == -1){
        // Ignore index.
        return;
    }

    int query_idx = index / local_size;
    int batch_idx = index_pair_batch[query_idx];
    int key_start_idx = 0;
    for (int i = 0; i < batch_idx; i++){
        key_start_idx += key_batch_cnt[i];
    }

    // 1. Obtain key features.
    key_start_idx += index_pair[index];
    key_features += key_start_idx * nhead * hdim + head_idx * hdim + hdim_idx;
    grad_key_features += key_start_idx * nhead * hdim + head_idx * hdim + hdim_idx;

    // 2. Obtain quantize relative position.
    relpos += index;
    int quantize_relpos = min(max(int(floor(relpos[0])), 0), l - 1);
    lookup_table += quantize_relpos * nhead * hdim + head_idx * hdim + hdim_idx;
    grad_lookup_table += quantize_relpos * nhead * hdim + head_idx * hdim + hdim_idx;

    // 3. Obtain output position.
    float gradient = grad_out[index * nhead + head_idx];
    atomicAdd(
        grad_key_features,
        gradient * lookup_table[0]);
    atomicAdd(
        grad_lookup_table,
        gradient * key_features[0]);
}


void rpe_k_grad_launcher_v2(
    int b, int total_query_num, int local_size,
    int total_key_num, int nhead, int hdim, int l,
    const int *query_batch_cnt, const int *key_batch_cnt, const int* index_pair_batch,
    const int *index_pair,
    const float *relpos, const float* lookup_table, const float* key_features,
    float *grad_out, float* grad_lookup_table, float* grad_key_features){
    // params query_batch_cnt: [b]
    // params key_batch_cnt: [b]
    // params index_pair_batch: [total_query_num]
    // params index_pair: [total_query_num, local_size]
    // params relpos: [total_query_num, local_size]
    // params lookup_table: [l, nhead, hdim]
    // params key_features: [total_key_num, nhead, hdim]
    // params grad_out: [total_query_num, local_size, nhead]
    // params grad_lookup_table: [l, nhead, hdim]
    // params grad_key_features: [total_key_num, nhead, hdim]

    dim3 blocks(DIVUP(total_query_num * local_size, THREADS_PER_BLOCK), nhead, hdim);
    dim3 threads(THREADS_PER_BLOCK);
    rpe_k_backward_v2<<<blocks, threads>>>(
        b, total_query_num, local_size, total_key_num, nhead, hdim, l,
        query_batch_cnt, key_batch_cnt, index_pair_batch,
        index_pair, relpos, lookup_table, key_features,
        grad_out, grad_lookup_table, grad_key_features);
}

