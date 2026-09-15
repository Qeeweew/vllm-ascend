#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
#include <type_traits>

using namespace AscendC;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
constexpr int32_t BLOCK_SIZE       = 128; // 一次处理的计算块大小
constexpr int32_t GROUP_SIZE       = 32;  // 现在的量化 group 大小
constexpr int32_t GROUPS_PER_BLOCK = BLOCK_SIZE / GROUP_SIZE; // 4个
constexpr int32_t COMPUTE_ROWS     = 32;  // 每次内循环拷贝计算的行数
constexpr int32_t PACK_RATIO       = 8;
constexpr int32_t GROUP_TILE       = 8;

// -----------------------------------------------------------------------------
// HardEvent IDs (only 0/1/2/3)
// Same HardEvent must use the same event_id everywhere.
// -----------------------------------------------------------------------------
static constexpr int EID_MTE2_V = 0;  // MTE2 -> V
static constexpr int EID_V_MTE2 = 1;  // V -> MTE2
static constexpr int EID_V_MTE3 = 2;  // V -> MTE3
static constexpr int EID_MTE3_V = 3;  // MTE3 -> V

// -----------------------------------------------------------------------------
// Simple Low-level ZeroOut (no pipe)
// -----------------------------------------------------------------------------
template<typename T>
class KernelZeroOutLL {
public:
    __aicore__ inline void Init(GM_ADDR dst, uint64_t elem_cnt) {
        dstGm.SetGlobalBuffer((__gm__ T*)dst);
        total = elem_cnt;
    }

    __aicore__ inline void Process() {
        uint32_t addr = 0;
        LocalTensor<T> tile = LocalTensor<T>(TPosition::VECOUT, addr, TILE_LEN);

        int32_t core_idx = GetBlockIdx();
        int32_t core_num = GetBlockNum();

        Duplicate(tile, (T)0, TILE_LEN);
        PipeBarrier<PIPE_V>();
        // V -> MTE3
        SetFlag<HardEvent::V_MTE3>(EID_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EID_V_MTE3);

        for (uint64_t base = (uint64_t)core_idx * TILE_LEN; base < total; base += (uint64_t)core_num * TILE_LEN) {
            uint32_t len = (uint32_t)((total - base < (uint64_t)TILE_LEN) ? (total - base) : TILE_LEN);
            DataCopy(dstGm[base], tile, len);
        }
        DataSyncBarrier<MemDsbT::DDR>();
    }

private:
    GlobalTensor<T> dstGm;
    uint64_t total = 0;
    static constexpr int32_t TILE_LEN = 2048;
};

// -----------------------------------------------------------------------------
// Kernel SwiGLU (Low-level, NO double-buffer; simple)
// Layout: [Gate | Value] split at inter_dim
// Input: FP32, Output: T
// -----------------------------------------------------------------------------
template<typename T>
class KernelSwiGLU {
public:
    __aicore__ inline KernelSwiGLU() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, int32_t total_rows, int32_t inter_dim,
                                float swiglu_limit) {
        inputGm.SetGlobalBuffer((__gm__ float*)input);
        outputGm.SetGlobalBuffer((__gm__ T*)output);
        this->total_rows = total_rows;
        this->inter_dim  = inter_dim;
        this->swiglu_limit = swiglu_limit;

        uint32_t addr = 0;
        f_in  = LocalTensor<float>(TPosition::VECCALC, addr, 2 * TILE_LEN);
        addr += (uint32_t)(2 * TILE_LEN * sizeof(float));
        f_out = LocalTensor<float>(TPosition::VECCALC, addr, TILE_LEN);
        addr += (uint32_t)(TILE_LEN * sizeof(float));

        t_out = LocalTensor<T>(TPosition::VECOUT, addr, TILE_LEN);
        addr += (uint32_t)(TILE_LEN * sizeof(T));
    }

    __aicore__ inline void Process() {
        int32_t core_idx = GetBlockIdx();
        int32_t core_num = GetBlockNum();

        for (int32_t r = core_idx; r < total_rows; r += core_num) {
            ComputeRow(r);
        }
    }

private:
    __aicore__ inline void ComputeRow(int32_t row_idx) {
        uint64_t in_base  = (uint64_t)row_idx * 2u * (uint64_t)inter_dim;
        uint64_t out_base = (uint64_t)row_idx * (uint64_t)inter_dim;

        bool first_tile = true;
        for (int32_t i = 0; i < inter_dim; i += (int32_t)TILE_LEN) {
            int32_t len = (inter_dim - i < (int32_t)TILE_LEN) ? (inter_dim - i) : (int32_t)TILE_LEN;

            // reverse deps to avoid overwrite
            if (!first_tile) {
                WaitFlag<HardEvent::V_MTE2>(EID_V_MTE2);     // previous compute done before overwriting t_in
                WaitFlag<HardEvent::MTE3_V>(EID_MTE3_V);     // previous store done before overwriting t_out
            }

            // Copy FP32 input directly
            DataCopyParams params;
            params.blockCount = 2;
            params.blockLen   = (uint16_t)(len * (int32_t)sizeof(float) / 32);
            params.srcStride  = (uint16_t)((inter_dim - len) * (int32_t)sizeof(float) / 32);
            params.dstStride  = 0;

            DataCopy(f_in, inputGm[in_base + (uint64_t)i], params);
            SetFlag<HardEvent::MTE2_V>(EID_MTE2_V);
            WaitFlag<HardEvent::MTE2_V>(EID_MTE2_V);

            if (swiglu_limit > 0.0f) {
                Mins(f_in, f_in, swiglu_limit, len);
                PipeBarrier<PIPE_V>();
                Mins(f_in[len], f_in[len], swiglu_limit, len);
                PipeBarrier<PIPE_V>();
                Maxs(f_in[len], f_in[len], -swiglu_limit, len);
                PipeBarrier<PIPE_V>();
            }
            Silu(f_out, f_in, len);
            PipeBarrier<PIPE_V>();
            Mul(f_out, f_out, f_in[len], len);
            PipeBarrier<PIPE_V>();
            Cast(t_out, f_out, RoundMode::CAST_ROUND, len);
            PipeBarrier<PIPE_V>();

            // mark V done for next tile's CopyIn overwrite protection
            SetFlag<HardEvent::V_MTE2>(EID_V_MTE2);

            // store
            SetFlag<HardEvent::V_MTE3>(EID_V_MTE3);
            WaitFlag<HardEvent::V_MTE3>(EID_V_MTE3);

            DataCopy(outputGm[out_base + (uint64_t)i], t_out, len);

            SetFlag<HardEvent::MTE3_V>(EID_MTE3_V);

            first_tile = false;
        }

        // close last outstanding pairs (important across rows, avoid "Set twice" later)
        if (!first_tile) {
            WaitFlag<HardEvent::V_MTE2>(EID_V_MTE2);
            WaitFlag<HardEvent::MTE3_V>(EID_MTE3_V);
        }
    }

private:
    GlobalTensor<float> inputGm;
    GlobalTensor<T> outputGm;

    LocalTensor<float> f_in, f_out;
    LocalTensor<T>     t_out;

    int32_t total_rows = 0;
    int32_t inter_dim  = 0;
    float swiglu_limit = 0.0f;
    static constexpr int TILE_LEN = 1024;
};

template<typename T, typename OutputT = T>
class KernelGroupedGemvW4A16Moe {
public:
    __aicore__ inline KernelGroupedGemvW4A16Moe() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR scales,
                                GM_ADDR expert_ids, GM_ADDR y, GM_ADDR topk_weights,
                                int32_t total_tokens, int32_t in_dim, int32_t out_dim,
                                int32_t num_experts, int32_t top_k,
                                bool is_broadcast_x, bool is_weighted_sum) {
        this->total_tokens = total_tokens;
        this->top_k = top_k;
        this->in_dim = in_dim;
        this->out_dim = out_dim;
        this->out_dim_packed = out_dim / PACK_RATIO;
        this->num_experts = num_experts;
        // Block 是数据加载单元，Group 是 Scale 单元
        this->num_blocks = (in_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
        this->total_groups = in_dim / GROUP_SIZE;

        this->is_broadcast_x = is_broadcast_x;
        this->is_weighted_sum = is_weighted_sum;

        xGm.SetGlobalBuffer((__gm__ T*)x);
        weightGm.SetGlobalBuffer((__gm__ int32_t*)weight);
        scalesGm.SetGlobalBuffer((__gm__ T*)scales);
        expertIdsGm.SetGlobalBuffer((__gm__ int32_t*)expert_ids);
        yGm.SetGlobalBuffer((__gm__ OutputT*)y);

        if (is_weighted_sum) {
            topkWeightsGm.SetGlobalBuffer((__gm__ float*)topk_weights);
        }

        BuildLocalTensors();
    }

    __aicore__ inline void Process() {
        int32_t total_tasks = total_tokens * num_blocks;
        int32_t core_idx = GetBlockIdx();
        int32_t core_num = GetBlockNum();

        int32_t base = total_tasks / core_num;
        int32_t rem  = total_tasks % core_num;

        int32_t start_task, task_cnt;
        if (core_idx < rem) {
            task_cnt = base + 1;
            start_task = core_idx * (base + 1);
        } else {
            task_cnt = base;
            start_task = rem * (base + 1) + (core_idx - rem) * base;
        }
        int32_t end_task = start_task + task_cnt;
        if (task_cnt <= 0) return;

        for (int32_t n_start = 0; n_start < out_dim; n_start += TILE_N) {
            int32_t cur_n_len = (out_dim - n_start < TILE_N) ? (out_dim - n_start) : TILE_N;
            Duplicate(y_fp32, 0.0f, cur_n_len);
            PipeBarrier<PIPE_V>();

            int32_t current_row_idx = -1;
            if (start_task < end_task) current_row_idx = start_task / num_blocks;

            for (int32_t task_id = start_task; task_id < end_task; ++task_id) {
                int32_t row_idx = task_id / num_blocks;
                int32_t b_idx   = task_id % num_blocks;

                if (row_idx != current_row_idx) {
                    CopyOut(current_row_idx, n_start, cur_n_len);
                    Duplicate(y_fp32, 0.0f, cur_n_len);
                    PipeBarrier<PIPE_V>();
                    current_row_idx = row_idx;
                }

                int32_t expert_id = expertIdsGm.GetValue(row_idx);
                ProcessBlock(b_idx, expert_id, row_idx, n_start, cur_n_len, y_fp32);
            }

            if (current_row_idx != -1) {
                CopyOut(current_row_idx, n_start, cur_n_len);
            }
        }
    }

private:
    __aicore__ inline void BuildLocalTensors() {
        constexpr int32_t tile_n_packed = TILE_N / PACK_RATIO;
        uint32_t addr = 0;

        x_local = LocalTensor<T>(TPosition::VECIN, addr, BLOCK_SIZE);
        addr += (uint32_t)(BLOCK_SIZE * sizeof(T));

        w_local[0] = LocalTensor<int32_t>(TPosition::VECIN, addr, COMPUTE_ROWS * tile_n_packed);
        addr += (uint32_t)(COMPUTE_ROWS * tile_n_packed * sizeof(int32_t));
        w_local[1] = LocalTensor<int32_t>(TPosition::VECIN, addr, COMPUTE_ROWS * tile_n_packed);
        addr += (uint32_t)(COMPUTE_ROWS * tile_n_packed * sizeof(int32_t));

        // 保留 4 个 Scale 行缓存
        s_local = LocalTensor<T>(TPosition::VECIN, addr, GROUPS_PER_BLOCK * TILE_N);
        addr += (uint32_t)(GROUPS_PER_BLOCK * TILE_N * sizeof(T));

        y_fp32 = LocalTensor<float>(TPosition::VECOUT, addr, TILE_N);
        addr += (uint32_t)(TILE_N * sizeof(float));

        // Four FP32 group accumulators; never narrow BF16 activations to FP16.
        group_acc = LocalTensor<float>(TPosition::VECCALC, addr, GROUPS_PER_BLOCK * TILE_N);
        addr += (uint32_t)(GROUPS_PER_BLOCK * TILE_N * sizeof(float));

        w_half = LocalTensor<half>(TPosition::VECCALC, addr, GROUP_TILE * TILE_N);
        addr += (uint32_t)(GROUP_TILE * TILE_N * sizeof(half));
        w_float = LocalTensor<float>(TPosition::VECCALC, addr, GROUP_TILE * TILE_N);
        addr += GROUP_TILE * TILE_N * sizeof(float);
        x_float = LocalTensor<float>(TPosition::VECCALC, addr, BLOCK_SIZE);
    }

    __aicore__ inline void PrefetchW(int32_t expert_id, int32_t b_idx, int32_t k_inner_start,
                                     int32_t n_offset, int32_t cur_n_len, int buf_id) {
        uint64_t w_stride_k = (uint64_t)out_dim_packed;
        int32_t cur_n_packed = cur_n_len / PACK_RATIO;
        int32_t n_offset_packed = n_offset / PACK_RATIO;

        uint64_t w_offset = (uint64_t)expert_id * (uint64_t)in_dim * w_stride_k +
                            (uint64_t)(b_idx * BLOCK_SIZE + k_inner_start) * w_stride_k +
                            (uint64_t)n_offset_packed;

        DataCopyExtParams p{
            (uint16_t)COMPUTE_ROWS,
            (uint32_t)(cur_n_packed * sizeof(int32_t)),
            (uint32_t)((w_stride_k - (uint64_t)cur_n_packed) * sizeof(int32_t)),
            0, 0
        };
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(w_local[buf_id], weightGm[w_offset], p, pad);
    }

    __aicore__ inline void PrefetchS(int32_t expert_id, int32_t b_idx, // codespell:ignore prefetchs
                                     int32_t n_offset, int32_t cur_n_len) {
        // 当前Block的基础 group 索引
        uint64_t base_group = (uint64_t)b_idx * GROUPS_PER_BLOCK;

        DataCopyExtParams p{
            (uint16_t)((total_groups - b_idx * GROUPS_PER_BLOCK < GROUPS_PER_BLOCK) ?
                total_groups - b_idx * GROUPS_PER_BLOCK : GROUPS_PER_BLOCK),
            (uint32_t)(cur_n_len * sizeof(T)),
            (uint32_t)((out_dim - (uint64_t)cur_n_len) * sizeof(T)),
            (uint32_t)((TILE_N - cur_n_len) * sizeof(T) / 32), 0
        };

        DataCopyPadExtParams<T> pad{false, 0, 0, 0};

        // 将属于这个Block里的 4 行 Scale 读出来
        uint64_t sz_offset = (uint64_t)expert_id * (uint64_t)total_groups * (uint64_t)out_dim +
                             base_group * (uint64_t)out_dim +
                             (uint64_t)n_offset;
        DataCopyPad(s_local, scalesGm[sz_offset], p, pad);
    }

    __aicore__ inline void ComputeChunk(LocalTensor<int32_t>& w_i32,
                                        int32_t x_offset_idx,
                                        int32_t tile_n,
                                        int32_t tile_n_packed,
                                        LocalTensor<float>& current_group_acc) {
        for (int i = 0; i < COMPUTE_ROWS; i += GROUP_TILE) {
            LocalTensor<int4b_t> w_int4 = w_i32[i * tile_n_packed].template ReinterpretCast<int4b_t>();
            // INT4 -> FP16 is exact; all activation products and reductions use FP32.
            Cast(w_half, w_int4, RoundMode::CAST_NONE, GROUP_TILE * tile_n);
            PipeBarrier<PIPE_V>();
            Cast(w_float, w_half, RoundMode::CAST_NONE, GROUP_TILE * tile_n);
            PipeBarrier<PIPE_V>();
            for (int k = 0; k < GROUP_TILE; ++k) {
                Axpy(current_group_acc, w_float[k * tile_n],
                     x_float.GetValue(x_offset_idx + i + k), tile_n);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void ProcessBlock(int32_t b_idx, int32_t expert_id, int32_t row_idx,
                                        int32_t n_offset, int32_t cur_n_len,
                                        LocalTensor<float>& global_acc) {
        // A negative id is the cache-miss sentinel.  The row accumulator is
        // already zero-initialised by Process(), so skipping every K block
        // makes this route contribute exactly zero without ever indexing the
        // cache buffers with an invalid address.  The normal (full expert)
        // operator never emits negative ids, therefore this is backwards
        // compatible with its existing semantics.
        if (expert_id < 0 || expert_id >= num_experts) {
            return;
        }

        uint64_t x_offset;
        if (is_broadcast_x) {
            int32_t batch_idx = row_idx / top_k;
            x_offset = (uint64_t)batch_idx * (uint64_t)in_dim + (uint64_t)b_idx * BLOCK_SIZE;
        } else {
            x_offset = (uint64_t)row_idx * (uint64_t)in_dim + (uint64_t)b_idx * BLOCK_SIZE;
        }

        const int32_t remaining = in_dim - b_idx * BLOCK_SIZE;
        const int32_t block_k = remaining < BLOCK_SIZE ? remaining : BLOCK_SIZE;
        DataCopy(x_local, xGm[x_offset], block_k);
        SetFlag<HardEvent::MTE2_V>(EID_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(EID_MTE2_V);
        Cast(x_float, x_local, RoundMode::CAST_NONE, block_k);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);

        PrefetchW(expert_id, b_idx, 0, n_offset, cur_n_len, 0);
        SetFlag<HardEvent::MTE2_V>(EID_MTE2_V);

        // Initialize all groups; the final K block may contain only one group.
        Duplicate(group_acc, 0.0f, GROUPS_PER_BLOCK * TILE_N);
        PipeBarrier<PIPE_V>();

        int32_t cur_n_packed = cur_n_len / PACK_RATIO;
        int ping = 0;

        for (int32_t k_inner = 0; k_inner < block_k; k_inner += COMPUTE_ROWS) {
            WaitFlag<HardEvent::MTE2_V>(EID_MTE2_V);
            if (k_inner != 0) {
                WaitFlag<HardEvent::V_MTE2>(EID_V_MTE2);
            }

            if (k_inner + COMPUTE_ROWS < block_k) {
                int next_buf = ping ^ 1;
                PrefetchW(expert_id, b_idx, k_inner + COMPUTE_ROWS, n_offset, cur_n_len, next_buf);
                SetFlag<HardEvent::MTE2_V>(EID_MTE2_V);
            } else {
                // 最后一次拷贝 4 个对应的 Scale
                PrefetchS(expert_id, b_idx, n_offset, cur_n_len); // codespell:ignore prefetchs
                SetFlag<HardEvent::MTE2_V>(EID_MTE2_V);
            }

            // 计算该 32 行该放入哪个 group_acc
            int step = k_inner / COMPUTE_ROWS;
            LocalTensor<float> current_acc = group_acc[step * TILE_N];

            ComputeChunk(w_local[ping], k_inner, cur_n_len, cur_n_packed, current_acc);

            if (k_inner + COMPUTE_ROWS < block_k) {
                SetFlag<HardEvent::V_MTE2>(EID_V_MTE2);
            }
            ping ^= 1;
        }

        WaitFlag<HardEvent::MTE2_V>(EID_MTE2_V);

        // Scale stays signed. Only the valid groups in the final K block are read.
        LocalTensor<float> f_scale = w_float;
        for (int i = 0; i < block_k / GROUP_SIZE; ++i) {
            LocalTensor<float> current_acc = group_acc[i * TILE_N];
            LocalTensor<T> current_scale = s_local[i * TILE_N];
            Cast(f_scale, current_scale, RoundMode::CAST_NONE, cur_n_len);
            PipeBarrier<PIPE_V>();
            MulAddDst(global_acc, current_acc, f_scale, cur_n_len);
            PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_MTE2>(EID_V_MTE2);
        WaitFlag<HardEvent::V_MTE2>(EID_V_MTE2);
    }

    __aicore__ inline void CopyOut(int32_t row_idx, int32_t n_offset, int32_t cur_n_len) {
        if (row_idx < 0) return;

        if (is_weighted_sum) {
            float w_val = topkWeightsGm.GetValue(row_idx);
            Muls(y_fp32, y_fp32, w_val, cur_n_len);
            PipeBarrier<PIPE_V>();
        }

        SetFlag<HardEvent::V_MTE3>(EID_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EID_V_MTE3);

        if constexpr (std::is_same_v<OutputT, float>) {
            // FP32 output: copy directly without cast
            SetAtomicAdd<float>();

            if (is_weighted_sum) {
                int32_t batch_idx = row_idx / top_k;
                uint64_t y_offset = (uint64_t)batch_idx * (uint64_t)out_dim + (uint64_t)n_offset;
                DataCopy(yGm[y_offset], y_fp32, cur_n_len);
            } else {
                uint64_t y_offset = (uint64_t)row_idx * (uint64_t)out_dim + (uint64_t)n_offset;
                DataCopy(yGm[y_offset], y_fp32, cur_n_len);
            }

            SetAtomicNone();
        } else {
            // T output: cast from FP32 to T
            LocalTensor<T> y_half = y_fp32.template ReinterpretCast<T>();
            Cast(y_half, y_fp32, RoundMode::CAST_ROUND, cur_n_len);
            PipeBarrier<PIPE_V>();

            SetAtomicAdd<T>();

            if (is_weighted_sum) {
                int32_t batch_idx = row_idx / top_k;
                uint64_t y_offset = (uint64_t)batch_idx * (uint64_t)out_dim + (uint64_t)n_offset;
                DataCopy(yGm[y_offset], y_half, cur_n_len);
            } else {
                uint64_t y_offset = (uint64_t)row_idx * (uint64_t)out_dim + (uint64_t)n_offset;
                DataCopy(yGm[y_offset], y_half, cur_n_len);
            }

            SetAtomicNone();
        }

        SetFlag<HardEvent::MTE3_V>(EID_MTE3_V);
        WaitFlag<HardEvent::MTE3_V>(EID_MTE3_V);
    }

private:
    GlobalTensor<float> topkWeightsGm;

    GlobalTensor<T>       xGm;
    GlobalTensor<int32_t> weightGm;
    GlobalTensor<T>       scalesGm;
    GlobalTensor<int32_t> expertIdsGm;
    GlobalTensor<OutputT> yGm;

    // UB
    LocalTensor<T>        x_local;
    LocalTensor<int32_t>  w_local[2];
    LocalTensor<T>        s_local;

    LocalTensor<float>    y_fp32;
    LocalTensor<float>    group_acc;
    LocalTensor<half>     w_half;
    LocalTensor<float>    w_float, x_float;

    int32_t top_k = 0;
    int32_t in_dim = 0;
    int32_t out_dim = 0;
    int32_t out_dim_packed = 0;
    int32_t num_experts = 0;
    int32_t num_blocks = 0;
    int32_t total_groups = 0;
    int32_t total_tokens = 0;

    // Runtime flags
    bool is_broadcast_x = false;
    bool is_weighted_sum = false;

    static constexpr int32_t TILE_N = 1024;
};

// -----------------------------------------------------------------------------
// Kernel Cast FP32 to T (for final output conversion)
// -----------------------------------------------------------------------------
template<typename T>
class KernelCastFP32ToT {
public:
    __aicore__ inline KernelCastFP32ToT() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, int32_t total_elems) {
        inputGm.SetGlobalBuffer((__gm__ float*)input);
        outputGm.SetGlobalBuffer((__gm__ T*)output);
        this->total_elems = total_elems;

        uint32_t addr = 0;
        f_local = LocalTensor<float>(TPosition::VECIN, addr, TILE_LEN);
        addr += (uint32_t)(TILE_LEN * sizeof(float));
        t_local = LocalTensor<T>(TPosition::VECOUT, addr, TILE_LEN);
        addr += (uint32_t)(TILE_LEN * sizeof(T));
    }

    __aicore__ inline void Process() {
        int32_t core_idx = GetBlockIdx();
        int32_t core_num = GetBlockNum();

        for (int64_t base = (int64_t)core_idx * TILE_LEN; base < total_elems; base += (int64_t)core_num * TILE_LEN) {
            int32_t len = (int32_t)((total_elems - base < (int64_t)TILE_LEN) ? (total_elems - base) : TILE_LEN);

            DataCopy(f_local, inputGm[base], len);
            SetFlag<HardEvent::MTE2_V>(EID_MTE2_V);
            WaitFlag<HardEvent::MTE2_V>(EID_MTE2_V);

            Cast(t_local, f_local, RoundMode::CAST_ROUND, len);
            PipeBarrier<PIPE_V>();

            SetFlag<HardEvent::V_MTE3>(EID_V_MTE3);
            WaitFlag<HardEvent::V_MTE3>(EID_V_MTE3);

            DataCopy(outputGm[base], t_local, len);

            // A second tile must not overwrite f_local before the previous
            // Cast has consumed it. MTE3_V only stalls Vector, not MTE2.
            SetFlag<HardEvent::MTE3_MTE2>(EID_MTE3_V);
            WaitFlag<HardEvent::MTE3_MTE2>(EID_MTE3_V);
        }
    }

private:
    GlobalTensor<float> inputGm;
    GlobalTensor<T> outputGm;

    LocalTensor<float> f_local;
    LocalTensor<T> t_local;

    int64_t total_elems = 0;
    static constexpr uint32_t TILE_LEN = 2048;
};

// -----------------------------------------------------------------------------
// Fused Kernel Implementation
// -----------------------------------------------------------------------------
template<typename T>
__aicore__ inline void fused_moe_small_bs_impl(
    GM_ADDR x,
    GM_ADDR w13_weight, GM_ADDR w13_scales,
    GM_ADDR w2_weight,  GM_ADDR w2_scales,
    GM_ADDR expert_ids, GM_ADDR topk_weights,
    GM_ADDR y,
    GM_ADDR workspace,
    int32_t batch_size, int32_t hidden_size, int32_t inter_size,
    int32_t num_experts, int32_t top_k, float swiglu_limit)
{
    int32_t total_tokens = batch_size * top_k;

    // workspace layout:
    // [0 : w13_out_size) : W13 output (FP32) [TotalTokens, 2 * InterDim]
    // [w13_out_size : w13_out_size + w2_in_size) : SwiGLU output (T) [TotalTokens, InterDim]
    // [w13_out_size + w2_in_size : end) : W2 output (FP32) [BatchSize, HiddenSize]
    GM_ADDR w13_out_ptr = workspace;
    uint64_t w13_out_size = (uint64_t)total_tokens * (uint64_t)(inter_size * 2) * sizeof(float);

    GM_ADDR w2_in_ptr = (GM_ADDR)((__gm__ uint8_t*)w13_out_ptr + w13_out_size);
    uint64_t w2_in_size = (uint64_t)total_tokens * (uint64_t)inter_size * sizeof(T);

    GM_ADDR w2_out_ptr = (GM_ADDR)((__gm__ uint8_t*)w2_in_ptr + w2_in_size);
    uint64_t w2_out_size = (uint64_t)batch_size * (uint64_t)hidden_size * sizeof(float);

    // Phase 0: Zero out entire workspace at once (W13 FP32 + W2_in T + W2 FP32)
    {
        uint64_t total_workspace_size = w13_out_size + w2_in_size + w2_out_size;
        uint64_t total_elems = total_workspace_size / sizeof(int32_t);

        KernelZeroOutLL<int32_t> zeroOp;
        zeroOp.Init(workspace, total_elems);
        zeroOp.Process();
    }

    AscendC::SyncAll();

    // Phase 1: W13 Gemv (Broadcast X = true, Weighted = false, FP32 output)
    {
        KernelGroupedGemvW4A16Moe<T, float> op_w13;
        op_w13.Init(x, w13_weight, w13_scales,
                    expert_ids, w13_out_ptr, nullptr,
                    total_tokens, hidden_size, inter_size * 2, num_experts, top_k,
                    true, false);
        op_w13.Process();
    }

    AscendC::SyncAll();

    // Phase 2: SwiGLU (FP32 input, T output)
    // SwiGLU reads from w13_out and writes to w2_in (separate memory areas)
    {
        KernelSwiGLU<T> op_act;
        op_act.Init(w13_out_ptr, w2_in_ptr, total_tokens, inter_size, swiglu_limit);
        op_act.Process();
    }

    AscendC::SyncAll();

    // Phase 3: W2 Gemv (Broadcast X = false, Weighted = true, FP32 output)
    {
        KernelGroupedGemvW4A16Moe<T, float> op_w2;
        op_w2.Init(w2_in_ptr, w2_weight, w2_scales,
                   expert_ids, w2_out_ptr, topk_weights,
                   total_tokens, inter_size, hidden_size, num_experts, top_k,
                   false, true);
        op_w2.Process();
    }

    AscendC::SyncAll();

    // Phase 4: Cast FP32 to T (final output to y)
    {
        KernelCastFP32ToT<T> op_cast;
        op_cast.Init(w2_out_ptr, y, (int32_t)((uint64_t)batch_size * (uint64_t)hidden_size));
        op_cast.Process();
    }
}

extern "C" __global__ __aicore__ void fused_moe_small_bs_w4a16_fp16(
    GM_ADDR x,
    GM_ADDR w13_weight, GM_ADDR w13_scales,
    GM_ADDR w2_weight, GM_ADDR w2_scales,
    GM_ADDR expert_ids, GM_ADDR topk_weights,
    GM_ADDR y, GM_ADDR workspace,
    int32_t batch_size, int32_t hidden_size, int32_t inter_size,
    int32_t num_experts, int32_t top_k, float swiglu_limit)
{
    // The phased decode kernel uses hardware SyncAll. On arch22 the direct
    // launch stub supplies ffts_addr only for MIX modes (CANN
    // add_ffts_addr_func_param_by_mode); AIV_ONLY would deadlock at SyncAll.
    // 1_0 executes vector cores only, with the required FFTS launch metadata.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    AscendC::InitSocState();
    AscendC::ICachePreLoad(5);

    fused_moe_small_bs_impl<half>(
        x, w13_weight, w13_scales, w2_weight, w2_scales,
        expert_ids, topk_weights, y, workspace,
        batch_size, hidden_size, inter_size, num_experts, top_k,
        swiglu_limit);
}

extern "C" __global__ __aicore__ void fused_moe_small_bs_w4a16_bf16(
    GM_ADDR x,
    GM_ADDR w13_weight, GM_ADDR w13_scales,
    GM_ADDR w2_weight, GM_ADDR w2_scales,
    GM_ADDR expert_ids, GM_ADDR topk_weights,
    GM_ADDR y, GM_ADDR workspace,
    int32_t batch_size, int32_t hidden_size, int32_t inter_size,
    int32_t num_experts, int32_t top_k, float swiglu_limit)
{
    // The phased decode kernel uses hardware SyncAll. On arch22 the direct
    // launch stub supplies ffts_addr only for MIX modes (CANN
    // add_ffts_addr_func_param_by_mode); AIV_ONLY would deadlock at SyncAll.
    // 1_0 executes vector cores only, with the required FFTS launch metadata.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    AscendC::InitSocState();
    AscendC::ICachePreLoad(5);

    fused_moe_small_bs_impl<bfloat16_t>(
        x, w13_weight, w13_scales, w2_weight, w2_scales,
        expert_ids, topk_weights, y, workspace,
        batch_size, hidden_size, inter_size, num_experts, top_k,
        swiglu_limit);
}
