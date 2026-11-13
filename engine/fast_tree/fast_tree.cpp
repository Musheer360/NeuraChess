// Copyright (c) 2026 Musheer Alam. Licensed under GPL-3.0.
/*
 * NeuraChess Compiled Zero-GIL C++23 PUCT Search Arena, 64B-Aligned SoA AVX2
 * Selector, Win32 Hybrid P-Core Pinning & 64-bit Zobrist LRU Cache
 * (engine/fast_tree/fast_tree.cpp)
 * ===========================================================================
 * Compiled with Clang 22 / Zig C++23:
 *   `-std=c++23 -O3 -ffast-math -march=native -flto -shared`
 *
 * Key Upgrades (Step 2 + Steering Upgrade 1):
 *   1. Zero Legal-Move Pruning: 100% of legal moves are expanded at the root
 *      AND every internal node (`branching_k_for_ply` = all legal moves).
 *   2. 64-Byte Cache-Line Aligned Struct-of-Arrays (`alignas(64) ChildSoAArena`):
 *      Evaluates 8 children per AVX2 instruction (`_mm256_fmadd_ps`, `_mm256_sqrt_ps`,
 *      `_mm256_div_ps`, `_mm256_blendv_ps`) with zero gather stalls and C++23 `[[assume]]`.
 *   3. Core Ultra 5 125U Hybrid P-Core Pinning & HighQoS:
 *      Disables Windows 11 EcoQoS / LP-E power throttling (`SetThreadInformation`
 *      `THREAD_POWER_THROTTLING_EXECUTION_SPEED`, `StateMask=0`) and pins search/ponder
 *      threads to highest-`EfficiencyClass` Redwood Cove P-cores.
 *   4. Zero-Copy OpenVINO Input/Output Tensor Staging:
 *      Packs `(B, 18, 8, 8)` float32 planes directly into `InferRequest` host tensor memory.
 */

#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <malloc.h>
#endif

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <immintrin.h>
#include "bitboard_chess.hpp"

#ifdef _WIN32
#define FT_EXPORT extern "C" __declspec(dllexport)
#else
#define FT_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// ============================================================================
// 64-Byte Cache-Line Aligned RAII Slab for Struct-of-Arrays (SoA) AVX2
// ============================================================================
template <typename T>
class alignas(64) AlignedSlab {
public:
    T* ptr = nullptr;
    size_t cap = 0;

    AlignedSlab() = default;
    explicit AlignedSlab(size_t initial_cap) { reserve(initial_cap); }

    ~AlignedSlab() { release(); }

    AlignedSlab(const AlignedSlab&) = delete;
    AlignedSlab& operator=(const AlignedSlab&) = delete;

    AlignedSlab(AlignedSlab&& other) noexcept : ptr(other.ptr), cap(other.cap) {
        other.ptr = nullptr;
        other.cap = 0;
    }
    AlignedSlab& operator=(AlignedSlab&& other) noexcept {
        if (this != &other) {
            release();
            ptr = other.ptr;
            cap = other.cap;
            other.ptr = nullptr;
            other.cap = 0;
        }
        return *this;
    }

    void release() noexcept {
        if (ptr) {
#ifdef _WIN32
            _aligned_free(ptr);
#else
            std::free(ptr);
#endif
            ptr = nullptr;
            cap = 0;
        }
    }

    void reserve(size_t needed) {
        if (needed <= cap) return;
        size_t new_cap = cap ? cap : 32768ULL;
        while (new_cap < needed) new_cap *= 2ULL;
        new_cap = (new_cap + 15ULL) & ~15ULL; // multiple of 16 elements (64B for 32-bit types)
        size_t bytes = ((new_cap * sizeof(T)) + 63ULL) & ~63ULL;
#ifdef _WIN32
        T* new_ptr = static_cast<T*>(_aligned_malloc(bytes, 64));
#else
        T* new_ptr = static_cast<T*>(std::aligned_alloc(64, bytes));
#endif
        if (ptr && cap > 0) {
            std::memcpy(new_ptr, ptr, cap * sizeof(T));
#ifdef _WIN32
            _aligned_free(ptr);
#else
            std::free(ptr);
#endif
        }
        ptr = new_ptr;
        cap = new_cap;
    }

    inline T& operator[](size_t i) noexcept { return ptr[i]; }
    inline const T& operator[](size_t i) const noexcept { return ptr[i]; }
};

struct alignas(64) ChildSoAArena {
    AlignedSlab<float>   prior{32768};
    AlignedSlab<float>   value_sum{32768};
    AlignedSlab<int32_t> visit_count{32768};
    AlignedSlab<int32_t> virtual_loss{32768};
    AlignedSlab<float>   slv_q{32768};
    AlignedSlab<float>   slv_mask{32768}; // -1.0f (sign bit=1) => use slv_q; +1.0f (sign bit=0) => use normal Q
    int32_t used = 0;

    void clear() noexcept { used = 0; }

    // Allocates a 16-lane (64-byte cache-line aligned) block for `nc` children, padding tail lanes to -1e30f
    int32_t alloc_block(int32_t nc) {
        int32_t padded = (nc + 15) & ~15;
        if (padded == 0) padded = 16;
        int32_t off = used;
        [[assume((off & 15) == 0)]];
        size_t needed = (size_t)(used + padded);
        prior.reserve(needed);
        value_sum.reserve(needed);
        visit_count.reserve(needed);
        virtual_loss.reserve(needed);
        slv_q.reserve(needed);
        slv_mask.reserve(needed);

        for (int32_t i = nc; i < padded; ++i) {
            int32_t s = off + i;
            prior[(size_t)s] = 0.0f;
            value_sum[(size_t)s] = 0.0f;
            visit_count[(size_t)s] = 0;
            virtual_loss[(size_t)s] = 0;
            slv_q[(size_t)s] = -1.0e30f;
            slv_mask[(size_t)s] = -1.0f; // sign bit 1 => always selects -1e30f
        }
        used += padded;
        return off;
    }
};

struct Node {
    double value_sum;           // Exact IEEE-754 double sum of backed-up values (STM perspective)
    double prior;               // Policy prior P(s, a)
    double terminal_value;      // Terminal value (-1.0 checkmate, 0.0 draw)
    double solved_value;        // Exact solver value (-1.0, 0.0, +1.0)
    double self_value_sum;      // Direct leaf evaluation value sum
    uint64_t zobrist_hash;      // Primary 64-bit Zobrist / transposition hash
    uint64_t verify_hash;       // Secondary 64-bit occupancy mix (128-bit collision-free)
    int32_t visit_count;        // N(s, a)
    int32_t virtual_loss;       // In-flight virtual loss counter
    int32_t first_child_idx;    // Contiguous child block start in `nodes` (-1 if none)
    int32_t num_children;       // Number of children
    int32_t soa_offset;         // 64B-aligned offset into `ChildSoAArena` for this node's children (-1 if none)
    int32_t soa_slot;           // Exact slot inside parent's `ChildSoAArena` block (-1 for root)
    int32_t parent_idx;         // Parent node index (-1 for root)
    int32_t ply;                // Search depth from current root
    int32_t action_from_parent; // Action index [0..4671]
    int32_t mate_plies;         // Distance to proven mate
    int32_t self_visits;        // Direct evaluations at this node
    int32_t num_legal_moves;    // Legal move count (-1 if unknown)
    int32_t board_idx;          // Index into FastTreeArena::boards (-1 if uninstantiated)
    uint16_t move_packed;       // from_sq | (to_sq << 6) | (promo << 12)
    uint8_t is_expanded;
    uint8_t is_instantiated;
    uint8_t is_terminal;
    uint8_t is_solved;
    uint8_t path_dep_draw;
    uint8_t has_cache_key;
};

struct CacheEntry {
    uint64_t key_hi;
    uint64_t key_lo;
    double leaf_val;
    size_t byte_size;
    std::vector<uint16_t> unsorted_acts;
    std::vector<uint16_t> unsorted_moves;
    std::vector<float> unsorted_logits;
};

class LruEvalCache {
public:
    explicit LruEvalCache(size_t max_bytes = 64ULL * 1024ULL * 1024ULL)
        : max_bytes_(max_bytes), used_bytes_(0), hits_(0), misses_(0), evictions_(0) {
        map_.reserve(65536);
    }

    const CacheEntry* get(uint64_t key_hi, uint64_t key_lo) {
        auto it = map_.find(key_hi);
        if (it != map_.end()) {
            for (auto list_it : it->second) {
                if (list_it->key_lo == key_lo) {
                    entries_.splice(entries_.begin(), entries_, list_it);
                    hits_++;
                    return &(*list_it);
                }
            }
        }
        misses_++;
        return nullptr;
    }

    const CacheEntry* peek(uint64_t key_hi, uint64_t key_lo) const {
        auto it = map_.find(key_hi);
        if (it != map_.end()) {
            for (auto list_it : it->second) {
                if (list_it->key_lo == key_lo) {
                    return &(*list_it);
                }
            }
        }
        return nullptr;
    }

    const CacheEntry* put_entry(
        uint64_t key_hi,
        uint64_t key_lo,
        double leaf_val,
        int32_t n_legal,
        const uint16_t* acts,
        const uint16_t* moves_packed,
        const float* full_logits_4672
    ) {
        size_t eb = 192ULL + (size_t)n_legal * 16ULL;
        if (eb > max_bytes_) {
            return nullptr;
        }

        auto map_it = map_.find(key_hi);
        if (map_it != map_.end()) {
            auto& vec = map_it->second;
            for (size_t i = 0; i < vec.size(); ++i) {
                if (vec[i]->key_lo == key_lo) {
                    used_bytes_ -= vec[i]->byte_size;
                    entries_.erase(vec[i]);
                    vec.erase(vec.begin() + (ptrdiff_t)i);
                    if (vec.empty()) {
                        map_.erase(map_it);
                    }
                    break;
                }
            }
        }

        while (used_bytes_ + eb > max_bytes_ && !entries_.empty()) {
            auto last_it = std::prev(entries_.end());
            uint64_t ev_hi = last_it->key_hi;
            uint64_t ev_lo = last_it->key_lo;
            used_bytes_ -= last_it->byte_size;
            auto m_it = map_.find(ev_hi);
            if (m_it != map_.end()) {
                auto& vec = m_it->second;
                for (size_t i = 0; i < vec.size(); ++i) {
                    if (vec[i]->key_lo == ev_lo) {
                        vec.erase(vec.begin() + (ptrdiff_t)i);
                        break;
                    }
                }
                if (vec.empty()) {
                    map_.erase(m_it);
                }
            }
            entries_.pop_back();
            evictions_++;
        }

        CacheEntry entry;
        entry.key_hi = key_hi;
        entry.key_lo = key_lo;
        entry.leaf_val = leaf_val;
        entry.byte_size = eb;
        if (n_legal > 0) {
            entry.unsorted_acts.assign(acts, acts + n_legal);
            entry.unsorted_moves.assign(moves_packed, moves_packed + n_legal);
            entry.unsorted_logits.resize((size_t)n_legal);
            for (int32_t i = 0; i < n_legal; ++i) {
                entry.unsorted_logits[(size_t)i] = full_logits_4672[acts[i]];
            }
        }

        entries_.push_front(std::move(entry));
        map_[key_hi].push_back(entries_.begin());
        used_bytes_ += eb;
        return &(*entries_.begin());
    }

    void clear() {
        entries_.clear();
        map_.clear();
        used_bytes_ = 0;
        hits_ = 0;
        misses_ = 0;
        evictions_ = 0;
    }

    size_t size() const { return entries_.size(); }
    size_t used_bytes() const { return used_bytes_; }
    uint64_t hits() const { return hits_; }
    uint64_t misses() const { return misses_; }
    uint64_t evictions() const { return evictions_; }

private:
    size_t max_bytes_;
    size_t used_bytes_;
    uint64_t hits_;
    uint64_t misses_;
    uint64_t evictions_;
    std::list<CacheEntry> entries_;
    std::unordered_map<uint64_t, std::vector<std::list<CacheEntry>::iterator>> map_;
};

struct BatchStats {
    int64_t terminal_backups = 0;
    int64_t collisions = 0;
    int64_t nn_cache_hits = 0;
    int64_t nn_cache_misses = 0;
    int64_t cache_expanded_nodes = 0;
    int64_t dedup_avoided = 0;
    int32_t max_depth_cache = 0;
};

struct KeyPairHash {
    size_t operator()(const std::pair<uint64_t, uint64_t>& p) const noexcept {
        return (size_t)(p.first ^ (p.second * 0x9E3779B97F4A7C15ULL));
    }
};

// Exact IEEE-754 softmax helper (protected against -ffast-math reassociation for Gate 1 parity)
#if defined(__clang__)
#pragma float_control(precise, on, push)
#endif
static inline void compute_exact_f32_softmax(const float* logits, int32_t n, float* out_exp, float& out_sum) noexcept {
    float max_l = logits[0];
    for (int32_t i = 1; i < n; ++i) {
        if (logits[i] > max_l) max_l = logits[i];
    }
    float sum_e = 0.0f;
    for (int32_t i = 0; i < n; ++i) {
        float ev = std::exp(logits[i] - max_l);
        out_exp[i] = ev;
        sum_e += ev;
    }
    out_sum = sum_e;
}
#if defined(__clang__)
#pragma float_control(pop)
#endif

static inline float hsum256_ps(__m256 v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(s);
    s = _mm_add_ps(s, shuf);
    shuf = _mm_movehl_ps(shuf, s);
    s = _mm_add_ss(s, shuf);
    return _mm_cvtss_f32(s);
}

class FastTreeArena {
public:
    std::vector<Node> nodes;
    ChildSoAArena soa;
    LruEvalCache* cache;

    double c_puct;
    int use_fpu_reduction;
    double fpu_reduction;
    double root_fpu_reduction;
    int use_exact_solver;

    // Active batch collection state
    int32_t cur_batch_target = 0;
    int32_t cur_slot = 0;
    int32_t cur_max_depth = 14;
    int cur_use_nn_cache = 0;
    int cur_use_pending = 0;
    std::vector<int32_t> cur_path;

    std::vector<int32_t> leaves_to_eval;
    std::vector<std::vector<int32_t>> paths_to_leaves;
    std::vector<std::vector<int32_t>> vl_applied_paths;
    std::vector<std::vector<int32_t>> collision_paths;

    struct PendingCacheHit {
        int32_t leaf_idx;
        std::vector<int32_t> path;
        uint64_t key_hi;
        uint64_t key_lo;
    };
    std::vector<PendingCacheHit> cache_hits_to_process;
    std::unordered_set<std::pair<uint64_t, uint64_t>, KeyPairHash> batch_keys;
    std::unordered_set<int32_t> pending_nodes;
    std::vector<bb::CBoard> boards;
    std::unordered_set<std::pair<uint64_t, uint64_t>, KeyPairHash> seen_npu_keys;
    int native_board_mode = 0;

    BatchStats stats;

    FastTreeArena(
        LruEvalCache* eval_cache,
        double c_puct_,
        int use_fpu_,
        double fpu_red_,
        double root_fpu_red_,
        int use_solver_
    )
        : cache(eval_cache),
          c_puct(c_puct_),
          use_fpu_reduction(use_fpu_),
          fpu_reduction(fpu_red_),
          root_fpu_reduction(root_fpu_red_),
          use_exact_solver(use_solver_) {
        nodes.reserve(32768);
        cur_path.reserve(64);
    }

    // Zero Legal-Move Pruning: 100% of legal moves expanded at root AND all internal nodes!
    static inline int32_t branching_k_for_ply(int32_t /*ply*/, int32_t n_legal = -1) noexcept {
        return (n_legal >= 0) ? n_legal : 256;
    }

    inline void sync_node_to_soa(int32_t idx) noexcept {
        const Node& n = nodes[(size_t)idx];
        int32_t s = n.soa_slot;
        if (s < 0) return;
        soa.prior[(size_t)s] = (float)n.prior;
        soa.value_sum[(size_t)s] = (float)n.value_sum;
        soa.visit_count[(size_t)s] = n.visit_count;
        soa.virtual_loss[(size_t)s] = n.virtual_loss;
        if (n.is_solved && n.solved_value <= -1.0) {
            soa.slv_q[(size_t)s] = 1000.0f - (float)n.mate_plies;
            soa.slv_mask[(size_t)s] = -1.0f;
        } else if (n.is_solved && n.solved_value >= 1.0) {
            soa.slv_q[(size_t)s] = -1000.0f + (float)n.mate_plies;
            soa.slv_mask[(size_t)s] = -1.0f;
        } else {
            soa.slv_q[(size_t)s] = 0.0f;
            soa.slv_mask[(size_t)s] = 1.0f;
        }
    }

    inline void add_virtual_loss(int32_t idx, int32_t delta) noexcept {
        Node& n = nodes[(size_t)idx];
        n.virtual_loss += delta;
        if (n.soa_slot >= 0) {
            soa.virtual_loss[(size_t)n.soa_slot] = n.virtual_loss;
        }
    }

    void rebuild_all_soa() {
        soa.clear();
        for (size_t ni = 0; ni < nodes.size(); ++ni) {
            nodes[ni].soa_offset = -1;
            if (nodes[ni].parent_idx < 0) {
                nodes[ni].soa_slot = -1;
            }
        }
        for (size_t ni = 0; ni < nodes.size(); ++ni) {
            int32_t fc = nodes[ni].first_child_idx;
            int32_t nc = nodes[ni].num_children;
            if (fc >= 0 && nc > 0) {
                int32_t off = soa.alloc_block(nc);
                nodes[ni].soa_offset = off;
                for (int32_t c = 0; c < nc; ++c) {
                    int32_t ci = fc + c;
                    nodes[(size_t)ci].soa_slot = off + c;
                    sync_node_to_soa(ci);
                }
            }
        }
    }

    int32_t reset_root() {
        nodes.clear();
        soa.clear();
        boards.clear();
        pending_nodes.clear();
        native_board_mode = 0;
        Node root{};
        root.first_child_idx = -1;
        root.soa_offset = -1;
        root.soa_slot = -1;
        root.parent_idx = -1;
        root.action_from_parent = -1;
        root.num_legal_moves = -1;
        root.board_idx = -1;
        root.is_instantiated = 1;
        nodes.push_back(root);
        return 0;
    }

    inline void instantiate_node_native(int32_t node_idx) {
        if (nodes[(size_t)node_idx].is_instantiated && nodes[(size_t)node_idx].board_idx >= 0) return;
        int32_t p_idx = nodes[(size_t)node_idx].parent_idx;
        int32_t p_bidx = nodes[(size_t)p_idx].board_idx;
        bb::CBoard cb = boards[(size_t)p_bidx];
        cb.push(nodes[(size_t)node_idx].move_packed);
        int32_t new_bidx = (int32_t)boards.size();
        boards.push_back(cb);

        int32_t iterm, isolv, pdep;
        double tval, sval;
        boards[(size_t)new_bidx].evaluate_terminal(iterm, tval, isolv, sval, pdep);

        Node& n = nodes[(size_t)node_idx];
        n.board_idx = new_bidx;
        n.is_instantiated = 1;
        n.is_terminal = (uint8_t)iterm;
        n.terminal_value = tval;
        n.is_solved = (uint8_t)isolv;
        n.solved_value = sval;
        n.path_dep_draw = (uint8_t)pdep;
        n.zobrist_hash = boards[(size_t)new_bidx].zobrist_hi;
        n.verify_hash = boards[(size_t)new_bidx].zobrist_lo;
        n.has_cache_key = 1;
        if (isolv) {
            sync_node_to_soa(node_idx);
        }
    }

    inline double compute_fpu_exact(int32_t node_idx) const noexcept {
#if defined(__clang__)
#pragma float_control(precise, on)
#endif
        const Node& n = nodes[(size_t)node_idx];
        if (!use_fpu_reduction || n.visit_count == 0) {
            return 0.0;
        }
        double visited_mass = 0.0;
        int32_t fc = n.first_child_idx;
        int32_t nc = n.num_children;
        for (int32_t i = 0; i < nc; ++i) {
            const Node& ch = nodes[(size_t)(fc + i)];
            if (ch.visit_count + ch.virtual_loss > 0) {
                visited_mass += ch.prior;
            }
        }
        double q_val = n.value_sum / (double)n.visit_count;
        double c_fpu = (n.ply == 0) ? root_fpu_reduction : fpu_reduction;
        double res = q_val - c_fpu * std::sqrt(visited_mass);
        if (res < -1.0) return -1.0;
        if (res > 1.0) return 1.0;
        return res;
    }

    static inline double q_from_parent_view_exact(const Node& ch, double fpu_value) noexcept {
#if defined(__clang__)
#pragma float_control(precise, on)
#endif
        if (ch.is_solved) {
            if (ch.solved_value <= -1.0) {
                return 1000.0 - (double)ch.mate_plies;
            }
            if (ch.solved_value >= 1.0) {
                return -1000.0 + (double)ch.mate_plies;
            }
        }
        int32_t eff_visits = ch.visit_count + ch.virtual_loss;
        if (eff_visits == 0) {
            return fpu_value;
        }
        double parent_total = -ch.value_sum - (double)ch.virtual_loss;
        return parent_total / (double)eff_visits;
    }

    inline double child_score_exact(const Node& ch, double fpu_val, double sqrt_n) const noexcept {
#if defined(__clang__)
#pragma float_control(precise, on)
#endif
        double q_parent = q_from_parent_view_exact(ch, fpu_val);
        int32_t eff_v = ch.visit_count + ch.virtual_loss;
        double u = c_puct * ch.prior * (sqrt_n / (double)(1 + eff_v));
        return q_parent + u;
    }

    // 64B-Aligned SoA AVX2 PUCT Child Selector (8 children per instruction via _mm256_fmadd_ps & _mm256_sqrt_ps)
    inline int32_t select_best_child_avx2(int32_t curr) const noexcept {
        const Node& cn = nodes[(size_t)curr];
        const int32_t fc = cn.first_child_idx;
        const int32_t nc = cn.num_children;
        const int32_t off = cn.soa_offset;
        if (nc <= 0 || fc < 0 || off < 0) return -1;
        [[assume(nc > 0 && nc <= 256)]];
        [[assume((off & 15) == 0)]];

        const float*   __restrict pr_p  = soa.prior.ptr + off;
        const float*   __restrict vs_p  = soa.value_sum.ptr + off;
        const int32_t* __restrict vc_p  = soa.visit_count.ptr + off;
        const int32_t* __restrict vl_p  = soa.virtual_loss.ptr + off;
        const float*   __restrict sq_p  = soa.slv_q.ptr + off;
        const float*   __restrict sm_p  = soa.slv_mask.ptr + off;

        // 1. AVX2 visited-mass reduction + _mm256_sqrt_ps for FPU and sqrt(parent_visits)
        float fpu_f = 0.0f;
        float sqrt_n_f = 1.0f;
        if (cn.visit_count > 0) {
            if (use_fpu_reduction) {
                __m256 v_mass = _mm256_setzero_ps();
                const __m256i v_zero_i = _mm256_setzero_si256();
                for (int32_t i = 0; i < nc; i += 8) {
                    __m256i vis = _mm256_load_si256((const __m256i*)(vc_p + i));
                    __m256i vl  = _mm256_load_si256((const __m256i*)(vl_p + i));
                    __m256i eff = _mm256_add_epi32(vis, vl);
                    __m256i gt0 = _mm256_cmpgt_epi32(eff, v_zero_i);
                    __m256 pr   = _mm256_load_ps(pr_p + i);
                    v_mass = _mm256_add_ps(v_mass, _mm256_and_ps(pr, _mm256_castsi256_ps(gt0)));
                }
                float mass_f = hsum256_ps(v_mass);
                // Simultaneous AVX2 _mm256_sqrt_ps on [parent_visits, visited_mass, ...]
                __m256 sq_in = _mm256_set_ps(1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, mass_f, (float)cn.visit_count);
                __m256 sq_out = _mm256_sqrt_ps(sq_in);
                alignas(32) float sq_buf[8];
                _mm256_store_ps(sq_buf, sq_out);
                sqrt_n_f = sq_buf[0];
                float q_node = (float)cn.value_sum / (float)cn.visit_count;
                float c_fpu = (cn.ply == 0) ? (float)root_fpu_reduction : (float)fpu_reduction;
                fpu_f = q_node - c_fpu * sq_buf[1];
                if (fpu_f < -1.0f) fpu_f = -1.0f;
                if (fpu_f > 1.0f)  fpu_f = 1.0f;
            } else {
                __m256 sq_out = _mm256_sqrt_ps(_mm256_set1_ps((float)cn.visit_count));
                sqrt_n_f = _mm256_cvtss_f32(sq_out);
            }
        }

        // 2. 8-wide SoA AVX2 Q + U evaluation (`_mm256_fmadd_ps`)
        const __m256 v_fpu      = _mm256_set1_ps(fpu_f);
        const __m256 v_c_sqrtn  = _mm256_set1_ps((float)c_puct * sqrt_n_f);
        const __m256 v_one      = _mm256_set1_ps(1.0f);
        const __m256 v_zero_f   = _mm256_setzero_ps();
        const __m256i v_zero_i  = _mm256_setzero_si256();

        alignas(64) float scores_buf[256];
        __m256 v_max = _mm256_set1_ps(-1.0e30f);

        int32_t n_blocks = (nc + 7) >> 3;
        [[assume(n_blocks >= 1 && n_blocks <= 32)]];
        for (int32_t b = 0; b < n_blocks; ++b) {
            int32_t i = b << 3;
            __m256i v_vis = _mm256_load_si256((const __m256i*)(vc_p + i));
            __m256i v_vl  = _mm256_load_si256((const __m256i*)(vl_p + i));
            __m256i v_eff_i = _mm256_add_epi32(v_vis, v_vl);
            __m256  v_eff_f = _mm256_cvtepi32_ps(v_eff_i);

            __m256 v_vsum = _mm256_load_ps(vs_p + i);
            __m256 v_vl_f = _mm256_cvtepi32_ps(v_vl);
            __m256 v_ptot = _mm256_sub_ps(_mm256_sub_ps(v_zero_f, v_vsum), v_vl_f);

            __m256 v_unvis_m = _mm256_castsi256_ps(_mm256_cmpeq_epi32(v_eff_i, v_zero_i));
            __m256 v_safe_den = _mm256_blendv_ps(v_eff_f, v_one, v_unvis_m);
            __m256 v_q_vis    = _mm256_div_ps(v_ptot, v_safe_den);
            __m256 v_q        = _mm256_blendv_ps(v_q_vis, v_fpu, v_unvis_m);

            __m256 v_slv_m = _mm256_load_ps(sm_p + i);
            __m256 v_slv_q = _mm256_load_ps(sq_p + i);
            v_q = _mm256_blendv_ps(v_q, v_slv_q, v_slv_m);

            __m256 v_prior = _mm256_load_ps(pr_p + i);
            __m256 v_u_div = _mm256_div_ps(v_prior, _mm256_add_ps(v_one, v_eff_f));
            __m256 v_score = _mm256_fmadd_ps(v_c_sqrtn, v_u_div, v_q);

            _mm256_store_ps(scores_buf + i, v_score);
            v_max = _mm256_max_ps(v_max, v_score);
        }

        // Horizontal max across 8 lanes of v_max
        __m128 m_lo = _mm256_castps256_ps128(v_max);
        __m128 m_hi = _mm256_extractf128_ps(v_max, 1);
        __m128 m4 = _mm_max_ps(m_lo, m_hi);
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
        float max_score = _mm_cvtss_f32(m4);

        // 3. Fast AVX2 near-max candidate extraction:
        //    If a single unique child wins by > 1e-4f (99.9%+ of visits), return it directly in O(1)!
        //    Only if >= 2 children lie within 1e-4f of max_score, break the near-tie in exact IEEE-754 double.
        const __m256 v_thresh = _mm256_set1_ps(max_score - 1.0e-4f);
        int32_t first_cand = -1;
        int32_t cand_count = 0;
        uint32_t block_masks[32];

        for (int32_t b = 0; b < n_blocks; ++b) {
            __m256 vs = _mm256_load_ps(scores_buf + (b << 3));
            uint32_t m = (uint32_t)_mm256_movemask_ps(_mm256_cmp_ps(vs, v_thresh, _CMP_GE_OQ));
            if ((b + 1 == n_blocks) && (nc & 7)) {
                m &= (1U << (nc & 7)) - 1U;
            }
            block_masks[b] = m;
            if (m) {
                int c = (int)_mm_popcnt_u32(m);
                if (first_cand < 0) {
                    first_cand = (b << 3) + (int)_tzcnt_u32(m);
                }
                cand_count += c;
            }
        }

        if (cand_count == 1) {
            return fc + first_cand;
        }

        // Exact IEEE-754 double tie-resolver on ONLY the near-tied candidates
        double exact_sqrt_n = (cn.visit_count > 0) ? std::sqrt((double)cn.visit_count) : 1.0;
        double exact_fpu = compute_fpu_exact(curr);
        double best_exact = -1.0e300;
        int32_t best_child = (first_cand >= 0) ? (fc + first_cand) : fc;

        if (cand_count > 1) {
            for (int32_t b = 0; b < n_blocks; ++b) {
                uint32_t m = block_masks[b];
                while (m) {
                    int lane = (int)_tzcnt_u32(m);
                    m &= (m - 1U);
                    int32_t c_idx = fc + (b << 3) + lane;
                    double sc = child_score_exact(nodes[(size_t)c_idx], exact_fpu, exact_sqrt_n);
                    if (sc > best_exact) {
                        best_exact = sc;
                        best_child = c_idx;
                    }
                }
            }
        } else {
            for (int32_t i = 0; i < nc; ++i) {
                int32_t c_idx = fc + i;
                double sc = child_score_exact(nodes[(size_t)c_idx], exact_fpu, exact_sqrt_n);
                if (sc > best_exact) {
                    best_exact = sc;
                    best_child = c_idx;
                }
            }
        }
        return best_child;
    }

    void update_solver_along_path(const std::vector<int32_t>& path) {
        if (!use_exact_solver) return;
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            int32_t n_idx = *it;
            Node& n = nodes[(size_t)n_idx];
            if (n.is_terminal || !n.is_expanded || n.num_children == 0) {
                continue;
            }
            int32_t min_win_plies = -1;
            int32_t max_loss_plies = 0;
            bool all_losing_children = true;
            int32_t fc = n.first_child_idx;
            int32_t nc = n.num_children;
            for (int32_t i = 0; i < nc; ++i) {
                const Node& ch = nodes[(size_t)(fc + i)];
                if (ch.is_solved && ch.solved_value <= -1.0) {
                    int32_t cand = 1 + ch.mate_plies;
                    if (min_win_plies < 0 || cand < min_win_plies) {
                        min_win_plies = cand;
                    }
                }
                if (!(ch.is_solved && ch.solved_value >= 1.0)) {
                    all_losing_children = false;
                } else {
                    int32_t cand_loss = 1 + ch.mate_plies;
                    if (cand_loss > max_loss_plies) {
                        max_loss_plies = cand_loss;
                    }
                }
            }
            if (min_win_plies >= 0) {
                n.is_solved = 1;
                n.solved_value = 1.0;
                if (n.mate_plies == 0 || min_win_plies < n.mate_plies) {
                    n.mate_plies = min_win_plies;
                }
                sync_node_to_soa(n_idx);
            } else if (!n.is_solved && all_losing_children && n.num_legal_moves > 0 && nc >= n.num_legal_moves) {
                n.is_solved = 1;
                n.solved_value = -1.0;
                n.mate_plies = max_loss_plies;
                sync_node_to_soa(n_idx);
            }
        }
    }

    double expand_node_from_cache_entry(int32_t node_idx, const CacheEntry& entry, int32_t /*top_k*/) {
        int32_t n_legal = (int32_t)entry.unsorted_acts.size();
        nodes[(size_t)node_idx].num_legal_moves = n_legal;
        if (n_legal == 0) {
            return entry.leaf_val;
        }

        float e_l[256];
        float sum_e = 0.0f;
        compute_exact_f32_softmax(entry.unsorted_logits.data(), n_legal, e_l, sum_e);

        attach_or_update_children(
            node_idx,
            n_legal,
            entry.unsorted_acts.data(),
            entry.unsorted_moves.data(),
            e_l,
            sum_e
        );
        return entry.leaf_val;
    }

    static inline double exact_prior_div(float ev, float sum_exp) noexcept {
#if defined(__clang__)
#pragma float_control(precise, on)
#endif
        return (double)(ev / sum_exp);
    }

    void attach_or_update_children(
        int32_t node_idx,
        int32_t sel_len,
        const uint16_t* c_acts,
        const uint16_t* c_moves,
        const float* exp_logits,
        float sum_exp
    ) {
        int32_t old_fc = nodes[(size_t)node_idx].first_child_idx;
        int32_t old_nc = nodes[(size_t)node_idx].num_children;
        int32_t child_ply = nodes[(size_t)node_idx].ply + 1;
        int32_t soa_off = soa.alloc_block(sel_len);

        if (old_nc == 0 || old_fc < 0) {
            int32_t new_fc = (int32_t)nodes.size();
            nodes.resize(nodes.size() + (size_t)sel_len);
            for (int32_t i = 0; i < sel_len; ++i) {
                int32_t c_idx = new_fc + i;
                Node ch{};
                ch.prior = exact_prior_div(exp_logits[i], sum_exp);
                ch.first_child_idx = -1;
                ch.soa_offset = -1;
                ch.soa_slot = soa_off + i;
                ch.parent_idx = node_idx;
                ch.ply = child_ply;
                ch.action_from_parent = (int32_t)c_acts[i];
                ch.num_legal_moves = -1;
                ch.board_idx = -1;
                ch.move_packed = c_moves[i];
                nodes[(size_t)c_idx] = ch;
                sync_node_to_soa(c_idx);
            }
            nodes[(size_t)node_idx].first_child_idx = new_fc;
            nodes[(size_t)node_idx].num_children = sel_len;
            nodes[(size_t)node_idx].soa_offset = soa_off;
            nodes[(size_t)node_idx].is_expanded = 1;
        } else {
            int32_t new_fc = (int32_t)nodes.size();
            nodes.resize(nodes.size() + (size_t)sel_len);
            for (int32_t i = 0; i < sel_len; ++i) {
                int32_t act = (int32_t)c_acts[i];
                double pr = exact_prior_div(exp_logits[i], sum_exp);
                int32_t matched_old = -1;
                for (int32_t j = 0; j < old_nc; ++j) {
                    if (nodes[(size_t)(old_fc + j)].action_from_parent == act) {
                        matched_old = old_fc + j;
                        break;
                    }
                }
                int32_t new_c_idx = new_fc + i;
                if (matched_old >= 0) {
                    Node copied = nodes[(size_t)matched_old];
                    copied.prior = pr;
                    copied.ply = child_ply;
                    copied.parent_idx = node_idx;
                    copied.soa_slot = soa_off + i;
                    nodes[(size_t)new_c_idx] = copied;
                    if (copied.first_child_idx >= 0 && copied.num_children > 0) {
                        for (int32_t g = 0; g < copied.num_children; ++g) {
                            nodes[(size_t)(copied.first_child_idx + g)].parent_idx = new_c_idx;
                        }
                    }
                } else {
                    Node ch{};
                    ch.prior = pr;
                    ch.first_child_idx = -1;
                    ch.soa_offset = -1;
                    ch.soa_slot = soa_off + i;
                    ch.parent_idx = node_idx;
                    ch.ply = child_ply;
                    ch.action_from_parent = act;
                    ch.num_legal_moves = -1;
                    ch.board_idx = -1;
                    ch.move_packed = c_moves[i];
                    nodes[(size_t)new_c_idx] = ch;
                }
                sync_node_to_soa(new_c_idx);
            }
            nodes[(size_t)node_idx].first_child_idx = new_fc;
            nodes[(size_t)node_idx].num_children = sel_len;
            nodes[(size_t)node_idx].soa_offset = soa_off;
            nodes[(size_t)node_idx].is_expanded = 1;
        }
    }

    inline void backup_value(int32_t leaf_idx, const std::vector<int32_t>& path, double leaf_val) {
#if defined(__clang__)
#pragma float_control(precise, on)
#endif
        nodes[(size_t)leaf_idx].self_visits += 1;
        nodes[(size_t)leaf_idx].self_value_sum += leaf_val;
        double val = leaf_val;
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            int32_t idx = *it;
            Node& n = nodes[(size_t)idx];
            n.visit_count += 1;
            n.value_sum += val;
            if (n.soa_slot >= 0) {
                soa.visit_count[(size_t)n.soa_slot] = n.visit_count;
                soa.value_sum[(size_t)n.soa_slot] = (float)n.value_sum;
            }
            val = -val;
        }
        update_solver_along_path(path);
    }

    int32_t run_selection_slots(int32_t* out_node_idx, int32_t* out_parent_idx, uint16_t* out_move_packed) {
        while (cur_slot < cur_batch_target) {
            if (use_exact_solver && nodes[0].is_solved && nodes[0].mate_plies <= 3) {
                break;
            }

            int32_t curr = 0;
            cur_path.clear();
            cur_path.push_back(0);

            while (nodes[(size_t)curr].is_expanded &&
                   !nodes[(size_t)curr].is_terminal &&
                   !(use_exact_solver && nodes[(size_t)curr].is_solved)) {
                int32_t best_child = select_best_child_avx2(curr);
                if (best_child < 0) break;
                curr = best_child;
                cur_path.push_back(curr);
            }

            if (!nodes[(size_t)curr].is_instantiated) {
                if (native_board_mode) {
                    instantiate_node_native(curr);
                } else {
                    *out_node_idx = curr;
                    *out_parent_idx = nodes[(size_t)curr].parent_idx;
                    *out_move_packed = nodes[(size_t)curr].move_packed;
                    return 1;
                }
            }

            process_instantiated_leaf(curr);
            cur_slot++;
        }

        finalize_batch();
        *out_node_idx = -1;
        *out_parent_idx = -1;
        *out_move_packed = 0;
        return 0;
    }

    void process_instantiated_leaf(int32_t curr) {
        Node& leaf = nodes[(size_t)curr];

        if (leaf.is_terminal || (use_exact_solver && leaf.is_solved)) {
            double val = (use_exact_solver && leaf.is_solved) ? leaf.solved_value : leaf.terminal_value;
            backup_value(curr, cur_path, val);
            stats.terminal_backups += 1;
            return;
        }

        if (cur_use_pending && pending_nodes.find(curr) != pending_nodes.end()) {
            for (size_t i = 1; i < cur_path.size(); ++i) {
                add_virtual_loss(cur_path[i], 1);
            }
            collision_paths.push_back(cur_path);
            stats.collisions += 1;
            return;
        }

        if (cur_use_nn_cache && cache != nullptr && !leaf.is_expanded && leaf.self_visits == 0 && leaf.has_cache_key) {
            uint64_t khi = leaf.zobrist_hash;
            uint64_t klo = leaf.verify_hash;
            const CacheEntry* hit = cache->get(khi, klo);
            if (hit != nullptr) {
                cache_hits_to_process.push_back({curr, cur_path, khi, klo});
                for (size_t i = 1; i < cur_path.size(); ++i) {
                    add_virtual_loss(cur_path[i], 1);
                }
                collision_paths.push_back(cur_path);
                stats.nn_cache_hits += 1;
                stats.cache_expanded_nodes += 1;
                if (leaf.ply > stats.max_depth_cache) {
                    stats.max_depth_cache = leaf.ply;
                }
                return;
            } else {
                stats.nn_cache_misses += 1;
                auto kp = std::make_pair(khi, klo);
                if (batch_keys.find(kp) != batch_keys.end()) {
                    stats.dedup_avoided += 1;
                    return;
                }
                batch_keys.insert(kp);
            }
        }

        bool already_queued = false;
        for (int32_t existing : leaves_to_eval) {
            if (existing == curr) {
                already_queued = true;
                break;
            }
        }
        if (!already_queued) {
            leaves_to_eval.push_back(curr);
            paths_to_leaves.push_back(cur_path);
            for (size_t i = 1; i < cur_path.size(); ++i) {
                add_virtual_loss(cur_path[i], 1);
            }
            vl_applied_paths.push_back(cur_path);
            if (cur_use_pending) {
                pending_nodes.insert(curr);
            }
        }
    }

    void finalize_batch() {
        for (const auto& cpath : collision_paths) {
            for (size_t i = 1; i < cpath.size(); ++i) {
                add_virtual_loss(cpath[i], -1);
            }
        }
        collision_paths.clear();

        for (const auto& ch : cache_hits_to_process) {
            const CacheEntry* entry = cache ? cache->peek(ch.key_hi, ch.key_lo) : nullptr;
            if (entry == nullptr) continue;
            double leaf_val = entry->leaf_val;
            if (nodes[(size_t)ch.leaf_idx].ply < cur_max_depth && !nodes[(size_t)ch.leaf_idx].is_expanded) {
                int32_t bk = branching_k_for_ply(nodes[(size_t)ch.leaf_idx].ply, (int32_t)entry->unsorted_acts.size());
                leaf_val = expand_node_from_cache_entry(ch.leaf_idx, *entry, bk);
            }
            backup_value(ch.leaf_idx, ch.path, leaf_val);
        }
        cache_hits_to_process.clear();
    }
};

// ============================================================================
// Core Ultra 5 125U Hybrid P-Core Pinning & HighQoS (Win32)
// ============================================================================

FT_EXPORT int32_t ft_pin_current_thread_to_pcores(
    uint64_t* out_affinity_mask,
    int32_t* out_max_eff_class,
    int32_t* out_high_qos_ok
) {
#ifdef _WIN32
    HANDLE hThread = GetCurrentThread();

    // 1. Disable Windows 11 EcoQoS / LP-E power throttling (HighQoS execution speed)
    struct FtPowerThrottlingState {
        ULONG Version;
        ULONG ControlMask;
        ULONG StateMask;
    };
    FtPowerThrottlingState throttling{};
    throttling.Version = 1;      // THREAD_POWER_THROTTLING_CURRENT_VERSION
    throttling.ControlMask = 1;  // THREAD_POWER_THROTTLING_EXECUTION_SPEED
    throttling.StateMask = 0;    // 0 = HighQoS (never throttle or park onto LP-E cores)
    BOOL qos_ok = SetThreadInformation(
        hThread,
        static_cast<THREAD_INFORMATION_CLASS>(3), // ThreadPowerThrottling = 3
        &throttling,
        sizeof(throttling)
    );
    SetThreadPriority(hThread, THREAD_PRIORITY_ABOVE_NORMAL);

    // 2. Discover highest EfficiencyClass cores (Redwood Cove P-cores on Core Ultra 5 125U)
    DWORD buf_len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &buf_len);
    uint64_t pcore_mask = 0;
    int32_t max_eff = -1;

    if (buf_len > 0) {
        std::vector<uint8_t> buf((size_t)buf_len);
        auto* base = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data());
        if (GetLogicalProcessorInformationEx(RelationProcessorCore, base, &buf_len)) {
            DWORD offset = 0;
            while (offset < buf_len) {
                auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + offset);
                if (info->Relationship == RelationProcessorCore) {
                    int32_t eff = (int32_t)info->Processor.EfficiencyClass;
                    if (eff > max_eff) {
                        max_eff = eff;
                    }
                }
                offset += info->Size;
            }
            offset = 0;
            while (offset < buf_len) {
                auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + offset);
                if (info->Relationship == RelationProcessorCore &&
                    (int32_t)info->Processor.EfficiencyClass == max_eff &&
                    info->Processor.GroupCount >= 1) {
                    pcore_mask |= (uint64_t)info->Processor.GroupMask[0].Mask;
                }
                offset += info->Size;
            }
        }
    }

    int32_t pinned_ok = 0;
    if (pcore_mask != 0) {
        DWORD_PTR prev = SetThreadAffinityMask(hThread, (DWORD_PTR)pcore_mask);
        pinned_ok = (prev != 0) ? 1 : 0;
    }

    if (out_affinity_mask) *out_affinity_mask = pcore_mask;
    if (out_max_eff_class) *out_max_eff_class = max_eff;
    if (out_high_qos_ok) *out_high_qos_ok = qos_ok ? 1 : 0;
    return pinned_ok;
#else
    if (out_affinity_mask) *out_affinity_mask = 0;
    if (out_max_eff_class) *out_max_eff_class = 0;
    if (out_high_qos_ok) *out_high_qos_ok = 0;
    return 0;
#endif
}

// ============================================================================
// Exported C ABI Functions
// ============================================================================

FT_EXPORT void* ft_cache_create(uint64_t max_bytes) {
    return new LruEvalCache((size_t)max_bytes);
}

FT_EXPORT void ft_cache_destroy(void* cache_ptr) {
    delete static_cast<LruEvalCache*>(cache_ptr);
}

FT_EXPORT void ft_cache_clear(void* cache_ptr) {
    if (cache_ptr) static_cast<LruEvalCache*>(cache_ptr)->clear();
}

FT_EXPORT void ft_cache_stats(
    void* cache_ptr,
    uint64_t* out_entries,
    uint64_t* out_used_bytes,
    uint64_t* out_hits,
    uint64_t* out_misses,
    uint64_t* out_evictions
) {
    auto* c = static_cast<LruEvalCache*>(cache_ptr);
    if (!c) return;
    *out_entries = (uint64_t)c->size();
    *out_used_bytes = (uint64_t)c->used_bytes();
    *out_hits = c->hits();
    *out_misses = c->misses();
    *out_evictions = c->evictions();
}

FT_EXPORT int32_t ft_cache_contains(void* cache_ptr, uint64_t key_hi, uint64_t key_lo) {
    auto* c = static_cast<LruEvalCache*>(cache_ptr);
    if (!c) return 0;
    return (c->peek(key_hi, key_lo) != nullptr) ? 1 : 0;
}

FT_EXPORT void* ft_tree_create(
    void* cache_ptr,
    double c_puct,
    int32_t use_fpu_reduction,
    double fpu_reduction,
    double root_fpu_reduction,
    int32_t use_exact_solver
) {
    return new FastTreeArena(
        static_cast<LruEvalCache*>(cache_ptr),
        c_puct,
        use_fpu_reduction,
        fpu_reduction,
        root_fpu_reduction,
        use_exact_solver
    );
}

FT_EXPORT void ft_tree_destroy(void* tree_ptr) {
    delete static_cast<FastTreeArena*>(tree_ptr);
}

FT_EXPORT int32_t ft_tree_reset_root(void* tree_ptr, uint64_t key_hi, uint64_t key_lo) {
    auto* t = static_cast<FastTreeArena*>(tree_ptr);
    int32_t r = t->reset_root();
    t->nodes[0].zobrist_hash = key_hi;
    t->nodes[0].verify_hash = key_lo;
    t->nodes[0].has_cache_key = 1;
    t->stats = BatchStats{};
    return r;
}

FT_EXPORT void ft_tree_reset_stats(void* tree_ptr) {
    auto* t = static_cast<FastTreeArena*>(tree_ptr);
    t->stats = BatchStats{};
}

FT_EXPORT int32_t ft_tree_num_nodes(void* tree_ptr) {
    return (int32_t)static_cast<FastTreeArena*>(tree_ptr)->nodes.size();
}

FT_EXPORT void ft_tree_set_root_fpu_visit(void* tree_ptr, double root_v) {
    auto* t = static_cast<FastTreeArena*>(tree_ptr);
    Node& r = t->nodes[0];
    if (r.visit_count == 0) {
        r.self_visits = 1;
        r.self_value_sum = root_v;
        r.visit_count = 1;
        r.value_sum = root_v;
    }
}

FT_EXPORT int32_t ft_expand_root_from_cache(void* tree_ptr, uint64_t key_hi, uint64_t key_lo, int32_t n_legals, double* out_root_val) {
    auto* t = static_cast<FastTreeArena*>(tree_ptr);
    if (!t->cache) return 0;
    const CacheEntry* entry = t->cache->get(key_hi, key_lo);
    if (!entry) {
        t->stats.nn_cache_misses += 1;
        return 0;
    }
    t->stats.nn_cache_hits += 1;
    t->stats.cache_expanded_nodes += 1;
    *out_root_val = t->expand_node_from_cache_entry(0, *entry, n_legals);
    return 1;
}

FT_EXPORT double ft_expand_node_with_logits(
    void* tree_ptr,
    int32_t node_idx,
    uint64_t key_hi,
    uint64_t key_lo,
    double leaf_val,
    int32_t n_legal,
    const uint16_t* acts,
    const uint16_t* moves_packed,
    const float* full_logits_4672,
    const double* /*tactical_bonuses_unused_no_pruning*/,
    int32_t top_k,
    int32_t store_in_cache
) {
    auto* t = static_cast<FastTreeArena*>(tree_ptr);
    t->nodes[(size_t)node_idx].num_legal_moves = n_legal;
    t->nodes[(size_t)node_idx].zobrist_hash = key_hi;
    t->nodes[(size_t)node_idx].verify_hash = key_lo;
    t->nodes[(size_t)node_idx].has_cache_key = 1;

    if (store_in_cache && t->cache != nullptr) {
        const CacheEntry* ce = t->cache->put_entry(
            key_hi, key_lo, leaf_val, n_legal, acts, moves_packed, full_logits_4672
        );
        if (ce != nullptr && top_k > 0) {
            return t->expand_node_from_cache_entry(node_idx, *ce, n_legal);
        }
    }

    if (n_legal <= 0 || top_k <= 0) {
        return leaf_val;
    }

    // 100% of legal moves expanded (zero pruning!)
    float l_sel[256];
    float e_l[256];
    for (int32_t i = 0; i < n_legal; ++i) {
        l_sel[i] = full_logits_4672[acts[i]];
    }
    float sum_e = 0.0f;
    compute_exact_f32_softmax(l_sel, n_legal, e_l, sum_e);

    t->attach_or_update_children(node_idx, n_legal, acts, moves_packed, e_l, sum_e);
    return leaf_val;
}

FT_EXPORT int32_t ft_select_begin(
    void* tree_ptr,
    int32_t batch_size,
    int32_t max_depth,
    int32_t use_nn_cache,
    int32_t use_pending,
    int32_t* out_node_idx,
    int32_t* out_parent_idx,
    uint16_t* out_move_packed
) {
    auto* t = static_cast<FastTreeArena*>(tree_ptr);
    t->cur_batch_target = batch_size;
    t->cur_slot = 0;
    t->cur_max_depth = max_depth;
    t->cur_use_nn_cache = use_nn_cache;
    t->cur_use_pending = use_pending;
    t->leaves_to_eval.clear();
    t->paths_to_leaves.clear();
    t->vl_applied_paths.clear();
    t->collision_paths.clear();
    t->cache_hits_to_process.clear();
    t->batch_keys.clear();

    return t->run_selection_slots(out_node_idx, out_parent_idx, out_move_packed);
}

FT_EXPORT int32_t ft_select_provide_instantiation(
    void* tree_ptr,
    int32_t node_idx,
    int32_t is_terminal,
    double terminal_value,
    int32_t is_solved,
    double solved_value,
    int32_t path_dep_draw,
    uint64_t key_hi,
    uint64_t key_lo,
    int32_t* out_node_idx,
    int32_t* out_parent_idx,
    uint16_t* out_move_packed
) {
    auto* t = static_cast<FastTreeArena*>(tree_ptr);
    Node& n = t->nodes[(size_t)node_idx];
    n.is_instantiated = 1;
    n.is_terminal = (uint8_t)is_terminal;
    n.terminal_value = terminal_value;
    n.is_solved = (uint8_t)is_solved;
    n.solved_value = solved_value;
    n.path_dep_draw = (uint8_t)path_dep_draw;
    n.zobrist_hash = key_hi;
    n.verify_hash = key_lo;
    n.has_cache_key = 1;
    if (is_solved) {
        t->sync_node_to_soa(node_idx);
    }

    t->process_instantiated_leaf(node_idx);
    t->cur_slot++;
    return t->run_selection_slots(out_node_idx, out_parent_idx, out_move_packed);
}

struct InflightBatchHandle {
    std::vector<int32_t> leaves;
    std::vector<std::vector<int32_t>> paths;
    std::vector<std::vector<int32_t>> vl_paths;
    int32_t max_leaf_ply = 0;
};

FT_EXPORT void* ft_pop_inflight_batch(void* tree_ptr, int32_t* out_leaves, int32_t* out_n_eval, int32_t* out_max_ply) {
    auto* t = static_cast<FastTreeArena*>(tree_ptr);
    auto* h = new InflightBatchHandle();
    h->leaves.swap(t->leaves_to_eval);
    h->paths.swap(t->paths_to_leaves);
    h->vl_paths.swap(t->vl_applied_paths);

    int32_t n_eval = (int32_t)h->leaves.size();
    int32_t max_ply = 0;
    for (int32_t i = 0; i < n_eval; ++i) {
        int32_t li = h->leaves[(size_t)i];
        out_leaves[i] = li;
        int32_t p = t->nodes[(size_t)li].ply;
        if (p > max_ply) max_ply = p;
    }
    h->max_leaf_ply = max_ply;
    *out_n_eval = n_eval;
    *out_max_ply = max_ply;
    return h;
}

FT_EXPORT void ft_revert_batch_vl_and_clear_pending(void* tree_ptr, void* batch_handle) {
    auto* t = static_cast<FastTreeArena*>(tree_ptr);
    auto* h = static_cast<InflightBatchHandle*>(batch_handle);
    for (const auto& vp : h->vl_paths) {
        for (size_t i = 1; i < vp.size(); ++i) {
            t->add_virtual_loss(vp[i], -1);
        }
    }
    h->vl_paths.clear();
    for (int32_t li : h->leaves) {
        t->pending_nodes.erase(li);
    }
}

FT_EXPORT void ft_backup_single_leaf_from_batch(
    void* tree_ptr,
    void* batch_handle,
    int32_t slot_idx,
    uint64_t key_hi,
    uint64_t key_lo,
    double leaf_val,
    int32_t n_legal,
    const uint16_t* acts,
    const uint16_t* moves_packed,
    const float* full_logits_4672,
    const double* tactical_bonuses,
    int32_t max_depth,
    int32_t store_in_cache
) {
    auto* t = static_cast<FastTreeArena*>(tree_ptr);
    auto* h = static_cast<InflightBatchHandle*>(batch_handle);
    int32_t leaf_idx = h->leaves[(size_t)slot_idx];
    const auto& path = h->paths[(size_t)slot_idx];

    int32_t ply = t->nodes[(size_t)leaf_idx].ply;
    int32_t do_expand = (ply < max_depth && !t->nodes[(size_t)leaf_idx].is_expanded) ? 1 : 0;
    int32_t bk = do_expand ? FastTreeArena::branching_k_for_ply(ply, n_legal) : 0;

    ft_expand_node_with_logits(
        tree_ptr,
        leaf_idx,
        key_hi,
        key_lo,
        leaf_val,
        n_legal,
        acts,
        moves_packed,
        full_logits_4672,
        tactical_bonuses,
        bk,
        store_in_cache
    );

    t->backup_value(leaf_idx, path, leaf_val);
}

FT_EXPORT void ft_destroy_inflight_batch(void* batch_handle) {
    delete static_cast<InflightBatchHandle*>(batch_handle);
}

FT_EXPORT void ft_get_stats(
    void* tree_ptr,
    int64_t* out_term_backups,
    int64_t* out_collisions,
    int64_t* out_cache_hits,
    int64_t* out_cache_misses,
    int64_t* out_cache_expanded,
    int64_t* out_dedup_avoided,
    int32_t* out_max_depth_cache,
    int32_t* out_root_visits
) {
    auto* t = static_cast<FastTreeArena*>(tree_ptr);
    *out_term_backups = t->stats.terminal_backups;
    *out_collisions = t->stats.collisions;
    *out_cache_hits = t->stats.nn_cache_hits;
    *out_cache_misses = t->stats.nn_cache_misses;
    *out_cache_expanded = t->stats.cache_expanded_nodes;
    *out_dedup_avoided = t->stats.dedup_avoided;
    *out_max_depth_cache = t->stats.max_depth_cache;
    *out_root_visits = t->nodes.empty() ? 0 : t->nodes[0].visit_count;
}

#if defined(__clang__)
#pragma float_control(precise, on, push)
#endif
FT_EXPORT int32_t ft_get_node_children_info(
    void* tree_ptr,
    int32_t node_idx,
    int32_t* out_child_indices,
    int32_t* out_actions,
    uint16_t* out_moves_packed,
    int32_t* out_visits,
    double* out_q_values,
    double* out_priors,
    uint8_t* out_is_solved,
    double* out_solved_values,
    int32_t* out_mate_plies
) {
    auto* t = static_cast<FastTreeArena*>(tree_ptr);
    if (node_idx < 0 || (size_t)node_idx >= t->nodes.size()) return 0;
    const Node& n = t->nodes[(size_t)node_idx];
    int32_t fc = n.first_child_idx;
    int32_t nc = n.num_children;
    if (fc < 0 || nc <= 0) return 0;

    for (int32_t i = 0; i < nc; ++i) {
        int32_t ci = fc + i;
        const Node& ch = t->nodes[(size_t)ci];
        out_child_indices[i] = ci;
        out_actions[i] = ch.action_from_parent;
        out_moves_packed[i] = ch.move_packed;
        out_visits[i] = ch.visit_count;
        out_q_values[i] = (ch.visit_count > 0) ? (ch.value_sum / (double)ch.visit_count) : 0.0;
        out_priors[i] = ch.prior;
        out_is_solved[i] = ch.is_solved;
        out_solved_values[i] = ch.solved_value;
        out_mate_plies[i] = ch.mate_plies;
    }
    return nc;
}
#if defined(__clang__)
#pragma float_control(pop)
#endif

FT_EXPORT int32_t ft_reroot_compact(
    void* tree_ptr,
    int32_t new_root_idx,
    int32_t* out_old_indices,
    int32_t max_out_cap
) {
    auto* t = static_cast<FastTreeArena*>(tree_ptr);
    if (new_root_idx < 0 || (size_t)new_root_idx >= t->nodes.size()) return 0;

    std::vector<Node> new_nodes;
    new_nodes.reserve(4096);
    std::vector<int32_t> old_map;
    old_map.reserve(4096);

    Node r = t->nodes[(size_t)new_root_idx];
    r.parent_idx = -1;
    r.soa_slot = -1;
    r.soa_offset = -1;
    r.action_from_parent = -1;
    r.move_packed = 0;
    r.ply = 0;
    r.is_instantiated = 1;
    r.virtual_loss = 0;
    r.is_terminal = 0;
    r.terminal_value = 0.0;
    r.path_dep_draw = 0;
    r.is_solved = 0;
    r.solved_value = 0.0;
    r.mate_plies = 0;

    new_nodes.push_back(r);
    old_map.push_back(new_root_idx);

    size_t head = 0;
    while (head < new_nodes.size()) {
        int32_t old_fc = new_nodes[head].first_child_idx;
        int32_t nc = new_nodes[head].num_children;
        int32_t parent_ply = new_nodes[head].ply;
        if (old_fc >= 0 && nc > 0) {
            int32_t new_fc = (int32_t)new_nodes.size();
            new_nodes[head].first_child_idx = new_fc;
            for (int32_t i = 0; i < nc; ++i) {
                int32_t old_ci = old_fc + i;
                Node ch = t->nodes[(size_t)old_ci];
                ch.parent_idx = (int32_t)head;
                ch.ply = parent_ply + 1;
                ch.virtual_loss = 0;
                new_nodes.push_back(ch);
                old_map.push_back(old_ci);
            }
        } else {
            new_nodes[head].first_child_idx = -1;
            new_nodes[head].num_children = 0;
            new_nodes[head].soa_offset = -1;
        }
        head++;
    }

    t->nodes.swap(new_nodes);
    t->pending_nodes.clear();
    t->rebuild_all_soa();

    int32_t n_surv = (int32_t)old_map.size();
    if (out_old_indices != nullptr && max_out_cap > 0) {
        int32_t to_copy = std::min(n_surv, max_out_cap);
        for (int32_t i = 0; i < to_copy; ++i) {
            out_old_indices[i] = old_map[(size_t)i];
        }
    }
    return n_surv;
}

FT_EXPORT int32_t ft_get_instantiated_preorder(
    void* tree_ptr,
    int32_t* out_node_indices,
    int32_t* out_parent_indices,
    uint16_t* out_moves_packed,
    uint8_t* out_old_path_dep,
    int32_t max_cap
) {
    auto* t = static_cast<FastTreeArena*>(tree_ptr);
    if (t->nodes.empty()) return 0;

    std::vector<int32_t> stack;
    stack.push_back(0);
    int32_t count = 0;
    while (!stack.empty() && count < max_cap) {
        int32_t idx = stack.back();
        stack.pop_back();
        const Node& n = t->nodes[(size_t)idx];
        out_node_indices[count] = idx;
        out_parent_indices[count] = n.parent_idx;
        out_moves_packed[count] = n.move_packed;
        out_old_path_dep[count] = n.path_dep_draw;
        count++;

        if (n.first_child_idx >= 0 && n.num_children > 0) {
            for (int32_t i = 0; i < n.num_children; ++i) {
                int32_t ci = n.first_child_idx + i;
                if (t->nodes[(size_t)ci].is_instantiated) {
                    stack.push_back(ci);
                }
            }
        }
    }
    return count;
}

FT_EXPORT void ft_invalidate_reused_node(
    void* tree_ptr,
    int32_t node_idx,
    int32_t is_terminal,
    double terminal_value,
    int32_t is_solved,
    double solved_value,
    int32_t path_dep_draw
) {
    auto* t = static_cast<FastTreeArena*>(tree_ptr);
    Node& n = t->nodes[(size_t)node_idx];
    n.self_visits = 0;
    n.self_value_sum = 0.0;
    n.first_child_idx = -1;
    n.num_children = 0;
    n.soa_offset = -1;
    n.is_expanded = 0;
    n.is_terminal = (uint8_t)is_terminal;
    n.terminal_value = terminal_value;
    n.is_solved = (uint8_t)is_solved;
    n.solved_value = solved_value;
    n.mate_plies = 0;
    n.path_dep_draw = (uint8_t)path_dep_draw;
    t->sync_node_to_soa(node_idx);
}

FT_EXPORT void ft_finalize_reused_subtree(void* tree_ptr, const int32_t* /*unused_preorder*/, int32_t /*unused_count*/, int32_t max_depth) {
    auto* t = static_cast<FastTreeArena*>(tree_ptr);
    if (t->nodes.empty()) return;

    std::vector<int32_t> order;
    order.reserve(2048);
    std::vector<int32_t> stack;
    stack.reserve(256);
    stack.push_back(0);

    while (!stack.empty()) {
        int32_t idx = stack.back();
        stack.pop_back();
        order.push_back(idx);

        if (idx != 0 && !t->nodes[(size_t)idx].is_terminal) {
            Node& n = t->nodes[(size_t)idx];
            n.is_solved = 0;
            n.solved_value = 0.0;
            n.mate_plies = 0;
            if (!n.is_expanded && n.self_visits > 0 && n.ply < max_depth) {
                n.self_visits = 0;
                n.self_value_sum = 0.0;
            }
        }

        if (!t->nodes[(size_t)idx].is_terminal &&
            t->nodes[(size_t)idx].is_expanded &&
            t->nodes[(size_t)idx].has_cache_key &&
            t->nodes[(size_t)idx].ply < max_depth &&
            t->cache != nullptr) {
            uint64_t khi = t->nodes[(size_t)idx].zobrist_hash;
            uint64_t klo = t->nodes[(size_t)idx].verify_hash;
            const CacheEntry* ce = t->cache->peek(khi, klo);
            if (ce != nullptr) {
                int32_t n_leg = (int32_t)ce->unsorted_acts.size();
                if (t->nodes[(size_t)idx].num_children < n_leg) {
                    t->expand_node_from_cache_entry(idx, *ce, n_leg);
                }
            }
        }

        int32_t fc = t->nodes[(size_t)idx].first_child_idx;
        int32_t nc = t->nodes[(size_t)idx].num_children;
        if (fc >= 0 && nc > 0) {
            for (int32_t c = nc - 1; c >= 0; --c) {
                int32_t ci = fc + c;
                if (t->nodes[(size_t)ci].is_instantiated) {
                    stack.push_back(ci);
                }
            }
        }
    }

    {
#if defined(__clang__)
#pragma float_control(precise, on)
#endif
        for (int32_t i = (int32_t)order.size() - 1; i >= 0; --i) {
            int32_t idx = order[(size_t)i];
            int32_t v_sum = t->nodes[(size_t)idx].self_visits;
            double val_sum = t->nodes[(size_t)idx].self_value_sum;
            int32_t fc = t->nodes[(size_t)idx].first_child_idx;
            int32_t nc = t->nodes[(size_t)idx].num_children;
            if (fc >= 0 && nc > 0) {
                for (int32_t c = 0; c < nc; ++c) {
                    const Node& ch = t->nodes[(size_t)(fc + c)];
                    v_sum += ch.visit_count;
                    val_sum += -ch.value_sum;
                }
            }
            t->nodes[(size_t)idx].visit_count = v_sum;
            t->nodes[(size_t)idx].value_sum = val_sum;
        }
    }

    // Rebuild compact SoA blocks & recompute exact solver status bottom-up
    t->rebuild_all_soa();
    if (t->use_exact_solver) {
        std::vector<int32_t> single_path(1);
        for (int32_t i = (int32_t)order.size() - 1; i >= 0; --i) {
            int32_t idx = order[(size_t)i];
            if (!t->nodes[(size_t)idx].is_terminal &&
                t->nodes[(size_t)idx].is_expanded &&
                t->nodes[(size_t)idx].num_children > 0) {
                single_path[0] = idx;
                t->update_solver_along_path(single_path);
            }
        }
    }
}

FT_EXPORT void ft_clear_all_vl(void* tree_ptr) {
    auto* t = static_cast<FastTreeArena*>(tree_ptr);
    t->pending_nodes.clear();
    t->collision_paths.clear();
    t->vl_applied_paths.clear();
    for (size_t i = 0; i < t->nodes.size(); ++i) {
        if (t->nodes[i].virtual_loss != 0) {
            t->nodes[i].virtual_loss = 0;
            if (t->nodes[i].soa_slot >= 0) {
                t->soa.virtual_loss[(size_t)t->nodes[i].soa_slot] = 0;
            }
        }
    }
}

// Fast native C++23 AVX2 bitboard -> 18-plane (gpu_unpack) float32 tensor encoder
FT_EXPORT void ft_encode_gpu_unpack_single(
    const uint64_t* piece_bbs_12,
    int32_t turn_is_black,
    int32_t castling_mask_4bit,
    int32_t halfmove_clock,
    float* out_tensor_1152
) {
    __m256 z = _mm256_setzero_ps();
    for (int off = 0; off < 18 * 64; off += 32) {
        _mm256_storeu_ps(out_tensor_1152 + off + 0,  z);
        _mm256_storeu_ps(out_tensor_1152 + off + 8,  z);
        _mm256_storeu_ps(out_tensor_1152 + off + 16, z);
        _mm256_storeu_ps(out_tensor_1152 + off + 24, z);
    }
    for (int p = 0; p < 12; ++p) {
        uint64_t b = piece_bbs_12[p];
        float* plane = out_tensor_1152 + p * 64;
        while (b) {
            int sq = bb::ctz64(b);
            plane[sq] = 1.0f;
            b &= (b - 1);
        }
    }
    if (turn_is_black) {
        bb::fill_plane_64_avx2(out_tensor_1152 + 12 * 64, 1.0f);
    }
    for (int b = 0; b < 4; ++b) {
        if ((castling_mask_4bit >> b) & 1) {
            bb::fill_plane_64_avx2(out_tensor_1152 + (13 + b) * 64, 1.0f);
        }
    }
    float hm = (halfmove_clock >= 100) ? 1.0f : ((float)halfmove_clock / 100.0f);
    if (hm > 0.0f) {
        bb::fill_plane_64_avx2(out_tensor_1152 + 17 * 64, hm);
    }
}

// ============================================================================
// Pure C++23 Zero-GIL Native Bitboard Search Pipeline + Zero-Copy Staging
// ============================================================================

static bb::CBoard make_cboard_from_args(
    const uint64_t* bbs_8,
    uint64_t castling_rights,
    int32_t ep_square,
    int32_t turn_is_white,
    int32_t halfmove_clock,
    int32_t rev_len,
    const uint64_t* rev_hi,
    const uint64_t* rev_lo
) {
    bb::CBoard b{};
    b.pawns = bbs_8[0];
    b.knights = bbs_8[1];
    b.bishops = bbs_8[2];
    b.rooks = bbs_8[3];
    b.queens = bbs_8[4];
    b.kings = bbs_8[5];
    b.occ_w = bbs_8[6];
    b.occ_b = bbs_8[7];
    b.castling_rights = castling_rights;
    b.ep_square = (int8_t)ep_square;
    b.turn_is_white = (uint8_t)(turn_is_white ? 1 : 0);
    b.halfmove_clock = (int16_t)halfmove_clock;
    int32_t rl = std::min(rev_len, 104);
    b.rev_len = (uint8_t)rl;
    for (int32_t i = 0; i < rl; ++i) {
        b.rev_hi[i] = rev_hi[i];
        b.rev_lo[i] = rev_lo[i];
    }
    b.castling_rights = b.clean_castling_rights();
    b.refresh_zobrist();
    return b;
}

FT_EXPORT void ft_init_action_table(const uint16_t* table_49152) {
    auto& tb = bb::tables();
    std::memcpy(tb.action_table, table_49152, 2 * 64 * 64 * 6 * sizeof(uint16_t));
    tb.action_table_ready = true;
}

FT_EXPORT void ft_clear_seen_keys(void* tree_ptr) {
    auto* t = static_cast<FastTreeArena*>(tree_ptr);
    t->seen_npu_keys.clear();
}

FT_EXPORT void ft_compute_cboard_zobrist(
    const uint64_t* bbs_8,
    uint64_t castling_rights,
    int32_t ep_square,
    int32_t turn_is_white,
    uint64_t* out_hi,
    uint64_t* out_lo
) {
    bb::CBoard b = make_cboard_from_args(bbs_8, castling_rights, ep_square, turn_is_white, 0, 0, nullptr, nullptr);
    *out_hi = b.zobrist_hi;
    *out_lo = b.zobrist_lo;
}

FT_EXPORT void ft_set_root_native(
    void* tree_ptr,
    const uint64_t* bbs_8,
    uint64_t castling_rights,
    int32_t ep_square,
    int32_t turn_is_white,
    int32_t halfmove_clock,
    int32_t rev_len,
    const uint64_t* rev_hi,
    const uint64_t* rev_lo,
    uint64_t* out_root_hi,
    uint64_t* out_root_lo
) {
    auto* t = static_cast<FastTreeArena*>(tree_ptr);
    t->reset_root();
    t->native_board_mode = 1;
    t->stats = BatchStats{};

    bb::CBoard rb = make_cboard_from_args(
        bbs_8, castling_rights, ep_square, turn_is_white, halfmove_clock, rev_len, rev_hi, rev_lo
    );
    t->boards.push_back(rb);
    t->nodes[0].board_idx = 0;
    t->nodes[0].zobrist_hash = rb.zobrist_hi;
    t->nodes[0].verify_hash = rb.zobrist_lo;
    t->nodes[0].has_cache_key = 1;
    *out_root_hi = rb.zobrist_hi;
    *out_root_lo = rb.zobrist_lo;
}

// Encodes root CBoard directly into OpenVINO `root_model.infer_request.get_input_tensor().data`
FT_EXPORT void ft_encode_root_native(void* tree_ptr, float* out_tensor_1152) {
    auto* t = static_cast<FastTreeArena*>(tree_ptr);
    if (out_tensor_1152 == nullptr || t->nodes.empty() || t->nodes[0].board_idx < 0) return;
    t->boards[(size_t)t->nodes[0].board_idx].encode_gpu_unpack(out_tensor_1152);
}

FT_EXPORT int32_t ft_try_reuse_root_native(
    void* tree_ptr,
    const uint64_t* bbs_8,
    uint64_t castling_rights,
    int32_t ep_square,
    int32_t turn_is_white,
    int32_t halfmove_clock,
    int32_t rev_len,
    const uint64_t* rev_hi,
    const uint64_t* rev_lo,
    int32_t n_delta_moves,
    const uint16_t* delta_moves_packed,
    int32_t stack_extended_flag,
    int32_t max_depth,
    int32_t* out_reused_visits,
    int32_t* out_reused_nodes,
    int32_t* out_invalidated_nodes,
    uint64_t* out_root_hi,
    uint64_t* out_root_lo
) {
    auto* t = static_cast<FastTreeArena*>(tree_ptr);
    if (!t->native_board_mode || t->nodes.empty() || t->boards.empty() || t->nodes[0].board_idx < 0) {
        return 0;
    }

    bb::CBoard target_root = make_cboard_from_args(
        bbs_8, castling_rights, ep_square, turn_is_white, halfmove_clock, rev_len, rev_hi, rev_lo
    );
    *out_root_hi = target_root.zobrist_hi;
    *out_root_lo = target_root.zobrist_lo;

    int32_t found_idx = -1;
    bool stack_matches = false;

    if (stack_extended_flag && n_delta_moves > 0 && n_delta_moves <= 4) {
        int32_t curr = 0;
        bool ok = true;
        for (int32_t d = 0; d < n_delta_moves; ++d) {
            if (!t->nodes[(size_t)curr].is_instantiated) {
                t->instantiate_node_native(curr);
            }
            int32_t fc = t->nodes[(size_t)curr].first_child_idx;
            int32_t nc = t->nodes[(size_t)curr].num_children;
            if (fc < 0 || nc <= 0) {
                ok = false;
                break;
            }
            uint16_t want_mp = delta_moves_packed[d];
            int32_t matched_c = -1;
            for (int32_t i = 0; i < nc; ++i) {
                if (t->nodes[(size_t)(fc + i)].move_packed == want_mp) {
                    matched_c = fc + i;
                    break;
                }
            }
            if (matched_c < 0) {
                ok = false;
                break;
            }
            if (!t->nodes[(size_t)matched_c].is_instantiated) {
                t->instantiate_node_native(matched_c);
            }
            curr = matched_c;
        }
        if (ok && t->nodes[(size_t)curr].board_idx >= 0) {
            const bb::CBoard& cb = t->boards[(size_t)t->nodes[(size_t)curr].board_idx];
            if (cb.zobrist_hi == target_root.zobrist_hi &&
                cb.zobrist_lo == target_root.zobrist_lo &&
                cb.halfmove_clock == target_root.halfmove_clock) {
                found_idx = curr;
                stack_matches = true;
            }
        }
    }

    if (found_idx < 0) {
        const bb::CBoard old_rb = t->boards[(size_t)t->nodes[0].board_idx];
        int32_t fc1 = t->nodes[0].first_child_idx;
        int32_t nc1 = t->nodes[0].num_children;
        for (int32_t i = 0; i < nc1 && found_idx < 0; ++i) {
            int32_t c1 = fc1 + i;
            bb::CBoard b1 = old_rb;
            b1.push(t->nodes[(size_t)c1].move_packed);
            if (b1.turn_is_white == target_root.turn_is_white &&
                b1.zobrist_hi == target_root.zobrist_hi &&
                b1.zobrist_lo == target_root.zobrist_lo &&
                b1.halfmove_clock == target_root.halfmove_clock) {
                if (!t->nodes[(size_t)c1].is_instantiated) t->instantiate_node_native(c1);
                found_idx = c1;
                break;
            }
            if (t->nodes[(size_t)c1].is_expanded) {
                int32_t fc2 = t->nodes[(size_t)c1].first_child_idx;
                int32_t nc2 = t->nodes[(size_t)c1].num_children;
                for (int32_t j = 0; j < nc2; ++j) {
                    int32_t c2 = fc2 + j;
                    bb::CBoard b2 = b1;
                    b2.push(t->nodes[(size_t)c2].move_packed);
                    if (b2.turn_is_white == target_root.turn_is_white &&
                        b2.zobrist_hi == target_root.zobrist_hi &&
                        b2.zobrist_lo == target_root.zobrist_lo &&
                        b2.halfmove_clock == target_root.halfmove_clock) {
                        if (!t->nodes[(size_t)c1].is_instantiated) t->instantiate_node_native(c1);
                        if (!t->nodes[(size_t)c2].is_instantiated) t->instantiate_node_native(c2);
                        found_idx = c2;
                        break;
                    }
                }
            }
        }
    }

    if (found_idx < 0) {
        return 0;
    }

    int32_t n_surv = ft_reroot_compact(tree_ptr, found_idx, nullptr, 0);
    if (n_surv <= 0) return 0;

    std::vector<bb::CBoard> new_boards;
    new_boards.reserve((size_t)std::min(n_surv, 4096));
    new_boards.push_back(target_root);
    t->nodes[0].board_idx = 0;
    t->nodes[0].zobrist_hash = target_root.zobrist_hi;
    t->nodes[0].verify_hash = target_root.zobrist_lo;
    t->nodes[0].has_cache_key = 1;

    for (int32_t ni = 1; ni < n_surv; ++ni) {
        Node& n = t->nodes[(size_t)ni];
        if (n.is_instantiated && n.board_idx >= 0 && (size_t)n.board_idx < t->boards.size()) {
            int32_t nb_idx = (int32_t)new_boards.size();
            new_boards.push_back(t->boards[(size_t)n.board_idx]);
            n.board_idx = nb_idx;
        } else {
            n.board_idx = -1;
            n.is_instantiated = 0;
        }
    }
    t->boards.swap(new_boards);

    std::vector<int32_t> stk;
    stk.reserve(256);
    stk.push_back(0);
    int32_t pre_count = 0;
    int32_t inv_count = 0;

    while (!stk.empty()) {
        int32_t ni = stk.back();
        stk.pop_back();
        pre_count++;

        if (ni != 0) {
            Node& n = t->nodes[(size_t)ni];
            bool old_pd = (n.path_dep_draw != 0);
            bool new_pd = false;
            if (!stack_matches || n.board_idx < 0) {
                int32_t pi = n.parent_idx;
                bb::CBoard cb = t->boards[(size_t)t->nodes[(size_t)pi].board_idx];
                cb.push(n.move_packed);
                if (n.board_idx >= 0) {
                    t->boards[(size_t)n.board_idx] = cb;
                } else {
                    n.board_idx = (int32_t)t->boards.size();
                    t->boards.push_back(cb);
                }
                n.zobrist_hash = cb.zobrist_hi;
                n.verify_hash = cb.zobrist_lo;
                n.has_cache_key = 1;
                bool is_50m = (cb.halfmove_clock >= 100);
                bool is_ins = (!is_50m) && cb.is_insufficient_material();
                bool is_rep = (!is_50m) && (!is_ins) && cb.is_repetition_2();
                new_pd = is_50m || is_rep;
            }

            if (old_pd || new_pd) {
                inv_count++;
                int32_t iterm, isolv, pdep;
                double tval, sval;
                t->boards[(size_t)n.board_idx].evaluate_terminal(iterm, tval, isolv, sval, pdep);
                ft_invalidate_reused_node(tree_ptr, ni, iterm, tval, isolv, sval, pdep);
            }
        }

        int32_t fc = t->nodes[(size_t)ni].first_child_idx;
        int32_t nc = t->nodes[(size_t)ni].num_children;
        if (fc >= 0 && nc > 0) {
            for (int32_t c = nc - 1; c >= 0; --c) {
                int32_t ci = fc + c;
                if (t->nodes[(size_t)ci].is_instantiated) {
                    stk.push_back(ci);
                }
            }
        }
    }

    ft_finalize_reused_subtree(tree_ptr, nullptr, 0, max_depth);

    *out_reused_visits = t->nodes[0].visit_count;
    *out_reused_nodes = pre_count;
    *out_invalidated_nodes = inv_count;
    return 1;
}

FT_EXPORT int32_t ft_record_seen_root_native(void* tree_ptr) {
    auto* t = static_cast<FastTreeArena*>(tree_ptr);
    auto kp = std::make_pair(t->nodes[0].zobrist_hash, t->nodes[0].verify_hash);
    if (t->seen_npu_keys.find(kp) != t->seen_npu_keys.end()) {
        return 0;
    }
    t->seen_npu_keys.insert(kp);
    return 1;
}

FT_EXPORT double ft_expand_root_native(
    void* tree_ptr,
    double root_val,
    const float* full_logits_4672,
    int32_t store_in_cache
) {
    auto* t = static_cast<FastTreeArena*>(tree_ptr);
    const bb::CBoard& rb = t->boards[(size_t)t->nodes[0].board_idx];
    uint16_t acts[256];
    uint16_t moves[256];
    int32_t n_leg = rb.generate_legal_moves_and_actions(acts, moves);
    return ft_expand_node_with_logits(
        tree_ptr,
        0,
        rb.zobrist_hi,
        rb.zobrist_lo,
        root_val,
        n_leg,
        acts,
        moves,
        full_logits_4672,
        nullptr,
        n_leg,
        store_in_cache
    );
}

// Writes `(B, 18, 8, 8)` directly into OpenVINO `InferRequest.get_input_tensor().data` pointer!
FT_EXPORT void* ft_select_and_pack_native(
    void* tree_ptr,
    int32_t batch_size,
    int32_t max_depth,
    int32_t use_nn_cache,
    int32_t use_pending,
    float* out_batch_tensors,
    int32_t* out_n_eval,
    int32_t* out_max_ply
) {
    auto* t = static_cast<FastTreeArena*>(tree_ptr);
    int32_t dummy_n = -1, dummy_p = -1;
    uint16_t dummy_m = 0;
    ft_select_begin(tree_ptr, batch_size, max_depth, use_nn_cache, use_pending, &dummy_n, &dummy_p, &dummy_m);

    int32_t leaves_tmp[128];
    void* handle = ft_pop_inflight_batch(tree_ptr, leaves_tmp, out_n_eval, out_max_ply);
    int32_t n_eval = *out_n_eval;

    if (n_eval > 0 && out_batch_tensors != nullptr) {
        for (int32_t i = 0; i < n_eval; ++i) {
            int32_t li = leaves_tmp[i];
            int32_t bidx = t->nodes[(size_t)li].board_idx;
            t->boards[(size_t)bidx].encode_gpu_unpack(out_batch_tensors + (size_t)i * 1152ULL);
        }
        for (int32_t i = n_eval; i < batch_size; ++i) {
            std::memcpy(out_batch_tensors + (size_t)i * 1152ULL, out_batch_tensors, 1152ULL * sizeof(float));
        }
    }
    return handle;
}

FT_EXPORT void ft_backup_batch_native(
    void* tree_ptr,
    void* batch_handle,
    const float* p_logits_batch,
    const float* val_batch,
    int32_t val_cols,
    int32_t max_depth,
    int32_t store_in_cache,
    int32_t* out_novel_npu,
    int32_t* out_wasted_dup
) {
    auto* t = static_cast<FastTreeArena*>(tree_ptr);
    auto* h = static_cast<InflightBatchHandle*>(batch_handle);

    ft_revert_batch_vl_and_clear_pending(tree_ptr, batch_handle);

    int32_t novel = 0;
    int32_t dup = 0;
    int32_t n_eval = (int32_t)h->leaves.size();

    uint16_t acts[256];
    uint16_t moves[256];

    for (int32_t i = 0; i < n_eval; ++i) {
        int32_t leaf_idx = h->leaves[(size_t)i];
        int32_t bidx = t->nodes[(size_t)leaf_idx].board_idx;
        const bb::CBoard& cb = t->boards[(size_t)bidx];

        auto kp = std::make_pair(cb.zobrist_hi, cb.zobrist_lo);
        if (t->seen_npu_keys.find(kp) != t->seen_npu_keys.end()) {
            dup++;
        } else {
            t->seen_npu_keys.insert(kp);
            novel++;
        }

        const float* v_row = val_batch + (size_t)i * (size_t)val_cols;
        double leaf_val = (val_cols == 3) ? (double)(v_row[0] - v_row[2]) : (double)v_row[0];
        const float* logits_row = p_logits_batch + (size_t)i * 4672ULL;

        int32_t ply = t->nodes[(size_t)leaf_idx].ply;
        int32_t do_expand = (ply < max_depth && !t->nodes[(size_t)leaf_idx].is_expanded) ? 1 : 0;

        int32_t n_leg = 0;
        if (do_expand || store_in_cache || t->use_exact_solver) {
            n_leg = cb.generate_legal_moves_and_actions(acts, moves);
        }
        int32_t bk = do_expand ? n_leg : 0;

        ft_expand_node_with_logits(
            tree_ptr,
            leaf_idx,
            cb.zobrist_hi,
            cb.zobrist_lo,
            leaf_val,
            n_leg,
            acts,
            moves,
            logits_row,
            nullptr,
            bk,
            store_in_cache
        );

        t->backup_value(leaf_idx, h->paths[(size_t)i], leaf_val);
    }

    *out_novel_npu = novel;
    *out_wasted_dup = dup;
    ft_destroy_inflight_batch(batch_handle);
}

FT_EXPORT int32_t ft_test_generate_moves_native(
    const uint64_t* bbs_8,
    uint64_t castling_rights,
    int32_t ep_square,
    int32_t turn_is_white,
    uint16_t* out_acts,
    uint16_t* out_moves,
    double* out_bonuses
) {
    bb::CBoard cb = make_cboard_from_args(bbs_8, castling_rights, ep_square, turn_is_white, 0, 0, nullptr, nullptr);
    return cb.generate_scored_legal_moves(out_acts, out_moves, out_bonuses);
}
