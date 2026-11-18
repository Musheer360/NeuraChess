// Copyright (c) 2026 Musheer Alam. Licensed under GPL-3.0.
/*
 * Shared Packed V1 (107B) & Unified Sidecar (68B) Binary Format, Action Table,
 * Holdout 98B Key Filter, and SHA-256 Prefix Hasher for Lc0 T80 & Lichess SF Converters.
 */

#pragma once

#include "../engine/fast_tree/bitboard_chess.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace nc_pack {

constexpr double CP_TANH_SCALE = 0.0028782313662425573; // ln(10) / 800

#pragma pack(push, 1)
struct PackedV1 {
    uint64_t bbs[12];
    uint8_t turn;
    uint8_t castling;
    uint8_t ep_square;
    uint8_t halfmove;
    uint16_t action_index;
    int16_t eval_cp;
    uint8_t wdl[3];
};

// Bit flags in `UnifiedSidecar68::flags`:
//   bit 0 (1): side-to-move is black
//   bit 1 (2): en passant square present
//   bit 2 (4): best_q used (Lc0) / source is Lc0
//   bit 3 (8): valid MLH target present (`plies_left > 0`)
struct UnifiedSidecar68 {
    uint8_t num_topk;
    uint8_t flags;
    uint16_t visits_clamped;
    uint16_t top8_actions[8];
    float top8_probs[8];
    float backed_q;
    float backed_d;
    float result_q;
    float plies_left;
};
#pragma pack(pop)

static_assert(sizeof(PackedV1) == 107, "PackedV1 must be 107 bytes");
static_assert(sizeof(UnifiedSidecar68) == 68, "UnifiedSidecar68 must be 68 bytes");

inline void init_nc_action_table() noexcept {
    auto& tb = bb::tables();
    if (tb.action_table_ready) return;

    static constexpr int dr[8] = {+1, +1,  0, -1, -1, -1,  0, +1};
    static constexpr int dc[8] = { 0, +1, +1, +1,  0, -1, -1, -1};
    static constexpr int k_dr[8] = {+2, +1, -1, -2, -2, -1, +1, +2};
    static constexpr int k_dc[8] = {+1, +2, +2, +1, -1, -2, -2, -1};
    static constexpr int p_dc[3] = {-1, 0, +1};
    static constexpr int p_id[3] = {bb::PT_KNIGHT, bb::PT_BISHOP, bb::PT_ROOK};

    uint16_t raw_tbl[64][64][6];
    std::memset(raw_tbl, 0, sizeof(raw_tbl));

    for (int f_sq = 0; f_sq < 64; ++f_sq) {
        int fr = f_sq >> 3, fc = f_sq & 7;
        for (int d = 0; d < 8; ++d) {
            for (int dist = 1; dist <= 7; ++dist) {
                int tr = fr + dr[d] * dist;
                int tc = fc + dc[d] * dist;
                if (tr >= 0 && tr < 8 && tc >= 0 && tc < 8) {
                    int t_sq = tr * 8 + tc;
                    uint16_t act = (uint16_t)(f_sq * 73 + d * 7 + (dist - 1));
                    raw_tbl[f_sq][t_sq][0] = act;
                    if (fr == 6 && tr == 7 && dist == 1 && std::abs(dc[d]) <= 1) {
                        raw_tbl[f_sq][t_sq][bb::PT_QUEEN] = act;
                    }
                }
            }
        }
        for (int k = 0; k < 8; ++k) {
            int tr = fr + k_dr[k], tc = fc + k_dc[k];
            if (tr >= 0 && tr < 8 && tc >= 0 && tc < 8) {
                int t_sq = tr * 8 + tc;
                raw_tbl[f_sq][t_sq][0] = (uint16_t)(f_sq * 73 + 56 + k);
            }
        }
        if (fr == 6) {
            int tr = 7;
            for (int di = 0; di < 3; ++di) {
                int tc = fc + p_dc[di];
                if (tc >= 0 && tc < 8) {
                    int t_sq = tr * 8 + tc;
                    for (int pi = 0; pi < 3; ++pi) {
                        raw_tbl[f_sq][t_sq][p_id[pi]] = (uint16_t)(f_sq * 73 + 64 + 3 * di + pi);
                    }
                }
            }
        }
    }

    for (int is_blk = 0; is_blk < 2; ++is_blk) {
        for (int f_sq = 0; f_sq < 64; ++f_sq) {
            int cf = is_blk ? ((7 - (f_sq >> 3)) * 8 + (f_sq & 7)) : f_sq;
            for (int t_sq = 0; t_sq < 64; ++t_sq) {
                int ct = is_blk ? ((7 - (t_sq >> 3)) * 8 + (t_sq & 7)) : t_sq;
                for (int pr = 0; pr < 6; ++pr) {
                    tb.action_table[is_blk][f_sq][t_sq][pr] = raw_tbl[cf][ct][pr];
                }
            }
        }
    }
    tb.action_table_ready = true;
}

inline uint64_t hash98_from_v1(const PackedV1& p) noexcept {
    static constexpr uint64_t kMult[12] = {
        0x9E3779B97F4A7C15ULL, 0xBF58476D1CE4E5B9ULL, 0x94D049BB133111EBULL,
        0xD6E8FEB86659FD93ULL, 0xA0761D6478BD642FULL, 0xE7037ED1A0B428DBULL,
        0x8EBC6AF09C88C6E3ULL, 0x589965CC75374CC3ULL, 0x1D8E4E27C47D124FULL,
        0x2545F4914F6CDD1DULL, 0x6C62272E07BB0142ULL, 0xB2B5C735F98D9F1BULL,
    };
    uint64_t acc = 0;
    for (int i = 0; i < 12; ++i) {
        acc += p.bbs[i] * kMult[i];
    }
    acc ^= ((uint64_t)p.turn << 1) | ((uint64_t)p.castling << 9);
    acc ^= acc >> 30;
    acc *= 0xBF58476D1CE4E5B9ULL;
    acc ^= acc >> 27;
    return acc;
}

inline uint64_t sha256_prefix64(const uint8_t* data, size_t len) noexcept {
    static constexpr uint32_t K[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
    };
    auto rotr = [](uint32_t x, uint32_t n) -> uint32_t { return (x >> n) | (x << (32 - n)); };
    uint32_t H[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };

    const uint64_t total_bits = (uint64_t)len * 8ULL;
    size_t padded_len = ((len + 9ULL + 63ULL) / 64ULL) * 64ULL;
    std::array<uint8_t, 128> stack_buf{};
    std::vector<uint8_t> heap_buf;
    uint8_t* msg = stack_buf.data();
    if (padded_len > stack_buf.size()) {
        heap_buf.assign(padded_len, 0);
        msg = heap_buf.data();
    } else {
        std::memset(msg, 0, padded_len);
    }
    if (len > 0) std::memcpy(msg, data, len);
    msg[len] = 0x80;
    for (int i = 0; i < 8; ++i) {
        msg[padded_len - 1 - i] = (uint8_t)((total_bits >> (8 * i)) & 0xFFU);
    }

    for (size_t off = 0; off < padded_len; off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            const uint8_t* b = msg + off + i * 4;
            w[i] = ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) | ((uint32_t)b[2] << 8) | (uint32_t)b[3];
        }
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = H[0], b = H[1], c = H[2], d = H[3];
        uint32_t e = H[4], f = H[5], g = H[6], h = H[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ ((~e) & g);
            uint32_t t1 = h + S1 + ch + K[i] + w[i];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + maj;
            h = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        H[0] += a; H[1] += b; H[2] += c; H[3] += d;
        H[4] += e; H[5] += f; H[6] += g; H[7] += h;
    }
    return ((uint64_t)H[0] << 32) | (uint64_t)H[1];
}

struct ExcludeSet {
    std::unordered_set<uint64_t> sha_prefixes;
    std::unordered_set<uint64_t> hash98_set;
    std::vector<std::array<uint8_t, 98>> canonical98_exact;

    void load_hash_hex_file(const std::string& path) {
        std::ifstream fin(path);
        if (!fin) return;
        std::string content((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
        size_t i = 0;
        while (i < content.size()) {
            while (i < content.size() && !(std::isxdigit((unsigned char)content[i]))) i++;
            if (i + 16 <= content.size()) {
                bool ok = true;
                for (size_t j = 0; j < 16; ++j) {
                    if (!std::isxdigit((unsigned char)content[i + j])) { ok = false; break; }
                }
                if (ok && (i + 16 == content.size() || !std::isxdigit((unsigned char)content[i + 16]))) {
                    uint64_t v = std::stoull(content.substr(i, 16), nullptr, 16);
                    sha_prefixes.insert(v);
                    i += 16;
                    continue;
                }
            }
            while (i < content.size() && std::isxdigit((unsigned char)content[i])) i++;
        }
    }

    void load_bin_98(const std::string& path, size_t stride = sizeof(PackedV1)) {
        std::ifstream fin(path, std::ios::binary);
        if (!fin) return;
        std::vector<uint8_t> rec(stride);
        while (fin.read(reinterpret_cast<char*>(rec.data()), (std::streamsize)stride)) {
            PackedV1 p{};
            std::memcpy(&p, rec.data(), sizeof(PackedV1));
            hash98_set.insert(hash98_from_v1(p));
            std::array<uint8_t, 98> k{};
            std::memcpy(k.data(), rec.data(), 98);
            canonical98_exact.push_back(k);
        }
    }

    inline bool is_forbidden_sha(const char* fen4, size_t len) const noexcept {
        if (sha_prefixes.empty()) return false;
        uint64_t h = sha256_prefix64(reinterpret_cast<const uint8_t*>(fen4), len);
        return sha_prefixes.find(h) != sha_prefixes.end();
    }

    inline bool is_forbidden_v1(const PackedV1& p) const noexcept {
        if (hash98_set.empty()) return false;
        uint64_t h = hash98_from_v1(p);
        if (hash98_set.find(h) == hash98_set.end()) return false;
        std::array<uint8_t, 98> k{};
        std::memcpy(k.data(), &p, 98);
        for (const auto& ex : canonical98_exact) {
            if (ex == k) return true;
        }
        return false;
    }
};

} // namespace nc_pack
