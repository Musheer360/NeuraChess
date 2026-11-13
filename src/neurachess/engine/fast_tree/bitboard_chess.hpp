// Copyright (c) 2026 Musheer Alam. Licensed under GPL-3.0.
/*
 * Native C++23 Bitboard Chess Engine with AVX2 + BMI2 PEXT Intrinsics
 * (src/neurachess/engine/fast_tree/bitboard_chess.hpp)
 * ===================================================================
 * Implements bit-for-bit `python-chess` legal move generation (exact move set
 * and exact `scan_reversed` MSB-first move order), BMI2 `_pext_u64` O(1) slider
 * attack tables, `_tzcnt_u64` / `_lzcnt_u64` / `_mm_popcnt_u64` hardware intrinsics,
 * C++23 `[[assume(...)]]` hints, incremental O(1) Zobrist hashing, O(1) 2-fold
 * repetition & insufficient material detection, and AVX2 18-plane `gpu_unpack`
 * zero-copy float32 tensor encoding.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <immintrin.h>

namespace bb {

constexpr int WHITE = 1;
constexpr int BLACK = 0;

constexpr int PT_NONE   = 0;
constexpr int PT_PAWN   = 1;
constexpr int PT_KNIGHT = 2;
constexpr int PT_BISHOP = 3;
constexpr int PT_ROOK   = 4;
constexpr int PT_QUEEN  = 5;
constexpr int PT_KING   = 6;

constexpr uint64_t BB_ALL           = 0xFFFFFFFFFFFFFFFFULL;
constexpr uint64_t BB_RANK_1        = 0x00000000000000FFULL;
constexpr uint64_t BB_RANK_3        = 0x0000000000FF0000ULL;
constexpr uint64_t BB_RANK_4        = 0x00000000FF000000ULL;
constexpr uint64_t BB_RANK_5        = 0x000000FF00000000ULL;
constexpr uint64_t BB_RANK_6        = 0x0000FF0000000000ULL;
constexpr uint64_t BB_RANK_8        = 0xFF00000000000000ULL;
constexpr uint64_t BB_FILE_A        = 0x0101010101010101ULL;
constexpr uint64_t BB_FILE_H        = 0x8080808080808080ULL;
constexpr uint64_t BB_DARK_SQUARES  = 0xAA55AA55AA55AA55ULL;
constexpr uint64_t BB_LIGHT_SQUARES = 0x55AA55AA55AA55AAULL;

static inline int ctz64(uint64_t bb) noexcept {
    [[assume(bb != 0ULL)]];
    return (int)_tzcnt_u64(bb);
}

static inline int msb64(uint64_t bb) noexcept {
    [[assume(bb != 0ULL)]];
    return 63 - (int)_lzcnt_u64(bb);
}

static inline int popcnt64(uint64_t bb) noexcept {
    return (int)_mm_popcnt_u64(bb);
}

static inline uint16_t pack_mv(int from_sq, int to_sq, int promo = 0) noexcept {
    return (uint16_t)(from_sq | (to_sq << 6) | (promo << 12));
}

// Directions: 0=N(+8), 1=E(+1), 2=NE(+9), 3=NW(+7) [positive: ctz]
//             4=S(-8), 5=W(-1), 6=SE(-7), 7=SW(-9) [negative: msb]
struct alignas(64) Tables {
    uint64_t knight_attacks[64];
    uint64_t king_attacks[64];
    uint64_t pawn_attacks[2][64]; // [0=BLACK, 1=WHITE]
    uint64_t dir_ray[8][64];
    uint64_t between_bb[64][64];
    uint64_t line_ray_bb[64][64];

    // BMI2 PEXT Slider Attack Tables (102,400 rook + 5,248 bishop entries = 841 KB)
    uint64_t rook_mask[64];
    uint64_t bishop_mask[64];
    uint32_t rook_offset[64];
    uint32_t bishop_offset[64];
    uint64_t rook_pext[102400];
    uint64_t bishop_pext[5248];

    uint16_t action_table[2][64][64][6]; // [is_black][from_sq][to_sq][promo 0..5]
    bool action_table_ready = false;

    uint64_t rook_attacks_ray(int sq, uint64_t occ) const noexcept {
        uint64_t att = 0;
        for (int d = 0; d < 2; ++d) {
            uint64_t r = dir_ray[d][sq];
            uint64_t b = r & occ;
            att |= b ? (r ^ dir_ray[d][ctz64(b)]) : r;
        }
        for (int d = 4; d < 6; ++d) {
            uint64_t r = dir_ray[d][sq];
            uint64_t b = r & occ;
            att |= b ? (r ^ dir_ray[d][msb64(b)]) : r;
        }
        return att;
    }

    uint64_t bishop_attacks_ray(int sq, uint64_t occ) const noexcept {
        uint64_t att = 0;
        for (int d = 2; d < 4; ++d) {
            uint64_t r = dir_ray[d][sq];
            uint64_t b = r & occ;
            att |= b ? (r ^ dir_ray[d][ctz64(b)]) : r;
        }
        for (int d = 6; d < 8; ++d) {
            uint64_t r = dir_ray[d][sq];
            uint64_t b = r & occ;
            att |= b ? (r ^ dir_ray[d][msb64(b)]) : r;
        }
        return att;
    }

    Tables() {
        std::memset(this, 0, sizeof(Tables));
        const int kn_dr[8] = {-2, -2, -1, -1, 1, 1, 2, 2};
        const int kn_dc[8] = {-1, 1, -2, 2, -2, 2, -1, 1};
        for (int sq = 0; sq < 64; ++sq) {
            int r = sq >> 3, c = sq & 7;
            uint64_t kn = 0, kg = 0;
            for (int i = 0; i < 8; ++i) {
                int nr = r + kn_dr[i], nc = c + kn_dc[i];
                if (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) kn |= (1ULL << (nr * 8 + nc));
            }
            for (int dr = -1; dr <= 1; ++dr) {
                for (int dc = -1; dc <= 1; ++dc) {
                    if (dr == 0 && dc == 0) continue;
                    int nr = r + dr, nc = c + dc;
                    if (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) kg |= (1ULL << (nr * 8 + nc));
                }
            }
            knight_attacks[sq] = kn;
            king_attacks[sq] = kg;

            uint64_t pw = 0, pb = 0;
            if (r + 1 < 8) {
                if (c - 1 >= 0) pw |= (1ULL << ((r + 1) * 8 + (c - 1)));
                if (c + 1 < 8)  pw |= (1ULL << ((r + 1) * 8 + (c + 1)));
            }
            if (r - 1 >= 0) {
                if (c - 1 >= 0) pb |= (1ULL << ((r - 1) * 8 + (c - 1)));
                if (c + 1 < 8)  pb |= (1ULL << ((r - 1) * 8 + (c + 1)));
            }
            pawn_attacks[WHITE][sq] = pw;
            pawn_attacks[BLACK][sq] = pb;

            const int drs[8] = {1, 0, 1, 1, -1, 0, -1, -1};
            const int dcs[8] = {0, 1, 1, -1, 0, -1, 1, -1};
            for (int d = 0; d < 8; ++d) {
                uint64_t ray = 0;
                int nr = r + drs[d], nc = c + dcs[d];
                while (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {
                    ray |= (1ULL << (nr * 8 + nc));
                    nr += drs[d];
                    nc += dcs[d];
                }
                dir_ray[d][sq] = ray;
            }
        }

        // Between and full-line ray bitboards (matching python-chess `between(a, b)` and `ray(a, b)`)
        const int drs[8] = {1, 0, 1, 1, -1, 0, -1, -1};
        const int dcs[8] = {0, 1, 1, -1, 0, -1, 1, -1};
        const int opp_d[8] = {4, 5, 7, 6, 0, 1, 3, 2};
        for (int a = 0; a < 64; ++a) {
            for (int d = 0; d < 8; ++d) {
                int r = (a >> 3) + drs[d], c = (a & 7) + dcs[d];
                uint64_t btw = 0;
                uint64_t full_line = (1ULL << a) | dir_ray[d][a] | dir_ray[opp_d[d]][a];
                while (r >= 0 && r < 8 && c >= 0 && c < 8) {
                    int b = r * 8 + c;
                    between_bb[a][b] = btw;
                    line_ray_bb[a][b] = full_line;
                    btw |= (1ULL << b);
                    r += drs[d];
                    c += dcs[d];
                }
            }
        }

        // Initialize BMI2 PEXT masks & attack lookup tables
        uint32_t r_off = 0;
        uint32_t b_off = 0;
        constexpr uint64_t outer_edges = BB_RANK_1 | BB_RANK_8 | BB_FILE_A | BB_FILE_H;
        for (int sq = 0; sq < 64; ++sq) {
            uint64_t rmask = ((dir_ray[0][sq] & ~BB_RANK_8) |
                              (dir_ray[4][sq] & ~BB_RANK_1) |
                              (dir_ray[1][sq] & ~BB_FILE_H) |
                              (dir_ray[5][sq] & ~BB_FILE_A));
            uint64_t bmask = (dir_ray[2][sq] | dir_ray[3][sq] | dir_ray[6][sq] | dir_ray[7][sq]) & ~outer_edges;

            rook_mask[sq] = rmask;
            bishop_mask[sq] = bmask;
            rook_offset[sq] = r_off;
            bishop_offset[sq] = b_off;

            uint32_t r_cnt = 1U << popcnt64(rmask);
            for (uint32_t idx = 0; idx < r_cnt; ++idx) {
                uint64_t occ = _pdep_u64((uint64_t)idx, rmask);
                rook_pext[r_off + idx] = rook_attacks_ray(sq, occ);
            }
            r_off += r_cnt;

            uint32_t b_cnt = 1U << popcnt64(bmask);
            for (uint32_t idx = 0; idx < b_cnt; ++idx) {
                uint64_t occ = _pdep_u64((uint64_t)idx, bmask);
                bishop_pext[b_off + idx] = bishop_attacks_ray(sq, occ);
            }
            b_off += b_cnt;
        }
    }
};

inline Tables& tables() noexcept {
    static Tables t;
    return t;
}

static inline uint64_t rook_attacks(int sq, uint64_t occ) noexcept {
    [[assume(sq >= 0 && sq < 64)]];
    const auto& tb = tables();
    return tb.rook_pext[tb.rook_offset[sq] + _pext_u64(occ, tb.rook_mask[sq])];
}

static inline uint64_t bishop_attacks(int sq, uint64_t occ) noexcept {
    [[assume(sq >= 0 && sq < 64)]];
    const auto& tb = tables();
    return tb.bishop_pext[tb.bishop_offset[sq] + _pext_u64(occ, tb.bishop_mask[sq])];
}

static inline uint64_t splitmix64(uint64_t x) noexcept {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

static inline void fill_plane_64_avx2(float* dst, float val) noexcept {
    __m256 v = _mm256_set1_ps(val);
    _mm256_storeu_ps(dst + 0,  v);
    _mm256_storeu_ps(dst + 8,  v);
    _mm256_storeu_ps(dst + 16, v);
    _mm256_storeu_ps(dst + 24, v);
    _mm256_storeu_ps(dst + 32, v);
    _mm256_storeu_ps(dst + 40, v);
    _mm256_storeu_ps(dst + 48, v);
    _mm256_storeu_ps(dst + 56, v);
}

struct CBoard {
    uint64_t pawns;
    uint64_t knights;
    uint64_t bishops;
    uint64_t rooks;
    uint64_t queens;
    uint64_t kings;
    uint64_t occ_w;
    uint64_t occ_b;
    uint64_t castling_rights; // bitmask on a1(0), h1(7), a8(56), h8(63)
    uint64_t zobrist_hi;
    uint64_t zobrist_lo;
    int16_t halfmove_clock;
    int8_t ep_square;         // -1 if None, else 0..63
    uint8_t turn_is_white;    // 1=White, 0=Black
    uint8_t rev_len;          // reversible history count [0..104]
    uint64_t rev_hi[104];
    uint64_t rev_lo[104];

    inline uint64_t occupied() const noexcept { return occ_w | occ_b; }
    inline uint64_t occ_co(int color) const noexcept { return color ? occ_w : occ_b; }

    inline int piece_type_at(int sq) const noexcept {
        [[assume(sq >= 0 && sq < 64)]];
        uint64_t m = 1ULL << sq;
        if (!((occ_w | occ_b) & m)) return PT_NONE;
        if (pawns & m)   return PT_PAWN;
        if (knights & m) return PT_KNIGHT;
        if (bishops & m) return PT_BISHOP;
        if (rooks & m)   return PT_ROOK;
        if (queens & m)  return PT_QUEEN;
        if (kings & m)   return PT_KING;
        return PT_NONE;
    }

    inline uint64_t clean_castling_rights() const noexcept {
        uint64_t cr = castling_rights;
        uint64_t out = 0;
        if ((kings & occ_w) == (1ULL << 4)) {
            out |= cr & rooks & occ_w & ((1ULL << 0) | (1ULL << 7));
        }
        if ((kings & occ_b) == (1ULL << 60)) {
            out |= cr & rooks & occ_b & ((1ULL << 56) | (1ULL << 63));
        }
        return out;
    }

    inline uint64_t attackers_mask(int attacker_color, int sq, uint64_t occ) const noexcept {
        [[assume(sq >= 0 && sq < 64)]];
        const auto& tb = tables();
        uint64_t att_occ = attacker_color ? occ_w : occ_b;
        uint64_t rq = rooks | queens;
        uint64_t bq = bishops | queens;
        uint64_t a = (tb.pawn_attacks[attacker_color ^ 1][sq] & pawns)
                   | (tb.knight_attacks[sq] & knights)
                   | (tb.king_attacks[sq] & kings)
                   | (rook_attacks(sq, occ) & rq)
                   | (bishop_attacks(sq, occ) & bq);
        return a & att_occ;
    }

    inline bool is_attacked_by(int attacker_color, int sq, uint64_t occ) const noexcept {
        return attackers_mask(attacker_color, sq, occ) != 0;
    }

    inline bool attacked_for_king(uint64_t path, uint64_t occ) const noexcept {
        int opp = turn_is_white ^ 1;
        while (path) {
            int sq = ctz64(path);
            if (is_attacked_by(opp, sq, occ)) return true;
            path &= (path - 1);
        }
        return false;
    }

    inline uint64_t slider_blockers(int king_sq) const noexcept {
        [[assume(king_sq >= 0 && king_sq < 64)]];
        const auto& tb = tables();
        uint64_t rq = rooks | queens;
        uint64_t bq = bishops | queens;
        uint64_t opp_occ = turn_is_white ? occ_b : occ_w;
        uint64_t us_occ = turn_is_white ? occ_w : occ_b;
        uint64_t occ = us_occ | opp_occ;

        uint64_t snipers = (((tb.dir_ray[0][king_sq] | tb.dir_ray[4][king_sq] |
                              tb.dir_ray[1][king_sq] | tb.dir_ray[5][king_sq]) & rq) |
                            ((tb.dir_ray[2][king_sq] | tb.dir_ray[3][king_sq] |
                              tb.dir_ray[6][king_sq] | tb.dir_ray[7][king_sq]) & bq)) & opp_occ;

        uint64_t blockers = 0;
        while (snipers) {
            int sniper = msb64(snipers);
            snipers ^= (1ULL << sniper);
            uint64_t b = tb.between_bb[king_sq][sniper] & occ;
            if (b && (b & (b - 1)) == 0) {
                blockers |= b;
            }
        }
        return blockers & us_occ;
    }

    inline bool is_ep_move_safe(int king_sq, int from_sq, int to_sq) const noexcept {
        int cap_sq = turn_is_white ? (to_sq - 8) : (to_sq + 8);
        uint64_t from_m = 1ULL << from_sq;
        uint64_t to_m = 1ULL << to_sq;
        uint64_t cap_m = 1ULL << cap_sq;
        uint64_t occ_ep = ((occ_w | occ_b) ^ from_m ^ cap_m) | to_m;

        int opp = turn_is_white ^ 1;
        uint64_t opp_occ = (opp ? occ_w : occ_b) & ~cap_m;
        const auto& tb = tables();
        uint64_t rq = rooks | queens;
        uint64_t bq = bishops | queens;
        uint64_t att = (tb.pawn_attacks[turn_is_white][king_sq] & (pawns & ~cap_m))
                     | (tb.knight_attacks[king_sq] & knights)
                     | (tb.king_attacks[king_sq] & kings)
                     | (rook_attacks(king_sq, occ_ep) & rq)
                     | (bishop_attacks(king_sq, occ_ep) & bq);
        return (att & opp_occ) == 0;
    }

    inline bool is_safe(int king_sq, uint64_t blockers, uint16_t mv) const noexcept {
        int from_sq = mv & 0x3F;
        int to_sq = (mv >> 6) & 0x3F;
        if (from_sq == king_sq) {
            if (std::abs((to_sq & 7) - (from_sq & 7)) == 2) {
                return true;
            }
            return !is_attacked_by(turn_is_white ^ 1, to_sq, (occ_w | occ_b) ^ (1ULL << king_sq));
        }
        if (ep_square >= 0 && to_sq == ep_square && (pawns & (1ULL << from_sq)) && !((occ_w | occ_b) & (1ULL << to_sq))) {
            return is_ep_move_safe(king_sq, from_sq, to_sq);
        }
        return !(blockers & (1ULL << from_sq)) || (tables().line_ray_bb[from_sq][to_sq] & (1ULL << king_sq));
    }

    inline bool has_legal_en_passant() const noexcept {
        if (ep_square < 0) return false;
        uint64_t occ = occ_w | occ_b;
        if (occ & (1ULL << ep_square)) return false;
        uint64_t us_occ = turn_is_white ? occ_w : occ_b;
        uint64_t rank_mask = turn_is_white ? BB_RANK_5 : BB_RANK_4;
        uint64_t capturers = pawns & us_occ & tables().pawn_attacks[turn_is_white ^ 1][ep_square] & rank_mask;
        if (!capturers) return false;
        uint64_t km = kings & us_occ;
        if (!km) return false;
        int king_sq = msb64(km);
        while (capturers) {
            int cap_from = msb64(capturers);
            capturers ^= (1ULL << cap_from);
            if (is_ep_move_safe(king_sq, cap_from, ep_square)) return true;
        }
        return false;
    }

    // En-passant Zobrist convention (matches Polyglot / python-chess _transposition_key()):
    // `ep_square` is hashed into `zobrist_hi`/`zobrist_lo` ONLY when at least one legal
    // en-passant capture actually exists (`has_legal_en_passant()`), NOT unconditionally
    // after every double pawn push. Castling rights are masked to intact kings/rooks.
    inline void refresh_zobrist() noexcept {
        uint64_t cr = clean_castling_rights();
        int eff_ep = has_legal_en_passant() ? (int)ep_square : -1;

        uint64_t h1 = splitmix64(pawns ^ 0x100000001B3ULL);
        h1 ^= splitmix64(knights + 0x200000001B3ULL);
        h1 ^= splitmix64(bishops + 0x300000001B3ULL);
        h1 ^= splitmix64(rooks   + 0x400000001B3ULL);
        h1 ^= splitmix64(queens  + 0x500000001B3ULL);
        h1 ^= splitmix64(kings   + 0x600000001B3ULL);
        h1 ^= splitmix64(occ_w   + 0x700000001B3ULL);
        h1 ^= splitmix64(occ_b   + 0x800000001B3ULL);
        h1 ^= splitmix64(cr      + 0x900000001B3ULL);
        h1 ^= splitmix64((uint64_t)(eff_ep + 2) * 0xA24BAED4963EE407ULL + (uint64_t)turn_is_white);

        uint64_t h2 = splitmix64(occ_w ^ splitmix64(occ_b));
        h2 ^= splitmix64((pawns << 1) ^ (knights << 3) ^ (bishops << 5));
        h2 ^= splitmix64((rooks << 2) ^ (queens << 4) ^ (kings << 7));
        h2 ^= splitmix64(cr ^ ((uint64_t)(eff_ep + 17) << 19) ^ (turn_is_white ? 0xF00DBAADBEEF1234ULL : 0ULL));

        zobrist_hi = h1;
        zobrist_lo = h2;
    }

    inline bool is_repetition_2() const noexcept {
        for (int i = (int)rev_len - 1; i >= 0; --i) {
            if (rev_hi[i] == zobrist_hi && rev_lo[i] == zobrist_lo) {
                return true;
            }
        }
        return false;
    }

    inline bool has_insufficient_material(int color) const noexcept {
        uint64_t us = color ? occ_w : occ_b;
        uint64_t them = color ? occ_b : occ_w;
        if (us & (pawns | rooks | queens)) return false;
        if (us & knights) {
            return (popcnt64(us) <= 2) && ((them & ~kings & ~queens) == 0);
        }
        if (us & bishops) {
            bool same_color = ((bishops & BB_DARK_SQUARES) == 0) || ((bishops & BB_LIGHT_SQUARES) == 0);
            return same_color && (pawns == 0) && (knights == 0);
        }
        return true;
    }

    inline bool is_insufficient_material() const noexcept {
        return has_insufficient_material(WHITE) && has_insufficient_material(BLACK);
    }

    inline uint64_t attacks_mask_piece(int sq, uint64_t occ) const noexcept {
        [[assume(sq >= 0 && sq < 64)]];
        uint64_t m = 1ULL << sq;
        const auto& tb = tables();
        if (pawns & m) {
            int col = (occ_w & m) ? WHITE : BLACK;
            return tb.pawn_attacks[col][sq];
        }
        if (knights & m) return tb.knight_attacks[sq];
        if (kings & m)   return tb.king_attacks[sq];
        uint64_t a = 0;
        if ((bishops | queens) & m) a |= bishop_attacks(sq, occ);
        if ((rooks | queens) & m)   a |= rook_attacks(sq, occ);
        return a;
    }

    // Appends pseudo-legal moves in exact `python-chess` `scan_reversed` order
    int generate_pseudo_legal(uint16_t* out, uint64_t from_mask, uint64_t to_mask) const noexcept {
        int n = 0;
        const auto& tb = tables();
        uint64_t us_occ = turn_is_white ? occ_w : occ_b;
        uint64_t opp_occ = turn_is_white ? occ_b : occ_w;
        uint64_t occ = us_occ | opp_occ;

        // 1. Non-pawn piece moves (scan_reversed)
        uint64_t non_pawns = us_occ & ~pawns & from_mask;
        while (non_pawns) {
            int from_sq = msb64(non_pawns);
            non_pawns ^= (1ULL << from_sq);
            uint64_t mv_bb = attacks_mask_piece(from_sq, occ) & ~us_occ & to_mask;
            while (mv_bb) {
                int to_sq = msb64(mv_bb);
                mv_bb ^= (1ULL << to_sq);
                out[n++] = pack_mv(from_sq, to_sq, 0);
            }
        }

        // 2. Castling moves (if from_mask & kings)
        if (from_mask & kings) {
            uint64_t backrank = turn_is_white ? BB_RANK_1 : BB_RANK_8;
            uint64_t k_bb = us_occ & kings & backrank & from_mask;
            k_bb &= (~k_bb + 1ULL);
            if (k_bb) {
                int k_sq = msb64(k_bb);
                uint64_t cr_cands = clean_castling_rights() & backrank & to_mask;
                while (cr_cands) {
                    int cand = msb64(cr_cands);
                    cr_cands ^= (1ULL << cand);
                    uint64_t rook_bb = 1ULL << cand;
                    bool a_side = (rook_bb < k_bb);
                    int kto_sq = a_side ? (turn_is_white ? 2 : 58) : (turn_is_white ? 6 : 62);
                    int rto_sq = a_side ? (turn_is_white ? 3 : 59) : (turn_is_white ? 5 : 61);
                    uint64_t king_to = 1ULL << kto_sq;
                    uint64_t rook_to = 1ULL << rto_sq;
                    uint64_t king_path = tb.between_bb[k_sq][kto_sq];
                    uint64_t rook_path = tb.between_bb[cand][rto_sq];
                    if (!(((occ ^ k_bb ^ rook_bb) & (king_path | rook_path | king_to | rook_to)) ||
                          attacked_for_king(king_path | k_bb, occ ^ k_bb) ||
                          attacked_for_king(king_to, occ ^ k_bb ^ rook_bb ^ rook_to))) {
                        out[n++] = pack_mv(k_sq, kto_sq, 0);
                    }
                }
            }
        }

        // 3. Pawn moves
        uint64_t our_pawns = pawns & us_occ & from_mask;
        if (!our_pawns) return n;

        // 3a. Pawn captures (scan_reversed)
        uint64_t capturers = our_pawns;
        while (capturers) {
            int from_sq = msb64(capturers);
            capturers ^= (1ULL << from_sq);
            uint64_t targets = tb.pawn_attacks[turn_is_white][from_sq] & opp_occ & to_mask;
            while (targets) {
                int to_sq = msb64(targets);
                targets ^= (1ULL << to_sq);
                int r = to_sq >> 3;
                if (r == 0 || r == 7) {
                    out[n++] = pack_mv(from_sq, to_sq, PT_QUEEN);
                    out[n++] = pack_mv(from_sq, to_sq, PT_ROOK);
                    out[n++] = pack_mv(from_sq, to_sq, PT_BISHOP);
                    out[n++] = pack_mv(from_sq, to_sq, PT_KNIGHT);
                } else {
                    out[n++] = pack_mv(from_sq, to_sq, 0);
                }
            }
        }

        // 3b. Single and double pawn advances
        uint64_t single_moves, double_moves;
        if (turn_is_white) {
            single_moves = (our_pawns << 8) & ~occ;
            double_moves = (single_moves << 8) & ~occ & (BB_RANK_3 | BB_RANK_4);
        } else {
            single_moves = (our_pawns >> 8) & ~occ;
            double_moves = (single_moves >> 8) & ~occ & (BB_RANK_6 | BB_RANK_5);
        }
        single_moves &= to_mask;
        double_moves &= to_mask;

        while (single_moves) {
            int to_sq = msb64(single_moves);
            single_moves ^= (1ULL << to_sq);
            int from_sq = turn_is_white ? (to_sq - 8) : (to_sq + 8);
            int r = to_sq >> 3;
            if (r == 0 || r == 7) {
                out[n++] = pack_mv(from_sq, to_sq, PT_QUEEN);
                out[n++] = pack_mv(from_sq, to_sq, PT_ROOK);
                out[n++] = pack_mv(from_sq, to_sq, PT_BISHOP);
                out[n++] = pack_mv(from_sq, to_sq, PT_KNIGHT);
            } else {
                out[n++] = pack_mv(from_sq, to_sq, 0);
            }
        }

        while (double_moves) {
            int to_sq = msb64(double_moves);
            double_moves ^= (1ULL << to_sq);
            int from_sq = turn_is_white ? (to_sq - 16) : (to_sq + 16);
            out[n++] = pack_mv(from_sq, to_sq, 0);
        }

        // 3c. En passant captures
        if (ep_square > 0 && ((1ULL << ep_square) & to_mask) && !(occ & (1ULL << ep_square))) {
            uint64_t rank_m = turn_is_white ? BB_RANK_5 : BB_RANK_4;
            uint64_t ep_caps = our_pawns & tb.pawn_attacks[turn_is_white ^ 1][ep_square] & rank_m;
            while (ep_caps) {
                int cap_sq = msb64(ep_caps);
                ep_caps ^= (1ULL << cap_sq);
                out[n++] = pack_mv(cap_sq, ep_square, 0);
            }
        }

        return n;
    }

    int generate_legal_moves(uint16_t* out_moves) const noexcept {
        const auto& tb = tables();
        uint64_t us_occ = turn_is_white ? occ_w : occ_b;
        uint64_t occ = occ_w | occ_b;
        uint64_t king_mask = kings & us_occ;
        if (!king_mask) {
            return generate_pseudo_legal(out_moves, BB_ALL, BB_ALL);
        }

        int king_sq = msb64(king_mask);
        uint64_t blockers = slider_blockers(king_sq);
        uint64_t checkers = attackers_mask(turn_is_white ^ 1, king_sq, occ);

        uint16_t pseudo[256];
        int n_pseudo = 0;

        if (checkers) {
            uint64_t sliders = checkers & (bishops | rooks | queens);
            uint64_t attacked = 0;
            while (sliders) {
                int chk = msb64(sliders);
                sliders ^= (1ULL << chk);
                attacked |= tb.line_ray_bb[king_sq][chk] & ~(1ULL << chk);
            }
            uint64_t k_targets = tb.king_attacks[king_sq] & ~us_occ & ~attacked;
            while (k_targets) {
                int to_sq = msb64(k_targets);
                k_targets ^= (1ULL << to_sq);
                pseudo[n_pseudo++] = pack_mv(king_sq, to_sq, 0);
            }

            int checker = msb64(checkers);
            if ((1ULL << checker) == checkers) {
                uint64_t target = tb.between_bb[king_sq][checker] | checkers;
                n_pseudo += generate_pseudo_legal(pseudo + n_pseudo, ~kings, target);

                if (ep_square > 0 && !((1ULL << ep_square) & target) && !(occ & (1ULL << ep_square))) {
                    int last_double = ep_square + (turn_is_white ? -8 : 8);
                    if (last_double == checker) {
                        uint64_t rank_m = turn_is_white ? BB_RANK_5 : BB_RANK_4;
                        uint64_t ep_caps = pawns & us_occ & tb.pawn_attacks[turn_is_white ^ 1][ep_square] & rank_m;
                        while (ep_caps) {
                            int cap_sq = msb64(ep_caps);
                            ep_caps ^= (1ULL << cap_sq);
                            pseudo[n_pseudo++] = pack_mv(cap_sq, ep_square, 0);
                        }
                    }
                }
            }
        } else {
            n_pseudo = generate_pseudo_legal(pseudo, BB_ALL, BB_ALL);
        }

        int n_leg = 0;
        for (int i = 0; i < n_pseudo; ++i) {
            if (is_safe(king_sq, blockers, pseudo[i])) {
                out_moves[n_leg++] = pseudo[i];
            }
        }
        return n_leg;
    }

    // Fast unpruned legal move + canonical action lookup (100% legal moves kept, zero sorting/pruning)
    int generate_legal_moves_and_actions(uint16_t* out_acts, uint16_t* out_moves) const noexcept {
        int n_leg = generate_legal_moves(out_moves);
        if (n_leg == 0) return 0;
        const auto& act_tbl = tables().action_table[turn_is_white ? 0 : 1];
        for (int i = 0; i < n_leg; ++i) {
            uint16_t mp = out_moves[i];
            int from_sq = mp & 0x3F;
            int to_sq = (mp >> 6) & 0x3F;
            int promo = (mp >> 12) & 0xF;
            [[assume(promo >= 0 && promo < 6)]];
            out_acts[i] = act_tbl[from_sq][to_sq][promo];
        }
        return n_leg;
    }

    inline bool has_any_legal_move() const noexcept {
        uint16_t mvs[256];
        return generate_legal_moves(mvs) > 0;
    }

    inline bool is_check() const noexcept {
        uint64_t us_occ = turn_is_white ? occ_w : occ_b;
        uint64_t km = kings & us_occ;
        if (!km) return false;
        return is_attacked_by(turn_is_white ^ 1, msb64(km), occ_w | occ_b);
    }

    inline void evaluate_terminal(
        int32_t& out_term,
        double& out_tval,
        int32_t& out_solved,
        double& out_sval,
        int32_t& out_pdep
    ) const noexcept {
        if (halfmove_clock >= 100) {
            out_term = 1; out_tval = 0.0; out_solved = 0; out_sval = 0.0; out_pdep = 1;
            return;
        }
        if (is_insufficient_material()) {
            out_term = 1; out_tval = 0.0; out_solved = 0; out_sval = 0.0; out_pdep = 0;
            return;
        }
        if (is_repetition_2()) {
            out_term = 1; out_tval = 0.0; out_solved = 0; out_sval = 0.0; out_pdep = 1;
            return;
        }
        if (!has_any_legal_move()) {
            if (is_check()) {
                out_term = 1; out_tval = -1.0; out_solved = 1; out_sval = -1.0; out_pdep = 0;
            } else {
                out_term = 1; out_tval = 0.0; out_solved = 0; out_sval = 0.0; out_pdep = 0;
            }
            return;
        }
        out_term = 0; out_tval = 0.0; out_solved = 0; out_sval = 0.0; out_pdep = 0;
    }

    // Retained for test harness compatibility (MVV-LVA move ordering bonus; never prunes legal moves)
    int generate_scored_legal_moves(uint16_t* out_acts, uint16_t* out_moves, double* out_bonuses) const noexcept {
        int n_leg = generate_legal_moves(out_moves);
        if (n_leg == 0) return 0;

        static constexpr int PIECE_VAL[7] = {100, 100, 320, 330, 500, 900, 20000};
        const auto& act_tbl = tables().action_table[turn_is_white ? 0 : 1];
        uint64_t opp_occ = turn_is_white ? occ_b : occ_w;

        for (int i = 0; i < n_leg; ++i) {
            uint16_t mp = out_moves[i];
            int from_sq = mp & 0x3F;
            int to_sq = (mp >> 6) & 0x3F;
            int promo = (mp >> 12) & 0xF;
            [[assume(promo >= 0 && promo < 6)]];
            out_acts[i] = act_tbl[from_sq][to_sq][promo];

            if (out_bonuses != nullptr) {
                uint64_t to_m = 1ULL << to_sq;
                bool is_ep = (ep_square >= 0 && to_sq == ep_square && (pawns & (1ULL << from_sq)) && !(opp_occ & to_m));
                bool is_cap = ((opp_occ & to_m) != 0) || is_ep;
                if (is_cap) {
                    int att_pt = piece_type_at(from_sq);
                    int att_v = PIECE_VAL[att_pt ? att_pt : 1];
                    int vic_v = 100;
                    if (!is_ep) {
                        int vic_pt = piece_type_at(to_sq);
                        vic_v = PIECE_VAL[vic_pt ? vic_pt : 1];
                    }
                    out_bonuses[i] = (vic_v >= att_v) ? 0.35 : 0.10;
                } else {
                    out_bonuses[i] = 0.0;
                }
            }
        }
        return n_leg;
    }

    inline void remove_piece_at(int sq) noexcept {
        [[assume(sq >= 0 && sq < 64)]];
        uint64_t m = ~(1ULL << sq);
        pawns &= m; knights &= m; bishops &= m; rooks &= m; queens &= m; kings &= m;
        occ_w &= m; occ_b &= m;
    }

    inline void set_piece_at(int sq, int pt, int color) noexcept {
        [[assume(sq >= 0 && sq < 64)]];
        remove_piece_at(sq);
        uint64_t m = 1ULL << sq;
        if (color) occ_w |= m; else occ_b |= m;
        switch (pt) {
            case PT_PAWN:   pawns   |= m; break;
            case PT_KNIGHT: knights |= m; break;
            case PT_BISHOP: bishops |= m; break;
            case PT_ROOK:   rooks   |= m; break;
            case PT_QUEEN:  queens  |= m; break;
            case PT_KING:   kings   |= m; break;
            default: break;
        }
    }

    void push(uint16_t mp) noexcept {
        int from_sq = mp & 0x3F;
        int to_sq = (mp >> 6) & 0x3F;
        int promo = (mp >> 12) & 0xF;

        uint64_t prev_cr = clean_castling_rights();
        bool had_ep = has_legal_en_passant();
        uint64_t prev_hi = zobrist_hi;
        uint64_t prev_lo = zobrist_lo;

        int pt = piece_type_at(from_sq);
        int cap_pt = piece_type_at(to_sq);
        int old_ep = ep_square;
        ep_square = -1;

        bool is_zeroing = (pt == PT_PAWN) || (cap_pt != PT_NONE);
        halfmove_clock = is_zeroing ? 0 : (int16_t)(halfmove_clock + 1);

        int us = turn_is_white ? WHITE : BLACK;
        remove_piece_at(from_sq);

        uint64_t new_cr = prev_cr & ~(1ULL << from_sq) & ~(1ULL << to_sq);
        if (pt == PT_KING) {
            new_cr &= us ? ~BB_RANK_1 : ~BB_RANK_8;
        }

        bool is_castling = (pt == PT_KING) && (std::abs((to_sq & 7) - (from_sq & 7)) == 2);
        if (is_castling) {
            bool a_side = (to_sq & 7) < (from_sq & 7);
            int r_from = a_side ? (us ? 0 : 56) : (us ? 7 : 63);
            int r_to   = a_side ? (us ? 3 : 59) : (us ? 5 : 61);
            remove_piece_at(r_from);
            set_piece_at(to_sq, PT_KING, us);
            set_piece_at(r_to, PT_ROOK, us);
        } else {
            if (pt == PT_PAWN) {
                int diff = to_sq - from_sq;
                if (us && diff == 16 && (from_sq >> 3) == 1) {
                    ep_square = (int8_t)(from_sq + 8);
                } else if (!us && diff == -16 && (from_sq >> 3) == 6) {
                    ep_square = (int8_t)(from_sq - 8);
                } else if (old_ep >= 0 && to_sq == old_ep && (std::abs(diff) == 7 || std::abs(diff) == 9) && cap_pt == PT_NONE) {
                    int cap_sq = us ? (old_ep - 8) : (old_ep + 8);
                    remove_piece_at(cap_sq);
                }
            }
            if (promo != 0) {
                pt = promo;
            }
            set_piece_at(to_sq, pt, us);
        }

        castling_rights = new_cr;
        castling_rights = clean_castling_rights();
        turn_is_white ^= 1;

        bool irrev = is_zeroing || (castling_rights != prev_cr) || had_ep;
        if (irrev) {
            rev_len = 0;
        } else if (rev_len < 104) {
            rev_hi[rev_len] = prev_hi;
            rev_lo[rev_len] = prev_lo;
            rev_len++;
        }

        refresh_zobrist();
    }

    // Encodes 18-plane `gpu_unpack` float32 tensor directly into `out_1152` (AVX2 vectorized)
    inline void encode_gpu_unpack(float* out_1152) const noexcept {
        __m256 z = _mm256_setzero_ps();
        for (int off = 0; off < 18 * 64; off += 32) {
            _mm256_storeu_ps(out_1152 + off + 0,  z);
            _mm256_storeu_ps(out_1152 + off + 8,  z);
            _mm256_storeu_ps(out_1152 + off + 16, z);
            _mm256_storeu_ps(out_1152 + off + 24, z);
        }
        const uint64_t bbs[12] = {
            occ_w & pawns, occ_w & knights, occ_w & bishops, occ_w & rooks, occ_w & queens, occ_w & kings,
            occ_b & pawns, occ_b & knights, occ_b & bishops, occ_b & rooks, occ_b & queens, occ_b & kings
        };
        for (int p = 0; p < 12; ++p) {
            uint64_t b = bbs[p];
            float* plane = out_1152 + p * 64;
            while (b) {
                int sq = ctz64(b);
                plane[sq] = 1.0f;
                b &= (b - 1);
            }
        }
        if (!turn_is_white) {
            fill_plane_64_avx2(out_1152 + 12 * 64, 1.0f);
        }
        uint64_t cr = clean_castling_rights();
        int cmask = (int)(((cr >> 7) & 1ULL) | ((cr & 1ULL) << 1) | (((cr >> 63) & 1ULL) << 2) | (((cr >> 56) & 1ULL) << 3));
        for (int b = 0; b < 4; ++b) {
            if ((cmask >> b) & 1) {
                fill_plane_64_avx2(out_1152 + (13 + b) * 64, 1.0f);
            }
        }
        float hm = (halfmove_clock >= 100) ? 1.0f : ((float)halfmove_clock / 100.0f);
        if (hm > 0.0f) {
            fill_plane_64_avx2(out_1152 + 17 * 64, hm);
        }
    }
};

} // namespace bb
