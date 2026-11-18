// Copyright (c) 2026 Musheer Alam. Licensed under GPL-3.0.
/*
 * High-Speed C++23 Lichess Eval JSONL Streaming Converter
 * (src/neurachess/training/lichess_eval_converter.cpp)
 * =======================================================
 * Streams uncompressed JSONL lines (e.g. from `zstdcat lichess_db_eval.jsonl.zst | -`)
 * skipping lines `< skip_lines`, checks SHA-256 core FEN holdout hashes + 98-byte canonical keys,
 * runs fast bitboard FEN parser + C++23 legal movegen, computes Top-5 Boltzmann soft distribution
 * (`T=0.5`) matching NeuraChess S1, and outputs unified `(PackedV1 107B, UnifiedSidecar68 68B)` shards
 * at >200,000 pos/s without Python/python-chess bottlenecks.
 */

#include "unified_pack_common.hpp"

#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <vector>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace fs = std::filesystem;

namespace nc_lichess {

using nc_pack::ExcludeSet;
using nc_pack::PackedV1;
using nc_pack::UnifiedSidecar68;

struct Stats {
    std::atomic<uint64_t> lines_seen{0};
    std::atomic<uint64_t> records_written{0};
    std::atomic<uint64_t> skipped_overlap{0};
    std::atomic<uint64_t> skipped_invalid{0};
};

static bool parse_fen_to_cboard(std::string_view fen, bb::CBoard& cb, std::string_view& out_core4) noexcept {
    cb = bb::CBoard{};
    cb.ep_square = -1;
    size_t n = fen.size();
    size_t i = 0;
    while (i < n && fen[i] == ' ') i++;
    size_t core_start = i;

    int rank = 7, file = 0;
    while (i < n && fen[i] != ' ') {
        char ch = fen[i++];
        if (ch == '/') {
            if (file != 8 || rank <= 0) return false;
            rank--;
            file = 0;
        } else if (ch >= '1' && ch <= '8') {
            file += (ch - '0');
            if (file > 8) return false;
        } else {
            if (file >= 8) return false;
            int sq = rank * 8 + file++;
            uint64_t m = 1ULL << sq;
            bool is_w = (ch >= 'A' && ch <= 'Z');
            char lc = (char)(is_w ? (ch - 'A' + 'a') : ch);
            if (is_w) cb.occ_w |= m; else cb.occ_b |= m;
            switch (lc) {
                case 'p': cb.pawns |= m; break;
                case 'n': cb.knights |= m; break;
                case 'b': cb.bishops |= m; break;
                case 'r': cb.rooks |= m; break;
                case 'q': cb.queens |= m; break;
                case 'k': cb.kings |= m; break;
                default: return false;
            }
        }
    }
    if (rank != 0 || file != 8) return false;

    while (i < n && fen[i] == ' ') i++;
    if (i >= n) return false;
    char stm = fen[i++];
    if (stm == 'w') cb.turn_is_white = 1;
    else if (stm == 'b') cb.turn_is_white = 0;
    else return false;

    while (i < n && fen[i] == ' ') i++;
    if (i >= n) return false;
    while (i < n && fen[i] != ' ') {
        char c = fen[i++];
        if (c == 'K') cb.castling_rights |= (1ULL << 7);
        else if (c == 'Q') cb.castling_rights |= (1ULL << 0);
        else if (c == 'k') cb.castling_rights |= (1ULL << 63);
        else if (c == 'q') cb.castling_rights |= (1ULL << 56);
        else if (c == '-') {}
        else return false;
    }

    while (i < n && fen[i] == ' ') i++;
    if (i < n && fen[i] != '-') {
        if (i + 2 <= n && fen[i] >= 'a' && fen[i] <= 'h' && fen[i + 1] >= '1' && fen[i + 1] <= '8') {
            int f = fen[i] - 'a';
            int r = fen[i + 1] - '1';
            cb.ep_square = (int8_t)(r * 8 + f);
            i += 2;
        } else {
            return false;
        }
    } else if (i < n && fen[i] == '-') {
        i++;
    }
    size_t core_end = i;
    out_core4 = fen.substr(core_start, core_end - core_start);

    while (i < n && fen[i] == ' ') i++;
    int hm = 0;
    while (i < n && std::isdigit((unsigned char)fen[i])) {
        hm = std::min(100, hm * 10 + (fen[i] - '0'));
        i++;
    }
    cb.halfmove_clock = (int16_t)hm;
    cb.refresh_zobrist();
    return (cb.occ_w & cb.kings) != 0ULL && (cb.occ_b & cb.kings) != 0ULL;
}

static inline uint16_t parse_uci_packed(std::string_view u) noexcept {
    if (u.size() < 4) return 0;
    int ff = u[0] - 'a', fr = u[1] - '1';
    int tf = u[2] - 'a', tr = u[3] - '1';
    if (ff < 0 || ff > 7 || fr < 0 || fr > 7 || tf < 0 || tf > 7 || tr < 0 || tr > 7) return 0;
    int pr = 0;
    if (u.size() >= 5) {
        switch (u[4]) {
            case 'n': pr = bb::PT_KNIGHT; break;
            case 'b': pr = bb::PT_BISHOP; break;
            case 'r': pr = bb::PT_ROOK; break;
            case 'q': pr = bb::PT_QUEEN; break;
            default: pr = 0; break;
        }
    }
    return bb::pack_mv(fr * 8 + ff, tr * 8 + tf, pr);
}

static inline bool extract_json_str(std::string_view sv, std::string_view key, size_t start_pos, std::string_view& val, size_t& after_pos) noexcept {
    size_t p = sv.find(key, start_pos);
    if (p == std::string_view::npos) return false;
    p += key.size();
    while (p < sv.size() && (sv[p] == ' ' || sv[p] == ':' || sv[p] == '\t')) p++;
    if (p >= sv.size() || sv[p] != '"') return false;
    p++;
    size_t e = sv.find('"', p);
    if (e == std::string_view::npos) return false;
    val = sv.substr(p, e - p);
    after_pos = e + 1;
    return true;
}

static inline bool extract_json_int(std::string_view sv, std::string_view key, int& out_v) noexcept {
    size_t p = sv.find(key);
    if (p == std::string_view::npos) return false;
    p += key.size();
    while (p < sv.size() && (sv[p] == ' ' || sv[p] == ':' || sv[p] == '\t')) p++;
    if (p >= sv.size()) return false;
    bool neg = false;
    if (sv[p] == '-') { neg = true; p++; }
    if (p >= sv.size() || !std::isdigit((unsigned char)sv[p])) return false;
    int64_t acc = 0;
    while (p < sv.size() && std::isdigit((unsigned char)sv[p])) {
        acc = acc * 10 + (sv[p] - '0');
        if (acc > 1000000LL) break;
        p++;
    }
    out_v = (int)(neg ? -acc : acc);
    return true;
}

static bool convert_line(
    std::string_view line,
    double temperature,
    const ExcludeSet& excl,
    PackedV1& out_v1,
    UnifiedSidecar68& out_sc,
    uint64_t& out_dedup_hash,
    Stats& stats
) noexcept {
    if (line.find("\"fen\"") == std::string_view::npos || line.find("\"pvs\"") == std::string_view::npos) {
        return false;
    }

    std::string_view fen_sv;
    size_t fen_after = 0;
    if (!extract_json_str(line, "\"fen\"", 0, fen_sv, fen_after)) {
        stats.skipped_invalid.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    bb::CBoard cb{};
    std::string_view core4;
    if (!parse_fen_to_cboard(fen_sv, cb, core4)) {
        stats.skipped_invalid.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    out_dedup_hash = nc_pack::sha256_prefix64(reinterpret_cast<const uint8_t*>(core4.data()), core4.size());
    if (!excl.sha_prefixes.empty() && excl.sha_prefixes.find(out_dedup_hash) != excl.sha_prefixes.end()) {
        stats.skipped_overlap.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    size_t pvs_pos = line.find("\"pvs\"", fen_after);
    if (pvs_pos == std::string_view::npos) return false;
    size_t arr_start = line.find('[', pvs_pos);
    if (arr_start == std::string_view::npos) return false;

    uint16_t leg_acts[256];
    uint16_t leg_moves[256];
    int n_leg = cb.generate_legal_moves_and_actions(leg_acts, leg_moves);
    if (n_leg <= 0) {
        stats.skipped_invalid.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    bool is_black = (cb.turn_is_white == 0);
    uint16_t pv_acts[5] = {};
    int pv_cps[5] = {};
    int n_pv = 0;

    size_t cur = arr_start + 1;
    while (cur < line.size() && n_pv < 5) {
        while (cur < line.size() && line[cur] != '{' && line[cur] != ']') cur++;
        if (cur >= line.size() || line[cur] == ']') break;
        size_t obj_start = cur;
        int depth = 0;
        bool in_str = false;
        while (cur < line.size()) {
            char c = line[cur++];
            if (c == '"' && (cur < 2 || line[cur - 2] != '\\')) in_str = !in_str;
            else if (!in_str) {
                if (c == '{') depth++;
                else if (c == '}') {
                    depth--;
                    if (depth == 0) break;
                }
            }
        }
        if (depth != 0) break;
        std::string_view obj = line.substr(obj_start, cur - obj_start);

        std::string_view mv_line;
        size_t dummy_pos = 0;
        if (!extract_json_str(obj, "\"line\"", 0, mv_line, dummy_pos)) continue;
        size_t sp = mv_line.find(' ');
        if (sp != std::string_view::npos) mv_line = mv_line.substr(0, sp);
        uint16_t mp = parse_uci_packed(mv_line);
        if (mp == 0) continue;

        uint16_t act = 65535;
        for (int m = 0; m < n_leg; ++m) {
            if (leg_moves[m] == mp) {
                act = leg_acts[m];
                break;
            }
        }
        if (act == 65535) continue;

        int raw_cp = 0;
        int mate = 0;
        int stm_cp = 0;
        if (extract_json_int(obj, "\"cp\"", raw_cp)) {
            stm_cp = is_black ? -raw_cp : raw_cp;
        } else if (extract_json_int(obj, "\"mate\"", mate)) {
            stm_cp = ((mate > 0 && !is_black) || (mate < 0 && is_black)) ? 10000 : -10000;
        } else {
            continue;
        }
        stm_cp = std::clamp(stm_cp, -32767, 32767);

        bool dup = false;
        for (int k = 0; k < n_pv; ++k) {
            if (pv_acts[k] == act) { dup = true; break; }
        }
        if (!dup) {
            pv_acts[n_pv] = act;
            pv_cps[n_pv] = stm_cp;
            n_pv++;
        }
    }

    if (n_pv == 0) {
        stats.skipped_invalid.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    int eval_cp = pv_cps[0];
    double p_win = 1.0 / (1.0 + std::pow(10.0, -(double)eval_cp / 400.0));
    double p_loss = 1.0 / (1.0 + std::pow(10.0, (double)eval_cp / 400.0));
    double p_draw = std::max(0.0, 1.0 - p_win - p_loss);

    uint8_t cast_mask = 0;
    if (cb.castling_rights & (1ULL << 7))  cast_mask |= 1;
    if (cb.castling_rights & (1ULL << 0))  cast_mask |= 2;
    if (cb.castling_rights & (1ULL << 63)) cast_mask |= 4;
    if (cb.castling_rights & (1ULL << 56)) cast_mask |= 8;

    out_v1 = PackedV1{};
    out_v1.bbs[0]  = cb.pawns   & cb.occ_w;
    out_v1.bbs[1]  = cb.knights & cb.occ_w;
    out_v1.bbs[2]  = cb.bishops & cb.occ_w;
    out_v1.bbs[3]  = cb.rooks   & cb.occ_w;
    out_v1.bbs[4]  = cb.queens  & cb.occ_w;
    out_v1.bbs[5]  = cb.kings   & cb.occ_w;
    out_v1.bbs[6]  = cb.pawns   & cb.occ_b;
    out_v1.bbs[7]  = cb.knights & cb.occ_b;
    out_v1.bbs[8]  = cb.bishops & cb.occ_b;
    out_v1.bbs[9]  = cb.rooks   & cb.occ_b;
    out_v1.bbs[10] = cb.queens  & cb.occ_b;
    out_v1.bbs[11] = cb.kings   & cb.occ_b;
    out_v1.turn = is_black ? 1 : 0;
    out_v1.castling = cast_mask;
    out_v1.ep_square = (cb.ep_square >= 0) ? (uint8_t)cb.ep_square : 255;
    out_v1.halfmove = (uint8_t)cb.halfmove_clock;
    out_v1.action_index = pv_acts[0];
    out_v1.eval_cp = (int16_t)eval_cp;
    out_v1.wdl[0] = (uint8_t)std::clamp((int)std::llround(p_win * 255.0), 0, 255);
    out_v1.wdl[1] = (uint8_t)std::clamp((int)std::llround(p_draw * 255.0), 0, 255);
    out_v1.wdl[2] = (uint8_t)std::clamp((int)std::llround(p_loss * 255.0), 0, 255);

    if (excl.is_forbidden_v1(out_v1)) {
        stats.skipped_overlap.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    // Boltzmann soft policy distribution over top-K Stockfish PVs (matches S1 training)
    double w_exp[5] = {};
    double max_s = -1e30;
    double inv_temp = 1.0 / (100.0 * std::max(1e-3, temperature));
    for (int k = 0; k < n_pv; ++k) {
        double s = (double)pv_cps[k] * inv_temp;
        w_exp[k] = s;
        if (s > max_s) max_s = s;
    }
    double sum_e = 0.0;
    for (int k = 0; k < n_pv; ++k) {
        w_exp[k] = std::exp(w_exp[k] - max_s);
        sum_e += w_exp[k];
    }

    out_sc = UnifiedSidecar68{};
    out_sc.num_topk = (uint8_t)n_pv;
    out_sc.flags = (uint8_t)((is_black ? 1 : 0) | ((cb.ep_square >= 0) ? 2 : 0)); // bit2=0 (SF), bit3=0 (no MLH)
    out_sc.visits_clamped = (uint16_t)std::min(500, n_pv * 100);
    for (int k = 0; k < 8; ++k) {
        out_sc.top8_actions[k] = (k < n_pv) ? pv_acts[k] : 65535;
        out_sc.top8_probs[k] = (k < n_pv && sum_e > 0.0) ? (float)(w_exp[k] / sum_e) : 0.0f;
    }
    out_sc.backed_q = (float)std::tanh((double)eval_cp * nc_pack::CP_TANH_SCALE);
    out_sc.backed_d = (float)p_draw;
    out_sc.result_q = out_sc.backed_q;
    out_sc.plies_left = -1.0f;
    return true;
}

static int run_converter(int argc, char** argv) {
    nc_pack::init_nc_action_table();

    std::string out_prefix;
    std::string input_path = "-";
    std::vector<std::string> exclude_hash_files;
    std::vector<std::string> exclude_bin_files;
    uint64_t skip_lines = 0;
    size_t max_records = 0;
    int threads = (int)std::max(1u, std::thread::hardware_concurrency());
    double temperature = 0.5;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--out-prefix" && i + 1 < argc) out_prefix = argv[++i];
        else if (a == "--skip-lines" && i + 1 < argc) skip_lines = std::stoull(argv[++i]);
        else if (a == "--max-records" && i + 1 < argc) max_records = (size_t)std::stoull(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) threads = std::max(1, std::stoi(argv[++i]));
        else if (a == "--temperature" && i + 1 < argc) temperature = std::stod(argv[++i]);
        else if (a == "--exclude-hashes" && i + 1 < argc) exclude_hash_files.emplace_back(argv[++i]);
        else if (a == "--exclude-bin" && i + 1 < argc) exclude_bin_files.emplace_back(argv[++i]);
        else input_path = a;
    }

    if (out_prefix.empty()) {
        std::cerr << "Usage: lichess_eval_converter --out-prefix <path> [--skip-lines N] [--max-records N] [--exclude-hashes f] [--exclude-bin f] [input.jsonl|-]\n";
        return 2;
    }

    ExcludeSet excl;
    for (const auto& f : exclude_hash_files) excl.load_hash_hex_file(f);
    for (const auto& f : exclude_bin_files) excl.load_bin_98(f, sizeof(PackedV1));

    fs::path out_bin(out_prefix + ".bin");
    fs::path out_sc(out_prefix + ".sidecar68.bin");
    if (out_bin.has_parent_path()) fs::create_directories(out_bin.parent_path());
    std::ofstream fb(out_bin, std::ios::binary | std::ios::trunc);
    std::ofstream fsc(out_sc, std::ios::binary | std::ios::trunc);
    if (!fb || !fsc) {
        std::cerr << "Failed opening outputs at: " << out_prefix << "\n";
        return 3;
    }

    std::ifstream fin;
    std::istream* in_ptr = nullptr;
    if (input_path == "-") {
#ifdef _WIN32
        _setmode(_fileno(stdin), _O_BINARY);
#endif
        in_ptr = &std::cin;
    } else {
        fin.open(input_path, std::ios::binary);
        if (!fin) {
            std::cerr << "Failed opening input: " << input_path << "\n";
            return 4;
        }
        in_ptr = &fin;
    }

    Stats stats{};
    std::unordered_set<uint64_t> seen_hashes;
    seen_hashes.reserve(1024 * 1024);

    auto t0 = std::chrono::steady_clock::now();
    std::string line;
    uint64_t line_idx = 0;
    size_t total_written = 0;
    constexpr size_t kBatchLines = 8192;
    std::vector<std::string> batch;
    batch.reserve(kBatchLines);

    auto flush_lines = [&]() {
        if (batch.empty()) return;
        size_t nb = batch.size();
        std::vector<uint8_t> ok_flags(nb, 0);
        std::vector<PackedV1> b_v1(nb);
        std::vector<UnifiedSidecar68> b_sc(nb);
        std::vector<uint64_t> b_hash(nb, 0);

        size_t nw = std::min<size_t>((size_t)threads, nb);
        std::atomic<size_t> next_i{0};
        std::vector<std::thread> pool;
        pool.reserve(nw);
        for (size_t wi = 0; wi < nw; ++wi) {
            pool.emplace_back([&]() {
                while (true) {
                    size_t i = next_i.fetch_add(1, std::memory_order_relaxed);
                    if (i >= nb) break;
                    if (convert_line(batch[i], temperature, excl, b_v1[i], b_sc[i], b_hash[i], stats)) {
                        ok_flags[i] = 1;
                    }
                }
            });
        }
        for (auto& th : pool) th.join();

        for (size_t i = 0; i < nb; ++i) {
            if (!ok_flags[i]) continue;
            if (!seen_hashes.insert(b_hash[i]).second) {
                if (seen_hashes.size() > 8000000ULL) seen_hashes.clear();
                continue;
            }
            fb.write(reinterpret_cast<const char*>(&b_v1[i]), sizeof(PackedV1));
            fsc.write(reinterpret_cast<const char*>(&b_sc[i]), sizeof(UnifiedSidecar68));
            total_written++;
            stats.records_written.fetch_add(1, std::memory_order_relaxed);
            if (max_records > 0 && total_written >= max_records) break;
        }
        batch.clear();
    };

    while (std::getline(*in_ptr, line)) {
        stats.lines_seen.fetch_add(1, std::memory_order_relaxed);
        if (line_idx++ < skip_lines) continue;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        batch.push_back(std::move(line));
        if (batch.size() >= kBatchLines) {
            flush_lines();
            if (max_records > 0 && total_written >= max_records) break;
        }
    }
    if (!batch.empty() && (max_records == 0 || total_written < max_records)) {
        flush_lines();
    }

    fb.flush();
    fsc.flush();
    double elapsed_s = std::max(1e-6, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    std::cout << "{"
              << "\"lines_seen\":" << stats.lines_seen.load()
              << ",\"records_written\":" << total_written
              << ",\"skipped_overlap\":" << stats.skipped_overlap.load()
              << ",\"skipped_invalid\":" << stats.skipped_invalid.load()
              << ",\"elapsed_s\":" << elapsed_s
              << ",\"written_pos_per_s\":" << (double)total_written / elapsed_s
              << ",\"out_bin\":\"" << out_bin.generic_string() << "\""
              << ",\"out_sidecar\":\"" << out_sc.generic_string() << "\""
              << "}\n";
    return 0;
}

} // namespace nc_lichess

int main(int argc, char** argv) {
    return nc_lichess::run_converter(argc, argv);
}
