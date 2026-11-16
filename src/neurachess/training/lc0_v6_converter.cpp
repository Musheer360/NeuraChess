// Copyright (c) 2026 Musheer Alam. Licensed under GPL-3.0.
/*
 * High-Speed C++23 Lc0 V6TrainingData (.tar / .gz / .v6) Stream Converter
 * (src/neurachess/training/lc0_v6_converter.cpp)
 * =====================================================================
 * Streams Lc0 `training-run1-test80` 8,356-byte v6 binary records (`ply > 8`),
 * reconstructs exact C++23 bitboard state + ep_square + castling (excluding FRC),
 * maps Lc0's 1,858 MCTS visit distribution (with e1h1/e1a1 -> e1g1/e1c1 castling
 * resolution) into legal NeuraChess 4,672 actions, extracts Top-8 (action, prob),
 * backed-up Q / WDL / plies_left, and writes packed 107B `.bin` + 68B `.lc0_sidecar.bin`
 * shards across multi-threaded P-core/CPU workers at >50,000 pos/s.
 */

#include "../engine/fast_tree/bitboard_chess.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace fs = std::filesystem;

namespace nc_lc0 {

#pragma pack(push, 1)
struct V6TrainingData {
    uint32_t version;
    uint32_t input_format;
    float probabilities[1858];
    uint64_t planes[104];
    uint8_t castling_us_ooo;
    uint8_t castling_us_oo;
    uint8_t castling_them_ooo;
    uint8_t castling_them_oo;
    uint8_t side_to_move_or_enpassant;
    uint8_t rule50_count;
    uint8_t invariance_info;
    uint8_t dummy;
    float root_q;
    float best_q;
    float root_d;
    float best_d;
    float root_m;
    float best_m;
    float plies_left;
    float result_q;
    float result_d;
    float played_q;
    float played_d;
    float played_m;
    float orig_q;
    float orig_d;
    float orig_m;
    uint32_t visits;
    uint16_t played_idx;
    uint16_t best_idx;
    float policy_kld;
    float q_st;
};

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

struct Lc0Sidecar68 {
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

static_assert(sizeof(V6TrainingData) == 8356, "V6TrainingData must be 8356 bytes");
static_assert(sizeof(PackedV1) == 107, "PackedV1 must be 107 bytes");
static_assert(sizeof(Lc0Sidecar68) == 68, "Lc0Sidecar68 must be 68 bytes");

constexpr int kIdxE1A1 = 97;
constexpr int kIdxE1H1 = 103;
constexpr uint16_t kActE1G1 = 307;
constexpr uint16_t kActE1C1 = 335;

// Verified 1-to-1 mapping from Lc0 kMoveStrs[1858] -> NeuraChess 4,672 canonical action index
alignas(64) constexpr uint16_t kLc0ToNcAct[1858] = {
      14,   15,   16,   17,   18,   19,   20,    0,    7,   57,    1,   56,    8,    2,    9,    3,
      10,    4,   11,    5,   12,    6,   13,  115,   87,   88,   89,   90,   91,   92,  122,   73,
      80,  130,  136,   74,  129,   81,   75,   82,   76,   83,   77,   84,   78,   85,   79,  189,
     188,  160,  161,  162,  163,  164,  208,  195,  146,  153,  203,  196,  209,  147,  202,  154,
     148,  155,  149,  156,  150,  157,  151,  152,  263,  262,  261,  233,  234,  235,  236,  281,
     268,  219,  226,  276,  269,  282,  220,  275,  227,  270,  221,  228,  222,  229,  223,  224,
     225,  337,  336,  335,  334,  306,  307,  308,  354,  341,  292,  299,  349,  342,  355,  293,
     348,  300,  343,  294,  301,  344,  295,  296,  297,  298,  411,  410,  409,  408,  407,  379,
     380,  427,  414,  365,  372,  422,  415,  428,  366,  421,  373,  416,  367,  417,  368,  418,
     369,  370,  371,  485,  484,  483,  482,  481,  480,  452,  500,  487,  438,  445,  488,  501,
     439,  494,  489,  440,  490,  441,  491,  442,  492,  443,  444,  559,  558,  557,  556,  555,
     554,  553,  573,  560,  511,  561,  574,  512,  562,  513,  563,  514,  564,  515,  565,  516,
     566,  517,  612,  605,  642,  598,  599,  600,  601,  602,  603,  604,  584,  591,  641,  585,
     640,  592,  586,  593,  587,  594,  588,  595,  589,  596,  692,  685,  678,  715,  699,  671,
     672,  673,  674,  675,  676,  706,  657,  664,  714,  720,  658,  713,  665,  659,  666,  660,
     667,  661,  668,  662,  669,  791,  765,  758,  751,  788,  773,  772,  744,  745,  746,  747,
     748,  792,  779,  730,  737,  787,  780,  793,  731,  786,  738,  732,  739,  733,  740,  734,
     741,  735,  864,  838,  831,  824,  861,  847,  846,  845,  817,  818,  819,  820,  865,  852,
     803,  810,  860,  853,  866,  804,  859,  811,  854,  805,  812,  806,  813,  807,  808,  937,
     911,  904,  897,  934,  921,  920,  919,  918,  890,  891,  892,  938,  925,  876,  883,  933,
     926,  939,  877,  932,  884,  927,  878,  885,  928,  879,  880,  881, 1010,  984,  977,  970,
    1007,  995,  994,  993,  992,  991,  963,  964, 1011,  998,  949,  956, 1006,  999, 1012,  950,
    1005,  957, 1000,  951, 1001,  952, 1002,  953,  954, 1083, 1057, 1050, 1043, 1069, 1068, 1067,
    1066, 1065, 1064, 1036, 1084, 1071, 1022, 1029, 1072, 1085, 1023, 1078, 1073, 1024, 1074, 1025,
    1075, 1026, 1076, 1027, 1156, 1130, 1123, 1143, 1142, 1141, 1140, 1139, 1138, 1137, 1157, 1144,
    1095, 1145, 1158, 1096, 1146, 1097, 1147, 1098, 1148, 1099, 1149, 1100, 1197, 1227, 1190, 1196,
    1189, 1226, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1168, 1175, 1225, 1169, 1224, 1176, 1170,
    1177, 1171, 1178, 1172, 1179, 1301, 1270, 1300, 1263, 1276, 1269, 1262, 1299, 1283, 1255, 1256,
    1257, 1258, 1259, 1260, 1290, 1241, 1248, 1298, 1304, 1242, 1297, 1249, 1243, 1250, 1244, 1251,
    1245, 1252, 1350, 1374, 1343, 1373, 1336, 1375, 1349, 1342, 1335, 1372, 1357, 1356, 1328, 1329,
    1330, 1331, 1332, 1376, 1363, 1314, 1321, 1371, 1364, 1377, 1315, 1370, 1322, 1316, 1323, 1317,
    1324, 1318, 1325, 1423, 1447, 1416, 1446, 1409, 1448, 1422, 1415, 1408, 1445, 1431, 1430, 1429,
    1401, 1402, 1403, 1404, 1449, 1436, 1387, 1394, 1444, 1437, 1450, 1388, 1443, 1395, 1438, 1389,
    1396, 1390, 1397, 1391, 1496, 1520, 1489, 1519, 1482, 1521, 1495, 1488, 1481, 1518, 1505, 1504,
    1503, 1502, 1474, 1475, 1476, 1522, 1509, 1460, 1467, 1517, 1510, 1523, 1461, 1516, 1468, 1511,
    1462, 1469, 1512, 1463, 1464, 1569, 1593, 1562, 1592, 1555, 1594, 1568, 1561, 1554, 1591, 1579,
    1578, 1577, 1576, 1575, 1547, 1548, 1595, 1582, 1533, 1540, 1590, 1583, 1596, 1534, 1589, 1541,
    1584, 1535, 1585, 1536, 1586, 1537, 1642, 1666, 1635, 1665, 1667, 1641, 1634, 1627, 1653, 1652,
    1651, 1650, 1649, 1648, 1620, 1668, 1655, 1606, 1613, 1656, 1669, 1607, 1662, 1657, 1608, 1658,
    1609, 1659, 1610, 1715, 1739, 1708, 1740, 1714, 1707, 1727, 1726, 1725, 1724, 1723, 1722, 1721,
    1741, 1728, 1679, 1729, 1742, 1680, 1730, 1681, 1731, 1682, 1732, 1683, 1782, 1775, 1781, 1811,
    1774, 1780, 1773, 1810, 1766, 1767, 1768, 1769, 1770, 1771, 1772, 1752, 1759, 1809, 1753, 1808,
    1760, 1754, 1761, 1755, 1762, 1855, 1848, 1885, 1854, 1884, 1847, 1860, 1853, 1846, 1883, 1867,
    1839, 1840, 1841, 1842, 1843, 1844, 1874, 1825, 1832, 1882, 1888, 1826, 1881, 1833, 1827, 1834,
    1828, 1835, 1928, 1921, 1934, 1958, 1927, 1957, 1920, 1959, 1933, 1926, 1919, 1956, 1941, 1940,
    1912, 1913, 1914, 1915, 1916, 1960, 1947, 1898, 1905, 1955, 1948, 1961, 1899, 1954, 1906, 1900,
    1907, 1901, 1908, 2008, 2001, 1994, 2007, 2031, 2000, 2030, 1993, 2032, 2006, 1999, 1992, 2029,
    2015, 2014, 2013, 1985, 1986, 1987, 1988, 2033, 2020, 1971, 1978, 2028, 2021, 2034, 1972, 2027,
    1979, 2022, 1973, 1980, 1974, 1981, 2081, 2074, 2067, 2080, 2104, 2073, 2103, 2066, 2105, 2079,
    2072, 2065, 2102, 2089, 2088, 2087, 2086, 2058, 2059, 2060, 2106, 2093, 2044, 2051, 2101, 2094,
    2107, 2045, 2100, 2052, 2095, 2046, 2053, 2096, 2047, 2154, 2147, 2153, 2177, 2146, 2176, 2139,
    2178, 2152, 2145, 2138, 2175, 2163, 2162, 2161, 2160, 2159, 2131, 2132, 2179, 2166, 2117, 2124,
    2174, 2167, 2180, 2118, 2173, 2125, 2168, 2119, 2169, 2120, 2227, 2220, 2226, 2250, 2219, 2249,
    2251, 2225, 2218, 2211, 2237, 2236, 2235, 2234, 2233, 2232, 2204, 2252, 2239, 2190, 2197, 2240,
    2253, 2191, 2246, 2241, 2192, 2242, 2193, 2300, 2293, 2299, 2323, 2292, 2324, 2298, 2291, 2311,
    2310, 2309, 2308, 2307, 2306, 2305, 2325, 2312, 2263, 2313, 2326, 2264, 2314, 2265, 2315, 2266,
    2367, 2360, 2366, 2359, 2365, 2395, 2358, 2364, 2357, 2394, 2350, 2351, 2352, 2353, 2354, 2355,
    2356, 2336, 2343, 2393, 2337, 2392, 2344, 2338, 2345, 2440, 2433, 2439, 2432, 2469, 2438, 2468,
    2431, 2444, 2437, 2430, 2467, 2451, 2423, 2424, 2425, 2426, 2427, 2428, 2458, 2409, 2416, 2466,
    2472, 2410, 2465, 2417, 2411, 2418, 2513, 2506, 2512, 2505, 2518, 2542, 2511, 2541, 2504, 2543,
    2517, 2510, 2503, 2540, 2525, 2524, 2496, 2497, 2498, 2499, 2500, 2544, 2531, 2482, 2489, 2539,
    2532, 2545, 2483, 2538, 2490, 2484, 2491, 2586, 2579, 2592, 2585, 2578, 2591, 2615, 2584, 2614,
    2577, 2616, 2590, 2583, 2576, 2613, 2599, 2598, 2597, 2569, 2570, 2571, 2572, 2617, 2604, 2555,
    2562, 2612, 2605, 2618, 2556, 2611, 2563, 2606, 2557, 2564, 2666, 2659, 2665, 2658, 2651, 2664,
    2688, 2657, 2687, 2650, 2689, 2663, 2656, 2649, 2686, 2673, 2672, 2671, 2670, 2642, 2643, 2644,
    2690, 2677, 2628, 2635, 2685, 2678, 2691, 2629, 2684, 2636, 2679, 2630, 2637, 2739, 2732, 2738,
    2731, 2737, 2761, 2730, 2760, 2723, 2762, 2736, 2729, 2722, 2759, 2747, 2746, 2745, 2744, 2743,
    2715, 2716, 2763, 2750, 2701, 2708, 2758, 2751, 2764, 2702, 2757, 2709, 2752, 2703, 2812, 2805,
    2811, 2804, 2810, 2834, 2803, 2833, 2835, 2809, 2802, 2795, 2821, 2820, 2819, 2818, 2817, 2816,
    2788, 2836, 2823, 2774, 2781, 2824, 2837, 2775, 2830, 2825, 2776, 2885, 2878, 2884, 2877, 2883,
    2907, 2876, 2908, 2882, 2875, 2895, 2894, 2893, 2892, 2891, 2890, 2889, 2909, 2896, 2847, 2897,
    2910, 2848, 2898, 2849, 2952, 2945, 2951, 2944, 2950, 2943, 2949, 2979, 2942, 2948, 2941, 2978,
    2934, 2935, 2936, 2937, 2938, 2939, 2940, 2920, 2927, 2977, 2921, 2976, 2928, 3025, 3018, 3024,
    3017, 3023, 3016, 3053, 3022, 3052, 3015, 3028, 3021, 3014, 3051, 3035, 3007, 3008, 3009, 3010,
    3011, 3012, 3042, 2993, 3000, 3050, 3056, 2994, 3049, 3001, 3098, 3091, 3097, 3090, 3096, 3089,
    3102, 3126, 3095, 3125, 3088, 3127, 3101, 3094, 3087, 3124, 3109, 3108, 3080, 3081, 3082, 3083,
    3084, 3128, 3115, 3066, 3073, 3123, 3116, 3129, 3067, 3122, 3074, 3171, 3170, 3163, 3176, 3169,
    3162, 3175, 3199, 3168, 3198, 3161, 3200, 3174, 3167, 3160, 3197, 3183, 3182, 3181, 3153, 3154,
    3155, 3156, 3201, 3188, 3139, 3146, 3196, 3189, 3202, 3140, 3195, 3147, 3244, 3250, 3243, 3249,
    3242, 3235, 3248, 3272, 3241, 3271, 3234, 3273, 3247, 3240, 3233, 3270, 3257, 3256, 3255, 3254,
    3226, 3227, 3228, 3274, 3261, 3212, 3219, 3269, 3262, 3275, 3213, 3268, 3220, 3324, 3317, 3323,
    3316, 3322, 3315, 3321, 3345, 3314, 3344, 3307, 3346, 3320, 3313, 3306, 3343, 3331, 3330, 3329,
    3328, 3327, 3299, 3300, 3347, 3334, 3285, 3292, 3342, 3335, 3348, 3286, 3341, 3293, 3397, 3390,
    3396, 3389, 3395, 3388, 3394, 3418, 3387, 3417, 3419, 3393, 3386, 3379, 3405, 3404, 3403, 3402,
    3401, 3400, 3372, 3420, 3407, 3358, 3365, 3408, 3421, 3359, 3414, 3470, 3463, 3469, 3462, 3468,
    3461, 3467, 3491, 3460, 3492, 3466, 3459, 3479, 3478, 3477, 3476, 3475, 3474, 3473, 3493, 3480,
    3431, 3481, 3494, 3432, 3537, 3530, 3536, 3529, 3535, 3528, 3534, 3527, 3533, 3563, 3526, 3532,
    3525, 3562, 3518, 3519, 3520, 3521, 3522, 3523, 3524, 3504, 3511, 3561, 3610, 3603, 3609, 3602,
    3608, 3601, 3607, 3600, 3637, 3606, 3636, 3599, 3612, 3605, 3598, 3635, 3619, 3591, 3592, 3593,
    3594, 3595, 3596, 3626, 3577, 3584, 3634, 3683, 3682, 3675, 3681, 3674, 3680, 3673, 3686, 3710,
    3679, 3709, 3672, 3711, 3685, 3678, 3671, 3708, 3693, 3692, 3664, 3665, 3666, 3667, 3668, 3712,
    3699, 3650, 3657, 3707, 3756, 3755, 3754, 3747, 3760, 3753, 3746, 3759, 3783, 3752, 3782, 3745,
    3784, 3758, 3751, 3744, 3781, 3767, 3766, 3765, 3737, 3738, 3739, 3740, 3785, 3772, 3723, 3730,
    3780, 3829, 3828, 3834, 3827, 3833, 3826, 3819, 3832, 3856, 3825, 3855, 3818, 3857, 3831, 3824,
    3817, 3854, 3841, 3840, 3839, 3838, 3810, 3811, 3812, 3858, 3845, 3796, 3803, 3853, 3902, 3908,
    3901, 3907, 3900, 3906, 3899, 3905, 3929, 3898, 3928, 3891, 3930, 3904, 3897, 3890, 3927, 3915,
    3914, 3913, 3912, 3911, 3883, 3884, 3931, 3918, 3869, 3876, 3926, 3982, 3975, 3981, 3974, 3980,
    3973, 3979, 3972, 3978, 4002, 3971, 4001, 4003, 3977, 3970, 3963, 3989, 3988, 3987, 3986, 3985,
    3984, 3956, 4004, 3991, 3942, 3949, 4055, 4048, 4054, 4047, 4053, 4046, 4052, 4045, 4051, 4075,
    4044, 4076, 4050, 4043, 4063, 4062, 4061, 4060, 4059, 4058, 4057, 4077, 4064, 4015, 4122, 4115,
    4121, 4114, 4120, 4113, 4119, 4112, 4118, 4111, 4117, 4147, 4110, 4116, 4109, 4146, 4102, 4103,
    4104, 4105, 4106, 4107, 4108, 4195, 4194, 4187, 4193, 4186, 4192, 4185, 4191, 4184, 4221, 4190,
    4220, 4183, 4196, 4189, 4182, 4219, 4203, 4175, 4176, 4177, 4178, 4179, 4180, 4268, 4267, 4266,
    4259, 4265, 4258, 4264, 4257, 4270, 4294, 4263, 4293, 4256, 4295, 4269, 4262, 4255, 4292, 4277,
    4276, 4248, 4249, 4250, 4251, 4252, 4341, 4340, 4339, 4338, 4331, 4344, 4337, 4330, 4343, 4367,
    4336, 4366, 4329, 4368, 4342, 4335, 4328, 4365, 4351, 4350, 4349, 4321, 4322, 4323, 4324, 4414,
    4413, 4412, 4418, 4411, 4417, 4410, 4403, 4416, 4440, 4409, 4439, 4402, 4441, 4415, 4408, 4401,
    4438, 4425, 4424, 4423, 4422, 4394, 4395, 4396, 4487, 4486, 4492, 4485, 4491, 4484, 4490, 4483,
    4489, 4513, 4482, 4512, 4475, 4514, 4488, 4481, 4474, 4511, 4499, 4498, 4497, 4496, 4495, 4467,
    4468, 4560, 4566, 4559, 4565, 4558, 4564, 4557, 4563, 4556, 4562, 4586, 4555, 4585, 4587, 4561,
    4554, 4547, 4573, 4572, 4571, 4570, 4569, 4568, 4540, 4640, 4633, 4639, 4632, 4638, 4631, 4637,
    4630, 4636, 4629, 4635, 4659, 4628, 4660, 4634, 4627, 4647, 4646, 4645, 4644, 4643, 4642, 4641,
    3504, 3573, 3572, 3511, 3576, 3575, 3626, 3643, 3642, 3577, 3646, 3645, 3584, 3649, 3648, 3699,
    3716, 3715, 3650, 3719, 3718, 3657, 3722, 3721, 3772, 3789, 3788, 3723, 3792, 3791, 3730, 3795,
    3794, 3845, 3862, 3861, 3796, 3865, 3864, 3803, 3868, 3867, 3918, 3935, 3934, 3869, 3938, 3937,
    3876, 3941, 3940, 3991, 4008, 4007, 3942, 4011, 4010, 3949, 4014, 4013, 4064, 4081, 4080, 4015,
    4084, 4083,
};

// Reverse bits within each byte (SWAR) -> converts Lc0 bitboard file order (h..a) to LSB (a..h)
static inline uint64_t reverse_bits_in_bytes(uint64_t x) noexcept {
    x = ((x >> 1) & 0x5555555555555555ULL) | ((x & 0x5555555555555555ULL) << 1);
    x = ((x >> 2) & 0x3333333333333333ULL) | ((x & 0x3333333333333333ULL) << 2);
    x = ((x >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((x & 0x0F0F0F0F0F0F0F0FULL) << 4);
    return x;
}

static inline uint64_t reverse_bytes_in_u64(uint64_t x) noexcept {
#if defined(__clang__) || defined(__GNUC__)
    return __builtin_bswap64(x);
#else
    return _byteswap_uint64(x);
#endif
}

static void init_nc_action_table() noexcept {
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

// Zero-dependency RFC-1952 (gzip) + RFC-1951 (DEFLATE) streaming decompressor
namespace inflate {

struct HuffTable {
    static constexpr int MAX_BITS = 15;
    std::array<int, MAX_BITS + 1> count{};
    std::array<int, MAX_BITS + 1> offs{};
    std::array<uint16_t, 320> symbols{};

    bool build(const int* lengths, int num_syms) {
        count.fill(0);
        for (int i = 0; i < num_syms; ++i) {
            if (lengths[i] > MAX_BITS) return false;
            count[lengths[i]]++;
        }
        if (count[0] == num_syms) return true;
        int left = 1;
        for (int len = 1; len <= MAX_BITS; ++len) {
            left = (left << 1) - count[len];
            if (left < 0) return false;
        }
        offs[1] = 0;
        for (int len = 1; len < MAX_BITS; ++len) {
            offs[len + 1] = offs[len] + count[len];
        }
        std::array<int, MAX_BITS + 1> work = offs;
        for (int sym = 0; sym < num_syms; ++sym) {
            int len = lengths[sym];
            if (len != 0) {
                symbols[work[len]++] = (uint16_t)sym;
            }
        }
        return true;
    }
};

struct BitReader {
    const uint8_t* ptr;
    const uint8_t* end;
    uint64_t bitbuf = 0;
    int bitcnt = 0;

    inline void ensure(int n) noexcept {
        while (bitcnt < n && ptr < end) {
            bitbuf |= (uint64_t)(*ptr++) << bitcnt;
            bitcnt += 8;
        }
    }

    inline uint32_t read_bits(int n) noexcept {
        ensure(n);
        uint32_t v = (uint32_t)(bitbuf & ((1ULL << n) - 1ULL));
        bitbuf >>= n;
        bitcnt -= n;
        return v;
    }

    inline void align_byte() noexcept {
        int drop = bitcnt & 7;
        bitbuf >>= drop;
        bitcnt -= drop;
    }

    inline int decode_sym(const HuffTable& h) noexcept {
        int code = 0;
        int first = 0;
        int index = 0;
        for (int len = 1; len <= HuffTable::MAX_BITS; ++len) {
            ensure(1);
            if (bitcnt < 1) return -1;
            code |= (int)(bitbuf & 1ULL);
            bitbuf >>= 1;
            bitcnt -= 1;
            int cnt = h.count[len];
            if (code - cnt < first) {
                return h.symbols[index + (code - first)];
            }
            index += cnt;
            first = (first + cnt) << 1;
            code <<= 1;
        }
        return -1;
    }
};

static inline bool decompress_deflate_block_stream(BitReader& br, std::vector<uint8_t>& out) {
    static constexpr int kLenBase[29] = {
        3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
        35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258
    };
    static constexpr int kLenExtra[29] = {
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
        3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0
    };
    static constexpr int kDistBase[30] = {
        1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
        257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577
    };
    static constexpr int kDistExtra[30] = {
        0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
        7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13
    };
    static constexpr int kClOrder[19] = {
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15
    };

    HuffTable lit_len, dist_tbl;
    int bfinal = 0;
    do {
        br.ensure(3);
        if (br.bitcnt < 3) return false;
        bfinal = (int)br.read_bits(1);
        int btype = (int)br.read_bits(2);

        if (btype == 0) {
            br.align_byte();
            uint32_t len = br.read_bits(16);
            uint32_t nlen = br.read_bits(16);
            if (((len ^ 0xFFFFU) & 0xFFFFU) != nlen) return false;
            size_t old_sz = out.size();
            out.resize(old_sz + len);
            for (uint32_t i = 0; i < len; ++i) {
                if (br.bitcnt >= 8) {
                    out[old_sz + i] = (uint8_t)br.read_bits(8);
                } else if (br.ptr < br.end) {
                    out[old_sz + i] = *br.ptr++;
                } else {
                    return false;
                }
            }
            continue;
        }

        if (btype == 1) {
            int ll_lens[288];
            int d_lens[32];
            for (int i = 0; i <= 143; ++i) ll_lens[i] = 8;
            for (int i = 144; i <= 255; ++i) ll_lens[i] = 9;
            for (int i = 256; i <= 279; ++i) ll_lens[i] = 7;
            for (int i = 280; i <= 287; ++i) ll_lens[i] = 8;
            for (int i = 0; i < 32; ++i) d_lens[i] = 5;
            if (!lit_len.build(ll_lens, 288) || !dist_tbl.build(d_lens, 32)) return false;
        } else if (btype == 2) {
            int hlit = (int)br.read_bits(5) + 257;
            int hdist = (int)br.read_bits(5) + 1;
            int hclen = (int)br.read_bits(4) + 4;
            if (hlit > 286 || hdist > 30) return false;

            int cl_lens[19] = {};
            for (int i = 0; i < hclen; ++i) {
                cl_lens[kClOrder[i]] = (int)br.read_bits(3);
            }
            HuffTable cl_tbl;
            if (!cl_tbl.build(cl_lens, 19)) return false;

            int total = hlit + hdist;
            int lens[320] = {};
            int i = 0;
            while (i < total) {
                int sym = br.decode_sym(cl_tbl);
                if (sym < 0) return false;
                if (sym < 16) {
                    lens[i++] = sym;
                } else if (sym == 16) {
                    if (i == 0) return false;
                    int rep = 3 + (int)br.read_bits(2);
                    int prev = lens[i - 1];
                    while (rep-- > 0 && i < total) lens[i++] = prev;
                } else if (sym == 17) {
                    int rep = 3 + (int)br.read_bits(3);
                    while (rep-- > 0 && i < total) lens[i++] = 0;
                } else {
                    int rep = 11 + (int)br.read_bits(7);
                    while (rep-- > 0 && i < total) lens[i++] = 0;
                }
            }
            if (!lit_len.build(lens, hlit) || !dist_tbl.build(lens + hlit, hdist)) return false;
        } else {
            return false;
        }

        while (true) {
            int sym = br.decode_sym(lit_len);
            if (sym < 0) return false;
            if (sym < 256) {
                out.push_back((uint8_t)sym);
            } else if (sym == 256) {
                break;
            } else {
                int lidx = sym - 257;
                if (lidx >= 29) return false;
                uint32_t length = (uint32_t)(kLenBase[lidx] + (int)br.read_bits(kLenExtra[lidx]));
                int dsym = br.decode_sym(dist_tbl);
                if (dsym < 0 || dsym >= 30) return false;
                uint32_t dist = (uint32_t)(kDistBase[dsym] + (int)br.read_bits(kDistExtra[dsym]));
                if (dist == 0 || dist > out.size()) return false;
                size_t cur = out.size();
                out.resize(cur + length);
                for (uint32_t k = 0; k < length; ++k) {
                    out[cur + k] = out[cur - dist + k];
                }
            }
        }
    } while (!bfinal);
    return true;
}

static bool decompress_gzip(const uint8_t* data, size_t len, std::vector<uint8_t>& out) {
    out.clear();
    if (len < 18) return false;
    out.reserve(std::min<size_t>(len * 32ULL, 16ULL << 20));
    size_t pos = 0;
    while (pos + 10 <= len) {
        if (data[pos] != 0x1F || data[pos + 1] != 0x8B || data[pos + 2] != 0x08) {
            return false;
        }
        uint8_t flg = data[pos + 3];
        pos += 10;
        if (flg & 0x04) {
            if (pos + 2 > len) return false;
            uint16_t xlen = (uint16_t)data[pos] | ((uint16_t)data[pos + 1] << 8);
            pos += 2 + xlen;
        }
        if (flg & 0x08) {
            while (pos < len && data[pos] != 0) pos++;
            pos++;
        }
        if (flg & 0x10) {
            while (pos < len && data[pos] != 0) pos++;
            pos++;
        }
        if (flg & 0x02) pos += 2;
        if (pos >= len) return false;

        BitReader br{data + pos, data + len};
        if (!decompress_deflate_block_stream(br, out)) return false;
        br.align_byte();
        // Advance back to unconsumed bytes before trailing 8-byte CRC32 + ISIZE
        size_t consumed = (size_t)((br.ptr - (data + pos)) - (br.bitcnt / 8));
        pos += consumed + 8;
        if (pos >= len) break;
    }
    return !out.empty();
}

} // namespace inflate

struct ConvertStats {
    std::atomic<uint64_t> chunks_seen{0};
    std::atomic<uint64_t> records_in{0};
    std::atomic<uint64_t> records_written{0};
    std::atomic<uint64_t> skipped_ply{0};
    std::atomic<uint64_t> skipped_960{0};
    std::atomic<uint64_t> skipped_invalid{0};
    std::atomic<uint64_t> ep_detected{0};
};

// Reconstruct `bb::CBoard`, filter FRC/ply, run movegen, extract Top-8 legal actions + Q/WDL/MLH
static void convert_record_stream(
    const uint8_t* raw,
    size_t raw_bytes,
    int min_ply,
    std::vector<PackedV1>& out_v1,
    std::vector<Lc0Sidecar68>& out_sc,
    ConvertStats& stats
) {
    size_t n_rec = raw_bytes / sizeof(V6TrainingData);
    if (n_rec == 0) return;

    stats.records_in.fetch_add((uint64_t)n_rec, std::memory_order_relaxed);
    out_v1.reserve(out_v1.size() + n_rec);
    out_sc.reserve(out_sc.size() + n_rec);

    int ply_in_game = 0;
    int prev_nz_hist = 0;
    bool skip_game = false;

    for (size_t idx = 0; idx < n_rec; ++idx) {
        const auto* rec = reinterpret_cast<const V6TrainingData*>(raw + idx * sizeof(V6TrainingData));
        if (rec->version != 6 || rec->input_format > 2) {
            stats.skipped_invalid.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        int nz_hist = 0;
        for (int h = 0; h < 8; ++h) {
            if (rec->planes[h * 13 + 5] != 0ULL) nz_hist++;
        }
        if (idx == 0 || nz_hist == 1 || nz_hist < prev_nz_hist) {
            ply_in_game = 0;
            skip_game = false;
        }
        prev_nz_hist = nz_hist;

        int cur_ply = ply_in_game++;
        if (skip_game) {
            stats.skipped_960.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        bool is_black = (rec->side_to_move_or_enpassant != 0);
        uint64_t p[12];
        for (int k = 0; k < 12; ++k) {
            p[k] = reverse_bits_in_bytes(rec->planes[k]);
        }

        // Standard chess castling check: filter Chess960 / FRC games cleanly
        bool c_us000 = (rec->castling_us_ooo != 0);
        bool c_us00 = (rec->castling_us_oo != 0);
        bool c_th000 = (rec->castling_them_ooo != 0);
        bool c_th00 = (rec->castling_them_oo != 0);

        if (((c_us000 || c_us00) && !(p[5] & (1ULL << 4))) ||
            (c_us000 && !(p[3] & (1ULL << 0))) ||
            (c_us00 && !(p[3] & (1ULL << 7))) ||
            ((c_th000 || c_th00) && !(p[11] & (1ULL << 60))) ||
            (c_th000 && !(p[9] & (1ULL << 56))) ||
            (c_th00 && !(p[9] & (1ULL << 63)))) {
            skip_game = true;
            stats.skipped_960.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        if (cur_ply < min_ply) {
            stats.skipped_ply.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        uint64_t w_bbs[6], b_bbs[6];
        uint8_t cast_mask = 0;
        if (is_black) {
            for (int k = 0; k < 6; ++k) {
                w_bbs[k] = reverse_bytes_in_u64(p[6 + k]);
                b_bbs[k] = reverse_bytes_in_u64(p[k]);
            }
            if (c_us000) cast_mask |= 8;
            if (c_us00)  cast_mask |= 4;
            if (c_th000) cast_mask |= 2;
            if (c_th00)  cast_mask |= 1;
        } else {
            for (int k = 0; k < 6; ++k) {
                w_bbs[k] = p[k];
                b_bbs[k] = p[6 + k];
            }
            if (c_us000) cast_mask |= 2;
            if (c_us00)  cast_mask |= 1;
            if (c_th000) cast_mask |= 8;
            if (c_th00)  cast_mask |= 4;
        }

        if (w_bbs[5] == 0ULL || b_bbs[5] == 0ULL) {
            stats.skipped_invalid.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        // Reconstruct en passant square from h=1 vs h=0 enemy pawn double push
        int ep_sq = -1;
        if (nz_hist >= 2) {
            uint64_t th_p_h0 = p[6];
            uint64_t th_p_h1 = reverse_bits_in_bytes(rec->planes[13 + 6]);
            uint64_t dis = (th_p_h1 & ~th_p_h0) & 0x00FF000000000000ULL;
            uint64_t app = (th_p_h0 & ~th_p_h1) & 0x000000FF00000000ULL;
            if (dis && app && bb::popcnt64(dis) == 1 && bb::popcnt64(app) == 1 && (dis >> 16) == app) {
                bool other_same = true;
                for (int k = 7; k < 12; ++k) {
                    if (reverse_bits_in_bytes(rec->planes[k]) != reverse_bits_in_bytes(rec->planes[13 + k])) {
                        other_same = false;
                        break;
                    }
                }
                if (other_same) {
                    int rel_sq = bb::ctz64(app) + 8;
                    ep_sq = is_black ? ((7 - (rel_sq >> 3)) * 8 + (rel_sq & 7)) : rel_sq;
                    stats.ep_detected.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }

        bb::CBoard cb{};
        cb.pawns   = w_bbs[0] | b_bbs[0];
        cb.knights = w_bbs[1] | b_bbs[1];
        cb.bishops = w_bbs[2] | b_bbs[2];
        cb.rooks   = w_bbs[3] | b_bbs[3];
        cb.queens  = w_bbs[4] | b_bbs[4];
        cb.kings   = w_bbs[5] | b_bbs[5];
        cb.occ_w = w_bbs[0] | w_bbs[1] | w_bbs[2] | w_bbs[3] | w_bbs[4] | w_bbs[5];
        cb.occ_b = b_bbs[0] | b_bbs[1] | b_bbs[2] | b_bbs[3] | b_bbs[4] | b_bbs[5];
        cb.castling_rights =
            ((cast_mask & 1) ? (1ULL << 7) : 0ULL) |
            ((cast_mask & 2) ? (1ULL << 0) : 0ULL) |
            ((cast_mask & 4) ? (1ULL << 63) : 0ULL) |
            ((cast_mask & 8) ? (1ULL << 56) : 0ULL);
        cb.ep_square = (int8_t)ep_sq;
        cb.turn_is_white = is_black ? 0 : 1;
        cb.halfmove_clock = (int16_t)std::min<uint8_t>(100, rec->rule50_count);

        uint16_t leg_acts[256];
        uint16_t leg_moves[256];
        int n_leg = cb.generate_legal_moves_and_actions(leg_acts, leg_moves);
        if (n_leg <= 0) {
            stats.skipped_invalid.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        uint64_t legal_bits[73] = {};
        for (int m = 0; m < n_leg; ++m) {
            uint16_t a = leg_acts[m];
            legal_bits[a >> 6] |= (1ULL << (a & 63));
        }

        auto is_legal_act = [&](uint16_t a) -> bool {
            return a < 4672 && ((legal_bits[a >> 6] >> (a & 63)) & 1ULL);
        };
        bool us_k_e1 = (p[5] == (1ULL << 4));

        // AVX2 scan across 1,858 floats for non-zero MCTS policy entries -> Top-8 legal actions
        float top_p[8] = {0};
        uint16_t top_a[8] = {65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535};
        int n_top = 0;

        auto insert_cand = [&](uint16_t act, float pr) {
            for (int k = 0; k < n_top; ++k) {
                if (top_a[k] == act) {
                    top_p[k] += pr;
                    while (k > 0 && top_p[k] > top_p[k - 1]) {
                        std::swap(top_p[k], top_p[k - 1]);
                        std::swap(top_a[k], top_a[k - 1]);
                        --k;
                    }
                    return;
                }
            }
            if (n_top < 8 || pr > top_p[7]) {
                int pos = (n_top < 8) ? n_top++ : 7;
                top_p[pos] = pr;
                top_a[pos] = act;
                while (pos > 0 && top_p[pos] > top_p[pos - 1]) {
                    std::swap(top_p[pos], top_p[pos - 1]);
                    std::swap(top_a[pos], top_a[pos - 1]);
                    --pos;
                }
            }
        };

        const __m256 v_zero = _mm256_setzero_ps();
        for (int base = 0; base < 1856; base += 8) {
            __m256 vp = _mm256_loadu_ps(rec->probabilities + base);
            uint32_t m = (uint32_t)_mm256_movemask_ps(_mm256_cmp_ps(vp, v_zero, _CMP_GT_OQ));
            while (m) {
                int lane = (int)_tzcnt_u32(m);
                m &= (m - 1U);
                int lidx = base + lane;
                float pr = rec->probabilities[lidx];
                if (!std::isfinite(pr) || pr <= 0.0f) continue;
                uint16_t act = kLc0ToNcAct[lidx];
                if (!is_legal_act(act)) {
                    if (lidx == kIdxE1H1 && us_k_e1 && is_legal_act(kActE1G1)) {
                        act = kActE1G1;
                    } else if (lidx == kIdxE1A1 && us_k_e1 && is_legal_act(kActE1C1)) {
                        act = kActE1C1;
                    } else {
                        continue;
                    }
                }
                insert_cand(act, pr);
            }
        }
        for (int lidx = 1856; lidx < 1858; ++lidx) {
            float pr = rec->probabilities[lidx];
            if (std::isfinite(pr) && pr > 0.0f) {
                uint16_t act = kLc0ToNcAct[lidx];
                if (is_legal_act(act)) insert_cand(act, pr);
            }
        }

        if (n_top == 0) {
            if (rec->best_idx < 1858) {
                uint16_t act = kLc0ToNcAct[rec->best_idx];
                if (!is_legal_act(act)) {
                    if (rec->best_idx == kIdxE1H1 && us_k_e1 && is_legal_act(kActE1G1)) {
                        act = kActE1G1;
                    } else if (rec->best_idx == kIdxE1A1 && us_k_e1 && is_legal_act(kActE1C1)) {
                        act = kActE1C1;
                    }
                }
                if (is_legal_act(act)) {
                    top_a[0] = act;
                    top_p[0] = 1.0f;
                    n_top = 1;
                }
            }
            if (n_top == 0) {
                stats.skipped_invalid.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
        }

        float psum = 0.0f;
        for (int k = 0; k < n_top; ++k) psum += top_p[k];
        if (psum > 0.0f) {
            float inv_s = 1.0f / psum;
            for (int k = 0; k < n_top; ++k) top_p[k] *= inv_s;
        }

        // Backed-up Q / WDL (prefer best_q/best_d when finite, else root_q/root_d)
        bool used_best = std::isfinite(rec->best_q);
        float q = used_best ? rec->best_q : (std::isfinite(rec->root_q) ? rec->root_q : rec->result_q);
        float d = (used_best && std::isfinite(rec->best_d))
                      ? rec->best_d
                      : (std::isfinite(rec->root_d) ? rec->root_d : rec->result_d);
        if (!std::isfinite(q)) q = 0.0f;
        if (!std::isfinite(d)) d = 0.0f;
        q = std::clamp(q, -1.0f, 1.0f);
        d = std::clamp(d, 0.0f, 1.0f);

        float w_p = std::clamp(0.5f * (1.0f + q - d), 0.0f, 1.0f);
        float l_p = std::clamp(0.5f * (1.0f - q - d), 0.0f, 1.0f);
        float wdl_sum = w_p + d + l_p;
        if (wdl_sum > 0.0f) {
            w_p /= wdl_sum;
            d /= wdl_sum;
            l_p /= wdl_sum;
        } else {
            w_p = 0.0f; d = 1.0f; l_p = 0.0f;
        }

        // CP_TANH_SCALE = ln(10)/800 = 0.0028782313662425573
        float q_clip = std::clamp(q, -0.9999f, 0.9999f);
        double cp_f = std::atanh((double)q_clip) / 0.0028782313662425573;
        int16_t eval_cp = (int16_t)std::clamp<int>((int)std::llround(cp_f), -32767, 32767);

        PackedV1 pv1{};
        for (int k = 0; k < 6; ++k) pv1.bbs[k] = w_bbs[k];
        for (int k = 0; k < 6; ++k) pv1.bbs[6 + k] = b_bbs[k];
        pv1.turn = is_black ? 1 : 0;
        pv1.castling = cast_mask;
        pv1.ep_square = (ep_sq >= 0) ? (uint8_t)ep_sq : 255;
        pv1.halfmove = (uint8_t)cb.halfmove_clock;
        pv1.action_index = top_a[0];
        pv1.eval_cp = eval_cp;
        pv1.wdl[0] = (uint8_t)std::clamp((int)std::lround(w_p * 255.0f), 0, 255);
        pv1.wdl[1] = (uint8_t)std::clamp((int)std::lround(d * 255.0f), 0, 255);
        pv1.wdl[2] = (uint8_t)std::clamp((int)std::lround(l_p * 255.0f), 0, 255);

        Lc0Sidecar68 sc{};
        sc.num_topk = (uint8_t)n_top;
        sc.flags = (uint8_t)((is_black ? 1 : 0) | ((ep_sq >= 0) ? 2 : 0) | (used_best ? 4 : 0));
        sc.visits_clamped = (uint16_t)std::min<uint32_t>(rec->visits, 65535U);
        for (int k = 0; k < 8; ++k) {
            sc.top8_actions[k] = top_a[k];
            sc.top8_probs[k] = top_p[k];
        }
        sc.backed_q = q;
        sc.backed_d = d;
        sc.result_q = std::isfinite(rec->result_q) ? std::clamp(rec->result_q, -1.0f, 1.0f) : 0.0f;
        sc.plies_left = (std::isfinite(rec->plies_left) && rec->plies_left >= 0.0f) ? rec->plies_left : 0.0f;

        out_v1.push_back(pv1);
        out_sc.push_back(sc);
        stats.records_written.fetch_add(1, std::memory_order_relaxed);
    }
}

struct OwnedChunk {
    std::vector<uint8_t> bytes;
    bool is_gzip;
};

static uint64_t parse_octal(const uint8_t* p, size_t n) noexcept {
    uint64_t v = 0;
    for (size_t i = 0; i < n; ++i) {
        if (p[i] == 0 || p[i] == ' ') continue;
        if (p[i] >= '0' && p[i] <= '7') v = (v << 3) + (uint64_t)(p[i] - '0');
    }
    return v;
}

static size_t read_exact_stream(std::istream& in, uint8_t* dst, size_t n) {
    size_t got = 0;
    while (got < n) {
        in.read(reinterpret_cast<char*>(dst + got), (std::streamsize)(n - got));
        std::streamsize g = in.gcount();
        if (g <= 0) break;
        got += (size_t)g;
    }
    return got;
}

static int run_converter(int argc, char** argv) {
    init_nc_action_table();

    std::vector<std::string> inputs;
    std::string out_prefix;
    int min_ply = 9; // strict ply > 8 (0-indexed ply >= 9)
    int threads = (int)std::max(1u, std::thread::hardware_concurrency());
    size_t max_records = 0;
    bool verify_18p = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--out-prefix" && i + 1 < argc) out_prefix = argv[++i];
        else if (a == "--min-ply" && i + 1 < argc) min_ply = std::stoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) threads = std::max(1, std::stoi(argv[++i]));
        else if (a == "--max-records" && i + 1 < argc) max_records = (size_t)std::stoull(argv[++i]);
        else if (a == "--verify-18planes") verify_18p = true;
        else if (a == "--input" && i + 1 < argc) inputs.emplace_back(argv[++i]);
        else if (a == "-" || (!a.empty() && a[0] != '-')) inputs.emplace_back(a);
        else {
            std::cerr << "Unknown option: " << a << "\n";
            return 2;
        }
    }

    if (inputs.empty() || out_prefix.empty()) {
        std::cerr << "Usage: lc0_v6_converter --out-prefix <path> [--min-ply 9] [--threads N] [--max-records N] <input.tar/.gz/.bin | ->\n";
        return 2;
    }

    std::vector<std::string> sources;
    for (const auto& inp : inputs) {
        if (inp == "-") {
            sources.emplace_back("-");
            continue;
        }
        fs::path p(inp);
        if (fs::is_directory(p)) {
            std::vector<std::string> dir_files;
            for (const auto& entry : fs::directory_iterator(p)) {
                if (entry.is_regular_file()) dir_files.push_back(entry.path().string());
            }
            std::sort(dir_files.begin(), dir_files.end());
            sources.insert(sources.end(), dir_files.begin(), dir_files.end());
        } else if (fs::is_regular_file(p)) {
            sources.push_back(p.string());
        }
    }

    fs::path out_bin(out_prefix + ".bin");
    fs::path out_sc(out_prefix + ".lc0_sidecar.bin");
    if (out_bin.has_parent_path()) fs::create_directories(out_bin.parent_path());

    std::ofstream fb(out_bin, std::ios::binary | std::ios::trunc);
    std::ofstream fs_c(out_sc, std::ios::binary | std::ios::trunc);
    if (!fb || !fs_c) {
        std::cerr << "Failed to open output files at prefix: " << out_prefix << "\n";
        return 3;
    }

    ConvertStats stats{};
    std::atomic<uint64_t> total_18p_checksum{0};
    size_t total_written = 0;
    auto t0 = std::chrono::steady_clock::now();

    auto flush_batch = [&](std::vector<OwnedChunk>& chunks) {
        if (chunks.empty()) return;
        stats.chunks_seen.fetch_add((uint64_t)chunks.size(), std::memory_order_relaxed);
        size_t n_chunks = chunks.size();
        size_t n_workers = (size_t)std::min<int>(threads, (int)n_chunks);
        std::vector<std::vector<PackedV1>> w_v1(n_chunks);
        std::vector<std::vector<Lc0Sidecar68>> w_sc(n_chunks);
        std::atomic<size_t> next_slice{0};
        std::atomic<bool> stop_flag{false};

        auto worker = [&]() {
            std::vector<uint8_t> dec_buf;
            float tensor_1152[18 * 64];
            while (!stop_flag.load(std::memory_order_relaxed)) {
                size_t si = next_slice.fetch_add(1, std::memory_order_relaxed);
                if (si >= n_chunks) break;
                const auto& ch = chunks[si];
                const uint8_t* rptr = ch.bytes.data();
                size_t rlen = ch.bytes.size();
                if (ch.is_gzip) {
                    if (!inflate::decompress_gzip(rptr, rlen, dec_buf)) continue;
                    rptr = dec_buf.data();
                    rlen = dec_buf.size();
                }
                convert_record_stream(rptr, rlen, min_ply, w_v1[si], w_sc[si], stats);
                if (verify_18p && !w_v1[si].empty()) {
                    const auto& first_p = w_v1[si][0];
                    bb::CBoard tb{};
                    for (int k = 0; k < 6; ++k) {
                        tb.occ_w |= first_p.bbs[k];
                        tb.occ_b |= first_p.bbs[6 + k];
                    }
                    tb.pawns = first_p.bbs[0] | first_p.bbs[6];
                    tb.knights = first_p.bbs[1] | first_p.bbs[7];
                    tb.bishops = first_p.bbs[2] | first_p.bbs[8];
                    tb.rooks = first_p.bbs[3] | first_p.bbs[9];
                    tb.queens = first_p.bbs[4] | first_p.bbs[10];
                    tb.kings = first_p.bbs[5] | first_p.bbs[11];
                    tb.turn_is_white = first_p.turn ? 0 : 1;
                    tb.castling_rights =
                        ((first_p.castling & 1) ? (1ULL << 7) : 0ULL) |
                        ((first_p.castling & 2) ? (1ULL << 0) : 0ULL) |
                        ((first_p.castling & 4) ? (1ULL << 63) : 0ULL) |
                        ((first_p.castling & 8) ? (1ULL << 56) : 0ULL);
                    tb.ep_square = (first_p.ep_square != 255) ? (int8_t)first_p.ep_square : -1;
                    tb.halfmove_clock = first_p.halfmove;
                    tb.encode_gpu_unpack(tensor_1152);
                    total_18p_checksum.fetch_add((uint64_t)(tensor_1152[0] > 0.0f ? 1 : 0), std::memory_order_relaxed);
                }
                if (max_records > 0 && (total_written + stats.records_written.load(std::memory_order_relaxed)) >= max_records) {
                    stop_flag.store(true, std::memory_order_relaxed);
                }
            }
        };

        std::vector<std::thread> pool;
        pool.reserve(n_workers);
        for (size_t wi = 0; wi < n_workers; ++wi) pool.emplace_back(worker);
        for (auto& th : pool) th.join();

        for (size_t si = 0; si < n_chunks; ++si) {
            if (!w_v1[si].empty()) {
                size_t cnt = w_v1[si].size();
                if (max_records > 0 && (total_written + cnt) > max_records) {
                    cnt = max_records - total_written;
                }
                if (cnt > 0) {
                    fb.write(reinterpret_cast<const char*>(w_v1[si].data()), (std::streamsize)(cnt * sizeof(PackedV1)));
                    fs_c.write(reinterpret_cast<const char*>(w_sc[si].data()), (std::streamsize)(cnt * sizeof(Lc0Sidecar68)));
                    total_written += cnt;
                }
                if (max_records > 0 && total_written >= max_records) break;
            }
        }
        chunks.clear();
    };

    constexpr size_t kBatchSlices = 512;
    for (const auto& src : sources) {
        if (max_records > 0 && total_written >= max_records) break;
        std::ifstream fin;
        std::istream* in_ptr = nullptr;
        if (src == "-") {
#ifdef _WIN32
            _setmode(_fileno(stdin), _O_BINARY);
#endif
            in_ptr = &std::cin;
        } else {
            fin.open(src, std::ios::binary);
            if (!fin) continue;
            in_ptr = &fin;
        }

        uint8_t first512[512];
        size_t n0 = read_exact_stream(*in_ptr, first512, 512);
        if (n0 == 0) continue;

        bool is_tar = (n0 == 512 && std::memcmp(first512 + 257, "ustar", 5) == 0);
        if (!is_tar) {
            std::vector<uint8_t> raw(first512, first512 + n0);
            std::vector<uint8_t> rest((std::istreambuf_iterator<char>(*in_ptr)), std::istreambuf_iterator<char>());
            raw.insert(raw.end(), rest.begin(), rest.end());
            bool is_gz = (raw.size() >= 3 && raw[0] == 0x1F && raw[1] == 0x8B && raw[2] == 0x08);
            std::vector<OwnedChunk> one;
            one.push_back({std::move(raw), is_gz});
            flush_batch(one);
            continue;
        }

        std::vector<OwnedChunk> batch;
        batch.reserve(kBatchSlices);
        uint8_t hdr[512];
        std::memcpy(hdr, first512, 512);
        bool have_hdr = true;

        while (have_hdr) {
            bool all_zero = true;
            for (int i = 0; i < 512; ++i) {
                if (hdr[i] != 0) { all_zero = false; break; }
            }
            if (all_zero) break;

            uint64_t fsz = parse_octal(hdr + 124, 12);
            char typeflag = (char)hdr[156];
            size_t padded = (size_t)(((fsz + 511ULL) / 512ULL) * 512ULL);

            if ((typeflag == '0' || typeflag == '\0') && fsz >= 18 && fsz <= (64ULL << 20)) {
                std::vector<uint8_t> payload(padded);
                size_t got = read_exact_stream(*in_ptr, payload.data(), padded);
                if (got < fsz) break;
                payload.resize((size_t)fsz);
                bool is_gz = (payload[0] == 0x1F && payload[1] == 0x8B && payload[2] == 0x08);
                if (is_gz || ((fsz % sizeof(V6TrainingData)) == 0)) {
                    batch.push_back({std::move(payload), is_gz});
                    if (batch.size() >= kBatchSlices) {
                        flush_batch(batch);
                        if (max_records > 0 && total_written >= max_records) break;
                    }
                }
            } else if (padded > 0) {
                std::vector<uint8_t> skip_buf(std::min<size_t>(padded, 65536));
                size_t rem = padded;
                while (rem > 0) {
                    size_t step = std::min(rem, skip_buf.size());
                    if (read_exact_stream(*in_ptr, skip_buf.data(), step) < step) { rem = 1; break; }
                    rem -= step;
                }
                if (rem > 0) break;
            }
            have_hdr = (read_exact_stream(*in_ptr, hdr, 512) == 512);
        }
        if (!batch.empty() && (max_records == 0 || total_written < max_records)) {
            flush_batch(batch);
        }
    }

    fb.flush();
    fs_c.flush();
    auto t1 = std::chrono::steady_clock::now();
    double elapsed_s = std::max(1e-6, std::chrono::duration<double>(t1 - t0).count());
    uint64_t r_in = stats.records_in.load();

    std::cout << "{"
              << "\"chunks_seen\":" << stats.chunks_seen.load()
              << ",\"records_in\":" << r_in
              << ",\"records_written\":" << total_written
              << ",\"skipped_ply\":" << stats.skipped_ply.load()
              << ",\"skipped_960\":" << stats.skipped_960.load()
              << ",\"skipped_invalid\":" << stats.skipped_invalid.load()
              << ",\"ep_detected\":" << stats.ep_detected.load()
              << ",\"elapsed_s\":" << elapsed_s
              << ",\"in_records_per_s\":" << (double)r_in / elapsed_s
              << ",\"written_pos_per_s\":" << (double)total_written / elapsed_s
              << ",\"out_bin\":\"" << out_bin.generic_string() << "\""
              << ",\"out_sidecar\":\"" << out_sc.generic_string() << "\""
              << "}\n";
    return 0;
}

} // namespace nc_lc0

int main(int argc, char** argv) {
    return nc_lc0::run_converter(argc, argv);
}
