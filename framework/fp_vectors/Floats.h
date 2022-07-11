/*
 * Copyright 2022 Intel Corporation.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __FP_VECTORS_H
#define __FP_VECTORS_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
#include <limits>
#endif

// GCC supports _Float16 on x86 and __fp16 on AArch64, in both cases it
// only supports IEEE-754 format.
// https://gcc.gnu.org/onlinedocs/gcc/Half-Precision.html
#ifdef SANDSTONE_FP16_TYPE
typedef SANDSTONE_FP16_TYPE fp16_t;
#else
#  define SANDSTONE_FLOAT16_EMULATED
#endif

#ifndef __FLT128_DECIMAL_DIG__
#  define __FLT128_DECIMAL_DIG__ 36
#endif
#ifndef __FLT128_DENORM_MIN__
#  define __FLT128_DENORM_MIN__ 6.47517511943802511092443895822764655e-4966F128
#endif

#define FLOAT16_EXPONENT_MASK  0x1fu
#define BFLOAT16_EXPONENT_MASK 0xffu
#define FLOAT32_EXPONENT_MASK  0xffu
#define FLOAT64_EXPONENT_MASK  0x7ffu
#define FLOAT80_EXPONENT_MASK  0x7fffu

#define FLOAT16_INFINITY_EXPONENT  0x1fu
#define BFLOAT16_INFINITY_EXPONENT 0xffu
#define FLOAT32_INFINITY_EXPONENT  0xffu
#define FLOAT64_INFINITY_EXPONENT  0x7ffu
#define FLOAT80_INFINITY_EXPONENT  0x7fffu

#define FLOAT16_NAN_EXPONENT  0x1fu
#define BFLOAT16_NAN_EXPONENT 0xffu
#define FLOAT32_NAN_EXPONENT  0xffu
#define FLOAT64_NAN_EXPONENT  0x7ffu
#define FLOAT80_NAN_EXPONENT  0x7fffu

#define FLOAT16_DENORM_EXPONENT  0x00
#define BFLOAT16_DENORM_EXPONENT 0x00
#define FLOAT32_DENORM_EXPONENT  0x00
#define FLOAT64_DENORM_EXPONENT  0x00
#define FLOAT80_DENORM_EXPONENT  0x00

#define FLOAT16_EXPONENT_BIAS  0x0fu
#define BFLOAT16_EXPONENT_BIAS 0x7fu
#define FLOAT32_EXPONENT_BIAS  0x7fu
#define FLOAT64_EXPONENT_BIAS  0x3ffu
#define FLOAT80_EXPONENT_BIAS  0x3fffu

#define FLOAT16_MANTISSA_MASK  0x3ffu
#define BFLOAT16_MANTISSA_MASK 0x7fu
#define FLOAT32_MANTISSA_MASK  0x7fffffu
#define FLOAT64_MANTISSA_MASK  0xfffffffffffffu
#define FLOAT80_MANTISSA_MASK  0x7fffffffffffffffu

#define FLOAT16_MANTISSA_QUIET_NAN_MASK  0x200u
#define BFLOAT16_MANTISSA_QUIET_NAN_MASK 0x40u
#define FLOAT32_MANTISSA_QUIET_NAN_MASK  0x400000u
#define FLOAT64_MANTISSA_QUIET_NAN_MASK  0x8000000000000u
#define FLOAT80_MANTISSA_QUIET_NAN_MASK  0x4000000000000000u

#define FP16_SIGN_BITS        1
#define FP16_EXPONENT_BITS    5
#define FP16_MANTISSA_BITS    10
#define FP16_QUIET_BITS       1

#define BFLT16_SIGN_BITS      1
#define BFLT16_EXPONENT_BITS  8
#define BFLT16_MANTISSA_BITS  7
#define BFLT16_QUIET_BITS     1

#define FLOAT32_SIGN_BITS     1
#define FLOAT32_EXPONENT_BITS 8
#define FLOAT32_MANTISSA_BITS 23
#define FLOAT32_QUIET_BITS    1

#define FLOAT64_SIGN_BITS     1
#define FLOAT64_EXPONENT_BITS 11
#define FLOAT64_MANTISSA_BITS 52
#define FLOAT64_QUIET_BITS    1

#define FLOAT80_SIGN_BITS     1
#define FLOAT80_EXPONENT_BITS 15
#define FLOAT80_JBIT_BITS     1
#define FLOAT80_MANTISSA_BITS 63
#define FLOAT80_QUIET_BITS    1

#define FP16_DECIMAL_DIG        5
#define FP16_DENORM_MIN         5.96046447753906250000000000000000000e-8
#define FP16_DIG                3
#define FP16_EPSILON            9.76562500000000000000000000000000000e-4
#define FP16_HAS_DENORM         1
#define FP16_HAS_INFINITY       1
#define FP16_HAS_QUIET_NAN      1
#define FP16_MANT_DIG           11
#define FP16_MAX_10_EXP         4
#define FP16_MAX                6.55040000000000000000000000000000000e+4
#define FP16_MAX_EXP            16
#define FP16_MIN_10_EXP         (-4)
#define FP16_MIN                6.10351562500000000000000000000000000e-5
#define FP16_MIN_EXP            (-13)
#define FP16_NORM_MAX           6.55040000000000000000000000000000000e+4

#define BFLT16_DECIMAL_DIG      3
#define BFLT16_DENORM_MIN       (0x1p-133)
#define BFLT16_DIG              2
#define BFLT16_EPSILON          (FLT_EPSILON * 65536)
#define BFLT16_HAS_DENORM       1
#define BFLT16_HAS_INFINITY     1
#define BFLT16_HAS_QUIET_NAN    1
#define BFLT16_MANT_DIG         (FLT_MANT_DIG - 16)
#define BFLT16_MAX_10_EXP       FLT_MAX_10_EXP
#define BFLT16_MAX_EXP          FLT_MAX_EXP
#define BFLT16_MAX              (0x1.fep+127f)
#define BFLT16_MIN_10_EXP       FLT_MIN_10_EXP
#define BFLT16_MIN_EXP          FLT_MIN_EXP
#define BFLT16_MIN              (0x1p-126f)
#define BFLT16_NORM_MAX         BFLT16_MAX

#ifdef __cplusplus
extern "C" {
#endif

struct Float16
{
    union {
        struct __attribute__((packed)) {
            uint16_t mantissa : FP16_MANTISSA_BITS;
            uint16_t exponent : FP16_EXPONENT_BITS;
            uint16_t sign     : FP16_SIGN_BITS;
        };
        struct __attribute__((packed)) {
            uint16_t payload  : FP16_MANTISSA_BITS - FP16_QUIET_BITS;
            uint16_t quiet    : FP16_QUIET_BITS;
            uint16_t exponent : FP16_EXPONENT_BITS;
            uint16_t sign     : FP16_SIGN_BITS;
        } as_nan;

        uint16_t as_hex;
#ifdef SANDSTONE_FP16_TYPE
        fp16_t as_float;
#endif
        uint16_t payload;
    };

#ifdef __cplusplus
    inline Float16() = default;
    inline Float16(float f);

    constexpr inline Float16(uint16_t s, uint16_t e, uint16_t m): mantissa(m), exponent(e), sign(s) { }

    static constexpr int digits = FP16_MANT_DIG;
    static constexpr int digits10 = FP16_DIG;
    static constexpr int max_digits10 = 6;  // log2(digits)
    static constexpr int min_exponent = FP16_MIN_EXP;
    static constexpr int min_exponent10 = FP16_MIN_10_EXP;
    static constexpr int max_exponent = FP16_MAX_EXP;
    static constexpr int max_exponent10 = FP16_MAX_10_EXP;

    static constexpr bool radix = 2;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = FP16_HAS_INFINITY;
    static constexpr bool has_quiet_NaN = FP16_HAS_QUIET_NAN;
    static constexpr bool has_signaling_NaN = has_quiet_NaN;
    static constexpr std::float_denorm_style has_denorm = std::denorm_present;
    static constexpr bool has_denorm_loss = false;
    static constexpr bool is_iec559 = true;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;
    static constexpr std::float_round_style round_style =
            std::round_toward_zero;   // unlike std::numeric_limits<float>::round_style

    static constexpr Float16 min()              { return Float16(Holder{0x0400}); }
    static constexpr Float16 max()              { return Float16(Holder{0x7bff}); }
    static constexpr Float16 lowest()           { return Float16(Holder{0xfbff}); }
    static constexpr Float16 denorm_min()       { return Float16(Holder{0x0001}); }
    static constexpr Float16 epsilon()          { return Float16(Holder{0x1400}); }
    static constexpr Float16 round_error()      { return Float16(Holder{0x3800}); }
    static constexpr Float16 infinity()         { return Float16(Holder{0x7c00}); }
    static constexpr Float16 neg_infinity()     { return Float16(Holder{0xfc00}); }
    static constexpr Float16 quiet_NaN()        { return Float16(Holder{0x7e00}); }
    static constexpr Float16 signaling_NaN()    { return Float16(Holder{0x7d00}); }

    constexpr inline bool     is_negative() const         { return sign != 0; }
    constexpr inline bool     is_zero() const             { return (exponent == FLOAT16_DENORM_EXPONENT) && (mantissa == 0); }
    constexpr inline bool     is_denormal() const         { return (exponent == FLOAT16_DENORM_EXPONENT) && (mantissa != 0); }

    // NaNs
    constexpr inline bool     is_general_nan() const      { return exponent == FLOAT16_NAN_EXPONENT; }
    constexpr inline bool     is_inf() const              { return is_general_nan() && (mantissa == 0); }
    constexpr inline bool     is_nan() const              { return is_general_nan() && (mantissa != 0); }
    constexpr inline bool     is_snan() const             { return is_nan() && ((mantissa & FLOAT16_MANTISSA_QUIET_NAN_MASK) == 0); }
    constexpr inline bool     is_qnan() const             { return is_nan() && ((mantissa & FLOAT16_MANTISSA_QUIET_NAN_MASK) != 0); }

    constexpr inline uint16_t get_nan_payload() const     { return mantissa & (~FLOAT16_MANTISSA_QUIET_NAN_MASK); }

private:
    struct Holder { uint16_t payload; };
    explicit constexpr Float16(Holder h) : as_hex(h.payload) {}
#endif
};
typedef struct Float16 Float16;

// C interface
static inline bool     Float16_is_negative(Float16 f)         { return f.sign != 0; }
static inline bool     Float16_is_zero(Float16 f)             { return (f.exponent == FLOAT16_DENORM_EXPONENT) && (f.mantissa == 0); }
static inline bool     Float16_is_denormal(Float16 f)         { return (f.exponent == FLOAT16_DENORM_EXPONENT) && (f.mantissa != 0); }

static inline bool     Float16_is_general_nan(Float16 f)      { return f.exponent == FLOAT16_NAN_EXPONENT; }
static inline bool     Float16_is_inf(Float16 f)              { return Float16_is_general_nan(f) && (f.mantissa == 0); }
static inline bool     Float16_is_nan(Float16 f)              { return Float16_is_general_nan(f) && (f.mantissa != 0); }
static inline bool     Float16_is_snan(Float16 f)             { return Float16_is_nan(f) && (f.as_nan.quiet == 0); }
static inline bool     Float16_is_qnan(Float16 f)             { return Float16_is_nan(f) && (f.as_nan.quiet != 0); }

static inline uint16_t Float16_get_nan_payload(Float16 f)     { return f.as_nan.payload; }


struct BFloat16
{
    union {
        struct __attribute__((packed)) {
            uint16_t mantissa : BFLT16_MANTISSA_BITS;
            uint16_t exponent : BFLT16_EXPONENT_BITS;
            uint16_t sign     : BFLT16_SIGN_BITS;
        };
        struct __attribute__((packed)) {
            uint16_t payload  : BFLT16_MANTISSA_BITS - BFLT16_QUIET_BITS;
            uint16_t quiet    : BFLT16_QUIET_BITS;
            uint16_t exponent : BFLT16_EXPONENT_BITS;
            uint16_t sign     : BFLT16_SIGN_BITS;
        } as_nan;
        uint16_t as_hex;
        uint16_t payload;
    };

#ifdef __cplusplus
    inline BFloat16() = default;
    inline BFloat16(float f);
    constexpr inline BFloat16(uint16_t s, uint16_t e, uint16_t m): mantissa(m), exponent(e), sign(s) { }

    // same API as std::numeric_limits:
    static constexpr int digits = BFLT16_MANT_DIG;
    static constexpr int digits10 = BFLT16_DIG;
    static constexpr int max_digits10 = 3;  // log2(digits)
    static constexpr int min_exponent = std::numeric_limits<float>::min_exponent;
    static constexpr int min_exponent10 = std::numeric_limits<float>::min_exponent10;
    static constexpr int max_exponent = std::numeric_limits<float>::max_exponent;
    static constexpr int max_exponent10 = std::numeric_limits<float>::max_exponent10;

    static constexpr bool radix = 2;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = std::numeric_limits<float>::has_infinity;
    static constexpr bool has_quiet_NaN = std::numeric_limits<float>::has_quiet_NaN;
    static constexpr bool has_signaling_NaN = has_quiet_NaN;
    static constexpr std::float_denorm_style has_denorm = std::denorm_present;
    static constexpr bool has_denorm_loss = false;
    static constexpr bool is_iec559 = true;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;
    static constexpr std::float_round_style round_style =
            std::round_toward_zero;   // unlike std::numeric_limits<float>::round_style

    static constexpr BFloat16 max()           { return BFloat16(Holder{0x7f7f}); }
    static constexpr BFloat16 min()           { return BFloat16(Holder{0x0080}); }
    static constexpr BFloat16 lowest()        { return BFloat16(Holder{0xff7f}); }
    static constexpr BFloat16 denorm_min()    { return BFloat16(Holder{0x0001}); }
    static constexpr BFloat16 epsilon()       { return BFloat16(Holder{0x3c00}); }
    static constexpr BFloat16 round_error()   { return BFloat16(Holder{0x3f00}); }
    static constexpr BFloat16 infinity()      { return BFloat16(Holder{0x7f80}); }
    static constexpr BFloat16 neg_infinity()  { return BFloat16(Holder{0xff80}); }
    static constexpr BFloat16 quiet_NaN()     { return BFloat16(Holder{0x7fc0}); }
    static constexpr BFloat16 signaling_NaN() { return BFloat16(Holder{0x7fa0}); }

    // extra
    static constexpr float epsilon_v()        { return std::numeric_limits<float>::epsilon() * 65536; }

    constexpr inline bool     is_negative() const       { return sign != 0; }
    constexpr inline bool     is_zero() const           { return (exponent == BFLOAT16_DENORM_EXPONENT) && (mantissa == 0); }
    constexpr inline bool     is_denormal() const       { return (exponent == BFLOAT16_DENORM_EXPONENT) && (mantissa != 0); }

    // NaNs
    constexpr inline bool     is_general_nan() const    { return exponent == FLOAT32_NAN_EXPONENT; }
    constexpr inline bool     is_inf() const            { return is_general_nan() && (mantissa == 0); }
    constexpr inline bool     is_nan() const            { return is_general_nan() && (mantissa != 0); }
    constexpr inline bool     is_snan() const           { return is_nan() && ((mantissa & BFLOAT16_MANTISSA_QUIET_NAN_MASK) == 0); }
    constexpr inline bool     is_qnan() const           { return is_nan() && ((mantissa & BFLOAT16_MANTISSA_QUIET_NAN_MASK) != 0); }

    constexpr inline uint16_t get_nan_payload() const   { return mantissa & (~BFLOAT16_MANTISSA_QUIET_NAN_MASK); }

private:
    struct Holder { uint16_t payload; };
    explicit constexpr BFloat16(Holder h) : as_hex(h.payload) {}
#endif
};
typedef struct BFloat16 BFloat16;

// C interface
static inline bool     BFloat16_is_negative(BFloat16 f)         { return f.sign != 0; }
static inline bool     BFloat16_is_zero(BFloat16 f)             { return (f.exponent == BFLOAT16_DENORM_EXPONENT) && (f.mantissa == 0); }
static inline bool     BFloat16_is_denormal(BFloat16 f)         { return (f.exponent == BFLOAT16_DENORM_EXPONENT) && (f.mantissa != 0); }

static inline bool     BFloat16_is_general_nan(BFloat16 f)      { return f.exponent == FLOAT32_NAN_EXPONENT; }
static inline bool     BFloat16_is_inf(BFloat16 f)              { return BFloat16_is_general_nan(f) && (f.mantissa == 0); }
static inline bool     BFloat16_is_nan(BFloat16 f)              { return BFloat16_is_general_nan(f) && (f.mantissa != 0); }
static inline bool     BFloat16_is_snan(BFloat16 f)             { return BFloat16_is_nan(f) && (f.as_nan.quiet == 0); }
static inline bool     BFloat16_is_qnan(BFloat16 f)             { return BFloat16_is_nan(f) && (f.as_nan.quiet != 0); }

static inline uint16_t BFloat16_get_nan_payload(BFloat16 f)     { return f.as_nan.payload; }


struct Float32 {
    union {
        struct {
            uint32_t mantissa : FLOAT32_MANTISSA_BITS;
            uint32_t exponent : FLOAT32_EXPONENT_BITS;
            uint32_t sign     : FLOAT32_SIGN_BITS;
        };
        struct {
            uint32_t payload  : FLOAT32_MANTISSA_BITS - FLOAT32_QUIET_BITS;
            uint32_t quiet    : FLOAT32_QUIET_BITS;
            uint32_t exponent : FLOAT32_EXPONENT_BITS;
            uint32_t sign     : FLOAT32_SIGN_BITS;
        } as_nan;
        float as_float;
        uint32_t as_hex;
    };

#ifdef __cplusplus
    inline Float32() = default;
    constexpr inline Float32(float f) : as_float(f) { }
    constexpr inline Float32(uint32_t s, uint32_t e, uint32_t m): mantissa(m), exponent(e), sign(s) { }
#endif
};
typedef struct Float32 Float32;

struct Float64 {
    union {
        struct {
            uint64_t mantissa : FLOAT64_MANTISSA_BITS;
            uint64_t exponent : FLOAT64_EXPONENT_BITS;
            uint64_t sign     : FLOAT64_SIGN_BITS;
        };
        struct {
            uint64_t payload  : FLOAT64_MANTISSA_BITS - FLOAT64_QUIET_BITS;
            uint64_t quiet    : FLOAT64_QUIET_BITS;
            uint64_t exponent : FLOAT64_EXPONENT_BITS;
            uint64_t sign     : FLOAT64_SIGN_BITS;
        } as_nan;
        struct {
            uint32_t low32;
            uint32_t high32;
        } as_hex32;
        double as_float;
        uint64_t as_hex;
    };

#ifdef __cplusplus
    inline Float64() = default;
    constexpr inline Float64(float f) : as_float(f) { }
    constexpr inline Float64(uint64_t s, uint64_t e, uint64_t m): mantissa(m), exponent(e), sign(s) { }
#endif
};
typedef struct Float64 Float64;

struct Float80 {
    union {
        struct {
            uint64_t mantissa : FLOAT80_MANTISSA_BITS;
            uint64_t jbit     : FLOAT80_JBIT_BITS;
            uint64_t exponent : FLOAT80_EXPONENT_BITS;
            uint64_t sign     : FLOAT80_SIGN_BITS;
        };
        struct {
            uint64_t payload  : FLOAT80_MANTISSA_BITS - FLOAT80_QUIET_BITS;
            uint64_t quiet    : FLOAT80_QUIET_BITS;
            uint64_t jbit     : FLOAT80_JBIT_BITS;
            uint64_t exponent : FLOAT80_EXPONENT_BITS;
            uint64_t sign     : FLOAT80_SIGN_BITS;
        } as_nan;
        struct {
            uint64_t low64;
            uint16_t high16;
        } as_hex;
        struct {
            uint32_t low32;
            uint32_t high32;
            uint16_t extra16;
        } as_hex32;
        long double as_float;
    };

#ifdef __cplusplus
    inline Float80() = default;
    constexpr inline Float80(long double f) : as_float(f) { }
    constexpr inline Float80(uint64_t s, uint64_t e, uint64_t j, uint64_t m): mantissa(m), jbit(j), exponent(e), sign(s) { }
#endif
};
typedef struct Float80 Float80;

/**
 * @brief C/C++ builders (inlined)
 *
 * "variadic" C/C++ builders, either constructor (C++) or direct struct initialization (C)
 *
 * @{
 */

#ifdef __cplusplus
#define STATIC_INLINE static inline constexpr
#else
#define STATIC_INLINE static inline
#endif

STATIC_INLINE Float16 new_float16(uint16_t sign, uint16_t exponent, uint16_t mantissa)
{
#ifdef __cplusplus
    return Float16(sign, exponent, mantissa);
#else
    return (Float16) {{{ .sign = sign, .exponent = exponent, .mantissa = mantissa }}};
#endif
}

STATIC_INLINE BFloat16 new_bfloat16(uint16_t sign, uint16_t exponent, uint16_t mantissa)
{
#ifdef __cplusplus
    return BFloat16(sign, exponent, mantissa);
#else
    return (BFloat16) {{{ .sign = sign, .exponent = exponent, .mantissa = mantissa }}};
#endif
}

STATIC_INLINE Float32 new_float32(uint32_t sign, uint32_t exponent, uint32_t mantissa)
{
#ifdef __cplusplus
    return Float32(sign, exponent, mantissa);
#else
    return (Float32) {{{ .sign = sign, .exponent = exponent, .mantissa = mantissa }}};
#endif
}

STATIC_INLINE Float64 new_float64(uint64_t sign, uint64_t exponent, uint64_t mantissa)
{
#ifdef __cplusplus
    return Float64(sign, exponent, mantissa);
#else
    return (Float64) {{{ .sign = sign, .exponent = exponent, .mantissa = mantissa }}};
#endif
}

STATIC_INLINE Float80 new_float80(uint64_t sign, uint64_t exponent, uint64_t jbit, uint64_t mantissa)
{
#ifdef __cplusplus
    return Float80(sign, exponent, jbit, mantissa);
#else
    return (Float80) {{{ .sign = sign, .exponent = exponent, .jbit = jbit, .mantissa = mantissa }}};
#endif
}
/** @} */

Float16 new_random_float16();
BFloat16 new_random_bfloat16();
Float32 new_random_float32();
Float64 new_random_float64();
Float80 new_random_float80();

// __builtins
#define IS_NEGATIVE(v) \
    _Generic((v),\
        Float16: Float16_is_negative,\
        BFloat16: BFloat16_is_negative\
    )(v)
#define IS_ZERO(v) \
    _Generic((v),\
        Float16: Float16_is_zero,\
        BFloat16: BFloat16_is_zero\
    )(v)
#define IS_DENORMAL(v) \
    _Generic((v),\
        Float16: Float16_is_denormal,\
        BFloat16: BFloat16_is_denormal\
    )(v)
#define IS_INF(v) \
    _Generic((v),\
        Float16: Float16_is_inf,\
        BFloat16: BFloat16_is_inf\
    )(v)
#define IS_NAN(v) \
    _Generic((v),\
        Float16: Float16_is_nan,\
        BFloat16: BFloat16_is_nan\
    )(v)
#define IS_SNAN(v) \
    _Generic((v),\
        Float16: Float16_is_snan,\
        BFloat16: BFloat16_is_snan\
    )(v)
#define IS_QNAN(v) \
    _Generic((v),\
        Float16: Float16_is_qnan,\
        BFloat16: BFloat16_is_qnan\
    )(v)
#define GET_NAN_PAYLOAD(v) \
    _Generic((v),\
        Float16: Float16_get_nan_payload,\
        BFloat16: BFloat16_get_nan_payload\
    )(v)

#ifdef __cplusplus
} // extern "C"
#endif

#endif //PROJECT_FP_VECTORS_H
