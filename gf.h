/* 
 * `Finding path motifs in large temporal graphs using algebraic fingerprints`
 *
 * This experimental source code is supplied to accompany the 
 * aforementioned paper. 
 * 
 * The source code is configured for a gcc build to a native microarchitecture
 * that must support the AVX2 and PCLMULQDQ instruction set extensions. Other
 * builds are possible but require manual configuration of 'Makefile' and
 * 'builds.h'.
 * 
 * The source code is subject to the following license.
 * 
 * The MIT License (MIT)
 * 
 * Copyright (c) 2020 S. Thejaswi, A. Gionis, J. Lauri
 * Copyright (c) 2019 S. Thejaswi, A. Gionis
 * Copyright (c) 2014 A. Björklund, P. Kaski L. Kowalik, J. Lauri
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 * 
 */

#ifndef GF_H
#define GF_H

#include<immintrin.h> 
// https://software.intel.com/sites/landingpage/IntrinsicsGuide/

/*********************************** Basic types for finite field arithmetic. */

/* 
 * Terminology
 * -----------
 *
 *   Scalar:      one field element (e.g. from GF(2^64))
 *
 *   Limb:        a vector of scalars, arithmetic is executed 
 *                one limb at a time
 *
 *   Line:        a vector of limbs, memory transactions on scalars 
 *                are executed one line at a time, e.g. one cache line 
 *                at a time
 *
 *   Line array:  an array of lines
 *
 */

/* 
 * Remarks: 
 *
 * 1)
 * CPU instruction set extensions (Intel SSE2/AVX2/PCLMULQDQ) are supported.
 *
 * 2)
 * Assumes an index data type "index_t" is defined. 
 *
 * 3)
 * Variables of type "int" are assumed to be large enough
 * to hold quantities for purposes of shifting a variable. 
 *
 * 4) 
 * The primitive polynomials used in field construction:
 *
 * degree  8:  [x^8 +] x^4 + x^3 + x^2 + 1            ~ 0x1D = 29
 * degree 16: [x^16 +] x^5 + x^3 + x^2 + 1            ~ 0x2D = 45
 * degree 32: [x^32 +] x^7 + x^5 + x^3 + x^2 + x + 1  ~ 0xAF = 175
 * degree 64: [x^64 +] x^4 + x^3 + x + 1              ~ 0x1B = 27
 *
 * http://www.ams.org/journals/mcom/1962-16-079/S0025-5718-1962-0148256-1/S0025-5718-1962-0148256-1.pdf
 *
 */

#define GF2_8_MODULUS   0x01D
#define GF2_16_MODULUS  0x02D
#define GF2_32_MODULUS  0x0AF
#define GF2_64_MODULUS  0x01BL

#define GF_PRECOMPUTE ;
    
/**************************************************************** Limb types. */

/*********************** One B-bit word representing one element of GF(2^B). */

#if (defined LIMB_1_GF2_8 || defined LIMB_1_GF2_64 || defined LIMB_1_GF2_8_EXPLOG || defined LIMB_1_GF2_64_UNROLL)

#if (defined LIMB_1_GF2_8 || defined LIMB_1_GF2_8_EXPLOG)
#define GF2_B       8
#define GF2_MODULUS GF2_8_MODULUS
#ifdef LIMB_1_GF2_8
#define LIMB_TYPE "8-bit word [1 x GF(2^{8}) with one 8-bit word]"
#endif
#ifdef LIMB_1_GF2_8_EXPLOG
#define LIMB_TYPE "8-bit word [1 x GF(2^{8}) with one 8-bit word, exp/log multiply]"
#endif
typedef unsigned char scalar_t; // work with 8-bit scalars
#define WORD_TO_SCALAR(x) ((x)&0x0FF)
#endif

#if (defined LIMB_1_GF2_64 || defined LIMB_1_GF2_64_UNROLL)
#define GF2_B       64
#define GF2_MODULUS GF2_64_MODULUS
#ifdef LIMB_1_GF2_64
#define LIMB_TYPE "64-bit word [1 x GF(2^{64}) with one 64-bit word]"
#endif
#ifdef LIMB_1_GF2_64_UNROLL
#define LIMB_TYPE "64-bit word [1 x GF(2^{64}) with one 64-bit word, unrolled multiply]"
#endif
typedef unsigned long scalar_t; // work with 64-bit scalars
#define WORD_TO_SCALAR(x) ((x)&0xFFFFFFFFFFFFFFFFUL)
#endif

typedef scalar_t limb_t;  // limb is one scalar

#define SCALARS_IN_LIMB 1

#define LIMB_PREFETCH(ptr,idx)                                           \
{                                                                        \
    limb_t *p = ((limb_t *) ptr) + (idx);                                \
    _mm_prefetch(p, _MM_HINT_T0);                                        \
}

#define LIMB_LOAD(target,ptr,idx)                                        \
{                                                                        \
    target = ((limb_t *) ptr)[idx];                                      \
}

#define LIMB_STORE(ptr,idx,source)                                       \
{                                                                        \
    ((limb_t *) ptr)[idx] = source;                                      \
}

#define LIMB_ARRAY_LOAD_SCALAR(target,ptr,idx)                           \
{                                                                        \
    target = ((scalar_t *) ptr)[idx];                                    \
}

#define LIMB_ARRAY_STORE_SCALAR(ptr,idx,source)                          \
{                                                                        \
    ((scalar_t *) ptr)[idx] = (source);                                  \
}

#define LIMB_MOV(target, source)                                         \
{                                                                        \
    target = source;                                                     \
}

#define LIMB_ADD(target, left, right)                                    \
{                                                                        \
    target = (left)^(right);                                             \
}

#if (defined LIMB_1_GF2_8 || defined LIMB_1_GF2_64)    
#define LIMB_MUL(target, left, right)                                    \
{                                                                        \
    SCALAR_MUL(target, left, right);                                     \
}
#endif

#ifdef LIMB_1_GF2_8_EXPLOG
#define LIMB_MUL(target, left, right)                                        \
{                                                                            \
    scalar_t lll = left;                                                     \
    scalar_t rrr = right;                                                    \
    scalar_t ttt;                                                            \
    if(lll == 0 || rrr == 0) {                                               \
        ttt = 0;                                                             \
    } else {                                                                 \
        ttt = gf2_8_lookup_exp[gf2_8_lookup_log[lll]+gf2_8_lookup_log[rrr]]; \
    }                                                                        \
    target = ttt;                                                            \
}
#endif

#ifdef LIMB_1_GF2_64_UNROLL
#define REP2(x) x x
#define REP4(x) REP2(x) REP2(x)
#define REP8(x) REP4(x) REP4(x)
#define REP16(x) REP8(x) REP8(x)
#define REP32(x) REP16(x) REP16(x)
#define REP64(x) REP32(x) REP32(x)
#define LIMB_MUL(target, left, right)                                    \
{                                                                        \
    scalar_t lll = left;                                                 \
    scalar_t rrr = right;                                                \
    scalar_t ttt = 0UL;                                                  \
    scalar_t mmm;                                                        \
    REP64({                                                              \
      ttt = ttt ^ (lll & (-(rrr & 1UL)));                                \
      mmm = ((signed long)lll)>>63;                                      \
      rrr >>= 1;                                                         \
      lll = (lll << 1)^(((scalar_t)GF2_64_MODULUS)&mmm);                 \
    })                                                                   \
    target = ttt;                                                        \
}
#endif

#define LIMB_MUL_SCALAR(target, source, scalar)                          \
{                                                                        \
    LIMB_MUL(target, source, scalar);                                    \
}

#define LIMB_SUM(target, source)                                         \
{                                                                        \
    target = source;                                                     \
}

#define LIMB_SET_ZERO(target)                                            \
{                                                                        \
    target = (scalar_t) 0;                                               \
}

#define LIMB_SET_ONE(target)                                             \
{                                                                        \
    target = (scalar_t) 1;                                               \
}

#define LIMB_STORE_SCALAR(target,idx,source)                             \
{                                                                        \
    target = source;                                                     \
}

#define SCALAR_SET_ZERO(target)                                          \
{                                                                        \
    target = 0;                                                          \
}

#define SCALAR_ADD(target, left, right)                                  \
{                                                                        \
    target = left^right;                                                 \
}


#define SCALAR_MUL(target, left, right)                                  \
{                                                                        \
    REF_SCALAR_MUL(target, left, right);                                 \
}

#endif




/*********************** One 64-bit word representing 1 element of GF(2^64). */

#ifdef LIMB_1_GF2_64_SSE2_PCLMULQDQ

#define LIMB_TYPE "64-bit word [1 x GF(2^{64}) with one 64-bit word, PCLMULQDQ]"

#define GF2_B       64
#define GF2_MODULUS GF2_64_MODULUS

typedef unsigned long scalar_t; // work with 64-bit scalars
typedef scalar_t limb_t;        // limb is one scalar

#define SCALARS_IN_LIMB 1

#define LIMB_PREFETCH(ptr,idx)                                           \
{                                                                        \
    limb_t *p = ((limb_t *) ptr) + (idx);                                \
    _mm_prefetch(p, _MM_HINT_T0);                                        \
}

#define LIMB_LOAD(target,ptr,idx)                                        \
{                                                                        \
    target = ((limb_t *) ptr)[idx];                                      \
}

#define LIMB_STORE(ptr,idx,source)                                       \
{                                                                        \
    ((limb_t *) ptr)[idx] = source;                                      \
}

#define LIMB_ARRAY_LOAD_SCALAR(target,ptr,idx)                           \
{                                                                        \
    target = ((scalar_t *) ptr)[idx];                                    \
}

#define LIMB_ARRAY_STORE_SCALAR(ptr,idx,source)                          \
{                                                                        \
    ((scalar_t *) ptr)[idx] = (source);                                  \
}

#define LIMB_MOV(target, source)                                         \
{                                                                        \
    target = source;                                                     \
}

#define LIMB_ADD(target, left, right)                                    \
{                                                                        \
    target = (left)^(right);                                             \
}

#define LIMB_MUL(target, left, right)                                    \
{                                                                        \
    GF2_64_MUL_SSE2_PCLMULQDQ(target, left, right);                      \
}

#define LIMB_MUL_SCALAR(target, source, scalar)                          \
{                                                                        \
    LIMB_MUL(target, source, scalar);                                    \
}

#define LIMB_SUM(target, source)                                         \
{                                                                        \
    target = source;                                                     \
}

#define LIMB_SET_ZERO(target)                                            \
{                                                                        \
    target = (scalar_t) 0;                                               \
}

#define LIMB_SET_ONE(target)                                             \
{                                                                        \
    target = (scalar_t) 1;                                               \
}

#define LIMB_STORE_SCALAR(target,idx,source)                             \
{                                                                        \
    target = source;                                                     \
}

#define SCALAR_SET_ZERO(target)                                          \
{                                                                        \
    target = 0;                                                          \
}

#define SCALAR_ADD(target, left, right)                                  \
{                                                                        \
    target = left^right;                                                 \
}

#define SCALAR_MUL(target, left, right)                                  \
{                                                                        \
    REF_SCALAR_MUL(target, left, right);                                 \
}

#define WORD_TO_SCALAR(x) ((x)&0xFFFFFFFFFFFFFFFFUL)

#endif



/**************** One AVX2 256-bit word representing 4 elements of GF(2^64). */

#ifdef LIMB_4_GF2_64_AVX2_PCLMULQDQ

#define LIMB_TYPE "256-bit AVX2 [4 x GF(2^{64}) with four 64-bit words]"

#define GF2_B       64
#define GF2_MODULUS GF2_64_MODULUS

typedef unsigned long scalar_t; // work with 64-bit scalars
typedef __m256i limb_t;         // one 256-bit AVX2 word

#define SCALARS_IN_LIMB 4

#define LIMB_PREFETCH(ptr,idx)                                           \
{                                                                        \
    __m256i *p = ((__m256i *) ptr) + (idx);                              \
    _mm_prefetch(p, _MM_HINT_T0);                                        \
}

#define LIMB_LOAD(target,ptr,idx)                                        \
{                                                                        \
    target = _mm256_loadu_si256(((__m256i *) ptr)+(idx));                \
}

#define LIMB_STORE(ptr,idx,source)                                       \
{                                                                        \
    _mm256_storeu_si256(((__m256i *) ptr)+(idx), source);                \
}

#define LIMB_ARRAY_LOAD_SCALAR(target,ptr,idx)                           \
{                                                                        \
    target = ((scalar_t *) ptr)[idx];                                    \
}

#define LIMB_ARRAY_STORE_SCALAR(ptr,idx,source)                          \
{                                                                        \
    ((scalar_t *) ptr)[idx] = (source);                                  \
}

#define LIMB_MOV(target, source)                                         \
{                                                                        \
    target = source;                                                     \
}

#define LIMB_ADD(target, left, right)                                    \
{                                                                        \
    target = _mm256_xor_si256(left, right);                              \
}
    
#define LIMB_MUL(target, left, right)                                    \
{                                                                        \
    GF2_64_MUL_QUAD_AVX2_PCLMULQDQ(target, left, right);                 \
}

#define LIMB_MUL_SCALAR(target, source, scalar)                          \
{                                                                        \
    GF2_64_MUL_QUAD_SCALAR_AVX2_PCLMULQDQ(target, source, scalar);       \
}

#define LIMB_SUM(target, source)                                         \
{                                                                        \
    __m128i tmp1 = _mm256_extracti128_si256(source, 0);                  \
    __m128i tmp2 = _mm256_extracti128_si256(source, 1);                  \
    __m128i tmp3 = _mm_xor_si128(tmp1, tmp2);                            \
    scalar_t  tmp4 = _mm_extract_epi64(tmp3, 0);                         \
    scalar_t  tmp5 = _mm_extract_epi64(tmp3, 1);                         \
    target = tmp4^tmp5;                                                  \
}

#define LIMB_SET_ZERO(target)                                            \
{                                                                        \
    target = _mm256_setzero_si256();                                     \
}

#define LIMB_SET_ONE(target)                                             \
{                                                                        \
    target = _mm256_set1_epi64x(1L);                                     \
}

#define LIMB_STORE_SCALAR(target,idx,source)                             \
{                                                                        \
    __m256i data = _mm256_broadcastq_epi64(_mm_set_epi64x(0,source));    \
    __m256i mask;                                                        \
    switch((idx)&3) {                                                    \
        case 0:                                                          \
            mask = _mm256_set_epi64x(0,0,0,0xFFFFFFFFFFFFFFFFL); break;  \
        case 1:                                                          \
            mask = _mm256_set_epi64x(0,0,0xFFFFFFFFFFFFFFFFL,0); break;  \
        case 2:                                                          \
            mask = _mm256_set_epi64x(0,0xFFFFFFFFFFFFFFFFL,0,0); break;  \
        case 3:                                                          \
            mask = _mm256_set_epi64x(0xFFFFFFFFFFFFFFFFL,0,0,0); break;  \
    }                                                                    \
    target = _mm256_xor_si256(_mm256_and_si256(mask,data),               \
                              _mm256_andnot_si256(mask,target));         \
}

#define SCALAR_SET_ZERO(target)                                          \
{                                                                        \
    target = 0;                                                          \
}

#define SCALAR_ADD(target, left, right)                                  \
{                                                                        \
    target = left^right;                                                 \
}

#define SCALAR_MUL(target, left, right)                                  \
{                                                                        \
    GF2_64_MUL_SSE2_PCLMULQDQ(target, left, right);                      \
}

#define WORD_TO_SCALAR(x) ((x)&0xFFFFFFFFFFFFFFFFUL)

#endif


/**************** One SSE2 128-bit word representing 2 elements of GF(2^64). */

#ifdef LIMB_2_GF2_64_SSE2_PCLMULQDQ

#define LIMB_TYPE "128-bit SSE2 [2 x GF(2^{64}) with two 64-bit words]"

#define GF2_B       64
#define GF2_MODULUS GF2_64_MODULUS

typedef unsigned long scalar_t; // work with 64-bit scalars
typedef __m128i limb_t;         // one 128-bit SSE2 word

#define SCALARS_IN_LIMB 2

#define LIMB_PREFETCH(ptr,idx)                                           \
{                                                                        \
    __m128i *p = ((__m128i *) ptr) + (idx);                              \
    _mm_prefetch(p, _MM_HINT_T0);                                        \
}

#define LIMB_ARRAY_LOAD_SCALAR(target,ptr,idx)                           \
{                                                                        \
    target = ((scalar_t *) ptr)[idx];                                    \
}
    
#define LIMB_ARRAY_STORE_SCALAR(ptr,idx,source)                          \
{                                                                        \
    ((scalar_t *) ptr)[idx] = (source);                                  \
}

#define LIMB_LOAD(target,ptr,idx)                                        \
{                                                                        \
    target = _mm_loadu_si128(((__m128i*) ptr)+(idx));                    \
}

#define LIMB_STORE(ptr,idx,source)                                       \
{                                                                        \
    _mm_storeu_si128(((__m128i*)ptr)+(idx), source);                     \
}

#define LIMB_MOV(target, source)                                         \
{                                                                        \
    target = source;                                                     \
}

#define LIMB_ADD(target, left, right)                                    \
{                                                                        \
    target = _mm_xor_si128(left,right);                                  \
}
    
#define LIMB_MUL(target, left, right)                                    \
{                                                                        \
    GF2_64_MUL_PAIR_SSE2_PCLMULQDQ(target, left, right);                 \
}

#define LIMB_MUL_SCALAR(target, source, scalar)                          \
{                                                                        \
    GF2_64_MUL_PAIR_SCALAR_SSE2_PCLMULQDQ(target, source, scalar);       \
}

#define LIMB_SUM(target, source)                                         \
{                                                                        \
    scalar_t lo = _mm_cvtsi128_si64(source);                             \
    __m128i t   = _mm_unpackhi_epi64(source,source);                     \
    scalar_t hi = _mm_cvtsi128_si64(t);                                  \
    target = lo^hi;                                                      \
}

#define LIMB_SET_ZERO(target)                                            \
{                                                                        \
    target = _mm_setzero_si128();                                        \
}

#define LIMB_SET_ONE(target)                                             \
{                                                                        \
    target = _mm_set1_epi64x(1L);                                        \
}

#define LIMB_STORE_SCALAR(target,idx,source)                             \
{                                                                        \
    __m128i data = _mm_set1_epi64x(source);                              \
    __m128i mask;                                                        \
    switch((idx)&1) {                                                    \
        case 0:                                                          \
            mask = _mm_set_epi64x(0,0xFFFFFFFFFFFFFFFFL); break;         \
        case 1:                                                          \
            mask = _mm_set_epi64x(0xFFFFFFFFFFFFFFFFL,0); break;         \
    }                                                                    \
    target = _mm_xor_si128(_mm_and_si128(mask,data),                     \
                              _mm_andnot_si128(mask,target));            \
}

#define SCALAR_SET_ZERO(target)                                          \
{                                                                        \
    target = 0;                                                          \
}

#define SCALAR_ADD(target, left, right)                                  \
{                                                                        \
    target = left^right;                                                 \
}

#define SCALAR_MUL(target, left, right)                                  \
{                                                                        \
    GF2_64_MUL_SSE2_PCLMULQDQ(target, left, right);                      \
}

#define WORD_TO_SCALAR(x) ((x)&0xFFFFFFFFFFFFFFFFUL)

#endif


/**************** An 8-word limb that packs K bit-sliced elements of GF(2^8). */

#if (defined LIMB_32_GF2_8 || defined LIMB_64_GF2_8)

#ifdef LIMB_32_GF2_8 
#define LIMB_TYPE "256 bits [32 x GF(2^8) with eight 32-bit words, bit sliced]"
#define SCALARS_IN_LIMB 32
typedef unsigned int slice_t;   // one slice is 32 bits
#endif

#ifdef LIMB_64_GF2_8 
#define LIMB_TYPE "512 bits [64 x GF(2^8) with eight 64-bit words, bit sliced]"
#define SCALARS_IN_LIMB 64
typedef unsigned long slice_t;   // one slice is 64 bits
#endif

/* A limb consists of eight K-bit words, each word represents _one bit_
 * in each element of GF(2^8). */

#define GF2_B       8
#define GF2_MODULUS GF2_8_MODULUS

typedef unsigned char scalar_t;
typedef slice_t limb_t[8];

#define LIMB_ARRAY_LOAD_SCALAR(target,ptr,idx)                           \
{                                                                        \
    index_t j = idx;                                                     \
    index_t jlo = j%SCALARS_IN_LIMB;                                     \
    index_t jhi = j/SCALARS_IN_LIMB;                                     \
    slice_t *p = ((slice_t *)ptr) + 8*jhi;                               \
    target = (((p[0] >> jlo)&1)<<0)|                                     \
             (((p[1] >> jlo)&1)<<1)|                                     \
             (((p[2] >> jlo)&1)<<2)|                                     \
             (((p[3] >> jlo)&1)<<3)|                                     \
             (((p[4] >> jlo)&1)<<4)|                                     \
             (((p[5] >> jlo)&1)<<5)|                                     \
             (((p[6] >> jlo)&1)<<6)|                                     \
             (((p[7] >> jlo)&1)<<7);                                     \
}

#define LIMB_ARRAY_STORE_SCALAR(ptr,idx,source)                             \
{                                                                           \
    index_t j = idx;                                                        \
    index_t jlo = j%SCALARS_IN_LIMB;                                        \
    index_t jhi = j/SCALARS_IN_LIMB;                                        \
    slice_t *p = ((slice_t *)ptr) + 8*jhi;                                  \
    scalar_t byte = source;                                                 \
    p[0] = (p[0] & (~(((slice_t)1)<<jlo)))|(((slice_t)((byte>>0)&1))<<jlo); \
    p[1] = (p[1] & (~(((slice_t)1)<<jlo)))|(((slice_t)((byte>>1)&1))<<jlo); \
    p[2] = (p[2] & (~(((slice_t)1)<<jlo)))|(((slice_t)((byte>>2)&1))<<jlo); \
    p[3] = (p[3] & (~(((slice_t)1)<<jlo)))|(((slice_t)((byte>>3)&1))<<jlo); \
    p[4] = (p[4] & (~(((slice_t)1)<<jlo)))|(((slice_t)((byte>>4)&1))<<jlo); \
    p[5] = (p[5] & (~(((slice_t)1)<<jlo)))|(((slice_t)((byte>>5)&1))<<jlo); \
    p[6] = (p[6] & (~(((slice_t)1)<<jlo)))|(((slice_t)((byte>>6)&1))<<jlo); \
    p[7] = (p[7] & (~(((slice_t)1)<<jlo)))|(((slice_t)((byte>>7)&1))<<jlo); \
}

#define LIMB_PREFETCH(ptr,idx)                                            \
{                                                                         \
    slice_t *p = ((slice_t *)ptr) + 8*(idx);                              \
    _mm_prefetch(p, _MM_HINT_T0);                                         \
}

#define LIMB_LOAD(target,ptr,idx)                                         \
{                                                                         \
    slice_t *p = ((slice_t *)ptr) + 8*(idx);                              \
    (target)[0] = p[0];                                                   \
    (target)[1] = p[1];                                                   \
    (target)[2] = p[2];                                                   \
    (target)[3] = p[3];                                                   \
    (target)[4] = p[4];                                                   \
    (target)[5] = p[5];                                                   \
    (target)[6] = p[6];                                                   \
    (target)[7] = p[7];                                                   \
}

#define LIMB_STORE(ptr,idx,source)                                        \
{                                                                         \
    slice_t *p = ((slice_t *) ptr) + 8*(idx);                             \
    p[0] = (source)[0];                                                   \
    p[1] = (source)[1];                                                   \
    p[2] = (source)[2];                                                   \
    p[3] = (source)[3];                                                   \
    p[4] = (source)[4];                                                   \
    p[5] = (source)[5];                                                   \
    p[6] = (source)[6];                                                   \
    p[7] = (source)[7];                                                   \
}

#define LIMB_MOV(target, source)                                          \
{                                                                         \
    (target)[0] = (source)[0];                                            \
    (target)[1] = (source)[1];                                            \
    (target)[2] = (source)[2];                                            \
    (target)[3] = (source)[3];                                            \
    (target)[4] = (source)[4];                                            \
    (target)[5] = (source)[5];                                            \
    (target)[6] = (source)[6];                                            \
    (target)[7] = (source)[7];                                            \
}

#define LIMB_ADD(target, left, right)                                     \
{                                                                         \
    (target)[0] = (left)[0]^(right)[0];                                   \
    (target)[1] = (left)[1]^(right)[1];                                   \
    (target)[2] = (left)[2]^(right)[2];                                   \
    (target)[3] = (left)[3]^(right)[3];                                   \
    (target)[4] = (left)[4]^(right)[4];                                   \
    (target)[5] = (left)[5]^(right)[5];                                   \
    (target)[6] = (left)[6]^(right)[6];                                   \
    (target)[7] = (left)[7]^(right)[7];                                   \
}

#define LIMB_MUL(target, left, right)\
{                                                                         \
    GF2_8_MUL_SLICE((target)[0], (target)[1], (target)[2], (target)[3],   \
                    (target)[4], (target)[5], (target)[6], (target)[7],   \
                    (left)[0], (left)[1], (left)[2], (left)[3],           \
                    (left)[4], (left)[5], (left)[6], (left)[7],           \
                    (right)[0], (right)[1], (right)[2], (right)[3],       \
                    (right)[4], (right)[5], (right)[6], (right)[7]);      \
}

#define LIMB_MUL_SCALAR(target, source, scalar)                           \
{                                                                         \
    limb_t temp;                                                          \
    temp[0] = -((slice_t)(((scalar)>>0)&1));                              \
    temp[1] = -((slice_t)(((scalar)>>1)&1));                              \
    temp[2] = -((slice_t)(((scalar)>>2)&1));                              \
    temp[3] = -((slice_t)(((scalar)>>3)&1));                              \
    temp[4] = -((slice_t)(((scalar)>>4)&1));                              \
    temp[5] = -((slice_t)(((scalar)>>5)&1));                              \
    temp[6] = -((slice_t)(((scalar)>>6)&1));                              \
    temp[7] = -((slice_t)(((scalar)>>7)&1));                              \
    LIMB_MUL(target, source, temp);                                       \
}

#define LIMB_SUM(target, source)                                          \
{                                                                         \
    scalar_t temp = 0;                                                    \
    for(int i = 0; i < SCALARS_IN_LIMB; i++) {                            \
      temp ^= (((source)[0]>>i)&1)<<0;                                    \
      temp ^= (((source)[1]>>i)&1)<<1;                                    \
      temp ^= (((source)[2]>>i)&1)<<2;                                    \
      temp ^= (((source)[3]>>i)&1)<<3;                                    \
      temp ^= (((source)[4]>>i)&1)<<4;                                    \
      temp ^= (((source)[5]>>i)&1)<<5;                                    \
      temp ^= (((source)[6]>>i)&1)<<6;                                    \
      temp ^= (((source)[7]>>i)&1)<<7;                                    \
    }                                                                     \
    target = temp;                                                        \
}

#define LIMB_SET_ZERO(target)                                             \
{                                                                         \
    (target)[0] = 0;                                                      \
    (target)[1] = 0;                                                      \
    (target)[2] = 0;                                                      \
    (target)[3] = 0;                                                      \
    (target)[4] = 0;                                                      \
    (target)[5] = 0;                                                      \
    (target)[6] = 0;                                                      \
    (target)[7] = 0;                                                      \
}

#define LIMB_SET_ONE(target)                                              \
{                                                                         \
    (target)[0] = (slice_t) -1;                                           \
    (target)[1] = 0;                                                      \
    (target)[2] = 0;                                                      \
    (target)[3] = 0;                                                      \
    (target)[4] = 0;                                                      \
    (target)[5] = 0;                                                      \
    (target)[6] = 0;                                                      \
    (target)[7] = 0;                                                      \
}

#define LIMB_STORE_SCALAR(target,idx,source)                                                      \
{                                                                                                 \
    (target)[0] = ((target)[0] & (~(((slice_t)1)<<(idx))))|(((slice_t)(((source)>>0)&1))<<(idx)); \
    (target)[1] = ((target)[1] & (~(((slice_t)1)<<(idx))))|(((slice_t)(((source)>>1)&1))<<(idx)); \
    (target)[2] = ((target)[2] & (~(((slice_t)1)<<(idx))))|(((slice_t)(((source)>>2)&1))<<(idx)); \
    (target)[3] = ((target)[3] & (~(((slice_t)1)<<(idx))))|(((slice_t)(((source)>>3)&1))<<(idx)); \
    (target)[4] = ((target)[4] & (~(((slice_t)1)<<(idx))))|(((slice_t)(((source)>>4)&1))<<(idx)); \
    (target)[5] = ((target)[5] & (~(((slice_t)1)<<(idx))))|(((slice_t)(((source)>>5)&1))<<(idx)); \
    (target)[6] = ((target)[6] & (~(((slice_t)1)<<(idx))))|(((slice_t)(((source)>>6)&1))<<(idx)); \
    (target)[7] = ((target)[7] & (~(((slice_t)1)<<(idx))))|(((slice_t)(((source)>>7)&1))<<(idx)); \
}

#define SCALAR_SET_ZERO(target)                                           \
{                                                                         \
    target = 0;                                                           \
}
    

#define SCALAR_ADD(target, left, right)                                   \
{                                                                         \
    target = left^right;                                                  \
}

#define SCALAR_MUL(target, left, right)                                   \
{                                                                         \
    REF_SCALAR_MUL(target, left, right);                                  \
}

#define WORD_TO_SCALAR(x) ((x)&0x0FF)

#endif



/********************* A 256-bit AVX2 limb that packs 32 elements of GF(2^8). */

#ifdef LIMB_32_GF2_8_AVX2
#define LIMB_TYPE "256-bit AVX2 [32 x GF(2^8)]"

#define GF2_B       8
#define GF2_MODULUS GF2_8_MODULUS

typedef unsigned char scalar_t;
typedef __m256i limb_t;         // one 256-bit AVX2 word

#define SCALARS_IN_LIMB 32

#define LIMB_ARRAY_LOAD_SCALAR(target,ptr,idx)                            \
{                                                                         \
    target = ((scalar_t *) ptr)[idx];                                     \
}

#define LIMB_ARRAY_STORE_SCALAR(ptr,idx,source)                           \
{                                                                         \
    ((scalar_t *) ptr)[idx] = (source);                                   \
}

#define LIMB_MOV(target,source)                                           \
{                                                                         \
    target = source;                                                      \
}

#define LIMB_LOAD(target,ptr,idx)                                         \
{                                                                         \
    target = _mm256_loadu_si256(((__m256i *) ptr)+(idx));                 \
}

#define LIMB_PREFETCH(ptr,idx)                                            \
{                                                                         \
    __m256i *p = ((__m256i *) ptr) + (idx);                               \
    _mm_prefetch(p, _MM_HINT_T0);                                         \
}

#define LIMB_STORE(ptr,idx,source)                                        \
{                                                                         \
    _mm256_storeu_si256(((__m256i*)ptr)+(idx), source);                   \
}

#define LIMB_MUL(target, left, right)                                        \
{                                                                            \
    __m256i x = left;                                                        \
    __m256i y = right;                                                       \
    __m256i mask1 = _mm256_set1_epi8(0x0F);                                  \
    y = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(mask1,y),4),      \
                        _mm256_srli_epi16(_mm256_andnot_si256(mask1,y),4));  \
    __m256i mask2 = _mm256_set1_epi8(0x33);                                  \
    y = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(mask2,y),2),      \
                        _mm256_srli_epi16(_mm256_andnot_si256(mask2,y),2));  \
    __m256i mask3 = _mm256_set1_epi8(0x55);                                  \
    y = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(mask3,y),1),      \
                        _mm256_srli_epi16(_mm256_andnot_si256(mask3,y),1));  \
    __m256i zero = _mm256_setzero_si256();                                   \
    __m256i mod = _mm256_set1_epi8(GF2_8_MODULUS);                           \
    __m256i z = zero;                                                        \
    __m256i xhim;                                                            \
    __m256i inc;                                                             \
    xhim = _mm256_blendv_epi8(zero,mod,x);                                   \
    inc  = _mm256_blendv_epi8(zero,x,y);                                     \
    y    = _mm256_add_epi8(y,y);                                             \
    x    = _mm256_add_epi8(x,x);                                             \
    z    = _mm256_xor_si256(z,inc);                                          \
    x    = _mm256_xor_si256(x,xhim);                                         \
    xhim = _mm256_blendv_epi8(zero,mod,x);                                   \
    inc  = _mm256_blendv_epi8(zero,x,y);                                     \
    y    = _mm256_add_epi8(y,y);                                             \
    x    = _mm256_add_epi8(x,x);                                             \
    z    = _mm256_xor_si256(z,inc);                                          \
    x    = _mm256_xor_si256(x,xhim);                                         \
    xhim = _mm256_blendv_epi8(zero,mod,x);                                   \
    inc  = _mm256_blendv_epi8(zero,x,y);                                     \
    y    = _mm256_add_epi8(y,y);                                             \
    x    = _mm256_add_epi8(x,x);                                             \
    z    = _mm256_xor_si256(z,inc);                                          \
    x    = _mm256_xor_si256(x,xhim);                                         \
    xhim = _mm256_blendv_epi8(zero,mod,x);                                   \
    inc  = _mm256_blendv_epi8(zero,x,y);                                     \
    y    = _mm256_add_epi8(y,y);                                             \
    x    = _mm256_add_epi8(x,x);                                             \
    z    = _mm256_xor_si256(z,inc);                                          \
    x    = _mm256_xor_si256(x,xhim);                                         \
    xhim = _mm256_blendv_epi8(zero,mod,x);                                   \
    inc  = _mm256_blendv_epi8(zero,x,y);                                     \
    y    = _mm256_add_epi8(y,y);                                             \
    x    = _mm256_add_epi8(x,x);                                             \
    z    = _mm256_xor_si256(z,inc);                                          \
    x    = _mm256_xor_si256(x,xhim);                                         \
    xhim = _mm256_blendv_epi8(zero,mod,x);                                   \
    inc  = _mm256_blendv_epi8(zero,x,y);                                     \
    y    = _mm256_add_epi8(y,y);                                             \
    x    = _mm256_add_epi8(x,x);                                             \
    z    = _mm256_xor_si256(z,inc);                                          \
    x    = _mm256_xor_si256(x,xhim);                                         \
    xhim = _mm256_blendv_epi8(zero,mod,x);                                   \
    inc  = _mm256_blendv_epi8(zero,x,y);                                     \
    y    = _mm256_add_epi8(y,y);                                             \
    x    = _mm256_add_epi8(x,x);                                             \
    z    = _mm256_xor_si256(z,inc);                                          \
    x    = _mm256_xor_si256(x,xhim);                                         \
    inc  = _mm256_blendv_epi8(zero,x,y);                                     \
    z    = _mm256_xor_si256(z,inc);                                          \
    target = z;                                                              \
}

#define LIMB_MUL_SCALAR(target, source, scalar)                           \
{                                                                         \
    __m256i ext = _mm256_set1_epi8(scalar);                               \
    LIMB_MUL(target,source,ext);                                          \
}
    
#define LIMB_SUM(target, source)                                          \
{                                                                         \
    __m128i tmp1 = _mm256_extracti128_si256(source, 0);                   \
    __m128i tmp2 = _mm256_extracti128_si256(source, 1);                   \
    __m128i tmp3 = _mm_xor_si128(tmp1,tmp2);                              \
    unsigned long tmp4 = _mm_extract_epi64(tmp3, 0);                      \
    unsigned long tmp5 = _mm_extract_epi64(tmp3, 1);                      \
    unsigned long tmp6 = tmp4^tmp5;                                       \
    unsigned int tmp7 = (tmp6>>32)^tmp6;                                  \
    unsigned short tmp8 = (tmp7>>16)^tmp7;                                \
    target = (tmp8>>8)^tmp8;                                              \
}

#define LIMB_SET_ZERO(target)                                             \
{                                                                         \
    target = _mm256_setzero_si256();                                      \
}

#define LIMB_SET_ONE(target)                                              \
{                                                                         \
    target = _mm256_set1_epi8(1);                                         \
}

#define LIMB_STORE_SCALAR(target,idx,source)                                                             \
{                                                                                                        \
    __m256i data = _mm256_broadcastq_epi64(_mm_set_epi64x(0,((unsigned long) (source))<<(8*((idx)&7)))); \
    __m256i lomask = _mm256_broadcastq_epi64(_mm_set_epi64x(0,((unsigned long) 0xFF))<<(8*((idx)&7)));   \
    __m256i mask;                                                                                        \
    switch(((idx)>>3)&3) {                                                                               \
        case 0:                                                                                          \
            mask = _mm256_set_epi64x(0,0,0,0xFFFFFFFFFFFFFFFFL); break;                                  \
        case 1:                                                                                          \
            mask = _mm256_set_epi64x(0,0,0xFFFFFFFFFFFFFFFFL,0); break;                                  \
        case 2:                                                                                          \
            mask = _mm256_set_epi64x(0,0xFFFFFFFFFFFFFFFFL,0,0); break;                                  \
        case 3:                                                                                          \
            mask = _mm256_set_epi64x(0xFFFFFFFFFFFFFFFFL,0,0,0); break;                                  \
    }                                                                                                    \
    mask = _mm256_and_si256(mask,lomask);                                                                \
    target = _mm256_xor_si256(_mm256_and_si256(mask,data),                                               \
                              _mm256_andnot_si256(mask,target));                                         \
}

#define LIMB_ADD(target, left, right)                                     \
{                                                                         \
    target = _mm256_xor_si256(left,right);                                \
}


#define SCALAR_SET_ZERO(target)                                           \
{                                                                         \
    target = 0;                                                           \
}

#define SCALAR_ADD(target, left, right)                                   \
{                                                                         \
    target = left^right;                                                  \
}

#define SCALAR_MUL(target, left, right)                                   \
{                                                                         \
    REF_SCALAR_MUL(target, left, right);                                  \
}

#define WORD_TO_SCALAR(x) ((x)&0x0FF)

#endif



/************************* AVX2/SSE2 + PCLMULQDQ multiplication subroutines. */

// Modulus: [x^64 +] x^4 + x^3 + x + 1                ~ 0x1B = 27 (primitive)

#define REDUCE_QUAD_AVX2(target,p_20,p_31)                                \
{                                                                         \
    __m256i p_lo = _mm256_unpacklo_epi64(p_20, p_31);                     \
    __m256i p_hi = _mm256_unpackhi_epi64(p_20, p_31);                     \
    __m256i up   = _mm256_xor_si256(p_hi,                                 \
                   _mm256_xor_si256(_mm256_srli_epi64(p_hi, 63),          \
                   _mm256_xor_si256(_mm256_srli_epi64(p_hi, 61),          \
                                    _mm256_srli_epi64(p_hi, 60))));       \
          target = _mm256_xor_si256(up,                                   \
                   _mm256_xor_si256(_mm256_slli_epi64(up, 4),             \
                   _mm256_xor_si256(_mm256_slli_epi64(up, 3),             \
                   _mm256_xor_si256(_mm256_slli_epi64(up, 1),             \
                                    p_lo))));                             \
}

#define GF2_64_MUL_QUAD_AVX2_PCLMULQDQ(target, left, right)               \
{                                                                         \
    __m128i l_lo = _mm256_extracti128_si256(left, 0);                     \
    __m128i l_hi = _mm256_extracti128_si256(left, 1);                     \
    __m128i r_lo = _mm256_extracti128_si256(right, 0);                    \
    __m128i r_hi = _mm256_extracti128_si256(right, 1);                    \
    __m128i p_0  = _mm_clmulepi64_si128(l_lo, r_lo, 0x00);                \
    __m256i p_20 = _mm256_castsi128_si256(p_0);                           \
    __m128i p_2  = _mm_clmulepi64_si128(l_hi, r_hi, 0x00);                \
            p_20 = _mm256_inserti128_si256(p_20, p_2, 1);                 \
    __m128i p_1  = _mm_clmulepi64_si128(l_lo, r_lo, 0x11);                \
    __m256i p_31 = _mm256_castsi128_si256(p_1);                           \
    __m128i p_3  = _mm_clmulepi64_si128(l_hi, r_hi, 0x11);                \
            p_31 = _mm256_inserti128_si256(p_31, p_3, 1);                 \
    REDUCE_QUAD_AVX2(target, p_20, p_31);                                 \
}
          
#define GF2_64_MUL_QUAD_SCALAR_AVX2_PCLMULQDQ(target, source, scalar)     \
{                                                                         \
    __m128i rhs  = _mm_set_epi64x(0, scalar);                             \
    __m128i s_lo = _mm256_extracti128_si256(source, 0);                   \
    __m128i s_hi = _mm256_extracti128_si256(source, 1);                   \
    __m128i p_0  = _mm_clmulepi64_si128(s_lo, rhs, 0x00);                 \
    __m256i p_20 = _mm256_castsi128_si256(p_0);                           \
    __m128i p_2  = _mm_clmulepi64_si128(s_hi, rhs, 0x00);                 \
            p_20 = _mm256_inserti128_si256(p_20, p_2, 1);                 \
    __m128i p_1  = _mm_clmulepi64_si128(s_lo, rhs, 0x01);                 \
    __m256i p_31 = _mm256_castsi128_si256(p_1);                           \
    __m128i p_3  = _mm_clmulepi64_si128(s_hi, rhs, 0x01);                 \
            p_31 = _mm256_inserti128_si256(p_31, p_3, 1);                 \
    REDUCE_QUAD_AVX2(target, p_20, p_31);                                 \
}

#define REDUCE_PAIR_SSE2(target,p_0,p_1)                                  \
{                                                                         \
    __m128i p_lo = _mm_unpacklo_epi64(p_0, p_1);                          \
    __m128i p_hi = _mm_unpackhi_epi64(p_0, p_1);                          \
    __m128i up   = _mm_xor_si128(p_hi,                                    \
                   _mm_xor_si128(_mm_srli_epi64(p_hi, 63),                \
                   _mm_xor_si128(_mm_srli_epi64(p_hi, 61),                \
                                 _mm_srli_epi64(p_hi, 60))));             \
          target = _mm_xor_si128(up,                                      \
                   _mm_xor_si128(_mm_slli_epi64(up, 4),                   \
                   _mm_xor_si128(_mm_slli_epi64(up, 3),                   \
                   _mm_xor_si128(_mm_slli_epi64(up, 1),                   \
                                 p_lo))));                                \
}

#define GF2_64_MUL_PAIR_SSE2_PCLMULQDQ(target, left, right)               \
{                                                                         \
    __m128i p_0 = _mm_clmulepi64_si128(left, right, 0x00);                \
    __m128i p_1 = _mm_clmulepi64_si128(left, right, 0x11);                \
    REDUCE_PAIR_SSE2(target, p_0, p_1);                                   \
}

#define GF2_64_MUL_PAIR_SCALAR_SSE2_PCLMULQDQ(target, source, scalar)     \
{                                                                         \
    __m128i rhs = _mm_set_epi64x(0, scalar);                              \
    __m128i p_0 = _mm_clmulepi64_si128(source, rhs, 0x00);                \
    __m128i p_1 = _mm_clmulepi64_si128(source, rhs, 0x01);                \
    REDUCE_PAIR_SSE2(target, p_0, p_1);                                   \
}

#define GF2_64_MUL_SSE2_PCLMULQDQ(target, left, right)                    \
{                                                                         \
    __m128i lhs = _mm_set_epi64x(0, left);                                \
    __m128i rhs = _mm_set_epi64x(0, right);                               \
    __m128i p_0 = _mm_clmulepi64_si128(lhs, rhs, 0x00);                   \
    __m128i t;                                                            \
    REDUCE_PAIR_SSE2(t,p_0,_mm_setzero_si128());                          \
    target = _mm_extract_epi64(t, 0);                                     \
}


/************************************ Bit-sliced multiplication subroutines. */

/*
 * A simplified bit-sliced Mastrovito multiplier for GF(2^8)
 * (141 gates)
 *
 * Modulus:  [x^8 +] x^4 + x^3 + x^2 + 1            ~ 0x1D = 29 (primitive)
 *
 * First few powers of the generator & degrees of nonzero monomials
 *
 *  0: 0x01 0
 *  1: 0x02 1
 *  2: 0x04 2
 *  3: 0x08 3
 *  4: 0x10 4
 *  5: 0x20 5
 *  6: 0x40 6
 *  7: 0x80 7
 * -------------------
 *  8: 0x1D 0 2 3 4
 *  9: 0x3A 1 3 4 5
 * 10: 0x74 2 4 5 6
 * 11: 0xE8 3 5 6 7
 * 12: 0xCD 0 2 3 6 7
 * 13: 0x87 0 1 2 7
 * 14: 0x13 0 1 4
 * 
 */

#define GF2_8_MUL_SLICE(z0,z1,z2,z3,z4,z5,z6,z7,x0,x1,x2,x3,x4,x5,x6,x7,y0,y1,y2,y3,y4,y5,y6,y7)\
{\
    slice_t a;\
    slice_t t8;\
    slice_t t9;\
    slice_t t10;\
    slice_t t11;\
    slice_t t12;\
    slice_t t13;\
    slice_t t14;\
    z0  = x0  &  y0;\
    z1  = x0  &  y1;\
    a   = x1  &  y0;\
    z1  = z1  ^  a;\
    z2  = x0  &  y2;\
    a   = x1  &  y1;\
    z2  = z2  ^  a;\
    a   = x2  &  y0;\
    z2  = z2  ^  a;\
    z3  = x0  &  y3;\
    a   = x1  &  y2;\
    z3  = z3  ^  a;\
    a   = x2  &  y1;\
    z3  = z3  ^  a;\
    a   = x3  &  y0;\
    z3  = z3  ^  a;\
    z4  = x0  &  y4;\
    a   = x1  &  y3;\
    z4  = z4  ^  a;\
    a   = x2  &  y2;\
    z4  = z4  ^  a;\
    a   = x3  &  y1;\
    z4  = z4  ^  a;\
    a   = x4  &  y0;\
    z4  = z4  ^  a;\
    z5  = x0  &  y5;\
    a   = x1  &  y4;\
    z5  = z5  ^  a;\
    a   = x2  &  y3;\
    z5  = z5  ^  a;\
    a   = x3  &  y2;\
    z5  = z5  ^  a;\
    a   = x4  &  y1;\
    z5  = z5  ^  a;\
    a   = x5  &  y0;\
    z5  = z5  ^  a;\
    z6  = x0  &  y6;\
    a   = x1  &  y5;\
    z6  = z6  ^  a;\
    a   = x2  &  y4;\
    z6  = z6  ^  a;\
    a   = x3  &  y3;\
    z6  = z6  ^  a;\
    a   = x4  &  y2;\
    z6  = z6  ^  a;\
    a   = x5  &  y1;\
    z6  = z6  ^  a;\
    a   = x6  &  y0;\
    z6  = z6  ^  a;\
    z7  = x0  &  y7;\
    a   = x1  &  y6;\
    z7  = z7  ^  a;\
    a   = x2  &  y5;\
    z7  = z7  ^  a;\
    a   = x3  &  y4;\
    z7  = z7  ^  a;\
    a   = x4  &  y3;\
    z7  = z7  ^  a;\
    a   = x5  &  y2;\
    z7  = z7  ^  a;\
    a   = x6  &  y1;\
    z7  = z7  ^  a;\
    a   = x7  &  y0;\
    z7  = z7  ^  a;\
    t8  = x1  &  y7;\
    a   = x2  &  y6;\
    t8  = t8  ^  a;\
    a   = x3  &  y5;\
    t8  = t8  ^  a;\
    a   = x4  &  y4;\
    t8  = t8  ^  a;\
    a   = x5  &  y3;\
    t8  = t8  ^  a;\
    a   = x6  &  y2;\
    t8  = t8  ^  a;\
    a   = x7  &  y1;\
    t8  = t8  ^  a;\
    t9  = x2  &  y7;\
    a   = x3  &  y6;\
    t9  = t9  ^  a;\
    a   = x4  &  y5;\
    t9  = t9  ^  a;\
    a   = x5  &  y4;\
    t9  = t9  ^  a;\
    a   = x6  &  y3;\
    t9  = t9  ^  a;\
    a   = x7  &  y2;\
    t9  = t9  ^  a;\
    t10 = x3  &  y7;\
    a   = x4  &  y6;\
    t10 = t10 ^  a;\
    a   = x5  &  y5;\
    t10 = t10 ^  a;\
    a   = x6  &  y4;\
    t10 = t10 ^  a;\
    a   = x7  &  y3;\
    t10 = t10 ^  a;\
    t11 = x4  &  y7;\
    a   = x5  &  y6;\
    t11 = t11 ^  a;\
    a   = x6  &  y5;\
    t11 = t11 ^  a;\
    a   = x7  &  y4;\
    t11 = t11 ^  a;\
    t12 = x5  &  y7;\
    a   = x6  &  y6;\
    t12 = t12 ^  a;\
    a   = x7  &  y5;\
    t12 = t12 ^  a;\
    t13 = x6  &  y7;\
    a   = x7  &  y6;\
    t13 = t13 ^  a;\
    t14 = x7  &  y7;\
    z0  = z0  ^  t8;\
    z2  = z2  ^  t8;\
    z3  = z3  ^  t8;\
    z4  = z4  ^  t8;\
    z1  = z1  ^  t9;\
    z3  = z3  ^  t9;\
    z4  = z4  ^  t9;\
    z5  = z5  ^  t9;\
    z2  = z2  ^  t10;\
    z4  = z4  ^  t10;\
    z5  = z5  ^  t10;\
    z6  = z6  ^  t10;\
    z3  = z3  ^  t11;\
    z5  = z5  ^  t11;\
    z6  = z6  ^  t11;\
    z7  = z7  ^  t11;\
    z0  = z0  ^  t12;\
    z2  = z2  ^  t12;\
    z3  = z3  ^  t12;\
    z6  = z6  ^  t12;\
    z7  = z7  ^  t12;\
    z0  = z0  ^  t13;\
    z1  = z1  ^  t13;\
    z2  = z2  ^  t13;\
    z7  = z7  ^  t13;\
    z0  = z0  ^  t14;\
    z1  = z1  ^  t14;\
    z4  = z4  ^  t14;\
}

/******************************************** Reference/baseline subroutines. */

inline scalar_t gf2_add_ref(scalar_t x, scalar_t y)
{   
    return x^y;
}

#define REF_SCALAR_ADD(target, left, right) { target = gf2_add_ref(left, right); }

#if GF2_B == 8

// Modulus:  [x^8 +] x^4 + x^3 + x^2 + 1            ~ 0x1D = 29 (primitive)

inline scalar_t gf2_8_mul_ref(scalar_t x, scalar_t y)
{   
    scalar_t z = 0;
    for(int i = 0; i < 8; i++) {
        scalar_t f = (scalar_t) (x & 0x080);
        if(y & 1)
            z ^= x;
        y = y >> 1;
        x = (x&0x07F) << 1;
        if(f)
            x ^= (scalar_t) GF2_8_MODULUS;
    }
    return z;
}

#define REF_SCALAR_MUL(target, left, right) { target = gf2_8_mul_ref(left, right); }

#define SCALAR_FORMAT_STRING "0x%02lX"

#endif

#if GF2_B == 32

// Modulus: [x^32 +] x^7 + x^5 + x^3 + x^2 + x + 1   ~ 0xAF = 175 (primitive)

inline scalar_t gf2_32_mul_ref(scalar_t x, scalar_t y)
{   
    scalar_t z = 0;
    for(int i = 0; i < 32; i++) {
        scalar_t f = (scalar_t) (x & 0x80000000);
        if(y & 1)
            z ^= x;
        y >>= 1;
        x <<= 1;
        if(f)
            x ^= (scalar_t) GF2_32_MODULUS;
    }
    return z;
}

#define REF_SCALAR_MUL(target, left, right) { target = gf2_32_mul_ref(left, right); }

#define SCALAR_FORMAT_STRING "0x%08lX"

#endif

#if GF2_B == 64
#define REF_SCALAR_MUL(target, left, right) { target = gf2_64_mul_ref(left, right); }

// Modulus: [x^64 +] x^4 + x^3 + x + 1               ~ 0x1B = 27 (primitive)

inline scalar_t gf2_64_mul_ref(scalar_t x, scalar_t y)
{   
    scalar_t z = 0;
    for(int i = 0; i < 64; i++) {
        scalar_t f = (scalar_t) (x & 0x8000000000000000L);
        if(y & 1)
            z ^= x;
        y >>= 1;
        x <<= 1;
        if(f)
            x ^= (scalar_t) GF2_64_MODULUS;
    }
    return z;
}

#define SCALAR_FORMAT_STRING "0x%016lX"

#endif


scalar_t gf2pow(scalar_t g, scalar_t amount) 
{
    assert(amount >= 0);
    if(amount == 0)
        return 1;
    if(amount == 1)
        return g;
    scalar_t gg;
    SCALAR_MUL(gg, g, g);
    if(amount & 1) {
        scalar_t t;
        t = gf2pow(gg, amount/2);
        scalar_t tt;
        SCALAR_MUL(tt, g, t);
        return tt;
    } else {
        return gf2pow(gg, amount/2);
    }
}

scalar_t gf2inv(scalar_t g) 
{
    assert(g != 0);
    return gf2pow(g, (scalar_t) -2); 
}

#define SCALAR_INV(gg, g)                                                 \
{                                                                         \
    gg = gf2inv(g);                                                       \
}


// Returns the degree-j coefficient of the degree d Lagrange
// polynomial p_{d,k}(z) that has value 1 at point z=k+1 
// and value 0 at all points z=l+1 with l=0,1,...,k-1,k+1,...,deg-1.

#define MAX_LAGRANGE 128

scalar_t lagrange_coeff(index_t d, index_t k, index_t j)
{
    scalar_t coeff[MAX_LAGRANGE+1];

    assert(k >= 0 && k <= d);
    assert(j >= 0 && j <= d);
    assert(d < MAX_LAGRANGE);

    for(index_t i = 0; i <= d; i++)
        coeff[i] = (i == 0) ? 1 : 0;
    scalar_t zk = k+1;
    for(index_t l = 0; l <= d; l++) {
        if(l == k)
            continue;
        scalar_t zl = l+1;
        scalar_t g, denom;
        SCALAR_ADD(denom, zk, zl);
        SCALAR_INV(g, denom);
        for(index_t i = d; i >= 0; i--) {
            scalar_t q;
            SCALAR_MUL(q, coeff[i], g);
            SCALAR_MUL(q, q, zl);
            if(i > 0) {
                scalar_t qq;
                SCALAR_MUL(qq, coeff[i-1], g);
                SCALAR_ADD(q, q, qq);
            }
            coeff[i] = q;
        }
    }
    return coeff[j];
}



/********************************************************** Line definitions. */

typedef limb_t  line_t[LIMBS_IN_LINE];   
typedef limb_t  line_array_t;

#define SCALARS_IN_LINE (LIMBS_IN_LINE*SCALARS_IN_LIMB)

#define LINE_ARRAY_SIZE(b) (sizeof(scalar_t)*(size_t)(b))

#define LINE_ARRAY_LOAD_SCALAR(target,ptr,idx)                            \
{                                                                         \
    LIMB_ARRAY_LOAD_SCALAR(target,ptr,idx);                               \
}

#define LINE_ARRAY_STORE_SCALAR(ptr,idx,source)                           \
{                                                                         \
    LIMB_ARRAY_STORE_SCALAR(ptr,idx,source);                              \
}

#define LINE_PREFETCH(ptr,idx)                                            \
{                                                                         \
    LIMB_PREFETCH(ptr,LIMBS_IN_LINE*(idx)+0);                             \
}


#if LIMBS_IN_LINE == 2UL
#define LINE_LOAD(target,ptr,idx)                                         \
        LIMB_LOAD(target[0],ptr,LIMBS_IN_LINE*(idx)+0);                   \
        LIMB_LOAD(target[1],ptr,LIMBS_IN_LINE*(idx)+1);
#else
#if LIMBS_IN_LINE == 4UL
#define LINE_LOAD(target,ptr,idx)                                         \
        LIMB_LOAD(target[0],ptr,LIMBS_IN_LINE*(idx)+0);                   \
        LIMB_LOAD(target[1],ptr,LIMBS_IN_LINE*(idx)+1);                   \
        LIMB_LOAD(target[2],ptr,LIMBS_IN_LINE*(idx)+2);                   \
        LIMB_LOAD(target[3],ptr,LIMBS_IN_LINE*(idx)+3);
#else
#if LIMBS_IN_LINE == 8UL
#define LINE_LOAD(target,ptr,idx)                                         \
        LIMB_LOAD(target[0],ptr,LIMBS_IN_LINE*(idx)+0);                   \
        LIMB_LOAD(target[1],ptr,LIMBS_IN_LINE*(idx)+1);                   \
        LIMB_LOAD(target[2],ptr,LIMBS_IN_LINE*(idx)+2);                   \
        LIMB_LOAD(target[3],ptr,LIMBS_IN_LINE*(idx)+3);                   \
        LIMB_LOAD(target[4],ptr,LIMBS_IN_LINE*(idx)+4);                   \
        LIMB_LOAD(target[5],ptr,LIMBS_IN_LINE*(idx)+5);                   \
        LIMB_LOAD(target[6],ptr,LIMBS_IN_LINE*(idx)+6);                   \
        LIMB_LOAD(target[7],ptr,LIMBS_IN_LINE*(idx)+7);
#else
#define LINE_LOAD(target,ptr,idx)                                         \
{                                                                         \
    for(index_t ell = 0; ell < LIMBS_IN_LINE; ell++) {                    \
        LIMB_LOAD(target[ell],ptr,LIMBS_IN_LINE*(idx)+ell);               \
    }                                                                     \
}
#endif
#endif
#endif

#define LINE_STORE(ptr,idx,source)                                        \
{                                                                         \
    for(index_t ell = 0; ell < LIMBS_IN_LINE; ell++) {                    \
        LIMB_STORE(ptr,LIMBS_IN_LINE*(idx)+ell,source[ell]);              \
    }                                                                     \
}

#define LINE_MOV(target, source)                                          \
{                                                                         \
    for(index_t ell = 0; ell < LIMBS_IN_LINE; ell++) {                    \
        LIMB_MOV(target[ell],source[ell]);                                \
    }                                                                     \
}

#define LINE_ADD(target, left, right)                                     \
{                                                                         \
    for(index_t ell = 0; ell < LIMBS_IN_LINE; ell++) {                    \
        LIMB_ADD(target[ell],left[ell],right[ell]);                       \
    }                                                                     \
}

#if LIMBS_IN_LINE == 2UL
#define LINE_MUL(target, left, right)                                     \
        LIMB_MUL(target[0],left[0],right[0]);                             \
        LIMB_MUL(target[1],left[1],right[1]);                       
#else
#if LIMBS_IN_LINE == 4UL
#define LINE_MUL(target, left, right)                                     \
        LIMB_MUL(target[0],left[0],right[0]);                             \
        LIMB_MUL(target[1],left[1],right[1]);                             \
        LIMB_MUL(target[2],left[2],right[2]);                             \
        LIMB_MUL(target[3],left[3],right[3]);                       
#else
#if LIMBS_IN_LINE == 8UL
#define LINE_MUL(target, left, right)                                     \
        LIMB_MUL(target[0],left[0],right[0]);                             \
        LIMB_MUL(target[1],left[1],right[1]);                             \
        LIMB_MUL(target[2],left[2],right[2]);                             \
        LIMB_MUL(target[3],left[3],right[3]);                             \
        LIMB_MUL(target[4],left[4],right[4]);                             \
        LIMB_MUL(target[5],left[5],right[5]);                             \
        LIMB_MUL(target[6],left[6],right[6]);                             \
        LIMB_MUL(target[7],left[7],right[7]);                       
#else
#if LIMBS_IN_LINE == 16UL
#define LINE_MUL(target, left, right)                                     \
        LIMB_MUL(target[0],left[0],right[0]);                             \
        LIMB_MUL(target[1],left[1],right[1]);                             \
        LIMB_MUL(target[2],left[2],right[2]);                             \
        LIMB_MUL(target[3],left[3],right[3]);                             \
        LIMB_MUL(target[4],left[4],right[4]);                             \
        LIMB_MUL(target[5],left[5],right[5]);                             \
        LIMB_MUL(target[6],left[6],right[6]);                             \
        LIMB_MUL(target[7],left[7],right[7]);                             \
        LIMB_MUL(target[8],left[8],right[8]);                             \
        LIMB_MUL(target[9],left[9],right[9]);                             \
        LIMB_MUL(target[10],left[10],right[10]);                          \
        LIMB_MUL(target[11],left[11],right[11]);                          \
        LIMB_MUL(target[12],left[12],right[12]);                          \
        LIMB_MUL(target[13],left[13],right[13]);                          \
        LIMB_MUL(target[14],left[14],right[14]);                          \
        LIMB_MUL(target[15],left[15],right[15]);                       
#else
#if LIMBS_IN_LINE == 32UL
#define LINE_MUL(target, left, right)                                     \
        LIMB_MUL(target[0],left[0],right[0]);                             \
        LIMB_MUL(target[1],left[1],right[1]);                             \
        LIMB_MUL(target[2],left[2],right[2]);                             \
        LIMB_MUL(target[3],left[3],right[3]);                             \
        LIMB_MUL(target[4],left[4],right[4]);                             \
        LIMB_MUL(target[5],left[5],right[5]);                             \
        LIMB_MUL(target[6],left[6],right[6]);                             \
        LIMB_MUL(target[7],left[7],right[7]);                             \
        LIMB_MUL(target[8],left[8],right[8]);                             \
        LIMB_MUL(target[9],left[9],right[9]);                             \
        LIMB_MUL(target[10],left[10],right[10]);                          \
        LIMB_MUL(target[11],left[11],right[11]);                          \
        LIMB_MUL(target[12],left[12],right[12]);                          \
        LIMB_MUL(target[13],left[13],right[13]);                          \
        LIMB_MUL(target[14],left[14],right[14]);                          \
        LIMB_MUL(target[15],left[15],right[15]);                          \
        LIMB_MUL(target[16],left[16],right[16]);                          \
        LIMB_MUL(target[17],left[17],right[17]);                          \
        LIMB_MUL(target[18],left[18],right[18]);                          \
        LIMB_MUL(target[19],left[19],right[19]);                          \
        LIMB_MUL(target[20],left[20],right[20]);                          \
        LIMB_MUL(target[21],left[21],right[21]);                          \
        LIMB_MUL(target[22],left[22],right[22]);                          \
        LIMB_MUL(target[23],left[23],right[23]);                          \
        LIMB_MUL(target[24],left[24],right[24]);                          \
        LIMB_MUL(target[25],left[25],right[25]);                          \
        LIMB_MUL(target[26],left[26],right[26]);                          \
        LIMB_MUL(target[27],left[27],right[27]);                          \
        LIMB_MUL(target[28],left[28],right[28]);                          \
        LIMB_MUL(target[29],left[29],right[29]);                          \
        LIMB_MUL(target[30],left[30],right[30]);                          \
        LIMB_MUL(target[31],left[31],right[31]);                       
#else
#define LINE_MUL(target, left, right)                                     \
{                                                                         \
    for(index_t ell = 0; ell < LIMBS_IN_LINE; ell++) {                    \
        LIMB_MUL(target[ell],left[ell],right[ell]);                       \
    }                                                                     \
}
#endif
#endif
#endif
#endif
#endif

#define LINE_MUL_SCALAR(target, source, scalar)                           \
{                                                                         \
    for(index_t ell = 0; ell < LIMBS_IN_LINE; ell++) {                    \
        LIMB_MUL_SCALAR(target[ell],source[ell],scalar);                  \
    }                                                                     \
}
    
#define LINE_SUM(target, source)                                          \
{                                                                         \
    scalar_t stmp0, stmp1;                                                \
    SCALAR_SET_ZERO(stmp0);                                               \
    for(index_t ell = 0; ell < LIMBS_IN_LINE; ell++) {                    \
        LIMB_SUM(stmp1, source[ell]);                                     \
        SCALAR_ADD(stmp0, stmp0, stmp1);                                  \
    }                                                                     \
    target = stmp0;                                                       \
}

#define LINE_SET_ZERO(target)                                             \
{                                                                         \
    for(index_t ell = 0; ell < LIMBS_IN_LINE; ell++) {                    \
        LIMB_SET_ZERO(target[ell]);                                       \
    }                                                                     \
}

#define LINE_SET_ONE(target)                                              \
{                                                                         \
    for(index_t ell = 0; ell < LIMBS_IN_LINE; ell++) {                    \
        LIMB_SET_ONE(target[ell]);                                        \
    }                                                                     \
}

#define LINE_STORE_SCALAR(target,idx,source)                               \
{                                                                          \
    for(index_t ell = 0; ell < LIMBS_IN_LINE; ell++) {                     \
        if((idx) / SCALARS_IN_LIMB == ell) {                               \
            LIMB_STORE_SCALAR(target[ell], (idx)%SCALARS_IN_LIMB, source); \
        }                                                                  \
    }                                                                      \
}



/************************************************************ Precomputation. */

#ifdef LIMB_1_GF2_8_EXPLOG

typedef scalar_t scalar_tbl_t;

scalar_tbl_t gf2_8_lookup_exp[512];
scalar_tbl_t gf2_8_lookup_log[256];

void gf2_8_precompute_exp_log(void)
{
    scalar_t v, g;
    v = 0x01;
    g = 0x02;   // modulus is primitive so 0x02 == x generates the mult group   
    for(index_t i = 0; i < 511; i++) {
        gf2_8_lookup_exp[i] = v;
        if(i < 256)
            gf2_8_lookup_log[v] = i;
        REF_SCALAR_MUL(v, v, g);
    }
}

#undef GF_PRECOMPUTE
#define GF_PRECOMPUTE                                  \
{                                                      \
    gf2_8_precompute_exp_log();                        \
}

#endif


#endif

