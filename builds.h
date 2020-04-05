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
 * Copyright (c) 2014 A. Björklund, P. Kaski,L. Kowalik, J. Lauri
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

// Default build (needs AVX2 and PCLMULQDQ)

#ifdef LISTER_DEFAULT
#define BUILD_PARALLEL
#define BUILD_PREFETCH
#define LIMB_4_GF2_64_AVX2_PCLMULQDQ
#define LIMBS_IN_LINE    2UL
#define BUILD_GENF       2
#endif

#ifdef LISTER_PAR_GENF1
#define BUILD_PARALLEL
#define BUILD_PREFETCH
#define LIMB_4_GF2_64_AVX2_PCLMULQDQ
#define LIMBS_IN_LINE    2UL
#define BUILD_GENF       1
#endif

#ifdef LISTER_PAR_GENF2
#define BUILD_PARALLEL
#define BUILD_PREFETCH
#define LIMB_4_GF2_64_AVX2_PCLMULQDQ
#define LIMBS_IN_LINE    2UL
#define BUILD_GENF       2
#endif

#ifdef LISTER_SIN_GENF1
#define BUILD_PREFETCH
#define LIMB_4_GF2_64_AVX2_PCLMULQDQ
#define LIMBS_IN_LINE    2UL
#define BUILD_GENF       1
#endif

#ifdef LISTER_SIN_GENF2
#define BUILD_PREFETCH
#define LIMB_4_GF2_64_AVX2_PCLMULQDQ
#define LIMBS_IN_LINE    2UL
#define BUILD_GENF       2
#endif


// 64-bit scalar builds

#ifdef LISTER_1x64_UNROLL
#define LIMB_1_GF2_64_UNROLL
#define LIMBS_IN_LINE    1UL
#define BUILD_GENF       1
#endif

#ifdef LISTER_4x64_UNROLL
#define LIMB_1_GF2_64_UNROLL
#define LIMBS_IN_LINE    4UL
#define BUILD_GENF       1
#endif

#ifdef LISTER_8x64_UNROLL
#define LIMB_1_GF2_64_UNROLL
#define LIMBS_IN_LINE    8UL
#define BUILD_GENF       1
#endif

#ifdef LISTER_1x64_PACK
#define LIMB_1_GF2_64_SSE2_PCLMULQDQ
#define LIMBS_IN_LINE    1UL
#define BUILD_GENF       1
#endif

#ifdef LISTER_4x64_PACK
#define LIMB_4_GF2_64_AVX2_PCLMULQDQ
#define LIMBS_IN_LINE    1UL
#define BUILD_GENF       1
#endif

#ifdef LISTER_8x64_PACK
#define LIMB_4_GF2_64_AVX2_PCLMULQDQ
#define LIMBS_IN_LINE    2UL
#define BUILD_GENF       1
#endif

#ifdef LISTER_8x64_MEMSAVE
#define LIMB_4_GF2_64_AVX2_PCLMULQDQ
#define LIMBS_IN_LINE    2UL
#define BUILD_GENF       2
#endif

// 8-bit scalar builds

#ifdef LISTER_1x8_EXPLOG
#define LIMB_1_GF2_8_EXPLOG
#define LIMBS_IN_LINE    1UL
#define BUILD_GENF       1
#endif

#ifdef LISTER_32x8_EXPLOG
#define LIMB_1_GF2_8_EXPLOG
#define LIMBS_IN_LINE    32UL
#define BUILD_GENF       1
#endif

#ifdef LISTER_32x8_SLICE
#define LIMB_32_GF2_8
#define LIMBS_IN_LINE    1UL
#define BUILD_GENF       1
#endif

#ifdef LISTER_64x8_SLICE
#define LIMB_64_GF2_8
#define LIMBS_IN_LINE    1UL
#define BUILD_GENF       1
#endif

#ifdef LISTER_32x8_PACK
#define LIMB_32_GF2_8_AVX2
#define LIMBS_IN_LINE    1UL
#define BUILD_GENF       1
#endif

#ifdef LISTER_64x8_PACK
#define LIMB_32_GF2_8_AVX2
#define LIMBS_IN_LINE    2UL
#define BUILD_GENF       1
#endif

#ifdef LISTER_32x8_MEMSAVE
#define LIMB_32_GF2_8_AVX2
#define LIMBS_IN_LINE    1UL
#define BUILD_GENF       2
#endif

