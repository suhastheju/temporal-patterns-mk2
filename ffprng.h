/* 
 * `Finding path motifs in large temporal graphs using algebraic fingerprints`
 *
 * This experimental source code is supplied to accompany the 
 * aforementioned paper. 
 * 
 * The source code is configured for a gcc build to a native 
 * microarchitecture that must support the AVX2 and PCLMULQDQ 
 * instruction set extensions. Other builds are possible but 
 * require manual configuration of 'Makefile' and 'builds.h'.
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

#ifndef FFPRNG_H
#define FFPRNG_H

#include<assert.h>

/**************** A quick-and-dirty fast-forward pseudorandom number generator (PRNG). */

/* Idea: XOR two LFSRs, one clocked from left to right, the other from right to left. */

/*
 * The primitive polynomial used to construct GF(2^{64})
 *
 *   :64  x^{64}+                   -
 * 63:60  x^{62}+x^{60}+            5
 * 59:56  x^{58}+x^{56}+            5
 * 55:52  x^{55}+x^{53}+x^{52}+     B
 * 51:48  x^{51}+x^{50}+x^{48}+     D
 * 47:44  x^{46}+x^{45}+x^{44}+     7
 * 43:40  x^{43}+x^{41}+x^{40}+     B
 * 39:36  x^{39}+                   8
 * 35:32  x^{35}+x^{34}+            C
 * 31:28  x^{31}+x^{30}+x^{29}+     E
 * 27:24  x^{27}+x^{25}+x^{24}+     B
 * 23:20  x^{22}+x^{21}+x^{20}+     7
 * 19:16  x^{19}+x^{17}+x^{16}+     B
 * 15:12  x^{15}+x^{12}+            9
 * 11:8   x^{11}+x^{9}+             A
 * 7:4    x^{7}+x^{6}+x^{5}+        E
 * 3:0    x^{3}+1                   9
 *
 * http://fchabaud.free.fr/English/default.php?COUNT=3&FILE0=Poly&FILE1=GF(2)&FILE2=Primitive
 *
 */

#define FFPRNG_MODULUS         0x55BD7B8CEB7B9AE9UL
#define FFPRNG_MODULUS_REVERSE 0x9759DED731DEBDAAUL

typedef unsigned long int ffprng_scalar_t;

#define FFPRNG_SCALAR_MSB      0x8000000000000000L

typedef struct 
{
    ffprng_scalar_t   state_left;
    ffprng_scalar_t   state_right;
} ffprng_t;

// Reverse an N-bit quantity, from
// https://graphics.stanford.edu/~seander/bithacks.html

#define FFPRNG_REVERSE64(t)\
{\
    t = ((t >>  1) & 0x5555555555555555L)|((t & 0x5555555555555555L) <<  1);\
    t = ((t >>  2) & 0x3333333333333333L)|((t & 0x3333333333333333L) <<  2);\
    t = ((t >>  4) & 0x0F0F0F0F0F0F0F0FL)|((t & 0x0F0F0F0F0F0F0F0FL) <<  4);\
    t = ((t >>  8) & 0x00FF00FF00FF00FFL)|((t & 0x00FF00FF00FF00FFL) <<  8);\
    t = ((t >> 16) & 0x0000FFFF0000FFFFL)|((t & 0x0000FFFF0000FFFFL) << 16);\
    t = ((t >> 32) & 0x00000000FFFFFFFFL)|((t & 0x00000000FFFFFFFFL) << 32);\
}

#define FFPRNG_INIT(ctx, origin)\
{                                                       \
    ffprng_scalar_t t = (origin);                       \
    ctx.state_left  = t^0x1234567890ABCDEFL;            \
    FFPRNG_REVERSE64(t);                                \
    ctx.state_right = t^0xFEDCBA0987654321L;            \
}

#define FFPRNG_RAND(out, ctx)\
{                                                                      \
    ffprng_scalar_t tl = ctx.state_left;                               \
    ffprng_scalar_t tr = ctx.state_right;                              \
    out = tl^tr;                                                       \
    ffprng_scalar_t fl = tl & 0x8000000000000000L;                     \
    tl = (tl << 1)^(((((signed long) fl) >> 63))&FFPRNG_MODULUS);      \
    ffprng_scalar_t fr = tr & 0x0000000000000001L;                     \
    tr = (tr >> 1)^((-fr)&FFPRNG_MODULUS_REVERSE);                     \
    ctx.state_left  = tl;                                              \
    ctx.state_right = tr;                                              \
}

#ifdef FFPRNG_SSE2_PCLMULQDQ
#define FFPRNG_MUL(target, left, right)\
    FFPRNG_GF2_64_MUL_SSE2_PCLMULQDQ(target, left, right);
#else
#define FFPRNG_MUL(target, left, right)\
    target = ffprng_gf2_64_mul(left, right);
#endif

ffprng_scalar_t ffprng_gf2_64_mul(ffprng_scalar_t x, ffprng_scalar_t y)
{   
    ffprng_scalar_t z = 0;
    for(int i = 0; i < 64; i++) {
        ffprng_scalar_t f = x & FFPRNG_SCALAR_MSB;
        if(y & 1)
            z ^= x;
        y >>= 1;
        x <<= 1;
        if(f)
            x ^= FFPRNG_MODULUS;
    }
    return z;
}

ffprng_scalar_t ffprng_gf2pow(ffprng_scalar_t g, index_t amount) 
{
    assert(amount >= 0);
    if(amount == 0)
        return 1;
    if(amount == 1)
        return g;
    ffprng_scalar_t gg;
    FFPRNG_MUL(gg, g, g);
    if(amount & 1) {
        ffprng_scalar_t t;
        t = ffprng_gf2pow(gg, amount/2);
        ffprng_scalar_t tt;
        FFPRNG_MUL(tt, g, t);
        return tt;
    } else {
        return ffprng_gf2pow(gg, amount/2);
    }
}

#define FFPRNG_FWD(ctx, amount, base)                  \
{                                                      \
    ffprng_scalar_t tt = ffprng_gf2pow(0x02L, amount); \
    ffprng_scalar_t tl = base.state_left;              \
    ffprng_scalar_t tr = base.state_right;             \
    FFPRNG_REVERSE64(tr);                              \
    ffprng_scalar_t ttl, ttr;                          \
    FFPRNG_MUL(ttl, tt, tl);                           \
    FFPRNG_MUL(ttr, tt, tr);                           \
    FFPRNG_REVERSE64(ttr);                             \
    ctx.state_left  = ttl;                             \
    ctx.state_right = ttr;                             \
}

#endif
