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

#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<time.h>
#include<sys/utsname.h>
#include<string.h>
#include<stdarg.h>
#include<assert.h>
#include<ctype.h>
#include<omp.h>

/************************************************************* Configuration. */
#define MAX_K          32
#define MAX_SHADES     32

#define PREFETCH_PAD   32
#define MAX_THREADS   128

#define UNDEFINED -1
#define MATH_INF ((index_t)0x3FFFFFFF)

#include"builds.h"        // get build config

typedef long int index_t; // default to 64-bit indexing

#include"gf.h"       // finite fields
#include"ffprng.h"   // fast-forward pseudorandom number generator


#define MIN(x,y) (x)<(y) ? (x) : (y)
#define MAX(x,y) (x)>(y) ? (x) : (y)

/********************************************************************* Flags. */

index_t flag_bin_input    = 0; // default to ASCII input

/************************************************************* Common macros. */

/* Linked list navigation macros. */

#define pnlinknext(to,el) { (el)->next = (to)->next; (el)->prev = (to); (to)->next->prev = (el); (to)->next = (el); }
#define pnlinkprev(to,el) { (el)->prev = (to)->prev; (el)->next = (to); (to)->prev->next = (el); (to)->prev = (el); }
#define pnunlink(el) { (el)->next->prev = (el)->prev; (el)->prev->next = (el)->next; }
#define pnrelink(el) { (el)->next->prev = (el); (el)->prev->next = (el); }


/*********************************************************** Error reporting. */

#define ERROR(...) error(__FILE__,__LINE__,__func__,__VA_ARGS__);

static void error(const char *fn, int line, const char *func, 
                  const char *format, ...)
{
    va_list args;
    va_start(args, format);
    fprintf(stderr, 
            "ERROR [file = %s, line = %d]\n"
            "%s: ",
            fn,
            line,
            func);
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
    abort();    
}

/********************************************************* Get the host name. */

#define MAX_HOSTNAME 256

const char *sysdep_hostname(void)
{
    static char hn[MAX_HOSTNAME];

    struct utsname undata;
    uname(&undata);
    strcpy(hn, undata.nodename);
    return hn;
}

/********************************************************* Available threads. */

index_t num_threads(void)
{
#ifdef BUILD_PARALLEL
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/********************************************** Memory allocation & tracking. */

#define MALLOC(x) malloc_wrapper(x)
#define FREE(x) free_wrapper(x)

index_t malloc_balance = 0;

struct malloc_track_struct
{
    void *p;
    size_t size;
    struct malloc_track_struct *prev;
    struct malloc_track_struct *next;
};

typedef struct malloc_track_struct malloc_track_t;

malloc_track_t malloc_track_root;
size_t malloc_total = 0;

#define MEMTRACK_STACK_CAPACITY 256
size_t memtrack_stack[MEMTRACK_STACK_CAPACITY];
index_t memtrack_stack_top = -1;

void *malloc_wrapper(size_t size)
{
    if(malloc_balance == 0) {
        malloc_track_root.prev = &malloc_track_root;
        malloc_track_root.next = &malloc_track_root;
    }
    void *p = malloc(size);
    if(p == NULL)
        ERROR("malloc fails");
    malloc_balance++;

    malloc_track_t *t = (malloc_track_t *) malloc(sizeof(malloc_track_t));
    t->p = p;
    t->size = size;
    pnlinkprev(&malloc_track_root, t);
    malloc_total += size;
    for(index_t i = 0; i <= memtrack_stack_top; i++)
        if(memtrack_stack[i] < malloc_total)
            memtrack_stack[i] = malloc_total;    
    return p;
}

void free_wrapper(void *p)
{
    malloc_track_t *t = malloc_track_root.next;
    for(;
        t != &malloc_track_root;
        t = t->next) {
        if(t->p == p)
            break;
    }
    if(t == &malloc_track_root)
        ERROR("FREE issued on a non-tracked pointer %p", p);
    malloc_total -= t->size;
    pnunlink(t);
    free(t);
    
    free(p);
    malloc_balance--;
}

index_t *alloc_idxtab(index_t n)
{
    index_t *t = (index_t *) MALLOC(sizeof(index_t)*n);
    return t;
}

void push_memtrack(void) 
{
    assert(memtrack_stack_top + 1 < MEMTRACK_STACK_CAPACITY);
    memtrack_stack[++memtrack_stack_top] = malloc_total;
}

size_t pop_memtrack(void)
{
    assert(memtrack_stack_top >= 0);
    return memtrack_stack[memtrack_stack_top--];    
}

size_t current_mem(void)
{
    return malloc_total;
}

double inGiB(size_t s) 
{
    return (double) s / (1 << 30);
}

void print_current_mem(void)
{
    fprintf(stdout, "{curr: %.2lfGiB}", inGiB(current_mem()));
    fflush(stdout);
}

void print_pop_memtrack(void)
{
    fprintf(stdout, "{peak: %.2lfGiB}", inGiB(pop_memtrack()));
    fflush(stdout);
}

/******************************************************** Timing subroutines. */

#define TIME_STACK_CAPACITY 256
double start_stack[TIME_STACK_CAPACITY];
index_t start_stack_top = -1;

void push_time(void) 
{
    assert(start_stack_top + 1 < TIME_STACK_CAPACITY);
    start_stack[++start_stack_top] = omp_get_wtime();
}

double pop_time(void)
{
    double wstop = omp_get_wtime();
    assert(start_stack_top >= 0);
    double wstart = start_stack[start_stack_top--];
    return (double) (1000.0*(wstop-wstart));
}

/******************************************************************* Sorting. */

void shellsort(index_t n, index_t *a)
{
    index_t h = 1;
    index_t i;
    for(i = n/3; h < i; h = 3*h+1)
        ;
    do {
        for(i = h; i < n; i++) {
            index_t v = a[i];
            index_t j = i;
            do {
                index_t t = a[j-h];
                if(t <= v)
                    break;
                a[j] = t;
                j -= h;
            } while(j >= h);
            a[j] = v;
        }
        h /= 3;
    } while(h > 0);
}

#define LEFT(x)      (x<<1)
#define RIGHT(x)     ((x<<1)+1)
#define PARENT(x)    (x>>1)

void heapsort_indext(index_t n, index_t *a)
{
    /* Shift index origin from 0 to 1 for convenience. */
    a--; 
    /* Build heap */
    for(index_t i = 2; i <= n; i++) {
        index_t x = i;
        while(x > 1) {
            index_t y = PARENT(x);
            if(a[x] <= a[y]) {
                /* heap property ok */
                break;              
            }
            /* Exchange a[x] and a[y] to enforce heap property */
            index_t t = a[x];
            a[x] = a[y];
            a[y] = t;
            x = y;
        }
    }

    /* Repeat delete max and insert */
    for(index_t i = n; i > 1; i--) {
        index_t t = a[i];
        /* Delete max */
        a[i] = a[1];
        /* Insert t */
        index_t x = 1;
        index_t y, z;
        while((y = LEFT(x)) < i) {
            z = RIGHT(x);
            if(z < i && a[y] < a[z]) {
                index_t s = z;
                z = y;
                y = s;
            }
            /* Invariant: a[y] >= a[z] */
            if(t >= a[y]) {
                /* ok to insert here without violating heap property */
                break;
            }
            /* Move a[y] up the heap */
            a[x] = a[y];
            x = y;
        }
        /* Insert here */
        a[x] = t; 
    }
}

/******************************************************* Bitmap manipulation. */

void bitset(index_t *map, index_t j, index_t value)
{
    assert((value & (~1UL)) == 0);
    map[j/64] = (map[j/64] & ~(1UL << (j%64))) | ((value&1) << (j%64));  
}

index_t bitget(index_t *map, index_t j)
{
    return (map[j/64]>>(j%64))&1UL;
}

/*************************************************** Random numbers and such. */

index_t irand(void)
{
    return (((index_t) rand())<<31)^((index_t) rand());
}

void randshuffle_seq(index_t n, index_t *p, ffprng_t gen)
{
    for(index_t i = 0; i < n-1; i++) {
        ffprng_scalar_t rnd;
        FFPRNG_RAND(rnd, gen);
        index_t x = i+(rnd%(n-i));
        index_t t = p[x];
        p[x] = p[i];
        p[i] = t;
    }
}

void randperm(index_t n, index_t seed, index_t *p)
{
#ifdef BUILD_PARALLEL
    index_t nt = 64;
#else
    index_t nt = 1;
#endif
    index_t block_size = n/nt;
    index_t f[128][128];
    assert(nt < 128);

    ffprng_t base;
    FFPRNG_INIT(base, seed);
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t t = 0; t < nt; t++) {
        for(index_t j = 0; j < nt; j++)
            f[t][j] = 0;
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? n-1 : (start+block_size-1);
        ffprng_t gen;
        FFPRNG_FWD(gen, start, base);
        for(index_t i = start; i <= stop; i++) {
            ffprng_scalar_t rnd;
            FFPRNG_RAND(rnd, gen);
            index_t bin = (index_t) ((unsigned long) rnd)%((unsigned long)nt);
            f[t][bin]++;
        }
    }

    for(index_t bin = 0; bin < nt; bin++) {
        for(index_t t = 1; t < nt; t++) {
            f[0][bin] += f[t][bin];
        }
    }
    index_t run = 0;
    for(index_t j = 1; j <= nt; j++) {
        index_t fp = f[0][j-1];
        f[0][j-1] = run;
        run += fp;
    }
    f[0][nt] = run;

    FFPRNG_INIT(base, seed);
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t t = 0; t < nt; t++) {
        ffprng_t gen;
        index_t start = 0;
        index_t stop = n-1;
        index_t pos = f[0][t];
        FFPRNG_FWD(gen, start, base);
        for(index_t i = start; i <= stop; i++) {
            ffprng_scalar_t rnd;
            FFPRNG_RAND(rnd, gen);
            index_t bin = (index_t) ((unsigned long) rnd)%((unsigned long)nt);
            if(bin == t)
                p[pos++] = i;
        }
        assert(pos == f[0][t+1]);
    }


    FFPRNG_INIT(base, (seed^0x9078563412EFDCABL));
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t t = 0; t < nt; t++) {
        ffprng_t fwd, gen;
        index_t start = f[0][t];
        index_t stop = f[0][t+1]-1;
        index_t u;
        FFPRNG_FWD(fwd, (1234567890123456L*t), base);
        FFPRNG_RAND(u, fwd);
        FFPRNG_INIT(gen, u);
        randshuffle_seq(stop-start+1, p + start, gen);
    }
}

/********************************** Initialize an array with random scalars. */

void randinits_scalar(scalar_t *a, index_t s, ffprng_scalar_t seed) 
{
    ffprng_t base;
    FFPRNG_INIT(base, seed);
    index_t nt = num_threads();
    index_t block_size = s/nt;
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t t = 0; t < nt; t++) {
        ffprng_t gen;
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? s-1 : (start+block_size-1);
        FFPRNG_FWD(gen, start, base);
        for(index_t i = start; i <= stop; i++) {
            ffprng_scalar_t rnd;
            FFPRNG_RAND(rnd, gen);
            scalar_t rs = (scalar_t) rnd;           
            a[i] = rs;
        }
    }
}

/***************************************************** (Parallel) prefix sum. */

index_t prefixsum(index_t n, index_t *a, index_t k)
{

#ifdef BUILD_PARALLEL
    index_t s[MAX_THREADS];
    index_t nt = num_threads();
    assert(nt < MAX_THREADS);

    index_t length = n;
    index_t block_size = length/nt;

#pragma omp parallel for
    for(index_t t = 0; t < nt; t++) {
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? length-1 : (start+block_size-1);
        index_t tsum = (stop-start+1)*k;
        for(index_t u = start; u <= stop; u++)
            tsum += a[u];
        s[t] = tsum;
    }

    index_t run = 0;
    for(index_t t = 1; t <= nt; t++) {
        index_t v = s[t-1];
        s[t-1] = run;
        run += v;
    }
    s[nt] = run;

#pragma omp parallel for
    for(index_t t = 0; t < nt; t++) {
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? length-1 : (start+block_size-1);
        index_t trun = s[t];
        for(index_t u = start; u <= stop; u++) {
            index_t tv = a[u];
            a[u] = trun;
            trun += tv + k;
        }
        assert(trun == s[t+1]);    
    }

#else

    index_t run = 0;
    for(index_t u = 0; u < n; u++) {
        index_t tv = a[u];
        a[u] = run;
        run += tv + k;
    }

#endif

    return run; 
}

/************************************************************* Parallel sum. */

index_t parallelsum(index_t n, index_t *a)
{
    index_t sum = 0;
#ifdef BUILD_PARALLEL
    index_t s[MAX_THREADS];
    index_t nt = num_threads();
    assert(nt < MAX_THREADS);

    index_t length = n;
    index_t block_size = length/nt;

#pragma omp parallel for
    for(index_t t = 0; t < nt; t++) {
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? length-1 : (start+block_size-1);
        index_t tsum = 0;
        for(index_t u = start; u <= stop; u++)
            tsum += a[u];
        s[t] = tsum;
    }

    for(index_t t = 0; t < nt; t++)
        sum += s[t];
#else
    for(index_t i = 0; i < n; i++) {
        sum += a[i];
    }
#endif
    return sum;
}

// count number of non-zero values in an array
index_t parallelcount(index_t n, index_t *a)
{
    index_t total_cnt = 0;
#ifdef BUILD_PARALLEL
    index_t nt = num_threads();
    index_t block_size = n/nt;
    index_t *cnt_nt = alloc_idxtab(nt);
#pragma omp parallel for
    for(index_t th = 0; th <nt; th++) {
        index_t start = th*block_size;
        index_t stop = (th == nt-1) ? n-1 : (start+block_size-1);
        index_t cnt = 0;
        for(index_t i = start; i <= stop; i++)
            cnt += (a[i] ? 1 : 0);

        cnt_nt[th] = cnt;
    }

    for(index_t th = 0; th < nt; th++)
        total_cnt += cnt_nt[th];  
#else
    for(index_t i = 0; i < n; i++)
        total_cnt += (a[i] ? 1 : 0);
#endif
    return total_cnt;
}


/************************ Search for an interval of values in a sorted array. */

index_t get_interval(index_t n, index_t *a, 
                            index_t lo_val, index_t hi_val,
                            index_t *iv_start, index_t *iv_end)
{
    assert(n >= 0);
    if(n == 0) {
        *iv_start = 0; 
        return 0;
    }
    assert(lo_val <= hi_val);
    // find first element in interval (if any) with binary search
    index_t lo = 0;
    index_t hi = n-1;
    // at or above lo, and at or below hi (if any)
    while(lo < hi) {
        index_t mid = (lo+hi)/2; // lo <= mid < hi
        index_t v = a[mid];
        if(hi_val < v) {
            hi = mid-1;     // at or below hi (if any)
        } else {
            if(v < lo_val)
                lo = mid+1; // at or above lo (if any), lo <= hi
            else
                hi = mid;   // at or below hi (exists) 
        }
        // 0 <= lo <= n-1
    }
    if(a[lo] < lo_val || a[lo] > hi_val) {
        // array contains no values in interval
        if(a[lo] < lo_val) {
            lo++;
            assert(lo == n || a[lo+1] > hi_val);
        } else {
            assert(lo == 0 || a[lo-1] < lo_val);
        }
        *iv_start = lo; 
        *iv_end   = hi;
        return 0; 
    }
    assert(lo_val <= a[lo] && a[lo] <= hi_val);
    *iv_start = lo;
    // find interval end (last index in interval) with binary search
    lo = 0;
    hi = n-1;
    // last index (if any) is at or above lo, and at or below hi
    while(lo < hi) {
        index_t mid = (lo+hi+1)/2; // lo < mid <= hi
        index_t v = a[mid];
        if(hi_val < v) {
            hi = mid-1;     // at or below hi, lo <= hi
        } else {
            if(v < lo_val)
                lo = mid+1; // at or above lo
            else
                lo = mid;   // at or above lo, lo <= hi
        }
    }
    assert(lo == hi);
    *iv_end = lo; // lo == hi
    return 1+*iv_end-*iv_start; // return cut size
}


/******************************************************************** Stack. */

typedef struct stack_node {
    index_t u;
    index_t l;
    index_t t;
} stack_node_t;

typedef struct stack {
    index_t size; // size of stack
    index_t n; // number of elements
    stack_node_t *a;
}stk_t;

stk_t * stack_alloc(index_t size)
{
    stk_t *s = (stk_t *) malloc(sizeof(stk_t)); 
    s->size = size;
    s->n = 0;
    s->a = (stack_node_t *) malloc(s->size*sizeof(stack_node_t));

#ifdef DEBUG
    for(index_t i = 0; i < s->n; i++) {
        stack_node_t *e = s->a + i;
        e->u = UNDEFINED;
        e->l = UNDEFINED;
        e->t = UNDEFINED;
    }
#endif
    return s;
}

void stack_free(stk_t *s)
{
    free(s->a);
    free(s);
}

void stack_push(stk_t *s, stack_node_t *e_in)
{
    assert(s->n < s->size);
    stack_node_t *e = s->a + s->n;
    e->u = e_in->u;
    //e->l = e_in->l;
    e->t = e_in->t;
    s->n++;
}

void stack_pop(stk_t *s, stack_node_t *e_out)
{
    assert(s->n > 0);
    s->n--;
    stack_node_t *e = s->a + s->n;
    e_out->u = e->u;
    //e_out->l = e->l;
    e_out->t = e->t;

#ifdef DEBUG
    e->u = UNDEFINED;
    //e->l = UNDEFINED;
    e->t = UNDEFINED;
#endif
}

void stack_top(stk_t *s, stack_node_t *e_out)
{
    assert(s->n >= 0);
    stack_node_t *e = s->a + s->n-1;
    e_out->u = e->u;
    e_out->l = e->l;
    e_out->t = e->t;
}

void stack_empty(stk_t *s)
{
    s->n = 0;
}

void stack_get_vertices(stk_t *s, index_t *uu)
{
    for(index_t i = 0; i < s->n; i++) {
        stack_node_t *e = s->a + i;
        uu[i] = e->u;
    }
}

void stack_get_timestamps(stk_t *s, index_t *tt)
{
    for(index_t i = 0; i < s->n; i++) {
        stack_node_t *e = s->a + i;
        tt[i] = e->t;
    }
}

#ifdef DEBUG
void print_stack(stk_t *s)
{
    fprintf(stdout, "-----------------------------------------------\n");
    fprintf(stdout, "print stack\n");
    fprintf(stdout, "-----------------------------------------------\n");
    fprintf(stdout, "size: %ld\n", s->size);
    fprintf(stdout, "n: %ld\n", s->n);
    fprintf(stdout, "a: ");
    for(index_t i = 0; i < s->n; i++) {
        stack_node_t *e = s->a + i;
        fprintf(stdout, "[%ld, %ld, %ld]%s", e->u, e->l, e->t, (i==s->n-1)?"\n":" ");
    }
    fprintf(stdout, "-----------------------------------------------\n");
}

void print_stacknode(stack_node_t *e)
{
    fprintf(stdout, "print stack-node: [%ld, %ld, %ld]\n", e->u, e->l, e->t);
}

#endif

/****************************************************************** Sieving. */

long long int num_muls;
long long int trans_bytes;

#define SHADE_LINES ((MAX_SHADES+SCALARS_IN_LINE-1)/SCALARS_IN_LINE)
typedef unsigned int shade_map_t;

void constrained_sieve_pre(index_t         n,
                           index_t         k,
                           index_t         g,
                           index_t         pfx,
                           index_t         num_shades,
                           shade_map_t     *d_s,
                           ffprng_scalar_t seed,
                           line_array_t    *d_x)
{
    assert(g == SCALARS_IN_LINE);   
    assert(num_shades <= MAX_SHADES);

    line_t   wdj[SHADE_LINES*MAX_K];

    ffprng_t base;
    FFPRNG_INIT(base, seed);
    for(index_t j = 0; j < k; j++) {
        for(index_t dl = 0; dl < SHADE_LINES; dl++) {
            index_t jsdl = j*SHADE_LINES+dl;
            LINE_SET_ZERO(wdj[jsdl]);
            for(index_t a = 0; a < SCALARS_IN_LINE; a++) {
                ffprng_scalar_t rnd;
                FFPRNG_RAND(rnd, base);
                scalar_t rs = (scalar_t) rnd;
                LINE_STORE_SCALAR(wdj[jsdl], a, rs);   // W: [cached]
            }
        }
    }

    index_t nt = num_threads();
    index_t length = n;
    index_t block_size = length/nt;
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t t = 0; t < nt; t++) {
        ffprng_t gen;
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? length-1 : (start+block_size-1);
        FFPRNG_FWD(gen, SHADE_LINES*SCALARS_IN_LINE*start, base);
        line_t vd[SHADE_LINES];
        for(index_t j = 0; j < SHADE_LINES; j++) {
            LINE_SET_ZERO(vd[j]); // to cure an annoying compiler warning
        }       
        for(index_t u = start; u <= stop; u++) {
            scalar_t uu[MAX_K];
            shade_map_t shades_u = d_s[u];            // R: n   shade_map_t
            for(index_t dl = 0; dl < SHADE_LINES; dl++) {
                for(index_t a = 0; a < SCALARS_IN_LINE; a++) {
                    index_t d = dl*SCALARS_IN_LINE + a;
                    ffprng_scalar_t rnd;
                    FFPRNG_RAND(rnd, gen);
                    scalar_t rs = (scalar_t) rnd;
                    rs = rs & (-((scalar_t)((shades_u >> d)&(d < num_shades))));  
                    LINE_STORE_SCALAR(vd[dl], a, rs); // W: [cached]
                }
            }
            for(index_t j = 0; j < k; j++) {
                scalar_t uj;
                SCALAR_SET_ZERO(uj);
                for(index_t dl = 0; dl < SHADE_LINES; dl++) {
                    index_t jsdl = j*SHADE_LINES+dl;
                    line_t ln;
                    LINE_MUL(ln, wdj[jsdl], vd[dl]);  // R: [cached]
                                                      // MUL: n*SHADE_LINES*g*k
                    scalar_t lns;
                    LINE_SUM(lns, ln);
                    SCALAR_ADD(uj, uj, lns);
                }
                uu[j] = uj;
            }
            line_t ln;
            LINE_SET_ZERO(ln);
            for(index_t a = 0; a < SCALARS_IN_LINE; a++) {
                index_t ap = a < (1L << k) ? pfx+a : 0;
                scalar_t xua;
                SCALAR_SET_ZERO(xua);
                for(index_t j = 0; j < k; j++) {
                    scalar_t z_uj = uu[j];            // R: [cached]
                    z_uj = z_uj & (-((scalar_t)(((ap) >> j)&1)));
                    SCALAR_ADD(xua, xua, z_uj);
                }
                LINE_STORE_SCALAR(ln, a, xua);
            }
            LINE_STORE(d_x, u, ln);                  // W: ng scalar_t
        }
    }

    num_muls    += n*SHADE_LINES*g*k;
    trans_bytes += sizeof(scalar_t)*n*g + sizeof(shade_map_t)*n;
}

/***************************************************************** Line sum. */

scalar_t line_sum(index_t      l, 
                  index_t      g,
                  line_array_t *d_s)
{

    index_t nt = num_threads();
    index_t block_size = l/nt;
    assert(nt < MAX_THREADS);
    scalar_t ts[MAX_THREADS];
    
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t t = 0; t < nt; t++) {
        SCALAR_SET_ZERO(ts[t]);
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? l-1 : (start+block_size-1);
        line_t ln;
        line_t acc;
        LINE_SET_ZERO(acc);
        for(index_t i = start; i <= stop; i++) {    
            LINE_LOAD(ln, d_s, i);    // R: lg scalar_t
            LINE_ADD(acc, acc, ln);
        }
        scalar_t lsum;
        LINE_SUM(lsum, acc);
        ts[t] = lsum;
    }
    scalar_t sum;
    SCALAR_SET_ZERO(sum);
    for(index_t t = 0; t < nt; t++) {
        SCALAR_ADD(sum, sum, ts[t]);
    }

    trans_bytes += sizeof(scalar_t)*l*g;
    return sum;
}

void vertex_acc(index_t      l,
                index_t      g,
                index_t      stride,
                line_array_t *d_s,
                scalar_t     *out)
{
    index_t nt = num_threads();
    index_t block_size = l/nt;
    assert(nt < MAX_THREADS);
    //scalar_t ts[MAX_THREADS];

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t th = 0; th < nt; th++) {
        //SCALAR_SET_ZERO(ts[t]);
        index_t start = th*block_size;
        index_t stop = (th == nt-1) ? l-1 : (start+block_size-1);
        line_t ln;
        scalar_t lsum;
        for(index_t i = start; i <= stop; i++) {
            LINE_LOAD(ln, d_s, i);    // R: lg scalar_t
            LINE_SUM(lsum, ln);
            out[i] = lsum;            // R: scalar_t,  W: scalar_t
        }
    }
    //scalar_t sum;
    //SCALAR_SET_ZERO(sum);
    //for(index_t t = 0; t < nt; t++) {
    //    SCALAR_ADD(sum, sum, ts[t]);
    //}

    trans_bytes += sizeof(scalar_t)*(l*g+2);
}


scalar_t line_sum_stride(index_t      l, 
                         index_t      g,
                         index_t      stride,
                         line_array_t *d_s)
{

    index_t nt = num_threads();
    index_t block_size = l/nt;
    assert(nt < MAX_THREADS);
    scalar_t ts[MAX_THREADS];
    
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t th = 0; th < nt; th++) {
        SCALAR_SET_ZERO(ts[th]);
        index_t start = th*block_size;
        index_t stop = (th == nt-1) ? l-1 : (start+block_size-1);
        line_t ln;
        line_t acc;
        LINE_SET_ZERO(acc);
        for(index_t i = start; i <= stop; i++) {    
            index_t ii = i*stride;
            LINE_LOAD(ln, d_s, ii);    // R: lg scalar_t
            LINE_ADD(acc, acc, ln);
        }
        scalar_t lsum;
        LINE_SUM(lsum, acc);
        ts[th] = lsum;
    }
    scalar_t sum;
    SCALAR_SET_ZERO(sum);
    for(index_t th = 0; th < nt; th++) {
        SCALAR_ADD(sum, sum, ts[th]);
    }

    trans_bytes += sizeof(scalar_t)*l*g;

    return sum;
}

void vertex_acc_stride(index_t      l,
                       index_t      g,
                       index_t      stride,
                       line_array_t *d_s,
                       scalar_t     *out)
{
    index_t nt = num_threads();
    index_t block_size = l/nt;
    assert(nt < MAX_THREADS);
    //scalar_t ts[MAX_THREADS];

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t th = 0; th < nt; th++) {
        //SCALAR_SET_ZERO(ts[th]);
        index_t start = th*block_size;
        index_t stop = (th == nt-1) ? l-1 : (start+block_size-1);
        line_t ln;
        scalar_t lsum;
        for(index_t i = start; i <= stop; i++) {
            index_t ii = i*stride;
            LINE_LOAD(ln, d_s, ii);    // R: lg scalar_t
            LINE_SUM(lsum, ln);
            out[i] = lsum;            // R: scalar_t,  W: scalar_t
        }
    }
    //scalar_t sum;
    //SCALAR_SET_ZERO(sum);
    //for(index_t th = 0; th < nt; th++) {
    //    SCALAR_ADD(sum, sum, ts[th]);
    //}

    trans_bytes += sizeof(scalar_t)*(l*g+2);
}

/********************************** k-temppath generating function (mark 1). */

#if BUILD_GENF == 1

#define TEMP_PATH_LINE_IDX(n, k, tmax, l, t, u) (((l-1)*(tmax+1)*(n))+((n)*(t))+(u))

#ifdef DEBUG
#define PRINT_LINE(source)                                                  \
{                                                                           \
    scalar_t *s = (scalar_t *)&source;                                      \
    for(index_t i = 0; i < SCALARS_IN_LINE; i++) {                          \
        fprintf(stdout, SCALAR_FORMAT_STRING"%s",                           \
                        (long) s[i],                                        \
                        i==SCALARS_IN_LINE-1 ? "\n":" ");                   \
    }                                                                       \
}

void print_dx(index_t n,
              line_array_t *d_x)
{
    fprintf(stdout, "d_x:\n");
    for(index_t u = 0; u < n; u ++) {
        line_t xu;
        LINE_LOAD(xu, d_x, u);
        fprintf(stdout, "%ld: ", u);
        PRINT_LINE(xu);
    }
}

void print_ds(index_t n,
              index_t k, 
              index_t tmax,
              line_array_t *d_s)
{
    fprintf(stdout, "d_s: \n");

    for(index_t l = 1; l <= k; l++) {
        fprintf(stdout, "--------------------------------------------------\n");
        fprintf(stdout, "--------------------------------------------------\n");
        fprintf(stdout, "l: %ld\n", l);
        fprintf(stdout, "--------------------------------------------------\n");
        fprintf(stdout, "--------------------------------------------------\n");
        for(index_t t = 0; t <= tmax; t++) {
            fprintf(stdout, "--------------------------------------------------\n");
            fprintf(stdout, "t: %ld\n", t);
            fprintf(stdout, "--------------------------------------------------\n");
            for(index_t u = 0; u < n; u++) {
                fprintf(stdout, "%ld: ", u+1);
                index_t i_ult = TEMP_PATH_LINE_IDX(n, k, tmax, l, t, u);
                line_t p_ult;
                LINE_LOAD(p_ult, d_s, i_ult);
                PRINT_LINE(p_ult);
                scalar_t sum;
                LINE_SUM(sum, p_ult);
                fprintf(stdout, "line sum: "SCALAR_FORMAT_STRING"\n",sum);
            }
        }
    }
    
}
#endif 


void init_ds_genf1(index_t n,
                   index_t k,
                   index_t tmax,
                   line_array_t *d_s)
{
    line_t p_zero;
    LINE_SET_ZERO(p_zero);
    for(index_t l = 1; l <= k; l++) {
        for(index_t t = 0; t <= tmax; t++) {
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
            for(index_t u = 0; u < n; u++) {
                index_t i_ult = TEMP_PATH_LINE_IDX(n, k, tmax, l, t, u);
                LINE_STORE(d_s, i_ult, p_zero);      // W: ng  scalar_t
            }
        }
    }
}

void k_temp_path_genf1_round(index_t n,
                             index_t m,
                             index_t k,
                             index_t tmax,
                             index_t t,
                             index_t g,
                             index_t l,
                             index_t *d_pos,
                             index_t *d_adj,
                             index_t yl_seed,
                             line_array_t *d_x,
                             line_array_t *d_s)
{
    assert(g == SCALARS_IN_LINE);

    index_t nt = num_threads();
    index_t length = n;
    index_t block_size = length/nt;

    ffprng_t y_base;
    FFPRNG_INIT(y_base, yl_seed);

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t th = 0; th < nt; th++) {
        index_t start = th*block_size;
        index_t stop = (th == nt-1) ? length-1 : (start+block_size-1);
        ffprng_t y_gen;
        index_t y_pos = d_pos[(t-1)*n+start]-((t-1)*n+start);
        FFPRNG_FWD(y_gen, y_pos, y_base);
        for(index_t u = start; u <= stop; u++) {
            index_t pu  = d_pos[n*(t-1)+u];             
            index_t deg = d_adj[pu];               
            line_t p_ult;
            LINE_SET_ZERO(p_ult);
            for(index_t j = 1; j <= deg; j++) {
                index_t v = d_adj[pu+j];          
                line_t p_vl1t1;
                index_t i_vl1t1 = TEMP_PATH_LINE_IDX(n, k, tmax, l-1, t-1, v);
                LINE_LOAD(p_vl1t1, d_s, i_vl1t1);

#ifdef BUILD_PREFETCH
                // prefetch next line
                index_t nv = d_adj[pu+j+(j < deg ? 1 : 2)];
                index_t i_nvl1t1 = TEMP_PATH_LINE_IDX(n, k, tmax, l-1, t-1, nv);
                LINE_PREFETCH(d_s, i_nvl1t1);
#endif
                ffprng_scalar_t rnd;
                FFPRNG_RAND(rnd, y_gen);
                scalar_t y_luvt = (scalar_t) rnd;
                line_t sy;
                LINE_MUL_SCALAR(sy, p_vl1t1, y_luvt);
                LINE_ADD(p_ult, p_ult, sy);
            }
            line_t xu;
            LINE_LOAD(xu, d_x, u);
            LINE_MUL(p_ult, p_ult, xu);

            line_t p_ult1;
            index_t i_ult1 = TEMP_PATH_LINE_IDX(n, k, tmax, l, t-1, u);
            LINE_LOAD(p_ult1, d_s, i_ult1);

            LINE_ADD(p_ult, p_ult, p_ult1);
            index_t i_ult = TEMP_PATH_LINE_IDX(n, k, tmax, l, t, u);
            LINE_STORE(d_s, i_ult, p_ult);      // W: ng  scalar_t
        }
    }

    // total edges at time `t`
    index_t m_t = d_pos[n*(t-1) + n-1] - d_pos[n*(t-1)] - (n-1) + 
                  d_adj[d_pos[n*(t-1)+(n-1)]];
    trans_bytes += ((2*n*tmax)+m_t)*sizeof(index_t) + (2*n+m_t)*g*sizeof(scalar_t);
    num_muls    += (n*g+m_t);
}

scalar_t k_temp_path_genf1(index_t         n,
                           index_t         m,
                           index_t         k,
                           index_t         tmax,
                           index_t         g,
                           index_t         vert_loc,
                           index_t         *d_pos,
                           index_t         *d_adj, 
                           ffprng_scalar_t y_seed,
                           line_array_t    *d_x,
                           scalar_t        *vs) 
{
    assert( g == SCALARS_IN_LINE);
    assert( k >= 1);

    line_array_t *d_s= (line_array_t *) MALLOC(LINE_ARRAY_SIZE(k*(tmax+1)*n*g));

    init_ds_genf1(n, k, tmax, d_s);

    // initialise: l = 1
    for(index_t u = 0; u < n; u++) {
        for(index_t t = 0; t <= tmax; t++) {
            line_t xu;
            LINE_LOAD(xu, d_x, u);
            index_t i_u_1 = TEMP_PATH_LINE_IDX(n, k, tmax, 1, t, u);
            LINE_STORE(d_s, i_u_1, xu);
        }
    }

    srand(y_seed);
    for(index_t l = 2; l <= k; l++) {
        ffprng_scalar_t yl_seed = irand(); // new seed for each l
        for(index_t t = l-1; t <= tmax; t++) {
            k_temp_path_genf1_round(n, m, k, tmax, t, g, l, 
                                    d_pos, d_adj, yl_seed, d_x, d_s);
        }
    }

    // sum up
    index_t ii = TEMP_PATH_LINE_IDX(n, k, tmax, k, tmax, 0);
    scalar_t sum = line_sum(n, g, ((line_array_t *)(((line_t *) d_s)+ii)));

    // vertex-localisation
    if(vert_loc) {
        vertex_acc(n, g, k, ((line_array_t *)(((line_t *) d_s)+ii)), vs);
    }

    //print_ds(n, k, tmax, d_s);
    // free memory
    FREE(d_s);

    return sum;
}

#endif

/*********************************** k-temppath generating function (mark 2) */

#if BUILD_GENF == 2

#define TEMP_PATH_LINE_IDX2(n, k, tmax, l, t, u) (((n)*(t))+(u))


void init_ds_genf2(index_t n,
                   index_t k,
                   index_t tmax,
                   line_array_t *d_s)
{
    line_t p_zero;
    LINE_SET_ZERO(p_zero);
    for(index_t t = 0; t <= tmax; t++) {
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
        for(index_t u = 0; u < n; u++) {
            index_t i_ult = TEMP_PATH_LINE_IDX2(n, k, tmax, 1, t, u);
            LINE_STORE(d_s, i_ult, p_zero);      // W: ng  scalar_t
        }
    }
}

void k_temp_path_genf2_round(index_t n,
                            index_t m,
                            index_t k,
                            index_t tmax,
                            index_t t,
                            index_t g,
                            index_t l,
                            index_t *d_pos,
                            index_t *d_adj,
                            index_t yl_seed,
                            line_array_t *d_x,
                            line_array_t *d_s1,
                            line_array_t *d_s2)
{
    assert(g == SCALARS_IN_LINE);

    index_t nt = num_threads();
    index_t length = n;
    index_t block_size = length/nt;

    ffprng_t y_base;
    FFPRNG_INIT(y_base, yl_seed);

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t th = 0; th < nt; th++) {
        index_t start = th*block_size;
        index_t stop = (th == nt-1) ? length-1 : (start+block_size-1);
        ffprng_t y_gen;
        index_t y_pos = d_pos[(t-1)*n+start]-((t-1)*n+start);
        FFPRNG_FWD(y_gen, y_pos, y_base);
        for(index_t u = start; u <= stop; u++) {
            index_t pu  = d_pos[n*(t-1)+u];             
            index_t deg = d_adj[pu];               
            line_t p_ult;
            LINE_SET_ZERO(p_ult);
            for(index_t j = 1; j <= deg; j++) {
                index_t v = d_adj[pu+j];          
                line_t p_vl1t1;
                index_t i_vl1t1 = TEMP_PATH_LINE_IDX2(n, k, tmax, l-1, t-1, v);
                LINE_LOAD(p_vl1t1, d_s1, i_vl1t1);

#ifdef BUILD_PREFETCH
                // prefetch next line
                index_t nv = d_adj[pu+j+(j < deg ? 1 : 2)];
                index_t i_nvl1t1 = TEMP_PATH_LINE_IDX2(n, k, tmax, l-1, t-1, nv);
                LINE_PREFETCH(d_s1, i_nvl1t1);
#endif
                ffprng_scalar_t rnd;
                FFPRNG_RAND(rnd, y_gen);
                scalar_t y_luvt = (scalar_t) rnd;
                line_t sy;
                LINE_MUL_SCALAR(sy, p_vl1t1, y_luvt);
                LINE_ADD(p_ult, p_ult, sy);
            }
            line_t xu;
            LINE_LOAD(xu, d_x, u);
            LINE_MUL(p_ult, p_ult, xu);

            line_t p_ult1;
            index_t i_ult1 = TEMP_PATH_LINE_IDX2(n, k, tmax, l, t-1, u);
            LINE_LOAD(p_ult1, d_s2, i_ult1);

            LINE_ADD(p_ult, p_ult, p_ult1);
            index_t i_ult = TEMP_PATH_LINE_IDX2(n, k, tmax, l, t, u);
            LINE_STORE(d_s2, i_ult, p_ult);      // W: ng  scalar_t
        }
    }

    // total edges at time `t`
    index_t m_t = d_pos[n*(t-1) + n-1] - d_pos[n*(t-1)] - (n-1) + 
                  d_adj[d_pos[n*(t-1)+(n-1)]];
    trans_bytes += ((2*n*tmax)+m_t)*sizeof(index_t) + (2*n+m_t)*g*sizeof(scalar_t);
    num_muls    += (n*g+m_t);
}



scalar_t k_temp_path_genf2(index_t         n,
                           index_t         m,
                           index_t         k,
                           index_t         tmax,
                           index_t         g,
                           index_t         vert_loc,
                           index_t         *d_pos,
                           index_t         *d_adj, 
                           ffprng_scalar_t y_seed,
                           line_array_t    *d_x,
                           scalar_t        *vs) 
{
    assert( g == SCALARS_IN_LINE);
    assert( k >= 1);

    line_array_t *d_s1= (line_array_t *) MALLOC(LINE_ARRAY_SIZE((tmax+1)*n*g));
    line_array_t *d_s2= (line_array_t *) MALLOC(LINE_ARRAY_SIZE((tmax+1)*n*g));

    init_ds_genf2(n, 1, tmax, d_s2);
    // initialise: l = 1
    for(index_t u = 0; u < n; u++) {
        for(index_t t = 0; t <= tmax; t++) {
            line_t xu;
            LINE_LOAD(xu, d_x, u);
            index_t i_u_1 = TEMP_PATH_LINE_IDX2(n, k, tmax, 1, t, u);
            LINE_STORE(d_s1, i_u_1, xu);
        }
    }

    srand(y_seed);
    for(index_t l = 2; l <= k; l++) {
        ffprng_scalar_t yl_seed = irand(); // new seed for each l
        for(index_t t = l-1; t <= tmax; t++) {
            k_temp_path_genf2_round(n, m, k, tmax, t, g, l, 
                                    d_pos, d_adj, yl_seed, d_x, 
                                    d_s1, d_s2);
        }

        //fprintf(stdout, "\n\n----------------\n");
        //fprintf(stdout, "l : %ld\n", l);
        //fprintf(stdout, "d_s1 before init\n");
        //print_ds(n, 1, tmax, d_s1);
        //fprintf(stdout, "\n\n----------------\n");
        //fprintf(stdout, "d_s2 before init\n");
        //print_ds(n, 1, tmax, d_s2);

        // swap and initialise
        line_array_t *d_temp = d_s1;
        d_s1 = d_s2;
        d_s2 = d_temp;
        init_ds_genf2(n, 1, tmax, d_s2);

        //fprintf(stdout, "\n\n----------------\n");
        //fprintf(stdout, "d_s1 after init\n");
        //print_ds(n, 1, tmax, d_s1);
        //fprintf(stdout, "\n\n----------------\n");
        //fprintf(stdout, "d_s1 after init\n");
        //print_ds(n, 1, tmax, d_s2);
    }

    // sum up
    index_t ii = TEMP_PATH_LINE_IDX2(n, k, tmax, 1, tmax, 0);
    scalar_t sum = line_sum(n, g, ((line_array_t *)(((line_t *) d_s1)+ii)));

    // vertex-localisation
    if(vert_loc) {
        vertex_acc(n, g, k, ((line_array_t *)(((line_t *) d_s1)+ii)), vs);
    }

    // free memory
    FREE(d_s1);
    FREE(d_s2);

    return sum;
}
#endif

/************************************************************ The oracle(s). */

index_t temppath_oracle(index_t         n,
                        index_t         k,
                        index_t         tmax,
                        index_t         *h_pos,
                        index_t         *h_adj,
                        index_t         num_shades,
                        shade_map_t     *h_s,
                        ffprng_scalar_t y_seed,
                        ffprng_scalar_t z_seed,
                        index_t         vert_loc,
                        scalar_t        *master_vsum) 
{
    push_memtrack();
    assert(k >= 1 && k < 31);
    //index_t m = h_pos[n-1]+h_adj[h_pos[n-1]]+1-n;
    index_t m = h_pos[n*(tmax-1)+n-1]+h_adj[h_pos[n*(tmax-1)+n-1]]+1-(n*tmax);
    index_t sum_size = 1 << k;       

    index_t g = SCALARS_IN_LINE;
    index_t outer = (sum_size + g-1) / g; 
    // number of iterations for outer loop

    num_muls = 0;
    trans_bytes = 0;

    index_t *d_pos     = h_pos;
    index_t *d_adj     = h_adj;
    line_array_t *d_x  = (line_array_t *) MALLOC(LINE_ARRAY_SIZE(n*g));

    /* Run the work & time it. */
    push_time();

    scalar_t master_sum;
    SCALAR_SET_ZERO(master_sum);

    if(vert_loc) {
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
        for(index_t i = 0; i < n; i++)
            master_vsum[i] = 0;
    }
        
    for(index_t out = 0; out < outer; out++) {

        constrained_sieve_pre(n, k, g, g*out, num_shades, h_s, z_seed, d_x);
#if BUILD_GENF == 1
#define GENF_TYPE "k_temp_path_genf1"
        scalar_t sum = k_temp_path_genf1(n, m, k, tmax, g, vert_loc, d_pos, d_adj, y_seed, d_x, master_vsum);
#elif BUILD_GENF == 2
#define GENF_TYPE "k_temp_path_genf2"
        scalar_t sum = k_temp_path_genf2(n, m, k, tmax, g, vert_loc, d_pos, d_adj, y_seed, d_x, master_vsum);
#else
#error BUILD_GENF should be either 1 or 2
#endif

        SCALAR_ADD(master_sum, master_sum, sum);
    }

    double time = pop_time();
    double trans_rate = trans_bytes / (time/1000.0);
    double mul_rate = num_muls / time;
    FREE(d_x);

    fprintf(stdout, 
            SCALAR_FORMAT_STRING
            " %.2lf ms [%.2lfGiB/s, %.2lfGHz] %d",
            (long) master_sum,
            time,
            trans_rate/((double) (1 << 30)),
            mul_rate/((double) 1e6),
            master_sum != 0);
    fprintf(stdout, " ");
    print_pop_memtrack();
    fprintf(stdout, " ");   
    print_current_mem();   
    fflush(stdout);

    return master_sum != 0;
}

/************************************** k-path generating function (mark 1). */

#if BUILD_GENF == 1

#define PATH_LINE_IDX(b, k, l, u) ((k)*(u)+(l)-1)

#ifdef DEBUG
void print_kpath_ds(index_t n,
                    index_t k,
                    line_array_t *d_s)
{
    for(index_t l = 1; l <= k; l++) {
        fprintf(stdout,"-------------------------------------------------\n");
        fprintf(stdout, "l: %ld\n", l);
        fprintf(stdout,"-------------------------------------------------\n");
        for(index_t u = 0; u < n; u++) {
            fprintf(stdout, "%ld: ", u+1);
            index_t i_u_l = PATH_LINE_IDX(b, k, l, u);
            line_t pul;
            LINE_LOAD(pul, d_s, i_u_l);
            PRINT_LINE(pul);
            scalar_t sum;
            LINE_SUM(sum, pul);
            fprintf(stdout, "line sum: "SCALAR_FORMAT_STRING"\n",sum);
        }
    }
}
#endif

void k_path_genf1_round(index_t         n,
                        index_t         m,
                        index_t         k,
                        index_t         g,
                        index_t         l,
                        index_t         *d_pos,
                        index_t         *d_adj,
                        ffprng_scalar_t yl_seed,
                        line_array_t    *d_s)
{
    assert(g == SCALARS_IN_LINE);

    index_t nt = num_threads();
    index_t length = n;
    index_t block_size = length/nt;

    ffprng_t y_base;
    FFPRNG_INIT(y_base, yl_seed);

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t t = 0; t < nt; t++) {
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? length-1 : (start+block_size-1);
        ffprng_t y_gen;
        index_t y_pos = d_pos[start]-start;
        FFPRNG_FWD(y_gen, y_pos, y_base);
        for(index_t u = start; u <= stop; u++) {
            index_t pu  = d_pos[u];                // R: n  index_t   [hw pref]
            index_t deg = d_adj[pu];               // R: n  index_t   [hw pref]
            line_t pul;
            LINE_SET_ZERO(pul);
            for(index_t j = 1; j <= deg; j++) {
                index_t v = d_adj[pu+j];           // R: m  index_t   [hw pref]
                line_t pvl1;
                index_t i_v_l1 = PATH_LINE_IDX(b, k, l-1, v);
                LINE_LOAD(pvl1, d_s, i_v_l1);

#ifdef BUILD_PREFETCH
                // prefetch next line
                index_t nv = d_adj[pu+j+(j < deg ? 1 : 2)];
                index_t i_nv_l1 = PATH_LINE_IDX(b, k, l-1, nv);
                LINE_PREFETCH(d_s, i_nv_l1);
#endif

                ffprng_scalar_t rnd;
                FFPRNG_RAND(rnd, y_gen);
                scalar_t y_luv = (scalar_t) rnd;
                line_t sy;
                LINE_MUL_SCALAR(sy, pvl1, y_luv);     // MUL: ng
                LINE_ADD(pul, pul, sy);
            }
            line_t pul0;
            index_t i_u_l0 = PATH_LINE_IDX(b, k, 1, u);
            LINE_LOAD(pul0, d_s, i_u_l0);
            LINE_MUL(pul, pul, pul0);
            index_t i_u_l = PATH_LINE_IDX(b, k, l, u);
            LINE_STORE(d_s, i_u_l, pul);      // W: ng  scalar_t
        }
    }

    trans_bytes += (2*n+m)*sizeof(index_t) + (m+n)*g*sizeof(scalar_t);
    num_muls    += (n*g+m);
}


scalar_t k_path_genf1(index_t         n,
                     index_t         m,
                     index_t         k,
                     index_t         g,
                     index_t         *d_pos,
                     index_t         *d_adj,
                     ffprng_scalar_t y_seed,
                     line_array_t    *d_x,
                     scalar_t        *vs)
{

    assert(g == SCALARS_IN_LINE);
    assert(k >= 1);

    line_array_t *d_s  = (line_array_t *) MALLOC(LINE_ARRAY_SIZE(k*n*g));

    // Save the base case to d_s

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++) {
        line_t xu;
        LINE_LOAD(xu, d_x, u);              // R: ng  scalar_t [hw prefetched]
        index_t i_u_1 = PATH_LINE_IDX(b, k, 1, u);
        LINE_STORE(d_s, i_u_1, xu);         // W: ng  scalar_t
    }

    // Run the recurrence
    srand(y_seed);
    for(index_t l = 2; l <= k; l++) {
        ffprng_scalar_t yl_seed = irand(); // different y-values for every round
        k_path_genf1_round(n,m,k,g,l,d_pos,d_adj,yl_seed,d_s);
    }

    // Sum up
    scalar_t sum = line_sum_stride(n, g, k,
                                   ((line_array_t *)(((line_t *) d_s) + k-1)));

    // vertex localisation
    vertex_acc_stride(n,
                      g,
                      k,
                      ((line_array_t *)(((line_t *) d_s) + k-1)),
                      vs);
    FREE(d_s);

    trans_bytes += 2*n*g*sizeof(scalar_t);
    num_muls    += 0;

    return sum;
}

#endif

/************************************** k-path generating function (mark 2). */

#if BUILD_GENF == 2

#define PATH_LINE_IDX2(k, l, u) ((u))

#ifdef DEBUG
void print_kpath_ds_genf2(index_t n,
                          index_t k,
                          line_array_t *d_s)
{
    for(index_t l = 1; l <= k; l++) {
        fprintf(stdout,"-------------------------------------------------\n");
        fprintf(stdout, "l: %ld\n", l);
        fprintf(stdout,"-------------------------------------------------\n");
        for(index_t u = 0; u < n; u++) {
            fprintf(stdout, "%ld: ", u+1);
            index_t i_u_l = PATH_LINE_IDX2(k, l, u);
            line_t pul;
            LINE_LOAD(pul, d_s, i_u_l);
            PRINT_LINE(pul);
            scalar_t sum;
            LINE_SUM(sum, pul);
            fprintf(stdout, "line sum: "SCALAR_FORMAT_STRING"\n",sum);
        }
    }
}
#endif

void k_path_genf2_round(index_t         n,
                        index_t         m,
                        index_t         k,
                        index_t         g,
                        index_t         l,
                        index_t         *d_pos,
                        index_t         *d_adj,
                        ffprng_scalar_t yl_seed,
                        line_array_t    *d_x,
                        line_array_t    *d_s1,
                        line_array_t    *d_s2)
{
    assert(g == SCALARS_IN_LINE);

    index_t nt = num_threads();
    index_t length = n;
    index_t block_size = length/nt;

    ffprng_t y_base;
    FFPRNG_INIT(y_base, yl_seed);

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t t = 0; t < nt; t++) {
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? length-1 : (start+block_size-1);
        ffprng_t y_gen;
        index_t y_pos = d_pos[start]-start;
        FFPRNG_FWD(y_gen, y_pos, y_base);
        for(index_t u = start; u <= stop; u++) {
            index_t pu  = d_pos[u];                // R: n  index_t   [hw pref]
            index_t deg = d_adj[pu];               // R: n  index_t   [hw pref]
            line_t pul;
            LINE_SET_ZERO(pul);
            for(index_t j = 1; j <= deg; j++) {
                index_t v = d_adj[pu+j];           // R: m  index_t   [hw pref]
                line_t pvl1;
                index_t i_v_l1 = PATH_LINE_IDX2(k, l-1, v);
                LINE_LOAD(pvl1, d_s1, i_v_l1);

#ifdef BUILD_PREFETCH
                // prefetch next line
                index_t nv = d_adj[pu+j+(j < deg ? 1 : 2)];
                index_t i_nv_l1 = PATH_LINE_IDX2(k, l-1, nv);
                LINE_PREFETCH(d_s1, i_nv_l1);
#endif

                ffprng_scalar_t rnd;
                FFPRNG_RAND(rnd, y_gen);
                scalar_t y_luv = (scalar_t) rnd;
                line_t sy;
                LINE_MUL_SCALAR(sy, pvl1, y_luv);     // MUL: ng
                LINE_ADD(pul, pul, sy);
            }

            line_t xu;
            LINE_LOAD(xu, d_x, u);
            LINE_MUL(pul, pul, xu);
            index_t i_u_l = PATH_LINE_IDX2(k, l, u);
            LINE_STORE(d_s2, i_u_l, pul);      // W: ng  scalar_t
        }
    }

    trans_bytes += (2*n+m)*sizeof(index_t) + (m+n)*g*sizeof(scalar_t);
    num_muls    += (n*g+m);
}


scalar_t k_path_genf2(index_t         n,
                      index_t         m,
                      index_t         k,
                      index_t         g,
                      index_t         *d_pos,
                      index_t         *d_adj,
                      ffprng_scalar_t y_seed,
                      line_array_t    *d_x,
                      scalar_t        *vs)
{

    assert(g == SCALARS_IN_LINE);
    assert(k >= 1);

    line_array_t *d_s1  = (line_array_t *) MALLOC(LINE_ARRAY_SIZE(n*g));
    line_array_t *d_s2  = (line_array_t *) MALLOC(LINE_ARRAY_SIZE(n*g));

    // initialise x_u to d_s1

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++) {
        line_t xu;
        LINE_LOAD(xu, d_x, u);              // R: ng  scalar_t [hw prefetched]
        index_t i_u_1 = PATH_LINE_IDX2(k, 1, u);
        LINE_STORE(d_s1, i_u_1, xu);         // W: ng  scalar_t
    }

    // Run the recurrence
    srand(y_seed);
    for(index_t l = 2; l <= k; l++) {
        ffprng_scalar_t yl_seed = irand(); // different y-values for every round
        k_path_genf2_round(n, m, k, g, l, d_pos, d_adj, yl_seed, d_x, d_s1, d_s2);

        // swap array pointers
        line_array_t *d_temp = d_s1;
        d_s1 = d_s2;
        d_s2 = d_temp;
    }

    // Sum up
    scalar_t sum = line_sum(n, g, ((line_array_t *)(((line_t *) d_s1))));

    // vertex localisation
    vertex_acc(n, g, k, ((line_array_t *)(((line_t *) d_s1))), vs);

    // free memory
    FREE(d_s1);
    FREE(d_s2);

    trans_bytes += 2*n*g*sizeof(scalar_t);
    num_muls    += 0;

    return sum;
}

#endif
/************************************************************ The oracle(s). */

index_t path_oracle(index_t         n,
                    index_t         k,
                    index_t         *h_pos,
                    index_t         *h_adj,
                    index_t         num_shades,
                    shade_map_t     *h_s,
                    ffprng_scalar_t y_seed,
                    ffprng_scalar_t z_seed,
                    scalar_t        *master_vsum)
{
    push_memtrack();
    assert(k >= 1 && k < 31);
    index_t m = h_pos[n-1]+h_adj[h_pos[n-1]]+1-n;
    index_t sum_size = 1 << k;

    index_t g = SCALARS_IN_LINE;
    index_t outer = (sum_size + g-1) / g;
    // number of iterations for outer loop

    num_muls = 0;
    trans_bytes = 0;

    index_t *d_pos     = h_pos;
    index_t *d_adj     = h_adj;
    line_array_t *d_x  = (line_array_t *) MALLOC(LINE_ARRAY_SIZE(n*g));

    /* Run the work & time it. */

    push_time();

    scalar_t master_sum;
    SCALAR_SET_ZERO(master_sum);

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t i = 0; i < n; i++)
        master_vsum[i] = 0;

    for(index_t out = 0; out < outer; out++) {
        constrained_sieve_pre(n, k, g, g*out, num_shades, h_s, z_seed, d_x);
#if BUILD_GENF == 1
        scalar_t sum = k_path_genf1(n, m, k, g, d_pos, d_adj, y_seed, d_x, master_vsum);
#elif BUILD_GENF == 2
        scalar_t sum = k_path_genf2(n, m, k, g, d_pos, d_adj, y_seed, d_x, master_vsum);
#else
#error BUILD_GENF should be either 1 or 2
#endif

        SCALAR_ADD(master_sum, master_sum, sum);
    }

    double time = pop_time();
    double trans_rate = trans_bytes / (time/1000.0);
    double mul_rate = num_muls / time;
    FREE(d_x);

    fprintf(stdout,
            SCALAR_FORMAT_STRING
            " %.2lf ms [%.2lfGiB/s, %.2lfGHz] %d",
            (long) master_sum,
            time,
            trans_rate/((double) (1 << 30)),
            mul_rate/((double) 1e6),
            master_sum != 0);
    fprintf(stdout, " ");
    print_pop_memtrack();
    fprintf(stdout, " ");
    print_current_mem();
    fflush(stdout);

    return master_sum != 0;
}

/************************************************* Rudimentary graph builder. */

typedef struct 
{
    index_t is_directed;
    index_t num_vertices;
    index_t num_edges;
    index_t max_time;
    index_t edge_capacity;
    index_t *edges;
    index_t *colors;
} graph_t;

static index_t *enlarge(index_t m, index_t m_was, index_t *was)
{
    assert(m >= 0 && m_was >= 0);

    index_t *a = (index_t *) MALLOC(sizeof(index_t)*m);
    index_t i;
    if(was != (void *) 0) {
        for(i = 0; i < m_was; i++) {
            a[i] = was[i];
        }
        FREE(was);
    }
    return a;
}

graph_t *graph_alloc(index_t n)
{
    assert(n >= 0);

    index_t i;
    graph_t *g = (graph_t *) MALLOC(sizeof(graph_t));
    g->is_directed   = 0; // default: undirected graph
    g->num_vertices  = n;
    g->num_edges     = 0;
    g->edge_capacity = 100;
    g->edges  = enlarge(3*g->edge_capacity, 0, (void *) 0);
    g->colors = (index_t *) MALLOC(sizeof(index_t)*n);
    for(i = 0; i < n; i++)
        g->colors[i] = UNDEFINED;
    return g;
}

void graph_free(graph_t *g)
{
    FREE(g->edges);
    FREE(g->colors);
    FREE(g);
}

void graph_add_edge(graph_t *g, index_t u, index_t v, index_t t)
{
    assert(u >= 0 && 
           v >= 0 && 
           u < g->num_vertices &&
           v < g->num_vertices);
    assert(t>=0);
    //assert(t>=0 && t < g->max_time);

    if(g->num_edges == g->edge_capacity) {
        g->edges = enlarge(6*g->edge_capacity, 3*g->edge_capacity, g->edges);
        g->edge_capacity *= 2;
    }

    assert(g->num_edges < g->edge_capacity);

    index_t *e = g->edges + 3*g->num_edges;
    e[0] = u;
    e[1] = v;
    e[2] = t;
    g->num_edges++;
}

index_t *graph_edgebuf(graph_t *g, index_t cap)
{
    g->edges = enlarge(3*g->edge_capacity+3*cap, 3*g->edge_capacity, g->edges);
    index_t *e = g->edges + 3*g->num_edges;
    g->edge_capacity += cap;
    g->num_edges += cap;
    return e;
}

void graph_set_color(graph_t *g, index_t u, index_t c)
{
    assert(u >= 0 && u < g->num_vertices && c >= 0);
    g->colors[u] = c;
}

void graph_set_is_directed(graph_t *g, index_t is_dir)
{
    assert(is_dir == 0 || is_dir == 1);
    g->is_directed = is_dir;
}

void graph_set_max_time(graph_t *g, index_t tmax)
{
    assert(tmax > 0);
    g->max_time = tmax;
}

#ifdef DEBUG
void print_graph(graph_t *g)
{
    index_t n = g->num_vertices;
    index_t m = g->num_edges;
    index_t tmax = g->max_time;
    fprintf(stdout, "p motif %ld %ld %ld\n", n, m, tmax);

    index_t *e = g->edges;
    for(index_t i = 0; i < 3*m; i+=3) {
        fprintf(stdout, "e %ld %ld %ld\n", 
                        e[i]+1, e[i+1]+1, e[i+2]+1);
    }

    index_t *c = g->colors;
    for(index_t i = 0; i < n; i++)
        fprintf(stdout, "n %ld %ld\n", i+1, c[i]+1);
}
#endif


/************************************* Basic motif query processing routines. */

struct temppathq_struct
{
    index_t     is_stub;
    index_t     n;
    index_t     k;
    index_t     tmax;
    index_t     *pos;
    index_t     *adj;
    index_t     nl;
    index_t     *l;  
    index_t     ns;
    shade_map_t *shade;
    index_t     vert_loc;
    scalar_t    *vsum;
};

typedef struct temppathq_struct temppathq_t;

void adjsort(index_t n, index_t *pos, index_t *adj)
{
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++) {
        index_t pu = pos[u];
        index_t deg = adj[pu];
        heapsort_indext(deg, adj + pu + 1);
    }
}

void temppathq_free(temppathq_t *q)
{
    if(!q->is_stub) {
        FREE(q->pos);
        FREE(q->adj);
        FREE(q->l);
        FREE(q->shade);
        FREE(q->vsum);
    }
    FREE(q);
}

index_t temppathq_execute(temppathq_t *q)
{
    if(q->is_stub)
        return 0;
    return temppath_oracle(q->n, q->k, q->tmax, q->pos, q->adj, q->ns, q->shade, 
                           irand(), irand(), q->vert_loc, q->vsum);
}

#ifdef DEBUG
void print_temppathq(temppathq_t *q)
{
    index_t n       = q->n;
    index_t k       = q->k;
    index_t tmax    = q->tmax;
    index_t *pos    = q->pos;
    index_t *adj    = q->adj;
    fprintf(stdout, "-----------------------------------------------\n");
    fprintf(stdout, "printing temppathq\n");
    fprintf(stdout, "is_stub = %ld\n", q->is_stub);
    fprintf(stdout, "n = %ld\n", n);
    fprintf(stdout, "k = %ld\n", k);
    fprintf(stdout, "tmax = %ld\n", tmax);
    fprintf(stdout, "pos\n");
    fprintf(stdout, "----\n ");
    for(index_t i = 0; i < n*tmax; i++) {
        fprintf(stdout, "%ld%s", pos[i], i%n==n-1 ? "\n ":" ");
    }

    fprintf(stdout, "adjacency list:\n");
    fprintf(stdout, "---------------\n");
    for(index_t t = 0; t < tmax; t++) {
        fprintf(stdout, "t: %ld\n", t+1);
        fprintf(stdout, "---------------\n");

        index_t *pos_t = pos + n*t;
        for(index_t u = 0; u < n; u++) {
            index_t pu = pos_t[u];
            index_t nu = adj[pu];
            index_t *adj_u = adj + pu + 1;
            fprintf(stdout, "%4ld:", u+1);
            for(index_t i = 0; i < nu; i++) {
                fprintf(stdout, " %4ld", adj_u[i]+1);
            }
            fprintf(stdout, "\n");
        }
    }

    index_t nl          = q->nl;
    index_t *l          = q->l;
    fprintf(stdout, "nl = %ld\n", nl);
    fprintf(stdout, "l:\n");
    for(index_t i = 0; i < nl; i++)
        fprintf(stdout, "%8ld : %8ld\n", nl, l[i]);

    index_t ns = q ->ns;
    shade_map_t *shade  = q->shade;
    fprintf(stdout, "ns : %ld\n", ns);
    fprintf(stdout, "shades:\n");
    for(index_t u = 0; u < n; u++) {
        fprintf(stdout, "%10ld : 0x%08X\n", u+1, shade[u]);
    }

    scalar_t *vsum = q->vsum;
    fprintf(stdout, "vert_loc: %ld\n", q->vert_loc);
    fprintf(stdout, "vsum:\n");
    for(index_t u = 0; u < n; u++) 
        fprintf(stdout, "%10ld : "SCALAR_FORMAT_STRING"\n", u+1, vsum[u]);
    fprintf(stdout, "-----------------------------------------------\n");
}

void print_array(const char *name, index_t n, index_t *a, index_t offset)
{
    fprintf(stdout, "%s (%ld):", name, n);
    for(index_t i = 0; i < n; i++) {
        fprintf(stdout, " %ld", a[i] == -1 ? -1 : a[i]+offset);
    }
    fprintf(stdout, "\n"); 
}
#endif

/*************************************************** basic path query routine */
struct pathq_struct
{
    index_t     is_stub;
    index_t     n;
    index_t     k;
    index_t     *pos;
    index_t     *adj;
    index_t     nl;
    index_t     *l;
    index_t     ns;
    shade_map_t *shade;
    scalar_t    *vsum;
};

typedef struct pathq_struct pathq_t;

void pathq_free(pathq_t *q)
{
    if(!q->is_stub) {
        FREE(q->pos);
        FREE(q->adj);
        FREE(q->l);
        FREE(q->shade);
        FREE(q->vsum);
    }
    FREE(q);
}

#ifdef DEBUG
void print_pathq(pathq_t *q)
{
    index_t n           = q->n;
    index_t k           = q->k;
    index_t *pos        = q->pos;
    index_t *adj        = q->adj;
    index_t nl          = q->nl;
    index_t *l          = q->l;
    index_t ns          = q ->ns;
    shade_map_t *shade  = q->shade;
    scalar_t *vsum      = q->vsum; 

    fprintf(stdout, "-----------------------------------------------\n");
    fprintf(stdout, "printing pathq\n");
    fprintf(stdout, "is_stub : %ld\n", q->is_stub);
    fprintf(stdout, "n : %ld\n", n);
    fprintf(stdout, "k : %ld\n", k);
    fprintf(stdout, "pos\n");
    fprintf(stdout, "----\n ");
    for(index_t i = 0; i < n; i++) {
        fprintf(stdout, "%4ld%s", pos[i], i%n==n-1 ? "\n ":" ");
    }

    fprintf(stdout, "adjacency list:\n");
    fprintf(stdout, "---------------\n");
    for(index_t u = 0; u < n; u++) {
        index_t pu = pos[u];
        index_t nu = adj[pu];
        index_t *adj_u = adj + pu + 1;
        fprintf(stdout, "%4ld:", u+1);
        for(index_t i = 0; i < nu; i++) {
            fprintf(stdout, " %4ld", adj_u[i]+1);
        }
        fprintf(stdout, "\n");
    }

    fprintf(stdout, "nl = %ld\n", nl);
    fprintf(stdout, "l:\n");
    for(index_t i = 0; i < nl; i++)
        fprintf(stdout, "%8ld : %8ld\n", nl, l[i]);

    fprintf(stdout, "ns : %ld\n", ns);
    fprintf(stdout, "shades:\n");
    for(index_t u = 0; u < n; u++) {
        fprintf(stdout, "%10ld : 0x%08X\n", u+1, shade[u]);
    }

    fprintf(stdout, "vsum:\n");
    for(index_t u = 0; u < n; u++) 
        fprintf(stdout, "%10ld : "SCALAR_FORMAT_STRING"\n", u+1, vsum[u]);
    fprintf(stdout, "-----------------------------------------------\n");
}
#endif

pathq_t * build_pathq(temppathq_t *in)
{
    push_memtrack();

    index_t n = in->n;
    index_t tmax = in->tmax;
    index_t *i_pos = in->pos;
    index_t *i_adj = in->adj;
    shade_map_t *i_shade = in->shade;

    push_time();
    fprintf(stdout, "build pathq: ");
    fflush(stdout);
    push_time();

    // output position list
    index_t *c_pos = (index_t *) MALLOC(sizeof(index_t)*n);
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++)
        c_pos[u] = 0;

    for(index_t t = 0; t < tmax; t++) {
        index_t *i_pos_t = i_pos + t*n;
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
        for(index_t u = 0; u < n; u++) {
            index_t i_pu = i_pos_t[u];
            index_t i_nu = i_adj[i_pu];
            c_pos[u] += i_nu;
        }
    }

    index_t c_m = parallelsum(n, c_pos);
    index_t c_run = prefixsum(n, c_pos, 1);
    assert(c_run == n+c_m);

    fprintf(stdout, "[pos: %.2lf ms] ", pop_time());
    fflush(stdout);
    push_time();

    index_t *c_adj = (index_t *) MALLOC(sizeof(index_t)*(n+c_m));
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++)
        c_adj[c_pos[u]] = 0;

    for(index_t t = 0; t < tmax; t++) {
        index_t *i_pos_t = i_pos + t*n;
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
        for(index_t u = 0; u < n; u++) {
            index_t i_pu = i_pos_t[u];
            index_t i_nu = i_adj[i_pu];
            index_t *i_adj_u = i_adj + i_pu;
            index_t o_pu = c_pos[u];
            for(index_t j = 1; j <= i_nu; j++) {
                index_t v = i_adj_u[j];
                c_adj[o_pu + 1 + c_adj[o_pu]++] = v;
            }
        }
    }

    adjsort(n, c_pos, c_adj);

    index_t *o_pos = (index_t *) MALLOC(sizeof(index_t)*n);
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++)
        o_pos[u] = 0;
        
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++) {
        index_t c_pu = c_pos[u];
        index_t c_nu = c_adj[c_pu];

        if(c_nu == 0 || c_nu == 1) {
            o_pos[u] = c_nu;
            continue;
        }
        o_pos[u] = 1;
        index_t *c_adj_u = c_adj + c_pu;
        for(index_t j = 2; j <= c_nu; j++) {
            if(c_adj_u[j-1] != c_adj_u[j]) {
                o_pos[u]++;
            }
        }
    }

    index_t o_m = parallelsum(n, o_pos);
    index_t o_run = prefixsum(n, o_pos, 1);
    assert(o_run==n+o_m);
    
    index_t *o_adj = (index_t *) MALLOC(sizeof(index_t)*(n+o_m));
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++)
        o_adj[o_pos[u]] = 0;

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++) {
        index_t c_pu = c_pos[u];
        index_t c_nu = c_adj[c_pu];

        if(c_nu == 0) continue;

        index_t o_pu = o_pos[u];
        index_t *c_adj_u = c_adj + c_pu;
        o_adj[o_pu + 1 + o_adj[o_pu]++] = c_adj_u[1];

        for(index_t j = 2; j <= c_nu; j++) {
            if(c_adj_u[j-1] != c_adj_u[j]) {
                o_adj[o_pu + 1 + o_adj[o_pu]++] = c_adj_u[j];
            }
        }
    }

    fprintf(stdout, "[adj: %.2lf ms] ", pop_time());
    fflush(stdout);
    push_time();

    shade_map_t *o_shade = (shade_map_t *) MALLOC(sizeof(shade_map_t)*n);
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++)
        o_shade[u] = i_shade[u];
    
    fprintf(stdout, "[shade: %.2lf ms] ", pop_time());
    fprintf(stdout, "done. [%.2lf ms] ", pop_time());
    print_pop_memtrack();
    fprintf(stdout, " ");
    print_current_mem();
    fprintf(stdout, "\n");
    fflush(stdout);

    FREE(c_pos);
    FREE(c_adj);

    pathq_t *out    = (pathq_t *) MALLOC(sizeof(pathq_t));
    out->is_stub    = 0;
    out->n          = n;
    out->k          = in->k;
    out->pos        = o_pos;
    out->adj        = o_adj;
    out->nl         = 0;
    out->l          = (index_t *) MALLOC(sizeof(index_t)*out->nl);
    out->ns         = in->ns;
    out->shade      = o_shade;
    out->vsum       = (scalar_t *) MALLOC(sizeof(scalar_t)*n);

    return out;
}


// A quick fix to support vertex localised sieving for undirected graphs
pathq_t * build_pathq_dir(temppathq_t *in)
{
    push_memtrack();

    index_t n       = in->n;
    index_t tmax    = in->tmax;
    index_t *i_pos  = in->pos;
    index_t *i_adj  = in->adj;
    shade_map_t *i_shade = in->shade;

    push_time();
    fprintf(stdout, "build pathq: ");
    fflush(stdout);
    push_time();

    // output position list
    index_t *c_pos = (index_t *) MALLOC(sizeof(index_t)*n);
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++)
        c_pos[u] = 0;

    for(index_t t = 0; t < tmax; t++) {
        index_t *i_pos_t = i_pos + t*n;
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
        for(index_t u = 0; u < n; u++) {
            index_t i_pu = i_pos_t[u];
            index_t i_nu = i_adj[i_pu];
            c_pos[u] += i_nu;
        }
    }

    index_t c_m = parallelsum(n, c_pos);
    index_t c_run = prefixsum(n, c_pos, 1);
    assert(c_run == n+c_m);

    fprintf(stdout, "[pos: %.2lf ms] ", pop_time());
    fflush(stdout);
    push_time();

    index_t *c_adj = (index_t *) MALLOC(sizeof(index_t)*(n+c_m));
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++)
        c_adj[c_pos[u]] = 0;

    for(index_t t = 0; t < tmax; t++) {
        index_t *i_pos_t = i_pos + t*n;
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
        for(index_t u = 0; u < n; u++) {
            index_t i_pu = i_pos_t[u];
            index_t i_nu = i_adj[i_pu];
            index_t *i_adj_u = i_adj + i_pu;
            index_t o_pu = c_pos[u];
            for(index_t j = 1; j <= i_nu; j++) {
                index_t v = i_adj_u[j];
                c_adj[o_pu + 1 + c_adj[o_pu]++] = v;
            }
        }
    }

    adjsort(n, c_pos, c_adj);

    index_t *o_pos = (index_t *) MALLOC(sizeof(index_t)*n);
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++)
        o_pos[u] = 0;
        
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++) {
        index_t c_pu = c_pos[u];
        index_t c_nu = c_adj[c_pu];

        if(c_nu == 0 || c_nu == 1) {
            o_pos[u] = c_nu;
            continue;
        }
        o_pos[u] = 1;
        index_t *c_adj_u = c_adj + c_pu;
        for(index_t j = 2; j <= c_nu; j++) {
            if(c_adj_u[j-1] != c_adj_u[j]) {
                o_pos[u]++;
            }
        }
    }

    index_t o_m = parallelsum(n, o_pos);
    index_t o_run = prefixsum(n, o_pos, 1);
    assert(o_run==n+o_m);
    
    index_t *o_adj = (index_t *) MALLOC(sizeof(index_t)*(n+o_m));
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++)
        o_adj[o_pos[u]] = 0;

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++) {
        index_t c_pu = c_pos[u];
        index_t c_nu = c_adj[c_pu];

        if(c_nu == 0) continue;

        index_t o_pu = o_pos[u];
        index_t *c_adj_u = c_adj + c_pu;
        o_adj[o_pu + 1 + o_adj[o_pu]++] = c_adj_u[1];

        for(index_t j = 2; j <= c_nu; j++) {
            if(c_adj_u[j-1] != c_adj_u[j]) {
                o_adj[o_pu + 1 + o_adj[o_pu]++] = c_adj_u[j];
            }
        }
    }

    // convert directed to undirected graph
    index_t *o_pos_ud = (index_t *) MALLOC(n*sizeof(index_t));
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++)
        o_pos_ud[u] = o_adj[o_pos[u]];

    index_t nt = num_threads();
    index_t block_size = n/nt;
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t th = 0; th < nt; th++) {
        index_t start = th*block_size;
        index_t stop = (th == nt-1) ? n-1 : (start+block_size-1);
        for(index_t u = 0; u < n; u++) {
            index_t o_pu = o_pos[u];
            index_t *o_adj_u = o_adj + o_pu;
            index_t o_nu = o_adj_u[0];
            for(index_t j = 1; j <= o_nu; j++) {
                index_t v = o_adj_u[j];
                if(start <= v && v <= stop)
                    o_pos_ud[v]++;
            }
        }
    }

    index_t o_m_ud   = parallelsum(n, o_pos_ud);
    index_t o_run_ud = prefixsum(n, o_pos_ud, 1);
    assert(o_run_ud == n+o_m_ud);
    assert(o_m_ud == 2*o_m);

    index_t *o_adj_ud = (index_t *) MALLOC((n+o_m_ud)*sizeof(index_t));

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++)
        o_adj_ud[o_pos_ud[u]] = 0; 

    // first copy the adjacency list as it is
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++) {
        index_t o_pu = o_pos[u];
        index_t *o_adj_u = o_adj + o_pu;
        index_t o_nu = o_adj_u[0];

        index_t o_pu_ud = o_pos_ud[u];
        index_t *o_adj_ud_u = o_adj_ud + o_pu_ud;
        o_adj_ud_u[0] = o_nu;
        for(index_t j = 1; j <= o_nu; j++) {
            index_t v = o_adj_u[j];
            o_adj_ud_u[j] = v;
        }
    }

    // add edges in other direction now
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t th = 0; th < nt; th++) {
        index_t start = th*block_size;
        index_t stop = (th == nt-1) ? n-1 : (start+block_size-1);
        for(index_t u = 0; u < n; u++) {
            index_t o_pu = o_pos[u];
            index_t *o_adj_u = o_adj + o_pu;
            index_t o_nu = o_adj_u[0];
            for(index_t j = 1; j <= o_nu; j++) {
                index_t v = o_adj_u[j];
                if(start <= v && v <= stop) {
                    index_t o_pv_ud = o_pos_ud[v];
                    o_adj_ud[o_pv_ud + 1 + o_adj_ud[o_pv_ud]++] = u;
                }
            }
        }
    }


    FREE(o_pos);
    FREE(o_adj);

    fprintf(stdout, "[adj: %.2lf ms] ", pop_time());
    fflush(stdout);
    push_time();

    shade_map_t *o_shade = (shade_map_t *) MALLOC(sizeof(shade_map_t)*n);
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++)
        o_shade[u] = i_shade[u];
    
    fprintf(stdout, "[shade: %.2lf ms] ", pop_time());
    fprintf(stdout, "done. [%.2lf ms] ", pop_time());
    print_pop_memtrack();
    fprintf(stdout, " ");
    print_current_mem();
    fprintf(stdout, "\n");
    fflush(stdout);

    FREE(c_pos);
    FREE(c_adj);

    pathq_t *out    = (pathq_t *) MALLOC(sizeof(pathq_t));
    out->is_stub    = 0;
    out->n          = n;
    out->k          = in->k;
    out->pos        = o_pos_ud;
    out->adj        = o_adj_ud;
    out->nl         = 0;
    out->l          = (index_t *) MALLOC(sizeof(index_t)*out->nl);
    out->ns         = in->ns;
    out->shade      = o_shade;
    out->vsum       = (scalar_t *) MALLOC(sizeof(scalar_t)*n);

    return out;
}


scalar_t pathq_execute(pathq_t *q)
{
    if(q->is_stub)
        return 0;
    return path_oracle(q->n, q->k, q->pos, q->adj, q->ns, q->shade, irand(), irand(), q->vsum);
}


/*************** Project a query by cutting out a given interval of vertices. */

index_t get_poscut(index_t n, index_t tmax, 
                   index_t *pos, index_t *adj, 
                   index_t lo_v, index_t hi_v,
                   index_t *poscut)
{
    // Note: assumes the adjacency lists are sorted
    assert(lo_v <= hi_v);

    index_t ncut = n - (hi_v-lo_v+1);

    for(index_t t = 0; t < tmax; t++) {
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
        for(index_t u = 0; u < lo_v; u++) {
            index_t pu = pos[t*n + u];
            index_t deg = adj[pu];
            index_t cs, ce;
            index_t l = get_interval(deg, adj + pu + 1,
                                     lo_v, hi_v,
                                     &cs, &ce);
            poscut[t*ncut + u] = deg - l;
        }
    }

    for(index_t t = 0; t < tmax; t++) {
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
        for(index_t u = hi_v+1; u < n; u++) {
            index_t pu = pos[t*n + u];
            index_t deg = adj[pu];
            index_t cs, ce;
            index_t l = get_interval(deg, adj + pu + 1,
                                     lo_v, hi_v,
                                     &cs, &ce);
            poscut[t*ncut + (u-hi_v-1+lo_v)] = deg - l;
        }
    }

    index_t run = prefixsum(tmax*ncut, poscut, 1);
    return run;
}

temppathq_t *temppathq_cut(temppathq_t *q, index_t lo_v, index_t hi_v)
{
    // Note: assumes the adjacency lists are sorted

    //fprintf(stdout, "-------------------------------\n");
    //fprintf(stdout, "low: %ld, high: %ld\n", lo_v, hi_v);
    //print_temppathq(q);

    index_t n = q->n;
    index_t tmax = q->tmax;
    index_t *pos = q->pos;
    index_t *adj = q->adj;    
    assert(0 <= lo_v && lo_v <= hi_v && hi_v < n);

    // Fast-forward a stub NO when the interval 
    // [lo_v,hi_v] contains an element in q->l
    for(index_t i = 0; i < q->nl; i++) {
        if(q->l[i] >= lo_v && q->l[i] <= hi_v) {
            temppathq_t *qs = (temppathq_t *) MALLOC(sizeof(temppathq_t));
            qs->is_stub = 1;
            return qs;
        }
    }

    index_t ncut = n - (hi_v-lo_v+1); // number of vertices after cut
    index_t *poscut = alloc_idxtab(tmax*ncut);
    index_t bcut = get_poscut(n, tmax, pos, adj, lo_v, hi_v, poscut);
    index_t *adjcut = alloc_idxtab(bcut);
    index_t gap = hi_v-lo_v+1;

    //print_array("poscut", tmax*ncut, poscut, 0);

    for(index_t t = 0; t < tmax; t++) {
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
        for(index_t v = 0; v < ncut; v++) {
            index_t u = v;
            if(u >= lo_v)
                u += gap;
            index_t pu = pos[t*n + u];
            index_t degu = adj[pu];
            index_t cs, ce;
            index_t l = get_interval(degu, adj + pu + 1,
                                     lo_v, hi_v,
                                     &cs, &ce);
            index_t pv = poscut[t*ncut + v];
            index_t degv = degu - l;
            adjcut[pv] = degv;
            // could parallelize this too
            for(index_t i = 0; i < cs; i++)
                adjcut[pv + 1 + i] = adj[pu + 1 + i];
            // could parallelize this too
            for(index_t i = cs; i < degv; i++)
                adjcut[pv + 1 + i] = adj[pu + 1 + i + l] - gap;
        }
    }

    //print_array("adj_cut", bcut, adjcut, 0);

    temppathq_t *qq = (temppathq_t *) MALLOC(sizeof(temppathq_t));
    qq->is_stub = 0;
    qq->n = ncut;
    qq->k = q->k;
    qq->tmax = q->tmax;
    qq->pos = poscut;
    qq->adj = adjcut;
    qq->nl = q->nl;
    qq->l = (index_t *) MALLOC(sizeof(index_t)*qq->nl);
    for(index_t i = 0; i < qq->nl; i++) {
        index_t u = q->l[i];
        assert(u < lo_v || u > hi_v);
        if(u > hi_v)
            u -= gap;
        qq->l[i] = u;
    }
    qq->ns = q->ns;
    qq->shade = (shade_map_t *) MALLOC(sizeof(shade_map_t)*ncut);
    for(index_t v = 0; v < ncut; v++) {
        index_t u = v;
        if(u >= lo_v)
            u += gap;
        qq->shade[v] = q->shade[u];
    }

    qq->vsum = (scalar_t *) MALLOC(sizeof(scalar_t)*ncut);
    //print_temppathq(qq);
    //exit(0);
    return qq;
}

/****************** Project a query with given projection & embedding arrays. */

#define PROJ_UNDEF 0xFFFFFFFFFFFFFFFFUL

index_t get_posproj(index_t n, index_t *pos, index_t *adj, 
                    index_t nproj, index_t *proj, index_t *embed,
                    index_t *posproj)
{

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t v = 0; v < nproj; v++) {
        index_t u = embed[v];
        index_t pu = pos[u];
        index_t deg = adj[pu];
        index_t degproj = 0;
        for(index_t i = 0; i < deg; i++) {
            index_t w = proj[adj[pu + 1 + i]];
            if(w != PROJ_UNDEF)
                degproj++;
        }
        posproj[v] = degproj;
    }

    index_t run = prefixsum(nproj, posproj, 1);
    return run;
}

temppathq_t *temppathq_project(temppathq_t *q, 
                         index_t nproj, index_t *proj, index_t *embed,
                         index_t nl, index_t *l)
{
    index_t n = q->n;
    index_t *pos = q->pos;
    index_t *adj = q->adj;    
 
    index_t *posproj = alloc_idxtab(nproj);
    index_t bproj = get_posproj(n, pos, adj, nproj, proj, embed, posproj);
    index_t *adjproj = alloc_idxtab(bproj);

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t v = 0; v < nproj; v++) {
        index_t pv = posproj[v];
        index_t u = embed[v];
        index_t pu = pos[u];
        index_t deg = adj[pu];
        index_t degproj = 0;
        for(index_t i = 0; i < deg; i++) {
            index_t w = proj[adj[pu + 1 + i]];
            if(w != PROJ_UNDEF)
                adjproj[pv + 1 + degproj++] = w;
        }
        adjproj[pv] = degproj;
    }

    temppathq_t *qq = (temppathq_t *) MALLOC(sizeof(temppathq_t));
    qq->is_stub = 0;
    qq->n = nproj;
    qq->k = q->k;
    qq->pos = posproj;
    qq->adj = adjproj;

    // Now project the l array

    assert(q->nl == 0); // l array comes from lister    
    qq->nl = nl;
    qq->l = (index_t *) MALLOC(sizeof(index_t)*nl);
    for(index_t i = 0; i < nl; i++) {
        index_t u = proj[l[i]];
        assert(u != PROJ_UNDEF); // query is a trivial NO !
        qq->l[i] = u;
    }

    // Next set up the projected shades

    qq->ns = q->ns;
    qq->shade = (shade_map_t *) MALLOC(sizeof(shade_map_t)*nproj);

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++) {
        index_t v = proj[u];
        if(v != PROJ_UNDEF)
            qq->shade[v] = q->shade[u];
    }

    // Reserve a unique shade to every vertex in l
    // while keeping the remaining shades available

    // Reserve shades first ... 
    index_t *l_shade = (index_t *) MALLOC(sizeof(index_t)*nl);
    shade_map_t reserved_shades = 0;
    for(index_t i = 0; i < nl; i++) {
        index_t v = qq->l[i];
        index_t j = 0;
        for(; j < qq->ns; j++)
            if(((qq->shade[v] >> j)&1) == 1 && 
               ((reserved_shades >> j)&1) == 0)
                break;
        assert(j < qq->ns);
        reserved_shades |= 1UL << j;
        l_shade[i] = j;
    }
    // ... then clear all reserved shades in one pass

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t v = 0; v < nproj; v++)
        qq->shade[v] &= ~reserved_shades;

    // ... and finally set reserved shades
    for(index_t i = 0; i < nl; i++) {
        index_t v = qq->l[i];
        qq->shade[v] = 1UL << l_shade[i];
    }
    FREE(l_shade);

    return qq;
}

/**************************************************** The interval extractor. */

struct ivlist_struct
{
    index_t start;
    index_t end;
    struct ivlist_struct *prev;
    struct ivlist_struct *next;
};

typedef struct ivlist_struct ivlist_t;

typedef struct ivext_struct 
{
    index_t     n;
    index_t     k;
    ivlist_t    *queue;
    ivlist_t    *active_queue_head;
    ivlist_t    *spare_queue_head;
    ivlist_t    *embed_list;
} ivext_t;

void ivext_enqueue_spare(ivext_t *e, ivlist_t *iv)
{
    pnlinknext(e->spare_queue_head,iv);
}

void ivext_enqueue_active(ivext_t *e, ivlist_t *iv)
{
    pnlinkprev(e->active_queue_head,iv);
}

ivlist_t *ivext_dequeue_first_nonsingleton(ivext_t *e)
{
    ivlist_t *iv = e->active_queue_head->next;  
    for(; 
        iv != e->active_queue_head; 
        iv = iv->next)
        if(iv->end - iv->start + 1 > 1)
            break;
    assert(iv != e->active_queue_head);
    pnunlink(iv);
    return iv;
}

ivlist_t *ivext_get_spare(ivext_t *e)
{
    assert(e->spare_queue_head->next != e->spare_queue_head);
    ivlist_t *iv = e->spare_queue_head->next;
    pnunlink(iv);
    return iv;
}

void ivext_reset(ivext_t *e)
{
    e->active_queue_head = e->queue + 0;
    e->spare_queue_head  = e->queue + 1;
    e->active_queue_head->next = e->active_queue_head;
    e->active_queue_head->prev = e->active_queue_head;
    e->spare_queue_head->prev  = e->spare_queue_head;
    e->spare_queue_head->next  = e->spare_queue_head;  
    e->embed_list = (ivlist_t *) 0;

    for(index_t i = 0; i < e->k + 2; i++)
        ivext_enqueue_spare(e, e->queue + 2 + i); // rot-safe
    ivlist_t *iv = ivext_get_spare(e);
    iv->start = 0;
    iv->end = e->n-1;
    ivext_enqueue_active(e, iv);
}

ivext_t *ivext_alloc(index_t n, index_t k)
{
    ivext_t *e = (ivext_t *) MALLOC(sizeof(ivext_t));
    e->n = n;
    e->k = k;
    e->queue = (ivlist_t *) MALLOC(sizeof(ivlist_t)*(k+4)); // rot-safe
    ivext_reset(e);
    return e;
}

void ivext_free(ivext_t *e)
{
    ivlist_t *el = e->embed_list;
    while(el != (ivlist_t *) 0) {
        ivlist_t *temp = el;
        el = el->next;
        FREE(temp);
    }
    FREE(e->queue);
    FREE(e);
}

void ivext_project(ivext_t *e, ivlist_t *iv)
{
    for(ivlist_t *z = e->active_queue_head->next; 
        z != e->active_queue_head; 
        z = z->next) {
        assert(z->end < iv->start ||
               z->start > iv->end);
        if(z->start > iv->end) {
            z->start -= iv->end-iv->start+1;
            z->end   -= iv->end-iv->start+1;
        }
    }

    ivlist_t *em = (ivlist_t *) MALLOC(sizeof(ivlist_t));
    em->start    = iv->start;
    em->end      = iv->end;
    em->next     = e->embed_list;
    e->embed_list = em;
}

index_t ivext_embed(ivext_t *e, index_t u)
{
    ivlist_t *el = e->embed_list;
    while(el != (ivlist_t *) 0) {
        if(u >= el->start)
            u += el->end - el->start + 1;
        el = el->next;
    }
    return u;
}

ivlist_t *ivext_halve(ivext_t *e, ivlist_t *iv)
{
    assert(iv->end - iv->start + 1 >= 2);
    index_t mid = (iv->start + iv->end)/2;  // mid < iv->end    
    ivlist_t *h = ivext_get_spare(e);
    h->start = iv->start;
    h->end = mid;
    iv->start = mid+1;
    return h;
}
    
index_t ivext_queue_size(ivext_t *e)
{
    index_t s = 0;
    for(ivlist_t *iv = e->active_queue_head->next; 
        iv != e->active_queue_head; 
        iv = iv->next)
        s += iv->end-iv->start+1;
    return s;
}

index_t ivext_num_active_intervals(ivext_t *e)
{
    index_t s = 0;
    for(ivlist_t *iv = e->active_queue_head->next; 
        iv != e->active_queue_head; 
        iv = iv->next)
        s++;
    return s;
}

void ivext_queue_print(FILE *out, ivext_t *e, index_t rot)
{
    index_t j = 0;
    char x[16384];
    char y[16384];
    y[0] = '\0';
    sprintf(x, "%c%12ld [", 
            rot == 0 ? ' ' : 'R',
            ivext_queue_size(e));
    strcat(y, x);
    for(ivlist_t *iv = e->active_queue_head->next; 
        iv != e->active_queue_head; 
        iv = iv->next) {
        assert(iv->start <= iv->end);
        if(iv->start < iv->end)
            sprintf(x, 
                    "%s[%ld:%ld]", 
                    j++ == 0 ? "" : ",",
                    ivext_embed(e, iv->start),
                    ivext_embed(e, iv->end));
        else
            sprintf(x, 
                    "%s[%ld]", 
                    j++ == 0 ? "[" : ",",
                    ivext_embed(e, iv->start));
        strcat(y, x);
    }   
    strcat(y, "] ");
    fprintf(out, "%-120s", y);
    fflush(out);
}

index_t extract_match(index_t is_root, temppathq_t *query, index_t *match)
{
    // Assumes adjancency lists of query are sorted.

    fprintf(stdout, "extract: %ld %ld %ld\n", query->n, query->k, query->nl);
    push_time();
    assert(query->k <= query->n);
    ivext_t *e = ivext_alloc(query->n, query->k);
    ivext_queue_print(stdout, e, 0);
    if(!temppathq_execute(query)) {
        fprintf(stdout, " -- false\n");
        ivext_free(e);
        if(!is_root)
            temppathq_free(query);
        double time = pop_time();
        fprintf(stdout, "extract done [%.2lf ms]\n", time);
        return 0;
    }
    fprintf(stdout, " -- true\n");
           
    while(ivext_queue_size(e) > e->k) {
        ivlist_t *iv = ivext_dequeue_first_nonsingleton(e);
        ivlist_t *h = ivext_halve(e, iv);
        ivext_enqueue_active(e, iv);
        temppathq_t *qq = temppathq_cut(query, h->start, h->end);
        ivext_queue_print(stdout, e, 0);
        if(temppathq_execute(qq)) {
            fprintf(stdout, " -- true\n");
            if(!is_root)
                temppathq_free(query);
            query = qq;
            is_root = 0;
            ivext_project(e, h);
            ivext_enqueue_spare(e, h);
        } else {
            fprintf(stdout, " -- false\n");
            temppathq_free(qq);
            pnunlink(iv);
            ivext_enqueue_active(e, h);
            qq = temppathq_cut(query, iv->start, iv->end);
            ivext_queue_print(stdout, e, 0);
            if(temppathq_execute(qq)) {
                fprintf(stdout, " -- true\n");
                if(!is_root)
                    temppathq_free(query);
                query = qq;
                is_root = 0;
                ivext_project(e, iv);
                ivext_enqueue_spare(e, iv);
            } else {
                fprintf(stdout, " -- false\n");
                temppathq_free(qq);
                ivext_enqueue_active(e, iv);
                while(ivext_num_active_intervals(e) > e->k) {
                    // Rotate queue until outlier is out ...
                    ivlist_t *iv = e->active_queue_head->next;  
                    pnunlink(iv);
                    qq = temppathq_cut(query, iv->start, iv->end);
                    ivext_queue_print(stdout, e, 1);
                    if(temppathq_execute(qq)) {
                        fprintf(stdout, " -- true\n");
                        if(!is_root)
                            temppathq_free(query);
                        query = qq;
                        is_root = 0;
                        ivext_project(e, iv);
                        ivext_enqueue_spare(e, iv);
                    } else {
                        fprintf(stdout, " -- false\n");
                        temppathq_free(qq);
                        ivext_enqueue_active(e, iv);
                    }
                }
            }
        }
    }
    for(index_t i = 0; i < query->k; i++)
        match[i] = ivext_embed(e, i);
    ivext_free(e);
    if(!is_root)
        temppathq_free(query);
    double time = pop_time();
    fprintf(stdout, "extract done [%.2lf ms]\n", time);
    return 1;
}

/**************************************************************** The lister. */

#define M_QUERY       0
#define M_OPEN        1
#define M_CLOSE       2
#define M_REWIND_U    3
#define M_REWIND_L    4

index_t command_mnemonic(index_t command) 
{
    return command >> 60;   
}

index_t command_index(index_t command)
{
    return command & (~(0xFFUL<<60));
}

index_t to_command_idx(index_t mnemonic, index_t idx)
{
    assert(idx < (1UL << 60));
    return (mnemonic << 60)|idx;
}

index_t to_command(index_t mnemonic)
{
    return to_command_idx(mnemonic, 0UL);
}

typedef struct 
{
    index_t n;              // number of elements in universe
    index_t k;              // size of the sets to be listed
    index_t *u;             // upper bound as a bitmap
    index_t u_size;         // size of upper bound
    index_t *l;             // lower bound 
    index_t l_size;         // size of lower bound
    index_t *stack;         // a stack for maintaining state
    index_t stack_capacity; // ... the capacity of the stack    
    index_t top;            // index of stack top
    temppathq_t *root;         // the root query
} lister_t;

void lister_push(lister_t *t, index_t word)
{
    assert(t->top + 1 < t->stack_capacity);
    t->stack[++t->top] = word;
}

index_t lister_pop(lister_t *t)
{
    return t->stack[t->top--];
}

index_t lister_have_work(lister_t *t)
{
    return t->top >= 0;
}

index_t lister_in_l(lister_t *t, index_t j)
{
    for(index_t i = 0; i < t->l_size; i++)
        if(t->l[i] == j)
            return 1;
    return 0;
}

void lister_push_l(lister_t *t, index_t j)
{
    assert(!lister_in_l(t, j) && t->l_size < t->k);
    t->l[t->l_size++] = j;
}

void lister_pop_l(lister_t *t)
{
    assert(t->l_size > 0);
    t->l_size--;
}

void lister_reset(lister_t *t)
{
    t->l_size = 0;
    t->top = -1;
    lister_push(t, to_command(M_QUERY));
    for(index_t i = 0; i < t->n; i++)
        bitset(t->u, i, 1);
    t->u_size = t->n;
}

lister_t *lister_alloc(index_t n, index_t k, temppathq_t *root)
{
    assert(n >= 1 && n < (1UL << 60) && k >= 1 && k <= n);
    lister_t *t = (lister_t *) MALLOC(sizeof(lister_t));
    t->n = n;
    t->k = k;
    t->u = alloc_idxtab((n+63)/64);
    t->l = alloc_idxtab(k);
    t->stack_capacity = n + k*(k+1+2*k) + 1;
    t->stack = alloc_idxtab(t->stack_capacity);
    lister_reset(t);
    t->root = root;
    if(t->root != (temppathq_t *) 0) {
        assert(t->root->n == t->n);
        assert(t->root->k == t->k);
        assert(t->root->nl == 0);
    }
    return t;
}

void lister_free(lister_t *t)
{
    if(t->root != (temppathq_t *) 0)
        temppathq_free(t->root);
    FREE(t->u);
    FREE(t->l);
    FREE(t->stack);
    FREE(t);
}

void lister_get_proj_embed(lister_t *t, index_t **proj_out, index_t **embed_out)
{
    index_t n = t->n;
    index_t usize = t->u_size;

    index_t *embed = (index_t *) MALLOC(sizeof(index_t)*usize);
    index_t *proj  = (index_t *) MALLOC(sizeof(index_t)*n);

    // could parallelize this (needs parallel prefix sum)
    index_t run = 0;
    for(index_t i = 0; i < n; i++) {
        if(bitget(t->u, i)) {
            proj[i]    = run;
            embed[run] = i;
            run++;
        } else {
            proj[i] = PROJ_UNDEF;
        }
    }
    assert(run == usize);

    *proj_out  = proj;
    *embed_out = embed;
}

void lister_query_setup(lister_t *t, temppathq_t **q_out, index_t **embed_out)
{
    index_t *proj;
    index_t *embed;

    // set up the projection with u and l
    lister_get_proj_embed(t, &proj, &embed);
    temppathq_t *qq = temppathq_project(t->root, 
                                  t->u_size, proj, embed, 
                                  t->l_size, t->l);
    FREE(proj);

    *q_out     = qq;
    *embed_out = embed;
}

index_t lister_extract(lister_t *t, index_t *s)
{
    // assumes t->u contains all elements of t->l
    // (otherwise query is trivial no)

    assert(t->root != (temppathq_t *) 0);
    
    if(t->u_size == t->n) {
        // rush the root query without setting up a copy
        return extract_match(1, t->root, s);
    } else {
        // a first order of business is to set up the query 
        // based on the current t->l and t->u; this includes
        // also setting up the embedding back to the root,
        // in case we are lucky and actually discover a match
        temppathq_t *qq; // will be released by extractor
        index_t *embed;
        lister_query_setup(t, &qq, &embed);
        
        // now execute the interval extractor ...
        index_t got_match = extract_match(0, qq, s);
        
        // ... and embed the match (if any) 
        if(got_match) {
            for(index_t i = 0; i < t->k; i++)
                s[i] = embed[s[i]];
        }
        FREE(embed);
        return got_match;
    }
}

index_t lister_run(lister_t *t, index_t *s)
{
    while(lister_have_work(t)) {
        index_t cmd = lister_pop(t);
        index_t mnem = command_mnemonic(cmd);
        index_t idx = command_index(cmd);
        switch(mnem) {
        case M_QUERY:
            if(t->k <= t->u_size && lister_extract(t, s)) {
                // we have discovered a match, which we need to
                // put on the stack to continue work when the user
                // requests this
                for(index_t i = 0; i < t->k; i++)
                    lister_push(t, s[i]);
                lister_push(t, to_command_idx(M_OPEN, t->k-1));
                // now report our discovery to user
                return 1;
            }
            break;
        case M_OPEN:
            {
                index_t *x = t->stack + t->top - t->k + 1;
                index_t k = 0;
                for(; k < idx; k++)
                    if(!lister_in_l(t, x[k]))
                        break;
                if(k == idx) {
                    // opening on last element of x not in l
                    // so we can dispense with x as long as we remember to 
                    // insert x[idx] back to u when rewinding
                    for(index_t j = 0; j < t->k; j++)
                        lister_pop(t); // axe x from stack
                    if(!lister_in_l(t, x[idx])) {
                        bitset(t->u, x[idx], 0); // remove x[idx] from u
                        t->u_size--;
                        lister_push(t, to_command_idx(M_REWIND_U, x[idx]));
                        lister_push(t, to_command(M_QUERY));
                    }
                } else {
                    // have still other elements of x that we need to
                    // open on, so must keep x in stack 
                    // --
                    // invariant that controls stack size:
                    // each open increases l by at least one
                    lister_push(t, to_command_idx(M_CLOSE, idx));
                    if(!lister_in_l(t, x[idx])) {
                        bitset(t->u, x[idx], 0); // remove x[idx] from u
                        t->u_size--;
                        lister_push(t, to_command_idx(M_REWIND_U, x[idx]));
                        // force x[0],x[1],...,x[idx-1] to l
                        index_t j = 0;
                        for(; j < idx; j++) {
                            if(!lister_in_l(t, x[j])) {
                                if(t->l_size >= t->k)
                                    break;
                                lister_push_l(t, x[j]);
                                lister_push(t, 
                                            to_command_idx(M_REWIND_L, x[j]));
                            }
                        }
                        if(j == idx)
                            lister_push(t, to_command(M_QUERY));
                    }
                }
            }
            break;
        case M_CLOSE:
            assert(idx > 0);
            lister_push(t, to_command_idx(M_OPEN, idx-1));
            break;
        case M_REWIND_U:
            bitset(t->u, idx, 1);
            t->u_size++;
            break;
        case M_REWIND_L:
            lister_pop_l(t);
            break;
        }
    }
    lister_push(t, to_command(M_QUERY));
    return 0;
}

/******************************************************** Root query builder. */

// Query builder for directed graphs
//
temppathq_t *build_temppathq_dir(graph_t *g, index_t k, index_t *kk)
{
    push_memtrack();

    index_t n = g->num_vertices;
    index_t m = g->num_edges;
    index_t tmax = g->max_time;
    index_t *pos = alloc_idxtab(n*tmax);
    index_t *adj = alloc_idxtab(n*tmax+2*m);
    index_t ns = k;
    shade_map_t *shade = (shade_map_t *) MALLOC(sizeof(shade_map_t)*n);

    temppathq_t *root = (temppathq_t *) MALLOC(sizeof(temppathq_t));
    root->is_stub  = 0;
    root->n        = g->num_vertices;
    root->k        = k;
    root->tmax     = tmax;
    root->pos      = pos;
    root->adj      = adj;
    root->nl       = 0;
    root->l        = (index_t *) MALLOC(sizeof(index_t)*root->nl);
    root->ns       = ns;
    root->shade    = shade;
    root->vert_loc = 0;
    root->vsum     = (scalar_t *) MALLOC(sizeof(scalar_t)*root->n);

    assert(tmax >= k-1);

    push_time();
    fprintf(stdout, "build query: ");
    fflush(stdout);

    push_time();
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n*tmax; u++)
        pos[u] = 0;
    double time = pop_time();
    fprintf(stdout, "[zero: %.2lf ms] ", time);
    fflush(stdout);
    
    push_time();
    index_t *e = g->edges;
#ifdef BUILD_PARALLEL
   // Parallel occurrence count
   // -- each thread is responsible for a group of bins, 
   //    all threads scan the entire list of edges
    index_t nt = num_threads();
    index_t block_size = n/nt;
#pragma omp parallel for
    for(index_t th = 0; th < nt; th++) {
        index_t start = th*block_size;
        index_t stop = (th == nt-1) ? n-1 : (start+block_size-1);
        for(index_t j = 0; j < 3*m; j+=3) {
            //index_t u = e[j];
            index_t v = e[j+1];
            index_t t = e[j+2];
            index_t *pos_t = (pos + (n*t));
            //if(start <= u && u <= stop) {
            //    // I am responsible for u, record adjacency to u
            //    pos_t[u]++;
            //}
            if(start <= v && v <= stop) {
                // I am responsible for v, record adjacency to v
                pos_t[v]++;
            }
        }
    }
#else
    for(index_t j = 0; j < 3*m; j+=3) {
        //index_t u = e[j];
        index_t v = e[j+1];
        index_t t = e[j+2];
        index_t *pos_t = pos + n*t;
        //pos_t[u]++;
        pos_t[v]++;
    }
#endif

    index_t run = prefixsum(n*tmax, pos, 1);
    assert(run == (n*tmax+m));
    time = pop_time();
    fprintf(stdout, "[pos: %.2lf ms] ", time);
    fflush(stdout);

    push_time();
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n*tmax; u++)
            adj[pos[u]] = 0;

    e = g->edges;
#ifdef BUILD_PARALLEL
    // Parallel aggregation to bins 
    // -- each thread is responsible for a group of bins, 
    //    all threads scan the entire list of edges
    nt = num_threads();
    block_size = n/nt;
#pragma omp parallel for
    for(index_t th = 0; th < nt; th++) {
        index_t start = th*block_size;
        index_t stop = (th == nt-1) ? n-1 : (start+block_size-1);
        for(index_t j = 0; j < 3*m; j+=3) {
            index_t u = e[j+0];
            index_t v = e[j+1];
            index_t t = e[j+2];
            //if(start <= u && u <= stop) {
            //    // I am responsible for u, record adjacency to u
            //    index_t pu = pos[n*t+u];
            //    adj[pu + 1 + adj[pu]++] = v;
            //}
            if(start <= v && v <= stop) {
                // I am responsible for v, record adjacency to v
                index_t pv = pos[n*t+v];
                adj[pv + 1 + adj[pv]++] = u;
            }
        }
    }
#else
    for(index_t j = 0; j < 3*m; j+=3) {
        index_t u = e[j+0];
        index_t v = e[j+1];
        index_t t = e[j+2];
        //index_t pu = pos[n*t+u];
        index_t pv = pos[n*t+v];       
        //adj[pu + 1 + adj[pu]++] = v;
        adj[pv + 1 + adj[pv]++] = u;
    }
#endif
    time = pop_time();
    fprintf(stdout, "[adj: %.2lf ms] ", time);
    fflush(stdout);

    //print_temppathq(root);
    push_time();
    adjsort(n*tmax, pos, adj);
    time = pop_time();
    fprintf(stdout, "[adjsort: %.2lf ms] ", time);
    fflush(stdout);

    push_time();
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++) {
        shade_map_t s = 0;
        for(index_t j = 0; j < k; j++)
            if(g->colors[u] == kk[j])
                s |= 1UL << j;
        shade[u] = s;
        //fprintf(stdout, "%4ld: 0x%08X\n", u, shade[u]);
    }
    time = pop_time();
    fprintf(stdout, "[shade: %.2lf ms] ", time);
    fflush(stdout);

    time = pop_time();
    fprintf(stdout, "done. [%.2lf ms] ", time);
    print_pop_memtrack();
    fprintf(stdout, " ");
    print_current_mem();
    fprintf(stdout, "\n");
    fflush(stdout);

    return root;
}

// Query builder for undirected graphs
//
temppathq_t *build_temppathq(graph_t *g, index_t k, index_t *kk)
{
    push_memtrack();

    index_t n = g->num_vertices;
    index_t m = g->num_edges;
    index_t tmax = g->max_time;
    index_t *pos = alloc_idxtab(n*tmax);
    index_t *adj = alloc_idxtab(n*tmax+2*m);
    index_t ns = k;
    shade_map_t *shade = (shade_map_t *) MALLOC(sizeof(shade_map_t)*n);

    temppathq_t *root = (temppathq_t *) MALLOC(sizeof(temppathq_t));
    root->is_stub   = 0;
    root->n         = g->num_vertices;
    root->k         = k;
    root->tmax      = tmax;
    root->pos       = pos;
    root->adj       = adj;
    root->nl        = 0;
    root->l         = (index_t *) MALLOC(sizeof(index_t)*root->nl);
    root->ns        = ns;
    root->shade     = shade;
    root->vert_loc  = 0;
    root->vsum      = (scalar_t *) MALLOC(sizeof(index_t)*root->n);

    assert(tmax >= k-1);

    push_time();
    fprintf(stdout, "build query: ");
    fflush(stdout);

    push_time();
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n*tmax; u++)
        pos[u] = 0;
    double time = pop_time();
    fprintf(stdout, "[zero: %.2lf ms] ", time);
    fflush(stdout);
    
    push_time();
    index_t *e = g->edges;
#ifdef BUILD_PARALLEL
   // Parallel occurrence count
   // -- each thread is responsible for a group of bins, 
   //    all threads scan the entire list of edges
    index_t nt = num_threads();
    index_t block_size = n/nt;
#pragma omp parallel for
    for(index_t th = 0; th < nt; th++) {
        index_t start = th*block_size;
        index_t stop = (th == nt-1) ? n-1 : (start+block_size-1);
        for(index_t j = 0; j < 3*m; j+=3) {
            index_t u = e[j];
            index_t v = e[j+1];
            index_t t = e[j+2];
            index_t *pos_t = (pos + (n*t));
            if(start <= u && u <= stop) {
                // I am responsible for u, record adjacency to u
                pos_t[u]++;
            }
            if(start <= v && v <= stop) {
                // I am responsible for v, record adjacency to v
                pos_t[v]++;
            }
        }
    }
#else
    for(index_t j = 0; j < 3*m; j+=3) {
        index_t u = e[j];
        index_t v = e[j+1];
        index_t t = e[j+2];
        index_t *pos_t = pos + n*t;
        pos_t[u]++;
        pos_t[v]++;
    }
#endif

    index_t run = prefixsum(n*tmax, pos, 1);
    assert(run == (n*tmax+2*m));
    time = pop_time();
    fprintf(stdout, "[pos: %.2lf ms] ", time);
    fflush(stdout);

    push_time();
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n*tmax; u++) {
            adj[pos[u]] = 0;
    }

    e = g->edges;
#ifdef BUILD_PARALLEL
    // Parallel aggregation to bins 
    // -- each thread is responsible for a group of bins, 
    //    all threads scan the entire list of edges
    nt = num_threads();
    block_size = n/nt;
#pragma omp parallel for
    for(index_t th = 0; th < nt; th++) {
        index_t start = th*block_size;
        index_t stop = (th == nt-1) ? n-1 : (start+block_size-1);
        for(index_t j = 0; j < 3*m; j+=3) {
            index_t u = e[j+0];
            index_t v = e[j+1];
            index_t t = e[j+2];
            if(start <= u && u <= stop) {
                // I am responsible for u, record adjacency to u
                index_t pu = pos[n*t+u];
                adj[pu + 1 + adj[pu]++] = v;
            }
            if(start <= v && v <= stop) {
                // I am responsible for v, record adjacency to v
                index_t pv = pos[n*t+v];
                adj[pv + 1 + adj[pv]++] = u;
            }
        }
    }
#else
    for(index_t j = 0; j < 3*m; j+=3) {
        index_t u = e[j+0];
        index_t v = e[j+1];
        index_t t = e[j+2];
        index_t pu = pos[n*t+u];
        index_t pv = pos[n*t+v];       
        adj[pu + 1 + adj[pu]++] = v;
        adj[pv + 1 + adj[pv]++] = u;
    }
#endif
    time = pop_time();
    fprintf(stdout, "[adj: %.2lf ms] ", time);
    fflush(stdout);

    push_time();
    adjsort(n*tmax, pos, adj);
    time = pop_time();
    fprintf(stdout, "[adjsort: %.2lf ms] ", time);
    fflush(stdout);

    push_time();
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++) {
        shade_map_t s = 0;
        for(index_t j = 0; j < k; j++)
            if(g->colors[u] == kk[j])
                s |= 1UL << j;
        shade[u] = s;
//        fprintf(stdout, "%4ld: 0x%08X\n", u, shade[u]);
    }
    time = pop_time();
    fprintf(stdout, "[shade: %.2lf ms] ", time);
    fflush(stdout);

    time = pop_time();
    fprintf(stdout, "done. [%.2lf ms] ", time);
    print_pop_memtrack();
    fprintf(stdout, " ");
    print_current_mem();
    fprintf(stdout, "\n");
    fflush(stdout);

    //print_temppathq(root);
    return root;
}

void query_pre_mk1(temppathq_t *in, temppathq_t **out_q, index_t **out_map)
{
    push_memtrack();

    index_t nt = num_threads();
    index_t i_n          = in->n;
    index_t k            = in->k;
    index_t tmax         = in->tmax;
    index_t *i_pos       = in->pos;
    index_t *i_adj       = in->adj;
    index_t ns           = in->ns;
    shade_map_t *i_shade = in->shade;

    push_time();
    fprintf(stdout, "query pre [1]: ");
    fflush(stdout);

    push_time();
    // input-to-output vertex map
    index_t *v_map_i2o   = (index_t *) MALLOC(sizeof(index_t)*i_n);
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < i_n; u++)
        v_map_i2o[u] = UNDEFINED;

    index_t v_cnt = 0;
#ifdef BUILD_PARALLEL
    // parallely construct input-to-output vertex map
    index_t block_size = i_n/nt;
    index_t t_vcnt[nt];

#pragma omp parallel for
    for(index_t th = 0; th < nt; th++) {
        index_t start = th*block_size;
        index_t stop = (th == nt-1) ? i_n-1 : (start+block_size-1);
        t_vcnt[th] = 0;
        for(index_t u = start; u <= stop; u++) {
            if(i_shade[u])
                v_map_i2o[u] = t_vcnt[th]++;
        }
    }
  
    // prefix sum
    for(index_t th = 1; th < nt; th++)
        t_vcnt[th] += t_vcnt[th-1];

#pragma omp parallel for
    for(index_t th = 0; th < nt; th++) {
        index_t start = th*block_size;
        index_t stop = (th == nt-1) ? i_n-1 : (start+block_size-1);
        index_t tsum = (th==0 ? 0 : t_vcnt[th-1]);
        for(index_t u = start; u <= stop; u++) {
            if(i_shade[u])
                v_map_i2o[u] += tsum;
        }
    }
    v_cnt = t_vcnt[nt-1];

#else
    // serially construct input-to-output vertex map
    for(index_t u = 0; u < i_n; u++) {
        if(i_shade[u])
            v_map_i2o[u] = v_cnt++;
    }
#endif

    // output-to-input vertex map 
    // required to reconstruct solution in original graph
    index_t o_n = v_cnt;
    index_t *v_map_o2i = (index_t *) MALLOC(sizeof(index_t)*o_n);
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < i_n; u++) {
        index_t o_u = v_map_i2o[u];
        if(o_u != UNDEFINED)
            v_map_o2i[o_u] = u;
    }

    fprintf(stdout, "[map: %.2lf ms] ", pop_time());
    fflush(stdout);
    push_time();

    // output position list
    index_t *o_pos = alloc_idxtab(o_n*tmax);
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < o_n*tmax; u++)
        o_pos[u] = 0;

    for(index_t t = 0; t < tmax; t++) {
        index_t *o_pos_t = o_pos + o_n*t;
        index_t *i_pos_t = i_pos + i_n*t;
        index_t block_size = i_n/nt;
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
        for(index_t th = 0; th < nt; th++) {
            index_t start = th*block_size;
            index_t stop = (th == nt-1) ? i_n-1 : (start+block_size-1);
            for(index_t u = start; u <= stop; u++) {
                index_t o_u =  v_map_i2o[u];
                if(o_u == UNDEFINED) continue;
                index_t i_pu = i_pos_t[u];
                index_t i_nu = i_adj[i_pu];
                index_t *i_adj_u = i_adj + i_pu;
                for(index_t j = 1; j <= i_nu; j++) {
                    index_t v = i_adj_u[j];
                    index_t o_v = v_map_i2o[v];
                    if(o_v == UNDEFINED) continue;
                    o_pos_t[o_u]++;
                }
            }
        }
    }

    index_t o_m   = parallelsum(o_n*tmax, o_pos);
    index_t run   = prefixsum(o_n*tmax, o_pos, 1);
    assert(run == (o_n*tmax+o_m));

    fprintf(stdout, "[pos: %.2lf ms] ", pop_time());
    fflush(stdout);
    push_time();

    // output adjacency list
    index_t *o_adj = alloc_idxtab(o_n*tmax + o_m);

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < o_n*tmax; u++)
        o_adj[o_pos[u]] = 0;

    for(index_t t = 0; t < tmax; t++) {
        index_t *o_pos_t = o_pos + o_n*t;
        index_t *i_pos_t = i_pos + i_n*t;
        index_t block_size = i_n/nt;
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
        for(index_t th = 0; th < nt; th++) {
            index_t start = th*block_size;
            index_t stop = (th == nt-1) ? i_n-1 : (start+block_size-1);
            for(index_t u = start; u <= stop; u++) {
                index_t o_u = v_map_i2o[u];
                if(o_u == UNDEFINED) continue;

                index_t i_pu = i_pos_t[u];
                index_t i_nu = i_adj[i_pu];
                index_t *i_adj_u = i_adj + i_pu;
                index_t o_pu = o_pos_t[o_u];
                for(index_t j = 1; j <= i_nu; j++) {
                    index_t v = i_adj_u[j];
                    index_t o_v = v_map_i2o[v];
                    if(o_v == UNDEFINED) continue;
                    
                    o_adj[o_pu + 1 + o_adj[o_pu]++] = o_v;
                }
            }
        }
    }

    fprintf(stdout, "[adj: %.2lf ms] ", pop_time());
    fflush(stdout);
    push_time();

    // output shade map
    shade_map_t *o_shade = (shade_map_t *) MALLOC(sizeof(shade_map_t)*o_n);
#ifdef BUILD_PARALLEL
#pragma omp parallel
#endif
    for(index_t u = 0; u < i_n; u++) {
        index_t o_u = v_map_i2o[u];
        if(o_u != UNDEFINED)
            o_shade[o_u] = i_shade[u];
    }

    fprintf(stdout, "[shade: %.2lf ms] ", pop_time());
    fflush(stdout);

    temppathq_t *out = (temppathq_t *) MALLOC(sizeof(temppathq_t));
    out->is_stub     = 0;
    out->n           = o_n;
    out->k           = k;
    out->tmax        = tmax;
    out->pos         = o_pos;
    out->adj         = o_adj;
    out->nl          = 0;
    out->l           = (index_t *) MALLOC(sizeof(index_t)*out->nl);
    out->ns          = ns;
    out->shade       = o_shade;
    out->vert_loc    = in->vert_loc;
    out->vsum        = (scalar_t *) MALLOC(sizeof(scalar_t)*out->n);

    *out_q           = out;
    *out_map         = v_map_o2i;

    FREE(v_map_i2o);

    fprintf(stdout, "done. [%.2lf ms] ", pop_time());
    print_pop_memtrack();
    fprintf(stdout, " ");
    print_current_mem();
    fprintf(stdout, "\n");
    fflush(stdout);
}

void query_pre_mk2(index_t is_dir, temppathq_t *in, temppathq_t **out_q, index_t **out_map)
{
    push_memtrack();

    index_t nt           = num_threads();
    index_t i_n          = in->n;
    index_t k            = in->k;
    index_t tmax         = in->tmax;
    index_t *i_pos       = in->pos;
    index_t *i_adj       = in->adj;
    index_t ns           = in->ns;
    shade_map_t *i_shade = in->shade;

    // Preprocessing steps
    // 1. merge graph temporal graph to a static instance
    // 2. build vertex localised sieve for static graph
    // 3. remove all vertices which are not incident to a match

    push_time();
    // building path query
    pathq_t * pathq = (pathq_t *) 0;
    if(is_dir) {
        pathq = build_pathq_dir(in); 
    } else {
        pathq = build_pathq(in);
    }

    // evaluate vertex localised sieve
    scalar_t master_sum = 0;
    scalar_t *master_vsum = (scalar_t *) MALLOC(sizeof(scalar_t)*i_n);
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < i_n; u++)
        master_vsum[u] = 0;

    // Note: restricting number of repetitions to one. Field size used is
    // GF(2^64) and per-vertex false negative probability is (2k-1)/2^{64}.
    // TODO: cross-verify experimental results
    // DONE: verified, a single run of sieve is sufficient
    index_t repeats = 1; 
    for(index_t r = 0; r < repeats; r++) {
        fprintf(stdout, "oracle [path]: ");

        scalar_t sum = pathq_execute(pathq);
        scalar_t *vsum = pathq->vsum;

        // Support size
        index_t support_size = 0;
#ifdef BUILD_PARALLEL
        index_t nt = num_threads();
        index_t block_size = i_n/nt;
        index_t ts_size[MAX_THREADS];
#pragma omp parallel for
        for(index_t th = 0; th < nt; th++) {
            ts_size[th] = 0;
            index_t start = th*block_size;
            index_t stop = (th == nt-1) ? i_n-1 : (start+block_size-1);
            for(index_t u = start; u <= stop; u++) {
                if(vsum[u] != 0)
                    ts_size[th]++;
            }
        }

        for(index_t th = 0; th < nt; th++){
            support_size += ts_size[th];
        }
#else
        for(index_t u = 0; u < i_n; u++) {
            if(vsum[u] != 0)
                support_size++;
        }
#endif
        fprintf(stdout, " -- %s [%ld]\n", sum!=0?"true":"false", support_size);
        fflush(stdout);

        // update master sum
        master_sum = (master_sum!=0 ? master_sum : sum);
        // update master vsum
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
        for(index_t u = 0; u < i_n; u++)
            master_vsum[u] = (master_vsum[u]!=0 ? master_vsum[u] : vsum[u]); 
    }
    
    // free memory
    pathq_free(pathq);

    //for(index_t u = 0; u < i_n; u++)
    //    fprintf(stdout, "%4ld:"SCALAR_FORMAT_STRING"\n", u+1, master_vsum[u]);

    // retain vertices which are incident to at least one match
    push_time();

    // input-to-output vertex map
    index_t *v_map_i2o   = (index_t *) MALLOC(sizeof(index_t)*i_n);
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < i_n; u++)
        v_map_i2o[u] = UNDEFINED;

    index_t v_cnt = 0;
#ifdef BUILD_PARALLEL
    // parallely construct input-to-output vertex map
    index_t block_size = i_n/nt;
    index_t t_vcnt[nt];

#pragma omp parallel for
    for(index_t th = 0; th < nt; th++) {
        index_t start = th*block_size;
        index_t stop = (th == nt-1) ? i_n-1 : (start+block_size-1);
        t_vcnt[th] = 0;
        for(index_t u = start; u <= stop; u++) {
            if(master_vsum[u])
                v_map_i2o[u] = t_vcnt[th]++;
        }
    }
  
    // prefix sum
    for(index_t th = 1; th < nt; th++)
        t_vcnt[th] += t_vcnt[th-1];

#pragma omp parallel for
    for(index_t th = 0; th < nt; th++) {
        index_t start = th*block_size;
        index_t stop = (th == nt-1) ? i_n-1 : (start+block_size-1);
        index_t tsum = (th==0 ? 0 : t_vcnt[th-1]);
        for(index_t u = start; u <= stop; u++) {
            if(master_vsum[u])
                v_map_i2o[u] += tsum;
        }
    }
    v_cnt = t_vcnt[nt-1];

#else
    // serially construct input-to-output vertex map
    for(index_t u = 0; u < i_n; u++) {
        if(master_vsum[u])
            v_map_i2o[u] = v_cnt++;
    }
#endif

    // output-to-input vertex map 
    // required to reconstruct solution in original graph
    index_t o_n = v_cnt;
    index_t *v_map_o2i = (index_t *) MALLOC(sizeof(index_t)*o_n);
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < i_n; u++) {
        index_t o_u = v_map_i2o[u];
        if(o_u != UNDEFINED)
            v_map_o2i[o_u] = u;
    }

    fprintf(stdout, "query pre [2]: ");
    fprintf(stdout, "[map: %.2lf ms] ", pop_time());
    fflush(stdout);
    push_time();

    // output position list
    index_t *o_pos = alloc_idxtab(o_n*tmax);
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < o_n*tmax; u++)
        o_pos[u] = 0;

    for(index_t t = 0; t < tmax; t++) {
        index_t *o_pos_t = o_pos + o_n*t;
        index_t *i_pos_t = i_pos + i_n*t;
        index_t block_size = i_n/nt;
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
        for(index_t th = 0; th < nt; th++) {
            index_t start = th*block_size;
            index_t stop = (th == nt-1) ? i_n-1 : (start+block_size-1);
            for(index_t u = start; u <= stop; u++) {
                index_t o_u =  v_map_i2o[u];
                if(o_u == UNDEFINED) continue;
                index_t i_pu = i_pos_t[u];
                index_t i_nu = i_adj[i_pu];
                index_t *i_adj_u = i_adj + i_pu;
                for(index_t j = 1; j <= i_nu; j++) {
                    index_t v = i_adj_u[j];
                    index_t o_v = v_map_i2o[v];
                    if(o_v == UNDEFINED) continue;
                    o_pos_t[o_u]++;
                }
            }
        }
    }

    index_t o_m   = parallelsum(o_n*tmax, o_pos);
    index_t run   = prefixsum(o_n*tmax, o_pos, 1);
    assert(run == (o_n*tmax+o_m));

    fprintf(stdout, "[pos: %.2lf ms] ", pop_time());
    fflush(stdout);
    push_time();

    // output adjacency list
    index_t *o_adj = alloc_idxtab(o_n*tmax + o_m);

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < o_n*tmax; u++)
        o_adj[o_pos[u]] = 0;

    for(index_t t = 0; t < tmax; t++) {
        index_t *o_pos_t = o_pos + o_n*t;
        index_t *i_pos_t = i_pos + i_n*t;
        index_t block_size = i_n/nt;
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
        for(index_t th = 0; th < nt; th++) {
            index_t start = th*block_size;
            index_t stop = (th == nt-1) ? i_n-1 : (start+block_size-1);
            for(index_t u = start; u <= stop; u++) {
                index_t o_u = v_map_i2o[u];
                if(o_u == UNDEFINED) continue;

                index_t i_pu = i_pos_t[u];
                index_t i_nu = i_adj[i_pu];
                index_t *i_adj_u = i_adj + i_pu;
                index_t o_pu = o_pos_t[o_u];
                for(index_t j = 1; j <= i_nu; j++) {
                    index_t v = i_adj_u[j];
                    index_t o_v = v_map_i2o[v];
                    if(o_v == UNDEFINED) continue;
                    
                    o_adj[o_pu + 1 + o_adj[o_pu]++] = o_v;
                }
            }
        }
    }

    fprintf(stdout, "[adj: %.2lf ms] ", pop_time());
    fflush(stdout);
    push_time();

    // output shade map
    shade_map_t *o_shade = (shade_map_t *) MALLOC(sizeof(shade_map_t)*o_n);
#ifdef BUILD_PARALLEL
#pragma omp parallel
#endif
    for(index_t u = 0; u < i_n; u++) {
        index_t o_u = v_map_i2o[u];
        if(o_u != UNDEFINED)
            o_shade[o_u] = i_shade[u];
    }
    fprintf(stdout, "[shade: %.2lf ms] ", pop_time());
    fprintf(stdout, "done. [%.2lf ms] ", pop_time());
    print_pop_memtrack();
    fprintf(stdout, " ");
    print_current_mem();
    fprintf(stdout, "\n");
    fflush(stdout);


    temppathq_t *out = (temppathq_t *) MALLOC(sizeof(temppathq_t));
    out->is_stub     = 0;
    out->n           = o_n;
    out->k           = k;
    out->tmax        = tmax;
    out->pos         = o_pos;
    out->adj         = o_adj;
    out->nl          = 0;
    out->l           = (index_t *) MALLOC(sizeof(index_t)*out->nl);
    out->ns          = ns;
    out->shade       = o_shade;
    out->vert_loc    = in->vert_loc;
    out->vsum        = (scalar_t *) MALLOC(sizeof(scalar_t)*out->n);

    *out_q           = out;
    *out_map         = v_map_o2i;

    FREE(master_vsum);
    FREE(v_map_i2o);
}

void query_post_mk1(index_t *uu, temppathq_t *in, temppathq_t **out_q, 
                    index_t **out_map)
{
    push_memtrack();

    index_t nt           = num_threads();
    index_t i_n          = in->n;
    index_t k            = in->k;
    index_t tmax         = in->tmax;
    index_t *i_pos       = in->pos;
    index_t *i_adj       = in->adj;
    index_t ns           = in->ns;
    shade_map_t *i_shade = in->shade;

    // output graph
    index_t o_n = k;

    push_time();
    fprintf(stdout, "subgraph: ");
    fflush(stdout);

    shellsort(k, uu);

    push_time();
    // input-to-output vertex map
    index_t *v_map_i2o   = (index_t *) MALLOC(sizeof(index_t)*i_n);
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < i_n; u++)
        v_map_i2o[u] = UNDEFINED;

    // serially construct input-to-output vertex map
    for(index_t i = 0; i < k; i++)
        v_map_i2o[uu[i]] = i;

    // output-to-input vertex map
    // required to reconstruct solution in original graph
    index_t *v_map_o2i = (index_t *) MALLOC(sizeof(index_t)*o_n);
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t i = 0; i < o_n; i++) {
        v_map_o2i[i] = uu[i];
    }

    fprintf(stdout, "[map: %.2lf ms] ", pop_time());
    fflush(stdout);
    push_time();

    // output position list
    index_t *o_pos = alloc_idxtab(o_n*tmax);
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < o_n*tmax; u++)
        o_pos[u] = 0;

    for(index_t t = 0; t < tmax; t++) {
        index_t *o_pos_t = o_pos + o_n*t;
        index_t *i_pos_t = i_pos + i_n*t;
        index_t block_size = i_n/nt;
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
        for(index_t th = 0; th < nt; th++) {
            index_t start = th*block_size;
            index_t stop = (th == nt-1) ? i_n-1 : (start+block_size-1);
            for(index_t u = start; u <= stop; u++) {
                index_t o_u =  v_map_i2o[u];
                if(o_u == UNDEFINED) continue;
                index_t i_pu = i_pos_t[u];
                index_t i_nu = i_adj[i_pu];
                index_t *i_adj_u = i_adj + i_pu;
                for(index_t j = 1; j <= i_nu; j++) {
                    index_t v = i_adj_u[j];
                    index_t o_v = v_map_i2o[v];
                    if(o_v == UNDEFINED) continue;
                    o_pos_t[o_u]++;
                }
            }
        }
    }

    index_t o_m   = parallelsum(o_n*tmax, o_pos);
    index_t run   = prefixsum(o_n*tmax, o_pos, 1);
    assert(run == (o_n*tmax+o_m));

    fprintf(stdout, "[pos: %.2lf ms] ", pop_time());
    fflush(stdout);
    push_time();

    // output adjacency list
    index_t *o_adj = alloc_idxtab(o_n*tmax + o_m);

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < o_n*tmax; u++)
        o_adj[o_pos[u]] = 0;

    for(index_t t = 0; t < tmax; t++) {
        index_t *o_pos_t = o_pos + o_n*t;
        index_t *i_pos_t = i_pos + i_n*t;
        index_t block_size = i_n/nt;
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
        for(index_t th = 0; th < nt; th++) {
            index_t start = th*block_size;
            index_t stop = (th == nt-1) ? i_n-1 : (start+block_size-1);
            for(index_t u = start; u <= stop; u++) {
                index_t o_u = v_map_i2o[u];
                if(o_u == UNDEFINED) continue;

                index_t i_pu = i_pos_t[u];
                index_t i_nu = i_adj[i_pu];
                index_t *i_adj_u = i_adj + i_pu;
                index_t o_pu = o_pos_t[o_u];
                for(index_t j = 1; j <= i_nu; j++) {
                    index_t v = i_adj_u[j];
                    index_t o_v = v_map_i2o[v];
                    if(o_v == UNDEFINED) continue;

                    o_adj[o_pu + 1 + o_adj[o_pu]++] = o_v;
                }
            }
        }
    }
    fprintf(stdout, "[adj: %.2lf ms] ", pop_time());
    fflush(stdout);
    push_time();

    // output shade map
    shade_map_t *o_shade = (shade_map_t *) MALLOC(sizeof(shade_map_t)*o_n);
#ifdef BUILD_PARALLEL
#pragma omp parallel
#endif
    for(index_t u = 0; u < i_n; u++) {
        index_t o_u = v_map_i2o[u];
        if(o_u != UNDEFINED)
            o_shade[o_u] = i_shade[u];
    }

    fprintf(stdout, "[shade: %.2lf ms] ", pop_time());
    fflush(stdout);

    temppathq_t *out = (temppathq_t *) MALLOC(sizeof(temppathq_t));
    out->is_stub     = 0;
    out->n           = o_n;
    out->k           = k;
    out->tmax        = tmax;
    out->pos         = o_pos;
    out->adj         = o_adj;
    out->nl          = 0;
    out->l           = (index_t *) MALLOC(sizeof(index_t)*out->nl);
    out->ns          = ns;
    out->shade       = o_shade;
    out->vert_loc    = in->vert_loc;
    out->vsum        = (scalar_t *) MALLOC(sizeof(scalar_t)*out->n);

    *out_q           = out;
    *out_map         = v_map_o2i;

    FREE(v_map_i2o);

    fprintf(stdout, "done. [%.2lf ms] ", pop_time());
    print_pop_memtrack();
    fprintf(stdout, " ");
    print_current_mem();
    fprintf(stdout, "\n");
    fflush(stdout); 
}

/****************************************************** Input reader (ASCII). */

void skipws(FILE *in)
{
    int c;
    do {
        c = fgetc(in);
        if(c == '#') {
            do {
                c = fgetc(in);
            } while(c != EOF && c != '\n');
        }
    } while(c != EOF && isspace(c));
    if(c != EOF)
        ungetc(c, in);
}

#define CMD_NOP              0
#define CMD_TEST_UNIQUE      1
#define CMD_TEST_COUNT       2
#define CMD_LIST_FIRST       3
#define CMD_LIST_FIRST_VLOC  4
#define CMD_LIST_ALL_VLOC    5
#define CMD_RUN_ORACLE       6

char *cmd_legend[] = { "no operation", "test unique", "test count", "list first", "list first (localised)", "list all (localised)", "run oracle" };

void reader_ascii(FILE *in, 
                  graph_t **g_out, index_t *k_out, index_t **kk_out, 
                  index_t *cmd_out, index_t **cmd_args_out)
{
    push_time();
    push_memtrack();
    
    index_t n         = 0;
    index_t m         = 0;
    index_t tmax      = 0;
    index_t is_dir    = 0;
    graph_t *g        = (graph_t *) 0;
    index_t *kk       = (index_t *) 0;
    index_t cmd       = CMD_NOP;
    index_t *cmd_args = (index_t *) 0;
    index_t i, j, d, k, t;
    skipws(in);
    while(!feof(in)) {
        skipws(in);
        int c = fgetc(in);
        switch(c) {
        case 'p':
            if(g != (graph_t *) 0)
                ERROR("duplicate parameter line");
            skipws(in);
            if(fscanf(in, "motif %ld %ld %ld %ld", &n, &m, &tmax, &is_dir) != 4)
                ERROR("invalid parameter line");
            if(n <= 0 || m < 0 ) {
                ERROR("invalid input parameters (n = %ld, m = %ld, tmax = %ld)",
                       n, m, tmax);
            }
            g = graph_alloc(n);
            graph_set_is_directed(g, is_dir);
            graph_set_max_time(g, tmax);
            break;
        case 'e':
            if(g == (graph_t *) 0)
                ERROR("parameter line must be given before edges");
            skipws(in);
            if(fscanf(in, "%ld %ld %ld", &i, &j, &t) != 3)
                ERROR("invalid edge line");
            //if(i < 1 || i > n || j < 1 || j > n || t < 1 || t > tmax) {
            //    ERROR("invalid edge (i = %ld, j = %ld t = %ld with n = %ld, tmax = %ld)", 
            //          i, j, t, n, tmax);
            //}
            graph_add_edge(g, i-1, j-1, t-1);
            break;
        case 'n':
            if(g == (graph_t *) 0)
                ERROR("parameter line must be given before vertex colors");
            skipws(in);
            if(fscanf(in, "%ld %ld", &i, &d) != 2)
                ERROR("invalid color line");
            if(i < 1 || i > n || d < 1)
                ERROR("invalid color line (i = %ld, d = %ld with n = %ld)", 
                      i, d, n);
            graph_set_color(g, i-1, d-1);
            break;
        case 'k':
            if(g == (graph_t *) 0)
                ERROR("parameter line must be given before motif");
            skipws(in);
            if(fscanf(in, "%ld", &k) != 1)
                ERROR("invalid motif line");
            if(k < 1 || k > n)
                ERROR("invalid motif line (k = %ld with n = %d)", k, n);
            kk = alloc_idxtab(k);
            for(index_t u = 0; u < k; u++) {
                skipws(in);
                if(fscanf(in, "%ld", &i) != 1)
                    ERROR("error parsing motif line");
                if(i < 1)
                    ERROR("invalid color on motif line (i = %ld)", i);
                kk[u] = i-1;
            }
            break;
        case 't':
            if(g == (graph_t *) 0 || kk == (index_t *) 0)
                ERROR("parameter and motif lines must be given before test");
            skipws(in);
            {
                char cmdstr[128];
                if(fscanf(in, "%100s", cmdstr) != 1)
                    ERROR("invalid test command");
                if(!strcmp(cmdstr, "unique")) {
                    cmd_args = alloc_idxtab(k);
                    for(index_t u = 0; u < k; u++) {
                        skipws(in);
                        if(fscanf(in, "%ld", &i) != 1)
                            ERROR("error parsing test line");
                        if(i < 1 || i > n)
                            ERROR("invalid test line entry (i = %ld)", i);
                        cmd_args[u] = i-1;
                    }
                    heapsort_indext(k, cmd_args);
                    for(index_t u = 1; u < k; u++)
                        if(cmd_args[u-1] >= cmd_args[u])
                            ERROR("test line contains duplicate entries");
                    cmd = CMD_TEST_UNIQUE;
                } else {
                    if(!strcmp(cmdstr, "count")) {
                        cmd_args = alloc_idxtab(1);
                        skipws(in);
                        if(fscanf(in, "%ld", &i) != 1)
                            ERROR("error parsing test line");
                        if(i < 0)
                            ERROR("count on test line cannot be negative");
                        cmd = CMD_TEST_COUNT;
                        cmd_args[0] = i;
                    } else {
                        ERROR("unrecognized test command \"%s\"", cmdstr);
                    }
                }
            }
            break;
        case EOF:
            break;
        default:
            ERROR("parse error");
        }
    }

    if(g == (graph_t *) 0)
        ERROR("no graph given in input");
    if(kk == (index_t *) 0)
        ERROR("no motif given in input");

    for(index_t i = 0; i < n; i++) {
        if(g->colors[i] == -1)
            ERROR("no color assigned to vertex i = %ld", i);
    }
    double time = pop_time();
    fprintf(stdout, 
            "input: n = %ld, m = %ld, k = %ld, t = %ld [%.2lf ms] ", 
            g->num_vertices,
            g->num_edges,
            k,
            g->max_time,
            time);
    print_pop_memtrack();
    fprintf(stdout, " ");
    print_current_mem();
    fprintf(stdout, "\n");    
    
    *g_out = g;
    *k_out = k;
    *kk_out = kk;
    *cmd_out = cmd;
    *cmd_args_out = cmd_args;
}

/***************************************************** Input reader (binary). */

#define BIN_MAGIC 0x1234567890ABCDEFUL

void reader_bin(FILE *in, 
                graph_t **g_out, index_t *k_out, index_t **kk_out, 
                index_t *cmd_out, index_t **cmd_args_out)
{
    push_time();
    push_memtrack();
    
    index_t magic = 0;
    index_t n = 0;
    index_t m = 0;
    graph_t *g = (graph_t *) 0;
    index_t k = 0;
    index_t has_target = 0;
    index_t *kk = (index_t *) 0;
    index_t cmd = CMD_NOP;
    index_t *cmd_args = (index_t *) 0;
    
    if(fread(&magic, sizeof(index_t), 1UL, in) != 1UL)
        ERROR("error reading input");
    if(magic != BIN_MAGIC)
        ERROR("error reading input");
    if(fread(&n, sizeof(index_t), 1UL, in) != 1UL)
        ERROR("error reading input");
    if(fread(&m, sizeof(index_t), 1UL, in) != 1UL)
        ERROR("error reading input");
    assert(n >= 0 && m >= 0 && m%2 == 0);
    g = graph_alloc(n);
    index_t *e = graph_edgebuf(g, m/2);
    if(fread(e, sizeof(index_t), m, in) != m)
        ERROR("error reading input");
    if(fread(g->colors, sizeof(index_t), n, in) != n)
        ERROR("error reading input");
    if(fread(&has_target, sizeof(index_t), 1UL, in) != 1UL)
        ERROR("error reading input");
    assert(has_target == 0 || has_target == 1);
    if(has_target) {
        if(fread(&k, sizeof(index_t), 1UL, in) != 1UL)
            ERROR("error reading input");
        assert(k >= 0);
        kk = alloc_idxtab(k);
        if(fread(kk, sizeof(index_t), k, in) != k)
            ERROR("error reading input");         
        if(fread(&cmd, sizeof(index_t), 1UL, in) != 1UL)
            ERROR("error reading input");         
        switch(cmd) {
        case CMD_NOP:
            break;
        case CMD_TEST_UNIQUE:
            cmd_args = alloc_idxtab(k);
            if(fread(cmd_args, sizeof(index_t), k, in) != k)
                ERROR("error reading input");         
            shellsort(k, cmd_args);
            break;          
        case CMD_TEST_COUNT:
            cmd_args = alloc_idxtab(1);
            if(fread(cmd_args, sizeof(index_t), 1UL, in) != 1UL)
                ERROR("error reading input");                         
            break;          
        default:
            ERROR("invalid command in binary input stream");
            break;          
        }
    }

    double time = pop_time();
    fprintf(stdout, 
            "input: n = %ld, m = %ld, k = %ld [%.2lf ms] ", 
            g->num_vertices,
            g->num_edges,
            k,
            time);
    print_pop_memtrack();
    fprintf(stdout, " ");
    print_current_mem();
    fprintf(stdout, "\n");
    
    *g_out = g;
    *k_out = k;
    *kk_out = kk;
    *cmd_out = cmd;
    *cmd_args_out = cmd_args;
}

/************************************************************ Temporal DFS. */

index_t temp_dfs(index_t n, index_t k, index_t tmax, index_t *pos, 
                 index_t *adj, index_t *in_stack, stk_t *s)
{
    stack_node_t e;
    stack_top(s, &e);
    index_t u    = e.u;
    index_t l    = e.l;
    index_t tmin = e.t;

    // reached depth 'k'
    if(s->n == k) // TODO: fix this to s->n == k
        return 1;

    for(index_t t = tmin; t < tmax; t++) {
        index_t *pos_t = pos + t*n;
        index_t pu = pos_t[u];
        index_t nu = adj[pu];
        if(nu == 0) continue;
        index_t *adj_u = adj + pu;
        for(index_t i = 1; i <= nu; i++) {
            index_t v = adj_u[i];
            if(in_stack[v]) continue;

            stack_node_t e;
            e.u = v;
            e.l = l+1;
            e.t = t+1;
            stack_push(s, &e);
            in_stack[v] = 1;

            if(temp_dfs(n, k, tmax, pos, adj, in_stack, s))
                return 1;
            stack_pop(s, &e);
            in_stack[v] = 0;
        } 
    }
    return 0; // not found
}

index_t find_temppath(temppathq_t *root, index_t *uu, index_t *tt) 
{
    index_t n           = root->n;
    index_t k           = root->k;
    index_t tmax        = root->tmax;
    index_t *pos        = root->pos;
    index_t *adj        = root->adj;

    // alloc memory
    index_t *v_rand     = alloc_idxtab(n);
    // random permutation of vertices
    index_t seed = irand();
    randperm(n, seed, v_rand);

    index_t *in_stack   = alloc_idxtab(n);
    stk_t *s          = stack_alloc(k);
    for(index_t j = 0; j < n; j++) {
        for(index_t i = 0; i < n; i++)
            in_stack[i] = 0;

        index_t u = v_rand[j];
        stack_node_t e;
        e.u = u;
        e.l = 1;
        e.t = 0;
        stack_push(s, &e);
        in_stack[u] = 1;

        if(temp_dfs(n, k, tmax, pos, adj, in_stack, s)) {
            index_t cnt = 0;
            while(s->n) {
                stack_node_t e;
                stack_pop(s, &e);
                index_t u = e.u;
                index_t t = e.t;
                uu[cnt] = u;
                tt[cnt] = t;
                cnt++;
            }
            break;
        } else {
            stack_empty(s);
        }
    }

    FREE(v_rand);
    FREE(in_stack);
    stack_free(s);
    return 1; 
}

/********************************************************** temporal rev-DFS. */
// 

index_t temp_revdfs(index_t n, index_t k, index_t tmax, index_t *pos,
                    index_t *adj, index_t *color, index_t *kk_in,
                    index_t *in_stack, stk_t *s, index_t *uu_out,
                    index_t *tt_out, index_t *t_opt)
{
    if(s->n >= k) {
        // reached depth k
        assert(s->n <= k);
        // allocate memory
        index_t *uu_sol = (index_t *) malloc(k*sizeof(index_t));
        index_t *kk_sol = (index_t *) malloc(k*sizeof(index_t));
        index_t *tt_sol = (index_t *) malloc(k*sizeof(index_t));

        // get vertices in stack
        stack_get_vertices(s, uu_sol);
        stack_get_timestamps(s, tt_sol);
        // get vertex colors
        for(index_t i = 0; i < k; i++)
            kk_sol[i] = color[uu_sol[i]];
        shellsort(k, kk_sol);

        // check if colors match
        index_t is_motif = 1;
        for(index_t i = 0; i < k; i++) {
            if(kk_sol[i] != kk_in[i]) {
                is_motif = 0;
                break;
            }
        }

        // match found
        if(is_motif) {
            stack_node_t e;
            stack_top(s, &e);
            if(*t_opt > e.t) {
                // copy solution vertices
                for(index_t i = 0; i < k; i++)
                    uu_out[i] = uu_sol[i];
                // copy solution timestamps
                for(index_t i = 0; i < k; i++)
                    tt_out[i] = tt_sol[i];
                *t_opt = e.t;
            }
        }

        // free memory
        free(uu_sol);
        free(kk_sol);
        free(tt_sol);
        return 1;
    } else {
        stack_node_t e;
        stack_top(s, &e);
        index_t u    = e.u;
        //index_t l    = e.l;
        index_t t_start = e.t;
        index_t t_end = 0;

        for(index_t t = t_start-1; t >= t_end; t--) {
            index_t *pos_t = pos + t*n;
            index_t pu = pos_t[u];
            index_t nu = adj[pu];

            if(nu == 0) continue;
            index_t *adj_u = adj + pu;
            for(index_t i = 1; i <= nu; i++) {
                index_t v = adj_u[i];
                if(in_stack[v]) continue;

                stack_node_t e;
                e.u = v;
                //e.l = l+1;
                e.t = t;
                stack_push(s, &e);
                in_stack[v] = 1;

                // recursive call to depth k
                temp_revdfs(n, k, tmax, pos, adj, color, kk_in, in_stack, s,
                            uu_out, tt_out, t_opt);

                stack_pop(s, &e);
                in_stack[v] = 0;
            }
        }
    }
    return 1; // not found
}

index_t exhaustive_search(temppathq_t *root, 
                          index_t *kk, 
                          index_t *color,
                          index_t *v_map,
                          index_t cmd)
{
    push_time();
    push_memtrack();

    index_t nt = num_threads();

    index_t n           = root->n;
    index_t k           = root->k;
    index_t tmax        = root->tmax;
    index_t *pos        = root->pos;
    index_t *adj        = root->adj;
    scalar_t *vsum      = root->vsum;
    index_t *vsum_cnt_nt = alloc_idxtab(nt+1);

    push_time();
    index_t block_size = n/nt;
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t th = 0; th < nt; th++) {
        index_t start = th*block_size;
        index_t stop = (th == nt-1) ? n-1 : (start+block_size-1);
        index_t cnt = 0;
        for(index_t i = start; i <= stop; i++)
            cnt += (vsum[i] ? 1 : 0);
        vsum_cnt_nt[th] = cnt;
    }

    // cosolidate thread counts
    vsum_cnt_nt[nt] = 0;
    prefixsum(nt+1, vsum_cnt_nt, 0);
    index_t vsum_cnt = vsum_cnt_nt[nt];

    // get vertices with non-zero value in `vsum`
    index_t *vsum_vertices = alloc_idxtab(vsum_cnt);
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t th = 0; th < nt; th++) {
        index_t start = th*block_size;
        index_t stop = (th == nt-1) ? n-1 : (start+block_size-1);
        index_t j = vsum_cnt_nt[th];
        for(index_t i = start; i <= stop; i++) {
            if(vsum[i]) vsum_vertices[j++] = i;
        }
    }

    index_t *v_seq = alloc_idxtab(vsum_cnt);
    index_t seed = irand();
    randperm(vsum_cnt, seed, v_seq);

    double init_time = pop_time();
    push_time();

    index_t *uu_sol_nt = alloc_idxtab(k*nt);
    index_t *tt_sol_nt = alloc_idxtab(k*nt);
    index_t *in_stack_nt = alloc_idxtab(n*nt);
    index_t *t_opt_nt = alloc_idxtab(nt);

    block_size = vsum_cnt/nt;

    double dfs_time = 0.0;

    volatile index_t found = 0;
    if(cmd == CMD_LIST_FIRST_VLOC) {
#ifdef BUILD_PARALLEL
#pragma omp parallel for shared(found)
#endif
        for(index_t th = 0; th < nt; th++) {
            index_t start = th*block_size;
            index_t stop = (th == nt-1) ? vsum_cnt-1 : (start+block_size-1);

            index_t *uu_sol = uu_sol_nt + th*k;
            index_t *tt_sol = tt_sol_nt + th*k;
            index_t *in_stack = in_stack_nt +th*n;
            index_t t_opt = MATH_INF;
            stk_t *s = stack_alloc(k);
            for(index_t j = start; j <= stop; j++) {
                if(found) break;

                index_t u = vsum_vertices[v_seq[j]];
                for(index_t i = 0; i < n; i++)
                    in_stack[i] = 0;
            
                stack_node_t e;
                e.u = u;
                //e.l = k;
                e.t = tmax;
                stack_push(s, &e);
                in_stack[u] = 1;
                temp_revdfs(n, k, tmax, pos, adj, color, kk, in_stack, s, uu_sol, tt_sol, &t_opt);
                if(t_opt != MATH_INF) {
                    found = th+1;
                }
                stack_empty(s);
            }
            stack_free(s);
        }
        // found a solution
        if(found) {
            index_t th = found-1;
            index_t *uu_sol = uu_sol_nt + th*k;
            index_t *tt_sol = tt_sol_nt + th*k;

            dfs_time = pop_time();
            fprintf(stdout, "solution [%ld, %.2lfms]: ", tt_sol[0], pop_time());
            for(index_t i = k-1; i > 0; i--) {
                index_t u = v_map[uu_sol[i]];
                index_t v = v_map[uu_sol[i-1]];
                index_t t = tt_sol[i-1];
                fprintf(stdout, "[%ld, %ld, %ld]%s", u+1, v+1, t, i==1?"\n":" ");
            }
        }
    } else {
        //TODO: implement listing all solutions 
        fprintf(stdout, "listing all solutions not supported\n");
    }

    if(!found) dfs_time = pop_time();

    FREE(vsum_cnt_nt);
    FREE(uu_sol_nt);
    FREE(tt_sol_nt);
    FREE(in_stack_nt);
    FREE(t_opt_nt);
    FREE(v_seq);
    FREE(vsum_vertices);
   
    fprintf(stdout, "exhaustive-search: [init: %.2lfms] [dfs: %.2lfms] done."
                    " [%.2lfms] ", init_time, dfs_time, pop_time());
    print_pop_memtrack();
    fprintf(stdout, " ");
    print_current_mem();
    fprintf(stdout, " -- %s\n", found?"true":"false");
    fflush(stdout);

    return (found ? 1 : 0);
}


/******************************************************* Program entry point. */

#define PRE_NOP 0
#define PRE_MK1 1
#define PRE_MK2 2
#define PRE_MK3 3

int main(int argc, char **argv)
{
    GF_PRECOMPUTE;

    push_time();
    push_memtrack();
    
    index_t precomp         = PRE_NOP;
    index_t arg_cmd         = CMD_NOP;
    index_t flag_help       = 0;
    index_t have_seed       = 0;
    index_t have_input      = 0;
    index_t find_optimal    = 0;
    index_t seed            = 123456789;
    char *filename          = (char *) 0;
    for(index_t f = 1; f < argc; f++) {
        if(argv[f][0] == '-') {
            if(!strcmp(argv[f], "-h") || !strcmp(argv[f], "-help")) { 
                flag_help = 1;
                break;
            }
            if(!strcmp(argv[f], "-bin")) {
                flag_bin_input = 1;
            }
            if(!strcmp(argv[f], "-ascii")) {
                flag_bin_input = 0;
            }
            if(!strcmp(argv[f], "-pre")) {
                if(f == argc -1)
                    ERROR("preprocessing argument missing from command line");
                precomp = atol(argv[++f]);
            }
            if(!strcmp(argv[f], "-optimal")) {
                find_optimal = 1;
            }
            if(!strcmp(argv[f], "-oracle")) {
                arg_cmd = CMD_RUN_ORACLE;
            }
            if(!strcmp(argv[f], "-first")) {
                arg_cmd = CMD_LIST_FIRST;
            }
            if(!strcmp(argv[f], "-first-vloc")) {
               arg_cmd = CMD_LIST_FIRST_VLOC; 
            }
            if(!strcmp(argv[f], "-all-vloc")) {
                arg_cmd = CMD_LIST_ALL_VLOC;
            }
            if(!strcmp(argv[f], "-seed")) {
                if(f == argc - 1)
                    ERROR("random seed missing from command line");
                seed = atol(argv[++f]);
                have_seed = 1;
            }
            if(!strcmp(argv[f], "-in")) {
                if(f == argc - 1)
                    ERROR("input file missing from command line");
                have_input = 1;
                filename = argv[++f];
            }
        }
    }

    fprintf(stdout, "invoked as:");
    for(index_t f = 0; f < argc; f++)
        fprintf(stdout, " %s", argv[f]);
    fprintf(stdout, "\n");

    if(flag_help) {
        fprintf(stdout,
                "usage: %s -pre <value> -optimal -<command-type> -seed <value> -in <input-file> -<file-type> \n"
                "       %s -h/help\n"
                "\n"
                "  -pre <value>     : <0>   -  no preprocessing (default)\n"
                "                     <1>   -  preprocess step-1\n"
                "                     <2>   -  preprocess step-2\n"
                "                     <3>   -  preprocess step-1 and step-2\n"
                "  -optimal         : obtain optimal solution (optional)\n"
                "  -<command-type>  : <oracle> 	   - decide existence of a solution\n"
                "                     <first>  	   - extract one solution\n"
                "                     <first-vloc> - extract one solution (vertex localisation)\n"
                " -seed <value>     : integer value in range 1 to 2^32 -1\n"
				"					  default value `%ld`\n"
                " -in <input-file>  : read from <input file>\n"
                "                     read from <stdin> by default\n"
                " -<file-type>      : ascii  - ascii input file (default) \n"
                "                     bin    - binary input file \n"
                " -h or -help       : help\n"
                "\n"
                , argv[0], argv[0], seed);
        return 0;
    }

    if(have_seed == 0) {
        fprintf(stdout, 
                "no random seed given, defaulting to %ld\n", seed);
    }
    fprintf(stdout, "random seed = %ld\n", seed);
   
    FILE *in = stdin;
    if(have_input) {
        in = fopen(filename, "r");
        if(in == NULL)
            ERROR("unable to open file '%s'", filename);
    } else {
        fprintf(stdout, "no input file specified, defaulting to stdin\n");
    }
    fflush(stdout);

    srand(seed); 

    graph_t *g;
    index_t k;
    index_t *kk;
    index_t input_cmd;
    index_t *cmd_args;
    if(flag_bin_input) {
        reader_bin(in, &g, &k, &kk, &input_cmd, &cmd_args);
    } else {
        reader_ascii(in, &g, &k, &kk, &input_cmd, &cmd_args);
    }
    index_t cmd = input_cmd;  // by default execute command in input stream
    if(arg_cmd != CMD_NOP)
        cmd = arg_cmd;        // override command in input stream

    // build root query
    index_t is_dir = 0;
    temppathq_t *root = (temppathq_t *) 0;
    if(g->is_directed) {
        is_dir = 1;
        root = build_temppathq_dir(g, k, kk); 
    } else {
        root = build_temppathq(g, k, kk);
    }

    // keep a copy of colors
    index_t *color = (index_t *) 0;
    if(cmd == CMD_LIST_FIRST_VLOC || cmd == CMD_LIST_ALL_VLOC) {
        color = alloc_idxtab(root->n);
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
        for(index_t u = 0; u < root->n; u++)
            color[u] = g->colors[u];
    }
    //free graph
    graph_free(g);
    
    push_time();

    push_time();
    // preprocess query
    index_t *v_map_pre1;
    index_t *v_map_pre2;
    switch(precomp) {
    case PRE_NOP:
    {
        // no precomputation
        fprintf(stdout, "no preprocessing, default execution\n");
        break;
    }
    case PRE_MK1:
    {
        // preprocess: remove vertices with no matching colors
        temppathq_t *root_pre;
        query_pre_mk1(root, &root_pre, &v_map_pre1);
        temppathq_free(root);
        root = root_pre;

        // preprocessed graph statistics
        index_t o_n    = root->n;
        index_t tmax   = root->tmax;
        index_t *o_pos = root->pos;
        index_t *o_adj = root->adj;
        index_t o_m = (o_pos[o_n*(tmax-1) + o_n-1] + 
                      o_adj[o_pos[o_n*(tmax-1) + o_n-1]] - (o_n*tmax) + 1)/2;
        fprintf(stdout, "output pre [1]: n = %ld, m = %ld, k = %ld \n", 
                         o_n, o_m, k);

        // required to reconstruct the solution in original graph
        //FREE(v_map_pre1);
        break;
    }
    case PRE_MK2:
    {
        // preprocess: constructing vertex localised sieve in static graph
        temppathq_t *root_pre;
        query_pre_mk2(is_dir, root, &root_pre, &v_map_pre2);
        temppathq_free(root);
        root= root_pre;

        // preprocessed graph statistics
        index_t o_n    = root->n;
        index_t tmax   = root->tmax;
        index_t *o_pos = root->pos;
        index_t *o_adj = root->adj;
        index_t o_m = (o_pos[o_n*(tmax-1) + o_n-1] + 
                      o_adj[o_pos[o_n*(tmax-1) + o_n-1]] - (o_n*tmax)+1)/2;
        fprintf(stdout, "output pre [2]: n = %ld, m = %ld, k = %ld \n", 
                         o_n, o_m, k);

        // required to reconstruct the solution in original graph
        //FREE(v_map_pre2);
        break;
    }
    case PRE_MK3:
    {
        // -- execute all preprocessing steps --
        //
        // preprocess: remove vertices with no matching colors
        temppathq_t *root_pre1;
        query_pre_mk1(root, &root_pre1, &v_map_pre1);
        temppathq_free(root);
        root = root_pre1;

        // preprocessed graph statistics
        index_t o_n    = root->n;
        index_t tmax   = root->tmax;
        index_t *o_pos = root->pos;
        index_t *o_adj = root->adj;
        index_t o_m = (o_pos[o_n*(tmax-1) + o_n-1] + 
                      o_adj[o_pos[o_n*(tmax-1) + o_n-1]] - (o_n*tmax)+1)/2;
        fprintf(stdout, "output pre [1]: n = %ld, m = %ld, k = %ld \n", 
                         o_n, o_m, k);
 
        // preprocess: constructing vertex localised sieve in static graph
        temppathq_t *root_pre2;
        query_pre_mk2(is_dir, root, &root_pre2, &v_map_pre2);
        temppathq_free(root);
        root= root_pre2;

        // preprocessed graph statistics
        o_n   = root->n;
        o_pos = root->pos;
        o_adj = root->adj;
        o_m = (o_pos[o_n*(tmax-1) + o_n-1] + 
               o_adj[o_pos[o_n*(tmax-1) + o_n-1]] - (o_n*tmax)+1)/2;
        fprintf(stdout, "output pre [2]: n = %ld, m = %ld, k = %ld \n", 
                         o_n, o_m, k);
 
        // required to reconstruct the solution in original graph
        //FREE(v_map_pre1);
        //FREE(v_map_pre2);
        break;
    }
    default:
        break;
    }

    double precomp_time = pop_time();
    push_time();

    index_t SOLUTION_EXISTS = 0; // default: assume solution do not exists
    // find optimal solution
    if(find_optimal) {
        // --- optimal solution ---
        // 
        fprintf(stdout, "optimal : min = %ld, max = %ld\n", k-1, root->tmax);
        // binary search: obtain optimal value of `t`
        index_t t_opt = root->tmax;
        index_t tmax = root->tmax;
        index_t low = k-1;
        index_t high = tmax;
        while(low < high) {
            index_t mid = (low+high)/2;
            root->tmax = mid;
            fprintf(stdout, "%13ld [%ld:%ld]\t\t", mid, low, high);
            if(temppathq_execute(root)) {
                if(t_opt > root->tmax) 
                    t_opt = root->tmax;
                high = mid;
                fprintf(stdout, " -- true\n");
                fflush(stdout);
                SOLUTION_EXISTS = 1;
            } else {
                low = mid + 1;
                fprintf(stdout, " -- false\n");
                fflush(stdout);
            }
        }
        root->tmax = t_opt;
        if(!SOLUTION_EXISTS && CMD_NOP) {
        //    fprintf(stdout, " -- false\n");
            fflush(stdout);
            temppathq_free(root);
        }
    }

    double opt_time = pop_time();
    fprintf(stdout, "command: %s\n", cmd_legend[cmd]);
    fflush(stdout);
    push_time();

    // execute command
    switch(cmd) {
    case CMD_NOP:
        {
            // no operation
            temppathq_free(root);
            break;
        }
    case CMD_TEST_UNIQUE:
        {
            // ---- test unique ---
            //
            // check if the solution is unique
            index_t n = root->n;
            index_t k = root->k;
            lister_t *t = lister_alloc(n, k, root);
            index_t *get = alloc_idxtab(k);
            index_t ct = 0;
            while(lister_run(t, get)) {
                assert(ct == 0);
                fprintf(stdout, "found %ld: ", ct);
                for(index_t i = 0; i < k; i++)
                    fprintf(stdout, "%ld%s", get[i], i == k-1 ? "\n" : " ");
                for(index_t l = 0; l < k; l++)
                    assert(get[l] == cmd_args[l]);
                ct++;
            }
            assert(ct == 1);
            FREE(get);
            lister_free(t);
        }
        break;
    case CMD_LIST_FIRST:
        {
            // --- list first solution ---
            //
            // list vertices: obtain `k` vertices satisfying our constraints
            index_t n = root->n;
            index_t k = root->k;
            lister_t *t = lister_alloc(n, k, root);
            index_t *get = alloc_idxtab(k);
            index_t ct = 0;
            if(lister_run(t, get)) {
                fprintf(stdout, "found %ld: ", ct);
                switch(precomp) {
                case PRE_NOP:
                    for(index_t i = 0; i < k; i++)
                        fprintf(stdout, "%ld%s", get[i]+1, i == k-1 ? "\n" : " ");
                    break;
                case PRE_MK1:
                    for(index_t i = 0; i < k; i++)
                        fprintf(stdout, "%ld%s", v_map_pre1[get[i]]+1, i == k-1 ? "\n" : " ");
                    break;
                case PRE_MK2:
                    for(index_t i = 0; i < k; i++)
                        fprintf(stdout, "%ld%s", v_map_pre2[get[i]]+1, i == k-1 ? "\n" : " ");
                    break;
                case PRE_MK3:
                    for(index_t i = 0; i < k; i++)
                        fprintf(stdout, "%ld%s", v_map_pre1[v_map_pre2[get[i]]]+1, i == k-1 ? "\n" : " ");
                    break;
                default:
                    break;
                }
                ct++;
                //if(cmd == CMD_LIST_FIRST || CMD_LIST_OPT)
                //    break;
            }

            // post-processing: obtain vertex-induced subgraph using 
            // k-vertices of previous step
            if(ct) {
                push_time();
                index_t *v_map_post     = (index_t *) 0;
                temppathq_t *root_post  = (temppathq_t *) 0;
                query_post_mk1(get, root, &root_post, &v_map_post);

                // final vertex map
                index_t *v_map = alloc_idxtab(k);
                switch(precomp) {
                case PRE_NOP:
                    for(index_t i = 0; i < k; i++)
                        v_map[i] = v_map_post[i];
                    break;
                case PRE_MK1:
                    for(index_t i = 0; i < k; i++)
                        v_map[i] = v_map_pre1[v_map_post[i]];
                    break;
                case PRE_MK2:
                    for(index_t i = 0; i < k; i++)
                        v_map[i] = v_map_pre2[v_map_post[i]];
                    break;
                case PRE_MK3:
                    for(index_t i = 0; i < k; i++)
                        v_map[i] = v_map_pre1[v_map_pre2[v_map_post[i]]];
                    break;
                default:
                    break;
                }

                // find itenary: temporal-DFS to get travel itenary
                index_t *uu_sol = alloc_idxtab(k);
                index_t *tt_sol = alloc_idxtab(k);
                find_temppath(root_post, uu_sol, tt_sol) ;

                fprintf(stdout, "solution [%ld, %.2lfms]: ", tt_sol[0], pop_time());
                for(index_t i = k-1; i > 0; i--) {
                    index_t u = v_map[uu_sol[i]];
                    index_t v = v_map[uu_sol[i-1]];
                    index_t t = tt_sol[i-1];
                    fprintf(stdout, "[%ld, %ld, %ld]%s", u+1, v+1, t, i==1?"\n":" ");
                }

                FREE(v_map_post);
                FREE(v_map);
                FREE(uu_sol);
                FREE(tt_sol);
                temppathq_free(root_post);
            }
            FREE(get);
            lister_free(t);
        }
        break;
    case CMD_LIST_FIRST_VLOC:
    case CMD_LIST_ALL_VLOC:
        {
            fprintf(stdout, "oracle [temppath]: ");
            fflush(stdout);
            root->vert_loc = 1;
            if(temppathq_execute(root)) {
                fprintf(stdout, " -- true\n");
                index_t n           = root->n; 
                index_t *v_map      = alloc_idxtab(n);

                // build vertex map
                switch(precomp) {
                case PRE_NOP:
                {
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
                    for(index_t i = 0; i < n; i++)
                        v_map[i] = i;
                }
                break;
                case PRE_MK1:
                {
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
                    for(index_t i = 0; i < n; i++)
                        v_map[i] = v_map_pre1[i];
                }
                break;
                case PRE_MK2:
                {
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
                    for(index_t i = 0; i < n; i++)
                        v_map[i] = v_map_pre2[i];
                }
                break;
                case PRE_MK3:
                {
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
                    for(index_t i = 0; i < n; i++)
                        v_map[i] = v_map_pre1[v_map_pre2[i]];
                }
                break;
                default:
                    break;
                }

                // build color map
                index_t *color_map  = alloc_idxtab(n);
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
                for(index_t i = 0; i < n; i++)
                    color_map[i] = color[v_map[i]];

                exhaustive_search(root, kk, color_map, v_map, cmd);

                FREE(v_map);
                FREE(color_map);
            } else {
                fprintf(stdout, " -- false\n");
            }

            FREE(color);
            temppathq_free(root);
        }
        break;
    case CMD_RUN_ORACLE:
        {
            //if(!SOLUTION_EXISTS) break;

            // --- run oracle ---
            fprintf(stdout, "oracle [temppath]: ");
            fflush(stdout);
            if(temppathq_execute(root))
                fprintf(stdout, " -- true\n");
            else
                fprintf(stdout, " -- false\n");
            temppathq_free(root);
        }
        break;
    default:
        assert(0);
        break;
    }

    // free vertex map
    if(precomp == PRE_MK1)
        FREE(v_map_pre1);
    if(precomp == PRE_MK2)
        FREE(v_map_pre2);
    if(precomp == PRE_MK3) {
        FREE(v_map_pre1); 
        FREE(v_map_pre2);
    }
    FREE(kk);
    
    double cmd_time = pop_time();
    double time = pop_time();
    fprintf(stdout, "command done [%.2lf ms %.2lfms %.2lf ms %.2lf ms]\n", 
                    precomp_time, opt_time, cmd_time, time);
    if(input_cmd != CMD_NOP)
        FREE(cmd_args);

    time = pop_time();
    fprintf(stdout, "grand total [%.2lf ms] ", time);
    print_pop_memtrack();
    fprintf(stdout, "\n");
    fprintf(stdout, "host: %s\n", sysdep_hostname());
    fprintf(stdout, 
            "build: %s, %s, %s, %ld x %s\n",
#ifdef BUILD_PARALLEL
            "multithreaded",
#else
            "single thread",
#endif
#ifdef BUILD_PREFETCH
            "prefetch",
#else
            "no prefetch",
#endif
            GENF_TYPE,
            LIMBS_IN_LINE,
            LIMB_TYPE);
    fprintf(stdout, 
            "compiler: gcc %d.%d.%d\n",
            __GNUC__,
            __GNUC_MINOR__,
            __GNUC_PATCHLEVEL__);
    fflush(stdout);
    assert(malloc_balance == 0);
    assert(memtrack_stack_top < 0);
    assert(start_stack_top < 0);

    return 0;
}
