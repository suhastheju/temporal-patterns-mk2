# Temporal patterns (mark-2)

## Overview
This software repository contains an experimental software implementation of algorithms for solving a set of pattern-detection problems in temporal graphs. The software is written in C programming language.

This version of the source code is realeased for BIG DATA special issue -- best of SDM2020 submission titled "Finding path motifs in large temporal graphs using algebraic fingerprints".

## License
The source code is subject to MIT license.

## Compilation
The source code is configured for a gcc build. Other builds are possible but it might require manual configuration of the 'Makefile'.

Use GNU make to build the software. Check 'Makefile' for more details.

`make clean all`

## Using the software
usage:  `./LISTER_PAR_GENF2 -pre <value> -optimal -<command-type> -seed <value> -in <input-file> -<file-type>`
        `./LISTER_PAR_GENF2 -h/help`

Arguments:
        -pre <value>      : <0>   -  no preprocessing (default)
                            <1>   -  preprocess step-1
                            <2>   -  preprocess step-2
                            <3>   -  preprocess step-1 and step-2
        -optimal          : obtain optimal solution (optional)
        -<command-type>   : <oracle>     - decide existence of a solution
                            <first>      - extract one solution
                            <first-vloc> - extract one solution (vertex localisation)
        -seed <value>     : integer value in range 1 to 2^32 -1
                            default value `123456789`
        -in <input-file>  : read from <input file>
                            read from <stdin> by default
        -<file-type>      : <ascii>  - ascii input file (default)
                            <bin>    - binary input file
        -h or -help       : help

## Input file format
We use dimacs format for the input graph. An example of input graph is available in `input-graph.g`. See `graph-gen` from `temporal-patterns` repository for using the graph generator. 

## Example

`$ /LISTER_PAR_GENF2 -ascii -pre 0 -first -optimal -in input-graph.g`  

        invoked as: ./LISTER_PAR_GENF2 -ascii -pre 0 -first -optimal -in input-graph.g
        no random seed given, defaulting to 123456789
        random seed = 123456789
        input: n = 100, m = 1040, k = 5, t = 100 [0.71 ms] {peak: 0.00GiB} {curr: 0.00GiB}
        build query: [zero: 0.38 ms] [pos: 0.10 ms] [adj: 0.10 ms] [adjsort: 0.08 ms] [shade: 0.04 ms] done. [0.75 ms] {peak: 0.00GiB} {curr: 0.00GiB}
        no preprocessing, default execution
        optimal : 100
                  100 [1:100]       0x28FA11B2AF010654 84.30 ms [3.04GiB/s, 0.02GHz] 1 {peak: 0.00GiB} {curr: 0.00GiB} -- true
                   52 [4:100]       0x571743D4C273CBDA 41.90 ms [1.75GiB/s, 0.02GHz] 1 {peak: 0.00GiB} {curr: 0.00GiB} -- true
                   28 [4:52]        0xFE53DB1DDEFC62CE 26.80 ms [0.87GiB/s, 0.02GHz] 1 {peak: 0.00GiB} {curr: 0.00GiB} -- true
                   16 [4:28]        0x270A23F153165088 12.52 ms [0.69GiB/s, 0.02GHz] 1 {peak: 0.00GiB} {curr: 0.00GiB} -- true
                   10 [4:16]        0x0000000000000000 8.46 ms [0.46GiB/s, 0.02GHz] 0 {peak: 0.00GiB} {curr: 0.00GiB} -- false
                   13 [11:16]       0x493DEFD79A72D4CD 14.02 ms [0.43GiB/s, 0.02GHz] 1 {peak: 0.00GiB} {curr: 0.00GiB} -- true
                   12 [11:13]       0xB4F17067E64D2CEC 10.17 ms [0.52GiB/s, 0.02GHz] 1 {peak: 0.00GiB} {curr: 0.00GiB} -- true
                   11 [11:12]       0x66CB3E304689A621 10.03 ms [0.45GiB/s, 0.02GHz] 1 {peak: 0.00GiB} {curr: 0.00GiB} -- true
        command: list first
        extract: 100 5 0
                  100 [[0:99]]                                                                                                  0xCA26578CEE26F488 10.98 ms [0.42GiB/s, 0.02GHz] 1 {peak: 0.00GiB} {curr: 0.00GiB} -- true
                   50 [[50:99]]                                                                                                 0x0000000000000000 10.98 ms [0.20GiB/s, 0.01GHz] 0 {peak: 0.00GiB} {curr: 0.00GiB} -- false
                   50 [[0:49]]                                                                                                  0x0000000000000000 8.42 ms [0.26GiB/s, 0.01GHz] 0 {peak: 0.00GiB} {curr: 0.00GiB} -- false
                   75 [[50:99],[25:49]]                                                                                         0x0000000000000000 7.65 ms [0.44GiB/s, 0.02GHz] 0 {peak: 0.00GiB} {curr: 0.00GiB} -- false
                   75 [[50:99],[0:24]]                                                                                          0x0000000000000000 9.07 ms [0.38GiB/s, 0.02GHz] 0 {peak: 0.00GiB} {curr: 0.00GiB} -- false
                   75 [[0:24],[25:49],[75:99]]                                                                                  0x0000000000000000 9.18 ms [0.37GiB/s, 0.02GHz] 0 {peak: 0.00GiB} {curr: 0.00GiB} -- false
                   75 [[0:24],[25:49],[50:74]]                                                                                  0x0000000000000000 9.22 ms [0.37GiB/s, 0.02GHz] 0 {peak: 0.00GiB} {curr: 0.00GiB} -- false
                   87 [[25:49],[50:74],[75:99],[13:24]]                                                                         0x3AC2C64D5891226A 9.18 ms [0.43GiB/s, 0.02GHz] 1 {peak: 0.00GiB} {curr: 0.00GiB} -- true
                   74 [[50:74],[75:99],[13:24],[38:49]]                                                                         0x97B8B67EC63CF8EB 9.54 ms [0.35GiB/s, 0.01GHz] 1 {peak: 0.00GiB} {curr: 0.00GiB} -- true
                   61 [[75:99],[13:24],[38:49],[63:74]]                                                                         0x0000000000000000 9.46 ms [0.29GiB/s, 0.01GHz] 0 {peak: 0.00GiB} {curr: 0.00GiB} -- false
                   62 [[75:99],[13:24],[38:49],[50:62]]                                                                         0x0000000000000000 7.69 ms [0.36GiB/s, 0.02GHz] 0 {peak: 0.00GiB} {curr: 0.00GiB} -- false
                   61 [[13:24],[38:49],[50:62],[63:74],[88:99]]                                                                 0x0000000000000000 10.47 ms [0.26GiB/s, 0.01GHz] 0 {peak: 0.00GiB} {curr: 0.00GiB} -- false
                   62 [[13:24],[38:49],[50:62],[63:74],[75:87]]                                                                 0x9833E5574DB79CEF 12.42 ms [0.23GiB/s, 0.01GHz] 1 {peak: 0.00GiB} {curr: 0.00GiB} -- true
                   56 [[38:49],[50:62],[63:74],[75:87],[19:24]]                                                                 0x0000000000000000 7.51 ms [0.34GiB/s, 0.01GHz] 0 {peak: 0.00GiB} {curr: 0.00GiB} -- false
                   56 [[38:49],[50:62],[63:74],[75:87],[13:18]]                                                                 0x7E0549A3C1D992F7 9.98 ms [0.25GiB/s, 0.01GHz] 1 {peak: 0.00GiB} {curr: 0.00GiB} -- true
                   50 [[50:62],[63:74],[75:87],[13:18],[44:49]]                                                                 0x0000000000000000 9.80 ms [0.23GiB/s, 0.01GHz] 0 {peak: 0.00GiB} {curr: 0.00GiB} -- false
                   50 [[50:62],[63:74],[75:87],[13:18],[38:43]]                                                                 0x41F6BCD1FA6A42D1 9.19 ms [0.25GiB/s, 0.01GHz] 1 {peak: 0.00GiB} {curr: 0.00GiB} -- true
                   43 [[63:74],[75:87],[13:18],[38:43],[57:62]]                                                                 0xB26FD63E83D81707 9.87 ms [0.19GiB/s, 0.01GHz] 1 {peak: 0.00GiB} {curr: 0.00GiB} -- true
                   37 [[75:87],[13:18],[38:43],[57:62],[69:74]]                                                                 0x0000000000000000 8.52 ms [0.19GiB/s, 0.01GHz] 0 {peak: 0.00GiB} {curr: 0.00GiB} -- false
                   37 [[75:87],[13:18],[38:43],[57:62],[63:68]]                                                                 0x6FE1B276F25A7AF1 9.34 ms [0.18GiB/s, 0.01GHz] 1 {peak: 0.00GiB} {curr: 0.00GiB} -- true
                   30 [[13:18],[38:43],[57:62],[63:68],[82:87]]                                                                 0x5BCED4B993DE4846 8.79 ms [0.15GiB/s, 0.01GHz] 1 {peak: 0.00GiB} {curr: 0.00GiB} -- true
                   27 [[38:43],[57:62],[63:68],[82:87],[16:18]]                                                                 0x54B6FC6B14F3FACA 12.77 ms [0.09GiB/s, 0.00GHz] 1 {peak: 0.00GiB} {curr: 0.00GiB} -- true
                   24 [[57:62],[63:68],[82:87],[16:18],[41:43]]                                                                 0x0000000000000000 10.89 ms [0.10GiB/s, 0.00GHz] 0 {peak: 0.00GiB} {curr: 0.00GiB} -- false
                   24 [[57:62],[63:68],[82:87],[16:18],[38:40]]                                                                 0xA3061A7C8AAE7D18 9.81 ms [0.11GiB/s, 0.00GHz] 1 {peak: 0.00GiB} {curr: 0.00GiB} -- true
                   21 [[63:68],[82:87],[16:18],[38:40],[60:62]]                                                                 0x0000000000000000 8.82 ms [0.11GiB/s, 0.00GHz] 0 {peak: 0.00GiB} {curr: 0.00GiB} -- false
                   21 [[63:68],[82:87],[16:18],[38:40],[57:59]]                                                                 0x34276ED4AED2E1A4 9.15 ms [0.10GiB/s, 0.00GHz] 1 {peak: 0.00GiB} {curr: 0.00GiB} -- true
                   18 [[82:87],[16:18],[38:40],[57:59],[66:68]]                                                                 0xDE160638CFC07E26 9.01 ms [0.09GiB/s, 0.00GHz] 1 {peak: 0.00GiB} {curr: 0.00GiB} -- true
                   15 [[16:18],[38:40],[57:59],[66:68],[85:87]]                                                                 0x0000000000000000 9.38 ms [0.07GiB/s, 0.00GHz] 0 {peak: 0.00GiB} {curr: 0.00GiB} -- false
                   15 [[16:18],[38:40],[57:59],[66:68],[82:84]]                                                                 0x4C24C74C2532AB8F 9.73 ms [0.07GiB/s, 0.00GHz] 1 {peak: 0.00GiB} {curr: 0.00GiB} -- true
                   13 [[38:40],[57:59],[66:68],[82:84],[18]]                                                                    0x0000000000000000 8.82 ms [0.07GiB/s, 0.00GHz] 0 {peak: 0.00GiB} {curr: 0.00GiB} -- false
                   14 [[38:40],[57:59],[66:68],[82:84],[16:17]]                                                                 0xD249750ED4FD73FB 8.51 ms [0.07GiB/s, 0.00GHz] 1 {peak: 0.00GiB} {curr: 0.00GiB} -- true
                   12 [[57:59],[66:68],[82:84],[16:17],[40]]                                                                    0x0000000000000000 9.33 ms [0.06GiB/s, 0.00GHz] 0 {peak: 0.00GiB} {curr: 0.00GiB} -- false
                   13 [[57:59],[66:68],[82:84],[16:17],[38:39]]                                                                 0x77D05B815AE50216 8.77 ms [0.07GiB/s, 0.00GHz] 1 {peak: 0.00GiB} {curr: 0.00GiB} -- true
                   11 [[66:68],[82:84],[16:17],[38:39],[59]]                                                                    0x0000000000000000 10.34 ms [0.05GiB/s, 0.00GHz] 0 {peak: 0.00GiB} {curr: 0.00GiB} -- false
                   12 [[66:68],[82:84],[16:17],[38:39],[57:58]]                                                                 0x10ED7443A2E1617F 8.89 ms [0.06GiB/s, 0.00GHz] 1 {peak: 0.00GiB} {curr: 0.00GiB} -- true
                   10 [[82:84],[16:17],[38:39],[57:58],[68]]                                                                    0x0069CBEB00405C0C 8.94 ms [0.05GiB/s, 0.00GHz] 1 {peak: 0.00GiB} {curr: 0.00GiB} -- true
                    8 [[16:17],[38:39],[57:58],[68],[84]]                                                                       0x0000000000000000 8.54 ms [0.04GiB/s, 0.00GHz] 0 {peak: 0.00GiB} {curr: 0.00GiB} -- false
                    9 [[16:17],[38:39],[57:58],[68],[82:83]]                                                                    0x3309622379CA7D45 8.65 ms [0.05GiB/s, 0.00GHz] 1 {peak: 0.00GiB} {curr: 0.00GiB} -- true
                    8 [[38:39],[57:58],[68],[82:83],[17]]                                                                       0x0000000000000000 9.32 ms [0.04GiB/s, 0.00GHz] 0 {peak: 0.00GiB} {curr: 0.00GiB} -- false
                    8 [[38:39],[57:58],[68],[82:83],[16]]                                                                       0x3139087A21D34865 8.37 ms [0.04GiB/s, 0.00GHz] 1 {peak: 0.00GiB} {curr: 0.00GiB} -- true
                    7 [[57:58],[68],[82:83],[16],[39]]                                                                          0x0000000000000000 8.48 ms [0.04GiB/s, 0.00GHz] 0 {peak: 0.00GiB} {curr: 0.00GiB} -- false
                    7 [[57:58],[68],[82:83],[16],[38]]                                                                          0x686BD4BE0CDD6AA5 9.70 ms [0.03GiB/s, 0.00GHz] 1 {peak: 0.00GiB} {curr: 0.00GiB} -- true
                    6 [[[68],[82:83],[16],[38],[58]]                                                                            0x0000000000000000 11.12 ms [0.02GiB/s, 0.00GHz] 0 {peak: 0.00GiB} {curr: 0.00GiB} -- false
                    6 [[[68],[82:83],[16],[38],[57]]                                                                            0x2432EFBA55222EE8 8.62 ms [0.03GiB/s, 0.00GHz] 1 {peak: 0.00GiB} {curr: 0.00GiB} -- true
                    5 [[[68],[16],[38],[57],[83]]                                                                               0x58A9F0DF41EBE3CE 12.56 ms [0.02GiB/s, 0.00GHz] 1 {peak: 0.00GiB} {curr: 0.00GiB} -- true
        extract done [462.59 ms]
        found 0: 17 39 58 69 84
        subgraph: [map: 0.08 ms] [pos: 0.51 ms] [adj: 0.28 ms] [shade: 0.03 ms] done. [0.95 ms] {peak: 0.00GiB} {curr: 0.00GiB}
        solution [11, 1.33ms]: [39, 58, 2] [58, 17, 6] [17, 69, 10] [69, 84, 11]
        command done [0.00 ms 208.50ms 463.97 ms 672.48 ms]
        grand total [674.03 ms] {peak: 0.00GiB}
        host: maagha
        build: multithreaded, prefetch, k_temp_path_genf, 2 x 256-bit AVX2 [4 x GF(2^{64}) with four 64-bit words]
        compiler: gcc 9.1.0

The line `command done [0.00 ms 208.50ms 463.97 ms 672.48 ms]` specifies the runtime of execution. Here the first time `0.00 ms` is preprocessing time, `208.50 ms` time to find optimal timestamp, `463.97 ms` time to find solution and total time is `672.48 ms`.  

The reported runtimes are in milliseconds.

The line `solution [11, 1.33ms]: [39, 58, 2] [58, 17, 6] [17, 69, 10] [69, 84, 11]` reports a solution and each term is of the form `[u, v, i]` where `u, v` are vertices and `i` is the timestamp of transition.
