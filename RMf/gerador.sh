#!/bin/bash
./genrmf -a 192 -b 24 -c1 1 -c2 1000 -out genrmf_long_192_24_1_1000
./genrmf -a 224 -b 28 -c1 1 -c2 1000 -out genrmf_long_224_28_1_1000
./genrmf -a 256 -b 32 -c1 1 -c2 1000 -out genrmf_long_256_32_1_1000
./genrmf -a 36 -b 36 -c1 1 -c2 1000 -out genrmf_wide_36_36_1_1000
./genrmf -a 48 -b 48 -c1 1 -c2 1000 -out genrmf_wide_48_48_1_1000
./genrmf -a 64 -b 64 -c1 1 -c2 1000 -out genrmf_wide_64_64_1_1000
./genrmf -a 128 -b 128 -c1 1 -c2 1000 -out genrmf_wide_128_128_1_1000
