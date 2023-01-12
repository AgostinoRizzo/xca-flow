#!/usr/bin/sh

PERF_OUT_FILE=perf.out
rm $PERF_OUT_FILE

#
# serial version
#
cd mbusu_cpu
echo '=====SERIAL_VERSION=====' >> ../$PERF_OUT_FILE
make clean
make
make run >> ../$PERF_OUT_FILE
md5sum output_* >> ../$PERF_OUT_FILE
make clean
echo '' >> ../$PERF_OUT_FILE
cd ..

#
# cuda version
#
cd mbusu_cuda
for CUDA_VERSION in CUDA_VERSION_BASIC CUDA_VERSION_TILED_HALO CUDA_VERSION_TILED_NO_HALO
do
    for BLOCK_SIZE in "8 8 1" "16 32 1" "32 16 1" "32 32 1"
    do
        set -- $BLOCK_SIZE
        echo "=====${CUDA_VERSION}_$1x$2x$3=====" >> ../$PERF_OUT_FILE
        make clean
        make BUILD=$CUDA_VERSION
        make run BLOCK_SIZE_0=$1 BLOCK_SIZE_1=$2 BLOCK_SIZE_2=$3 >> ../$PERF_OUT_FILE
        md5sum output_* >> ../$PERF_OUT_FILE
        make clean
        echo '' >> ../$PERF_OUT_FILE
    done
done
cd ..