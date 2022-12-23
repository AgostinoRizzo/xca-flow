# Simulation parameters
ROWS=100
COLS=100
SLICES=50
INPUT_KS_PATH=../data/generatedKs/matrice_3d_linearizzata_2_n_31.csv
MIN_VAR=237.528
SIMULATION_TIME=864000
#SIMULATION_TIME=4320
OUTPUT_PREFIX=./output_

# GPU execution parameters
BLOCK_SIZE_0=8
BLOCK_SIZE_1=8
BLOCK_SIZE_2=1

# CUDA architectures
ifeq ($(CUDA_ARCH),AMPERE)
	CUDA_COMP_ARCH = sm_80
else
ifeq ($(CUDA_ARCH),TURING)
	CUDA_COMP_ARCH = sm_75
else
ifeq ($(CUDA_ARCH),VOLTA)
	CUDA_COMP_ARCH = sm_70
else
ifeq ($(CUDA_ARCH),PASCAL)
	CUDA_COMP_ARCH = sm_61
else
ifeq ($(CUDA_ARCH),MAXWELL)
  CUDA_COMP_ARCH = sm_52
else
ifeq ($(CUDA_ARCH),KEPLER)
  CUDA_COMP_ARCH = sm_35
endif
endif
endif
endif
endif
endif
