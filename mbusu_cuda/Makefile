###############################################################################
# COMMON 																																			#
###############################################################################

include ../common.mk

###############################################################################
# COMPILERS                                                                   #
###############################################################################

# definisce la macro CPPC
ifndef CPPC
	CPPC=nvcc
endif

###############################################################################
# BUILD                                                                       #
###############################################################################

BUILD = CUDA_VERSION_BASIC
#BUILD = CUDA_VERSION_TILED_HALO
#BUILD = CUDA_VERSION_TILED_NO_HALO

SRC = mbusu_dhpccpp.cu
BIN = mbusu_dhpccpp.o
EXE = mbusu_dhpccpp

#SRC_MPUI = mbusu_dhpccpp.cu ../mpui/mpui.cpp ../mpui/mpui_sck.cpp
#BIN_MPUI = mbusu_dhpccpp.o mpui.o mpui_sck.o
#EXE_MPUI = mbusu_dhpccpp

SRC_MULTI = mbusu_dhpccpp_multi.cu
BIN_MULTI = mbusu_dhpccpp_multi.o
EXE_MULTI = mbusu_dhpccpp_multi

#default:$(EXE)
#default:$(EXE_MPUI)
default:$(EXE_MULTI)


$(EXE): $(BIN)
	$(CPPC) $^ -o $@ -O3 -g -G

$(BIN): $(SRC)
	$(CPPC) $^ -D ROWS=$(ROWS) -D COLS=$(COLS) -D SLICES=$(SLICES) -D $(BUILD) -c $(REDUCTION_FLAG) -O3 $(CUDA_GPU_ARCH) $(CUDA_GPU_CODE) #-g -G


$(EXE_MPUI): $(BIN_MPUI)
	$(CPPC) $^ -o $@ -O3 -g -G

$(BIN_MPUI): $(SRC_MPUI)
	$(CPPC) $^ -D ROWS=$(ROWS) -D COLS=$(COLS) -D SLICES=$(SLICES) -D $(BUILD) -D __MPUI__ -c $(REDUCTION_FLAG) -O3 $(CUDA_GPU_ARCH) $(CUDA_GPU_CODE) #-g -G


$(EXE_MULTI): $(BIN_MULTI)
	$(CPPC) $^ -L/usr/lib/openmpi/ -lmpi -o $@ -O3 -g -G

$(BIN_MULTI): $(SRC_MULTI)
	$(CPPC) $^ -D ROWS=$(ROWS) -D COLS=$(COLS) -D SLICES=$(SLICES) -D $(BUILD) -c $(REDUCTION_FLAG) -O3 $(CUDA_GPU_ARCH) $(CUDA_GPU_CODE) #-g -G


###############################################################################
# EXECUTION                                                                   #
###############################################################################

run:
	./$(EXE) $(INPUT_KS_PATH) $(SIMULATION_TIME) $(OUTPUT_PREFIX) $(BLOCK_SIZE_0) $(BLOCK_SIZE_1) $(BLOCK_SIZE_2)

run_mpui:
	./$(EXE_MPUI) $(INPUT_KS_PATH) $(SIMULATION_TIME) $(OUTPUT_PREFIX) $(BLOCK_SIZE_0) $(BLOCK_SIZE_1) $(BLOCK_SIZE_2)

run_multi:
	mpirun -np 2 ./$(EXE_MULTI) $(INPUT_KS_PATH) $(SIMULATION_TIME) $(OUTPUT_PREFIX) $(BLOCK_SIZE_0) $(BLOCK_SIZE_1) $(BLOCK_SIZE_2)


###############################################################################
# CLEAN                                                                       #
###############################################################################

clean:
	rm -f *.o $(OUTPUT_PREFIX)* $(EXE) $(EXE_MPUI) $(EXE_MULTI)

wipe:
	rm -f *.o $(OUTPUT_PREFIX)*
