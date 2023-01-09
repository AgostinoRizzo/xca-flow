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

SRC_MPUI = mbusu_dhpccpp.cu ../mpui/mpui.cpp ../mpui/mpui_net.cpp
BIN_MPUI = mbusu_dhpccpp_mpui.o mpui.o mpui_net.o
EXE_MPUI = mbusu_dhpccpp_mpui

default:$(EXE)
#default:$(EXE_MPUI)


$(EXE): $(BIN)
	$(CPPC) $^ -o $@ -O3 -g -G

$(BIN): $(SRC)
	$(CPPC) $^ -D $(BUILD) -c $(REDUCTION_FLAG) -O3 $(CUDA_GPU_ARCH) $(CUDA_GPU_CODE) #-g -G


$(EXE_MPUI): $(BIN_MPUI)
	$(CPPC) $^ -o $@ -O3 -g -G

$(BIN_MPUI): $(SRC_MPUI)
	$(CPPC) $^ -D $(BUILD) -D __MPUI__ -c $(REDUCTION_FLAG) -O3 $(CUDA_GPU_ARCH) $(CUDA_GPU_CODE) #-g -G


###############################################################################
# EXECUTION                                                                   #
###############################################################################

run:
	./$(EXE) $(ROWS) $(COLS) $(SLICES) $(INPUT_KS_PATH) $(SIMULATION_TIME) $(OUTPUT_PREFIX) $(BLOCK_SIZE_0) $(BLOCK_SIZE_1) $(BLOCK_SIZE_2)

run_mpui:
	./$(EXE_TILED) $(ROWS) $(COLS) $(SLICES) $(INPUT_KS_PATH) $(SIMULATION_TIME) $(OUTPUT_TILED_PREFIX) $(BLOCK_SIZE_0) $(BLOCK_SIZE_1) $(BLOCK_SIZE_2)


###############################################################################
# CLEAN                                                                       #
###############################################################################

clean:
	rm -f *.o $(OUTPUT_PREFIX)* $(EXE) $(EXE_MPUI)

wipe:
	rm -f *.o $(OUTPUT_PREFIX)*
