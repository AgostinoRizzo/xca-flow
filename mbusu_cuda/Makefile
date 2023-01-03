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

SRC = mbusu_dhpccpp.cu #../mpui/mpui.cpp ../mpui/mpui_hub.cpp
BIN = mbusu_dhpccpp.o #mpui.o mpui_hub.o
EXE = mbusu_dhpccpp

SRC_TILED = mbusu_dhpccpp_cuda_tiled.cu
BIN_TILED = mbusu_dhpccpp_cuda_tiled.o
EXE_TILED = mbusu_dhpccpp_cuda_tiled

default:$(EXE)


$(EXE): $(BIN)
	$(CPPC) $^ -o $@ -O3 -g -G 
# $(CPPC) $^ -lGL -lGLU -lglut -o $@ -O3 -g -G

$(BIN): $(SRC)
	$(CPPC) $^ -c $(REDUCTION_FLAG) -O3 $(CUDA_GPU_ARCH) $(CUDA_GPU_CODE) #-g -G


$(EXE_TILED): $(BIN_TILED)
	$(CPPC) $^ -o $@ -O3 -g -G

$(BIN_TILED): $(SRC_TILED)
	$(CPPC) $^ -c $(REDUCTION_FLAG) -O3 $(CUDA_GPU_ARCH) $(CUDA_GPU_CODE) #-g -G


###############################################################################
# EXECUTION                                                                   #
###############################################################################

run:
	./$(EXE) $(ROWS) $(COLS) $(SLICES) $(INPUT_KS_PATH) $(SIMULATION_TIME) $(OUTPUT_PREFIX) $(BLOCK_SIZE_0) $(BLOCK_SIZE_1) $(BLOCK_SIZE_2)

run_gpu1:
	CUDA_VISIBLE_DEVICES=1 ./$(EXE) $(ROWS) $(COLS) $(SLICES) $(INPUT_KS_PATH) $(SIMULATION_TIME) $(OUTPUT_PREFIX) $(BLOCK_SIZE_0) $(BLOCK_SIZE_1) $(BLOCK_SIZE_2)

run_tiled:
	./$(EXE_TILED) $(ROWS) $(COLS) $(SLICES) $(INPUT_KS_PATH) $(SIMULATION_TIME) $(OUTPUT_TILED_PREFIX) $(BLOCK_SIZE_0) $(BLOCK_SIZE_1) $(BLOCK_SIZE_2)

run_tiled_gpu1:
	CUDA_VISIBLE_DEVICES=1 ./$(EXE_TILED) $(ROWS) $(COLS) $(SLICES) $(INPUT_KS_PATH) $(SIMULATION_TIME) $(OUTPUT_TILED_PREFIX) $(BLOCK_SIZE_0) $(BLOCK_SIZE_1) $(BLOCK_SIZE_2)

debug_tiled:
	cuda-gdb --args ./$(EXE_TILED) $(ROWS) $(COLS) $(SLICES) $(INPUT_KS_PATH) $(SIMULATION_TIME) $(OUTPUT_TILED_PREFIX) $(BLOCK_SIZE_0) $(BLOCK_SIZE_1) $(BLOCK_SIZE_2)


###############################################################################
# CLEAN                                                                       #
###############################################################################

clean:
	rm -f *.o $(OUTPUT_PREFIX)* $(EXE) $(EXE_TILED)

wipe:
	rm -f *.o $(OUTPUT_PREFIX)*
