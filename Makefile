# General Makefile

.PHONY: all clean wipe

all:
	$(MAKE) -C mbusu_cpu
	#$(MAKE) -C mbusu_cuda

clean:
	$(MAKE) -C mbusu_cpu clean
	#$(MAKE) -C mbusu_cuda clean

wipe:
	$(MAKE) -C mbusu_cpu wipe
	#$(MAKE) -C mbusu_cuda wipe
