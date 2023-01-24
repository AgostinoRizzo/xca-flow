#ifndef MBUSU_DHPCCPP_HPP
#define MBUSU_DHPCCPP_HPP

#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include "../mbusu_cpu/util.hpp"

// ----------------------------------------------------------------------------
// MBUSU-CUDA implementation
// ----------------------------------------------------------------------------
//#define CUDA_VERSION_BASIC
//#define CUDA_VERSION_TILED_HALO
//#define CUDA_VERSION_TILED_NO_HALO
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// MBUSU-DOMAIN size and boundaries
// ----------------------------------------------------------------------------
#define ROWS   100
#define COLS   100
#define SLICES  50
#define SIZE       (ROWS*COLS*SLICES)
#define SLICE_SIZE (ROWS*COLS)
#define MB_BOUNDS_i_start 1
#define MB_BOUNDS_i_end   (ROWS-1)
#define MB_BOUNDS_j_start 1
#define MB_BOUNDS_j_end   (COLS-1)
#define MB_BOUNDS_k_start 0
#define MB_BOUNDS_k_end   SLICES
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// I/O parameters used to index argv[]
// ----------------------------------------------------------------------------
#define INPUT_KS_ID 1
#define SIMUALITION_TIME_ID 2
#define OUTPUT_PREFIX_ID 3
#define BLOCKSIZE_X_ID 4
#define BLOCKSIZE_Y_ID 5
#define BLOCKSIZE_Z_ID 6
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Simulation parameters
// ----------------------------------------------------------------------------
#define ADJACENT_CELLS 6
#define VON_NEUMANN_NEIGHBORHOOD_3D_CELLS 7
#define MIN_VAR 55.0 //237.528
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Read/Write access macros linearizing single/multy layer buffer 2D/3D indices
// ----------------------------------------------------------------------------
#define SET3D(M, rows, columns, i, j, k, value) ( (M)[( ((k)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )
#define GET3D(M, rows, columns, i, j, k) ( M[( ((k)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )
#define BUF_SET3D(M, rows, columns, slices, n, i, j, k, value) ( (M)[( ((n)*(rows)*(columns)*(slices)) + ((k)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )
#define BUF_GET3D(M, rows, columns, slices, n, i, j, k) ( M[( ((n)*(rows)*(columns)*(slices)) + ((k)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )
// ----------------------------------------------------------------------------

//
// Substates struct is declared as a linear 4D buffer (__substates__)
//  - each of the 16 substates is concatenated linearly
//
struct Substates
{
  double *__substates__;

  double *ks;

  double *teta;
  double *teta_next;
  double *moist_cont;
  double *moist_cont_next;
  double *psi;
  double *psi_next;
  double *k;
  double *k_next;
  double *h;
  double *h_next;
  double *dqdh;
  double *dqdh_next;
  double *convergence;
  double *convergence_next;

  double *F;
};

// ----------------------------------------------------------------------------
// Substates size and offsets
// ----------------------------------------------------------------------------
#define __SUBSTATE_SIZE__             SIZE
#define __SUBSTATE_SIZE_BYTES__       SIZE * sizeof(double)
#define __Q_h__OFFSET__               (SIZE*9)
#define __Q_convergence_OFFSET__      (SIZE*13)
#define __Q_convergence_next_OFFSET__ (SIZE*14)
// ----------------------------------------------------------------------------

__host__ __device__
void syncSubstatesPtrs(Substates &Q, bool substates_swap=false)
{
  Q.ks               = Q.__substates__;

  if ( substates_swap )
  {
    Q.teta             = Q.__substates__ + SIZE*2;
    Q.teta_next        = Q.__substates__ + SIZE;
    Q.moist_cont       = Q.__substates__ + SIZE*4;
    Q.moist_cont_next  = Q.__substates__ + SIZE*3;
    Q.psi              = Q.__substates__ + SIZE*6;
    Q.psi_next         = Q.__substates__ + SIZE*5;
    Q.k                = Q.__substates__ + SIZE*8;
    Q.k_next           = Q.__substates__ + SIZE*7;
    Q.h                = Q.__substates__ + SIZE*10;
    Q.h_next           = Q.__substates__ + SIZE*9;
    Q.dqdh             = Q.__substates__ + SIZE*12;
    Q.dqdh_next        = Q.__substates__ + SIZE*11;
    Q.convergence      = Q.__substates__ + SIZE*14;
    Q.convergence_next = Q.__substates__ + SIZE*13;
  }
  else
  {
    Q.teta             = Q.__substates__ + SIZE;
    Q.teta_next        = Q.__substates__ + SIZE*2;
    Q.moist_cont       = Q.__substates__ + SIZE*3;
    Q.moist_cont_next  = Q.__substates__ + SIZE*4;
    Q.psi              = Q.__substates__ + SIZE*5;
    Q.psi_next         = Q.__substates__ + SIZE*6;
    Q.k                = Q.__substates__ + SIZE*7;
    Q.k_next           = Q.__substates__ + SIZE*8;
    Q.h                = Q.__substates__ + SIZE*9;
    Q.h_next           = Q.__substates__ + SIZE*10;
    Q.dqdh             = Q.__substates__ + SIZE*11;
    Q.dqdh_next        = Q.__substates__ + SIZE*12;
    Q.convergence      = Q.__substates__ + SIZE*13;
    Q.convergence_next = Q.__substates__ + SIZE*14;
  }

  Q.F                = Q.__substates__ + SIZE*15;
}

int allocSubstates(Substates &Q)
{
  Q.__substates__ = util::allocBuffer4D(15 + ADJACENT_CELLS, ROWS, COLS, SLICES);
  syncSubstatesPtrs( Q );
  return (15 + ADJACENT_CELLS) * SIZE;
}

void deleteSubstates(Substates& Q) { free(Q.__substates__); }

#define P_YOUT (COLS-1)
#define P_YIN 0
#define P_XE (ROWS-1)
#define P_XW 0
#define P_ZSUP (SLICES-1)
#define P_ZFONDO 0

#define P_h_init 734
#define P_tetas 0.348
#define P_tetar 0.095467
#define P_alfa 0.034733333
#define P_n 1.729
#define P_rain 0.000023148148
#define P_psi_zero (-0.1)
#define P_ss 0.0001
#define P_lato 30.0

struct Parameters
{
  double delta_t;
  double delta_t_cum;
  double delta_t_cum_prec;
  double simulation_time;
};

void initParameters(Parameters& P, double simulation_time)
{
  P.simulation_time = simulation_time;
  P.delta_t = 10.0;
  P.delta_t_cum = 0.0;
  P.delta_t_cum_prec = 0.0;
}

// ----------------------------------------------------------------------------
// I/O functions
// ----------------------------------------------------------------------------
void readKs(double* ks, std::string path)
{
  FILE *f = fopen(path.c_str(), "r");
  if (f == NULL)
  {
    printf("can not open file %s", path.c_str());
    exit(0);
  }
  //printf("read succefully %s \n", path.c_str());
  char str[256];
  int i, j, k;
  for (k = 0; k < SLICES; k++)
    for (i = 0; i < ROWS; i++)
      for (j = 0; j < COLS; j++)
      {
        fscanf(f, "%s", str);
        SET3D(ks, ROWS, COLS, i, j, k, atof(str));
      }
  fclose(f);
}

void saveFile(double* sub, std::string nameFile)
{
  int i, j, k;
  double moist_print;

  FILE *stream = fopen(nameFile.c_str(), "w");
  for (k = 0; k < SLICES; k++)
  {
    for (i = 0; i < ROWS; i++)
    {
      for (j = 0; j < COLS; j++)
      {
        moist_print = GET3D(sub, ROWS, COLS, i, j, k);
        fprintf(stream, "%.8f ", moist_print);
      }
      fprintf(stream, "\n");
    }
    fprintf(stream, "\n");
  }
  fclose(stream);
}

#endif