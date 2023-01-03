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
//#define CUDA_VERSION_TILED_HALO
//#define CUDA_VERSION_TILED_NO_HALO
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// I/O parameters used to index argv[]
// ----------------------------------------------------------------------------
#define ROWS_ID 1
#define COLS_ID 2
#define LAYERS_ID 3
#define INPUT_KS_ID 4
#define SIMUALITION_TIME_ID 5
#define OUTPUT_PREFIX_ID 6
#define BLOCKSIZE_X_ID 7
#define BLOCKSIZE_Y_ID 8
#define BLOCKSIZE_Z_ID 9
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Simulation parameters
// ----------------------------------------------------------------------------
#define ADJACENT_CELLS 6
#define VON_NEUMANN_NEIGHBORHOOD_3D_CELLS 7
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Read/Write access macros linearizing single/multy layer buffer 2D/3D indices
// ----------------------------------------------------------------------------
#define SET3D(M, rows, columns, i, j, k, value) ( (M)[( ((k)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )
#define GET3D(M, rows, columns, i, j, k) ( M[( ((k)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )
#define BUF_SET3D(M, rows, columns, slices, n, i, j, k, value) ( (M)[( ((n)*(rows)*(columns)*(slices)) + ((k)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )
#define BUF_GET3D(M, rows, columns, slices, n, i, j, k) ( M[( ((n)*(rows)*(columns)*(slices)) + ((k)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )
// ----------------------------------------------------------------------------

struct DomainBoundaries
{
  int i_start;
  int i_end;
  int j_start;
  int j_end;
  int k_start;
  int k_end;
};

void initDomainBoundaries(DomainBoundaries& B, int i_start, int i_end, int j_start, int j_end, int k_start, int k_end)
{
  B.i_start = i_start;
  B.i_end   = i_end; 
  B.j_start = j_start;
  B.j_end   = j_end;
  B.k_start = k_start;
  B.k_end   = k_end;
}

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

__host__ __device__
void syncSubstatesPtrs(Substates &Q, int offset_size )
{
  Q.ks               = Q.__substates__;

  Q.teta             = Q.__substates__ + offset_size;
  Q.teta_next        = Q.__substates__ + offset_size*2;
  Q.moist_cont       = Q.__substates__ + offset_size*3;
  Q.moist_cont_next  = Q.__substates__ + offset_size*4;
  Q.psi              = Q.__substates__ + offset_size*5;
  Q.psi_next         = Q.__substates__ + offset_size*6;
  Q.k                = Q.__substates__ + offset_size*7;
  Q.k_next           = Q.__substates__ + offset_size*8;
  Q.h                = Q.__substates__ + offset_size*9;
  Q.h_next           = Q.__substates__ + offset_size*10;
  Q.dqdh             = Q.__substates__ + offset_size*11;
  Q.dqdh_next        = Q.__substates__ + offset_size*12;
  Q.convergence      = Q.__substates__ + offset_size*13;
  Q.convergence_next = Q.__substates__ + offset_size*14;

  Q.F                = Q.__substates__ + offset_size*15;
}

int allocSubstates(Substates &Q, int r, int c, int s)
{
  const int size  = r*c*s;
  Q.__substates__ = util::allocBuffer4D(15 + ADJACENT_CELLS, r, c, s);
  syncSubstatesPtrs( Q, size );
  return (15 + ADJACENT_CELLS) * size;
}

void deleteSubstates(Substates& Q) { free(Q.__substates__); }

struct Parameters
{
  int YOUT;
  int YIN;
  int XE;
  int XW;
  int ZFONDO;
  int ZSUP;

  double h_init;
  double tetas;
  double tetar;
  double alfa;
  double n;
  double rain;
  double psi_zero;
  double ss;
  double lato;
  double delta_t;
  double delta_t_cum;
  double delta_t_cum_prec;
  double simulation_time;
};

void initParameters(Parameters& P, double simulation_time, int r, int c, int s)
{
  P.YOUT = c-1;
  P.YIN = 0;
  P.XE = r-1;
  P.XW = 0;
  P.ZSUP = s-1;
  P.ZFONDO = 0;

  P.h_init = 734;
  P.tetas = 0.348;
  P.tetar = 0.095467;
  P.alfa = 0.034733333;
  P.n = 1.729;
  P.rain = 0.000023148148;
  P.psi_zero = -0.1;
  P.ss = 0.0001;
  P.lato = 30.0;	
  P.simulation_time = simulation_time;
  P.delta_t = 10.0;
  P.delta_t_cum = 0.0;
  P.delta_t_cum_prec = 0.0;
}

// ----------------------------------------------------------------------------
// I/O functions
// ----------------------------------------------------------------------------
void readKs(double* ks, int r, int c, int s, std::string path)
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
  for (k = 0; k < s; k++)
    for (i = 0; i < r; i++)
      for (j = 0; j < c; j++)
      {
        fscanf(f, "%s", str);
        SET3D(ks, r, c, i, j, k, atof(str));
      }
  fclose(f);
}

void saveFile(double* sub, int r, int c, int s, std::string nameFile)
{
  int i, j, k;
  double moist_print;

  FILE *stream = fopen(nameFile.c_str(), "w");
  for (k = 0; k < s; k++)
  {
    for (i = 0; i < r; i++)
    {
      for (j = 0; j < c; j++)
      {
        moist_print = GET3D(sub, r, c, i, j, k);
        fprintf(stream, "%.8f ", moist_print);
      }
      fprintf(stream, "\n");
    }
    fprintf(stream, "\n");
  }
  fclose(stream);
}

#endif