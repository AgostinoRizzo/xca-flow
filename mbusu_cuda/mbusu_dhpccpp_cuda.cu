#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include "../mbusu_cpu/util.hpp"
//#include "../mpui/mpui.h"
// ----------------------------------------------------------------------------
// CUDA implementation
// ----------------------------------------------------------------------------
#define CUDA_VERSION_TILED_HALO
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
// Simulation parameters
// ----------------------------------------------------------------------------
#define ADJACENT_CELLS 6
#define VON_NEUMANN_NEIGHBORHOOD_3D_CELLS 7
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

__device__ int Xi[] = {0, -1,  0,  0,  1,  0,  0};
__device__ int Xj[] = {0,  0, -1,  1,  0,  0,  0};
__device__ int Xk[] = {0,  0,  0,  0,  0, -1,  1};

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

__device__
void updateSubstates( Substates &d__Q, int r, int c, int i, int j, int k )
{
  SET3D( d__Q.dqdh,        r, c, i, j, k, GET3D(d__Q.dqdh_next,        r, c, i, j, k) );
  SET3D( d__Q.psi,         r, c, i, j, k, GET3D(d__Q.psi_next,         r, c, i, j, k) );
  SET3D( d__Q.k,           r, c, i, j, k, GET3D(d__Q.k_next,           r, c, i, j, k) );
  SET3D( d__Q.h,           r, c, i, j, k, GET3D(d__Q.h_next,           r, c, i, j, k) );
  SET3D( d__Q.teta,        r, c, i, j, k, GET3D(d__Q.teta_next,        r, c, i, j, k) );
  SET3D( d__Q.moist_cont,  r, c, i, j, k, GET3D(d__Q.moist_cont_next,  r, c, i, j, k) );
  SET3D( d__Q.convergence, r, c, i, j, k, GET3D(d__Q.convergence_next, r, c, i, j, k) );
}

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

// ----------------------------------------------------------------------------
// init kernel, called once before the simulation loop
// ----------------------------------------------------------------------------
__device__
void simulation_init(int i, int j, int k, Substates &Q, int r, int c, int s, Parameters &P)
{
  double quota, teta, satur, psi, h, _k, uno_su_dqdh;
  double ksTmp, moist_cont;
  double denom_pow, denompow_uno, denompow_due, denompow_tre;
  double exp_c, exp_d, satur_expc, satur_expd;
  double convergence;

  int k_inv = (s-1) - k;
  quota = P.lato * k_inv;
  ksTmp = GET3D(Q.ks, r, c, i, j, k);
  h = -P.h_init;

  psi = h - quota;
  if (psi < P.psi_zero)
  {
    denompow_uno = pow(P.alfa * (-psi), (1 - P.n));
    denompow_due = pow(P.alfa * (-psi), P.n);
    denompow_tre = pow((1 / (1 + denompow_due)), (1 / P.n - 2));
    uno_su_dqdh = (denompow_uno / (P.alfa * (P.n - 1) * (P.tetas - P.tetar))) * denompow_tre;
  }
  else
    uno_su_dqdh = 1 / P.ss;

  denom_pow = pow(P.alfa * (-psi), P.n);
  teta = P.tetar + ((P.tetas - P.tetar) * pow((1 / (1 + denom_pow)), (1 - 1 / P.n)));
  moist_cont = teta / P.tetas;

  satur = (teta - P.tetar) / (P.tetas - P.tetar);
  exp_c = P.n / (P.n - 1);
  satur_expc = pow(satur, exp_c);
  exp_d = 1 - (1 / P.n);
  satur_expd = pow((1 - satur_expc), exp_d);
  _k = ksTmp * pow(satur, 0.5) * pow((1 - satur_expd), 2);
  if ((_k > 0) && (uno_su_dqdh > 0))
    convergence = P.lato * P.lato / (ADJACENT_CELLS * _k * uno_su_dqdh);
  else
    convergence = 1.0;

  SET3D(Q.dqdh_next       , r, c, i, j, k, uno_su_dqdh);  
  SET3D(Q.psi_next        , r, c, i, j, k, psi);  
  SET3D(Q.k_next          , r, c, i, j, k, _k);  
  SET3D(Q.h_next          , r, c, i, j, k, h);  
  SET3D(Q.teta_next       , r, c, i, j, k, teta);  
  SET3D(Q.moist_cont_next , r, c, i, j, k, moist_cont);  
  SET3D(Q.convergence_next, r, c, i, j, k, convergence);  

  for(int n = 0; n < ADJACENT_CELLS; n++)
    BUF_SET3D(Q.F, r, c, s, n, i, j, k, 0.0);
}

// ----------------------------------------------------------------------------
// computing kernels, aka elementary processes in the XCA terminology
// ----------------------------------------------------------------------------
__device__
void reset_flows(int i, int j, int k, Substates &Q, int r, int c, int s)
{
  BUF_SET3D(Q.F, r, c, s, 0, i, j, k, 0.0);
  BUF_SET3D(Q.F, r, c, s, 1, i, j, k, 0.0);
  BUF_SET3D(Q.F, r, c, s, 2, i, j, k, 0.0);
  BUF_SET3D(Q.F, r, c, s, 3, i, j, k, 0.0);
  BUF_SET3D(Q.F, r, c, s, 4, i, j, k, 0.0);
  BUF_SET3D(Q.F, r, c, s, 5, i, j, k, 0.0);
}

#ifdef CUDA_VERSION_TILED_HALO
  #define GET3D_Q_h(xi, xj, xk) (GET3D(Q.h, s__rows, s__cols, s__i+xi, s__j+xj, s__k+xk))
  #define GET3D_Q_k(xi, xj, xk) (GET3D(Q.k, s__rows, s__cols, s__i+xi, s__j+xj, s__k+xk))
  #define BUF_GET3D_Q_F(n, xi, xj, xk) (BUF_GET3D(Q.F, s__rows, s__cols, s__slices, n, s__i+xi, s__j+xj, s__k+xk))
#else
  #define GET3D_Q_h(xi, xj, xk) (GET3D(Q.h, r, c, i+xi, j+xj, k+xk))
  #define GET3D_Q_k(xi, xj, xk) (GET3D(Q.k, r, c, i+xi, j+xj, k+xk))
  #define BUF_GET3D_Q_F(n, xi, xj, xk) (BUF_GET3D(Q.F, r, c, s, n, i+xi, j+xj, k+xk))
#endif

__device__
#ifdef CUDA_VERSION_TILED_HALO
void compute_flows(int i, int j, int k, int s__i, int s__j, int s__k, Substates &Q, int r, int c, int s, int s__rows, int s__cols, Parameters &P)
#else
void compute_flows(int i, int j, int k, Substates &Q, int r, int c, int s, Parameters &P)
#endif
{
  int k_inv = (s-1) - k;
  double Delta_h = 0.0;
  double h = GET3D_Q_h(0, 0, 0); 

  if (k_inv > P.ZFONDO && h > GET3D_Q_h(Xi[6], Xj[6], Xk[6]))
  {
    Delta_h = h - GET3D_Q_h(Xi[6], Xj[6], Xk[6]);
    BUF_SET3D(Q.F, r, c, s, 0, i, j, k, Delta_h);
  }

  if (k_inv < P.ZSUP && h > GET3D_Q_h(Xi[5], Xj[5], Xk[5]))
  {
    Delta_h = h - GET3D_Q_h(Xi[5], Xj[5], Xk[5]);
    BUF_SET3D(Q.F, r, c, s, 1, i, j, k, Delta_h);
  }

  if (i > P.XW && h > GET3D_Q_h(Xi[1], Xj[1], Xk[1]))
  {
    Delta_h = h - GET3D_Q_h(Xi[1], Xj[1], Xk[1]);
    BUF_SET3D(Q.F, r, c, s, 2, i, j, k, Delta_h);
  }

  if (i < P.XE && h > GET3D_Q_h(Xi[4], Xj[4], Xk[4]))
  {
    Delta_h = h - GET3D_Q_h(Xi[4], Xj[4], Xk[4]);
    BUF_SET3D(Q.F, r, c, s, 3, i, j, k, Delta_h);
  }

  if (j > P.YIN && h > GET3D_Q_h(Xi[2], Xj[2], Xk[2]))
  {
    Delta_h = h - GET3D_Q_h(Xi[2], Xj[2], Xk[2]);
    BUF_SET3D(Q.F, r, c, s, 4, i, j, k, Delta_h);
  }

  if (j < P.YOUT && h > GET3D_Q_h(Xi[3], Xj[3], Xk[3]))
  {
    Delta_h = h - GET3D_Q_h(Xi[3], Xj[3], Xk[3]);
    BUF_SET3D(Q.F, r, c, s, 5, i, j, k, Delta_h);
  }
}
__device__
#ifdef CUDA_VERSION_TILED_HALO
void mass_balance(int i, int j, int k, int s__i, int s__j, int s__k, Substates &Q, int r, int c, int s, int s__rows, int s__cols, int s__slices, Parameters &P)
#else
void mass_balance(int i, int j, int k, Substates &Q, int r, int c, int s, Parameters &P)
#endif
{
  int k_inv = (s-1) - k;
  double quota = P.lato * k_inv;

  double teta, satur, psi, h_next, uno_su_dqdh, teta_pioggia;
  double ks, moist_cont;
  double denom_pow, denompow_uno, denompow_due, denompow_tre;
  double exp_c, exp_d, satur_expc, satur_expd;
  double convergence;
  double temp_value;

  ks = GET3D(Q.ks, r, c, i, j, k);
  h_next = GET3D(Q.h, r, c, i, j, k);

  double currentK = GET3D_Q_k(0, 0, 0);
  double currentDQDH = GET3D(Q.dqdh, r, c, i, j, k);

  temp_value = ((currentK + GET3D_Q_k(Xi[4], Xj[4], Xk[4])) / 2.0) * currentDQDH;
  h_next = h_next - (BUF_GET3D_Q_F(3, 0, 0, 0) / (P.lato * P.lato)) * P.delta_t * temp_value;
  h_next = h_next + (BUF_GET3D_Q_F(2, Xi[4], Xj[4], Xk[4]) / (P.lato * P.lato)) * P.delta_t * temp_value;

  temp_value = ((currentK + GET3D_Q_k(Xi[1], Xj[1], Xk[1])) / 2.0) * currentDQDH;
  h_next = h_next - (BUF_GET3D_Q_F(2, 0, 0, 0) / (P.lato * P.lato)) * P.delta_t * temp_value;
  h_next = h_next + (BUF_GET3D_Q_F(3, Xi[1], Xj[1], Xk[1]) / (P.lato * P.lato)) * P.delta_t * temp_value;

  if( k_inv != P.ZSUP )
  {
    temp_value = ((currentK +GET3D_Q_k(Xi[5], Xj[5], Xk[5])) / 2.0) * currentDQDH;
    h_next = h_next - (BUF_GET3D_Q_F(1, 0, 0, 0) / (P.lato * P.lato)) * P.delta_t * temp_value;
    h_next = h_next + (BUF_GET3D_Q_F(0, Xi[5], Xj[5], Xk[5]) / (P.lato * P.lato)) * P.delta_t * temp_value;
  }

  if( k_inv != P.ZFONDO )
  {
    temp_value = ((currentK + GET3D_Q_k(Xi[6], Xj[6], Xk[6])) / 2.0) * currentDQDH;
    h_next = h_next - (BUF_GET3D_Q_F(0, 0, 0, 0) / (P.lato * P.lato)) * P.delta_t * temp_value;
    h_next = h_next + (BUF_GET3D_Q_F(1, Xi[6], Xj[6], Xk[6]) / (P.lato * P.lato)) * P.delta_t * temp_value;
  }

  temp_value = ((currentK + GET3D_Q_k(Xi[3], Xj[3], Xk[3])) / 2.0) * currentDQDH;
  h_next = h_next - (BUF_GET3D_Q_F(5, 0, 0, 0) / (P.lato * P.lato)) * P.delta_t * temp_value;
  h_next = h_next + (BUF_GET3D_Q_F(4, Xi[3], Xj[3], Xk[3]) / (P.lato * P.lato)) * P.delta_t * temp_value;

  temp_value = ((currentK + GET3D_Q_k(Xi[2], Xj[2], Xk[2])) / 2.0) * currentDQDH;
  h_next = h_next - (BUF_GET3D_Q_F(4, 0, 0, 0) / (P.lato * P.lato)) * P.delta_t * temp_value;
  h_next = h_next + (BUF_GET3D_Q_F(5, Xi[2], Xj[2], Xk[2]) / (P.lato * P.lato)) * P.delta_t * temp_value;

  if (k_inv == P.ZSUP && i <= r * 0.7 && i > r * 0.3 && j <= c * 0.7 && j > c * 0.3)
  {
    teta_pioggia = P.lato * P.lato * P.rain * P.delta_t / pow(P.lato, 3.0);
    h_next = h_next + teta_pioggia * currentDQDH;
  }

  psi = h_next - quota;
  if (psi < P.psi_zero)
  {
    denompow_uno = pow(P.alfa * (-psi), (1 - P.n));
    denompow_due = pow(P.alfa * (-psi), P.n);
    denompow_tre = pow((1 / (1 + denompow_due)), (1 / P.n - 2));
    uno_su_dqdh = (denompow_uno / (P.alfa * (P.n - 1) * (P.tetas - P.tetar))) * denompow_tre;
  }
  else
    uno_su_dqdh = 1 / P.ss;

  if (psi < 0)
    denom_pow = pow(P.alfa * (-psi), P.n);
  else
    denom_pow = pow(P.alfa * (psi), P.n);

  teta = P.tetar + ((P.tetas - P.tetar) * pow((1 / (1 + denom_pow)), (1 - 1 / P.n)));
  moist_cont = teta / P.tetas;

  satur = (teta - P.tetar) / (P.tetas - P.tetar);
  exp_c = P.n / (P.n - 1);
  satur_expc = pow(satur, exp_c);
  exp_d = 1 - (1 / P.n);
  satur_expd = pow((1 - satur_expc), exp_d);

  double _k = ks * pow(satur, 0.5) * pow((1 - satur_expd), 2);

  if ((_k > 0) && (uno_su_dqdh > 0))
    convergence = P.lato * P.lato / (ADJACENT_CELLS * _k * uno_su_dqdh); 
  else
    convergence = 1.0;

  SET3D(Q.dqdh_next       , r, c, i, j, k, uno_su_dqdh);  
  SET3D(Q.psi_next        , r, c, i, j, k, psi);  
  SET3D(Q.k_next          , r, c, i, j, k, _k);  
  SET3D(Q.h_next          , r, c, i, j, k, h_next);  
  SET3D(Q.teta_next       , r, c, i, j, k, teta);  
  SET3D(Q.moist_cont_next , r, c, i, j, k, moist_cont);  
  SET3D(Q.convergence_next, r, c, i, j, k, convergence);  
}

// ----------------------------------------------------------------------------
// cuda kernel routines
// ----------------------------------------------------------------------------

// BEGIN - common kernel block preface
// ----------------------------------------------------------------------------
#define ____KERNEL_BLOCK_PREFACE____                                       \
  const int i = blockIdx.y*blockDim.y + threadIdx.y;                       \
  const int j = blockIdx.x*blockDim.x + threadIdx.x;                       \
  const int k = blockIdx.z*blockDim.z + threadIdx.z;                       \
                                                                           \
  if ( i >= d__wsize[1] || j >= d__wsize[0] || k >= d__wsize[2] )          \
    return;                                                                \
                                                                           \
  const bool zerothread = threadIdx.x == threadIdx.y == threadIdx.z == 0;  \
  __shared__ Substates d__Q;                                               \
                                                                           \
  if ( zerothread )  /* thread 0 */                                        \
  {                                                                        \
    d__Q.__substates__ = d__substates__;                                   \
    syncSubstatesPtrs( d__Q, d__wsize[0]*d__wsize[1]*d__wsize[2] );        \
  }                                                                        \
  __syncthreads();
// ----------------------------------------------------------------------------
// END - common kernel block preface

__global__
void update_substates_kernel( double *d__substates__, int d__wsize[] )
{
  ____KERNEL_BLOCK_PREFACE____

  updateSubstates( d__Q, d__wsize[0], d__wsize[1], i, j, k );
}

__global__
void simul_init_kernel( double *d__substates__, Parameters *d__P, int d__wsize[] )
{
  ____KERNEL_BLOCK_PREFACE____

  simulation_init( i, j, k, d__Q, d__wsize[0], d__wsize[1], d__wsize[2], *d__P );
}

__global__
void reset_flows_kernel( double *d__substates__, Parameters *d__P, DomainBoundaries *d__mb_bounds, int d__wsize[] )
{
  ____KERNEL_BLOCK_PREFACE____

  //
  // Apply the reset flow kernel to the whole domain
  //
  reset_flows( i, j, k, d__Q, d__wsize[0], d__wsize[1], d__wsize[2] );
}

__global__
void compute_flows_kernel( double *d__substates__, Parameters *d__P, DomainBoundaries *d__mb_bounds, int d__wsize[] )
{
#ifdef CUDA_VERSION_TILED_HALO

  #define TILE_SIZE_X (blockDim.x-2)
  #define TILE_SIZE_Y (blockDim.y-2)
  #define TILE_SIZE_Z (blockDim.z-2)
  const int      i = blockIdx.y*TILE_SIZE_Y + threadIdx.y;
  const int      j = blockIdx.x*TILE_SIZE_X + threadIdx.x;
  const int      k = blockIdx.z*TILE_SIZE_Z + threadIdx.z;
  const int i_halo = i-1;
  const int j_halo = j-1;
  const int k_halo = k-1;

  Substates d__Q;
  extern __shared__ double s__Q_h[];

  d__Q.__substates__ = d__substates__;
  syncSubstatesPtrs( d__Q, d__wsize[0]*d__wsize[1]*d__wsize[2] );
  
  if ( i_halo >= 0 && j_halo >= 0 && k_halo >= 0 &&
       i_halo < d__wsize[0] && j_halo < d__wsize[1] && k_halo < d__wsize[2] )
    SET3D( s__Q_h, blockDim.y, blockDim.x, threadIdx.y, threadIdx.x, threadIdx.z, GET3D(d__Q.h, d__wsize[0], d__wsize[1], i_halo, j_halo, k_halo) );
  
  __syncthreads();

  if ( threadIdx.x >= TILE_SIZE_X || threadIdx.y >= TILE_SIZE_Y || threadIdx.z >= TILE_SIZE_Z ||
       i >= d__wsize[1] || j >= d__wsize[0] || k >= d__wsize[2] )
    return;

  d__Q.h = s__Q_h;
  //
  // Apply the flow computation kernel to the whole domain
  //
  compute_flows( i, j, k, threadIdx.y+1, threadIdx.x+1, threadIdx.z+1, d__Q, d__wsize[0], d__wsize[1], d__wsize[2], blockDim.y, blockDim.x, *d__P );

#else
  ____KERNEL_BLOCK_PREFACE____
  //
  // Apply the flow computation kernel to the whole domain
  //
  compute_flows( i, j, k, d__Q, d__wsize[0], d__wsize[1], d__wsize[2], *d__P );
#endif
}

__global__
void mass_balance_kernel( double *d__substates__, Parameters *d__P, DomainBoundaries *d__mb_bounds, int d__wsize[] )
{
#ifdef CUDA_VERSION_TILED_HALO

  #define TILE_SIZE_X (blockDim.x-2)
  #define TILE_SIZE_Y (blockDim.y-2)
  #define TILE_SIZE_Z (blockDim.z-2)
  const int      i = blockIdx.y*TILE_SIZE_Y + threadIdx.y;
  const int      j = blockIdx.x*TILE_SIZE_X + threadIdx.x;
  const int      k = blockIdx.z*TILE_SIZE_Z + threadIdx.z;
  const int i_halo = i-1;
  const int j_halo = j-1;
  const int k_halo = k-1;

  Substates d__Q;
  extern __shared__ double s__Q_kF[];
  double *s__Q_k = s__Q_kF;
  double *s__Q_F = s__Q_kF + blockDim.x*blockDim.y*blockDim.z;

  d__Q.__substates__ = d__substates__;
  syncSubstatesPtrs( d__Q, d__wsize[0]*d__wsize[1]*d__wsize[2] );
  
  if ( i_halo >= 0 && j_halo >= 0 && k_halo >= 0 &&
       i_halo < d__wsize[0] && j_halo < d__wsize[1] && k_halo < d__wsize[2] )
  {
    SET3D( s__Q_k, blockDim.y, blockDim.x, threadIdx.y, threadIdx.x, threadIdx.z, GET3D(d__Q.k, d__wsize[0], d__wsize[1], i_halo, j_halo, k_halo) );
    for ( int adjc=0; adjc < ADJACENT_CELLS; ++adjc )
      BUF_SET3D( s__Q_F, blockDim.y, blockDim.x, blockDim.z, adjc, threadIdx.y, threadIdx.x, threadIdx.z, BUF_GET3D(d__Q.F, d__wsize[0], d__wsize[1], d__wsize[2], adjc, i_halo, j_halo, k_halo) );
  }
  
  __syncthreads();

  if ( threadIdx.x >= TILE_SIZE_X || threadIdx.y >= TILE_SIZE_Y || threadIdx.z >= TILE_SIZE_Z ||
       i >= d__wsize[1] || j >= d__wsize[0] || k >= d__wsize[2] )
    return;

  d__Q.k = s__Q_k;
  d__Q.F = s__Q_F;
  //
  // Apply the flow computation kernel to the whole domain
  //
  if ( i >= d__mb_bounds->i_start && i < d__mb_bounds->i_end &&
       j >= d__mb_bounds->j_start && j < d__mb_bounds->j_end &&
       k >= d__mb_bounds->k_start && k < d__mb_bounds->k_end )
  mass_balance( i, j, k, threadIdx.y+1, threadIdx.x+1, threadIdx.z+1, d__Q, d__wsize[0], d__wsize[1], d__wsize[2], blockDim.y, blockDim.x, blockDim.z, *d__P );

#else
  ____KERNEL_BLOCK_PREFACE____

  //
  // Apply the mass balance kernel to the domain bounded by mb_bounds 
  //
  if ( i >= d__mb_bounds->i_start && i < d__mb_bounds->i_end &&
       j >= d__mb_bounds->j_start && j < d__mb_bounds->j_end &&
       k >= d__mb_bounds->k_start && k < d__mb_bounds->k_end )
    mass_balance( i, j, k, d__Q, d__wsize[0], d__wsize[1], d__wsize[2], *d__P );
#endif
}

__global__
void simul_steering( double *d__convergence, int size, double *minvar )
{
  const int t = threadIdx.x;
  const int i = blockDim.x*blockIdx.x + t;
  if ( i >= size ) return;

  extern __shared__ double s__minvar[];
  s__minvar[t] = d__convergence[i];

  for ( int stride = blockDim.x/2; stride >= 1; stride = stride>>1 )
  {
    __syncthreads();
    if ( t < stride && s__minvar[t+stride] < s__minvar[t] )
      s__minvar[t] = s__minvar[t+stride];
  }

  // write the result of this block to the global memory.
  if ( t == 0 ) minvar[blockIdx.x] = s__minvar[0];
}

// ----------------------------------------------------------------------------
// main() function
// ----------------------------------------------------------------------------
int main(int argc, char **argv)
{
  const int     r               = atoi(argv[ROWS_ID]);
  const int     c               = atoi(argv[COLS_ID]);
  const int     s               = atoi(argv[LAYERS_ID]);
  const char   *input_ks_path   = argv[INPUT_KS_ID];
  const double  simulation_time = atoi(argv[SIMUALITION_TIME_ID]);
  const char   *output_prefix   = argv[OUTPUT_PREFIX_ID];
  const int     blocksize_x     = atoi(argv[BLOCKSIZE_X_ID]);
  const int     blocksize_y     = atoi(argv[BLOCKSIZE_Y_ID]);
  const int     blocksize_z     = atoi(argv[BLOCKSIZE_Z_ID]);

  Substates        h__Q;
  Parameters       h__P;
  DomainBoundaries h__mb_bounds;
  int              h__wsize[] = { r, c, s };

  const int substsize = allocSubstates(h__Q, r, c, s);
  readKs(h__Q.ks, r, c, s, input_ks_path);
  initParameters(h__P, simulation_time, r, c, s); 
  initDomainBoundaries(h__mb_bounds, 1, r-1, 1, c-1, 0, s);

  //mpui::MPUI_WSize wsize = {c, r, s};
  //mpui::MPUI_Session *session;
  //mpui::MPUI_Init(mpui::MPUI_Mode::HUB, wsize, session);
  
  const dim3 block_size( blocksize_x, blocksize_y, blocksize_z );
  const dim3 grid_size ( ceil(c / (float)block_size.x), ceil(r / (float)block_size.y), ceil(s / (float)block_size.z) );

  const unsigned int steering_block_size = block_size.x * block_size.y * block_size.z;
        unsigned int steering_grid_size  = ceil(r*c*s / (float)steering_block_size);

  const int substate_offset_size = r*c*s;
  int reduction_size;

  #ifdef CUDA_VERSION_TILED_HALO
  const dim3 tiled_halo_block_size( blocksize_x+2, blocksize_y+2, blocksize_z+2 );
  const unsigned int sharedmem_size = tiled_halo_block_size.x * tiled_halo_block_size.y * tiled_halo_block_size.z;
  #endif

  double           *d__substates__;
  Parameters       *d__P;
  DomainBoundaries *d__mb_bounds;
  int              *d__wsize;
  double           *d__minvar;
  cudaMalloc( &d__substates__,     substsize * sizeof(double) );
  cudaMalloc( &d__P,                           sizeof(Parameters) );
  cudaMalloc( &d__mb_bounds,                   sizeof(DomainBoundaries) );
  cudaMalloc( &d__wsize,                   3 * sizeof(int) );
  cudaMalloc( &d__minvar, steering_grid_size * sizeof(double) );

  cudaMemcpy( d__substates__,  h__Q.__substates__, substsize * sizeof(double),           cudaMemcpyHostToDevice );
  cudaMemcpy( d__P,           &h__P,                           sizeof(Parameters),       cudaMemcpyHostToDevice );
  cudaMemcpy( d__mb_bounds,   &h__mb_bounds,                   sizeof(DomainBoundaries), cudaMemcpyHostToDevice );
  cudaMemcpy( d__wsize,        h__wsize,                   3 * sizeof(int),              cudaMemcpyHostToDevice );

  //
  // Apply the simulation init kernel to the whole domain
  //
  simul_init_kernel      <<< grid_size, block_size >>>( d__substates__, d__P, d__wsize );
  update_substates_kernel<<< grid_size, block_size >>>( d__substates__, d__wsize );

  //
  // simulation loop
  //
  util::Timer cl_timer;
  while( !(h__P.delta_t_cum >= h__P.simulation_time && h__P.delta_t_cum_prec <= h__P.simulation_time) )
  {
    cudaMemcpy( d__P, &h__P, sizeof(Parameters), cudaMemcpyHostToDevice );
    
    //
    // Apply the whole simulation cycle:
    //     1. apply the reset flow kernel to the whole domain
    //     2. apply the flow computation kernel to the whole domain
    //     3. apply the mass balance kernel to the domain bounded by mb_bounds
    //     4. simulation steering
    //
    reset_flows_kernel     <<< grid_size, block_size >>>( d__substates__, d__P, d__mb_bounds, d__wsize );
    compute_flows_kernel   <<< grid_size,
    #ifdef CUDA_VERSION_TILED_HALO
      tiled_halo_block_size, sharedmem_size * sizeof(double)
    #else
      block_size
    #endif
       >>>( d__substates__, d__P, d__mb_bounds, d__wsize );
    
    mass_balance_kernel    <<< grid_size,
    #ifdef CUDA_VERSION_TILED_HALO
      tiled_halo_block_size, sharedmem_size * sizeof(double) + sharedmem_size * sizeof(double) * ADJACENT_CELLS
    #else
      block_size
    #endif
       >>>( d__substates__, d__P, d__mb_bounds, d__wsize );
    update_substates_kernel<<< grid_size, block_size >>>( d__substates__, d__wsize );

    reduction_size = substate_offset_size;
    do
    {
      steering_grid_size = ceil(reduction_size / (float)steering_block_size);
      simul_steering<<< steering_grid_size, steering_block_size, steering_block_size * sizeof(double) >>>( d__substates__ + substate_offset_size*13, substate_offset_size, d__minvar );
      reduction_size = steering_grid_size;
    }
    while( steering_grid_size > 1 );

    double minVar;
    cudaMemcpy( &minVar, d__minvar, sizeof(double), cudaMemcpyDeviceToHost );
    
    if (minVar > 55.0)
      minVar = 55.0;
    
    h__P.delta_t = minVar;
    h__P.delta_t_cum_prec = h__P.delta_t_cum;
    h__P.delta_t_cum += h__P.delta_t;
    
    cudaError err = cudaGetLastError();
    if ( err != cudaSuccess )
    {
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    //mpui::MPUI_Recv_local(session, h__Q.h);
  }

  double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
  //printf("Elapsed time: %lf [s]\n", cl_time);
  printf("%lf\n", cl_time);

  // only h__Q.h is necessary at this point.
  cudaMemcpy( h__Q.__substates__ + substate_offset_size*9, d__substates__ + substate_offset_size*9, substate_offset_size * sizeof(double), cudaMemcpyDeviceToHost );
  
  cudaFree( d__substates__ );
  cudaFree( d__P           );
  cudaFree( d__mb_bounds   );
  cudaFree( d__wsize       );
  cudaFree( d__minvar      );

  //mpui::MPUI_Finalize(session);
  
  std::string s_path = (std::string)output_prefix + "h_LAST_simulation_time_" + util::converttostringint(simulation_time) + "s.txt";
  saveFile(h__Q.h, r, c, s, s_path);

  //printf("Releasing memory...\n");
  deleteSubstates(h__Q);

  return 0;
}
