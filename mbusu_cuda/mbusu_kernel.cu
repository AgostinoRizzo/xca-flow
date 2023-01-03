#include "mbusu_core.cu"

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

// ----------------------------------------------------------------------------
// CUDA KERNEL ROUTINES
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// common kernel block preface
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
#if defined(CUDA_VERSION_TILED_HALO)

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

#elif defined(CUDA_VERSION_TILED_NO_HALO)

  const int i = blockIdx.y*blockDim.y + threadIdx.y;
  const int j = blockIdx.x*blockDim.x + threadIdx.x;
  const int k = blockIdx.z*blockDim.z + threadIdx.z;

  if ( i >= d__wsize[1] || j >= d__wsize[0] || k >= d__wsize[2] )
    return;
  
  Substates d__Q;
  extern __shared__ double s__Q_h[];

  d__Q.__substates__ = d__substates__;
  syncSubstatesPtrs( d__Q, d__wsize[0]*d__wsize[1]*d__wsize[2] );

  SET3D( s__Q_h, blockDim.y, blockDim.x, threadIdx.y, threadIdx.x, threadIdx.z, GET3D(d__Q.h, d__wsize[0], d__wsize[1], i, j, k) );
  __syncthreads();

  //
  // Apply the flow computation kernel to the whole domain
  //
  compute_flows( i, j, k, threadIdx.y, threadIdx.x, threadIdx.z, d__Q, s__Q_h,
                 d__wsize[0], d__wsize[1], d__wsize[2], blockDim.y, blockDim.x, blockDim.z, *d__P );
  
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
#if defined(CUDA_VERSION_TILED_HALO)

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

#elif defined(CUDA_VERSION_TILED_NO_HALO)

  const int i = blockIdx.y*blockDim.y + threadIdx.y;
  const int j = blockIdx.x*blockDim.x + threadIdx.x;
  const int k = blockIdx.z*blockDim.z + threadIdx.z;

  if ( i >= d__wsize[1] || j >= d__wsize[0] || k >= d__wsize[2] )
    return;
  
  Substates d__Q;
  extern __shared__ double s__Q_kF[];
  double *s__Q_k = s__Q_kF;
  double *s__Q_F = s__Q_kF + blockDim.x*blockDim.y*blockDim.z;

  d__Q.__substates__ = d__substates__;
  syncSubstatesPtrs( d__Q, d__wsize[0]*d__wsize[1]*d__wsize[2] );

  SET3D( s__Q_k, blockDim.y, blockDim.x, threadIdx.y, threadIdx.x, threadIdx.z, GET3D(d__Q.k, d__wsize[0], d__wsize[1], i, j, k) );
  for ( int adjc=0; adjc < ADJACENT_CELLS; ++adjc )
    BUF_SET3D( s__Q_F, blockDim.y, blockDim.x, blockDim.z, adjc, threadIdx.y, threadIdx.x, threadIdx.z, BUF_GET3D(d__Q.F, d__wsize[0], d__wsize[1], d__wsize[2], adjc, i, j, k) );
  __syncthreads();

  //
  // Apply the mass balance kernel to the domain bounded by mb_bounds 
  //
  if ( i >= d__mb_bounds->i_start && i < d__mb_bounds->i_end &&
       j >= d__mb_bounds->j_start && j < d__mb_bounds->j_end &&
       k >= d__mb_bounds->k_start && k < d__mb_bounds->k_end )
    mass_balance( i, j, k, threadIdx.y, threadIdx.x, threadIdx.z, d__Q, s__Q_k, s__Q_F,
                  d__wsize[0], d__wsize[1], d__wsize[2], blockDim.y, blockDim.x, blockDim.z, *d__P );

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