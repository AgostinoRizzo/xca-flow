#ifndef MBUSU_KERNEL_HALO
#define MBUSU_KERNEL_HALO

#include "mbusu_dhpccpp.hpp"
#include "mbusu_kernel_basic.cu"
#ifdef CUDA_VERSION_TILED_HALO

// ----------------------------------------------------------------------------
// CUDA KERNEL ROUTINES
// ----------------------------------------------------------------------------

__global__
void compute_flows_kernel( double *d__substates__, bool substates_swap ____SLICE_LIMIT_PARAMS____ )
{
  const int i = blockIdx.y*blockDim.y + threadIdx.y;
  const int j = blockIdx.x*blockDim.x + threadIdx.x;
  const int k = blockIdx.z*blockDim.z + threadIdx.z;

  ____SLICE_LIMIT_CHECK____

  if ( i >= ROWS || j >= COLS || k >= SLICES )
    return;

  Substates d__Q;
  extern __shared__ double s__Q_h[];

  d__Q.__substates__ = d__substates__;
  syncSubstatesPtrs( d__Q, substates_swap );
  
  #define BLOCK_I (threadIdx.y)
  #define BLOCK_J (threadIdx.x)
  #define BLOCK_K (threadIdx.z)
  #define LAST_BLOCK_I (blockDim.y-1)
  #define LAST_BLOCK_J (blockDim.x-1)
  #define LAST_BLOCK_K (blockDim.z-1)
  #define LAST_I (ROWS-1)
  #define LAST_J (COLS-1)
  #define LAST_K (SLICES-1)

  #define SHMEM_ROWS   (blockDim.y+2)
  #define SHMEM_COLS   (blockDim.x+2)
  #define SHMEM_SLICES (blockDim.z+2)

  const int shmem_i = threadIdx.y + 1;
  const int shmem_j = threadIdx.x + 1;
  const int shmem_k = threadIdx.z + 1;

  SET3D( s__Q_h, SHMEM_ROWS, SHMEM_COLS, shmem_i, shmem_j, shmem_k, GET3D(d__Q.h, ROWS, COLS, i, j, k) );
  
  if ( BLOCK_I == 0 && i > 0 )                 SET3D( s__Q_h, SHMEM_ROWS, SHMEM_COLS, shmem_i-1, shmem_j, shmem_k, GET3D(d__Q.h, ROWS, COLS, i-1, j, k) );
  if ( BLOCK_I == LAST_BLOCK_I && i < LAST_I ) SET3D( s__Q_h, SHMEM_ROWS, SHMEM_COLS, shmem_i+1, shmem_j, shmem_k, GET3D(d__Q.h, ROWS, COLS, i+1, j, k) );

  if ( BLOCK_J == 0 && j > 0 )                 SET3D( s__Q_h, SHMEM_ROWS, SHMEM_COLS, shmem_i, shmem_j-1, shmem_k, GET3D(d__Q.h, ROWS, COLS, i, j-1, k) );
  if ( BLOCK_J == LAST_BLOCK_J && j < LAST_J ) SET3D( s__Q_h, SHMEM_ROWS, SHMEM_COLS, shmem_i, shmem_j+1, shmem_k, GET3D(d__Q.h, ROWS, COLS, i, j+1, k) );

  if ( BLOCK_K == 0 && k > 0 )                 SET3D( s__Q_h, SHMEM_ROWS, SHMEM_COLS, shmem_i, shmem_j, shmem_k-1, GET3D(d__Q.h, ROWS, COLS, i, j, k-1) );
  if ( BLOCK_K == LAST_BLOCK_K && k < LAST_K ) SET3D( s__Q_h, SHMEM_ROWS, SHMEM_COLS, shmem_i, shmem_j, shmem_k+1, GET3D(d__Q.h, ROWS, COLS, i, j, k+1) );

  __syncthreads();

  d__Q.h = s__Q_h;
  //
  // Apply the flow computation kernel to the whole domain
  //
  compute_flows( i, j, k, shmem_i, shmem_j, shmem_k, d__Q, SHMEM_ROWS, SHMEM_COLS );

  #undef BLOCK_I
  #undef BLOCK_J
  #undef BLOCK_K
  #undef LAST_BLOCK_I
  #undef LAST_BLOCK_J
  #undef LAST_BLOCK_K
  #undef LAST_I
  #undef LAST_J
  #undef LAST_K

  #undef SHMEM_ROWS
  #undef SHMEM_COLS
  #undef SHMEM_SLICES
}


__global__
void mass_balance_kernel( double *d__substates__, Parameters *d__P, bool substates_swap ____SLICE_LIMIT_PARAMS____ )
{
  const int i = blockIdx.y*blockDim.y + threadIdx.y + 1;
  const int j = blockIdx.x*blockDim.x + threadIdx.x + 1;
  const int k = blockIdx.z*blockDim.z + threadIdx.z;

  ____SLICE_LIMIT_CHECK____
  
  if ( i >= (ROWS-1) || j >= (COLS-1) || k >= SLICES )
    return;
  
  Substates d__Q;
  extern __shared__ double s__Q_k[];

  d__Q.__substates__ = d__substates__;
  syncSubstatesPtrs( d__Q, substates_swap );

  #define BLOCK_I (threadIdx.y)
  #define BLOCK_J (threadIdx.x)
  #define BLOCK_K (threadIdx.z)
  #define LAST_BLOCK_I (blockDim.y-1)
  #define LAST_BLOCK_J (blockDim.x-1)
  #define LAST_BLOCK_K (blockDim.z-1)
  #define LAST_I (ROWS-1)
  #define LAST_J (COLS-1)
  #define LAST_K (SLICES-1)

  #define SHMEM_ROWS   (blockDim.y+2)
  #define SHMEM_COLS   (blockDim.x+2)
  #define SHMEM_SLICES (blockDim.z+2)

  const int shmem_i = threadIdx.y + 1;
  const int shmem_j = threadIdx.x + 1;
  const int shmem_k = threadIdx.z + 1;

  SET3D( s__Q_k, SHMEM_ROWS, SHMEM_COLS, shmem_i, shmem_j, shmem_k, GET3D(d__Q.k, ROWS, COLS, i, j, k) );
  
  if ( BLOCK_I == 0 && i > 0 )                 SET3D( s__Q_k, SHMEM_ROWS, SHMEM_COLS, shmem_i-1, shmem_j, shmem_k, GET3D(d__Q.k, ROWS, COLS, i-1, j, k) );
  if ( BLOCK_I == LAST_BLOCK_I && i < LAST_I ) SET3D( s__Q_k, SHMEM_ROWS, SHMEM_COLS, shmem_i+1, shmem_j, shmem_k, GET3D(d__Q.k, ROWS, COLS, i+1, j, k) );

  if ( BLOCK_J == 0 && j > 0 )                 SET3D( s__Q_k, SHMEM_ROWS, SHMEM_COLS, shmem_i, shmem_j-1, shmem_k, GET3D(d__Q.k, ROWS, COLS, i, j-1, k) );
  if ( BLOCK_J == LAST_BLOCK_J && j < LAST_J ) SET3D( s__Q_k, SHMEM_ROWS, SHMEM_COLS, shmem_i, shmem_j+1, shmem_k, GET3D(d__Q.k, ROWS, COLS, i, j+1, k) );

  if ( BLOCK_K == 0 && k > 0 )                 SET3D( s__Q_k, SHMEM_ROWS, SHMEM_COLS, shmem_i, shmem_j, shmem_k-1, GET3D(d__Q.k, ROWS, COLS, i, j, k-1) );
  if ( BLOCK_K == LAST_BLOCK_K && k < LAST_K ) SET3D( s__Q_k, SHMEM_ROWS, SHMEM_COLS, shmem_i, shmem_j, shmem_k+1, GET3D(d__Q.k, ROWS, COLS, i, j, k+1) );

  __syncthreads();

  d__Q.k = s__Q_k;
  //
  // Apply the mass balance kernel to the domain bounded by mb_bounds 
  //
  mass_balance( i, j, k, shmem_i, shmem_j, shmem_k, d__Q, SHMEM_ROWS, SHMEM_COLS, SHMEM_SLICES, *d__P );
}

#endif
#endif