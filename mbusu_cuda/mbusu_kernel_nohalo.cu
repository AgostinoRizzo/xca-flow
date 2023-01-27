#ifndef MBUSU_KERNEL_NOHALO
#define MBUSU_KERNEL_NOHALO

#include "mbusu_dhpccpp.hpp"
#include "mbusu_kernel_basic.cu"
#ifdef CUDA_VERSION_TILED_NO_HALO

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

  SET3D( s__Q_h, blockDim.y, blockDim.x, threadIdx.y, threadIdx.x, threadIdx.z, GET3D(d__Q.h, ROWS, COLS, i, j, k) );
  __syncthreads();

  //
  // Apply the flow computation kernel to the whole domain
  //
  compute_flows( i, j, k, threadIdx.y, threadIdx.x, threadIdx.z, d__Q, s__Q_h,
                 blockDim.y, blockDim.x, blockDim.z );
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
  
  SET3D( s__Q_k, blockDim.y, blockDim.x, threadIdx.y, threadIdx.x, threadIdx.z, GET3D(d__Q.k, ROWS, COLS, i, j, k) );  
  __syncthreads();

  //
  // Apply the mass balance kernel to the domain bounded by mb_bounds 
  //
  mass_balance( i, j, k, threadIdx.y, threadIdx.x, threadIdx.z, d__Q, s__Q_k,
                blockDim.y, blockDim.x, blockDim.z, *d__P );
}

#endif
#endif