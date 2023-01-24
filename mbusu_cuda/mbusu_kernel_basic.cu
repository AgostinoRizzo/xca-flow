#ifndef MBUSU_KERNEL_BASIC
#define MBUSU_KERNEL_BASIC

#include "mbusu_core.cu"

__device__
void updateSubstates( Substates &d__Q, int i, int j, int k )
{
  SET3D( d__Q.dqdh,        ROWS, COLS, i, j, k, GET3D(d__Q.dqdh_next,        ROWS, COLS, i, j, k) );
  SET3D( d__Q.psi,         ROWS, COLS, i, j, k, GET3D(d__Q.psi_next,         ROWS, COLS, i, j, k) );
  SET3D( d__Q.k,           ROWS, COLS, i, j, k, GET3D(d__Q.k_next,           ROWS, COLS, i, j, k) );
  SET3D( d__Q.h,           ROWS, COLS, i, j, k, GET3D(d__Q.h_next,           ROWS, COLS, i, j, k) );
  SET3D( d__Q.teta,        ROWS, COLS, i, j, k, GET3D(d__Q.teta_next,        ROWS, COLS, i, j, k) );
  SET3D( d__Q.moist_cont,  ROWS, COLS, i, j, k, GET3D(d__Q.moist_cont_next,  ROWS, COLS, i, j, k) );
  SET3D( d__Q.convergence, ROWS, COLS, i, j, k, GET3D(d__Q.convergence_next, ROWS, COLS, i, j, k) );
}

__device__
void updateSubstatesReverse( Substates &d__Q, int i, int j, int k )
{
  SET3D( d__Q.dqdh_next,        ROWS, COLS, i, j, k, GET3D(d__Q.dqdh,        ROWS, COLS, i, j, k) );
  SET3D( d__Q.psi_next,         ROWS, COLS, i, j, k, GET3D(d__Q.psi,         ROWS, COLS, i, j, k) );
  SET3D( d__Q.k_next,           ROWS, COLS, i, j, k, GET3D(d__Q.k,           ROWS, COLS, i, j, k) );
  SET3D( d__Q.h_next,           ROWS, COLS, i, j, k, GET3D(d__Q.h,           ROWS, COLS, i, j, k) );
  SET3D( d__Q.teta_next,        ROWS, COLS, i, j, k, GET3D(d__Q.teta,        ROWS, COLS, i, j, k) );
  SET3D( d__Q.moist_cont_next,  ROWS, COLS, i, j, k, GET3D(d__Q.moist_cont,  ROWS, COLS, i, j, k) );
  SET3D( d__Q.convergence_next, ROWS, COLS, i, j, k, GET3D(d__Q.convergence, ROWS, COLS, i, j, k) );
}


// ----------------------------------------------------------------------------
// slice limit parameters + check
// ----------------------------------------------------------------------------
#ifdef __MPI__
  #define ____SLICE_LIMIT_PARAMS____ , unsigned int slice_start, unsigned int slice_end
#else
  #define ____SLICE_LIMIT_PARAMS____
#endif
#ifdef __MPI__
  #define ____SLICE_LIMIT_CHECK____ if ( k < slice_start || k > slice_end ) return;
#else
  #define ____SLICE_LIMIT_CHECK____
#endif



// ----------------------------------------------------------------------------
// CUDA KERNEL ROUTINES
// ----------------------------------------------------------------------------

__global__
void update_substates_kernel( double *d__substates__, Substates *__d__Q, bool substates_swap ____SLICE_LIMIT_PARAMS____ )
{
  const int i = blockIdx.y*blockDim.y + threadIdx.y;
  const int j = blockIdx.x*blockDim.x + threadIdx.x;
  const int k = blockIdx.z*blockDim.z + threadIdx.z;

  if ( i >= ROWS || j >= COLS || k >= SLICES )
    return;

  ____SLICE_LIMIT_CHECK____

  Substates d__Q;
  d__Q.__substates__ = d__substates__;
  syncSubstatesPtrs( d__Q, substates_swap );

  if ( i < MB_BOUNDS_i_start || i >= MB_BOUNDS_i_end ||
       j < MB_BOUNDS_j_start || j >= MB_BOUNDS_j_end ||
       k < MB_BOUNDS_k_start || k >= MB_BOUNDS_k_end )
    updateSubstatesReverse( d__Q, i, j, k );
}



__global__
void simul_init_kernel( double *d__substates__, Substates *__d__Q )
{
  const int i = blockIdx.y*blockDim.y + threadIdx.y;                       
  const int j = blockIdx.x*blockDim.x + threadIdx.x;                       
  const int k = blockIdx.z*blockDim.z + threadIdx.z;                       

  if ( i >= ROWS || j >= COLS || k >= SLICES )                             
    return;
  
  Substates d__Q;
  d__Q.__substates__ = d__substates__;
  syncSubstatesPtrs( d__Q, false );
  
  simulation_init( i, j, k, d__Q );
}

__global__
void reset_flows_kernel( double *d__substates__, Parameters *d__P, Substates *__d__Q, bool substates_swap ____SLICE_LIMIT_PARAMS____ )
{
  const int i = blockIdx.y*blockDim.y + threadIdx.y;
  const int j = blockIdx.x*blockDim.x + threadIdx.x;
  const int k = blockIdx.z*blockDim.z + threadIdx.z;

  if ( i >= ROWS || j >= COLS || k >= SLICES )
    return;

  ____SLICE_LIMIT_CHECK____

  Substates d__Q;
  d__Q.__substates__ = d__substates__;
  syncSubstatesPtrs( d__Q, substates_swap );

  //
  // Apply the reset flow kernel to the whole domain
  //
  reset_flows( i, j, k, d__Q );
}



#ifdef CUDA_VERSION_BASIC

__global__
void compute_flows_kernel( double *d__substates__, Substates *__d__Q, bool substates_swap ____SLICE_LIMIT_PARAMS____ )
{
  const int i = blockIdx.y*blockDim.y + threadIdx.y;
  const int j = blockIdx.x*blockDim.x + threadIdx.x;
  const int k = blockIdx.z*blockDim.z + threadIdx.z;

  if ( i >= ROWS || j >= COLS || k >= SLICES )
    return;

  ____SLICE_LIMIT_CHECK____

  Substates d__Q;
  d__Q.__substates__ = d__substates__;
  syncSubstatesPtrs( d__Q, substates_swap );

  //
  // Apply the flow computation kernel to the whole domain
  //
  compute_flows( i, j, k, d__Q );
}

__global__
void mass_balance_kernel( double *d__substates__, Parameters *d__P, Substates *__d__Q, bool substates_swap ____SLICE_LIMIT_PARAMS____ )
{
  const int i = blockIdx.y*blockDim.y + threadIdx.y + 1;                       
  const int j = blockIdx.x*blockDim.x + threadIdx.x + 1;                       
  const int k = blockIdx.z*blockDim.z + threadIdx.z;                       
                                                                           
  Substates d__Q;
  d__Q.__substates__ = d__substates__;
  syncSubstatesPtrs( d__Q, substates_swap );                            
                                                                           
  if ( i >= (ROWS-1) || j >= (COLS-1) || k >= SLICES )                             
    return; 

  ____SLICE_LIMIT_CHECK____

  //
  // Apply the mass balance kernel to the domain bounded by mb_bounds 
  //
  mass_balance( i, j, k, d__Q, *d__P );
}

#endif

__global__
void simul_steering( double *d__convergence, int size, double *minvar, bool substates_swap, Parameters *d__P, bool *d__next_step )
{
  const int t = threadIdx.x;
  const int i = blockDim.x*blockIdx.x + t;
  if ( i >= size ) return;

  extern __shared__ double s__minvar[];
  s__minvar[t] = d__convergence[i];

  for ( int stride = blockDim.x>>1; stride >= 1; stride = stride>>1 )
  {
    __syncthreads();
    if ( t < stride && i+stride < size && s__minvar[t+stride] < s__minvar[t] )
      s__minvar[t] = s__minvar[t+stride];
  }

  // write the result of this block to the global memory.
  if ( t == 0 )
  {
    if ( gridDim.x == 1 )
    {
      double minVar = s__minvar[0];
      if (minVar > MIN_VAR)
      minVar = MIN_VAR;
    
      d__P->delta_t           = minVar;
      d__P->delta_t_cum_prec  = d__P->delta_t_cum;
      d__P->delta_t_cum      += minVar;
      (*d__next_step) = !(d__P->delta_t_cum >= d__P->simulation_time && d__P->delta_t_cum_prec <= d__P->simulation_time);
    }
    else
      minvar[blockIdx.x] = s__minvar[0];
  }
}

#endif