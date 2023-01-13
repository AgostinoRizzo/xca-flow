#include "mbusu_dhpccpp.hpp"
#include "mbusu_kernel.cu"

#if defined(__MPUI__)
#include "../mpui/mpui.h"
#define  __MPUI_HOSTNAME__ "127.0.0.1"
#define  __MPUI_DT__ 500 
#endif

struct CudaKernelBlockConfig3D { dim3 block_size; unsigned long sharedmem_size; };
struct CudaKernelBlockConfig1D { unsigned int block_size; unsigned long sharedmem_size; };

// ----------------------------------------------------------------------------
// main() function
// ----------------------------------------------------------------------------
int main(int argc, char **argv)
{
  const char   *input_ks_path   = argv[INPUT_KS_ID];
  const double  simulation_time = atoi(argv[SIMUALITION_TIME_ID]);
  const char   *output_prefix   = argv[OUTPUT_PREFIX_ID];
  const int     blocksize_x     = atoi(argv[BLOCKSIZE_X_ID]);
  const int     blocksize_y     = atoi(argv[BLOCKSIZE_Y_ID]);
  const int     blocksize_z     = atoi(argv[BLOCKSIZE_Z_ID]);

  Substates  h__Q;
  Parameters h__P;

  const int substsize = allocSubstates(h__Q);
  readKs(h__Q.ks, input_ks_path);
  initParameters(h__P, simulation_time);

  #ifdef __MPUI__
  mpui::MPUI_WSize wsize = {c, r, s};
  mpui::MPUI_Session *session;
  mpui::MPUI_Init(mpui::MPUI_Mode::SOURCE, wsize, session);
  #endif
  
  //
  // CUDA kernel configurations.
  //
  const dim3 block_size( blocksize_x, blocksize_y, blocksize_z );
  const dim3 grid_size ( ceil(COLS / (float)block_size.x), ceil(ROWS / (float)block_size.y), ceil(SLICES / (float)block_size.z) );

  const unsigned int steering_block_size = block_size.x * block_size.y * block_size.z;
        unsigned int steering_grid_size  = ceil(SIZE / (float)steering_block_size);

  CudaKernelBlockConfig3D compute_flows_blkconfig;
  CudaKernelBlockConfig3D mass_balance_blkconfig;
  CudaKernelBlockConfig1D simul_steering_blkconfig{steering_block_size, steering_block_size * sizeof(double)};

#if defined(CUDA_VERSION_TILED_HALO)

  const dim3 tiled_halo_block_size    ( blocksize_x+2, blocksize_y+2, blocksize_z+2 );
  const unsigned int sharedmem_size = tiled_halo_block_size.x * tiled_halo_block_size.y * tiled_halo_block_size.z * sizeof(double);

  compute_flows_blkconfig.block_size     = tiled_halo_block_size;
  compute_flows_blkconfig.sharedmem_size = block_size.x*block_size.y*block_size.z * sizeof(double);

  mass_balance_blkconfig.block_size      = tiled_halo_block_size;
  mass_balance_blkconfig.sharedmem_size  = 2 * sharedmem_size * ADJACENT_CELLS

#elif defined(CUDA_VERSION_TILED_NO_HALO)

  compute_flows_blkconfig.block_size     = block_size;
  compute_flows_blkconfig.sharedmem_size = block_size.x*block_size.y*block_size.z * sizeof(double);

  mass_balance_blkconfig.block_size      = block_size;
  mass_balance_blkconfig.sharedmem_size  = block_size.x*block_size.y*block_size.z * sizeof(double) * (1 + ADJACENT_CELLS);

#else // CUDA_VERSION_BASIC

  compute_flows_blkconfig.block_size     = block_size;
  compute_flows_blkconfig.sharedmem_size = 0;

  mass_balance_blkconfig.block_size      = block_size;
  mass_balance_blkconfig.sharedmem_size  = 0;

#endif

  //
  // Auxiliary local variables.
  //
  bool          substates_swap = false;
  int           reduction_size;
  double       *d__reduction_buffer;
  double        minVar;
  unsigned int  minVarSize = sizeof(double);
  cudaError     err;

  //
  // GPU data structures alloc+memcpy.
  //
  double           *d__substates__;
  Parameters       *d__P;
  double           *d__minvar;
  cudaMalloc( &d__substates__,     substsize * sizeof(double) );
  cudaMalloc( &d__P,                           sizeof(Parameters) );
  cudaMalloc( &d__minvar, steering_grid_size * sizeof(double) );

  cudaMemcpy( d__substates__,  h__Q.__substates__, substsize * sizeof(double),     cudaMemcpyHostToDevice );
  cudaMemcpy( d__P,           &h__P,                           sizeof(Parameters), cudaMemcpyHostToDevice );

  //
  // Apply the simulation init kernel to the whole domain
  //
  simul_init_kernel <<< grid_size, block_size >>>( d__substates__, d__P );
  substates_swap = true; // update substates

  //
  // simulation loop
  //
  util::Timer cl_timer;
  while( !(h__P.delta_t_cum >= h__P.simulation_time && h__P.delta_t_cum_prec <= h__P.simulation_time) )
  {
    cudaMemcpy( d__P, &h__P, sizeof(Parameters), cudaMemcpyHostToDevice );
    
    //
    // Apply the whole simulation cycle:
    // 1. apply the reset flow kernel to the whole domain
    // 2. apply the flow computation kernel to the whole domain
    // 3. apply the mass balance kernel to the domain bounded by mb_bounds
    // 4. simulation steering
    //
    reset_flows_kernel      <<< grid_size, block_size >>>( d__substates__, d__P, substates_swap );
    compute_flows_kernel    <<< grid_size, compute_flows_blkconfig.block_size, compute_flows_blkconfig.sharedmem_size >>>( d__substates__, d__P, substates_swap );
    mass_balance_kernel     <<< grid_size, mass_balance_blkconfig.block_size, mass_balance_blkconfig.sharedmem_size >>>( d__substates__, d__P, substates_swap );
    update_substates_kernel <<< grid_size, block_size >>>( d__substates__, substates_swap );
    substates_swap = !substates_swap; // update substates

    reduction_size = __SUBSTATE_SIZE__;
    d__reduction_buffer = d__substates__ + __Q_convergence_OFFSET__;
    do
    {
      steering_grid_size = ceil(reduction_size / (float)steering_block_size);
      simul_steering<<< steering_grid_size, simul_steering_blkconfig.block_size, simul_steering_blkconfig.sharedmem_size >>>( d__reduction_buffer, reduction_size, d__minvar, substates_swap );
      
      d__reduction_buffer = d__minvar;
      reduction_size = steering_grid_size;
    }
    while( steering_grid_size > 1 );

    cudaMemcpy( &minVar, d__minvar, minVarSize, cudaMemcpyDeviceToHost );
    
    if (minVar > 55.0)
      minVar = 55.0;
    
    h__P.delta_t           = minVar;
    h__P.delta_t_cum_prec  = h__P.delta_t_cum;
    h__P.delta_t_cum      += h__P.delta_t;
    
    //
    // Manage CUDA errors.
    //
    err = cudaGetLastError();
    if ( err != cudaSuccess )
    {
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
      break;
    }
    
    #ifdef __MPUI__
    cudaMemcpy( h__Q.__substates__ + __Q_h__OFFSET__, d__substates__ + __Q_h__OFFSET__, __SUBSTATE_SIZE_BYTES__, cudaMemcpyDeviceToHost );
    mpui::MPUI_Send(session, h__Q.h, __MPUI_HOSTNAME__, __MPUI_DT__);
    #endif
  }

  double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
  //printf("Elapsed time: %lf [s]\n", cl_time);
  printf("%lf\n", cl_time);

  // only h__Q.h is necessary at this point.
  cudaMemcpy( h__Q.__substates__ + __Q_h__OFFSET__, d__substates__ + __Q_h__OFFSET__, __SUBSTATE_SIZE_BYTES__, cudaMemcpyDeviceToHost );
  
  //
  // GPU data structures free.
  //
  cudaFree( d__substates__ );
  cudaFree( d__P           );
  cudaFree( d__minvar      );
  
  std::string s_path = (std::string)output_prefix + "h_LAST_simulation_time_" + util::converttostringint(simulation_time) + "s.txt";
  saveFile(h__Q.h, s_path);

  #ifdef __MPUI__
  mpui::MPUI_Send(session, h__Q.h, __MPUI_HOSTNAME__, __MPUI_DT__);
  mpui::MPUI_Finalize(session);
  #endif
  
  //printf("Releasing memory...\n");
  deleteSubstates(h__Q);

  return 0;
}
