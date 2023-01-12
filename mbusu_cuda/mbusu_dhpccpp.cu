#include "mbusu_dhpccpp.hpp"
#include "mbusu_kernel.cu"

#if defined(__MPUI__)
#include "../mpui/mpui.h"
#define  __MPUI_HOSTNAME__ "127.0.0.1"
#define  __MPUI_DT__ 500 
#endif

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

  Substates        h__Q;
  Parameters       h__P;

  const int substsize = allocSubstates(h__Q);
  readKs(h__Q.ks, input_ks_path);
  initParameters(h__P, simulation_time);

  #ifdef __MPUI__
  mpui::MPUI_WSize wsize = {c, r, s};
  mpui::MPUI_Session *session;
  mpui::MPUI_Init(mpui::MPUI_Mode::SOURCE, wsize, session);
  #endif
  
  const dim3 block_size( blocksize_x, blocksize_y, blocksize_z );
  const dim3 grid_size ( ceil(COLS / (float)block_size.x), ceil(ROWS / (float)block_size.y), ceil(SLICES / (float)block_size.z) );

  const unsigned int steering_block_size = block_size.x * block_size.y * block_size.z;
        unsigned int steering_grid_size  = ceil(SIZE / (float)steering_block_size);

  const int substate_offset_size = SIZE;
  int reduction_size;

  #ifdef CUDA_VERSION_TILED_HALO
  const dim3 tiled_halo_block_size( blocksize_x+2, blocksize_y+2, blocksize_z+2 );
  const unsigned int sharedmem_size = tiled_halo_block_size.x * tiled_halo_block_size.y * tiled_halo_block_size.z;
  #endif

  double           *d__substates__;
  Parameters       *d__P;
  double           *d__minvar;
  cudaMalloc( &d__substates__,     substsize * sizeof(double) );
  cudaMalloc( &d__P,                           sizeof(Parameters) );
  cudaMalloc( &d__minvar, steering_grid_size * sizeof(double) );

  cudaMemcpy( d__substates__,  h__Q.__substates__, substsize * sizeof(double),           cudaMemcpyHostToDevice );
  cudaMemcpy( d__P,           &h__P,                           sizeof(Parameters),       cudaMemcpyHostToDevice );

  //
  // Apply the simulation init kernel to the whole domain
  //
  simul_init_kernel      <<< grid_size, block_size >>>( d__substates__, d__P );
  update_substates_kernel<<< grid_size, block_size >>>( d__substates__ );

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
    reset_flows_kernel     <<< grid_size, block_size >>>( d__substates__, d__P );
    compute_flows_kernel   <<< grid_size,
    #if defined(CUDA_VERSION_TILED_HALO)
      tiled_halo_block_size, sharedmem_size * sizeof(double)
    #elif defined(CUDA_VERSION_TILED_NO_HALO)
      block_size, block_size.x*block_size.y*block_size.z * sizeof(double)
    #else
      block_size
    #endif
       >>>( d__substates__, d__P );
    
    mass_balance_kernel    <<< grid_size,
    #if defined(CUDA_VERSION_TILED_HALO)
      tiled_halo_block_size, sharedmem_size * sizeof(double) + sharedmem_size * sizeof(double) * ADJACENT_CELLS
    #elif defined(CUDA_VERSION_TILED_NO_HALO)
      block_size, block_size.x*block_size.y*block_size.z * sizeof(double) * (1 + ADJACENT_CELLS)
    #else
      block_size
    #endif
       >>>( d__substates__, d__P );
    update_substates_kernel<<< grid_size, block_size >>>( d__substates__ );

    reduction_size = substate_offset_size;
    double *d__reduction_buffer = d__substates__ + substate_offset_size*13;
    do
    {
      steering_grid_size = ceil(reduction_size / (float)steering_block_size);
      simul_steering<<< steering_grid_size, steering_block_size, steering_block_size * sizeof(double) >>>( d__reduction_buffer, reduction_size, d__minvar );
      
      d__reduction_buffer = d__minvar;
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
      break;
    }
    
    #ifdef __MPUI__
    cudaMemcpy( h__Q.__substates__ + substate_offset_size*9, d__substates__ + substate_offset_size*9, substate_offset_size * sizeof(double), cudaMemcpyDeviceToHost );
    mpui::MPUI_Send(session, h__Q.h, __MPUI_HOSTNAME__, __MPUI_DT__);
    #endif
  }

  double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
  //printf("Elapsed time: %lf [s]\n", cl_time);
  printf("%lf\n", cl_time);

  // only h__Q.h is necessary at this point.
  cudaMemcpy( h__Q.__substates__ + substate_offset_size*9, d__substates__ + substate_offset_size*9, substate_offset_size * sizeof(double), cudaMemcpyDeviceToHost );
  
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
