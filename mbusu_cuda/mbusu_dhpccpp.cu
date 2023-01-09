#include "mbusu_dhpccpp.hpp"
#include "mbusu_kernel.cu"

#ifdef __MPUI__
#include "../mpui/mpui.h"
#define __MPUI__HOSTNAME__ ""
#endif

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

  #ifdef __MPUI__
  mpui::MPUI_WSize wsize = {c, r, s};
  mpui::MPUI_Session *session;
  mpui::MPUI_Init(mpui::MPUI_Mode::SOURCE, wsize, session);
  #endif
  
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
    #if defined(CUDA_VERSION_TILED_HALO)
      tiled_halo_block_size, sharedmem_size * sizeof(double)
    #elif defined(CUDA_VERSION_TILED_NO_HALO)
      block_size, block_size.x*block_size.y*block_size.z * sizeof(double)
    #else
      block_size
    #endif
       >>>( d__substates__, d__P, d__mb_bounds, d__wsize );
    
    mass_balance_kernel    <<< grid_size,
    #if defined(CUDA_VERSION_TILED_HALO)
      tiled_halo_block_size, sharedmem_size * sizeof(double) + sharedmem_size * sizeof(double) * ADJACENT_CELLS
    #elif defined(CUDA_VERSION_TILED_NO_HALO)
      block_size, block_size.x*block_size.y*block_size.z * sizeof(double) * (1 + ADJACENT_CELLS)
    #else
      block_size
    #endif
       >>>( d__substates__, d__P, d__mb_bounds, d__wsize );
    update_substates_kernel<<< grid_size, block_size >>>( d__substates__, d__wsize );

    
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
    }
    
    #ifdef __MPUI__
    //mpui::MPUI_Send(session, h__Q.h);
    #endif
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
  
  std::string s_path = (std::string)output_prefix + "h_LAST_simulation_time_" + util::converttostringint(simulation_time) + "s.txt";
  saveFile(h__Q.h, r, c, s, s_path);

  #ifdef __MPUI__
  mpui::MPUI_Send(session, h__Q.h, __MPUI__HOSTNAME__);
  mpui::MPUI_Finalize(session);
  #endif
  
  //printf("Releasing memory...\n");
  deleteSubstates(h__Q);

  return 0;
}
