#define __MPI__
#include "mbusu_dhpccpp.hpp"
#include "mbusu_kernel.cu"
#include <mpi.h>

#define LOCAL_ROWS        ROWS
#define LOCAL_COLS        COLS
#define LOCAL_SLICES (SLICES/2)
#define LOCAL_SIZE     (SIZE/2)

struct CudaKernelBlockConfig3D { dim3 block_size; unsigned long sharedmem_size; };
struct CudaKernelBlockConfig1D { unsigned int block_size; unsigned long sharedmem_size; };

// ----------------------------------------------------------------------------
// main() function
// ----------------------------------------------------------------------------
int main(int argc, char **argv)
{
  int pid = -1;
  int np  = -1;
  MPI_Init( &argc, &argv );
  MPI_Comm_rank( MPI_COMM_WORLD, &pid );
  MPI_Comm_size( MPI_COMM_WORLD, &np  );

  const bool is_data_server = pid == 0;
  const int  other_pid = (pid + 1) % np;

  if ( np != 2 )
  {
    if ( is_data_server ) printf("A number of 2 processes is required\n");
    MPI_Abort( MPI_COMM_WORLD, 1 );
    return 1;
  }
  cudaSetDevice(pid);

  //
  // load input configs.
  //
  const char   *input_ks_path   = argv[INPUT_KS_ID];
  const double  simulation_time = atoi(argv[SIMUALITION_TIME_ID]);
  const char   *output_prefix   = argv[OUTPUT_PREFIX_ID];
  const int     blocksize_x     = atoi(argv[BLOCKSIZE_X_ID]);
  const int     blocksize_y     = atoi(argv[BLOCKSIZE_Y_ID]);
  const int     blocksize_z     = atoi(argv[BLOCKSIZE_Z_ID]);

  Substates  h__Q;
  Parameters h__P;
  Substates  h__Q_next_boundary;
  Substates  h__Q_next_halo;

  const int substsize = allocSubstates(h__Q);
  readKs(h__Q.ks, input_ks_path);
  initParameters(h__P, simulation_time);

  const unsigned long slice_bytes = SLICE_SIZE * sizeof(double);
  cudaHostAlloc( &(h__Q_next_boundary.k_next), slice_bytes, cudaHostAllocDefault );
  cudaHostAlloc( &(h__Q_next_boundary.h_next), slice_bytes, cudaHostAllocDefault );
  cudaHostAlloc( &(h__Q_next_boundary.F), slice_bytes, cudaHostAllocDefault );
  cudaHostAlloc( &(h__Q_next_halo.k_next), slice_bytes, cudaHostAllocDefault );
  cudaHostAlloc( &(h__Q_next_halo.h_next), slice_bytes, cudaHostAllocDefault );
  cudaHostAlloc( &(h__Q_next_halo.F), slice_bytes, cudaHostAllocDefault );

  //
  // CUDA kernel configurations.
  //
  const dim3 block_size       ( blocksize_x, blocksize_y, blocksize_z );
  const dim3 grid_size        ( ceil(COLS / (float)block_size.x), ceil(ROWS / (float)block_size.y), ceil(SLICES / (float)block_size.z) );

  const unsigned int steering_block_size = block_size.x * block_size.y * block_size.z;
        unsigned int steering_grid_size  = ceil(LOCAL_SIZE / (float)steering_block_size);

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
  unsigned int  boundary_slice     = pid == 0 ? LOCAL_SLICES - 1 : LOCAL_SLICES;
  unsigned int  halo_slice         = pid == 0 ? LOCAL_SLICES     : LOCAL_SLICES - 1;
  unsigned int  inner_slices_start = pid == 0 ? 0 : boundary_slice + 1;
  unsigned int  inner_slices_end   = pid == 0 ? boundary_slice - 1 : SIZE - 1;
  MPI_Status    status;
  cudaStream_t  stream0, stream1;
  Substates     d__Q;
  bool          substates_swap = false;
  int           reduction_size;
  double       *d__reduction_buffer;
  unsigned int  reduction_buffer_local_offset = pid == 0 ? 0 : LOCAL_SIZE;
  double        minVar, other_minVar;
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

  cudaStreamCreate( &stream0 );
  cudaStreamCreate( &stream1 );
  d__Q.__substates__ = d__substates__;

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

    //
    // Stage 1
    // Compute boundaries
    //
    reset_flows_kernel      <<< grid_size, block_size, 0, stream0 >>>( d__substates__, d__P, substates_swap, boundary_slice, boundary_slice );
    compute_flows_kernel    <<< grid_size, compute_flows_blkconfig.block_size, compute_flows_blkconfig.sharedmem_size, stream0 >>>( d__substates__, d__P, substates_swap, boundary_slice, boundary_slice );
    mass_balance_kernel     <<< grid_size, mass_balance_blkconfig.block_size, mass_balance_blkconfig.sharedmem_size, stream0 >>>( d__substates__, d__P, substates_swap, boundary_slice, boundary_slice );
    update_substates_kernel <<< grid_size, block_size, 0, stream0 >>>( d__substates__, substates_swap, boundary_slice, boundary_slice );

    //
    // Stage 2 (in parallel)
    // Send boundaries and receive halos
    // Compute Inner chunk
    //
    reset_flows_kernel      <<< grid_size, block_size, 0, stream1 >>>( d__substates__, d__P, substates_swap, inner_slices_start, inner_slices_end );
    compute_flows_kernel    <<< grid_size, compute_flows_blkconfig.block_size, compute_flows_blkconfig.sharedmem_size, stream1 >>>( d__substates__, d__P, substates_swap, inner_slices_start, inner_slices_end );
    mass_balance_kernel     <<< grid_size, mass_balance_blkconfig.block_size, mass_balance_blkconfig.sharedmem_size, stream1 >>>( d__substates__, d__P, substates_swap, inner_slices_start, inner_slices_end );
    update_substates_kernel <<< grid_size, block_size, 0, stream1 >>>( d__substates__, substates_swap, inner_slices_start, inner_slices_end );
    substates_swap = !substates_swap; // update substates

    cudaStreamSynchronize( stream0 );
    reduction_size = LOCAL_SIZE;
    d__reduction_buffer = d__substates__ + ( substates_swap ? __Q_convergence_next_OFFSET__ : __Q_convergence_OFFSET__ ) + reduction_buffer_local_offset;
    do
    {
      steering_grid_size = ceil(reduction_size / (float)steering_block_size);
      simul_steering<<< steering_grid_size, simul_steering_blkconfig.block_size, simul_steering_blkconfig.sharedmem_size, stream1 >>>( d__reduction_buffer, reduction_size, d__minvar, substates_swap );
      
      d__reduction_buffer = d__minvar;
      reduction_size = steering_grid_size;
    }
    while( steering_grid_size > 1 );

    //
    // Copy data needed by the other process to the host.
    //
    syncSubstatesPtrs( d__Q, !substates_swap );
    cudaMemcpyAsync( h__Q_next_boundary.k_next, d__Q.k_next + boundary_slice * SLICE_SIZE, slice_bytes, cudaMemcpyDeviceToHost, stream0 );
    cudaMemcpyAsync( h__Q_next_boundary.h_next, d__Q.h_next + boundary_slice * SLICE_SIZE, slice_bytes, cudaMemcpyDeviceToHost, stream0 );
    cudaMemcpyAsync( h__Q_next_boundary.F, d__Q.F + boundary_slice * SLICE_SIZE, slice_bytes, cudaMemcpyDeviceToHost, stream0 );
    cudaStreamSynchronize( stream0 );
    MPI_Sendrecv( h__Q_next_boundary.k_next, SLICE_SIZE, MPI_DOUBLE, other_pid, 0, h__Q_next_halo.k_next, SLICE_SIZE, MPI_DOUBLE, other_pid, 0, MPI_COMM_WORLD, &status );
    MPI_Sendrecv( h__Q_next_boundary.h_next, SLICE_SIZE, MPI_DOUBLE, other_pid, 0, h__Q_next_halo.h_next, SLICE_SIZE, MPI_DOUBLE, other_pid, 0, MPI_COMM_WORLD, &status );
    MPI_Sendrecv( h__Q_next_boundary.F,      SLICE_SIZE, MPI_DOUBLE, other_pid, 0, h__Q_next_halo.F,      SLICE_SIZE, MPI_DOUBLE, other_pid, 0, MPI_COMM_WORLD, &status );
    cudaMemcpyAsync( d__Q.k_next + halo_slice * SLICE_SIZE, h__Q_next_halo.k_next, slice_bytes, cudaMemcpyHostToDevice, stream0 );

    cudaMemcpy( &minVar, d__minvar, minVarSize, cudaMemcpyDeviceToHost );
    MPI_Sendrecv( &minVar, 1, MPI_DOUBLE, other_pid, 0, &other_minVar, 1, MPI_DOUBLE, other_pid, 0, MPI_COMM_WORLD, &status );
    if ( other_minVar < minVar ) minVar = other_minVar;
    
    if (minVar > 55.0)
      minVar = 55.0;
    
    h__P.delta_t           = minVar;
    h__P.delta_t_cum_prec  = h__P.delta_t_cum;
    h__P.delta_t_cum      += h__P.delta_t;

    //
    // block
    //
    cudaDeviceSynchronize();
    
    //
    // Manage CUDA errors.
    //
    err = cudaGetLastError();
    if ( err != cudaSuccess )
    {
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
      break;
    }
  }

  double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
  //printf("Elapsed time: %lf [s]\n", cl_time);
  if ( is_data_server ) printf("%lf\n", cl_time);

  // only h__Q.h is necessary at this point.
  cudaMemcpy( h__Q.__substates__ + __Q_h__OFFSET__, d__substates__ + __Q_h__OFFSET__, __SUBSTATE_SIZE_BYTES__, cudaMemcpyDeviceToHost );
  if ( is_data_server ) MPI_Recv( h__Q.h + halo_slice     * SLICE_SIZE, LOCAL_SIZE, MPI_DOUBLE, other_pid, 0, MPI_COMM_WORLD, &status );
  else                  MPI_Send( h__Q.h + boundary_slice * SLICE_SIZE, LOCAL_SIZE, MPI_DOUBLE, other_pid, 0, MPI_COMM_WORLD );
  
  //
  // GPU data structures free.
  //
  cudaFree( d__substates__ );
  cudaFree( d__P           );
  cudaFree( d__minvar      );
  
  if ( is_data_server )
  {
    std::string s_path = (std::string)output_prefix + "h_LAST_simulation_time_" + util::converttostringint(simulation_time) + "s.txt";
    saveFile(h__Q.h, s_path);
  }
  
  //printf("Releasing memory...\n");
  deleteSubstates(h__Q);
  cudaFreeHost( h__Q_next_boundary.k_next );
  cudaFreeHost( h__Q_next_boundary.h_next );
  cudaFreeHost( h__Q_next_boundary.F );
  cudaFreeHost( h__Q_next_halo.k_next );
  cudaFreeHost( h__Q_next_halo.h_next );
  cudaFreeHost( h__Q_next_halo.F );

  MPI_Finalize();

  return 0;
}
