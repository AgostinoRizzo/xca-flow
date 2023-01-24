#ifndef MBUSU_CUDA
#define MBUSU_CUDA

#include "mbusu_dhpccpp.hpp"
#include "mbusu_kernel_basic.cu"
#include "mbusu_kernel_halo.cu"
#include "mbusu_kernel_nohalo.cu"

struct CudaKernelBlockConfig3D { dim3 grid_size; dim3 block_size; unsigned long sharedmem_size; };
struct CudaKernelBlockConfig1D { unsigned int block_size; unsigned long sharedmem_size; };

class MbusuCuda
{
public:
	MbusuCuda( Substates &h__Q, Parameters &h__P, unsigned int substsize,
               unsigned int blocksize_x, unsigned int blocksize_y, unsigned int blocksize_z );
	virtual ~MbusuCuda();
	bool simul_init();
	#ifndef __MPI__
	bool simul_step();
	#endif
	void get_substate( double *substate, unsigned int substate_offset );
protected:

	Parameters   *h__P;
	double       *h__minvar;
	unsigned int  h__minvar_size;

	//
	// GPU data structures.
	//
	double           *d__substates__;
	Substates        *d__Q;
	Parameters       *d__P;
	double           *d__minvar;
	bool             *d__next_step;

	//
	// Auxiliary local variables.
	//
	bool          substates_swap;
	bool          next_step;
	int           reduction_size;
	double       *d__reduction_buffer;
	unsigned int  next_step_size;

	//
	// CUDA kernel configurations.
	//
	dim3 block_size;
	dim3 grid_size;
	
	const unsigned int steering_block_size = block_size.x * block_size.y * block_size.z;
  	unsigned int steering_grid_size  = ceil(SIZE / (float)steering_block_size);
	
	CudaKernelBlockConfig3D compute_flows_blkconfig;
	CudaKernelBlockConfig3D mass_balance_blkconfig;
	CudaKernelBlockConfig1D simul_steering_blkconfig{steering_block_size, steering_block_size * sizeof(double)};
};

class MbusuCudaPiped: public MbusuCuda
{
public:
	MbusuCudaPiped( Substates &h__Q, Parameters &h__P, unsigned int substsize,
                    unsigned int blocksize_x, unsigned int blocksize_y, unsigned int blocksize_z )
					: MbusuCuda( h__Q, h__P, substsize, blocksize_x, blocksize_y, blocksize_z )
		{ cudaStreamCreateWithFlags( &pipeStream, cudaStreamNonBlocking ); }
	virtual ~MbusuCudaPiped() { cudaStreamDestroy( pipeStream); }
	bool simul_init();
	#ifndef __MPI__
	bool simul_step();
	#endif
protected:
	cudaStream_t pipeStream;
};


//------------------------------------------------------------/
// Mbusu Cuda
//------------------------------------------------------------/

MbusuCuda::MbusuCuda(  Substates &h__Q, Parameters &h__P, unsigned int substsize,
                       unsigned int blocksize_x, unsigned int blocksize_y, unsigned int blocksize_z )
	: substates_swap(false), 
	  next_step(!(h__P.delta_t_cum >= h__P.simulation_time && h__P.delta_t_cum_prec <= h__P.simulation_time)),
	  next_step_size(sizeof(bool)),
	  block_size( blocksize_x, blocksize_y, blocksize_z )
{
	this->h__P = &h__P;

	//
	// GPU data structures alloc+memcpy.
	//
	cudaMalloc( &d__substates__,     substsize * sizeof(double) );
	cudaMalloc( &d__Q,                           sizeof(Substates) );
	cudaMalloc( &d__P,                           sizeof(Parameters) );
	cudaMalloc( &d__minvar, steering_grid_size * sizeof(double) );
	cudaMalloc( &d__next_step, sizeof(bool) );

	cudaMemcpy( d__substates__,  h__Q.__substates__, substsize * sizeof(double),     cudaMemcpyHostToDevice );
	cudaMemcpy( d__P,           &h__P,                           sizeof(Parameters), cudaMemcpyHostToDevice );

	//
	// CUDA kernel configurations.
	//
	const unsigned int grid_size_x      = (unsigned int) ceil(COLS       / (float)block_size.x);
	const unsigned int grid_size_y      = (unsigned int) ceil(ROWS       / (float)block_size.y);
	const unsigned int grid_size_z      = (unsigned int) ceil(SLICES     / (float)block_size.z);
	const unsigned int padd_grid_size_x = (unsigned int) ceil((COLS-2)   / (float)block_size.x);
	const unsigned int padd_grid_size_y = (unsigned int) ceil((ROWS-2)   / (float)block_size.y);
	const unsigned int padd_grid_size_z = (unsigned int) ceil((SLICES-2) / (float)block_size.z);
	grid_size = { grid_size_x, grid_size_y, grid_size_z };
	
	const unsigned int steering_block_size = block_size.x * block_size.y * block_size.z;
  	unsigned int steering_grid_size  = ceil(SIZE / (float)steering_block_size);

	h__minvar_size = ceil(__SUBSTATE_SIZE__ / (float)steering_block_size);
	h__minvar = new double[h__minvar_size];
	
	simul_steering_blkconfig = { steering_block_size, steering_block_size * sizeof(double) };

#if defined(CUDA_VERSION_TILED_HALO)

	const unsigned int sharedmem_size      = (blocksize_x+2) * (blocksize_y+2) * (blocksize_z+2) * sizeof(double);

	compute_flows_blkconfig.grid_size      = grid_size;
	compute_flows_blkconfig.block_size     = block_size;
	compute_flows_blkconfig.sharedmem_size = sharedmem_size;

	mass_balance_blkconfig.grid_size       = { padd_grid_size_x, padd_grid_size_y, grid_size_z };
	mass_balance_blkconfig.block_size      = block_size;
	mass_balance_blkconfig.sharedmem_size  = sharedmem_size;

#elif defined(CUDA_VERSION_TILED_NO_HALO)

	compute_flows_blkconfig.grid_size      = grid_size;
	compute_flows_blkconfig.block_size     = block_size;
	compute_flows_blkconfig.sharedmem_size = block_size.x*block_size.y*block_size.z * sizeof(double);

	mass_balance_blkconfig.grid_size       = { padd_grid_size_x, padd_grid_size_y, grid_size_z };
	mass_balance_blkconfig.block_size      = block_size;
	mass_balance_blkconfig.sharedmem_size  = block_size.x*block_size.y*block_size.z * sizeof(double);

#else // CUDA_VERSION_BASIC

	compute_flows_blkconfig.grid_size      = grid_size;
	compute_flows_blkconfig.block_size     = block_size;
	compute_flows_blkconfig.sharedmem_size = 0;

	mass_balance_blkconfig.grid_size       = { padd_grid_size_x, padd_grid_size_y, grid_size_z };
	mass_balance_blkconfig.block_size      = block_size;
	mass_balance_blkconfig.sharedmem_size  = 0;

#endif
}

MbusuCuda::~MbusuCuda()
{
	//
	// GPU data structures free.
	//
	cudaFree( d__substates__ );
	cudaFree( d__Q           );
	cudaFree( d__P           );
	cudaFree( d__minvar      );
	cudaFree( d__next_step   );

	delete[] h__minvar;
}

bool MbusuCuda::simul_init()
{
	simul_init_kernel <<< grid_size, block_size >>>( d__substates__, d__Q );
	substates_swap = true; // update substates
	return next_step;
}

#ifndef __MPI__
bool MbusuCuda::simul_step()
{
	/*cudaMemcpy( d__P, h__P, sizeof(Parameters), cudaMemcpyHostToDevice );*/

	//
    // Apply the whole simulation cycle:
    // 1. apply the reset flow kernel to the whole domain
    // 2. apply the flow computation kernel to the whole domain
    // 3. apply the mass balance kernel to the domain bounded by mb_bounds
    // 4. simulation steering
    //
    reset_flows_kernel      <<< grid_size, block_size >>>( d__substates__, d__P, d__Q, substates_swap );
    compute_flows_kernel    <<< compute_flows_blkconfig.grid_size, compute_flows_blkconfig.block_size, compute_flows_blkconfig.sharedmem_size >>>( d__substates__, d__Q, substates_swap );
    mass_balance_kernel     <<< mass_balance_blkconfig.grid_size, mass_balance_blkconfig.block_size, mass_balance_blkconfig.sharedmem_size >>>( d__substates__, d__P, d__Q, substates_swap );
    update_substates_kernel <<< grid_size, block_size >>>( d__substates__, d__Q, substates_swap );
    //substates_swap_kernel <<< 1, 1 >>> ( d__Q );
    substates_swap = !substates_swap; // update substates

    reduction_size = __SUBSTATE_SIZE__;
    d__reduction_buffer = d__substates__ + ( substates_swap ? __Q_convergence_next_OFFSET__ : __Q_convergence_OFFSET__ );
    do
    {
      steering_grid_size = ceil(reduction_size / (float)steering_block_size);
      simul_steering<<< steering_grid_size, simul_steering_blkconfig.block_size, simul_steering_blkconfig.sharedmem_size >>>( d__reduction_buffer, reduction_size, d__minvar, substates_swap, d__P, d__next_step );
      
      d__reduction_buffer = d__minvar;
      reduction_size = steering_grid_size;
    }
    while( steering_grid_size > 1 );

    cudaMemcpy( &next_step, d__next_step, next_step_size, cudaMemcpyDeviceToHost );

	/*reduction_size = __SUBSTATE_SIZE__;
    d__reduction_buffer = d__substates__ + ( substates_swap ? __Q_convergence_next_OFFSET__ : __Q_convergence_OFFSET__ );
    //do
    {
      steering_grid_size = ceil(reduction_size / (float)steering_block_size);
      simul_steering<<< steering_grid_size, simul_steering_blkconfig.block_size, simul_steering_blkconfig.sharedmem_size >>>( d__reduction_buffer, reduction_size, d__minvar, substates_swap, d__P, d__next_step );
      
      d__reduction_buffer = d__minvar;
      reduction_size = steering_grid_size;
    }
    //while( steering_grid_size > 1 );

	cudaMemcpy( h__minvar, d__minvar, h__minvar_size * sizeof(double), cudaMemcpyDeviceToHost );
	
	double minVar = h__minvar[0];
    for ( int __mv_i=1; __mv_i < h__minvar_size; ++__mv_i )
		if ( h__minvar[__mv_i] < minVar ) minVar = h__minvar[__mv_i];
    
    if (minVar > MIN_VAR)
      minVar = MIN_VAR;
    
    h__P->delta_t           = minVar;
    h__P->delta_t_cum_prec  = h__P->delta_t_cum;
    h__P->delta_t_cum      += h__P->delta_t;

	next_step = !(h__P->delta_t_cum >= h__P->simulation_time && h__P->delta_t_cum_prec <= h__P->simulation_time);*/

#ifdef __CUDA_DEBUG__
	//
    // Manage CUDA errors.
    //
    cudaError err = cudaGetLastError();
    if ( err != cudaSuccess )
    {
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
      return false;
    }
#endif

	return next_step;
}
#endif

void MbusuCuda::get_substate( double *substate, unsigned int substate_offset )
{
	cudaMemcpy( substate, d__substates__ + substate_offset, __SUBSTATE_SIZE_BYTES__, cudaMemcpyDeviceToHost );
}


//------------------------------------------------------------/
// Mbusu Cuda Piped
//------------------------------------------------------------/

bool MbusuCudaPiped::simul_init()
{
	simul_init_kernel <<< grid_size, block_size >>>( d__substates__, d__Q );
	substates_swap = true; // update substates

	reset_flows_kernel      <<< grid_size, block_size, 0 >>>( d__substates__, d__P, d__Q, substates_swap );
	return next_step;
}

#ifndef __MPI__
bool MbusuCudaPiped::simul_step()
{
	//
    // Apply the whole simulation cycle:
    // 1. apply the reset flow kernel to the whole domain
    // 2. apply the flow computation kernel to the whole domain
    // 3. apply the mass balance kernel to the domain bounded by mb_bounds
    // 4. simulation steering
    //
    compute_flows_kernel    <<< compute_flows_blkconfig.grid_size, compute_flows_blkconfig.block_size, compute_flows_blkconfig.sharedmem_size >>>( d__substates__, d__Q, substates_swap );
	mass_balance_kernel     <<< mass_balance_blkconfig.grid_size, mass_balance_blkconfig.block_size, mass_balance_blkconfig.sharedmem_size >>>( d__substates__, d__P, d__Q, substates_swap );
    update_substates_kernel <<< grid_size, block_size >>>( d__substates__, d__Q, substates_swap );
    substates_swap = !substates_swap; // update substates

	cudaDeviceSynchronize();
	reset_flows_kernel      <<< grid_size, block_size, 0, pipeStream >>>( d__substates__, d__P, d__Q, substates_swap );
    
    reduction_size = __SUBSTATE_SIZE__;
    d__reduction_buffer = d__substates__ + ( substates_swap ? __Q_convergence_next_OFFSET__ : __Q_convergence_OFFSET__ );
    do
    {
      steering_grid_size = ceil(reduction_size / (float)steering_block_size);
      simul_steering<<< steering_grid_size, simul_steering_blkconfig.block_size, simul_steering_blkconfig.sharedmem_size >>>( d__reduction_buffer, reduction_size, d__minvar, substates_swap, d__P, d__next_step );
      
      d__reduction_buffer = d__minvar;
      reduction_size = steering_grid_size;
    }
    while( steering_grid_size > 1 );

    cudaMemcpy( &next_step, d__next_step, next_step_size, cudaMemcpyDeviceToHost );

	cudaStreamSynchronize( pipeStream );

#ifdef __CUDA_DEBUG__
	//
    // Manage CUDA errors.
    //
    cudaError err = cudaGetLastError();
    if ( err != cudaSuccess )
    {
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
      return false;
    }
#endif

	return next_step;
}
#endif

#endif