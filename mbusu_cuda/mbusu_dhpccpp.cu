#include "mbusu_dhpccpp.hpp"
#include "mbusu_cuda.cuh"

#define __MPUI__
#if defined(__MPUI__)
#include "../mpui/mpui.h"
#define  __MPUI_HOSTNAME__ "87.7.112.212"
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

  Substates  h__Q;
  Parameters h__P;

  const int substsize = allocSubstates(h__Q);
  readKs(h__Q.ks, input_ks_path);
  initParameters(h__P, simulation_time);

  #ifdef __MPUI__
  mpui::MPUI_WSize wsize = {COLS, ROWS, SLICES};
  mpui::MPUI_Session *session;
  mpui::MPUI_Init(mpui::MPUI_Mode::SOURCE, wsize, session);
  #endif
  
  //
  // MBUSU Cuda simulation object.
  //
  MbusuCuda mbusu( h__Q, h__P, substsize, blocksize_x, blocksize_y, blocksize_z );

  //
  // Auxiliary local variables.
  //
  bool      next_step;
  int       steps = 0;

  //
  // Apply the simulation init kernel to the whole domain
  //
  next_step = mbusu.simul_init();

  //
  // simulation loop
  //
  util::Timer cl_timer;
  while( next_step )
  {
    //
    // Apply the whole simulation cycle
    //
    next_step = mbusu.simul_step();

    #ifdef __MPUI__
    mbusu.get_substate( h__Q.h, __Q_h__OFFSET__ );
    mpui::MPUI_Send(session, h__Q.h, __MPUI_HOSTNAME__, __MPUI_DT__);
    #endif

    ++steps;
  }

  double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
  //printf("Elapsed time: %lf [s]\n", cl_time);
  printf("%lf\n", cl_time);
  printf("Steps: %d\n", steps);

  // only h__Q.h is necessary at this point.
  mbusu.get_substate( h__Q.h, __Q_h__OFFSET__ );
  
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
