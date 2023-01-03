#include "mbusu_dhpccpp.hpp"

__device__ int Xi[] = {0, -1,  0,  0,  1,  0,  0};
__device__ int Xj[] = {0,  0, -1,  1,  0,  0,  0};
__device__ int Xk[] = {0,  0,  0,  0,  0, -1,  1};

// ----------------------------------------------------------------------------
// MBUSU KERNEL ROUTINES
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// init kernel, called once before the simulation loop
// ----------------------------------------------------------------------------
__device__
void simulation_init(int i, int j, int k, Substates &Q, int r, int c, int s, Parameters &P)
{
  double quota, teta, satur, psi, h, _k, uno_su_dqdh;
  double ksTmp, moist_cont;
  double denom_pow, denompow_uno, denompow_due, denompow_tre;
  double exp_c, exp_d, satur_expc, satur_expd;
  double convergence;

  int k_inv = (s-1) - k;
  quota = P.lato * k_inv;
  ksTmp = GET3D(Q.ks, r, c, i, j, k);
  h = -P.h_init;

  psi = h - quota;
  if (psi < P.psi_zero)
  {
    denompow_uno = pow(P.alfa * (-psi), (1 - P.n));
    denompow_due = pow(P.alfa * (-psi), P.n);
    denompow_tre = pow((1 / (1 + denompow_due)), (1 / P.n - 2));
    uno_su_dqdh = (denompow_uno / (P.alfa * (P.n - 1) * (P.tetas - P.tetar))) * denompow_tre;
  }
  else
    uno_su_dqdh = 1 / P.ss;

  denom_pow = pow(P.alfa * (-psi), P.n);
  teta = P.tetar + ((P.tetas - P.tetar) * pow((1 / (1 + denom_pow)), (1 - 1 / P.n)));
  moist_cont = teta / P.tetas;

  satur = (teta - P.tetar) / (P.tetas - P.tetar);
  exp_c = P.n / (P.n - 1);
  satur_expc = pow(satur, exp_c);
  exp_d = 1 - (1 / P.n);
  satur_expd = pow((1 - satur_expc), exp_d);
  _k = ksTmp * pow(satur, 0.5) * pow((1 - satur_expd), 2);
  if ((_k > 0) && (uno_su_dqdh > 0))
    convergence = P.lato * P.lato / (ADJACENT_CELLS * _k * uno_su_dqdh);
  else
    convergence = 1.0;

  SET3D(Q.dqdh_next       , r, c, i, j, k, uno_su_dqdh);  
  SET3D(Q.psi_next        , r, c, i, j, k, psi);  
  SET3D(Q.k_next          , r, c, i, j, k, _k);  
  SET3D(Q.h_next          , r, c, i, j, k, h);  
  SET3D(Q.teta_next       , r, c, i, j, k, teta);  
  SET3D(Q.moist_cont_next , r, c, i, j, k, moist_cont);  
  SET3D(Q.convergence_next, r, c, i, j, k, convergence);  

  for(int n = 0; n < ADJACENT_CELLS; n++)
    BUF_SET3D(Q.F, r, c, s, n, i, j, k, 0.0);
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// computing kernels, aka elementary processes in the XCA terminology
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// reset flows kernels
// ----------------------------------------------------------------------------
__device__
void reset_flows(int i, int j, int k, Substates &Q, int r, int c, int s)
{
  BUF_SET3D(Q.F, r, c, s, 0, i, j, k, 0.0);
  BUF_SET3D(Q.F, r, c, s, 1, i, j, k, 0.0);
  BUF_SET3D(Q.F, r, c, s, 2, i, j, k, 0.0);
  BUF_SET3D(Q.F, r, c, s, 3, i, j, k, 0.0);
  BUF_SET3D(Q.F, r, c, s, 4, i, j, k, 0.0);
  BUF_SET3D(Q.F, r, c, s, 5, i, j, k, 0.0);
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Read/Write access macros linearizing single/multy layer buffer 2D/3D indices
// for each cuda implementation
// ----------------------------------------------------------------------------
#if defined(CUDA_VERSION_TILED_HALO)
  #define GET3D_Q_h(xi, xj, xk) (GET3D(Q.h, s__rows, s__cols, s__i+xi, s__j+xj, s__k+xk))
  #define GET3D_Q_k(xi, xj, xk) (GET3D(Q.k, s__rows, s__cols, s__i+xi, s__j+xj, s__k+xk))
  #define BUF_GET3D_Q_F(n, xi, xj, xk) (BUF_GET3D(Q.F, s__rows, s__cols, s__slices, n, s__i+xi, s__j+xj, s__k+xk))
#elif defined(CUDA_VERSION_TILED_NO_HALO)
  __device__
  inline double GET3D_SHARED( double *M, double *s__M,
                              int rows, int cols, int i, int j, int k,
                              int s__rows, int s__cols, int s__slices, int s__i, int s__j, int s__k )
  {
    return ( s__i < 0 || s__j < 0 || s__k < 0 || s__i >= s__rows || s__j >= s__cols || s__k >= s__slices )
           ? GET3D(M, rows, cols, i, j, k)
           : GET3D(s__M, s__rows, s__cols, s__i, s__j, s__k);
  }
  __device__
  inline double BUF_GET3D_SHARED( double *M, double *s__M,
                                  int rows, int cols, int slices, int n, int i, int j, int k,
                                  int s__rows, int s__cols, int s__slices, int s__i, int s__j, int s__k )
  {
    return ( s__i < 0 || s__j < 0 || s__k < 0 || s__i >= s__rows || s__j >= s__cols || s__k >= s__slices )
           ? BUF_GET3D(M, rows, cols, slices, n, i, j, k)
           : BUF_GET3D(s__M, s__rows, s__cols, s__slices, n, s__i, s__j, s__k);
  }
  #define GET3D_Q_h(xi, xj, xk) (GET3D_SHARED(Q.h, s__Q_h, r, c, i+xi, j+xj, k+xk, s__rows, s__cols, s__slices, s__i+xi, s__j+xj, s__k+xk ))
  #define GET3D_Q_k(xi, xj, xk) (GET3D_SHARED(Q.k, s__Q_k, r, c, i+xi, j+xj, k+xk, s__rows, s__cols, s__slices, s__i+xi, s__j+xj, s__k+xk ))
  #define BUF_GET3D_Q_F(n, xi, xj, xk) (BUF_GET3D_SHARED(Q.F, s__Q_F, r, c, s, n, i+xi, j+xj, k+xk, s__rows, s__cols, s__slices, s__i+xi, s__j+xj, s__k+xk ))
#else
  #define GET3D_Q_h(xi, xj, xk) (GET3D(Q.h, r, c, i+xi, j+xj, k+xk))
  #define GET3D_Q_k(xi, xj, xk) (GET3D(Q.k, r, c, i+xi, j+xj, k+xk))
  #define BUF_GET3D_Q_F(n, xi, xj, xk) (BUF_GET3D(Q.F, r, c, s, n, i+xi, j+xj, k+xk))
#endif
// ----------------------------------------------------------------------------


// ----------------------------------------------------------------------------
// compute flows kernels
// ----------------------------------------------------------------------------
__device__
#if defined(CUDA_VERSION_TILED_HALO)
void compute_flows(int i, int j, int k, int s__i, int s__j, int s__k, Substates &Q,
                   int r, int c, int s, int s__rows, int s__cols, Parameters &P)
#elif defined(CUDA_VERSION_TILED_NO_HALO)
void compute_flows(int i, int j, int k, int s__i, int s__j, int s__k, Substates &Q, double *s__Q_h, 
                   int r, int c, int s, int s__rows, int s__cols, int s__slices, Parameters &P )
#else
void compute_flows(int i, int j, int k, Substates &Q, int r, int c, int s, Parameters &P)
#endif
{
  int k_inv = (s-1) - k;
  double Delta_h = 0.0;
  double h = GET3D_Q_h(0, 0, 0); 

  if (k_inv > P.ZFONDO && h > GET3D_Q_h(Xi[6], Xj[6], Xk[6]))
  {
    Delta_h = h - GET3D_Q_h(Xi[6], Xj[6], Xk[6]);
    BUF_SET3D(Q.F, r, c, s, 0, i, j, k, Delta_h);
  }

  if (k_inv < P.ZSUP && h > GET3D_Q_h(Xi[5], Xj[5], Xk[5]))
  {
    Delta_h = h - GET3D_Q_h(Xi[5], Xj[5], Xk[5]);
    BUF_SET3D(Q.F, r, c, s, 1, i, j, k, Delta_h);
  }

  if (i > P.XW && h > GET3D_Q_h(Xi[1], Xj[1], Xk[1]))
  {
    Delta_h = h - GET3D_Q_h(Xi[1], Xj[1], Xk[1]);
    BUF_SET3D(Q.F, r, c, s, 2, i, j, k, Delta_h);
  }

  if (i < P.XE && h > GET3D_Q_h(Xi[4], Xj[4], Xk[4]))
  {
    Delta_h = h - GET3D_Q_h(Xi[4], Xj[4], Xk[4]);
    BUF_SET3D(Q.F, r, c, s, 3, i, j, k, Delta_h);
  }

  if (j > P.YIN && h > GET3D_Q_h(Xi[2], Xj[2], Xk[2]))
  {
    Delta_h = h - GET3D_Q_h(Xi[2], Xj[2], Xk[2]);
    BUF_SET3D(Q.F, r, c, s, 4, i, j, k, Delta_h);
  }

  if (j < P.YOUT && h > GET3D_Q_h(Xi[3], Xj[3], Xk[3]))
  {
    Delta_h = h - GET3D_Q_h(Xi[3], Xj[3], Xk[3]);
    BUF_SET3D(Q.F, r, c, s, 5, i, j, k, Delta_h);
  }
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// mass balance kernels
// ----------------------------------------------------------------------------
__device__
#if defined(CUDA_VERSION_TILED_HALO)
void mass_balance(int i, int j, int k, int s__i, int s__j, int s__k, Substates &Q, 
                  int r, int c, int s, int s__rows, int s__cols, int s__slices, Parameters &P)
#elif defined(CUDA_VERSION_TILED_NO_HALO)
void mass_balance(int i, int j, int k, int s__i, int s__j, int s__k, Substates &Q, double *s__Q_k, double *s__Q_F,
                  int r, int c, int s, int s__rows, int s__cols, int s__slices, Parameters &P )
#else
void mass_balance(int i, int j, int k, Substates &Q, int r, int c, int s, Parameters &P)
#endif
{
  int k_inv = (s-1) - k;
  double quota = P.lato * k_inv;

  double teta, satur, psi, h_next, uno_su_dqdh, teta_pioggia;
  double ks, moist_cont;
  double denom_pow, denompow_uno, denompow_due, denompow_tre;
  double exp_c, exp_d, satur_expc, satur_expd;
  double convergence;
  double temp_value;

  ks = GET3D(Q.ks, r, c, i, j, k);
  h_next = GET3D(Q.h, r, c, i, j, k);

  double currentK = GET3D_Q_k(0, 0, 0);
  double currentDQDH = GET3D(Q.dqdh, r, c, i, j, k);

  temp_value = ((currentK + GET3D_Q_k(Xi[4], Xj[4], Xk[4])) / 2.0) * currentDQDH;
  h_next = h_next - (BUF_GET3D_Q_F(3, 0, 0, 0) / (P.lato * P.lato)) * P.delta_t * temp_value;
  h_next = h_next + (BUF_GET3D_Q_F(2, Xi[4], Xj[4], Xk[4]) / (P.lato * P.lato)) * P.delta_t * temp_value;

  temp_value = ((currentK + GET3D_Q_k(Xi[1], Xj[1], Xk[1])) / 2.0) * currentDQDH;
  h_next = h_next - (BUF_GET3D_Q_F(2, 0, 0, 0) / (P.lato * P.lato)) * P.delta_t * temp_value;
  h_next = h_next + (BUF_GET3D_Q_F(3, Xi[1], Xj[1], Xk[1]) / (P.lato * P.lato)) * P.delta_t * temp_value;

  if( k_inv != P.ZSUP )
  {
    temp_value = ((currentK +GET3D_Q_k(Xi[5], Xj[5], Xk[5])) / 2.0) * currentDQDH;
    h_next = h_next - (BUF_GET3D_Q_F(1, 0, 0, 0) / (P.lato * P.lato)) * P.delta_t * temp_value;
    h_next = h_next + (BUF_GET3D_Q_F(0, Xi[5], Xj[5], Xk[5]) / (P.lato * P.lato)) * P.delta_t * temp_value;
  }

  if( k_inv != P.ZFONDO )
  {
    temp_value = ((currentK + GET3D_Q_k(Xi[6], Xj[6], Xk[6])) / 2.0) * currentDQDH;
    h_next = h_next - (BUF_GET3D_Q_F(0, 0, 0, 0) / (P.lato * P.lato)) * P.delta_t * temp_value;
    h_next = h_next + (BUF_GET3D_Q_F(1, Xi[6], Xj[6], Xk[6]) / (P.lato * P.lato)) * P.delta_t * temp_value;
  }

  temp_value = ((currentK + GET3D_Q_k(Xi[3], Xj[3], Xk[3])) / 2.0) * currentDQDH;
  h_next = h_next - (BUF_GET3D_Q_F(5, 0, 0, 0) / (P.lato * P.lato)) * P.delta_t * temp_value;
  h_next = h_next + (BUF_GET3D_Q_F(4, Xi[3], Xj[3], Xk[3]) / (P.lato * P.lato)) * P.delta_t * temp_value;

  temp_value = ((currentK + GET3D_Q_k(Xi[2], Xj[2], Xk[2])) / 2.0) * currentDQDH;
  h_next = h_next - (BUF_GET3D_Q_F(4, 0, 0, 0) / (P.lato * P.lato)) * P.delta_t * temp_value;
  h_next = h_next + (BUF_GET3D_Q_F(5, Xi[2], Xj[2], Xk[2]) / (P.lato * P.lato)) * P.delta_t * temp_value;

  if (k_inv == P.ZSUP && i <= r * 0.7 && i > r * 0.3 && j <= c * 0.7 && j > c * 0.3)
  {
    teta_pioggia = P.lato * P.lato * P.rain * P.delta_t / pow(P.lato, 3.0);
    h_next = h_next + teta_pioggia * currentDQDH;
  }

  psi = h_next - quota;
  if (psi < P.psi_zero)
  {
    denompow_uno = pow(P.alfa * (-psi), (1 - P.n));
    denompow_due = pow(P.alfa * (-psi), P.n);
    denompow_tre = pow((1 / (1 + denompow_due)), (1 / P.n - 2));
    uno_su_dqdh = (denompow_uno / (P.alfa * (P.n - 1) * (P.tetas - P.tetar))) * denompow_tre;
  }
  else
    uno_su_dqdh = 1 / P.ss;

  if (psi < 0)
    denom_pow = pow(P.alfa * (-psi), P.n);
  else
    denom_pow = pow(P.alfa * (psi), P.n);

  teta = P.tetar + ((P.tetas - P.tetar) * pow((1 / (1 + denom_pow)), (1 - 1 / P.n)));
  moist_cont = teta / P.tetas;

  satur = (teta - P.tetar) / (P.tetas - P.tetar);
  exp_c = P.n / (P.n - 1);
  satur_expc = pow(satur, exp_c);
  exp_d = 1 - (1 / P.n);
  satur_expd = pow((1 - satur_expc), exp_d);

  double _k = ks * pow(satur, 0.5) * pow((1 - satur_expd), 2);

  if ((_k > 0) && (uno_su_dqdh > 0))
    convergence = P.lato * P.lato / (ADJACENT_CELLS * _k * uno_su_dqdh); 
  else
    convergence = 1.0;

  SET3D(Q.dqdh_next       , r, c, i, j, k, uno_su_dqdh);  
  SET3D(Q.psi_next        , r, c, i, j, k, psi);  
  SET3D(Q.k_next          , r, c, i, j, k, _k);  
  SET3D(Q.h_next          , r, c, i, j, k, h_next);  
  SET3D(Q.teta_next       , r, c, i, j, k, teta);  
  SET3D(Q.moist_cont_next , r, c, i, j, k, moist_cont);  
  SET3D(Q.convergence_next, r, c, i, j, k, convergence);  
}
// ----------------------------------------------------------------------------