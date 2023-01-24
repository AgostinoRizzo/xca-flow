#include "mbusu_dhpccpp.hpp"

// ----------------------------------------------------------------------------
// Read/Write access macros linearizing single/multy layer buffer 2D/3D indices
// for each cuda implementation
// ----------------------------------------------------------------------------
#if defined(CUDA_VERSION_TILED_HALO)
  #define GET3D_Q_h(xi, xj, xk) (GET3D(Q.h, s__rows, s__cols, s__i+(xi), s__j+(xj), s__k+(xk)))
  #define GET3D_Q_k(xi, xj, xk) (GET3D(Q.k, s__rows, s__cols, s__i+(xi), s__j+(xj), s__k+(xk)))
  #define BUF_GET3D_Q_F(n, xi, xj, xk) (BUF_GET3D(Q.F, ROWS, COLS, SLICES, (n), i+(xi), j+(xj), k+(xk)))
#elif defined(CUDA_VERSION_TILED_NO_HALO)
  #define GET3D_SHARED( M, s__M, i, j, k, s__rows, s__cols, s__slices, s__i, s__j, s__k ) \
    (( s__i < 0 || s__j < 0 || s__k < 0 || s__i >= s__rows || s__j >= s__cols || s__k >= s__slices ) \
           ? GET3D(M, ROWS, COLS, (i), (j), (k)) \
           : GET3D(s__M, s__rows, s__cols, (s__i), (s__j), (s__k)))
  #define GET3D_Q_h(xi, xj, xk) (GET3D_SHARED(Q.h, s__Q_h, i+(xi), j+(xj), k+(xk), s__rows, s__cols, s__slices, s__i+(xi), s__j+(xj), s__k+(xk) ))
  #define GET3D_Q_k(xi, xj, xk) (GET3D_SHARED(Q.k, s__Q_k, i+(xi), j+(xj), k+(xk), s__rows, s__cols, s__slices, s__i+(xi), s__j+(xj), s__k+(xk) ))
  #define BUF_GET3D_Q_F(n, xi, xj, xk) (BUF_GET3D(Q.F, ROWS, COLS, SLICES, n, i+(xi), j+(xj), k+(xk)))
#else
  #define GET3D_Q_h(xi, xj, xk) (GET3D(Q.h, ROWS, COLS, i+(xi), j+(xj), k+(xk)))
  #define GET3D_Q_k(xi, xj, xk) (GET3D(Q.k, ROWS, COLS, i+(xi), j+(xj), k+(xk)))
  #define BUF_GET3D_Q_F(n, xi, xj, xk) (BUF_GET3D(Q.F, ROWS, COLS, SLICES, (n), i+(xi), j+(xj), k+(xk)))
#endif
// ----------------------------------------------------------------------------

#define Xi_0  0
#define Xi_1 -1
#define Xi_2  0
#define Xi_3  0
#define Xi_4  1
#define Xi_5  0
#define Xi_6  0

#define Xj_0  0
#define Xj_1  0
#define Xj_2 -1
#define Xj_3  1
#define Xj_4  0
#define Xj_5  0
#define Xj_6  0

#define Xk_0  0
#define Xk_1  0
#define Xk_2  0
#define Xk_3  0
#define Xk_4  0
#define Xk_5 -1
#define Xk_6  1


// ----------------------------------------------------------------------------
// MBUSU KERNEL ROUTINES
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// init kernel, called once before the simulation loop
// ----------------------------------------------------------------------------
__device__
void simulation_init(int i, int j, int k, Substates &Q)
{
  double quota, teta, satur, psi, h, _k, uno_su_dqdh;
  double ksTmp, moist_cont;
  double denom_pow, denompow_uno, denompow_due, denompow_tre;
  double exp_c, exp_d, satur_expc, satur_expd;
  double convergence;

  int k_inv = (SLICES-1) - k;
  quota = P_lato * k_inv;
  ksTmp = GET3D(Q.ks, ROWS, COLS, i, j, k);
  h = -P_h_init;

  psi = h - quota;

  denom_pow = pow(P_alfa * (-psi), P_n);
  teta = P_tetar + ((P_tetas - P_tetar) * pow((1 / (1 + denom_pow)), (1 - 1 / P_n)));
  moist_cont = teta / P_tetas;

  satur = (teta - P_tetar) / (P_tetas - P_tetar);
  exp_c = P_n / (P_n - 1);
  satur_expc = pow(satur, exp_c);
  exp_d = 1 - (1 / P_n);
  satur_expd = pow((1 - satur_expc), exp_d);
  _k = ksTmp * pow(satur, 0.5) * pow((1 - satur_expd), 2);

  SET3D(Q.psi_next        , ROWS, COLS, i, j, k, psi);  
  SET3D(Q.k_next          , ROWS, COLS, i, j, k, _k);  
  SET3D(Q.h_next          , ROWS, COLS, i, j, k, h);  
  SET3D(Q.teta_next       , ROWS, COLS, i, j, k, teta);  
  SET3D(Q.moist_cont_next , ROWS, COLS, i, j, k, moist_cont);  

  for(int n = 0; n < ADJACENT_CELLS; n++)
    BUF_SET3D(Q.F, ROWS, COLS, SLICES, n, i, j, k, 0.0);
  
  if (psi < P_psi_zero)
  {
    denompow_uno = pow(P_alfa * (-psi), (1 - P_n));
    denompow_due = pow(P_alfa * (-psi), P_n);
    denompow_tre = pow((1 / (1 + denompow_due)), (1 / P_n - 2));
    uno_su_dqdh = (denompow_uno / (P_alfa * (P_n - 1) * (P_tetas - P_tetar))) * denompow_tre;
  }
  else
    uno_su_dqdh = 1 / P_ss;
  
  if ((_k > 0) && (uno_su_dqdh > 0))
    convergence = P_lato * P_lato / (ADJACENT_CELLS * _k * uno_su_dqdh);
  else
    convergence = 1.0;

  SET3D(Q.dqdh_next       , ROWS, COLS, i, j, k, uno_su_dqdh);  
  SET3D(Q.convergence_next, ROWS, COLS, i, j, k, convergence);  
}
// ----------------------------------------------------------------------------


// ----------------------------------------------------------------------------
// computing kernels, aka elementary processes in the XCA terminology
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// reset flows kernels
// ----------------------------------------------------------------------------
__device__
void reset_flows(int i, int j, int k, Substates &Q)
{
  BUF_SET3D(Q.F, ROWS, COLS, SLICES, 0, i, j, k, 0.0);
  BUF_SET3D(Q.F, ROWS, COLS, SLICES, 1, i, j, k, 0.0);
  BUF_SET3D(Q.F, ROWS, COLS, SLICES, 2, i, j, k, 0.0);
  BUF_SET3D(Q.F, ROWS, COLS, SLICES, 3, i, j, k, 0.0);
  BUF_SET3D(Q.F, ROWS, COLS, SLICES, 4, i, j, k, 0.0);
  BUF_SET3D(Q.F, ROWS, COLS, SLICES, 5, i, j, k, 0.0);
}
// ----------------------------------------------------------------------------


// ----------------------------------------------------------------------------
// compute flows kernels
// ----------------------------------------------------------------------------
__device__
#if defined(CUDA_VERSION_TILED_HALO)
void compute_flows(int i, int j, int k, int s__i, int s__j, int s__k, Substates &Q,
                   int s__rows, int s__cols)
#elif defined(CUDA_VERSION_TILED_NO_HALO)
void compute_flows(int i, int j, int k, int s__i, int s__j, int s__k, Substates &Q, double *s__Q_h, 
                   int s__rows, int s__cols, int s__slices )
#else
void compute_flows(int i, int j, int k, Substates &Q)
#endif
{
  int k_inv = (SLICES-1) - k;
  double h = GET3D_Q_h(0, 0, 0);
  double Q_h_temp;
  
  if ( k_inv > P_ZFONDO && h > (Q_h_temp=GET3D_Q_h(Xi_6, Xj_6, Xk_6)) )
    BUF_SET3D(Q.F, ROWS, COLS, SLICES, 0, i, j, k, h - Q_h_temp);

  if ( k_inv < P_ZSUP && h > (Q_h_temp=GET3D_Q_h(Xi_5, Xj_5, Xk_5)) )
    BUF_SET3D(Q.F, ROWS, COLS, SLICES, 1, i, j, k, h - Q_h_temp);

  if ( i > P_XW && h > (Q_h_temp=GET3D_Q_h(Xi_1, Xj_1, Xk_1)) )
    BUF_SET3D(Q.F, ROWS, COLS, SLICES, 2, i, j, k, h - Q_h_temp);

  if ( i < P_XE && h > (Q_h_temp=GET3D_Q_h(Xi_4, Xj_4, Xk_4)) )
    BUF_SET3D(Q.F, ROWS, COLS, SLICES, 3, i, j, k, h - Q_h_temp);

  if ( j > P_YIN && h > (Q_h_temp=GET3D_Q_h(Xi_2, Xj_2, Xk_2)) )
    BUF_SET3D(Q.F, ROWS, COLS, SLICES, 4, i, j, k, h - Q_h_temp);

  if ( j < P_YOUT && h > (Q_h_temp=GET3D_Q_h(Xi_3, Xj_3, Xk_3)) )
    BUF_SET3D(Q.F, ROWS, COLS, SLICES, 5, i, j, k, h - Q_h_temp);
}
// ----------------------------------------------------------------------------


// ----------------------------------------------------------------------------
// mass balance kernels
// ----------------------------------------------------------------------------
__device__
#if defined(CUDA_VERSION_TILED_HALO)
void mass_balance(int i, int j, int k, int s__i, int s__j, int s__k, Substates &Q, 
                  int s__rows, int s__cols, int s__slices, Parameters &P)
#elif defined(CUDA_VERSION_TILED_NO_HALO)
void mass_balance(int i, int j, int k, int s__i, int s__j, int s__k, Substates &Q, double *s__Q_k,
                  int s__rows, int s__cols, int s__slices, Parameters &P )
#else
void mass_balance(int i, int j, int k, Substates &Q, Parameters &P)
#endif
{
  int k_inv = (SLICES-1) - k;

  double teta, satur, psi, h_next, uno_su_dqdh;
  double denom_pow, satur_expd;

  /****************************************  h_next  **************************************************************/
  h_next = GET3D(Q.h, ROWS, COLS, i, j, k);

  double ks = GET3D(Q.ks, ROWS, COLS, i, j, k);
  double currentK = GET3D_Q_k(0, 0, 0);
  double currentDQDH = GET3D(Q.dqdh, ROWS, COLS, i, j, k);

  double P_delta_t__currentDQDH = P.delta_t * currentDQDH;

  #define temp_value ( ((currentK + GET3D_Q_k(Xi_4, Xj_4, Xk_4)) / 2.0) * P_delta_t__currentDQDH )
  h_next += ( (BUF_GET3D_Q_F(2, Xi_4, Xj_4, Xk_4) / (P_lato * P_lato)) - (BUF_GET3D_Q_F(3, 0, 0, 0) / (P_lato * P_lato)) ) * temp_value;
  #undef temp_value

  #define temp_value ( ((currentK + GET3D_Q_k(Xi_1, Xj_1, Xk_1)) / 2.0) * P_delta_t__currentDQDH )
  h_next += ( (BUF_GET3D_Q_F(3, Xi_1, Xj_1, Xk_1) / (P_lato * P_lato)) - (BUF_GET3D_Q_F(2, 0, 0, 0) / (P_lato * P_lato)) ) * temp_value;
  #undef temp_value

  if( k_inv != P_ZSUP )
  {
    #define temp_value ( ((currentK + GET3D_Q_k(Xi_5, Xj_5, Xk_5)) / 2.0) * P_delta_t__currentDQDH )
    h_next += ( (BUF_GET3D_Q_F(0, Xi_5, Xj_5, Xk_5) / (P_lato * P_lato)) - (BUF_GET3D_Q_F(1, 0, 0, 0) / (P_lato * P_lato)) ) * temp_value;
    #undef temp_value
  }
  
  if( k_inv != P_ZFONDO )
  {
    #define temp_value ( ((currentK + GET3D_Q_k(Xi_6, Xj_6, Xk_6)) / 2.0) * P_delta_t__currentDQDH )
    h_next += ( (BUF_GET3D_Q_F(1, Xi_6, Xj_6, Xk_6) / (P_lato * P_lato)) - (BUF_GET3D_Q_F(0, 0, 0, 0) / (P_lato * P_lato)) ) * temp_value;
    #undef temp_value
  }

  #define temp_value ( ((currentK + GET3D_Q_k(Xi_3, Xj_3, Xk_3)) / 2.0) * P_delta_t__currentDQDH )
  h_next += ( (BUF_GET3D_Q_F(4, Xi_3, Xj_3, Xk_3) / (P_lato * P_lato)) - (BUF_GET3D_Q_F(5, 0, 0, 0) / (P_lato * P_lato)) ) * temp_value;
  #undef temp_value

  #define temp_value ( ((currentK + GET3D_Q_k(Xi_2, Xj_2, Xk_2)) / 2.0) * P_delta_t__currentDQDH )
  h_next += ( (BUF_GET3D_Q_F(5, Xi_2, Xj_2, Xk_2) / (P_lato * P_lato)) - (BUF_GET3D_Q_F(4, 0, 0, 0) / (P_lato * P_lato)) ) * temp_value;
  #undef temp_value

  if (k_inv == P_ZSUP && i <= (ROWS * 0.7) && i > (ROWS * 0.3) && j <= (COLS * 0.7) && j > (COLS * 0.3))
    h_next += ((P_lato * P_lato * P_rain) / (P_lato * P_lato * P_lato)) * P_delta_t__currentDQDH;
  /******************************************************************************************************/

  #define satur_expc ( pow( satur, (P_n / (P_n - 1)) ) )

  psi        = h_next - (P_lato * k_inv);

  double __y_base__ = P_alfa * ( psi < 0 ? -psi : psi );
  double __y__ = pow( __y_base__, (1 - P_n) );
  denom_pow  = __y_base__ * (1/__y__);
  //denom_pow  = ( psi < 0 ? pow(P_alfa * (-psi), P_n) : pow(P_alfa * (psi), P_n) );

  double __x__ = pow ( (1 / (1 + denom_pow)), (1 - 1 / P_n) );

  teta       = P_tetar + ( (P_tetas - P_tetar) * __x__ );  // complicate expression
  
  satur      = (teta - P_tetar) / (P_tetas - P_tetar);

  satur_expd = ( pow( (1 - satur_expc ), (1 - (1 / P_n)) ) );

  //pow(satur, 0.5) * pow((1 - satur_expd), 2)
  //(1 + satur_expd * (-2 + satur_expd))
  double _k  = ks * sqrt(satur) * (1 - 2*satur_expd + satur_expd*satur_expd);  // complicate expression
  

  SET3D(Q.psi_next        , ROWS, COLS, i, j, k, psi);
  SET3D(Q.k_next          , ROWS, COLS, i, j, k, _k);  
  SET3D(Q.h_next          , ROWS, COLS, i, j, k, h_next);
  SET3D(Q.teta_next       , ROWS, COLS, i, j, k, teta);
  SET3D(Q.moist_cont_next , ROWS, COLS, i, j, k, teta / P_tetas);

  /****************************************  last divergence  *************************************************/
  if (psi < P_psi_zero)
  {
    #define denompow_uno ( __y__ )
    #define denompow_due denom_pow
    #define denompow_tre ( 1 / ( __x__ * (1 / (1 + denompow_due))) )
    uno_su_dqdh = (denompow_uno / (P_alfa * (P_n - 1) * (P_tetas - P_tetar))) * denompow_tre;  // complicate expression

  }
  else uno_su_dqdh = 1 / P_ss;

  SET3D(Q.dqdh_next       , ROWS, COLS, i, j, k, uno_su_dqdh);  
  SET3D(Q.convergence_next, ROWS, COLS, i, j, k, ( ((_k > 0) && (uno_su_dqdh > 0)) ? (P_lato * P_lato) / (ADJACENT_CELLS * _k * uno_su_dqdh) : 1.0 )  );  
}
// ----------------------------------------------------------------------------