#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "util.hpp"
// ----------------------------------------------------------------------------
// I/O parameters used to index argv[]
// ----------------------------------------------------------------------------
#define ROWS_ID 1
#define COLS_ID 2
#define LAYERS_ID 3
#define INPUT_KS_ID 4
#define SIMUALITION_TIME_ID 5
#define OUTPUT_PREFIX_ID 6
// ----------------------------------------------------------------------------
// Simulation parameters
// ----------------------------------------------------------------------------
#define ADJACENT_CELLS 6
#define VON_NEUMANN_NEIGHBORHOOD_3D_CELLS 7
// ----------------------------------------------------------------------------
// Read/Write access macros linearizing single/multy layer buffer 2D/3D indices
// ----------------------------------------------------------------------------
#define SET3D(M, rows, columns, i, j, k, value) ( (M)[( ((k)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )
#define GET3D(M, rows, columns, i, j, k) ( M[( ((k)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )
#define BUF_SET3D(M, rows, columns, slices, n, i, j, k, value) ( (M)[( ((n)*(rows)*(columns)*(slices)) + ((k)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )
#define BUF_GET3D(M, rows, columns, slices, n, i, j, k) ( M[( ((n)*(rows)*(columns)*(slices)) + ((k)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )
// ----------------------------------------------------------------------------

struct DomainBoundaries
{
  int i_start;
  int i_end;
  int j_start;
  int j_end;
  int k_start;
  int k_end;
};

void initDomainBoundaries(DomainBoundaries& B, int i_start, int i_end, int j_start, int j_end, int k_start, int k_end)
{
  B.i_start = i_start;
  B.i_end   = i_end; 
  B.j_start = j_start;
  B.j_end   = j_end;
  B.k_start = k_start;
  B.k_end   = k_end;
}

int Xi[] = {0, -1,  0,  0,  1,  0,  0};
int Xj[] = {0,  0, -1,  1,  0,  0,  0};
int Xk[] = {0,  0,  0,  0,  0, -1,  1};

struct Substates
{
  double* ks;

  double* teta;
  double* teta_next;
  double* moist_cont;
  double* moist_cont_next;
  double* psi;
  double* psi_next;
  double* k;
  double* k_next;
  double* h;
  double* h_next;
  double* dqdh;
  double* dqdh_next;
  double* convergence;
  double* convergence_next;

  double* F;
};

void allocSubstates(Substates& Q, int r, int c, int s)
{
  Q.ks               = util::allocBuffer3D(r, c, s);

  Q.teta             = util::allocBuffer3D(r, c, s);
  Q.teta_next        = util::allocBuffer3D(r, c, s);
  Q.moist_cont       = util::allocBuffer3D(r, c, s);
  Q.moist_cont_next  = util::allocBuffer3D(r, c, s);
  Q.psi              = util::allocBuffer3D(r, c, s);
  Q.psi_next         = util::allocBuffer3D(r, c, s);
  Q.k                = util::allocBuffer3D(r, c, s);
  Q.k_next           = util::allocBuffer3D(r, c, s);
  Q.h                = util::allocBuffer3D(r, c, s);
  Q.h_next           = util::allocBuffer3D(r, c, s);
  Q.dqdh             = util::allocBuffer3D(r, c, s);
  Q.dqdh_next        = util::allocBuffer3D(r, c, s);
  Q.convergence      = util::allocBuffer3D(r, c, s);
  Q.convergence_next = util::allocBuffer3D(r, c, s);

  Q.F                = util::allocBuffer4D(ADJACENT_CELLS, r, c, s);
}

void deleteSubstates(Substates& Q)
{
  free(Q.ks);

  free(Q.teta);
  free(Q.teta_next);
  free(Q.moist_cont);
  free(Q.moist_cont_next);
  free(Q.psi);
  free(Q.psi_next);
  free(Q.k);
  free(Q.k_next);
  free(Q.h);
  free(Q.h_next);
  free(Q.dqdh);
  free(Q.dqdh_next);
  free(Q.convergence);
  free(Q.convergence_next);

  free(Q.F);
}

void updateSubstates(Substates& Q, int r, int c, int s)
{
  memcpy(Q.dqdh,        Q.dqdh_next,        sizeof(double)*r*c*s);
  memcpy(Q.psi,         Q.psi_next,         sizeof(double)*r*c*s);
  memcpy(Q.k,           Q.k_next,           sizeof(double)*r*c*s);
  memcpy(Q.h,           Q.h_next,           sizeof(double)*r*c*s);
  memcpy(Q.teta,        Q.teta_next,        sizeof(double)*r*c*s);
  memcpy(Q.moist_cont,  Q.moist_cont_next,  sizeof(double)*r*c*s);
  memcpy(Q.convergence, Q.convergence_next, sizeof(double)*r*c*s);
}

struct Parameters
{
  int YOUT;
  int YIN;
  int XE;
  int XW;
  int ZFONDO;
  int ZSUP;

  double h_init;
  double tetas;
  double tetar;
  double alfa;
  double n;
  double rain;
  double psi_zero;
  double ss;
  double lato;
  double delta_t;
  double delta_t_cum;
  double delta_t_cum_prec;
  double simulation_time;
};

void initParameters(Parameters& P, double simulation_time, int r, int c, int s)
{
  P.YOUT = c-1;
  P.YIN = 0;
  P.XE = r-1;
  P.XW = 0;
  P.ZSUP = s-1;
  P.ZFONDO = 0;

  P.h_init = 734;
  P.tetas = 0.348;
  P.tetar = 0.095467;
  P.alfa = 0.034733333;
  P.n = 1.729;
  P.rain = 0.000023148148;
  P.psi_zero = -0.1;
  P.ss = 0.0001;
  P.lato = 30.0;	
  P.simulation_time = simulation_time;
  P.delta_t = 10.0;
  P.delta_t_cum = 0.0;
  P.delta_t_cum_prec = 0.0;
}

// ----------------------------------------------------------------------------
// I/O functions
// ----------------------------------------------------------------------------
void readKs(double* ks, int r, int c, int s, std::string path)
{
  FILE *f = fopen(path.c_str(), "r");
  if (f == NULL)
  {
    printf("can not open file %s", path.c_str());
    exit(0);
  }
  //printf("read succefully %s \n", path.c_str());
  char str[256];
  int i, j, k;
  for (k = 0; k < s; k++)
    for (i = 0; i < r; i++)
      for (j = 0; j < c; j++)
      {
        fscanf(f, "%s", str);
        SET3D(ks, r, c, i, j, k, atof(str));
      }
  fclose(f);
}

void saveFile(double* sub, int r, int c, int s, std::string nameFile)
{
  int i, j, k;
  double moist_print;

  FILE *stream = fopen(nameFile.c_str(), "w");
  for (k = 0; k < s; k++)
  {
    for (i = 0; i < r; i++)
    {
      for (j = 0; j < c; j++)
      {
        moist_print = GET3D(sub, r, c, i, j, k);
        fprintf(stream, "%.8f ", moist_print);
      }
      fprintf(stream, "\n");
    }
    fprintf(stream, "\n");
  }
  fclose(stream);
}

// ----------------------------------------------------------------------------
// init kernel, called once before the simulation loop
// ----------------------------------------------------------------------------

void simulation_init(int i, int j, int k, Substates Q, int r, int c, int s, Parameters P)
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
// computing kernels, aka elementary processes in the XCA terminology
// ----------------------------------------------------------------------------
void reset_flows(int i, int j, int k, Substates Q, int r, int c, int s)
{
  BUF_SET3D(Q.F, r, c, s, 0, i, j, k, 0.0);
  BUF_SET3D(Q.F, r, c, s, 1, i, j, k, 0.0);
  BUF_SET3D(Q.F, r, c, s, 2, i, j, k, 0.0);
  BUF_SET3D(Q.F, r, c, s, 3, i, j, k, 0.0);
  BUF_SET3D(Q.F, r, c, s, 4, i, j, k, 0.0);
  BUF_SET3D(Q.F, r, c, s, 5, i, j, k, 0.0);
}

void compute_flows(int i, int j, int k, Substates Q, int r, int c, int s, Parameters P)
{
  int k_inv = (s-1) - k;
  double Delta_h = 0.0;
  double h = GET3D(Q.h, r, c, i, j, k); 

  if (k_inv > P.ZFONDO && h > GET3D(Q.h, r, c, i+Xi[6], j+Xj[6], k+Xk[6]))
  {
    Delta_h = h - GET3D(Q.h, r, c, i+Xi[6], j+Xj[6], k+Xk[6]);
    BUF_SET3D(Q.F, r, c, s, 0, i, j, k, Delta_h);
  }

  if (k_inv < P.ZSUP && h > GET3D(Q.h, r, c, i+Xi[5], j+Xj[5], k+Xk[5]))
  {
    Delta_h = h - GET3D(Q.h, r, c, i+Xi[5], j+Xj[5], k+Xk[5]);
    BUF_SET3D(Q.F, r, c, s, 1, i, j, k, Delta_h);
  }

  if (i > P.XW && h > GET3D(Q.h, r, c, i+Xi[1], j+Xj[1], k+Xk[1]))
  {
    Delta_h = h - GET3D(Q.h, r, c, i+Xi[1], j+Xj[1], k+Xk[1]);
    BUF_SET3D(Q.F, r, c, s, 2, i, j, k, Delta_h);
  }

  if (i < P.XE && h > GET3D(Q.h, r, c, i+Xi[4], j+Xj[4], k+Xk[4]))
  {
    Delta_h = h - GET3D(Q.h, r, c, i+Xi[4], j+Xj[4], k+Xk[4]);
    BUF_SET3D(Q.F, r, c, s, 3, i, j, k, Delta_h);
  }

  if (j > P.YIN && h > GET3D(Q.h, r, c, i+Xi[2], j+Xj[2], k+Xk[2]))
  {
    Delta_h = h - GET3D(Q.h, r, c, i+Xi[2], j+Xj[2], k+Xk[2]);
    BUF_SET3D(Q.F, r, c, s, 4, i, j, k, Delta_h);
  }

  if (j < P.YOUT && h > GET3D(Q.h, r, c, i+Xi[3], j+Xj[3], k+Xk[3]))
  {
    Delta_h = h - GET3D(Q.h, r, c, i+Xi[3], j+Xj[3], k+Xk[3]);
    BUF_SET3D(Q.F, r, c, s, 5, i, j, k, Delta_h);
  }
}

void mass_balance(int i, int j, int k, Substates Q, int r, int c, int s, Parameters P)
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

  double currentK = GET3D(Q.k, r, c, i, j, k);
  double currentDQDH = GET3D(Q.dqdh, r, c, i, j, k);

  temp_value = ((currentK + GET3D(Q.k, r, c, i+Xi[4], j+Xj[4], k+Xk[4])) / 2.0) * currentDQDH;
  h_next = h_next - (BUF_GET3D(Q.F, r, c, s, 3, i, j, k) / (P.lato * P.lato)) * P.delta_t * temp_value;
  h_next = h_next + (BUF_GET3D(Q.F, r, c, s, 2, i+Xi[4], j+Xj[4], k+Xk[4]) / (P.lato * P.lato)) * P.delta_t * temp_value;

  temp_value = ((currentK + GET3D(Q.k, r, c, i+Xi[1], j+Xj[1], k+Xk[1])) / 2.0) * currentDQDH;
  h_next = h_next - (BUF_GET3D(Q.F, r, c, s, 2, i, j, k) / (P.lato * P.lato)) * P.delta_t * temp_value;
  h_next = h_next + (BUF_GET3D(Q.F, r, c, s, 3, i+Xi[1], j+Xj[1], k+Xk[1]) / (P.lato * P.lato)) * P.delta_t * temp_value;

  if( k_inv != P.ZSUP )
  {
    temp_value = ((currentK +GET3D(Q.k, r, c, i+Xi[5], j+Xj[5], k+Xk[5])) / 2.0) * currentDQDH;
    h_next = h_next - (BUF_GET3D(Q.F, r, c, s, 1, i, j, k) / (P.lato * P.lato)) * P.delta_t * temp_value;
    h_next = h_next + (BUF_GET3D(Q.F, r, c, s, 0, i+Xi[5], j+Xj[5], k+Xk[5]) / (P.lato * P.lato)) * P.delta_t * temp_value;
  }

  if( k_inv != P.ZFONDO )
  {
    temp_value = ((currentK + GET3D(Q.k, r, c, i+Xi[6], j+Xj[6], k+Xk[6])) / 2.0) * currentDQDH;
    h_next = h_next - (BUF_GET3D(Q.F, r, c, s, 0, i, j, k) / (P.lato * P.lato)) * P.delta_t * temp_value;
    h_next = h_next + (BUF_GET3D(Q.F, r, c, s, 1, i+Xi[6], j+Xj[6], k+Xk[6]) / (P.lato * P.lato)) * P.delta_t * temp_value;
  }

  temp_value = ((currentK + GET3D(Q.k, r, c, i+Xi[3], j+Xj[3], k+Xk[3])) / 2.0) * currentDQDH;
  h_next = h_next - (BUF_GET3D(Q.F, r, c, s, 5, i, j, k) / (P.lato * P.lato)) * P.delta_t * temp_value;
  h_next = h_next + (BUF_GET3D(Q.F, r, c, s, 4, i+Xi[3], j+Xj[3], k+Xk[3]) / (P.lato * P.lato)) * P.delta_t * temp_value;

  temp_value = ((currentK + GET3D(Q.k, r, c, i+Xi[2], j+Xj[2], k+Xk[2])) / 2.0) * currentDQDH;
  h_next = h_next - (BUF_GET3D(Q.F, r, c, s, 4, i, j, k) / (P.lato * P.lato)) * P.delta_t * temp_value;
  h_next = h_next + (BUF_GET3D(Q.F, r, c, s, 5, i+Xi[2], j+Xj[2], k+Xk[2]) / (P.lato * P.lato)) * P.delta_t * temp_value;

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
// main() function
// ----------------------------------------------------------------------------
int main(int argc, char **argv)
{
  int r = atoi(argv[ROWS_ID]);
  int c = atoi(argv[COLS_ID]);
  int s = atoi(argv[LAYERS_ID]);
  char* input_ks_path = argv[INPUT_KS_ID];
  double simulation_time = atoi(argv[SIMUALITION_TIME_ID]);
  char* output_prefix = argv[OUTPUT_PREFIX_ID];

  Substates Q;
  Parameters P;
  DomainBoundaries mb_bounds;

  allocSubstates(Q, r, c, s);
  readKs(Q.ks, r, c, s, input_ks_path);
  initParameters(P, simulation_time, r, c, s); 
  initDomainBoundaries(mb_bounds, 1, r-1, 1, c-1, 0, s);

  // Apply the simulation init kernel to the whole domain
#pragma omp parallel for
    for (int i = 0; i < r; i++)
      for (int j = 0; j < c; j++)
        for (int k = 0; k <s; k++)
        simulation_init(i, j, k, Q, r, c, s, P);
  updateSubstates(Q, r, c, s);

  // simulation loop
  util::Timer cl_timer;
  while(!(P.delta_t_cum >= P.simulation_time && P.delta_t_cum_prec <= P.simulation_time))
  {
    // Apply the reset flow kernel to the whole domain
#pragma omp parallel for
    for (int i = 0; i < r; i++)
      for (int j = 0; j < c; j++)
        for (int k = 0; k <s; k++)
          reset_flows(i, j, k, Q, r, c, s);

    // Apply the flow computation kernel to the whole domain
#pragma omp parallel for
    for (int i = 0; i < r; i++)
      for (int j = 0; j < c; j++)
        for (int k = 0; k <s; k++)
          compute_flows(i, j, k, Q, r, c, s, P);

    // Apply the mass balance kernel to the domain bounded by mb_bounds 
#pragma omp parallel for
    for (int i = mb_bounds.i_start; i < mb_bounds.i_end; i++)
      for (int j = mb_bounds.j_start; j < mb_bounds.j_end; j++)
        for (int k = mb_bounds.k_start; k < mb_bounds.k_end; k++)
          mass_balance(i, j, k, Q, r, c, s, P);
    updateSubstates(Q, r, c, s);

    // Simulation Steering
    double minVar = GET3D(Q.convergence, r, c, 0, 0, 0);
    int i, j, k;
#pragma omp parallel for reduction(min : minVar) 
    for (i = 0; i < r; i++)
      for (j = 0; j < c; j++)
        for (k = 0; k <s; k++)
        {
          double tmpmin = GET3D(Q.convergence, r, c, i, j, k);
          if (minVar > tmpmin)
            minVar = tmpmin;
        }
    if (minVar > 55.0)
      minVar = 55.0;

    P.delta_t = minVar;
    P.delta_t_cum_prec = P.delta_t_cum;
    P.delta_t_cum += P.delta_t;
  }

  double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
  //printf("Elapsed time: %lf [s]\n", cl_time);
  printf("%lf\n", cl_time);

  std::string s_path = (std::string)output_prefix + "h_LAST_simulation_time_" + util::converttostringint(simulation_time) + "s.txt";
  saveFile(Q.h, r, c, s, s_path);

  //printf("Releasing memory...\n");
  deleteSubstates(Q);

  return 0;
}
