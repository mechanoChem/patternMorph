#ifndef parameters_
#define parameters_

#define DIMS 3
#define totalDOF 7
#define alen 40.0
#define blen 40.0
#define clen 2.0
#define restart_step 0
#define restart_time 0.0
double youngsModulus = 5.0e3, poissonRatio = 0.30;
double kappa1 = 1.0, omega1 = 1.0, kT = 0.0, alpha1 = 0.0, tau1 = 1.0;
double kappa2 = 1.0, omega2 = 1.0, alpha2 = 0.0, tau2 = 1.0;
double dc = 0.4, sc = 0.7;
const double inner_radius = 10.0, outer_radius = 20.0;
const Point<3> center (alen/2.0,blen/2.0,clen/2.0);
unsigned int n_cells;
#endif
