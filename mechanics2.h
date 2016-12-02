/*
  Elasto-growth code. Uses a single species for inhomogeneous growth.
  Written by Krishna Garikipati and Shiva Rudraraju
 */

#ifndef MECHANICS_H_
#define MECHANICS_H_
#include <deal.II/dofs/dof_handler.h>
#include "functionEvaluations1.h"
#include "supplementaryFunctions1.h"
#include "parameters.h"

//Saint-Venant Kirchhoff constitutive model
template <int dim>
inline double SVK3D(unsigned int i, unsigned int j, unsigned int k, unsigned int l, double E){
  double nu=0.3;
  double lambda=(E*nu)/((1+nu)*(1-2*nu)), mu=E/(2*(1+nu));
  return lambda*(i==j)*(k==l) + mu*((i==k)*(j==l) + (i==l)*(j==k));
}

template <int dim>
inline double SVK2D(unsigned int i, unsigned int j, double E){
  double nu=0.3;
  double lambda=(E*nu)/((1+nu)*(1-2*nu)), mu=E/(2*(1+nu));
  if (i==j && i<2) return lambda + 2*mu;
  else if (i==2 && j==2) return mu;
  else if ((i+j)==1) return lambda;
  else return 0.0;
}

template <int dim>
inline double SVK1D(double E){
  return E;
}

//Compressible neo-Hookean constitutive model
template <class T, int dim>
  void neoHookean(double& youngsModulusContrast, Table<2, T>& C, Table<2, T>& S, Table<4, T>& CTang, T& engy){
  double lambda=(youngsModulusContrast*poissonRatio)/((1+poissonRatio)*(1-2*poissonRatio)), mu=youngsModulusContrast/(2*(1+poissonRatio));// Lame' parameters
  T detC = 0.0, logdetC = 0.0;
  Table<2, T> invC(dim, dim); 
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      {invC[i][j] = 0.0;}
  getInverse<T, dim>(C, invC, detC);
  logdetC = std::log(detC);
  for (unsigned int i = 0; i < dim; ++i){
    for (unsigned int j = 0; j < dim; ++j){
      S[i][j] += 0.5*lambda*detC*invC[i][j] - (0.5*lambda + mu)*invC[i][j] + mu*(i==j);
      for (unsigned int k = 0; k < dim; ++k){
	for (unsigned int l = 0; l< dim; ++l){
	  CTang[i][j][k][l] += lambda*detC*invC[i][j]*invC[k][l] + (2.0*mu + (1.0 - detC)*lambda)*0.5*(invC[i][k]*invC[j][l] + invC[i][l]*invC[j][k]);
	}
      }
    }
  }
  engy = 0.25*lambda*(detC - 1.0) - 0.5*(0.5*lambda + mu)*logdetC + 0.5*mu*(C[0][0] + C[1][1] + C[2][2] - 3.0);
}

//Mechanics implementation
template <class T, int dim>
  void getDeformationMap(const FEValues<dim>& fe_values, unsigned int DOF, Table<1, T>& ULocal, deformationMap<T, dim>& defMap, const unsigned int iteration){
  unsigned int n_q_points= fe_values.n_quadrature_points;
  //evaluate dx/dX
  Table<3, T> gradU(n_q_points, dim, dim);
  evaluateVectorFunctionGradient<T, dim>(fe_values, DOF, ULocal, gradU);

  //Loop over quadrature points
  for (unsigned int q=0; q<n_q_points; ++q){
    Table<2, T > Fq(dim, dim), invFq(dim, dim); T detFq;
    for (unsigned int i=0; i<dim; ++i){
      for (unsigned int j=0; j<dim; ++j){
	defMap.F[q][i][j] = Fq[i][j] = (i==j) + gradU[q][i][j]; //F (as double value)
      }
    }
    getInverse<T, dim>(Fq, invFq, detFq); //get inverse(F)
    defMap.detF[q] = detFq;
    for (unsigned int i=0; i<dim; ++i){
      for (unsigned int j=0; j<dim; ++j){
	defMap.invF[q][i][j] = invFq[i][j];
      }
    }
    //detF
    if (defMap.detF[q].val()<=1.0e-15 && iteration==0){
      printf("**************Non positive jacobian detected**************. Value: %12.4e\n", defMap.detF[q].val());
      for (unsigned int i=0; i<dim; ++i){
	for (unsigned int j=0; j<dim; ++j) printf("%12.6e  ", defMap.F[q][i][j].val());
      }
      printf("\n"); exit(-1);
      //throw "Non positive jacobian detected";
    }
  }
}

//Mechanics implementation
template <class T, int dim>
  void evaluateStress(const FEValues<dim>& fe_values,const unsigned int DOF, const Table<1, T>& ULocal, Table<3, T>& P, const deformationMap<T, dim>& defMap, typename hp::DoFHandler<dim>::active_cell_iterator& cell, dealii::Table<1,double>& c_conv, dealii::Table<1, Sacado::Fad::DFad<double> >& c, dealii::Table<1, Sacado::Fad::DFad<double> >& c_0){
  unsigned int n_q_points= fe_values.n_quadrature_points;

  // Stiffer cortical layer with lower saturation 
  double youngsModulusContrast = 0.0, sat = 0.0;
  const Point<dim> cellcenter = cell->center();
  if (cellcenter.distance(center) >= 0.1*inner_radius + 0.9*outer_radius)
    {
      youngsModulusContrast = youngsModulus*2.0;
      sat = 1.0;
    }
  else
    {
      youngsModulusContrast = youngsModulus;    
      sat = 10.0;
    }  
  dealii::Table<1,Sacado::Fad::DFad<double> > fac(n_q_points);

  //Loop over quadrature points
  for (unsigned int q=0; q<n_q_points; ++q){
    if (c_0[q] > 0) 
      {//fac[q] = 1.0;
	//	fac[q]=std::pow((c[q]/c_0[q]), 1.0/3.0); //Isotropic
	fac[q]=std::pow((c[q]/c_0[q]), 1.0/2.0); //Tangential
      }
    else
      {fac[q]=1.0;
      }
    if (fac[q] <= 1.0e-15)
      {
	printf("*************Non positive growth factor*************. Value %12.4e\n", fac[q].val());
      }
    //Scale swelling by saturation function after crossing saturation threshold
    if (fac[q] < sat)
      fac[q] = 1.0;
    else
      fac[q] /= sat;    

    Table<2, Sacado::Fad::DFad<double> > Fe (dim, dim);
    //Fe, assuming that Fg = fac*Istropic. Change if different
    /*    for (unsigned int i=0; i<dim; ++i){
      for (unsigned int j=0; j<dim; ++j){
		  Fe[i][j]=defMap.F[q][i][j]/fac[q];
      }
      } */ 
    
    //Fg is tengential
    const Point<dim> posR = fe_values.quadrature_point(q);
    Table<1, double> eR(dim);
    for (unsigned int j = 0; j<dim; ++j){
    eR[j] = posR[j]/std::sqrt(posR.square());
    }
    Table<1, Sacado::Fad::DFad<double> > FeR(dim);
    for (unsigned int i=0; i<dim; ++i){
      FeR[i] = 0.0;
      for (unsigned int j=0; j<dim; ++j){
		  FeR[i]+=defMap.F[q][i][j]*eR[j];
      }
    }  
    for (unsigned int i=0; i<dim; ++i){
      for (unsigned int j=0; j<dim; ++j){
	Fe[i][j]=defMap.F[q][i][j]/fac[q] + (1.0 - 1.0/fac[q])*FeR[i]*eR[j];
      }
    }  

    //Ee, Ce
    Table<2, Sacado::Fad::DFad<double> > Ee (dim, dim);
    Table<2, Sacado::Fad::DFad<double> > Ce (dim, dim);
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
	{Ce[i][j] = 0.0;}
    for (unsigned int i=0; i<dim; ++i){
      for (unsigned int j=0; j<dim; ++j){
	Ee[i][j] = -0.5*(i==j);
	for (unsigned int k=0; k<dim; ++k){
	  Ee[i][j] += 0.5*Fe[k][i]*Fe[k][j];
	  Ce[i][j] += Fe[k][i]*Fe[k][j];
	}
      }
    }
    
    //determine second Piola-Kirchhoff stress tensor, S = 2dpsi/dCe and material tangent C = 4d^2 psi/dCedCe
    Table<2, Sacado::Fad::DFad<double> > S(dim, dim); 
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
	{S[i][j] = 0.0;}
    Table<4, Sacado::Fad::DFad<double> > CTang(dim, dim, dim, dim); 
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
	for (unsigned int k = 0; k < dim; ++k)
	  for (unsigned int l = 0; l < dim; ++l)	    
	    {CTang[i][j][k][l] = 0.0;}
    T stengy = 0.0; 
    neoHookean<Sacado::Fad::DFad<double>, dim>(youngsModulusContrast, Ce, S, CTang, stengy);
       
    //P
    for (unsigned int i=0; i<dim; ++i){
      for (unsigned int j=0; j<dim; ++j){
	P[q][i][j]=0;
	for (unsigned int k=0; k<dim; ++k){
	  P[q][i][j]+=Fe[i][k]*S[k][j];
	}
      }
    }
    Table<2, Sacado::Fad::DFad<double> > Sigma (dim, dim);
    //Sigma
    for (unsigned int i=0; i<dim; ++i){
      for (unsigned int j=0; j<dim; ++j){
	Sigma[i][j]=0;
	for (unsigned int k=0; k<dim; ++k){
	  Sigma[i][j]+=P[q][i][k]*defMap.F[q][j][k];
	}
	Sigma[i][j]/=defMap.detF[q];
      }
    }
    //strain[cell][q]=E[0][0].val()+E[1][1].val(); //trace(E)
    //    strain[cell][q]= defMap.detF[q].val();
  }
}

//Mechanics residual implementation
template <int dim>
void residualForMechanics(const FEValues<dim>& fe_values, unsigned int DOF, FEFaceValues<dim>& fe_face_values, Table<1, Sacado::Fad::DFad<double> >& ULocal, Table<1, double>& ULocalConv, Table<1, Sacado::Fad::DFad<double> >& R, deformationMap<Sacado::Fad::DFad<double>, dim>& defMap, typename hp::DoFHandler<dim>::active_cell_iterator& cell, dealii::Table<1,double>& c_conv, dealii::Table<1, Sacado::Fad::DFad<double> >& c, dealii::Table<1, Sacado::Fad::DFad<double> >& c_0, double currentTime, double totalTime){
  unsigned int dofs_per_cell= fe_values.dofs_per_cell;
  unsigned int n_q_points= fe_values.n_quadrature_points;
  //Temporary arrays
  Table<3,Sacado::Fad::DFad<double> > P (n_q_points, dim, dim);
  //evaluate mechanics=====added 
  //  evaluateStress<Sacado::Fad::DFad<double>, dim>(fe_values, DOF, ULocal, P, defMap, youngsModulus, strain, cell, c_conv, c, c_0);
  evaluateStress<Sacado::Fad::DFad<double>, dim>(fe_values, DOF, ULocal, P, defMap, cell, c_conv, c, c_0);

  //evaluate Residual
  for (unsigned int i=0; i<dofs_per_cell; ++i) {
    const unsigned int ck = fe_values.get_fe().system_to_component_index(i).first - DOF;
    if (ck<dim){
      // R = Grad(w)*P
      for (unsigned int q=0; q<n_q_points; ++q){
	for (unsigned int d = 0; d < dim; d++){
	  R[i] +=  fe_values.shape_grad(i, q)[d]*P[q][ck][d]*fe_values.JxW(q);
	}
      }
    }
  }

 /*/input Force boundary condition
  double t=currentTime/totalTime;
  double force= 15; //10N 
	
  //Neumann conditions 
//Flux like tracking bounday coding
  for (unsigned int faceID=0; faceID<2*dim; faceID++){
    if (cell->face(faceID)->at_boundary()){
      const Point<dim> face_center = cell->face(faceID)->center();
      if (face_center[1] == 1.0){ //flux on Y=1 boundary
	fe_face_values.reinit (cell, faceID);
	for (unsigned int i=0; i<dofs_per_cell; ++i) {
	  const unsigned int ck = fe_values.get_fe().system_to_component_index(i).first - DOF;
	  if (ck==1){
	    for (unsigned int q=0; q<fe_face_values.n_quadrature_points; ++q){
	      R[i] += -fe_face_values.shape_value(i, q)*force*fe_face_values.JxW(q); 
	    }
	  }
	}
      }
      else if (face_center[1] == 0.0){ //flux on Y=0 boundary
	fe_face_values.reinit (cell, faceID);
	for (unsigned int i=0; i<dofs_per_cell; ++i) {
	  const unsigned int ck = fe_values.get_fe().system_to_component_index(i).first - DOF;
	  if (ck==1){
	    for (unsigned int q=0; q<fe_face_values.n_quadrature_points; ++q){
	      R[i] += fe_face_values.shape_value(i, q)*force*fe_face_values.JxW(q); 
	    }
	  }
	}
      }
    }
  }/*/
}

#endif /* MECHANICS_H_ */
