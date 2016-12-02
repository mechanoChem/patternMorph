/*Transport physics model for second species. 
  Implementation is via the chemical potential. Flux is defined in terms of chem pot.
  Chem pot is related to concentration in a separate pde.
  Created by Shiva Rudraraju, modified by Krishna Garikipati*/
#ifndef CHEMO2_H_
#define CHEMO2_H_
#include <deal.II/dofs/dof_handler.h>
#include "functionEvaluations1.h"
#include "supplementaryFunctions1.h"
#include "parameters.h"

//Homogeneous chem potential
template <class T, int dim>
  void evaluateHomChemPot2(const FEValues<dim>& fe_values, dealii::Table<1, Sacado::Fad::DFad<double> >& c1, dealii::Table<1, Sacado::Fad::DFad<double> >& c2, Table<1, T>& fhomPrime)
{
  unsigned int n_q_points = fe_values.n_quadrature_points;
  for (unsigned int q = 0; q < n_q_points; ++q)
    {
      fhomPrime[q] = 0.0;
      //      fhomPrime[q]+= 2.0*omega2*(c2[q] - 0.05)*(c2[q] - 0.95)*(2.0*c2[q] - 1.0); //Double well
      //      fhomPrime[q]+= 2.0*omega2*(c2[q] - 0.05)*(c2[q] - 0.5)*(c2[q] - 0.95)*(3.0*std::pow(c2[q],2)-3.0*c2[q] + 0.05*0.95 + 0.05*0.5 + 0.5*0.95); //Triple well, single field
      fhomPrime[q]+= (6.0*dc/std::pow(sc,4))*(std::pow(c2[q],3) + c2[q]*pow(c1[q],2)) + 3.0*(dc/std::pow(sc,3))*(std::pow(c2[q],2) - std::pow(c1[q],2)) - 3.0*(dc/std::pow(sc,2))*c2[q];//Triple well, two field
      //      fhomPrime[q]+= 2.0*omega2*(c2[q] - 0.5)*(c2[q] - 0.5)*(2.0*c2[q] - 1.0); //Single well
      //      fhomPrime[q]+= c2[q]; //Fickian
      //      fhomPrime[q]+= kT*std::log(c2[q]/(1.0 - c2[q])); //Mixing entropy from Stirling approximation
      fhomPrime[q]+= 0.25*kT*std::pow((c2[q] - 0.5),3); //Polynomial approximation of logarithmic mixing entropy
      /*      if (c2[q].val() > 1.0 || c2[q].val() < 0.0)
	{
	  printf("**************Composition lies outside (0,1)**************. Value: %12.4e\n", c2[q].val());
	  printf("\n"); exit(-1);
	}
      else if (c2[q].val() == 1.0 || c2[q].val() == 0.0)
	{
	  fhomPrime[q] = 0.0;
	}
      else
	{
	  //	  fhomPrime[q] = 2.0*omega2*c2[q]*(1 - c2[q])*(1.0 - 2.0*c2[q]);
	  //	  fhomPrime[q]+= kT*std::log(c2[q]/(1.0 - c2[q]));
	  //	  fhomPrime[q]+= kT*(c2[q]/(1.0-c2[q]) - std::pow(c2[q]/(1.0-c2[q]),2)/2.0 + std::pow(c2[q]/(1.0-c2[q]),3)/6.0 - std::pow(c2[q]/(1.0-c2[q]),4)/24.0 + std::pow(c2[q]/(1.0-c2[q]),5)/120.0);
	  fhomPrime[q] = 2.0*omega2*(c2[q] - 0.05)*(c2[q] - 0.95)*(2.0*c2[q] - 1.0);
	  fhomPrime[q]+= 0.25*kT*std::pow((c2[q] - 0.5),3);
	  }*/
    }
}

//Transport equations
template <int dim>
void residualForChemo2(const FEValues<dim>& fe_values, unsigned int DOF, FEFaceValues<dim>& fe_face_values, const typename hp::DoFHandler<dim>::active_cell_iterator &cell, double dt, dealii::Table<1, Sacado::Fad::DFad<double> >& ULocal, dealii::Table<1, double>& ULocalConv, dealii::Table<1, Sacado::Fad::DFad<double> >& R, deformationMap<Sacado::Fad::DFad<double>, dim>& defMap, dealii::Table<1,double>& mu2_conv, dealii::Table<1, Sacado::Fad::DFad<double> >& mu2, dealii::Table<2, Sacado::Fad::DFad<double> >& gradmu2, dealii::Table<1,double>& c2_conv, dealii::Table<1, Sacado::Fad::DFad<double> >& c1, dealii::Table<1, Sacado::Fad::DFad<double> >& c2, dealii::Table<2, Sacado::Fad::DFad<double> >& gradc2){

  unsigned int dofs_per_cell= fe_values.dofs_per_cell;
  unsigned int n_q_points= fe_values.n_quadrature_points;

  // Mobility and gradient parameter
  Table<1, Sacado::Fad::DFad<double> > mob(n_q_points); 
  for (unsigned int q = 0; q < n_q_points; ++q)
    {mob[q] = 0.1;}

  // Advection velocity
  Table<2, Sacado::Fad::DFad<double> > vel(n_q_points, dim);
  for (unsigned int q = 0; q < n_q_points; ++q)
    {
      const Point<dim> posR = fe_values.quadrature_point(q);
      for (unsigned int j = 0; j < dim; ++j)
	{
	  vel[q][j] = alpha2*posR[j]/std::sqrt(posR.square());//Radial velocity
	  //	  vel[q][j] = j == 0;//Unit velocity along x
	  /*	  if (j == 0)
	    vel[q][j] = 1.0;
	  else
	  vel[q][j] = 0.0;*/
	}
    }

  // Reaction
  Table<1, Sacado::Fad::DFad<double> > reac1(n_q_points);
  Table<1, Sacado::Fad::DFad<double> > reac2(n_q_points);
  for (unsigned int q = 0; q < n_q_points; ++q){
    reac1[q] = 0.0;
    reac2[q] = 0.0;} //reac[q] = 0.01;

  //Threshold concentration for reaction
  Table<1, Sacado::Fad::DFad<double> > c1_reac(n_q_points);
  Table<1, Sacado::Fad::DFad<double> > c2_reac(n_q_points);
  for (unsigned int q = 0; q < n_q_points; ++q){
    c1_reac[q] = 0.5;
    c2_reac[q] = 0.5;}
  
  //Spatial gradients
  Table<3, Sacado::Fad::DFad<double> > shapeGradSpat(n_q_points, dofs_per_cell, dim); 
  for (unsigned int q = 0; q < n_q_points; ++q)
    for (unsigned int i =0; i < dofs_per_cell; ++i)
      for (unsigned int j = 0; j < dim; ++j)
	{shapeGradSpat[q][i][j] = 0.0;}
  Table<2, Sacado::Fad::DFad<double> > gradSpatmu2(n_q_points, dim);
  for (unsigned int q = 0; q < n_q_points; ++q)
    for (unsigned int j = 0; j < dim; ++j)
      {gradSpatmu2[q][j] = 0.0;}
  Table<2, Sacado::Fad::DFad<double> > gradSpatc2(n_q_points, dim); 
  for (unsigned int q = 0; q < n_q_points; ++q)
    for (unsigned int j = 0; j < dim; ++j)
      {gradSpatc2[q][j] = 0.0;}
  for (unsigned int q=0; q<n_q_points; ++q)
    {
      for (unsigned int j=0; j<dim; ++j)
	{
	  for (unsigned int k=0; k<dim; ++k)
	    {
	      gradSpatmu2[q][j] += gradmu2[q][k]*defMap.invF[q][k][j];
	      gradSpatc2[q][j] += gradc2[q][k]*defMap.invF[q][k][j];
	      for (unsigned int i=0; i<dofs_per_cell; ++i)
		{
		  const int ck = fe_values.get_fe().system_to_component_index(i).first - DOF;
		  if (ck == 2 || ck == 3)
		    {
		      shapeGradSpat[q][i][j] += fe_values.shape_grad(i, q)[k]*defMap.invF[q][k][j];
		    }
		}
	    }
	}
    }

  //velocity dot gradC2
  Table<1, Sacado::Fad::DFad<double> > velDotGradSpatc2(n_q_points); 
  for (unsigned int q = 0; q < n_q_points; ++q)
    {velDotGradSpatc2[q] = 0.0;
      for (unsigned int j =0; j < dim; ++j)
	{velDotGradSpatc2[q] = vel[q][j]*gradSpatc2[q][j];}
    }


  //Homogeneous chem potential
  Table<1,Sacado::Fad::DFad<double> > fhomPrime (n_q_points);
  evaluateHomChemPot2<Sacado::Fad::DFad<double>, dim>(fe_values, c1, c2, fhomPrime);
  
  //evaluate Residual
  for (unsigned int i=0; i<dofs_per_cell; ++i) 
    {
      const int ck = fe_values.get_fe().system_to_component_index(i).first - DOF;
      for (unsigned int q=0; q<n_q_points; ++q)
	{
	  if (ck==2)
	    {
	      R[i] +=  fe_values.shape_value(i, q)*(c2[q]-c2_conv[q])*defMap.detF[q]*fe_values.JxW(q);//time-dependent term
	      R[i] += -fe_values.shape_value(i, q)*dt*reac1[q]*(c1[q]-c1_reac[q])*defMap.detF[q]*fe_values.JxW(q);//reaction (source) term for c1
	      R[i] += -fe_values.shape_value(i, q)*dt*reac2[q]*(c2[q]-c2_reac[q])*defMap.detF[q]*fe_values.JxW(q);//reaction (source) term for c2
	      //	      R[i] += -fe_values.shape_value(i, q)*dt*reac0*defMap.detF[q]*fe_values.JxW(q);//const reaction (source) term 
	      //	      R[i] += -fe_values.shape_value(i, q)*dt*reac1[q]*(c1[q]-c1_reac[q])*defMap.detF[q]*fe_values.JxW(q);//reaction (source) term from c1
	      //	      R[i] += -fe_values.shape_value(i, q)*dt*reac2[q]*std::pow((c1[q]-c1_reac[q]),2)*(c2[q]-c2_reac[q])*defMap.detF[q]*fe_values.JxW(q);//reaction (source) term from c1^2*c2
	      for (unsigned int j = 0; j < dim; ++j)
		{
		  		  R[i] +=  dt*shapeGradSpat[q][i][j]*mob[q]*gradSpatmu2[q][j]*defMap.detF[q]*fe_values.JxW(q);//diffusive flux
		  		  R[i] -=  dt*shapeGradSpat[q][i][j]*c2[q]*vel[q][j]*defMap.detF[q]*fe_values.JxW(q);//advective flux
		  		  R[i] +=  tau2*shapeGradSpat[q][i][j]*vel[q][j]*(c2[q]-c2_conv[q])*defMap.detF[q]*fe_values.JxW(q);//stabilization term on time derivative
		  /*stabilization term on advective flux*/
		  		  R[i] +=  tau2*dt*shapeGradSpat[q][i][j]*vel[q][j]*velDotGradSpatc2[q]*defMap.detF[q]*fe_values.JxW(q);
		  		  R[i] += -tau2*shapeGradSpat[q][i][j]*vel[q][j]*dt*reac1[q]*(c1[q]-c1_reac[q])*defMap.detF[q]*fe_values.JxW(q);//stabilization term on source for c1
		  		  R[i] += -tau2*shapeGradSpat[q][i][j]*vel[q][j]*dt*reac2[q]*(c2[q]-c2_reac[q])*defMap.detF[q]*fe_values.JxW(q);//stabilization term on source for c2
		}
	    }
	  else if (ck==3)
	    {
	      R[i] +=  fe_values.shape_value(i, q)*(mu2[q]-fhomPrime[q])*defMap.detF[q]*fe_values.JxW(q);//mu21 - fprime_hom
	      for (unsigned int j = 0; j < dim; ++j)
		{
		  R[i] -= shapeGradSpat[q][i][j]*mob[q]*kappa2*gradSpatc2[q][j]*defMap.detF[q]*fe_values.JxW(q);//Laplacian or Hessian of conc c21, after int by parts
		}
	    }
	}
    }

  //Neumann boundary conditions
  double jn= 0.0;//0.01;//influx defined per unit reference area, by exploiting configuration-independent mass supply and Piola transform
  double gradC2N = 0.0;//normal gradient of c2 at boundary
  for (unsigned int faceID=0; faceID<2*dim; faceID++)
    {
      if (cell->face(faceID)->at_boundary())
	{
	  const Point<dim> face_center = cell->face(faceID)->center();
	  //	  if (face_center.distance(center) >= 0.99*outer_radius)
	  if (std::abs(face_center[2]) >= 0.8*clen && std::abs(face_center[2]) < 0.99*clen)
	    { //influx on outer boundary or on x-y faces of omega1 domain
	      fe_face_values.reinit (cell, faceID);
	      for (unsigned int i=0; i<dofs_per_cell; ++i) 
		{
		  const int ck = fe_values.get_fe().system_to_component_index(i).first - DOF;
		  if (ck==2)
		    {
		      for (unsigned int q=0; q<fe_face_values.n_quadrature_points; ++q)
			{
			  R[i] += -dt*fe_face_values.shape_value(i, q)*jn*fe_face_values.JxW(q); 
			}
		    }
		  else if (ck==3)
		    {
		      for (unsigned int q=0; q<fe_face_values.n_quadrature_points; ++q)
			{
			  R[i] += fe_face_values.shape_value(i, q)*kappa2*gradC2N*fe_face_values.JxW(q); 
			}
		    }
		}
	    }
	}
    }
}

#endif /* CHEMO2_H_ */
