/*
 * projectFields.h
 *  Created on: March 5, 2015, K. Garikipati.
 */

#ifndef project_Fields1_
#define project_Fields1_
#include "mechanics2.h"
#include "initialConditions1.h"
#include "parameters.h"
//
using namespace dealii;

template <int dim>
class ComputeManyProjections : public DataPostprocessor<dim>
{
 public:
  ComputeManyProjections ();
  virtual
  void
    compute_derived_quantities_vector (const std::vector<Vector<double> >              &uh,
				       const std::vector<std::vector<Tensor<1,dim> > > &duh,
				       const std::vector<std::vector<Tensor<2,dim> > > &dduh,
				       const std::vector<Point<dim> >                  &normals,
				       const std::vector<Point<dim> >                  &evaluation_points,
				       std::vector<Vector<double> >                    &computed_quantities) const;
  virtual std::vector<std::string> get_names () const;
  virtual
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation () const;
  virtual UpdateFlags get_needed_update_flags () const;
  // Initial conditions for projection. Used in defining Fe
  InitialConditions<dim> init_conds;

};

//Constructor
template <int dim>
ComputeManyProjections<dim>::ComputeManyProjections ()
{}

template <int dim>
std::vector<std::string>
ComputeManyProjections<dim>::get_names() const
{
  std::vector<std::string> solution_names;
  solution_names.push_back ("traceF");
  solution_names.push_back ("detF");
  solution_names.push_back ("p");
  return solution_names;
}

template <int dim>
std::vector<DataComponentInterpretation::DataComponentInterpretation>
ComputeManyProjections<dim>::get_data_component_interpretation () const
{
  std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation;
    //    interpretation (dim,DataComponentInterpretation::component_is_part_of_vector);
  interpretation.push_back (DataComponentInterpretation::component_is_scalar);
  interpretation.push_back (DataComponentInterpretation::component_is_scalar);
  interpretation.push_back (DataComponentInterpretation::component_is_scalar);
  return interpretation;
}

template <int dim>
UpdateFlags
ComputeManyProjections<dim>::get_needed_update_flags() const
{
  return update_values | update_gradients | update_q_points;
}

template <int dim>
void
ComputeManyProjections<dim>::compute_derived_quantities_vector (
				   const std::vector<Vector<double> >              &uh,
                                   const std::vector<std::vector<Tensor<1,dim> > > &duh,
                                   const std::vector<std::vector<Tensor<2,dim> > > &dduh,
                                   const std::vector<Point<dim> >                  &normals,
                                   const std::vector<Point<dim> >                  &evaluation_points,
                                   std::vector<Vector<double> >                    &computed_quantities) const
{ 
  const unsigned int n_quadrature_points = uh.size();
  Assert (duh.size() == n_quadrature_points,                  ExcInternalError());
  Assert (computed_quantities.size() == n_quadrature_points,  ExcInternalError());
  Assert (uh[0].size() == totalDOF,                           ExcInternalError());
  for (unsigned int q=0; q<n_quadrature_points; ++q)
    {
      Table<2,double> F(dim,dim), Fe(dim,dim), Ce(dim,dim), invF(dim,dim), invCe(dim,dim), S(dim,dim), Sigma(dim,dim);
      for (unsigned int i = 0; i < dim; ++i){
	for (unsigned int j = 0; j < dim; ++j){
	  F[i][j] = (i==j) + duh[q][i][j];
	}
      }
      double detF;
      getInverse<double, dim>(F, invF, detF);
      computed_quantities[q](0) = F[0][0] + F[1][1] + F[2][2];
      computed_quantities[q](1) = detF;
      //      computed_quantities[q](2) = 2*std::log(detF)/detF;      
 
      Vector<double> initvalues(totalDOF);
      init_conds.vector_value(evaluation_points[q], initvalues);
      for (unsigned int i = 0; i < dim; ++i){
	for (unsigned int j = 0; j < dim; ++j){
	  Fe[i][j] = F[i][j]/std::pow((uh[q][dim+1]/initvalues[dim+1]),1/3);
	}
      }
      for (unsigned int i = 0; i < dim; ++i)
	for (unsigned int j = 0; j < dim; ++j)
	  {Ce[i][j] = 0.0;}
      for (unsigned int i=0; i<dim; ++i){
	for (unsigned int j=0; j<dim; ++j){
	  for (unsigned int k=0; k<dim; ++k){
	    Ce[i][j] += Fe[k][i]*Fe[k][j];
	  }
	}
      }
      double lambda=(youngsModulus*poissonRatio)/((1+poissonRatio)*(1-2*poissonRatio)), mu=youngsModulus/(2*(1+poissonRatio));// Lame' parameters
      double detCe = 0.0;
      for (unsigned int i = 0; i < dim; ++i)
	for (unsigned int j = 0; j < dim; ++j)
	  {invCe[i][j] = 0.0;}
      getInverse<double, dim>(Ce, invCe, detCe);
      for (unsigned int i = 0; i < dim; ++i){
	for (unsigned int j = 0; j < dim; ++j){
	  S[i][j] += 0.5*lambda*detCe*invCe[i][j] - (0.5*lambda + mu)*invCe[i][j] + mu*(i==j);
	}
      }
      for (unsigned int i = 0; i < dim; ++i){
	for (unsigned int j = 0; j < dim; ++j){
	  Sigma[i][j] = 0.0;
	  for (unsigned int k = 0; k < dim; ++k){
	    for (unsigned int l = 0; l < dim; ++l){
	      Sigma[i][j] += Fe[i][k]*S[k][l]*Fe[l][j]/std::pow(detCe,0.5);
	    }
	  }
	}
      }
      computed_quantities[q](2) = Sigma[0][0] + Sigma[1][1] + Sigma[2][2];      
    }     
}

#endif /* project_Fields1_ */
