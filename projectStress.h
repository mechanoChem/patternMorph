/*
 * projectStress.h
 *  Created on: March 4, 2015. S. Rudraraju, K. Garikipati.
 */

#ifndef project_Stress_
#define project_Stress_
#include "mechanics2.h"

template <int dim>
class ComputeProjection : public DataPostprocessorScalar<dim>
{
 public:
  ComputeProjection ();
  virtual
  void
    compute_derived_quantities_vector (const std::vector< Vector< double > > &uh,
				       const std::vector< std::vector< Tensor< 1, dim > > > &duh,
				       const std::vector< std::vector< Tensor< 2, dim > > > &dduh,
				       const std::vector< Point< dim > > &normals,
				       const std::vector<Point<dim> > &evaluation_points,
				       std::vector< Vector< double > > &computed_quantities) const;
};

//Constructor
template <int dim>
ComputeProjection<dim>::ComputeProjection ()
:
DataPostprocessorScalar<dim> ("Stress",
			      update_gradients)
{}

template <int dim>
void
ComputeProjection<dim>::compute_derived_quantities_vector (
							  const std::vector< Vector< double > >                  &uh,
							  const std::vector< std::vector< Tensor< 1, dim > > >  &duh,
							  const std::vector< std::vector< Tensor< 2, dim > > >  &dduh,
							  const std::vector< Point< dim > >                     &normals,
							  const std::vector<Point<dim> >                        &evaluation_points,
							  std::vector< Vector< double > >                       &computed_quantities
							  ) const
{
  Assert(computed_quantities.size() == duh.size(),
         ExcDimensionMismatch (computed_quantities.size(), duh.size()));
  for (unsigned int i=0; i<computed_quantities.size(); i++)
    {
      Assert(computed_quantities[i].size() == 1,
             ExcDimensionMismatch (computed_quantities[i].size(), 1));
      Assert(duh[i].size() == dim + 2, ExcDimensionMismatch (duh[i].size(), dim + 2));
      Table<2, double> F(dim,dim);
      computed_quantities[i](0) = 3 + duh[i][0][0] + duh[i][1][1] + duh[i][2][2];
    }
}

#endif /* project_Stress_ */
