/* Boundary Conditions for the coupled transport-mechanics code
   Written by Krishna Garikipati
*/
#ifndef boundary_conds_
#define boundary_conds_
#include "parameters.h"
//
using namespace dealii;

//Dirichlet BC
template <int dim>
class BoundaryConditions :  public Function<dim>{
  double time;
public:
 BoundaryConditions(double currentTime): Function<dim> (totalDOF), time(currentTime) {}
  void vector_value (const Point<dim>   &p, Vector<double>   &values) const{
    Assert (values.size() == totalDOF, ExcDimensionMismatch (values.size(), totalDOF));
    values(totalDOF-7) = 0.0;
    values(totalDOF-6) = 0.0;
    values(totalDOF-5) = 0.0;
    if (time == 0.0)
      {
	values(totalDOF-4) = 0.0;
	values(totalDOF-2) = 0.0;
      }
    else
      {
	values(totalDOF-4) = 0.0;
	values(totalDOF-2) = 0.0;
      }
    values(totalDOF-3) = 0.0;
    values(totalDOF-1) = 0.0;
  }
};

#endif
