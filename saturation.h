/* Concentration saturation function for the coupled transport-mechanics code
   Written by Krishna Garikipati
*/
#ifndef saturation_
#define saturation_
#include "parameters.h"
#include <cstdlib>
//
using namespace dealii;

//Saturation function
template <int dim>
class SaturationFunction: public Function<dim>{
public:
  Saturation (): Function<dim>(){}
  void vector_value (const Point<dim>   &p, double & value) const{
    value = 10;
  }
};

#endif
