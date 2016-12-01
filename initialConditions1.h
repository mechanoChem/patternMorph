/* Iniital Conditions for the coupled transport-mechanics code
   March 2015 
*/
#ifndef initial_conds_
#define initial_conds_
#include "parameters.h"
#include <cstdlib>
//
using namespace dealii;

//Initial conditions
template <int dim>
class InitialConditions: public Function<dim>{
public:
  InitialConditions (): Function<dim>(totalDOF){}
  void vector_value (const Point<dim>   &p, Vector<double>   &values) const{
    Assert (values.size() == totalDOF, ExcDimensionMismatch (values.size(), totalDOF));
    values(totalDOF-7)=0; // u=0
    values(totalDOF-6)=0; 
    values(totalDOF-5)=0; 
    //    const Point<dim> center(alen/2, blen/2, clen/2);
    //    double radius = std::pow(std::pow(alen,2)+std::pow(blen,2)+std::pow(clen,2),0.5)/2.0;
    //    values(totalDOF-1) = 2.0 - p.distance(center)/radius;
    //    if (p.distance(center) >= 0.8*outer_radius){

    //            values(totalDOF-2) = 0.25*kT*std::pow((std::exp(-(alen - p(0))/(alen)) - 0.5),3);
    //                values(totalDOF-1) = std::exp(-(alen - p(0))/(alen));

    //    values(totalDOF-1) = 0.5 + 0.01*std::sin(numbers::PI*8.0*p(0)/alen)*std::sin(numbers::PI*8.0*p(1)/blen);
    //    values(totalDOF-1) = std::exp(-4.0*std::pow(p(0),2)) + std::exp(-4.0*std::pow(p(0)-alen,2)) + std::exp(-4.0*std::pow(p(1),2)) + std::exp(-4.0*std::pow(p(1)-blen,2));
    //    if (p(0) < 0.1*alen || p(0) > 0.9*alen || p(1) < 0.1*blen || p(1) > 0.9*blen)
    //    if (p(0) < 0.1*blen || p(0) > (alen-0.1*blen) || p(1) < 0.1*blen || p(1) > 0.9*blen)
    //    if ((std::pow(p(0)-0.5*alen,8) + std::pow(p(1)-0.5*blen,8)*std::pow(alen/blen,8) - std::pow(0.25*alen,8) < 50000000) && (std::pow(p(0)-0.5*alen,8) + std::pow(p(1)-0.5*blen,8)*std::pow(alen/blen,8) - std::pow(0.25*alen,8) > -50000000))
    //    if (p(0) > 0.25*alen && p(0) < 0.75*alen && p(1) > 0.25*blen && p(1) < 0.75*blen)// && p(0) < 0.26*alen && p(0) > 0.74*alen && p(1) < 0.30*blen && p(1) > 0.70*blen)
    //    if ((std::pow(p(0)-0.5*alen,2) + std::pow(p(1)-0.5*blen,2) - std::pow(0.25*alen,2) < 0.05) )

    //    values(totalDOF-1) = std::exp(-(p.distance(center)-inner_radius)/(outer_radius-inner_radius));
    /*    if (p.distance(center) <= 1.5*inner_radius)
      {
	values(totalDOF-1) = 0.0;
      }
    else
      {
	values(totalDOF-1) = 0.0;
	}*/

    //    values(totalDOF-2) = 0.25*kT*std::pow((std::exp(-(p.distance(center)-inner_radius)/((outer_radius-inner_radius))) - 0.5),3);
    //    values(totalDOF-1) = std::exp(-(p.distance(center)-inner_radius)/((outer_radius-inner_radius)));
    /* Double sinosoidal function
    values(totalDOF-4) = 0.25*kT*std::pow((0.5 + 0.01*std::sin(numbers::PI*8.0*p(0)/alen)*std::sin(numbers::PI*8.0*p(1)/blen) - 0.5),3);
    values(totalDOF-3) = 0.5 + 0.01*std::sin(numbers::PI*8.0*p(0)/alen)*std::sin(numbers::PI*8.0*p(1)/blen);
    values(totalDOF-2) = 0.25*kT*std::pow((0.5 + 0.01*std::sin(numbers::PI*8.0*p(0)/alen)*std::sin(numbers::PI*8.0*p(1)/blen) - 0.5),3);
    values(totalDOF-1) = 0.5 + 0.01*std::sin(numbers::PI*8.0*p(0)/alen)*std::sin(numbers::PI*8.0*p(1)/blen);
    */
       
       //Random
    /*    if (p.distance(center) <= alen/4.0){
      values(totalDOF-3) = 0.23 + static_cast <double> (rand())/(static_cast <double>(RAND_MAX/0.02));//Randomized over [0.23,0.25]
    }
    else{
      values(totalDOF-3) = 0.75 + static_cast <double> (rand())/(static_cast <double>(RAND_MAX/0.02));//Randomized over [0.75,0.77]
      }*/
    values(totalDOF-3) = -1.0 + static_cast <double> (rand())/(static_cast <double>(RAND_MAX/2.0));//Randomized over [-1.0,1.0]
    values(totalDOF-4) = 0.25*kT*std::pow((values(totalDOF-3) - 0.5),3);
    values(totalDOF-1) = -1.0 + static_cast <double> (rand())/(static_cast <double>(RAND_MAX/2.0));//Randomized over [-1.0,1.0]
    values(totalDOF-2) = 0.25*kT*std::pow((values(totalDOF-1) - 0.5),3);    

       //Random
    /*    values(totalDOF-3) = 0.49 + static_cast <double> (rand())/(static_cast <double>(RAND_MAX/0.02));
    values(totalDOF-4) = 0.25*kT*std::pow((values(totalDOF-3) - 0.5),3);
    values(totalDOF-1) = 0.49 + static_cast <double> (rand())/(static_cast <double>(RAND_MAX/0.02));
    values(totalDOF-2) = 0.25*kT*std::pow((values(totalDOF-1) - 0.5),3);    
    */

    /*
        //Step function
    values(totalDOF-3) = std::exp(-2.0*p.distance(center)/alen);
    values(totalDOF-4) = 0.25*kT*std::pow((values(totalDOF-3) - 0.5),3);
    values(totalDOF-1) = std::exp(-2.0*p.distance(center)/alen);
    values(totalDOF-2) = 0.25*kT*std::pow((values(totalDOF-1) - 0.5),3);        
    */
    //values(totalDOF-2) = 0.25*kT*std::pow((1.0 - 0.5),3);
    //values(totalDOF-1) = 1.0;

  }
};

#endif
