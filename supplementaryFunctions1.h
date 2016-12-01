/*
 * supplementaryFunctions.h
 *
 *  Created on: May 11, 2011
 */

#ifndef SUPPLEMENTARYFUNCTIONS_H_
#define SUPPLEMENTARYFUNCTIONS_H_
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/numerics/vector_tools.h>
using namespace dealii;

void solveSystem(SparseMatrix<double>& system_matrix, Vector<double>& system_rhs, Vector<double>& dU, double tolerance){
    //Iterative solve
	dU=0; SolverControl solver_control1 (2000, tolerance, true, true); solver_control1.set_failure_criterion(1.0e8); solver_control1.log_frequency(100); solver_control1.log_result(true);
    try{
    	//BICG, Jacobi
    	printf("GMRES, Jacobi:  ");
    	SolverGMRES<> solver (solver_control1); PreconditionJacobi<> preconditioner; preconditioner.initialize(system_matrix, 1.0); solver.solve (system_matrix, dU, system_rhs, preconditioner);
    	printf("iterative solve complete (Steps:%4u, Tol:%11.4e, InitialRes:%11.4e, Res:%11.4e).\n", solver_control1.last_step(), tolerance,  solver_control1.initial_value(), solver_control1.last_value());
    }
    catch(...){
    	dU=0; SolverControl solver_control2 (2000, tolerance, true, true); solver_control2.set_failure_criterion(1.0e8); solver_control2.log_frequency(100); solver_control2.log_result(true);
    	try{
    		//GMRES, Jacobi
    		printf("failed (Steps:%4u, Res:%11.4e) \nBICG, Jacobi: ", solver_control1.last_step(), solver_control1.last_value());
    		SolverBicgstab<> solver (solver_control2); PreconditionJacobi<> preconditioner; preconditioner.initialize(system_matrix, 1.0); solver.solve (system_matrix, dU, system_rhs, preconditioner);
			printf("iterative solve complete (Steps:%4u, Tol:%11.4e, InitialRes:%11.4e, Res:%11.4e).\n", solver_control2.last_step(), tolerance, solver_control2.initial_value(), solver_control2.last_value());
    	}
    	catch(...){
    		dU=0; SolverControl solver_control3 (2000, tolerance, true, true); solver_control3.set_failure_criterion(1.0e8); solver_control3.log_frequency(100); solver_control3.log_result(true);
        	try{
    			//BiCG, SOR
        		printf("failed (Steps:%4u, Res:%11.4e) \nBiCG, SOR:     ", solver_control2.last_step(), solver_control2.last_value());
        		SolverBicgstab<> solver (solver_control3); PreconditionSOR<> preconditioner; preconditioner.initialize(system_matrix, 1.0); solver.solve (system_matrix, dU, system_rhs, preconditioner);
        		printf("iterative solve complete (Steps:%4u, Tol:%11.4e, InitialRes:%11.4e, Res:%11.4e).\n", solver_control3.last_step(), tolerance, solver_control3.initial_value(), solver_control3.last_value());
        	}
        	catch(...){
        		dU=0; SolverControl solver_control4 (2000, tolerance, true, true); solver_control4.set_failure_criterion(1.0e8); solver_control4.log_frequency(100); solver_control4.log_result(true);
        		try{
        			//GMRES, SOR
        			printf("failed (Steps:%4u, Res:%11.4e) \nGMRES, SOR:    ", solver_control3.last_step(), solver_control3.last_value());
        		   	SolverGMRES<> solver (solver_control4); PreconditionSOR<> preconditioner; preconditioner.initialize(system_matrix, 1.0); solver.solve (system_matrix, dU, system_rhs, preconditioner);
        		   	printf("iterative solve complete (Steps:%4u, Tol:%11.4e, InitialRes:%11.4e, Res:%11.4e).\n", solver_control4.last_step(), tolerance, solver_control4.initial_value(), solver_control4.last_value());
        		}
        		catch(...){
        			printf("failed (Steps:%4u, Res:%11.4e) \nDirect Solve: ", solver_control4.last_step(), solver_control4.last_value());
        			dU=0; SparseDirectUMFPACK  A_direct; A_direct.initialize(system_matrix);  A_direct.vmult (dU, system_rhs);
        			printf("direct solve complete.\n");
        		}
    		}
		}
    }
}

template <class T, int dim>
T determinantOfMinor(unsigned int theRowHeightY, unsigned int theColumnWidthX, Table<2, T>& matrix){
  unsigned int x1 = theColumnWidthX == 0 ? 1 : 0;  /* always either 0 or 1 */
  unsigned int x2 = theColumnWidthX == 2 ? 1 : 2;  /* always either 1 or 2 */
  unsigned int y1 = theRowHeightY   == 0 ? 1 : 0;  /* always either 0 or 1 */
  unsigned int y2 = theRowHeightY   == 2 ? 1 : 2;  /* always either 1 or 2 */
  return matrix[y1][x1]*matrix[y2][x2] - matrix[y1][x2]*matrix[y2][x1];
}


template <class T, int dim>
void getInverse(Table<2, T>& matrix, Table<2, T>& invMatrix, T& det){
	if (dim==1){
		det=matrix[0][0];
		invMatrix[0][0]=1.0/det;
	}
	else if(dim==2){
		det=matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0];
		invMatrix[0][0]=matrix[1][1]/det;
		invMatrix[1][0]=-matrix[1][0]/det;
		invMatrix[0][1]=-matrix[0][1]/det;
		invMatrix[1][1]=matrix[0][0]/det;
	}
	else if(dim==3){
		det=  matrix[0][0]*determinantOfMinor<T, dim>(0, 0, matrix) - matrix[0][1]*determinantOfMinor<T, dim>(0, 1, matrix) +  matrix[0][2]*determinantOfMinor<T, dim>(0, 2, matrix);
		for (int y=0;  y< dim;  y++){
			for (int x=0; x< dim;  x++){
				invMatrix[y][x] = determinantOfMinor<T, dim>(x, y, matrix)/det;
				if( ((x + y) % 2)==1){invMatrix[y][x]*=-1;}
			}
		}
	}
	else throw "dim>3";
	if (std::abs(det)< 1.0e-15){
		printf("**************Near zero determinant in Matrix inversion***********************\n"); throw "Near zero determinant in Matrix inversion";
	}
}



#endif /* SUPPLEMENTARYFUNCTIONS_H_ */

